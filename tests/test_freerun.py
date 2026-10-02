"""G5 tests: the C4 free-run (src/freerun.py, robustness.c4_*, run_c4; passes.run_pass1 capture hook; §10.2, §14, §15.1;
IMP-086). Expected values are hand arithmetic (89 grid bins, 2-s segments, power x100 = 2 log10 units, 10% of 11 windows),
closed-form noise levels and the existing filter / predict as oracles; never read back from the code under test."""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import baseline, freerun as fr, model, passes, regression, robustness as rb, tuning, ukf, ukf_ext
from src import state_space as ss
from src.config import load_config
from sim_data import make_recording
from legacy_rule import legacy

CFG = load_config()
REPO = Path(__file__).resolve().parent.parent
FS = 256
LAYOUT = ss.make_layout(CFG)
# a smaller q than the declared 1e-2 for the end-to-end tests: at 1e-2 most synthetic windows are unstable (a finding, IMP-086)
CFGQ = copy.deepcopy(CFG)
CFGQ["ukf"]["process_noise"]["q_fixed"] = 1.0e-3
SPEC = ukf_ext.Spec("A", s2=(0.5, 0.5), tau=(0.05, 0.05))


def start_state(spec=SPEC, with_pots=False, shift=0.0):
    x0, _, buf = ukf.initial_state(LAYOUT, CFG)
    x = np.concatenate([x0, np.zeros(spec.nx)])
    x[0] += shift
    pots = ukf.buffer_snapshot(ss.make_potential_buffer(CFG)) if with_pots else None
    return {"x": x, "ring": ukf.buffer_snapshot(buf), "pots": pots}


def delay_buffer(snapshot):
    buf = ss.DelayBuffer(snapshot.shape[0] - 1)
    for lag in range(snapshot.shape[0]):
        buf._buf[(-lag) % snapshot.shape[0]] = snapshot[lag]
    buf._head = 0
    return buf


class MixResidual(ss.ResidualHook):
    def value(self, u_tgt, u_src, S_src):
        return 0.3 * np.tanh(np.asarray(u_src)) + 0.1 * np.asarray(S_src)


# ---- the ring and the propagation -----------------------------------------------------------------------------------------

def test_batch_ring_reads_by_lag_writes_and_replaces_per_realization():
    snap = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])                  # lag 0, 1, 2
    ring = fr.BatchRing(snap, 3)
    assert ring.delay == 2 and ring.read(0).shape == (3, 2)
    for lag in range(3):
        assert np.array_equal(ring.read(lag), np.tile(snap[lag], (3, 1)))
    ring.write(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert np.array_equal(ring.read(0)[:, 0], [1.0, 3.0, 5.0]) and np.array_equal(ring.read(1)[0], snap[0])
    ring.replace_latest(np.zeros((3, 2)))
    assert not ring.read(0).any() and np.array_equal(ring.read(1)[0], snap[0])
    with pytest.raises(fr.FreeRunError):
        ring.read(3)


@pytest.mark.parametrize("residual", [False, True])
def test_propagate_one_row_is_bit_identical_to_state_space_predict(residual):
    st = start_state(with_pots=residual, shift=1.5)
    X = st["x"][:LAYOUT.n][None]
    buf = delay_buffer(st["ring"])
    hook = None
    if residual:
        hook = MixResidual().bind(delay_buffer(st["pots"]))
    want = ss.predict(X, np.array([1.0]), buf, LAYOUT, CFG, residual=hook)
    ring = fr.BatchRing(st["ring"], 1)
    pots = fr.BatchRing(st["pots"], 1) if residual else None
    got = fr.propagate(X, ring, pots, LAYOUT, CFG, MixResidual() if residual else None)
    assert np.array_equal(got, want)
    assert np.array_equal(ring.snapshot()[:, 0], np.array([buf.read(lag) for lag in range(buf.delay + 1)]))
    if residual:
        assert np.array_equal(pots.snapshot()[:, 0], np.array([hook.pots.read(lag) for lag in range(hook.pots.delay + 1)]))


def test_propagate_rows_are_independent_of_each_other():
    st = start_state()
    X = np.tile(st["x"][:LAYOUT.n], (3, 1))
    X[1, 0] += 2.0
    ring = fr.BatchRing(st["ring"], 3)
    out = fr.propagate(X, ring, None, LAYOUT, CFG)
    one = fr.propagate(X[1:2], fr.BatchRing(st["ring"], 1), None, LAYOUT, CFG)
    assert np.array_equal(out[0], out[2]) and not np.array_equal(out[0], out[1]) and np.array_equal(out[1], one[0])


# ---- the stochastic simulation ----------------------------------------------------------------------------------------------

Q = 1.0e-2


def quiet_cfg(r_fraction=0.0):
    cfg = copy.deepcopy(CFG)
    cfg["ukf"]["observation_noise"]["R_fraction_of_rescaled_variance"] = r_fraction
    return cfg


def test_process_noise_has_the_filters_variance_q_times_the_steady_state_variance():
    st = start_state(ukf_ext.Spec("A", s2=(0.0, 0.0), tau=(1.0, 1.0)))
    out = fr.simulate(st, 1, np.random.default_rng(1), CFG, LAYOUT, ukf_ext.Spec("A", s2=(0.0, 0.0), tau=(1.0, 1.0)), Q,
                      n_real=4000, keep_state=True)
    X = out["X"]
    for state in (0, 1):                                  # y0 and y3 of node 1: identical deterministic part across rows
        want = np.sqrt(Q * CFG["ukf"]["initial_state"]["neural_variance"][state])
        assert X[:, state].std() == pytest.approx(want, rel=0.05)
    assert np.allclose(X[:, ss.N_NEURAL:], X[0, ss.N_NEURAL:])          # the parameters never move


def test_white_observation_noise_has_the_filters_R_and_the_ou_state_its_stationary_variance():
    spec0 = ukf_ext.Spec("A", s2=(0.0, 0.0), tau=(1.0, 1.0))
    st = start_state(spec0)
    det = fr.simulate(st, 100, np.random.default_rng(5), quiet_cfg(), LAYOUT, spec0, 0.0, n_real=200)["y"]
    assert np.allclose(det, det[:1])                                                # no noise at all: rows identical
    noisy = fr.simulate(st, 100, np.random.default_rng(5), CFG, LAYOUT, spec0, 0.0, n_real=200)["y"]
    want = np.sqrt(0.25) * CFG["rescaling"]["sigma_ref"]                            # R = 0.25 sigma_ref^2 (section 7.6)
    assert (noisy - det).std() == pytest.approx(want, rel=0.03) and abs((noisy - det).mean()) < 0.01 * want
    spec = ukf_ext.Spec("A", s2=(1.0, 4.0), tau=(0.05, 0.05))
    st = start_state(spec)
    ou = fr.simulate(st, 400, np.random.default_rng(6), quiet_cfg(), LAYOUT, spec, 0.0, n_real=300)["y"]
    dev = ou[:, 200:] - det[:, :1].mean()                                           # the deterministic level, no noise
    base = fr.simulate(start_state(spec0), 400, np.random.default_rng(6), quiet_cfg(), LAYOUT, spec0, 0.0, n_real=1)["y"][0, 200:]
    resid = ou[:, 200:] - base
    assert resid[:, :, 0].var() == pytest.approx(1.0, rel=0.1) and resid[:, :, 1].var() == pytest.approx(4.0, rel=0.1)
    assert dev.shape == resid.shape


def test_a_start_far_from_the_fixed_point_or_a_non_finite_residual_is_unstable():
    spec0 = ukf_ext.Spec("A", s2=(0.0, 0.0), tau=(1.0, 1.0))
    sd0 = np.sqrt(CFG["ukf"]["initial_state"]["neural_variance"][0])
    far = fr.simulate(start_state(spec0, shift=30.0 * sd0), 5, np.random.default_rng(1), legacy(CFG), LAYOUT, spec0, Q, n_real=4)
    assert far["unstable"].all() and np.isnan(far["y"][:, 1:]).all()
    ok = fr.simulate(start_state(spec0), 20, np.random.default_rng(1), CFG, LAYOUT, spec0, Q, n_real=4)
    assert not ok["unstable"].any() and np.isfinite(ok["y"]).all()

    class Bad(ss.ResidualHook):
        def value(self, u_tgt, u_src, S_src):
            return np.full(np.shape(u_tgt), np.nan)

    bad = fr.simulate(start_state(spec0, with_pots=True), 5, np.random.default_rng(1), CFG, LAYOUT, spec0, Q, residual=Bad(), n_real=4)
    assert bad["unstable"].all()
    with pytest.raises(fr.FreeRunError):
        fr.simulate(start_state(spec0, with_pots=True), 2, np.random.default_rng(1), CFG, LAYOUT, spec0, Q, n_real=2)


def test_after_every_step_the_newest_ring_entries_are_s_and_the_potential_of_the_noisy_state():
    spec0 = ukf_ext.Spec("A", s2=(0.0, 0.0), tau=(1.0, 1.0))
    out = fr.simulate(start_state(spec0, with_pots=True), 3, np.random.default_rng(2), CFG, LAYOUT, spec0, Q, residual=MixResidual(),
                      n_real=5, keep_state=True)
    k = model.constants(CFG)
    pot = ss.potentials(out["X"])
    assert np.allclose(out["pots"].read(0), pot) and np.allclose(out["ring"].read(0), model.sigmoid(pot, k["e0"], k["v0"], k["r"]))
    assert not np.allclose(out["pots"].read(0)[0], out["pots"].read(0)[1])               # every realization has its own ring


def test_the_same_key_gives_the_same_realizations_and_a_different_key_does_not():
    spec0 = ukf_ext.Spec("A", s2=(0.5, 0.5), tau=(0.05, 0.05))
    st = start_state(spec0)
    a = fr.simulate(st, 30, np.random.default_rng([42, 0, 2, 0]), CFG, LAYOUT, spec0, Q, n_real=3)["y"]
    b = fr.simulate(st, 30, np.random.default_rng([42, 0, 2, 0]), CFG, LAYOUT, spec0, Q, n_real=3)["y"]
    c = fr.simulate(st, 30, np.random.default_rng([42, 0, 2, 1]), CFG, LAYOUT, spec0, Q, n_real=3)["y"]
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a[0], a[1])


# ---- spectra and the window error -----------------------------------------------------------------------------------------

def test_the_grid_is_89_bins_of_half_a_hertz_from_1_to_45_hz():
    fs, seg_n, sel = fr.grid(CFG)
    f = np.fft.rfftfreq(seg_n, 1.0 / fs)[sel]
    assert (fs, seg_n) == (256, 512) and sel.sum() == 89 and f[0] == 1.0 and f[-1] == 45.0 and np.allclose(np.diff(f), 0.5)
    assert [fr.n_segments(L, CFG) for L in (2, 5, 10)] == [1, 2, 5]


def test_window_error_is_two_log_units_for_a_hundredfold_power_and_zero_for_a_mean_of_logs_balance():
    rng = np.random.default_rng(3)
    obs = rng.standard_normal((2, 10 * FS))
    sim_ten = np.stack([obs.T * 10.0, obs.T * 10.0])                       # (R=2, n, 2): power x100 -> +2 in log10
    e, segs = fr.window_error(obs, sim_ten, 10, CFG)
    assert e == pytest.approx(2.0, abs=1e-9) and len(segs) == 5 and all(s == pytest.approx(2.0, abs=1e-9) for s in segs)
    balanced = np.stack([obs.T * 10.0, obs.T / 10.0])                      # +2 and -2: the mean of logs is the observed log
    assert fr.window_error(obs, balanced, 10, CFG)[0] == pytest.approx(0.0, abs=1e-9)
    half = obs.T.copy()
    half[2 * FS:] *= 1.0                                                   # segment 0 equal, segment 1 scaled
    half[FS * 2:FS * 4] *= 10.0
    e5, segs5 = fr.window_error(obs[:, :5 * FS], np.stack([half[:5 * FS]] * 2), 5, CFG)
    assert len(segs5) == 2 and segs5[0] == pytest.approx(0.0, abs=1e-9) and segs5[1] == pytest.approx(2.0, abs=1e-9)
    assert e5 == pytest.approx(1.0, abs=1e-9)                              # a 5-s window is its 2 whole segments, mean 1.0


def test_window_error_refuses_a_non_finite_spectrum():
    obs = np.zeros((2, 2 * FS))                                            # a flat zero segment: log10(0)
    with pytest.raises(fr.FreeRunError):
        fr.window_error(obs, np.ones((1, 2 * FS, 2)), 2, CFG)


# ---- the window plan ------------------------------------------------------------------------------------------------------------

def test_mask_starts_and_windows_are_anchored_at_the_first_scored_sample():
    lengths = [3000, 1000, 40]
    first = fr.mask_starts(passes.scoring_mask(lengths, CFG))
    assert first == [1536, 128, None]                                      # 6 s cumulative burn-in, then 0.5 s per segment
    assert fr.window_plan(lengths, first, [True, True, True], 512) == [(0, 1536), (0, 2048), (1, 128)]
    assert fr.window_plan(lengths, first, [False, True, True], 512) == [(1, 128)]          # a diverged segment gives none
    assert fr.window_plan(lengths, first, [True, True, True], 1280) == [(0, 1536)]        # the tail never fills a window
    assert fr.capture_indices(lengths, first, [512, 1280]) == [[1536, 2048], [128], []]


# ---- the capture hook in passes.run_pass1 ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_rec():
    rec = make_recording(CFG, (9.0, 5.0), 11, g12=3.0, g21=1.0, m=0.3, p=np.array([220.0, 250.0]), gap_seconds=1.0, burn_seconds=2.0)
    return rec["segments"], rec["starts"]


def test_the_capture_hook_changes_nothing_and_holds_the_filtered_state_before_the_index(small_rec):
    segs, starts = small_rec
    spec = passes.make_spec(CFG, "A", segs)
    plain = passes.run_pass1(segs, starts, CFG, forward_only=True, spec=spec)
    cap = passes.run_pass1(segs, starts, CFG, forward_only=True, spec=spec, capture_idx=[[1600, 2000], [200]])
    assert plain.segments[0].captured is None
    for a, b in zip(plain.segments, cap.segments):
        assert np.array_equal(a.sq_err, b.sq_err) and np.array_equal(a.z_pred, b.z_pred)
    assert plain.gain_estimate == cap.gain_estimate
    # oracle: the extended Numba filter run directly on segment 0 from the same prior
    z = np.ascontiguousarray(segs[0].T)
    full = ukf_ext.run_filter_ext(z, CFG, LAYOUT, passes.resolve_q(CFG, None, "A"), spec)
    for t0 in (1600, 2000):
        c = cap.segments[0].captured[t0]
        assert c["x"].shape == (LAYOUT.n + 2,) and c["pots"] is None
        assert np.array_equal(c["x"], full.x[t0 - 1]) and np.array_equal(c["ring"], full.snapshots[t0])
    k = model.constants(CFG)
    s0 = model.sigmoid(ss.potentials(cap.segments[0].captured[1600]["x"][None, :ss.N_NEURAL])[0], k["e0"], k["v0"], k["r"])
    assert np.allclose(cap.segments[0].captured[1600]["ring"][0], s0)      # the newest ring entry is S of the filtered state
    with pytest.raises(passes.PassError):
        passes.run_pass1(segs, starts, CFG, forward_only=True, spec=spec, capture_idx=[[0], []])
    with pytest.raises(passes.PassError):
        passes.run_pass1(segs, starts, CFG, forward_only=True, spec=spec, capture_idx=[[1600], [segs[1].shape[1]]])


def test_the_m3_filter_also_captures_the_potential_ring(small_rec):
    segs, starts = small_rec
    spec = passes.make_spec(CFG, "A", segs)
    z = np.zeros(3)
    resid = regression.FrozenEquation(doc={"no_term": False, "equation": "tanh(u_tgt)",
                                           "zscore": {"mean_X": [0, 0, 0], "sd_X": [1, 1, 1], "mean_y": 0.0, "sd_y": 1.0, "n": 1}},
                                      sha256="x").residual()
    assert z.shape == (3,)
    cap = passes.run_pass1(segs, starts, CFG, forward_only=True, spec=spec, residual=resid, capture_idx=[[1700], []])
    c = cap.segments[0].captured[1700]
    assert c["pots"].shape == c["ring"].shape == (CFG["coupling"]["delay_substeps"] + 1, 2)
    assert np.allclose(c["pots"][0], ss.potentials(c["x"][None, :ss.N_NEURAL])[0])      # lag 0 = potential of the filtered state


# ---- stability, the condition and the statistics ---------------------------------------------------------------------------------

def windows(L, n, n_unstable, err=1.0):
    return [{"length_s": L, "segment": 0, "t0": i, "stable": i >= n_unstable, "error": err if i >= n_unstable else None,
             "segment_errors": None} for i in range(n)]


@pytest.mark.parametrize("n,bad,unstable", [(10, 1, False), (11, 2, True), (20, 2, False), (20, 3, True), (10, 0, False)])
def test_a_recording_is_unstable_at_a_length_only_above_ten_percent_of_its_windows(n, bad, unstable):
    res = {"pass1_diverged": False, "windows": windows(2, n, bad)}
    s = rb.c4_recording_summary(res, [2], CFG)
    assert s["per_length"][2]["unstable"] is unstable and s["stable_all"] is (not unstable)
    assert s["per_length"][2]["n_unstable"] == bad


def test_pass1_divergence_and_no_windows_make_a_recording_unstable():
    assert rb.c4_recording_summary({"pass1_diverged": True, "windows": []}, [2, 5], CFG)["stable_all"] is False
    s = rb.c4_recording_summary({"pass1_diverged": False, "windows": windows(2, 5, 0)}, [2, 5], CFG)
    assert s["stable_all"] is False and s["per_length"][5]["reason"] == "no_windows" and s["per_length"][2]["error"] == 1.0


@pytest.mark.parametrize("n_ok,n,met", [(4, 5, True), (3, 4, False), (8, 10, True), (7, 10, False), (0, 0, False), (1, 1, True)])
def test_the_stability_condition_is_at_least_80_percent(n_ok, n, met):
    sums = [{"stable_all": i < n_ok} for i in range(n)]
    out = rb.c4_stable_fraction(sums, CFG)
    assert out["met"] is met and out["n_stable"] == n_ok and out["min_fraction"] == 0.8


@pytest.mark.parametrize("gate,ok", [({"low_confidence": False, "hard_stop": False, "complete": True}, True),
                                     ({"low_confidence": True, "hard_stop": False, "complete": True}, False),
                                     ({"low_confidence": False, "hard_stop": True, "complete": True}, False),
                                     ({"low_confidence": False, "hard_stop": False, "complete": False}, False),
                                     ({"hard_stop": False}, False), (None, False)])
def test_g0_full_pass_needs_no_low_confidence_no_hard_stop_and_a_complete_gate(gate, ok):
    assert rb.c4_gate_full_pass(gate) is ok


def test_the_ratio_is_a_ratio_of_means_and_its_upper_bound_is_the_97_5th_percentile_of_the_resampled_ratio():
    e2 = np.array([1.0, 1.0, 1.0, 1.0])
    e5 = np.array([1.1, 1.3, 1.2, 1.4])
    idx = np.array([[0] * 4, [1] * 4, [2] * 4, [3] * 4])                  # each resample is one subject repeated
    r = rb.c4_ratio(e2, e5, idx, CFG)
    assert r["ratio"] == pytest.approx(1.25) and r["n"] == 4 and r["B"] == 4
    assert r["upper"] == pytest.approx(1.3 + 0.925 * 0.1)                  # linear percentile 97.5 of [1.1, 1.2, 1.3, 1.4]
    assert r["ci"][0] == pytest.approx(1.1 + 0.075 * 0.1)
    unequal = rb.c4_ratio(np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([[0, 0], [1, 1]]), CFG)
    assert unequal["ratio"] == pytest.approx(4.0 / 3.0) and unequal["upper"] == pytest.approx(1.0 + 0.975 * 1.0)
    with pytest.raises(rb.RobustnessError):
        rb.c4_ratio(e2, e5, np.zeros((3, 3), dtype=int), CFG)


def summary(errs, stable_all=True):
    return {"stable_all": stable_all, "per_length": {L: {"error": e, "unstable": not stable_all, "n_windows": 3, "n_unstable": 0}
                                                    for L, e in zip((2, 5, 10), errs)}}


def test_the_ratio_analysis_uses_only_recordings_stable_at_every_length_and_judges_the_limit():
    good = [summary((1.0, 1.05, 1.1)) for _ in range(6)] + [summary((9.0, 9.0, 9.0), stable_all=False)]
    out = rb.c4_ratio_analysis(good, [2, 5, 10], CFG, 42, 200)
    assert out["n_subjects"] == 6 and out["status"] == "ok" and out["limit"] == 1.2
    assert out["ratios"]["5"]["ratio"] == pytest.approx(1.05) and out["ratios"]["10"]["upper"] == pytest.approx(1.1)
    assert out["passed"] is True
    bad = [summary((1.0, 1.0, 1.3)) for _ in range(6)]
    assert rb.c4_ratio_analysis(bad, [2, 5, 10], CFG, 42, 200)["passed"] is False
    pilot = rb.c4_ratio_analysis([summary((1.0, 1.0, 1.0)) for _ in range(3)] and
                                 [{"stable_all": True, "per_length": {2: {"error": 1.0}, 10: {"error": 1.1}}} for _ in range(3)],
                                 [2, 10], CFG, 42, 100)
    assert pilot["passed"] is None and "10" in pilot["ratios"]                    # not every configured length was run
    assert rb.c4_ratio_analysis([summary((1.0, 1.0, 1.0))], [2, 5, 10], CFG, 42, 10)["status"] == "too_few_subjects"


def with_windows(errs, L=2):
    rows = [{"length_s": L, "segment": 0, "t0": i, "stable": True, "error": e, "segment_errors": None} for i, e in enumerate(errs)]
    s = rb.c4_recording_summary({"pass1_diverged": False, "windows": rows}, [L], CFG)
    return dict(s, windows=rows)


def test_the_holm_contrast_is_the_paired_difference_m3_minus_m2_over_common_stable_windows():
    m2 = {f"s{i}": with_windows([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) for i in range(5)}
    m3 = {f"s{i}": with_windows([0.9] * 10) for i in range(5)}
    out = rb.c4_freerun_contrast(m2, m3, [2], CFG, 42, 400)["freerun:2s"]
    assert out["mean"] == pytest.approx(-0.1) and out["n"] == 5 and out["difference"] == "M3 - M2" and out["p"] < 0.05
    # a window unstable in one model only is dropped from both
    m3b = {k: dict(v, windows=[dict(w, stable=(i != 0), error=(None if i == 0 else 0.5)) for i, w in enumerate(v["windows"])])
           for k, v in m3.items()}
    m2b = {k: dict(v, windows=[dict(w, error=(100.0 if i == 0 else 1.0)) for i, w in enumerate(v["windows"])])
           for k, v in m2.items()}
    # 1 of 10 windows unstable is within the 10% limit, so the recording stays and only the common windows are compared
    m3b = {k: dict(rb.c4_recording_summary({"pass1_diverged": False, "windows": v["windows"]}, [2], CFG), windows=v["windows"])
           for k, v in m3b.items()}
    out2 = rb.c4_freerun_contrast(m2b, m3b, [2], CFG, 42, 400)["freerun:2s"]
    assert out2["mean"] == pytest.approx(-0.5)                                   # (0.5 - 1.0) on the 9 common windows
    same = rb.c4_freerun_contrast(m2, m2, [2], CFG, 42, 400, same_model=True)["freerun:2s"]
    assert same["p"] == 1.0 and same["mean"] == 0.0 and same["ci"] == [0.0, 0.0] and same["same_model"] is True


def test_the_holm_family_is_eleven_with_free_run_and_eight_without_and_the_size_is_checked():
    names = [m["name"] for m in rb.holm_members(CFG, c4_attempted=False)]
    assert len(names) == 8 and not any(n.startswith("freerun") for n in names) and CFG["statistics"]["holm_family_size"] == 11
    assert len(rb.holm_members(CFG)) == 11
    raw = {n: 0.01 * (i + 1) for i, n in enumerate(names)}
    rep = rb.holm_report(CFG, raw, c4_attempted=False)
    assert rep["status"] == "complete" and rep["members"][0]["adjusted_p"] == pytest.approx(0.08)           # smallest p x 8
    assert rb.holm_report(CFG, raw)["status"] == "partial_family"                  # with C4 attempted the three p are missing
    with pytest.raises(rb.RobustnessError):
        rb.holm_report(CFG, dict(raw, **{"freerun:2s": 0.5}), c4_attempted=False)
    cfg = copy.deepcopy(CFG)
    cfg["statistics"]["holm_family_size_without_freerun"] = 9
    with pytest.raises(rb.RobustnessError, match="without_freerun"):
        rb.holm_members(cfg, c4_attempted=False)


# ---- run_c4 end to end (synthetic, stub loader, temp root) ----------------------------------------------------------------------

def _stable_rec(seed, seconds=(14.0, 14.0)):
    rec = make_recording(CFG, seconds, seed, g12=0.0, g21=0.0, m=0.2, p=np.array([200.0, 210.0]), gap_seconds=1.0, burn_seconds=2.0)
    return {"reason": None, "segments": rec["segments"], "starts": rec["starts"], "key": f"k{seed}"}


def _world(monkeypatch, n_train=14, n_test=4):
    import main as main_mod
    train = [f"sub-{i:03d}" for i in range(1, n_train + 1)]
    test = [f"sub-{i:03d}" for i in range(n_train + 1, n_train + n_test + 1)]
    data = {s: _stable_rec(500 + k) for k, s in enumerate(train + test)}
    blown = _stable_rec(900)
    blown["segments"] = [np.ascontiguousarray(seg * 40.0 + 25.0) for seg in blown["segments"]]       # pass 1 diverges
    data[train[0]] = blown
    asked = []

    def loader(s, session=None):
        asked.append(s)
        return data[s]

    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(train))
    monkeypatch.setattr(main_mod, "test_subjects", lambda seed, root=None, cfg=None: list(test))
    return train, test, loader, asked


@pytest.fixture(scope="module")
def pilot_c4(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("c4root")
    shutil.copy(REPO / "config.yml", root / "config.yml")
    train, test, loader, asked = _world(mp)
    pilot = train[:4]
    out = rb.run_c4(CFGQ, root, 42, pilot=True, loader=loader, pilot_ids=pilot, n_jobs=2)
    yield {"root": root, "out": out, "asked": asked, "pilot": pilot, "loader": loader, "train": train}
    mp.undo()


def test_pilot_c4_reads_only_pilot_subjects_and_is_mechanics_only(pilot_c4):
    d = pilot_c4["out"]["doc"]
    assert set(pilot_c4["asked"]) <= set(pilot_c4["pilot"]) and d["subjects"] == sorted(pilot_c4["pilot"])
    assert d["pilot"] and d["mechanics_only"] and d["status"] == "mechanics_only" and d["lengths_s"] == [2, 10]
    assert pilot_c4["out"]["path"].parent == pilot_c4["root"] / "results" / "pilot"
    assert not (pilot_c4["root"] / "results" / "c4_42.json").exists()
    assert d["model"]["mode"] == "absent" and d["model"]["scored"] == "M2 stand-in"
    assert d["c4_verdict"]["passed"] is None and d["holm_raw_p"] == {} and d["holm_c4_attempted"] is None
    assert d["gate"]["used_as_condition"] is False and d["n_realizations"] == 20


def test_pilot_c4_counts_a_pass1_diverged_recording_as_unstable_and_reports_the_condition(pilot_c4):
    d = pilot_c4["out"]["doc"]
    blown = pilot_c4["pilot"][0]
    row = d["per_subject"][blown]["M2"]
    assert row["pass1_diverged"] is True and row["stable_all"] is False
    cond = d["stability_condition"]
    n_ok = sum(v["M2"]["stable_all"] for v in d["per_subject"].values())
    assert cond["n_recordings"] == 4 and cond["n_stable"] == n_ok and cond["met"] is (n_ok * 5 >= 4 * 4)
    assert d["ratio_analysis"]["n_subjects"] == n_ok


def test_pilot_c4_rerun_is_unchanged_whatever_the_worker_count(pilot_c4):
    again = rb.run_c4(CFGQ, pilot_c4["root"], 42, pilot=True, loader=pilot_c4["loader"], pilot_ids=pilot_c4["pilot"], n_jobs=1,
                      use_cache=False)
    assert again["status"] == "unchanged"


def test_pilot_c4_guard_refuses_a_subject_outside_the_plan(pilot_c4):
    plan = rb.resolve_diag_subjects(CFG, pilot_c4["root"], 42, pilot=True, pilot_ids=pilot_c4["pilot"])
    with pytest.raises(rb.GuardError):
        rb.guarded_loader(pilot_c4["loader"], plan)(pilot_c4["train"][8])
    with pytest.raises(rb.GuardError, match="confirmatory=True"):
        rb.run_c4(CFGQ, pilot_c4["root"], 42, pilot=False, loader=pilot_c4["loader"])
    with pytest.raises(rb.GuardError):
        rb.run_c4(CFGQ, pilot_c4["root"], 42, pilot=True, confirmatory=True, loader=pilot_c4["loader"])


def _gate(root, **kw):
    path = root / "outputs" / "gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict({"low_confidence": False, "hard_stop": False, "complete": True}, **kw)), encoding="utf-8")


def test_confirmatory_c4_is_not_attempted_when_g0_has_low_confidence_and_reads_no_recording(tmp_path, monkeypatch):
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    train, test, loader, asked = _world(monkeypatch)
    _gate(tmp_path, low_confidence=True)
    out = rb.run_c4(CFGQ, tmp_path, 42, pilot=False, confirmatory=True, loader=loader)
    d = out["doc"]
    assert d["status"] == "not_attempted" and d["holm_c4_attempted"] is False and d["holm_raw_p"] == {} and asked == []
    assert d["gate"]["full_pass"] is False and d["gate"]["used_as_condition"] is True
    assert out["path"] == tmp_path / "results" / "c4_42.json"
    assert rb.run_c4(CFGQ, tmp_path, 42, pilot=False, confirmatory=True, loader=loader)["status"] == "unchanged"


def test_confirmatory_c4_needs_the_frozen_equation_once_the_gate_passes(tmp_path, monkeypatch):
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    train, test, loader, asked = _world(monkeypatch)
    _gate(tmp_path)
    with pytest.raises(rb.GuardError, match="frozen equation"):
        rb.run_c4(CFGQ, tmp_path, 42, pilot=False, confirmatory=True, loader=loader)
    assert asked == []


def _frozen_file(root, no_term, eq="tanh(u_tgt)"):
    z = regression.ZScore(mean_X=np.zeros(3), sd_X=np.ones(3), mean_y=0.0, sd_y=1.0, n=10)
    rec = {"role": "primary", "no_term": no_term, "equation": "0.37" if no_term else eq, "zscore": z.to_dict(), "front": [],
           "fit_subjects": ["sub-001"], "val_subjects": ["sub-002"], "complexity": 2, "val_loss": 1.0,
           "signatures": [] if no_term else [eq]}
    regression.write_frozen_equation(regression.frozen_equation_path(CFG, root, 42, False),
                                     regression.build_frozen_document(rec, CFG, REPO, 42, False), force=True)


def test_confirmatory_c4_attempted_with_a_no_term_equation_gives_p_one_for_every_free_run_member(tmp_path, monkeypatch):
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    train, test, loader, asked = _world(monkeypatch, n_test=5)
    _gate(tmp_path)
    _frozen_file(tmp_path, True)
    out = rb.run_c4(CFGQ, tmp_path, 42, pilot=False, confirmatory=True, loader=loader, n_jobs=2)
    d = out["doc"]
    assert set(asked) == set(test) and d["mode"] == "confirmatory" and not d["mechanics_only"]
    assert d["stability_condition"]["met"] and d["status"] == "attempted" and d["holm_c4_attempted"] is True
    assert d["model"]["mode"] == "no_term" and set(d["holm_raw_p"]) == {"freerun:2s", "freerun:5s", "freerun:10s"}
    assert set(d["holm_raw_p"].values()) == {1.0}
    assert d["ratio_analysis"]["status"] == "ok" and d["c4_verdict"]["passed"] in (True, False)


def test_confirmatory_c4_is_not_attempted_when_too_few_recordings_are_stable(tmp_path, monkeypatch):
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    train, test, loader, asked = _world(monkeypatch, n_test=2)
    import main as main_mod
    blown = _stable_rec(901)
    blown["segments"] = [np.ascontiguousarray(seg * 40.0 + 25.0) for seg in blown["segments"]]
    inner = loader
    monkeypatch.setattr(main_mod, "test_subjects", lambda seed, root=None, cfg=None: list(test))

    def loader2(s, session=None):
        return blown if s == test[0] else inner(s)

    _gate(tmp_path)
    _frozen_file(tmp_path, True)
    out = rb.run_c4(CFGQ, tmp_path, 42, pilot=False, confirmatory=True, loader=loader2, n_jobs=2)
    d = out["doc"]
    assert d["stability_condition"]["met"] is False and d["status"] == "not_attempted" and d["holm_raw_p"] == {}
    assert d["holm_c4_attempted"] is False and "ratio_analysis" not in d


def test_confirmatory_c4_scored_equation_runs_m3_and_m2_and_the_contrast_is_not_trivial(tmp_path, monkeypatch):
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    train, test, loader, asked = _world(monkeypatch, n_test=3)
    _gate(tmp_path)
    _frozen_file(tmp_path, False, eq="0.5*tanh(u_tgt) + 0.2*S_src")
    out = rb.run_c4(CFGQ, tmp_path, 42, pilot=False, confirmatory=True, loader=loader, n_jobs=3)
    d = out["doc"]
    assert d["model"]["mode"] == "scored" and d["model"]["scored"] == "M3" and d["status"] in ("attempted", "not_attempted")
    for sid in test:
        assert set(d["per_subject"][sid]) == {"M2", "M3"}
    if d["status"] == "attempted":
        c = d["freerun_contrast"]
        assert set(c) == {"freerun:2s", "freerun:5s", "freerun:10s"} and all(v["same_model"] is False for v in c.values())
        assert all(v["mean"] != 0.0 and 0.0 <= v["p"] <= 1.0 for v in c.values())
        assert d["holm_raw_p"] == {k: v["p"] for k, v in c.items()}
    m2 = [d["per_subject"][s]["M2"]["per_length"][2]["error"] for s in test]
    m3 = [d["per_subject"][s]["M3"]["per_length"][2]["error"] for s in test]
    assert all(e is not None for e in m2 + m3) and m2 != m3                         # the residual really changes the free-run


def test_source_hygiene_of_freerun_py():
    import ast
    import re
    text = (REPO / "src" / "freerun.py").read_text(encoding="utf-8")
    assert not re.search(r"(?<![A-Za-z_])print\(", text) and "np.random.seed" not in text and "np.random.rand" not in text
    assert "import pysr" not in text and "load_pysr" not in text
    tree = ast.parse(text)
    bad = [(n.lineno, n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool) and n.value not in {0, 1, 2, 0.5}]                         # 0.5: the Heun average
    assert not bad, bad
    assert "default_rng" not in text or "rng" in text                              # every random draw comes from a passed generator
