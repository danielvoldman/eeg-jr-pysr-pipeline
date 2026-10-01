"""C3 tests: two-pass parameter handling (§5.1, §7.5, §8.3, §10.2, §11.3; IMP-030 to IMP-036).

Expected values come from the simulation truth, from plain numpy/filterpy code written here (an
independent numpy Heun propagation and mixing observation, filterpy's UKF and RTS smoother as the
oracle), or from arithmetic done in the test; never from src/passes.py itself. Nothing touches
config.yml except reading it.
"""
import ast
import copy
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from src import model, passes, state_space as ss, ukf
from src.config import load_config
from sim_data import make_recording

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
FS = CFG["preprocessing"]["observation_fs_hz"]
BURN = 128                      # 0.5 s at 256 Hz (windows.training_burn_in_s)
WIN = 512                       # 2 s at 256 Hz (windows.training_window_s)
DELAY = CFG["coupling"]["delay_substeps"]
Q_TEST = 1.0e-2                 # fixed a priori (inside the 1e-4 .. 1e-1 grid); C4 tunes q, not C3

# truth of the main scenario and the recovery bands: fixed BEFORE the first run (C3 approval)
TRUE = {"g12": 12.0, "g21": 0.0, "p": (220.0, 260.0), "m": 0.35, "log_rho": math.log(3.25 / 22.0)}
BAND_G12 = (0.5 * TRUE["g12"], 1.5 * TRUE["g12"])
BAND_G21 = 3.0
BAND_P = 50.0
BAND_M = 0.15
BAND_LOG_RHO = 0.2
SEG_SECONDS = (12, 10, 8)


def S(v):
    e0, v0, r = CFG["jansen_rit"]["e0"], CFG["jansen_rit"]["v0"], CFG["jansen_rit"]["r"]
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - np.asarray(v, dtype=float))))


def fake_params(**kw):
    """RecordingParams with chosen values (no filtering involved)."""
    base = dict(p1=225.0, p2=245.0, log_rho1=math.log(3.25 / 22.0) + 0.05, log_rho2=math.log(3.25 / 22.0) - 0.08,
                g12=9.0, g21=-4.0, m=0.3)
    base.update(kw)
    ab = CFG["priors"]["AB_product"]
    rho1, rho2 = math.exp(base["log_rho1"]), math.exp(base["log_rho2"])
    return passes.RecordingParams(
        p1=base["p1"], p2=base["p2"], log_rho1=base["log_rho1"], log_rho2=base["log_rho2"], rho1=rho1, rho2=rho2,
        A1=math.sqrt(ab * rho1), B1=math.sqrt(ab / rho1), A2=math.sqrt(ab * rho2), B2=math.sqrt(ab / rho2),
        g12=base["g12"], g21=base["g21"], m=base["m"], m_raw=base["m"], m_clipped_fraction=0.0,
        posterior_sd={}, n_samples=1, layout_names=())


# ---- the main scenario, run once ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def layout():
    return ss.make_layout(CFG)


@pytest.fixture(scope="module")
def scen():
    rec = make_recording(CFG, SEG_SECONDS, seed=31, g12=TRUE["g12"], g21=TRUE["g21"], m=TRUE["m"], p=np.array(TRUE["p"]))
    p1 = passes.run_pass1(rec["segments"], rec["starts"], CFG, Q_TEST, filter_name="19D")
    p2 = passes.run_pass2(rec["segments"], rec["starts"], p1.params, CFG, Q_TEST, filter_name="19D")
    return {"rec": rec, "p1": p1, "p2": p2}


# ---- recovery on simulated data with known truth (bands fixed before the first run) ------------------------

def test_recording_level_parameters_recovered_within_the_fixed_bands(scen):
    prm = scen["p1"].params
    sd = prm.posterior_sd
    print(f"\nrecording-level values from {prm.n_samples} post-burn-in samples (truth in brackets):"
          f"\n  g12 {prm.g12:.2f} [{TRUE['g12']}]  g21 {prm.g21:.2f} [{TRUE['g21']}]"
          f"\n  p1 {prm.p1:.1f} [{TRUE['p'][0]}]  p2 {prm.p2:.1f} [{TRUE['p'][1]}]"
          f"\n  m {prm.m:.3f} [{TRUE['m']}] (raw {prm.m_raw:.3f}, clipped fraction {prm.m_clipped_fraction:.3f})"
          f"\n  log_rho1 {prm.log_rho1:.3f} log_rho2 {prm.log_rho2:.3f} [{TRUE['log_rho']:.3f}]"
          f"\n  posterior SD g12 {sd['g12']:.2f} g21 {sd['g21']:.2f} p1 {sd['p1']:.1f} p2 {sd['p2']:.1f} m {sd['m']:.3f}"
          f"\n  filtered-gain estimator (null gate): {scen['p1'].gain_estimate}")
    assert not scen["p1"].recording_diverged and scen["p1"].n_diverged == 0
    assert BAND_G12[0] <= prm.g12 <= BAND_G12[1]
    assert abs(prm.g21 - TRUE["g21"]) < BAND_G21
    assert abs(prm.p1 - TRUE["p"][0]) < BAND_P and abs(prm.p2 - TRUE["p"][1]) < BAND_P
    assert abs(prm.m - TRUE["m"]) < BAND_M
    assert abs(prm.log_rho1 - TRUE["log_rho"]) < BAND_LOG_RHO and abs(prm.log_rho2 - TRUE["log_rho"]) < BAND_LOG_RHO


def test_recording_level_mean_is_the_mean_of_the_smoothed_trajectory_after_burn_in(scen, layout):
    """Recomputed here from the returned smoothed trajectories, sample 128 being the first one counted."""
    p1 = scen["p1"]
    cols = {name: layout.idx[name] - ss.N_NEURAL for name in ss.PARAM_NAMES_FULL}
    rows = np.concatenate([s.x_smooth_params[BURN:] for s in p1.segments])
    assert p1.params.n_samples == rows.shape[0] == sum(n * FS - BURN for n in SEG_SECONDS)
    for name, c in cols.items():
        expected = rows[:, c].clip(*CFG["priors"]["m_truncate"]).mean() if name == "m" else rows[:, c].mean()
        assert getattr(p1.params, name) == pytest.approx(expected, rel=1e-12, abs=1e-12), name
    assert p1.params.m_raw == pytest.approx(rows[:, cols["m"]].mean(), rel=1e-12)
    assert p1.params.rho1 == pytest.approx(math.exp(p1.params.log_rho1), rel=1e-14)
    ab = CFG["priors"]["AB_product"]
    assert p1.params.A1 == pytest.approx(math.sqrt(ab * p1.params.rho1), rel=1e-13)
    assert p1.params.B1 == pytest.approx(math.sqrt(ab / p1.params.rho1), rel=1e-13)


# ---- windows -----------------------------------------------------------------------------------------------

def test_window_count_shapes_burn_in_and_segment_containment(scen):
    rec, p2 = scen["rec"], scen["p2"]
    expected = sum((sec * FS) // WIN for sec in SEG_SECONDS)             # tails shorter than 2 s dropped
    assert len(p2.windows) == expected == 6 + 5 + 4 and not p2.recording_diverged and p2.n_diverged == 0
    for w in p2.windows:
        k = w.segment
        lo, hi = rec["starts"][k], rec["starts"][k] + rec["segments"][k].shape[1]
        assert lo <= w.start and w.start + WIN <= hi                     # never crosses a segment boundary
        assert (w.start - lo) % WIN == 0
        assert w.x_smooth.shape == (WIN - BURN, 12) and w.s_delayed.shape == (WIN - BURN, 2)
        assert w.x_smooth.dtype == np.float64 and np.isfinite(w.x_smooth).all()
    # no window contains a sample of a gap
    covered = np.zeros(rec["z_all"].shape[0], dtype=bool)
    for k in range(len(rec["segments"])):
        covered[rec["starts"][k]:rec["starts"][k] + rec["segments"][k].shape[1]] = True
    assert all(covered[w.start:w.start + WIN].all() for w in p2.windows)


def test_windows_hold_the_parameters_fixed_and_have_only_neural_states(scen):
    p2 = scen["p2"]
    assert p2.fixed_layout.n == 12 and p2.fixed_layout.names == ss.NEURAL_NAMES
    assert all(w.x_smooth.shape[1] == 12 for w in p2.windows)
    assert p2.fixed_layout.fixed_params == scen["p1"].params.fixed()


# ---- independent oracle for one window ---------------------------------------------------------------------

def indep_deriv12(X, delayed, prm):
    """Independent vectorized §7.1 right-hand side of the 12 neural states with FIXED parameters."""
    jr = CFG["jansen_rit"]
    a, b, C = jr["a"], jr["b"], jr["C"]
    C1, C2, C3, C4 = (C * jr[f"C{i}_multiplier"] for i in (1, 2, 3, 4))
    ab = CFG["priors"]["AB_product"]
    out = np.zeros_like(X)
    drive = (prm["g21"] * delayed[1], prm["g12"] * delayed[0])
    for j in range(2):
        y0, y1, y2, y3, y4, y5 = (X[:, 6 * j + i] for i in range(6))
        rho = math.exp(prm[f"log_rho{j + 1}"])
        A, B = math.sqrt(ab * rho), math.sqrt(ab / rho)
        out[:, 6 * j:6 * j + 6] = np.stack([
            y3, y4, y5,
            A * a * S(y1 - y2) - 2 * a * y3 - a * a * y0,
            A * a * (prm[f"p{j + 1}"] + C2 * S(C1 * y0) + drive[j]) - 2 * a * y4 - a * a * y1,
            B * b * C4 * S(C3 * y0) - 2 * b * y5 - b * b * y2], axis=1)
    return out


def indep_propagate(X, hist, Wm, prm):
    """4 Heun sub-steps of one observation step; hist is a plain list where hist[-1 - lag] is `lag` old.
    Appends the predicted-mean S after every sub-step."""
    dt = 1.0 / (FS * CFG["ukf"]["substeps_per_observation"])
    X = np.array(X, dtype=np.float64)
    for _ in range(CFG["ukf"]["substeps_per_observation"]):
        k1 = indep_deriv12(X, hist[-1 - DELAY], prm)
        k2 = indep_deriv12(X + dt * k1, hist[-DELAY], prm)
        X = X + 0.5 * dt * (k1 + k2)
        hist.append(S(Wm @ np.stack([X[:, 1] - X[:, 2], X[:, 7] - X[:, 8]], axis=1)))
    return X


def indep_hx(sig, m):
    mu = CFG["rescaling"]["mu_ref"]
    m = min(max(m, CFG["priors"]["m_truncate"][0]), CFG["priors"]["m_truncate"][1])
    dev = np.stack([sig[..., 1] - sig[..., 2], sig[..., 7] - sig[..., 8]], axis=-1) - mu
    return mu + np.stack([dev[..., 0] + m * dev[..., 1], m * dev[..., 0] + dev[..., 1]], axis=-1)


class WindowOracle(UnscentedKalmanFilter):
    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        self.sigmas_f[:] = indep_propagate(sigmas, self.hist, self.Wm, self.prm)


def test_window_reproduces_an_independent_12d_filter_and_smoother(scen):
    """filterpy's UKF and rts_smoother with an independent numpy propagation, mixing observation, noise
    matrices and initial state, run on the same 2-s window with the recording-level parameters fixed."""
    prm = scen["p1"].params.fixed()
    w = scen["p2"].windows[1]
    seg = scen["rec"]["segments"][w.segment]
    off = w.start - scen["rec"]["starts"][w.segment]
    z = seg[:, off:off + WIN].T
    sp = CFG["ukf"]["sigma_points"]
    f = WindowOracle(dim_x=12, dim_z=2, dt=1.0, hx=lambda s: indep_hx(s, prm["m"]), fx=lambda s, dt: s,
                     points=MerweScaledSigmaPoints(12, alpha=sp["alpha"], beta=sp["beta"], kappa=sp["kappa"]))
    var = np.tile(np.asarray(CFG["ukf"]["initial_state"]["neural_variance"]), 2)
    f.Q = Q_TEST * np.diag(var)
    f.R = CFG["ukf"]["observation_noise"]["R_fraction_of_rescaled_variance"] * CFG["rescaling"]["sigma_ref"] ** 2 * np.eye(2)
    node = ss.initial_node_state(CFG)
    f.x, f.P = np.tile(node, 2), np.diag(var)
    s0 = float(S(node[1] - node[2]))
    f.hist, f.prm = [np.array([s0, s0])] * (DELAY + 1), prm
    xs, Ps, after = [], [], []
    for t in range(WIN):
        f.predict()
        f.update(z[t])
        f.hist[-1] = S(np.array([f.x[1] - f.x[2], f.x[7] - f.x[8]]))
        xs.append(f.x.copy())
        Ps.append(f.P.copy())
        after.append(list(f.hist[-(DELAY + 1):]))
    xs, Ps = np.array(xs), np.array(Ps)
    # the smoother's propagation of step k -> k + 1 reads the buffer as it stood after step k
    calls = {"i": 0}
    n_sig = 25

    def fx(s, dt):
        k = (WIN - 2) - calls["i"] // n_sig
        calls["i"] += 1
        return indep_propagate(s[None], list(after[k]), np.array([1.0]), prm)[0]
    f.fx = fx
    xsm, Psm, _ = f.rts_smoother(xs, Ps)

    scale = np.abs(xsm).max(axis=0)
    np.testing.assert_allclose(w.x_smooth, xsm[BURN:], rtol=1e-8, atol=1e-10 * scale.max())
    # the delayed source S handed on: lag `delay` of the buffer after step t, forward means
    expected_s = np.array([after[t][0] for t in range(WIN)])[BURN:]
    np.testing.assert_allclose(w.s_delayed, expected_s, rtol=1e-9)
    # and it is the very same thing ukf.run_filter gives with the fixed layout
    flayout = ss.make_fixed_layout(CFG, prm)
    res = ukf.run_filter(z, CFG, flayout, Q_TEST, keep_cov=True)
    np.testing.assert_allclose(res.x, xs, rtol=1e-8, atol=1e-10 * np.abs(xs).max())


# ---- no leakage of observations, no crossing of segments ---------------------------------------------------

def test_a_window_never_reads_observations_outside_its_2s_span(monkeypatch):
    rng = np.random.default_rng(5)
    seg = make_recording(CFG, (7,), seed=12, g12=6.0, p=np.array([220.0, 240.0]))["segments"][0]      # 1792 samples
    prm = fake_params()
    seen = []
    real = ukf.run_filter

    def spy(z, *a, **k):
        seen.append(z.shape)
        return real(z, *a, **k)
    monkeypatch.setattr(ukf, "run_filter", spy)
    base = passes.run_pass2([seg], [0], prm, CFG, Q_TEST, filter_name="19D")
    assert seen == [(WIN, 2)] * 3 and len(base.windows) == 3
    other = seg.copy()
    other[:, :WIN] += rng.normal(size=(2, WIN)) * 3.0                     # windows 0 ...
    other[:, 2 * WIN:] += rng.normal(size=(2, seg.shape[1] - 2 * WIN)) * 3.0   # ... and 2 changed, window 1 not
    pert = passes.run_pass2([other], [0], prm, CFG, Q_TEST, filter_name="19D")
    assert np.array_equal(base.windows[1].x_smooth, pert.windows[1].x_smooth)
    assert np.array_equal(base.windows[1].s_delayed, pert.windows[1].s_delayed)
    assert not np.array_equal(base.windows[0].x_smooth, pert.windows[0].x_smooth)
    assert not np.array_equal(base.windows[2].x_smooth, pert.windows[2].x_smooth)
    inside = seg.copy()
    inside[:, WIN + 10] += 2.0                                            # one sample INSIDE window 1
    assert not np.array_equal(passes.run_pass2([inside], [0], prm, CFG, Q_TEST, filter_name="19D").windows[1].x_smooth,
                              base.windows[1].x_smooth)


def test_windows_do_not_cross_segment_gaps_and_tails_are_dropped():
    rec = make_recording(CFG, (5.2, 4.4), seed=13, g12=6.0, p=np.array([220.0, 240.0]), gap_seconds=2.0)
    segs, starts = rec["segments"], rec["starts"]
    lens = [s.shape[1] for s in segs]
    assert lens == [1331, 1126]                                           # neither is a multiple of 512
    prm = fake_params()
    out = passes.run_pass2(segs, starts, prm, CFG, Q_TEST, filter_name="19D")
    assert [(w.segment, w.window) for w in out.windows] == [(0, 0), (0, 1), (1, 0), (1, 1)]   # floor(1331/512)=2, floor(1126/512)=2
    assert [w.start for w in out.windows] == [starts[0], starts[0] + WIN, starts[1], starts[1] + WIN]
    for w in out.windows:
        assert w.start + WIN <= starts[w.segment] + lens[w.segment]
    # changing the OTHER segment leaves this segment's windows bit-identical
    altered = [segs[0], segs[1] + 1.5]
    out2 = passes.run_pass2(altered, starts, prm, CFG, Q_TEST, filter_name="19D")
    assert all(np.array_equal(a.x_smooth, b.x_smooth) for a, b in zip(out.windows[:2], out2.windows[:2]))
    assert not any(np.array_equal(a.x_smooth, b.x_smooth) for a, b in zip(out.windows[2:], out2.windows[2:]))
    assert passes.cut_windows(1331, WIN) == [0, 512] and passes.cut_windows(511, WIN) == []


def test_overlapping_or_unordered_segments_are_refused():
    seg = np.zeros((2, 600))
    with pytest.raises(passes.PassError):
        passes.run_pass2([seg, seg], [0, 300], fake_params(), CFG, Q_TEST, filter_name="19D")
    with pytest.raises(passes.PassError):
        passes.run_pass1([np.zeros((3, 600))], [0], CFG, Q_TEST, filter_name="19D")


# ---- carry across segments (IMP-030) -------------------------------------------------------------------------

def test_parameters_are_carried_across_segments_and_neural_states_reset(monkeypatch):
    rec = make_recording(CFG, (3.0, 3.0, 3.0), seed=14, g12=8.0, p=np.array([220.0, 240.0]), gap_seconds=2.5)
    layout = ss.make_layout(CFG)
    real = ukf.run_filter
    calls, outs = [], []

    def spy(z, cfg, lay, q, x0=None, P0=None, buffer=None, keep_cov=False):
        calls.append((x0.copy(), P0.copy()))
        out = real(z, cfg, lay, q, x0=x0, P0=P0, buffer=buffer, keep_cov=keep_cov)
        outs.append(out)
        return out
    monkeypatch.setattr(ukf, "run_filter", spy)
    res = passes.run_pass1(rec["segments"], rec["starts"], CFG, Q_TEST, filter_name="19D")
    N = ss.N_NEURAL
    prior_x, prior_P = ss.prior_mean(layout, CFG), ss.prior_cov(layout, CFG)
    walk = CFG["ukf"]["process_noise"]["parameter_random_walk_factor"]
    var_prior = np.diag(prior_P)[N:]
    assert np.array_equal(calls[0][0], prior_x) and np.array_equal(calls[0][1], prior_P)       # first: the prior
    for k in (1, 2):
        x0, P0 = calls[k]
        gap_s = (rec["starts"][k] - (rec["starts"][k - 1] + rec["segments"][k - 1].shape[1])) / FS
        assert gap_s == pytest.approx(2.5)
        prev = outs[k - 1]
        np.testing.assert_array_equal(x0[:N], prior_x[:N])                    # 12 neural means: the prior
        np.testing.assert_array_equal(P0[:N, :N], prior_P[:N, :N])            # their covariance: the prior
        assert not P0[:N, N:].any() and not P0[N:, :N].any()                  # no neural-parameter cross-covariance
        np.testing.assert_array_equal(x0[N:], prev.x[-1, N:])                 # parameter mean carried
        expected = prev.P[-1][N:, N:] + np.diag(walk * var_prior * gap_s)     # covariance carried + gap random walk
        np.testing.assert_allclose(P0[N:, N:], expected, rtol=1e-14, atol=0)
        assert res.segments[k].gap_s == pytest.approx(gap_s)
        np.testing.assert_allclose(res.segments[k].carry_in_var, np.diag(expected), rtol=1e-14)
        np.testing.assert_array_equal(res.segments[k].carry_in_mean, prev.x[-1, N:])


# ---- stub-based checks of the bookkeeping (no filtering cost) -------------------------------------------------

def install_stub(monkeypatch, layout, plans):
    """Replace run_filter / run_smoother by stubs. plans: one dict per segment with
    diverged (bool) and overrides {state name: value} for the smoothed AND filtered trajectories."""
    calls = {"i": 0, "x0": []}
    base = ss.prior_mean(layout, CFG)

    def fake_filter(z, cfg, lay, q, x0=None, P0=None, buffer=None, keep_cov=False):
        plan = plans[calls["i"]]
        calls["i"] += 1
        calls["x0"].append((x0.copy(), P0.copy()))
        T = z.shape[0]
        x = np.tile(base, (T, 1))
        for name, v in plan.get("values", {}).items():
            x[:, lay.idx[name]] = v
        P = np.tile(np.eye(lay.n) * 0.01, (T, 1, 1))
        return SimpleNamespace(x=x, P=P, diverged=plan["diverged"], divergence_reason="stub" if plan["diverged"] else None,
                               divergence_step=3 if plan["diverged"] else None, monitor={}, n_done=T)
    monkeypatch.setattr(ukf, "run_filter", fake_filter)
    monkeypatch.setattr(ukf, "run_smoother", lambda res, cfg: (res.x, res.P))
    return calls


def stub_segments(lengths, gap=300):
    starts, pos = [], 0
    for n in lengths:
        starts.append(pos)
        pos += n + gap
    return [np.zeros((2, n)) for n in lengths], starts


def test_ten_percent_rule_in_exact_integer_arithmetic():
    assert passes.exceeds_fraction(100, 1000, 0.1) is False               # exactly 10.0 percent: not diverged
    assert passes.exceeds_fraction(101, 1000, 0.1) is True
    assert passes.exceeds_fraction(0, 0, 0.1) is False
    assert passes.exceeds_fraction(100, 999, 0.1) is True                 # 100 > 99.9
    assert passes.exceeds_fraction(100, 1001, 0.1) is False               # 100 < 100.1
    assert passes.exceeds_fraction(99, 990, 0.1) is False                 # exactly 10.0 percent again


@pytest.mark.parametrize("lengths,diverged_index,expected", [
    ((900, 100), 1, False),           # 100 / 1000 = exactly 10.0 percent
    ((899, 100), 1, True),            # 100 / 999 > 10 percent
    ((901, 100), 1, False),           # 100 / 1001 < 10 percent
])
def test_recording_divergence_boundary_pass1(monkeypatch, layout, lengths, diverged_index, expected):
    plans = [{"diverged": i == diverged_index, "values": {"p1": 200.0 + 100 * i}} for i in range(len(lengths))]
    install_stub(monkeypatch, layout, plans)
    segs, starts = stub_segments(lengths)
    res = passes.run_pass1(segs, starts, CFG, Q_TEST, filter_name="19D")
    assert res.recording_diverged is expected
    assert res.n_diverged == lengths[diverged_index] and res.n_clean == sum(lengths)
    assert res.diverged_fraction == pytest.approx(lengths[diverged_index] / sum(lengths))


def test_diverged_segments_are_excluded_from_the_mean_and_carry_resumes_after_the_gap(monkeypatch, layout):
    plans = [{"diverged": False, "values": {"p1": 200.0}},
             {"diverged": True, "values": {"p1": 1000.0}},
             {"diverged": False, "values": {"p1": 300.0}}]
    calls = install_stub(monkeypatch, layout, plans)
    segs, starts = stub_segments((400, 300, 500), gap=200)
    res = passes.run_pass1(segs, starts, CFG, Q_TEST, filter_name="19D")
    burn = BURN
    # weighted by post-burn-in sample counts of the two GOOD segments only
    n1, n3 = 400 - burn, 500 - burn
    assert res.params.p1 == pytest.approx((200.0 * n1 + 300.0 * n3) / (n1 + n3), rel=1e-13)
    assert res.params.n_samples == n1 + n3 and res.segments[1].n_used == 0 and res.segments[1].diverged
    # segment 3 continues from the end of segment 1: the gap spans the dropped segment
    walk = CFG["ukf"]["process_noise"]["parameter_random_walk_factor"]
    gap_s = (starts[2] - (starts[0] + 400)) / FS
    assert res.segments[2].gap_s == pytest.approx(gap_s) and gap_s == pytest.approx((200 + 300 + 200) / FS)
    layout_var = np.diag(ss.prior_cov(layout, CFG))[ss.N_NEURAL:]
    np.testing.assert_allclose(res.segments[2].carry_in_var, 0.01 + walk * layout_var * gap_s, rtol=1e-13)


def test_recording_level_m_is_the_mean_of_the_clipped_values(monkeypatch, layout):
    lo, hi = CFG["priors"]["m_truncate"]
    plans = [{"diverged": False, "values": {"m": 0.9}}, {"diverged": False, "values": {"m": 0.3}},
             {"diverged": False, "values": {"m": -0.1}}]
    install_stub(monkeypatch, layout, plans)
    segs, starts = stub_segments((300, 300, 300))
    res = passes.run_pass1(segs, starts, CFG, Q_TEST, filter_name="19D")
    n = 300 - BURN
    assert res.params.m == pytest.approx((hi + 0.3 + lo) / 3, rel=1e-13)                 # (0.5 + 0.3 + 0.0) / 3
    assert res.params.m_raw == pytest.approx((0.9 + 0.3 - 0.1) / 3, rel=1e-13)          # 0.3667, whose clip is itself
    assert res.params.m != pytest.approx(min(max(res.params.m_raw, lo), hi))    # not the clip of the mean
    assert res.params.m_clipped_fraction == pytest.approx(2 / 3)


def test_reduction_layouts_follow_in_the_recording_level_values(monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["state"]["reduction_switches"].update(fix_EI_terms=True, tie_p1_p2=True, tie_g12_g21=True)
    lay = ss.make_layout(cfg)
    monkeypatch.setattr(ss, "make_layout", lambda c, include_gains=True: lay)
    install_stub(monkeypatch, lay, [{"diverged": False, "values": {"p1": 240.0, "g12": 7.0}}])
    segs, starts = stub_segments((400,))
    res = passes.run_pass1(segs, starts, cfg, Q_TEST, layout=lay, filter_name="19D")
    prm = res.params
    assert prm.p2 == prm.p1 == pytest.approx(240.0) and prm.g21 == prm.g12 == pytest.approx(7.0)
    assert prm.log_rho1 == prm.log_rho2 == pytest.approx(math.log(3.25 / 22))
    assert set(prm.posterior_sd) == {"p1", "g12", "m"}


def test_gains_less_layout_has_no_gain_estimate(monkeypatch):
    m1 = ss.make_layout(CFG, include_gains=False)
    install_stub(monkeypatch, m1, [{"diverged": False}])
    segs, starts = stub_segments((400,))
    res = passes.run_pass1(segs, starts, CFG, Q_TEST, layout=m1, filter_name="19D")
    assert res.gain_estimate is None and res.params.g12 == 0.0 and res.params.g21 == 0.0


def test_filtered_gain_estimator_skips_both_burn_ins(monkeypatch, layout):
    """Per-segment 0.5 s and the recording-level estimator burn-in (processed clean samples)."""
    est_burn = round(CFG["passes"]["estimator_burn_in_s"] * FS)
    n = est_burn + 3000
    plans = [{"diverged": False, "values": {"g12": 4.0}}, {"diverged": False, "values": {"g12": 10.0}}]
    install_stub(monkeypatch, layout, plans)
    segs, starts = stub_segments((n, 1000))
    res = passes.run_pass1(segs, starts, CFG, Q_TEST, filter_name="19D")
    # segment 1: samples est_burn.. counted; segment 2: all after its own 128 (cumulative count already past)
    n1, n2 = n - est_burn, 1000 - BURN
    assert res.gain_estimate["n"] == n1 + n2
    assert res.gain_estimate["g12"] == pytest.approx((4.0 * n1 + 10.0 * n2) / (n1 + n2), rel=1e-13)


# ---- pass 2 divergence bookkeeping ----------------------------------------------------------------------------

def test_pass2_drops_diverged_windows_and_applies_the_ten_percent_rule(monkeypatch):
    prm = fake_params()
    real_calls = {"i": 0}

    def fake_filter(z, cfg, lay, q, x0=None, P0=None, buffer=None, keep_cov=False):
        i = real_calls["i"]
        real_calls["i"] += 1
        bad = i in {0}
        T = z.shape[0]
        x = np.tile(ss.prior_mean(lay, cfg), (T, 1))
        return SimpleNamespace(x=x, P=np.tile(np.eye(lay.n), (T, 1, 1)), diverged=bad, divergence_reason="stub" if bad else None,
                               divergence_step=5 if bad else None, monitor={}, n_done=T,
                               snapshots=np.zeros((T, DELAY + 1, 2)))
    monkeypatch.setattr(ukf, "run_filter", fake_filter)
    monkeypatch.setattr(ukf, "run_smoother", lambda res, cfg: (res.x, res.P))
    monkeypatch.setattr(ukf, "buffer_snapshot", lambda buf: np.zeros((DELAY + 1, 2)))
    segs = [np.zeros((2, WIN * 10))]
    out = passes.run_pass2(segs, [0], prm, CFG, Q_TEST, filter_name="19D")                   # 1 of 10 windows = exactly 10.0 percent
    assert out.n_attempted == 10 * WIN and out.n_diverged == WIN and out.recording_diverged is False
    assert [w.diverged for w in out.windows] == [True] + [False] * 9 and len(out.kept) == 9
    real_calls["i"] = 0
    out = passes.run_pass2([np.zeros((2, WIN * 9))], [0], prm, CFG, Q_TEST, filter_name="19D")   # 1 of 9 > 10 percent
    assert out.recording_diverged is True


def test_recording_flag_is_the_or_of_both_passes_and_pass2_skips_diverged_segments(monkeypatch, layout):
    plans = [{"diverged": False, "values": {"p1": 220.0}}, {"diverged": True}]
    install_stub(monkeypatch, layout, plans)
    seen = []

    def fake_p2(segments, starts, params, cfg, q, skip_segments=(), filter_name=None, spec=None):
        seen.append(set(skip_segments))
        return SimpleNamespace(recording_diverged=True)
    monkeypatch.setattr(passes, "run_pass2", fake_p2)
    segs, starts = stub_segments((2000, 100))                              # 100 / 2100 < 10 percent: pass 1 ok
    out = passes.run_recording(segs, starts, CFG, Q_TEST, filter_name="19D")
    assert seen == [{1}] and out.pass1.recording_diverged is False and out.recording_diverged is True
    install_stub(monkeypatch, layout, [{"diverged": True}, {"diverged": False}])
    segs, starts = stub_segments((500, 1000))                              # 500 / 1500 = 33 percent
    seen.clear()
    out = passes.run_recording(segs, starts, CFG, Q_TEST, filter_name="19D")
    assert out.pass1.recording_diverged and out.pass2 is None and out.recording_diverged and seen == []


# ---- base model dy4/dt (§8.3): inputs only, no residual -------------------------------------------------------

def test_base_dy4dt_matches_the_model_equation():
    prm = fake_params()
    rng = np.random.default_rng(3)
    x = np.concatenate([np.tile(ss.initial_node_state(CFG), 2)[None] + rng.normal(size=(9, 12)) * 0.5], axis=0)
    s = rng.uniform(0.5, 4.0, size=(9, 2))
    got = passes.base_dy4dt(x, s, prm, CFG)
    jr = CFG["jansen_rit"]
    a, C = jr["a"], jr["C"]
    C1, C2 = C * jr["C1_multiplier"], C * jr["C2_multiplier"]
    for j, (p, A, g, src) in enumerate(((prm.p1, prm.A1, prm.g21, 1), (prm.p2, prm.A2, prm.g12, 0))):
        y0, y1, y4 = x[:, 6 * j], x[:, 6 * j + 1], x[:, 6 * j + 4]
        expected = A * a * (p + C2 * S(C1 * y0) + g * s[:, src]) - 2 * a * y4 - a * a * y1
        np.testing.assert_allclose(got[:, j], expected, rtol=1e-13)
        via_model = model.jr_rhs(x[:, 6 * j:6 * j + 6], p, g * s[:, src], A, prm.B1 if j == 0 else prm.B2, CFG)[:, 4]
        np.testing.assert_allclose(got[:, j], via_model, rtol=1e-12)
    assert np.abs(got[:, 0] - got[:, 1]).max() > 1.0                       # nodes differ: a swap would show


# ---- the shrinkage problem (§7.5): no gain is estimated inside a window ---------------------------------------

def test_window_output_does_not_depend_on_the_priors_of_the_parameters():
    seg = make_recording(CFG, (3,), seed=15, g12=8.0, p=np.array([220.0, 240.0]))["segments"][0][:, :WIN]
    prm = fake_params()
    changed = copy.deepcopy(CFG)
    changed["coupling"]["gain_prior_mean"] = 5.0
    changed["coupling"]["gain_prior_sd_factor_of_C2"] = 0.3
    changed["priors"]["p_sd"] = 120.0
    changed["priors"]["m_mean"] = 0.4
    changed["priors"]["log_rho_sd"] = 0.5
    base = passes.run_pass2([seg], [0], prm, CFG, Q_TEST, filter_name="19D")
    other = passes.run_pass2([seg], [0], prm, changed, Q_TEST, filter_name="19D")
    assert np.array_equal(base.windows[0].x_smooth, other.windows[0].x_smooth)
    assert np.array_equal(base.windows[0].s_delayed, other.windows[0].s_delayed)
    # control: a window that DID carry the gains in its state depends on those priors
    full, full2 = ss.make_layout(CFG), ss.make_layout(changed)
    a = ukf.run_filter(seg.T, CFG, full, Q_TEST)
    b = ukf.run_filter(seg.T, changed, full2, Q_TEST)
    assert not np.array_equal(a.x[:, :12], b.x[:, :12])


# ---- hygiene ------------------------------------------------------------------------------------------------------

def test_no_numeric_parameter_literals_no_preprocess_import_no_print():
    path = REPO_ROOT / "src" / "passes.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    allowed = {0, 1, 2, 0.5, 0.0, 1.0}
    bad = [(n.lineno, n.value) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
           and n.value not in allowed]
    assert not bad, bad
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
            imported |= {a.name for a in node.names if node.module == "src"}
    assert not imported & {"filterpy", "preprocess", "numba", "scipy"}, imported
    assert not re.search(r"\bprint\(", text)


def test_new_config_leaf_is_tagged():
    import yaml
    from src.config import DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["passes"]["estimator_burn_in_s"]
    assert raw["prov"] == "placeholder" and raw["value"] == 6 and raw["ref"]
