"""G2 tests, part 2: wPLI and imaginary coherency, the AAFT surrogate test, the diagnostics guard and driver
(src/robustness.py; §14; IMP-082). Expected values: an explicit DFT of one epoch, quadrature and zero-lag constructions with
known answers, exact rank properties of AAFT, a coherence measure from scipy, hand p-values; never read back from the code
under test."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src import robustness as rb, tuning
from src.config import load_config
from sim_data import make_recording

CFG = load_config()
REPO = Path(__file__).resolve().parent.parent
FS = CFG["preprocessing"]["observation_fs_hz"]          # 256
EP = 512                                                # 2-s epochs


def band_cfg(lo, hi):
    cfg = copy.deepcopy(CFG)
    cfg["statistics"]["wpli_bands_hz"] = [[lo, hi]]
    return cfg


def tone_pair(n_epochs, f0=10.0, phase=np.pi / 2, noise=0.0, seed=0):
    t = np.arange(n_epochs * EP) / FS
    rng = np.random.default_rng(seed)
    x1 = np.cos(2 * np.pi * f0 * t) + noise * rng.normal(size=t.size)
    x2 = np.cos(2 * np.pi * f0 * t - phase) + noise * rng.normal(size=t.size)
    return np.stack([x1, x2])


# ---- wPLI and imaginary coherency -------------------------------------------------------------------------------------

def test_one_epoch_matches_an_explicit_dft_of_the_two_channels():
    rng = np.random.default_rng(2)
    seg = rng.normal(size=(2, EP))
    k = 20                                                                              # bin 20 = 10 Hz at 0.5 Hz per bin
    n = np.arange(EP)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / (EP - 1))                                    # numpy's symmetric Hann
    X = [np.sum(w * (c - c.mean()) * np.exp(-2j * np.pi * k * n / EP)) for c in seg]
    S = X[0] * np.conj(X[1])
    want_icoh = S.imag / np.sqrt(abs(X[0]) ** 2 * abs(X[1]) ** 2)
    out = rb.connectivity([seg], band_cfg(10.0, 10.0))
    b = out["bands"]["10.0-10.0"]
    assert out["n_epochs"] == 1 and b["n_bins"] == 1
    assert b["icoh"] == pytest.approx(want_icoh, rel=1e-9) and b["wpli"] == pytest.approx(1.0)      # one epoch: wPLI is 1


def test_quadrature_coupling_gives_wpli_one_and_signed_imaginary_coherency_one():
    seg = tone_pair(20)
    b = rb.connectivity([seg], band_cfg(9.5, 10.5))["bands"]["9.5-10.5"]
    assert b["wpli"] > 0.999 and b["icoh"] > 0.999 and b["abs_icoh"] > 0.999            # x2 lags x1 by 90 degrees: Im S > 0
    swapped = rb.connectivity([seg[::-1]], band_cfg(9.5, 10.5))["bands"]["9.5-10.5"]
    assert swapped["icoh"] < -0.999 and swapped["abs_icoh"] > 0.999 and swapped["wpli"] > 0.999   # signed, wPLI is not


def test_zero_lag_shared_signal_gives_no_imaginary_part():
    rng = np.random.default_rng(6)
    common = rng.normal(size=EP * 200)
    seg = np.stack([common + 0.5 * rng.normal(size=common.size), common + 0.5 * rng.normal(size=common.size)])
    for lo, hi in ((8.0, 13.0), (1.0, 45.0)):
        b = rb.connectivity([seg], band_cfg(lo, hi))["bands"][f"{lo}-{hi}"]
        assert b["wpli"] < 0.15 and b["abs_icoh"] < 0.1 and b["abs_icoh_binwise"] < 0.2
    in_phase = rb.connectivity([tone_pair(100, phase=0.0, noise=0.05)], band_cfg(9.5, 10.5))["bands"]["9.5-10.5"]
    assert in_phase["abs_icoh"] < 0.05 and in_phase["wpli"] < 0.3                     # zero lag: the real part carries it


def test_independent_noise_has_small_wpli_and_epochs_are_counted_per_segment():
    rng = np.random.default_rng(1)
    segs = [rng.normal(size=(2, 1100)), rng.normal(size=(2, 600)), rng.normal(size=(2, 300))]
    out = rb.connectivity(segs, band_cfg(1.0, 45.0))
    assert out["n_epochs"] == 2 + 1 + 0                                                # partial epochs are dropped
    big = rb.connectivity([rng.normal(size=(2, EP * 300))], band_cfg(1.0, 45.0))["bands"]["1.0-45.0"]
    assert big["wpli"] < 0.12 and big["abs_icoh"] < 0.05
    with pytest.raises(rb.RobustnessError):
        rb.connectivity([np.zeros((2, 100))], CFG)
    with pytest.raises(rb.RobustnessError):
        rb.connectivity([np.zeros((3, EP))], CFG)


def test_bands_use_both_edges_and_the_half_hertz_grid():
    out = rb.connectivity([np.random.default_rng(0).normal(size=(2, EP * 4))], CFG)["bands"]
    assert out["8-13"]["n_bins"] == 11 and out["1-45"]["n_bins"] == 89                 # 0.5-Hz bins, edges included


def test_the_summary_is_the_median_and_iqr_over_recordings_with_no_test():
    per = {f"r{i}": {"bands": {"8-13": {"wpli": v, "abs_icoh": v / 2, "abs_icoh_binwise": v}}} for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.9])}
    s = rb.connectivity_summary(per, [25, 50, 75])["8-13"]
    assert s["wpli"]["median"] == pytest.approx(0.3) and s["wpli"]["iqr"] == [pytest.approx(0.2), pytest.approx(0.4)]
    assert s["abs_icoh"]["median"] == pytest.approx(0.15) and s["wpli"]["n"] == 5 and "p" not in s["wpli"]


# ---- AAFT ----------------------------------------------------------------------------------------------------------

def coherence(a, b):
    from scipy import signal
    f, c = signal.coherence(a, b, fs=FS, nperseg=EP)
    return float(np.mean(c[(f >= 1) & (f <= 45)]))


def _correlated_pair(n=EP * 40, seed=3):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    return s + 0.3 * rng.normal(size=n), s + 0.3 * rng.normal(size=n)


def test_a_surrogate_has_exactly_the_original_values_in_a_different_order_and_is_deterministic():
    x = np.random.default_rng(1).normal(size=999) ** 3                                  # odd length, heavy tails
    a = rb.aaft_surrogate(x, np.random.default_rng(7))
    assert np.array_equal(np.sort(a), np.sort(x)) and not np.array_equal(a, x) and a.shape == x.shape
    assert np.array_equal(a, rb.aaft_surrogate(x, np.random.default_rng(7)))
    assert not np.array_equal(a, rb.aaft_surrogate(x, np.random.default_rng(8)))
    even = rb.aaft_surrogate(x[:998], np.random.default_rng(7))
    assert np.array_equal(np.sort(even), np.sort(x[:998]))


def test_a_surrogate_keeps_the_power_spectrum_of_a_linear_series_approximately():
    rng = np.random.default_rng(4)
    e = rng.normal(size=EP * 60)
    x = np.zeros_like(e)
    for t in range(2, e.size):                                                          # AR(2), a resonance near 12 Hz
        x[t] = 1.6 * x[t - 1] - 0.85 * x[t - 2] + e[t]
    from scipy import signal
    f, p0 = signal.welch(x, fs=FS, nperseg=EP)
    _, p1 = signal.welch(rb.aaft_surrogate(x, np.random.default_rng(9)), fs=FS, nperseg=EP)
    assert np.corrcoef(np.log(p0[1:100]), np.log(p1[1:100]))[0, 1] > 0.95
    assert f[np.argmax(p1)] == pytest.approx(f[np.argmax(p0)], abs=2.0)


def test_per_channel_surrogates_destroy_the_cross_channel_coupling():
    a, b = _correlated_pair()
    assert coherence(a, b) > 0.8
    rng = np.random.default_rng(5)
    sa, sb = rb.aaft_segments([np.stack([a, b])], rng)[0]
    assert coherence(sa, sb) < 0.15                                                     # independent phases per channel
    assert np.array_equal(np.sort(sa), np.sort(a)) and np.array_equal(np.sort(sb), np.sort(b))


def test_segments_and_starts_are_kept_and_each_segment_gets_its_own_draw():
    rng = np.random.default_rng(0)
    segs = [rng.normal(size=(2, 600)), rng.normal(size=(2, 800))]
    out = rb.aaft_segments(segs, np.random.default_rng(1))
    assert [o.shape for o in out] == [(2, 600), (2, 800)]
    for o, s in zip(out, segs):
        assert not np.array_equal(o, s) and np.array_equal(np.sort(o, axis=1), np.sort(s, axis=1))


def test_aaft_p_is_one_plus_k_over_n_plus_one_and_a_diverged_surrogate_counts_as_exceeding():
    p, k, nd = rb.aaft_p(3.0, [1.0, 2.0, 3.0, 4.0], 4)
    assert (k, nd) == (2, 0) and p == pytest.approx(3 / 5)                              # 3 and 4 are at least as large
    p, k, nd = rb.aaft_p(2.0, [None, 1.0], 2)
    assert (k, nd) == (1, 1) and p == pytest.approx(2 / 3)
    p, k, nd = rb.aaft_p(9.0, [1.0] * 50, 50)
    assert k == 0 and p == pytest.approx(1 / 51) and p < 0.05                           # never exactly 0
    assert rb.aaft_p(9.0, [1.0] * 48 + [9.0, 10.0], 50)[0] == pytest.approx(3 / 51) and rb.aaft_p(9.0, [1.0] * 48 + [9.0, 10.0], 50)[0] > 0.05
    assert rb.aaft_p(9.0, [1.0] * 49 + [9.0], 50)[0] == pytest.approx(2 / 51)           # k = 1 is still significant


def test_the_statistic_is_the_larger_absolute_gain_and_none_when_there_is_no_estimate():
    assert rb.aaft_statistic({"g12": -4.0, "g21": 3.0, "recording_diverged": False}) == 4.0
    assert rb.aaft_statistic({"g12": 1.0, "g21": -2.5, "recording_diverged": False}) == 2.5
    assert rb.aaft_statistic({"g12": 1.0, "g21": 2.0, "recording_diverged": True}) is None
    assert rb.aaft_statistic({"g12": None, "g21": None, "recording_diverged": False}) is None


# ---- AAFT driver ---------------------------------------------------------------------------------------------------

def small_cfg(n_rec=2, n_sur=4, per_task=2):
    cfg = copy.deepcopy(CFG)
    cfg["statistics"]["aaft"].update(n_recordings=n_rec, n_surrogates=n_sur, surrogates_per_task=per_task)
    return cfg


def _recs(n=4, seed=300):
    data = {}
    for k in range(n):
        rec = make_recording(CFG, (6.7, 3.5), seed + k, g12=3.0 + k, g21=5.0, m=0.3, p=np.array([220.0, 250.0]),
                             gap_seconds=1.0, burn_seconds=2.0)
        data[f"sub-{k + 1:03d}"] = {"reason": None, "segments": rec["segments"], "starts": rec["starts"], "key": f"k{k}",
                                    "session": "ses-t1", "vigilance": {"ratios": [[1.0], [1.0]]}}
    return data


@pytest.fixture(scope="module")
def recs():
    return _recs()


def _plan(data):
    return rb.SubjectPlan("pilot", tuple(sorted(data)), frozenset(data), False)


def test_the_worker_uses_the_documented_rng_layout(recs):
    cfg = small_cfg()
    r = recs["sub-002"]
    got = rb._aaft_worker(("surrogates", 3, [2], r["segments"], r["starts"], cfg, 42, "A"))[0]
    surr = rb.aaft_segments(r["segments"], np.random.default_rng([42, 3, 2]))
    want = rb.aaft_statistic(rb.session_gain(surr, r["starts"], cfg))
    assert got["index"] == 2 and got["stat"] == pytest.approx(want, rel=1e-12)
    other = rb._aaft_worker(("surrogates", 3, [2], r["segments"], r["starts"], cfg, 43, "A"))[0]
    assert other["stat"] != got["stat"]


def test_the_recordings_are_drawn_by_seed_and_the_p_values_follow_the_stored_statistics(recs):
    cfg = small_cfg(n_rec=2, n_sur=4)
    out = rb.run_aaft(cfg, _plan(recs), lambda s: recs[s], n_jobs=1)
    ids = sorted(recs)
    order = [ids[i] for i in np.random.default_rng([42, 4]).permutation(4)]               # the documented draw, written out here
    assert out["draw_order"] == order and [r["id"] for r in out["recordings"]] == sorted(order[:2])
    assert out["n_recordings"] == 2 and out["n_surrogates"] == 4 and out["replaced"] == []
    for r in out["recordings"]:
        rec = recs[r["id"]]
        real = rb.aaft_statistic(rb.session_gain(rec["segments"], rec["starts"], cfg))
        stats = []
        for k in range(4):
            surr = rb.aaft_segments(rec["segments"], np.random.default_rng([42, r["recording_index"], k]))
            stats.append(rb.aaft_statistic(rb.session_gain(surr, rec["starts"], cfg)))
        k_hand = sum(s >= real for s in stats)
        assert r["real_statistic"] == pytest.approx(real, rel=1e-12) and r["k_at_least_real"] == k_hand
        assert r["p"] == pytest.approx((1 + k_hand) / 5) and r["significant"] == (r["p"] < 0.05)
    assert out["n_significant"] == sum(r["significant"] for r in out["recordings"])


def test_an_excluded_or_diverged_real_recording_is_replaced_by_the_next_draw_and_counted(recs, monkeypatch):
    cfg = small_cfg(n_rec=2, n_sur=2)
    ids = sorted(recs)
    order = [ids[i] for i in np.random.default_rng([42, 4]).permutation(4)]
    data = dict(recs)
    data[order[0]] = {"reason": "excluded: insufficient_clean_data"}
    real = rb.session_gain
    bad = recs[order[1]]["segments"]

    def fake(segments, starts, cfg_, residual=None, filter_name=None):
        out = real(segments, starts, cfg_, residual=residual, filter_name=filter_name)
        if segments is bad:
            out["recording_diverged"] = True
        return out
    monkeypatch.setattr(rb, "session_gain", fake)
    out = rb.run_aaft(cfg, _plan(recs), lambda s: data[s], n_jobs=1)
    assert [r["id"] for r in out["replaced"]] == [order[0], order[1]]
    assert sorted(r["id"] for r in out["recordings"]) == sorted(order[2:4]) and out["n_recordings"] == 2


def test_fewer_eligible_recordings_than_requested_is_reported_not_padded(recs):
    cfg = small_cfg(n_rec=6, n_sur=2)
    out = rb.run_aaft(cfg, _plan(recs), lambda s: recs[s], n_jobs=1)
    assert out["n_recordings"] == 4 and out["n_recordings_requested"] == 6


def test_a_diverged_surrogate_counts_as_exceeding_and_is_reported(recs, monkeypatch):
    cfg = small_cfg(n_rec=1, n_sur=3, per_task=3)
    real = rb.session_gain
    target = recs[sorted(recs)[0]]["segments"]

    def fake(segments, starts, cfg_, residual=None, filter_name=None):
        out = real(segments, starts, cfg_, residual=residual, filter_name=filter_name)
        if segments is not target and not any(segments is r["segments"] for r in recs.values()):
            out["recording_diverged"] = True                                            # every surrogate diverges
        return out
    monkeypatch.setattr(rb, "session_gain", fake)
    out = rb.run_aaft(cfg, _plan({k: v for k, v in recs.items()}), lambda s: recs[s], n_jobs=1)
    row = out["recordings"][0]
    assert row["n_surrogates_diverged"] == 3 and row["k_at_least_real"] == 3 and row["p"] == pytest.approx(4 / 4) and not row["significant"]


def test_the_pool_gives_the_same_numbers_as_the_inline_run_and_the_cache_is_reused(recs, tmp_path):
    cfg = small_cfg(n_rec=2, n_sur=4, per_task=2)
    inline = rb.run_aaft(cfg, _plan(recs), lambda s: recs[s], n_jobs=1)
    pooled = rb.run_aaft(cfg, _plan(recs), lambda s: recs[s], n_jobs=2, cache_dir=tmp_path)
    assert inline == pooled
    assert len(list(tmp_path.glob("*.json"))) == 2
    again = rb.run_aaft(cfg, _plan(recs), lambda s: recs[s], n_jobs=1, cache_dir=tmp_path)
    assert again == inline


# ---- guard ---------------------------------------------------------------------------------------------------------

def _world(monkeypatch, n_train=14, n_test=6):
    import main as main_mod
    train = [f"sub-{i:03d}" for i in range(1, n_train + 1)]
    test = [f"sub-{i:03d}" for i in range(n_train + 1, n_train + n_test + 1)]
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(train))
    return main_mod, train, test


def test_the_pilot_diagnostics_plan_is_the_pilot_subjects_only(monkeypatch, temp_root):
    main_mod, train, test = _world(monkeypatch)
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test list loaded")))
    pilot = train[:12]
    plan = rb.resolve_diag_subjects(CFG, temp_root, 42, pilot=True, pilot_ids=pilot)
    assert plan.mode == "pilot" and not plan.allow_all and plan.ids == tuple(sorted(pilot))
    with pytest.raises(rb.GuardError):
        rb.resolve_diag_subjects(CFG, temp_root, 42, pilot=True, pilot_ids=train[:11] + [test[0]])
    with pytest.raises(rb.GuardError):
        rb.resolve_diag_subjects(CFG, temp_root, 42, pilot=True, confirmatory=True, pilot_ids=pilot)


def test_the_confirmatory_diagnostics_plan_needs_flag_and_gate_and_is_the_test_partition(monkeypatch, temp_root):
    main_mod, train, test = _world(monkeypatch)
    monkeypatch.setattr(main_mod, "test_subjects", lambda seed, root=None, cfg=None: list(test))
    with pytest.raises(rb.GuardError, match="confirmatory=True"):
        rb.resolve_diag_subjects(CFG, temp_root, 42, pilot=False)
    with pytest.raises(tuning.GateError):
        rb.resolve_diag_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    (temp_root / "outputs").mkdir()
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": True}))
    with pytest.raises(tuning.GateError):
        rb.resolve_diag_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": False}))
    plan = rb.resolve_diag_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    assert plan.allow_all and plan.ids == tuple(sorted(test))


# ---- end to end ----------------------------------------------------------------------------------------------------

def test_e2e_diagnostics_pilot_reads_only_pilot_t1_and_writes_to_the_pilot_folder(monkeypatch, tmp_path):
    import shutil
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    main_mod, train, test = _world(monkeypatch)
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test list loaded")))
    pilot = train[:4]
    data = _recs(4, seed=400)
    renamed = {pilot[i]: v for i, v in enumerate(data.values())}
    asked = []

    def loader(s, session=None):
        asked.append((s, session))
        return renamed[s]
    cfg = small_cfg(n_rec=2, n_sur=3, per_task=3)
    out = rb.run_diagnostics(cfg, tmp_path, 42, pilot=True, loader=loader, pilot_ids=pilot, n_jobs=1)
    d = out["doc"]
    assert d["pilot"] and d["mechanics_only"] and out["path"].parent == tmp_path / "results" / "pilot"
    assert not (tmp_path / "results" / "diagnostics_42.json").exists()
    assert {s for s, _ in asked} <= set(pilot) and {x for _, x in asked} == {None}
    assert set(d["connectivity"]["per_recording"]) == set(pilot) and set(d["connectivity"]["summary"]) == {"8-13", "1-45"}
    assert d["aaft"]["n_recordings"] == 2 and d["aaft"]["p_rule"] == "one_plus_k_over_n_plus_one"
    for sid, rec in d["connectivity"]["per_recording"].items():
        want = rb.connectivity(renamed[sid]["segments"], cfg)
        assert rec["bands"]["8-13"]["wpli"] == pytest.approx(want["bands"]["8-13"]["wpli"], rel=1e-12)
    again = rb.run_diagnostics(cfg, tmp_path, 42, pilot=True, loader=loader, pilot_ids=pilot, n_jobs=1)
    assert again["status"] == "unchanged"
    plan = rb.resolve_diag_subjects(cfg, tmp_path, 42, pilot=True, pilot_ids=pilot)
    with pytest.raises(rb.GuardError):
        rb.guarded_loader(loader, plan)("sub-099")
