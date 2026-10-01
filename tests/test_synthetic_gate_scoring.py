"""E4 tests: the filter switch, series evaluation, tuning with any filter, the scoring of §9.1 to §9.3, the flags and
gate.json of src/synthetic_gate.py. Expected values are hand tables, closed forms or independent numpy / scipy routes."""
import copy
import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim_data  # noqa: E402
from src import passes, regression as R, synthetic_gate as sg, tuning, ukf, ukf_ext  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()
SRC = Path(sg.__file__)
Q = 1.0e-2
PRIOR = {"p1": 50.0, "p2": 50.0, "log_rho1": 0.2, "log_rho2": 0.2, "g12": 0.1 * 108.0, "g21": 0.1 * 108.0, "m": 0.15}


# ---- the switch --------------------------------------------------------------------------------------

def test_pilot_runs_every_option_and_a_full_run_uses_the_adopted_filter_and_refuses_when_unset():
    assert sg.gate_filters(CFG, True) == ["19D", "A", "B"]
    assert sg.gate_filters(CFG, False) == ["A"]                              # DEV-005 adopted A
    c = copy.deepcopy(CFG)
    c["g0"]["filter"] = None
    with pytest.raises(sg.GateError, match="unset"):
        sg.gate_filters(c, False)
    c["g0"]["filter"] = "B"
    assert sg.gate_filters(c, False) == ["B"] and sg.gate_filters(c, True) == ["19D", "A", "B"]
    c["g0"]["filter"] = "C"
    with pytest.raises(sg.GateError):
        sg.gate_filters(c, False)
    assert set(sg.FILTER_KINDS) == set(CFG["g0"]["filter_options"]) and CFG["g0"]["filter"] == "A"


def _series(seed=3, seconds=30.0, g=8.0):
    r = sim_data.make_recording(CFG, [seconds], seed, g12=g, g21=g, p=(220.0, 250.0))
    return SimpleNamespace(segments=r["segments"], starts=r["starts"], arm="positive", index=seed, stream="pilot",
                           level_index=2, gains=(g, g), truth=None,
                           operating_point={"distance": 0.5, "regime": "limit_cycle"})


def _est_tuple(p):
    return None if p is None else (p.g12, p.g21, p.m, p.p1, p.p2, p.log_rho1, p.log_rho2, tuple(sorted(p.posterior_sd.items())))


def test_19d_option_equals_plain_run_pass1_and_leaves_the_filters_alone():
    s = _series()
    gc = sg.g0_cfg(CFG)
    orig = (ukf.run_filter, ukf.run_smoother)
    plain = passes.run_pass1(s.segments, s.starts, gc, Q)
    with sg.filter_context(gc, "19D", s.segments):
        assert (ukf.run_filter, ukf.run_smoother) == orig
        opt = passes.run_pass1(s.segments, s.starts, gc, Q)
    assert _est_tuple(opt.params) == _est_tuple(plain.params) and opt.gain_estimate == plain.gain_estimate
    rec, _ = sg.evaluate_series(CFG, s, "19D", Q, diagnostic=False)
    assert rec["estimates"]["g12"] == plain.params.g12 and rec["estimates"]["g21_filt"] == plain.gain_estimate["g21"]
    with pytest.raises(sg.GateError):
        sg.filter_context(gc, "Z", s.segments)


@pytest.mark.parametrize("name", ["A", "B"])
def test_a_and_b_equal_a_direct_ukf_ext_call_and_pass_2_windows_are_14d(name, monkeypatch):
    s = _series(seed=4)
    gc = sg.g0_cfg(CFG)
    orig = (ukf.run_filter, ukf.run_smoother)
    spec = ukf_ext.spec_for(name, s.segments, gc)
    seen = []
    real = ukf_ext.run_filter_ext
    monkeypatch.setattr(ukf_ext, "run_filter_ext", lambda z, cfg, layout, q, sp, **kw: (
        seen.append((layout.n, sp.nx)), real(z, cfg, layout, q, sp, **kw))[1])
    with sg.filter_context(gc, name, s.segments):
        res = passes.run_recording(s.segments, s.starts, gc, Q)
    assert (ukf.run_filter, ukf.run_smoother) == orig                         # restored on exit
    # pass 1 (19 neural + parameter states) and pass 2 (12 neural states) both carry the two noise states: 21-D and 14-D
    assert seen[0] == (19, 2) and len(seen) == 1 + len(res.pass2.windows) and all(v == (12, 2) for v in seen[1:])
    assert res.pass2.windows[0].x_smooth.shape == (384, 12)
    # the pass-1 filtered parameter trajectory equals a direct call of the extended filter on the first segment
    layout = res.pass1.layout
    direct = real(np.ascontiguousarray(s.segments[0].T), gc, layout, Q, spec, keep_cov=False)
    gi = list(layout.names).index("g12")
    np.testing.assert_allclose(res.pass1.segments[0].x_filt_params[:, list(layout.names[12:]).index("g12")],
                               direct.x[:, gi], rtol=1e-12, atol=1e-12)


def test_evaluate_series_record_fields_and_diagnostic_label():
    s = _series(seed=5)
    rec, p2 = sg.evaluate_series(CFG, s, "A", Q)
    assert rec["filter"] == "A" and rec["level_index"] == 2 and rec["g_true"] == 8.0 and rec["z_distance"] == 0.5
    assert rec["diverged"] is False and p2 is not None and len(p2.windows) > 0
    est = rec["estimates"]
    assert {"g12", "g21", "g12_filt", "g21_filt", "m", "rho1", "rho2", "posterior_sd"} <= set(est)
    assert rec["stability"]["n_runs"] == 1 + len(p2.windows) and sg.stability_ok(rec["stability"])
    d = rec["diagnostic_flag_off"]
    assert d["diagnostic_only"] is True and d["estimates"]["g12"] is not None
    assert rec["has_term"] is None and rec["nrmse"] is None and rec["linear_floor"] is None


def test_a_diverged_series_has_no_estimates_but_keeps_the_diagnostic(monkeypatch):
    s = _series(seed=6)
    c = copy.deepcopy(CFG)
    c["ukf"]["divergence"]["state_sd_multiple"] = 1.0e-3                       # the rule fires at once
    rec, p2 = sg.evaluate_series(c, s, "19D", Q)
    assert rec["diverged"] and rec["estimates"] is None and p2 is None
    assert rec["diagnostic_flag_off"]["diagnostic_only"] and rec["diagnostic_flag_off"]["estimates"] is not None
    assert sg.pooled_gain_errors(rec) == [float("inf"), float("inf")]


def test_a_diverged_recording_with_usable_parameters_still_reports_no_estimates():
    """A healthy segment gives pass-1 parameters, a +100 mV segment diverges (IMP-047): 40% of the clean samples are in
    a diverged segment, so the recording is diverged by the 10% rule and the series has no estimates (failed series)."""
    r = sim_data.make_recording(CFG, [30.0, 20.0], 9, g12=8.0, g21=8.0, p=(220.0, 250.0))
    segs = [r["segments"][0], r["segments"][1] + 100.0]
    s = SimpleNamespace(segments=segs, starts=r["starts"], arm="positive", index=9, stream="pilot", level_index=2,
                        gains=(8.0, 8.0), truth=None, operating_point={"distance": 0.1, "regime": "limit_cycle"})
    gc = sg.g0_cfg(CFG)
    direct = passes.run_pass1(s.segments, s.starts, gc, Q)
    assert direct.params is not None and direct.recording_diverged and 0.1 < direct.diverged_fraction < 1.0
    rec, p2 = sg.evaluate_series(CFG, s, "19D", Q)
    assert rec["diverged"] is True and rec["estimates"] is None and p2 is None
    assert rec["n_segments_dropped_pass1"] == 1 and rec["diverged_fraction_pass1"] == direct.diverged_fraction
    assert rec["diagnostic_flag_off"]["estimates"] is not None


# ---- tuning with any filter --------------------------------------------------------------------------

def test_tune_g0_q_option_uses_the_gates_worker_cache_and_the_select_rule(tmp_path, monkeypatch):
    s1, s2 = _series(seed=7, seconds=20.0), _series(seed=8, seconds=20.0)
    ser = [SimpleNamespace(index=i, segments=s.segments, starts=s.starts) for i, s in enumerate((s1, s2))]
    c = copy.deepcopy(CFG)
    c["ukf"]["qr_rule"]["min_recordings"] = 2
    calls = []
    real = sg._tune_worker
    monkeypatch.setattr(sg, "_tune_worker", lambda p: (calls.append(p[4]), real(p))[1])
    res = sg.tune_g0_q_option(c, ser, "B", cache_dir=tmp_path, n_jobs=1, min_recordings=2)
    assert calls == ["B"] * 16 and len(res.table) == 8
    calls.clear()
    res2 = sg.tune_g0_q_option(c, ser, "B", cache_dir=tmp_path, n_jobs=1, min_recordings=2)
    assert calls == [] and res2.q == res.q                                    # fully cached
    assert res.q is None or any(np.isclose(res.q, g) for g in tuning.q_grid(c))
    assert [r["id"] for r in res.recordings] == ["tuning_0", "tuning_1"]
    sg.tune_g0_q_option(c, ser, "A", cache_dir=tmp_path, n_jobs=1, min_recordings=2)
    assert calls == ["A"] * 16                                                # another filter is another cache entry


# ---- pooled errors, positive verdict -----------------------------------------------------------------

def _rec(level, g_true, e12=0.0, e21=0.0, *, diverged=False, nrmse=0.1, floor=0.2, sd=None, f12=None, f21=None):
    est = {"g12": g_true * (1 + e12), "g21": g_true * (1 + e21), "g12_filt": g_true * (1 + (e12 if f12 is None else f12)),
           "g21_filt": g_true * (1 + (e21 if f21 is None else f21)), "m": 0.25,
           "posterior_sd": dict(sd) if sd is not None else {k: 0.2 * v for k, v in PRIOR.items()}}
    return {"level_index": level, "g_true": g_true, "diverged": diverged, "estimates": None if diverged else est,
            "nrmse": nrmse, "linear_floor": floor, "stability": {"n_runs": 1, "n_negative_eig_steps": 0, "n_nan_inf": 0,
                                                                  "n_linalg_divergences": 0, "n_jitter_fallbacks": 0,
                                                                  "min_eig_overall": 1e-6}}


G = (2.16, 5.4, 10.8, 27.0)


def _positives(per_level=3, **kw):
    return [_rec(lv, G[lv], **kw) for lv in range(4) for _ in range(per_level)]


def test_pooled_gain_errors_use_both_directions_and_inf_when_diverged():
    r = _rec(2, 10.8, 0.10, -0.30)
    np.testing.assert_allclose(sg.pooled_gain_errors(r), [0.10, 0.30])
    np.testing.assert_allclose(sg.pooled_gain_errors(_rec(2, 10.8, 0.1, 0.2, f12=0.5, f21=0.6), ("g12_filt", "g21_filt")),
                               [0.5, 0.6])
    assert sg.pooled_gain_errors(_rec(2, 10.8, diverged=True)) == [float("inf")] * 2


def test_positive_verdict_levels_two_and_up_median_rules_and_floor():
    recs = _positives()
    for r in recs[:3]:                                                          # level 1 (the floor): hopeless, ignored
        r["estimates"]["g12"] = r["estimates"]["g21"] = 100.0
        r["nrmse"] = 5.0
    v = sg.positive_verdict(CFG, recs)
    assert v["pass"] is True and v["levels"][0]["detection_floor"] and v["levels"][0]["ok"] is None
    assert all(v["levels"][k]["required"] and v["levels"][k]["ok"] for k in (1, 2, 3))
    assert v["levels"][2]["linear_floor_median"] == pytest.approx(0.2)
    # one level above the 15% median gain error fails the verdict
    bad = _positives()
    for r in bad:
        if r["level_index"] == 2:
            r["estimates"]["g12"] = r["estimates"]["g21"] = 10.8 * 1.2
    assert sg.positive_verdict(CFG, bad)["pass"] is False
    # exactly 15% passes, 0.16 fails (<=)
    edge = _positives()
    for r in edge:
        r["estimates"]["g12"] = r["estimates"]["g21"] = r["g_true"] * 1.15
    assert sg.positive_verdict(CFG, edge)["levels"][1]["gain_ok"] is True
    edge2 = _positives(e12=0.16, e21=0.16)
    assert sg.positive_verdict(CFG, edge2)["levels"][1]["gain_ok"] is False
    # the gain criterion is a MEDIAN: two good series and one bad one pass although the mean error is 30%
    skew = _positives()
    third = [r for r in skew if r["level_index"] == 2][2]
    third["estimates"]["g12"] = third["estimates"]["g21"] = 10.8 * 1.9
    assert sg.positive_verdict(CFG, skew)["pass"] is True


def test_positive_verdict_is_a_median_nrmse_pending_and_diverged_as_inf():
    recs = _positives()
    # median NRMSE 0.2 although one of three series per level is terrible; 0.26 median fails
    for lv in (1, 2, 3):
        rs = [r for r in recs if r["level_index"] == lv]
        rs[0]["nrmse"], rs[1]["nrmse"], rs[2]["nrmse"] = 0.2, 0.2, 9.0
    assert sg.positive_verdict(CFG, recs)["pass"] is True
    for r in recs:
        if r["level_index"] == 3:
            r["nrmse"] = 0.26
    assert sg.positive_verdict(CFG, recs)["pass"] is False
    pend = _positives(nrmse=None)
    assert sg.positive_verdict(CFG, pend)["pass"] is None
    pend[3]["estimates"]["g12"] = pend[3]["estimates"]["g21"] = 100.0           # gains fail already: False beats pending
    pend[4]["estimates"]["g12"] = pend[4]["estimates"]["g21"] = 100.0
    pend[5]["estimates"]["g12"] = pend[5]["estimates"]["g21"] = 100.0
    assert sg.positive_verdict(CFG, pend)["pass"] is False
    dv = _positives()
    for r in [x for x in dv if x["level_index"] == 1][:2]:                       # 2 of 3 diverged: the median is inf
        r.update(diverged=True, estimates=None)
    out = sg.positive_verdict(CFG, dv)
    assert out["pass"] is False and out["levels"][1]["n_diverged"] == 2 and out["levels"][1]["gain_error_median"] == float("inf")


# ---- nulls --------------------------------------------------------------------------------------------

def _null(g12f, g21f, has_term=False, diverged=False):
    return {"diverged": diverged, "has_term": has_term,
            "estimates": None if diverged else {"g12_filt": g12f, "g21_filt": g21f}}


def test_null_arm_verdict_strict_delta_term_diverged_and_pending():
    d = 1.08
    assert sg.delta(CFG) == pytest.approx(d)
    ok = [_null(0.5, -1.0), _null(1.07, 0.0)]
    v = sg.null_arm_verdict(CFG, ok)
    assert v["pass"] is True and v["n_fail"] == 0 and v["status"] == ["pass", "pass"]
    assert sg.null_arm_verdict(CFG, [_null(1.08, 0.0)])["pass"] is False                 # |g| = delta is outside
    assert sg.null_arm_verdict(CFG, [_null(0.1, 0.1, has_term=True)])["pass"] is False
    assert sg.null_arm_verdict(CFG, [_null(0, 0, diverged=True)])["status"] == ["fail"]
    pend = sg.null_arm_verdict(CFG, [_null(0.1, 0.1, has_term=None)])
    assert pend["pass"] is None and pend["n_pending"] == 1
    assert sg.null_arm_verdict(CFG, [_null(0.1, 0.1, has_term=None), _null(5.0, 0.0, has_term=None)])["pass"] is False


def test_upper_bound_matches_the_exact_binomial_inversion():
    from scipy.optimize import brentq
    from scipy.stats import binom
    assert sg.upper_bound_95(0, 20) == pytest.approx(1 - 0.05 ** (1 / 20)) == pytest.approx(0.1391, abs=1e-4)
    assert sg.upper_bound_95(0, 60) == pytest.approx(0.0487, abs=1e-4)
    for k, n in ((1, 20), (2, 60)):
        exact = brentq(lambda p: binom.cdf(k, n, p) - 0.05, 1e-9, 1 - 1e-9)
        assert sg.upper_bound_95(k, n) == pytest.approx(exact, rel=1e-6)
    assert sg.upper_bound_95(20, 20) == 1.0


# ---- contraction, reduction, stability ----------------------------------------------------------------

def _with_sd(**override):
    sd = {k: 0.2 * v for k, v in PRIOR.items()}
    sd.update(override)
    return sd


def test_contraction_median_per_level_boundary_and_the_never_dropped_quantities():
    recs = _positives(sd=_with_sd(g12=0.5 * PRIOR["g12"]))                     # contraction exactly 0.5: passes
    v = sg.contraction_verdict(CFG, recs)
    assert v["pass"] is True and v["table"]["g12"]["per_level"] == {1: pytest.approx(0.5), 2: pytest.approx(0.5), 3: pytest.approx(0.5)}
    assert set(v["table"]) == {"p1", "p2", "log_rho1", "log_rho2", "g12", "g21", "m"}
    assert sg.contraction_verdict(CFG, _positives(sd=_with_sd(g12=0.51 * PRIOR["g12"])))["table"]["g12"]["ok"] is False
    # m: not identifiable, never reduced
    vm = sg.contraction_verdict(CFG, _positives(sd=_with_sd(m=0.6 * PRIOR["m"])))
    assert vm["failed"] == ["m"] and vm["not_identifiable"] == ["m"] and vm["reduction_required"] is None
    assert all(abs(x - 0.4) < 1e-12 for x in vm["m_contraction"].values())
    # a p quantity fails: the next unapplied step of the section 7.4 order is named
    vp = sg.contraction_verdict(CFG, _positives(sd=_with_sd(p1=0.8 * PRIOR["p1"])))
    assert vp["failed"] == ["p1"] and vp["reduction_required"] == "fix_EI_terms" and vp["pass"] is False
    c = copy.deepcopy(CFG)
    c["state"]["reduction_switches"]["fix_EI_terms"] = True
    assert sg.contraction_verdict(c, _positives(sd=_with_sd(p1=0.8 * PRIOR["p1"])))["reduction_required"] == "tie_p1_p2"
    c["state"]["reduction_switches"]["tie_p1_p2"] = c["state"]["reduction_switches"]["tie_g12_g21"] = True
    assert sg.next_reduction_step(c) is None


def test_contraction_is_a_median_over_series_diverged_count_as_zero_and_level_one_is_ignored():
    recs = _positives()
    for r in recs:                                                              # the weakest level may fail freely
        if r["level_index"] == 0:
            r["estimates"]["posterior_sd"] = _with_sd(g12=2.0 * PRIOR["g12"], m=2.0 * PRIOR["m"])
    assert sg.contraction_verdict(CFG, recs)["pass"] is True
    one = [r for r in recs if r["level_index"] == 2]
    one[0].update(diverged=True, estimates=None)                                # 1 of 3 diverged: the median still passes
    assert sg.contraction_verdict(CFG, recs)["pass"] is True
    one[1].update(diverged=True, estimates=None)                                # 2 of 3: median 0
    v = sg.contraction_verdict(CFG, recs)
    assert v["pass"] is False and v["table"]["g12"]["per_level"][2] == 0.0


def _mon(neg=0, nan=False, jit=0, min_eig=1e-6, reason=None):
    return {"min_eig_overall": min_eig, "n_negative_eig_steps": neg, "nan_inf_seen": nan, "n_jitter_fallbacks": jit,
            "divergence_reason": reason}


def test_stability_summary_and_verdict():
    s = sg.stability_summary([_mon(), _mon(jit=2, min_eig=3e-7)])
    assert s == {"n_runs": 2, "n_negative_eig_steps": 0, "n_nan_inf": 0, "n_linalg_divergences": 0,
                 "n_jitter_fallbacks": 2, "min_eig_overall": 3e-7} and sg.stability_ok(s)
    assert not sg.stability_ok(sg.stability_summary([_mon(neg=1, min_eig=-1e-9)]))
    assert not sg.stability_ok(sg.stability_summary([_mon(nan=True)]))
    assert not sg.stability_ok(sg.stability_summary([_mon(reason="linalg_error: Matrix is not positive definite")]))
    assert sg.stability_ok(sg.stability_summary([_mon(reason="state_beyond_sd_multiple")]))      # the state rule is no stability event
    recs = [_rec(1, 5.4), _rec(2, 10.8)]
    assert sg.stability_verdict(recs)["pass"] is True and sg.stability_verdict(recs)["n_runs"] == 2
    recs[1]["stability"] = dict(recs[1]["stability"], n_negative_eig_steps=3)
    assert sg.stability_verdict(recs)["pass"] is False


def test_gate_flags_truth_table_against_an_independent_rule():
    names = ("positive", "null_A", "null_B", "contraction", "stability", "preproc_bias", "preproc_null")
    hard, soft = {"null_A", "null_B", "stability", "preproc_null"}, {"positive", "contraction", "preproc_bias"}
    for combo in itertools.product((True, False, None), repeat=len(names)):
        v = dict(zip(names, combo))
        for pilot in (False, True):
            f = sg.gate_flags(pilot, v)
            would = any(v[k] is False for k in hard)
            assert f["would_hard_stop"] == would and f["hard_stop"] == (would and not pilot)
            assert f["low_confidence"] == any(v[k] is False for k in soft)
            assert f["pending"] == sorted(k for k in names if v[k] is None) and f["complete"] == (not f["pending"])


# ---- profile, parsimony, truth measures ---------------------------------------------------------------

def test_gain_profile_by_hand():
    recs = [_rec(1, 5.4, 0.0, 0.0), _rec(1, 5.4, 0.2, -0.2), _rec(1, 5.4, diverged=True)]
    p = sg.gain_profile(CFG, recs)[1]
    vals = [5.4, 5.4, 5.4 * 1.2, 5.4 * 0.8]
    assert p["g_true"] == 5.4 and p["n"] == 3 and p["n_diverged"] == 1
    assert p["g_median"] == pytest.approx(np.median(vals)) and p["g_q25"] == pytest.approx(np.percentile(vals, 25))
    assert p["posterior_sd_median"] == pytest.approx(0.2 * PRIOR["g12"]) and p["contraction_median"] == pytest.approx(0.8)


def test_choose_parsimony_lowest_median_ties_to_the_larger_penalty():
    r = sg.choose_parsimony(CFG, {0.003: [0.5, 0.2, 0.9], 0.01: [0.3, 0.3, 0.3], 0.03: [0.4, 0.1, 0.35]})
    assert r["penalty"] == 0.01 and r["median_nrmse"]["0.003"] == 0.5 and r["median_nrmse"]["0.03"] == 0.35
    tie = sg.choose_parsimony(CFG, {0.003: [0.3, 0.3], 0.01: [0.3, 0.3], 0.03: [0.9, 0.9]})
    assert tie["penalty"] == 0.01                                                # the larger of the tied penalties
    inf = sg.choose_parsimony(CFG, {0.003: [float("inf")] * 3, 0.01: [0.3, float("inf"), 0.2], 0.03: [float("inf")] * 3})
    assert inf["penalty"] == 0.01


def _truth(n, f):
    rng = np.random.default_rng(5)
    ut, us, ss_ = rng.standard_normal((3, n, 2))
    return {"u_tgt": ut, "u_src": us, "s_src": ss_, "basis": ut * us, "planted": f(ut, us, ss_)}


def test_linear_floor_closed_forms():
    n = 200_000
    assert sg.linear_floor_nrmse(_truth(n, lambda a, b, c: 3.0 * a - 2.0 * b + 0.5 * c + 4.0)) < 1e-9
    assert sg.linear_floor_nrmse(_truth(n, lambda a, b, c: a * b)) == pytest.approx(1.0, abs=0.01)
    # x y + 2 x: variance 1 + 4 = 5, the linear part explains 4: floor sqrt(1 / 5)
    assert sg.linear_floor_nrmse(_truth(n, lambda a, b, c: a * b + 2.0 * a)) == pytest.approx(np.sqrt(0.2), abs=0.01)


def test_equation_nrmse_exact_expansion_and_the_constant_equation():
    n = 5000
    rng = np.random.default_rng(6)
    truth = {"u_tgt": rng.standard_normal((n, 2)) * 2 + 5, "u_src": rng.standard_normal((n, 2)) * 3 - 1,
             "s_src": rng.standard_normal((n, 2))}
    c = 1.7
    truth["planted"] = c * truth["u_tgt"] * truth["u_src"]
    X_raw = np.concatenate([np.column_stack([truth["u_tgt"][:, j], truth["u_src"][:, j], truth["s_src"][:, j]]) for j in (0, 1)])
    y = np.concatenate([truth["planted"][:, 0], truth["planted"][:, 1]])
    z = R.fit_zscore([R.Rows(X_raw, y, np.zeros(2 * n, int), np.zeros(2 * n, np.int64), np.zeros(2 * n, int),
                             np.array(["s"] * (2 * n)))])
    m1, m2, s1, s2 = z.mean_X[0], z.mean_X[1], z.sd_X[0], z.sd_X[1]
    # planted = c (m1 + s1 zt)(m2 + s2 zs) = c [m1 m2 + m1 s2 zs + m2 s1 zt + s1 s2 zt zs]; target z-scored: (. - mean_y) / sd_y
    a0, a1, a2, a3 = (c * m1 * m2 - z.mean_y) / z.sd_y, c * m2 * s1 / z.sd_y, c * m1 * s2 / z.sd_y, c * s1 * s2 / z.sd_y
    a0, a1, a2, a3 = (float(v) for v in (a0, a1, a2, a3))
    eq = f"({a0!r}) + ({a1!r}) * u_tgt + ({a2!r}) * u_src + ({a3!r}) * u_tgt_u_src"
    assert sg.equation_nrmse(eq, z, truth) < 1e-9
    assert sg.equation_nrmse("0.0", z, truth) == pytest.approx(1.0, abs=1e-9)           # the mean: RMS error = SD(planted)
    assert sg.equation_nrmse("u_tgt_u_src", z, truth) > 0.05                              # the bare product alone is not it


# ---- the per-option report ----------------------------------------------------------------------------

def test_option_report_counts_side_by_side_and_z_distances():
    pos = [dict(_rec(1, 5.4), arm="positive", index=0, z_distance=0.5, diagnostic_flag_off={"diagnostic_only": True, "estimates": {"g12": 99.0}}),
           dict(_rec(1, 5.4, diverged=True), arm="positive", index=1, z_distance=2.0, diagnostic_flag_off={"diagnostic_only": True, "estimates": {"g12": 55.0}}),
           dict(_rec(2, 10.8, floor=0.4), arm="positive", index=2, z_distance=1.0)]
    nul = [dict(_null(0.1, 0.1), level_index=None, index=0, z_distance=0.3, linear_floor=None, g_true=0.0)]
    rep = sg.option_report(CFG, "B", {"positive": pos, "null_A": nul})
    a = rep["arms"]["positive"]
    assert rep["filter"] == "B" and a["n"] == 3 and a["n_dropped_by_standard_rule"] == 1
    assert a["levels"]["1"] == {"n": 2, "n_dropped_by_standard_rule": 1, "linear_floor_nrmse_median": 0.2}
    assert a["levels"]["2"]["linear_floor_nrmse_median"] == 0.4
    assert a["z_distance"]["median"] == 1.0 and a["z_distance"]["max"] == 2.0 and a["z_distance"]["per_series"] == [0.5, 2.0, 1.0]
    sbs = a["side_by_side"]
    assert sbs[0]["rule_on"]["g12"] == pytest.approx(5.4) and sbs[0]["diagnostic_flag_off"]["diagnostic_only"]
    assert sbs[1]["rule_on"] is None and sbs[1]["diagnostic_flag_off"]["estimates"]["g12"] == 55.0
    assert "never a G0 verdict" in a["diagnostic_note"]
    assert rep["arms"]["null_A"]["levels"]["None"]["n"] == 1


# ---- gate.json ----------------------------------------------------------------------------------------

VERDICTS_OK = {"positive": True, "null_A": True, "null_B": True, "contraction": True, "stability": True,
               "preproc_bias": True, "preproc_null": True}


def _doc(root, pilot, **verdict_changes):
    v = dict(VERDICTS_OK, **verdict_changes)
    return sg.build_gate_document(CFG, root, pilot=pilot, filter_name="A", verdicts=v,
                                  sections={"positive": {"pass": v["positive"]}, "gain_profile": {}},
                                  q={"q": 1e-3, "in_band": False}, parsimony={"penalty": 0.01})


def test_gate_document_schema_and_the_file_roundtrip_through_check_gate(tmp_path):
    root = tmp_path
    doc = _doc(root, False, positive=False)                                    # positive failure: low_confidence only
    assert doc["schema_version"] == 1 and doc["pilot"] is False and doc["filter"] == "A"
    assert doc["hard_stop"] is False and doc["low_confidence"] is True and doc["would_hard_stop"] is False
    assert doc["delta"] == pytest.approx(1.08) and doc["reasons"] == ["positive failed"] and doc["complete"] is True
    assert {"git_commit", "git_dirty", "config_sha256", "code_sha256"} <= set(doc["provenance"])
    path = sg.write_gate(CFG, root, doc, sg.gate_path(CFG, root, False))
    assert path == root / "outputs" / "gate.json"
    got = tuning.check_gate(CFG, root)
    assert got["low_confidence"] is True and got["hard_stop"] is False and got["verdicts"] == doc["verdicts"]
    assert got["q"] == {"q": 1e-3, "in_band": False}
    # a hard stop is refused by the Q/R gate check
    stop = _doc(root, False, null_B=False)
    assert stop["hard_stop"] is True and stop["reasons"] == ["null_B failed"]
    sg.write_gate(CFG, root, stop, path)
    with pytest.raises(tuning.GateError):
        tuning.check_gate(CFG, root)


def test_pilot_never_writes_outputs_gate_json_and_never_stops(tmp_path):
    root = tmp_path
    doc = _doc(root, True, null_A=False, stability=False)
    assert doc["pilot"] is True and doc["hard_stop"] is False and doc["would_hard_stop"] is True
    p = sg.gate_path(CFG, root, True, "A")
    assert p == root / "results" / "pilot" / "gate_A.json" and p != root / "outputs" / "gate.json"
    sg.write_gate(CFG, root, doc, p)
    assert json.loads(p.read_text(encoding="utf-8"))["would_hard_stop"] is True
    assert not (root / "outputs" / "gate.json").exists()
    with pytest.raises(sg.GateError, match="pilot"):
        sg.write_gate(CFG, root, doc, root / "outputs" / "gate.json")
    with pytest.raises(sg.GateError, match="full gate"):
        sg.write_gate(CFG, root, _doc(root, False), p)
    with pytest.raises(sg.GateError):
        sg.gate_path(CFG, root, True)                                            # a pilot gate is per option
    assert not (root / "outputs" / "gate.json").exists()


def test_gate_json_is_valid_json_with_non_finite_numbers_as_strings(tmp_path):
    doc = _doc(tmp_path, True)
    doc["positive"] = {"levels": {1: {"gain_error_median": float("inf"), "x": np.float64(0.5), "n": np.int64(3),
                                      "ok": np.bool_(True), "arr": np.array([1.0, np.nan]), "t": (1, 2)}}}
    p = sg.write_gate(CFG, tmp_path, doc, sg.gate_path(CFG, tmp_path, True, "A"))
    got = json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(f"non-standard constant {c}"))
    lv = got["positive"]["levels"]["1"]
    assert lv["gain_error_median"] == "inf" and lv["x"] == 0.5 and lv["n"] == 3 and lv["ok"] is True
    assert lv["arr"] == [1.0, "nan"] and lv["t"] == [1, 2]


# ---- hygiene ------------------------------------------------------------------------------------------

def test_source_and_config_for_e4():
    text = SRC.read_text(encoding="utf-8")
    assert not re.search(r"(^|\s)print\(", text)
    assert not re.search(r"np\.random\.(seed|rand|randn|normal|uniform|randint|choice|shuffle|permutation)\b", text)
    assert not re.search(r'filter_name\s*==\s*"(A|B)"\s*and', text)
    g0 = CFG["g0"]
    assert g0["pass"]["from_level_index"] == 1 and g0["gate_schema_version"] == 1
    assert g0["pass"]["median_nrmse_max"] == 0.25 and g0["pass"]["median_gain_rel_error_max"] == 0.15
    assert g0["pass"]["contraction_min"] == 0.5 and g0["pass"]["null_false_positives_allowed"] == 0
