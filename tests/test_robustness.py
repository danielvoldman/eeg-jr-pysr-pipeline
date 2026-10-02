"""G1 tests: C1 scoring, matched sets, M3 loss, bootstrap, Holm, the guard and the writer (src/robustness.py; §10.2, §11.4,
§12, §15; IMP-078). Expected values are hand arithmetic from the literals 128 (0.5 s) and 1536 (6 s) at 256 Hz, hand-built
index matrices, textbook Holm numbers; never read back from src/robustness.py."""
import ast
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import baseline, passes, regression, robustness as rb, tuning
from src.config import load_config
from sim_data import make_recording

CFG = load_config()
REPO = Path(__file__).resolve().parent.parent
LENGTHS = [1600, 400, 400]
# hand mask: segment 1 -> samples 1536..1599 (cumulative 6 s burn-in), segments 2 and 3 -> samples 128.. (0.5 s per segment)
N_SCORED = [64, 272, 272]


def hand_mask():
    out = []
    for n, first in zip(LENGTHS, (1536, 128, 128)):
        m = np.zeros(n, dtype=bool)
        m[first:] = True
        out.append(m)
    return out


def const_err(values):
    """Per-sample squared error: one constant per segment (so a subject mean is hand-computable)."""
    return np.concatenate([np.full(n, v) for n, v in zip(LENGTHS, values)])


def base_entry(m0=(5.0, 5.0, 5.0), m0b=(4.0, 4.0, 4.0), mask=None):
    return {"lengths": np.array(LENGTHS), "mask": np.concatenate(hand_mask()) if mask is None else mask,
            "M0": const_err(m0), "M0b": const_err(m0b), "starts": np.array([0, 2000, 3000])}


def run(vals, seg_div=(False, False, False), rec_div=False):
    return {"sq_err": [None if d else np.full(n, v) for n, v, d in zip(LENGTHS, vals, seg_div)],
            "seg_diverged": list(seg_div), "seg_reason": [("state_beyond_sd_multiple" if d else None) for d in seg_div],
            "seg_n": list(LENGTHS), "recording_diverged": rec_div, "n_clean": sum(LENGTHS),
            "n_diverged": sum(n for n, d in zip(LENGTHS, seg_div) if d)}


def wmean(vals, weights):
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights)


# ---- per-subject scores, masks, matched sets ----------------------------------------------------------------------

def test_hand_mask_is_the_shared_scoring_mask():
    for a, b in zip(hand_mask(), passes.scoring_mask(LENGTHS, CFG)):
        assert np.array_equal(a, b)


def test_matched_subject_score_is_the_masked_mean_per_variant():
    runs = {"M1": run((1.0, 2.0, 3.0)), "M2": run((2.0, 4.0, 6.0)), "M3": run((1.5, 3.0, 5.0))}
    row = rb.subject_scores("s", CFG, base_entry(m0=(7.0, 8.0, 9.0), m0b=(1.0, 1.0, 1.0)), runs, "scored")
    assert row["status"] == "matched" and row["n_scored"] == 608 and row["n_mask"] == 608
    sc = row["scores"]
    assert sc["M1"] == pytest.approx(wmean((1.0, 2.0, 3.0), N_SCORED))
    assert sc["M2"] == pytest.approx((64 * 2 + 272 * 4 + 272 * 6) / 608)
    assert sc["M3"] == pytest.approx(wmean((1.5, 3.0, 5.0), N_SCORED))
    assert sc["M0"] == pytest.approx(wmean((7.0, 8.0, 9.0), N_SCORED)) and sc["M0b"] == pytest.approx(1.0)


def test_a_diverged_segment_in_any_variant_removes_those_samples_from_every_variant():
    runs = {"M1": run((1.0, 2.0, 3.0)), "M2": run((2.0, 4.0, 6.0)),
            "M3": run((1.5, 3.0, 5.0), seg_div=(False, True, False))}
    row = rb.subject_scores("s", CFG, base_entry(m0=(7.0, 8.0, 9.0)), runs, "scored")
    assert row["status"] == "matched" and row["n_scored"] == 64 + 272 and row["n_segments_dropped"] == 1
    w = (64, 0, 272)
    assert row["scores"]["M2"] == pytest.approx(wmean((2.0, 4.0, 6.0), w))      # M2 did not diverge, still loses segment 2
    assert row["scores"]["M0"] == pytest.approx(wmean((7.0, 8.0, 9.0), w))
    assert row["scores"]["M1"] == pytest.approx(wmean((1.0, 2.0, 3.0), w))


def test_recording_level_divergence_in_any_of_m1_to_m3_excludes_the_subject_with_its_reason():
    for bad in ("M1", "M2", "M3"):
        runs = {"M1": run((1, 1, 1)), "M2": run((2, 2, 2)), "M3": run((3, 3, 3))}
        runs[bad] = run((1, 1, 1), rec_div=True)
        row = rb.subject_scores("s", CFG, base_entry(), runs, "scored")
        if bad == "M3":
            assert row["status"] == "m3_loss"        # M3 only: kept for the sensitivity analysis, not the primary set
        else:
            assert row["status"] == "excluded" and row["reason"] == f"recording_diverged_in_{bad}"


def test_m2_and_m3_both_diverged_is_an_exclusion_not_an_m3_loss():
    runs = {"M1": run((1, 1, 1)), "M2": run((2, 2, 2), rec_div=True), "M3": run((3, 3, 3), rec_div=True)}
    row = rb.subject_scores("s", CFG, base_entry(), runs, "scored")
    assert row["status"] == "excluded" and row["reason"] == "recording_diverged_in_M2_M3"


def test_m3_loss_takes_the_worst_of_m0_m1_m2_and_scores_on_the_m0_m1_m2_samples():
    # M3 diverged at recording level, M2 did not: M3 error := max(M0, M1, M2) of the subject
    runs = {"M1": run((3.0, 3.0, 3.0)), "M2": run((4.0, 4.0, 4.0)), "M3": run((0.0, 0.0, 0.0), rec_div=True)}
    row = rb.subject_scores("s", CFG, base_entry(m0=(5.0, 5.0, 5.0)), runs, "scored")
    assert row["status"] == "m3_loss" and row["scores"]["M3"] == pytest.approx(5.0)        # M0 is the worst
    row = rb.subject_scores("s", CFG, base_entry(m0=(1.0, 1.0, 1.0)), runs, "scored")
    assert row["scores"]["M3"] == pytest.approx(4.0)                                        # M2 is the worst
    runs["M1"] = run((9.0, 9.0, 9.0))
    row = rb.subject_scores("s", CFG, base_entry(m0=(1.0, 1.0, 1.0)), runs, "scored")
    assert row["scores"]["M3"] == pytest.approx(9.0)                                        # M1 is the worst


def test_m3_loss_when_m1_also_diverged_uses_m0_and_m2_only():
    runs = {"M1": run((9.0, 9.0, 9.0), rec_div=True), "M2": run((4.0, 4.0, 4.0)), "M3": run((0, 0, 0), rec_div=True)}
    row = rb.subject_scores("s", CFG, base_entry(m0=(1.0, 1.0, 1.0)), runs, "scored")
    assert row["status"] == "m3_loss" and "M1" not in row["scores"] and row["scores"]["M3"] == pytest.approx(4.0)


def test_m3_loss_samples_ignore_the_segments_only_m3_lost():
    runs = {"M1": run((1, 1, 1)), "M2": run((2, 2, 2)), "M3": run((0, 0, 0), seg_div=(False, True, False), rec_div=True)}
    row = rb.subject_scores("s", CFG, base_entry(), runs, "scored")
    assert row["n_scored"] == 608            # M3's own segment loss does not shrink the M0/M1/M2 sample set


def test_no_scored_samples_is_an_explicit_exclusion():
    runs = {"M1": run((1, 1, 1), seg_div=(False, True, True)), "M2": run((2, 2, 2)), "M3": run((3, 3, 3))}
    runs["M1"]["seg_diverged"] = [True, True, True]
    row = rb.subject_scores("s", CFG, base_entry(), runs, "scored")
    assert row["status"] == "excluded" and row["reason"] == "no_scored_samples"


def test_no_term_makes_m3_exactly_m2_and_absent_has_no_m3():
    runs = {"M1": run((1.0, 2.0, 3.0)), "M2": run((2.0, 4.0, 6.0))}
    row = rb.subject_scores("s", CFG, base_entry(), runs, "no_term")
    assert row["scores"]["M3"] == row["scores"]["M2"]
    row = rb.subject_scores("s", CFG, base_entry(), runs, "absent")
    assert "M3" not in row["scores"]


def test_baseline_mask_or_lengths_that_differ_from_the_recording_are_refused():
    runs = {"M1": run((1, 1, 1)), "M2": run((2, 2, 2))}
    bad = hand_mask()
    bad[1][:] = False
    with pytest.raises(rb.GuardError, match="mask"):
        rb.subject_scores("s", CFG, base_entry(mask=np.concatenate(bad)), runs, "absent")
    runs["M2"]["seg_n"] = [1600, 400, 401]
    with pytest.raises(rb.GuardError, match="lengths"):
        rb.subject_scores("s", CFG, base_entry(), runs, "absent")


def test_nan_in_a_scored_baseline_sample_is_not_skipped():
    base = base_entry()
    base["M0"] = base["M0"].copy()
    base["M0"][1800] = np.nan      # segment 2, sample 200: inside the scored region
    with pytest.raises(passes.PassError):
        rb.subject_scores("s", CFG, base, {"M1": run((1, 1, 1)), "M2": run((2, 2, 2))}, "absent")


# ---- divergence report ---------------------------------------------------------------------------------------------

def test_divergence_counts_per_variant_and_the_m3_only_share():
    runs_by = {"a": {"M1": run((1, 1, 1)), "M2": run((2, 2, 2)), "M3": run((3, 3, 3))},
               "b": {"M1": run((1, 1, 1)), "M2": run((2, 2, 2)), "M3": run((3, 3, 3), seg_div=(False, True, False), rec_div=True)},
               "c": {"M1": run((1, 1, 1), rec_div=True), "M2": run((2, 2, 2)), "M3": run((3, 3, 3))}}
    rows = [rb.subject_scores(k, CFG, base_entry(), v, "scored") for k, v in runs_by.items()]
    rep = rb.divergence_report(rows, runs_by, "scored", CFG)
    pv = rep["per_variant"]
    assert pv["M3"]["recordings_diverged"] == 1 and pv["M1"]["recordings_diverged"] == 1 and pv["M2"]["recordings_diverged"] == 0
    assert pv["M3"]["segments_diverged"] == 1 and pv["M3"]["samples_diverged"] == 400 and pv["M3"]["segments"] == 9
    assert rep["m3_only_divergence"]["n"] == 1 and rep["m3_only_divergence"]["subjects"] == ["b"]
    assert rep["m3_only_divergence"]["fraction"] == pytest.approx(1 / 3) and rep["m3_only_divergence"]["exceeds_limit_state_in_limitations"]
    rm = rep["removed"]
    assert (rm["matched"], rm["recording_level_divergence"], rm["no_scored_samples"], rm["m3_loss_kept_in_sensitivity"]) == (1, 1, 0, 1)


def test_m3_only_share_at_exactly_five_percent_is_not_flagged():
    runs_by = {f"s{i}": {"M1": run((1, 1, 1)), "M2": run((2, 2, 2)), "M3": run((3, 3, 3))} for i in range(20)}
    runs_by["s0"]["M3"] = run((3, 3, 3), rec_div=True)
    rows = [rb.subject_scores(k, CFG, base_entry(), v, "scored") for k, v in runs_by.items()]
    rep = rb.divergence_report(rows, runs_by, "scored", CFG)
    assert rep["m3_only_divergence"]["n"] == 1 and rep["m3_only_divergence"]["fraction"] == pytest.approx(0.05)
    assert rep["m3_only_divergence"]["exceeds_limit_state_in_limitations"] is False


# ---- bootstrap ----------------------------------------------------------------------------------------------------

def test_index_matrix_shape_range_determinism_and_dependence_on_split_seed_and_n():
    a = rb.draw_index_matrix(33, 500, 42, 42)
    assert a.shape == (500, 33) and a.dtype == np.int64 and a.min() >= 0 and a.max() <= 32
    assert np.array_equal(a, rb.draw_index_matrix(33, 500, 42, 42))
    assert not np.array_equal(a, rb.draw_index_matrix(33, 500, 42, 43))
    assert not np.array_equal(a[:, :32], rb.draw_index_matrix(32, 500, 42, 42))
    assert not np.array_equal(a, rb.draw_index_matrix(33, 500, 41, 42))
    assert np.array_equal(a, np.random.default_rng([42, 42, 33]).integers(0, 33, size=(500, 33)))
    with pytest.raises(rb.RobustnessError):
        rb.draw_index_matrix(0, 10, 42, 42)


def test_resample_means_hand_cases_plain_and_clustered():
    d = np.array([1.0, 2.0, 3.0])
    idx = np.array([[0, 0, 0], [0, 1, 2], [2, 2, 1]])
    np.testing.assert_allclose(rb.resample_means(d, idx), [1.0, 2.0, 8.0 / 3.0])
    d4 = np.array([1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(rb.resample_means(d4, np.array([[0, 0], [0, 1], [1, 1]]), cluster=[0, 0, 1, 1]), [2.0, 4.0, 6.0])
    # unequal clusters: {d1, d2} and {d3}; resample [0, 1] -> (1 + 3 + 5) / 3, [1, 1] -> 5, [0, 0] -> 2
    np.testing.assert_allclose(rb.resample_means(np.array([1.0, 3.0, 5.0]), np.array([[0, 1], [1, 1], [0, 0]]),
                                                 cluster=["a", "a", "b"]), [3.0, 5.0, 2.0])
    with pytest.raises(rb.RobustnessError):
        rb.resample_means(d, np.zeros((4, 2), dtype=int))


def test_paired_bootstrap_ci_and_p_by_hand():
    cfg = copy.deepcopy(CFG)
    d = np.array([1.0, 2.0, 3.0, 4.0])
    idx = np.array([[0, 0, 0, 0], [3, 3, 3, 3], [0, 1, 2, 3], [1, 1, 2, 2]])        # means 1, 4, 2.5, 2.5
    r = rb.paired_bootstrap(d, idx, cfg)
    srt = [1.0, 2.5, 2.5, 4.0]
    lo = srt[0] + 0.025 * 3 * (srt[1] - srt[0])               # linear percentile at 2.5 % of 4 values
    hi = srt[2] + (0.975 * 3 - 2) * (srt[3] - srt[2])
    assert r["mean"] == pytest.approx(2.5) and r["ci"] == pytest.approx([lo, hi]) and r["ci_level"] == 0.95 and r["B"] == 4
    assert r["p"] == pytest.approx(min(1.0, 2 * min((1 + 0) / 5, (1 + 4) / 5)))         # no mean <= 0: 2 * 1/5
    zero = rb.paired_bootstrap(np.zeros(3), np.array([[0, 1, 2]] * 7), cfg)
    assert zero["p"] == 1.0 and zero["ci"] == [0.0, 0.0]
    neg = rb.paired_bootstrap(-np.ones(3), np.array([[0, 1, 2]] * 99), cfg)
    assert neg["p"] == pytest.approx(2.0 / 100) and neg["ci"] == [-1.0, -1.0]


def test_the_ci_is_a_percentile_ci_and_covers_the_true_mean_difference_on_a_known_problem():
    rng = np.random.default_rng(7)
    d = rng.normal(loc=-0.5, scale=1.0, size=60)
    idx = rb.draw_index_matrix(60, 10000, 42, 42)
    r = rb.paired_bootstrap(d, idx, CFG)
    se = d.std(ddof=1) / np.sqrt(60)
    assert r["ci"][0] == pytest.approx(d.mean() - 1.96 * se, abs=0.05) and r["ci"][1] == pytest.approx(d.mean() + 1.96 * se, abs=0.05)
    assert r["ci"][1] < 0.0 and r["p"] < 0.01


def _rows(m3, m2, m0, m0b=None, m1=None):
    out = []
    for i, (a, b, c) in enumerate(zip(m3, m2, m0)):
        sc = {"M3": a, "M2": b, "M0": c, "M0b": c if m0b is None else m0b[i], "M1": b if m1 is None else m1[i]}
        out.append({"id": f"s{i}", "status": "matched", "scores": sc})
    return out


def test_one_matrix_per_analysis_is_shared_by_every_comparison_and_never_refits(monkeypatch):
    calls = []
    real = rb.draw_index_matrix
    monkeypatch.setattr(rb, "draw_index_matrix", lambda *a, **k: calls.append(a) or real(*a, **k))
    for name in ("run_pass1", "run_recording"):
        monkeypatch.setattr(passes, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("filter called in the bootstrap")))
    monkeypatch.setattr(regression, "fit_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("PySR refit")))
    rows = _rows([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 5.0, 7.0, 9.0])
    out = rb.analyse(rows, CFG, 42, 200)
    assert len(calls) == 1 and calls[0] == (4, 200, 42, 42)
    names = set(out["comparisons"])
    assert {"M3_vs_M2", "M3_vs_M0", "M3_vs_M0b", "M1_vs_M0", "M2_vs_M1"} <= names
    assert out["comparisons"]["M3_vs_M2"]["mean"] == pytest.approx(-1.0)
    assert out["comparisons"]["M3_vs_M0"]["mean"] == pytest.approx(np.mean([-2.0, -3.0, -4.0, -5.0]))
    assert out["means"]["M3"] == pytest.approx(2.5)


def test_analysis_over_rows_without_m1_reports_only_the_comparisons_it_can():
    rows = _rows([1.0, 2.0], [2.0, 3.0], [3.0, 4.0])
    for r in rows:
        del r["scores"]["M1"]
    out = rb.analyse(rows, CFG, 42, 50)
    assert "M1_vs_M0" not in out["comparisons"] and "M3_vs_M2" in out["comparisons"]
    assert rb.analyse([], CFG, 42, 50)["n_subjects"] == 0


# ---- Holm ---------------------------------------------------------------------------------------------------------

def test_holm_textbook_numbers_and_cap_and_incomplete_family():
    # sorted 0.005 (x4 = 0.02), 0.01 (x3 = 0.03), 0.03 (x2 = 0.06), 0.04 (x1 = 0.04 -> raised to 0.06 by monotonicity)
    np.testing.assert_allclose(rb.holm([0.01, 0.04, 0.03, 0.005], 4), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(rb.holm([0.5, 0.6], 2), [1.0, 1.0])
    np.testing.assert_allclose(rb.holm([0.2], 1), [0.2])
    with pytest.raises(rb.RobustnessError, match="family"):
        rb.holm([0.01, 0.02], 4)


# ---- the C1 verdict ------------------------------------------------------------------------------------------------

def _an(ci_m2, ci_m0):
    return {"comparisons": {"M3_vs_M2": {"ci": ci_m2}, "M3_vs_M0": {"ci": ci_m0}}}


def test_c1_needs_both_whole_cis_below_zero():
    assert rb.c1_verdict(_an([-0.3, -0.1], [-0.5, -0.2]), "scored")["passed"] is True
    assert rb.c1_verdict(_an([-0.3, 0.1], [-0.5, -0.2]), "scored")["passed"] is False          # CI straddles zero
    assert rb.c1_verdict(_an([-0.3, -0.1], [-0.5, 0.0]), "scored")["passed"] is False          # upper bound exactly zero
    assert rb.c1_verdict(_an([0.1, 0.3], [-0.5, -0.2]), "scored")["passed"] is False           # excludes zero on the WRONG side
    assert rb.c1_verdict(_an([0.1, 0.3], [0.2, 0.4]), "scored")["passed"] is False
    v = rb.c1_verdict(_an([0.0, 0.0], [-0.5, -0.2]), "no_term")
    assert v["passed"] is False and "no residual term" in v["reason"]
    assert rb.c1_verdict({"comparisons": {}}, "absent")["passed"] is None


# ---- the guard -----------------------------------------------------------------------------------------------------

def _world(monkeypatch, n_train=20, n_test=10):
    import main as main_mod
    train = [f"sub-{i:03d}" for i in range(1, n_train + 1)]
    test = [f"sub-{i:03d}" for i in range(n_train + 1, n_train + n_test + 1)]
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(train))
    return main_mod, train, test


def test_pilot_plan_is_the_internal_test_side_and_never_loads_the_test_list(monkeypatch, temp_root):
    main_mod, train, test = _world(monkeypatch)
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test list loaded")))
    pilot = train[:12]
    plan = rb.resolve_subjects(CFG, temp_root, 42, pilot=True, pilot_ids=pilot)
    assert plan.mode == "pilot" and not plan.allow_all and len(plan.ids) == 4
    assert plan.ids == tuple(baseline.pilot_internal_split(pilot, CFG)[1]) and set(plan.ids) <= set(pilot)
    assert not set(plan.ids) & set(test)
    with pytest.raises(rb.GuardError):
        rb.resolve_subjects(CFG, temp_root, 42, pilot=True, pilot_ids=train[:11] + [test[0]])
    with pytest.raises(rb.GuardError):
        rb.resolve_subjects(CFG, temp_root, 42, pilot=True, confirmatory=True, pilot_ids=pilot)


def test_confirmatory_needs_the_flag_the_gate_and_the_frozen_equation(monkeypatch, temp_root):
    main_mod, train, test = _world(monkeypatch)
    monkeypatch.setattr(main_mod, "test_subjects", lambda seed, root=None, cfg=None: list(test))
    with pytest.raises(rb.GuardError, match="confirmatory=True"):
        rb.resolve_subjects(CFG, temp_root, 42, pilot=False)
    with pytest.raises(tuning.GateError):
        rb.resolve_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    (temp_root / "outputs").mkdir()
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": True}))
    with pytest.raises(tuning.GateError):
        rb.resolve_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": False}))
    with pytest.raises(rb.GuardError, match="frozen equation"):
        rb.resolve_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    doc = regression.build_frozen_document(_record(), CFG, REPO, 42, False)
    regression.write_frozen_equation(regression.frozen_equation_path(CFG, temp_root, 42, False), doc)
    plan = rb.resolve_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    assert plan.mode == "confirmatory" and plan.allow_all and plan.ids == tuple(sorted(test))


def _record():
    z = regression.ZScore(mean_X=np.zeros(3), sd_X=np.ones(3), mean_y=0.0, sd_y=1.0, n=10)
    return {"role": "primary", "no_term": False, "equation": "tanh(u_tgt)", "zscore": z.to_dict(), "front": [],
            "fit_subjects": ["sub-001"], "val_subjects": ["sub-002"], "complexity": 2, "val_loss": 1.0, "signatures": ["tanh(u_tgt)"]}


def test_guarded_loader_refuses_before_calling_the_loader():
    plan = rb.SubjectPlan("pilot", ("sub-001", "sub-002"), frozenset(["sub-001"]), False)
    asked = []
    load = rb.guarded_loader(lambda s: asked.append(s) or {"reason": None}, plan)
    load("sub-001")
    with pytest.raises(rb.GuardError):
        load("sub-099")
    assert asked == ["sub-001"]


def _write_baseline(root, ids, skipped=(), pilot=True, seed=42):
    jpath, npath = baseline.output_paths(CFG, root, seed, pilot)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps({"pilot": pilot, "split_seed": seed, "order": 3, "mechanics_only": pilot,
                                 "skipped_subjects": [{"subject": s, "reason": "x"} for s in skipped]}), encoding="utf-8")
    arrays = {}
    for s in ids:
        b = base_entry()
        for k in ("starts", "lengths", "mask", "M0", "M0b"):
            arrays[f"{s}|{k}"] = b[k]
    np.savez(npath, **arrays)


def test_baseline_scores_must_cover_exactly_the_planned_subjects(tmp_path):
    plan = rb.SubjectPlan("pilot", ("a", "b", "c"), frozenset("abc"), False)
    _write_baseline(tmp_path, ["a", "b", "c"])
    scores, doc, _, _ = rb.load_baseline(CFG, tmp_path, 42, plan)
    assert sorted(scores) == ["a", "b", "c"]
    _write_baseline(tmp_path, ["a", "b"], skipped=["c"])
    assert sorted(rb.load_baseline(CFG, tmp_path, 42, plan)[0]) == ["a", "b"]           # a skipped subject is accounted for
    _write_baseline(tmp_path, ["a", "b"])
    with pytest.raises(rb.GuardError, match="exactly"):
        rb.load_baseline(CFG, tmp_path, 42, plan)                                        # c is missing without a reason
    _write_baseline(tmp_path, ["a", "b", "c", "z"])
    with pytest.raises(rb.GuardError, match="exactly"):
        rb.load_baseline(CFG, tmp_path, 42, plan)                                        # z is not planned
    _write_baseline(tmp_path, ["a", "b", "c"], pilot=False)
    with pytest.raises(rb.GuardError):
        rb.load_baseline(CFG, tmp_path, 42, plan)
    (tmp_path / "results" / "pilot" / "baseline_var_42.json").unlink()
    with pytest.raises(rb.GuardError, match="not found"):
        rb.load_baseline(CFG, tmp_path, 42, plan)


# ---- writer ---------------------------------------------------------------------------------------------------------

def test_c1_output_is_written_once_and_a_different_result_is_refused(tmp_path):
    path = tmp_path / "r" / "c1_42.json"
    doc = {"a": 1, "provenance": {"git_commit": "x"}}
    assert rb.write_c1(path, doc) == "written"
    before = path.read_bytes()
    assert rb.write_c1(path, {"a": 1, "provenance": {"git_commit": "y"}}) == "unchanged" and path.read_bytes() == before
    with pytest.raises(rb.RobustnessError, match="force"):
        rb.write_c1(path, {"a": 2, "provenance": {}})
    assert path.read_bytes() == before
    assert rb.write_c1(path, {"a": 2, "provenance": {}}, force=True) == "written"


def test_output_paths_pilot_and_full():
    assert rb.output_path(CFG, "/r", 42, True).as_posix().endswith("results/pilot/c1_42.json")
    assert rb.output_path(CFG, "/r", 43, False).as_posix().endswith("results/c1_43.json")


# ---- source hygiene -----------------------------------------------------------------------------------------------

def test_source_hygiene_no_print_no_pysr_no_global_random_no_literals():
    text = (REPO / "src" / "robustness.py").read_text(encoding="utf-8")
    assert not re.search(r"(?<![A-Za-z_])print\(", text)
    assert "np.random.seed" not in text and "np.random.rand" not in text and "np.random.normal" not in text
    for banned in ("load_pysr", "fit_model", "run_fit", "run_ensemble", "import pysr"):
        assert banned not in text
    tree = ast.parse(text)
    bad = [(n.lineno, n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool) and n.value not in {0, 1, 2, 0.0, 1.0, 2.0}]
    assert not bad, bad
    code = "import sys; import src.robustness; print([m for m in ('pysr', 'juliacall') if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr


# ---- end to end on synthetic data (stub loader, temp root, pilot mode) -------------------------------------------------

def _pilot_world(monkeypatch, seed=71):
    import main as main_mod
    train = [f"sub-{i:03d}" for i in range(1, 21)]
    test = [f"sub-{i:03d}" for i in range(21, 31)]
    data = {}
    for k, s in enumerate(train + test):
        rec = make_recording(CFG, (6.7, 3.5), seed + k, g12=5.0, g21=2.0, m=0.3, p=np.array([220.0, 250.0]), gap_seconds=1.0,
                             burn_seconds=2.0)
        data[s] = {"reason": None, "segments": rec["segments"], "starts": rec["starts"], "key": f"k{s}"}
    asked = []

    def loader(s):
        asked.append(s)
        return data[s]
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(train))
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test list loaded")))
    return train, test, loader, asked


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """One pilot-mode run with a frozen equation (M3 scored), plus the same subjects without one."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("c1root")
    import shutil
    shutil.copy(REPO / "config.yml", root / "config.yml")
    train, test, loader, asked = _pilot_world(mp)
    pilot = train[:12]
    baseline.run_baseline(CFG, root, 42, pilot=True, loader=loader, pilot_ids=pilot)
    absent = rb.run_c1(CFG, root, 42, pilot=True, loader=loader, pilot_ids=pilot)
    doc = regression.build_frozen_document(_record(), CFG, REPO, 42, True)
    regression.write_frozen_equation(regression.frozen_equation_path(CFG, root, 42, True), doc)
    n_before = len(asked)
    scored = rb.run_c1(CFG, root, 42, pilot=True, loader=loader, pilot_ids=pilot, force=True)
    yield {"root": root, "absent": absent, "scored": scored, "asked": asked, "pilot": pilot, "n_before": n_before,
           "loader": loader}
    mp.undo()


def test_e2e_pilot_reads_only_pilot_subjects_and_writes_to_the_pilot_folder(e2e):
    assert set(e2e["asked"]) <= set(e2e["pilot"])
    for key in ("absent", "scored"):
        d = e2e[key]["doc"]
        assert d["pilot"] and d["mechanics_only"] and d["mode"] == "pilot"
        assert e2e[key]["path"].parent == e2e["root"] / "results" / "pilot"
        assert len(d["subjects"]) == 4 and set(d["subjects"]) <= set(e2e["pilot"])
    assert not (e2e["root"] / "results" / "c1_42.json").exists()


def test_e2e_without_a_frozen_equation_m3_is_absent_and_no_verdict_is_made(e2e):
    d = e2e["absent"]["doc"]
    assert d["m3"]["mode"] == "absent" and d["c1_verdict"]["passed"] is None and d["sensitivity_m3_loss"] is None
    assert "M3" not in d["primary"]["means"] and {"M0", "M0b", "M1", "M2"} <= set(d["primary"]["means"])
    assert "M1_vs_M0" in d["primary"]["comparisons"] and "M3_vs_M2" not in d["primary"]["comparisons"]
    assert d["primary"]["bootstrap"]["B"] == CFG["statistics"]["bootstrap_B_pilot"] == 100


def test_e2e_scored_run_has_m3_and_the_primary_verdict_machinery(e2e):
    d = e2e["scored"]["doc"]
    assert d["m3"]["mode"] == "scored" and d["m3"]["equation"] == "tanh(u_tgt)" and len(d["m3"]["frozen_equation_sha256"]) == 64
    prim = d["primary"]
    assert prim["n_subjects"] == d["divergence"]["removed"]["matched"] > 0
    cmp_ = prim["comparisons"]
    for k in ("M3_vs_M2", "M3_vs_M0", "M3_vs_M0b", "M1_vs_M0", "M2_vs_M1"):
        assert np.isfinite(cmp_[k]["mean"]) and cmp_[k]["ci"][0] <= cmp_[k]["mean"] <= cmp_[k]["ci"][1] + 1e-9
    assert cmp_["M3_vs_M2"]["difference"] == "M3 - M2" and cmp_["M3_vs_M2"]["role"] == "primary"
    assert d["c1_verdict"]["passed"] in (True, False)
    assert d["holm"]["holm_status"] == "partial_family" and d["holm"]["adjusted"] is None
    assert set(d["holm"]["raw_p"]) == {"M1_vs_M0", "M2_vs_M1"}
    assert set(d["divergence"]["per_variant"]) >= {"M0", "M0b", "M1", "M2", "M3"}
    # the same index matrix serves every comparison of the analysis (one hash per analysis)
    assert len(prim["bootstrap"]["index_matrix_sha256"]) == 64


def test_e2e_per_subject_means_equal_an_independent_recomputation_from_the_baseline_file(e2e):
    d = e2e["scored"]["doc"]
    bpath = baseline.output_paths(CFG, e2e["root"], 42, True)[1]
    base = baseline.load_scores(bpath)
    for row in d["per_subject"]:
        if row["status"] != "matched":
            continue
        b = base[row["id"]]
        mask = np.concatenate(passes.scoring_mask([int(n) for n in b["lengths"]], CFG))
        assert row["scores"]["M0"] == pytest.approx(float(np.mean(b["M0"][mask])), rel=1e-12)
        assert row["scores"]["M0b"] == pytest.approx(float(np.mean(b["M0b"][mask])), rel=1e-12)
        assert row["n_scored"] == int(mask.sum())
    means = d["primary"]["means"]
    ok = [r for r in d["per_subject"] if r["status"] == "matched"]
    assert means["M2"] == pytest.approx(np.mean([r["scores"]["M2"] for r in ok]))


def test_e2e_m3_is_m2_plus_a_real_residual_so_it_differs_from_m2(e2e):
    cmp_ = e2e["scored"]["doc"]["primary"]["comparisons"]["M3_vs_M2"]
    assert abs(cmp_["mean"]) > 0.0


def test_e2e_rerun_is_unchanged_and_uses_the_cache(e2e):
    again = rb.run_c1(CFG, e2e["root"], 42, pilot=True, loader=e2e["loader"], pilot_ids=e2e["pilot"])
    assert again["status"] == "unchanged"
    cache = e2e["root"] / "cache" / "c1_scores"
    assert len(list(cache.glob("*_M3_*.npz"))) == 4 and len(list(cache.glob("*_M2_*.npz"))) == 4


def test_e2e_no_term_equation_makes_m3_equal_to_m2_and_c1_fails(e2e, monkeypatch):
    root = e2e["root"]
    rec = _record()
    rec.update(no_term=True, equation="0.37", signatures=[])
    regression.write_frozen_equation(regression.frozen_equation_path(CFG, root, 42, True),
                                     regression.build_frozen_document(rec, CFG, REPO, 42, True), force=True)
    out = rb.run_c1(CFG, root, 42, pilot=True, loader=e2e["loader"], pilot_ids=e2e["pilot"], force=True)
    d = out["doc"]
    assert d["m3"]["mode"] == "no_term" and d["c1_verdict"]["passed"] is False
    c = d["primary"]["comparisons"]["M3_vs_M2"]
    assert c["mean"] == 0.0 and c["ci"] == [0.0, 0.0]
    assert d["divergence"]["per_variant"]["M3"]["note"].startswith("identical to M2")


def test_e2e_loader_asked_for_an_unplanned_subject_is_refused(e2e):
    plan = rb.resolve_subjects(CFG, e2e["root"], 42, pilot=True, pilot_ids=e2e["pilot"])
    outside = sorted(set(e2e["pilot"]) - set(plan.ids))[0]
    with pytest.raises(rb.GuardError):
        rb.guarded_loader(e2e["loader"], plan)(outside)
