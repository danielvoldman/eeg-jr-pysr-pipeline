"""G2 tests, part 1: the Holm family registry, ICC(3,1) with its F-based and cluster-bootstrap CIs, the vigilance-adjusted
ICC, pair building, the C3 guard and the C3 driver (src/robustness.py; §11.3, §11.4, §14, §15; IMP-080, IMP-081).
Expected values are the Shrout and Fleiss (1979) table, hand ANOVA written as explicit loops, closed-form properties and
simulation coverage; never read back from the code under test."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src import passes, regression, robustness as rb, tuning
from src.config import load_config
from sim_data import make_recording

CFG = load_config()
REPO = Path(__file__).resolve().parent.parent
LEVEL = 0.95

# Shrout and Fleiss 1979, Table 2: 6 targets x 4 judges. Published: BMS 11.24, JMS 32.49, EMS 1.02, ICC(3,1) = .71
SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)


def anova_by_loops(Y):
    """Two-way ANOVA by explicit sums (a different route from the code's vectorised means): returns MSR, MSE."""
    n, k = len(Y), len(Y[0])
    grand = sum(sum(r) for r in Y) / (n * k)
    row = [sum(r) / k for r in Y]
    col = [sum(Y[i][j] for i in range(n)) / n for j in range(k)]
    ss_total = sum((Y[i][j] - grand) ** 2 for i in range(n) for j in range(k))
    ss_rows = k * sum((m - grand) ** 2 for m in row)
    ss_cols = n * sum((m - grand) ** 2 for m in col)
    return ss_rows / (n - 1), (ss_total - ss_rows - ss_cols) / ((n - 1) * (k - 1))


# ---- Holm family ---------------------------------------------------------------------------------------------------

EXPECTED_MEMBERS = ["c1_step:M1_vs_M0", "c1_step:M2_vs_M1", "freerun:2s", "freerun:5s", "freerun:10s",
                    "c3_test_icc:g12", "c3_test_icc:g21", "c1_seed:43", "c1_seed:44", "c1_seed:45", "c1_seed:46"]


def test_the_holm_family_has_the_eleven_members_of_the_four_bullets_and_names_their_stage():
    members = rb.holm_members(CFG)
    assert [m["name"] for m in members] == EXPECTED_MEMBERS and CFG["statistics"]["holm_family_size"] == 11
    stage = {m["name"]: m["stage"] for m in members}
    assert {stage[n] for n in EXPECTED_MEMBERS[:2]} == {"G1"} and {stage[n] for n in EXPECTED_MEMBERS[2:5]} == {"G5"}
    assert {stage[n] for n in EXPECTED_MEMBERS[5:7]} == {"G4"} and {stage[n] for n in EXPECTED_MEMBERS[7:]} == {"G3"}
    assert len({m["bullet"] for m in members}) == 4


def test_a_family_size_that_disagrees_with_the_registry_raises():
    cfg = copy.deepcopy(CFG)
    cfg["statistics"]["holm_family_size"] = 4
    with pytest.raises(rb.RobustnessError, match="11 members"):
        rb.holm_members(cfg)


def test_no_adjusted_value_until_all_eleven_exist_and_then_the_textbook_numbers():
    raw = {n: 0.01 for n in EXPECTED_MEMBERS[:10]}
    part = rb.holm_report(CFG, raw)
    assert part["status"] == "partial_family" and part["missing"] == [EXPECTED_MEMBERS[10]]
    assert all(m["adjusted_p"] is None for m in part["members"])
    raw[EXPECTED_MEMBERS[10]] = None
    assert rb.holm_report(CFG, raw)["status"] == "partial_family"
    with pytest.raises(rb.RobustnessError, match="not members"):
        rb.holm_report(CFG, dict(raw, extra=0.5))
    # hand Holm: sorted p 0.001, 0.004, 0.01 x 8, 0.2; adjusted = running max of (11 - rank) p
    p = dict(zip(EXPECTED_MEMBERS, [0.2, 0.004, 0.001] + [0.01] * 8))
    rep = rb.holm_report(CFG, p)
    assert rep["status"] == "complete"
    adj = {m["name"]: m["adjusted_p"] for m in rep["members"]}
    assert adj["freerun:2s"] == pytest.approx(0.011) and adj["c1_step:M2_vs_M1"] == pytest.approx(0.04)
    assert adj["c1_step:M1_vs_M0"] == pytest.approx(0.2) and adj["c3_test_icc:g12"] == pytest.approx(0.09)
    assert all(a <= 1.0 for a in adj.values())
    with pytest.raises(rb.RobustnessError):
        rb.holm_report(CFG, dict(p, **{"freerun:5s": 1.5}))


# ---- ICC(3,1) ------------------------------------------------------------------------------------------------------

def test_icc31_reproduces_the_shrout_and_fleiss_table():
    icc, msr, mse = rb.icc31_values(SF)
    assert (msr, mse) == (pytest.approx(11.24, abs=0.006), pytest.approx(1.02, abs=0.006))
    assert float(icc) == pytest.approx(0.71, abs=0.005)


def test_icc31_matches_a_hand_anova_for_two_sessions_and_agrees_in_the_vectorised_form():
    Y = [[1.0, 2.0], [2.0, 2.0], [3.0, 5.0], [4.0, 4.0], [6.0, 7.5]]
    msr, mse = anova_by_loops(Y)
    want = (msr - mse) / (msr + mse)
    out = rb.icc31(np.array(Y), LEVEL)
    assert out["icc"] == pytest.approx(want, rel=1e-12) and out["MSR"] == pytest.approx(msr) and out["MSE"] == pytest.approx(mse)
    assert out["df"] == [4, 4] and out["F"] == pytest.approx(msr / mse)
    stack = np.stack([np.array(Y), np.array(Y)[::-1]])
    assert rb.icc31_values(stack)[0][0] == pytest.approx(want, rel=1e-12)


def test_the_consistency_form_ignores_a_constant_session_offset_but_absolute_agreement_does_not():
    rng = np.random.default_rng(3)
    subj = rng.normal(size=12)
    Y = np.column_stack([subj + 0.3 * rng.normal(size=12), subj + 0.3 * rng.normal(size=12)])
    shifted = Y + np.array([0.0, 5.0])
    assert rb.icc31(shifted, LEVEL)["icc"] == pytest.approx(rb.icc31(Y, LEVEL)["icc"], rel=1e-10)
    n, k = Y.shape
    msr, mse = anova_by_loops(shifted.tolist())
    msc = n * np.sum((shifted.mean(axis=0) - shifted.mean()) ** 2) / (k - 1)
    icc21 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)             # absolute agreement, ICC(2,1)
    assert icc21 < rb.icc31(shifted, LEVEL)["icc"] - 0.2


def test_the_f_based_ci_covers_the_true_icc_95_percent_of_the_time():
    rng = np.random.default_rng(11)
    n, reps, hit, order_ok = 42, 1500, 0, True
    for _ in range(reps):
        subj = rng.normal(size=n)
        Y = np.column_stack([subj + rng.normal(size=n), subj + rng.normal(size=n)])        # true ICC(3,1) = 0.5
        r = rb.icc31(Y, LEVEL)
        lo, hi = r["ci"]
        order_ok &= lo <= r["icc"] <= hi
        hit += lo <= 0.5 <= hi
    assert order_ok and 0.93 <= hit / reps <= 0.97


def test_the_ci_quoted_in_the_document_and_the_inverse_for_a_lower_bound_of_0_40():
    # §11.5 quotes an observed ICC of 0.50 at n = 42 as roughly 0.24 to 0.70; F = (1 + icc) / (1 - icc) for k = 2
    ci = rb._f_ci(3.0, 2, 41, 41, LEVEL)
    assert 0.20 <= ci[0] <= 0.26 and 0.68 <= ci[1] <= 0.72
    icc = rb.icc_for_ci_lower(0.40, 42, 2, LEVEL)
    assert 0.55 < icc < 0.65                                                         # §15.1: about 0.6
    back = rb._f_ci((1.0 + icc) / (1.0 - icc), 2, 41, 41, LEVEL)
    assert back[0] == pytest.approx(0.40, abs=1e-9)
    assert rb.icc_for_ci_lower(0.40, 100, 2, LEVEL) < icc                           # more pairs, a smaller observed ICC suffices


def test_the_f_based_ci_for_four_sessions_matches_the_published_shrout_and_fleiss_interval():
    # k = 4 makes the two df differ (5 and 15), which two sessions cannot: published ICC(3,1) .71, 95% CI .34 to .95
    out = rb.icc31(SF, LEVEL)
    assert out["df"] == [5, 15] and out["ci"][0] == pytest.approx(0.34, abs=0.01) and out["ci"][1] == pytest.approx(0.95, abs=0.01)
    assert out["F"] == pytest.approx(11.24 / 1.02, rel=0.01)


def test_icc_is_reported_as_is_with_flags_never_clipped():
    neg = rb.icc31(np.array([[1.0, 3.0], [3.0, 1.0], [2.0, 2.5], [4.0, 0.5], [0.0, 4.0]]), LEVEL)
    assert neg["icc"] < 0 and "negative_icc" in neg["flags"] and neg["p"] > 0.5
    perfect = rb.icc31(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]), LEVEL)
    assert perfect["icc"] == 1.0 and perfect["flags"] == ["mse_zero"] and perfect["ci"] == [1.0, 1.0] and perfect["p"] == 0.0
    flat = rb.icc31(np.ones((4, 2)), LEVEL)
    assert np.isnan(flat["icc"]) and "undefined" in flat["flags"]
    with pytest.raises(rb.RobustnessError):
        rb.icc31_values(np.ones((1, 2)))


def test_the_icc_p_value_is_the_upper_tail_f_test_of_icc_zero():
    from scipy import stats
    Y = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 5.0], [4.0, 4.0], [6.0, 7.5]])
    out = rb.icc31(Y, LEVEL)
    assert out["p"] == pytest.approx(1.0 - stats.f.cdf(out["F"], 4, 4), rel=1e-9)


# ---- vigilance adjustment ------------------------------------------------------------------------------------------

def test_vigilance_residuals_are_the_pooled_ols_residuals_with_an_intercept():
    rng = np.random.default_rng(5)
    V = rng.uniform(1, 4, size=(9, 2))
    Y = np.column_stack([2.0 * V[:, 0] + 1.0, -1.0 * V[:, 1] + 7.0]) + 0.1 * rng.normal(size=(9, 2))
    res = rb.vigilance_adjust(Y, V)
    slope, intercept = np.polyfit(V.ravel(), Y.ravel(), 1)                           # independent pooled fit
    assert res.shape == Y.shape and np.allclose(res.ravel(), Y.ravel() - (slope * V.ravel() + intercept))
    assert abs(res.sum()) < 1e-9 and abs(np.sum(res * V)) < 1e-9                     # normal equations: intercept and slope
    per_session = np.column_stack([np.polyfit(V[:, j], Y[:, j], 1)[0] for j in range(2)])
    assert not np.allclose(per_session, slope)                                       # a per-session fit would differ here
    exact = np.column_stack([3.0 * V[:, 0] + 2.0, 3.0 * V[:, 1] + 2.0])
    assert np.allclose(rb.vigilance_adjust(exact, V), 0.0, atol=1e-9)
    with pytest.raises(rb.RobustnessError):
        rb.vigilance_adjust(Y, V[:, :1])


def test_a_session_difference_that_vigilance_explains_is_removed_by_the_adjustment():
    rng = np.random.default_rng(8)
    base = rng.uniform(1, 3, size=(30, 1))
    v = base + 0.1 * rng.normal(size=(30, 2))                                        # a subject's vigilance persists
    y = 4.0 * v + 0.05 * rng.normal(size=(30, 2))                                    # the estimates are all vigilance
    assert rb.icc31(y, LEVEL)["icc"] > 0.9
    assert abs(rb.icc31(rb.vigilance_adjust(y, v), LEVEL)["icc"]) < 0.5


def test_vigilance_value_is_the_mean_over_all_epochs_and_both_channels():
    vig = {"ratios": [[1.0, 2.0, 6.0], [3.0, 4.0, 8.0]], "epoch_starts": [0, 512, 1024], "mean": [3.0, 5.0]}
    value, n = rb.vigilance_value(vig)
    assert value == pytest.approx(4.0) and n == 3
    with pytest.raises(rb.RobustnessError):
        rb.vigilance_value({"ratios": [[], []]})


# ---- cluster bootstrap ---------------------------------------------------------------------------------------------

def _reliable(n=30, seed=1, noise=0.15):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    return np.column_stack([s + noise * rng.normal(size=n), s + noise * rng.normal(size=n)])


def test_the_cluster_bootstrap_keeps_a_pairs_sessions_together():
    Y = _reliable()
    point = rb.icc31(Y, LEVEL)["icc"]
    idx = rb.draw_index_matrix(Y.shape[0], 500, 42, 42)
    out = rb.icc_cluster_bootstrap(Y, idx, LEVEL)
    assert out["ci"][0] > 0.8 and out["ci"][0] <= point <= out["ci"][1] and out["B"] == 500 and out["n_undefined"] == 0
    with pytest.raises(rb.RobustnessError):
        rb.icc_cluster_bootstrap(Y, idx[:, :5], LEVEL)


def test_the_bootstrap_ci_brackets_the_f_based_ci_roughly_and_is_reproducible():
    Y = _reliable(n=42, seed=4, noise=0.6)
    f_ci = rb.icc31(Y, LEVEL)["ci"]
    idx = rb.draw_index_matrix(42, 2000, 42, 42)
    a, b = rb.icc_cluster_bootstrap(Y, idx, LEVEL), rb.icc_cluster_bootstrap(Y, idx, LEVEL)
    assert a == b and abs(a["ci"][0] - f_ci[0]) < 0.15 and abs(a["ci"][1] - f_ci[1]) < 0.15


def test_degenerate_resamples_are_counted_not_dropped_silently():
    Y = np.array([[1.0, 2.0], [3.0, 5.0]])
    idx = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])                                 # two resamples repeat one subject
    out = rb.icc_cluster_bootstrap(Y, idx, LEVEL)
    assert out["n_undefined"] == 2 and out["B"] == 4


def test_c3_analysis_uses_one_index_matrix_for_both_directions_and_the_adjusted_icc():
    pairs = [{"id": f"s{i}", "g12": [float(i), i + 0.1 * (i % 3)], "g21": [10.0 - i, 10.0 - i + 0.2 * (i % 2)],
              "vigilance": [1.0 + 0.1 * i, 1.0 + 0.1 * i + 0.05]} for i in range(8)]
    an = rb.c3_analysis(pairs, CFG, 42, 100)
    idx = rb.draw_index_matrix(8, 100, CFG["statistics"]["bootstrap_seed"], 42)
    import hashlib
    assert an["index_matrix_sha256"] == hashlib.sha256(idx.tobytes()).hexdigest() and an["n_pairs"] == 8
    assert set(an["directions"]) == {"g12", "g21"}
    for d in an["directions"].values():
        assert {"unadjusted", "vigilance_adjusted"} <= set(d) and "cluster_bootstrap" in d["unadjusted"]
    want = rb.icc_cluster_bootstrap(np.array([p["g12"] for p in pairs]), idx, LEVEL)
    assert an["directions"]["g12"]["unadjusted"]["cluster_bootstrap"] == want
    assert rb.c3_analysis(pairs[:1], CFG, 42, 100)["status"] == "too_few_pairs"


def test_the_c3_verdict_is_the_unadjusted_f_ci_lower_bound_in_at_least_one_direction():
    def an(l12, l21):
        d = lambda lo: {"unadjusted": {"ci": [lo, 0.9]}, "vigilance_adjusted": {"ci": [0.99, 1.0]}}     # noqa: E731
        return {"status": "ok", "directions": {"g12": d(l12), "g21": d(l21)}}
    assert rb.c3_verdict(an(0.40, 0.1), CFG, "M3")["passed"] is True                  # at the minimum counts (>=)
    assert rb.c3_verdict(an(0.399, 0.39), CFG, "M3")["passed"] is False               # the adjusted ICC never rescues it
    assert rb.c3_verdict(an(0.1, 0.5), CFG, "M3_no_term_equals_M2")["passed"] is True
    assert rb.c3_verdict(an(0.9, 0.9), CFG, "M2_stand_in")["passed"] is None
    assert rb.c3_verdict({"status": "too_few_pairs"}, CFG, "M3")["passed"] is None


# ---- pairs ---------------------------------------------------------------------------------------------------------

def _row(sid, g1, g2, partition="train", div1=False, div2=False, reason2=None, none1=False):
    def ses(g, div, reason=None):
        if reason:
            return {"reason": reason}
        return {"reason": None, "gain": {"g12": None if none1 else g, "g21": g + 1.0, "recording_diverged": div},
                "vigilance": (1.5, 10)}
    return {"id": sid, "partition": partition, "sessions": {"ses-t1": ses(g1, div1, None), "ses-t2": ses(g2, div2, reason2)}}


def test_a_pair_needs_both_sessions_kept_and_not_diverged_and_every_exclusion_is_listed():
    rows = [_row("a", 1.0, 2.0), _row("b", 1.0, 2.0, div2=True), _row("c", 1.0, 2.0, reason2="excluded: bad_electrode"),
            _row("d", 1.0, 2.0, div1=True), _row("e", 1.0, 2.0, none1=True), _row("f", 3.0, 4.0, partition="test")]
    pairs, excluded = rb.build_pairs(rows, CFG)
    assert [p["id"] for p in pairs] == ["a", "f"] and pairs[1]["partition"] == "test"
    assert pairs[0]["g12"] == [1.0, 2.0] and pairs[0]["g21"] == [2.0, 3.0] and pairs[0]["vigilance"] == [1.5, 1.5]
    why = {e["id"]: e["reason"] for e in excluded}
    assert "ses-t2: diverged" in why["b"] and "bad_electrode" in why["c"] and "ses-t1: diverged" in why["d"]
    assert "no gain estimate" in why["e"] and set(why) == {"b", "c", "d", "e"}


# ---- guard ---------------------------------------------------------------------------------------------------------

def _data_tree(root, both, only_t1):
    for sid in both:
        for ses in ("ses-t1", "ses-t2"):
            (root / "data" / sid / ses / "eeg").mkdir(parents=True, exist_ok=True)
            (root / "data" / sid / ses / "eeg" / f"{sid}_{ses}_eeg.edf").write_bytes(b"x")
    for sid in only_t1:
        (root / "data" / sid / "ses-t1" / "eeg").mkdir(parents=True, exist_ok=True)
        (root / "data" / sid / "ses-t1" / "eeg" / f"{sid}_ses-t1_eeg.edf").write_bytes(b"x")


def _world(monkeypatch, n_train=14, n_test=6):
    import main as main_mod
    train = [f"sub-{i:03d}" for i in range(1, n_train + 1)]
    test = [f"sub-{i:03d}" for i in range(n_train + 1, n_train + n_test + 1)]
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(train))
    return main_mod, train, test


def test_the_pilot_c3_plan_is_the_pilot_subjects_with_both_sessions_and_never_loads_the_test_list(monkeypatch, temp_root):
    main_mod, train, test = _world(monkeypatch)
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test list loaded")))
    monkeypatch.setattr(main_mod, "load_split", lambda *a, **k: (_ for _ in ()).throw(AssertionError("split loaded")))
    pilot = train[:12]
    _data_tree(temp_root, pilot[:4], pilot[4:] + test)
    plan = rb.resolve_c3_subjects(CFG, temp_root, 42, pilot=True, pilot_ids=pilot)
    assert plan.mode == "pilot" and not plan.allow_all and plan.ids == tuple(pilot[:4])
    assert set(plan.partition.values()) == {"train"} and not set(plan.ids) & set(test)
    with pytest.raises(rb.GuardError):
        rb.resolve_c3_subjects(CFG, temp_root, 42, pilot=True, pilot_ids=train[:11] + [test[0]])
    with pytest.raises(rb.GuardError):
        rb.resolve_c3_subjects(CFG, temp_root, 42, pilot=True, confirmatory=True, pilot_ids=pilot)


def test_the_confirmatory_c3_plan_needs_flag_gate_and_equation_and_covers_both_partitions(monkeypatch, temp_root):
    main_mod, train, test = _world(monkeypatch)
    split = {"subjects_with_t2": [train[0], train[1], test[0]], "train": train, "test": test}
    monkeypatch.setattr(main_mod, "load_split", lambda seed, root=None, cfg=None: split)
    with pytest.raises(rb.GuardError, match="confirmatory=True"):
        rb.resolve_c3_subjects(CFG, temp_root, 42, pilot=False)
    with pytest.raises(tuning.GateError):
        rb.resolve_c3_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    (temp_root / "outputs").mkdir()
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": False}))
    with pytest.raises(rb.GuardError, match="frozen equation"):
        rb.resolve_c3_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    doc = regression.build_frozen_document(_record(), CFG, REPO, 42, False)
    regression.write_frozen_equation(regression.frozen_equation_path(CFG, temp_root, 42, False), doc)
    plan = rb.resolve_c3_subjects(CFG, temp_root, 42, pilot=False, confirmatory=True)
    assert plan.allow_all and plan.ids == tuple(sorted(split["subjects_with_t2"]))
    assert plan.partition == {train[0]: "train", train[1]: "train", test[0]: "test"}


def _record():
    z = regression.ZScore(mean_X=np.zeros(3), sd_X=np.ones(3), mean_y=0.0, sd_y=1.0, n=10)
    return {"role": "primary", "no_term": False, "equation": "tanh(u_tgt)", "zscore": z.to_dict(), "front": [],
            "fit_subjects": ["sub-001"], "val_subjects": ["sub-002"], "complexity": 2, "val_loss": 1.0, "signatures": ["tanh(u_tgt)"]}


def test_the_guarded_loader_passes_the_session_and_still_refuses_unplanned_subjects():
    plan = rb.SubjectPlan("pilot", ("sub-001",), frozenset(["sub-001"]), False)
    asked = []
    load = rb.guarded_loader(lambda s, session=None: asked.append((s, session)) or {"reason": None}, plan)
    load("sub-001", "ses-t2")
    with pytest.raises(rb.GuardError):
        load("sub-009", "ses-t2")
    assert asked == [("sub-001", "ses-t2")]


def test_make_real_loader_reads_the_asked_session_and_returns_vigilance(monkeypatch, temp_root):
    from src import preprocess as pp
    _data_tree(temp_root, ["sub-001"], [])
    seen = {}

    class Res:
        meta = {"sha256": "abc", "b5": {"vigilance": {"ratios": [[1.0], [2.0]]}}}
        segments, starts = [np.zeros((2, 4))], [0]

    def fake_segment(cfg, path, data_root, manifest, pilot_ids, allow_all, cache_root):
        seen["file"] = Path(path).name
        seen["allow_all"] = allow_all
        return Res
    monkeypatch.setattr(pp, "load_manifest", lambda p: {})
    monkeypatch.setattr(pp, "segment_recording", fake_segment)
    monkeypatch.setattr(pp, "recording_decision", lambda *a, **k: {"status": "kept"})
    monkeypatch.setattr(pp, "cache_key_b5", lambda *a, **k: "k")
    load = tuning.make_real_loader(CFG, temp_root, frozenset(["sub-001"]), allow_all=False)
    r2 = load("sub-001", "ses-t2")
    assert seen["file"] == "sub-001_ses-t2_eeg.edf" and r2["session"] == "ses-t2" and r2["vigilance"]["ratios"][1] == [2.0]
    r1 = load("sub-001")
    assert seen["file"] == "sub-001_ses-t1_eeg.edf" and r1["session"] == "ses-t1" and not seen["allow_all"]


# ---- end to end on synthetic data (stub loader, temp root, pilot mode) --------------------------------------------

def _c3_world(monkeypatch, temp_root, n_pairs=4):
    main_mod, train, test = _world(monkeypatch)
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test list loaded")))
    pilot = train[:12]
    both = pilot[:n_pairs]
    _data_tree(temp_root, both, pilot[n_pairs:])
    data = {}
    for k, s in enumerate(both):
        for j, ses in enumerate(("ses-t1", "ses-t2")):
            rec = make_recording(CFG, (6.7, 3.5), 200 + 10 * k + j, g12=3.0 + 2.0 * k, g21=9.0 - 2.0 * k, m=0.3,
                                 p=np.array([220.0, 250.0]), gap_seconds=1.0, burn_seconds=2.0)
            ratios = [[1.0 + 0.2 * k + 0.1 * j] * 5, [1.2 + 0.2 * k + 0.1 * j] * 5]
            data[s, ses] = {"reason": None, "segments": rec["segments"], "starts": rec["starts"], "key": f"{s}{ses}",
                            "session": ses, "vigilance": {"ratios": ratios}}
    asked = []

    def loader(s, session=None):
        asked.append((s, session))
        return data[s, session]
    return pilot, both, loader, asked, data


@pytest.fixture(scope="module")
def c3e2e(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("c3root")
    import shutil
    shutil.copy(REPO / "config.yml", root / "config.yml")
    pilot, both, loader, asked, data = _c3_world(mp, root)
    stand_in = rb.run_c3(CFG, root, 42, pilot=True, loader=loader, pilot_ids=pilot, n_jobs=1)
    n_first = len(asked)
    yield {"root": root, "pilot": pilot, "both": both, "loader": loader, "asked": asked, "data": data, "stand_in": stand_in,
           "n_first": n_first}
    mp.undo()


def test_e2e_c3_pilot_reads_only_pilot_pairs_and_writes_to_the_pilot_folder(c3e2e):
    d = c3e2e["stand_in"]["doc"]
    assert d["pilot"] and d["mechanics_only"] and d["mode"] == "pilot"
    assert c3e2e["stand_in"]["path"].parent == c3e2e["root"] / "results" / "pilot"
    assert not (c3e2e["root"] / "results" / "c3_42.json").exists()
    assert {s for s, _ in c3e2e["asked"]} == set(c3e2e["both"]) and {x for _, x in c3e2e["asked"]} == {"ses-t1", "ses-t2"}
    assert d["subjects"] == c3e2e["both"] and len(d["pairs"]) == 4 and d["excluded_pairs"] == []


def test_e2e_c3_without_a_frozen_equation_is_a_labelled_stand_in_with_no_verdict(c3e2e):
    d = c3e2e["stand_in"]["doc"]
    assert d["estimator"]["mode"] == "M2_stand_in" and d["verdict"]["passed"] is None
    assert d["test_partition_pairs"]["status"].startswith("no_test_partition_pairs") and set(d["holm_raw_p"].values()) == {None}
    assert d["all_pairs"]["B"] == CFG["statistics"]["bootstrap_B_pilot"] == 100


def test_e2e_c3_gains_and_icc_equal_an_independent_recomputation(c3e2e):
    d = c3e2e["stand_in"]["doc"]
    from src import state_space as ss
    for pair in d["pairs"]:
        for j, ses in enumerate(("ses-t1", "ses-t2")):
            rec = c3e2e["data"][pair["id"], ses]
            p1 = passes.run_pass1(rec["segments"], rec["starts"], CFG, layout=ss.make_layout(CFG), forward_only=True)
            assert pair["g12"][j] == pytest.approx(p1.gain_estimate["g12"], rel=1e-12)
            assert pair["g21"][j] == pytest.approx(p1.gain_estimate["g21"], rel=1e-12)
    Y = [p["g12"] for p in d["pairs"]]
    msr, mse = anova_by_loops(Y)
    assert d["all_pairs"]["directions"]["g12"]["unadjusted"]["icc"] == pytest.approx((msr - mse) / (msr + mse), rel=1e-9)
    assert d["pairs"][0]["vigilance"] == [pytest.approx(1.1), pytest.approx(1.2)]
    ref = d["criterion"]["observed_icc_equivalent_to_the_minimum"]
    assert ref["n_subjects_with_t2"] == 42 and 0.55 < ref["at_n_subjects_with_t2"] < 0.65


def test_e2e_c3_rerun_is_unchanged_and_uses_the_gain_cache(c3e2e):
    again = rb.run_c3(CFG, c3e2e["root"], 42, pilot=True, loader=c3e2e["loader"], pilot_ids=c3e2e["pilot"], n_jobs=1)
    assert again["status"] == "unchanged"
    assert len(list((c3e2e["root"] / "cache" / "c3_gains").glob("*.json"))) == 8


def test_e2e_c3_a_frozen_term_gives_m3_estimates_and_a_no_term_equation_gives_exactly_m2(c3e2e):
    root = c3e2e["root"]
    stand = c3e2e["stand_in"]["doc"]
    rec = _record()
    rec.update(no_term=True, equation="0.37", signatures=[])
    regression.write_frozen_equation(regression.frozen_equation_path(CFG, root, 42, True),
                                     regression.build_frozen_document(rec, CFG, REPO, 42, True))
    nt = rb.run_c3(CFG, root, 42, pilot=True, loader=c3e2e["loader"], pilot_ids=c3e2e["pilot"], n_jobs=1, force=True)["doc"]
    assert nt["estimator"]["mode"] == "M3_no_term_equals_M2" and nt["verdict"]["passed"] in (True, False)
    assert [p["g12"] for p in nt["pairs"]] == [p["g12"] for p in stand["pairs"]]
    regression.write_frozen_equation(regression.frozen_equation_path(CFG, root, 42, True),
                                     regression.build_frozen_document(_record(), CFG, REPO, 42, True), force=True)
    m3 = rb.run_c3(CFG, root, 42, pilot=True, loader=c3e2e["loader"], pilot_ids=c3e2e["pilot"], n_jobs=1, force=True)["doc"]
    assert m3["estimator"]["mode"] == "M3" and len(m3["estimator"]["frozen_equation_sha256"]) == 64
    assert [p["g12"] for p in m3["pairs"]] != [p["g12"] for p in stand["pairs"]]       # a real residual changes the filter
    regression.frozen_equation_path(CFG, root, 42, True).unlink()


def test_e2e_c3_a_diverged_session_removes_the_pair_with_its_reason(monkeypatch, temp_root):
    pilot, both, loader, asked, data = _c3_world(monkeypatch, temp_root, n_pairs=3)
    real = rb.session_gain

    def fake(segments, starts, cfg, residual=None, filter_name=None):
        out = real(segments, starts, cfg, residual=residual, filter_name=filter_name)
        if segments is data[both[1], "ses-t2"]["segments"]:
            out["recording_diverged"] = True
        return out
    monkeypatch.setattr(rb, "session_gain", fake)
    d = rb.run_c3(CFG, temp_root, 42, pilot=True, loader=loader, pilot_ids=pilot, use_cache=False, n_jobs=1)["doc"]
    assert [p["id"] for p in d["pairs"]] == [both[0], both[2]]
    assert d["excluded_pairs"] == [{"id": both[1], "partition": "train", "reason": "ses-t2: diverged at recording level"}]


def test_e2e_c3_loader_asked_for_an_unplanned_subject_is_refused(c3e2e):
    plan = rb.resolve_c3_subjects(CFG, c3e2e["root"], 42, pilot=True, pilot_ids=c3e2e["pilot"])
    outside = sorted(set(c3e2e["pilot"]) - set(plan.ids))[0]
    with pytest.raises(rb.GuardError):
        rb.guarded_loader(c3e2e["loader"], plan)(outside, "ses-t1")
