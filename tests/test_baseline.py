"""F1 tests: the VAR baselines M0 and M0b and the shared scoring mask (§10.2, §13.1; IMP-075, IMP-076).

Expected values are the planted VAR coefficients, analytic error variances, hand-built design rows and masks written
here from the literals 128 (0.5 s) and 1536 (6 s), or arithmetic done in the test; never read back from src/baseline.py.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src import baseline, passes, state_space as ss, tuning
from src.config import load_config
from sim_data import make_recording

CFG = load_config()
BURN, EST_BURN = 128, 1536      # 0.5 s and 6 s at 256 Hz (windows.training_burn_in_s, passes.estimator_burn_in_s)


def cfg_with(**baseline_kw):
    c = copy.deepcopy(CFG)
    c["baseline"].update(baseline_kw)
    return c


def sim_var(A, c, n, rng, sigma=1.0):
    """z(t) = c + sum_l A[l] z(t - l) + sigma e(t); (2, n), started from zero with 300 samples of burn-in thrown away."""
    p, tot = len(A), n + 300
    z = np.zeros((2, tot))
    e = rng.normal(scale=sigma, size=(2, tot))
    for t in range(p, tot):
        z[:, t] = c + sum(A[l] @ z[:, t - 1 - l] for l in range(p)) + e[:, t]
    return z[:, 300:]


def recs_from(A, c, n_sub, n_seg, n, seed, sigma=1.0, prefix="s"):
    rng = np.random.default_rng(seed)
    return [{"id": f"{prefix}{i:03d}", "segments": [sim_var(A, c, n, rng, sigma) for _ in range(n_seg)],
             "starts": [k * (n + 500) for k in range(n_seg)]} for i in range(n_sub)]


# ---- design rows never cross a gap ----------------------------------------------------------------------------

def test_lag_design_matches_hand_built_rows():
    seg = np.array([[1., 2., 3., 4., 5., 6.], [10., 20., 30., 40., 50., 60.]])
    x, y, rows = baseline.lag_design(seg, 2, 2, True)
    # t = 2: [1, z(1), z(0)] = [1, 2, 20, 1, 10], target z(2) = [3, 30]
    np.testing.assert_array_equal(x[0], [1, 2, 20, 1, 10])
    np.testing.assert_array_equal(y[0], [3, 30])
    np.testing.assert_array_equal(rows, [2, 3, 4, 5])
    np.testing.assert_array_equal(x[3], [1, 5, 50, 4, 40])
    x0, _, _ = baseline.lag_design(seg, 2, 2, False)
    np.testing.assert_array_equal(x0[0], [2, 20, 1, 10])


def test_recording_gram_equals_a_hand_loop_over_segments_only():
    rng = np.random.default_rng(3)
    segs = [rng.normal(size=(2, n)) + 10 * k for k, n in enumerate((40, 25, 33))]     # jumps of 10 between segments
    rec = {"id": "x", "segments": segs, "starts": [0, 100, 200]}
    cfg2 = cfg_with(var_order_range=[1, 3])
    r = baseline.recording_r(rec, 3, True)
    rows = []
    for s in segs:                                           # plain loops, lags from the same segment only
        for t in range(3, s.shape[1]):
            rows.append([1.0] + [s[ch, t - l] for l in (1, 2, 3) for ch in (0, 1)] + [s[0, t], s[1, t]])
    a = np.array(rows)
    np.testing.assert_allclose(r.T @ r, a.T @ a, rtol=1e-10)
    assert a.shape[0] == (40 - 3) + (25 - 3) + (33 - 3)      # no row reaches across a gap
    assert cfg2["baseline"]["var_order_range"] == [1, 3]


def test_pooled_fit_recovers_the_planted_coefficients_with_many_short_segments():
    A = [np.array([[0.6, 0.1], [-0.2, 0.5]]), np.array([[-0.2, 0.0], [0.1, -0.15]])]
    c = np.array([0.3, -0.2])
    rng = np.random.default_rng(5)
    segs = [sim_var(A, c, 14, rng) for _ in range(500)]       # short segments: a fit across the gaps would be badly biased
    rec = {"id": "r", "segments": segs, "starts": list(range(0, 500 * 100, 100))}
    cfg2 = cfg_with(var_order_range=[1, 2])
    fit = baseline.fit_pooled([rec], 2, cfg2)
    for l in range(2):
        np.testing.assert_allclose(fit.A[l], A[l], atol=0.05)
    np.testing.assert_allclose(fit.c, c, atol=0.1)


def test_every_order_is_fitted_on_the_same_rows_and_equals_plain_least_squares():
    rng = np.random.default_rng(8)
    A = [np.array([[0.5, 0.0], [0.0, 0.4]])]
    z = sim_var(A, np.zeros(2), 3000, rng)
    rec = {"id": "r", "segments": [z], "starts": [0]}
    for order in (1, 3):
        fit = baseline.fit_pooled([rec], order, CFG)          # config range 1..32: rows t >= 32 for every order
        t = np.arange(32, 3000)
        X = np.column_stack([np.ones(t.size)] + [z[ch, t - l] for l in range(1, order + 1) for ch in (0, 1)])
        Y = z[:, t].T
        ref = np.linalg.lstsq(X, Y, rcond=None)[0]            # numpy's own least squares on the same rows
        np.testing.assert_allclose(fit.coef, ref, rtol=1e-8, atol=1e-10)


# ---- order selection --------------------------------------------------------------------------------------------

A3 = [np.array([[0.5, 0.0], [0.0, 0.4]]), np.zeros((2, 2)), np.array([[0.3, 0.1], [0.0, -0.35]])]


def test_cv_picks_the_known_order_three():
    recs = recs_from(A3, np.zeros(2), n_sub=10, n_seg=2, n=3000, seed=11)
    cfg2 = cfg_with(var_order_range=[1, 8])
    fit, cv = baseline.fit_m0(recs, cfg2)
    assert cv.order == fit.order == 3 and cv.orders == list(range(1, 9))
    assert cv.mse[2] < 0.95 * cv.mse[1]                       # the third lag matters
    assert cv.mse[2] <= min(cv.mse)


def test_cv_order_from_config_range_and_fold_count():
    recs = recs_from(A3[:1], np.zeros(2), n_sub=6, n_seg=1, n=2500, seed=12)
    cv = baseline.cv_select_order(recs, cfg_with(var_order_range=[2, 4], var_cv_folds=3))
    assert cv.orders == [2, 3, 4] and len(cv.folds) == 3 and len(cv.mse) == 3


def test_ties_go_to_the_smaller_order_and_the_minimum_is_chosen():
    assert baseline.choose_order([3.0, 1.0, 1.0, 2.0], [1, 2, 3, 4]) == 2
    assert baseline.choose_order([1.0, 1.0, 1.0], [1, 2, 3]) == 1
    assert baseline.choose_order([5.0, 4.0, 3.0], [1, 2, 3]) == 3         # the smallest error, not the largest


# ---- folds: subject-wise, seeded, no leakage -------------------------------------------------------------------------

def test_folds_partition_the_subjects_and_depend_on_the_seed():
    ids = [f"s{i:03d}" for i in range(78)]
    folds = baseline.make_folds(ids, 5, 42)
    assert len(folds) == 5 and sorted(i for f in folds for i in f) == ids
    assert sorted(len(f) for f in folds) == [15, 15, 16, 16, 16]
    assert baseline.make_folds(ids, 5, 42) == folds and baseline.make_folds(ids, 5, 43) != folds
    assert baseline.make_folds(ids[::-1], 5, 42) == folds     # order of the input does not matter (sorted first)
    with pytest.raises(baseline.BaselineError):
        baseline.make_folds(ids[:3], 5, 42)


def test_fold_fits_never_see_the_held_out_subjects():
    recs = recs_from(A3[:1], np.zeros(2), n_sub=7, n_seg=1, n=2500, seed=14)
    cfg2 = cfg_with(var_order_range=[1, 3], var_cv_folds=3)
    cv = baseline.cv_select_order(recs, cfg2)
    by_id = {r["id"]: r for r in recs}
    for k, held in enumerate(cv.folds):
        train = [by_id[i] for i in by_id if i not in held]
        for p in cv.orders:
            np.testing.assert_allclose(cv.fold_coefs[k][p], baseline.fit_pooled(train, p, cfg2).coef, rtol=1e-9, atol=1e-12)
    # change a held-out subject's data completely: the coefficients of ITS fold are bit-identical
    altered = copy.deepcopy(recs)
    victim = cv.folds[0][0]
    for r in altered:
        if r["id"] == victim:
            r["segments"] = [np.random.default_rng(99).normal(size=s.shape) * 50 for s in r["segments"]]
    cv2 = baseline.cv_select_order(altered, cfg2)
    assert cv2.folds == cv.folds
    for p in cv.orders:
        np.testing.assert_array_equal(cv2.fold_coefs[0][p], cv.fold_coefs[0][p])
    assert cv2.per_subject[victim] != cv.per_subject[victim]  # its own score does change


# ---- the shared scoring mask ---------------------------------------------------------------------------------------

def test_scoring_mask_is_the_two_burn_ins_by_hand():
    lengths = [2048, 1536, 300]
    m = passes.scoring_mask(lengths, CFG)
    expect = []
    cum = 0
    for n in lengths:                                          # hand loop on the literals 128 and 1536
        expect.append(np.array([(i >= 128) and (cum + i >= 1536) for i in range(n)]))
        cum += n
    for a, b in zip(m, expect):
        np.testing.assert_array_equal(a, b)
    assert [int(a.sum()) for a in m] == [2048 - 1536, 1536 - 128, 300 - 128]


def test_mask_equals_the_filter_side_samples_when_nothing_diverges_and_the_scorer_count_agrees():
    rec = make_recording(CFG, (8, 6), seed=41, g12=6.0, g21=3.0, m=0.3, p=np.array([220.0, 250.0]))
    p1 = passes.run_pass1(rec["segments"], rec["starts"], CFG, forward_only=True)            # default filter A
    assert not any(s.diverged for s in p1.segments) and p1.filter_name == "A"
    lengths = [s.shape[1] for s in rec["segments"]]
    mask = passes.scoring_mask(lengths, CFG)
    for s, m in zip(p1.segments, mask):
        np.testing.assert_array_equal(s.nis_keep, m)
    hand = (2048 - 1536) + (1536 - 128)                         # = 1920, arithmetic from the burn-in literals
    assert sum(int(m.sum()) for m in mask) == hand == sum(int(s.nis_keep.sum()) for s in p1.segments)
    kept_filter = np.concatenate([s.sq_err[m] for s, m in zip(p1.segments, mask)])
    fit = baseline.VarFit(1, np.array([[0.0, 0.0], [0.9, 0.0], [0.0, 0.9]]), True)          # any fixed VAR
    scored = baseline.score_recording(fit, {"segments": rec["segments"], "starts": rec["starts"]}, CFG)
    assert int(scored["mask"].sum()) == hand == kept_filter.size
    assert np.isfinite(scored["err"][scored["mask"]]).all()


def test_mask_ignores_divergence_while_the_filters_own_keep_mask_does_not():
    nan_seg = np.full((2, 1000), np.nan)                        # diverges at step 0 (NaN/Inf)
    ok = np.random.default_rng(2).normal(size=(2, 3000)) * 2.0 + 0.0
    mu = CFG["rescaling"]["mu_ref"]
    ok = ok + mu
    p1 = passes.run_pass1([nan_seg, ok], [0, 2000], CFG, 1.0e-2, forward_only=True, filter_name="19D")
    assert p1.segments[0].diverged and not p1.segments[1].diverged
    m = passes.scoring_mask([1000, 3000], CFG)
    assert int(m[1].sum()) == 3000 - 536                        # cumulative index 1000 + i >= 1536 -> i >= 536
    assert int(p1.segments[1].nis_keep.sum()) == 3000 - 1536    # the filter's counter skips the diverged segment
    assert int(passes.scoring_mask([3000], CFG)[0].sum()) == 3000 - 1536


def test_burn_in_shorter_than_the_largest_order_raises_not_skips():
    c2 = copy.deepcopy(CFG)
    c2["windows"]["training_burn_in_s"] = 0.05                  # 13 samples < 32
    rng = np.random.default_rng(1)       # the second segment starts after the 6 s burn-in, so its first scored sample is 13
    rec = {"id": "r", "segments": [rng.normal(size=(2, 2000)), rng.normal(size=(2, 2000))], "starts": [0, 3000]}
    fit = baseline.VarFit(1, np.zeros((3, 2)), True)
    with pytest.raises(baseline.BaselineError, match="burn-in"):
        baseline.score_recording(fit, rec, c2)
    with pytest.raises(baseline.BaselineError, match="burn-in"):
        baseline.cv_select_order(recs_from(A3[:1], np.zeros(2), 6, 2, 2000, 1), c2)


def test_first_order_samples_of_each_segment_have_no_prediction():
    seg = np.random.default_rng(4).normal(size=(2, 50))
    fit = baseline.VarFit(3, np.zeros((7, 2)), True)
    pred = baseline.predict_segment(fit, seg)
    assert np.isnan(pred[:, :3]).all() and np.isfinite(pred[:, 3:]).all()
    scored = baseline.score_recording(fit, {"segments": [seg, seg], "starts": [0, 200]}, CFG)
    assert np.isnan(scored["err"][:3]).all() and np.isnan(scored["err"][50:53]).all()
    assert not scored["mask"].any()                             # 100 samples: all inside the burn-ins


# ---- scoring convention ----------------------------------------------------------------------------------------------

def test_subject_score_is_the_mean_over_masked_samples_and_raises_on_a_missing_prediction():
    e = [np.array([1., 2., 3., 4.]), np.array([5., 6.])]
    k = [np.array([False, True, True, False]), np.array([True, True])]
    assert passes.subject_score(e, k) == pytest.approx((2 + 3 + 5 + 6) / 4)
    with pytest.raises(passes.PassError):
        passes.subject_score([np.array([1., np.nan, 3.])], [np.array([True, True, False])])       # NaN on a scored sample
    with pytest.raises(passes.PassError):
        passes.subject_score([np.array([1., 2.])], [np.array([False, False])])                    # nothing scored


# ---- M0 versus M0b ---------------------------------------------------------------------------------------------------

def test_m0b_is_better_in_sample_and_both_match_the_analytic_errors():
    a_train, a_test, sigma = 0.5, 0.8, 1.0
    cfg2 = cfg_with(var_order_range=[1, 1], var_cv_folds=3)
    train = recs_from([np.diag([a_train, a_train])], np.zeros(2), n_sub=6, n_seg=2, n=6000, seed=21)
    test = recs_from([np.diag([a_test, a_test])], np.zeros(2), n_sub=1, n_seg=2, n=6000, seed=22, prefix="t")[0]
    fit, _ = baseline.fit_m0(train, cfg2)
    s0 = baseline.subject_mse(baseline.score_recording(fit, test, cfg2))
    s0b = baseline.subject_mse(baseline.score_recording(baseline.fit_recording(test, fit.order, cfg2), test, cfg2))
    expect_m0 = sigma ** 2 + (a_test - a_train) ** 2 * sigma ** 2 / (1 - a_test ** 2)     # = 1.25
    assert s0 == pytest.approx(expect_m0, rel=0.04)
    assert s0b == pytest.approx(sigma ** 2, rel=0.03) and s0b < 0.9 * s0


def test_m0b_on_the_training_dynamics_is_close_to_m0():
    cfg2 = cfg_with(var_order_range=[1, 1], var_cv_folds=3)
    train = recs_from([np.diag([0.5, 0.5])], np.zeros(2), n_sub=6, n_seg=2, n=6000, seed=23)
    test = recs_from([np.diag([0.5, 0.5])], np.zeros(2), n_sub=1, n_seg=2, n=6000, seed=24, prefix="t")[0]
    fit, _ = baseline.fit_m0(train, cfg2)
    s0 = baseline.subject_mse(baseline.score_recording(fit, test, cfg2))
    s0b = baseline.subject_mse(baseline.score_recording(baseline.fit_recording(test, 1, cfg2), test, cfg2))
    assert abs(s0 - s0b) < 0.03 * s0


# ---- config is read ------------------------------------------------------------------------------------------------------

def test_baseline_config_is_read_not_hard_coded():
    recs = recs_from(A3[:1], np.zeros(2), n_sub=6, n_seg=1, n=2500, seed=31)
    a = baseline.cv_select_order(recs, cfg_with(var_order_range=[1, 2], cv_seed=1))
    b = baseline.cv_select_order(recs, cfg_with(var_order_range=[1, 2], cv_seed=2))
    assert a.folds != b.folds and a.seed == 1 and b.seed == 2
    with pytest.raises(baseline.BaselineError):
        baseline.cv_select_order(recs, cfg_with(cv_aggregation="pooled"))
    with pytest.raises(baseline.BaselineError):
        baseline.cv_select_order(recs, cfg_with(cv_seed=None))
    no_ic = baseline.fit_pooled(recs, 1, cfg_with(var_order_range=[1, 2], intercept=False))
    assert no_ic.coef.shape == (2, 2) and np.all(no_ic.c == 0.0)
    assert baseline.fit_pooled(recs, 1, cfg_with(var_order_range=[1, 2])).coef.shape == (3, 2)


def test_pilot_internal_split_is_the_seeded_permutation_from_config():
    ids = [f"sub-{i:03d}" for i in (5, 8, 19, 51, 56, 63, 74, 81, 82, 86, 88, 107)]
    for seed in (42, 7):
        tr, te = baseline.pilot_internal_split(ids, cfg_with(pilot_internal_seed=seed))
        perm = np.random.default_rng(seed).permutation(12)          # expected, written here from the rule
        assert tr == sorted(sorted(ids)[i] for i in perm[:8]) and te == sorted(sorted(ids)[i] for i in perm[8:])
    assert baseline.pilot_internal_split(ids, cfg_with(pilot_internal_seed=42)) != baseline.pilot_internal_split(
        ids, cfg_with(pilot_internal_seed=7))
    with pytest.raises(baseline.BaselineError):
        baseline.pilot_internal_split(ids[:10], CFG)


def test_rank_deficient_design_raises():
    z = np.random.default_rng(1).normal(size=(1, 400))
    rec = {"id": "r", "segments": [np.vstack([z, 2.0 * z])], "starts": [0]}      # channel 2 = 2 x channel 1
    with pytest.raises(baseline.BaselineError, match="rank deficient"):
        baseline.fit_pooled([rec], 2, cfg_with(var_order_range=[1, 2]))


# ---- filter results carry z_pred and sq_err ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["19D", "A", "B"])
@pytest.mark.parametrize("forward_only", [True, False])
def test_every_filter_result_carries_prediction_and_squared_error(name, forward_only):
    rec = make_recording(CFG, (5,), seed=51, g12=5.0, g21=2.0, m=0.3, p=np.array([220.0, 250.0]))
    p1 = passes.run_pass1(rec["segments"], rec["starts"], CFG, 1.0e-2, forward_only=forward_only, filter_name=name)
    s = p1.segments[0]
    assert not s.diverged and s.z_pred.shape == (1280, 2) and s.sq_err.shape == (1280,)
    sq = passes.filter_sq_errors(p1)
    assert len(sq) == 1 and sq[0] is s.sq_err


def test_scoring_raises_when_the_filter_side_error_is_absent():
    ok = SimpleNamespace(diverged=False, sq_err=np.zeros(5), index=0)
    gone = SimpleNamespace(diverged=False, sq_err=None, index=1)
    dropped = SimpleNamespace(diverged=True, sq_err=None, index=2)
    assert passes.filter_sq_errors(SimpleNamespace(segments=[ok, dropped])) == [ok.sq_err, None]
    with pytest.raises(passes.PassError, match="segment 1"):
        passes.filter_sq_errors(SimpleNamespace(segments=[ok, gone]))


# ---- the driver: outputs, pilot interface, gate -------------------------------------------------------------------------

def _fake_world(monkeypatch, n_train, n_test, n_seg=1, n=2500, seed=0):
    import main as main_mod
    train = [f"sub-{i:03d}" for i in range(1, n_train + 1)]
    test = [f"sub-{i:03d}" for i in range(n_train + 1, n_train + n_test + 1)]
    rng = np.random.default_rng(seed)
    A = [np.array([[0.5, 0.1], [0.0, 0.4]])]
    data = {s: {"reason": None, "segments": [sim_var(A, np.zeros(2), n, rng) for _ in range(n_seg)],
                "starts": [k * (n + 500) for k in range(n_seg)]} for s in train + test}
    asked = []

    def loader(s):
        asked.append(s)
        return data[s]
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(train))
    monkeypatch.setattr(main_mod, "test_subjects", lambda seed, root=None, cfg=None: list(test))
    return train, test, loader, asked


def test_pilot_run_uses_only_pilot_subjects_and_the_8_4_split(temp_root, monkeypatch):
    train, test, loader, asked = _fake_world(monkeypatch, 20, 10, seed=61)
    pilot = train[:12]
    res = baseline.run_baseline(CFG, temp_root, 42, pilot=True, loader=loader, pilot_ids=pilot)
    tr8, te4 = baseline.pilot_internal_split(pilot, CFG)
    assert len(tr8) == 8 and len(te4) == 4 and set(tr8) | set(te4) == set(pilot) and not set(tr8) & set(te4)
    assert set(asked) == set(pilot)                              # no subject outside the pilot set, test side never read
    assert res["doc"]["fit"]["subjects"] == tr8 and res["doc"]["scored_subjects"] == te4
    assert res["doc"]["pilot"] and res["doc"]["mechanics_only"]
    vpath, npath = res["paths"]
    assert vpath.parent == temp_root / "results" / "pilot" and vpath.is_file() and npath.is_file()
    assert not (temp_root / "outputs" / "baseline_var_42.json").exists()


def test_the_same_driver_runs_unchanged_on_78_33_and_writes_the_format_stage_g_reads(temp_root, monkeypatch):
    train, test, loader, asked = _fake_world(monkeypatch, 78, 33, seed=62)
    (temp_root / "outputs").mkdir()
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": False}))
    res = baseline.run_baseline(CFG, temp_root, 42, loader=loader)
    doc = res["doc"]
    assert doc["fit"]["n_subjects"] == 78 and len(doc["scored_subjects"]) == 33 and not doc["pilot"]
    assert doc["cv"]["n_folds"] == 5 and sorted(len(f) for f in doc["cv"]["folds"]) == [15, 15, 16, 16, 16]
    assert doc["order"] in range(1, 33) and len(doc["A"]) == doc["order"] and len(doc["intercept"]) == 2
    assert doc["order"] == res["fit"].order and len(doc["cv"]["mse_by_order"]) == 32
    on_disk = json.loads(res["paths"][0].read_text(encoding="utf-8"))
    assert on_disk["order"] == doc["order"] and "provenance" in on_disk
    sc = baseline.load_scores(res["paths"][1])
    assert sorted(sc) == sorted(test)
    one = sc[test[0]]
    assert set(one) == {"starts", "lengths", "mask", "M0", "M0b"}
    assert one["M0"].shape == one["mask"].shape == (int(one["lengths"].sum()),)
    assert passes.subject_score([one["M0"]], [one["mask"]]) == pytest.approx(doc["summary"]["M0"]["per_subject"][test[0]])
    # the order is the one with the smallest CV error, the headline is the mean over subjects
    assert doc["order"] == doc["cv"]["orders"][int(np.argmin(doc["cv"]["mse_by_order"]))]
    per = doc["summary"]["M0"]["per_subject"]
    assert doc["summary"]["M0"]["mean"] == pytest.approx(np.mean(list(per.values())))


def test_outputs_are_written_once_and_a_different_result_is_refused(temp_root, monkeypatch):
    train, test, loader, _ = _fake_world(monkeypatch, 20, 10, seed=63)
    pilot = train[:12]
    first = baseline.run_baseline(CFG, temp_root, 42, pilot=True, loader=loader, pilot_ids=pilot)
    before = first["paths"][0].read_bytes()
    again = baseline.run_baseline(CFG, temp_root, 42, pilot=True, loader=loader, pilot_ids=pilot)
    assert again["status"] == "unchanged" and first["paths"][0].read_bytes() == before
    train2, test2, loader2, _ = _fake_world(monkeypatch, 20, 10, seed=64)
    with pytest.raises(baseline.BaselineError, match="force"):
        baseline.run_baseline(CFG, temp_root, 42, pilot=True, loader=loader2, pilot_ids=train2[:12])
    forced = baseline.run_baseline(CFG, temp_root, 42, pilot=True, loader=loader2, pilot_ids=train2[:12], force=True)
    assert forced["status"] == "written"


def test_a_real_fit_needs_a_gate_that_permits_it_but_pilot_does_not(temp_root, monkeypatch):
    train, test, loader, asked = _fake_world(monkeypatch, 20, 10, seed=65)
    with pytest.raises(tuning.GateError):
        baseline.run_baseline(CFG, temp_root, 42, loader=loader)
    assert asked == []                                           # nothing was read before the gate refused
    (temp_root / "outputs").mkdir()
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": True}))
    with pytest.raises(tuning.GateError):
        baseline.run_baseline(CFG, temp_root, 42, loader=loader)
    baseline.run_baseline(CFG, temp_root, 42, pilot=True, loader=loader, pilot_ids=train[:12])      # no gate needed


def test_held_out_recordings_are_read_after_the_frozen_fit(temp_root, monkeypatch):
    train, test, loader, asked = _fake_world(monkeypatch, 20, 10, seed=66)
    (temp_root / "outputs").mkdir()
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": False}))
    real_fit = baseline.fit_m0
    seen = {}

    def spy(recs, cfg):
        seen["asked_at_fit"] = list(asked)
        return real_fit(recs, cfg)
    monkeypatch.setattr(baseline, "fit_m0", spy)
    baseline.run_baseline(CFG, temp_root, 42, loader=loader)
    assert set(seen["asked_at_fit"]) == set(train) and not set(seen["asked_at_fit"]) & set(test)


def test_source_has_no_print_or_global_random_state():
    import re
    from pathlib import Path
    text = (Path(baseline.__file__)).read_text(encoding="utf-8")
    assert not re.search(r"(?<![A-Za-z_])print\(", text)
    assert "np.random.seed" not in text and "np.random.rand" not in text and "np.random.normal" not in text
    import ast
    top = [n for n in ast.parse(text).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in top for a in n.names} | {n.module for n in top if isinstance(n, ast.ImportFrom)}
    assert not ({"src.model", "model", "src.preprocess", "preprocess", "pysr"} & names)
    assert ss.N_NODES == 2
