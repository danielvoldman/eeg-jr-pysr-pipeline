"""Stage D1 tests (PLAN.md D1; §8.2, §8.3, §8.4, §9.1, §10.2; IMP-053 to IMP-058).

Expected values are independent of the code under test: closed forms, explicit loops written here,
np.polyfit, scipy.optimize for the TV objective, hand formulas with the model constants taken from config.
No Julia, no PySR, no real data, no test subject.
"""
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import regression as R  # noqa: E402
from src.config import load_config  # noqa: E402
import planted_sim as ps  # noqa: E402

CFG = load_config()
FS = CFG["preprocessing"]["observation_fs_hz"]
NS = 6                                                   # states per node


# ---- independent helpers -----------------------------------------------------------------------------

def brute_kernel(support_ms):
    """Gaussian weights g_k, K and sigma, written out from the definition (sigma = h / 3)."""
    h = support_ms / 2000.0
    K = int(math.floor(h * FS))
    sigma = h / 3.0
    k = np.arange(-K, K + 1)
    g = np.exp(-((k / FS) ** 2) / (2 * sigma ** 2))
    return K, k, g


def brute_slope(y, i, support_ms):
    K, k, g = brute_kernel(support_ms)
    num = sum(g[q] * k[q] * y[i + k[q]] for q in range(len(k)))
    den = sum(g[q] * k[q] ** 2 for q in range(len(k)))
    return FS * num / den


def brute_smooth(f, i, support_ms):
    K, k, g = brute_kernel(support_ms)
    phi = g - math.exp(-4.5)                              # g(h) = exp(-(h/sigma)^2 / 2) = exp(-4.5)
    return sum(phi[q] * f[i + k[q]] for q in range(len(k))) / phi.sum()


def hand_sigmoid(v):
    jr = CFG["jansen_rit"]
    return 2 * jr["e0"] / (1 + math.exp(jr["r"] * (jr["v0"] - v)))


def analytic_window(n=80):
    """Analytic two-node states with known derivative of y4, and a hand-computed base prediction.
    Returns x (n, 12), s (n, 2), params, f_base (n, 2), y4dot (n, 2)."""
    t = np.arange(n) / FS
    w = 2 * np.pi
    x = np.zeros((n, 12))
    y4dot = np.zeros((n, 2))
    s = np.zeros((n, 2))
    for j in (0, 1):
        x[:, NS * j + 0] = 2.0 + np.sin(w * 3 * t + j)
        x[:, NS * j + 1] = 1.0 + 0.8 * np.sin(w * 5 * t + 0.4 * j)
        x[:, NS * j + 2] = 0.5 + 0.3 * np.cos(w * 7 * t + j)
        x[:, NS * j + 4] = 4.0 * np.sin(w * 6 * t + 0.2 + j)
        y4dot[:, j] = 4.0 * w * 6 * np.cos(w * 6 * t + 0.2 + j)
        s[:, j] = 2.5 + 1.2 * np.sin(w * 4 * t + j)
    params = SimpleNamespace(p1=210.0, p2=240.0, A1=3.2, A2=3.6, g12=7.0, g21=19.0)
    jr = CFG["jansen_rit"]
    a, C = jr["a"], jr["C"]
    C1, C2 = C * jr["C1_multiplier"], C * jr["C2_multiplier"]
    f = np.zeros((n, 2))
    for j, (p, A, g, src) in enumerate(((params.p1, params.A1, params.g21, 1),
                                        (params.p2, params.A2, params.g12, 0))):
        for i in range(n):
            y0, y1, y4 = x[i, NS * j], x[i, NS * j + 1], x[i, NS * j + 4]
            f[i, j] = A * a * (p + C2 * hand_sigmoid(C1 * y0) + g * s[i, src]) - 2 * a * y4 - a * a * y1
    return x, s, params, f, y4dot


# ---- config ---------------------------------------------------------------------------------------------

def test_config_leaves_for_stage_d():
    rc, pc = CFG["residual"], CFG["pysr"]
    assert rc["tv_weight_grid"] == [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    assert rc["weak_sigma_divisor"] == 3 and rc["subtraction_form"] == "matched"
    assert pc["subsample"]["seed"] == 3000
    assert (pc["seeds"]["fold_offset"], pc["seeds"]["random_state_offset"], pc["seeds"]["refit_stride"]) == (2000, 4000, 100)


# ---- weak-form kernel --------------------------------------------------------------------------------------

@pytest.mark.parametrize("support,K", [(60, 7), (100, 12), (160, 20)])
def test_weak_kernel_geometry(support, K):
    k = R.weak_kernel(support, CFG)
    assert k.K == K and len(k.deriv) == 2 * K + 1
    np.testing.assert_allclose(k.deriv, -k.deriv[::-1], atol=1e-12)        # antisymmetric
    assert abs(k.smooth.sum() - 1.0) < 1e-14 and np.all(k.smooth > 0)
    assert abs(k.sigma_s - support / 2000 / 3) < 1e-15


def test_weak_derivative_exact_for_linear_and_quadratic():
    k = R.weak_kernel(100, CFG)
    t = np.arange(60) / FS
    y = 1.5 - 4.0 * t + 30.0 * t ** 2
    d = R.weak_derivative(y, k)
    np.testing.assert_allclose(d, -4.0 + 60.0 * t[k.K:-k.K], rtol=1e-10, atol=1e-9)
    assert len(d) == 60 - 2 * k.K


@pytest.mark.parametrize("support", [60, 100, 160])
def test_weak_derivative_equals_weighted_least_squares_slope(support):
    rng = np.random.default_rng(1)
    y = rng.normal(size=70)
    k = R.weak_kernel(support, CFG)
    K, kk, g = brute_kernel(support)
    d = R.weak_derivative(y, k)
    for i in range(K, 70 - K):
        slope = np.polyfit(kk / FS, y[i - K:i + K + 1], 1, w=np.sqrt(g))[0]     # independent implementation
        assert d[i - K] == pytest.approx(slope, rel=1e-9, abs=1e-9)
        assert d[i - K] == pytest.approx(brute_slope(y, i, support), rel=1e-12, abs=1e-12)


def test_weak_derivative_sinusoid_gain_is_the_gaussian_response():
    k = R.weak_kernel(100, CFG)
    t = np.arange(200) / FS
    f = 6.0
    y = np.sin(2 * np.pi * f * t)
    d = R.weak_derivative(y, k)
    gain = math.exp(-0.5 * (2 * np.pi * f * k.sigma_s) ** 2)               # closed form of the Gaussian response
    expect = gain * 2 * np.pi * f * np.cos(2 * np.pi * f * t[k.K:-k.K])
    assert np.max(np.abs(d - expect)) < 0.02 * 2 * np.pi * f                # tail beyond 2.8 sigma is cut


def test_weak_derivative_is_local_and_never_padded():
    k = R.weak_kernel(100, CFG)
    rng = np.random.default_rng(2)
    y = rng.normal(size=60)
    d0 = R.weak_derivative(y, k)
    y2 = y.copy()
    y2[5] += 100.0
    d1 = R.weak_derivative(y2, k)
    changed = np.nonzero(np.abs(d1 - d0) > 1e-12)[0] + k.K                   # sample index of each changed row
    assert list(changed) == list(range(k.K, 5 + k.K + 1))                   # exactly the rows whose kernel holds sample 5
    with pytest.raises(R.RegressionError):
        R.weak_derivative(np.zeros(2 * k.K), k)


def test_weak_kernel_rejects_support_below_one_sample():
    with pytest.raises(R.RegressionError):
        R.weak_kernel(2, CFG)


# ---- TV derivative ---------------------------------------------------------------------------------------------

def tv_objective(u, yp, alpha, eps):
    n = len(u)
    A = np.zeros((n, n))
    for i in range(1, n):
        A[i, 0] += 0.5
        A[i, 1:i] += 1.0
        A[i, i] += 0.5
    du = np.diff(u)
    return 0.5 / n * np.sum((A @ u - yp) ** 2) + alpha / (n - 1) * np.sum(np.sqrt(du ** 2 + eps ** 2))


def kink_signal(n=100):
    i = np.arange(n)
    return np.where(i < 50, 2.0 * i, 100.0 - (i - 50.0))                    # slopes +2 and -1 per sample


def test_tv_recovers_piecewise_linear_slopes():
    y = kink_signal()
    d, info = R.tv_derivative(y, FS, CFG["residual"]["tv_weight_grid"][0], CFG)
    per_sample = d / FS
    assert np.all(np.abs(per_sample[10:40] - 2.0) < 0.1) and np.all(np.abs(per_sample[60:90] + 1.0) < 0.1)
    assert info["converged"]


def test_tv_exact_for_linear_signal_and_scale_free():
    i = np.arange(80)
    y = 0.7 * i + 3.0
    for a in CFG["residual"]["tv_weight_grid"]:
        d, _ = R.tv_derivative(y, FS, a, CFG)
        np.testing.assert_allclose(d, 0.7 * FS, rtol=1e-3)
    y = kink_signal()
    d1, _ = R.tv_derivative(y, FS, 1e-3, CFG)
    d7, _ = R.tv_derivative(7.0 * y, FS, 1e-3, CFG)
    np.testing.assert_allclose(d7, 7.0 * d1, rtol=1e-6)


def test_tv_objective_matches_independent_minimiser():
    rng = np.random.default_rng(3)
    n = 60
    y = np.sin(2 * np.pi * 4 * np.arange(n) / FS) + 0.02 * rng.normal(size=n)
    alpha = 1e-2
    rc = copy.deepcopy(CFG)
    rc["residual"]["tv_max_iter"], rc["residual"]["tv_tol"] = 500, 1e-6
    d, info = R.tv_derivative(y, FS, alpha, rc)
    yp = (y - y[0]) / y.std()
    eps = CFG["residual"]["tv_eps"]
    res = minimize(tv_objective, np.gradient(yp), args=(yp, alpha, eps), method="L-BFGS-B",
                   options={"maxiter": 20000, "maxfun": 10 ** 6, "ftol": 1e-15, "gtol": 1e-12})
    u_hat = d / (FS * y.std())
    assert tv_objective(u_hat, yp, alpha, eps) == pytest.approx(info["objective"], rel=1e-9)
    assert info["objective"] <= res.fun * (1 + 1e-4)                            # at least as good as L-BFGS
    assert np.sqrt(np.mean((u_hat - res.x) ** 2)) < 0.05 * np.std(res.x)


def test_tv_more_weight_means_less_total_variation():
    rng = np.random.default_rng(4)
    y = np.sin(2 * np.pi * 4 * np.arange(80) / FS) + 0.05 * rng.normal(size=80)
    tvs = [np.sum(np.abs(np.diff(R.tv_derivative(y, FS, a, CFG)[0]))) for a in CFG["residual"]["tv_weight_grid"]]
    assert all(tvs[i + 1] <= tvs[i] * (1 + 1e-9) for i in range(len(tvs) - 1)) and tvs[-1] < tvs[0]


def test_tv_rejects_bad_input_and_constant_series_is_zero():
    with pytest.raises(R.RegressionError):
        R.tv_derivative(np.zeros((4, 4)), FS, 1e-3, CFG)
    d, info = R.tv_derivative(np.full(30, 2.0), FS, 1e-3, CFG)
    assert np.all(d == 0.0) and info["converged"]


# ---- residual construction and rows ------------------------------------------------------------------------------

def test_window_rows_match_brute_force_for_matched_and_literal():
    x, s, params, f, _ = analytic_window()
    n = x.shape[0]
    K = 12
    for form in ("matched", "literal"):
        rows = R.window_rows(x, s, params, CFG, estimator="weak_form", support_ms=100, form=form,
                             window_start=1000, subject="sub-001")
        assert len(rows) == 2 * (n - 2 * K)
        for j in (0, 1):
            src = 1 - j
            blk = slice(j * (n - 2 * K), (j + 1) * (n - 2 * K))
            y4 = x[:, NS * j + 4]
            pot_j = x[:, NS * j + 1] - x[:, NS * j + 2]
            pot_s = x[:, NS * src + 1] - x[:, NS * src + 2]
            for q, i in enumerate(range(K, n - K)):
                slope = brute_slope(y4, i, 100)
                base = brute_smooth(f[:, j], i, 100) if form == "matched" else f[i, j]
                assert rows.y[blk][q] == pytest.approx(slope - base, rel=1e-9, abs=1e-7)
                assert rows.X_raw[blk][q, 0] == pytest.approx(pot_j[i], abs=1e-12)
                assert rows.X_raw[blk][q, 1] == pytest.approx(0.5 * (pot_s[i - 2] + pot_s[i - 3]), abs=1e-12)
                assert rows.X_raw[blk][q, 2] == pytest.approx(s[i, src], abs=1e-12)
        assert set(rows.node) == {0, 1} and np.all(rows.window == 1000) and set(rows.subject) == {"sub-001"}
        assert rows.t.min() == K and rows.t.max() == n - K - 1


def test_matched_form_follows_the_true_residual_and_literal_does_not():
    x, s, params, f, y4dot = analytic_window(n=120)
    K = 12
    rows_m = R.window_rows(x, s, params, CFG, support_ms=100, form="matched")
    rows_l = R.window_rows(x, s, params, CFG, support_ms=100, form="literal")
    n = x.shape[0]
    r_true = y4dot - f                                                     # analytic dy4/dt minus the hand f_base
    scale = np.sqrt(np.mean(y4dot ** 2))
    err_m, err_l = [], []
    for j in (0, 1):
        blk = slice(j * (n - 2 * K), (j + 1) * (n - 2 * K))
        smooth_truth = np.array([brute_smooth(r_true[:, j], i, 100) for i in range(K, n - K)])
        err_m.append(np.sqrt(np.mean((rows_m.y[blk] - smooth_truth) ** 2)))
        err_l.append(np.sqrt(np.mean((rows_l.y[blk] - r_true[K:n - K, j]) ** 2)))
    assert max(err_m) < 5e-3 * scale                                        # matched: discretisation only
    assert min(err_l) > 5e-2 * scale                                        # literal: the spurious (1 - H) f_base


def test_edge_rows_margin_and_window_locality():
    x, s, params, *_ = analytic_window(n=90)
    rows = R.window_rows(x, s, params, CFG, support_ms=60, margin=20)
    assert rows.t.min() == 20 and rows.t.max() == 69 and len(rows) == 2 * 50
    # rows of a window depend on that window only
    w = lambda z: SimpleNamespace(x_smooth=z, s_delayed=s, start=0, diverged=False)
    w2 = lambda z: SimpleNamespace(x_smooth=z, s_delayed=s, start=512, diverged=False)
    x_b = x.copy()
    x_b[:, 4] += 50.0
    r1 = R.recording_rows([w(x), w2(x)], params, CFG, "sub-001")
    r2 = R.recording_rows([w(x), w2(x_b)], params, CFG, "sub-001")
    first = r1.window == 0
    np.testing.assert_array_equal(r1.y[first], r2.y[r2.window == 0])
    assert not np.allclose(r1.y[~first], r2.y[r2.window == 512])


def test_total_variation_rows_use_literal_subtraction_and_same_rows():
    x, s, params, f, _ = analytic_window(n=70)
    rows = R.window_rows(x, s, params, CFG, estimator="total_variation", tv_alpha_rel=1e-3, margin=12)
    w = R.window_rows(x, s, params, CFG, estimator="weak_form", support_ms=100, form="literal")
    assert len(rows) == len(w) and np.array_equal(rows.t, w.t) and np.array_equal(rows.node, w.node)
    np.testing.assert_allclose(rows.X_raw, w.X_raw)
    y4 = x[:, 4]
    d, _ = R.tv_derivative(y4, FS, 1e-3, CFG)
    assert rows.y[:len(rows) // 2] == pytest.approx(d[12:70 - 12] - f[12:70 - 12, 0])


def test_window_rows_reject_bad_requests_and_ignore_extra_columns():
    x, s, params, *_ = analytic_window(n=70)
    with pytest.raises(R.RegressionError):
        R.window_rows(x, s, params, CFG, support_ms=100, margin=5)            # margin < K
    with pytest.raises(R.RegressionError):
        R.window_rows(x[:20], s[:20], params, CFG, support_ms=100)            # no row left
    with pytest.raises(R.RegressionError):
        R.window_rows(x[:, :10], s, params, CFG)
    with pytest.raises(R.RegressionError):
        R.window_rows(x, s[:, :1], params, CFG)
    with pytest.raises(R.RegressionError):
        R.window_rows(x, s, params, CFG, estimator="spline")
    with pytest.raises(R.RegressionError):
        R.window_rows(x, s, params, CFG, estimator="total_variation")
    with pytest.raises(R.RegressionError):
        R.window_rows(x, s, params, CFG, form="weird")
    extra = np.column_stack([x, np.full((70, 9), 1e6)])                        # a 21-D layout's extra states
    a = R.window_rows(x, s, params, CFG)
    b = R.window_rows(extra, s, params, CFG)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.X_raw, b.X_raw)


def test_recording_rows_skip_diverged_windows():
    x, s, params, *_ = analytic_window(n=70)
    ok = SimpleNamespace(x_smooth=x, s_delayed=s, start=0, diverged=False)
    bad = SimpleNamespace(x_smooth=None, s_delayed=None, start=512, diverged=True)
    rows = R.recording_rows([ok, bad], params, CFG, "sub-002")
    assert set(rows.window) == {0}


# ---- z-scoring ---------------------------------------------------------------------------------------------------

def _rows(rng, n, shift=0.0, subject="s"):
    X = rng.normal(loc=[1.0 + shift, -2.0, 0.5], scale=[2.0, 0.3, 1.5], size=(n, 3))
    y = 5.0 + 3.0 * X[:, 0] * X[:, 1] + rng.normal(size=n)
    return R.Rows(X, y, np.zeros(n, int), np.zeros(n, np.int64), np.arange(n), np.full(n, subject))


def test_zscore_matches_numpy_over_the_concatenation():
    rng = np.random.default_rng(5)
    parts = [_rows(rng, 300, 0.0), _rows(rng, 50, 4.0), _rows(rng, 700, -2.0)]
    z = R.fit_zscore(parts)
    allX = np.concatenate([p.X_raw for p in parts])
    ally = np.concatenate([p.y for p in parts])
    np.testing.assert_allclose(z.mean_X, allX.mean(0), rtol=1e-12)
    np.testing.assert_allclose(z.sd_X, allX.std(0), rtol=1e-12)
    assert z.mean_y == pytest.approx(ally.mean(), rel=1e-12) and z.sd_y == pytest.approx(ally.std(), rel=1e-12)
    assert z.n == 1050
    assert set(z.to_dict()) >= {"mean_X", "sd_X", "mean_y", "sd_y", "n"}


def test_design_uses_the_fit_constants_and_forms_products_from_z_columns():
    rng = np.random.default_rng(6)
    fit = _rows(rng, 500)
    val = _rows(rng, 200, shift=3.0)
    z = R.fit_zscore([fit])
    X, y = R.design(val, z)
    Xz = (val.X_raw - z.mean_X) / z.sd_X
    np.testing.assert_allclose(X[:, :3], Xz)
    np.testing.assert_allclose(X[:, 3], Xz[:, 0] * Xz[:, 1])
    np.testing.assert_allclose(X[:, 4], Xz[:, 0] * Xz[:, 2])
    np.testing.assert_allclose(X[:, 5], Xz[:, 1] * Xz[:, 2])
    np.testing.assert_allclose(y, (val.y - z.mean_y) / z.sd_y)
    assert abs(X[:, 0].mean()) > 0.5                                        # validation is NOT centred by its own mean
    assert R.VARIABLE_NAMES == ("u_tgt", "u_src", "S_src", "u_tgt_u_src", "u_tgt_S_src", "u_src_S_src")


def test_zscore_rejects_degenerate_input():
    rng = np.random.default_rng(7)
    r = _rows(rng, 10)
    r.X_raw[:, 1] = 3.0
    with pytest.raises(R.RegressionError):
        R.fit_zscore([r])
    with pytest.raises(R.RegressionError):
        R.fit_zscore([])


# ---- seeds, folds, ensemble, subsample -----------------------------------------------------------------------------

def make_split(n_train=78, n_test=33):
    return {"train": [f"sub-{i:03d}" for i in range(1, n_train + 1)],
            "test": [f"sub-{i:03d}" for i in range(500, 500 + n_test)]}


def test_seeds_for_adds_offsets_split_seed_and_refit_stride():
    assert R.seeds_for(CFG, 42) == {"fold": 2042, "subsample": 3042, "random_state": 4042}
    assert R.seeds_for(CFG, 43, k=3) == {"fold": 2343, "subsample": 3343, "random_state": 4343}


def test_subject_folds_sizes_disjoint_deterministic_and_guarded():
    split = make_split()
    fit, val = R.split_subjects_fit_val(split["train"], split, 2042, CFG)
    assert (len(fit), len(val)) == (62, 16)                                 # round(0.8 x 78) = 62
    assert not set(fit) & set(val) and sorted(fit + val) == sorted(split["train"])
    assert R.split_subjects_fit_val(split["train"], split, 2042, CFG) == (fit, val)
    assert R.split_subjects_fit_val(split["train"], split, 2043, CFG) != (fit, val)
    f3, v3 = R.split_subjects_fit_val(split["train"][:7], split, 1, CFG)
    assert (len(f3), len(v3)) == (6, 1)                                     # 0.8 x 7 = 5.6 rounds up to 6


def test_no_test_subject_can_enter_any_fold_or_ensemble():
    split = make_split()
    bad = split["train"][:10] + [split["test"][0]]
    with pytest.raises(R.RegressionError, match="not training subjects"):
        R.split_subjects_fit_val(bad, split, 1, CFG)
    with pytest.raises(R.RegressionError, match="not training subjects"):
        R.draw_ensemble(bad, split, 42, CFG)
    with pytest.raises(R.RegressionError):
        R.split_subjects_fit_val(["sub-999"], split, 1, CFG)                # unknown IDs are refused too
    R.check_training_ids(split["train"], split)


def test_window_time_split_takes_the_last_fifth():
    starts = [512 * i for i in range(120)]
    fit, val = R.split_windows_time(reversed(starts), CFG)
    assert fit == starts[:96] and val == starts[96:]                        # 96 / 24, time order
    fit, val = R.split_windows_time(starts[:117], CFG)
    assert len(val) == 23 and val[0] > fit[-1]                              # round(23.4) = 23
    with pytest.raises(R.RegressionError):
        R.split_windows_time([0], CFG)


def test_ensemble_draw_halves_folds_and_determinism():
    split = make_split()
    ens = R.draw_ensemble(split["train"], split, 42, CFG)
    assert len(ens) == 25 and [e["k"] for e in ens] == list(range(1, 26))
    for e in ens:
        assert len(e["subjects"]) == 39 and (len(e["fit"]), len(e["val"])) == (31, 8)
        assert set(e["fit"]) | set(e["val"]) == set(e["subjects"]) and not set(e["fit"]) & set(e["val"])
        assert set(e["subjects"]) <= set(split["train"])
        assert e["seeds"] == R.seeds_for(CFG, 42, e["k"])
    assert len({tuple(e["subjects"]) for e in ens}) == 25
    assert R.draw_ensemble(split["train"], split, 42, CFG)[4] == ens[4]
    assert len(R.draw_ensemble(split["train"], split, 42, CFG, n_refits=3)) == 3
    assert R.draw_ensemble(split["train"], split, 43, CFG)[0]["subjects"] != ens[0]["subjects"]


def test_subsample_quotas_equal_remainder_and_short_subjects():
    rng = np.random.default_rng(0)
    q = R.subsample_quotas({f"s{i}": 10_000 for i in range(62)}, 50_000, rng)
    assert sum(q.values()) == 50_000 and set(q.values()) == {806, 807}     # 50000 = 62 x 806 + 28
    assert sum(1 for v in q.values() if v == 807) == 28
    q = R.subsample_quotas({"a": 100, "b": 5000, "c": 5000}, 6000, np.random.default_rng(0))
    assert q == {"a": 100, "b": 2950, "c": 2950}                            # the short subject gives everything
    q = R.subsample_quotas({"a": 10, "b": 20}, 100, np.random.default_rng(0))
    assert q == {"a": 10, "b": 20}                                          # fewer rows than requested: all taken


def test_subsample_rows_counts_reproducibility_and_streams():
    rng = np.random.default_rng(8)
    by_sub = {f"sub-{i:03d}": _rows(rng, 400 + 10 * i, subject=f"sub-{i:03d}") for i in range(10)}
    out = R.subsample_rows(by_sub, 1000, seed=3042, stream=0)
    counts = {s: int(np.sum(out.subject == s)) for s in by_sub}
    assert len(out) == 1000 and set(counts.values()) == {100}
    again = R.subsample_rows(by_sub, 1000, seed=3042, stream=0)
    np.testing.assert_array_equal(out.X_raw, again.X_raw)
    assert not np.array_equal(out.X_raw, R.subsample_rows(by_sub, 1000, seed=3043, stream=0).X_raw)
    assert not np.array_equal(out.X_raw, R.subsample_rows(by_sub, 1000, seed=3042, stream=1).X_raw)
    for s in by_sub:                                                        # no duplicated rows, original order kept
        t = out.t[out.subject == s]
        assert len(set(t)) == len(t) and np.all(np.diff(t) > 0)


def test_fit_and_validation_subsamples_draw_from_disjoint_subjects():
    split = make_split()
    fit, val = R.split_subjects_fit_val(split["train"], split, 2042, CFG)
    rng = np.random.default_rng(9)
    by = {s: _rows(rng, 60, subject=s) for s in split["train"]}
    a = R.subsample_rows({s: by[s] for s in fit}, 3000, 3042, 0)
    b = R.subsample_rows({s: by[s] for s in val}, 800, 3042, 1)
    assert set(a.subject).isdisjoint(set(b.subject)) and len(a) == 3000 and len(b) == 800


# ---- recovery of a planted product, PySR-free ----------------------------------------------------------------------------

def test_planted_product_recovered_on_clean_rows():
    rng = np.random.default_rng(10)
    n = 5000
    X = rng.normal(loc=[4.0, 4.0, 2.5], scale=[1.0, 1.0, 1.0], size=(n, 3))
    basis = X[:, 0] * X[:, 1]
    truth = 3.0 * basis
    y = truth + 7.0 + 0.01 * rng.normal(size=n)
    nr, c = R.recovery_nrmse(y, basis, truth)
    assert nr < 0.01 and c == pytest.approx(3.0, rel=1e-2)
    nr_bad, _ = R.recovery_nrmse(rng.normal(size=n), basis, truth)
    assert nr_bad > 0.9                                                      # a rescaled-noise recovery is not recovery


def test_recovery_nrmse_definition_against_hand_value():
    basis = np.array([1.0, 2.0, 3.0, 4.0])
    truth = 2.0 * basis
    r_hat = 3.0 * basis + 5.0                                                # c_hat = 3
    nr, c = R.recovery_nrmse(r_hat, basis, truth)
    assert c == pytest.approx(3.0)
    assert nr == pytest.approx(math.sqrt(np.mean((basis) ** 2)) / np.std(truth))   # RMS(3b - 2b) / SD(2b)


def test_planted_simulation_truth_is_self_consistent():
    sim = ps.simulate_planted(CFG, 77, 10.8, 10.8, 12.0)
    lo = 4 * FS
    # c_j was set so that RMS(r_res) = 50% of RMS(drive) on the level pass; the planted RMS has the same order
    jr = CFG["jansen_rit"]
    a = CFG["jansen_rit"]["a"]
    rms_planted = np.sqrt(np.mean(sim.planted[lo:] ** 2, axis=0))
    rms_drive = 10.8 * np.sqrt(np.mean(sim.s_delayed[lo:, ::-1] ** 2, axis=0)) * jr["A"] * a
    assert np.all(rms_planted > 0.3 * rms_drive) and np.all(rms_planted < 0.8 * rms_drive)
    null = ps.simulate_planted(CFG, 77, 0.0, 0.0, 6.0, c_rel=0.0)
    assert np.all(null.planted == 0.0)


def test_recovery_on_simulated_rows_both_forms_beat_no_term():
    """True states, exact parameters, planted product at level 4 (0.25 x C2). What is measured is the
    regression step alone. The D1 report (tools/d1_estimator_report.py) found that the matched form shrinks
    the recovered coefficient (the target is smoothed by phi, the inputs are not) while the literal form does
    not; the shrinkage is asserted here because smoothing a broadband basis can only attenuate it."""
    sim = ps.simulate_planted(CFG, 201, 27.0, 27.0, 40.0)
    burn = int(round(CFG["windows"]["training_burn_in_s"] * FS))
    out, zero = {}, None
    for form in ("matched", "literal"):
        parts, bs, ts = [], [], []
        for start, xs, sd_ in ps.windows(sim, CFG):
            parts.append(R.window_rows(xs, sd_, sim.params, CFG, support_ms=100, form=form, window_start=start))
            b, tr = ps.window_truth(sim, start, xs.shape[0], burn, 12)
            bs.append(b)
            ts.append(tr)
        rows = R.Rows.concat(parts)
        truth = np.concatenate(ts)
        out[form] = R.recovery_nrmse(rows.y, np.concatenate(bs), truth)
        zero = np.sqrt(np.mean(truth ** 2)) / np.std(truth)                   # NRMSE of 'no term' (a zero function)
    (nr_m, c_m), (nr_l, c_l) = out["matched"], out["literal"]
    c_true = sim.planted[1100, 0] / sim.basis[1100, 0]
    assert nr_m < zero / 2 and nr_l < zero / 2 and nr_l < 1.0                  # both recover something; 'no term' is ~2.2
    assert 0.8 < c_l / c_true < 1.3                                           # literal: coefficient about right
    assert 0.3 < c_m / c_true < 1.0                                           # matched: attenuated by the smoothing


# ---- Pareto selection ------------------------------------------------------------------------------------------------

def E(c, eq, loss, val):
    return R.FrontEntry(complexity=c, equation=eq, loss=loss, val_loss=val)


def test_pareto_filter_keeps_strict_improvements_only():
    es = [E(1, "0.1", 1.0, 1.0), E(3, "u_src", 0.8, 0.8), E(5, "u_tgt", 0.8, 0.7), E(7, "u_tgt*u_src", 0.3, 0.3),
          E(9, "u_tgt*u_src + u_src", 0.35, 0.3)]
    assert [e.complexity for e in R.pareto_filter(es)] == [1, 3, 7]


def test_selection_picks_simplest_within_five_percent():
    front = [E(1, "0.0", 1.0, 1.0), E(3, "u_src", 0.6, 0.62), E(5, "u_tgt*u_src", 0.3, 0.30),
             E(9, "u_tgt*u_src + tanh(u_src)", 0.25, 0.29)]
    s = R.select_equation(front, CFG)
    assert (s.entry.complexity, s.no_term, s.min_val_loss) == (5, False, 0.29)       # 0.30 <= 0.29 x 1.05 = 0.3045
    front[2] = E(5, "u_tgt*u_src", 0.3, 0.31)                                         # 0.31 > 0.3045: complexity 9 wins
    assert R.select_equation(front, CFG).entry.complexity == 9
    assert R.select_equation(front, CFG).eligible == [9]


def test_selection_boundary_is_inclusive_and_exact():
    front = [E(3, "u_src", 2.0, 21.0), E(5, "u_tgt*u_src", 1.0, 20.0)]              # 20 x 1.05 = 21 exactly
    assert R.select_equation(front, CFG).entry.complexity == 3
    front = [E(3, "u_src", 2.0, 21.000000000000004), E(5, "u_tgt*u_src", 1.0, 20.0)]
    assert R.select_equation(front, CFG).entry.complexity == 5                        # one ulp beyond: excluded


def test_selection_bare_constant_is_no_term():
    front = [E(1, "0.4", 1.0, 1.0), E(3, "u_src", 0.99, 1.02), E(5, "u_tgt*u_src", 0.9, 0.99)]
    s = R.select_equation(front, CFG)
    assert s.no_term and s.entry.complexity == 1 and "constant" in s.reason            # constant is within 5% of min
    front = [E(1, "0.4", 1.0, 1.0), E(3, "u_src", 0.5, 0.5)]
    s = R.select_equation(front, CFG)
    assert not s.no_term and s.entry.equation == "u_src"
    front = [E(1, "2*0.4", 1.0, 0.4), E(3, "u_src", 0.5, 0.5)]                        # constant is the minimum itself
    assert R.select_equation(front, CFG).no_term


def test_selection_ignores_non_finite_and_flags_empty():
    front = [E(1, "0.4", 1.0, 1.0), E(3, "exp(u_src)", 0.5, float("nan")), E(5, "u_tgt*u_src", 0.3, float("inf")),
             E(7, "u_tgt_u_src", 0.2, 0.6)]
    s = R.select_equation(front, CFG)
    assert s.entry.complexity == 7 and s.min_val_loss == 0.6
    allbad = [E(1, "0.4", 1.0, float("nan")), E(3, "u_src", 0.5, None)]
    s = R.select_equation(allbad, CFG)
    assert s.no_term and s.entry is None and "finite" in s.reason


def test_equal_fit_loss_at_equal_complexity_enters_the_front_once():
    front = [E(4, "u_src", 0.6, 0.50), E(4, "u_tgt", 0.6, 0.49), E(8, "u_tgt*u_src", 0.1, 0.49)]
    # the front is strict in the fit loss: the second entry does not improve on the first and is dropped
    s = R.select_equation(front, CFG)
    assert s.entry.complexity == 4 and s.entry.equation == "u_src"


# ---- term signature ------------------------------------------------------------------------------------------------

def sig(text):
    return R.term_signatures(text)


def test_signature_spec_example_tanh_constants_drop_out():
    assert sig("tanh(1.02*u_tgt + 0.01)") == sig("tanh(u_tgt)") == frozenset({"tanh(u_tgt)"})


def test_signature_multiplicative_constants_and_sign_drop_out():
    assert sig("-2.5*tanh(u_src)") == sig("0.3*tanh(u_src)") == frozenset({"tanh(u_src)"})
    assert sig("exp(0.3*u_src + 2.0)") == frozenset({"exp(u_src)"})
    assert sig("tanh(0.5*u_src + 1.5*u_tgt - 0.2)") == frozenset({"tanh(u_src + u_tgt)"})


def test_signature_denominator_constants_are_kept_literally():
    assert sig("u_src/(u_src + 0.3)") != sig("u_src/(u_src + 0.5)")
    assert sig("u_src/(u_src + 0.3)") == sig("2.0*u_src/(u_src + 0.3)")
    assert sig("u_src/(2.0*u_src + 0.3)") != sig("u_src/(u_src + 0.3)")


def test_signature_components_of_a_sum_and_constants():
    assert sig("2*tanh(u_src) + 0.5*u_tgt*S_src + 3.0") == frozenset({"tanh(u_src)", "S_src*u_tgt"})
    assert sig("0.37") == frozenset()
    assert sig("1.5 + 2.5") == frozenset()
    assert sig("u_src + u_src") == frozenset({"u_src"})                                    # expands to one component


def test_signature_substitutes_product_columns_by_their_factors():
    assert sig("u_tgt_u_src") == sig("u_tgt*u_src") == frozenset({"u_src*u_tgt"})
    assert sig("3*u_tgt_S_src + tanh(u_src_S_src)") == frozenset({"S_src*u_tgt", "tanh(S_src*u_src)"})


def test_bare_constant_detection():
    assert R.is_bare_constant("0.4") and R.is_bare_constant("tanh(0.2) + 1")
    assert not R.is_bare_constant("u_src") and not R.is_bare_constant("0.4*u_tgt")


def test_signature_recurrence_uses_exact_ceiling():
    make = lambda n_with: [frozenset({"tanh(u_src)"}) if i < n_with else frozenset() for i in range(25)]
    r = R.signature_recurrence(make(18), CFG)
    assert r["needed"] == 18 and r["stable"] == ["tanh(u_src)"] and r["counts"]["tanh(u_src)"] == 18
    assert R.signature_recurrence(make(17), CFG)["stable"] == []                          # 17 / 25 = 68% < 70%
    assert R.signature_recurrence([frozenset({"a"})] * 3, CFG)["needed"] == 3               # ceil(2.1) for the 3-refit pilot
