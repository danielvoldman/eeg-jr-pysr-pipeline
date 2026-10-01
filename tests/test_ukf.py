"""C2 tests: reference UKF and unscented RTS smoother (§7.5, §7.6, §9.3; IMP-019 to IMP-023).

Expected values come from closed-form arithmetic, an independent textbook Kalman filter and
RTS smoother written here in plain numpy, and filterpy 1.4.5 (installed as a TEST ORACLE
only; the production filter is src/ukf.py). Nothing touches config.yml except reading it.
"""
import ast
import copy
import math
import re
from pathlib import Path

import numpy as np
import pytest
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from src import model, state_space as ss, ukf
from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
CFG["ukf"]["numba"]["enabled"] = False      # this module tests the NumPy REFERENCE filter (the default backend is Numba since DEV-005)
SP = CFG["ukf"]["sigma_points"]
MU_REF = CFG["rescaling"]["mu_ref"]
SIGMA_REF = CFG["rescaling"]["sigma_ref"]
Q_TEST = 1.0e-2          # fixed a priori: inside the 1e-4 .. 1e-1 grid of §7.6; never tuned to a result
JITTER = CFG["ukf"]["divergence"]["covariance_jitter"]
SEEDS = (7, 21, 22, 23)  # the four seeds measured in the C2 review
Y1_SMOOTHER_TOL = 0.15   # C2b: smoother may be worse than the filter on y1 by at most 0.15 SD (review: +0.125)
GAIN_MEAN_BOUND = 3.0    # C2b: |mean filtered/smoothed gain| for zero coupling (review: max 1.83)
G12_TRUE = 12.0          # C2b coupled test, fixed before the first run: g12 = 12, g21 = 0
COUPLED_BAND = (0.5, 1.5)   # mean of the filtered (and smoothed) g12 over the last half must lie in 0.5..1.5 x truth
COUPLED_G21_BOUND = 3.0
COUPLED_P = (220.0, 260.0)  # asymmetric nodes (C2b tried this and the old prior-reference divergence rule stopped it; C2c, IMP-029)


# ---- helpers: simulated data with known truth ---------------------------------------------

def simulate_data(seconds, seed, g12=0.0, g21=0.0, m=0.2, p=None):
    """Two-node A2 run at 2048 Hz, sampled every 8th step (256 Hz), mixed on the deviations
    from mu_ref (IMP-015), observation noise with the filter's R. Returns (truth states, z)."""
    n = int(seconds * 2048)
    res = model.simulate(CFG, n, seed=seed, n_nodes=2, g12=g12, g21=g21, p=p)
    states = res.states[8::8]                                  # (T, 2, 6) at the observation times
    y = states[:, :, 1] - states[:, :, 2]
    M = np.array([[1.0, m], [m, 1.0]])
    rng = np.random.default_rng(seed + 1000)
    z = MU_REF + (y - MU_REF) @ M.T + rng.normal(size=y.shape) * math.sqrt(0.25) * SIGMA_REF
    return states.reshape(-1, 12), z


def S(v):
    e0, v0, r = CFG["jansen_rit"]["e0"], CFG["jansen_rit"]["v0"], CFG["jansen_rit"]["r"]
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - np.asarray(v, dtype=float))))


@pytest.fixture(scope="module")
def layout():
    return ss.make_layout(CFG)


def filter_and_smooth(layout, truth_z, q=Q_TEST):
    truth, z = truth_z
    res = ukf.run_filter(z, CFG, layout, q, keep_cov=True)
    xs, Ps = ukf.run_smoother(res, CFG)
    return {"truth": truth, "z": z, "res": res, "xs": xs, "Ps": Ps}


@pytest.fixture(scope="module")
def runs(layout):
    """Four 12 s coupling-free series (C2 review seeds), q = Q_TEST, filter with covariances and smoother."""
    return {seed: filter_and_smooth(layout, simulate_data(12, seed=seed)) for seed in SEEDS}


@pytest.fixture(scope="module")
def run(runs):
    return runs[7]


BURN = 512               # 2 s at 256 Hz, discarded before every score


# ---- weights and sigma points --------------------------------------------------------------

@pytest.mark.parametrize("n", [19, 17])
def test_weights_and_spread_closed_form(n):
    lam, Wm, Wc = ukf.sigma_weights(n, SP["alpha"], SP["beta"], SP["kappa"])
    # alpha = 1, beta = 2, kappa = 0: lambda = 0, Wm0 = 0, Wc0 = 2, others 1 / (2 n)
    assert lam == 0.0 and Wm[0] == 0.0 and Wc[0] == 2.0
    np.testing.assert_allclose(Wm[1:], 1.0 / (2 * n), rtol=1e-15)
    np.testing.assert_allclose(Wc[1:], 1.0 / (2 * n), rtol=1e-15)
    assert Wm.shape == (2 * n + 1,) and Wm.sum() == pytest.approx(1.0, rel=1e-14)
    assert (Wm >= 0).all() and (Wc >= 0).all()
    pts = ukf.sigma_points(np.zeros(n), np.eye(n), lam)
    np.testing.assert_allclose(np.abs(pts[1:n + 1]).max(axis=1), math.sqrt(n), rtol=1e-14)
    assert n == 19 and Wm[1] == pytest.approx(1 / 38) or n == 17 and Wm[1] == pytest.approx(1 / 34)


def test_sigma_points_upper_triangular_rows_of_U():
    """P = [[4, 2], [2, 3]] has the upper factor U = [[2, 1], [0, sqrt 2]] (P = U'U). With
    n = 2, lambda = 0 the offsets are +-sqrt(2) * rows of U (hand arithmetic)."""
    x = np.array([1.0, -1.0])
    P = np.array([[4.0, 2.0], [2.0, 3.0]])
    pts = ukf.sigma_points(x, P, 0.0)
    r2 = math.sqrt(2.0)
    expected = np.array([x, x + r2 * np.array([2.0, 1.0]), x + r2 * np.array([0.0, r2]),
                         x - r2 * np.array([2.0, 1.0]), x - r2 * np.array([0.0, r2])])
    np.testing.assert_allclose(pts, expected, rtol=1e-14, atol=1e-14)
    # and they reproduce the covariance exactly (a lower factor with rows would not)
    _, Wm, Wc = ukf.sigma_weights(2, 1.0, 2.0, 0.0)
    d = pts - x
    np.testing.assert_allclose(d.T @ np.diag(np.array([0.0, .25, .25, .25, .25])) @ d, P, rtol=1e-13)


# ---- linear Gaussian sanity ----------------------------------------------------------------

def linear_model(n=3, k=2, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, n))
    F *= 0.9 / max(abs(np.linalg.eigvals(F)))
    H = rng.normal(size=(k, n))
    A = rng.normal(size=(n, n))
    P0 = A @ A.T + np.eye(n)
    return F, H, P0


def textbook_kf(F, H, Q, R, x0, P0, zs):
    x, P, xs, Ps = x0.copy(), P0.copy(), [], []
    for z in zs:
        x = F @ x
        P = F @ P @ F.T + Q
        Sm = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(Sm)
        x = x + K @ (z - H @ x)
        P = (np.eye(len(x)) - K @ H) @ P
        xs.append(x)
        Ps.append(P)
    return np.array(xs), np.array(Ps)


def filterpy_form_kf(F, H, Q, R, x0, P0, zs):
    """The filterpy recursion on a linear model (IMP-019): K and S built from F P F' (no Q),
    only the predicted covariance carries Q."""
    x, P, xs, Ps = x0.copy(), P0.copy(), [], []
    for z in zs:
        x = F @ x
        Pf = F @ P @ F.T
        Pp = Pf + Q
        Sm = H @ Pf @ H.T + R
        K = Pf @ H.T @ np.linalg.inv(Sm)
        x = x + K @ (z - H @ x)
        P = Pp - K @ Sm @ K.T
        xs.append(x)
        Ps.append(P)
    return np.array(xs), np.array(Ps)


def textbook_rts(F, Q, xs, Ps):
    xsm, Psm = xs.copy(), Ps.copy()
    for k in reversed(range(len(xs) - 1)):
        xb = F @ xs[k]
        Pb = F @ Ps[k] @ F.T + Q
        G = Ps[k] @ F.T @ np.linalg.inv(Pb)
        xsm[k] = xs[k] + G @ (xsm[k + 1] - xb)
        Psm[k] = Ps[k] + G @ (Psm[k + 1] - Pb) @ G.T
    return xsm, Psm


def run_linear(F, H, Q, R, x0, P0, zs):
    n = F.shape[0]
    w = ukf.sigma_weights(n, SP["alpha"], SP["beta"], SP["kappa"])
    f = ukf.UnscentedFilter(n, w, Q, R, propagate=lambda s, wm: s @ F.T, observe=lambda s: s @ H.T)
    f.x, f.P = x0.copy(), P0.copy()
    xs, Ps = [], []
    for z in zs:
        f.predict()
        f.update(z)
        xs.append(f.x.copy())
        Ps.append(f.P.copy())
    return np.array(xs), np.array(Ps), w


def linear_case(q_scale):
    F, H, P0 = linear_model()
    n, k = 3, 2
    rng = np.random.default_rng(5)
    x_true, zs = np.zeros(n), []
    for _ in range(40):
        x_true = F @ x_true + rng.normal(size=n) * 0.3
        zs.append(H @ x_true + rng.normal(size=k) * 0.2)
    Q = q_scale * np.diag([1.0, 2.0, 0.5])
    R = 0.04 * np.eye(k)
    return F, H, Q, R, np.array([0.5, -0.5, 1.0]), P0, np.array(zs)


def test_linear_Q0_ukf_equals_exact_kalman_filter_and_rts():
    F, H, Q, R, x0, P0, zs = linear_case(0.0)
    zs = zs[:10]       # with Q = 0 the covariance collapses towards singular; 10 steps stay positive definite
    xs, Ps, w = run_linear(F, H, Q, R, x0, P0, zs)
    xr, Pr = textbook_kf(F, H, Q, R, x0, P0, zs)
    np.testing.assert_allclose(xs, xr, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(Ps, Pr, rtol=1e-8, atol=1e-11)
    # smoother: with Q = 0 the predicted covariance F P F' is nearly singular and its inverse
    # amplifies the 1e-9 filter differences (seen: 3e-5 at 10 steps), so compare on 5 steps
    k5 = 5
    xsm, Psm = ukf.rts_smooth(xs[:k5], Ps[:k5], w, Q, lambda k, s, wm: s @ F.T)
    xrs, Prs = textbook_rts(F, Q, xr[:k5], Pr[:k5])
    np.testing.assert_allclose(xsm, xrs, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(Psm, Prs, rtol=1e-7, atol=1e-10)


def test_linear_Qpositive_matches_filterpy_form_and_differs_from_textbook():
    """Documents IMP-019. With Q > 0 the filterpy-form UKF is NOT the exact Kalman filter
    (K and S carry no Q). This is the specified algorithm; do not 'fix' it."""
    F, H, Q, R, x0, P0, zs = linear_case(0.05)
    xs, Ps, w = run_linear(F, H, Q, R, x0, P0, zs)
    xf, Pf = filterpy_form_kf(F, H, Q, R, x0, P0, zs)
    np.testing.assert_allclose(xs, xf, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(Ps, Pf, rtol=1e-8, atol=1e-11)
    xt, Pt = textbook_kf(F, H, Q, R, x0, P0, zs)
    assert np.abs(Ps - Pt).max() > 1e-3          # a real, non-rounding difference
    assert np.abs(xs - xt).max() > 1e-3
    # and the smoother is the textbook recursion on the FILTERED sequence
    xsm, Psm = ukf.rts_smooth(xs, Ps, w, Q, lambda k, s, wm: s @ F.T)
    xrs, Prs = textbook_rts(F, Q, xs, Ps)
    np.testing.assert_allclose(xsm, xrs, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(Psm, Prs, rtol=1e-7, atol=1e-10)


# ---- filterpy oracle on the Jansen-Rit model ------------------------------------------------

class OracleUKF(UnscentedKalmanFilter):
    """filterpy's UKF whose process sigmas come from C1's predict (the buffer needs the
    mean of all sigma points at every sub-step, which a per-point fx cannot do)."""

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        self.sigmas_f[:] = ss.predict(sigmas, self.Wm, self.jr_buffer, self.jr_layout, CFG)


def make_oracle(layout, q, buf):
    n = layout.n
    points = MerweScaledSigmaPoints(n, alpha=SP["alpha"], beta=SP["beta"], kappa=SP["kappa"])
    f = OracleUKF(dim_x=n, dim_z=2, dt=1.0, hx=lambda s: ss.observe(s, layout, CFG),
                  fx=lambda s, dt: s, points=points)
    f.Q, f.R = ukf.process_noise(layout, CFG, q), ukf.obs_noise(CFG)
    f.x, f.P = ss.prior_mean(layout, CFG), ss.prior_cov(layout, CFG)
    f.jr_buffer, f.jr_layout = buf, layout
    return f


def test_forward_filter_and_smoother_match_filterpy(layout):
    _, z = simulate_data(2, seed=11)
    z = z[:400]
    res = ukf.run_filter(z, CFG, layout, Q_TEST, keep_cov=True)
    assert res.n_done == 400
    buf = ss.make_buffer(CFG)
    f = make_oracle(layout, Q_TEST, buf)
    Xs, Ps = [], []
    for t in range(400):
        f.predict()
        f.update(z[t])
        buf.replace_latest(ss.sigmoid_of_mean_potential(f.x[None], [1.0], CFG))
        Xs.append(f.x.copy())
        Ps.append(f.P.copy())
    Xs, Ps = np.array(Xs), np.array(Ps)
    np.testing.assert_allclose(res.x, Xs, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(res.P, Ps, rtol=1e-10, atol=0.0)

    # smoother: filterpy's rts_smoother with an fx that reads the forward buffer snapshot of step k + 1
    xs_mine, Ps_mine = ukf.run_smoother(res, CFG)
    calls = {"i": 0}
    n_sig = 2 * layout.n + 1

    def fx(s, dt):
        k = (len(Xs) - 2) - calls["i"] // n_sig
        calls["i"] += 1
        b = ukf.buffer_from_snapshot(res.snapshots[k + 1])
        return ss.predict(s[None], [1.0], b, layout, CFG)[0]
    f.fx = fx
    xs_fp, Ps_fp, _ = f.rts_smoother(Xs, Ps)
    np.testing.assert_allclose(xs_mine, xs_fp, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(Ps_mine, Ps_fp, rtol=1e-10, atol=0.0)


# ---- the buffer inside the filter (S of the mean; filtered mean at every 4th) ---------------------

def test_buffer_holds_predicted_mean_then_filtered_mean(layout, monkeypatch):
    _, z = simulate_data(1, seed=3)
    z = z[:6]
    buf = ss.make_buffer(CFG)
    seen = []
    real = ss.predict

    def spy(points, wm, buffer, layout_, cfg_, n_substeps=None):
        out = real(points, wm, buffer, layout_, cfg_, n_substeps)
        v = ss.potentials(out)
        expected = S(np.asarray(wm) @ v)                 # S of the MEAN potential
        mean_of_s = np.asarray(wm) @ S(v)                # what a wrong buffer would hold
        seen.append((buffer.read(0), expected, mean_of_s))
        return out
    monkeypatch.setattr(ss, "predict", spy)
    res = ukf.run_filter(z, CFG, layout, Q_TEST, buffer=buf)
    assert res.n_done == 6 and len(seen) == 6
    for held, expected, mean_of_s in seen:
        np.testing.assert_allclose(held, expected, rtol=1e-12)
    # sigma points spread enough that S(mean) and mean(S) really differ at some step
    assert max(np.abs(e - m).max() for _, e, m in seen) > 1e-9
    # after the update the newest entry is S of the FILTERED mean potential
    v_filt = ss.potentials(res.x[res.n_done - 1])
    np.testing.assert_allclose(buf.read(0), S(v_filt), rtol=1e-12)
    assert np.abs(buf.read(0) - seen[-1][1]).max() > 1e-12       # and it replaced the predicted one


# ---- simulated two-node data with known truth ------------------------------------------------------

def normalized_rmse(est, truth):
    sd = np.sqrt(np.tile(ss.neural_variance(CFG), 2))
    return np.sqrt(((est[BURN:, :12] - truth[BURN:len(est)]) ** 2).mean(axis=0)) / sd


def test_filter_beats_prior_and_smoother_beats_filter(runs):
    """C2b: mean over four seeds. Per seed: smoother better on average, and per state except y1 of each
    node; y1 (seen only through y1 - y2, SD 0.25 mV against an observation noise SD of 0.57 mV, so
    essentially unobservable) may be worse after smoothing by at most Y1_SMOOTHER_TOL SD."""
    rp_all, rf_all = [], []
    y1 = [1, 7]
    not_y1 = [i for i in range(12) if i % 6 != 1]
    for seed, r in runs.items():
        res, truth = r["res"], r["truth"]
        assert res.n_done == len(truth) and not res.diverged
        prior = np.tile(ss.initial_neural_state(CFG), (len(truth), 1))
        rf, rs, rp = (normalized_rmse(r["res"].x, truth), normalized_rmse(r["xs"], truth),
                      normalized_rmse(prior, truth))
        rf_all.append(rf)
        rp_all.append(rp)
        print(f"\nseed {seed}: normalized RMSE (burn-in 2 s discarded)"
              f"\n  prior    {np.round(rp, 3)}\n  filter   {np.round(rf, 3)}\n  smoother {np.round(rs, 3)}"
              f"\n  smoother - filter on y1: {np.round(rs[y1] - rf[y1], 3)}")
        assert rs.mean() < rf.mean()
        assert (rs[not_y1] <= rf[not_y1]).all()
        assert (rs[y1] - rf[y1] <= Y1_SMOOTHER_TOL).all()
    mf, mp = np.mean(rf_all, axis=0), np.mean(rp_all, axis=0)
    print(f"\nmean over seeds: prior {np.round(mp, 3)}\n                 filter {np.round(mf, 3)}")
    assert (mf < mp).all()


def test_nis_mean_in_fixed_band(run):
    nis = run["res"].nis[BURN:]
    mean = float(np.mean(nis))
    print(f"\nmean NIS at correct R and q = {Q_TEST}: {mean:.3f} (target 2, band [1.0, 4.0])")
    assert 1.0 <= mean <= 4.0


def test_zero_coupling_gains_stay_near_zero(runs, layout):
    """C2b: fixed bound GAIN_MEAN_BOUND (the prior SD is 10.8; the review measured |mean| <= 1.83)."""
    for seed, r in runs.items():
        for name in ("g12", "g21"):
            k = layout.idx[name]
            mf, ms = float(r["res"].x[BURN:, k].mean()), float(r["xs"][BURN:, k].mean())
            print(f"\nseed {seed} {name}: filtered mean {mf:.3f}, smoothed mean {ms:.3f} (truth 0)")
            assert abs(mf) < GAIN_MEAN_BOUND and abs(ms) < GAIN_MEAN_BOUND


def test_monitor_reported_and_healthy(run):
    mon = run["res"].monitor
    assert set(mon) >= {"min_eig_overall", "n_negative_eig_steps", "nan_inf_seen", "diverged",
                        "divergence_step", "divergence_reason", "n_done", "n_jitter_fallbacks"}
    print(f"\nmin covariance eigenvalue over the run: {mon['min_eig_overall']:.3e}")
    assert mon["min_eig_overall"] > 0 and mon["n_negative_eig_steps"] == 0
    assert mon["nan_inf_seen"] is False and mon["diverged"] is False
    assert mon["n_jitter_fallbacks"] == 0
    assert np.isfinite(run["res"].min_eig[:run["res"].n_done]).all()


def test_huge_q_triggers_divergence_and_stops(layout):
    _, z = simulate_data(1, seed=5)
    res = ukf.run_filter(z, CFG, layout, 1.0e15)     # q up to 1e12 still converges: the data keep the means in range
    assert res.diverged and res.divergence_step is not None
    assert res.n_done <= res.divergence_step < len(z)
    assert res.divergence_reason in ("nan_inf", "covariance_not_pd", "state_beyond_sd_multiple") \
        or res.divergence_reason.startswith("linalg_error")
    assert res.monitor["diverged"] is True and np.isnan(res.x[res.divergence_step]).all()


def test_divergence_rule_state_beyond_ten_sd(layout):
    center = ss.initial_neural_state(CFG)
    sd = np.sqrt(np.tile(ss.neural_variance(CFG), 2))
    x = ss.prior_mean(layout, CFG)
    P = np.eye(layout.n)
    assert ukf._first_divergence(x, P, CFG, center, sd, 1.0) is None
    x2 = x.copy()
    x2[4] += 10.5 * sd[4]
    assert ukf._first_divergence(x2, P, CFG, center, sd, 1.0) == "state_beyond_sd_multiple"
    x3 = x.copy()
    x3[4] += 9.5 * sd[4]
    assert ukf._first_divergence(x3, P, CFG, center, sd, 1.0) is None
    assert ukf._first_divergence(x, P, CFG, center, sd, -1e-3) == "covariance_not_pd"
    assert ukf._first_divergence(x, P, CFG, center, sd, -5e-10) is None                 # -5e-10 + 1e-9 > 0
    assert ukf._first_divergence(x, P, CFG, center, sd, -2e-9) == "covariance_not_pd"   # -2e-9 + 1e-9 < 0
    Pn = P.copy()
    Pn[0, 0] = np.nan
    assert ukf._first_divergence(x, Pn, CFG, center, sd, 1.0) == "nan_inf"


def test_reproducible_bit_identical(layout):
    _, z = simulate_data(1, seed=9)
    a = ukf.run_filter(z[:120], CFG, layout, Q_TEST, keep_cov=True)
    b = ukf.run_filter(z[:120], CFG, layout, Q_TEST, keep_cov=True)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.P, b.P) and np.array_equal(a.nis, b.nis)


def test_smoother_needs_covariances(layout):
    _, z = simulate_data(1, seed=9)
    res = ukf.run_filter(z[:20], CFG, layout, Q_TEST)
    assert res.P is None
    with pytest.raises(ukf.UKFError):
        ukf.run_smoother(res, CFG)


def test_m1_and_reduced_layouts_run(layout):
    _, z = simulate_data(1, seed=13)
    for lay in (ss.make_layout(CFG, include_gains=False), ):
        res = ukf.run_filter(z[:80], CFG, lay, Q_TEST)
        assert res.n_done == 80 and res.x.shape[1] == 17


def test_noise_matrices(layout):
    Q = ukf.process_noise(layout, CFG, 0.05)
    dt = 1.0 / CFG["preprocessing"]["observation_fs_hz"]
    var = np.tile(ss.neural_variance(CFG), 2)
    np.testing.assert_allclose(np.diag(Q)[:12], 0.05 * var, rtol=1e-14)
    prior = np.diag(ss.prior_cov(layout, CFG))
    np.testing.assert_allclose(np.diag(Q)[12:], 1.0e-3 * prior[12:] * dt, rtol=1e-14)
    assert np.array_equal(Q, np.diag(np.diag(Q)))
    np.testing.assert_allclose(ukf.obs_noise(CFG), 0.25 * SIGMA_REF ** 2 * np.eye(2), rtol=1e-14)


# ---- C2b: exceptions, monitor, jitter --------------------------------------------------------------------

def test_config_error_propagates_instead_of_divergence(layout):
    cfg = copy.deepcopy(CFG)
    cfg["rescaling"]["mu_ref"] = None
    _, z = simulate_data(1, seed=3)
    with pytest.raises(ss.StateSpaceError):
        ukf.run_filter(z[:5], cfg, layout, Q_TEST)


def test_other_value_errors_propagate(layout, monkeypatch):
    _, z = simulate_data(1, seed=3)

    def boom(*args, **kwargs):
        raise ValueError("a bug, not a divergence")
    monkeypatch.setattr(ss, "predict", boom)
    with pytest.raises(ValueError, match="a bug"):
        ukf.run_filter(z[:5], CFG, layout, Q_TEST)


def test_linalg_error_is_recorded_as_divergence(layout, monkeypatch):
    _, z = simulate_data(1, seed=3)

    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")
    monkeypatch.setattr(ss, "predict", boom)
    res = ukf.run_filter(z[:5], CFG, layout, Q_TEST)
    assert res.diverged and res.divergence_step == 0 and res.n_done == 0
    assert res.divergence_reason.startswith("linalg_error")
    assert res.monitor["diverged"] is True


def test_monitor_includes_the_offending_step_on_not_pd(layout):
    """q = 1e15 makes the covariance indefinite at step 1 (IMP-021); the negative eigenvalue of that
    step must be in the monitor even though the step is not counted as done."""
    _, z = simulate_data(1, seed=5)
    res = ukf.run_filter(z, CFG, layout, 1.0e15)
    assert res.divergence_reason == "covariance_not_pd"
    mon = res.monitor
    assert mon["n_negative_eig_steps"] >= 1 and mon["min_eig_overall"] <= 0.0
    assert res.min_eig[res.divergence_step] == mon["min_eig_overall"] < 0.0
    assert res.n_done == res.divergence_step and np.isnan(res.x[res.divergence_step]).all()


def test_sigma_points_jitter_only_when_plain_cholesky_fails():
    x = np.zeros(3)
    P = np.diag([1.0, 2.0, 3.0])
    used = []
    assert np.array_equal(ukf.sigma_points(x, P, 0.0), ukf.sigma_points(x, P, 0.0, JITTER, used))
    assert used == []                                     # PD: bit-identical to the plain factorization
    P_lo = np.diag([1.0, 2.0, -0.5 * JITTER])             # P + jitter I is PD, P is not
    with pytest.raises(np.linalg.LinAlgError):
        ukf.sigma_points(x, P_lo, 0.0)
    pts = ukf.sigma_points(x, P_lo, 0.0, JITTER, used)
    assert used == [1] and np.isfinite(pts).all()
    assert np.array_equal(pts, ukf.sigma_points(x, P_lo + JITTER * np.eye(3), 0.0))
    with pytest.raises(np.linalg.LinAlgError):            # P + jitter I is still indefinite
        ukf.sigma_points(x, np.diag([1.0, 2.0, -2.0 * JITTER]), 0.0, JITTER, used)


def test_filter_predict_uses_jitter_fallback_and_counts_it():
    w = ukf.sigma_weights(2, SP["alpha"], SP["beta"], SP["kappa"])
    P0 = np.diag([1.0, -0.5 * JITTER])
    f = ukf.UnscentedFilter(2, w, np.zeros((2, 2)), np.eye(1), lambda s, wm: s, lambda s: s[:, :1], jitter=JITTER)
    f.x, f.P = np.zeros(2), P0.copy()
    f.predict()
    assert f.n_jitter == 1
    g = ukf.UnscentedFilter(2, w, np.zeros((2, 2)), np.eye(1), lambda s, wm: s, lambda s: s[:, :1])
    g.x, g.P = np.zeros(2), P0.copy()
    with pytest.raises(np.linalg.LinAlgError):
        g.predict()


# ---- C2b: coupled data, asymmetric nodes -----------------------------------------------------------------

def test_coupled_filter_moves_g12_toward_truth_and_keeps_g21_near_zero(layout):
    """g12 = G12_TRUE (node 1 -> node 2), g21 = 0, p = COUPLED_P. Band fixed before the first run:
    mean of the filtered and of the smoothed g12 over the last half in COUPLED_BAND x truth,
    |mean g21| < COUPLED_G21_BOUND."""
    truth, z = simulate_data(12, seed=31, g12=G12_TRUE, g21=0.0, p=np.array(COUPLED_P))
    r = filter_and_smooth(layout, (truth, z))
    assert r["res"].n_done == len(z) and not r["res"].diverged
    half = len(z) // 2
    g12, g21 = layout.idx["g12"], layout.idx["g21"]
    f12, f21 = (float(r["res"].x[half:, k].mean()) for k in (g12, g21))
    s12, s21 = (float(r["xs"][half:, k].mean()) for k in (g12, g21))
    p1, p2 = (float(r["res"].x[half:, layout.idx[n]].mean()) for n in ("p1", "p2"))
    print(f"\ncoupled (truth g12 = {G12_TRUE}, g21 = 0, p = {COUPLED_P}), last half:"
          f"\n  filtered g12 {f12:.2f}, g21 {f21:.2f}; smoothed g12 {s12:.2f}, g21 {s21:.2f}; "
          f"filtered p1 {p1:.1f}, p2 {p2:.1f}; posterior SD g12 "
          f"{math.sqrt(r['res'].P[-1][g12, g12]):.2f}")
    lo, hi = COUPLED_BAND[0] * G12_TRUE, COUPLED_BAND[1] * G12_TRUE
    assert lo <= f12 <= hi and lo <= s12 <= hi
    assert abs(f21) < COUPLED_G21_BOUND and abs(s21) < COUPLED_G21_BOUND


# ---- C2b: buffer protocol against an independent implementation ------------------------------------------

def indep_deriv(X, delayed):
    """Independent numpy §7.1 right-hand side of every sigma point of the 19-D M2 layout."""
    jr = CFG["jansen_rit"]
    a, b, C = jr["a"], jr["b"], jr["C"]
    C1, C2, C3, C4 = (C * jr[f"C{i}_multiplier"] for i in (1, 2, 3, 4))
    ab = CFG["priors"]["AB_product"]
    out = np.zeros_like(X)
    for i, x in enumerate(X):
        p, lr, g12, g21 = (x[12], x[13]), (x[14], x[15]), x[16], x[17]
        drive = (g21 * delayed[1], g12 * delayed[0])
        for j in range(2):
            y0, y1, y2, y3, y4, y5 = x[6 * j:6 * j + 6]
            rho = math.exp(lr[j])
            A, B = math.sqrt(ab * rho), math.sqrt(ab / rho)
            out[i, 6 * j:6 * j + 6] = [
                y3, y4, y5,
                A * a * S(y1 - y2) - 2 * a * y3 - a * a * y0,
                A * a * (p[j] + C2 * S(C1 * y0) + drive[j]) - 2 * a * y4 - a * a * y1,
                B * b * C4 * S(C3 * y0) - 2 * b * y5 - b * b * y2]
    return out


def indep_step(x, P, hist, Wm, lam):
    """One observation step with a plain-list history: hist[-1 - lag] is the entry `lag` sub-steps old.
    Appends the predicted-mean S after each sub-step and returns the list of those four entries."""
    dt = 1.0 / (CFG["preprocessing"]["observation_fs_hz"] * CFG["ukf"]["substeps_per_observation"])
    delay = CFG["coupling"]["delay_substeps"]
    X = ukf.sigma_points(x, P, lam)
    added = []
    for _ in range(CFG["ukf"]["substeps_per_observation"]):
        k1 = indep_deriv(X, hist[-1 - delay])
        k2 = indep_deriv(X + dt * k1, hist[-delay])
        X = X + 0.5 * dt * (k1 + k2)
        v = np.stack([X[:, 1] - X[:, 2], X[:, 7] - X[:, 8]], axis=1)
        hist.append(S(Wm @ v))
        added.append(hist[-1])
    return added


def test_buffer_protocol_matches_independent_history(layout):
    """After three filter steps: lags 1-3 hold the predicted-mean S of the last step, lag 0 holds S of the
    FILTERED mean, and every earlier step's 4th entry was replaced by its filtered-mean S while its
    other three entries kept the predicted-mean S. Expected values come from a plain-list history and
    a separate numpy right-hand side (indep_deriv), not from state_space."""
    _, z = simulate_data(1, seed=3)
    z = z[:3]
    buf = ss.make_buffer(CFG)
    res = ukf.run_filter(z, CFG, layout, Q_TEST, buffer=buf, keep_cov=True)
    assert res.n_done == 3
    lam, Wm, _ = ukf.weights_for(layout.n, CFG)
    delay = CFG["coupling"]["delay_substeps"]
    node = ss.initial_node_state(CFG)
    s0 = S(node[1] - node[2])
    hist = [np.array([s0, s0])] * (delay + 1)
    x, P = ss.prior_mean(layout, CFG), ss.prior_cov(layout, CFG)
    for t in range(3):
        predicted = indep_step(x, P, hist, Wm, lam)
        v = ss.potentials(res.x[t][None])[0]
        hist[-1] = S(v)                                   # the filtered-mean entry replaces the 4th
        x, P = res.x[t], res.P[t]
    for lag in range(delay + 1):
        np.testing.assert_allclose(buf.read(lag), hist[-1 - lag], rtol=1e-10, err_msg=f"lag {lag}")
    for lag in (1, 2, 3):
        np.testing.assert_allclose(buf.read(lag), predicted[3 - lag], rtol=1e-10)
    assert np.abs(buf.read(0) - predicted[3]).max() > 1e-9      # lag 0 is the filtered S, not the predicted one


# ---- C2b: divergence reference point ---------------------------------------------------------------------

def test_divergence_is_measured_from_the_steady_state_not_from_zero(layout):
    center = ss.initial_neural_state(CFG)
    sd = np.sqrt(np.tile(ss.neural_variance(CFG), 2))
    mult = CFG["ukf"]["divergence"]["state_sd_multiple"]
    P = np.eye(layout.n)
    x_prior = ss.prior_mean(layout, CFG)
    checked = 0
    for i in (0, 1, 2, 6, 7, 8):
        def verdict(value):
            x = x_prior.copy()
            x[i] = value
            return ukf._first_divergence(x, P, CFG, center, sd, 1.0)
        assert verdict(center[i]) is None
        assert verdict(center[i] + 0.95 * mult * sd[i]) is None
        assert verdict(center[i] - 0.95 * mult * sd[i]) is None
        assert verdict(center[i] + 1.05 * mult * sd[i]) == "state_beyond_sd_multiple"
        assert verdict(center[i] - 1.05 * mult * sd[i]) == "state_beyond_sd_multiple"
        if abs(center[i]) > 1.1 * mult * sd[i]:
            # far from zero but AT the steady state: fine; AT zero but far from the steady state: diverged
            assert verdict(center[i]) is None
            assert verdict(0.0) == "state_beyond_sd_multiple"
            checked += 1
    assert checked >= 4          # y1 and y2 of both nodes have a steady state beyond 10 SD of zero


# ---- C2b: reduction layouts ------------------------------------------------------------------------------

@pytest.mark.parametrize("fix_ei", [False, True])
@pytest.mark.parametrize("tie_p", [False, True])
@pytest.mark.parametrize("tie_g", [False, True])
@pytest.mark.parametrize("include_gains", [True, False])
def test_every_switch_combination_runs_one_filter_step(fix_ei, tie_p, tie_g, include_gains):
    cfg = copy.deepcopy(CFG)
    cfg["state"]["reduction_switches"].update(fix_EI_terms=fix_ei, tie_p1_p2=tie_p, tie_g12_g21=tie_g)
    lay = ss.make_layout(cfg, include_gains=include_gains)
    # independent dimension count: 19 minus the removed quantities (a tie removes g21 only with gains)
    n = 19 - 2 * fix_ei - tie_p - (2 if not include_gains else tie_g)
    assert lay.n == n
    Q, R = ukf.process_noise(lay, cfg, Q_TEST), ukf.obs_noise(cfg)
    assert Q.shape == (n, n) and R.shape == (2, 2)
    x0 = ss.prior_mean(lay, cfg)
    assert ss.observe(x0, lay, cfg).shape == (2,) and ss.observe(np.tile(x0, (5, 1)), lay, cfg).shape == (5, 2)
    _, z = simulate_data(1, seed=13)
    res = ukf.run_filter(z[:1], cfg, lay, Q_TEST)
    assert res.n_done == 1 and res.x.shape == (1, n) and np.isfinite(res.x).all()
    assert not res.diverged and res.S.shape == (1, 2, 2)
    # removed quantities have no Q entry: the parameter part of Q is 1e-3 * prior variance * dt, one per kept name
    prior_var = np.diag(ss.prior_cov(lay, cfg))[12:]
    assert len(lay.names[12:]) == n - 12
    np.testing.assert_allclose(np.diag(Q)[12:], 1.0e-3 * prior_var / cfg["preprocessing"]["observation_fs_hz"],
                               rtol=1e-14)


# ---- C2c: divergence reference at the current parameters (IMP-029) --------------------------------------

SD12 = np.sqrt(np.tile(ss.neural_variance(CFG), 2))


def shifted_x(layout, p2=None, g12=None):
    x = ss.prior_mean(layout, CFG)
    if p2 is not None:
        x[layout.idx["p2"]] = p2
    if g12 is not None:
        x[layout.idx["g12"]] = g12
    return x


@pytest.mark.parametrize("p,g12", [((220.0, 260.0), 0.0), ((220.0, 220.0), 12.0), ((220.0, 260.0), 12.0)])
def test_healthy_shifted_data_is_not_flagged(layout, p, g12):
    """p2 = 260 alone, g12 = 12 alone, and both together (the case that stopped the C2b coupled test)."""
    _, z = simulate_data(8, seed=31, g12=g12, g21=0.0, p=np.array(p))
    res = ukf.run_filter(z, CFG, layout, Q_TEST)
    print(f"\np = {p}, g12 = {g12}: n_done {res.n_done} of {len(z)}, monitor {res.monitor}")
    assert not res.diverged and res.n_done == len(z)
    assert res.monitor["n_reference_fallbacks"] == 0
    if p[1] != p[0] or g12 != 0.0:
        assert res.monitor["n_reference_updates"] >= 1


def test_reference_flags_blown_up_state_and_spares_near_current_reference(layout):
    x = shifted_x(layout, p2=260.0, g12=12.0)
    ref = ukf.DivergenceReference(layout, CFG)
    cur = ref.center(x)
    prior12 = ss.initial_neural_state(CFG)
    assert (np.abs(cur - prior12) / SD12)[7] > 10.0          # precondition: far from the PRIOR steady state
    P = np.eye(layout.n)

    def verdict(state12):
        xx = x.copy()
        xx[:12] = state12
        return ukf._first_divergence(xx, P, CFG, ref.center(xx), SD12, 1.0)
    assert verdict(cur) is None                              # near the current reference: healthy
    assert verdict(prior12) == "state_beyond_sd_multiple"    # near the prior one, far from the current: flagged
    for i in (0, 6):                                         # y0 of each node
        up, down, near = cur.copy(), cur.copy(), cur.copy()
        up[i] += 10.5 * SD12[i]
        down[i] -= 10.5 * SD12[i]
        near[i] += 9.5 * SD12[i]
        assert verdict(up) == "state_beyond_sd_multiple" and verdict(down) == "state_beyond_sd_multiple"
        assert verdict(near) is None


def test_blown_up_run_is_flagged_after_the_startup_exemption(layout):
    _, z = simulate_data(2, seed=31)
    res = ukf.run_filter(z + 100.0, CFG, layout, Q_TEST)         # 100 mV offset: nothing healthy about it
    assert res.diverged and res.divergence_reason == "state_beyond_sd_multiple"
    assert res.divergence_step >= STARTUP_STEPS


# ---- C2c: start-up exemption (DEV-003) ---------------------------------------------------------------------

STARTUP_STEPS = round(CFG["ukf"]["divergence"]["startup_exempt_s"] * CFG["preprocessing"]["observation_fs_hz"])


def shifted_reference(monkeypatch, n_sd=11.0):
    """A reference 11 SD away from every neural state of a healthy run: 'beyond 10 SD' at every step."""
    monkeypatch.setattr(ukf.DivergenceReference, "center",
                        lambda self, x: ss.initial_neural_state(CFG) + n_sd * SD12)


def test_startup_exemption_length_is_half_a_second():
    assert CFG["ukf"]["divergence"]["startup_exempt_s"] == 0.5 and STARTUP_STEPS == 128


def test_state_flag_is_suppressed_at_step_2_and_raised_at_step_129(layout, monkeypatch):
    shifted_reference(monkeypatch)
    _, z = simulate_data(1, seed=3)
    short = ukf.run_filter(z[:STARTUP_STEPS], CFG, layout, Q_TEST)      # steps 1..128: all exempt
    assert not short.diverged and short.n_done == STARTUP_STEPS
    res = ukf.run_filter(z[:STARTUP_STEPS + 10], CFG, layout, Q_TEST)
    assert res.diverged and res.divergence_reason == "state_beyond_sd_multiple"
    assert res.divergence_step == STARTUP_STEPS and res.n_done == STARTUP_STEPS   # step 129 (1-based)
    assert res.monitor["startup_exempt_steps"] == STARTUP_STEPS


def test_only_the_state_flag_is_exempt(layout, monkeypatch):
    _, z = simulate_data(1, seed=3)
    monkeypatch.setattr(ss, "predict", lambda pts, wm, buf, lay, cfg, n_substeps=None: np.full_like(pts, np.nan))
    res = ukf.run_filter(z[:5], CFG, layout, Q_TEST)                    # NaN at step 1
    assert res.diverged and res.divergence_step == 0
    assert res.divergence_reason == "nan_inf" or res.divergence_reason.startswith("linalg_error")
    monkeypatch.undo()
    _, z5 = simulate_data(1, seed=5)
    res = ukf.run_filter(z5, CFG, layout, 1.0e15)                       # covariance not PD at step 2, inside the window
    assert res.divergence_reason == "covariance_not_pd" and res.divergence_step < STARTUP_STEPS
    P = np.eye(layout.n)
    x = ss.prior_mean(layout, CFG)
    far = ss.initial_neural_state(CFG) + 20.0 * SD12
    x[:12] = far
    assert ukf._first_divergence(x, P, CFG, ss.initial_neural_state(CFG), SD12, 1.0, check_state=False) is None
    assert ukf._first_divergence(x, P, CFG, ss.initial_neural_state(CFG), SD12, 1.0) == "state_beyond_sd_multiple"
    Pn = P.copy()
    Pn[0, 0] = np.nan
    assert ukf._first_divergence(x, Pn, CFG, ss.initial_neural_state(CFG), SD12, 1.0, check_state=False) == "nan_inf"
    assert ukf._first_divergence(x, P, CFG, ss.initial_neural_state(CFG), SD12, -1e-3, check_state=False)         == "covariance_not_pd"


def test_reference_cache_refreshes_only_after_a_five_percent_move(layout, monkeypatch):
    calls = []
    real = ss.coupled_steady_state
    monkeypatch.setattr(ss, "coupled_steady_state", lambda cfg, *a: (calls.append(a), real(cfg, *a))[1])
    frac = CFG["ukf"]["divergence"]["reference_refresh_fraction_of_prior_sd"]
    sd = ss.parameter_prior_sd(CFG)
    ref = ukf.DivergenceReference(layout, CFG)
    x = ss.prior_mean(layout, CFG)
    assert np.array_equal(ref.center(x), ss.initial_neural_state(CFG)) and calls == []      # at the prior: no solve
    p1_0 = x[layout.idx["p1"]]
    x[layout.idx["p1"]] = p1_0 + 0.9 * frac * sd["p1"]
    ref.center(x)
    assert len(calls) == 0
    x[layout.idx["p1"]] = p1_0 + 1.1 * frac * sd["p1"]
    ref.center(x)
    assert len(calls) == 1 and ref.n_updates == 1
    x[layout.idx["p1"]] += 0.9 * frac * sd["p1"]                      # relative to the CACHED parameters now
    ref.center(x)
    assert len(calls) == 1
    x[layout.idx["p1"]] += 0.9 * frac * sd["p1"]
    ref.center(x)
    assert len(calls) == 2
    for name in ("g12", "g21", "log_rho1", "log_rho2", "p2"):        # each fixed-point parameter triggers it
        n0 = len(calls)
        x[layout.idx[name]] += 1.1 * frac * sd[name]
        ref.center(x)
        assert len(calls) == n0 + 1, name
    n0 = len(calls)
    x[layout.idx["m"]] += 0.4                                          # m is not part of the fixed point
    ref.center(x)
    assert len(calls) == n0
    x_nan = x.copy()
    x_nan[layout.idx["p1"]] = np.nan
    assert np.array_equal(ref.center(x_nan), ref.ref) and len(calls) == n0    # non-finite: cached reference


def test_reference_follows_removed_and_tied_quantities(monkeypatch):
    calls = []
    real = ss.coupled_steady_state
    monkeypatch.setattr(ss, "coupled_steady_state", lambda cfg, *a: (calls.append(a), real(cfg, *a))[1])
    cfg = copy.deepcopy(CFG)
    cfg["state"]["reduction_switches"].update(fix_EI_terms=True, tie_p1_p2=True, tie_g12_g21=True)
    lay = ss.make_layout(cfg)
    ref = ukf.DivergenceReference(lay, cfg)
    x = ss.prior_mean(lay, cfg)
    x[lay.idx["p1"]] = 250.0
    x[lay.idx["g12"]] = 9.0
    ref.center(x)
    lr = math.log(3.25 / 22)
    assert calls == [(250.0, 250.0, lr, lr, 9.0, 9.0)]              # p2 tied, log_rho fixed, g21 tied
    m1 = ss.make_layout(CFG, include_gains=False)
    ref1 = ukf.DivergenceReference(m1, CFG)
    x1 = ss.prior_mean(m1, CFG)
    x1[m1.idx["p2"]] = 260.0
    ref1.center(x1)
    assert calls[-1][4:] == (0.0, 0.0)                                # M1: no gains


def test_failed_fixed_point_falls_back_to_prior_reference_and_is_not_divergence(layout, monkeypatch):
    calls = []

    def boom(cfg, *a):
        calls.append(a)
        raise model.ModelError("no unique fixed point")
    monkeypatch.setattr(ss, "coupled_steady_state", boom)
    ref = ukf.DivergenceReference(layout, CFG)
    x = shifted_x(layout, p2=260.0)
    assert np.array_equal(ref.center(x), ss.initial_neural_state(CFG))       # the prior-parameter reference
    assert ref.n_fallbacks == 1 and ref.n_updates == 0
    ref.center(x)
    assert len(calls) == 1                                            # not retried until a further move
    _, z = simulate_data(3, seed=31, p=np.array(COUPLED_P))
    res = ukf.run_filter(z, CFG, layout, Q_TEST)
    assert not res.diverged and res.monitor["n_reference_fallbacks"] >= 1 and res.monitor["n_reference_updates"] == 0


# ---- hygiene -------------------------------------------------------------------------------------------

def test_no_numeric_parameter_literals_and_imports():
    path = REPO_ROOT / "src" / "ukf.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    allowed = {0, 1, 2, 0.5, 0.0, 1.0}
    bad = [(n.lineno, n.value) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool) and n.value not in allowed]
    assert not bad, bad
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert not imported & {"filterpy", "numba", "preprocess"}, imported
    assert not re.search(r"\bprint\(", text)
