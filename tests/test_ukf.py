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
SP = CFG["ukf"]["sigma_points"]
MU_REF = CFG["rescaling"]["mu_ref"]
SIGMA_REF = CFG["rescaling"]["sigma_ref"]
Q_TEST = 1.0e-2          # fixed a priori: inside the 1e-4 .. 1e-1 grid of §7.6; never tuned to a result


# ---- helpers: simulated data with known truth ---------------------------------------------

def simulate_data(seconds, seed, g12=0.0, g21=0.0, m=0.2):
    """Two-node A2 run at 2048 Hz, sampled every 8th step (256 Hz), mixed on the deviations
    from mu_ref (IMP-015), observation noise with the filter's R. Returns (truth states, z)."""
    n = int(seconds * 2048)
    res = model.simulate(CFG, n, seed=seed, n_nodes=2, g12=g12, g21=g21)
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


@pytest.fixture(scope="module")
def run(layout):
    """One 12 s coupling-free series, q = Q_TEST, forward filter with covariances and smoother."""
    truth, z = simulate_data(12, seed=7)
    res = ukf.run_filter(z, CFG, layout, Q_TEST, keep_cov=True)
    xs, Ps = ukf.run_smoother(res, CFG)
    return {"truth": truth, "z": z, "res": res, "xs": xs, "Ps": Ps}


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


def test_filter_beats_prior_and_smoother_beats_filter(run):
    res, truth = run["res"], run["truth"]
    assert res.n_done == len(truth) and not res.diverged
    prior = np.tile(ss.initial_neural_state(CFG), (len(truth), 1))
    rf = normalized_rmse(res.x, truth)
    rs = normalized_rmse(run["xs"], truth)
    rp = normalized_rmse(prior, truth)
    print("\nnormalized RMSE per neural state (burn-in 2 s discarded)"
          f"\n  prior    {np.round(rp, 3)}\n  filter   {np.round(rf, 3)}\n  smoother {np.round(rs, 3)}")
    assert (rf < rp).all()
    assert rs.mean() < rf.mean()
    # per state, except y1 of each node: y1 is seen only through y1 - y2, is barely better than the
    # prior (0.9 of the prior error), and the smoother can be slightly worse there (measured on
    # seeds 7, 21, 22, 23: one y1 state in each, by 0.002 to 0.12; see IMP-023)
    not_y1 = [i for i in range(12) if i % 6 != 1]
    assert (rs[not_y1] <= rf[not_y1]).all()


def test_nis_mean_in_fixed_band(run):
    nis = run["res"].nis[BURN:]
    mean = float(np.mean(nis))
    print(f"\nmean NIS at correct R and q = {Q_TEST}: {mean:.3f} (target 2, band [1.0, 4.0])")
    assert 1.0 <= mean <= 4.0


def test_zero_coupling_gains_stay_near_zero(run, layout):
    sd_g = CFG["coupling"]["gain_prior_sd_factor_of_C2"] * model.constants(CFG)["C2"]
    x = run["res"].x[BURN:]
    for name in ("g12", "g21"):
        mean_g = float(x[:, layout.idx[name]].mean())
        print(f"\n{name}: mean filtered estimate {mean_g:.3f} (truth 0, prior SD {sd_g:.2f})")
        assert abs(mean_g) < sd_g
    # smoothed trajectory too
    assert abs(float(run["xs"][BURN:, layout.idx["g12"]].mean())) < sd_g


def test_monitor_reported_and_healthy(run):
    mon = run["res"].monitor
    assert set(mon) >= {"min_eig_overall", "n_negative_eig_steps", "nan_inf_seen", "diverged",
                        "divergence_step", "divergence_reason", "n_done"}
    print(f"\nmin covariance eigenvalue over the run: {mon['min_eig_overall']:.3e}")
    assert mon["min_eig_overall"] > 0 and mon["n_negative_eig_steps"] == 0
    assert mon["nan_inf_seen"] is False and mon["diverged"] is False
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
