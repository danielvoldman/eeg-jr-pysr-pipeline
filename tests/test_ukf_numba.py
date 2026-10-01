"""C5: the Numba UKF and smoother against the NumPy reference and filterpy (§7.5, §16.5.2; IMP-044 to IMP-050).

Tolerance for every filtered and smoothed mean and covariance: rtol 1e-8, atol 1e-10 (§7.5, config
ukf.numba.rtol / atol). The NumPy path (src/ukf.py) is the oracle and stays the default.
"""
import ast
import copy
import re
import time
from pathlib import Path

import numpy as np
import pytest
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from scipy.linalg import cholesky

from src import passes, tuning, ukf, ukf_numba
from src import state_space as ss
from src.config import load_config
from tests import sim_data

CFG = load_config()
RTOL, ATOL = CFG["ukf"]["numba"]["rtol"], CFG["ukf"]["numba"]["atol"]
RTOL_SC, ATOL_SC = CFG["ukf"]["numba"]["rtol_smoothed_cov"], CFG["ukf"]["numba"]["atol_smoothed_cov"]   # DEV-004
Q = 1.0e-2
SP = CFG["ukf"]["sigma_points"]
KERNELS = ("_chol_upper", "_sigma_points", "_ut", "_drift", "_predict", "_observe", "_params6", "_sig", "_steady_node",
           "_nodes", "_coupled_fixed_point", "_filter_kernel", "_smooth_kernel")


def cfg_with(**switches):
    c = copy.deepcopy(CFG)
    c["state"]["reduction_switches"].update(switches)
    return c


def series(seconds, seed, g12=0.0, g21=0.0, p=None, m=0.2):
    r = sim_data.make_recording(CFG, [seconds], seed, g12=g12, g21=g21, p=p, m=m, burn_seconds=2.0)
    return np.ascontiguousarray(r["segments"][0].T)


def same(a, b, what=""):
    np.testing.assert_allclose(b, a, rtol=RTOL, atol=ATOL, err_msg=what)


def same_smoothed_cov(a, b, what=""):
    """Smoothed covariances only (DEV-004): rtol 1e-6, atol 1e-8; the smoother inverts a predicted covariance of condition
    number about 1e8, so last-bit differences show at the 1e-8 level."""
    np.testing.assert_allclose(b, a, rtol=RTOL_SC, atol=ATOL_SC, err_msg=what)


def assert_filters_agree(a, b, keep_cov=True):
    assert (a.n_done, a.diverged, a.divergence_step) == (b.n_done, b.diverged, b.divergence_step)
    assert (a.divergence_reason or "").split(":")[0] == (b.divergence_reason or "").split(":")[0]
    for name in ("x", "z_pred", "S", "innovation", "nis", "snapshots"):
        same(getattr(a, name), getattr(b, name), name)
    same(a.min_eig, b.min_eig, "min_eig")
    if keep_cov:
        same(a.P, b.P, "P")
    if a.P_last is not None:
        same(a.P_last, b.P_last, "P_last")
    for key in ("n_negative_eig_steps", "n_jitter_fallbacks", "startup_exempt_steps", "n_reference_updates",
                "n_reference_fallbacks", "nan_inf_seen", "diverged", "divergence_step", "n_done"):
        assert a.monitor[key] == b.monitor[key], key
    same(a.monitor["min_eig_overall"], b.monitor["min_eig_overall"], "min_eig_overall")


def both(z, layout, q=Q, cfg=CFG, keep_cov=True, **kw):
    a = ukf.run_filter(z, cfg, layout, q, keep_cov=keep_cov, backend="numpy", **kw)
    b = ukf.run_filter(z, cfg, layout, q, keep_cov=keep_cov, backend="numba", **kw)
    return a, b


@pytest.fixture(scope="module")
def layout():
    return ss.make_layout(CFG)


# ---- weights, Cholesky convention, sigma points ----------------------------------------------------------------

@pytest.mark.parametrize("n", [19, 17, 15, 12])
def test_weights_closed_form_and_equal_to_the_reference(n):
    lam, Wm, Wc = ukf_numba.sigma_weights(n, SP["alpha"], SP["beta"], SP["kappa"])
    assert SP["alpha"] == 1.0 and SP["beta"] == 2.0 and SP["kappa"] == 0.0       # the locked values
    assert lam == 0.0 and Wm[0] == 0.0 and Wc[0] == 2.0                          # centre: mean 0, covariance 2
    np.testing.assert_allclose(Wm[1:], 1.0 / (2.0 * n), rtol=1e-15)              # 1/38 at n = 19
    np.testing.assert_allclose(Wc[1:], 1.0 / (2.0 * n), rtol=1e-15)
    lam2, Wm2, Wc2 = ukf.sigma_weights(n, SP["alpha"], SP["beta"], SP["kappa"])
    assert (lam, list(Wm), list(Wc)) == (lam2, list(Wm2), list(Wc2))
    assert ukf_numba.weights_for(n, CFG)[2][0] == 2.0


def _spd(n, seed):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


@pytest.mark.parametrize("n", [3, 12, 19])
def test_chol_is_scipy_upper_triangular_and_fails_when_not_pd(n):
    A = _spd(n, n)
    U = np.empty((n, n))
    assert ukf_numba._chol_upper(A, U)
    np.testing.assert_allclose(U, cholesky(A), rtol=1e-12, atol=1e-13)          # scipy's default = upper
    np.testing.assert_allclose(U.T @ U, A, rtol=1e-12)
    assert np.all(np.tril(U, -1) == 0.0)
    B = A.copy()
    B[0, 0] = -1.0
    assert not ukf_numba._chol_upper(B, U)
    C = A.copy()
    C[1, 1] = np.nan
    assert not ukf_numba._chol_upper(C, U)


def test_sigma_points_rows_of_u_match_the_reference_and_use_jitter_only_on_failure():
    n = 19
    P = np.diag(np.linspace(1.0, 2.0, n)) + 0.1 * (_spd(n, 5) / n)
    x = np.linspace(-1.0, 1.0, n)
    lam = ukf.weights_for(n, CFG)[0]
    sig, U, A = np.empty((2 * n + 1, n)), np.empty((n, n)), np.empty((n, n))
    assert ukf_numba._sigma_points(x, P, lam, 1e-9, sig, U, A) == 0
    ref = ukf.sigma_points(x, P, lam)
    np.testing.assert_allclose(sig, ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(sig[1], x + cholesky((lam + n) * P)[0], rtol=1e-12, atol=1e-13)   # a ROW of U
    Pn = np.diag(np.linspace(1.0, 2.0, n))
    Pn[0, 0] = -1e-12                                    # plain fails, P + 1e-9 I is fine (IMP-024)
    used = []
    ref = ukf.sigma_points(x, Pn, lam, 1e-9, used)
    assert ukf_numba._sigma_points(x, Pn, lam, 1e-9, sig, U, A) == 1 and len(used) == 1
    np.testing.assert_allclose(sig, ref, rtol=1e-12, atol=1e-13)
    assert ukf_numba._sigma_points(x, -P, lam, 1e-9, sig, U, A) == 2               # a second failure: LinAlgError
    assert ukf_numba._sigma_points(x, Pn, lam, 0.0, sig, U, A) == 2                # no jitter allowed: fails


# ---- the divergence reference solver ------------------------------------------------------------------------------

@pytest.mark.parametrize("params", [(220.0, 220.0, -1.912, -1.912, 0.0, 0.0), (220.0, 260.0, -1.912, -1.7, 12.0, 0.0),
                                    (180.0, 250.0, -2.0, -1.8, 5.0, 8.0), (300.0, 150.0, -1.9, -2.1, 20.0, -3.0)])
def test_compiled_fixed_point_equals_coupled_steady_state(params):
    from src import model
    k = model.constants(CFG)
    dv = CFG["ukf"]["divergence"]
    out = np.empty(12)
    fp = (int(CFG["simulator"]["steady_state_scan_points"]), int(CFG["simulator"]["steady_state_bisection_iterations"]),
          dv["fixed_point_tolerance"], int(dv["fixed_point_max_iterations"]), dv["fixed_point_fd_step"])
    ab = CFG["priors"]["AB_product"]
    code = ukf_numba._coupled_fixed_point(*params, ab, k["a"], k["b"], k["C1"], k["C2"], k["C3"], k["C4"], k["e0"], k["v0"],
                                          k["r"], *fp, out)
    from src import model as model_mod
    try:
        want = ss.coupled_steady_state(CFG, *params)
    except (model_mod.ModelError, ss.StateSpaceError):      # the reference raises: the compiled solver must fail too
        assert code != 0
        return
    assert code == 0
    np.testing.assert_allclose(out, want, rtol=1e-9, atol=1e-10)
    fp_short = fp[:3] + (0,) + fp[4:]                    # no iterations allowed: not converged, as StateSpaceError
    assert ukf_numba._coupled_fixed_point(*params, ab, k["a"], k["b"], k["C1"], k["C2"], k["C3"], k["C4"], k["e0"], k["v0"],
                                          k["r"], *fp_short, out) == 2


# ---- filter and smoother against the reference -------------------------------------------------------------------------

def test_coupled_asymmetric_filter_and_smoother_match_the_reference(layout):
    z = series(4.0, 31, g12=12.0, g21=3.0, p=(220.0, 260.0))
    a, b = both(z, layout)
    assert a.n_done == z.shape[0] and not a.diverged
    assert_filters_agree(a, b)
    assert b.monitor["n_reference_updates"] > 0
    xa, Pa = ukf.run_smoother(a, CFG, backend="numpy")
    xb, Pb = ukf.run_smoother(b, CFG, backend="numba")
    same(xa, xb, "smoothed x")
    same_smoothed_cov(Pa, Pb, "smoothed P")
    print("max abs errors: filtered x %.2e, P %.2e; smoothed x %.2e, P %.2e" % (
        np.abs(a.x - b.x).max(), np.abs(a.P - b.P).max(), np.abs(xa - xb).max(), np.abs(Pa - Pb).max()))


class OracleUKF(UnscentedKalmanFilter):
    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        self.sigmas_f[:] = ss.predict(sigmas, self.Wm, self.jr_buffer, self.jr_layout, CFG)


def test_numba_matches_filterpy_forward_and_smoother(layout):
    z = series(2.0, 11, g12=6.0, g21=0.0, p=(220.0, 240.0))[:400]
    res = ukf.run_filter(z, CFG, layout, Q, keep_cov=True, backend="numba")
    buf = ss.make_buffer(CFG)
    n = layout.n
    f = OracleUKF(dim_x=n, dim_z=2, dt=1.0, hx=lambda s: ss.observe(s, layout, CFG), fx=lambda s, dt: s,
                  points=MerweScaledSigmaPoints(n, alpha=SP["alpha"], beta=SP["beta"], kappa=SP["kappa"]))
    f.Q, f.R = ukf.process_noise(layout, CFG, Q), ukf.obs_noise(CFG)
    f.x, f.P = ss.prior_mean(layout, CFG), ss.prior_cov(layout, CFG)
    f.jr_buffer, f.jr_layout = buf, layout
    Xs, Ps = [], []
    for t in range(z.shape[0]):
        f.predict()
        f.update(z[t])
        buf.replace_latest(ss.sigmoid_of_mean_potential(f.x[None], [1.0], CFG))
        Xs.append(f.x.copy())
        Ps.append(f.P.copy())
    Xs, Ps = np.array(Xs), np.array(Ps)
    same(Xs, res.x, "x vs filterpy")
    same(Ps, res.P, "P vs filterpy")
    xs_nb, Ps_nb = ukf.run_smoother(res, CFG, backend="numba")
    calls = {"i": 0}
    n_sig = 2 * n + 1

    def fx(s, dt):
        k = (len(Xs) - 2) - calls["i"] // n_sig
        calls["i"] += 1
        b = ukf.buffer_from_snapshot(res.snapshots[k + 1])
        return ss.predict(s[None], [1.0], b, layout, CFG)[0]
    f.fx = fx
    xs_fp, Ps_fp, _ = f.rts_smoother(Xs, Ps)
    same(xs_fp, xs_nb, "smoothed x vs filterpy")
    same_smoothed_cov(Ps_fp, Ps_nb, "smoothed P vs filterpy")


@pytest.mark.parametrize("name,cfg,make_layout", [
    ("M1 (17-D)", CFG, lambda c: ss.make_layout(c, include_gains=False)),
    ("fix_EI (17-D)", cfg_with(fix_EI_terms=True), lambda c: ss.make_layout(c)),
    ("tie_p1_p2 (18-D)", cfg_with(tie_p1_p2=True), lambda c: ss.make_layout(c)),
    ("tie_g12_g21 (18-D)", cfg_with(tie_g12_g21=True), lambda c: ss.make_layout(c)),
    ("all three (15-D)", cfg_with(fix_EI_terms=True, tie_p1_p2=True, tie_g12_g21=True), lambda c: ss.make_layout(c)),
    ("M1 with fix_EI (15-D)", cfg_with(fix_EI_terms=True), lambda c: ss.make_layout(c, include_gains=False)),
])
def test_reduced_layouts_match_the_reference(name, cfg, make_layout):
    lay = make_layout(cfg)
    z = series(2.0, 41, g12=8.0, g21=2.0, p=(220.0, 250.0))
    a, b = both(z, lay, cfg=cfg)
    assert a.n_done == z.shape[0], name
    assert_filters_agree(a, b)
    xa, Pa = ukf.run_smoother(a, cfg, backend="numpy")
    xb, Pb = ukf.run_smoother(b, cfg, backend="numba")
    same(xa, xb, name + " smoothed x")
    same_smoothed_cov(Pa, Pb, name + " smoothed P")


def test_fixed_parameter_window_layout_matches_the_reference():
    params = {"p1": 220.0, "p2": 250.0, "log_rho1": -1.912, "log_rho2": -1.85, "g12": 9.0, "g21": 1.5, "m": 0.3}
    lay = ss.make_fixed_layout(CFG, params)
    assert lay.n == 12
    z = series(2.5, 51, g12=9.0, g21=1.5, p=(220.0, 250.0), m=0.3)[:512]
    bufs = [ss.make_buffer(CFG), ss.make_buffer(CFG)]
    a = ukf.run_filter(z, CFG, lay, Q, buffer=bufs[0], keep_cov=True, backend="numpy")
    b = ukf.run_filter(z, CFG, lay, Q, buffer=bufs[1], keep_cov=True, backend="numba")
    assert a.n_done == 512
    assert_filters_agree(a, b)
    np.testing.assert_allclose(bufs[1]._buf, bufs[0]._buf, rtol=RTOL, atol=ATOL)   # buffer left in the same state
    assert bufs[0]._head == bufs[1]._head
    xa, Pa = ukf.run_smoother(a, CFG, backend="numpy")
    xb, Pb = ukf.run_smoother(b, CFG, backend="numba")
    same(xa, xb, "window smoothed x")
    same_smoothed_cov(Pa, Pb, "window smoothed P")


def test_the_numpy_reference_smoother_is_as_sensitive_as_the_smoothed_covariance_bound(layout):
    """DEV-004: the smoothed-covariance bound is justified by the code, not only by one run. A 1e-16 relative
    perturbation of the reference smoother's own input covariances changes its output by less than the bound
    (rtol 1e-6, atol 1e-8), while it exceeds the strict filter tolerance (1e-8, 1e-10): the smoother amplifies
    last-bit noise through the inverse of a predicted covariance with condition number about 1e8."""
    z = series(4.0, 31, g12=12.0, g21=3.0, p=(220.0, 260.0))
    a = ukf.run_filter(z, CFG, layout, Q, keep_cov=True, backend="numpy")
    ref = ukf.run_smoother(a, CFG, backend="numpy")
    rng = np.random.default_rng(0)
    worst = 0.0
    for eps in (1e-16, 1e-15):
        noise = rng.normal(size=a.P.shape)
        noise = 0.5 * (noise + np.swapaxes(noise, 1, 2))
        pert = copy.copy(a)
        pert.P = a.P * (1.0 + eps * noise)
        out = ukf.run_smoother(pert, CFG, backend="numpy")
        same_smoothed_cov(ref[1], out[1], f"reference self-sensitivity, eps {eps:g}")
        same(ref[0], out[0], "smoothed means are far inside the strict tolerance")
        worst = max(worst, float(np.max(np.abs(out[1] - ref[1]) / (ATOL + RTOL * np.abs(ref[1])))))
    assert worst > 1.0                       # the strict tolerance would fail: this is what DEV-004 records
    # the inverted matrix really is that ill-conditioned
    n = layout.n
    lam, Wm, Wc = ukf.weights_for(n, CFG)
    buf = ukf.buffer_from_snapshot(a.snapshots[50])
    sf = ss.predict(ukf.sigma_points(a.x[49], a.P[49], lam), Wm, buf, layout, CFG)
    Pb = ukf.unscented_transform(sf, Wm, Wc, ukf.process_noise(layout, CFG, Q))[1]
    assert np.linalg.cond(Pb) > 1e6


# ---- divergence: flag, step, reason and monitors must match -----------------------------------------------------------

def _same_divergence(a, b):
    assert (a.diverged, a.divergence_step, a.n_done) == (b.diverged, b.divergence_step, b.n_done)
    assert (a.divergence_reason or "").split(":")[0] == (b.divergence_reason or "").split(":")[0]
    assert a.monitor["n_negative_eig_steps"] == b.monitor["n_negative_eig_steps"]


def test_nan_observation_diverges_at_the_same_step_with_nan_inf(layout):
    z = series(1.0, 61)
    z[50, 0] = np.nan
    a, b = both(z, layout, keep_cov=False)
    assert a.diverged and a.divergence_step == 50 and a.divergence_reason == "nan_inf"
    _same_divergence(a, b)
    assert b.monitor["nan_inf_seen"] and np.isnan(b.x[50]).all()
    same(a.x[:50], b.x[:50], "steps before the divergence")


def test_non_pd_initial_covariance_is_a_linalg_divergence_at_step_zero_in_both(layout):
    z = series(0.5, 62)
    x0, P0, _ = ukf.initial_state(layout, CFG)
    P0 = P0.copy()
    P0[0, 0] = -1.0                                        # neither the plain nor the jittered Cholesky works (IMP-025)
    a, b = both(z, layout, keep_cov=False, x0=x0, P0=P0)
    assert a.diverged and a.divergence_step == 0 and a.divergence_reason.startswith("linalg_error")
    _same_divergence(a, b)
    assert b.divergence_reason.startswith("linalg_error")


def test_state_flag_is_raised_after_the_startup_exemption_at_the_same_step(layout):
    """A 100 mV offset on the data drives the states far from the reference. The state flag is suppressed for the first
    startup_exempt_s (128 steps) and raised at the first step at or after it, in both implementations. The run blows up
    (the parameters leave any sensible range), so only the flag, step and reason are compared, plus the number of
    reference refreshes; the split between solved and failed fixed points depends on rounding there."""
    z = sim_data.simulate_stream(CFG, 2, 31)[1] + 100.0
    a, b = both(z, layout, keep_cov=False)
    n_exempt = int(round(CFG["ukf"]["divergence"]["startup_exempt_s"] * CFG["preprocessing"]["observation_fs_hz"]))
    assert a.diverged and a.divergence_reason == "state_beyond_sd_multiple" and a.divergence_step >= n_exempt
    assert (b.diverged, b.divergence_step, b.divergence_reason) == (a.diverged, a.divergence_step, a.divergence_reason)
    same(a.x[:n_exempt], b.x[:n_exempt], "exempt steps")
    total = lambda r: r.monitor["n_reference_updates"] + r.monitor["n_reference_fallbacks"]     # noqa: E731
    assert total(a) == total(b)


def test_jitter_fallback_path_is_taken_and_counted_identically(layout):
    z = series(1.0, 81)
    x0, P0, _ = ukf.initial_state(layout, CFG)
    P0 = P0.copy()
    P0[13, 13] = -1e-10                                   # plain Cholesky fails, P + 1e-9 I works (IMP-024)
    a = ukf.run_filter(z, CFG, layout, Q, x0=x0, P0=P0, keep_cov=True, backend="numpy")
    b = ukf.run_filter(z, CFG, layout, Q, x0=x0, P0=P0, keep_cov=True, backend="numba")
    assert a.monitor["n_jitter_fallbacks"] >= 1
    assert_filters_agree(a, b)


def test_failed_fixed_point_counts_a_fallback_in_both_and_is_not_a_divergence(layout):
    cfg = copy.deepcopy(CFG)
    cfg["ukf"]["divergence"]["fixed_point_max_iterations"] = 1        # too few Newton steps: not converged
    z = series(2.0, 91, g12=10.0, p=(220.0, 260.0))
    a, b = both(z, layout, cfg=cfg)
    assert a.monitor["n_reference_fallbacks"] > 0
    assert not a.diverged or a.divergence_reason != "linalg_error"
    assert_filters_agree(a, b)


def test_reference_refresh_counts_match_on_a_moving_parameter_series(layout):
    z = series(6.0, 101, g12=12.0, p=(180.0, 260.0))
    a, b = both(z, layout, keep_cov=False)
    assert a.monitor["n_reference_updates"] == b.monitor["n_reference_updates"] > 10
    assert a.monitor["n_reference_fallbacks"] == b.monitor["n_reference_fallbacks"]


# ---- buffer protocol ----------------------------------------------------------------------------------------------------

def test_buffer_object_and_snapshots_follow_the_reference_protocol(layout):
    z = series(1.5, 111, g12=12.0, g21=0.0, p=(220.0, 260.0))
    b1, b2 = ss.make_buffer(CFG), ss.make_buffer(CFG)
    a = ukf.run_filter(z, CFG, layout, Q, buffer=b1, backend="numpy")
    b = ukf.run_filter(z, CFG, layout, Q, buffer=b2, backend="numba")
    np.testing.assert_allclose(b2._buf, b1._buf, rtol=RTOL, atol=ATOL)
    assert b1._head == b2._head
    assert not np.any(np.isnan(b.snapshots))
    # the newest entry of the buffer after every step is S of the FILTERED mean (replace_latest), the three
    # entries before it S of the predicted mean: the final entry is S(x_filtered) exactly
    want = ss.sigmoid_of_mean_potential(b.x[b.n_done - 1][None], [1.0], CFG)
    np.testing.assert_allclose(b2.read(0), want, rtol=1e-12)


# ---- downstream statistics: pass 1, pass 2, NIS, tuning ------------------------------------------------------------------

@pytest.fixture(scope="module")
def recording():
    return sim_data.make_recording(CFG, [6.0, 5.0], 121, g12=8.0, g21=2.0, p=(220.0, 250.0))


@pytest.fixture(scope="module")
def cfg_numba():
    c = copy.deepcopy(CFG)
    c["ukf"]["numba"]["enabled"] = True
    return c


@pytest.fixture(scope="module")
def passes_numpy(recording):
    return passes.run_recording(recording["segments"], recording["starts"], CFG, Q, filter_name="19D")


def test_pass1_and_pass2_are_unchanged_by_the_backend(recording, cfg_numba, passes_numpy):
    nb = passes.run_recording(recording["segments"], recording["starts"], cfg_numba, Q, filter_name="19D")
    a, b = passes_numpy, nb
    assert a.recording_diverged == b.recording_diverged is False
    for k in ("p1", "p2", "log_rho1", "log_rho2", "rho1", "rho2", "A1", "B1", "A2", "B2", "g12", "g21", "m", "m_raw"):
        np.testing.assert_allclose(getattr(b.pass1.params, k), getattr(a.pass1.params, k), rtol=RTOL, atol=ATOL, err_msg=k)
    for k in a.pass1.params.posterior_sd:
        np.testing.assert_allclose(b.pass1.params.posterior_sd[k], a.pass1.params.posterior_sd[k], rtol=RTOL, atol=ATOL)
    for k in ("g12", "g21"):
        np.testing.assert_allclose(b.pass1.gain_estimate[k], a.pass1.gain_estimate[k], rtol=RTOL, atol=ATOL)
    assert b.pass1.gain_estimate["n"] == a.pass1.gain_estimate["n"]
    for sa, sb in zip(a.pass1.segments, b.pass1.segments):
        assert sa.diverged == sb.diverged
        same(sa.x_smooth_params, sb.x_smooth_params, "pass-1 smoothed parameters")
        same(sa.x_filt_params, sb.x_filt_params, "pass-1 filtered parameters")
        same(sa.carry_out_mean, sb.carry_out_mean, "carry mean")
        same(sa.carry_out_var, sb.carry_out_var, "carry variance")
    assert len(a.pass2.windows) == len(b.pass2.windows) > 0
    for wa, wb in zip(a.pass2.windows, b.pass2.windows):
        assert wa.diverged == wb.diverged is False
        same(wa.x_smooth, wb.x_smooth, "window smoothed states")
        same(wa.s_delayed, wb.s_delayed, "window delayed S")
        assert wa.monitor["n_reference_updates"] == wb.monitor["n_reference_updates"]
    print("max abs errors: window x %.2e, window s_delayed %.2e" % (
        max(np.abs(wa.x_smooth - wb.x_smooth).max() for wa, wb in zip(a.pass2.windows, b.pass2.windows)),
        max(np.abs(wa.s_delayed - wb.s_delayed).max() for wa, wb in zip(a.pass2.windows, b.pass2.windows))))


def test_nis_and_tuning_statistic_are_unchanged_by_the_backend(recording, cfg_numba):
    cfg = copy.deepcopy(CFG)
    cfg["passes"]["estimator_burn_in_s"] = 1
    cfg_n = copy.deepcopy(cfg_numba)
    cfg_n["passes"]["estimator_burn_in_s"] = 1
    ea = tuning.recording_nis(recording["segments"], recording["starts"], cfg, Q, filter_name="19D")
    eb = tuning.recording_nis(recording["segments"], recording["starts"], cfg_n, Q, filter_name="19D")
    assert ea["n_samples"] == eb["n_samples"] > 0
    np.testing.assert_allclose(eb["mean_nis"], ea["mean_nis"], rtol=RTOL, atol=ATOL)
    assert {k: v for k, v in ea.items() if k != "mean_nis"} == {k: v for k, v in eb.items() if k != "mean_nis"}


# ---- switch, determinism, hygiene ---------------------------------------------------------------------------------------------

def test_default_is_numba_since_dev005_the_switch_selects_numpy_and_backend_overrides(layout, monkeypatch):
    assert CFG["ukf"]["numba"]["enabled"] is True                       # DEV-005 / C5: Numba is the validated default
    called = []
    monkeypatch.setattr(ukf_numba, "run_filter", lambda *a, **k: called.append(1) or (_ for _ in ()).throw(RuntimeError("numba used")))
    z = series(0.5, 131)
    with pytest.raises(RuntimeError):
        ukf.run_filter(z, CFG, layout, Q)                               # default: Numba, the stub is reached
    off = copy.deepcopy(CFG)
    off["ukf"]["numba"]["enabled"] = False
    called.clear()
    ukf.run_filter(z, off, layout, Q)                                   # switched off: NumPy, the stub is not reached
    assert not called
    ukf.run_filter(z, CFG, layout, Q, backend="numpy")                  # explicit override wins over the switch
    with pytest.raises(ukf.UKFError):
        ukf.run_filter(z, CFG, layout, Q, backend="cuda")


def test_numba_run_is_bit_identical_twice(layout):
    z = series(1.5, 141, g12=6.0, p=(220.0, 240.0))
    a = ukf.run_filter(z, CFG, layout, Q, keep_cov=True, backend="numba")
    b = ukf.run_filter(z, CFG, layout, Q, keep_cov=True, backend="numba")
    for name in ("x", "P", "S", "nis", "min_eig", "snapshots"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    sa = ukf.run_smoother(a, CFG, backend="numba")
    sb = ukf.run_smoother(b, CFG, backend="numba")
    np.testing.assert_array_equal(sa[0], sb[0])
    np.testing.assert_array_equal(sa[1], sb[1])


def test_smoother_needs_covariances(layout):
    z = series(0.5, 151)
    res = ukf.run_filter(z, CFG, layout, Q, backend="numba")
    with pytest.raises(ukf.UKFError):
        ukf.run_smoother(res, CFG, backend="numba")


def test_every_kernel_is_compiled_without_fastmath():
    for name in KERNELS:
        opts = getattr(ukf_numba, name).targetoptions
        assert opts.get("fastmath") is False, (name, opts)
        assert opts.get("nopython") is True, name


def test_kernels_are_cached_on_disk():
    for name in ("_filter_kernel", "_smooth_kernel", "_coupled_fixed_point"):
        assert getattr(ukf_numba, name)._cache is not None, name


def test_no_magic_numbers_prints_or_forbidden_imports_in_ukf_numba():
    path = Path(ukf_numba.__file__)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    allowed = {0, 1, 2, 0.5, 0.0, 1.0, 2.0, -1, -1.0, 3, 4, 5, 6, 7}         # array ranks, layout codes, the sigmoid's 2 e0
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
    assert not imported & {"filterpy", "preprocess", "pysr", "scipy"}, imported
    assert not re.search(r"(?<![A-Za-z_])print\(", text)
    assert "fastmath=True" not in text
