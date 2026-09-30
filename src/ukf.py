"""Reference unscented Kalman filter and unscented RTS smoother (PLAN.md C2; §7.5, §7.6, §9.3;
IMP-019 to IMP-023).

Pure NumPy, no Numba (that is C5). The algorithm is filterpy 1.4.5's, step for step:
scaled sigma points (alpha, beta, kappa from config), SciPy's UPPER-triangular Cholesky
with the ROWS of U as offsets, the same weights and the same summation order. In
particular, as in filterpy, update() uses the sigma points from predict() BEFORE the
process noise Q is added; only the predicted covariance gets Q (IMP-019). So the
innovation covariance S and the gain K contain no Q, and for Q > 0 the filter is not
the exact Kalman filter on a linear model. Do not "fix" this: §7.5 fixes the algorithm.

The core is model-agnostic (propagate and observe callables). The Jansen-Rit glue builds
on C1 (state_space.py): predict with the exogenous delay buffer, observe with the clip of
m, the initial state, and the buffer fill.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cholesky

from src import model
from src import state_space as ss

log = logging.getLogger(__name__)


class UKFError(ValueError):
    """Raised for an inconsistent filter request."""


# ---- sigma points and the unscented transform (filterpy 1.4.5) ---------------------------

def sigma_weights(n, alpha, beta, kappa):
    """(lambda, Wm, Wc) of the scaled unscented transform: 2n + 1 points, centre first."""
    lam = alpha ** 2 * (n + kappa) - n
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=np.float64)
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1.0 - alpha ** 2 + beta)
    return lam, Wm, Wc


def sigma_points(x, P, lam, jitter=0.0, jitter_used=None):
    """[x, x + U_k, x - U_k], U = upper-triangular cholesky((n + lambda) P), k = rows of U.

    jitter (IMP-024): only if the plain Cholesky fails, retry once with P + jitter * I (§7.5:
    P + 1e-9 I must be positive definite); a success then appends to `jitter_used`, and a
    second failure raises LinAlgError. Whenever the plain factorization works the result is
    bit-identical to filterpy's."""
    n = x.shape[0]
    try:
        U = cholesky((lam + n) * P)
    except np.linalg.LinAlgError:
        if jitter <= 0.0:
            raise
        U = cholesky((lam + n) * (P + jitter * np.eye(n)))
        if jitter_used is not None:
            jitter_used.append(1)
    sig = np.empty((2 * n + 1, n), dtype=np.float64)
    sig[0] = x
    sig[1:n + 1] = x + U
    sig[n + 1:] = x - U
    return sig


def unscented_transform(sigmas, Wm, Wc, noise_cov=None):
    x = np.dot(Wm, sigmas)
    y = sigmas - x
    P = np.dot(y.T, np.dot(np.diag(Wc), y))
    if noise_cov is not None:
        P += noise_cov
    return x, P


class UnscentedFilter:
    """filterpy's UnscentedKalmanFilter with an injectable propagation.

    propagate(sigmas, Wm) -> sigmas_f (may hold state, e.g. the delay buffer);
    observe(sigmas_f) -> sigmas_h, both batched over the rows.
    """

    def __init__(self, n, weights, Q, R, propagate, observe, jitter=0.0):
        self.n = n
        self.jitter = jitter
        self._jitter_used = []
        self.lam, self.Wm, self.Wc = weights
        self.Q, self.R = Q, R
        self.propagate, self.observe = propagate, observe
        self.x = np.zeros(n)
        self.P = np.eye(n)
        self.sigmas_f = None

    def predict(self):
        sigmas = sigma_points(self.x, self.P, self.lam, self.jitter, self._jitter_used)
        self.sigmas_f = np.asarray(self.propagate(sigmas, self.Wm), dtype=np.float64)
        self.x, self.P = unscented_transform(self.sigmas_f, self.Wm, self.Wc, self.Q)

    @property
    def n_jitter(self):
        """Number of sigma-point draws that needed the jitter fallback."""
        return len(self._jitter_used)

    def update(self, z):
        sigmas_h = np.atleast_2d(self.observe(self.sigmas_f))
        zp, self.S = unscented_transform(sigmas_h, self.Wm, self.Wc, self.R)
        self.SI = np.linalg.inv(self.S)
        Pxz = np.zeros((self.n, sigmas_h.shape[1]))
        for i in range(self.sigmas_f.shape[0]):
            Pxz += self.Wc[i] * np.outer(self.sigmas_f[i] - self.x, sigmas_h[i] - zp)
        self.K = np.dot(Pxz, self.SI)
        self.zp = zp
        self.y = z - zp
        self.x = self.x + np.dot(self.K, self.y)
        self.P = self.P - np.dot(self.K, np.dot(self.S, self.K.T))


def rts_smooth(xs, Ps, weights, Q, propagate_k, jitter=0.0, jitter_used=None):
    """Unscented RTS smoother, filterpy's rts_smoother recursion (Freestone et al. 2011 style).

    xs, Ps are the filtered means and covariances. propagate_k(k, sigmas, Wm) maps sigma
    points of step k to step k + 1 (it supplies whatever exogenous state that step used).
    Returns smoothed copies of xs and Ps.
    """
    lam, Wm, Wc = weights
    n_steps, n = xs.shape
    xsm, Psm = xs.copy(), Ps.copy()
    for k in reversed(range(n_steps - 1)):
        sigmas = sigma_points(xs[k], Ps[k], lam, jitter, jitter_used)
        sigmas_f = np.asarray(propagate_k(k, sigmas, Wm), dtype=np.float64)
        xb, Pb = unscented_transform(sigmas_f, Wm, Wc, Q)
        Pxb = 0
        for i in range(sigmas_f.shape[0]):
            Pxb += Wc[i] * np.outer(sigmas[i] - xs[k], sigmas_f[i] - xb)
        K = np.dot(Pxb, np.linalg.inv(Pb))
        xsm[k] += np.dot(K, xsm[k + 1] - xb)
        Psm[k] += np.dot(K, Psm[k + 1] - Pb).dot(K.T)
    return xsm, Psm


# ---- Jansen-Rit glue ---------------------------------------------------------------------

def weights_for(n, cfg):
    sp = cfg["ukf"]["sigma_points"]
    return sigma_weights(n, sp["alpha"], sp["beta"], sp["kappa"])


def process_noise(layout, cfg, q):
    """Q per observation step (§7.6): neural states q * (stored steady-state variance);
    parameters a slow random walk, factor * prior variance per second * the step length."""
    dt = 1.0 / cfg["preprocessing"]["observation_fs_hz"]
    prior_var = np.diag(ss.prior_cov(layout, cfg))
    diag = np.empty(layout.n)
    diag[:ss.N_NEURAL] = q * np.tile(ss.neural_variance(cfg), ss.N_NODES)
    diag[ss.N_NEURAL:] = cfg["ukf"]["process_noise"]["parameter_random_walk_factor"] \
        * prior_var[ss.N_NEURAL:] * dt
    return np.diag(diag)


def obs_noise(cfg):
    """R = 0.25 sigma_ref^2 I2 (§7.6)."""
    frac = cfg["ukf"]["observation_noise"]["R_fraction_of_rescaled_variance"]
    return frac * cfg["rescaling"]["sigma_ref"] ** 2 * np.eye(ss.N_NODES)


def initial_state(layout, cfg):
    """Prior mean and covariance at a (re)initialization, plus a freshly filled buffer."""
    return ss.prior_mean(layout, cfg), ss.prior_cov(layout, cfg), ss.make_buffer(cfg)


def buffer_snapshot(buf):
    """Entries by lag 0..delay, shape (delay + 1, nodes)."""
    return np.array([buf.read(lag) for lag in range(buf.delay + 1)])


def buffer_from_snapshot(snap):
    buf = ss.DelayBuffer(snap.shape[0] - 1, snap.shape[1])
    buf.reset(0.0)
    for lag in range(snap.shape[0] - 1, -1, -1):
        buf.write(snap[lag])
    return buf


@dataclass
class FilterResult:
    x: np.ndarray                 # (T, n) filtered means (NaN after n_done)
    P: object                     # (T, n, n) filtered covariances, or None
    z_pred: np.ndarray            # (T, 2) one-step-ahead predicted observation
    S: np.ndarray                 # (T, 2, 2) innovation covariance (no Q in it, IMP-019)
    innovation: np.ndarray        # (T, 2)
    nis: np.ndarray               # (T,) innovation' S^-1 innovation
    min_eig: np.ndarray           # (T,) smallest eigenvalue of the filtered covariance
    snapshots: np.ndarray         # (T, delay + 1, 2) buffer at the start of each step
    layout: object
    q: float
    P_last: object = None         # (n, n) filtered covariance of the last completed step (C4, IMP-040)
    n_done: int = 0
    diverged: bool = False
    divergence_step: object = None
    divergence_reason: object = None
    monitor: dict = field(default_factory=dict)


class DivergenceReference:
    """The reference point of the §7.5 divergence rule (IMP-029): the deterministic fixed point of the
    coupled two-node model at the current posterior-mean parameters (p, rho, g), not at the prior ones.

    center(x) returns the 12-vector reference for the state x. It is cached and recomputed only when
    some parameter mean has moved more than reference_refresh_fraction_of_prior_sd of its prior SD from
    the cached parameters. Quantities removed by the layout follow it (a tied p2 moves with p1, a fixed
    log_rho never moves; M1 has zero gains). If the solve fails, the prior-parameter reference is used,
    the failure is counted, and it is NOT a divergence. A non-finite x returns the cached reference
    (the finiteness check of the divergence rule reports it).
    """
    KEYS = ("p1", "p2", "log_rho1", "log_rho2", "g12", "g21")

    def __init__(self, layout, cfg):
        self.layout, self.cfg = layout, cfg
        self.prior_center = ss.initial_neural_state(cfg)
        self.prior_params = self._params(ss.prior_mean(layout, cfg))
        self.prior_sd = ss.parameter_prior_sd(cfg)
        self.frac = cfg["ukf"]["divergence"]["reference_refresh_fraction_of_prior_sd"]
        self.params, self.ref = self.prior_params, self.prior_center
        self.n_updates = 0
        self.n_fallbacks = 0
        if layout.fixed_params is not None:
            # pass 2 window: the parameters never change, so the reference is solved once, at them (IMP-032)
            self._solve(self.prior_params)

    def _params(self, x):
        q = self.layout.params(np.asarray(x, dtype=np.float64)[None])
        return {k: float(q[k][0]) for k in self.KEYS}

    def center(self, x):
        cur = self._params(x)
        if not all(np.isfinite(v) for v in cur.values()):
            return self.ref
        if all(abs(cur[k] - self.params[k]) <= self.frac * self.prior_sd[k] for k in self.KEYS):
            return self.ref
        self._solve(cur)
        return self.ref

    def _solve(self, cur):
        try:
            self.ref = ss.coupled_steady_state(self.cfg, *(cur[k] for k in self.KEYS))
            self.n_updates += 1
        except (model.ModelError, ss.StateSpaceError, np.linalg.LinAlgError):
            self.ref = self.prior_center
            self.n_fallbacks += 1
        self.params = cur          # also after a failure: retried only after a further move


def _first_divergence(x_post, P_post, cfg, center, sd, min_eig, check_state=True):
    """Reason string if this step is diverged (§7.5), else None. check_state=False suppresses only the
    state-beyond-10-SD flag (the start-up exemption, DEV-003)."""
    if not (np.all(np.isfinite(x_post)) and np.all(np.isfinite(P_post))):
        return "nan_inf"
    if min_eig + cfg["ukf"]["divergence"]["covariance_jitter"] <= 0.0:
        return "covariance_not_pd"
    mult = cfg["ukf"]["divergence"]["state_sd_multiple"]
    if check_state and np.any(np.abs(x_post[:ss.N_NEURAL] - center) > mult * sd):
        return "state_beyond_sd_multiple"
    return None


def use_numba(cfg, backend=None):
    """backend "numba" / "numpy" force a path; None follows ukf.numba.enabled (null or false: NumPy, the
    reference and the default until the C5 validation is accepted, IMP-044)."""
    if backend is None:
        return bool(cfg["ukf"]["numba"]["enabled"])
    if backend not in ("numba", "numpy"):
        raise UKFError(f"backend must be 'numba', 'numpy' or None, got {backend!r}")
    return backend == "numba"


def run_filter(z, cfg, layout, q, x0=None, P0=None, buffer=None, keep_cov=False, backend=None):
    """Forward UKF over one segment z of shape (T, 2), in rescaled mV.

    backend selects the implementation (see use_numba); this function is the NumPy reference and the oracle
    of src/ukf_numba.py.

    Predict then update for every sample, the first included (IMP-020). Stops at the first
    diverged step (IMP-021). After every update the newest buffer entry is replaced by S of
    the FILTERED mean potential; the sub-steps in between hold predicted means (§7.5).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != ss.N_NODES:
        raise UKFError(f"z must have shape (T, {ss.N_NODES}), got {z.shape}")
    if use_numba(cfg, backend):
        from src import ukf_numba
        return ukf_numba.run_filter(z, cfg, layout, q, x0=x0, P0=P0, buffer=buffer, keep_cov=keep_cov)
    T, n = z.shape[0], layout.n
    d_x0, d_P0, d_buf = initial_state(layout, cfg)
    x0 = d_x0 if x0 is None else np.asarray(x0, dtype=np.float64)
    P0 = d_P0 if P0 is None else np.asarray(P0, dtype=np.float64)
    buf = d_buf if buffer is None else buffer
    reference = DivergenceReference(layout, cfg)
    n_exempt = int(round(cfg["ukf"]["divergence"]["startup_exempt_s"] * cfg["preprocessing"]["observation_fs_hz"]))
    sd = np.sqrt(np.tile(ss.neural_variance(cfg), ss.N_NODES))
    filt = UnscentedFilter(
        n, weights_for(n, cfg), process_noise(layout, cfg, q), obs_noise(cfg),
        propagate=lambda s, wm: ss.predict(s, wm, buf, layout, cfg),
        observe=lambda s: ss.observe(s, layout, cfg),
        jitter=cfg["ukf"]["divergence"]["covariance_jitter"])
    filt.x, filt.P = x0.copy(), P0.copy()

    res = FilterResult(
        x=np.full((T, n), np.nan), P=np.full((T, n, n), np.nan) if keep_cov else None,
        z_pred=np.full((T, 2), np.nan), S=np.full((T, 2, 2), np.nan),
        innovation=np.full((T, 2), np.nan), nis=np.full(T, np.nan),
        min_eig=np.full(T, np.nan), snapshots=np.full((T, buf.delay + 1, 2), np.nan),
        layout=layout, q=q)
    for t in range(T):
        res.snapshots[t] = buffer_snapshot(buf)
        try:
            filt.predict()
            filt.update(z[t])
        except np.linalg.LinAlgError as exc:      # IMP-025: nothing else is a divergence
            res.diverged, res.divergence_step = True, t
            res.divergence_reason = f"linalg_error: {exc}"
            break
        sym = 0.5 * (filt.P + filt.P.T)
        finite = bool(np.all(np.isfinite(filt.P)))
        min_eig = float(np.linalg.eigvalsh(sym)[0]) if finite else float("nan")
        res.min_eig[t] = min_eig          # stored before the check: the offending step counts (IMP-025)
        reason = _first_divergence(filt.x, filt.P, cfg, reference.center(filt.x), sd, min_eig,
                                   check_state=t >= n_exempt)
        if reason is not None:
            res.diverged, res.divergence_step, res.divergence_reason = True, t, reason
            break
        buf.replace_latest(ss.sigmoid_of_mean_potential(filt.x[None], [1.0], cfg))
        res.x[t], res.z_pred[t], res.S[t] = filt.x, filt.zp, filt.S
        res.innovation[t] = filt.y
        res.nis[t] = float(filt.y @ filt.SI @ filt.y)
        if keep_cov:
            res.P[t] = filt.P
        res.P_last = filt.P.copy()
        res.n_done = t + 1
    seen = res.min_eig[np.isfinite(res.min_eig)]      # completed steps plus the diverged one, if computed
    res.monitor = {
        "min_eig_overall": float(seen.min()) if seen.size else float("nan"),
        "n_negative_eig_steps": int(np.count_nonzero(seen < 0.0)),
        "n_jitter_fallbacks": filt.n_jitter,
        "startup_exempt_steps": n_exempt,
        "n_reference_updates": reference.n_updates, "n_reference_fallbacks": reference.n_fallbacks,
        "nan_inf_seen": bool(res.divergence_reason == "nan_inf"),
        "diverged": res.diverged, "divergence_step": res.divergence_step,
        "divergence_reason": res.divergence_reason, "n_done": res.n_done,
    }
    return res


def run_smoother(result, cfg, backend=None):
    """Unscented RTS over the completed steps of a forward run with keep_cov=True.

    Each backward step propagates with a copy of the forward buffer as it stood at the start
    of the next forward step, so the delay buffer holds forward-filter means (not smoothed
    ones), as §7.5 requires. Returns (x_smooth, P_smooth) for the first n_done steps.
    """
    if result.P is None:
        raise UKFError("run_filter was called without keep_cov=True")
    if use_numba(cfg, backend):
        from src import ukf_numba
        return ukf_numba.run_smoother(result, cfg)
    n_done = result.n_done
    layout = result.layout
    xs, Ps = result.x[:n_done], result.P[:n_done]

    def propagate_k(k, sigmas, wm):
        buf = buffer_from_snapshot(result.snapshots[k + 1])
        return ss.predict(sigmas, wm, buf, layout, cfg)

    used = []
    out = rts_smooth(xs, Ps, weights_for(layout.n, cfg), process_noise(layout, cfg, result.q),
                     propagate_k, cfg["ukf"]["divergence"]["covariance_jitter"], used)
    if used:
        log.warning("smoother needed the covariance jitter fallback at %d steps", len(used))
    return out
