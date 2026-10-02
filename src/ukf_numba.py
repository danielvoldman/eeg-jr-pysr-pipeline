"""Numba port of the reference UKF and unscented RTS smoother (PLAN.md C5; §7.5, §16.5.2; IMP-044 to IMP-050).

The same algorithm as src/ukf.py, step for step, compiled: filterpy's scaled unscented transform
(alpha, beta, kappa from config, centre weights 0 / 2), SciPy's UPPER-triangular Cholesky convention with
the ROWS of U as sigma-point offsets, the filterpy-form update (sigma points from predict() BEFORE Q; S and
K contain no Q, IMP-019), 4 Heun sub-steps per observation with the exogenous delay ring buffer
(filtered-mean replacement of the newest entry after each update, IMP-018/020), the covariance jitter only
when the plain Cholesky fails (IMP-024), the §7.5 divergence rules with the DEV-003 start-up exemption, the
IMP-029 reference tracker (cached, refreshed after a 5% of prior SD move, counted fallback) and the IMP-028
linalg divergence. float64 throughout; every kernel is compiled with fastmath=False.

The NumPy reference in src/ukf.py stays the oracle and the default; this path is selected by
run_filter(..., backend="numba") or ukf.numba.enabled (IMP-044). Two things stay in Python: the divergence
reference's fixed-point solve (the kernel hands control back to Python when a refresh is due and is then
re-entered at the same step) and the result objects.

Where the port differs from the reference on purpose (IMP-045): the upper Cholesky factor is computed by an
explicit loop (same convention, same triangle read as SciPy's LAPACK potrf, but not the LAPACK routine); the
2 x 2 innovation covariance is inverted in closed form; the matrix products go through np.dot (BLAS) as in the
reference, in the reference's association order.
"""
import logging

import numpy as np
from numba import njit

from src import model
from src import state_space as ss
from src import ukf

log = logging.getLogger(__name__)

NS = model.N_STATES                                   # states per node
NN = ss.N_NEURAL                                      # neural states of the two nodes
Y1, Y2 = model.Y1, model.Y2
NREF = len(ukf.DivergenceReference.KEYS)              # parameters the divergence reference depends on
# divergence reason codes (info[I_REASON]) and the info layout
R_NAN, R_PD, R_STATE, R_CHOL, R_SINGULAR = 1, 2, 3, 4, 5
R_PARAM = 6                                           # DEV-007: parameter_beyond_prior_sd
I_STEP, I_REASON, I_DONE, I_JITTER, I_UPDATES, I_FALLBACKS, N_INFO = 0, 1, 2, 3, 4, 5, 6
_REASONS = {R_NAN: "nan_inf", R_PD: "covariance_not_pd", R_STATE: "state_beyond_sd_multiple",
            R_PARAM: "parameter_beyond_prior_sd"}
_LINALG = {R_CHOL: "Matrix is not positive definite", R_SINGULAR: "Singular matrix"}

# kernel return codes
_DONE, _DIVERGED = 0, 1


# ---- weights (independent of ukf.sigma_weights on purpose) ------------------------------------------

def sigma_weights(n, alpha, beta, kappa):
    """(lambda, Wm, Wc): scaled unscented transform, centre point first (filterpy 1.4.5)."""
    lam = alpha ** 2 * (n + kappa) - n
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=np.float64)
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1.0 - alpha ** 2 + beta)
    return lam, Wm, Wc


def weights_for(n, cfg):
    sp = cfg["ukf"]["sigma_points"]
    return sigma_weights(n, sp["alpha"], sp["beta"], sp["kappa"])


# ---- kernels ---------------------------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _chol_upper(A, U):
    """Upper-triangular U with A = U' U, reading only the upper triangle of A (LAPACK potrf 'U'). Returns
    False if a pivot is not positive (the LinAlgError case)."""
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            U[i, j] = 0.0
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= U[k, j] * U[k, j]
        if not (s > 0.0):
            return False
        d = np.sqrt(s)
        U[j, j] = d
        for i in range(j + 1, n):
            t = A[j, i]
            for k in range(j):
                t -= U[k, j] * U[k, i]
            U[j, i] = t / d
    return True


@njit(cache=True, fastmath=False)
def _sigma_points(x, P, lam, jitter, sig, U, A):
    """[x, x + U_k, x - U_k] with k the ROWS of the upper factor of (lam + n) P. Returns 0 (plain), 1 (jitter
    fallback used) or 2 (failed: LinAlgError)."""
    n = x.shape[0]
    status = 0
    for i in range(n):
        for j in range(n):
            A[i, j] = (lam + n) * P[i, j]
    ok = _chol_upper(A, U)
    if not ok:
        if jitter <= 0.0:
            return 2
        for i in range(n):
            for j in range(n):
                v = P[i, j]
                if i == j:
                    v = v + jitter
                A[i, j] = (lam + n) * v
        ok = _chol_upper(A, U)
        if not ok:
            return 2
        status = 1
    for j in range(n):
        sig[0, j] = x[j]
    for k in range(n):
        for j in range(n):
            sig[1 + k, j] = x[j] + U[k, j]
            sig[n + 1 + k, j] = x[j] - U[k, j]
    return status


@njit(cache=True, fastmath=False)
def _ut(sig, Wm, Wc, noise, x_out, P_out):
    """x = Wm . sigmas, P = y' diag(Wc) y + noise with y = sigmas - x (filterpy order)."""
    npts, n = sig.shape
    x = np.dot(Wm, sig)
    y = np.empty((npts, n))
    wy = np.empty((npts, n))
    for i in range(npts):
        for j in range(n):
            y[i, j] = sig[i, j] - x[j]
            wy[i, j] = Wc[i] * y[i, j]
    P = np.dot(np.ascontiguousarray(y.T), wy)
    for j in range(n):
        x_out[j] = x[j]
        for k in range(n):
            P_out[j, k] = P[j, k] + noise[j, k]


@njit(cache=True, fastmath=False)
def _drift(X, s0, s1, pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, out):
    """Drift of every point; s0, s1 are the exogenous delayed S of node 1 and node 2. Parameters have zero drift."""
    npts, n = X.shape
    for i in range(npts):
        for j in range(n):
            out[i, j] = 0.0
        p1 = X[i, pidx[0]] if pidx[0] >= 0 else pconst[0]
        p2 = X[i, pidx[1]] if pidx[1] >= 0 else pconst[1]
        lr1 = X[i, pidx[2]] if pidx[2] >= 0 else pconst[2]
        lr2 = X[i, pidx[3]] if pidx[3] >= 0 else pconst[3]
        g12 = X[i, pidx[4]] if pidx[4] >= 0 else pconst[4]
        g21 = X[i, pidx[5]] if pidx[5] >= 0 else pconst[5]
        rho1 = np.exp(lr1)
        rho2 = np.exp(lr2)
        A1 = np.sqrt(ab * rho1)
        B1 = np.sqrt(ab / rho1)
        A2 = np.sqrt(ab * rho2)
        B2 = np.sqrt(ab / rho2)
        model._rhs_node(X[i, 0:NS], p1, g21 * s1, A1, B1, a, b, C1, C2, C3, C4, e0, v0, r, out[i, 0:NS])
        model._rhs_node(X[i, NS:NN], p2, g12 * s0, A2, B2, a, b, C1, C2, C3, C4, e0, v0, r, out[i, NS:NN])


@njit(cache=True, fastmath=False)
def _predict(points, wm, ring, head, delay, n_sub, dt, pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, X):
    """Heun sub-steps (no process noise) of the points into X; after each sub-step the ring buffer gets S of
    the weighted-mean potential (the predicted mean)."""
    npts, n = points.shape
    size = ring.shape[0]
    for i in range(npts):
        for j in range(n):
            X[i, j] = points[i, j]
    k1 = np.empty((npts, n))
    k2 = np.empty((npts, n))
    xt = np.empty((npts, n))
    half = 0.5 * dt
    for _ in range(n_sub):
        h1 = (head[0] - delay) % size
        _drift(X, ring[h1, 0], ring[h1, 1], pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, k1)
        for i in range(npts):
            for j in range(n):
                xt[i, j] = X[i, j] + dt * k1[i, j]
        h2 = (head[0] - (delay - 1)) % size
        _drift(xt, ring[h2, 0], ring[h2, 1], pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, k2)
        for i in range(npts):
            for j in range(n):
                X[i, j] = X[i, j] + half * (k1[i, j] + k2[i, j])
        head[0] = (head[0] + 1) % size
        for nd in range(2):
            mv = 0.0
            for i in range(npts):
                mv += wm[i] * (X[i, nd * NS + Y1] - X[i, nd * NS + Y2])
            ring[head[0], nd] = 2.0 * e0 / (1.0 + np.exp(r * (v0 - mv)))


@njit(cache=True, fastmath=False)
def _observe(X, pidx, pconst, mu, lo, hi, out):
    npts = X.shape[0]
    for i in range(npts):
        m = X[i, pidx[6]] if pidx[6] >= 0 else pconst[6]
        if m < lo:
            m = lo
        elif m > hi:
            m = hi
        d0 = (X[i, Y1] - X[i, Y2]) - mu
        d1 = (X[i, NS + Y1] - X[i, NS + Y2]) - mu
        out[i, 0] = mu + (d0 + m * d1)
        out[i, 1] = mu + (m * d0 + d1)


@njit(cache=True, fastmath=False)
def _params6(x, pidx, pconst, out):
    """The six parameters the divergence reference depends on (p1, p2, log_rho1, log_rho2, g12, g21)."""
    for k in range(NREF):
        out[k] = x[pidx[k]] if pidx[k] >= 0 else pconst[k]


@njit(cache=True, fastmath=False)
def _sig(v, e0, v0, r):
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@njit(cache=True, fastmath=False)
def _steady_node(p, A, B, a, b, C1, C2, C3, C4, e0, v0, r, npts, n_bisect, out):
    """model.steady_state for one node: scan on np.linspace(0, 2 e0 A / a, npts), exactly one root or return
    False (ModelError), then bisection. out gets (y0, y1, y2)."""
    stop = 2.0 * e0 * A / a
    step = stop / (npts - 1)
    g = np.empty(npts)
    for i in range(npts):
        y0 = i * step + 0.0
        if i == npts - 1:
            y0 = stop
        y1 = (A / a) * (p + C2 * _sig(C1 * y0, e0, v0, r))
        y2 = (B / b) * C4 * _sig(C3 * y0, e0, v0, r)
        g[i] = y0 - (A / a) * _sig(y1 - y2, e0, v0, r)
    n_roots = 0
    first_zero = -1
    first_change = -1
    for i in range(npts):
        if g[i] == 0.0:
            n_roots += 1
            if first_zero < 0:
                first_zero = i
        if i < npts - 1:
            if (g[i] > 0.0 and g[i + 1] < 0.0) or (g[i] < 0.0 and g[i + 1] > 0.0):    # sign(g)[:-1] * sign(g)[1:] < 0
                n_roots += 1
            if first_change < 0 and g[i] * g[i + 1] < 0.0:                             # g[:-1] * g[1:] < 0 (locating)
                first_change = i
    if n_roots != 1:
        return False
    if first_zero >= 0:
        y0v = first_zero * step
        if first_zero == npts - 1:
            y0v = stop
    else:
        if first_change < 0:
            return False
        i0 = first_change
        lo = i0 * step
        hi = (i0 + 1) * step
        if i0 + 1 == npts - 1:
            hi = stop
        g_lo = g[i0]
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            y1 = (A / a) * (p + C2 * _sig(C1 * mid, e0, v0, r))
            y2 = (B / b) * C4 * _sig(C3 * mid, e0, v0, r)
            g_mid = mid - (A / a) * _sig(y1 - y2, e0, v0, r)
            if g_mid * g_lo < 0.0:
                hi = mid
            else:
                lo = mid
                g_lo = g_mid
        y0v = 0.5 * (lo + hi)
    y1 = (A / a) * (p + C2 * _sig(C1 * y0v, e0, v0, r))
    y2 = (B / b) * C4 * _sig(C3 * y0v, e0, v0, r)
    out[0] = y0v
    out[1] = y1
    out[2] = y2
    return True


@njit(cache=True, fastmath=False)
def _nodes(s, p1, p2, A1, B1, A2, B2, g12, g21, a, b, C1, C2, C3, C4, e0, v0, r, npts, n_bisect, out2, n1, n2):
    ok1 = _steady_node(p1 + g21 * s[1], A1, B1, a, b, C1, C2, C3, C4, e0, v0, r, npts, n_bisect, n1)
    ok2 = _steady_node(p2 + g12 * s[0], A2, B2, a, b, C1, C2, C3, C4, e0, v0, r, npts, n_bisect, n2)
    if not (ok1 and ok2):
        return False
    out2[0] = _sig(n1[1] - n1[2], e0, v0, r)
    out2[1] = _sig(n2[1] - n2[2], e0, v0, r)
    return True


@njit(cache=True, fastmath=False)
def _coupled_fixed_point(p1, p2, lr1, lr2, g12, g21, ab, a, b, C1, C2, C3, C4, e0, v0, r,
                         npts, n_bisect, tol, max_iter, h, out12):
    """state_space.coupled_steady_state (IMP-029): Newton on the two coupled S values with a finite-difference
    Jacobian. Returns 0 (converged; out12 filled) or non-zero (the reference raises ModelError,
    StateSpaceError or LinAlgError: the caller counts a fallback)."""
    rho1 = np.exp(lr1)
    rho2 = np.exp(lr2)
    A1 = np.sqrt(ab * rho1)
    B1 = np.sqrt(ab / rho1)
    A2 = np.sqrt(ab * rho2)
    B2 = np.sqrt(ab / rho2)
    n1 = np.empty(3)
    n2 = np.empty(3)
    f = np.empty(2)
    f2 = np.empty(2)
    s = np.zeros(2)
    if not _nodes(s, p1, p2, A1, B1, A2, B2, g12, g21, a, b, C1, C2, C3, C4, e0, v0, r, npts, n_bisect, f, n1, n2):
        return 1
    for k in range(2):
        s[k] = f[k]
    for _ in range(max_iter):
        if not _nodes(s, p1, p2, A1, B1, A2, B2, g12, g21, a, b, C1, C2, C3, C4, e0, v0, r, npts, n_bisect, f, n1, n2):
            return 1
        r0 = s[0] - f[0]
        r1 = s[1] - f[1]
        if max(abs(r0), abs(r1)) <= tol:
            for k in range(3):
                out12[k] = n1[k]
                out12[6 + k] = n2[k]
            for k in range(3, 6):
                out12[k] = 0.0
                out12[6 + k] = 0.0
            return 0
        jac = np.eye(2)
        for j in range(2):
            d = s.copy()
            d[j] += h
            if not _nodes(d, p1, p2, A1, B1, A2, B2, g12, g21, a, b, C1, C2, C3, C4, e0, v0, r, npts, n_bisect, f2, n1, n2):
                return 1
            jac[0, j] -= (f2[0] - f[0]) / h
            jac[1, j] -= (f2[1] - f[1]) / h
        det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        if det == 0.0:
            return 3
        d0 = (r0 * jac[1, 1] - jac[0, 1] * r1) / det
        d1 = (jac[0, 0] * r1 - jac[1, 0] * r0) / det
        hi = 2.0 * e0
        s[0] = min(max(s[0] - d0, 0.0), hi)
        s[1] = min(max(s[1] - d1, 0.0), hi)
    return 2


@njit(cache=True, fastmath=False)
def _filter_kernel(z, x, P, ring, head, lam, Wm, Wc, Qm, R, jitter, n_sub, dt, delay,
                   pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, mu, m_lo, m_hi,
                   ref, ref_params, prior_sd, frac, sd12, mult, n_exempt, n_dwell, n_pdwell, pcols, pmean, plim, keep_cov, prior_center, npts_scan, n_bisect, fp_tol, fp_iter, fp_h,
                   out_x, out_P, out_zp, out_S, out_innov, out_nis, out_mineig, out_snap, p_last,
                   info, zp, S, yv, nisv):
    """Forward filter over all steps. Returns 0 (all steps done) or 1 (diverged; info[1] the reason code,
    info[0] the step). info: 0 step, 1 reason, 2 n_done, 3 n_jitter, 4 reference updates, 5 reference
    fallbacks. The reference tracker (IMP-029) is in-kernel: ref and ref_params are updated in place."""
    T = z.shape[0]
    n = x.shape[0]
    cnt_s = 0
    cnt_p = 0
    npts = 2 * n + 1
    size = ring.shape[0]
    sig = np.empty((npts, n))
    sigf = np.empty((npts, n))
    U = np.empty((n, n))
    A = np.empty((n, n))
    sh = np.empty((npts, 2))
    SI = np.zeros((2, 2))
    xp = np.empty(n)
    Pp = np.empty((n, n))
    sym = np.empty((n, n))
    cur = np.empty(6)
    fpres = np.empty(NN)
    for t in range(T):
        for lag in range(delay + 1):
            for nd in range(2):
                out_snap[t, lag, nd] = ring[(head[0] - lag) % size, nd]
        st = _sigma_points(x, P, lam, jitter, sig, U, A)
        if st == 2:
            info[I_STEP] = t
            info[I_REASON] = R_CHOL
            return 1
        if st == 1:
            info[I_JITTER] += 1
        _predict(sig, Wm, ring, head, delay, n_sub, dt, pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, sigf)
        _ut(sigf, Wm, Wc, Qm, xp, Pp)
        for j in range(n):
            x[j] = xp[j]
            for k in range(n):
                P[j, k] = Pp[j, k]
        # update (filterpy form: the points are the propagated ones, before Q)
        _observe(sigf, pidx, pconst, mu, m_lo, m_hi, sh)
        zpv = np.empty(2)
        Sv = np.empty((2, 2))
        _ut(sh, Wm, Wc, R, zpv, Sv)
        det = Sv[0, 0] * Sv[1, 1] - Sv[0, 1] * Sv[1, 0]
        if det == 0.0:
            info[I_STEP] = t
            info[I_REASON] = R_SINGULAR
            return 1
        SI[0, 0] = Sv[1, 1] / det
        SI[0, 1] = -Sv[0, 1] / det
        SI[1, 0] = -Sv[1, 0] / det
        SI[1, 1] = Sv[0, 0] / det
        Pxz = np.zeros((n, 2))
        for i in range(npts):
            for j in range(n):
                dxj = sigf[i, j] - x[j]
                for k in range(2):
                    Pxz[j, k] += Wc[i] * (dxj * (sh[i, k] - zpv[k]))
        K = np.dot(Pxz, SI)
        for k in range(2):
            yv[k] = z[t, k] - zpv[k]
            zp[k] = zpv[k]
            for l in range(2):
                S[k, l] = Sv[k, l]
        x_new = x + np.dot(K, yv)
        T1 = np.dot(S, np.ascontiguousarray(K.T))
        P_new = P - np.dot(K, T1)
        for j in range(n):
            x[j] = x_new[j]
            for k in range(n):
                P[j, k] = P_new[j, k]
        w0 = yv[0] * SI[0, 0] + yv[1] * SI[1, 0]
        w1 = yv[0] * SI[0, 1] + yv[1] * SI[1, 1]
        nisv[0] = w0 * yv[0] + w1 * yv[1]
        # ---- divergence checks on the filtered result (§7.5) ----
        finite_P = True
        for j in range(n):
            for k in range(n):
                if not np.isfinite(P[j, k]):
                    finite_P = False
        if finite_P:
            for j in range(n):
                for k in range(n):
                    sym[j, k] = 0.5 * (P[j, k] + P[k, j])
            min_eig = np.linalg.eigvalsh(sym)[0]
        else:
            min_eig = np.nan
        out_mineig[t] = min_eig
        # reference tracker (IMP-029): refresh when a parameter mean moved more than frac of its prior SD
        _params6(x, pidx, pconst, cur)
        all_fin = True
        for k in range(NREF):
            if not np.isfinite(cur[k]):
                all_fin = False
        if all_fin:
            moved = False
            for k in range(NREF):
                if not (abs(cur[k] - ref_params[k]) <= frac * prior_sd[k]):
                    moved = True
            if moved:
                ok = _coupled_fixed_point(cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], ab, a, b, C1, C2, C3, C4,
                                          e0, v0, r, npts_scan, n_bisect, fp_tol, fp_iter, fp_h, fpres)
                if ok == 0:
                    for k in range(NN):
                        ref[k] = fpres[k]
                    info[I_UPDATES] += 1
                else:
                    for k in range(NN):
                        ref[k] = prior_center[k]
                    info[I_FALLBACKS] += 1
                for k in range(NREF):
                    ref_params[k] = cur[k]
        reason = 0
        finite_x = True
        for j in range(n):
            if not np.isfinite(x[j]):
                finite_x = False
        if not (finite_x and finite_P):
            reason = R_NAN
        elif min_eig + jitter <= 0.0:
            reason = R_PD
        else:
            # DEV-007: the state clause needs n_dwell CONSECUTIVE samples beyond the multiple (n_dwell = 1 is the legacy
            # first-crossing rule); the parameter clause needs n_pdwell consecutive samples beyond plim (inf = off)
            if t >= n_exempt:
                beyond = False
                for j in range(NN):
                    if abs(x[j] - ref[j]) > mult * sd12[j]:
                        beyond = True
                if beyond:
                    cnt_s += 1
                    if cnt_s >= n_dwell:
                        reason = R_STATE
                else:
                    cnt_s = 0
            if reason == 0:
                pbeyond = False
                for k in range(7):
                    c = pcols[k]
                    if c >= 0:
                        if abs(x[c] - pmean[k]) > plim[k]:
                            pbeyond = True
                if pbeyond:
                    cnt_p += 1
                    if cnt_p >= n_pdwell:
                        reason = R_PARAM
                else:
                    cnt_p = 0
        if reason != 0:
            info[I_STEP] = t
            info[I_REASON] = reason
            return 1
        # filtered-mean replacement of the newest buffer entry
        for nd in range(2):
            mv = x[nd * NS + Y1] - x[nd * NS + Y2]
            ring[head[0], nd] = 2.0 * e0 / (1.0 + np.exp(r * (v0 - mv)))
        for j in range(n):
            out_x[t, j] = x[j]
        for k in range(2):
            out_zp[t, k] = zp[k]
            out_innov[t, k] = yv[k]
            for l in range(2):
                out_S[t, k, l] = S[k, l]
        out_nis[t] = nisv[0]
        for j in range(n):
            for k in range(n):
                p_last[j, k] = P[j, k]
                if keep_cov:
                    out_P[t, j, k] = P[j, k]
        info[I_DONE] = t + 1
    return 0


@njit(cache=True, fastmath=False)
def _smooth_kernel(xs, Ps, snaps, lam, Wm, Wc, Qm, jitter, n_sub, dt, delay, pidx, pconst, ab,
                   a, b, C1, C2, C3, C4, e0, v0, r, xsm, Psm, info):
    """Unscented RTS recursion (filterpy's rts_smoother). Returns 0, or 1 on a Cholesky failure at step
    info[0] (a LinAlgError in the reference). info[3] counts the jitter fallbacks. Singular Pb raises
    numpy's LinAlgError from np.linalg.inv, as in the reference."""
    T, n = xs.shape
    npts = 2 * n + 1
    size = delay + 1
    sig = np.empty((npts, n))
    sigf = np.empty((npts, n))
    U = np.empty((n, n))
    A_ = np.empty((n, n))
    ring = np.empty((size, 2))
    head = np.zeros(1, dtype=np.int64)
    xb = np.empty(n)
    Pb = np.empty((n, n))
    for j in range(T):
        for k in range(n):
            xsm[j, k] = xs[j, k]
            for l in range(n):
                Psm[j, k, l] = Ps[j, k, l]
    for k in range(T - 2, -1, -1):
        st = _sigma_points(xs[k], Ps[k], lam, jitter, sig, U, A_)
        if st == 2:
            info[I_STEP] = k
            info[I_REASON] = R_CHOL
            return 1
        if st == 1:
            info[I_JITTER] += 1
        for lag in range(delay + 1):
            for nd in range(2):
                ring[(-lag) % size, nd] = snaps[k + 1, lag, nd]
        head[0] = 0
        _predict(sig, Wm, ring, head, delay, n_sub, dt, pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, sigf)
        _ut(sigf, Wm, Wc, Qm, xb, Pb)
        Pxb = np.zeros((n, n))
        for i in range(npts):
            for j in range(n):
                dj = sig[i, j] - xs[k, j]
                for l in range(n):
                    Pxb[j, l] += Wc[i] * (dj * (sigf[i, l] - xb[l]))
        K = np.dot(Pxb, np.linalg.inv(Pb))
        dx = np.empty(n)
        for j in range(n):
            dx[j] = xsm[k + 1, j] - xb[j]
        upd = np.dot(K, dx)
        for j in range(n):
            xsm[k, j] += upd[j]
        dP = np.empty((n, n))
        for j in range(n):
            for l in range(n):
                dP[j, l] = Psm[k + 1, j, l] - Pb[j, l]
        upP = np.dot(np.dot(K, dP), np.ascontiguousarray(K.T))
        for j in range(n):
            for l in range(n):
                Psm[k, j, l] += upP[j, l]
    return 0


# ---- Python wrappers ------------------------------------------------------------------------------------------

def layout_arrays(layout):
    """(pidx, pconst): for the seven parameters p1, p2, log_rho1, log_rho2, g12, g21, m the state index that
    supplies each (-1: a constant), following Layout.params."""
    pidx = np.full(7, -1, dtype=np.int64)
    pconst = np.zeros(7, dtype=np.float64)
    if layout.fixed_params is not None:
        pconst[:] = [layout.fixed_params[k] for k in ss.PARAM_NAMES_FULL]
        return pidx, pconst
    ix = layout.idx
    pidx[0] = ix["p1"]
    pidx[1] = ix["p1"] if layout.tie_p else ix["p2"]
    for k, name in ((2, "log_rho1"), (3, "log_rho2")):
        if name in ix:
            pidx[k] = ix[name]
        else:
            pconst[k] = layout.log_rho_fixed
    if layout.include_gains:
        pidx[4] = ix["g12"]
        pidx[5] = ix["g12"] if layout.tie_g else ix["g21"]
    pidx[6] = ix["m"]
    return pidx, pconst


def _model_consts(cfg):
    k = model.constants(cfg)
    return (float(cfg["priors"]["AB_product"]), k["a"], k["b"], k["C1"], k["C2"], k["C3"], k["C4"], k["e0"], k["v0"], k["r"])


def _reference_init(layout, cfg, consts, fp):
    """The DivergenceReference state at construction (IMP-029): the reference (12), the cached parameters (6, in
    ukf.DivergenceReference.KEYS order), the prior centre, the prior SDs, and the initial update/fallback counts. A
    fixed-parameter layout (pass 2 windows) solves once at the prior parameters, as the reference class does."""
    keys = ukf.DivergenceReference.KEYS
    q = layout.params(ss.prior_mean(layout, cfg)[None])
    prior_params = np.array([float(q[k][0]) for k in keys], dtype=np.float64)
    prior_center = np.asarray(ss.initial_neural_state(cfg), dtype=np.float64)
    prior_sd = ss.parameter_prior_sd(cfg)
    ref, n_upd, n_fb = prior_center.copy(), 0, 0
    if layout.fixed_params is not None:
        out = np.empty(NN)
        if _coupled_fixed_point(*prior_params, *consts, *fp, out) == 0:
            ref, n_upd = out, 1
        else:
            n_fb = 1
    return (ref, prior_params.copy(), prior_center, np.array([prior_sd[k] for k in keys], dtype=np.float64),
            n_upd, n_fb)


def _linalg_reason(code):
    return f"linalg_error: {_LINALG[code]}"


def run_filter(z, cfg, layout, q, x0=None, P0=None, buffer=None, keep_cov=False):
    """Drop-in for ukf.run_filter (same arguments, same FilterResult)."""
    z = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
    if z.ndim != 2 or z.shape[1] != ss.N_NODES:
        raise ukf.UKFError(f"z must have shape (T, {ss.N_NODES}), got {z.shape}")
    T, n = z.shape[0], layout.n
    d_x0, d_P0, d_buf = ukf.initial_state(layout, cfg)
    x = np.array(d_x0 if x0 is None else x0, dtype=np.float64)
    P = np.array(d_P0 if P0 is None else P0, dtype=np.float64)
    buf = d_buf if buffer is None else buffer
    fs = cfg["preprocessing"]["observation_fs_hz"]
    n_exempt = int(round(cfg["ukf"]["divergence"]["startup_exempt_s"] * fs))
    dwell = ukf.dwell_settings(layout, cfg)
    sd12 = np.sqrt(np.tile(ss.neural_variance(cfg), ss.N_NODES))
    lam, Wm, Wc = weights_for(n, cfg)
    Qm, R = ukf.process_noise(layout, cfg, q), ukf.obs_noise(cfg)
    dt, n_sub = ss.substep_dt(cfg)
    pidx, pconst = layout_arrays(layout)
    ab, a, b, C1, C2, C3, C4, e0, v0, r = _model_consts(cfg)
    mu = cfg["rescaling"]["mu_ref"]
    if mu is None:
        raise ss.StateSpaceError("rescaling.mu_ref is null")
    m_lo, m_hi = cfg["priors"]["m_truncate"]
    jitter = cfg["ukf"]["divergence"]["covariance_jitter"]
    ring = np.array(buf._buf, dtype=np.float64)
    head = np.array([buf._head], dtype=np.int64)
    dv = cfg["ukf"]["divergence"]
    fp = (int(cfg["simulator"]["steady_state_scan_points"]), int(cfg["simulator"]["steady_state_bisection_iterations"]),
          dv["fixed_point_tolerance"], int(dv["fixed_point_max_iterations"]), dv["fixed_point_fd_step"])
    ref, ref_params, prior_center, prior_sd, n_upd0, n_fb0 = _reference_init(layout, cfg, (ab, a, b, C1, C2, C3, C4, e0, v0, r), fp)

    out_x = np.full((T, n), np.nan)
    out_P = np.full((T, n, n), np.nan) if keep_cov else np.empty((1, n, n))
    out_zp, out_S = np.full((T, 2), np.nan), np.full((T, 2, 2), np.nan)
    out_innov, out_nis = np.full((T, 2), np.nan), np.full(T, np.nan)
    out_mineig = np.full(T, np.nan)
    out_snap = np.full((T, buf.delay + 1, 2), np.nan)
    p_last = np.empty((n, n))
    info = np.zeros(N_INFO, dtype=np.int64)
    zp_a, S_a, yv_a, nis_a = np.zeros(2), np.zeros((2, 2)), np.zeros(2), np.zeros(1)
    res = ukf.FilterResult(x=out_x, P=out_P if keep_cov else None, z_pred=out_zp, S=out_S, innovation=out_innov,
                           nis=out_nis, min_eig=out_mineig, snapshots=out_snap, layout=layout, q=q)
    code = _filter_kernel(z, x, P, ring, head, lam, Wm, Wc, Qm, R, jitter, n_sub, dt, buf.delay,
                          pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, mu, m_lo, m_hi,
                          ref, ref_params, prior_sd, dv["reference_refresh_fraction_of_prior_sd"], sd12,
                          float(dv["state_sd_multiple"]), n_exempt, *dwell, bool(keep_cov), prior_center, *fp,
                          out_x, out_P, out_zp, out_S, out_innov, out_nis, out_mineig, out_snap, p_last,
                          info, zp_a, S_a, yv_a, nis_a)
    res.n_done = int(info[I_DONE])
    if res.n_done > 0:
        res.P_last = p_last.copy()
    if code == _DIVERGED:
        res.diverged, res.divergence_step = True, int(info[I_STEP])
        rc = int(info[I_REASON])
        res.divergence_reason = _REASONS[rc] if rc in _REASONS else _linalg_reason(rc)
    buf._buf[:] = ring
    buf._head = int(head[0])
    seen = out_mineig[np.isfinite(out_mineig)]
    res.monitor = {
        "min_eig_overall": float(seen.min()) if seen.size else float("nan"),
        "n_negative_eig_steps": int(np.count_nonzero(seen < 0.0)),
        "n_jitter_fallbacks": int(info[I_JITTER]),
        "startup_exempt_steps": n_exempt,
        "n_reference_updates": n_upd0 + int(info[I_UPDATES]), "n_reference_fallbacks": n_fb0 + int(info[I_FALLBACKS]),
        "nan_inf_seen": bool(res.divergence_reason == "nan_inf"),
        "diverged": res.diverged, "divergence_step": res.divergence_step,
        "divergence_reason": res.divergence_reason, "n_done": res.n_done,
    }
    return res


def run_smoother(result, cfg):
    """Drop-in for ukf.run_smoother: unscented RTS over the completed steps of a forward run with keep_cov=True."""
    if result.P is None:
        raise ukf.UKFError("run_filter was called without keep_cov=True")
    n_done = result.n_done
    layout = result.layout
    xs = np.ascontiguousarray(result.x[:n_done])
    Ps = np.ascontiguousarray(result.P[:n_done])
    snaps = np.ascontiguousarray(result.snapshots[:n_done])
    n = layout.n
    lam, Wm, Wc = weights_for(n, cfg)
    Qm = ukf.process_noise(layout, cfg, result.q)
    dt, n_sub = ss.substep_dt(cfg)
    pidx, pconst = layout_arrays(layout)
    ab, a, b, C1, C2, C3, C4, e0, v0, r = _model_consts(cfg)
    xsm, Psm = np.empty_like(xs), np.empty_like(Ps)
    info = np.zeros(N_INFO, dtype=np.int64)
    delay = snaps.shape[1] - 1 if n_done else cfg["coupling"]["delay_substeps"]
    if n_done == 0:
        return xsm, Psm
    code = _smooth_kernel(xs, Ps, snaps, lam, Wm, Wc, Qm, cfg["ukf"]["divergence"]["covariance_jitter"], n_sub, dt,
                          delay, pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, xsm, Psm, info)
    if code != 0:
        raise np.linalg.LinAlgError(_LINALG[int(info[I_REASON])])
    if info[I_JITTER]:
        log.warning("smoother needed the covariance jitter fallback at %d steps", int(info[I_JITTER]))
    return xsm, Psm
