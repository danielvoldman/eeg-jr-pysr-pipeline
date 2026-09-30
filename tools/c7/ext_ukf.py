"""C7 experimental UKF with extra observation-noise states (candidates A and B) and Sage-Husa adaptive R (candidate C).

Experimental, tools only; src/ is not edited. The kernels are edited copies of src/ukf_numba._filter_kernel and
_smooth_kernel (everything not marked below is the original text); tests/test_c7_experimental.py checks that with no
extra states and no adaptation they reproduce src.ukf_numba. The differences:
  * _extra_decay after _predict: the last nx states of every sigma point are multiplied by phi (OU: exp(-dt/tau),
    random-walk offset: 1). The drift of _predict leaves them untouched (it only writes the 12 neural columns).
  * _extra_observe after _observe: the noise states are added to the two observed channels.
  * candidate C: R is a running per-channel estimate (rad = [on, 1/W, cap]) updated from the innovation after the
    NIS is formed; it enters S from the next step on.
  * the extra states are the LAST nx entries of the state; everything else (divergence reference, state-SD rule on
    the 12 neural states, ring buffer) is unchanged.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np
from numba import njit

from src import state_space as ss
from src import ukf
from src import ukf_numba as U
from src.ukf_numba import (I_DONE, I_FALLBACKS, I_JITTER, I_REASON, I_STEP, I_UPDATES, N_INFO, NN, NREF, NS, R_CHOL,  # noqa: F401
                           R_NAN, R_PD, R_SINGULAR, R_STATE, Y1, Y2, _coupled_fixed_point, _observe, _params6,
                           _predict, _sigma_points, _ut)


@njit(cache=True, fastmath=False)
def _extra_decay(X, nx, phi):
    n = X.shape[1]
    for i in range(X.shape[0]):
        for c in range(nx):
            X[i, n - nx + c] = X[i, n - nx + c] * phi[c]


@njit(cache=True, fastmath=False)
def _extra_observe(X, nx, out):
    n = X.shape[1]
    for i in range(X.shape[0]):
        for c in range(nx):
            out[i, c] = out[i, c] + X[i, n - nx + c]


@njit(cache=True, fastmath=False)
def _filter_kernel_ext(z, x, P, ring, head, lam, Wm, Wc, Qm, R, jitter, n_sub, dt, delay,
                   pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, mu, m_lo, m_hi,
                   ref, ref_params, prior_sd, frac, sd12, mult, n_exempt, keep_cov, prior_center, npts_scan, n_bisect, fp_tol, fp_iter, fp_h,
                   nx, phi, rad,
                   out_x, out_P, out_zp, out_S, out_innov, out_nis, out_mineig, out_snap, p_last,
                   info, zp, S, yv, nisv, out_R):
    """Forward filter over all steps. Returns 0 (all steps done) or 1 (diverged; info[1] the reason code,
    info[0] the step). info: 0 step, 1 reason, 2 n_done, 3 n_jitter, 4 reference updates, 5 reference
    fallbacks. The reference tracker (IMP-029) is in-kernel: ref and ref_params are updated in place."""
    T = z.shape[0]
    n = x.shape[0]
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
    Rc = R.copy()
    Rhat = np.array([R[0, 0], R[1, 1]])
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
        _extra_decay(sigf, nx, phi)
        _ut(sigf, Wm, Wc, Qm, xp, Pp)
        for j in range(n):
            x[j] = xp[j]
            for k in range(n):
                P[j, k] = Pp[j, k]
        # update (filterpy form: the points are the propagated ones, before Q)
        _observe(sigf, pidx, pconst, mu, m_lo, m_hi, sh)
        _extra_observe(sigf, nx, sh)
        zpv = np.empty(2)
        Sv = np.empty((2, 2))
        _ut(sh, Wm, Wc, Rc, zpv, Sv)
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
        if rad[0] > 0.5:                       # candidate C: Sage-Husa running R, clipped to [R0, cap]
            for k in range(2):
                e = yv[k] * yv[k] - (Sv[k, k] - Rc[k, k])
                v = (1.0 - rad[1]) * Rhat[k] + rad[1] * e
                v = min(max(v, R[k, k]), rad[2])
                Rhat[k] = v
                Rc[k, k] = v
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
        elif t >= n_exempt:
            for j in range(NN):
                if abs(x[j] - ref[j]) > mult * sd12[j]:
                    reason = R_STATE
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
        out_R[t, 0] = Rc[0, 0]
        out_R[t, 1] = Rc[1, 1]
        for j in range(n):
            for k in range(n):
                p_last[j, k] = P[j, k]
                if keep_cov:
                    out_P[t, j, k] = P[j, k]
        info[I_DONE] = t + 1
    return 0


@njit(cache=True, fastmath=False)
def _smooth_kernel_ext(xs, Ps, snaps, lam, Wm, Wc, Qm, jitter, n_sub, dt, delay, pidx, pconst, ab,
                   a, b, C1, C2, C3, C4, e0, v0, r, nx, phi, xsm, Psm, info):
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
        _extra_decay(sigf, nx, phi)
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


# ---- Python side ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    """kind "A" (OU), "B" (random-walk offset) or "C" (adaptive R); s2, tau per channel for A and B."""
    kind: str
    s2: tuple = (0.0, 0.0)
    tau: tuple = (1.0, 1.0)
    window_s: float = 2.0           # candidate C: forgetting window (the pipeline's 2-s window scale, §10.2)
    r_cap_fraction: float = 1.0     # candidate C: cap on R as a fraction of the rescaled variance

    @property
    def nx(self):
        return 2 if self.kind in ("A", "B") else 0


def extra_terms(spec, cfg):
    """(phi, q_extra, p0_extra) per extra state: exact OU discretization, or a random walk with the same drive."""
    dt = 1.0 / cfg["preprocessing"]["observation_fs_hz"]
    s2, tau = np.asarray(spec.s2, dtype=np.float64), np.asarray(spec.tau, dtype=np.float64)
    if spec.kind == "A":
        phi = np.exp(-dt / tau)
        return phi, s2 * (1.0 - phi ** 2), s2.copy()
    if spec.kind == "B":
        return np.ones(2), 2.0 * s2 / tau * dt, s2.copy()
    return np.ones(0), np.zeros(0), np.zeros(0)


def ext_layout(base, spec):
    return dataclasses.replace(base, names=tuple(base.names) + tuple(f"noise{c + 1}" for c in range(spec.nx)))


def _trim(res, base_layout):
    n0 = base_layout.n
    out = ukf.FilterResult(x=res.x[:, :n0], P=None if res.P is None else res.P[:, :n0, :n0], z_pred=res.z_pred,
                           S=res.S, innovation=res.innovation, nis=res.nis, min_eig=res.min_eig,
                           snapshots=res.snapshots, layout=base_layout, q=res.q,
                           P_last=None if getattr(res, "P_last", None) is None else res.P_last[:n0, :n0],
                           n_done=res.n_done, diverged=res.diverged, divergence_step=res.divergence_step,
                           divergence_reason=res.divergence_reason, monitor=res.monitor)
    out.full = res
    return out


def run_filter_ext(z, cfg, base_layout, q, spec, x0=None, P0=None, buffer=None, keep_cov=False):
    """Extended drop-in for ukf_numba.run_filter: x0, P0 have the base dimension; the noise states start at 0 with
    their stationary variance at every call (every segment). Returns the FULL result (base + extra states) with
    .R_used (T, 2), the R the filter used at each step."""
    z = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
    T, nb, nx = z.shape[0], base_layout.n, spec.nx
    n = nb + nx
    d_x0, d_P0, d_buf = ukf.initial_state(base_layout, cfg)
    x0 = np.array(d_x0 if x0 is None else x0, dtype=np.float64)
    P0 = np.array(d_P0 if P0 is None else P0, dtype=np.float64)
    phi, q_ex, p0_ex = extra_terms(spec, cfg)
    x = np.concatenate([x0, np.zeros(nx)])
    P = np.zeros((n, n))
    P[:nb, :nb] = P0
    for c in range(nx):
        P[nb + c, nb + c] = p0_ex[c]
    Qm = np.zeros((n, n))
    Qm[:nb, :nb] = ukf.process_noise(base_layout, cfg, q)
    for c in range(nx):
        Qm[nb + c, nb + c] = q_ex[c]
    buf = d_buf if buffer is None else buffer
    fs = cfg["preprocessing"]["observation_fs_hz"]
    n_exempt = int(round(cfg["ukf"]["divergence"]["startup_exempt_s"] * fs))
    sd12 = np.sqrt(np.tile(ss.neural_variance(cfg), ss.N_NODES))
    lam, Wm, Wc = U.weights_for(n, cfg)
    R = ukf.obs_noise(cfg)
    dt, n_sub = ss.substep_dt(cfg)
    pidx, pconst = U.layout_arrays(base_layout)
    ab, a, b, C1, C2, C3, C4, e0, v0, r = U._model_consts(cfg)
    mu = cfg["rescaling"]["mu_ref"]
    m_lo, m_hi = cfg["priors"]["m_truncate"]
    jitter = cfg["ukf"]["divergence"]["covariance_jitter"]
    ring = np.array(buf._buf, dtype=np.float64)
    head = np.array([buf._head], dtype=np.int64)
    dv = cfg["ukf"]["divergence"]
    fp = (int(cfg["simulator"]["steady_state_scan_points"]), int(cfg["simulator"]["steady_state_bisection_iterations"]),
          dv["fixed_point_tolerance"], int(dv["fixed_point_max_iterations"]), dv["fixed_point_fd_step"])
    ref, ref_params, prior_center, prior_sd, n_upd0, n_fb0 = U._reference_init(base_layout, cfg, (ab, a, b, C1, C2, C3, C4, e0, v0, r), fp)
    adapt = spec.kind == "C"
    rad = np.array([1.0 if adapt else 0.0, 1.0 / (spec.window_s * fs),
                    spec.r_cap_fraction * cfg["rescaling"]["sigma_ref"] ** 2])
    out_x = np.full((T, n), np.nan)
    out_P = np.full((T, n, n), np.nan) if keep_cov else np.empty((1, n, n))
    out_zp, out_S = np.full((T, 2), np.nan), np.full((T, 2, 2), np.nan)
    out_innov, out_nis = np.full((T, 2), np.nan), np.full(T, np.nan)
    out_mineig = np.full(T, np.nan)
    out_snap = np.full((T, buf.delay + 1, 2), np.nan)
    out_R = np.full((T, 2), np.nan)
    p_last = np.empty((n, n))
    info = np.zeros(N_INFO, dtype=np.int64)
    zp_a, S_a, yv_a, nis_a = np.zeros(2), np.zeros((2, 2)), np.zeros(2), np.zeros(1)
    res = ukf.FilterResult(x=out_x, P=out_P if keep_cov else None, z_pred=out_zp, S=out_S, innovation=out_innov,
                           nis=out_nis, min_eig=out_mineig, snapshots=out_snap,
                           layout=ext_layout(base_layout, spec), q=q)
    code = _filter_kernel_ext(z, x, P, ring, head, lam, Wm, Wc, Qm, R, jitter, n_sub, dt, buf.delay,
                              pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, mu, m_lo, m_hi,
                              ref, ref_params, prior_sd, dv["reference_refresh_fraction_of_prior_sd"], sd12,
                              float(dv["state_sd_multiple"]), n_exempt, bool(keep_cov), prior_center, *fp,
                              nx, phi, rad, out_x, out_P, out_zp, out_S, out_innov, out_nis, out_mineig, out_snap,
                              p_last, info, zp_a, S_a, yv_a, nis_a, out_R)
    res.n_done = int(info[I_DONE])
    if res.n_done > 0:
        res.P_last = p_last.copy()
    if code == 1:
        res.diverged, res.divergence_step = True, int(info[I_STEP])
        rc = int(info[I_REASON])
        res.divergence_reason = U._REASONS[rc] if rc in U._REASONS else U._linalg_reason(rc)
    buf._buf[:] = ring
    buf._head = int(head[0])
    seen = out_mineig[np.isfinite(out_mineig)]
    res.monitor = {
        "min_eig_overall": float(seen.min()) if seen.size else float("nan"),
        "n_negative_eig_steps": int(np.count_nonzero(seen < 0.0)),
        "n_jitter_fallbacks": int(info[I_JITTER]), "startup_exempt_steps": n_exempt,
        "n_reference_updates": n_upd0 + int(info[I_UPDATES]), "n_reference_fallbacks": n_fb0 + int(info[I_FALLBACKS]),
        "nan_inf_seen": bool(res.divergence_reason == "nan_inf"),
        "diverged": res.diverged, "divergence_step": res.divergence_step,
        "divergence_reason": res.divergence_reason, "n_done": res.n_done}
    res.R_used = out_R
    return res


def run_smoother_ext(res, cfg, spec, base_layout):
    """Unscented RTS over the completed steps of a full (extended) forward result with covariances."""
    if res.P is None:
        raise ukf.UKFError("run_filter_ext was called without keep_cov=True")
    n_done, nb, nx = res.n_done, base_layout.n, spec.nx
    n = nb + nx
    xs, Ps = np.ascontiguousarray(res.x[:n_done]), np.ascontiguousarray(res.P[:n_done])
    snaps = np.ascontiguousarray(res.snapshots[:n_done])
    lam, Wm, Wc = U.weights_for(n, cfg)
    phi, q_ex, _ = extra_terms(spec, cfg)
    Qm = np.zeros((n, n))
    Qm[:nb, :nb] = ukf.process_noise(base_layout, cfg, res.q)
    for c in range(nx):
        Qm[nb + c, nb + c] = q_ex[c]
    dt, n_sub = ss.substep_dt(cfg)
    pidx, pconst = U.layout_arrays(base_layout)
    ab, a, b, C1, C2, C3, C4, e0, v0, r = U._model_consts(cfg)
    xsm, Psm = np.empty_like(xs), np.empty_like(Ps)
    info = np.zeros(N_INFO, dtype=np.int64)
    if n_done == 0:
        return xsm, Psm
    delay = snaps.shape[1] - 1
    code = _smooth_kernel_ext(xs, Ps, snaps, lam, Wm, Wc, Qm, cfg["ukf"]["divergence"]["covariance_jitter"], n_sub, dt,
                              delay, pidx, pconst, ab, a, b, C1, C2, C3, C4, e0, v0, r, nx, phi, xsm, Psm, info)
    if code != 0:
        raise np.linalg.LinAlgError(U._LINALG[int(info[I_REASON])])
    return xsm, Psm


class patched_filters:
    """Context manager: passes.run_pass1 (unchanged src code) runs the extended filter and smoother.

    passes.py calls ukf.run_filter / ukf.run_smoother with 19-D (base) x0, P0. The wrapper extends them, runs the
    kernel, and hands passes.py a result trimmed to the base dimension (the noise states are per segment and are not
    carried); the smoother wrapper finds the full result again. Each forward result's monitor is appended to
    `monitors` for the stability report. Candidate C has no extra state and uses the src Numba smoother."""

    def __init__(self, spec):
        self.spec, self.monitors, self.r_used, self.nis = spec, [], [], []

    def __enter__(self):
        self._orig = (ukf.run_filter, ukf.run_smoother)
        spec = self.spec

        def rf(z, cfg, layout, q, x0=None, P0=None, buffer=None, keep_cov=False, backend=None):
            full = run_filter_ext(z, cfg, layout, q, spec, x0=x0, P0=P0, buffer=buffer, keep_cov=keep_cov)
            self.monitors.append(dict(full.monitor, n=int(np.asarray(z).shape[0])))
            burn = int(round(cfg["passes"]["estimator_burn_in_s"] * cfg["preprocessing"]["observation_fs_hz"]))
            v = full.nis[burn:full.n_done]                  # NIS after the 6-s burn-in of each segment
            self.nis.append((float(np.sum(v)), int(v.size)))
            if spec.kind == "C":
                self.r_used.append(np.nanmedian(full.R_used, axis=0))
            return _trim(full, layout)

        def rs(result, cfg, backend=None):
            if spec.kind == "C":
                return U.run_smoother(result, cfg)
            xs, Ps = run_smoother_ext(result.full, cfg, spec, result.layout)
            nb = result.layout.n
            return xs[:, :nb], Ps[:, :nb, :nb]

        ukf.run_filter, ukf.run_smoother = rf, rs
        return self

    def __exit__(self, *exc):
        ukf.run_filter, ukf.run_smoother = self._orig
        return False
