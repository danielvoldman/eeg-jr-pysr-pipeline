"""NumPy extended forward filter with the M3 residual (PLAN G0.5; §8.1, §8.3, §10.2; IMP-077).

M1 and M2 keep their Numba kernels (src/ukf_ext.py, unchanged). M3 adds the frozen PySR residual to dy4/dt of the target
node inside the prediction step, which the Numba kernels cannot call, so M3 runs here, on the NumPy reference pieces
(ukf.UnscentedFilter, ss.predict with a ResidualHook, ss.observe). With no residual this is the same filter as the Numba
extended kernel (tests/test_ukf_resid.py compares them) and the unextended filter (spec kind "N"). Forward only.
"""
import numpy as np

from src import state_space as ss
from src import ukf
from src import ukf_ext


def run_filter_numpy(z, cfg, base_layout, q, spec, residual=None, x0=None, P0=None, buffer=None, keep_cov=False):
    """NumPy forward filter for the extended state with an optional M3 residual (G0.5, IMP-077). Same model, same noise
    states, same divergence rule as run_filter_ext (the Numba kernel is the oracle: tests compare the two with no
    residual), built from the reference pieces (ukf.UnscentedFilter, ss.predict, ss.observe) so the residual enters through
    ss.predict. Forward only: there is no smoother for it. A residual needs a fresh buffer (it owns a potential ring that
    starts at the steady state), so `buffer` must be None with a residual. Returns the FULL result (base + extra states)."""
    z = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
    if residual is not None and buffer is not None:
        raise ukf.UKFError("a residual run starts from the steady-state buffers; do not pass a buffer")
    T, nb, nx = z.shape[0], base_layout.n, spec.nx
    n = nb + nx
    d_x0, d_P0, d_buf = ukf.initial_state(base_layout, cfg)
    x0 = np.array(d_x0 if x0 is None else x0, dtype=np.float64)
    P0 = np.array(d_P0 if P0 is None else P0, dtype=np.float64)
    phi, q_ex, p0_ex = ukf_ext.extra_terms(spec, cfg)
    P = np.zeros((n, n))
    P[:nb, :nb] = P0
    Qm = np.zeros((n, n))
    Qm[:nb, :nb] = ukf.process_noise(base_layout, cfg, q)
    for c in range(nx):
        P[nb + c, nb + c] = p0_ex[c]
        Qm[nb + c, nb + c] = q_ex[c]
    buf = d_buf if buffer is None else buffer
    hook = None if residual is None else residual.bind(ss.make_potential_buffer(cfg))
    n_exempt = int(round(cfg["ukf"]["divergence"]["startup_exempt_s"] * cfg["preprocessing"]["observation_fs_hz"]))
    sd = np.sqrt(np.tile(ss.neural_variance(cfg), ss.N_NODES))
    reference = ukf.DivergenceReference(base_layout, cfg)
    filt = ukf.UnscentedFilter(
        n, ukf.weights_for(n, cfg), Qm, ukf.obs_noise(cfg),
        propagate=lambda s, wm: np.hstack([ss.predict(s[:, :nb], wm, buf, base_layout, cfg, residual=hook), s[:, nb:] * phi]),
        observe=lambda s: ss.observe(s[:, :nb], base_layout, cfg) + (s[:, nb:] if nx else 0.0),
        jitter=cfg["ukf"]["divergence"]["covariance_jitter"])
    filt.x, filt.P = np.concatenate([x0, np.zeros(nx)]), P
    res = ukf.FilterResult(
        x=np.full((T, n), np.nan), P=np.full((T, n, n), np.nan) if keep_cov else None,
        z_pred=np.full((T, 2), np.nan), S=np.full((T, 2, 2), np.nan), innovation=np.full((T, 2), np.nan),
        nis=np.full(T, np.nan), min_eig=np.full(T, np.nan), snapshots=np.full((T, buf.delay + 1, 2), np.nan),
        layout=ukf_ext.ext_layout(base_layout, spec), q=q)
    res.pots_snap = None if hook is None else np.full((T, buf.delay + 1, 2), np.nan)   # C4 free-run start states (IMP-086)
    for t in range(T):
        res.snapshots[t] = ukf.buffer_snapshot(buf)
        if hook is not None:
            res.pots_snap[t] = ukf.buffer_snapshot(hook.pots)
        try:
            filt.predict()
            filt.update(z[t])
        except np.linalg.LinAlgError as exc:
            res.diverged, res.divergence_step = True, t
            res.divergence_reason = f"linalg_error: {exc}"
            break
        sym = 0.5 * (filt.P + filt.P.T)
        finite = bool(np.all(np.isfinite(filt.P)))
        min_eig = float(np.linalg.eigvalsh(sym)[0]) if finite else float("nan")
        res.min_eig[t] = min_eig
        reason = ukf._first_divergence(filt.x, filt.P, cfg, reference.center(filt.x[:nb]), sd, min_eig,
                                       check_state=t >= n_exempt)
        if reason is not None:
            res.diverged, res.divergence_step, res.divergence_reason = True, t, reason
            break
        buf.replace_latest(ss.sigmoid_of_mean_potential(filt.x[None, :nb], [1.0], cfg))
        if hook is not None:
            hook.pots.replace_latest(ss.potentials(filt.x[None, :nb])[0])
        res.x[t], res.z_pred[t], res.S[t] = filt.x, filt.zp, filt.S
        res.innovation[t] = filt.y
        res.nis[t] = float(filt.y @ filt.SI @ filt.y)
        if keep_cov:
            res.P[t] = filt.P
        res.P_last = filt.P.copy()
        res.n_done = t + 1
    seen = res.min_eig[np.isfinite(res.min_eig)]
    res.monitor = {
        "min_eig_overall": float(seen.min()) if seen.size else float("nan"),
        "n_negative_eig_steps": int(np.count_nonzero(seen < 0.0)),
        "n_jitter_fallbacks": filt.n_jitter, "startup_exempt_steps": n_exempt,
        "n_reference_updates": reference.n_updates, "n_reference_fallbacks": reference.n_fallbacks,
        "nan_inf_seen": bool(res.divergence_reason == "nan_inf"),
        "diverged": res.diverged, "divergence_step": res.divergence_step,
        "divergence_reason": res.divergence_reason, "n_done": res.n_done}
    res.pots = None if hook is None else hook.pots
    return res


def make_runners(spec, residual=None):
    """(run_filter, run_smoother) of the NumPy extended filter, with the call signatures of make_runners. The smoother
    does not exist for it: asking for one raises (the M3 residual is scored forward-only, G1)."""

    def rf(z, cfg, layout, q, x0=None, P0=None, buffer=None, keep_cov=False, backend=None):
        return ukf_ext._trim(run_filter_numpy(z, cfg, layout, q, spec, residual=residual, x0=x0, P0=P0, buffer=buffer,
                                          keep_cov=keep_cov), layout)

    def rs(result, cfg, backend=None):
        raise ukf.UKFError("the NumPy extended filter has no smoother (forward-only residual runs)")

    return rf, rs
