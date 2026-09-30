"""Two-node Jansen-Rit simulation with a PLANTED residual and exact truth, for the Stage D tests (§9.1).

A copy of the Heun kernel of src.model with one extra additive term in the y4 bracket,
    r_res_j(t) = c_j * u_src(t - d) * u_tgt(t),   u = (y1 - y2) / SD,
held constant within a step like the input p (so the model is A a [p + C2 S + g S_src(t-d) + r_res] ...).
c_j is set so that RMS(r_res_j) = c_rel x RMS(base coupling drive_j), from a first pass with c = 0 (the §9.1
definition of the level). Test-only: src/model.py is untouched.

The truth at the 256-Hz sample rows (every 8th step) is exact: the states, the delayed source S at the
simulator delay (20 steps = 2.5 samples), the basis u_src(t - d) u_tgt(t) with the exact lag, and dy4/dt
from the right-hand side including the drawn input. The smoothed-state inputs of a window are therefore
the true states (no filter): what is measured is the regression step alone.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from numba import njit

from src import model
from src.model import N_STATES, Y0, Y1, Y2, Y4, _rhs_node, _sig


@njit(fastmath=False)
def _kernel(u_in, y_init, s_fill, pot_fill, G, c, sd, delay, dt, A, B, a, b, C1, C2, C3, C4, e0, v0, r):
    n_steps, n = u_in.shape
    states = np.empty((n_steps + 1, n, N_STATES), dtype=np.float64)
    s_all = np.empty((n_steps + 1 + delay, n), dtype=np.float64)
    pot_all = np.empty((n_steps + 1 + delay, n), dtype=np.float64)
    drive_out = np.empty((n_steps, n), dtype=np.float64)
    res_out = np.empty((n_steps, n), dtype=np.float64)
    for i in range(n):
        for k in range(delay):
            s_all[k, i] = s_fill[i]
            pot_all[k, i] = pot_fill[i]
        for m in range(N_STATES):
            states[0, i, m] = y_init[i, m]
    k1 = np.empty(N_STATES, dtype=np.float64)
    k2 = np.empty(N_STATES, dtype=np.float64)
    ypred = np.empty(N_STATES, dtype=np.float64)
    for k in range(n_steps):
        for i in range(n):
            pot_all[k + delay, i] = states[k, i, Y1] - states[k, i, Y2]
            s_all[k + delay, i] = _sig(pot_all[k + delay, i], e0, v0, r)
        for j in range(n):
            d1 = 0.0
            d2 = 0.0
            src = 1 - j
            for i in range(n):
                d1 += G[j, i] * s_all[k, i]
                d2 += G[j, i] * s_all[k + 1, i]
            res = c[j] * (pot_all[k, src] / sd[src]) * (pot_all[k + delay, j] / sd[j])
            drive_out[k, j] = d1
            res_out[k, j] = res
            _rhs_node(states[k, j], u_in[k, j], d1 + res, A[j], B[j], a, b, C1, C2, C3, C4, e0, v0, r, k1)
            for m in range(N_STATES):
                ypred[m] = states[k, j, m] + dt * k1[m]
            _rhs_node(ypred, u_in[k, j], d2 + res, A[j], B[j], a, b, C1, C2, C3, C4, e0, v0, r, k2)
            for m in range(N_STATES):
                states[k + 1, j, m] = states[k, j, m] + 0.5 * dt * (k1[m] + k2[m])
    return states, drive_out, res_out, s_all, pot_all


@dataclass
class Planted:
    x: np.ndarray            # (N, 12) true states at the observation rows
    s_delayed: np.ndarray    # (N, 2) S of the node's pyramidal potential one delay earlier
    basis: np.ndarray        # (N, 2) u_src(t-d) u_tgt(t), exact lag; column j = target node j
    planted: np.ndarray      # (N, 2) A_j a c_j basis: the planted residual in dy4/dt units
    params: object           # p1, p2, A1, A2, g12, g21 (truth)
    c: np.ndarray
    sd: np.ndarray
    step: int


def _run(cfg, u, g12, g21, p, c, sd):
    k = model.constants(cfg)
    jr = cfg["jansen_rit"]
    A = np.full(2, float(jr["A"]))
    B = np.full(2, float(jr["B"]))
    steady = [model.steady_state(p[i], A[i], B[i], cfg) for i in range(2)]
    s_fill = np.array([model.sigmoid(s[Y1] - s[Y2], k["e0"], k["v0"], k["r"]) for s in steady])
    pot_fill = np.array([s[Y1] - s[Y2] for s in steady])
    G = np.zeros((2, 2))
    G[1, 0], G[0, 1] = g12, g21
    fs = float(cfg["rescaling"]["reference_simulation"]["sim_fs_hz"])
    delay = int(cfg["coupling"]["sim_delay_steps"])
    out = _kernel(np.ascontiguousarray(u), np.array(steady, dtype=np.float64), s_fill, pot_fill, G,
                  np.asarray(c, dtype=np.float64), np.asarray(sd, dtype=np.float64), delay, 1.0 / fs, A, B,
                  k["a"], k["b"], k["C1"], k["C2"], k["C3"], k["C4"], k["e0"], k["v0"], k["r"])
    return out, A, delay


def simulate_planted(cfg, seed, g12, g21, seconds, c_rel=0.5, p=(220.0, 220.0)):
    """One series. c_rel = 0 gives no planted term. Returns a Planted at the 256-Hz rows."""
    fs = float(cfg["rescaling"]["reference_simulation"]["sim_fs_hz"])
    step = int(round(fs / cfg["preprocessing"]["observation_fs_hz"]))
    n_steps = int(seconds * fs)
    u = model.draw_input(np.random.default_rng(seed), n_steps, 2, np.asarray(p, dtype=np.float64), cfg)
    p = np.asarray(p, dtype=np.float64)
    (states, drive, _, s_all, pot_all), A, delay = _run(cfg, u, g12, g21, p, [0.0, 0.0], [1.0, 1.0])
    skip = int(4 * fs)                                              # the same 4-s burn-in for the level
    pot = states[skip:n_steps, :, Y1] - states[skip:n_steps, :, Y2]
    sd = pot.std(axis=0)
    c = np.zeros(2)
    if c_rel > 0.0:
        for j in (0, 1):
            src = 1 - j
            basis = ((pot_all[skip:n_steps, src] / sd[src])
                     * (pot_all[skip + delay:n_steps + delay, j] / sd[j]))
            rms_drive = np.sqrt(np.mean(drive[skip:, j] ** 2))
            c[j] = c_rel * rms_drive / np.sqrt(np.mean(basis ** 2))
    (states, drive, res, s_all, pot_all), A, delay = _run(cfg, u, g12, g21, p, c, sd)
    rows = np.arange(0, n_steps, step)                              # row r = state at step r
    a = model.constants(cfg)["a"]
    x = states[rows].reshape(len(rows), 12)
    s_delayed = s_all[rows]                                         # s_all[r] = S of the state at r - delay
    basis = np.empty((len(rows), 2))
    for j in (0, 1):
        src = 1 - j
        basis[:, j] = (pot_all[rows, src] / sd[src]) * (pot_all[rows + delay, j] / sd[j])
    planted = basis * (A[None, :] * a * c[None, :])
    params = SimpleNamespace(p1=float(p[0]), p2=float(p[1]), A1=float(A[0]), A2=float(A[1]),
                             g12=float(g12), g21=float(g21))
    return Planted(x=x, s_delayed=s_delayed, basis=basis, planted=planted, params=params, c=c, sd=sd, step=step)


def windows(sim, cfg, burn_s=4.0):
    """Non-overlapping 2-s windows of a Planted series after `burn_s`, like pass 2: each item is
    (start_row, x_smooth (n - burn, 12), s_delayed (n - burn, 2)) with the 0.5-s window burn-in cut off."""
    fs = cfg["preprocessing"]["observation_fs_hz"]
    win = int(round(cfg["windows"]["training_window_s"] * fs))
    burn = int(round(cfg["windows"]["training_burn_in_s"] * fs))
    out = []
    pos = int(round(burn_s * fs))
    while pos + win <= sim.x.shape[0]:
        out.append((pos, sim.x[pos + burn:pos + win].copy(), sim.s_delayed[pos + burn:pos + win].copy()))
        pos += win
    return out


def window_truth(sim, start, n_kept, burn, margin):
    """Exact basis and planted residual aligned with window_rows(...) output for the window at `start`:
    node 0 rows first, then node 1, rows t = margin .. n_kept - margin - 1 of the kept samples."""
    lo = start + burn + margin
    hi = start + burn + n_kept - margin
    return (np.concatenate([sim.basis[lo:hi, 0], sim.basis[lo:hi, 1]]),
            np.concatenate([sim.planted[lo:hi, 0], sim.planted[lo:hi, 1]]))
