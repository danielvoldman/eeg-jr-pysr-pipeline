"""Two-node Jansen-Rit generative simulator (PLAN.md A2; §7.1, §7.3, §8.1, IMP-003).

Six-state Grimbert-Faugeras form per node (§7.1):
    y0' = y3
    y3' = A a S(y1 - y2) - 2a y3 - a^2 y0
    y1' = y4
    y4' = A a [p + C2 S(C1 y0) + drive] - 2a y4 - a^2 y1
    y2' = y5
    y5' = B b C4 S(C3 y0) - 2b y5 - b^2 y2
The EEG proxy is y1 - y2. `drive` is the delayed coupling g_ij * S(y1_i - y2_i)(t - d)
(§8.1): it sits in the y4 bracket, alongside the input p.

Gain convention (§8.1): g_ij is the gain from node i to node j, so g12 drives
node 2 from node 1 and g21 drives node 1 from node 2.

Integrator (IMP-003): Heun, with the input drawn once per step and held constant
within the step. The delayed coupling in the second Heun stage reads the history
one step later than the first stage. Nothing here imports scipy or preprocess.py.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from src.config import ConfigError

STATE_NAMES = ("y0", "y1", "y2", "y3", "y4", "y5")
N_STATES = len(STATE_NAMES)
Y0, Y1, Y2, Y3, Y4, Y5 = range(N_STATES)


class ModelError(ValueError):
    """Raised for an inconsistent simulator request."""


@dataclass
class SimResult:
    states: np.ndarray   # (n_steps + 1, n_nodes, 6); row 0 is the initial state
    output: np.ndarray   # (n_steps + 1, n_nodes); y1 - y2
    input: np.ndarray    # (n_steps, n_nodes); the p-input used at each step
    drive: np.ndarray    # (n_steps, n_nodes); stage-1 coupling drive at each step
    fs_hz: float


# ---- constants ---------------------------------------------------------------

def constants(cfg):
    """Scalar model constants from config (§7.3). C1..C4 are multiples of C."""
    jr = cfg["jansen_rit"]
    C = jr["C"]
    return {
        "a": float(jr["a"]), "b": float(jr["b"]),
        "C1": float(C * jr["C1_multiplier"]), "C2": float(C * jr["C2_multiplier"]),
        "C3": float(C * jr["C3_multiplier"]), "C4": float(C * jr["C4_multiplier"]),
        "e0": float(jr["e0"]), "v0": float(jr["v0"]), "r": float(jr["r"]),
    }


def default_half_width(cfg):
    """Half-width of the input uniform, from the two config bounds (§7.3)."""
    ref = cfg["rescaling"]["reference_simulation"]
    return 0.5 * (ref["input_noise_uniform_high_s_inv"]
                  - ref["input_noise_uniform_low_s_inv"])


# ---- sigmoid and right-hand side --------------------------------------------

def sigmoid(v, e0, v0, r):
    """S(v) = 2 e0 / (1 + exp(r (v0 - v))); no offset subtracted (§7.3, LOCKED)."""
    v = np.asarray(v, dtype=np.float64)
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@njit(fastmath=False)
def _sig(v, e0, v0, r):
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@njit(fastmath=False)
def _rhs_node(y, p_in, drive, A, B, a, b, C1, C2, C3, C4, e0, v0, r, out):
    """One node's six derivatives into `out` (§7.1). `drive` is the coupling input."""
    out[Y0] = y[Y3]
    out[Y3] = A * a * _sig(y[Y1] - y[Y2], e0, v0, r) - 2.0 * a * y[Y3] - a * a * y[Y0]
    out[Y1] = y[Y4]
    out[Y4] = (A * a * (p_in + C2 * _sig(C1 * y[Y0], e0, v0, r) + drive)
               - 2.0 * a * y[Y4] - a * a * y[Y1])
    out[Y2] = y[Y5]
    out[Y5] = B * b * C4 * _sig(C3 * y[Y0], e0, v0, r) - 2.0 * b * y[Y5] - b * b * y[Y2]


def jr_rhs(y, p_in, drive, A, B, cfg):
    """Right-hand side for states y of shape (..., 6). p_in, drive, A, B broadcast
    against the leading dimensions of y. Returns an array shaped like y."""
    y = np.asarray(y, dtype=np.float64)
    lead = y.shape[:-1]
    flat = y.reshape(-1, N_STATES)
    p_f, d_f, A_f, B_f = (np.broadcast_to(np.asarray(x, dtype=np.float64), lead).reshape(-1)
                          for x in (p_in, drive, A, B))
    k = constants(cfg)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        _rhs_node(flat[i], p_f[i], d_f[i], A_f[i], B_f[i], k["a"], k["b"], k["C1"],
                  k["C2"], k["C3"], k["C4"], k["e0"], k["v0"], k["r"], out[i])
    return out.reshape(y.shape)


# ---- steady state ------------------------------------------------------------

def _count_grid_roots(g):
    """Roots of a scanned residual: each exact zero on a grid point counts once, and each
    strict sign change between neighbours counts once. A zero never forms a strict sign
    change with its neighbours, so a root on a grid point is not counted twice."""
    sign = np.sign(g)
    return int(np.count_nonzero(sign[:-1] * sign[1:] < 0.0) + np.count_nonzero(sign == 0.0))


def steady_state(p, A, B, cfg):
    """Deterministic fixed point of one uncoupled node at input p (y3 = y4 = y5 = 0).

    Reduces to a scalar root in y0: y0 = (A/a) S(y1 - y2), with
    y1 = (A/a)(p + C2 S(C1 y0)) and y2 = (B/b) C4 S(C3 y0). The residual is scanned
    on a grid for sign changes; if it does not have exactly one root a ModelError is
    raised, so a multi-root parameter set is never resolved silently.
    """
    k = constants(cfg)
    sim = cfg["simulator"]

    def residual(y0):
        y1 = (A / k["a"]) * (p + k["C2"] * sigmoid(k["C1"] * y0, k["e0"], k["v0"], k["r"]))
        y2 = (B / k["b"]) * k["C4"] * sigmoid(k["C3"] * y0, k["e0"], k["v0"], k["r"])
        return y0 - (A / k["a"]) * sigmoid(y1 - y2, k["e0"], k["v0"], k["r"]), y1, y2

    grid = np.linspace(0.0, 2.0 * k["e0"] * A / k["a"], int(sim["steady_state_scan_points"]))
    g = residual(grid)[0]
    n_roots = _count_grid_roots(g)
    if n_roots != 1:
        raise ModelError(f"steady_state: {n_roots} roots at p={p}, A={A}, B={B}; need exactly 1")
    zero = np.flatnonzero(g == 0.0)
    if zero.size:
        y0 = float(grid[zero[0]])
    else:
        i = int(np.flatnonzero(g[:-1] * g[1:] < 0.0)[0])
        lo, hi = float(grid[i]), float(grid[i + 1])
        g_lo = g[i]
        for _ in range(int(sim["steady_state_bisection_iterations"])):
            mid = 0.5 * (lo + hi)
            g_mid = residual(mid)[0]
            if g_mid * g_lo < 0.0:
                hi = mid
            else:
                lo, g_lo = mid, g_mid
        y0 = 0.5 * (lo + hi)
    _, y1, y2 = residual(y0)
    state = np.zeros(N_STATES, dtype=np.float64)
    state[Y0], state[Y1], state[Y2] = y0, y1, y2
    return state


# ---- input draw --------------------------------------------------------------

def draw_input(rng, n_steps, n_nodes, p_mean, cfg, half_width=None):
    """Input to the y4 bracket: uniform on [p - h, p + h], one fresh draw per node per
    step (IMP-003). h defaults to (high - low)/2 from config, so p = 220 gives the
    Jansen-Rit uniform(120, 320). One rng call, shape (n_steps, n_nodes)."""
    h = default_half_width(cfg) if half_width is None else half_width
    p = np.broadcast_to(np.asarray(p_mean, dtype=np.float64), (n_nodes,))
    return rng.uniform(p - h, p + h, size=(n_steps, n_nodes)).astype(np.float64)


# ---- integrator --------------------------------------------------------------

@njit(fastmath=False)
def _simulate_kernel(u_in, y_init, s_fill, G, delay, dt, A, B,
                     a, b, C1, C2, C3, C4, e0, v0, r):
    """Heun over n_steps. S_all[k + delay] holds S(y1 - y2) of the state at step k; the
    first `delay` rows hold the steady-state fill, so stage 1 at step k reads S_all[k]
    (step k - delay) and stage 2 reads S_all[k + 1] (step k + 1 - delay)."""
    n_steps, n = u_in.shape
    states = np.empty((n_steps + 1, n, N_STATES), dtype=np.float64)
    s_all = np.empty((n_steps + 1 + delay, n), dtype=np.float64)
    drive_out = np.empty((n_steps, n), dtype=np.float64)
    for i in range(n):
        for k in range(delay):
            s_all[k, i] = s_fill[i]
        for m in range(N_STATES):
            states[0, i, m] = y_init[i, m]
    k1 = np.empty(N_STATES, dtype=np.float64)
    k2 = np.empty(N_STATES, dtype=np.float64)
    ypred = np.empty(N_STATES, dtype=np.float64)
    for k in range(n_steps):
        for i in range(n):
            s_all[k + delay, i] = _sig(states[k, i, Y1] - states[k, i, Y2], e0, v0, r)
        for j in range(n):
            d1 = 0.0
            d2 = 0.0
            for i in range(n):
                d1 += G[j, i] * s_all[k, i]
                d2 += G[j, i] * s_all[k + 1, i]
            drive_out[k, j] = d1
            _rhs_node(states[k, j], u_in[k, j], d1, A[j], B[j], a, b, C1, C2, C3, C4,
                      e0, v0, r, k1)
            for m in range(N_STATES):
                ypred[m] = states[k, j, m] + dt * k1[m]
            _rhs_node(ypred, u_in[k, j], d2, A[j], B[j], a, b, C1, C2, C3, C4,
                      e0, v0, r, k2)
            for m in range(N_STATES):
                states[k + 1, j, m] = states[k, j, m] + 0.5 * dt * (k1[m] + k2[m])
    return states, drive_out


def simulate(cfg, n_steps, seed=None, *, input=None, n_nodes=2, p=None, A=None, B=None,
             g12=0.0, g21=0.0, y_init=None, half_width=None, kernel=None):
    """Simulate n_steps Heun steps at the config simulation rate.

    Give exactly one of `seed` (the input is drawn with default_rng(seed)) or `input`
    (a pre-drawn (n_steps, n_nodes) array). p, A, B are scalars or per-node arrays and
    default to the §7.3 values; `half_width` overrides the input uniform half-width.
    The history buffer is pre-filled with each node's steady-state S (§7.5), and the
    default initial state is each node's steady state. `kernel` is a test hook: it
    replaces the time loop (default: the compiled one; tests pass
    `_simulate_kernel.py_func` for pure Python) and is not part of the simulator API.
    Raises ConfigError unless the reference-simulation integrator is "heun" and the
    noise redraw interval is "every step", the only scheme implemented (IMP-003).
    """
    ref = cfg["rescaling"]["reference_simulation"]
    if ref["integrator"] != "heun":
        raise ConfigError(f"rescaling.reference_simulation.integrator is {ref['integrator']!r}; "
                          "only 'heun' is implemented")
    if ref["noise_redraw_interval"] != "every step":
        raise ConfigError("rescaling.reference_simulation.noise_redraw_interval is "
                          f"{ref['noise_redraw_interval']!r}; only 'every step' is implemented")
    if (seed is None) == (input is None):
        raise ModelError("give exactly one of seed or input")
    if n_nodes not in (1, 2):
        raise ModelError("n_nodes must be 1 or 2")
    delay = int(cfg["coupling"]["sim_delay_steps"])
    if delay < 1:
        raise ModelError("sim_delay_steps must be >= 1")
    jr = cfg["jansen_rit"]
    p = np.broadcast_to(np.asarray(jr["p_mean"] if p is None else p, dtype=np.float64), (n_nodes,))
    A = np.broadcast_to(np.asarray(jr["A"] if A is None else A, dtype=np.float64), (n_nodes,))
    B = np.broadcast_to(np.asarray(jr["B"] if B is None else B, dtype=np.float64), (n_nodes,))
    if input is None:
        u_in = draw_input(np.random.default_rng(seed), n_steps, n_nodes, p, cfg, half_width)
    else:
        u_in = np.ascontiguousarray(input, dtype=np.float64)
        if u_in.shape != (n_steps, n_nodes):
            raise ModelError(f"input shape {u_in.shape} != {(n_steps, n_nodes)}")
    k = constants(cfg)
    steady = [steady_state(p[i], A[i], B[i], cfg) for i in range(n_nodes)]
    s_fill = np.array([sigmoid(s[Y1] - s[Y2], k["e0"], k["v0"], k["r"]) for s in steady],
                      dtype=np.float64)
    y0 = (np.array(steady, dtype=np.float64) if y_init is None
          else np.ascontiguousarray(y_init, dtype=np.float64))
    if y0.shape != (n_nodes, N_STATES):
        raise ModelError(f"y_init shape {y0.shape} != {(n_nodes, N_STATES)}")
    G = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if n_nodes == 2:
        G[1, 0] = g12   # node 1 -> node 2
        G[0, 1] = g21   # node 2 -> node 1
    fs = float(cfg["rescaling"]["reference_simulation"]["sim_fs_hz"])
    run = _simulate_kernel if kernel is None else kernel
    states, drive = run(u_in, y0, s_fill, G, delay, 1.0 / fs,
                        np.ascontiguousarray(A), np.ascontiguousarray(B),
                        k["a"], k["b"], k["C1"], k["C2"], k["C3"], k["C4"],
                        k["e0"], k["v0"], k["r"])
    return SimResult(states=states, output=states[..., Y1] - states[..., Y2],
                     input=u_in, drive=drive, fs_hz=fs)
