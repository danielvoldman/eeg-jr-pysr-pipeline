"""Augmented state-space model, no filter (PLAN.md C1; §7.1-§7.6, §8.1; IMP-014 to IMP-018).

State layout (§7.4): 12 neural states (per node y0..y5, node 1 first) followed by the
slow parameters p1, p2, log_rho1, log_rho2, g12, g21, m (19-D). M1 drops g12 and g21
(17-D). The three §7.4 reduction switches remove log_rho1/log_rho2, p2 or g21 from the
state through the same Layout code path; a removed quantity is pinned (log_rho at its
prior mean) or tied (p2 = p1, g21 = g12).

Gains g_ij are dimensionless: they multiply S, which is already in 1/s (IMP-014).
The gain from node i to node j is g_ij, as in the simulator (IMP-003 d).

The delayed coupling input is an EXOGENOUS argument of the drift (§7.5). It is read from
a DelayBuffer of S(mean potential) of the source node, S of the mean and not the mean of
S. This module holds no sigma-point code, no filter update and no process noise.
"""
import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
from numba import njit

from src import model
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.model import N_STATES, Y1, Y2

log = logging.getLogger(__name__)

N_NODES = 2
N_NEURAL = N_NODES * N_STATES
PARAM_NAMES_FULL = ("p1", "p2", "log_rho1", "log_rho2", "g12", "g21", "m")
NEURAL_NAMES = tuple(f"n{j + 1}_{name}" for j in range(N_NODES) for name in model.STATE_NAMES)
FULL_NAMES = NEURAL_NAMES + PARAM_NAMES_FULL
N_AUG = len(FULL_NAMES)
# Index constants of the full 19-D layout. Other layouts use Layout.idx.
P1, P2, LOG_RHO1, LOG_RHO2, G12, G21, M = range(N_NEURAL, N_AUG)


class StateSpaceError(ValueError):
    """Raised for an inconsistent state-space request."""


# ---- layout ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Which quantities are in the state vector, and how the others are supplied."""
    names: tuple
    tie_p: bool
    tie_g: bool
    include_gains: bool
    log_rho_fixed: float     # value of log_rho when it is not in the state

    @property
    def n(self):
        return len(self.names)

    @property
    def idx(self):
        return {name: i for i, name in enumerate(self.names)}

    def params(self, X):
        """Dict of the seven parameters, each an array of shape X.shape[:-1]."""
        X = np.asarray(X, dtype=np.float64)
        ix = self.idx
        zero = np.zeros(X.shape[:-1], dtype=np.float64)

        def get(name, default):
            return X[..., ix[name]] if name in ix else default

        p1 = X[..., ix["p1"]]
        g12 = get("g12", zero) if self.include_gains else zero
        g21 = (g12 if self.tie_g else get("g21", zero)) if self.include_gains else zero
        fixed = np.full(X.shape[:-1], self.log_rho_fixed, dtype=np.float64)
        return {"p1": p1, "p2": p1 if self.tie_p else X[..., ix["p2"]],
                "log_rho1": get("log_rho1", fixed), "log_rho2": get("log_rho2", fixed),
                "g12": g12, "g21": g21, "m": X[..., ix["m"]]}


def log_rho_prior_mean(cfg):
    """log(A0/B0) from the literature A and B (§7.6)."""
    return float(np.log(cfg["jansen_rit"]["A"] / cfg["jansen_rit"]["B"]))


def make_layout(cfg, include_gains=True):
    """Layout from the config reduction switches (§7.4). include_gains=False is M1."""
    sw = cfg["state"]["reduction_switches"]
    drop = set()
    if sw["fix_EI_terms"]:
        drop |= {"log_rho1", "log_rho2"}
    if sw["tie_p1_p2"]:
        drop.add("p2")
    if not include_gains:
        drop |= {"g12", "g21"}
    elif sw["tie_g12_g21"]:
        drop.add("g21")
    names = tuple(n for n in FULL_NAMES if n not in drop)
    return Layout(names=names, tie_p=bool(sw["tie_p1_p2"]),
                  tie_g=bool(sw["tie_g12_g21"]) and include_gains,
                  include_gains=include_gains, log_rho_fixed=log_rho_prior_mean(cfg))


# ---- E/I reparameterization (§7.6) -----------------------------------------------------

def rho_to_AB(rho, cfg):
    """A = sqrt(AB rho), B = sqrt(AB / rho) with the product AB held fixed."""
    ab = cfg["priors"]["AB_product"]
    rho = np.asarray(rho, dtype=np.float64)
    return np.sqrt(ab * rho), np.sqrt(ab / rho)


def AB_to_rho(A, B):
    return np.asarray(A, dtype=np.float64) / np.asarray(B, dtype=np.float64)


# ---- initial state, priors (§7.6) -------------------------------------------------------

def neural_variance(cfg):
    """The 6 stored per-state variances (y0..y5), shared by both nodes (IMP-017)."""
    v = cfg["ukf"]["initial_state"]["neural_variance"]
    if v is None:
        raise StateSpaceError("ukf.initial_state.neural_variance is null; run "
                              "python -m src.state_space --compute-initial-covariance")
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (N_STATES,):
        raise StateSpaceError(f"neural_variance has shape {v.shape}, expected {(N_STATES,)}")
    return v


def initial_node_state(cfg):
    """Deterministic steady state of one node at the prior parameter means (§7.6)."""
    jr = cfg["jansen_rit"]
    return model.steady_state(cfg["priors"]["p_mean"], jr["A"], jr["B"], cfg)


def initial_neural_state(cfg):
    """12 neural states at the prior parameter means; used for every (re)initialization."""
    return np.tile(initial_node_state(cfg), N_NODES)


def initial_buffer_fill(cfg):
    """S(y1 - y2) at the initial state, per node: the delay buffer fill (§7.5)."""
    k = model.constants(cfg)
    s = initial_node_state(cfg)
    return np.full(N_NODES, model.sigmoid(s[Y1] - s[Y2], k["e0"], k["v0"], k["r"]),
                   dtype=np.float64)


def prior_mean(layout, cfg):
    pr = cfg["priors"]
    values = {"p1": pr["p_mean"], "p2": pr["p_mean"], "log_rho1": layout.log_rho_fixed,
              "log_rho2": layout.log_rho_fixed, "g12": cfg["coupling"]["gain_prior_mean"],
              "g21": cfg["coupling"]["gain_prior_mean"], "m": pr["m_mean"]}
    neural = initial_neural_state(cfg)
    return np.array([neural[i] if i < N_NEURAL else values[name]
                     for i, name in enumerate(layout.names)], dtype=np.float64)


def prior_cov(layout, cfg):
    """Diagonal prior covariance. m is the raw Gaussian: the truncation to [0, 0.5] is
    applied only inside observe() (IMP-016)."""
    pr = cfg["priors"]
    g_sd = cfg["coupling"]["gain_prior_sd_factor_of_C2"] * model.constants(cfg)["C2"]
    var = {"p1": pr["p_sd"] ** 2, "p2": pr["p_sd"] ** 2, "log_rho1": pr["log_rho_sd"] ** 2,
           "log_rho2": pr["log_rho_sd"] ** 2, "g12": g_sd ** 2, "g21": g_sd ** 2,
           "m": pr["m_sd"] ** 2}
    neural = np.tile(neural_variance(cfg), N_NODES)
    return np.diag(np.array([neural[i] if i < N_NEURAL else var[name]
                             for i, name in enumerate(layout.names)], dtype=np.float64))


# ---- drift (§7.1, §8.1) ----------------------------------------------------------------

@njit(fastmath=False)
def _drift_kernel(Xn, p, drive, A, B, a, b, C1, C2, C3, C4, e0, v0, r, out):
    n_pts = Xn.shape[0]
    for i in range(n_pts):
        for j in range(N_NODES):
            lo = j * N_STATES
            model._rhs_node(Xn[i, lo:lo + N_STATES], p[i, j], drive[i, j], A[i, j], B[i, j],
                            a, b, C1, C2, C3, C4, e0, v0, r, out[i, lo:lo + N_STATES])


def drift(X, s_delayed, layout, cfg):
    """Deterministic drift f(x, s_delayed) of the augmented state, shape like X.

    s_delayed is the EXOGENOUS delayed S of each node, shape (2,) or (n_points, 2):
    node 2's y4 bracket receives g12 * s_delayed[0], node 1's receives g21 * s_delayed[1].
    The parameter states have zero drift (their random walk is process noise, added by
    the filter, not here). A and B come from the point's own log_rho (§7.6).
    """
    X = np.asarray(X, dtype=np.float64)
    flat = np.ascontiguousarray(X.reshape(-1, layout.n))
    s = np.broadcast_to(np.asarray(s_delayed, dtype=np.float64), (flat.shape[0], N_NODES))
    q = layout.params(flat)
    A1, B1 = rho_to_AB(np.exp(q["log_rho1"]), cfg)
    A2, B2 = rho_to_AB(np.exp(q["log_rho2"]), cfg)
    p = np.ascontiguousarray(np.stack([q["p1"], q["p2"]], axis=1))
    drive = np.ascontiguousarray(np.stack([q["g21"] * s[:, 1], q["g12"] * s[:, 0]], axis=1))
    A = np.ascontiguousarray(np.stack([A1, A2], axis=1))
    B = np.ascontiguousarray(np.stack([B1, B2], axis=1))
    k = model.constants(cfg)
    out = np.zeros_like(flat)
    _drift_kernel(np.ascontiguousarray(flat[:, :N_NEURAL]), p, drive, A, B, k["a"], k["b"],
                  k["C1"], k["C2"], k["C3"], k["C4"], k["e0"], k["v0"], k["r"], out)
    return out.reshape(X.shape)


# ---- delay buffer (§7.5) ---------------------------------------------------------------

class DelayBuffer:
    """Ring buffer of S(mean potential) per node, indexed by sub-step.

    The newest entry belongs to the current state (sub-step n). read(lag) returns the
    entry written lag sub-steps ago. A Heun step from sub-step n reads lag = delay in
    stage 1 (entry n - delay) and lag = delay - 1 in stage 2 (entry n + 1 - delay), which
    is the simulator's convention (IMP-003): an entry written for the state at sub-step n
    is read by stage 1 of the step from n + delay and stage 2 of the step from n + delay - 1.
    """

    def __init__(self, delay, n_nodes=N_NODES):
        if int(delay) < 1:
            raise StateSpaceError("delay must be at least 1 sub-step")
        self.delay = int(delay)
        self._buf = np.zeros((self.delay + 1, n_nodes), dtype=np.float64)
        self._head = 0

    def reset(self, fill):
        """Fill every entry with the steady-state S (after any (re)initialization)."""
        self._buf[:] = np.asarray(fill, dtype=np.float64)
        self._head = 0

    def write(self, s):
        """Append the entry of the next sub-step."""
        self._head = (self._head + 1) % self._buf.shape[0]
        self._buf[self._head] = np.asarray(s, dtype=np.float64)

    def replace_latest(self, s):
        """Overwrite the newest entry (the filter replaces the predicted mean by the
        filtered mean at every observation)."""
        self._buf[self._head] = np.asarray(s, dtype=np.float64)

    def read(self, lag):
        if not 0 <= lag <= self.delay:
            raise StateSpaceError(f"lag {lag} outside 0..{self.delay}")
        return self._buf[(self._head - lag) % self._buf.shape[0]].copy()


def make_buffer(cfg):
    """A DelayBuffer of coupling.delay_substeps entries, filled with the steady-state S."""
    buf = DelayBuffer(cfg["coupling"]["delay_substeps"])
    buf.reset(initial_buffer_fill(cfg))
    return buf


def potentials(X):
    """y1 - y2 of both nodes, shape X.shape[:-1] + (2,)."""
    X = np.asarray(X, dtype=np.float64)
    return np.stack([X[..., j * N_STATES + Y1] - X[..., j * N_STATES + Y2]
                     for j in range(N_NODES)], axis=-1)


def sigmoid_of_mean_potential(X, wm, cfg):
    """S(weighted mean of y1 - y2) per node: S of the mean, not the mean of S (§7.5)."""
    k = model.constants(cfg)
    mean_v = np.asarray(wm, dtype=np.float64) @ potentials(X)
    return model.sigmoid(mean_v, k["e0"], k["v0"], k["r"])


# ---- predict (§7.5, §7.6) --------------------------------------------------------------

def substep_dt(cfg, n_substeps=None):
    n = cfg["ukf"]["substeps_per_observation"] if n_substeps is None else n_substeps
    return 1.0 / (cfg["preprocessing"]["observation_fs_hz"] * n), int(n)


def predict(points, wm, buffer, layout, cfg, n_substeps=None):
    """Propagate the points over one observation interval: Heun sub-steps, NO process noise.

    `wm` are the mean weights. After every sub-step the buffer gets S of the weighted
    mean potential (the predicted mean); the filter replaces the last entry by the
    filtered mean after its update (buffer.replace_latest). Returns the propagated points.
    """
    dt, n_sub = substep_dt(cfg, n_substeps)
    X = np.array(points, dtype=np.float64, copy=True)
    for _ in range(n_sub):
        k1 = drift(X, buffer.read(buffer.delay), layout, cfg)
        k2 = drift(X + dt * k1, buffer.read(buffer.delay - 1), layout, cfg)
        X = X + 0.5 * dt * (k1 + k2)
        buffer.write(sigmoid_of_mean_potential(X, wm, cfg))
    return X


# ---- observation (§7.2) -----------------------------------------------------------------

def observe(X, layout, cfg):
    """y_obs = mu_ref + M @ (y - mu_ref), M = [[1, m], [m, 1]], y = (y1 - y2) per node.

    Mixing acts on the deviation from mu_ref, so the DC level is kept (IMP-015). m is
    clipped to priors.m_truncate here and nowhere else (IMP-016). Units: mV.
    """
    mu = cfg["rescaling"]["mu_ref"]
    if mu is None:
        raise StateSpaceError("rescaling.mu_ref is null")
    lo, hi = cfg["priors"]["m_truncate"]
    m = np.clip(layout.params(X)["m"], lo, hi)
    dev = potentials(X) - mu
    return mu + np.stack([dev[..., 0] + m * dev[..., 1], m * dev[..., 0] + dev[..., 1]], axis=-1)


# ---- per-state variance from a noise-driven simulation (§7.6; IMP-017) --------------------

def neural_variance_from_simulation(cfg, seed=None):
    """Per-state variance of y0..y5: two nodes at g = 0, literature parameters, 2048 Hz,
    burn-in discarded, ddof 1, pooled (averaged) over the two nodes by state index."""
    ini = cfg["ukf"]["initial_state"]
    ref = cfg["rescaling"]["reference_simulation"]
    seed = ini["covariance_seed"] if seed is None else seed
    fs = ref["sim_fs_hz"]
    n_burn = int(round(ref["burn_in_s"] * fs))
    n_keep = int(round(ini["covariance_sim_duration_s"] * fs))
    res = model.simulate(cfg, n_burn + n_keep, seed=seed, n_nodes=N_NODES,
                         p=cfg["jansen_rit"]["p_mean"], g12=0.0, g21=0.0)
    kept = res.states[1 + n_burn:]
    if kept.shape[0] != n_keep:
        raise StateSpaceError(f"kept {kept.shape[0]} samples, expected {n_keep}")
    ddof = ref["ddof"]
    per_node = np.stack([kept[:, j, :].var(axis=0, ddof=ddof) for j in range(N_NODES)])
    return {"variance": per_node.mean(axis=0), "per_node": per_node, "seed": seed,
            "n_samples": int(n_keep)}


def provenance_record(cfg, result, git_hash, dirty):
    ini = cfg["ukf"]["initial_state"]
    ref = cfg["rescaling"]["reference_simulation"]
    return {
        "seed": result["seed"], "kept_duration_s": ini["covariance_sim_duration_s"],
        "burn_in_s": ref["burn_in_s"], "sim_fs_hz": ref["sim_fs_hz"], "ddof": ref["ddof"],
        "n_samples": result["n_samples"], "nodes": "two nodes at g=0, pooled by state index",
        "units": "mV^2 for y0-y2, mV^2/s^2 for y3-y5 (model output units)",
        "git_commit": git_hash, "git_dirty": dirty,
        "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"),
        "numba": metadata.version("numba"),
    }


_CONFIG_LEAVES = ("neural_variance", "neural_variance_provenance")


def write_initial_covariance(config_path, variance, provenance, force=False):
    """Replace only the `value:` of ukf.initial_state.{neural_variance, ..._provenance}.
    Targeted text replacement, every other byte preserved, yaml.dump never used."""
    path = Path(config_path)
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    new_values = {"neural_variance": "[" + ", ".join(model.yaml_float(v) for v in variance) + "]",
                  "neural_variance_provenance": json.dumps(provenance)}
    hits = {}
    for i, line in enumerate(lines):
        m = re.match(r"^    (neural_variance|neural_variance_provenance): \{value: (.*?)(?=, prov:)", line)
        if m:
            if m.group(1) in hits:
                raise StateSpaceError(f"leaf {m.group(1)} appears twice in {path}")
            hits[m.group(1)] = (i, m)
    missing = [k for k in _CONFIG_LEAVES if k not in hits]
    if missing:
        raise StateSpaceError(f"leaves not found in {path}: {missing}")
    filled = [k for k in _CONFIG_LEAVES if hits[k][1].group(2) != "null"]
    if filled and not force:
        raise StateSpaceError(f"refusing to overwrite non-null {filled}; pass --force, and then "
                              "the overwrite must be logged in DEVIATIONS.md section 1")
    if filled:
        log.warning("OVERWRITING %s: log this in DEVIATIONS.md section 1", filled)
    for key, (i, m) in hits.items():
        lines[i] = lines[i][:m.start(2)] + new_values[key] + lines[i][m.end(2):]
    path.write_bytes("".join(lines).encode("utf-8"))


# ---- command line -----------------------------------------------------------------------

def _fmt(x):
    return f"{x:.12g}"


def _report(cfg, result, head, dirty):
    ini = cfg["ukf"]["initial_state"]
    steady = initial_node_state(cfg)
    lines = [
        f"git HEAD {head}, tree dirty: {dirty}",
        f"seed {result['seed']}: two nodes, g=0, p={cfg['jansen_rit']['p_mean']}, "
        f"{cfg['rescaling']['reference_simulation']['burn_in_s']} s burn-in discarded, "
        f"{ini['covariance_sim_duration_s']} s kept at "
        f"{cfg['rescaling']['reference_simulation']['sim_fs_hz']} Hz ({result['n_samples']} samples)",
        f"steady state y1 - y2 at the prior means: {_fmt(steady[Y1] - steady[Y2])} mV",
        f"{'state':>6} {'pooled variance':>20} {'node 1':>20} {'node 2':>20}",
    ]
    for k, name in enumerate(model.STATE_NAMES):
        lines.append(f"{name:>6} {_fmt(result['variance'][k]):>20} "
                     f"{_fmt(result['per_node'][0][k]):>20} {_fmt(result['per_node'][1][k]):>20}")
    return "\n".join(lines)


def main(argv=None, config_path=None):
    parser = argparse.ArgumentParser(prog="python -m src.state_space")
    parser.add_argument("--compute-initial-covariance", action="store_true", required=True)
    parser.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite non-null values")
    parser.add_argument("--expect-commit", help="required for a real write: the HEAD hash the "
                        "dry run was made at")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.expect_commit:
        parser.error("a real write needs --expect-commit <hash of the dry run>")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    config_path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    cfg = load_config(config_path)
    try:
        if not args.dry_run:
            model.check_git_state(args.expect_commit)
            ini = cfg["ukf"]["initial_state"]
            filled = [k for k in _CONFIG_LEAVES if ini[k] is not None]
            if filled and not args.force:
                raise StateSpaceError(f"refusing to overwrite non-null {filled}; pass --force")
        result = neural_variance_from_simulation(cfg)
        head, dirty = model.git_state()
        log.info(_report(cfg, result, head, dirty))
        if args.dry_run:
            log.info("dry run: nothing written")
            return 0
        model.check_git_state(args.expect_commit)
        write_initial_covariance(config_path, result["variance"],
                                 provenance_record(cfg, result, head, dirty), force=args.force)
        log.info("written to %s", config_path)
        return 0
    except (StateSpaceError, model.ModelError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
