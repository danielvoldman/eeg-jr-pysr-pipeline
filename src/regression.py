"""Regression machinery for the residual on the coupling term (PLAN.md D1; §8.2, §8.3, §8.4, §10.2).

D1 has no Julia and no PySR: this module builds the residual target and its rows, the two derivative
estimators, the subject folds, the equal-per-subject subsample, the z-scoring constants, the Pareto
selection rule (on any list of front entries), the term signature and the ensemble bookkeeping.
PySR itself is imported lazily, never here, and never before main.py has set PYTHON_JULIACALL_THREADS
(D2).

Inputs are the plain outputs of pass 2 (IMP-032): per 2-s window the smoothed neural states
(n, >= 12; only the first 12 columns are read, so a filter layout with extra states does not matter),
the delayed source S (n, 2) and the recording-level parameters (p1, p2, A1, A2, g12, g21). Only
passes.base_dy4dt and model.constants are used from the filter side; no UKF layout is touched.

Residual (§8.3): for target node j, r_j(t) = dy4_j/dt estimated from the smoothed y4 minus the base
model's dy4_j/dt at the recording-level gain. The weak-form estimator is the Gaussian-window
test-function derivative; 'matched' smooths the base prediction with the same test function,
'literal' subtracts it pointwise (IMP-053). TV uses the literal form. Rows exist only where the whole
kernel lies inside the kept samples of a window (no padding, nothing crosses a window edge).
"""
import faulthandler
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import numpy as np
import sympy

from src import model
from src import passes
from src.config import REPO_ROOT

log = logging.getLogger(__name__)

N_INPUTS = 3                          # u_tgt, u_src, S_src (config residual.input_names)


class RegressionError(ValueError):
    """Raised for an inconsistent regression request."""


# ---- small exact helpers ---------------------------------------------------------------------------

def _frac(x):
    """Exact Fraction of a config number as written (0.8 is 4/5, not its binary value)."""
    return Fraction(str(x))


def _round_half_up(x):
    return int(math.floor(x + Fraction(1, 2)))


def _fs(cfg):
    return cfg["preprocessing"]["observation_fs_hz"]


def delay_samples(cfg):
    """Coupling delay in observation samples: delay_substeps / substeps_per_observation (2.5)."""
    return _frac(cfg["coupling"]["delay_substeps"]) / _frac(cfg["ukf"]["substeps_per_observation"])


# ---- derivative estimators (§8.3) ------------------------------------------------------------------

@dataclass(frozen=True)
class WeakKernel:
    K: int                 # half-width in samples; the kernel has 2K + 1 taps
    sigma_s: float
    deriv: np.ndarray      # (2K+1,) weights: out_i = sum_k deriv[k] y[i + k - K]  (units: per second)
    smooth: np.ndarray     # (2K+1,) weights of phi, normalised to sum 1 (for the matched form)


def weak_kernel(support_ms, cfg):
    """Gaussian-window test function with compact support (§8.3, IMP-053).

    h = support / 2, sigma = h / weak_sigma_divisor, K = floor(h fs) taps each side. The test function is
    phi = g - g(h), which vanishes at the support edge, so dy/dt = -int phi'(tau - t) y(tau) dtau / int phi
    has no boundary term. Since phi' is proportional to k g_k, the discrete derivative is
    (fs) sum_k k g_k y[i + k] / sum_k k^2 g_k, exact for linear signals (the Gaussian-weighted least
    squares slope). The matched smoothing weights are phi_k / sum phi."""
    fs = _fs(cfg)
    h = _frac(support_ms) / 2000
    K = int(math.floor(h * _frac(fs)))
    if K < 1:
        raise RegressionError(f"support {support_ms} ms holds no whole sample at {fs} Hz")
    div = cfg["residual"]["weak_sigma_divisor"]
    sigma = float(h) / div
    k = np.arange(-K, K + 1, dtype=np.float64)
    g = np.exp(-((k / fs) ** 2) / (2.0 * sigma ** 2))
    g_edge = math.exp(-0.5 * div ** 2)                    # g at |s| = h
    phi = g - g_edge
    deriv = fs * k * g / np.sum(k * k * g)
    return WeakKernel(K=K, sigma_s=sigma, deriv=deriv, smooth=phi / np.sum(phi))


def _valid_apply(y, w):
    """out[i] = sum_k w[k] y[i + k] for the n - len(w) + 1 positions where the kernel fits (axis 0)."""
    y = np.asarray(y, dtype=np.float64)
    if y.shape[0] < len(w):
        raise RegressionError(f"{y.shape[0]} samples are fewer than the {len(w)} kernel taps")
    win = np.lib.stride_tricks.sliding_window_view(y, len(w), axis=0)
    return win @ w


def weak_derivative(y, kernel):
    """Weak-form dy/dt at the n - 2K interior samples i = K .. n - K - 1 (no padding)."""
    return _valid_apply(y, kernel.deriv)


def weak_smooth(f, kernel):
    """phi-weighted mean of f at the same interior samples (the matched partner of the derivative)."""
    return _valid_apply(f, kernel.smooth)


@lru_cache(maxsize=8)
def _tv_operators(n):
    """A (trapezoid cumulative integral, sample units) and D (first difference)."""
    A = np.tril(np.ones((n, n)), -1)
    A[:, 0] = 0.5
    A[np.arange(n), np.arange(n)] = 0.5
    A[0, :] = 0.0
    D = np.diff(np.eye(n), axis=0)
    return A, D


def tv_derivative(y, fs_hz, alpha_rel, cfg):
    """TV-regularised derivative of one window (§8.3 comparison baseline, IMP-054).

    y' = (y - y[0]) / SD(y) (dimensionless), u = dy'/d(sample). Minimise
        J(u) = (1/2n) |A u - y'|^2 + alpha_rel/(n-1) sum_i sqrt((D u)_i^2 + eps^2)
    by lagged diffusivity (each iteration solves the linear system with the TV weights frozen), from the
    central-difference slope. Returns (dy/dt in y units per second at all n samples, info)."""
    rc = cfg["residual"]
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size < 3:
        raise RegressionError("tv_derivative needs a 1-D series of at least 3 samples")
    n = y.size
    s = float(np.std(y))
    if s == 0.0:
        return np.zeros(n), {"iterations": 0, "converged": True, "objective": 0.0}
    yp = (y - y[0]) / s
    A, D = _tv_operators(n)
    eps = rc["tv_eps"]
    c1, c2 = 1.0 / n, alpha_rel / (n - 1)
    AtA, rhs = c1 * (A.T @ A), c1 * (A.T @ yp)
    u = np.gradient(yp)
    converged, it = False, 0
    for it in range(1, rc["tv_max_iter"] + 1):
        du = D @ u
        w = 1.0 / np.sqrt(du * du + eps * eps)
        u_new = np.linalg.solve(AtA + c2 * (D.T * w) @ D, rhs)
        rel = np.linalg.norm(u_new - u) / max(np.linalg.norm(u), np.finfo(np.float64).tiny)
        u = u_new
        if rel < rc["tv_tol"]:
            converged = True
            break
    du = D @ u
    obj = 0.5 * c1 * float(np.sum((A @ u - yp) ** 2)) + c2 * float(np.sum(np.sqrt(du * du + eps * eps)))
    if not converged:
        log.warning("tv_derivative: no convergence in %d iterations (alpha_rel %g)", it, alpha_rel)
    return u * fs_hz * s, {"iterations": it, "converged": converged, "objective": obj}


def margin_samples(cfg, support_ms=None):
    """Rows trimmed at each end of a window: the half-width K of the (primary) weak kernel."""
    sup = cfg["residual"]["weak_support_ms"] if support_ms is None else support_ms
    return weak_kernel(sup, cfg).K


# ---- rows (§8.3, §8.4) -------------------------------------------------------------------------------

INPUT_NAMES = ("u_tgt", "u_src", "S_src")
PRODUCT_PAIRS = ((0, 1), (0, 2), (1, 2))
PRODUCT_NAMES = tuple(f"{INPUT_NAMES[a]}_{INPUT_NAMES[b]}" for a, b in PRODUCT_PAIRS)
VARIABLE_NAMES = INPUT_NAMES + PRODUCT_NAMES


@dataclass
class Rows:
    """Pooled rows of both directions (§8.4). X_raw columns: u_tgt, u_src, S_src (unscaled)."""
    X_raw: np.ndarray
    y: np.ndarray
    node: np.ndarray        # target node index (0 = node 1 is driven, 1 = node 2 is driven)
    window: np.ndarray      # start sample of the window (time order key)
    t: np.ndarray           # sample index inside the window
    subject: np.ndarray     # str

    def __len__(self):
        return int(self.y.shape[0])

    def take(self, idx):
        return Rows(self.X_raw[idx], self.y[idx], self.node[idx], self.window[idx], self.t[idx],
                    self.subject[idx])

    @staticmethod
    def concat(parts):
        parts = [p for p in parts if len(p)]
        if not parts:
            return Rows(np.empty((0, N_INPUTS)), np.empty(0), np.empty(0, int), np.empty(0, np.int64),
                        np.empty(0, int), np.empty(0, "U1"))
        return Rows(np.concatenate([p.X_raw for p in parts]), np.concatenate([p.y for p in parts]),
                    np.concatenate([p.node for p in parts]), np.concatenate([p.window for p in parts]),
                    np.concatenate([p.t for p in parts]), np.concatenate([p.subject for p in parts]))


def pyramidal_potential(x_smooth, node):
    """y1 - y2 of one node from the smoothed neural states (12 columns, six per node)."""
    x = np.asarray(x_smooth, dtype=np.float64)
    return x[:, node * model.N_STATES + model.Y1] - x[:, node * model.N_STATES + model.Y2]


def lagged(v, idx, d):
    """v at (idx - d) samples for a possibly fractional d, by linear interpolation (d = 2.5: the mean
    of the samples 2 and 3 back)."""
    lo = int(math.floor(d))
    frac = float(d - lo)
    if frac == 0.0:
        return v[idx - lo]
    return (1.0 - frac) * v[idx - lo] + frac * v[idx - lo - 1]


def window_rows(x_smooth, s_delayed, params, cfg, *, estimator="weak_form", support_ms=None,
                form=None, tv_alpha_rel=None, margin=None, window_start=0, subject=""):
    """Rows of one 2-s window, both nodes pooled (§8.3, §8.4, IMP-053, IMP-055).

    estimator 'weak_form' (form 'matched' or 'literal'; default residual.subtraction_form) or
    'total_variation' (always literal; needs tv_alpha_rel). `margin` >= the kernel half-width trims more
    rows at each end so that estimators can be compared on a common row set."""
    x = np.asarray(x_smooth, dtype=np.float64)
    s = np.asarray(s_delayed, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 * model.N_STATES:
        raise RegressionError(f"x_smooth must have at least {2 * model.N_STATES} columns, got {x.shape}")
    if s.shape != (x.shape[0], 2):
        raise RegressionError(f"s_delayed must have shape ({x.shape[0]}, 2), got {s.shape}")
    n = x.shape[0]
    rc = cfg["residual"]
    form = rc["subtraction_form"] if form is None else form
    d = delay_samples(cfg)
    base = passes.base_dy4dt(x[:, :2 * model.N_STATES], s, params, cfg)
    if estimator == "weak_form":
        if form not in ("matched", "literal"):
            raise RegressionError(f"unknown subtraction form {form!r}")
        kern = weak_kernel(rc["weak_support_ms"] if support_ms is None else support_ms, cfg)
        K = kern.K
    elif estimator == "total_variation":
        if tv_alpha_rel is None:
            raise RegressionError("total_variation needs tv_alpha_rel")
        kern, K = None, margin_samples(cfg)
    else:
        raise RegressionError(f"unknown estimator {estimator!r}")
    m = K if margin is None else int(margin)
    if m < K:
        raise RegressionError(f"margin {m} is smaller than the kernel half-width {K}")
    if m < math.ceil(d):
        raise RegressionError(f"margin {m} is smaller than the delay of {float(d)} samples")
    if n <= 2 * m:
        raise RegressionError(f"{n} samples leave no row with margin {m}")
    idx = np.arange(m, n - m)
    parts = []
    for j in (0, 1):
        src = 1 - j
        y4 = x[:, j * model.N_STATES + model.Y4]
        if estimator == "weak_form":
            dy = weak_derivative(y4, kern)[m - K:n - m - K]
            if form == "matched":
                r = dy - weak_smooth(base[:, j], kern)[m - K:n - m - K]
            else:
                r = dy - base[idx, j]
        else:
            dy, _ = tv_derivative(y4, _fs(cfg), tv_alpha_rel, cfg)
            r = dy[idx] - base[idx, j]
        X = np.column_stack([pyramidal_potential(x, j)[idx],
                             lagged(pyramidal_potential(x, src), idx, d),
                             s[idx, src]])
        parts.append(Rows(X_raw=X, y=r, node=np.full(idx.size, j, dtype=int),
                          window=np.full(idx.size, int(window_start), dtype=np.int64), t=idx.copy(),
                          subject=np.full(idx.size, str(subject))))
    return Rows.concat(parts)


def recording_rows(windows, params, cfg, subject, **kwargs):
    """Rows of every kept window of one recording (objects with x_smooth, s_delayed, start, diverged)."""
    parts = [window_rows(w.x_smooth, w.s_delayed, params, cfg, window_start=w.start, subject=subject,
                         **kwargs) for w in windows if not w.diverged]
    return Rows.concat(parts)


# ---- z-scoring (§8.2) -------------------------------------------------------------------------------

@dataclass
class ZScore:
    mean_X: np.ndarray
    sd_X: np.ndarray
    mean_y: float
    sd_y: float
    n: int

    def to_dict(self):
        return {"mean_X": self.mean_X.tolist(), "sd_X": self.sd_X.tolist(), "mean_y": float(self.mean_y),
                "sd_y": float(self.sd_y), "n": int(self.n), "inputs": list(INPUT_NAMES)}


def fit_zscore(chunks):
    """Mean and (population) SD of the three inputs and the target over every row of the FIT fold, merged
    chunk by chunk (Chan's formula), so the rows need not be held at once. Constants are stored with the
    frozen equation and applied unchanged to validation and test (§8.2)."""
    n, mean, m2 = 0, np.zeros(N_INPUTS + 1), np.zeros(N_INPUTS + 1)
    for rows in chunks:
        if not len(rows):
            continue
        data = np.column_stack([rows.X_raw, rows.y])
        nb, mb = len(rows), data.mean(axis=0)
        m2b = ((data - mb) ** 2).sum(axis=0)
        delta = mb - mean
        tot = n + nb
        mean = mean + delta * nb / tot
        m2 = m2 + m2b + delta ** 2 * n * nb / tot
        n = tot
    if n < 2:
        raise RegressionError("z-scoring needs at least two rows")
    sd = np.sqrt(m2 / n)
    if np.any(sd == 0.0):
        raise RegressionError("a z-scored column is constant")
    return ZScore(mean_X=mean[:N_INPUTS], sd_X=sd[:N_INPUTS], mean_y=float(mean[-1]), sd_y=float(sd[-1]), n=n)


def design(rows, z):
    """(X, y): the z-scored inputs plus their three pairwise products (formed from the z-scored columns,
    so they carry no constants of their own), and the z-scored target."""
    Xz = (rows.X_raw - z.mean_X) / z.sd_X
    prods = [Xz[:, a] * Xz[:, b] for a, b in PRODUCT_PAIRS]
    return np.column_stack([Xz] + prods), (rows.y - z.mean_y) / z.sd_y


# ---- seeds, folds, subsample (§8.2, §10.2; IMP-056) -----------------------------------------------------

def seeds_for(cfg, split_seed, k=0):
    """Fold, subsample and PySR seeds: offset + split seed + stride x k (k = 0 is the primary fit)."""
    pc = cfg["pysr"]
    add = int(split_seed) + pc["seeds"]["refit_stride"] * int(k)
    return {"fold": pc["seeds"]["fold_offset"] + add, "subsample": pc["subsample"]["seed"] + add,
            "random_state": pc["seeds"]["random_state_offset"] + add}


def check_training_ids(ids, split):
    """Refuse any subject that is not on the training side of the split file (CLAUDE.md rule 6)."""
    train, test = set(split["train"]), set(split["test"])
    bad = sorted(i for i in ids if i in test or i not in train)
    if bad:
        raise RegressionError(f"not training subjects (test or unknown): {bad[:5]}{'...' if len(bad) > 5 else ''}")


def split_subjects_fit_val(train_ids, split, seed, cfg):
    """80/20 subject partition of training subjects (§8.2): sorted IDs, rng = default_rng(seed),
    perm = rng.permutation(n); the first round_half_up(0.8 n) are the fit fold."""
    ids = sorted(train_ids)
    check_training_ids(ids, split)
    if len(ids) < 2:
        raise RegressionError("need at least two subjects for a fit/validation split")
    n_fit = _round_half_up(_frac(cfg["pysr"]["selection"]["fit_fraction"]) * len(ids))
    n_fit = min(max(n_fit, 1), len(ids) - 1)
    perm = np.random.default_rng(int(seed)).permutation(len(ids))
    return sorted(ids[i] for i in perm[:n_fit]), sorted(ids[i] for i in perm[n_fit:])


def split_windows_time(window_starts, cfg):
    """G0 folds: the last 20% of the recording's windows in time order validate (§8.2);
    n_val = round_half_up(0.2 n)."""
    starts = sorted(set(int(w) for w in window_starts))
    if len(starts) < 2:
        raise RegressionError("need at least two windows for a time split")
    n_val = _round_half_up(_frac(cfg["pysr"]["selection"]["validation_fraction"]) * len(starts))
    n_val = min(max(n_val, 1), len(starts) - 1)
    return starts[:-n_val], starts[-n_val:]


def draw_ensemble(train_ids, split, split_seed, cfg, n_refits=None):
    """The C2 refit ensemble's subject draws (§8.2): refit k = 1..n takes a random half of the training
    subjects (rng = default_rng(fold seed of k): permutation, first round_half_up(0.5 n)), then the 80/20
    fit/validation split of that half from the same generator. Pilot: n_refits = 3."""
    ids = sorted(train_ids)
    check_training_ids(ids, split)
    ec = cfg["pysr"]["ensemble"]
    n_refits = ec["n_refits"] if n_refits is None else n_refits
    n_half = _round_half_up(_frac(ec["subject_fraction"]) * len(ids))
    out = []
    for k in range(1, n_refits + 1):
        sd = seeds_for(cfg, split_seed, k)
        rng = np.random.default_rng(sd["fold"])
        half = sorted(ids[i] for i in rng.permutation(len(ids))[:n_half])
        n_fit = _round_half_up(_frac(cfg["pysr"]["selection"]["fit_fraction"]) * len(half))
        n_fit = min(max(n_fit, 1), len(half) - 1)
        perm = rng.permutation(len(half))
        out.append({"k": k, "seeds": sd, "subjects": half,
                    "fit": sorted(half[i] for i in perm[:n_fit]), "val": sorted(half[i] for i in perm[n_fit:])})
    return out


def subsample_quotas(available, total, rng):
    """Rows per subject: equal, or as equal as integers allow (counts differ by at most 1). A subject with
    fewer rows than its quota gives all of them and the deficit is shared by the others; the remainder of
    an uneven division goes to subjects picked by `rng`. If the subjects hold fewer than `total` rows,
    all are taken."""
    ids = sorted(available)
    if not ids:
        raise RegressionError("no subjects to subsample")
    quota = {}
    active, remaining = list(ids), int(total)
    while active:
        base, extra = divmod(remaining, len(active))
        short = [s for s in active if available[s] <= base]
        if not short:
            picked = set(int(i) for i in rng.permutation(len(active))[:extra])
            for i, s in enumerate(active):
                quota[s] = base + (1 if i in picked else 0)
            break
        for s in short:
            quota[s] = int(available[s])
            remaining -= int(available[s])
            active.remove(s)
    return quota


def subsample_rows(rows_by_subject, total, seed, stream, cfg=None):
    """Fixed-size subsample with equal rows per subject (§8.2): rng = default_rng([seed, stream]), stream
    0 for the fit fold and 1 for the validation fold; quotas as in subsample_quotas; each subject's rows
    drawn without replacement and kept in their original order."""
    rng = np.random.default_rng([int(seed), int(stream)])
    quota = subsample_quotas({s: len(r) for s, r in rows_by_subject.items()}, total, rng)
    parts = []
    for s in sorted(rows_by_subject):
        r = rows_by_subject[s]
        pick = np.sort(rng.choice(len(r), size=quota[s], replace=False)) if quota[s] else np.empty(0, int)
        parts.append(r.take(pick))
    out = Rows.concat(parts)
    if len(out) < total:
        log.warning("subsample: %d rows available, %d requested", len(out), total)
    return out


# ---- recovery measure (§9.1) --------------------------------------------------------------------------

def recovery_nrmse(r_hat, basis, truth):
    """NRMSE of the recovered residual against the planted one, both on the true input trajectories
    (§9.1): the recovered function is c_hat x basis, c_hat from OLS of r_hat on [1, basis] (the intercept
    is a bare constant, i.e. no term), and RMS(c_hat basis - truth) / SD(truth)."""
    r_hat, basis, truth = (np.asarray(v, dtype=np.float64) for v in (r_hat, basis, truth))
    Z = np.column_stack([np.ones_like(basis), basis])
    coef, *_ = np.linalg.lstsq(Z, r_hat, rcond=None)
    rec = coef[1] * basis
    return float(np.sqrt(np.mean((rec - truth) ** 2)) / np.std(truth)), float(coef[1])


# ---- Pareto selection (§8.2) --------------------------------------------------------------------------

@dataclass
class FrontEntry:
    complexity: int
    equation: str
    loss: float                       # fit-fold loss (PySR)
    val_loss: object = None           # validation loss (None until scored)


@dataclass
class Selection:
    entry: object                     # FrontEntry or None
    no_term: bool
    reason: str
    min_val_loss: object = None
    eligible: list = field(default_factory=list)   # complexities within the tolerance


def pareto_filter(entries):
    """Keep an entry only if its fit loss is strictly below that of every simpler entry."""
    front, best = [], math.inf
    for e in sorted(entries, key=lambda e: e.complexity):
        if e.loss < best:
            front.append(e)
            best = e.loss
    return front


def parse_equation(text):
    """sympy expression of a PySR equation string; the candidate-input names are plain symbols."""
    return sympy.sympify(text, locals={n: sympy.Symbol(n) for n in VARIABLE_NAMES})


def is_bare_constant(text):
    return not parse_equation(text).free_symbols


def select_equation(entries, cfg):
    """Simplest Pareto entry whose validation loss is within loss_tolerance of the minimum (§8.2).

    Non-finite validation losses are never the minimum and never eligible. The comparison is exact
    arithmetic on the float values: eligible if val_loss <= min_val (1 + tolerance), boundary included.
    A bare constant selected means 'no residual term'. Ties in complexity go to the lower validation loss."""
    sel = cfg["pysr"]["selection"]
    front = pareto_filter(entries)
    scored = [e for e in front if e.val_loss is not None and math.isfinite(e.val_loss)]
    if not scored:
        return Selection(None, True, "no entry has a finite validation loss")
    lo = Fraction(min(e.val_loss for e in scored))
    bound = lo * (1 + _frac(sel["loss_tolerance"]))
    eligible = [e for e in scored if Fraction(e.val_loss) <= bound]
    best = min(eligible, key=lambda e: (e.complexity, e.val_loss))
    if sel["bare_constant_counts_as_no_term"] and is_bare_constant(best.equation):
        return Selection(best, True, "selected equation is a bare constant", float(lo),
                         [e.complexity for e in eligible])
    return Selection(best, False, "selected", float(lo), [e.complexity for e in eligible])


# ---- term signature (§8.4) ---------------------------------------------------------------------------

def _product_substitution():
    sym = {n: sympy.Symbol(n) for n in VARIABLE_NAMES}
    return {sym[PRODUCT_NAMES[i]]: sym[INPUT_NAMES[a]] * sym[INPUT_NAMES[b]]
            for i, (a, b) in enumerate(PRODUCT_PAIRS)}


def _is_denominator(e):
    return e.is_Pow and e.exp.is_number and e.exp.is_negative


def _norm_mul(e):
    """Multiplicative constants -> 1 (sign included); denominators kept literally; recurse into
    function arguments and sums."""
    if e.is_Number:
        return sympy.S.One
    if e.is_Mul:
        return sympy.Mul(*[_norm_mul(f) for f in e.args])
    if _is_denominator(e):
        return e
    if e.is_Pow:
        return sympy.Pow(_norm_mul(e.base), e.exp)
    if e.is_Add:
        return sympy.Add(*[_norm_mul(t) for t in e.args])
    if isinstance(e, sympy.Function):
        return e.func(*[_norm_arg(a) for a in e.args])
    return e


def _norm_arg(a):
    """A function argument: additive numeric constants -> 0, then multiplicative constants -> 1."""
    if a.is_Add:
        return sympy.Add(*[_norm_mul(t) for t in a.args if not t.is_Number])
    return _norm_mul(a)


def term_signatures(text):
    """Signatures (§8.4, IMP-058) of the additive components of a selected equation: product columns are
    replaced by their factors, the expression is expanded, each component has multiplicative numeric
    constants set to 1 and additive constants inside function arguments set to 0 (constants inside
    denominators are kept literally), then simplified. A component without symbols has no signature."""
    expr = parse_equation(text).xreplace(_product_substitution())
    out = set()
    for comp in sympy.Add.make_args(sympy.expand(expr)):
        sig = sympy.simplify(_norm_mul(comp))
        if sig.free_symbols:
            out.add(str(sig))
    return frozenset(out)


def signature_recurrence(signature_sets, cfg):
    """Count of refits in which each signature appears, and the signatures present in at least
    statistics.criteria.c2_signature_recurrence_min of them (exact: count >= ceil(0.7 n))."""
    n = len(signature_sets)
    need = math.ceil(_frac(cfg["statistics"]["criteria"]["c2_signature_recurrence_min"]) * n)
    counts = {}
    for s in signature_sets:
        for sig in s:
            counts[sig] = counts.get(sig, 0) + 1
    return {"n_refits": n, "needed": need, "counts": counts,
            "stable": sorted(sig for sig, c in counts.items() if c >= need)}


# ---- PySR wrapper (PLAN.md D3, D5; §8.2, §16.5; IMP-059, IMP-060) -------------------------------------------

ROLE_TIMEOUT = {"primary": "timeout_primary_s", "refit": "timeout_refit_s",
                "synthetic": "timeout_synthetic_s", "pilot_primary": "timeout_pilot_primary_s"}


def load_pysr(cfg):
    """Import PySR once per process and check the pins (§8.2: one warm Julia session for every fit).

    PYTHON_JULIACALL_THREADS must already hold compute.julia_threads: Julia's thread count is fixed when
    juliacall is first imported (main.py sets it from config before anything else). JULIA_DEPOT_PATH is
    pointed at pysr.julia_depot_dir (inside .venv) unless the caller set it. Raises RegressionError if the
    thread count, PySR version or Julia version is not the pinned one."""
    want = str(cfg["compute"]["julia_threads"])
    if "juliacall" not in sys.modules:
        if os.environ.get("PYTHON_JULIACALL_THREADS") != want:
            raise RegressionError(f"PYTHON_JULIACALL_THREADS must be {want} before PySR is imported "
                                  f"(main.py sets it); it is {os.environ.get('PYTHON_JULIACALL_THREADS')!r}")
        os.environ.setdefault("JULIA_DEPOT_PATH", str(REPO_ROOT / cfg["pysr"]["julia_depot_dir"]))
    if faulthandler.is_enabled():
        # Julia's GC safepoints are page faults that Windows reports as access violations; Python's
        # faulthandler (pytest turns it on) prints each as a 'fatal exception' although Julia handles it
        # and the run continues (IMP-059).
        faulthandler.disable()
        log.info("faulthandler disabled before PySR is imported (Julia safepoints raise access violations)")
    import pysr
    from juliacall import Main as jl
    pins = cfg["pysr"]["version"]
    n_threads = int(jl.seval("Threads.nthreads()"))
    if n_threads != int(want):
        raise RegressionError(f"Julia runs {n_threads} threads, expected {want}")
    if pysr.__version__ != pins["pysr"]:
        raise RegressionError(f"PySR {pysr.__version__} is installed, config pins {pins['pysr']}")
    julia_version = str(jl.seval("string(VERSION)"))
    if julia_version != pins["julia"]:
        raise RegressionError(f"Julia {julia_version} runs, config pins {pins['julia']}")
    return pysr


def model_kwargs(cfg, role, *, random_state, serial=False, turbo=None, niterations=None, parsimony=None,
                 timeout_s=None):
    """PySRRegressor keyword arguments from config (§8.2). Populations, ncycles_per_iteration and fast_cycle
    are not passed (PySR's defaults, §16.5.3); batching is passed as False because PySR 2 defaults it to
    'auto'; precision is 64. serial=True is the determinism-check mode: one thread, deterministic, a fixed
    niterations and no timeout. PySR 2 takes variable_names at fit(), not here (fit_model passes them)."""
    pc = cfg["pysr"]
    kw = dict(binary_operators=list(pc["binary_operators"]), unary_operators=list(pc["unary_operators"]),
              maxsize=pc["maxsize"], parsimony=pc["parsimony_start"] if parsimony is None else parsimony,
              batching=pc["batching"], precision=pc["precision"],
              turbo=pc["turbo"] if turbo is None else turbo, random_state=int(random_state),
              verbosity=pc["verbosity"], progress=False,
              temp_equation_file=True, delete_tempfiles=True)
    if serial:
        if niterations is None:
            raise RegressionError("serial mode needs a fixed niterations")
        kw.update(parallelism="serial", deterministic=True, niterations=int(niterations),
                  timeout_in_seconds=None)
    else:
        kw["parallelism"] = pc["parallelism"]
        kw["niterations"] = pc["niterations_timeout_bound"] if niterations is None else int(niterations)
        kw["timeout_in_seconds"] = float(pc[ROLE_TIMEOUT[role]] if timeout_s is None else timeout_s)
        if pc["parallelism"] == "multiprocessing":
            kw["procs"] = pc["procs"]
    return kw


def evaluate_equation(text, X, names=VARIABLE_NAMES):
    """Value of a PySR equation string on the columns of X (float64, sympy lambdify; independent of PySR's
    own predict). Overflow and invalid operations give inf / nan, which the selection treats as a failed
    loss."""
    expr = parse_equation(text)
    fn = sympy.lambdify([sympy.Symbol(n) for n in names], expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        out = fn(*[X[:, i] for i in range(len(names))])
    return np.broadcast_to(np.asarray(out, dtype=np.float64), (X.shape[0],)).copy()


def front_entries(equations, X_val, y_val, names=VARIABLE_NAMES):
    """FrontEntry list from PySR's equations_ table (rows with complexity, loss, equation), each scored on
    the validation rows by mean squared error of the z-scored target (non-finite gives inf)."""
    out = []
    for row in equations.itertuples():
        pred = evaluate_equation(row.equation, X_val, names)
        loss = float(np.mean((pred - y_val) ** 2))
        out.append(FrontEntry(int(row.complexity), str(row.equation), float(row.loss),
                              loss if math.isfinite(loss) else math.inf))
    return out


def fit_model(cfg, X, y, role, *, random_state, names=VARIABLE_NAMES, **kw):
    """One PySR fit (a fresh PySRRegressor each time, the Julia session is warm). Returns (model, seconds)."""
    pysr = load_pysr(cfg)
    model_ = pysr.PySRRegressor(**model_kwargs(cfg, role, random_state=random_state, **kw))
    t0 = time.time()
    model_.fit(np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64),
               variable_names=list(names))
    return model_, time.time() - t0


@dataclass
class FitResult:
    selection: object
    entries: list
    zscore: object
    seeds: dict
    n_fit_rows: int
    n_val_rows: int
    seconds: float
    record: dict


def run_fit(rows_by_subject, fit_ids, val_ids, split, split_seed, cfg, *, role, k=0, **kw):
    """Fit on the fit fold's subsample, score the Pareto front on the validation subsample, select and
    freeze (§8.2). Only training subjects are accepted (check_training_ids); no refit on 100%."""
    check_training_ids(list(fit_ids) + list(val_ids), split)
    sc = cfg["pysr"]["subsample"]
    seeds = seeds_for(cfg, split_seed, k)
    fit_rows = subsample_rows({s: rows_by_subject[s] for s in fit_ids}, sc["rows_per_fit"], seeds["subsample"], 0)
    val_rows = subsample_rows({s: rows_by_subject[s] for s in val_ids}, sc["rows_validation"],
                              seeds["subsample"], 1)
    z = fit_zscore([rows_by_subject[s] for s in fit_ids])
    Xf, yf = design(fit_rows, z)
    Xv, yv = design(val_rows, z)
    model_, secs = fit_model(cfg, Xf, yf, role, random_state=seeds["random_state"], names=VARIABLE_NAMES, **kw)
    entries = front_entries(model_.equations_, Xv, yv)
    sel = select_equation(entries, cfg)
    chosen = sel.entry is not None and not sel.no_term
    record = {"role": role, "k": k, "seeds": seeds, "versions": cfg["pysr"]["version"],
              "turbo": cfg["pysr"]["turbo"] if kw.get("turbo") is None else kw["turbo"],
              "precision": cfg["pysr"]["precision"], "fit_seconds": secs, "n_fit_rows": len(fit_rows),
              "n_val_rows": len(val_rows), "fit_subjects": sorted(fit_ids), "val_subjects": sorted(val_ids),
              "zscore": z.to_dict(), "no_term": sel.no_term, "reason": sel.reason,
              "equation": sel.entry.equation if sel.entry is not None else None,
              "complexity": sel.entry.complexity if sel.entry is not None else None,
              "val_loss": sel.entry.val_loss if sel.entry is not None else None,
              "min_val_loss": sel.min_val_loss,
              "signatures": sorted(term_signatures(sel.entry.equation)) if chosen else [],
              "front": [{"complexity": e.complexity, "equation": e.equation, "loss": e.loss,
                         "val_loss": e.val_loss} for e in entries]}
    return FitResult(sel, entries, z, seeds, len(fit_rows), len(val_rows), secs, record)


def run_ensemble(rows_by_subject, train_ids, split, split_seed, cfg, *, n_refits=None, role="refit", **kw):
    """The C2 refit ensemble (§8.2): n half-sample refits (25; the pilot passes 3), each with the same
    selection rule, then the signature recurrence. Returns {"refits": [record], "recurrence": ...}."""
    draws = draw_ensemble(train_ids, split, split_seed, cfg, n_refits=n_refits)
    records, sig_sets = [], []
    for d in draws:
        res = run_fit(rows_by_subject, d["fit"], d["val"], split, split_seed, cfg, role=role, k=d["k"], **kw)
        res.record["half_subjects"] = d["subjects"]
        records.append(res.record)
        sig_sets.append(frozenset(res.record["signatures"]))
        log.info("refit %d: %s (complexity %s, no_term %s, %.0f s)", d["k"], res.record["equation"],
                 res.record["complexity"], res.record["no_term"], res.seconds)
    return {"refits": records, "recurrence": signature_recurrence(sig_sets, cfg)}


# ---- determinism and turbo comparison (§8.2, §16.5.2) -------------------------------------------------------

def _round_constants(text, sig_figs):
    expr = parse_equation(text)
    rep = {f: sympy.Float(float(f"{float(f):.{sig_figs}g}")) for f in expr.atoms(sympy.Float)}
    return str(expr.xreplace(rep))


def fronts_identical(a, b):
    """Bitwise: the same equations, complexities and fit losses (tables with complexity, loss, equation)."""
    return (len(a) == len(b)
            and all(int(x.complexity) == int(y.complexity) and str(x.equation) == str(y.equation)
                    and float(x.loss) == float(y.loss) for x, y in zip(a.itertuples(), b.itertuples())))


def fronts_equivalent(a, b, cfg):
    """The turbo pass definition (IMP-060): the same complexity sequence, the same sympy structure with
    constants rounded to constant_sig_figs, and fit losses within loss_rtol."""
    dc = cfg["pysr"]["determinism_check"]
    if len(a) != len(b):
        return False
    for x, y in zip(a.itertuples(), b.itertuples()):
        if int(x.complexity) != int(y.complexity):
            return False
        if (_round_constants(str(x.equation), dc["constant_sig_figs"])
                != _round_constants(str(y.equation), dc["constant_sig_figs"])):
            return False
        if not math.isclose(float(x.loss), float(y.loss), rel_tol=dc["loss_rtol"], abs_tol=0.0):
            return False
    return True
