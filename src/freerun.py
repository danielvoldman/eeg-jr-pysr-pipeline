"""C4 free-run simulation and spectral scoring (PLAN G5; §10.2, §14, §15.1; IMP-086).

Model side of C4, the same standing as ukf_resid.py: it simulates the fitted model forward from the filtered state at the
start of a scoring window WITHOUT observation updates and STOCHASTICALLY, and scores the simulated spectra against the
observed ones. Orchestration, the guard, the bootstrap and the verdict live in robustness.py (§18).

Simulation (one window, R independent realizations in one batch):
  * start: the filtered state after the sample before the window (passes.run_pass1 capture), the same for every
    realization: 12 neural states, the 7 parameters (frozen for the whole window), the 2 OU observation-noise states of
    filter A, the S delay ring and, for M3, the potential ring of the residual;
  * every realization owns its delay rings (the filter's single exogenous ring is shared by its sigma points, here the
    realizations diverge from each other);
  * one observation step = 4 Heun sub-steps (the filter's own propagation, bit-identical to state_space.predict for one
    realization), then process noise on the 12 neural states (variance q x steady-state variance, added once per
    observation step as in the filter), then the newest ring entries are overwritten with S and the potential of the noisy
    state (the filter does the same with the filtered mean), then the OU states advance (exact discretization);
  * output: the model observation (mixing) + the OU noise state + white noise of the filter's own R.
A realization is unstable if its neural state is not finite or leaves 10x the steady-state SD around the fixed point at the
window-start parameters (the section 7.5 divergence rule without a covariance); a window is unstable if any realization is.

Scoring: one 2-s Hann periodogram per segment on the 0.5-Hz grid over 1 to 45 Hz; the observed log10 spectrum of the segment
against the MEAN over realizations of the simulated log10 spectrum at the same position; the segment error is the mean
absolute difference over the grid and the two channels; a window's error is the mean over its whole 2-s segments
(L = 5 s gives 2 segments, the last second is not scored).
"""
import logging

import numpy as np
from scipy.signal import welch

from src import model
from src import state_space as ss
from src import ukf
from src import ukf_ext

log = logging.getLogger(__name__)


class FreeRunError(ValueError):
    """Raised for an unusable free-run request."""


# ---- per-realization delay ring --------------------------------------------------------------------------------------

class BatchRing:
    """The DelayBuffer of state_space with one ring per realization: entries of shape (R, nodes). Same indexing: read(lag)
    is the entry written lag sub-steps ago, write() appends, replace_latest() overwrites the newest."""

    def __init__(self, snapshot, n_real):
        snap = np.asarray(snapshot, dtype=np.float64)
        if snap.ndim != 2:
            raise FreeRunError("a ring snapshot has shape (delay + 1, nodes)")
        size = snap.shape[0]
        self.delay = size - 1
        self._buf = np.empty((size, int(n_real), snap.shape[1]), dtype=np.float64)
        for lag in range(size):
            self._buf[(-lag) % size] = snap[lag]
        self._head = 0

    def read(self, lag):
        if not 0 <= lag <= self.delay:
            raise FreeRunError(f"lag {lag} outside 0..{self.delay}")
        return self._buf[(self._head - lag) % self._buf.shape[0]]

    def write(self, s):
        self._head = (self._head + 1) % self._buf.shape[0]
        self._buf[self._head] = s

    def replace_latest(self, s):
        self._buf[self._head] = s

    def snapshot(self):
        """Lag-indexed entries, shape (delay + 1, R, nodes)."""
        return np.array([self.read(lag) for lag in range(self.delay + 1)])


def propagate(X, ring, pots, layout, cfg, residual=None):
    """One observation interval for every row of X (R, layout.n): Heun sub-steps, no noise, exactly state_space.predict but
    with the S (and potential) written per row. With one row this is predict(points, [1.0], buffer, ...) bit for bit."""
    dt, n_sub = ss.substep_dt(cfg)
    k = model.constants(cfg)
    X = np.array(X, dtype=np.float64, copy=True)
    for _ in range(n_sub):
        if pots is None:
            k1 = ss.drift(X, ring.read(ring.delay), layout, cfg)
            k2 = ss.drift(X + dt * k1, ring.read(ring.delay - 1), layout, cfg)
        else:
            k1 = ss.drift(X, ring.read(ring.delay), layout, cfg, residual, pots.read(pots.delay))
            k2 = ss.drift(X + dt * k1, ring.read(ring.delay - 1), layout, cfg, residual, pots.read(pots.delay - 1))
        X = X + 0.5 * dt * (k1 + k2)
        pot = ss.potentials(X)
        ring.write(model.sigmoid(pot, k["e0"], k["v0"], k["r"]))
        if pots is not None:
            pots.write(pot)
    return X


# ---- the stochastic simulation of one window ---------------------------------------------------------------------------

def simulate(start, n_steps, rng, cfg, layout, spec, q, residual=None, n_real=None, keep_state=False):
    """Free-run of one window. start: {"x": full filtered state (layout.n + spec.nx), "ring": (delay + 1, 2), "pots":
    (delay + 1, 2) or None}. Returns {"y": (R, n_steps, 2) simulated observation, "unstable": (R,) bool} (and "X", "noise", "ring", "pots"
    when keep_state). Stops at the first step at which any realization is unstable (the window is then unstable and its
    remaining output is NaN)."""
    n_real = int(cfg["windows"]["c4"]["n_realizations"] if n_real is None else n_real)
    nb, nx = layout.n, spec.nx
    x0 = np.asarray(start["x"], dtype=np.float64)
    if x0.shape != (nb + nx,):
        raise FreeRunError(f"the start state has {x0.shape}, expected ({nb + nx},)")
    if (residual is None) != (start.get("pots") is None):
        raise FreeRunError("a residual needs the potential ring of its filter, and no residual must not carry one")
    X = np.tile(x0[:nb], (n_real, 1))
    noise = np.tile(x0[nb:], (n_real, 1))
    ring = BatchRing(start["ring"], n_real)
    pots = None if residual is None else BatchRing(start["pots"], n_real)
    phi, q_ex, _ = ukf_ext.extra_terms(spec, cfg)
    sd_proc = np.sqrt(np.diag(ukf.process_noise(layout, cfg, q))[:ss.N_NEURAL])
    sd_ou = np.sqrt(q_ex)
    sd_obs = np.sqrt(np.diag(ukf.obs_noise(cfg)))
    sd12 = np.sqrt(np.tile(ss.neural_variance(cfg), ss.N_NODES))
    mult = float(cfg["ukf"]["divergence"]["state_sd_multiple"])
    centre = ukf.DivergenceReference(layout, cfg).center(x0[:nb])
    k = model.constants(cfg)
    draws = rng.standard_normal((int(n_steps), n_real, ss.N_NEURAL + nx + ss.N_NODES))
    y = np.full((n_real, int(n_steps), ss.N_NODES), np.nan)
    unstable = np.zeros(n_real, dtype=bool)
    with np.errstate(all="ignore"):
        for t in range(int(n_steps)):
            X = propagate(X, ring, pots, layout, cfg, residual)
            X[:, :ss.N_NEURAL] += sd_proc * draws[t, :, :ss.N_NEURAL]
            pot = ss.potentials(X)
            ring.replace_latest(model.sigmoid(pot, k["e0"], k["v0"], k["r"]))
            if pots is not None:
                pots.replace_latest(pot)
            if nx:
                noise = noise * phi + sd_ou * draws[t, :, ss.N_NEURAL:ss.N_NEURAL + nx]
            obs = ss.observe(X, layout, cfg) + (noise if nx else 0.0)
            y[:, t] = obs + sd_obs * draws[t, :, -ss.N_NODES:]
            unstable |= ~np.all(np.isfinite(X), axis=1) | np.any(np.abs(X[:, :ss.N_NEURAL] - centre) > mult * sd12, axis=1)
            if unstable.any():
                break
    out = {"y": y, "unstable": unstable}
    if keep_state:
        out.update(X=X, noise=noise, ring=ring, pots=pots)
    return out


# ---- spectra and the window error ------------------------------------------------------------------------------------------

def grid(cfg):
    """(sample rate, samples per 2-s segment, the 0.5-Hz grid bins over 1 to 45 Hz as a boolean mask of the rfft bins)."""
    c4 = cfg["windows"]["c4"]
    fs = cfg["preprocessing"]["observation_fs_hz"]
    seg_n = int(round(c4["welch_segment_s"] * fs))
    f = np.fft.rfftfreq(seg_n, 1.0 / fs)
    sel = (f >= c4["freq_range_hz"][0]) & (f <= c4["freq_range_hz"][1])
    if not np.allclose(np.diff(f[sel]), c4["grid_resolution_hz"]):
        raise FreeRunError("the Welch segment does not give the configured frequency grid")
    return fs, seg_n, sel


def log_spectra(x, cfg):
    """log10 of the Welch spectrum (one Hann segment of the configured length) of x (..., seg_n), on the 1 to 45 Hz grid."""
    fs, seg_n, sel = grid(cfg)
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != seg_n:
        raise FreeRunError(f"a segment has {seg_n} samples, got {x.shape[-1]}")
    _, p = welch(x, fs=fs, window="hann", nperseg=seg_n, noverlap=0, detrend="constant", axis=-1)
    with np.errstate(divide="ignore"):
        return np.log10(p[..., sel])


def n_segments(length_s, cfg):
    """Whole 2-s segments of a window of length_s seconds (5 s gives 2: the last second is not scored)."""
    return int(length_s // cfg["windows"]["c4"]["welch_segment_s"])


def window_error(observed, simulated, length_s, cfg):
    """(window error, [segment errors]). observed (2, L fs), simulated (R, n, 2) with n >= L fs. Segment j compares the
    observed log10 spectrum with the mean over the R realizations of the simulated log10 spectrum (a mean of logs, not the log
    of a mean) at the same position; the segment error is the mean absolute difference over the grid and both channels."""
    fs, seg_n, _ = grid(cfg)
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    errs = []
    for j in range(n_segments(length_s, cfg)):
        sl = slice(j * seg_n, (j + 1) * seg_n)
        lo = log_spectra(obs[:, sl], cfg)                                   # (2, bins)
        ls = log_spectra(np.moveaxis(sim[:, sl, :], 2, 1), cfg).mean(axis=0)  # (R, 2, bins) -> (2, bins)
        e = float(np.mean(np.abs(lo - ls)))
        if not np.isfinite(e):
            raise FreeRunError("a segment error is not finite")
        errs.append(e)
    if not errs:
        raise FreeRunError("a window shorter than one segment")
    return float(np.mean(errs)), errs


# ---- window plan ----------------------------------------------------------------------------------------------------------------

def mask_starts(mask_list):
    """First scored sample of every segment (None for a segment with none), from passes.scoring_mask."""
    out = []
    for m in mask_list:
        idx = np.flatnonzero(np.asarray(m, dtype=bool))
        out.append(int(idx[0]) if idx.size else None)
    return out


def window_plan(seg_lengths, first_scored, seg_ok, length_samples):
    """[(segment, t0)]: non-overlapping windows anchored at the first scored sample of each segment, each fully inside one
    segment, in segments marked ok (not diverged) only. The tail that does not fill a window is dropped."""
    plan = []
    for k, (n, a0, ok) in enumerate(zip(seg_lengths, first_scored, seg_ok)):
        if not ok or a0 is None:
            continue
        plan += [(k, a0 + h * int(length_samples)) for h in range((int(n) - a0) // int(length_samples))]
    return plan


def capture_indices(seg_lengths, first_scored, lengths_samples):
    """Per segment, every sample index at which a window of any length starts (for passes.run_pass1 capture_idx)."""
    out = []
    for n, a0 in zip(seg_lengths, first_scored):
        idx = set()
        if a0 is not None:
            for L in lengths_samples:
                idx |= {a0 + h * int(L) for h in range((int(n) - a0) // int(L))}
        out.append(sorted(idx))
    return out


def score_windows(segments, p1, spec, layout, cfg, q, lengths_s, rng_key, residual=None, n_real=None):
    """Free-run every planned window of a recording whose pass 1 (with captured states) did not diverge, at every length.
    Returns {length_s: [{"segment", "t0", "stable", "error", "segment_errors"}]}; an unstable window has error None.
    rng_key: (seed, subject index); the stream of a window is default_rng([seed, subject index, length_s, window index]),
    the same for M2 and M3."""
    from src import passes
    fs = cfg["preprocessing"]["observation_fs_hz"]
    seg_lengths = [s.n for s in p1.segments]
    first = mask_starts(passes.scoring_mask(seg_lengths, cfg))
    seg_ok = [not s.diverged for s in p1.segments]
    out = {}
    for L in lengths_s:
        Ln = int(round(L * fs))
        rows = []
        for h, (k, t0) in enumerate(window_plan(seg_lengths, first, seg_ok, Ln)):
            start = p1.segments[k].captured[t0]
            rng = np.random.default_rng([int(rng_key[0]), int(rng_key[1]), int(L), int(h)])
            sim = simulate(start, Ln, rng, cfg, layout, spec, q, residual=residual, n_real=n_real)
            row = {"segment": int(k), "t0": int(t0), "stable": not bool(sim["unstable"].any()), "error": None,
                   "segment_errors": None}
            if row["stable"]:
                obs = np.asarray(segments[k], dtype=np.float64)[:, t0:t0 + Ln]
                row["error"], row["segment_errors"] = window_error(obs, sim["y"], L, cfg)
            rows.append(row)
        out[L] = rows
    return out
