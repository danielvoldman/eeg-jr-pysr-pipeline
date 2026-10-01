"""Synthetic gate G0 (PLAN.md Stage E; §5.2, §9, §18.1). This file holds E1 (the operating-regime grid), E2
(series generator, Null A, Null B, the G0 tuning set), E3 (the artifacts of the preprocessing gate and its bias
scoring) and E4 (the filter option 19D / A / B, evaluation of a series, the scoring of §9.1 to §9.3, the flags and
gate.json).

E1 (§9.1, "operating regime matched to real data"): a grid of simulations over (p, input-noise SD, additive
observation-noise level), 12 x 8 x 5 by default, each reduced to three SCALE-FREE features (alpha peak
frequency, relative alpha power, specparam aperiodic exponent). A series draws one feature vector at random
from the TRAINING recordings and takes the nearest grid point; absolute power is never matched because it
depends on the rescaling reference (§5.1). Each grid point also carries a regime label, noise-driven or
limit-cycle, from a noise-free run at its p (reported, §9.1).

Training-only rule (CLAUDE.md rule 6): the feature table is built only from subjects on the training side of
the split (check_training_side) and, in development runs, only from pilot subjects (preprocess guard). Every
random draw uses numpy.random.default_rng with a seed from config g0.seeds (rule 3). Nothing here calls PySR.

E2 (§9.1, §9.2, §7.5): generate_series() makes one series with exact truth: stochastic Heun at 2,048 Hz with the
planted product residual (a compiled kernel here; src/model.py is untouched and the kernel equals model.simulate
when the planted coefficient is 0), the series' operating point from the E1 grid, mixing m ~ U(0.1, 0.4),
1/f plus white noise, scaling to the training median bipolar SD in uV, then the real notch, band-pass,
edge trim and segment_signal (blinks, 1-s rejection, anti-aliased downsampling, rescaling to mu_ref/sigma_ref).
Null A has zero coupling and no planted term; Null B adds a shared input with a lag of 0-20 ms carrying 30-50% of
each node's input variance. Every random draw is default_rng([block + round x stride, arm, substream, i]).

E3 (§5.2): artifact_signal() adds blinks, EMG bursts and 0.05 Hz drift in uV before the real preprocessing; the
positive arm draws them independently per channel, the artifact-only null (zero coupling) draws ONE set that hits both
channels with a 0-4 ms lag and a shared carrier. preprocessing_bias_verdict() scores the signed median bias of the gains
and of the E/I ratio (p excluded).

E4 (§9.1 to §9.3, §18.1): the filter is an OPTION (g0.filter_options 19D, A, B; ukf_ext is promoted, not adopted,
IMP-068); gate_filters() lets a pilot run every option and makes a full run refuse while g0.filter is unset.
evaluate_series() runs both passes with the standard divergence rule on (a diverged series is a failed series) and a
flag-off pass 1 labelled diagnostic only; positive_verdict / null_arm_verdict / contraction_verdict /
stability_verdict / gate_flags score; build_gate_document / write_gate produce gate.json (pilot: results/pilot only).

Entry points: python -m src.synthetic_gate --grid-report --pilot
              python -m src.synthetic_gate --time-series --pilot
              python -m src.synthetic_gate --time-gate-series --pilot
"""
import copy
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from numba import njit
from scipy.signal import welch

from src import model
from src import preprocess as pp
from src.config import REPO_ROOT, load_config
from src.model import N_STATES, Y1, Y2, _rhs_node, _sig

log = logging.getLogger(__name__)

_SOURCE = Path(__file__).resolve()
FEATURE_NAMES = ("alpha_peak_hz", "relative_alpha_power", "aperiodic_exponent")
LIMIT_CYCLE, NOISE_DRIVEN = "limit_cycle", "noise_driven"


class GateError(ValueError):
    """Raised for anything the gate must refuse (test subjects, unset decisions, malformed tables)."""


def emit(text=""):
    """The one place report text is written to stdout (development reports); everything else logs."""
    sys.stdout.write(text + "\n")


# ---------------------------------------------------------------- the grid axes

def axes(cfg):
    """(p values, input-SD factors, noise shares): linear, geometric, linear over the configured ranges,
    with the counts of g0.regime_grid_shape."""
    rc = cfg["g0"]["regime"]
    n_p, n_sd, n_ns = cfg["g0"]["regime_grid_shape"]
    p = np.linspace(rc["p_range"][0], rc["p_range"][1], n_p)
    sdf = np.geomspace(rc["input_sd_factor_range"][0], rc["input_sd_factor_range"][1], n_sd)
    share = np.linspace(rc["noise_share_range"][0], rc["noise_share_range"][1], n_ns)
    return p, sdf, share


def point_index(cfg, ip, isd, ins):
    """Flat grid index; p varies slowest, noise share fastest."""
    _, n_sd, n_ns = cfg["g0"]["regime_grid_shape"]
    return (ip * n_sd + isd) * n_ns + ins


# ---------------------------------------------------------------- observation noise and the real chain

def inband_power(x, fs_hz, band):
    """Sum of |FFT|^2 over the bins with band[0] <= f <= band[1] (mean removed). Only ratios are used."""
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / fs_hz)
    sel = (f >= band[0]) & (f <= band[1])
    return float(np.sum(np.abs(spec[sel]) ** 2))


def observe_noisy(cfg, y, fs_hz, m, share, exponent, rng):
    """Mixing (on the deviations from mu_ref, IMP-015) and additive noise on the model output y (n, 2) in model
    units. The noise carries `share` of the 1-45 Hz variance of the observed channel: its in-band power is
    share / (1 - share) times the signal's, split g0.regime.one_over_f_fraction_of_noise to 1/f (power
    exponent `exponent`) and the rest white, independent per channel. Returns z (2, n)."""
    rc = cfg["g0"]["regime"]
    band = cfg["g0"]["features"]["total_band_hz"]
    mu = cfg["rescaling"]["mu_ref"]
    M = np.array([[1.0, m], [m, 1.0]])
    mixed = M @ (np.asarray(y, dtype=np.float64) - mu).T
    n = mixed.shape[1]
    z = np.empty_like(mixed)
    frac = rc["one_over_f_fraction_of_noise"]
    for ch in range(2):
        target = share / (1.0 - share) * inband_power(mixed[ch], fs_hz, band)
        unit_f = pp._shaped_noise(rng, n, exponent)
        unit_w = rng.standard_normal(n)
        c_f = np.sqrt(frac * target / inband_power(unit_f, fs_hz, band))
        c_w = np.sqrt((1.0 - frac) * target / inband_power(unit_w, fs_hz, band))
        z[ch] = mu + mixed[ch] + c_f * unit_f + c_w * unit_w
    return z


def lite_chain(cfg, z, fs_hz):
    """The real band-pass, edge trim, anti-aliased downsampling and rescaling to mu_ref / sigma_ref on one
    continuous series (2, n) at fs_hz. Grid features only (no artifact injection, so no blink or EMG steps)."""
    bp = pp.bandpass(cfg, z, fs_hz)
    bp = pp.trim_edges(bp, fs_hz, pp.edge_trim_seconds(cfg))
    x = pp.downsample(cfg, bp, fs_hz)
    scaled, _ = pp.rescale(cfg, [x])
    return scaled[0]


# ---------------------------------------------------------------- scale-free features

def mean_psd(segments, fs_hz, seg_s):
    """Average Hann periodogram over non-overlapping seg_s blocks of every segment and channel (weighted by
    the number of blocks). Returns (freqs, psd)."""
    nper = int(round(seg_s * fs_hz))
    acc, count, freqs = None, 0, None
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float64)
        blocks = seg.shape[-1] // nper
        if blocks == 0:
            continue
        for ch in seg:
            freqs, p = welch(ch[:blocks * nper], fs=fs_hz, window="hann", nperseg=nper, noverlap=0,
                             detrend="constant")
            acc = p * blocks if acc is None else acc + p * blocks
            count += blocks
    if acc is None:
        raise GateError("mean_psd: no segment holds a whole block")
    return freqs, acc / count


def spectral_features(cfg, freqs, psd):
    """(alpha_peak_hz, relative_alpha_power, aperiodic_exponent) of one PSD, or NaNs for a failed fit.
    Alpha peak: the strongest specparam peak with centre in the alpha band, else the PSD maximum in the band.
    Relative alpha power: alpha-band share of the total-band power (trapezoid). Exponent: specparam
    aperiodic fit over g0.features.specparam_freq_range_hz."""
    from specparam import SpectralModel      # lazy: only the gate needs it
    fc = cfg["g0"]["features"]
    lo, hi = fc["alpha_band_hz"]
    tot_lo, tot_hi = fc["total_band_hz"]
    sel_a = (freqs >= lo) & (freqs <= hi)
    sel_t = (freqs >= tot_lo) & (freqs <= tot_hi)
    rel = float(np.trapezoid(psd[sel_a], freqs[sel_a]) / np.trapezoid(psd[sel_t], freqs[sel_t]))
    peak = np.nan
    exponent = np.nan
    try:
        sm = SpectralModel(aperiodic_mode=fc["specparam_aperiodic_mode"],
                           peak_width_limits=tuple(fc["specparam_peak_width_limits_hz"]),
                           max_n_peaks=fc["specparam_max_n_peaks"], verbose=False)
        sm.fit(freqs, psd, list(fc["specparam_freq_range_hz"]))
        exponent = float(sm.results.params.aperiodic.params[-1])
        pk = np.atleast_2d(sm.results.params.periodic.params)
        pk = pk[(pk[:, 0] >= lo) & (pk[:, 0] <= hi)] if pk.size else pk
        if pk.size:
            peak = float(pk[np.argmax(pk[:, 1]), 0])
    except Exception as exc:                 # specparam raises its own fit errors; a failed fit is a NaN feature
        log.warning("specparam fit failed: %s", exc)
    if np.isnan(peak) and np.isfinite(exponent):
        peak = float(freqs[sel_a][np.argmax(psd[sel_a])])
    return np.array([peak, rel, exponent], dtype=np.float64)


# ---------------------------------------------------------------- the training feature table

@dataclass
class FeatureTable:
    names: list                  # "sub-005_ses-t1"
    features: np.ndarray         # (n, 3): FEATURE_NAMES
    sd_uv: np.ndarray            # (n, 2) clean-sample SD per channel after the band-pass, before rescaling (uV)
    skipped: list = field(default_factory=list)   # (name, reason)

    @property
    def target_sd_uv(self):
        """Median over recordings and channels of the real bipolar SD in uV (§9.1 'Units')."""
        return float(np.median(self.sd_uv))

    @property
    def scale(self):
        sd = self.features.std(axis=0, ddof=1)
        if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
            raise GateError(f"feature table SD is zero or not finite ({sd}); cannot scale for matching")
        return sd

    @property
    def exponent(self):
        """Median aperiodic exponent of the training recordings: the 1/f noise generator's exponent."""
        return float(np.median(self.features[:, 2]))


def load_split(cfg, root, seed=None):
    seed = cfg["split"]["primary_seed"] if seed is None else seed
    path = Path(root) / cfg["paths"]["split_file_pattern"].format(seed=seed)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def check_training_side(subjects, split):
    """Every subject must be on the training side of the split (rule 6); test subjects never enter G0."""
    train = set(split["train"])
    bad = sorted(set(subjects) - train)
    if bad:
        raise GateError(f"subjects not on the training side of the split: {bad}")


def make_recording_loader(cfg, root, pilot_ids, allow_all=False):
    """Loader for build_feature_table: every ses-t1 / ses-t2 EEG file of a subject through B4/B5 (cached) and the
    B6 decision. Returns a list of {name, reason, segments, sd_uv}."""
    root = Path(root)
    manifest = pp.load_manifest(root / cfg["paths"]["manifest_file"])
    cache_root = root / cfg["paths"]["cache_dir"]
    data_root = root / cfg["paths"]["data_dir"]

    def load(subject):
        pattern = cfg["dataset"]["eeg_glob"].replace("sub-*", subject, 1)
        out = []
        for f in sorted(data_root.glob(pattern)):
            res = pp.segment_recording(cfg, f, data_root, manifest, pilot_ids, allow_all=allow_all,
                                       cache_root=cache_root)
            name = f"{res.meta['subject']}_{res.meta['session']}"
            decision = pp.recording_decision(cfg, res.meta, res.meta)
            if decision["status"] != "kept":
                out.append({"name": name, "reason": f"excluded: {decision['reason']}"})
                continue
            out.append({"name": name, "reason": None, "segments": res.segments,
                        "sd_uv": res.meta["b5"]["clean_stats_before"]["sd"]})
        return out

    return load


def build_feature_table(cfg, subjects, split, loader):
    """Feature table of the given TRAINING subjects (both sessions, every recording that is kept)."""
    check_training_side(subjects, split)
    fs = cfg["preprocessing"]["observation_fs_hz"]
    seg_s = cfg["g0"]["features"]["welch_segment_s"]
    names, feats, sds, skipped = [], [], [], []
    for subject in sorted(subjects):
        for rec in loader(subject):
            if rec["reason"] is not None:
                skipped.append((rec["name"], rec["reason"]))
                continue
            f, psd = mean_psd(rec["segments"], fs, seg_s)
            vec = spectral_features(cfg, f, psd)
            if not np.all(np.isfinite(vec)):
                skipped.append((rec["name"], "feature fit failed"))
                continue
            names.append(rec["name"])
            feats.append(vec)
            sds.append(np.asarray(rec["sd_uv"], dtype=np.float64))
    if not names:
        raise GateError("no usable training recording for the feature table")
    return FeatureTable(names, np.vstack(feats), np.vstack(sds), skipped)


# ---------------------------------------------------------------- regime label

def classify_regime(cfg, p):
    """Noise-free single-node run at input p from a perturbed steady state: peak-to-peak of y1 - y2 over the last
    classify_tail_s above the threshold means limit cycle, else noise-driven (the fixed point is stable)."""
    rc = cfg["g0"]["regime"]
    fs = cfg["rescaling"]["reference_simulation"]["sim_fs_hz"]
    jr = cfg["jansen_rit"]
    steady = model.steady_state(float(p), jr["A"], jr["B"], cfg)
    y0 = np.array(steady, dtype=np.float64)
    y0[model.Y1] += rc["classify_perturbation_mv"]
    n = int(round(rc["classify_duration_s"] * fs))
    res = model.simulate(cfg, n, input=np.full((n, 1), float(p)), n_nodes=1, p=float(p), y_init=y0[None, :])
    tail = res.output[-int(round(rc["classify_tail_s"] * fs)):, 0]
    return LIMIT_CYCLE if float(tail.max() - tail.min()) > rc["classify_pp_threshold_mv"] else NOISE_DRIVEN


# ---------------------------------------------------------------- the grid

@dataclass
class Grid:
    p: np.ndarray
    sd_factor: np.ndarray
    share: np.ndarray
    features: np.ndarray         # (n_points, 3), NaN where the fit failed
    regime: list                 # per point
    exponent: float              # the 1/f exponent used by the generator
    key: str
    from_cache: bool = False

    @property
    def valid(self):
        return np.all(np.isfinite(self.features), axis=1)

    def point(self, index):
        _, n_sd, n_ns = (len(self.p), len(self.sd_factor), len(self.share))
        ip, rem = divmod(index, n_sd * n_ns)
        isd, ins = divmod(rem, n_ns)
        return {"index": int(index), "p": float(self.p[ip]), "input_sd_factor": float(self.sd_factor[isd]),
                "noise_share": float(self.share[ins]), "regime": self.regime[index]}


def grid_point_features(cfg, index, p, sdf, share, exponent):
    """Features of one grid point: 2 uncoupled nodes at p, input half-width x sdf, mixing m, noise `share`."""
    rc = cfg["g0"]["regime"]
    fs = cfg["g0"]["generation_fs_hz"]
    seed_base = cfg["g0"]["seeds"]["grid"]
    n_burn = int(round(rc["grid_burn_in_s"] * fs))
    n_keep = int(round(rc["grid_sim_duration_s"] * fs))
    res = model.simulate(cfg, n_burn + n_keep, seed=seed_base + int(index), n_nodes=2, p=float(p),
                         half_width=model.default_half_width(cfg) * float(sdf))
    y = np.ascontiguousarray(res.output[1 + n_burn:, :], dtype=np.float64)
    rng = np.random.default_rng([seed_base, int(index)])
    z = observe_noisy(cfg, y, fs, rc["grid_mixing_m"], float(share), exponent, rng)
    x = lite_chain(cfg, z, fs)
    f, psd = mean_psd([x], cfg["preprocessing"]["observation_fs_hz"], cfg["g0"]["features"]["welch_segment_s"])
    return spectral_features(cfg, f, psd)


def _grid_worker(job):
    from threadpoolctl import threadpool_limits
    cfg, index, p, sdf, share, exponent = job
    with threadpool_limits(limits=1):
        return grid_point_features(cfg, index, p, sdf, share, exponent)


def grid_key(cfg, exponent):
    sections = {k: cfg[k] for k in ("g0", "rescaling", "jansen_rit", "simulator", "coupling", "preprocessing")}
    blob = json.dumps(sections, sort_keys=True, default=str) + repr(float(exponent))
    code = b"".join((_SOURCE.parent / name).read_bytes() for name in ("synthetic_gate.py", "model.py", "preprocess.py"))
    return hashlib.sha256(blob.encode("utf-8") + code).hexdigest()


def build_grid(cfg, exponent, cache_dir=None, n_jobs=1):
    """The regime grid (features and regime labels), cached under cache_dir by a key over the config, the code
    and the exponent. n_jobs > 1 uses joblib/loky (spawn-safe worker at module level)."""
    key = grid_key(cfg, exponent)
    path = None if cache_dir is None else Path(cache_dir) / f"regime_grid_{key[:16]}.npz"
    p, sdf, share = axes(cfg)
    if path is not None and path.is_file():
        with np.load(path, allow_pickle=False) as d:
            return Grid(p, sdf, share, d["features"], [str(r) for r in d["regime"]], float(exponent), key, True)
    jobs = [(cfg, point_index(cfg, ip, isd, ins), p[ip], sdf[isd], share[ins], exponent)
            for ip in range(len(p)) for isd in range(len(sdf)) for ins in range(len(share))]
    if n_jobs > 1:
        from joblib import Parallel, delayed
        feats = Parallel(n_jobs=n_jobs, backend=cfg["compute"]["joblib_backend"])(delayed(_grid_worker)(j) for j in jobs)
    else:
        feats = [_grid_worker(j) for j in jobs]
    features = np.vstack(feats)
    labels = {float(v): classify_regime(cfg, v) for v in p}
    regime = [labels[float(j[2])] for j in jobs]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, features=features, regime=np.array(regime))
    return Grid(p, sdf, share, features, regime, float(exponent), key, False)


# ---------------------------------------------------------------- matching a series to a grid point

def draw_feature_vector(table, rng):
    """One recording's feature vector drawn at random from the training table."""
    i = int(rng.integers(len(table.names)))
    return i, table.features[i]


def match_grid_point(grid_features, valid, target, scale):
    """Nearest valid grid point to `target` in features divided by `scale` (Euclidean); ties go to the first.
    Returns (index, distance)."""
    d = np.sqrt((((grid_features - target) / scale) ** 2).sum(axis=1))
    d = np.where(valid, d, np.inf)
    i = int(np.argmin(d))
    if not np.isfinite(d[i]):
        raise GateError("no valid grid point")
    return i, float(d[i])


def stream_base(cfg, stream, round_=0):
    """Seed block of a stream ('pilot', 'full', 'tuning', 'preprocessing_gate') in fresh-seed round `round_`."""
    sd = cfg["g0"]["seeds"]
    return sd[stream] + int(round_) * sd["fresh_round_stride"]


def series_operating_point(cfg, grid, table, stream, i, round_=0):
    """The operating point of series i of a seed stream: random training feature vector, nearest grid point.
    Arm code 0, so series i of every arm of a stream shares its operating point (paired comparison)."""
    rng = np.random.default_rng([stream_base(cfg, stream, round_), 0, int(i)])
    row, vec = draw_feature_vector(table, rng)
    idx, dist = match_grid_point(grid.features, grid.valid, vec, table.scale)
    return {"series": int(i), "recording": table.names[row], "target": [float(v) for v in vec],
            "distance": dist, **grid.point(idx)}


# ---------------------------------------------------------------- E2: the planted-residual simulator

@njit(fastmath=False)
def _planted_kernel(u_in, y_init, s_fill, pot_fill, G, c, sd, delay, dt, A, B, a, b, C1, C2, C3, C4, e0, v0, r):
    """Heun kernel of src.model with one extra additive term in the y4 bracket,
    r_res_j(t) = c_j u_src(t - d) u_tgt(t), u = (y1 - y2) / sd, held within a step like the input (§9.1).
    With c = 0 the states equal model._simulate_kernel bit for bit."""
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


def _run_kernel(cfg, u, g12, g21, p, c, sd):
    k = model.constants(cfg)
    jr = cfg["jansen_rit"]
    A = np.full(2, float(jr["A"]))
    B = np.full(2, float(jr["B"]))
    steady = [model.steady_state(float(p[i]), A[i], B[i], cfg) for i in range(2)]
    s_fill = np.array([model.sigmoid(v[Y1] - v[Y2], k["e0"], k["v0"], k["r"]) for v in steady])
    pot_fill = np.array([v[Y1] - v[Y2] for v in steady])
    G = np.zeros((2, 2))
    G[1, 0], G[0, 1] = g12, g21
    fs = float(cfg["g0"]["generation_fs_hz"])
    delay = int(cfg["coupling"]["sim_delay_steps"])
    out = _planted_kernel(np.ascontiguousarray(u, dtype=np.float64), np.array(steady, dtype=np.float64), s_fill,
                          pot_fill, G, np.asarray(c, dtype=np.float64), np.asarray(sd, dtype=np.float64), delay,
                          1.0 / fs, A, B, k["a"], k["b"], k["C1"], k["C2"], k["C3"], k["C4"], k["e0"], k["v0"], k["r"])
    return out, A, delay


def run_planted(cfg, u, g12, g21, p, c_rel, burn_samples):
    """Two-pass planted simulation (§9.1): a first run with c = 0 gives each node's SD of y1 - y2 and the RMS of
    its base coupling drive after the burn-in; c_j is set so RMS(r_res_j) = c_rel x RMS(drive_j) in that run, and the
    series is simulated again with the term (the realised ratio of the final run is returned, not retuned).
    c_rel = 0 simulates once without the term. Returns a dict with the states and the exact bookkeeping."""
    p = np.asarray(p, dtype=np.float64)
    n_steps = u.shape[0]
    (states, drive, res, s_all, pot_all), A, delay = _run_kernel(cfg, u, g12, g21, p, [0.0, 0.0], [1.0, 1.0])
    pot = states[burn_samples:n_steps, :, Y1] - states[burn_samples:n_steps, :, Y2]
    sd = pot.std(axis=0)
    c = np.zeros(2)
    if c_rel > 0.0:
        for j in (0, 1):
            src = 1 - j
            basis = (pot_all[burn_samples:n_steps, src] / sd[src]) * (pot_all[burn_samples + delay:n_steps + delay, j] / sd[j])
            c[j] = c_rel * np.sqrt(np.mean(drive[burn_samples:, j] ** 2)) / np.sqrt(np.mean(basis ** 2))
        (states, drive, res, s_all, pot_all), A, delay = _run_kernel(cfg, u, g12, g21, p, c, sd)
    ratio = [float(np.sqrt(np.mean(res[burn_samples:, j] ** 2)) / np.sqrt(np.mean(drive[burn_samples:, j] ** 2)))
             if np.any(drive[burn_samples:, j]) else float("nan") for j in (0, 1)]
    return {"states": states, "drive": drive, "res": res, "s_all": s_all, "pot_all": pot_all, "c": c, "sd": sd,
            "A": A, "delay": delay, "rms_ratio": ratio}


def level_gain(cfg, level_index):
    """Coupling gain of a level: coupling_levels_x_C2[level] x C2 (about 2.16, 5.4, 10.8, 27 s^-1)."""
    return float(cfg["g0"]["coupling_levels_x_C2"][level_index]) * model.constants(cfg)["C2"]


def delta(cfg):
    """The null band (§9.2): half the weakest planted level."""
    return cfg["g0"]["pass"]["null_delta_fraction_of_weakest_level"] * level_gain(cfg, 0)


# ---------------------------------------------------------------- E2: seeds and the Null B input

def series_rng(cfg, stream, arm, substream, i, round_=0):
    sd = cfg["g0"]["seeds"]
    return np.random.default_rng([stream_base(cfg, stream, round_), sd["arm_codes"][arm],
                                  sd["substreams"][substream], int(i)])


def null_b_input(cfg, rng_input, rng_common, n_steps, p, half_width):
    """Input of Null B (§9.2): the independent uniform input of every node plus a shared stochastic component, mixed
    with weights sqrt(1 - s) and sqrt(s) so each node's total input variance is unchanged and the shared part carries
    the share s ~ U(common_input_variance_share). The shared signal reaches one node (random per series) `lag` steps
    before the other, lag ~ U(common_input_lag_ms) quantised to generation steps. Returns (u (n, 2), meta)."""
    nb = cfg["g0"]["null_B"]
    fs = cfg["g0"]["generation_fs_hz"]
    share = float(rng_common.uniform(*nb["common_input_variance_share"]))
    lag_ms = float(rng_common.uniform(*nb["common_input_lag_ms"]))
    lag = int(round(lag_ms * 1e-3 * fs))
    lead = int(rng_common.integers(2))
    ec = rng_common.uniform(-1.0, 1.0, size=n_steps + lag)
    own = rng_input.uniform(-1.0, 1.0, size=(n_steps, 2))
    common = np.empty((n_steps, 2))
    common[:, lead] = ec[lag:]
    common[:, 1 - lead] = ec[:n_steps]
    u = np.asarray(p, dtype=np.float64) + half_width * (np.sqrt(1.0 - share) * own + np.sqrt(share) * common)
    return u, {"share": share, "lag_ms_drawn": lag_ms, "lag_steps": lag, "leading_node": lead}


# ---------------------------------------------------------------- E2: units, the real chain, truth rows

def scale_to_uv(cfg, z, fs_hz, target_sd_uv):
    """Scale each channel (mean removed) so that its SD after the real band-pass and edge trim equals the training
    median bipolar SD in uV (§9.1 'Units'). Returns (x_uv, per-channel factor)."""
    bp = pp.trim_edges(pp.bandpass(cfg, z, fs_hz), fs_hz, pp.edge_trim_seconds(cfg))
    factor = target_sd_uv / bp.std(axis=1, ddof=1)
    return (z - z.mean(axis=1, keepdims=True)) * factor[:, None], factor


def preprocess_series(cfg, x_uv, fs_hz, strict=False):
    """The real chain on bipolar channels in uV at the generation rate: line-noise notch, zero-phase band-pass, edge
    trim, then preprocess.segment_signal (blinks, 1-s rejection, padding, anti-aliased downsampling to the
    observation rate, clean stretches, rescaling to mu_ref and sigma_ref). The electrode-level steps (bad-channel
    flags on the four electrodes) have no synthetic counterpart."""
    trim_s = pp.edge_trim_seconds(cfg)
    notched = pp.notch(cfg, x_uv, fs_hz)
    bp = pp.trim_edges(pp.bandpass(cfg, notched, fs_hz), fs_hz, trim_s)
    return pp.segment_signal(cfg, bp, pp.trim_edges(notched, fs_hz, trim_s), fs_hz, strict)


def observation_rows(cfg, n_obs):
    """Simulator state rows of the first n_obs observation samples of the trimmed series: observation sample j sits
    at generation index trim + step x j of the kept output, i.e. state row 1 + burn + trim + step x j."""
    fs = cfg["g0"]["generation_fs_hz"]
    step = int(round(fs / cfg["preprocessing"]["observation_fs_hz"]))
    first = 1 + int(round(cfg["g0"]["generation_burn_in_s"] * fs)) + int(round(pp.edge_trim_seconds(cfg) * fs))
    return first + step * np.arange(n_obs)


@dataclass
class Series:
    arm: str
    index: int
    stream: str
    level_index: object          # None for the nulls
    gains: tuple                 # true (g12, g21)
    m: float
    operating_point: dict        # includes the matched z-distance
    segments: list
    starts: list
    truth: object                # dict of (N, 2) arrays on the observation axis, or None
    meta: dict

    def summary(self):
        """Flat record of a series for the G0 outputs, including the matched z-distance of its operating point."""
        op = self.operating_point
        out = {"arm": self.arm, "index": self.index, "stream": self.stream, "level_index": self.level_index,
               "g12": self.gains[0], "g21": self.gains[1], "m": self.m, "p": op["p"],
               "input_sd_factor": op["input_sd_factor"], "noise_share": op["noise_share"], "regime": op["regime"],
               "matched_recording": op["recording"], "z_distance": op["distance"],
               "clean_s": self.meta["clean_s"], "n_segments": len(self.segments)}
        out.update({k: v for k, v in self.meta.items() if k in ("rms_ratio", "c", "null_B", "round")})
        return out


def generate_series(cfg, arm, i, grid, table, stream, round_=0, level_index=None, with_truth=True, artifacts=None):
    """One synthetic series of arm 'positive', 'null_A', 'null_B' or 'artifact_null' (§9.1, §9.2, §5.2); see the module
    docstring. artifacts: None, 'independent' or 'bilateral' (the artifact-only null always uses 'bilateral')."""
    import time
    if arm not in ("positive", "null_A", "null_B", "artifact_null"):
        raise GateError(f"unknown arm {arm!r}")
    if artifacts not in (None, "independent", "bilateral"):
        raise GateError(f"unknown artifact mode {artifacts!r}")
    if arm == "positive" and level_index is None:
        level_index = int(i) % len(cfg["g0"]["coupling_levels_x_C2"])
    g0 = cfg["g0"]
    fs = g0["generation_fs_hz"]
    obs_fs = cfg["preprocessing"]["observation_fs_hz"]
    trim_s = pp.edge_trim_seconds(cfg)
    n_burn = int(round(g0["generation_burn_in_s"] * fs))
    n_keep = int(round((g0["series_duration_s"] + 2 * trim_s) * fs))
    t0 = time.perf_counter()
    op = series_operating_point(cfg, grid, table, stream, i, round_)
    p_vec = np.array([op["p"], op["p"]])
    hw = model.default_half_width(cfg) * op["input_sd_factor"]
    rng_in = series_rng(cfg, stream, arm, "input", i, round_)
    meta = {"round": int(round_)}
    if arm == "null_B":
        u, meta["null_B"] = null_b_input(cfg, rng_in, series_rng(cfg, stream, arm, "common_input", i, round_),
                                         n_burn + n_keep, p_vec, hw)
    else:
        u = model.draw_input(rng_in, n_burn + n_keep, 2, p_vec, cfg, hw)
    g = level_gain(cfg, level_index) if arm == "positive" else 0.0
    c_rel = g0["planted_residual_rms_fraction_of_base"] if arm == "positive" else 0.0
    sim = run_planted(cfg, u, g, g, p_vec, c_rel, n_burn)
    meta["c"] = [float(v) for v in sim["c"]]
    meta["rms_ratio"] = sim["rms_ratio"]
    t_sim = time.perf_counter()
    st = sim["states"]
    y = np.ascontiguousarray(st[1 + n_burn:1 + n_burn + n_keep, :, Y1] - st[1 + n_burn:1 + n_burn + n_keep, :, Y2])
    m = float(series_rng(cfg, stream, arm, "mixing", i, round_).uniform(*g0["mixing_m_range"]))
    z = observe_noisy(cfg, y, fs, m, op["noise_share"], grid.exponent, series_rng(cfg, stream, arm, "noise", i, round_))
    x_uv, factor = scale_to_uv(cfg, z, fs, table.target_sd_uv)
    meta["uv_factor"] = [float(v) for v in factor]
    mode = "bilateral" if arm == "artifact_null" else artifacts
    if mode is not None:
        art, meta["artifacts"] = artifact_signal(cfg, series_rng(cfg, stream, arm, "artifacts", i, round_), x_uv, fs, mode)
        x_uv = x_uv + art
    t_obs = time.perf_counter()
    res = preprocess_series(cfg, x_uv, fs)
    t_pre = time.perf_counter()
    meta["clean_s"] = float(res.meta["b5"]["log"]["clean_s"])
    meta["preprocessing_log"] = res.meta["b5"]["log"]
    truth = None
    if with_truth and arm == "positive":
        n_obs = int(round((y.shape[0] - 2 * int(round(trim_s * fs))) * obs_fs / fs))
        rows, d = observation_rows(cfg, n_obs), sim["delay"]
        pot_all, s_all = sim["pot_all"], sim["s_all"]
        a = model.constants(cfg)["a"]
        u_tgt = pot_all[rows + d, :]
        u_src = pot_all[rows, :][:, ::-1]                    # column j: the OTHER node's potential, one delay earlier
        s_src = s_all[rows, :][:, ::-1]
        basis = (u_src / sim["sd"][::-1][None, :]) * (u_tgt / sim["sd"][None, :])
        truth = {"u_tgt": u_tgt, "u_src": u_src, "s_src": s_src, "basis": basis,
                 "planted": basis * (sim["A"][None, :] * a * sim["c"][None, :])}
    meta["timing_s"] = {"simulate": t_sim - t0, "observe_and_scale": t_obs - t_sim, "preprocess": t_pre - t_obs}
    return Series(arm, int(i), stream, level_index, (g, g), m, op, res.segments, res.starts, truth, meta)


# ---------------------------------------------------------------- E2: the G0 tuning set

def g0_cfg(cfg):
    """Config copy G0 runs with: the Numba backend per g0.backend (the global switch stays as it is)."""
    c = copy.deepcopy(cfg)
    c["ukf"]["numba"]["enabled"] = cfg["g0"]["backend"] == "numba"
    return c


def tuning_set(cfg, grid, table, n=None, round_=0):
    """The G0 tuning set (§7.5): n (20) series generated like the positive control at the mid level, fresh seeds
    (block g0.seeds.tuning), never used for pass/fail."""
    n = cfg["g0"]["n_tuning_series"] if n is None else n
    return [generate_series(cfg, "positive", i, grid, table, "tuning", round_, cfg["g0"]["tuning_level_index"], False)
            for i in range(n)]


def tune_g0_q(cfg, series, filter_name="19D", cache_dir=None, n_jobs=1, min_recordings=None):
    """G0's own q by the section 7.5 NIS rule on the tuning set (tuning.tune_q, the real-data core; filters A and B
    through tune_g0_q_option). Returns the QRResult; the caller stores it under G0's own output, never in
    qr_<seed>.json."""
    from src import tuning
    if filter_name != "19D":
        return tune_g0_q_option(cfg, series, filter_name, cache_dir, n_jobs, min_recordings)
    recs = [{"id": f"tuning_{s.index}", "segments": s.segments, "starts": s.starts} for s in series]
    return tuning.tune_q(recs, g0_cfg(cfg), cache_dir=cache_dir, n_jobs=n_jobs, min_recordings=min_recordings)


# ---------------------------------------------------------------- E2: timing of one series

def time_one_series(cfg, root):
    """Generate one positive series, run the real chain and both UKF passes (19-D, Numba) and print the time per stage.
    Compilation and cache loading are warmed up first and excluded (CLAUDE.md)."""
    import time
    from src import passes
    root = Path(root)
    split = load_split(cfg, root)
    pilot_ids = pp.load_pilot_ids(cfg, root)
    table = build_feature_table(cfg, sorted(pilot_ids), split, make_recording_loader(cfg, root, pilot_ids))
    grid = build_grid(cfg, table.exponent, root / cfg["paths"]["cache_dir"] / cfg["g0"]["cache_subdir"],
                      cfg["compute"]["joblib_n_jobs"])
    gc = g0_cfg(cfg)
    warm = copy.deepcopy(gc)
    warm["g0"]["series_duration_s"] = 12
    w = generate_series(warm, "positive", 0, grid, table, "pilot", with_truth=False)
    passes.run_recording([w.segments[0][:, :20 * 256]], [0], warm, 1e-2)
    t = time.perf_counter()
    s = generate_series(gc, "positive", 2, grid, table, "pilot")
    t_gen = time.perf_counter() - t
    on = passes.run_pass1(s.segments, s.starts, gc, 1e-2, forward_only=True)
    off = copy.deepcopy(gc)                    # state-SD flag disabled in this copy only, as in C4b to C8 (timing)
    off["ukf"]["divergence"]["state_sd_multiple"] = float("inf")
    t = time.perf_counter()
    p1 = passes.run_pass1(s.segments, s.starts, off, 1e-2)
    t_p1 = time.perf_counter() - t
    t = time.perf_counter()
    p2 = passes.run_pass2(s.segments, s.starts, p1.params, off, 1e-2) if p1.params is not None else None
    t_p2 = time.perf_counter() - t
    emit("== E2: one positive series (pilot stream, i = 2, level index 2), 19-D, Numba, q = 1e-2 (timing only) ==")
    emit(f"summary {s.summary()}")
    emit(f"generation {t_gen:.2f} s  (simulate {s.meta['timing_s']['simulate']:.2f}, observe+scale "
         f"{s.meta['timing_s']['observe_and_scale']:.2f}, real preprocessing {s.meta['timing_s']['preprocess']:.2f})")
    emit(f"standard rule ON, forward-only pass 1: diverged {on.recording_diverged} (fraction {on.diverged_fraction:.3f}); "
         f"segments {[(sg.index, sg.diverged, sg.reason, sg.step) for sg in on.segments]}")
    emit(f"state-SD flag OFF (timing copy): pass 1 {t_p1:.2f} s, pass 2 {t_p2:.2f} s, total series "
         f"{t_gen + t_p1 + t_p2:.2f} s")
    if p1.params is not None:
        emit(f"true g = {s.gains[0]:.2f}; smoothed g12 {p1.params.g12:.2f}, g21 {p1.params.g21:.2f}; filtered-gain "
             f"estimate {p1.gain_estimate}; pass 2 windows {len(p2.windows) if p2 is not None else None}")
    return s


# ---------------------------------------------------------------- E3: artifacts of the preprocessing gate (§5.2)

def _event_times(rng, rate_per_s, duration_s):
    """Event times of a Poisson process (exponential inter-arrival times) on [0, duration_s)."""
    times, t = [], 0.0
    while True:
        t += float(rng.exponential(1.0 / rate_per_s))
        if t >= duration_s:
            return times
        times.append(t)


def _add(channel, start, wave):
    """Add wave at sample index start, clipped to the channel; negative starts are cut."""
    n = channel.shape[0]
    a, b = max(start, 0), min(start + wave.shape[0], n)
    if a < b:
        channel[a:b] += wave[a - start:b - start]


def blink_wave(cfg, fs_hz):
    """One blink: raised cosine of blink.duration_s with peak blink.amplitude_uv (uV)."""
    bk = cfg["g0"]["preprocessing_gate"]["blink"]
    n = int(round(bk["duration_s"] * fs_hz))
    return bk["amplitude_uv"] * 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


def emg_carrier(cfg, rng, n, fs_hz):
    """Unit-RMS Gaussian noise band-passed (zero phase) to emg.band_hz, n samples, ramped at both ends."""
    from scipy.signal import butter, sosfiltfilt
    pg = cfg["g0"]["preprocessing_gate"]
    lo, hi = pg["emg"]["band_hz"]
    sos = butter(pg["emg_filter_order"], [lo, hi], btype="band", fs=fs_hz, output="sos")
    pad = int(round(0.5 * fs_hz))
    x = sosfiltfilt(sos, rng.standard_normal(n + 2 * pad))[pad:pad + n]
    x = x / np.sqrt(np.mean(x ** 2))
    ramp = min(int(round(pg["emg_taper_s"] * fs_hz)), n // 2)
    env = np.ones(n)
    env[:ramp] = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))
    env[n - ramp:] = env[:ramp][::-1]
    return x * env


def artifact_signal(cfg, rng, x_uv, fs_hz, mode):
    """The §5.2 artifacts in uV for two channels: blinks (0.3 s, 30 uV, about 1 per 15 s), EMG bursts (20-150 Hz,
    1-3 s, RMS 30% of the channel SD, about 3 per minute) and slow drift (0.05 Hz, 50 uV).
    mode 'independent': events, carriers and phases drawn separately per channel (positive arm).
    mode 'bilateral': ONE set of events, carriers and drift phase hits both channels; the second channel receives
    them lag_steps later (lag ~ U(bilateral_lag_ms)) and each channel has its own amplitude factor
    ~ U(bilateral_amplitude_factor) (artifact-only null). Returns (art (2, n), meta)."""
    if mode not in ("independent", "bilateral"):
        raise GateError(f"unknown artifact mode {mode!r}")
    pg = cfg["g0"]["preprocessing_gate"]
    n = x_uv.shape[1]
    T = n / fs_hz
    sd = x_uv.std(axis=1, ddof=1)
    art = np.zeros((2, n))
    wave = blink_wave(cfg, fs_hz)
    bl, em, dr = pg["blink"], pg["emg"], pg["drift"]
    t = np.arange(n) / fs_hz
    meta = {"mode": mode}
    if mode == "bilateral":
        lag = int(round(rng.uniform(*pg["bilateral_lag_ms"]) * 1e-3 * fs_hz))
        fac = rng.uniform(*pg["bilateral_amplitude_factor"], size=2)
        blinks = _event_times(rng, bl["rate_per_s"], T)
        bursts = [(s0, rng.uniform(*em["duration_s"])) for s0 in _event_times(rng, em["rate_per_min"] / 60.0, T)]
        carriers = [emg_carrier(cfg, rng, int(round(d * fs_hz)), fs_hz) for _, d in bursts]
        phase = rng.uniform(0.0, 2.0 * np.pi)
        for ch in (0, 1):
            shift = lag if ch == 1 else 0
            for t0 in blinks:
                _add(art[ch], int(round(t0 * fs_hz)) + shift, fac[ch] * wave)
            for (s0, _), car in zip(bursts, carriers):
                _add(art[ch], int(round(s0 * fs_hz)) + shift, fac[ch] * em["rms_fraction_of_channel_sd"] * sd[ch] * car)
            art[ch] += fac[ch] * dr["amplitude_uv"] * np.sin(2.0 * np.pi * dr["freq_hz"] * (t - shift / fs_hz) + phase)
        meta.update(lag_steps=lag, factors=[float(v) for v in fac], n_blinks=len(blinks), n_emg=len(bursts))
    else:
        counts = {"n_blinks": 0, "n_emg": 0}
        for ch in (0, 1):
            for t0 in _event_times(rng, bl["rate_per_s"], T):
                _add(art[ch], int(round(t0 * fs_hz)), wave)
                counts["n_blinks"] += 1
            for s0 in _event_times(rng, em["rate_per_min"] / 60.0, T):
                d = rng.uniform(*em["duration_s"])
                car = emg_carrier(cfg, rng, int(round(d * fs_hz)), fs_hz)
                _add(art[ch], int(round(s0 * fs_hz)), em["rms_fraction_of_channel_sd"] * sd[ch] * car)
                counts["n_emg"] += 1
            art[ch] += dr["amplitude_uv"] * np.sin(2.0 * np.pi * dr["freq_hz"] * t + rng.uniform(0.0, 2.0 * np.pi))
        meta.update(counts)
    return art, meta


# ---------------------------------------------------------------- E3: scoring of the preprocessing gate (§5.2)

def null_pass(cfg, g12, g21, has_term):
    """§9.2 null rule, also the artifact-only null's rule (§5.2): no residual term AND both estimated gains strictly
    inside delta of zero. A series without estimates (diverged) is passed as None and fails."""
    if g12 is None or g21 is None:
        return False
    d = delta(cfg)
    return (not has_term) and abs(g12) < d and abs(g21) < d


def _signed_error(est, truth):
    """Signed relative error (est - truth) / truth; a missing estimate (diverged series) counts as +inf (IMP-066)."""
    return float("inf") if est is None else float((est - truth) / truth)


def preprocessing_bias_verdict(cfg, records):
    """Positive arm of the preprocessing gate (§5.2): the SIGNED median relative bias of the coupling gains (g12 and g21
    pooled) and of the E/I ratio rho = A / B (both nodes pooled) against the truth, over the series from level
    g0.preprocessing_gate.bias_from_level_index up, must lie within +-bias_tolerance; p is excluded (only meaningful
    against the fixed rescaling reference). Each record: level_index, g_true, g12, g21, rho1, rho2 (None when the
    series diverged) and optionally nrmse. If every eligible record carries an nrmse the median must also be
    <= g0.pass.median_nrmse_max, else the verdict is None (the PySR part is pending).
    E/I truth is the LITERATURE value A / B (the generator does not vary it), so shrinkage of the estimate toward the
    prior mean, which is the same value, is invisible to this check (reported in the verdict)."""
    pg = cfg["g0"]["preprocessing_gate"]
    tol = pg["bias_tolerance"]
    rho_true = cfg["jansen_rit"]["A"] / cfg["jansen_rit"]["B"]
    elig = [r for r in records if r["level_index"] >= pg["bias_from_level_index"]]
    if not elig:
        raise GateError("no eligible series for the preprocessing bias")
    g_err = [e for r in elig for e in (_signed_error(r["g12"], r["g_true"]), _signed_error(r["g21"], r["g_true"]))]
    ei_err = [e for r in elig for e in (_signed_error(r["rho1"], rho_true), _signed_error(r["rho2"], rho_true))]
    gain_bias, ei_bias = float(np.median(g_err)), float(np.median(ei_err))
    out = {"n_series": len(elig), "n_diverged": int(sum(r["g12"] is None for r in elig)),
           "gain_bias": gain_bias, "ei_bias": ei_bias, "tolerance": tol,
           "gain_ok": abs(gain_bias) <= tol, "ei_ok": abs(ei_bias) <= tol, "p_excluded": True,
           "rho_true": rho_true,
           "ei_note": "E/I truth is the literature value A/B (not varied by the generator); shrinkage toward the "
                      "prior, whose mean is the same value, is invisible to this check",
           "per_level": {}}
    for lv in sorted({r["level_index"] for r in elig}):
        rs = [r for r in elig if r["level_index"] == lv]
        out["per_level"][int(lv)] = {
            "n": len(rs),
            "gain_bias": float(np.median([e for r in rs for e in (_signed_error(r["g12"], r["g_true"]),
                                                                    _signed_error(r["g21"], r["g_true"]))])),
            "ei_bias": float(np.median([e for r in rs for e in (_signed_error(r["rho1"], rho_true),
                                                                  _signed_error(r["rho2"], rho_true))]))}
    nr = [r.get("nrmse") for r in elig]
    if all(v is not None for v in nr):
        out["nrmse_median"] = float(np.median(nr))
        out["nrmse_ok"] = out["nrmse_median"] <= cfg["g0"]["pass"]["median_nrmse_max"]
        out["pass"] = bool(out["gain_ok"] and out["ei_ok"] and out["nrmse_ok"])
    else:
        out["nrmse_median"], out["nrmse_ok"] = None, None
        out["pass"] = False if not (out["gain_ok"] and out["ei_ok"]) else None
    return out


def preprocessing_gate_set(cfg, grid, table, round_=0):
    """The 20 positive and 20 artifact-only null series of the preprocessing gate (§9.4): positive series carry the
    §5.2 artifacts per g0.preprocessing_gate.positive_artifact_mode (5 per level), the null series have zero coupling and
    bilateral near-zero-lag artifacts. Seeds from block g0.seeds.preprocessing_gate."""
    mode = cfg["g0"]["preprocessing_gate"]["positive_artifact_mode"]
    pos = [generate_series(cfg, "positive", i, grid, table, "preprocessing_gate", round_, artifacts=mode)
           for i in range(cfg["g0"]["n_preprocessing_gate_positive"])]
    nul = [generate_series(cfg, "artifact_null", i, grid, table, "preprocessing_gate", round_)
           for i in range(cfg["g0"]["n_preprocessing_gate_null"])]
    return pos, nul


def time_gate_series(cfg, root):
    """Generate one gate positive and one artifact-only null series, report what the real preprocessing did to the
    artifacts, and time both UKF passes (19-D, Numba; the standard rule ON and, for the timing only, the state-SD
    flag disabled in a copy)."""
    import time
    from src import passes
    root = Path(root)
    split = load_split(cfg, root)
    pilot_ids = pp.load_pilot_ids(cfg, root)
    table = build_feature_table(cfg, sorted(pilot_ids), split, make_recording_loader(cfg, root, pilot_ids))
    grid = build_grid(cfg, table.exponent, root / cfg["paths"]["cache_dir"] / cfg["g0"]["cache_subdir"],
                      cfg["compute"]["joblib_n_jobs"])
    gc = g0_cfg(cfg)
    warm = copy.deepcopy(gc)
    warm["g0"]["series_duration_s"] = 12
    w = generate_series(warm, "positive", 0, grid, table, "pilot", with_truth=False)
    passes.run_recording([w.segments[0][:, :20 * 256]], [0], warm, 1e-2)
    off = copy.deepcopy(gc)
    off["ukf"]["divergence"]["state_sd_multiple"] = float("inf")
    mode = cfg["g0"]["preprocessing_gate"]["positive_artifact_mode"]
    emit("== E3: one preprocessing-gate positive and one artifact-only null series (19-D, Numba, q = 1e-2, timing only) ==")
    for label, arm, art in (("positive (i = 2, L3)", "positive", mode), ("artifact-only null (i = 0)", "artifact_null", None)):
        t = time.perf_counter()
        s = generate_series(gc, arm, 2 if arm == "positive" else 0, grid, table, "preprocessing_gate", artifacts=art)
        t_gen = time.perf_counter() - t
        lg = s.meta["preprocessing_log"]
        emit(f"-- {label}: artifacts {s.meta.get('artifacts')}")
        emit(f"   real preprocessing: blinks {lg['blinks']}; rejected segments {lg['rejected_segments']} of {lg['n_segments']} "
             f"({lg['rejected_before_padding_s']:.0f} s, {lg['rejected_after_padding_s']:.0f} s after padding); rules "
             f"{ {k: v['n_segments'] for k, v in lg['rules'].items()} }; clean {lg['clean_s']:.0f} s in {lg['n_clean_segments']} segments")
        emit(f"   summary {s.summary()}")
        on = passes.run_pass1(s.segments, s.starts, gc, 1e-2, forward_only=True)
        t = time.perf_counter()
        p1 = passes.run_pass1(s.segments, s.starts, off, 1e-2)
        t_p1 = time.perf_counter() - t
        t = time.perf_counter()
        p2 = passes.run_pass2(s.segments, s.starts, p1.params, off, 1e-2) if p1.params is not None else None
        t_p2 = time.perf_counter() - t
        emit(f"   generation {t_gen:.2f} s (simulate {s.meta['timing_s']['simulate']:.2f}, observe+scale+artifacts "
             f"{s.meta['timing_s']['observe_and_scale']:.2f}, real preprocessing {s.meta['timing_s']['preprocess']:.2f}); "
             f"flag-off pass 1 {t_p1:.2f} s, pass 2 {t_p2:.2f} s, total {t_gen + t_p1 + t_p2:.2f} s")
        emit(f"   standard rule ON: recording diverged {on.recording_diverged} (fraction {on.diverged_fraction:.3f}), "
             f"segments diverged {sum(sg.diverged for sg in on.segments)} of {len(on.segments)}")
        if p1.params is not None:
            emit(f"   DIAGNOSTIC ONLY (flag off): smoothed g12 {p1.params.g12:.2f}, g21 {p1.params.g21:.2f} "
                 f"(truth {s.gains[0]:.2f}); rho {p1.params.rho1:.3f}, {p1.params.rho2:.3f} "
                 f"(truth {cfg['jansen_rit']['A'] / cfg['jansen_rit']['B']:.3f}); pass 2 windows {len(p2.windows) if p2 else None}")


# ---------------------------------------------------------------- E4: the filter option (DEV-005)

FILTER_KINDS = {"19D": "N", "A": "A", "B": "B"}      # option name -> ukf_ext kind ("N" = no extra state = the 19-D filter)


def gate_filters(cfg, pilot):
    """The filter options a G0 run covers. Pilot mode: every option of g0.filter_options, reported side by side. A full
    run needs the decided g0.filter and REFUSES while it is unset (DEV-005 decides it; never hard-coded here)."""
    options = list(cfg["g0"]["filter_options"])
    if pilot:
        return options
    f = cfg["g0"]["filter"]
    if f is None:
        raise GateError("g0.filter is unset (the DEV-005 decision): the full G0 run refuses to start")
    if f not in options:
        raise GateError(f"g0.filter {f!r} is not one of {options}")
    return [f]


def filter_context(cfg, filter_name, segments):
    """Context in which passes.run_pass1 / run_pass2 (unchanged src code) run the chosen filter: nothing for '19D',
    the extended filter with this recording's own (s2, tau) for 'A' and 'B' (pass 2 then runs 14-D windows)."""
    import contextlib
    from src import ukf_ext
    if filter_name not in FILTER_KINDS:
        raise GateError(f"unknown filter option {filter_name!r}")
    if filter_name == "19D":
        return contextlib.nullcontext()
    return ukf_ext.patched_filters(ukf_ext.spec_for(FILTER_KINDS[filter_name], segments, cfg))


def prior_sds(cfg):
    """Prior SD of each estimated quantity of §7.4 (for the §9.3 contraction)."""
    from src import state_space as ss
    out = dict(ss.parameter_prior_sd(cfg))
    out["m"] = cfg["priors"]["m_sd"]
    return out


def stability_summary(monitors):
    """Aggregate of UKF monitors (§9.3): negative-eigenvalue steps, NaN/Inf, covariance-not-PD divergences, jitter
    fallbacks (reported only) and the smallest eigenvalue seen."""
    mins = [m["min_eig_overall"] for m in monitors if m.get("min_eig_overall") is not None and np.isfinite(m["min_eig_overall"])]
    return {"n_runs": len(monitors),
            "n_negative_eig_steps": int(sum(m["n_negative_eig_steps"] for m in monitors)),
            "n_nan_inf": int(sum(bool(m["nan_inf_seen"]) for m in monitors)),
            "n_linalg_divergences": int(sum(str(m.get("divergence_reason") or "").startswith("linalg_error") for m in monitors)),
            "n_jitter_fallbacks": int(sum(m["n_jitter_fallbacks"] for m in monitors)),
            "min_eig_overall": float(min(mins)) if mins else None}


def stability_ok(summary):
    return summary["n_negative_eig_steps"] == 0 and summary["n_nan_inf"] == 0 and summary["n_linalg_divergences"] == 0


def _estimates(params, gain_estimate):
    if params is None:
        return None
    out = {"g12": params.g12, "g21": params.g21, "m": params.m, "p1": params.p1, "p2": params.p2,
           "rho1": params.rho1, "rho2": params.rho2, "log_rho1": params.log_rho1, "log_rho2": params.log_rho2,
           "posterior_sd": dict(params.posterior_sd)}
    if gain_estimate is not None:
        out["g12_filt"], out["g21_filt"] = gain_estimate["g12"], gain_estimate["g21"]
    return out


def evaluate_series(cfg, series, filter_name, q, want_pass2=True, diagnostic=True):
    """One series through the chosen filter: pass 1 and pass 2 with the standard divergence rule ON (the estimates of
    the verdicts exist only for a recording that is not diverged; a diverged series is a failed series, IMP-069), and
    with diagnostic=True a second pass 1 with the state-SD flag disabled in a config copy, stored under
    'diagnostic_flag_off' and labelled DIAGNOSTIC ONLY (IMP-066). Returns (record, pass2 or None)."""
    from src import passes
    gc = g0_cfg(cfg)
    with filter_context(gc, filter_name, series.segments):
        p1 = passes.run_pass1(series.segments, series.starts, gc, q)
        p2 = None
        if want_pass2 and p1.params is not None and not p1.recording_diverged:
            skip = {s.index for s in p1.segments if s.diverged}
            p2 = passes.run_pass2(series.segments, series.starts, p1.params, gc, q, skip_segments=skip)
    diverged = bool(p1.recording_diverged or (p2 is not None and p2.recording_diverged))
    est = None if diverged else _estimates(p1.params, p1.gain_estimate)
    monitors = [s.monitor for s in p1.segments] + ([w.monitor for w in p2.windows] if p2 is not None else [])
    rec = {"arm": series.arm, "index": series.index, "stream": series.stream, "filter": filter_name, "q": float(q),
           "level_index": series.level_index, "g_true": series.gains[0], "z_distance": series.operating_point["distance"],
           "regime": series.operating_point["regime"], "diverged": diverged,
           "diverged_pass1": bool(p1.recording_diverged), "diverged_fraction_pass1": float(p1.diverged_fraction),
           "n_segments": len(p1.segments), "n_segments_dropped_pass1": int(sum(s.diverged for s in p1.segments)),
           "estimates": est, "stability": stability_summary(monitors), "has_term": None, "nrmse": None,
           "linear_floor": None if series.truth is None else linear_floor_nrmse(series.truth)}
    if diagnostic:
        off = copy.deepcopy(gc)
        off["ukf"]["divergence"]["state_sd_multiple"] = float("inf")
        with filter_context(off, filter_name, series.segments):
            d1 = passes.run_pass1(series.segments, series.starts, off, q)
        rec["diagnostic_flag_off"] = {"diagnostic_only": True, "estimates": _estimates(d1.params, d1.gain_estimate),
                                      "recording_diverged": bool(d1.recording_diverged)}
    return rec, p2


# ---------------------------------------------------------------- E4: tuning with any filter option

def _tune_worker(payload):
    """Top-level (Windows spawn): forward-only NIS run of one (recording, q) through the chosen filter."""
    from threadpoolctl import threadpool_limits
    from src import tuning
    segments, starts, cfg, q, filter_name = payload
    with threadpool_limits(limits=1):
        with filter_context(cfg, filter_name, segments):
            return tuning.recording_nis(segments, starts, cfg, q)


def tune_g0_q_option(cfg, series, filter_name, cache_dir=None, n_jobs=1, min_recordings=None):
    """G0's own q for filter A or B: the same NIS rule (tuning.select_q) over the same grid, with the gate's own worker
    because tuning.tune_q's worker cannot see the patched filter. The cache key carries the filter name and the source
    of ukf_ext. The rule is applied literally; at_grid_edge and in_band are reported (C7: mean NIS of A and B is far
    below 2)."""
    from src import tuning, ukf_ext
    gc = g0_cfg(cfg)
    grid = tuning.q_grid(gc)
    ext_hash = hashlib.sha256((_SOURCE.parent / "ukf_ext.py").read_bytes()).hexdigest()
    recs = [{"id": f"tuning_{s.index}", "segments": s.segments, "starts": s.starts,
             "key": tuning.arrays_key(s.segments, s.starts) + f"|{filter_name}|{ext_hash}"} for s in series]
    entries = [[None] * len(grid) for _ in recs]
    todo = []
    for i, r in enumerate(recs):
        for j, qv in enumerate(grid):
            ck = tuning.cache_key(r["key"], qv, gc)
            hit = tuning._cache_read(cache_dir, ck)
            if hit is not None:
                entries[i][j] = hit
            else:
                todo.append((i, j, ck))
    payloads = [(recs[i]["segments"], recs[i]["starts"], gc, float(grid[j]), filter_name) for i, j, _ in todo]
    if n_jobs > 1 and len(payloads) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend=gc["compute"]["joblib_backend"])(delayed(_tune_worker)(p) for p in payloads)
    else:
        results = [_tune_worker(p) for p in payloads]
    for (i, j, ck), entry in zip(todo, results):
        entries[i][j] = entry
        tuning._cache_write(cache_dir, ck, entry)
    return tuning.select_q(entries, [r["id"] for r in recs], [r["key"] for r in recs], grid, gc, min_recordings)


# ---------------------------------------------------------------- E4: truth-based measures

def linear_floor_nrmse(truth):
    """NRMSE of the BEST LINEAR approximation of the planted term (OLS on 1, u_tgt, u_src, S_src, both nodes pooled,
    node 0 first) on the true inputs: RMS(residual) / SD(planted). It is what an equation without the product could
    reach; reported next to every NRMSE (PLAN Stage E note, IMP-061)."""
    X = np.concatenate([np.column_stack([np.ones(len(truth["u_tgt"])), truth["u_tgt"][:, j], truth["u_src"][:, j],
                                         truth["s_src"][:, j]]) for j in (0, 1)])
    y = np.concatenate([truth["planted"][:, 0], truth["planted"][:, 1]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sqrt(np.mean((X @ coef - y) ** 2)) / np.std(y))


def equation_nrmse(equation, zscore, truth):
    """§9.1 NRMSE of a selected equation: the recovered function (the equation on the z-scored true inputs, returned to
    dy4/dt units with the stored target mean and SD) and the planted residual, both on the recording's TRUE simulated
    input trajectories, RMS difference divided by the SD of the planted residual."""
    from src import regression as R
    X_raw = np.concatenate([np.column_stack([truth["u_tgt"][:, j], truth["u_src"][:, j], truth["s_src"][:, j]]) for j in (0, 1)])
    y = np.concatenate([truth["planted"][:, 0], truth["planted"][:, 1]])
    X, _ = R.design(SimpleNamespace(X_raw=X_raw, y=y), zscore)
    yhat = R.evaluate_equation(equation, X) * zscore.sd_y + zscore.mean_y
    return float(np.sqrt(np.mean((yhat - y) ** 2)) / np.std(y))


# ---------------------------------------------------------------- E4: scoring (§9.1 to §9.3, §15.1)

def pooled_gain_errors(rec, key_pair=("g12", "g21"), source="estimates"):
    """Absolute relative gain errors of one record, g12 and g21 pooled; inf for a diverged series (IMP-069)."""
    est = rec[source] if source in rec else None
    if rec.get("diverged") or est is None:
        return [float("inf"), float("inf")]
    return [abs(est[k] - rec["g_true"]) / rec["g_true"] for k in key_pair]


def positive_verdict(cfg, records):
    """§9.1 positive control per level: median pooled gain error <= 15% and median NRMSE <= 0.25 at every level from the
    second-weakest up (g0.pass.from_level_index); the weakest level is reported as the detection floor. A series without
    estimates counts as inf; an NRMSE not yet available makes the level (and the verdict) None unless the gain
    criterion already failed. Records carry level_index, g_true, diverged, estimates, nrmse, linear_floor."""
    pc = cfg["g0"]["pass"]
    frm = pc["from_level_index"]
    levels = {}
    for lv in sorted({r["level_index"] for r in records}):
        rs = [r for r in records if r["level_index"] == lv]
        gain = float(np.median([e for r in rs for e in pooled_gain_errors(r)]))
        gain_f = float(np.median([e for r in rs for e in pooled_gain_errors(r, ("g12_filt", "g21_filt"))]))
        nr = [float("inf") if r["diverged"] else r["nrmse"] for r in rs]
        nr_known = all(v is not None for v in nr)
        nrmse = float(np.median(nr)) if nr_known else None
        floors = [r["linear_floor"] for r in rs if r.get("linear_floor") is not None]
        gain_ok = gain <= pc["median_gain_rel_error_max"]
        nr_ok = None if nrmse is None else nrmse <= pc["median_nrmse_max"]
        required = lv >= frm
        ok = (False if not gain_ok else nr_ok) if required else None
        levels[int(lv)] = {"n": len(rs), "n_diverged": int(sum(r["diverged"] for r in rs)), "required": required,
                           "gain_error_median": gain, "gain_error_median_filtered": gain_f,
                           "nrmse_median": nrmse, "linear_floor_median": float(np.median(floors)) if floors else None,
                           "gain_ok": gain_ok, "nrmse_ok": nr_ok, "ok": ok, "detection_floor": lv < frm}
    req = [v["ok"] for v in levels.values() if v["required"]]
    passed = False if any(v is False for v in req) else (None if any(v is None for v in req) else True)
    return {"pass": passed, "levels": levels, "from_level_index": frm}


def upper_bound_95(n_fail, n):
    """One-sided 95% Clopper-Pearson upper bound of a failure rate: 1 - 0.05^(1/n) for no failure (about 3/n)."""
    from scipy.stats import beta
    if n <= 0:
        raise GateError("no series")
    return 1.0 if n_fail >= n else float(beta.ppf(1.0 - 0.05, n_fail + 1, n - n_fail))


def null_arm_verdict(cfg, records):
    """§9.2 null arm: a series FAILS if it diverged (IMP-069), if either filtered-gain estimate |g| >= delta, or if a
    residual term was selected (has_term True); has_term None (PySR pending) leaves a series undecided. Pass = no
    failure and none undecided; the pilot reports the 95% upper bound (about 14% at 20 series), the full G0 requires
    0 of 60. Records carry diverged, estimates (g12_filt, g21_filt) and has_term."""
    d = delta(cfg)
    fails = pend = 0
    rows = []
    for r in records:
        est = None if r["diverged"] else r["estimates"]
        if est is None or "g12_filt" not in est:
            status = "fail"
        else:
            inside = abs(est["g12_filt"]) < d and abs(est["g21_filt"]) < d
            status = "fail" if not inside or r["has_term"] is True else ("pending" if r["has_term"] is None else "pass")
        fails += status == "fail"
        pend += status == "pending"
        rows.append(status)
    n = len(records)
    allowed = cfg["g0"]["pass"]["null_false_positives_allowed"]
    passed = False if fails > allowed else (None if pend else True)
    return {"pass": passed, "n": n, "n_fail": int(fails), "n_pending": int(pend), "delta": d,
            "upper_bound_95": upper_bound_95(fails, n), "status": rows}


def contraction_verdict(cfg, records):
    """§9.3: posterior contraction 1 - posterior SD / prior SD of every estimated quantity (p1, p2, log rho 1 and 2, g12,
    g21, m), the MEDIAN over a level's series (a diverged series contributes 0), at every level from the second-weakest
    up, must be >= g0.pass.contraction_min. Failure of g12, g21 or m is 'not identifiable at this level' (never dropped,
    §7.4); failure of another quantity triggers the next step of the §7.4 reduction order, which this function names but
    does not apply."""
    from src import state_space as ss
    prior = prior_sds(cfg)
    lo = cfg["g0"]["pass"]["contraction_min"]
    frm = cfg["g0"]["pass"]["from_level_index"]
    levels = sorted({r["level_index"] for r in records if r["level_index"] >= frm})
    table = {}
    for name in ss.PARAM_NAMES_FULL:
        per = {}
        for lv in levels:
            vals = []
            for r in (x for x in records if x["level_index"] == lv):
                est = None if r["diverged"] else r["estimates"]
                sd = None if est is None else est["posterior_sd"].get(name)
                vals.append(0.0 if sd is None else 1.0 - sd / prior[name])
            per[int(lv)] = float(np.median(vals))
        table[name] = {"per_level": per, "ok": all(v >= lo for v in per.values())}
    failed = [k for k, v in table.items() if not v["ok"]]
    never_drop = set(cfg["state"]["never_drop"])
    not_identifiable = [k for k in failed if k in never_drop]
    reducible = [k for k in failed if k not in never_drop]
    return {"pass": not failed, "table": table, "failed": failed, "not_identifiable": not_identifiable,
            "reduction_required": next_reduction_step(cfg) if reducible else None, "contraction_min": lo,
            "m_contraction": {int(lv): table["m"]["per_level"][int(lv)] for lv in levels}}


def next_reduction_step(cfg):
    """The next unapplied step of the preregistered §7.4 reduction order, or None if all are applied."""
    sw = cfg["state"]["reduction_switches"]
    for step in cfg["state"]["reduction_order"]:
        if not sw[step]:
            return step
    return None


def stability_verdict(records):
    """§9.3 numerical-stability gate over every run of the positive control (pass 1 and pass 2 monitors)."""
    tot = {"n_runs": 0, "n_negative_eig_steps": 0, "n_nan_inf": 0, "n_linalg_divergences": 0, "n_jitter_fallbacks": 0}
    mins = []
    for r in records:
        s = r["stability"]
        for k in tot:
            tot[k] += s[k]
        if s["min_eig_overall"] is not None:
            mins.append(s["min_eig_overall"])
    tot["min_eig_overall"] = min(mins) if mins else None
    return {"pass": stability_ok(tot), **tot}


HARD_STOP_VERDICTS = ("null_A", "null_B", "preproc_null", "stability")
LOW_CONFIDENCE_VERDICTS = ("positive", "contraction", "preproc_bias")


def gate_flags(pilot, verdicts):
    """§9.4 / §18.1 flags from the verdicts (True, False or None = pending): a failed null (A, B or artifact-only) or
    stability verdict is a hard stop, a failed positive-control, contraction or preprocessing-bias verdict sets
    low_confidence. Pilot mode never stops (would_hard_stop still records it)."""
    would = any(verdicts.get(k) is False for k in HARD_STOP_VERDICTS)
    low = any(verdicts.get(k) is False for k in LOW_CONFIDENCE_VERDICTS)
    pending = sorted(k for k, v in verdicts.items() if v is None)
    return {"would_hard_stop": bool(would), "hard_stop": bool(would and not pilot), "low_confidence": bool(low),
            "pending": pending, "complete": not pending}


def gain_profile(cfg, records):
    """The §9.3 gain profile as a table per level (all levels, the weakest included): truth, median and IQR of the
    recording-level estimate (g12 and g21 pooled), the filtered-gain estimate, the median posterior SD and contraction
    of g12 / g21; diverged series are counted and left out of the estimates."""
    prior = prior_sds(cfg)
    out = {}
    for lv in sorted({r["level_index"] for r in records}):
        rs = [r for r in records if r["level_index"] == lv]
        ok = [r for r in rs if not r["diverged"] and r["estimates"] is not None]
        est = [r["estimates"][k] for r in ok for k in ("g12", "g21")]
        flt = [r["estimates"][k] for r in ok for k in ("g12_filt", "g21_filt") if k in r["estimates"]]
        sds = [r["estimates"]["posterior_sd"][k] for r in ok for k in ("g12", "g21")]
        q = lambda v, p: float(np.percentile(v, p)) if v else None          # noqa: E731
        out[int(lv)] = {"g_true": rs[0]["g_true"], "n": len(rs), "n_diverged": len(rs) - len(ok),
                        "g_median": q(est, 50), "g_q25": q(est, 25), "g_q75": q(est, 75),
                        "g_filtered_median": q(flt, 50), "posterior_sd_median": q(sds, 50),
                        "contraction_median": (None if not sds else float(1.0 - np.median(sds) / prior["g12"]))}
    return out


def choose_parsimony(cfg, nrmse_by_penalty):
    """§9.1: the penalty of the grid with the lowest MEDIAN NRMSE over the pilot positive series; ties go to the larger
    penalty. nrmse_by_penalty maps a penalty to one NRMSE per series (inf for a failed series)."""
    grid = list(cfg["pysr"]["parsimony_grid"])
    meds = {p: float(np.median(nrmse_by_penalty[p])) for p in grid}
    best = min(meds.values())
    return {"penalty": max(p for p in grid if meds[p] == best), "median_nrmse": {str(p): meds[p] for p in grid}}


def option_report(cfg, filter_name, records_by_arm):
    """Per-option report fields of the DEV-005 comparison (IMP-066): per arm and level the number of series dropped by
    the standard divergence rule, per series the rule-ON estimates and the flag-off estimates side by side (the latter
    DIAGNOSTIC ONLY, never a verdict), the matched z-distance (median, maximum, per series) and the linear-floor NRMSE
    per level."""
    out = {"filter": filter_name, "arms": {}}
    for arm, recs in records_by_arm.items():
        levels = {}
        for lv in sorted({r["level_index"] for r in recs}, key=lambda v: (v is None, v)):
            rs = [r for r in recs if r["level_index"] == lv]
            fl = [r["linear_floor"] for r in rs if r.get("linear_floor") is not None]
            levels[str(lv)] = {"n": len(rs), "n_dropped_by_standard_rule": int(sum(r["diverged"] for r in rs)),
                               "linear_floor_nrmse_median": float(np.median(fl)) if fl else None}
        zs = [r["z_distance"] for r in recs]
        out["arms"][arm] = {
            "n": len(recs), "n_dropped_by_standard_rule": int(sum(r["diverged"] for r in recs)), "levels": levels,
            "z_distance": {"median": float(np.median(zs)), "max": float(np.max(zs)), "per_series": [float(z) for z in zs]},
            "side_by_side": [{"index": r["index"], "level_index": r["level_index"], "g_true": r["g_true"],
                              "rule_on": None if r["estimates"] is None or r["diverged"] else
                              {k: r["estimates"].get(k) for k in ("g12", "g21", "g12_filt", "g21_filt")},
                              "diagnostic_flag_off": r.get("diagnostic_flag_off")} for r in recs],
            "diagnostic_note": "diagnostic_flag_off estimates use a config copy with the state-SD flag disabled; they are "
                               "never a G0 verdict"}
    return out


# ---------------------------------------------------------------- E4: gate.json

def _jsonable(obj):
    """JSON-safe copy: numpy scalars and arrays to Python, tuples to lists, non-finite floats to 'inf' / '-inf' / 'nan'."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else ("nan" if np.isnan(v) else ("inf" if v > 0 else "-inf"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def gate_path(cfg, root, pilot, filter_name=None):
    """outputs/gate.json for a full run; results/pilot/gate_<filter>.json for the pilot (never outputs/gate.json)."""
    root = Path(root)
    if not pilot:
        return root / cfg["paths"]["gate_file"]
    if filter_name is None:
        raise GateError("a pilot gate file is per filter option")
    return root / cfg["paths"]["pilot_results_dir"] / f"gate_{filter_name}.json"


def build_gate_document(cfg, root, *, pilot, filter_name, verdicts, sections, q=None, parsimony=None, seeds=None):
    """The gate.json document: schema_version, pilot, filter, hard_stop, low_confidence, would_hard_stop, pending, delta,
    reasons, the verdicts, the per-section tables (positive, null_A, null_B, contraction, stability, preprocessing,
    gain_profile, regime, report), G0's own q, the parsimony penalty, the seeds and the provenance. tuning.check_gate
    reads hard_stop and low_confidence."""
    from src import tuning
    flags = gate_flags(pilot, verdicts)
    reasons = [f"{k} failed" for k in HARD_STOP_VERDICTS + LOW_CONFIDENCE_VERDICTS if verdicts.get(k) is False]
    head, dirty = tuning._git_state(Path(root))
    doc = {"schema_version": cfg["g0"]["gate_schema_version"], "pilot": bool(pilot), "filter": filter_name,
           "hard_stop": flags["hard_stop"], "low_confidence": flags["low_confidence"],
           "would_hard_stop": flags["would_hard_stop"], "pending": flags["pending"], "complete": flags["complete"],
           "reasons": reasons, "delta": delta(cfg), "verdicts": dict(verdicts), "q": q, "parsimony": parsimony,
           "seeds": seeds if seeds is not None else dict(cfg["g0"]["seeds"]),
           "provenance": {"git_commit": head, "git_dirty": dirty,
                          "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
                          "code_sha256": hashlib.sha256(_SOURCE.read_bytes()).hexdigest()}}
    doc.update(sections)
    return doc


def write_gate(cfg, root, doc, path):
    """Write the document atomically. A pilot document is never written to the full gate path and a full document
    never into the pilot folder (the pilot cannot unlock real fitting, CLAUDE.md rule 6)."""
    path, full = Path(path), Path(root) / cfg["paths"]["gate_file"]
    pilot_dir = Path(root) / cfg["paths"]["pilot_results_dir"]
    if doc["pilot"] and path.resolve() == full.resolve():
        raise GateError("a pilot gate document must not be written to outputs/gate.json")
    if not doc["pilot"] and (path.resolve() != full.resolve() or pilot_dir.resolve() in path.resolve().parents):
        raise GateError("a full gate document is written to outputs/gate.json only")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(doc), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------- the E1 report

def grid_report(cfg, root, pilot, n_jobs=None):
    """Build the pilot feature table and the grid, print the grid and regime report. Returns (table, grid)."""
    if not pilot:
        raise GateError("E1 report is wired for --pilot only (the full run reads all training subjects in phase 1)")
    root = Path(root)
    split = load_split(cfg, root)
    pilot_ids = pp.load_pilot_ids(cfg, root)
    table = build_feature_table(cfg, sorted(pilot_ids), split, make_recording_loader(cfg, root, pilot_ids))
    n_jobs = cfg["compute"]["joblib_n_jobs"] if n_jobs is None else n_jobs
    grid = build_grid(cfg, table.exponent, root / cfg["paths"]["cache_dir"] / cfg["g0"]["cache_subdir"], n_jobs)
    emit("== E1: feature table (pilot subjects, training side) ==")
    emit(f"recordings used {len(table.names)}; skipped {table.skipped}")
    emit(f"median bipolar SD {table.target_sd_uv:.3f} uV (post band-pass, clean samples, over recordings and channels)")
    emit(f"generator 1/f exponent (table median) {table.exponent:.3f}")
    emit(f"{'recording':<18}{'alpha_pk':>9}{'rel_alpha':>10}{'exponent':>9}  nearest grid point (z-dist, p, sdf, share, regime)")
    scale = table.scale
    for name, vec in zip(table.names, table.features):
        i, d = match_grid_point(grid.features, grid.valid, vec, scale)
        g = grid.point(i)
        emit(f"{name:<18}{vec[0]:9.2f}{vec[1]:10.3f}{vec[2]:9.3f}  {d:5.2f}  p={g['p']:.0f} sdf={g['input_sd_factor']:.2f} "
             f"ns={g['noise_share']:.1f} {g['regime']}  grid(pk {grid.features[i, 0]:.2f}, rel {grid.features[i, 1]:.3f}, "
             f"exp {grid.features[i, 2]:.3f})")
    emit("== E1: grid ==")
    emit(f"points {len(grid.features)} (shape {cfg['g0']['regime_grid_shape']}), valid {int(grid.valid.sum())}, "
         f"from cache {grid.from_cache}, key {grid.key[:16]}")
    emit(f"p values {np.round(grid.p, 1).tolist()}")
    emit(f"input-SD factors {np.round(grid.sd_factor, 3).tolist()}; noise shares {np.round(grid.share, 2).tolist()}")
    for k, name in enumerate(FEATURE_NAMES):
        v = grid.features[grid.valid, k]
        t = table.features[:, k]
        emit(f"{name:<22} grid min/median/max {v.min():7.3f} {np.median(v):7.3f} {v.max():7.3f}   "
             f"training min/median/max {t.min():7.3f} {np.median(t):7.3f} {t.max():7.3f}")
    emit("regime by p (noise-free run):")
    labels = {}
    for pv, r in zip(np.repeat(grid.p, len(grid.sd_factor) * len(grid.share)), grid.regime):
        labels[float(pv)] = r
    emit("  " + ", ".join(f"{pv:.0f}:{r}" for pv, r in labels.items()))
    n_lc = sum(r == LIMIT_CYCLE for r in grid.regime)
    emit(f"grid points limit-cycle {n_lc} of {len(grid.regime)}, noise-driven {len(grid.regime) - n_lc}")
    emit("== E1: operating points of the pilot series (stream 'pilot', first 20) ==")
    counts = {LIMIT_CYCLE: 0, NOISE_DRIVEN: 0}
    dists = []
    for i in range(cfg["g0"]["n_pilot_positive"]):
        op = series_operating_point(cfg, grid, table, "pilot", i)
        counts[op["regime"]] += 1
        dists.append(op["distance"])
        emit(f"  series {i:2d}: {op['recording']:<18} -> p={op['p']:.0f} sdf={op['input_sd_factor']:.2f} "
             f"ns={op['noise_share']:.1f} {op['regime']:<12} z-dist {op['distance']:.2f}")
    emit(f"regimes of the 20 pilot series {counts}; z-distance median {np.median(dists):.2f}, max {np.max(dists):.2f}")
    return table, grid


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="G0 synthetic gate (E1: regime grid report)")
    ap.add_argument("--grid-report", action="store_true")
    ap.add_argument("--time-series", action="store_true")
    ap.add_argument("--time-gate-series", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.grid_report:
        grid_report(cfg, REPO_ROOT, args.pilot, args.n_jobs)
        return 0
    if args.time_series:
        if not args.pilot:
            raise GateError("the timing run is wired for --pilot only")
        time_one_series(cfg, REPO_ROOT)
        return 0
    if args.time_gate_series:
        if not args.pilot:
            raise GateError("the timing run is wired for --pilot only")
        time_gate_series(cfg, REPO_ROOT)
        return 0
    ap.error("nothing to do")


if __name__ == "__main__":
    sys.exit(main())
