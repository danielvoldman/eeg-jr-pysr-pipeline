"""Synthetic gate G0 (PLAN.md Stage E; §5.2, §9, §18.1). This file holds E1 so far: the operating-regime grid.

E1 (§9.1, "operating regime matched to real data"): a grid of simulations over (p, input-noise SD, additive
observation-noise level), 12 x 8 x 5 by default, each reduced to three SCALE-FREE features (alpha peak
frequency, relative alpha power, specparam aperiodic exponent). A series draws one feature vector at random
from the TRAINING recordings and takes the nearest grid point; absolute power is never matched because it
depends on the rescaling reference (§5.1). Each grid point also carries a regime label, noise-driven or
limit-cycle, from a noise-free run at its p (reported, §9.1).

Training-only rule (CLAUDE.md rule 6): the feature table is built only from subjects on the training side of
the split (check_training_side) and, in development runs, only from pilot subjects (preprocess guard). Every
random draw uses numpy.random.default_rng with a seed from config g0.seeds (rule 3). Nothing here calls PySR.

Entry point: python -m src.synthetic_gate --grid-report --pilot
"""
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import welch

from src import model
from src import preprocess as pp
from src.config import REPO_ROOT, load_config

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


def series_operating_point(cfg, grid, table, stream, i):
    """The operating point of series i of a seed stream: random training feature vector, nearest grid point.
    stream is a key of g0.seeds ('pilot', 'full', 'tuning', 'preprocessing_gate')."""
    rng = np.random.default_rng([cfg["g0"]["seeds"][stream], 0, int(i)])
    row, vec = draw_feature_vector(table, rng)
    idx, dist = match_grid_point(grid.features, grid.valid, vec, table.scale)
    return {"series": int(i), "recording": table.names[row], "target": [float(v) for v in vec],
            "distance": dist, **grid.point(idx)}


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
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.grid_report:
        grid_report(cfg, REPO_ROOT, args.pilot, args.n_jobs)
        return 0
    ap.error("nothing to do")


if __name__ == "__main__":
    sys.exit(main())
