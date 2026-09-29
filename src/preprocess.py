"""Zero-phase band-pass, FFT downsampling, and the recording pipeline up to the
bipolar pairs (§4.2, §5.1, §6, §10.1; IMP-004, IMP-009).

A3 functions (bandpass, downsample) are pure. B3/B4 add file loading with
checksum verification, the units check, bad-electrode detection, the notch, edge
trimming, bipolar derivation, a per-recording cache, and the pilot report
(`python -m src.preprocess --pilot-report`). Blink correction, segment
rejection, rescaling, downsampling of real data and exclusions are B5/B6.
Never imports src.model (§18).

bandpass() is for whole continuous recordings before segmentation, never for
short segments. It raises for inputs of padlen samples or fewer, by design.

Forward-backward filtering squares the magnitude response, so the nominal
cutoffs in config.yml are the -6 dB points of the combined filter.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample, sosfiltfilt

from src.config import REPO_ROOT, load_config

logger = logging.getLogger("pipeline.preprocess")


class PreprocessError(ValueError):
    """Raised for an input or configuration the filter functions cannot handle."""


class ChecksumError(PreprocessError):
    """A file is missing from the manifest, or its size or SHA-256 differs from it."""


class ChannelsError(PreprocessError):
    """channels.tsv is missing or unusable, or a required electrode label is missing."""


class DevelopmentGuardError(PreprocessError):
    """A development run was asked to read a subject outside the pilot set."""


def _check_signal(x, name):
    x = np.asarray(x)
    if x.dtype != np.float64:
        raise TypeError(f"{name}: input must be float64, got {x.dtype}")
    if x.ndim not in (1, 2):
        raise PreprocessError(f"{name}: input must be 1-D or 2-D, got {x.ndim}-D")
    if not np.all(np.isfinite(x)):
        raise PreprocessError(f"{name}: input contains NaN or Inf")
    return x


def bandpass_sos(cfg, fs_hz, highpass_hz=None):
    """High-pass sections followed by low-pass sections, `filter_order` per pass each."""
    bp = cfg["preprocessing"]["bandpass"]
    if bp["filter_family"] != "butterworth":
        raise PreprocessError(f"unsupported filter_family {bp['filter_family']!r}")
    if not bp["zero_phase"]:
        raise PreprocessError("preprocessing.bandpass.zero_phase must be true (§5.1)")
    hp = bp["highpass_hz"] if highpass_hz is None else highpass_hz
    lp = bp["lowpass_hz"]
    if not 0 < hp < lp < fs_hz * 0.5:
        raise PreprocessError(
            f"need 0 < highpass ({hp}) < lowpass ({lp}) < Nyquist ({fs_hz * 0.5})")
    order = bp["filter_order"]
    sos_hp = butter(order, hp, btype="highpass", fs=fs_hz, output="sos")
    sos_lp = butter(order, lp, btype="lowpass", fs=fs_hz, output="sos")
    return np.vstack([sos_hp, sos_lp])


def bandpass(cfg, x, fs_hz, highpass_hz=None):
    """Zero-phase band-pass of whole continuous recordings; time on the last axis."""
    x = _check_signal(x, "bandpass")
    bp = cfg["preprocessing"]["bandpass"]
    padlen = int(round(bp["edge_pad_s"] * fs_hz))
    if x.shape[-1] <= padlen:
        raise PreprocessError(
            f"bandpass: {x.shape[-1]} samples is not more than the edge padding "
            f"({padlen} samples = {bp['edge_pad_s']} s); use whole recordings, not segments")
    sos = bandpass_sos(cfg, fs_hz, highpass_hz)
    return sosfiltfilt(sos, x, axis=-1, padtype=bp["edge_padtype"], padlen=padlen)


def bandpass_impulse_response(cfg, fs_hz, highpass_hz=None):
    """Impulse response of bandpass() itself: an odd-length array, impulse at index len // 2."""
    bp = cfg["preprocessing"]["bandpass"]
    half = int(round(bp["impulse_response_duration_s"] * fs_hz * 0.5))
    x = np.zeros(2 * half + 1, dtype=np.float64)
    x[half] = 1.0
    return bandpass(cfg, x, fs_hz, highpass_hz)


def downsample(cfg, x, fs_in_hz, fs_out_hz=None):
    """FFT resample by an integer ratio, reflect-padded to soften the periodic-signal assumption."""
    x = _check_signal(x, "downsample")
    if fs_out_hz is None:
        fs_out_hz = cfg["preprocessing"]["observation_fs_hz"]
    ratio = int(round(fs_in_hz / fs_out_hz))
    if ratio < 1 or ratio * fs_out_hz != fs_in_hz:
        raise PreprocessError(
            f"downsample: {fs_in_hz} Hz -> {fs_out_hz} Hz is not an integer ratio >= 1")
    if ratio == 1:
        return x.copy()
    n = x.shape[-1]
    if n % ratio:
        raise PreprocessError(
            f"downsample: length {n} is not divisible by the ratio {ratio}; trim it first")
    rs = cfg["preprocessing"]["resample"]
    pad = ratio * -(-int(round(rs["pad_s"] * fs_in_hz)) // ratio)
    if pad >= n:
        raise PreprocessError(
            f"downsample: {n} samples is not more than the reflect-pad ({pad} samples)")
    widths = [(0, 0)] * (x.ndim - 1) + [(pad, pad)]
    xp = np.pad(x, widths, mode="reflect", reflect_type=rs["reflect_type"])
    y = resample(xp, (n + 2 * pad) // ratio, axis=-1)
    trim = pad // ratio
    return np.ascontiguousarray(y[..., trim:y.shape[-1] - trim], dtype=np.float64)


# ---------------------------------------------------------------- high-pass (IMP-009)

def highpass(cfg, x, fs_hz, highpass_hz=None):
    """High-pass half of the bandpass() design: same order, forward-backward, same edge padding."""
    x = _check_signal(x, "highpass")
    bp = cfg["preprocessing"]["bandpass"]
    if bp["filter_family"] != "butterworth":
        raise PreprocessError(f"unsupported filter_family {bp['filter_family']!r}")
    hp = bp["highpass_hz"] if highpass_hz is None else highpass_hz
    if not 0 < hp < fs_hz * 0.5:
        raise PreprocessError(f"need 0 < highpass ({hp}) < Nyquist ({fs_hz * 0.5})")
    padlen = int(round(bp["edge_pad_s"] * fs_hz))
    if x.shape[-1] <= padlen:
        raise PreprocessError(
            f"highpass: {x.shape[-1]} samples is not more than the edge padding ({padlen} samples)")
    sos = butter(bp["filter_order"], hp, btype="highpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, x, axis=-1, padtype=bp["edge_padtype"], padlen=padlen)


# ---------------------------------------------------------------- manifest and checksum (§5.1 step 1)

def load_manifest(path):
    """{relative posix path: (sha256, size_bytes)} from data/MANIFEST.sha256 (IMP-007 layout)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            sha, size, rel = line.split("\t")
        except ValueError:
            raise ChecksumError(f"{path}: malformed manifest line {line!r}") from None
        if not size.isdigit():
            raise ChecksumError(f"{path}: malformed manifest line {line!r}")
        out[rel] = (sha, int(size))
    return out


def sha256_of_file(path, chunk_bytes):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(cfg, path, rel_path, manifest):
    """Check size, then SHA-256, against the manifest. Returns the SHA-256; raises ChecksumError."""
    if rel_path not in manifest:
        raise ChecksumError(f"{rel_path}: not listed in the manifest")
    want_sha, want_size = manifest[rel_path]
    size = Path(path).stat().st_size
    if size != want_size:
        raise ChecksumError(f"{rel_path}: size {size} differs from the manifest ({want_size})")
    sha = sha256_of_file(path, cfg["download"]["hash_chunk_bytes"])
    if sha != want_sha:
        raise ChecksumError(f"{rel_path}: SHA-256 {sha} differs from the manifest ({want_sha})")
    return sha


# ---------------------------------------------------------------- development guard (IMP-009)

def load_pilot_ids(cfg, root):
    """Pilot subject IDs from outputs/split_<primary_seed>.json (json only; never imports main)."""
    path = Path(root) / cfg["paths"]["split_file_pattern"].format(seed=cfg["split"]["primary_seed"])
    with open(path, "r", encoding="utf-8") as fh:
        return frozenset(json.load(fh)["pilot"])


def assert_dev_subject(subject_id, pilot_ids, allow_all=False):
    """Development runs may only read pilot subjects; only phase 1 passes allow_all=True."""
    if allow_all:
        return
    if pilot_ids is None or subject_id not in pilot_ids:
        raise DevelopmentGuardError(
            f"{subject_id} is not a pilot subject; refusing to read it in a development run")


# ---------------------------------------------------------------- channels.tsv and labels (§6)

def channels_tsv_path(cfg, edf_path):
    ds = cfg["dataset"]
    name = Path(edf_path).name
    if not name.endswith(ds["edf_suffix"]):
        raise ChannelsError(f"{name}: does not end with {ds['edf_suffix']!r}")
    return Path(edf_path).with_name(name[:-len(ds["edf_suffix"])] + ds["channels_suffix"])


def read_channels_tsv(path):
    """[(name, type)] in file order. Missing file, name or type column, or blank type: ChannelsError."""
    path = Path(path)
    if not path.is_file():
        raise ChannelsError(f"{path}: channels.tsv is missing")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "name" not in reader.fieldnames or "type" not in reader.fieldnames:
            raise ChannelsError(f"{path}: needs 'name' and 'type' columns, found {reader.fieldnames}")
        rows = [(r["name"].strip(), (r["type"] or "").strip()) for r in reader]
    if not rows or any(not t for _, t in rows):
        raise ChannelsError(f"{path}: channel types are missing; stop and ask (IMP-009)")
    return rows


def required_labels(cfg):
    ch = cfg["preprocessing"]["channels"]
    return list(ch["left_pair"]) + list(ch["right_pair"])


def verify_labels(cfg, tsv_rows, edf_names):
    """Every required electrode must be an EEG channel in channels.tsv and a name in the EDF."""
    eeg_type = cfg["preprocessing"]["units_check"]["channel_type"]
    tsv = {n: t for n, t in tsv_rows}
    for label in required_labels(cfg):
        if label not in tsv:
            raise ChannelsError(f"electrode label {label!r} is missing from channels.tsv")
        if tsv[label].upper() != eeg_type.upper():
            raise ChannelsError(f"electrode {label!r} has type {tsv[label]!r} in channels.tsv, not {eeg_type}")
        if label not in edf_names:
            raise ChannelsError(f"electrode label {label!r} is missing from the EDF channel names")


# ---------------------------------------------------------------- loading (§4.2, §5.1 step 1)

@dataclass
class Recording:
    data_uv: np.ndarray          # (n_channels, n_samples) float64, microvolts, EEG-type channels only
    ch_names: list
    fs_hz: float
    sha256: str
    rel_path: str
    subject: str
    session: str


def _split_rel(rel_path):
    subject, session, *rest = rel_path.split("/")
    if not rest:
        raise PreprocessError(f"{rel_path}: expected <subject>/<session>/... layout")
    return subject, session


def load_recording(cfg, edf_path, data_root, manifest, pilot_ids, allow_all=False, known_sha256=None):
    """Guard, verify against the manifest, read with units="uV", keep the EEG-type channels.

    sub-010 ses-t1: only its scans.tsv was never downloaded; nothing here reads a scans.tsv.
    """
    edf_path = Path(edf_path)
    rel = edf_path.relative_to(data_root).as_posix()
    subject, session = _split_rel(rel)
    assert_dev_subject(subject, pilot_ids, allow_all)
    sha = known_sha256 if known_sha256 is not None else verify_file(cfg, edf_path, rel, manifest)
    tsv_rows = read_channels_tsv(channels_tsv_path(cfg, edf_path))

    import mne  # lazy: MNE is slow to import and only file reading needs it
    units = cfg["dataset"]["read_units"]
    raw = mne.io.read_raw_edf(edf_path, units=units, preload=True, verbose="ERROR")
    verify_labels(cfg, tsv_rows, raw.ch_names)
    eeg_type = cfg["preprocessing"]["units_check"]["channel_type"]
    names = [n for n, t in tsv_rows if t.upper() == eeg_type.upper()]
    missing = [n for n in names if n not in raw.ch_names]
    if missing:
        raise ChannelsError(f"{rel}: EEG channels in channels.tsv but not in the EDF: {missing}")
    fs = float(raw.info["sfreq"])
    if fs != float(cfg["dataset"]["native_fs_hz"]):
        raise PreprocessError(f"{rel}: sampling rate {fs} Hz, expected {cfg['dataset']['native_fs_hz']} Hz")
    # MNE stores volts internally; get_data(units=...) returns the file's own unit (uV).
    data = np.ascontiguousarray(raw.get_data(picks=names, units=units), dtype=np.float64)
    return Recording(data, names, fs, sha, rel, subject, session)


# ---------------------------------------------------------------- units check and halt rule (§4.2)

@dataclass
class UnitsCheck:
    median_sd_uv: float
    passed: bool


def units_check(cfg, data_uv, fs_hz):
    """Median SD (ddof 1) across channels after a 0.5 Hz high-pass of a copy; pass if within 2 to 200 uV."""
    uc = cfg["preprocessing"]["units_check"]
    filtered = highpass(cfg, np.asarray(data_uv, dtype=np.float64), fs_hz, uc["highpass_hz"])
    med = float(np.median(filtered.std(axis=-1, ddof=1)))
    return UnitsCheck(med, bool(uc["median_sd_min_uv"] <= med <= uc["median_sd_max_uv"]))


@dataclass
class HaltDecision:
    n_failed: int
    n_total: int
    fraction: float
    halt: bool


def units_check_halt(cfg, n_failed, n_total):
    """Halt if MORE than halt_if_fail_fraction_above of the recordings failed (exact arithmetic).

    With 5 percent: 153 recordings halt at 8 failures or more (5 percent of 153 is 7.65).
    """
    if n_total <= 0 or not 0 <= n_failed <= n_total:
        raise PreprocessError(f"need 0 <= n_failed <= n_total > 0, got {n_failed}, {n_total}")
    limit = Fraction(repr(cfg["preprocessing"]["units_check"]["halt_if_fail_fraction_above"]))
    return HaltDecision(n_failed, n_total, n_failed / n_total, Fraction(n_failed, n_total) > limit)


# ---------------------------------------------------------------- notch (§5.1)

def notch_frequencies(cfg, fs_hz):
    ln = cfg["preprocessing"]["line_noise"]
    f0, freqs, k = ln["freq_hz"], [], 1
    while k * f0 < fs_hz * 0.5:
        freqs.append(float(k * f0))
        if not ln["include_harmonics"]:
            break
        k += 1
    return freqs


def notch(cfg, x, fs_hz):
    """Zero-phase IIR notch (iirnotch + filtfilt) at the line frequency and every harmonic below Nyquist."""
    x = _check_signal(x, "notch")
    bp = cfg["preprocessing"]["bandpass"]
    padlen = int(round(bp["edge_pad_s"] * fs_hz))
    if x.shape[-1] <= padlen:
        raise PreprocessError(f"notch: {x.shape[-1]} samples is not more than the edge padding ({padlen})")
    q = cfg["preprocessing"]["line_noise"]["notch_q"]
    y = x
    for f in notch_frequencies(cfg, fs_hz):
        b, a = iirnotch(f, q, fs=fs_hz)
        y = filtfilt(b, a, y, axis=-1, padtype=bp["edge_padtype"], padlen=padlen)
    return y


# ---------------------------------------------------------------- edge trim, bipolar (§5.1, §6)

def edge_trim_seconds(cfg, sensitivity_highpass=False):
    pp = cfg["preprocessing"]
    return pp["edge_trim_s_sensitivity_highpass"] if sensitivity_highpass else pp["edge_trim_s"]


def trim_edges(x, fs_hz, trim_s):
    """Remove exactly round(trim_s * fs) samples from both ends (time on the last axis)."""
    k = int(round(trim_s * fs_hz))
    n = x.shape[-1]
    if n <= 2 * k:
        raise PreprocessError(f"trim_edges: {n} samples cannot lose {k} from each end")
    return x[..., k:n - k]


def bipolar(cfg, electrodes):
    """(2, n): left pair then right pair, first electrode minus second. `electrodes`: name -> 1-D array."""
    ch = cfg["preprocessing"]["channels"]
    return np.stack([electrodes[ch[side][0]] - electrodes[ch[side][1]]
                     for side in ("left_pair", "right_pair")]).astype(np.float64)


# ---------------------------------------------------------------- bad electrodes (§5.1 step 3, IMP-009)

def longest_run(mask):
    """Length of the longest run of True in a 1-D boolean array."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    d = np.diff(m)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max()) if starts.size else 0


def saturation_run(x_raw_uv):
    """Longest run of samples exactly equal to the channel's own maximum, or to its own minimum.

    Exact float equality is intended: equal ADC counts scale to equal floats.
    """
    return max(longest_run(x_raw_uv == x_raw_uv.max()), longest_run(x_raw_uv == x_raw_uv.min()))


def saturation_limit_samples(cfg, fs_hz):
    return int(math.ceil(cfg["preprocessing"]["bad_channel"]["saturation_min_run_s"] * fs_hz))


def flat_segment_count(cfg, x_uv, fs_hz):
    """Number of whole 1-s segments whose peak-to-peak is below the flat-line threshold."""
    seg = int(round(cfg["preprocessing"]["rejection"]["segment_length_s"] * fs_hz))
    n_seg = x_uv.shape[-1] // seg
    if n_seg == 0:
        return 0
    blocks = x_uv[:n_seg * seg].reshape(n_seg, seg)
    ptp = blocks.max(axis=1) - blocks.min(axis=1)
    return int(np.count_nonzero(ptp < cfg["preprocessing"]["bad_channel"]["flatline_threshold"]))


def rms_bounds(cfg, strict=False):
    bc = cfg["preprocessing"]["bad_channel"]
    hi = cfg["preprocessing"]["strict_variant"]["rms_max_uv"] if strict else bc["rms_max_uv"]
    return bc["rms_min_uv"], hi


def electrode_report(cfg, raw_uv, filtered_uv, fs_hz, strict=False):
    """Bad-electrode rules for one electrode.

    raw_uv: the raw scaled-to-uV signal over the whole recording (saturation).
    filtered_uv: notched, band-passed and edge-trimmed (RMS and flat-line).
    """
    lo, hi = rms_bounds(cfg, strict)
    rms = float(np.sqrt(np.mean(filtered_uv ** 2)))
    sat_run = saturation_run(raw_uv)
    sat_limit = saturation_limit_samples(cfg, fs_hz)
    flat = flat_segment_count(cfg, filtered_uv, fs_hz)
    flags = []
    if rms < lo:
        flags.append("rms_below_min")
    if rms > hi:
        flags.append("rms_above_max")
    if flat:
        flags.append("flatline")
    if sat_run >= sat_limit:
        flags.append("saturation")
    return {"rms_uv": rms, "saturation_run": sat_run, "saturation_limit": sat_limit,
            "flat_segments": flat, "flags": flags}


# ---------------------------------------------------------------- recording pipeline (B3 + B4)

@dataclass
class RecordingResult:
    meta: dict
    bipolar: object = field(default=None)   # (2, n) float64, native rate, edge-trimmed; None if excluded


def _fingerprint(cfg):
    pp, ds = cfg["preprocessing"], cfg["dataset"]
    used = {
        "units_check": pp["units_check"], "channels": pp["channels"], "bad_channel": pp["bad_channel"],
        "line_noise": pp["line_noise"], "bandpass": pp["bandpass"], "edge_trim_s": pp["edge_trim_s"],
        "edge_trim_s_sensitivity_highpass": pp["edge_trim_s_sensitivity_highpass"],
        "segment_length_s": pp["rejection"]["segment_length_s"],
        "sensitivity_highpass_hz": pp["sensitivity_highpass_hz"], "strict_variant": pp["strict_variant"],
        "read_units": ds["read_units"], "native_fs_hz": ds["native_fs_hz"],
        "edf_suffix": ds["edf_suffix"], "channels_suffix": ds["channels_suffix"],
    }
    return json.dumps(used, sort_keys=True, default=str)


def cache_key(cfg, sha256, sensitivity_highpass, strict):
    payload = "\n".join([sha256, _fingerprint(cfg), str(bool(sensitivity_highpass)), str(bool(strict))])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cfg, cache_root, rel_path, key):
    subject, session = _split_rel(rel_path)
    return Path(cache_root) / cfg["preprocessing"]["cache"]["subdir"] / f"{subject}_{session}_{key}.npz"


def _cache_load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        bip = z["bipolar"] if meta["bipolar_stored"] else None
    return RecordingResult(meta, None if bip is None else np.asarray(bip, dtype=np.float64))


def _cache_save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(result.meta, bipolar_stored=result.bipolar is not None)
    arr = result.bipolar if result.bipolar is not None else np.zeros((0,), dtype=np.float64)
    np.savez(path, bipolar=arr, meta=np.array(json.dumps(meta, sort_keys=True)))


def preprocess_recording(cfg, edf_path, data_root, manifest, pilot_ids, *, allow_all=False,
                         sensitivity_highpass=False, strict=False, cache_root=None):
    """Verify and load, units check, notch, band-pass, bad-electrode flags, edge trim, bipolar (§5.1).

    Bad electrodes are flagged in the result only; excluding recordings is B6 (§12).
    """
    edf_path = Path(edf_path)
    rel = edf_path.relative_to(data_root).as_posix()
    subject, session = _split_rel(rel)
    assert_dev_subject(subject, pilot_ids, allow_all)
    sha = verify_file(cfg, edf_path, rel, manifest)
    key = cache_key(cfg, sha, sensitivity_highpass, strict)
    cpath = _cache_path(cfg, cache_root, rel, key) if cache_root is not None else None
    if cpath is not None and cpath.is_file():
        logger.info("%s: cache hit", rel)
        return _cache_load(cpath)

    rec = load_recording(cfg, edf_path, data_root, manifest, pilot_ids, allow_all, known_sha256=sha)
    fs = rec.fs_hz
    hp = (cfg["preprocessing"]["sensitivity_highpass_hz"] if sensitivity_highpass
          else cfg["preprocessing"]["bandpass"]["highpass_hz"])
    trim_s = edge_trim_seconds(cfg, sensitivity_highpass)
    uc = units_check(cfg, rec.data_uv, fs)
    meta = {"rel_path": rel, "subject": subject, "session": session, "sha256": sha, "sha256_ok": True,
            "fs_hz": fs, "n_samples_raw": int(rec.data_uv.shape[-1]),
            "sensitivity_highpass": bool(sensitivity_highpass), "strict": bool(strict),
            "highpass_hz": hp, "edge_trim_s": trim_s,
            "units_median_sd_uv": uc.median_sd_uv, "units_passed": uc.passed,
            "excluded_reasons": [], "electrodes": {}, "bad_electrodes": [], "n_samples_after_trim": 0}
    if not uc.passed:
        meta["excluded_reasons"].append("units_check")
        logger.warning("%s: excluded, units check failed (median SD %.6g uV)", rel, uc.median_sd_uv)
        result = RecordingResult(meta, None)
    else:
        raw_el = {n: rec.data_uv[rec.ch_names.index(n)] for n in required_labels(cfg)}
        filt_el = {n: trim_edges(bandpass(cfg, notch(cfg, x, fs), fs, hp), fs, trim_s)
                   for n, x in raw_el.items()}
        for n in raw_el:
            rep = electrode_report(cfg, raw_el[n], filt_el[n], fs, strict)
            meta["electrodes"][n] = rep
            if rep["flags"]:
                meta["bad_electrodes"].append(n)
                logger.warning("%s: bad electrode %s (%s)", rel, n, ", ".join(rep["flags"]))
        bip = bipolar(cfg, filt_el)
        meta["n_samples_after_trim"] = int(bip.shape[-1])
        result = RecordingResult(meta, bip)
    if cpath is not None:
        _cache_save(cpath, result)
    return result


# ---------------------------------------------------------------- edge-transient report (development)

def _shaped_noise(rng, n, exponent):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    scale = np.zeros_like(f)
    scale[1:] = f[1:] ** (-exponent * 0.5)
    y = np.fft.irfft(spec * scale, n)
    return y / y.std(ddof=1)


def _edge_row(cfg, seed, sens):
    """One variant, one seed: filter a recording-length noise-like signal alone, and inside a longer
    signal (transient-free reference); compare over the recording length."""
    er = cfg["preprocessing"]["edge_report"]
    fs = float(cfg["dataset"]["native_fs_hz"])
    n = int(round(cfg["dataset"]["recording_length_s"] * fs))
    m = int(round(er["margin_s"] * fs))
    rng = np.random.default_rng(seed)
    t = np.arange(n + 2 * m, dtype=np.float64) / fs
    x = (er["noise_sd_uv"] * _shaped_noise(rng, n + 2 * m, er["noise_power_exponent"])
         + er["sine_amplitude_uv"] * np.sin(2 * np.pi * er["sine_hz"] * t))
    core = x[m:m + n]
    frac = er["max_deviation_fraction_of_sd"]
    hp = (cfg["preprocessing"]["sensitivity_highpass_hz"] if sens
          else cfg["preprocessing"]["bandpass"]["highpass_hz"])
    trim_s = edge_trim_seconds(cfg, sens)
    ideal = bandpass(cfg, x, fs, hp)[m:m + n]
    dev = np.abs(bandpass(cfg, core, fs, hp) - ideal)
    sd = float(ideal.std(ddof=1))
    k = int(round(trim_s * fs))
    over = np.flatnonzero(dev > frac * sd)
    first, last = over[over < n // 2], over[over >= n // 2]
    return {
        "seed": int(seed), "highpass_hz": hp, "edge_trim_s": trim_s, "signal_sd": sd,
        "untrimmed_max_frac": float(dev.max() / sd),
        "seconds_above_limit_start": (first.max() + 1) / fs if first.size else 0.0,
        "seconds_above_limit_end": (n - last.min()) / fs if last.size else 0.0,
        "trimmed_max_frac": float(dev[k:n - k].max() / sd),
        "exceeds": bool(dev[k:n - k].max() > frac * sd),
    }


def edge_transient_report(cfg):
    """One dict per high-pass variant: per-seed rows (seeds seed + 0 ... seed + n - 1, n = n_seeds for the
    0.5 Hz variant, n_seeds_slow for the 0.1 Hz variant) and worst cases."""
    er = cfg["preprocessing"]["edge_report"]
    variants = []
    for sens in (False, True):
        n_seeds = er["n_seeds_slow"] if sens else er["n_seeds"]
        per_seed = [_edge_row(cfg, er["seed"] + i, sens) for i in range(n_seeds)]
        variants.append({
            "highpass_hz": per_seed[0]["highpass_hz"], "edge_trim_s": per_seed[0]["edge_trim_s"],
            "per_seed": per_seed,
            "worst_untrimmed_max_frac": max(r["untrimmed_max_frac"] for r in per_seed),
            "worst_seconds_above_limit_start": max(r["seconds_above_limit_start"] for r in per_seed),
            "worst_seconds_above_limit_end": max(r["seconds_above_limit_end"] for r in per_seed),
            "worst_trimmed_max_frac": max(r["trimmed_max_frac"] for r in per_seed),
            "exceeds": any(r["exceeds"] for r in per_seed),
        })
    return variants


# ---------------------------------------------------------------- electrode diagnostics (read-only, IMP-010)

def _periodogram(cfg, x, fs_hz):
    """Mean periodogram of non-overlapping, mean-removed Hann segments (the IMP-005 alpha-peak method);
    x is (n_channels, n_samples). Returns (freqs, psd)."""
    nseg = int(round(cfg["simulator"]["sanity_welch_segment_s"] * fs_hz))
    nblk = x.shape[-1] // nseg
    blocks = x[:, :nblk * nseg].reshape(x.shape[0], nblk, nseg)
    blocks = (blocks - blocks.mean(axis=-1, keepdims=True)) * np.hanning(nseg)
    psd = np.mean(np.abs(np.fft.rfft(blocks, axis=-1)) ** 2, axis=1)
    return np.fft.rfftfreq(nseg, 1.0 / fs_hz), psd


def electrode_diagnostics(cfg, filtered_uv, ch_names, fs_hz):
    """RMS of every channel and, for the four required electrodes, rank, alpha peak and relative alpha.

    filtered_uv: (n_channels, n_samples) after notch, band-pass and edge trim, as in the pipeline.
    Print-only: nothing here changes a threshold or is stored.
    """
    ed = cfg["preprocessing"]["electrode_diagnostics"]
    lo_a, hi_a = ed["alpha_band_hz"]
    lo_t, hi_t = ed["total_band_hz"]
    rms_all = np.sqrt(np.mean(filtered_uv ** 2, axis=-1))
    med = float(np.median(rms_all))
    rms_floor = cfg["preprocessing"]["bad_channel"]["rms_min_uv"]
    freqs, psd = _periodogram(cfg, filtered_uv, fs_hz)
    in_alpha = (freqs >= lo_a) & (freqs <= hi_a)
    in_total = (freqs >= lo_t) & (freqs <= hi_t)
    out = {"n_channels": int(rms_all.size), "rms_median": med, "rms_min": float(rms_all.min()),
           "rms_max": float(rms_all.max()), "n_below_floor": int(np.count_nonzero(rms_all < rms_floor)),
           "electrodes": {}}
    for name in required_labels(cfg):
        i = ch_names.index(name)
        out["electrodes"][name] = {
            "rms_uv": float(rms_all[i]), "rms_over_median": float(rms_all[i] / med),
            "rank": int(np.count_nonzero(rms_all < rms_all[i])) + 1,
            "alpha_peak_hz": float(freqs[in_alpha][int(np.argmax(psd[i][in_alpha]))]),
            "relative_alpha": float(psd[i][in_alpha].sum() / psd[i][in_total].sum()),
            "below_floor": bool(rms_all[i] < rms_floor),
        }
    return out


def recording_electrode_diagnostics(cfg, rec):
    """Diagnostics for one loaded Recording with the pipeline's notch, band-pass and edge trim."""
    filt = trim_edges(bandpass(cfg, notch(cfg, rec.data_uv, rec.fs_hz), rec.fs_hz,
                               cfg["preprocessing"]["bandpass"]["highpass_hz"]),
                      rec.fs_hz, edge_trim_seconds(cfg))
    return electrode_diagnostics(cfg, filt, rec.ch_names, rec.fs_hz)


def print_electrode_diagnostics(d):
    print(f"    all channels' RMS (uV): median {d['rms_median']:.3f}, min {d['rms_min']:.3f}, "
          f"max {d['rms_max']:.3f} | channels below the RMS floor: {d['n_below_floor']} of {d['n_channels']}")
    for name, e in d["electrodes"].items():
        print(f"      {name:>4}: RMS {e['rms_uv']:8.3f} uV = {e['rms_over_median']:.3f} x median | "
              f"rank {e['rank']:>2} of {d['n_channels']} (1 = lowest) | alpha peak {e['alpha_peak_hz']:6.3f} Hz | "
              f"relative alpha {e['relative_alpha']:.4f} | below floor: {'YES' if e['below_floor'] else 'no'}")


# ---------------------------------------------------------------- pilot report (CLI)

def pilot_recordings(cfg, root, pilot_ids):
    data = Path(root) / cfg["paths"]["data_dir"]
    pattern = cfg["dataset"]["eeg_glob"]
    return [p for s in sorted(pilot_ids) for p in sorted(data.glob(pattern.replace("sub-*", s, 1)))]


def print_edge_report(cfg, variants):
    er = cfg["preprocessing"]["edge_report"]
    limit = er["max_deviation_fraction_of_sd"]
    print(f"\n[edge transient] 1/f noise + {er['sine_hz']} Hz sine, base seed {er['seed']} "
          f"(0.5 Hz variant {er['n_seeds']} seeds, 0.1 Hz variant {er['n_seeds_slow']} seeds), "
          f"{cfg['dataset']['recording_length_s']} s at "
          f"{cfg['dataset']['native_fs_hz']} Hz; deviation from the transient-free reference as a fraction of "
          f"the filtered signal SD (limit {limit})")
    for v in variants:
        print(f"\n  high-pass {v['highpass_hz']} Hz, edge trim {v['edge_trim_s']} s: "
              f"worst deviation after trim {v['worst_trimmed_max_frac']:.6f} | worst seconds above the limit: "
              f"start {v['worst_seconds_above_limit_start']:.3f}, end {v['worst_seconds_above_limit_end']:.3f} | "
              f"{'STOP' if v['exceeds'] else 'ok'}")
        print(f"  {'seed':>10} {'sig SD':>8} {'max dev, no trim':>17} {'s>lim start':>12} {'s>lim end':>10} "
              f"{'max dev after trim':>19} {'verdict':>8}")
        for r in v["per_seed"]:
            print(f"  {r['seed']:>10} {r['signal_sd']:>8.4f} {r['untrimmed_max_frac']:>17.5f} "
                  f"{r['seconds_above_limit_start']:>12.3f} {r['seconds_above_limit_end']:>10.3f} "
                  f"{r['trimmed_max_frac']:>19.6f} {'STOP' if r['exceeds'] else 'ok':>8}")


def flag_summary(flagged_recordings, flagged_subjects, n_recordings, n_subjects):
    """Fractions of the pilot that the current bad-electrode rule flags."""
    return {"recordings": flagged_recordings, "n_recordings": n_recordings,
            "recording_fraction": flagged_recordings / n_recordings,
            "subjects": len(flagged_subjects), "n_subjects": n_subjects,
            "subject_fraction": len(flagged_subjects) / n_subjects}


def print_flag_summary(s):
    print(f"\n[current rule] recordings with at least one bad electrode: {s['recordings']} of {s['n_recordings']} "
          f"({s['recording_fraction']:.3f}); subjects with at least one such recording: "
          f"{s['subjects']} of {s['n_subjects']} ({s['subject_fraction']:.3f})")


def pilot_report(cfg, root, electrode_diag=False):
    """B3 + B4 on the pilot subjects' recordings; prints only, stores nothing in config.yml.

    electrode_diag adds the read-only all-channel RMS and alpha diagnostics (pilot subjects only).
    """
    pilot_ids = load_pilot_ids(cfg, root)
    manifest = load_manifest(Path(root) / cfg["paths"]["manifest_file"])
    cache_root = Path(root) / cfg["paths"]["cache_dir"]
    data_root = Path(root) / cfg["paths"]["data_dir"]
    files = pilot_recordings(cfg, root, pilot_ids)
    print(f"[pilot report] {len(pilot_ids)} pilot subjects, {len(files)} recordings, "
          f"units check high-pass {cfg['preprocessing']['units_check']['highpass_hz']} Hz, "
          f"edge trim {cfg['preprocessing']['edge_trim_s']} s")
    n_fail = 0
    flagged_recordings, flagged_subjects = 0, set()
    for f in files:
        res = preprocess_recording(cfg, f, data_root, manifest, pilot_ids, cache_root=cache_root)
        m = res.meta
        n_fail += not m["units_passed"]
        print(f"\n{m['subject']} {m['session']}: SHA-256 {'OK' if m['sha256_ok'] else 'FAIL'} ({m['sha256']}) | "
              f"median SD after 0.5 Hz high-pass {m['units_median_sd_uv']:.3f} uV | "
              f"units check {'PASS' if m['units_passed'] else 'FAIL'} | samples after trim {m['n_samples_after_trim']}")
        for name, e in m["electrodes"].items():
            flag = ", ".join(e["flags"]) if e["flags"] else "-"
            print(f"    {name:>4}: band-passed RMS {e['rms_uv']:8.3f} uV | longest raw run at own max/min "
                  f"{e['saturation_run']:>3} samples (limit {e['saturation_limit']}) | "
                  f"flat 1-s segments {e['flat_segments']} | bad-electrode flag: {flag}")
        if m["bad_electrodes"]:
            print(f"    BAD ELECTRODES: {', '.join(m['bad_electrodes'])}")
            flagged_recordings += 1
            flagged_subjects.add(m["subject"])
        if electrode_diag and m["units_passed"]:
            rec = load_recording(cfg, f, data_root, manifest, pilot_ids, known_sha256=m["sha256"])
            print_electrode_diagnostics(recording_electrode_diagnostics(cfg, rec))
    halt = units_check_halt(cfg, n_fail, len(files))
    print(f"\n[units check] {halt.n_failed} of {halt.n_total} failed ({halt.fraction:.4f}); "
          f"halt: {'YES' if halt.halt else 'no'}")
    print_flag_summary(flag_summary(flagged_recordings, flagged_subjects, len(files), len(pilot_ids)))
    variants = edge_transient_report(cfg)
    print_edge_report(cfg, variants)
    if any(v["exceeds"] for v in variants):
        print("\nSTOP: the deviation after the trim exceeds the limit for at least one variant and seed; "
              "the edge trims are too short. Nothing was changed.")
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3/B4 recording preprocessing")
    parser.add_argument("--pilot-report", action="store_true",
                        help="run B3 and B4 on the pilot subjects' recordings and print a report")
    parser.add_argument("--electrode-diagnostics", action="store_true",
                        help="with --pilot-report: print read-only all-channel RMS and alpha diagnostics")
    args = parser.parse_args(argv)
    if not args.pilot_report:
        parser.error("nothing to do; use --pilot-report")
    cfg = load_config()
    log_dir = REPO_ROOT / cfg["paths"]["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "preprocess_pilot_report.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("pipeline").addHandler(handler)
    logging.getLogger("pipeline").setLevel(logging.INFO)
    return pilot_report(cfg, REPO_ROOT, electrode_diag=args.electrode_diagnostics)


if __name__ == "__main__":
    sys.exit(main())
