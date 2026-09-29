"""Zero-phase band-pass, FFT downsampling, and the recording pipeline up to the
bipolar pairs (§4.2, §5.1, §6, §10.1; IMP-004, IMP-009).

A3 functions (bandpass, downsample) are pure. B3/B4 add file loading with
checksum verification, the units check, bad-electrode detection, the notch, edge
trimming, bipolar derivation, a per-recording cache, and the pilot report
(`python -m src.preprocess --pilot-report`). B5/B6 (IMP-012) add blink correction,
1-s segment rejection, downsampling, clean segments, rescaling, the vigilance proxy and the
exclusion rules (`--segment-report`).
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
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.signal import butter, filtfilt, iirnotch, resample, sosfiltfilt

from src.config import REPO_ROOT, load_config

logger = logging.getLogger("pipeline.preprocess")

_SOURCE = Path(__file__).resolve()


def emit(text=""):
    """The one place report text is written to stdout (development reports); everything else logs."""
    sys.stdout.write(text + "\n")


def code_hash():
    """SHA-256 of the source text of this module; part of the B4 and B5 cache keys, so that a change to
    the code invalidates cached results (IMP-013)."""
    return hashlib.sha256(_SOURCE.read_bytes()).hexdigest()


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


def bandpass_sos(cfg, fs_hz, highpass_hz=None, lowpass_hz=None):
    """High-pass sections followed by low-pass sections, `filter_order` per pass each."""
    bp = cfg["preprocessing"]["bandpass"]
    if bp["filter_family"] != "butterworth":
        raise PreprocessError(f"unsupported filter_family {bp['filter_family']!r}")
    if not bp["zero_phase"]:
        raise PreprocessError("preprocessing.bandpass.zero_phase must be true (§5.1)")
    hp = bp["highpass_hz"] if highpass_hz is None else highpass_hz
    lp = bp["lowpass_hz"] if lowpass_hz is None else lowpass_hz
    if not 0 < hp < lp < fs_hz * 0.5:
        raise PreprocessError(
            f"need 0 < highpass ({hp}) < lowpass ({lp}) < Nyquist ({fs_hz * 0.5})")
    order = bp["filter_order"]
    sos_hp = butter(order, hp, btype="highpass", fs=fs_hz, output="sos")
    sos_lp = butter(order, lp, btype="lowpass", fs=fs_hz, output="sos")
    return np.vstack([sos_hp, sos_lp])


def bandpass(cfg, x, fs_hz, highpass_hz=None, lowpass_hz=None):
    """Zero-phase band-pass of whole continuous recordings; time on the last axis."""
    x = _check_signal(x, "bandpass")
    bp = cfg["preprocessing"]["bandpass"]
    padlen = int(round(bp["edge_pad_s"] * fs_hz))
    if x.shape[-1] <= padlen:
        raise PreprocessError(
            f"bandpass: {x.shape[-1]} samples is not more than the edge padding "
            f"({padlen} samples = {bp['edge_pad_s']} s); use whole recordings, not segments")
    sos = bandpass_sos(cfg, fs_hz, highpass_hz, lowpass_hz)
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
    bipolar_notched: object = field(default=None)   # same, notched but NOT band-passed (EMG screen, IMP-012)


def _fingerprint(cfg):
    pp, ds = cfg["preprocessing"], cfg["dataset"]
    used = {
        "units_check": pp["units_check"], "channels": pp["channels"], "bad_channel": pp["bad_channel"],
        "line_noise": pp["line_noise"], "bandpass": pp["bandpass"], "edge_trim_s": pp["edge_trim_s"],
        "edge_trim_s_sensitivity_highpass": pp["edge_trim_s_sensitivity_highpass"],
        "segment_length_s": pp["rejection"]["segment_length_s"],
        "sensitivity_highpass_hz": pp["sensitivity_highpass_hz"], "strict_variant": pp["strict_variant"],
        "cache_schema_version": pp["cache"]["schema_version"],
        "read_units": ds["read_units"], "native_fs_hz": ds["native_fs_hz"],
        "edf_suffix": ds["edf_suffix"], "channels_suffix": ds["channels_suffix"],
    }
    return json.dumps(used, sort_keys=True, default=str)


def cache_key(cfg, sha256, sensitivity_highpass, strict):
    payload = "\n".join([sha256, code_hash(), _fingerprint(cfg), str(bool(sensitivity_highpass)), str(bool(strict))])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cfg, cache_root, rel_path, key):
    subject, session = _split_rel(rel_path)
    return Path(cache_root) / cfg["preprocessing"]["cache"]["subdir"] / f"{subject}_{session}_{key}.npz"


def _cache_load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        bip = z["bipolar"] if meta["bipolar_stored"] else None
        bn = z["bipolar_notched"] if meta["bipolar_stored"] else None
    return RecordingResult(meta, None if bip is None else np.asarray(bip, dtype=np.float64),
                           None if bn is None else np.asarray(bn, dtype=np.float64))


def _cache_save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(result.meta, bipolar_stored=result.bipolar is not None)
    arr = result.bipolar if result.bipolar is not None else np.zeros((0,), dtype=np.float64)
    arr_n = result.bipolar_notched if result.bipolar_notched is not None else np.zeros((0,), dtype=np.float64)
    np.savez(path, bipolar=arr, bipolar_notched=arr_n, meta=np.array(json.dumps(meta, sort_keys=True)))


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
        notched_el = {n: notch(cfg, x, fs) for n, x in raw_el.items()}
        filt_el = {n: trim_edges(bandpass(cfg, x, fs, hp), fs, trim_s) for n, x in notched_el.items()}
        notched_trim = {n: trim_edges(x, fs, trim_s) for n, x in notched_el.items()}
        for n in raw_el:
            rep = electrode_report(cfg, raw_el[n], filt_el[n], fs, strict)
            meta["electrodes"][n] = rep
            if rep["flags"]:
                meta["bad_electrodes"].append(n)
                logger.warning("%s: bad electrode %s (%s)", rel, n, ", ".join(rep["flags"]))
        bip = bipolar(cfg, filt_el)
        meta["n_samples_after_trim"] = int(bip.shape[-1])
        result = RecordingResult(meta, bip, bipolar(cfg, notched_trim))
    if cpath is not None:
        _cache_save(cpath, result)
    return result


# ---------------------------------------------------------------- B5: blinks (§5.1, IMP-012)

def robust_sd(cfg, x):
    """mad_to_sd_factor * MAD about the median, along the last axis."""
    med = np.median(x, axis=-1, keepdims=True)
    return cfg["preprocessing"]["ocular"]["mad_to_sd_factor"] * np.median(np.abs(x - med), axis=-1)


def blink_copy(cfg, x, fs_hz):
    """The detect_band_hz (0.5 to 5 Hz) copy of a 1-D channel and its median."""
    lo, hi = cfg["preprocessing"]["ocular"]["detect_band_hz"]
    c = bandpass(cfg, x, fs_hz, highpass_hz=lo, lowpass_hz=hi)
    return c, float(np.median(c))


def blink_excursion(cfg, x, fs_hz):
    """max |copy - median| / robust SD of the detect-band copy (print-only diagnostic, IMP-012)."""
    c, med = blink_copy(cfg, x, fs_hz)
    sd = float(robust_sd(cfg, c))
    return float(np.max(np.abs(c - med)) / sd) if sd > 0 else 0.0


def blink_flags(cfg, x, fs_hz):
    """Boolean mask of samples where |copy - median| exceeds detect_threshold_robust_sd robust SDs."""
    c, med = blink_copy(cfg, x, fs_hz)
    sd = float(robust_sd(cfg, c))
    if sd <= 0:
        return np.zeros(x.shape, dtype=bool)
    return np.abs(c - med) > cfg["preprocessing"]["ocular"]["detect_threshold_robust_sd"] * sd


def blink_window_weights(cfg, flagged, fs_hz):
    """(window mask, weight, n_windows). Window = every sample within correct_window_s of a flagged
    sample (flagged samples closer than twice that share one window). Weight is 1 on flagged samples
    and falls by a raised cosine to 0 at correct_window_s from the nearest flagged sample."""
    flagged = np.asarray(flagged, dtype=bool)
    if not flagged.any():
        return np.zeros(flagged.shape, dtype=bool), np.zeros(flagged.shape), 0
    margin = int(round(cfg["preprocessing"]["ocular"]["correct_window_s"] * fs_hz))
    dist = distance_transform_edt(~flagged)
    window = dist <= margin
    weight = np.where(dist < margin, 0.5 * (1.0 + np.cos(np.pi * dist / margin)), 0.0)
    n_windows = int(np.count_nonzero(np.diff(np.concatenate(([0], window.astype(np.int8)))) == 1))
    return window, weight, n_windows


def wavelet_levels(cfg, fs_hz):
    """(decomposition level, altered detail levels), derived from detect_band_hz and the rate.

    Detail level j covers fs / 2**(j + 1) to fs / 2**j. Level = ceil(log2(fs / (2 * low))); altered =
    detail levels whose band lies below the top of the detect band (and above its bottom)."""
    lo, hi = cfg["preprocessing"]["ocular"]["detect_band_hz"]
    level = int(math.ceil(math.log2(fs_hz / (2.0 * lo))))
    altered = [j for j in range(1, level + 1) if fs_hz / 2 ** (j + 1) < hi and fs_hz / 2 ** j > lo]
    return level, altered


def wavelet_half_support(family, level):
    """Half of the level-`level` undecimated filter length in samples: (dec_len - 1) * 2**(level - 1) + 1, halved."""
    import pywt
    return ((pywt.Wavelet(family).dec_len - 1) * 2 ** (level - 1) + 1) // 2


def wavelet_correct(cfg, x, fs_hz, window, weight, diagnostics=None):
    """Blink correction of a 1-D channel by clipping undecimated wavelet detail coefficients.

    At each altered level j the coefficients inside `window` extended on both sides by that level's
    filter half-support are clipped to +/- clip_multiplier times the robust SD of that level over the
    whole channel. The pad is ceil(edge_pad_s * fs) samples on both ends plus a tail on the right that
    makes the padded length a multiple of 2**level. The inverse transform c is spliced in as
    x + weight * (c - x) inside `window`; samples outside it are returned unchanged (bit-identical).
    """
    import pywt  # lazy: only blink correction needs it
    x = _check_signal(x, "wavelet_correct")
    if not window.any():
        return x.copy()
    wv = cfg["preprocessing"]["ocular"]["wavelet"]
    level, altered = wavelet_levels(cfg, fs_hz)
    n = x.shape[-1]
    pad = int(math.ceil(wv["edge_pad_s"] * fs_hz))
    if pad >= n:
        raise PreprocessError(f"wavelet_correct: {n} samples is not more than the edge pad ({pad})")
    block = 2 ** level
    tail = -(-(n + 2 * pad) // block) * block - (n + 2 * pad)
    xp = np.pad(x, (pad, pad + tail), mode="reflect", reflect_type=wv["reflect_type"])
    mp = np.pad(window, (pad, pad + tail), constant_values=False)
    dist = distance_transform_edt(~mp)
    coeffs = pywt.swt(xp, wv["family"], level=level, trim_approx=True)   # [cA_L, cD_L, ..., cD_1]
    for j in altered:
        d = coeffs[1 + level - j]
        ext = dist <= wavelet_half_support(wv["family"], j)
        sd = float(robust_sd(cfg, d[pad:pad + n]))
        lam = wv["clip_multiplier_robust_sd"] * sd
        if diagnostics is not None:
            over = np.abs(d) > lam
            diagnostics[j] = {"robust_sd": sd, "limit": lam, "half_support": wavelet_half_support(wv["family"], j),
                              "over_inside": int(np.count_nonzero(ext & over)),
                              "over_outside": int(np.count_nonzero(~ext & over))}
        coeffs[1 + level - j] = np.where(ext, np.clip(d, -lam, lam), d)
    c = pywt.iswt(coeffs, wv["family"])[pad:pad + n]
    return np.where(window, x + weight * (c - x), x)


def correct_blinks(cfg, x, fs_hz):
    """Detect and correct blinks in every row of x (n_ch, n). Returns (corrected, per-channel log)."""
    out, log = np.empty_like(x), []
    for i in range(x.shape[0]):
        flagged = blink_flags(cfg, x[i], fs_hz)
        window, weight, n_win = blink_window_weights(cfg, flagged, fs_hz)
        out[i] = wavelet_correct(cfg, x[i], fs_hz, window, weight) if n_win else x[i]
        log.append({"n_flagged_samples": int(flagged.sum()), "n_windows": n_win,
                    "corrected_s": float(window.sum() / fs_hz)})
    return out, log


# ---------------------------------------------------------------- B5: 1-s segment rejection (§5.1, IMP-012)

def segment_blocks(x, seg):
    """(n_ch, n) -> (n_ch, n // seg, seg); a partial last block is dropped."""
    n_seg = x.shape[-1] // seg
    return x[:, :n_seg * seg].reshape(x.shape[0], n_seg, seg)


def hann_power(blocks):
    """|rfft|^2 of mean-removed, Hann-windowed blocks along the last axis."""
    seg = blocks.shape[-1]
    return np.abs(np.fft.rfft((blocks - blocks.mean(axis=-1, keepdims=True)) * np.hanning(seg), axis=-1)) ** 2


def emg_power(cfg, x_notched, fs_hz):
    """(n_ch, n_seg): mean Hann periodogram over emg_band_hz (inclusive) of every whole 1-s segment."""
    rj = cfg["preprocessing"]["rejection"]
    seg = int(round(rj["segment_length_s"] * fs_hz))
    lo, hi = rj["emg_band_hz"]
    freqs = np.fft.rfftfreq(seg, 1.0 / fs_hz)
    band = (freqs >= lo) & (freqs <= hi)
    return hann_power(segment_blocks(x_notched, seg))[..., band].mean(axis=-1)


def segment_flags(cfg, x_bp, x_notched, fs_hz, strict=False):
    """Per-rule, per-channel 1-s segment failures (n_ch, n_seg) and the overall rejection (n_seg,).

    x_bp: blink-corrected band-passed bipolar channels; x_notched: the SAME channels notched but not
    band-passed (the EMG measure is taken before the band-pass and low-pass, §5.1). Both must have the
    same length. A segment is rejected if EITHER channel fails ANY rule.
    """
    if x_bp.shape != x_notched.shape:
        raise PreprocessError("segment_flags: x_bp and x_notched must have the same shape")
    pp = cfg["preprocessing"]
    rj = pp["rejection"]
    seg = int(round(rj["segment_length_s"] * fs_hz))
    blocks = segment_blocks(x_bp, seg)
    rms = np.sqrt(np.mean(blocks ** 2, axis=-1))
    hi = pp["strict_variant"]["rms_max_uv"] if strict else pp["bad_channel"]["rms_max_uv"]
    mult = pp["strict_variant"]["emg_multiple_of_median"] if strict else rj["emg_multiple_of_median"]
    ptp = blocks.max(axis=-1) - blocks.min(axis=-1)
    pw = emg_power(cfg, x_notched, fs_hz)
    med = np.median(pw, axis=-1, keepdims=True)
    flags = {"rms_low": rms < rj["segment_rms_min_uv"], "rms_high": rms > hi,
             "flat": ptp < pp["bad_channel"]["flatline_threshold"], "emg": pw > mult * med}
    rejected = np.any([f.any(axis=0) for f in flags.values()], axis=0)
    return flags, rejected


def pad_rejected(cfg, rejected, fs_hz):
    """Sample mask at fs_hz (True = rejected) of rejected 1-s segments, each padded by rejection.pad_s."""
    rj = cfg["preprocessing"]["rejection"]
    seg = int(round(rj["segment_length_s"] * fs_hz))
    pad = int(round(rj["pad_s"] * fs_hz))
    raw = np.repeat(np.asarray(rejected, dtype=bool), seg)
    return binary_dilation(raw, structure=np.ones(2 * pad + 1, dtype=bool))


def clean_runs(keep, min_len):
    """[(start, end)) of the runs of True in `keep` that are at least min_len samples long."""
    d = np.diff(np.concatenate(([0], np.asarray(keep, dtype=np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


# ---------------------------------------------------------------- B5: rescaling, vigilance (§5.1, IMP-006, IMP-012)

def rescale(cfg, segments):
    """Rescale every channel to mu_ref and sigma_ref using the mean and SD (ddof) of that channel's clean
    samples over ALL segments. Returns (scaled segments, {"mean", "sd"} per channel). Constants come from
    config (rescaling.mu_ref, rescaling.sigma_ref), never from model.py."""
    if not segments:
        raise PreprocessError("rescale: no clean segments")
    allx = np.concatenate(segments, axis=-1)
    mean = allx.mean(axis=-1)
    sd = allx.std(axis=-1, ddof=cfg["preprocessing"]["rescale"]["ddof"])
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise PreprocessError(f"rescale: a channel is constant or non-finite over its clean samples (SD {sd})")
    mu_ref, sigma_ref = cfg["rescaling"]["mu_ref"], cfg["rescaling"]["sigma_ref"]
    scaled = [mu_ref + sigma_ref * (s - mean[:, None]) / sd[:, None] for s in segments]
    return scaled, {"mean": [float(m) for m in mean], "sd": [float(v) for v in sd]}


def vigilance(cfg, segments, starts, fs_hz):
    """Alpha/theta power per non-overlapping epoch (epoch_s, aligned to each clean segment's start; a
    partial epoch is dropped), per channel, Hann periodogram; alpha bins in [lo, hi], theta bins in
    [lo, hi) so 8 Hz is counted once. Returns starts (at fs_hz, relative to the trimmed start),
    ratios (n_ch lists) and the per-channel mean."""
    vg = cfg["preprocessing"]["vigilance"]
    ep = int(round(vg["epoch_s"] * fs_hz))
    freqs = np.fft.rfftfreq(ep, 1.0 / fs_hz)
    a_lo, a_hi = vg["alpha_band_hz"]
    t_lo, t_hi = vg["theta_band_hz"]
    alpha = (freqs >= a_lo) & (freqs <= a_hi)
    theta = (freqs >= t_lo) & (freqs < t_hi)
    ratios, epoch_starts = [], []
    for seg, s0 in zip(segments, starts):
        blocks = segment_blocks(seg, ep)
        if blocks.shape[1] == 0:
            continue
        p = hann_power(blocks)
        ratios.append(p[..., alpha].sum(axis=-1) / p[..., theta].sum(axis=-1))
        epoch_starts.extend(s0 + k * ep for k in range(blocks.shape[1]))
    n_ch = segments[0].shape[0] if segments else 0
    r = np.concatenate(ratios, axis=-1) if ratios else np.zeros((n_ch, 0))
    return {"epoch_starts": epoch_starts, "ratios": [[float(v) for v in row] for row in r],
            "mean": [float(np.mean(row)) if row.size else None for row in r]}


# ---------------------------------------------------------------- B5: driver and cache (IMP-012)

@dataclass
class SegmentResult:
    meta: dict            # B4 meta plus the "b5" block (log, rescaling constants, vigilance, ...)
    segments: list = field(default_factory=list)   # (2, n) float64 at the observation rate
    starts: list = field(default_factory=list)     # start index at the observation rate, relative to the trimmed start


def _fingerprint_b5(cfg):
    pp = cfg["preprocessing"]
    used = {"b4": json.loads(_fingerprint(cfg)), "ocular": pp["ocular"], "rejection": pp["rejection"],
            "vigilance": pp["vigilance"], "rescale": pp["rescale"], "observation_fs_hz": pp["observation_fs_hz"],
            "resample": pp["resample"], "min_clean_data_per_recording_s": pp["min_clean_data_per_recording_s"],
            "mu_ref": cfg["rescaling"]["mu_ref"], "sigma_ref": cfg["rescaling"]["sigma_ref"]}
    return json.dumps(used, sort_keys=True, default=str)


def cache_key_b5(cfg, sha256, sensitivity_highpass, strict):
    payload = "\n".join(["b5", sha256, code_hash(), _fingerprint_b5(cfg), str(bool(sensitivity_highpass)),
                         str(bool(strict))])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _b5_path(cfg, cache_root, rel_path, key):
    subject, session = _split_rel(rel_path)
    return Path(cache_root) / cfg["preprocessing"]["cache"]["subdir_b5"] / f"{subject}_{session}_{key}.npz"


def _b5_save(path, res):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"seg_{i}": a for i, a in enumerate(res.segments)}
    np.savez(path, meta=np.array(json.dumps(dict(res.meta, starts=list(res.starts)), sort_keys=True)), **arrays)


def _b5_load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        starts = meta.pop("starts")
        segs = [np.asarray(z[f"seg_{i}"], dtype=np.float64) for i in range(len(starts))]
    return SegmentResult(meta, segs, starts)


def segment_signal(cfg, x_bp, x_notched, fs_hz, strict=False):
    """B5 on already-derived bipolar channels (native rate, edge-trimmed). Returns a SegmentResult
    whose meta holds only the "b5" block. Order: crop to whole 1-s blocks, blink correction, 1-s
    rejection, padding, downsample the continuous signal, map the mask, split into clean stretches,
    vigilance (pre-rescaling; the ratio is scale-free), rescale."""
    x_bp = _check_signal(x_bp, "segment_signal")
    x_notched = _check_signal(x_notched, "segment_signal")
    pp = cfg["preprocessing"]
    seg_s = pp["rejection"]["segment_length_s"]
    seg_in = int(round(seg_s * fs_hz))
    fs_out = pp["observation_fs_hz"]
    seg_out = int(round(seg_s * fs_out))
    n_seg = x_bp.shape[-1] // seg_in
    if n_seg == 0:
        raise PreprocessError("segment_signal: less than one whole 1-s segment")
    n_use = n_seg * seg_in
    trimmed_s = float(x_bp.shape[-1] / fs_hz)
    tail_s = float((x_bp.shape[-1] - n_use) / fs_hz)
    x_bp, x_notched = x_bp[:, :n_use], x_notched[:, :n_use]
    corrected, blink_log = correct_blinks(cfg, x_bp, fs_hz)
    flags, rejected = segment_flags(cfg, corrected, x_notched, fs_hz, strict)
    x_out = downsample(cfg, corrected, fs_hz, fs_out)
    padded = pad_rejected(cfg, rejected, fs_out)
    if x_out.shape[-1] != padded.shape[-1] or padded.shape[-1] != n_seg * seg_out:
        raise PreprocessError("segment_signal: downsampled length does not match the segment grid")
    min_len = int(round(pp["rejection"]["min_clean_stretch_s"] * fs_out))
    runs = clean_runs(~padded, min_len)
    segments = [np.ascontiguousarray(x_out[:, a:b]) for a, b in runs]
    starts = [a for a, _ in runs]
    n_keep = int(np.count_nonzero(~padded))
    n_clean = sum(b - a for a, b in runs)
    log = {"n_segments": int(n_seg), "trimmed_s": trimmed_s, "cropped_tail_s": tail_s,
           "rules": {k: {"n_segments": int(v.any(axis=0).sum()), "seconds": float(v.any(axis=0).sum() * seg_s),
                         "per_channel": [int(r.sum()) for r in v]} for k, v in flags.items()},
           "rejected_segments": int(rejected.sum()), "rejected_before_padding_s": float(rejected.sum() * seg_s),
           "rejected_after_padding_s": float(padded.sum() / fs_out),
           "dropped_short_stretch_s": float((n_keep - n_clean) / fs_out),
           "clean_s": float(n_clean / fs_out), "n_clean_segments": len(runs), "blinks": blink_log}
    b5 = {"log": log, "rescaled": False, "rescale": None, "vigilance": None, "clean_stats_before": None}
    if segments:
        allx = np.concatenate(segments, axis=-1)
        b5["clean_stats_before"] = {"mean": [float(v) for v in allx.mean(axis=-1)],
                                    "sd": [float(v) for v in allx.std(axis=-1, ddof=pp["rescale"]["ddof"])]}
        b5["vigilance"] = vigilance(cfg, segments, starts, fs_out)
        segments, consts = rescale(cfg, segments)
        b5["rescale"], b5["rescaled"] = consts, True
    return SegmentResult({"b5": b5}, segments, starts)


def segment_recording(cfg, edf_path, data_root, manifest, pilot_ids, *, allow_all=False,
                      sensitivity_highpass=False, strict=False, cache_root=None):
    """B4 (cached), then B5 for recordings that passed the units check and have no bad electrode
    (B6 order; a recording excluded earlier is not segmented). Guard and checksum are those of
    preprocess_recording. Cached under cache/<subdir_b5>/ by the file hash and every config leaf used."""
    b4 = preprocess_recording(cfg, edf_path, data_root, manifest, pilot_ids, allow_all=allow_all,
                              sensitivity_highpass=sensitivity_highpass, strict=strict, cache_root=cache_root)
    meta = dict(b4.meta)
    if b4.bipolar is None or meta["bad_electrodes"]:
        meta["b5"] = None
        return SegmentResult(meta)
    rel = meta["rel_path"]
    cpath = (_b5_path(cfg, cache_root, rel, cache_key_b5(cfg, meta["sha256"], sensitivity_highpass, strict))
             if cache_root is not None else None)
    if cpath is not None and cpath.is_file():
        logger.info("%s: B5 cache hit", rel)
        return _b5_load(cpath)
    res = segment_signal(cfg, b4.bipolar, b4.bipolar_notched, meta["fs_hz"], strict)
    res.meta = dict(meta, **res.meta)
    lg = res.meta["b5"]["log"]
    logger.info("%s: B5 rejected %d of %d segments (%.1f s before padding, %.1f s after), clean %.1f s in %d segments",
                rel, lg["rejected_segments"], lg["n_segments"], lg["rejected_before_padding_s"],
                lg["rejected_after_padding_s"], lg["clean_s"], lg["n_clean_segments"])
    if cpath is not None:
        _b5_save(cpath, res)
    return res


# ---------------------------------------------------------------- B6: exclusions (§12, IMP-012)

def corrected_fractions(log):
    """Corrected blink time of each bipolar channel as a fraction of the edge-trimmed recording (B5 log)."""
    return [b["corrected_s"] / log["trimmed_s"] for b in log["blinks"]]


def blink_correction_excessive(cfg, log):
    """True if the corrected time of EITHER channel is strictly more than max_corrected_fraction of the
    edge-trimmed recording (exact arithmetic on the stored seconds; equal is not excessive)."""
    limit = Fraction(repr(cfg["preprocessing"]["ocular"]["max_corrected_fraction"])) * Fraction(log["trimmed_s"])
    return any(Fraction(b["corrected_s"]) > limit for b in log["blinks"])


def recording_decision(cfg, meta, seg_meta=None):
    """Kept or excluded for ONE recording, in the §12 order: units check, bad electrode, excessive blink
    correction (IMP-013), clean data below min_clean_data_per_recording_s (from B5). Returns the FIRST
    failing rule; every exclusion is logged."""
    rel = meta["rel_path"]
    minimum = cfg["preprocessing"]["min_clean_data_per_recording_s"]
    clean_s = None
    if not meta["units_passed"]:
        d = {"status": "excluded", "reason": "units_check", "detail": {"median_sd_uv": meta["units_median_sd_uv"]}}
    elif meta["bad_electrodes"]:
        d = {"status": "excluded", "reason": "bad_electrode",
             "detail": {"electrodes": {n: meta["electrodes"][n]["flags"] for n in meta["bad_electrodes"]}}}
    else:
        b5 = None if seg_meta is None else seg_meta.get("b5")
        if b5 is None:
            raise PreprocessError(f"{rel}: B5 results are needed to judge the clean-data minimum")
        clean_s = b5["log"]["clean_s"]
        if blink_correction_excessive(cfg, b5["log"]):
            d = {"status": "excluded", "reason": "excessive_blink_correction",
                 "detail": {"corrected_fraction": corrected_fractions(b5["log"]),
                            "maximum_fraction": cfg["preprocessing"]["ocular"]["max_corrected_fraction"]}}
        elif clean_s < minimum:
            d = {"status": "excluded", "reason": "insufficient_clean_data",
                 "detail": {"clean_s": clean_s, "minimum_s": minimum}}
        else:
            d = {"status": "kept", "reason": None, "detail": {}}
    d["clean_s"] = clean_s
    if d["status"] == "excluded":
        logger.warning("%s: excluded (%s) %s", rel, d["reason"], d["detail"])
    return d


def subject_status(cfg, decisions):
    """decisions: {(subject, session): decision}. Per subject: t1, t2 (or None), in_c1, in_c3.

    t2-only failure: subject stays in C1, drops out of C3. t1 failure: out of C1 and C3, and the t2
    recording is unused. No t2 recording: in C1 if t1 is kept, never in C3. The split is never touched."""
    s1, s2 = cfg["dataset"]["session_first"], cfg["dataset"]["session_second"]
    out = {}
    for subject in sorted({s for s, _ in decisions}):
        if (subject, s1) not in decisions:
            raise PreprocessError(f"{subject}: no {s1} decision (every subject has a t1 recording)")
        t1 = {k: decisions[(subject, s1)][k] for k in ("status", "reason")}
        t2d = decisions.get((subject, s2))
        t2 = None if t2d is None else {k: t2d[k] for k in ("status", "reason")}
        if t1["status"] == "excluded" and t2 is not None:
            t2 = {"status": "excluded", "reason": "t1_excluded_t2_unused"}
        in_c1 = t1["status"] == "kept"
        in_c3 = in_c1 and t2 is not None and t2["status"] == "kept"
        out[subject] = {"t1": t1, "t2": t2, "in_c1": in_c1, "in_c3": in_c3}
    return out


def apply_exclusions(cfg, decisions):
    """Structure for outputs/exclusions.json (written by write_exclusions, phase 1 only): the halt decision
    (units-check failures over ALL recordings), the per-recording decisions and the per-subject flags."""
    n_failed = sum(d["reason"] == "units_check" for d in decisions.values())
    halt = units_check_halt(cfg, n_failed, len(decisions))
    recs = {f"{s}/{ses}": d for (s, ses), d in sorted(decisions.items())}
    return {"schema": cfg["preprocessing"]["exclusions"]["schema_version"],
            "units_check_halt": {"n_failed": halt.n_failed, "n_total": halt.n_total,
                                 "fraction": halt.fraction, "halt": halt.halt},
            "recordings": recs, "subjects": subject_status(cfg, decisions)}


def write_exclusions(path, structure, *, variant, config_sha256, manifest_summary):
    """UTF-8, sorted keys, indent 2, LF. Called by phase 1 only (never by a test on the real outputs/)."""
    doc = dict(structure, variant=variant, config_sha256=config_sha256, manifest_summary=manifest_summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


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
    emit(f"    all channels' RMS (uV): median {d['rms_median']:.3f}, min {d['rms_min']:.3f}, "
          f"max {d['rms_max']:.3f} | channels below the RMS floor: {d['n_below_floor']} of {d['n_channels']}")
    for name, e in d["electrodes"].items():
        emit(f"      {name:>4}: RMS {e['rms_uv']:8.3f} uV = {e['rms_over_median']:.3f} x median | "
              f"rank {e['rank']:>2} of {d['n_channels']} (1 = lowest) | alpha peak {e['alpha_peak_hz']:6.3f} Hz | "
              f"relative alpha {e['relative_alpha']:.4f} | below floor: {'YES' if e['below_floor'] else 'no'}")


# ---------------------------------------------------------------- bipolar 1-s segment RMS (print-only, IMP-011)

def segment_rms(cfg, x, fs_hz):
    """RMS of every whole, non-overlapping 1-s segment (rejection.segment_length_s); x is (n_ch, n)."""
    seg = int(round(cfg["preprocessing"]["rejection"]["segment_length_s"] * fs_hz))
    n_seg = x.shape[-1] // seg
    if n_seg == 0:
        raise PreprocessError(f"segment_rms: {x.shape[-1]} samples hold no whole {seg}-sample segment")
    blocks = x[:, :n_seg * seg].reshape(x.shape[0], n_seg, seg)
    return np.sqrt(np.mean(blocks ** 2, axis=-1))


def segment_rms_summary(cfg, seg_rms):
    """Percentiles and threshold fractions of one channel's 1-s segment RMS values (uV). Print-only."""
    ed = cfg["preprocessing"]["electrode_diagnostics"]
    lo, hi = rms_bounds(cfg)
    pcts = ed["segment_rms_percentiles"]
    ref = ed["segment_rms_reference_uv"]
    return {"n_segments": int(seg_rms.size), "percentiles": dict(zip(pcts, np.percentile(seg_rms, pcts))),
            "median": float(np.median(seg_rms)), "lo": lo, "ref": ref, "hi": hi,
            "frac_below_lo": float(np.mean(seg_rms < lo)), "frac_below_ref": float(np.mean(seg_rms < ref)),
            "frac_above_hi": float(np.mean(seg_rms > hi))}


def bipolar_segment_report(cfg, root):
    """Pilot subjects only: per recording and bipolar channel, the 1-s segment RMS summary. Nothing stored,
    no rejection applied. Uses the pipeline's notch, band-pass and edge trim (preprocess_recording)."""
    pilot_ids = load_pilot_ids(cfg, root)
    manifest = load_manifest(Path(root) / cfg["paths"]["manifest_file"])
    cache_root = Path(root) / cfg["paths"]["cache_dir"]
    data_root = Path(root) / cfg["paths"]["data_dir"]
    files = pilot_recordings(cfg, root, pilot_ids)
    names = [f"{cfg['preprocessing']['channels'][s][0]}-{cfg['preprocessing']['channels'][s][1]}"
             for s in ("left_pair", "right_pair")]
    rows = []
    for f in files:
        res = preprocess_recording(cfg, f, data_root, manifest, pilot_ids, cache_root=cache_root)
        if res.bipolar is None:
            continue
        fs = res.meta["fs_hz"]
        seg = segment_rms(cfg, res.bipolar, fs)
        for i, name in enumerate(names):
            rows.append({"subject": res.meta["subject"], "session": res.meta["session"], "channel": name,
                         **segment_rms_summary(cfg, seg[i])})
    return names, rows, len(files)


def print_bipolar_segment_report(cfg, names, rows, n_files):
    ed = cfg["preprocessing"]["electrode_diagnostics"]
    pcts = ed["segment_rms_percentiles"]
    lo, ref, hi = rows[0]["lo"], rows[0]["ref"], rows[0]["hi"]
    emit(f"\n[bipolar 1-s segment RMS] pilot subjects, {n_files} recordings, after the pipeline's notch, band-pass "
          f"and edge trim; print-only, no rejection applied")
    emit(f"  {'recording':<16} {'channel':<8} {'n seg':>5} " + " ".join(f"{'p' + str(p):>8}" for p in pcts)
          + f" {'<' + str(lo) + ' uV':>9} {'<' + str(ref) + ' uV':>9} {'>' + str(hi) + ' uV':>9}")
    for r in rows:
        emit(f"  {r['subject'] + ' ' + r['session']:<16} {r['channel']:<8} {r['n_segments']:>5} "
              + " ".join(f"{r['percentiles'][p]:>8.3f}" for p in pcts)
              + f" {r['frac_below_lo']:>9.4f} {r['frac_below_ref']:>9.4f} {r['frac_above_hi']:>9.4f}")
    for name in names:
        meds = [r["median"] for r in rows if r["channel"] == name]
        emit(f"  median over {len(meds)} recordings of the median segment RMS, {name}: {float(np.median(meds)):.3f} uV")
    emit(f"  median over all {len(rows)} recording-channel medians: "
          f"{float(np.median([r['median'] for r in rows])):.3f} uV")


# ---------------------------------------------------------------- pilot report (CLI)

def pilot_recordings(cfg, root, pilot_ids):
    data = Path(root) / cfg["paths"]["data_dir"]
    pattern = cfg["dataset"]["eeg_glob"]
    return [p for s in sorted(pilot_ids) for p in sorted(data.glob(pattern.replace("sub-*", s, 1)))]


def print_edge_report(cfg, variants):
    er = cfg["preprocessing"]["edge_report"]
    limit = er["max_deviation_fraction_of_sd"]
    emit(f"\n[edge transient] 1/f noise + {er['sine_hz']} Hz sine, base seed {er['seed']} "
          f"(0.5 Hz variant {er['n_seeds']} seeds, 0.1 Hz variant {er['n_seeds_slow']} seeds), "
          f"{cfg['dataset']['recording_length_s']} s at "
          f"{cfg['dataset']['native_fs_hz']} Hz; deviation from the transient-free reference as a fraction of "
          f"the filtered signal SD (limit {limit})")
    for v in variants:
        emit(f"\n  high-pass {v['highpass_hz']} Hz, edge trim {v['edge_trim_s']} s: "
              f"worst deviation after trim {v['worst_trimmed_max_frac']:.6f} | worst seconds above the limit: "
              f"start {v['worst_seconds_above_limit_start']:.3f}, end {v['worst_seconds_above_limit_end']:.3f} | "
              f"{'STOP' if v['exceeds'] else 'ok'}")
        emit(f"  {'seed':>10} {'sig SD':>8} {'max dev, no trim':>17} {'s>lim start':>12} {'s>lim end':>10} "
              f"{'max dev after trim':>19} {'verdict':>8}")
        for r in v["per_seed"]:
            emit(f"  {r['seed']:>10} {r['signal_sd']:>8.4f} {r['untrimmed_max_frac']:>17.5f} "
                  f"{r['seconds_above_limit_start']:>12.3f} {r['seconds_above_limit_end']:>10.3f} "
                  f"{r['trimmed_max_frac']:>19.6f} {'STOP' if r['exceeds'] else 'ok':>8}")


def flag_summary(flagged_recordings, flagged_subjects, n_recordings, n_subjects):
    """Fractions of the pilot that the current bad-electrode rule flags."""
    return {"recordings": flagged_recordings, "n_recordings": n_recordings,
            "recording_fraction": flagged_recordings / n_recordings,
            "subjects": len(flagged_subjects), "n_subjects": n_subjects,
            "subject_fraction": len(flagged_subjects) / n_subjects}


def print_flag_summary(s):
    emit(f"\n[current rule] recordings with at least one bad electrode: {s['recordings']} of {s['n_recordings']} "
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
    emit(f"[pilot report] {len(pilot_ids)} pilot subjects, {len(files)} recordings, "
          f"units check high-pass {cfg['preprocessing']['units_check']['highpass_hz']} Hz, "
          f"edge trim {cfg['preprocessing']['edge_trim_s']} s")
    n_fail = 0
    flagged_recordings, flagged_subjects = 0, set()
    for f in files:
        res = preprocess_recording(cfg, f, data_root, manifest, pilot_ids, cache_root=cache_root)
        m = res.meta
        n_fail += not m["units_passed"]
        emit(f"\n{m['subject']} {m['session']}: SHA-256 {'OK' if m['sha256_ok'] else 'FAIL'} ({m['sha256']}) | "
              f"median SD after 0.5 Hz high-pass {m['units_median_sd_uv']:.3f} uV | "
              f"units check {'PASS' if m['units_passed'] else 'FAIL'} | samples after trim {m['n_samples_after_trim']}")
        for name, e in m["electrodes"].items():
            flag = ", ".join(e["flags"]) if e["flags"] else "-"
            emit(f"    {name:>4}: band-passed RMS {e['rms_uv']:8.3f} uV | longest raw run at own max/min "
                  f"{e['saturation_run']:>3} samples (limit {e['saturation_limit']}) | "
                  f"flat 1-s segments {e['flat_segments']} | bad-electrode flag: {flag}")
        if m["bad_electrodes"]:
            emit(f"    BAD ELECTRODES: {', '.join(m['bad_electrodes'])}")
            flagged_recordings += 1
            flagged_subjects.add(m["subject"])
        if electrode_diag and m["units_passed"]:
            rec = load_recording(cfg, f, data_root, manifest, pilot_ids, known_sha256=m["sha256"])
            print_electrode_diagnostics(recording_electrode_diagnostics(cfg, rec))
    halt = units_check_halt(cfg, n_fail, len(files))
    emit(f"\n[units check] {halt.n_failed} of {halt.n_total} failed ({halt.fraction:.4f}); "
          f"halt: {'YES' if halt.halt else 'no'}")
    print_flag_summary(flag_summary(flagged_recordings, flagged_subjects, len(files), len(pilot_ids)))
    variants = edge_transient_report(cfg)
    print_edge_report(cfg, variants)
    if any(v["exceeds"] for v in variants):
        emit("\nSTOP: the deviation after the trim exceeds the limit for at least one variant and seed; "
              "the edge trims are too short. Nothing was changed.")
        return 1
    return 0


def segment_report(cfg, root):
    """B5 + B6 on the pilot subjects' recordings; prints only, stores nothing in config.yml (IMP-012)."""
    pilot_ids = load_pilot_ids(cfg, root)
    manifest = load_manifest(Path(root) / cfg["paths"]["manifest_file"])
    cache_root = Path(root) / cfg["paths"]["cache_dir"]
    data_root = Path(root) / cfg["paths"]["data_dir"]
    files = pilot_recordings(cfg, root, pilot_ids)
    pp = cfg["preprocessing"]
    thr, mu_ref, sigma_ref = pp["ocular"]["detect_threshold_robust_sd"], cfg["rescaling"]["mu_ref"], cfg["rescaling"]["sigma_ref"]
    chan = [f"{pp['channels'][s][0]}-{pp['channels'][s][1]}" for s in ("left_pair", "right_pair")]
    emit(f"[segment report] {len(pilot_ids)} pilot subjects, {len(files)} recordings; B5 then B6 (IMP-012); "
          f"mu_ref {mu_ref}, sigma_ref {sigma_ref}; print-only")
    decisions = {}
    for f in files:
        b4 = preprocess_recording(cfg, f, data_root, manifest, pilot_ids, cache_root=cache_root)
        res = segment_recording(cfg, f, data_root, manifest, pilot_ids, cache_root=cache_root)
        m = res.meta
        dec = recording_decision(cfg, m, res.meta)
        decisions[(m["subject"], m["session"])] = dec
        fracs = None if m["b5"] is None else corrected_fractions(m["b5"]["log"])
        marks = []
        if dec["status"] == "excluded":
            marks.append("excluded")
        if fracs is not None and max(fracs) > pp["ocular"]["report_corrected_fraction"]:
            marks.append(f"corrected fraction above {pp['ocular']['report_corrected_fraction']:.0%}")
        mark = f"  <<<< {'; '.join(marks)}" if marks else ""
        emit(f"\n{m['subject']} {m['session']}: B6 decision {dec['status'].upper()}"
             + (f" ({dec['reason']}: {dec['detail']})" if dec["reason"] else "") + mark)
        if m["b5"] is None:
            emit("    B5 not run (excluded before segmentation)")
            continue
        b5, lg = m["b5"], m["b5"]["log"]
        n_use = lg["n_segments"] * int(round(pp["rejection"]["segment_length_s"] * m["fs_hz"]))
        for i, name in enumerate(chan):
            exc = blink_excursion(cfg, b4.bipolar[i][:n_use], m["fs_hz"])
            bl = lg["blinks"][i]
            emit(f"    {name:>7}: blinks detected {bl['n_windows']}, corrected {bl['n_windows']} "
                  f"({bl['corrected_s']:.2f} s = {fracs[i]:.4f} of the {lg['trimmed_s']:.1f} s trimmed recording, "
                  f"limit {pp['ocular']['max_corrected_fraction']}; {bl['n_flagged_samples']} flagged samples) | largest excursion of the "
                  f"{pp['ocular']['detect_band_hz'][0]}-{pp['ocular']['detect_band_hz'][1]} Hz copy {exc:.2f} robust SD "
                  f"(threshold {thr})")
        r = lg["rules"]
        emit(f"    segments of {lg['n_segments']} rejected {lg['rejected_segments']} (either channel): "
              f"RMS low {r['rms_low']['n_segments']}, RMS high {r['rms_high']['n_segments']}, "
              f"flat-line {r['flat']['n_segments']}, EMG {r['emg']['n_segments']} | per channel (ch0/ch1): "
              f"RMS low {r['rms_low']['per_channel']}, RMS high {r['rms_high']['per_channel']}, "
              f"flat {r['flat']['per_channel']}, EMG {r['emg']['per_channel']}")
        emit(f"    seconds rejected: before padding {lg['rejected_before_padding_s']:.1f}, after padding "
              f"{lg['rejected_after_padding_s']:.1f}; dropped short stretches {lg['dropped_short_stretch_s']:.2f}; "
              f"clean {lg['clean_s']:.2f} s in {lg['n_clean_segments']} segments (cropped tail {lg['cropped_tail_s']:.3f} s)")
        if res.segments:
            allx = np.concatenate(res.segments, axis=-1)
            ddof = pp["rescale"]["ddof"]
            bf = b5["clean_stats_before"]
            for i, name in enumerate(chan):
                emit(f"    {name:>7}: clean-sample mean/SD before rescaling {bf['mean'][i]:.6f} / {bf['sd'][i]:.6f} | "
                      f"after {allx[i].mean():.12f} / {allx[i].std(ddof=ddof):.12f} (must be {mu_ref} / {sigma_ref})")
            v = b5["vigilance"]
            emit(f"    vigilance alpha/theta over {len(v['epoch_starts'])} epochs: mean "
                  + ", ".join(f"{name} {mv:.3f}" for name, mv in zip(chan, v["mean"])))
    struct = apply_exclusions(cfg, decisions)
    h = struct["units_check_halt"]
    emit(f"\n[units check] {h['n_failed']} of {h['n_total']} failed (fraction {h['fraction']:.4f}); halt: "
          f"{'YES' if h['halt'] else 'no'} -- not meaningful with n={h['n_total']} "
          f"(one failure would be {1 / h['n_total']:.2%})")
    emit("\n[subjects] session status and derived flags")
    for subj, st in struct["subjects"].items():
        t2 = "none" if st["t2"] is None else f"{st['t2']['status']}" + (f" ({st['t2']['reason']})" if st["t2"]["reason"] else "")
        t1 = st["t1"]["status"] + (f" ({st['t1']['reason']})" if st["t1"]["reason"] else "")
        flag = "  <<<<" if not (st["in_c1"] and st["in_c3"]) and (st["t1"]["status"] == "excluded" or (
            st["t2"] is not None and st["t2"]["status"] == "excluded")) else ""
        emit(f"  {subj}: t1 {t1}; t2 {t2}; in_c1 {st['in_c1']}; in_c3 {st['in_c3']}{flag}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3/B4 recording preprocessing")
    parser.add_argument("--pilot-report", action="store_true",
                        help="run B3 and B4 on the pilot subjects' recordings and print a report")
    parser.add_argument("--electrode-diagnostics", action="store_true",
                        help="with --pilot-report: print read-only all-channel RMS and alpha diagnostics")
    parser.add_argument("--segment-report", action="store_true",
                        help="B5 + B6 on the pilot subjects' recordings: print-only report (IMP-012)")
    parser.add_argument("--bipolar-segment-report", action="store_true",
                        help="print-only 1-s segment RMS diagnostic of the bipolar channels (pilot subjects only)")
    args = parser.parse_args(argv)
    if not (args.pilot_report or args.bipolar_segment_report or args.segment_report):
        parser.error("nothing to do; use --pilot-report, --bipolar-segment-report or --segment-report")
    cfg = load_config()
    log_dir = REPO_ROOT / cfg["paths"]["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "preprocess_pilot_report.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("pipeline").addHandler(handler)
    logging.getLogger("pipeline").setLevel(logging.INFO)
    if args.segment_report:
        return segment_report(cfg, REPO_ROOT)
    if args.bipolar_segment_report:
        names, rows, n_files = bipolar_segment_report(cfg, REPO_ROOT)
        print_bipolar_segment_report(cfg, names, rows, n_files)
        return 0
    return pilot_report(cfg, REPO_ROOT, electrode_diag=args.electrode_diagnostics)


if __name__ == "__main__":
    sys.exit(main())
