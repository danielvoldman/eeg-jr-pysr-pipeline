"""B3/B4 tests: loading, checksum, units check, labels, bad electrodes, notch, bipolar, variants,
guard, cache (§4.2, §5.1, §6; IMP-009).

Expected values are independent of the code under test: closed-form notch gain with plain math,
noise scaled to exact RMS values, a test-local EDF writer. No test touches the real data/, cache/,
outputs/, results/ or logs/; files live under tmp_path.
"""
import ast
import cmath
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import sosfiltfilt

from src import preprocess
from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
FS = float(CFG["dataset"]["native_fs_hz"])
PP = CFG["preprocessing"]
SEED = 20260929          # test-local noise seed; not a pipeline seed
N_SAMPLES = int(30 * FS)  # 30 s recordings: longer than the 10 s edge padding
EEG_NAMES = ["P3", "PO3", "P4", "PO4", "Fz", "Cz", "Oz", "O1"]
SUBJECT, SESSION = "sub-001", "ses-t1"


# ---------------------------------------------------------------- test-local EDF writer (EDF spec)

def _field(text, width):
    return text.encode("ascii").ljust(width)[:width]


def _num(value):
    for p in range(6, -1, -1):
        s = f"{value:.{p}f}"
        if len(s) <= 8:
            return s
    raise ValueError(value)


def write_edf(path, names, data_uv, fs, unit="uV"):
    """16-bit EDF, 1-s records. data_uv: (n_channels, n), a multiple of fs samples."""
    ns, n = data_uv.shape
    spr = int(fs)
    assert n % spr == 0
    nrec = n // spr
    pmins, pmaxs, digital = [], [], np.empty((ns, n), dtype="<i2")
    dmin, dmax = -32768, 32767
    for i in range(ns):
        rng = data_uv[i].max() - data_uv[i].min()
        lo = float(_num(data_uv[i].min() - 0.01 * rng - 0.01))
        hi = float(_num(data_uv[i].max() + 0.01 * rng + 0.01))
        assert lo <= data_uv[i].min() and hi >= data_uv[i].max()
        pmins.append(_num(lo))
        pmaxs.append(_num(hi))
        digital[i] = np.round((data_uv[i] - lo) / (hi - lo) * (dmax - dmin) + dmin).astype("<i2")
    head = b"".join([
        _field("0", 8), _field("X X X X", 80), _field("Startdate X X X X", 80),
        _field("01.01.20", 8), _field("00.00.00", 8), _field(str(256 * (ns + 1)), 8),
        _field("", 44), _field(str(nrec), 8), _field("1", 8), _field(str(ns), 4)])
    cols = [
        [_field(n_, 16) for n_ in names], [_field("", 80)] * ns, [_field(unit, 8)] * ns,
        [_field(s, 8) for s in pmins], [_field(s, 8) for s in pmaxs],
        [_field(str(dmin), 8)] * ns, [_field(str(dmax), 8)] * ns,
        [_field("", 80)] * ns, [_field(str(spr), 8)] * ns, [_field("", 32)] * ns]
    head += b"".join(b"".join(c) for c in cols)
    body = digital.reshape(ns, nrec, spr).transpose(1, 0, 2).tobytes()
    Path(path).write_bytes(head + body)


def make_recording(root, data_uv, names, types, subject=SUBJECT, session=SESSION, fs=FS):
    """Write EDF + channels.tsv under root/data; return (edf_path, manifest, data_root)."""
    data_root = Path(root) / "data"
    folder = data_root / subject / session / "eeg"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{session}_task-resteyesc"
    edf = folder / f"{stem}_eeg.edf"
    write_edf(edf, names, data_uv, fs)
    tsv = "name\ttype\tunits\tsampling_frequency\n" + "".join(
        f"{n}\t{t}\tuV\t{int(fs)}\n" for n, t in zip(names, types))
    (folder / f"{stem}_channels.tsv").write_text(tsv, encoding="utf-8")
    return edf, manifest_for(edf, data_root), data_root


def manifest_for(edf, data_root):
    rel = Path(edf).relative_to(data_root).as_posix()
    blob = Path(edf).read_bytes()
    return {rel: (hashlib.sha256(blob).hexdigest(), len(blob))}


def white(rng, sd, n=N_SAMPLES):
    return sd * rng.standard_normal(n)


def standard_data(seed=SEED, sd=60.0, overrides=None):
    rng = np.random.default_rng(seed)
    x = np.stack([white(rng, sd) for _ in EEG_NAMES])
    for name, scale in (overrides or {}).items():
        x[EEG_NAMES.index(name)] *= scale
    return x


@pytest.fixture
def rec(tmp_path):
    edf, man, root = make_recording(tmp_path, standard_data(), EEG_NAMES, ["EEG"] * len(EEG_NAMES))
    return edf, man, root, frozenset({SUBJECT})


def run(rec, **kw):
    edf, man, root, pilots = rec
    return preprocess.preprocess_recording(CFG, edf, root, man, pilots, **kw)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# ---------------------------------------------------------------- units check

def _noise_block(scale, n_ch=64, seed=SEED):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((n_ch, N_SAMPLES))


def test_units_check_white_noise_20uv_passes():
    r = preprocess.units_check(CFG, _noise_block(20.0), FS)
    assert r.passed and 19.0 < r.median_sd_uv < 21.0


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def test_units_check_fails_for_scaling_errors(factor):
    r = preprocess.units_check(CFG, _noise_block(20.0 * factor), FS)
    assert not r.passed


def test_units_check_uses_the_median():
    x = _noise_block(20.0)
    x[0] *= 1e4                       # one huge channel among 64
    assert preprocess.units_check(CFG, x, FS).passed
    y = _noise_block(20.0)
    y[:33] *= 1e4                     # a majority of huge channels
    assert not preprocess.units_check(CFG, y, FS).passed


def test_units_check_removes_slow_drift():
    t = np.arange(N_SAMPLES) / FS
    noise = _noise_block(20.0)
    drifted = noise + 5000.0 * np.sin(2 * np.pi * 0.1 * t)       # 5 mV drift at 0.1 Hz
    base = preprocess.units_check(CFG, noise, FS).median_sd_uv
    r = preprocess.units_check(CFG, drifted, FS)
    assert r.passed and abs(r.median_sd_uv - base) < 0.05 * base


def test_highpass_is_the_high_pass_half_of_the_bandpass_design():
    x = _noise_block(20.0, n_ch=2)
    order = CFG["preprocessing"]["bandpass"]["filter_order"]
    sos = preprocess.bandpass_sos(CFG, FS)[:order // 2]        # high-pass sections come first
    padlen = int(round(CFG["preprocessing"]["bandpass"]["edge_pad_s"] * FS))
    want = sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=padlen)
    assert np.array_equal(preprocess.highpass(CFG, x, FS), want)


# ---------------------------------------------------------------- 5 percent halt rule

@pytest.mark.parametrize("failed,total,halt", [
    (7, 153, False),      # 4.58 percent
    (8, 153, True),       # 5.23 percent; 5 percent of 153 is 7.65
    (7, 154, False),
    (8, 154, True),       # 5 percent of 154 is 7.7
    (8, 160, False),      # exactly 5 percent is not "more than"
    (9, 160, True),
    (0, 153, False),
])
def test_halt_rule_boundaries(failed, total, halt):
    d = preprocess.units_check_halt(CFG, failed, total)
    assert d.halt is halt
    assert d.fraction == failed / total


def test_halt_rule_rejects_bad_counts():
    with pytest.raises(preprocess.PreprocessError):
        preprocess.units_check_halt(CFG, 1, 0)
    with pytest.raises(preprocess.PreprocessError):
        preprocess.units_check_halt(CFG, 5, 4)


# ---------------------------------------------------------------- checksum and manifest

def test_manifest_line_is_parsed(tmp_path):
    p = tmp_path / "MANIFEST.sha256"
    p.write_text("abc123\t42\tsub-001/ses-t1/eeg/a.edf\n# files=1 manifest_sha256=x\n", encoding="utf-8")
    assert preprocess.load_manifest(p) == {"sub-001/ses-t1/eeg/a.edf": ("abc123", 42)}


def test_manifest_malformed_line_raises(tmp_path):
    p = tmp_path / "MANIFEST.sha256"
    p.write_text("abc123\tnotanumber\tx\n", encoding="utf-8")
    with pytest.raises(preprocess.ChecksumError):
        preprocess.load_manifest(p)


def test_verify_accepts_intact_file_and_records_sha(rec):
    edf, man, root, _ = rec
    rel = edf.relative_to(root).as_posix()
    assert preprocess.verify_file(CFG, edf, rel, man) == man[rel][0]


def test_modified_file_is_refused(rec):
    edf, man, root, pilots = rec
    blob = bytearray(edf.read_bytes())
    blob[-1] ^= 1                                   # same size, one bit different
    edf.write_bytes(bytes(blob))
    with pytest.raises(preprocess.ChecksumError, match="SHA-256"):
        preprocess.load_recording(CFG, edf, root, man, pilots)


def test_truncated_file_is_refused(rec):
    edf, man, root, pilots = rec
    edf.write_bytes(edf.read_bytes()[:-1024])
    with pytest.raises(preprocess.ChecksumError, match="size"):
        preprocess.load_recording(CFG, edf, root, man, pilots)


def test_file_not_in_manifest_is_refused(rec):
    edf, _, root, pilots = rec
    with pytest.raises(preprocess.ChecksumError, match="not listed"):
        preprocess.load_recording(CFG, edf, root, {}, pilots)


# ---------------------------------------------------------------- EDF round trip and channels

def test_edf_round_trip_returns_microvolts(rec):
    edf, man, root, pilots = rec
    loaded = preprocess.load_recording(CFG, edf, root, man, pilots)
    want = standard_data()
    step = (want.max(axis=1) - want.min(axis=1)) / 65535
    assert loaded.data_uv.dtype == np.float64
    assert loaded.ch_names == EEG_NAMES and loaded.fs_hz == FS
    assert np.all(np.abs(loaded.data_uv - want).max(axis=1) <= step + 1e-9)
    assert 20.0 < loaded.data_uv.std() < 100.0          # uV scale, not 1e-6 of it
    import mne
    raw = mne.io.read_raw_edf(edf, units=CFG["dataset"]["read_units"], preload=True, verbose="ERROR")
    assert abs(raw.get_data().std() * 1e6 - loaded.data_uv.std()) < 1e-6 * loaded.data_uv.std() * 1e3


def test_only_eeg_type_channels_enter_the_units_check(tmp_path):
    names = EEG_NAMES + [f"X{i}" for i in range(9)]
    types = ["EEG"] * len(EEG_NAMES) + ["MISC"] * 9
    data = np.concatenate([standard_data(), 20000.0 * standard_data(seed=SEED + 1)[:1].repeat(9, axis=0)])
    edf, man, root = make_recording(tmp_path, data, names, types)
    loaded = preprocess.load_recording(CFG, edf, root, man, frozenset({SUBJECT}))
    assert loaded.ch_names == EEG_NAMES
    assert preprocess.units_check(CFG, loaded.data_uv, FS).passed     # 9 huge non-EEG channels would fail it


def test_missing_channel_types_stop(tmp_path):
    edf, man, root = make_recording(tmp_path, standard_data(), EEG_NAMES, ["EEG"] * len(EEG_NAMES))
    tsv = preprocess.channels_tsv_path(CFG, edf)
    tsv.write_text("name\tunits\n" + "".join(f"{n}\tuV\n" for n in EEG_NAMES), encoding="utf-8")
    with pytest.raises(preprocess.ChannelsError):
        preprocess.load_recording(CFG, edf, root, man, frozenset({SUBJECT}))
    tsv.write_text("name\ttype\n" + "".join(f"{n}\t\n" for n in EEG_NAMES), encoding="utf-8")
    with pytest.raises(preprocess.ChannelsError, match="types are missing"):
        preprocess.load_recording(CFG, edf, root, man, frozenset({SUBJECT}))


@pytest.mark.parametrize("missing", ["P3", "PO3", "P4", "PO4"])
def test_missing_label_fails_loudly_naming_the_label(missing):
    rows = [(n, "EEG") for n in EEG_NAMES if n != missing]
    with pytest.raises(preprocess.ChannelsError, match=missing + "'"):
        preprocess.verify_labels(CFG, rows, EEG_NAMES)
    with pytest.raises(preprocess.ChannelsError, match=missing + "'"):
        preprocess.verify_labels(CFG, [(n, "EEG") for n in EEG_NAMES], [n for n in EEG_NAMES if n != missing])


def test_missing_p4_in_a_real_file_fails(tmp_path):
    names = [n for n in EEG_NAMES if n != "P4"]
    data = np.delete(standard_data(), EEG_NAMES.index("P4"), axis=0)
    edf, man, root = make_recording(tmp_path, data, names, ["EEG"] * len(names))
    with pytest.raises(preprocess.ChannelsError, match="'P4'"):
        preprocess.load_recording(CFG, edf, root, man, frozenset({SUBJECT}))


def test_sub010_ses_t1_needs_no_scans_tsv(tmp_path):
    edf, man, root = make_recording(tmp_path, standard_data(), EEG_NAMES, ["EEG"] * len(EEG_NAMES),
                                    subject="sub-010", session="ses-t1")
    assert not list((root / "sub-010").rglob("*scans.tsv"))
    res = preprocess.preprocess_recording(CFG, edf, root, man, frozenset({"sub-010"}))
    assert res.meta["units_passed"] and res.bipolar is not None


def test_failed_units_check_excludes_and_is_logged(tmp_path, caplog):
    edf, man, root = make_recording(tmp_path, standard_data(sd=60.0) * 1e-6,
                                    EEG_NAMES, ["EEG"] * len(EEG_NAMES))
    with caplog.at_level("WARNING", logger="pipeline.preprocess"):
        res = preprocess.preprocess_recording(CFG, edf, root, man, frozenset({SUBJECT}))
    assert res.bipolar is None and res.meta["excluded_reasons"] == ["units_check"]
    assert any("units check failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------- bad electrodes

def _scaled_noise(target_rms, seed=SEED, n=N_SAMPLES):
    x = np.random.default_rng(seed).standard_normal(n)
    return x / rms(x) * target_rms


def _report(filtered, raw=None, strict=False):
    raw = filtered if raw is None else raw
    return preprocess.electrode_report(CFG, raw, filtered, FS, strict)


@pytest.mark.parametrize("target,flag", [
    (4.99, "rms_below_min"), (5.01, None), (149.9, None), (150.1, "rms_above_max")])
def test_rms_bounds_default(target, flag):
    flags = _report(_scaled_noise(target))["flags"]
    assert flags == ([flag] if flag else [])


@pytest.mark.parametrize("target,flagged", [(99.9, False), (100.1, True)])
def test_rms_bound_strict_is_100(target, flagged):
    assert bool(_report(_scaled_noise(target), strict=True)["flags"]) is flagged
    assert not _report(_scaled_noise(target), strict=False)["flags"]


def test_flat_one_second_stretch_is_flagged():
    x = _scaled_noise(20.0)
    seg = int(FS)
    x[5 * seg:6 * seg] = 3.0                                         # exactly one whole 1-s segment
    r = _report(x)
    assert "flatline" in r["flags"] and r["flat_segments"] == 1
    y = _scaled_noise(20.0)
    y[5 * seg:6 * seg] = np.linspace(0, 0.099, seg)                  # peak-to-peak 0.099 uV: flat
    assert _report(y)["flat_segments"] == 1
    y[5 * seg:6 * seg] = np.linspace(0, 0.101, seg)                  # 0.101 uV: not flat
    assert _report(y)["flat_segments"] == 0


N_SAT = math.ceil(PP["bad_channel"]["saturation_min_run_s"] * FS)


def test_saturation_limit_is_ceil_of_run_seconds_times_fs():
    assert preprocess.saturation_limit_samples(CFG, FS) == N_SAT == 52


@pytest.mark.parametrize("side", ["max", "min"])
def test_plateau_of_exactly_n_samples_at_extreme_is_saturated(side):
    def make(length):
        raw = _scaled_noise(20.0, seed=SEED + 3)
        raw[1000:1000 + length] = raw.max() + 1.0 if side == "max" else raw.min() - 1.0
        return raw
    assert preprocess.saturation_run(make(N_SAT)) == N_SAT
    assert "saturation" in _report(_scaled_noise(20.0), raw=make(N_SAT))["flags"]
    assert "saturation" not in _report(_scaled_noise(20.0), raw=make(N_SAT - 1))["flags"]


def test_quantisation_like_runs_of_seven_do_not_flag():
    raw = _scaled_noise(20.0, seed=SEED + 4)
    raw[2000:2007] = raw.max() + 1.0
    raw[4000:4007] = raw.min() - 1.0
    assert preprocess.saturation_run(raw) == 7
    assert "saturation" not in _report(_scaled_noise(20.0), raw=raw)["flags"]


def test_constant_channel_is_saturated_and_flat():
    flags = _report(np.zeros(N_SAMPLES), raw=np.full(N_SAMPLES, 12.5))["flags"]
    assert "saturation" in flags and "flatline" in flags and "rms_below_min" in flags


def test_plateau_not_at_extreme_is_not_saturation_but_may_be_flat():
    raw = _scaled_noise(20.0, seed=SEED + 5)
    seg = int(FS)
    raw[3 * seg:4 * seg] = 0.0                       # a long plateau in the middle of the range
    assert raw.max() > 0 > raw.min()
    assert preprocess.saturation_run(raw) < N_SAT
    r = _report(raw, raw=raw)
    assert "saturation" not in r["flags"] and "flatline" in r["flags"]


def test_ordinary_recording_with_one_bad_electrode_names_it(tmp_path):
    edf, man, root = make_recording(tmp_path, standard_data(overrides={"P4": 0.05}),
                                    EEG_NAMES, ["EEG"] * len(EEG_NAMES))
    res = preprocess.preprocess_recording(CFG, edf, root, man, frozenset({SUBJECT}))
    assert res.meta["bad_electrodes"] == ["P4"]
    assert res.meta["electrodes"]["P4"]["flags"] == ["rms_below_min"]
    assert all(not res.meta["electrodes"][n]["flags"] for n in ("P3", "PO3", "PO4"))
    assert res.bipolar is not None                    # exclusion itself is B6


def test_only_the_four_electrodes_are_judged(tmp_path):
    data = standard_data(overrides={"Fz": 1e-3})     # a dead non-pair channel
    edf, man, root = make_recording(tmp_path, data, EEG_NAMES, ["EEG"] * len(EEG_NAMES))
    res = preprocess.preprocess_recording(CFG, edf, root, man, frozenset({SUBJECT}))
    assert res.meta["bad_electrodes"] == [] and set(res.meta["electrodes"]) == {"P3", "PO3", "P4", "PO4"}


# ---------------------------------------------------------------- notch

Q = PP["line_noise"]["notch_q"]


def notch_gain(f, f0):
    """|H|^2 of one iirnotch section applied forward and backward, from the bilinear-transform formula."""
    w0 = 2 * math.pi * f0 / FS
    beta = math.tan(w0 / Q / 2)
    b = (1 / (1 + beta), -2 * math.cos(w0) / (1 + beta), 1 / (1 + beta))
    a = (1.0, -2 * math.cos(w0) / (1 + beta), (1 - beta) / (1 + beta))
    zi = cmath.exp(-1j * 2 * math.pi * f / FS)
    return abs((b[0] + b[1] * zi + b[2] * zi ** 2) / (a[0] + a[1] * zi + a[2] * zi ** 2)) ** 2


def total_gain(f):
    return math.prod(notch_gain(f, f0) for f0 in range(50, 512, 50))


def _amp_phase(y, f):
    n = y.size
    t = np.arange(n) / FS
    mid = slice(n // 4, 3 * n // 4)
    basis = np.column_stack([np.sin(2 * np.pi * f * t[mid]), np.cos(2 * np.pi * f * t[mid])])
    c, *_ = np.linalg.lstsq(basis, y[mid], rcond=None)
    return math.hypot(*c), math.atan2(c[1], c[0])


def _sine(f, n=N_SAMPLES):
    return np.sin(2 * np.pi * f * np.arange(n) / FS)


def test_notch_frequencies_are_line_and_harmonics_below_nyquist():
    assert preprocess.notch_frequencies(CFG, FS) == [float(f) for f in range(50, 512, 50)]


def test_notch_removes_50hz_and_keeps_10hz_with_analytic_gain():
    for f in (10.0, 45.0, 49.0, 51.0, 55.0):
        amp, _ = _amp_phase(preprocess.notch(CFG, _sine(f), FS), f)
        assert amp == pytest.approx(total_gain(f), rel=1e-3, abs=1e-6), f
    assert total_gain(10.0) > 0.99                                    # documented: 10 Hz untouched
    amp50, _ = _amp_phase(preprocess.notch(CFG, _sine(50.0), FS), 50.0)
    assert amp50 < 1e-4 and total_gain(50.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("f", [100.0, 150.0, 250.0, 400.0, 500.0])
def test_notch_removes_harmonics_up_to_500hz(f):
    amp, _ = _amp_phase(preprocess.notch(CFG, _sine(f), FS), f)
    assert amp < 1e-3


def test_notch_is_zero_phase_at_10hz():
    _, phase_in = _amp_phase(_sine(10.0), 10.0)
    _, phase_out = _amp_phase(preprocess.notch(CFG, _sine(10.0), FS), 10.0)
    assert abs(phase_out - phase_in) < 1e-6


# ---------------------------------------------------------------- bipolar and edge trim

def test_bipolar_is_exact_difference_of_filtered_electrodes(rec):
    edf, man, root, pilots = rec
    loaded = preprocess.load_recording(CFG, edf, root, man, pilots)
    hp, trim = PP["bandpass"]["highpass_hz"], PP["edge_trim_s"]
    filt = {n: preprocess.trim_edges(
        preprocess.bandpass(CFG, preprocess.notch(CFG, loaded.data_uv[loaded.ch_names.index(n)], FS), FS, hp),
        FS, trim) for n in ("P3", "PO3", "P4", "PO4")}
    res = run(rec)
    assert res.bipolar.dtype == np.float64 and res.bipolar.shape[0] == 2
    assert np.array_equal(res.bipolar[0], filt["P3"] - filt["PO3"])
    assert np.array_equal(res.bipolar[1], filt["P4"] - filt["PO4"])


def test_common_mode_cancels_in_the_pair():
    rng = np.random.default_rng(SEED + 7)
    a, b, common = (rng.standard_normal(N_SAMPLES) * 30 for _ in range(3))
    common = common + 200.0 * _sine(10.0)

    def chain(x):
        return preprocess.bandpass(CFG, preprocess.notch(CFG, x, FS), FS)
    with_cm = preprocess.bipolar(CFG, {"P3": chain(a + common), "PO3": chain(b + common),
                                       "P4": chain(a), "PO4": chain(b)})
    assert np.allclose(with_cm[0], chain(a) - chain(b), atol=1e-9)
    assert np.allclose(with_cm[0], with_cm[1], atol=1e-9)


def test_trim_edges_removes_exactly_the_configured_seconds_from_each_end():
    x = np.arange(N_SAMPLES, dtype=np.float64)
    k = int(round(PP["edge_trim_s"] * FS))
    y = preprocess.trim_edges(x, FS, PP["edge_trim_s"])
    assert y.size == N_SAMPLES - 2 * k and y[0] == k and y[-1] == N_SAMPLES - 1 - k
    ks = int(round(PP["edge_trim_s_sensitivity_highpass"] * FS))
    z = preprocess.trim_edges(x, FS, preprocess.edge_trim_seconds(CFG, True))
    assert PP["edge_trim_s_sensitivity_highpass"] == 10.0 and z.size == N_SAMPLES - 2 * ks


def test_recording_lengths_after_trim(rec):
    assert run(rec).meta["n_samples_after_trim"] == N_SAMPLES - 2 * int(FS)
    assert run(rec, sensitivity_highpass=True).meta["n_samples_after_trim"] == N_SAMPLES - 2 * int(10 * FS)


# ---------------------------------------------------------------- variants

def test_strict_flag_changes_only_the_rms_bound(tmp_path):
    data = standard_data(overrides={"PO3": 6.8})       # band-passed RMS about 120 uV: inside 150, outside 100
    edf, man, root = make_recording(tmp_path, data, EEG_NAMES, ["EEG"] * len(EEG_NAMES))
    pilots = frozenset({SUBJECT})
    base = preprocess.preprocess_recording(CFG, edf, root, man, pilots)
    strict = preprocess.preprocess_recording(CFG, edf, root, man, pilots, strict=True)
    assert 100 < base.meta["electrodes"]["PO3"]["rms_uv"] < 150
    assert base.meta["bad_electrodes"] == [] and strict.meta["bad_electrodes"] == ["PO3"]
    assert strict.meta["electrodes"]["PO3"]["flags"] == ["rms_above_max"]
    assert np.array_equal(base.bipolar, strict.bipolar)
    for key in ("units_median_sd_uv", "units_passed", "n_samples_after_trim", "highpass_hz", "edge_trim_s"):
        assert base.meta[key] == strict.meta[key]
    for n, rep in base.meta["electrodes"].items():
        assert {k: v for k, v in rep.items() if k != "flags"} == \
               {k: v for k, v in strict.meta["electrodes"][n].items() if k != "flags"}


def test_sensitivity_highpass_flag_changes_only_the_highpass_and_its_trim(rec):
    base, sens = run(rec), run(rec, sensitivity_highpass=True)
    assert base.meta["highpass_hz"] == 0.5 and sens.meta["highpass_hz"] == PP["sensitivity_highpass_hz"] == 0.1
    assert base.meta["units_median_sd_uv"] == sens.meta["units_median_sd_uv"]   # units check stays at 0.5 Hz
    edf, man, root, pilots = rec
    loaded = preprocess.load_recording(CFG, edf, root, man, pilots)
    x = {n: loaded.data_uv[loaded.ch_names.index(n)] for n in ("P3", "PO3", "P4", "PO4")}
    filt = {n: preprocess.trim_edges(preprocess.bandpass(CFG, preprocess.notch(CFG, v, FS), FS, 0.1),
                                     FS, PP["edge_trim_s_sensitivity_highpass"]) for n, v in x.items()}
    assert np.array_equal(sens.bipolar, preprocess.bipolar(CFG, filt))
    assert not np.array_equal(base.bipolar[:, :sens.bipolar.shape[1]], sens.bipolar)


# ---------------------------------------------------------------- development guard

def test_guard_refuses_non_pilot_and_accepts_pilot(rec):
    edf, man, root, _ = rec
    with pytest.raises(preprocess.DevelopmentGuardError, match=SUBJECT):
        preprocess.load_recording(CFG, edf, root, man, frozenset({"sub-999"}))
    with pytest.raises(preprocess.DevelopmentGuardError):
        preprocess.preprocess_recording(CFG, edf, root, man, frozenset({"sub-999"}))
    with pytest.raises(preprocess.DevelopmentGuardError):
        preprocess.load_recording(CFG, edf, root, man, None)
    assert preprocess.load_recording(CFG, edf, root, man, frozenset({SUBJECT})).subject == SUBJECT
    assert preprocess.load_recording(CFG, edf, root, man, frozenset(), allow_all=True).subject == SUBJECT


def test_load_pilot_ids_reads_the_split_json(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / f"split_{CFG['split']['primary_seed']}.json").write_text(
        '{"pilot": ["sub-002", "sub-001"]}', encoding="utf-8")
    assert preprocess.load_pilot_ids(CFG, tmp_path) == frozenset({"sub-001", "sub-002"})


# ---------------------------------------------------------------- cache

@pytest.fixture
def count_loads(monkeypatch):
    calls = []
    orig = preprocess.load_recording

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    monkeypatch.setattr(preprocess, "load_recording", counting)
    return calls


def test_cache_reuse_and_result_identical(rec, tmp_path, count_loads):
    cache = tmp_path / "cache"
    first = run(rec, cache_root=cache)
    second = run(rec, cache_root=cache)
    assert len(count_loads) == 1 and len(list(cache.rglob("*.npz"))) == 1
    assert np.array_equal(first.bipolar, second.bipolar)
    assert {k: v for k, v in second.meta.items() if k != "bipolar_stored"} == first.meta
    assert second.bipolar.dtype == np.float64


def test_cache_invalidated_by_config_leaf(rec, tmp_path, count_loads):
    cache = tmp_path / "cache"
    edf, man, root, pilots = rec
    run(rec, cache_root=cache)
    cfg2 = copy.deepcopy(CFG)
    cfg2["preprocessing"]["line_noise"]["notch_q"] = Q + 1
    preprocess.preprocess_recording(cfg2, edf, root, man, pilots, cache_root=cache)
    assert len(count_loads) == 2 and len(list(cache.rglob("*.npz"))) == 2


def test_cache_invalidated_by_file_hash_and_variant(rec, tmp_path, count_loads):
    cache = tmp_path / "cache"
    edf, man, root, pilots = rec
    run(rec, cache_root=cache)
    run(rec, cache_root=cache, strict=True)                      # variant flag is part of the key
    assert len(count_loads) == 2
    blob = bytearray(edf.read_bytes())
    blob[-1] ^= 1
    edf.write_bytes(bytes(blob))
    new_man = manifest_for(edf, root)
    preprocess.preprocess_recording(CFG, edf, root, new_man, pilots, cache_root=cache)
    assert len(count_loads) == 3


def test_cache_keys_differ_with_inputs():
    k = preprocess.cache_key(CFG, "a" * 64, False, False)
    assert k == preprocess.cache_key(CFG, "a" * 64, False, False)
    assert k != preprocess.cache_key(CFG, "b" * 64, False, False)
    assert k != preprocess.cache_key(CFG, "a" * 64, True, False)
    assert k != preprocess.cache_key(CFG, "a" * 64, False, True)


# ---------------------------------------------------------------- hygiene

def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names.add(n.module or "")
    return names


def test_preprocess_imports_are_restricted_and_never_model():
    path = REPO_ROOT / "src" / "preprocess.py"
    names = _imports(path)
    allowed = {"argparse", "csv", "hashlib", "json", "logging", "math", "sys", "dataclasses", "fractions",
               "pathlib", "numpy", "scipy.signal", "src.config", "mne"}
    assert names <= {n.split(".")[0] for n in allowed} | allowed, names
    tree = ast.parse(path.read_text(encoding="utf-8"))
    full = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and m.split(".")[-1] == "model" for m in full), full


def test_no_numeric_literals_beyond_allowed_in_preprocess():
    tree = ast.parse((REPO_ROOT / "src" / "preprocess.py").read_text(encoding="utf-8"))
    allowed = {0, 1, 2, 0.5}
    bad = [(n.value, n.lineno) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool) and n.value not in allowed]
    assert not bad, f"numeric literals in src/preprocess.py: {bad}"


def test_imp009_config_leaves_are_placeholders_and_set():
    raw = __import__("yaml").safe_load((REPO_ROOT / "config.yml").read_text(encoding="utf-8"))
    pp = raw["preprocessing"]
    for leaf in (pp["bad_channel"]["flatline_threshold"], pp["bad_channel"]["saturation_min_run_s"],
                 pp["line_noise"]["notch_q"], pp["edge_trim_s"], pp["edge_trim_s_sensitivity_highpass"],
                 pp["units_check"]["channel_type"]):
        assert leaf["prov"] == "placeholder" and leaf["value"] is not None
    assert "saturation_threshold" not in pp["bad_channel"] and "notch_width" not in pp["line_noise"]
    assert pp["bad_channel"]["saturation_min_run_s"]["value"] == 0.05
    assert pp["line_noise"]["notch_q"]["value"] == 30
    assert pp["edge_trim_s"]["value"] == 1.0 and pp["edge_trim_s_sensitivity_highpass"]["value"] == 10.0


def test_functions_write_nothing_to_run_time_folders(rec):
    folders = [REPO_ROOT / CFG["paths"][k] for k in ("cache_dir", "outputs_dir", "results_dir", "logs_dir")]

    def snapshot():
        return {str(p): p.stat().st_mtime_ns for d in folders if d.exists() for p in [d, *d.rglob("*")]}
    before = snapshot()
    run(rec)
    assert snapshot() == before
