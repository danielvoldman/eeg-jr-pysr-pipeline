"""B5 tests: blink detection and correction, 1-s segment rejection, EMG screen, padding, mask mapping,
clean segments, rescaling, vigilance, cache (§5.1, §10.2; IMP-012).

Expected values are independent of the code under test: exact-RMS scaling with plain numpy, a
test-local FFT power computation, hand-made masks, closed-form sample counts. No test touches the real
data/, cache/, outputs/, results/ or logs/.
"""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from src import preprocess
from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
FS = 1024                 # native rate (§4.1), also the rate the B5 rules run at
FS_OUT = 256              # observation rate (§10.1)
SEG, SEG_OUT = FS, FS_OUT  # samples in 1 s
BLINK_SEEDS = (20260930, 20260931, 20260932)   # test-local seeds, not pipeline seeds
BLINK_FWHM_S = 0.3        # "0.3 s wide" is read as the full width at half maximum (IMP-012)
BLINK_SIGMA_S = BLINK_FWHM_S / (2.0 * np.sqrt(2.0 * np.log(2.0)))
BACKGROUND_SD_UV = 10.0
BLINK_UV = 150.0
NOISE_SEED = 20260929


def noise(seed, n_ch, n_s, sd=BACKGROUND_SD_UV):
    return sd * np.random.default_rng(seed).standard_normal((n_ch, int(n_s * FS)))


def exact_rms(z, rms):
    """Scale z so that its RMS (no mean removal) is exactly `rms`."""
    return z * (rms / np.sqrt(np.mean(z ** 2)))


def fluent(cfg=CFG):
    return copy.deepcopy(cfg)


# ---------------------------------------------------------------- bandpass lowpass_hz argument

def test_bandpass_default_unchanged_and_lowpass_argument():
    x = noise(1, 1, 30)[0]
    a = preprocess.bandpass(CFG, x, FS)
    assert np.array_equal(a, preprocess.bandpass(CFG, x, FS, None, None))
    assert np.array_equal(a, preprocess.bandpass(CFG, x, FS, lowpass_hz=CFG["preprocessing"]["bandpass"]["lowpass_hz"]))
    sine = np.sin(2.0 * np.pi * 20.0 * np.arange(30 * FS) / FS)     # 20 Hz: passed by 45 Hz, stopped by 5 Hz
    mid = slice(10 * FS, 20 * FS)
    assert np.std(preprocess.bandpass(CFG, sine, FS)[mid]) > 0.99 * np.std(sine[mid])
    assert np.std(preprocess.bandpass(CFG, sine, FS, highpass_hz=0.5, lowpass_hz=5.0)[mid]) < 0.01 * np.std(sine[mid])


# ---------------------------------------------------------------- blink detection and correction

def one_over_f(seed, n_s, sd=BACKGROUND_SD_UV):
    n = int(n_s * FS)
    b = preprocess._shaped_noise(np.random.default_rng(seed), n, 1.0)
    return sd * b / b.std()


def gaussian_blink(n, amp, centre_s):
    t = np.arange(n) / FS
    return amp * np.exp(-0.5 * ((t - centre_s) / BLINK_SIGMA_S) ** 2)


def independent_flags(x):
    """Detection re-implemented with plain scipy and the document's numbers (0.5 to 5 Hz, 6 robust SD)."""
    sos = np.vstack([butter(4, 0.5, btype="highpass", fs=FS, output="sos"),
                     butter(4, 5.0, btype="lowpass", fs=FS, output="sos")])
    c = sosfiltfilt(sos, x, padtype="odd", padlen=10 * FS)
    med = np.median(c)
    rsd = 1.4826 * np.median(np.abs(c - med))
    return np.abs(c - med) > 6.0 * rsd


def test_blink_detected_once_with_window_pm_quarter_second():
    n_s = 30
    bg = one_over_f(BLINK_SEEDS[0], n_s)
    x = preprocess.bandpass(CFG, bg + gaussian_blink(bg.size, 100.0, 15.0), FS)
    flagged = preprocess.blink_flags(CFG, x, FS)
    assert np.array_equal(flagged, independent_flags(x))
    window, weight, n_win = preprocess.blink_window_weights(CFG, flagged, FS)
    assert n_win == 1 and flagged[int(15.0 * FS)]
    idx, fidx = np.flatnonzero(window), np.flatnonzero(flagged)
    margin = int(0.25 * FS)
    assert idx[0] == fidx[0] - margin and idx[-1] == fidx[-1] + margin
    assert np.all(weight[flagged] == 1.0) and weight[idx[0]] == 0.0 and weight[idx[-1]] == 0.0
    assert np.all(weight[~window] == 0.0)


@pytest.mark.parametrize("kind", ["one_over_f", "white"])
def test_noise_alone_gives_no_detection(kind):
    n_s = 240
    bg = one_over_f(NOISE_SEED, n_s) if kind == "one_over_f" else noise(NOISE_SEED, 1, n_s)[0]
    # as in the pipeline, detection sees the band-passed, edge-trimmed signal (B4 output)
    x = preprocess.trim_edges(preprocess.bandpass(CFG, bg, FS), FS, CFG["preprocessing"]["edge_trim_s"])
    flagged = preprocess.blink_flags(CFG, x, FS)
    assert not flagged.any()
    assert preprocess.blink_window_weights(CFG, flagged, FS)[2] == 0


def test_two_flags_within_twice_the_margin_share_one_window():
    flagged = np.zeros(20 * FS, dtype=bool)
    flagged[5 * FS] = True
    flagged[5 * FS + int(0.4 * FS)] = True            # 0.4 s apart: windows overlap
    flagged[12 * FS] = True                           # far away: separate window
    assert preprocess.blink_window_weights(CFG, flagged, FS)[2] == 2


def test_wavelet_levels_and_half_supports_at_1024_hz():
    level, altered = preprocess.wavelet_levels(CFG, FS)
    assert level == 10 and altered == [7, 8, 9, 10]
    # sym4 has 8 filter taps; the level-j undecimated filter has (8 - 1) * 2**(j - 1) + 1 samples
    assert [preprocess.wavelet_half_support("sym4", j) for j in (7, 8, 9, 10)] == [224, 448, 896, 1792]


def blink_trial(kind, seed, amp, n_s=30, cfg=CFG):
    """Pipeline-faithful synthetic blink: background and background + blink both pass the 0.5 to 45 Hz
    band-pass; the known clean signal is the band-passed background alone; error is measured inside the
    flagged window plus its +/-0.25 s margin (IMP-012)."""
    bg = one_over_f(seed, n_s) if kind == "one_over_f" else noise(seed, 1, n_s)[0]
    clean = preprocess.bandpass(cfg, bg, FS)
    x = preprocess.bandpass(cfg, bg + gaussian_blink(bg.size, amp, n_s / 2.0), FS)
    flagged = preprocess.blink_flags(cfg, x, FS)
    window, weight, n_win = preprocess.blink_window_weights(cfg, flagged, FS)
    if n_win == 0:
        return None
    y = preprocess.wavelet_correct(cfg, x, FS, window, weight)
    before = np.sqrt(np.mean((x - clean)[window] ** 2))
    after = np.sqrt(np.mean((y - clean)[window] ** 2))
    return {"x": x, "y": y, "window": window, "before": before, "after": after, "ratio": after / before}


def test_blink_correction_halves_the_error_on_all_three_seeds():
    ratios = []
    for seed in BLINK_SEEDS:
        r = blink_trial("one_over_f", seed, BLINK_UV)
        assert r is not None
        ratios.append(r["ratio"])
        print(f"blink {BLINK_UV} uV, 1/f, seed {seed}: error before {r['before']:.3f} after {r['after']:.3f} "
              f"ratio {r['ratio']:.3f}")
        assert np.array_equal(r["y"][~r["window"]], r["x"][~r["window"]])      # bit-identical outside
    assert max(ratios) <= 0.5, f"worst ratio {max(ratios):.3f}, ratios {ratios}"


def test_blink_correction_leaves_outside_bit_identical_and_touches_inside():
    r = blink_trial("one_over_f", BLINK_SEEDS[1], BLINK_UV)
    assert np.array_equal(r["y"][~r["window"]], r["x"][~r["window"]])
    assert not np.array_equal(r["y"][r["window"]], r["x"][r["window"]])


def test_do_no_harm_sine_inside_forced_window():
    n = 30 * FS
    t = np.arange(n) / FS
    amp = BACKGROUND_SD_UV
    x = amp * np.sin(2.0 * np.pi * 10.0 * t)
    forced = np.zeros(n, dtype=bool)
    forced[15 * FS:15 * FS + 100] = True
    window, weight, n_win = preprocess.blink_window_weights(CFG, forced, FS)
    assert n_win == 1
    y = preprocess.wavelet_correct(CFG, x, FS, window, weight)
    assert np.max(np.abs(y - x)) < 0.01 * amp
    assert np.array_equal(y[~window], x[~window])


def test_no_window_returns_an_identical_copy():
    x = one_over_f(NOISE_SEED, 30)
    window, weight, _ = preprocess.blink_window_weights(CFG, np.zeros(x.size, dtype=bool), FS)
    y = preprocess.wavelet_correct(CFG, x, FS, window, weight)
    assert np.array_equal(y, x) and y is not x


# ---------------------------------------------------------------- 1-s segment rules

def rows_with_segment_rms(n_seg, seg_rms_ch0, seg_rms_ch1=None, seed=3):
    """(2, n_seg * FS): unit-scale noise, segment k scaled to exact RMS values given as {k: rms}."""
    x = noise(seed, 2, n_seg, sd=BACKGROUND_SD_UV)
    for ch, spec in enumerate((seg_rms_ch0, seg_rms_ch1 or {})):
        for k, rms in spec.items():
            x[ch, k * SEG:(k + 1) * SEG] = exact_rms(x[ch, k * SEG:(k + 1) * SEG], rms)
    return x


def test_rms_bounds_just_inside_and_outside():
    x = rows_with_segment_rms(8, {1: 0.99, 2: 1.01, 3: 149.9, 4: 150.1})
    flags, rej = preprocess.segment_flags(CFG, x, x, FS)
    assert flags["rms_low"][0].tolist() == [False, True, False, False, False, False, False, False]
    assert flags["rms_high"][0].tolist() == [False, False, False, False, True, False, False, False]
    assert not flags["rms_low"][1].any() and not flags["rms_high"][1].any()
    assert rej[1] and rej[4] and not rej[2]


def test_either_channel_failing_rejects_the_segment():
    x = rows_with_segment_rms(8, {}, {5: 0.5})              # only channel 1 has a low-RMS segment
    flags, rej = preprocess.segment_flags(CFG, x, x, FS)
    assert flags["rms_low"][1, 5] and not flags["rms_low"][0].any()
    assert rej.tolist() == [False] * 5 + [True] + [False] * 2
    clean = rows_with_segment_rms(8, {})
    assert not preprocess.segment_flags(CFG, clean, clean, FS)[1].any()


def test_strict_variant_upper_bound_is_100_uv():
    x = rows_with_segment_rms(6, {2: 99.9, 3: 100.1})
    flags, _ = preprocess.segment_flags(CFG, x, x, FS, strict=True)
    assert flags["rms_high"][0].tolist() == [False, False, False, True, False, False]
    flags_default, _ = preprocess.segment_flags(CFG, x, x, FS, strict=False)
    assert not flags_default["rms_high"].any()


def test_flatline_segment_rejected_normal_segment_kept():
    x = noise(4, 2, 8)
    x[0, 3 * SEG:4 * SEG] = 5.0                              # peak-to-peak 0, RMS 5: inside the RMS bounds
    flags, rej = preprocess.segment_flags(CFG, x, x, FS)
    assert flags["flat"][0].tolist() == [False, False, False, True, False, False, False, False]
    assert not flags["flat"][1].any()
    assert rej[3] and not rej[2]
    assert not flags["rms_low"][0, 3]


# ---------------------------------------------------------------- EMG rule

def numpy_band_power(seg, lo, hi):
    """Mean of the Hann-windowed periodogram over bins lo..hi Hz (plain numpy, 1-s segment: bin k = k Hz)."""
    w = np.hanning(seg.size)
    spec = np.abs(np.fft.fft((seg - seg.mean()) * w)) ** 2
    return float(np.mean(spec[lo:hi + 1]))


def sine_burst(n_s, k, freq, amp):
    x = np.zeros(int(n_s * FS))
    t = np.arange(SEG) / FS
    x[k * SEG:(k + 1) * SEG] = amp * np.sin(2.0 * np.pi * freq * t)
    return x


def test_emg_burst_at_90_hz_rejected_and_power_matches_numpy():
    base = noise(5, 2, 20)
    notched = base.copy()
    notched[0] += sine_burst(20, 4, 90.0, 20.0)                # channel 0 only
    powers = np.array([numpy_band_power(notched[0, k * SEG:(k + 1) * SEG], 70, 110) for k in range(20)])
    assert powers[4] > 5.0 * np.median(powers)              # expected result, computed independently
    got = preprocess.emg_power(CFG, notched, FS)
    assert np.allclose(got[0], powers, rtol=1e-9, atol=0.0)
    flags, rej = preprocess.segment_flags(CFG, notched, notched, FS)
    assert flags["emg"][0, 4] and not flags["emg"][1].any() and not np.delete(flags["emg"][0], 4).any()
    assert rej[4]


def test_same_burst_at_30_hz_is_not_rejected():
    base = noise(5, 2, 20)
    notched = base.copy()
    notched[0] += sine_burst(20, 4, 30.0, 20.0)
    powers = np.array([numpy_band_power(notched[0, k * SEG:(k + 1) * SEG], 70, 110) for k in range(20)])
    assert powers[4] < 5.0 * np.median(powers)
    flags, _ = preprocess.segment_flags(CFG, notched, notched, FS)
    assert not flags["emg"].any()


def test_emg_measure_is_taken_before_the_low_pass():
    base = noise(5, 2, 30)
    notched = base.copy()
    notched[0] += sine_burst(30, 4, 90.0, 20.0)
    x_bp = preprocess.bandpass(CFG, notched, FS)            # the 45 Hz low-pass removes the 90 Hz burst
    seg = x_bp[0, 4 * SEG:5 * SEG]
    assert numpy_band_power(seg, 70, 110) < 5.0 * numpy_band_power(base[0, 5 * SEG:6 * SEG], 70, 110)
    flags, rej = preprocess.segment_flags(CFG, x_bp, notched, FS)
    assert flags["emg"][0, 4] and rej[4]


def test_emg_strict_multiple_is_three():
    base = noise(6, 2, 20)
    powers = np.array([numpy_band_power(base[0, k * SEG:(k + 1) * SEG], 70, 110) for k in range(20)])
    med = np.median(powers)
    x = base.copy()
    k = 7
    # scale the segment's 70-110 Hz content so its power is 4x the median: rejected only when strict (3x)
    burst = sine_burst(20, k, 90.0, 1.0)
    unit = numpy_band_power(burst[k * SEG:(k + 1) * SEG], 70, 110)
    x[0] += burst * np.sqrt(4.0 * med / unit)
    p = numpy_band_power(x[0, k * SEG:(k + 1) * SEG], 70, 110)
    assert 3.0 * np.median([numpy_band_power(x[0, j * SEG:(j + 1) * SEG], 70, 110) for j in range(20)]) < p \
        < 5.0 * np.median([numpy_band_power(x[0, j * SEG:(j + 1) * SEG], 70, 110) for j in range(20)])
    assert not preprocess.segment_flags(CFG, x, x, FS, strict=False)[0]["emg"][0, k]
    assert preprocess.segment_flags(CFG, x, x, FS, strict=True)[0]["emg"][0, k]


# ---------------------------------------------------------------- padding and clean stretches

def test_single_rejected_segment_removes_one_second_plus_two_half_seconds():
    rej = np.zeros(30, dtype=bool)
    rej[10] = True
    for fs in (FS, FS_OUT):
        m = preprocess.pad_rejected(CFG, rej, fs)
        assert m.sum() == 2 * fs                               # 1 s + 2 x 0.5 s, exact in samples at both rates
        assert m[int(9.5 * fs)] and not m[int(9.5 * fs) - 1]
        assert m[int(11.5 * fs) - 1] and not m[int(11.5 * fs)]


def test_padding_is_the_same_time_span_at_both_rates():
    rej = np.zeros(30, dtype=bool)
    rej[[3, 4, 11, 20]] = True
    m_hi = preprocess.pad_rejected(CFG, rej, FS)
    m_lo = preprocess.pad_rejected(CFG, rej, FS_OUT)
    assert np.array_equal(m_hi.reshape(-1, FS // FS_OUT).all(axis=1), m_lo)
    assert np.array_equal(m_hi.reshape(-1, FS // FS_OUT).any(axis=1), m_lo)


def test_rejected_segments_two_seconds_apart_merge_and_three_seconds_apart_do_not():
    rej = np.zeros(30, dtype=bool)
    rej[[10, 12]] = True                                      # starts 2 s apart: padded spans touch at 11.5 s
    m = preprocess.pad_rejected(CFG, rej, FS_OUT)
    assert m.sum() == 4 * FS_OUT
    assert len(preprocess.clean_runs(~m, 1)) == 2             # one merged hole: clean before and after only
    rej = np.zeros(30, dtype=bool)
    rej[[10, 13]] = True                                      # starts 3 s apart: 1 s of clean data between
    m = preprocess.pad_rejected(CFG, rej, FS_OUT)
    assert m.sum() == 4 * FS_OUT and len(preprocess.clean_runs(~m, 1)) == 3


def test_clean_stretch_boundary_in_samples_at_256_hz():
    min_len = int(5 * FS_OUT)                                  # 1280 samples
    for length, kept in ((int(4.9 * FS_OUT), False), (min_len - 1, False), (min_len, True), (min_len + 1, True)):
        keep = np.concatenate([np.zeros(100, bool), np.ones(length, bool), np.zeros(100, bool)])
        assert preprocess.clean_runs(keep, min_len) == ([(100, 100 + length)] if kept else [])
    # through the padding: rejected segments 0 and 7 leave 6 whole seconds, 5.0 s after the 0.5 s pads
    rej = np.zeros(8, dtype=bool)
    rej[[0, 7]] = True
    m = preprocess.pad_rejected(CFG, rej, FS_OUT)
    runs = preprocess.clean_runs(~m, min_len)
    assert runs == [(int(1.5 * FS_OUT), int(6.5 * FS_OUT))] and runs[0][1] - runs[0][0] == min_len
    rej = np.zeros(7, dtype=bool)
    rej[[0, 6]] = True                                          # 5 whole seconds -> 4.0 s clean: dropped
    assert preprocess.clean_runs(~preprocess.pad_rejected(CFG, rej, FS_OUT), min_len) == []


# ---------------------------------------------------------------- segment_signal: segments, mask consistency, rescaling

def synthetic_recording(n_s=60, flat_seg=30):
    x = noise(NOISE_SEED, 2, n_s)
    x[0, flat_seg * SEG:(flat_seg + 1) * SEG] = 0.0            # flat-line, RMS 0
    return x


def test_segments_never_cross_a_rejected_gap_and_starts_match_the_mask():
    x = synthetic_recording()
    res = preprocess.segment_signal(CFG, x, x, FS)
    lengths = [s.shape[-1] for s in res.segments]
    # rejected second 30 padded by 0.5 s: samples [29.5 s, 31.5 s); clean before and after
    assert res.starts == [0, int(31.5 * FS_OUT)]
    assert lengths == [int(29.5 * FS_OUT), int(60 * FS_OUT) - int(31.5 * FS_OUT)]
    for s0, n in zip(res.starts, lengths):
        assert s0 + n <= int(29.5 * FS_OUT) or s0 >= int(31.5 * FS_OUT)
    log = res.meta["b5"]["log"]
    assert log["trimmed_s"] == 60.0 and len(log["blinks"]) == 2 and all(b["corrected_s"] >= 0.0 for b in log["blinks"])
    assert log["rejected_segments"] == 1 and log["rejected_before_padding_s"] == 1.0
    assert log["rejected_after_padding_s"] == 2.0 and log["clean_s"] == 58.0
    assert log["rules"]["flat"]["n_segments"] == 1 and log["rules"]["rms_low"]["n_segments"] == 1
    assert log["rules"]["emg"]["n_segments"] == 0 and log["rules"]["rms_high"]["n_segments"] == 0
    assert all(s.dtype == np.float64 for s in res.segments)


def test_rescaled_clean_samples_have_mu_ref_and_sigma_ref():
    x = synthetic_recording()
    res = preprocess.segment_signal(CFG, x, x, FS)
    allx = np.concatenate(res.segments, axis=-1)
    assert np.allclose(allx.mean(axis=-1), CFG["rescaling"]["mu_ref"], rtol=0, atol=1e-12)
    assert np.allclose(allx.std(axis=-1, ddof=1), CFG["rescaling"]["sigma_ref"], rtol=0, atol=1e-12)
    consts = res.meta["b5"]["rescale"]
    assert len(consts["mean"]) == 2 and all(s > 0 for s in consts["sd"])
    before = res.meta["b5"]["clean_stats_before"]
    assert before["mean"] == pytest.approx(consts["mean"], abs=0) and before["sd"] == pytest.approx(consts["sd"], abs=0)


def test_rescaling_constants_come_from_config():
    cfg = fluent()
    cfg["rescaling"]["mu_ref"], cfg["rescaling"]["sigma_ref"] = 3.0, 2.0
    segs = [noise(8, 2, 6) * 5.0 + 40.0]
    scaled, consts = preprocess.rescale(cfg, segs)
    assert np.allclose(scaled[0].mean(axis=-1), 3.0, atol=1e-12)
    assert np.allclose(scaled[0].std(axis=-1, ddof=1), 2.0, atol=1e-12)


def test_rescale_ddof_config_equals_reference_simulation_ddof():
    # IMP-006(c): the per-recording rescaling uses the same SD convention as sigma_ref (ddof 1)
    assert CFG["preprocessing"]["rescale"]["ddof"] == CFG["rescaling"]["reference_simulation"]["ddof"] == 1


def test_constant_channel_raises_a_clear_error():
    segs = [np.vstack([np.full(1280, 3.0), noise(9, 1, 1.25)[0]])]
    with pytest.raises(preprocess.PreprocessError, match="constant"):
        preprocess.rescale(CFG, segs)
    with pytest.raises(preprocess.PreprocessError, match="no clean segments"):
        preprocess.rescale(CFG, [])


def test_min_clean_stretch_drops_short_islands_in_segment_signal():
    x = noise(10, 2, 60)
    for k in (0, 5, 40):                                        # flat seconds 0, 5, 10, 40: islands 1.5-4.5 s and 6.5-9.5 s are 3 s each
        x[0, k * SEG:(k + 1) * SEG] = 0.0
    x[0, 10 * SEG:11 * SEG] = 0.0
    res = preprocess.segment_signal(CFG, x, x, FS)
    for s in res.segments:
        assert s.shape[-1] >= int(5 * FS_OUT)
    assert res.meta["b5"]["log"]["dropped_short_stretch_s"] > 0


# ---------------------------------------------------------------- vigilance

def test_vigilance_alpha_over_theta():
    n = 20 * FS_OUT
    t = np.arange(n) / FS_OUT
    rng = np.random.default_rng(11)
    for freq, high in ((10.0, True), (6.0, False)):
        seg = np.vstack([np.sin(2.0 * np.pi * freq * t), np.sin(2.0 * np.pi * freq * t + 1.0)])
        seg = seg + 0.01 * rng.standard_normal(seg.shape)
        v = preprocess.vigilance(CFG, [seg], [100], FS_OUT)
        r = np.array(v["ratios"])
        assert r.shape == (2, 10) and v["epoch_starts"] == [100 + 512 * k for k in range(10)]
        assert (np.all(r > 100.0) and min(v["mean"]) > 100.0) if high else (np.all(r < 0.01) and max(v["mean"]) < 0.01)


def test_vigilance_drops_partial_epochs():
    seg = noise(12, 2, 5)[:, ::4]                               # 5 s at 256 Hz -> 2 whole 2-s epochs
    v = preprocess.vigilance(CFG, [seg], [0], FS_OUT)
    assert len(v["epoch_starts"]) == 2 and len(v["ratios"][0]) == 2


# ---------------------------------------------------------------- driver and cache (fake B4 result)

class FakeB4:
    def __init__(self, x, sha="a" * 64):
        self.bipolar, self.bipolar_notched = x, x
        self.meta = {"rel_path": "sub-001/ses-t1/eeg/x.edf", "subject": "sub-001", "session": "ses-t1", "sha256": sha,
                     "fs_hz": float(FS), "bad_electrodes": [], "units_passed": True}


@pytest.fixture
def driver(monkeypatch, tmp_path):
    calls = {"b4": 0, "b5": 0}
    state = {"sha": "a" * 64, "x": synthetic_recording(40)}

    def fake_b4(cfg, *a, **k):
        calls["b4"] += 1
        return FakeB4(state["x"], state["sha"])

    real = preprocess.segment_signal

    def counting(*a, **k):
        calls["b5"] += 1
        return real(*a, **k)

    monkeypatch.setattr(preprocess, "preprocess_recording", fake_b4)
    monkeypatch.setattr(preprocess, "segment_signal", counting)

    def run(cfg=CFG, **kw):
        return preprocess.segment_recording(cfg, "x", "y", {}, set(), cache_root=tmp_path / "cache", **kw)
    return run, calls, state, tmp_path


def test_b5_cache_reuse_and_result_identical(driver):
    run, calls, _, tmp = driver
    a, b = run(), run()
    assert calls["b5"] == 1 and len(list((tmp / "cache").rglob("*.npz"))) == 1
    assert a.starts == b.starts and a.meta == b.meta
    assert all(np.array_equal(p, q) for p, q in zip(a.segments, b.segments))


def test_b5_cache_invalidates_on_config_change_file_hash_and_variant(driver):
    run, calls, state, tmp = driver
    run()
    cfg2 = fluent()
    cfg2["preprocessing"]["ocular"]["wavelet"]["clip_multiplier_robust_sd"] = 4
    run(cfg2)
    assert calls["b5"] == 2
    state["sha"] = "b" * 64
    run()
    assert calls["b5"] == 3
    run(strict=True)
    run(sensitivity_highpass=True)
    assert calls["b5"] == 5 and len(list((tmp / "cache").rglob("*.npz"))) == 5
    cfg3 = fluent()
    cfg3["rescaling"]["sigma_ref"] = 1.5                        # the rescaling constants are part of the key
    run(cfg3)
    assert calls["b5"] == 6


def test_b5_cache_invalidates_when_the_source_changes(driver, monkeypatch, tmp_path):
    run, calls, _, tmp = driver
    run()
    run()
    assert calls["b5"] == 1
    src = tmp_path / "preprocess_changed.py"
    src.write_text(Path(preprocess.__file__).read_text(encoding="utf-8") + "\n# one changed line\n", encoding="utf-8")
    monkeypatch.setattr(preprocess, "_SOURCE", src)
    run()
    assert calls["b5"] == 2 and len(list((tmp / "cache").rglob("*.npz"))) == 2


def test_code_hash_is_the_sha256_of_the_source_and_enters_both_keys(monkeypatch, tmp_path):
    import hashlib
    assert preprocess.code_hash() == hashlib.sha256(Path(preprocess.__file__).read_bytes()).hexdigest()
    k4, k5 = preprocess.cache_key(CFG, "a" * 64, False, False), preprocess.cache_key_b5(CFG, "a" * 64, False, False)
    src = tmp_path / "other.py"
    src.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(preprocess, "_SOURCE", src)
    assert preprocess.code_hash() != hashlib.sha256(Path(preprocess.__file__).read_bytes()).hexdigest()
    assert preprocess.cache_key(CFG, "a" * 64, False, False) != k4
    assert preprocess.cache_key_b5(CFG, "a" * 64, False, False) != k5


def test_b5_skipped_for_units_or_bad_electrode_recordings(monkeypatch):
    class Bad(FakeB4):
        def __init__(self):
            super().__init__(None)
            self.bipolar = None
    monkeypatch.setattr(preprocess, "preprocess_recording", lambda *a, **k: Bad())
    res = preprocess.segment_recording(CFG, "x", "y", {}, set())
    assert res.meta["b5"] is None and res.segments == []


# ---------------------------------------------------------------- static checks

def test_no_numeric_literals_beyond_allowed_and_never_imports_model():
    tree = ast.parse((REPO_ROOT / "src" / "preprocess.py").read_text(encoding="utf-8"))
    allowed = {0, 1, 2, 0.5}
    bad = [(n.value, n.lineno) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool) and n.value not in allowed]
    assert not bad, bad
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and m.split(".")[-1] == "model" for m in mods), mods


def test_segment_rms_min_equals_the_electrode_rms_min():
    # DEV-001: the 1-s segment lower bound and the electrode lower bound are both 1 uV and move together
    pp = CFG["preprocessing"]
    assert pp["rejection"]["segment_rms_min_uv"] == pp["bad_channel"]["rms_min_uv"] == 1.0


def test_no_bare_print_in_src_preprocess_and_emit_writes_to_stdout(capsys):
    tree = ast.parse((REPO_ROOT / "src" / "preprocess.py").read_text(encoding="utf-8"))
    prints = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "print"]
    assert not prints, prints
    preprocess.emit("line one")
    preprocess.emit()
    assert capsys.readouterr().out == "line one\n\n"


def test_wavelet_pad_settings_come_from_config():
    cfg = fluent()
    x = one_over_f(NOISE_SEED, 30)
    window = np.zeros(x.size, dtype=bool)
    window[15 * FS:15 * FS + 200] = True
    weight = window.astype(float)
    a = preprocess.wavelet_correct(cfg, x, FS, window, weight)
    cfg["preprocessing"]["ocular"]["wavelet"]["reflect_type"] = "even"
    b = preprocess.wavelet_correct(cfg, x, FS, window, weight)
    assert a.shape == b.shape and np.array_equal(a[~window], b[~window])
    cfg["preprocessing"]["ocular"]["wavelet"]["edge_pad_s"] = 2
    assert preprocess.wavelet_correct(cfg, x, FS, window, weight).shape == x.shape


def test_b5_config_leaves_are_placeholders_with_the_documented_values():
    raw = __import__("yaml").safe_load((REPO_ROOT / "config.yml").read_text(encoding="utf-8"))["preprocessing"]
    assert raw["rejection"]["segment_rms_min_uv"]["value"] == 1.0
    assert raw["rejection"]["segment_rms_min_uv"]["prov"] == "placeholder"
    assert raw["ocular"]["mad_to_sd_factor"]["value"] == 1.4826 and raw["ocular"]["mad_to_sd_factor"]["prov"] == "literature"
    wv = raw["ocular"]["wavelet"]
    assert wv["family"]["value"] == "sym4" and wv["clip_multiplier_robust_sd"]["value"] == 3
    assert wv["edge_pad_s"]["value"] == 4 and all(v["prov"] == "placeholder" for v in wv.values())
    assert wv["reflect_type"]["value"] == "odd"
    assert raw["exclusions"]["schema_version"]["value"] == 1 and raw["exclusions"]["schema_version"]["prov"] == "placeholder"
    mad = raw["ocular"]["mad_to_sd_factor"]
    assert "normal" in mad["ref"] and "Phi^-1(0.75)" in mad["ref"] and "§5.1" not in mad["ref"].replace("(not from §5.1)", "")
    assert abs(mad["value"] - 1.0 / __import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.75)) < 5e-5
