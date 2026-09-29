"""A3 tests: zero-phase band-pass and FFT downsampling (§5.1, §10.1; IMP-004).

Expected values come from closed-form Butterworth arithmetic with plain `math`,
never from the code under test.
"""
import ast
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import correlate, sosfiltfilt

from src import preprocess
from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
BP = CFG["preprocessing"]["bandpass"]
HP_DEFAULT = BP["highpass_hz"]
HP_SENS = CFG["preprocessing"]["sensitivity_highpass_hz"]
LP = BP["lowpass_hz"]
ORDER = BP["filter_order"]
FS_NATIVE = CFG["dataset"]["native_fs_hz"]
FS_GEN = CFG["g0"]["generation_fs_hz"]
FS_OBS = CFG["preprocessing"]["observation_fs_hz"]
DURATION_S = CFG["dataset"]["recording_length_s"]
SEED = CFG["split"]["primary_seed"]
HP_VARIANTS = (HP_DEFAULT, HP_SENS)


# ---- closed-form reference (independent of preprocess.py) --------------------
#
# One bilinear-transform Butterworth pass of order n has the exact digital magnitude
#   low-pass :  |H|   = 1 / sqrt(1 + r^(2n)),  r = tan(pi f / fs) / tan(pi fc / fs)
#   high-pass:  |H|   = 1 / sqrt(1 + r^(2n)),  r = tan(pi fc / fs) / tan(pi f / fs)
# Forward-backward filtering applies the pass twice, so the magnitudes are squared:
#   |H_fb| = 1 / (1 + r^(2n)).
# The band-pass is the high-pass times the low-pass. The analog formula is the same
# with r = f / fc (no frequency warping); it is printed beside the warped value only.
#
# Worked example, fs = 1024, n = 4, hp = 0.5, lp = 45, f = 60:
#   low-pass : r = tan(pi*60/1024) / tan(pi*45/1024) = 0.18620 / 0.13894 = 1.3402
#              r^8 = 10.41, |H_fb| = 1 / 11.41 = 0.0876   (analog: r = 1.3333, r^8 = 9.99, 0.0910)
#   high-pass: r = tan(pi*0.5/1024) / tan(pi*60/1024) = 0.001534 / 0.18620 = 0.00824
#              r^8 ~ 2e-17, |H_fb| = 1
#   band-pass: 0.0876 * 1 = 0.0876
# At f = fc the ratio is 1 and |H_fb| = 1/2 (-6 dB), for either warping.

def _fb_gain(r):
    return 1.0 / (1.0 + r ** (2 * ORDER))


def ref_gain_warped(f, hp, fs):
    lp_part = _fb_gain(math.tan(math.pi * f / fs) / math.tan(math.pi * LP / fs))
    hp_part = _fb_gain(math.tan(math.pi * hp / fs) / math.tan(math.pi * f / fs))
    return hp_part * lp_part


def ref_gain_analog(f, hp):
    return _fb_gain(f / LP) * _fb_gain(hp / f)


def sine(f, fs, seconds, amp=1.0, phase=0.0):
    t = np.arange(int(round(seconds * fs))) / fs
    return amp * np.sin(2.0 * math.pi * f * t + phase)


def fit_sine(y, f, fs):
    """Least-squares amplitude and phase of y ~ a*sin(2 pi f t + phi) (t from 0)."""
    t = np.arange(y.size) / fs
    basis = np.column_stack([np.sin(2.0 * math.pi * f * t), np.cos(2.0 * math.pi * f * t)])
    a, b = np.linalg.lstsq(basis, y, rcond=None)[0]
    return math.hypot(a, b), math.atan2(b, a)


def middle(y, fs, seconds=100):
    """The middle `seconds` of a recording and the offset (in samples) of its first sample."""
    n = y.shape[-1]
    half = int(round(seconds * fs / 2))
    start = n // 2 - half
    return y[..., start:start + 2 * half], start


# ---- gain table ---------------------------------------------------------------

@pytest.mark.parametrize("hp", HP_VARIANTS)
def test_gain_matches_closed_form(hp, capsys):
    fs = FS_NATIVE
    freqs = (0.1, 0.5, 10.0, 45.0, 60.0)
    lines = [f"\n[gain table] fs={fs} Hz, hp={hp} Hz, lp={LP} Hz, order {ORDER}/pass, forward-backward",
             f"{'f (Hz)':>8} {'warped ref':>14} {'analog ref':>14} {'measured':>14} {'|meas-warped|':>14}"]
    failures = []
    for f in freqs:
        y = preprocess.bandpass(CFG, sine(f, fs, DURATION_S), fs, highpass_hz=hp)
        yy, start = middle(y, fs)
        # fit with t measured from the original start so the phase is comparable
        t = (np.arange(yy.size) + start) / fs
        basis = np.column_stack([np.sin(2.0 * math.pi * f * t), np.cos(2.0 * math.pi * f * t)])
        a, b = np.linalg.lstsq(basis, yy, rcond=None)[0]
        measured = math.hypot(a, b)
        ref = ref_gain_warped(f, hp, fs)
        lines.append(f"{f:>8.2f} {ref:>14.6e} {ref_gain_analog(f, hp):>14.6e} "
                     f"{measured:>14.6e} {abs(measured - ref):>14.3e}")
        if abs(measured - ref) > 1e-3 * ref + 1e-9:
            failures.append((f, ref, measured))
    with capsys.disabled():
        print("\n".join(lines))
    assert not failures, failures


# ---- zero phase ---------------------------------------------------------------

@pytest.mark.parametrize("hp", HP_VARIANTS)
def test_zero_phase_lag_is_zero(hp, capsys):
    fs, f = FS_NATIVE, 10.0
    x = sine(f, fs, DURATION_S)
    y = preprocess.bandpass(CFG, x, fs, highpass_hz=hp)
    xm, start = middle(x, fs, seconds=20)
    ym, _ = middle(y, fs, seconds=20)
    cc = correlate(ym, xm, mode="full", method="fft")
    lag = int(np.argmax(cc)) - (xm.size - 1)
    t = (np.arange(ym.size) + start) / fs
    basis = np.column_stack([np.sin(2.0 * math.pi * f * t), np.cos(2.0 * math.pi * f * t)])
    a, b = np.linalg.lstsq(basis, ym, rcond=None)[0]
    gain, phase = math.hypot(a, b), math.atan2(b, a)
    ref = ref_gain_warped(f, hp, fs)
    with capsys.disabled():
        print(f"\n[zero phase] hp={hp}: lag={lag} samples, phase={phase:.3e} rad, "
              f"gain={gain:.9f} (warped ref {ref:.9f})")
    assert lag == 0
    assert abs(phase) < 1e-6
    assert abs(gain - ref) < 1e-3 * ref


# ---- edge transient report (not a pass rule) ----------------------------------

def _edge_seconds(y, ideal, fs, amp=1.0, frac=0.01):
    dev = np.abs(y - ideal) > frac * amp
    idx = np.flatnonzero(dev)
    if idx.size == 0:
        return 0.0, 0.0
    n = y.size
    first = idx[idx < n // 2]
    last = idx[idx >= n // 2]
    start_s = (first.max() + 1) / fs if first.size else 0.0
    end_s = (n - last.min()) / fs if last.size else 0.0
    return start_s, end_s


def test_edge_transient_report(capsys):
    fs, f = FS_NATIVE, 10.0
    x = sine(f, fs, DURATION_S)
    default_pad_sos_hp = {}
    lines = [f"\n[edge transient] {DURATION_S} s, {f} Hz sine, fs={fs}; seconds at each end deviating "
             f"> 1% of amplitude from the ideal gain-scaled sine",
             f"{'hp (Hz)':>8} {'padding':>26} {'start (s)':>10} {'end (s)':>10}"]
    for hp in HP_VARIANTS:
        ideal = ref_gain_warped(f, hp, fs) * x
        y = preprocess.bandpass(CFG, x, fs, highpass_hz=hp)
        s, e = _edge_seconds(y, ideal, fs)
        lines.append(f"{hp:>8} {'configured (' + str(BP['edge_pad_s']) + ' s, odd)':>26} {s:>10.3f} {e:>10.3f}")
        sos = preprocess.bandpass_sos(CFG, fs, hp)
        y0 = sosfiltfilt(sos, x)
        s0, e0 = _edge_seconds(y0, ideal, fs)
        lines.append(f"{hp:>8} {'scipy default padlen':>26} {s0:>10.3f} {e0:>10.3f}")
        default_pad_sos_hp[hp] = (s, e, s0, e0)
    with capsys.disabled():
        print("\n".join(lines))
    # not a pass rule; only that the report is finite and the interior is clean
    for vals in default_pad_sos_hp.values():
        assert all(np.isfinite(v) for v in vals)


# ---- impulse response -----------------------------------------------------------

@pytest.mark.parametrize("hp", HP_VARIANTS)
def test_impulse_response(hp, capsys):
    fs = FS_NATIVE
    h = preprocess.bandpass_impulse_response(CFG, fs, highpass_hz=hp)
    n, centre = h.size, h.size // 2
    peak = np.max(np.abs(h))
    assert n % 2 == 1
    assert int(np.argmax(np.abs(h))) == centre
    # symmetric about the centre
    np.testing.assert_allclose(h, h[::-1], rtol=0.0, atol=1e-9 * peak)
    # DC gain about 0 (a high-pass has zero DC gain)
    dc = float(h.sum())
    # response at both ends of the configured window is negligible (STOP-and-report rule)
    end_ratio = max(abs(h[0]), abs(h[-1])) / peak
    with capsys.disabled():
        print(f"\n[impulse response] hp={hp}: n={n}, peak={peak:.6e}, sum(h)={dc:.3e}, "
              f"|h[0]|/peak={abs(h[0]) / peak:.3e}, |h[-1]|/peak={abs(h[-1]) / peak:.3e}")
    assert end_ratio < 1e-6
    assert abs(dc) < 1e-6
    # frequency response matches the closed form; phase is zero once centred
    spec = np.fft.rfft(np.roll(h, -centre))
    # the window truncates the 0.1 Hz tail at ~6e-10 of peak, which leaves ~5e-9 imaginary
    # residue; 1e-7 is a loose bound on "zero phase", not a tuned value
    assert np.max(np.abs(spec.imag)) < 1e-7 * np.max(np.abs(spec))
    for target in (0.5, 1.0, 5.0, 10.0, 20.0, 45.0, 60.0, 100.0):
        k = int(round(target * n / fs))
        f_k = k * fs / n
        ref = ref_gain_warped(f_k, hp, fs)
        assert abs(spec[k].real - ref) <= 1e-3 * ref + 1e-6, (target, spec[k].real, ref)


# ---- downsampling ---------------------------------------------------------------

@pytest.mark.parametrize("fs_in", (FS_NATIVE, FS_GEN))
def test_downsample_length_amplitude_phase(fs_in, capsys):
    ratio = fs_in // FS_OBS
    seconds, f, amp, phase0 = 30, 10.37, 1.7, 0.6   # 311.1 cycles: not periodic in the window
    x = sine(f, fs_in, seconds, amp=amp, phase=phase0)
    y = preprocess.downsample(CFG, x, fs_in)
    assert y.size == x.size // ratio
    assert y.dtype == np.float64
    # analytic sine sampled at 256 Hz
    expected = sine(f, FS_OBS, seconds, amp=amp, phase=phase0)
    err = np.abs(y - expected)
    edge = FS_OBS  # 1 s
    with capsys.disabled():
        print(f"\n[downsample] {fs_in}->{FS_OBS}: max |err| interior={err[edge:-edge].max():.3e}, "
              f"first/last 1 s={max(err[:edge].max(), err[-edge:].max()):.3e} (amp {amp})")
    assert err[edge:-edge].max() < 1e-3 * amp


def test_downsample_removes_200hz_from_1024():
    seconds = 30
    x = sine(200.0, FS_NATIVE, seconds)
    y = preprocess.downsample(CFG, x, FS_NATIVE)
    edge = FS_OBS
    rms_out = math.sqrt(float(np.mean(y[edge:-edge] ** 2)))
    rms_in = math.sqrt(float(np.mean(x ** 2)))
    assert rms_out < 1e-3 * rms_in


def test_downsample_default_target_is_observation_rate():
    x = np.zeros(FS_NATIVE * 10)
    assert preprocess.downsample(CFG, x, FS_NATIVE).size == x.size // (FS_NATIVE // FS_OBS)


def test_downsample_2d_matches_1d():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((2, FS_NATIVE * 10))
    y2 = preprocess.downsample(CFG, x, FS_NATIVE)
    for i in range(2):
        np.testing.assert_array_equal(y2[i], preprocess.downsample(CFG, x[i], FS_NATIVE))


def test_downsample_raises():
    x = np.zeros(FS_NATIVE * 10)
    with pytest.raises(preprocess.PreprocessError):
        preprocess.downsample(CFG, np.zeros(1000 * 10), 1000)           # 1000/256 not an integer
    with pytest.raises(preprocess.PreprocessError):
        preprocess.downsample(CFG, x, FS_NATIVE, fs_out_hz=FS_NATIVE * 2)  # upsampling
    with pytest.raises(preprocess.PreprocessError):
        preprocess.downsample(CFG, np.zeros(FS_NATIVE * 10 + 1), FS_NATIVE)  # not divisible by 4
    with pytest.raises(preprocess.PreprocessError):
        preprocess.downsample(CFG, np.zeros(FS_NATIVE), FS_NATIVE)     # 1 s < 2 s reflect-pad
    with pytest.raises(TypeError):
        preprocess.downsample(CFG, x.astype(np.float32), FS_NATIVE)
    with pytest.raises(preprocess.PreprocessError):
        bad = x.copy()
        bad[5] = np.nan
        preprocess.downsample(CFG, bad, FS_NATIVE)


def test_downsample_ratio_one_returns_copy():
    x = np.arange(FS_OBS * 10, dtype=np.float64)
    y = preprocess.downsample(CFG, x, FS_OBS)
    np.testing.assert_array_equal(x, y)
    assert y is not x


# ---- band-pass guards -------------------------------------------------------------

def test_bandpass_guards():
    fs = FS_NATIVE
    padlen = int(round(BP["edge_pad_s"] * fs))
    x = sine(10.0, fs, DURATION_S)
    with pytest.raises(TypeError):
        preprocess.bandpass(CFG, x.astype(np.float32), fs)
    with pytest.raises(TypeError):
        preprocess.bandpass(CFG, x.astype(np.int64), fs)
    for bad_value in (np.nan, np.inf):
        bad = x.copy()
        bad[100] = bad_value
        with pytest.raises(preprocess.PreprocessError):
            preprocess.bandpass(CFG, bad, fs)
    with pytest.raises(preprocess.PreprocessError):
        preprocess.bandpass(CFG, x[:padlen], fs)          # padlen samples or fewer raise
    assert np.all(np.isfinite(preprocess.bandpass(CFG, x[:padlen + 1], fs)))
    with pytest.raises(preprocess.PreprocessError):
        preprocess.bandpass(CFG, x, fs, highpass_hz=LP)   # highpass >= lowpass
    with pytest.raises(preprocess.PreprocessError):
        preprocess.bandpass(CFG, x, 64)                    # lowpass (45 Hz) above Nyquist (32 Hz)
    with pytest.raises(preprocess.PreprocessError):
        preprocess.bandpass(CFG, np.zeros((2, 2, 2000)), fs)  # 3-D


def test_bandpass_config_guards():
    import copy
    fs = FS_NATIVE
    x = sine(10.0, fs, DURATION_S)
    cfg = copy.deepcopy(CFG)
    cfg["preprocessing"]["bandpass"]["zero_phase"] = False
    with pytest.raises(preprocess.PreprocessError):
        preprocess.bandpass(cfg, x, fs)
    cfg = copy.deepcopy(CFG)
    cfg["preprocessing"]["bandpass"]["filter_family"] = "cheby1"
    with pytest.raises(preprocess.PreprocessError):
        preprocess.bandpass(cfg, x, fs)


def test_bandpass_2d_matches_1d_and_time_last():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((2, FS_OBS * 60))
    y2 = preprocess.bandpass(CFG, x, FS_OBS)
    assert y2.shape == x.shape
    for i in range(2):
        np.testing.assert_array_equal(y2[i], preprocess.bandpass(CFG, x[i], FS_OBS))


def test_bandpass_default_highpass_is_config_value():
    x = sine(0.5, FS_NATIVE, DURATION_S)
    np.testing.assert_array_equal(
        preprocess.bandpass(CFG, x, FS_NATIVE),
        preprocess.bandpass(CFG, x, FS_NATIVE, highpass_hz=HP_DEFAULT))


# ---- determinism --------------------------------------------------------------------

def test_bit_identical_repeat():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(FS_NATIVE * 30)
    a = preprocess.downsample(CFG, preprocess.bandpass(CFG, x, FS_NATIVE), FS_NATIVE)
    b = preprocess.downsample(CFG, preprocess.bandpass(CFG, x, FS_NATIVE), FS_NATIVE)
    assert a.tobytes() == b.tobytes()


# ---- 1024 Hz and 2048 Hz inputs agree at the 256 Hz output --------------------------

def test_cross_rate_agreement(capsys):
    # The two inputs carry the same 10 Hz sine but are filtered at different fs, so the
    # bilinear frequency warping gives slightly different gains. The expected amplitude
    # disagreement is therefore the difference of the two warped closed-form gains:
    #   delta_g = |g(10 Hz; fs=1024) - g(10 Hz; fs=2048)|.
    # Each measured amplitude (least-squares fit over the middle 100 s of the 256 Hz output)
    # must sit within delta_g of its own warped reference (a fs-dependent filter design may
    # not be off by more than the disagreement the warping itself predicts), and the two
    # amplitudes may differ from each other by at most 2 * delta_g. Fitting over whole
    # cycles averages out the point-wise FFT-resample seam ripple (about 1e-6 point-wise
    # at 30 s from the edge), so it does not enter the amplitude comparison.
    f = 10.0
    g1, g2 = (ref_gain_warped(f, HP_DEFAULT, fs) for fs in (FS_NATIVE, FS_GEN))
    delta_g = abs(g1 - g2)
    amps, phases = [], []
    for fs in (FS_NATIVE, FS_GEN):
        y = preprocess.downsample(CFG, preprocess.bandpass(CFG, sine(f, fs, DURATION_S), fs), fs)
        assert y.size == DURATION_S * FS_OBS
        ym, start = middle(y, FS_OBS)
        t = (np.arange(ym.size) + start) / FS_OBS
        basis = np.column_stack([np.sin(2.0 * math.pi * f * t), np.cos(2.0 * math.pi * f * t)])
        a, b = np.linalg.lstsq(basis, ym, rcond=None)[0]
        amps.append(math.hypot(a, b))
        phases.append(math.atan2(b, a))
    with capsys.disabled():
        print(f"\n[cross-rate] g(1024)={g1:.9f} g(2048)={g2:.9f} delta_g={delta_g:.3e}; "
              f"measured amps {amps[0]:.9f} / {amps[1]:.9f} (diff {abs(amps[0] - amps[1]):.3e}); "
              f"phases {phases[0]:.2e} / {phases[1]:.2e} rad")
    tol = delta_g
    assert abs(amps[0] - g1) <= tol
    assert abs(amps[1] - g2) <= tol
    assert abs(amps[0] - amps[1]) <= 2.0 * tol
    assert max(abs(p) for p in phases) < 1e-6


# ---- hygiene -------------------------------------------------------------------------

def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names |= {n.module or ""} | {f"{n.module}.{a.name}" for a in n.names}
    return names


def test_preprocess_never_imports_model():
    names = _imports(REPO_ROOT / "src" / "preprocess.py")
    assert not any(m.split(".")[-1] == "model" for m in names), names


def test_no_numeric_literals_beyond_allowed_in_preprocess():
    tree = ast.parse((REPO_ROOT / "src" / "preprocess.py").read_text(encoding="utf-8"))
    allowed = {0, 1, 2, 0.5}
    bad = [(n.value, n.lineno) for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
           and not isinstance(n.value, bool) and n.value not in allowed]
    assert not bad, f"numeric literals in src/preprocess.py: {bad}"


def test_functions_write_nothing_to_run_time_folders():
    paths = CFG["paths"]
    folders = [REPO_ROOT / paths[k] for k in ("cache_dir", "outputs_dir", "results_dir", "logs_dir")]

    def snapshot():
        return {str(p): p.stat().st_mtime_ns for d in folders if d.exists()
                for p in [d, *d.rglob("*")]}

    before = snapshot()
    x = sine(10.0, FS_NATIVE, DURATION_S)
    preprocess.downsample(CFG, preprocess.bandpass(CFG, x, FS_NATIVE), FS_NATIVE)
    preprocess.bandpass_impulse_response(CFG, FS_NATIVE)
    assert snapshot() == before


def test_config_leaves_for_imp004_are_placeholders():
    raw = __import__("yaml").safe_load((REPO_ROOT / "config.yml").read_text(encoding="utf-8"))
    bp = raw["preprocessing"]["bandpass"]
    for key in ("filter_family", "filter_order", "edge_pad_s", "edge_padtype",
                "impulse_response_duration_s"):
        assert bp[key]["prov"] == "placeholder", key
    for key in ("pad_s", "reflect_type"):
        assert raw["preprocessing"]["resample"][key]["prov"] == "placeholder", key
