"""Zero-phase band-pass and FFT downsampling (§5.1, §10.1; IMP-004).

Functions only: no file I/O, no channel handling, no segment rejection.
Never imports src.model (§18).

bandpass() is for whole continuous recordings before segmentation, never for
short segments. It raises for inputs of padlen samples or fewer, by design.
Trimming or masking the filtered edges is decided in B4.

Forward-backward filtering squares the magnitude response, so the nominal
cutoffs in config.yml are the -6 dB points of the combined filter.
"""
import numpy as np
from scipy.signal import butter, resample, sosfiltfilt


class PreprocessError(ValueError):
    """Raised for an input or configuration the filter functions cannot handle."""


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
