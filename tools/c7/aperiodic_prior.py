"""C7: per-recording aperiodic (1/f) background level from the recording's own spectrum (PLAN.md C7).

Experimental, tools only. Rule, fixed before any result exists (nothing here sees a gain estimate):
  1. Welch PSD (2-s Hann, 1-s overlap, constant detrend) of each clean segment, averaged over segments
     weighted by segment length, per channel.
  2. Use the bins FIT_HZ, excluding the alpha band EXCLUDE_HZ.
  3. Subtract the white floor of the filter's own R (one-sided density 2 R / fs); keep bins that stay positive.
  4. Fit a one-sided Lorentzian 4 s2 tau / (1 + (2 pi f tau)^2) for (s2, tau): tau on a log grid within TAU_BOUNDS_S,
     s2 in closed form (least squares of the log PSD). Deterministic, no optimizer.
The fit cannot tell 1/f from the model's own broadband shoulder; the S0 arm of the driver prints what it returns on
plain data.
"""
import numpy as np
from scipy.signal import welch

FIT_HZ = (1.0, 40.0)              # placeholder: upper edge below the 45 Hz low-pass transition
EXCLUDE_HZ = (7.0, 14.0)          # placeholder: as the slope metric of C4c / C4d
TAU_BOUNDS_S = (0.016, 0.32)      # placeholder: corner between 10 Hz and 0.5 Hz
N_TAU = 80


def pooled_psd(segments, fs):
    """(f, psd) of shape (n_f,), (2, n_f): one-sided density averaged over segments, weighted by length."""
    nper = int(round(2 * fs))
    tot, wsum, f = 0.0, 0.0, None
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float64)
        if seg.shape[1] < nper:
            continue
        f, p = welch(seg, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2, detrend="constant", axis=1)
        tot = tot + p * seg.shape[1]
        wsum += seg.shape[1]
    if f is None:
        raise ValueError("no segment is long enough for the 2-s Welch window")
    return f, tot / wsum


def lorentzian(f, s2, tau):
    return 4.0 * s2 * tau / (1.0 + (2.0 * np.pi * f * tau) ** 2)


def fit_channel(f, psd, white_floor):
    """(s2, tau) for one channel; (0.0, upper bound) if no bin stays positive after the floor."""
    sel = (f >= FIT_HZ[0]) & (f <= FIT_HZ[1]) & ~((f >= EXCLUDE_HZ[0]) & (f <= EXCLUDE_HZ[1]))
    res = psd[sel] - white_floor
    ok = res > 0.0
    fs_, ys = f[sel][ok], np.log(res[ok])
    if fs_.size < 3:
        return 0.0, TAU_BOUNDS_S[1]
    best = None
    for tau in np.geomspace(TAU_BOUNDS_S[0], TAU_BOUNDS_S[1], N_TAU):
        shape = np.log(lorentzian(fs_, 1.0, tau))
        ls2 = float(np.mean(ys - shape))
        sse = float(np.sum((ys - shape - ls2) ** 2))
        if best is None or sse < best[0]:
            best = (sse, np.exp(ls2), tau)
    return float(best[1]), float(best[2])


def estimate(segments, fs, R_variance):
    """Per channel (s2, tau) as arrays of shape (2,), plus the channel variances used for the report."""
    f, psd = pooled_psd(segments, fs)
    floor = 2.0 * R_variance / fs
    out = [fit_channel(f, psd[c], floor) for c in range(psd.shape[0])]
    allx = np.concatenate([np.asarray(s, dtype=np.float64) for s in segments], axis=1)
    return {"s2": np.array([o[0] for o in out]), "tau": np.array([o[1] for o in out]),
            "var": allx.var(axis=1, ddof=1)}
