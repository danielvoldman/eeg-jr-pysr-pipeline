"""C4d: read-only diagnostic on SYNTHETIC data: does a 1/f aperiodic background reproduce the real-data
behaviour (parameter runaway, y1 excursions, heavier NIS tail)? (PLAN.md C4d)

Changes no src rule, threshold or config value; writes nothing under outputs/. Everything is applied inside
this script. The state-SD flag is disabled as in tools/c4b_divergence_diagnostic.py (via the worker of
tools/c4c_parameter_freedom_diagnostic.py, variant V0). Item 5 is the only step that reads real data:
3 pilot recordings, ses-t1, through the development guard.

Series: the six C4b synthetic series (60 s as 2 x 30 s, same seeds and parameters).
  S0 plain (model units, as C4b)
  S1 + 1/f noise (power slope -1.6, FFT-shaped white noise, independent per channel) carrying 50% of the
     1-45 Hz variance of the (model + observation noise) signal
  S2 the same with 80%
The noise is added BEFORE the 0.5-45 Hz band-pass and the rescaling to mu_ref and sigma_ref (real pipeline
order); the segments are then cut out of the processed stream.

    python tools/c4d_aperiodic_diagnostic.py <raw_results.json> [n_jobs]
"""
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
os.environ["PYTHONPATH"] = os.pathsep.join([str(TOOLS), str(ROOT), os.environ.get("PYTHONPATH", "")])

import numpy as np  # noqa: E402

import c4c_parameter_freedom_diagnostic as C  # noqa: E402

Q_VALUES = (1e-2, 1e-3)
SLOPE = -1.6
SHARES = {"S1": 0.5, "S2": 0.8}
SERIES = [  # (label, seed, g12, g21, p): the C4b synthetic series
    ("g0 p=220,220", 101, 0.0, 0.0, None),
    ("g0 p=180,250", 102, 0.0, 0.0, (180.0, 250.0)),
    ("g0 p=260,150", 103, 0.0, 0.0, (260.0, 150.0)),
    ("g=5,5 p=240,200", 104, 5.0, 5.0, (240.0, 200.0)),
    ("g12=12 p=220,260", 105, 12.0, 0.0, (220.0, 260.0)),
    ("g=8,8 p=220,220", 106, 8.0, 8.0, None),
]
SEED_1F = 2000
CORR_RECORDINGS = ("sub-019", "sub-074", "sub-005")


def shaped_noise(rng, n, slope):
    """Unit-SD noise with power spectrum proportional to f^slope (FFT shaping of white noise)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (slope / 2.0)
    y = np.fft.irfft(spec * amp, n)
    return y / y.std(ddof=1)


def band_power(x, fs, lo=1.0, hi=45.0):
    """Sum of |FFT|^2 over lo..hi Hz (relative power; the same bins for both components)."""
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / fs)
    return float(X[(f >= lo) & (f <= hi)].sum())


def make_series(cfg, spec, share):
    """(segments, starts) of one synthetic series; share None = plain (S0)."""
    from src import preprocess
    from tests import sim_data
    label, seed, g12, g21, p = spec
    r = sim_data.make_recording(cfg, [30.0, 30.0], seed, g12=g12, g21=g21, p=p)
    if share is None:
        return r["segments"], r["starts"]
    fs = cfg["preprocessing"]["observation_fs_hz"]
    z = r["z_all"].T.copy()                                        # (2, n)
    out = np.empty_like(z)
    for ch in range(2):
        rng = np.random.default_rng(SEED_1F + seed * 10 + ch)
        nz = shaped_noise(rng, z.shape[1], SLOPE)
        c = np.sqrt(share / (1.0 - share) * band_power(z[ch], fs) / band_power(nz, fs))
        out[ch] = z[ch] + c * nz
    filt = preprocess.bandpass(cfg, out, fs)                       # 0.5-45 Hz, before the rescaling
    lens = [s.shape[1] for s in r["segments"]]
    segs = [filt[:, st:st + n] for st, n in zip(r["starts"], lens)]
    allx = np.concatenate(segs, axis=1)
    mu, sg = cfg["rescaling"]["mu_ref"], cfg["rescaling"]["sigma_ref"]
    mean, sd = allx.mean(axis=1, keepdims=True), allx.std(axis=1, ddof=1, keepdims=True)
    segs = [np.ascontiguousarray(mu + sg * (s - mean) / sd) for s in segs]
    return segs, r["starts"]


def _corr_worker(job):
    """Item 5: filtered y1 / y2 statistics per node for one real recording (V0, flag disabled)."""
    from threadpoolctl import threadpool_limits
    from src import passes, ukf
    from src import state_space as ss
    label, segments, starts, q, cfg = job
    calls = []
    orig_fd, orig_rf = ukf._first_divergence, ukf.run_filter

    def fd(x_post, P_post, cfg_, center, sd, min_eig, check_state=True):
        calls.append((np.array(x_post[:ss.N_NEURAL]), np.array(center)))
        return orig_fd(x_post, P_post, cfg_, center, sd, min_eig, check_state=False)

    recs = []

    def rf(*a, **k):
        calls.clear()
        res = orig_rf(*a, **k)
        recs.append({"x": np.array([c[0] for c in calls]), "c": np.array([c[1] for c in calls])})
        return res

    ukf._first_divergence, ukf.run_filter = fd, rf
    try:
        with threadpool_limits(limits=1):
            res = passes.run_pass1(segments, starts, cfg, q, forward_only=True, filter_name="19D")
    finally:
        ukf._first_divergence, ukf.run_filter = orig_fd, orig_rf
    X, Cc = [], []
    for seg, rec in zip(res.segments, recs):
        if seg.nis_keep is None or seg.diverged:
            continue
        X.append(rec["x"][seg.nis_keep])
        Cc.append(rec["c"][seg.nis_keep])
    X, Cc = np.concatenate(X), np.concatenate(Cc)
    out = {"label": label, "n": int(X.shape[0])}
    for node in (0, 1):
        y1, y2 = X[:, 6 * node + 1], X[:, 6 * node + 2]
        d1, d2 = y1 - Cc[:, 6 * node + 1], y2 - Cc[:, 6 * node + 2]
        out[f"n{node + 1}"] = {"corr_raw": float(np.corrcoef(y1, y2)[0, 1]), "corr_dev": float(np.corrcoef(d1, d2)[0, 1]),
                               "sd_y1": float(d1.std(ddof=1)), "sd_y2": float(d2.std(ddof=1)),
                               "sd_diff": float((d1 - d2).std(ddof=1)), "sd_sum": float((d1 + d2).std(ddof=1)),
                               "max_abs_y1": float(np.abs(d1).max()), "max_abs_y2": float(np.abs(d2).max())}
    return out


def main():
    from src.config import load_config
    from src import state_space as ss
    from joblib import Parallel, delayed
    out_path = Path(sys.argv[1])
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfg = load_config()
    fs = cfg["preprocessing"]["observation_fs_hz"]
    variants = [("S0", None), ("S1", SHARES["S1"]), ("S2", SHARES["S2"])]
    data, jobs = {}, []
    for vname, share in variants:
        for spec in SERIES:
            segs, starts = make_series(cfg, spec, share)
            data[(vname, spec[0])] = segs
            jobs += [((vname, spec[0]), segs, starts, q, cfg, "V0") for q in Q_VALUES]
    print(f"{len(jobs)} runs (6 series x 3 variants x {len(Q_VALUES)} q)", flush=True)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(C._worker)(dict_job(j)) for j in jobs)
    print(f"wall {time.perf_counter() - t0:.0f} s", flush=True)
    for (j, r) in zip(jobs, res):
        r["series"], r["variant"] = j[0][1], j[0][0]
    keys = [(v, q) for v, _ in variants for q in Q_VALUES]
    rows = {k: [r for r in res if (r["variant"], r["q"]) == k] for k in keys}
    labels = [s[0] for s in SERIES]

    print("\n=== 1. y1 exceedance (both nodes pooled); median [max] over the 6 series ===")
    print(f"  {'variant':<8}{'q':>7} " + "".join(f"{'>' + str(k) + ' SD':>17}" for k in C.THRESHOLDS) + f"{'max SD reached':>18}{'any state >10':>17}")
    for k in keys:
        rr = [r for r in rows[k] if "frac_y1" in r]
        print(f"  {k[0]:<8}{k[1]:>7g} " + "".join(f"{C._med_max([r['frac_y1'][str(t)] for r in rr]):>17}" for t in C.THRESHOLDS)
              + f"{np.median([r['max_y1'] for r in rr]):>9.1f} [{max(r['max_y1'] for r in rr):5.1f}]".rjust(18)
              + f"{C._med_max([r['any10'] for r in rr]):>17}")

    print("\n=== 2. parameter excursion from the prior mean, max per series, in prior SD; median [max] over series ===")
    pk = ss.PARAM_NAMES_FULL
    print(f"  {'variant':<8}{'q':>7} " + "".join(f"{p:>16}" for p in pk))
    for k in keys:
        rr = [r for r in rows[k] if "excursion_prior_sd" in r]
        print(f"  {k[0]:<8}{k[1]:>7g} " + "".join(f"{np.median([r['excursion_prior_sd'][p] for r in rr]):7.2f} [{max(r['excursion_prior_sd'][p] for r in rr):5.1f}]".rjust(16) for p in pk))

    print("\n=== 3. mean NIS per series [fraction of NIS samples above 10] ===")
    for q in Q_VALUES:
        print(f"  q = {q:g}")
        print(f"    {'series':<18}" + "".join(f"{v:>16}" for v, _ in variants))
        for lab in labels:
            cells = []
            for v, _ in variants:
                r = next(x for x in rows[(v, q)] if x["series"] == lab)
                cells.append((f"{r['mean_nis']:.2f} [{r['frac_nis_gt10']:.3f}]" if "mean_nis" in r else "none").rjust(16))
            print(f"    {lab:<18}" + "".join(cells))
        print(f"    {'median':<18}" + "".join(f"{np.median([x['mean_nis'] for x in rows[(v, q)] if 'mean_nis' in x]):>16.2f}" for v, _ in variants))

    print("\n=== 4. spectra after the pipeline (2-s Hann Welch, 1-45 Hz; slope excludes 7-14 Hz; alpha = 8-13 Hz share) ===")
    print(f"  {'variant':<8}{'series':<18}{'peak Hz':>8}{'peak resid':>11}{'alpha share':>12}{'slope':>8}")
    med = {}
    for vname, _ in variants:
        ms = []
        for lab in labels:
            f, p = C._observed_psd(data[(vname, lab)], fs)
            m = C._spectral_metrics(f, p)
            ms.append(m)
            print(f"  {vname:<8}{lab:<18}{m['peak_hz']:>8.1f}{m['peak_over_aperiodic_log10']:>11.2f}{m['alpha_share']:>12.3f}{m['slope']:>8.2f}")
        med[vname] = ms
    from src import preprocess as pp
    from src import tuning
    pilot = pp.load_pilot_ids(cfg, ROOT)
    loader = tuning.make_real_loader(cfg, ROOT, pilot, allow_all=False)      # development guard stays on
    real, real_data = [], {}
    for sid in sorted(pilot):
        r = loader(sid)
        if r["reason"] is not None:
            continue
        real_data[sid] = (r["segments"], r["starts"])
        f, p = C._observed_psd(r["segments"], fs)
        real.append(C._spectral_metrics(f, p))
    print("  summary, median [min, max] over series / recordings:")
    for vname, _ in variants:
        ms = med[vname]
        print(f"    {vname:<12} slope {np.median([m['slope'] for m in ms]):6.2f} [{min(m['slope'] for m in ms):5.2f}, {max(m['slope'] for m in ms):5.2f}]   "
              f"alpha share {np.median([m['alpha_share'] for m in ms]):.3f} [{min(m['alpha_share'] for m in ms):.3f}, {max(m['alpha_share'] for m in ms):.3f}]")
    print(f"    {'real pilot':<12} slope {np.median([m['slope'] for m in real]):6.2f} [{min(m['slope'] for m in real):5.2f}, {max(m['slope'] for m in real):5.2f}]   "
          f"alpha share {np.median([m['alpha_share'] for m in real]):.3f} [{min(m['alpha_share'] for m in real):.3f}, {max(m['alpha_share'] for m in real):.3f}]  ({len(real)} recordings)")

    print("\n=== 5. real data, V0, q = 1e-2: filtered y1 and y2 per node (deviation from the reference, mV; post-burn-in) ===")
    cj = [(s, real_data[s][0], real_data[s][1], 1e-2, cfg) for s in CORR_RECORDINGS if s in real_data]
    cres = Parallel(n_jobs=min(n_jobs, len(cj)), backend="loky")(delayed(_corr_worker)(j) for j in cj)
    print(f"  {'recording':<9}{'node':<5}{'corr y1,y2':>11}{'corr (dev)':>11}{'SD y1':>8}{'SD y2':>8}{'SD y1-y2':>9}{'SD y1+y2':>9}{'max|y1|':>8}{'max|y2|':>8}")
    for r in cres:
        for nd in ("n1", "n2"):
            s = r[nd]
            print(f"  {r['label']:<9}{nd:<5}{s['corr_raw']:>11.3f}{s['corr_dev']:>11.3f}{s['sd_y1']:>8.2f}{s['sd_y2']:>8.2f}{s['sd_diff']:>9.2f}{s['sd_sum']:>9.2f}{s['max_abs_y1']:>8.1f}{s['max_abs_y2']:>8.1f}")
    out_path.write_text(json.dumps({"runs": res, "corr": cres}))
    bad = [(r["variant"], r["series"], r["q"], r["failures"]) for r in res if r["failures"]]
    print("\nNaN/Inf or covariance-not-PD failures (state flag disabled): " + ("none" if not bad else str(bad)))


def dict_job(j):
    """Adapt (label, segments, starts, q, cfg, variant) with a (variant, series) label to the C4c worker."""
    (lab, segments, starts, q, cfg, variant) = j
    return (lab[1], segments, starts, q, cfg, variant)


if __name__ == "__main__":
    main()
