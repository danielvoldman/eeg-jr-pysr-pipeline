"""E8: DIAGNOSTIC ONLY, REAL PILOT DATA (PLAN.md E8; §7.4, §7.5, §17, DEV-003, DEV-005). 19-D filter vs candidates A and B on
the 12 pilot recordings.

Pilot subjects only, ses-t1 only (the development guard of tuning.make_real_loader stays on; no test subject is ever read).
Changes no src rule, threshold or config value, writes nothing under outputs/ (only results/pilot/e8_real_pilot_comparison.json),
runs no PySR. The state-SD flag is disabled DIAGNOSTICALLY in a config copy (state_sd_multiple = inf; NaN/Inf and covariance-PD
checks stay on), as in E4 to E7. Forward-only pass 1 (passes.run_pass1, Numba backend) at q = 1e-2 and 1e-3, per option
(19D, A, B), with the standard divergence rule ON and with the flag OFF.

Because the Numba kernels keep the divergence reference inside, the per-step deviations of the neural states from the reference
are recomputed AFTER each run by replaying ukf.DivergenceReference (the NumPy oracle of the in-kernel tracker, IMP-029) over
the filtered states; the deviation is in units of the prior neural SD, as in C4b / C4c.

Per option and q: (rule ON) segments and samples dropped, recordings with no surviving segment; (flag OFF) the maximum parameter
excursion in prior SD (p, log rho, g, m; m clipped as in pass 1), mean NIS and the fraction of NIS above 10 (samples after both
burn-ins), the fraction of samples with y1 beyond 10 SD and the largest y1 deviation, any negative-eigenvalue / NaN / PD event,
and the distribution of the filtered gains; for A and B the fitted aperiodic (s2, tau) per recording and whether the filtered
noise state carries an alpha peak (is it absorbing neural signal?).

    python tools/e8_real_pilot_comparison.py [--n-jobs 4] [--report-only]
"""
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

Q_VALUES = (1e-2, 1e-3)
OPTIONS = ("19D", "A", "B")
MODES = ("on", "off")
Y1 = (1, 7)                         # n1.y1, n2.y1 among the 12 neural states (as C4b / C4c)
EXCEED_SD = 10.0                    # the rule's multiple, for the y1 exceedance fraction (a measurement, not the rule)
NIS_HIGH = 10.0                     # as C4b / C4c
FLANK_GAP_HZ, FLANK_WIDTH_HZ = 1.0, 3.0   # alpha prominence: alpha band over the two flanks 1 to 4 Hz outside it (script-level)
GROUPS = {"p": ("p1", "p2"), "log_rho": ("log_rho1", "log_rho2"), "g": ("g12", "g21"), "m": ("m",)}
# C4b / C4c / C4d numbers of the 19-D filter on the same recordings, for the comparison (PLAN.md notes of C4b, C4c, C4d)
REFERENCE = {"y1_gt10_median_q1e-2": 0.021, "y1_max_sd_q1e-2": 39.5, "p_excursion_prior_sd_q1e-2": 21.0,
             "nis_median_q1e-2": 2.06, "excursion_range_prior_sd_q>=1e-3": (9.0, 26.0), "nis_range_q1e-2": (1.39, 2.84),
             "slope_model": (-0.8, -1.0), "slope_observed": (-1.6, -1.7)}


# ---------------------------------------------------------------- helpers (tested)

def med_max(values):
    """(median, max) of a list; (None, None) for an empty list."""
    v = [x for x in values if x is not None]
    return (float(np.median(v)), float(np.max(v))) if v else (None, None)


def param_excursions(layout, X, cfg):
    """Largest |parameter - prior mean| / prior SD over the rows of the filtered states X (T, n), per group of
    GROUPS (the maximum over the group's members); m is clipped to its truncation range first, as in pass 1."""
    from src import state_space as ss
    pm = layout.params(ss.prior_mean(layout, cfg)[None])
    psd = dict(ss.parameter_prior_sd(cfg))
    psd["m"] = cfg["priors"]["m_sd"]
    lo, hi = cfg["priors"]["m_truncate"]
    par = layout.params(np.asarray(X, dtype=np.float64))
    par["m"] = np.clip(par["m"], lo, hi)
    per = {k: float(np.max(np.abs(par[k] - float(pm[k][0]))) / psd[k]) for k in ss.PARAM_NAMES_FULL}
    return {g: max(per[k] for k in members) for g, members in GROUPS.items()}


def reference_deviations(layout, X, cfg):
    """|x_neural - reference| / sd per step in units of the prior neural SD (the quantity of the §7.5 rule), with the
    reference tracked exactly as the rule does (ukf.DivergenceReference, replayed over the filtered states in order)."""
    from src import state_space as ss
    from src import ukf
    sd = np.sqrt(np.tile(ss.neural_variance(cfg), ss.N_NODES))
    ref = ukf.DivergenceReference(layout, cfg)
    X = np.asarray(X, dtype=np.float64)
    out = np.empty((X.shape[0], ss.N_NEURAL))
    for t in range(X.shape[0]):
        out[t] = np.abs(X[t, :ss.N_NEURAL] - np.asarray(ref.center(X[t]))) / sd
    return out


def y1_exceedance(dev, keep, threshold=EXCEED_SD):
    """(fraction of kept (sample, node) pairs with y1 beyond `threshold` SD, largest y1 deviation) from the deviations dev
    (T, 12) and the boolean keep mask (T,)."""
    d = np.asarray(dev)[np.asarray(keep)][:, list(Y1)]
    if d.size == 0:
        return None, None
    return float(np.mean(d > threshold)), float(d.max())


def alpha_metrics(f, psd_noise, psd_obs, band):
    """Per channel: alpha prominence (mean PSD in the alpha band over the mean of the two flanks FLANK_GAP_HZ to
    FLANK_GAP_HZ + FLANK_WIDTH_HZ outside it) of the noise state and of the observed channel, and the share of the observed
    alpha-band power that the noise state carries (band sums of the two PSDs)."""
    lo, hi = band
    sel = (f >= lo) & (f <= hi)
    out = []
    for c in range(psd_noise.shape[0]):
        def prom(p):
            flank = np.concatenate([p[(f >= lo - FLANK_GAP_HZ - FLANK_WIDTH_HZ) & (f < lo - FLANK_GAP_HZ)],
                                    p[(f > hi + FLANK_GAP_HZ) & (f <= hi + FLANK_GAP_HZ + FLANK_WIDTH_HZ)]])
            return float(np.mean(p[sel]) / np.mean(flank))
        out.append({"prominence_noise": prom(psd_noise[c]), "prominence_observed": prom(psd_obs[c]),
                    "alpha_share_of_observed": float(np.sum(psd_noise[c][sel]) / np.sum(psd_obs[c][sel]))})
    return out


# ---------------------------------------------------------------- one run

def _worker(job):
    """One (recording, option, q, mode). Top level for the spawn start method; returns small scalars only."""
    from threadpoolctl import threadpool_limits
    from src import passes, ukf, ukf_ext
    from src import state_space as ss
    from src import synthetic_gate as sg

    sid, segments, starts, option, q, mode, cfg0 = job
    cfg = sg.g0_cfg(cfg0)
    if mode == "off":
        cfg["ukf"]["divergence"]["state_sd_multiple"] = float("inf")
    fs = cfg["preprocessing"]["observation_fs_hz"]
    captured = []
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        orig, orig_mk = ukf.run_filter, ukf_ext.make_runners

        def rf(*a, **k):                                   # 19D: the plain ukf filter
            r = orig(*a, **k)
            captured.append(r)
            return r

        def mk(spec, *a, **k):                             # A and B: the extended runners of passes.py
            f, sm = orig_mk(spec, *a, **k)

            def g(*aa, **kk):
                r = f(*aa, **kk)
                captured.append(r)
                return r
            return g, sm

        ukf.run_filter, ukf_ext.make_runners = rf, mk
        try:
            res = passes.run_pass1(segments, starts, cfg, q, forward_only=True, filter_name=option)
        finally:
            ukf.run_filter, ukf_ext.make_runners = orig, orig_mk
        wall = time.perf_counter() - t0
        if len(captured) != len(res.segments):
            raise RuntimeError("filter calls and segments do not match")
        layout = res.layout
        out = {"recording": sid, "option": option, "q": q, "mode": mode, "wall_s": wall, "n_segments": len(res.segments),
               "n_segments_dropped": int(sum(s.diverged for s in res.segments)), "n_samples": int(res.n_clean),
               "n_samples_dropped": int(res.n_diverged), "recording_diverged": bool(res.recording_diverged),
               "reasons": sorted({str(s.reason) for s in res.segments if s.diverged}),
               "stability": sg.stability_summary([s.monitor for s in res.segments])}
        nis_all, X_all, X_keep, dev_keep, noise, obs = [], [], [], [], [], []
        nb = layout.n
        for seg, r, z in zip(res.segments, captured, segments):
            n = int(r.n_done)
            if n == 0:
                continue
            X_all.append(np.asarray(r.x[:n]))
            if seg.diverged or seg.nis_keep is None:
                continue
            keep = np.asarray(seg.nis_keep)
            nis_all.append(np.asarray(seg.nis)[keep])
            X_keep.append(np.asarray(r.x[:n])[keep])
            if mode == "off":
                dev_keep.append(reference_deviations(layout, r.x[:n], cfg)[keep])
            if option != "19D":
                noise.append(np.asarray(r.full.x[:n, nb:nb + 2])[keep].T)
                obs.append(np.asarray(z, dtype=np.float64)[:, :n][:, keep])
        if X_all:
            out["excursion"] = param_excursions(layout, np.concatenate(X_all), cfg)
        if nis_all:
            N = np.concatenate(nis_all)
            out["nis_mean"], out["nis_frac_gt10"], out["n_kept"] = float(np.mean(N)), float(np.mean(N > NIS_HIGH)), int(N.size)
            par = layout.params(np.concatenate(X_keep))
            out["gains"] = {k: {"mean": float(np.mean(par[k])), "median": float(np.median(par[k])),
                                "q25": float(np.percentile(par[k], 25)), "q75": float(np.percentile(par[k], 75))} for k in ("g12", "g21")}
        if dev_keep:
            D = np.concatenate(dev_keep)
            d = D[:, list(Y1)]
            out["y1_gt10"], out["y1_max_sd"] = float(np.mean(d > EXCEED_SD)), float(d.max())
        if noise and mode == "off":
            try:
                f, pn = ukf_ext.pooled_psd(noise, fs, cfg)
                _, po = ukf_ext.pooled_psd(obs, fs, cfg)
                out["alpha"] = alpha_metrics(f, pn, po, cfg["g0"]["features"]["alpha_band_hz"])
            except ukf.UKFError:
                out["alpha"] = None
    return out


# ---------------------------------------------------------------- aggregation and report

def _cell(rows, key, sub=None, fmt="{:.2f}"):
    v = [r[key] if sub is None else (r.get(key) or {}).get(sub) for r in rows]
    med, mx = med_max(v)
    return "-" if med is None else (fmt + " [" + fmt + "]").format(med, mx)


def aggregate(runs):
    """{(option, q, mode): list of run dicts} from the flat run list."""
    out = {}
    for r in runs:
        out.setdefault((r["option"], r["q"], r["mode"]), []).append(r)
    return out


def report_lines(doc):
    """Compact tables (median [max] over the recordings) from the written document."""
    agg = aggregate(doc["runs"])
    L = []
    n_rec = len({r["recording"] for r in doc["runs"]})
    L.append(f"#### E8 (diagnostic only, REAL pilot data): {n_rec} recordings (ses-t1, training side); forward-only pass 1, Numba; "
             f"median [max] over the recordings")
    L.append("#### 1. standard divergence rule ON: segments and samples dropped")
    L.append(f"{'option':<7}{'q':>7}  {'segments dropped':>20}{'samples dropped %':>22}{'recordings with every segment dropped':>40}")
    for o in OPTIONS:
        for q in Q_VALUES:
            rows = agg.get((o, q, "on"), [])
            if not rows:
                continue
            seg = [r["n_segments_dropped"] for r in rows]
            smp = [100.0 * r["n_samples_dropped"] / r["n_samples"] for r in rows]
            allgone = sum(1 for r in rows if r["n_segments_dropped"] == r["n_segments"])
            L.append(f"{o:<7}{q:>7g}  {_fmt(seg, '{:.0f}'):>20}{_fmt(smp, '{:.1f}'):>22}{allgone:>34d} of {len(rows)}")
    L.append("#### 2. state-SD flag OFF (DIAGNOSTIC ONLY): parameter excursion (prior SD), NIS, y1 beyond 10 SD, stability")
    L.append(f"{'option':<7}{'q':>7}  {'p':>14}{'log rho':>14}{'g':>14}{'m':>14}  {'mean NIS':>13}{'NIS>10':>14}{'y1>10 SD':>16}{'max y1 SD':>13}  events")
    for o in OPTIONS:
        for q in Q_VALUES:
            rows = agg.get((o, q, "off"), [])
            if not rows:
                continue
            ev = sum(r["stability"]["n_negative_eig_steps"] + r["stability"]["n_nan_inf"] + r["stability"]["n_linalg_divergences"] for r in rows)
            L.append(f"{o:<7}{q:>7g}  " + "".join(f"{_cell(rows, 'excursion', g, '{:.1f}'):>14}" for g in GROUPS) +
                     f"  {_cell(rows, 'nis_mean'):>13}{_cell(rows, 'nis_frac_gt10', None, '{:.3f}'):>14}{_cell(rows, 'y1_gt10', None, '{:.3f}'):>16}"
                     f"{_cell(rows, 'y1_max_sd', None, '{:.1f}'):>13}  {ev}")
    L.append("#### 3. filtered gains over the recordings (time-mean per recording; median and IQR across the recordings), flag OFF; "
             "rule ON in brackets = surviving recordings / median of their mean")
    L.append(f"{'option':<7}{'q':>7}  {'g12 median (IQR)':>22}{'g21 median (IQR)':>22}   {'rule ON: recordings with estimates, g12/g21 median':>50}")
    for o in OPTIONS:
        for q in Q_VALUES:
            off = [r for r in agg.get((o, q, "off"), []) if r.get("gains")]
            on = [r for r in agg.get((o, q, "on"), []) if r.get("gains")]
            if not off:
                continue
            cells = []
            for k in ("g12", "g21"):
                v = [r["gains"][k]["mean"] for r in off]
                cells.append(f"{np.median(v):6.2f} ({np.percentile(v, 25):5.2f}, {np.percentile(v, 75):5.2f})")
            onc = (f"{len(on)} of {len(agg.get((o, q, 'on'), []))}: " +
                   ("/".join(f"{np.median([r['gains'][k]['mean'] for r in on]):.2f}" for k in ("g12", "g21")) if on else "-"))
            L.append(f"{o:<7}{q:>7g}  {cells[0]:>22}{cells[1]:>22}   {onc:>50}")
    L.append("#### 4. fitted aperiodic (s2 as a fraction of the channel variance, tau in s) per recording, channels 1 / 2 (A and B use these)")
    for r in doc["aperiodic"]:
        L.append(f"  {r['recording']:<8} s2/var {r['s2_over_var'][0]:.3f} / {r['s2_over_var'][1]:.3f}   tau {r['tau'][0]:.3f} / {r['tau'][1]:.3f}"
                 f"   at the tau upper bound: {r['tau_at_upper'][0]} / {r['tau_at_upper'][1]}")
    L.append("#### 5. does the filtered noise state carry an alpha peak? (flag OFF; prominence = alpha-band PSD over the flank PSD; "
             "share = noise-state alpha power / observed alpha power; channels pooled)")
    L.append(f"{'option':<7}{'q':>7}  {'noise-state prominence':>24}{'observed prominence':>22}{'alpha share':>16}")
    for o in ("A", "B"):
        for q in Q_VALUES:
            rows = [r for r in agg.get((o, q, "off"), []) if r.get("alpha")]
            if not rows:
                continue
            pn = [np.mean([c["prominence_noise"] for c in r["alpha"]]) for r in rows]
            po = [np.mean([c["prominence_observed"] for c in r["alpha"]]) for r in rows]
            sh = [np.mean([c["alpha_share_of_observed"] for c in r["alpha"]]) for r in rows]
            L.append(f"{o:<7}{q:>7g}  {_fmt(pn, '{:.2f}'):>24}{_fmt(po, '{:.2f}'):>22}{_fmt(sh, '{:.3f}'):>16}")
    ref = doc["reference"]
    L.append("#### reference, 19-D filter on the same recordings (C4b / C4c / C4d, state flag off): at q = 1e-2 the median y1 beyond 10 SD "
             f"{ref['y1_gt10_median_q1e-2']}, the largest y1 deviation up to {ref['y1_max_sd_q1e-2']} SD, the p excursion {ref['p_excursion_prior_sd_q1e-2']} "
             f"prior SD, the median NIS {ref['nis_median_q1e-2']} (range {ref['nis_range_q1e-2'][0]} to {ref['nis_range_q1e-2'][1]}); at q >= 1e-3 the "
             f"filtered parameters ran {ref['excursion_range_prior_sd_q>=1e-3'][0]:.0f} to {ref['excursion_range_prior_sd_q>=1e-3'][1]:.0f} prior SD from the prior")
    return L


def _fmt(values, fmt):
    med, mx = med_max(values)
    return "-" if med is None else (fmt + " [" + fmt + "]").format(med, mx)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    from src import preprocess as pp
    from src import synthetic_gate as sg
    from src import tuning, ukf_ext
    from src.config import load_config
    cfg = load_config()
    path = ROOT / cfg["paths"]["pilot_results_dir"] / "e8_real_pilot_comparison.json"
    if args.report_only:
        for line in report_lines(json.loads(path.read_text(encoding="utf-8"))):
            sg.emit(line)
        return 0
    from joblib import Parallel, delayed
    n_workers = sg.pool_size(cfg, args.n_jobs)
    pilot = pp.load_pilot_ids(cfg, ROOT)
    loader = tuning.make_real_loader(cfg, ROOT, pilot, allow_all=False)          # the development guard stays on
    data, skipped = {}, []
    for sid in sorted(pilot):
        r = loader(sid)
        if r["reason"] is not None:
            skipped.append((sid, r["reason"]))
            continue
        data[sid] = (r["segments"], r["starts"])
    sg.emit(f"== E8 (diagnostic only): {len(data)} pilot ses-t1 recordings, skipped {skipped}; options {list(OPTIONS)}; q {list(Q_VALUES)}; "
            f"{n_workers} workers ==")
    jobs = [(sid, seg, st, o, q, m, cfg) for sid, (seg, st) in data.items() for o in OPTIONS for q in Q_VALUES for m in MODES]
    t0 = time.perf_counter()
    runs = Parallel(n_jobs=n_workers, backend=cfg["compute"]["joblib_backend"])(delayed(_worker)(j) for j in jobs)
    wall = time.perf_counter() - t0
    aper = []
    for sid, (seg, _) in data.items():
        est = ukf_ext.estimate(seg, cfg)
        upper = cfg["ukf"]["aperiodic"]["tau_bounds_s"][1]
        aper.append({"recording": sid, "s2": est["s2"].tolist(), "tau": est["tau"].tolist(), "var": est["var"].tolist(),
                     "s2_over_var": (est["s2"] / est["var"]).tolist(), "tau_at_upper": [bool(t >= 0.999 * upper) for t in est["tau"]]})
    doc = {"stage": "E8 diagnostic only (REAL pilot data, ses-t1, no PySR, no verdict)", "dev005_decision": "not decided; evidence only",
           "recordings": sorted(data), "skipped": skipped, "q_values": list(Q_VALUES), "options": list(OPTIONS), "runs": runs,
           "aperiodic": aper, "reference": {k: v for k, v in REFERENCE.items()}, "wall_s": wall, "n_workers": n_workers}
    sg.write_pilot_json(cfg, ROOT, doc, path)
    for line in report_lines(json.loads(json.dumps(sg._jsonable(doc)))):
        sg.emit(line)
    sg.emit(f"wall {wall / 60:.1f} min; written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
