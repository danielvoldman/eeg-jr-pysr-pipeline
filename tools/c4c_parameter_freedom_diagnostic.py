"""C4c: read-only diagnostic of parameter freedom versus the y1 excursions (PLAN.md C4c).

Pilot subjects only (ses-t1). Changes no src rule, threshold or config value and writes nothing under
outputs/. Every variant is applied inside this script's own worker processes (a config copy, or a
monkeypatch that is undone), and the state-SD flag is disabled the way tools/c4b_divergence_diagnostic.py
does it (ukf._first_divergence replaced by a recording wrapper that calls the original with
check_state=False, so NaN/Inf and covariance-PD checks stay on).

Variants (each at q = 1e-2 and 1e-3, pass-1 style forward filtering):
  V0 baseline
  V1 parameters frozen at the prior means: parameter random-walk factor 0 and parameter prior
     covariance x 1e-12 (ss.prior_cov patched in the worker)
  V2 parameter random-walk factor x 0.1
  V3 parameter random-walk factor x 10

    python tools/c4c_parameter_freedom_diagnostic.py <raw_results.json> [n_jobs]
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
VARIANTS = ("V0", "V1", "V2", "V3")
THRESHOLDS = (3, 5, 10, 20)
Y1 = (1, 7)                                   # n1.y1, n2.y1 in the 12 neural states
SPECTRUM_RECORDINGS = ("sub-019", "sub-074", "sub-005")
N_REALIZATIONS = 20
SIM_SEED = 7000
SIM_SECONDS = 60.0
SIM_BURN_S = 4.0
SLOPE_EXCLUDE_HZ = (7.0, 14.0)


def _variant_cfg(cfg, variant):
    import copy
    c = copy.deepcopy(cfg)
    f = c["ukf"]["process_noise"]["parameter_random_walk_factor"]
    c["ukf"]["process_noise"]["parameter_random_walk_factor"] = {"V0": f, "V1": 0.0, "V2": f * 0.1, "V3": f * 10.0}[variant]
    return c


def _worker(job):
    """One (recording, variant, q). Top level for the spawn start method."""
    from threadpoolctl import threadpool_limits
    from src import passes, ukf
    from src import state_space as ss

    label, segments, starts, q, cfg0, variant = job
    cfg = _variant_cfg(cfg0, variant)
    fs = cfg["preprocessing"]["observation_fs_hz"]
    n_exempt = int(round(cfg["ukf"]["divergence"]["startup_exempt_s"] * fs))
    sd1 = float(np.sqrt(ss.neural_variance(cfg)[1]))
    calls = []
    orig_fd, orig_rf, orig_pc = ukf._first_divergence, ukf.run_filter, ss.prior_cov

    def fd(x_post, P_post, cfg_, center, sd, min_eig, check_state=True):
        if np.all(np.isfinite(x_post)) and np.all(np.isfinite(P_post)):
            dev = np.abs(x_post[:ss.N_NEURAL] - center) / sd
            psd = np.sqrt(np.maximum(np.diag(P_post)[list(Y1)], 0.0))
        else:
            dev, psd = np.full(ss.N_NEURAL, np.nan), np.full(2, np.nan)
        calls.append((dev, np.array(x_post), psd))
        return orig_fd(x_post, P_post, cfg_, center, sd, min_eig, check_state=False)

    segrecs = []

    def rf(*a, **k):
        calls.clear()
        res = orig_rf(*a, **k)
        segrecs.append({"dev": np.array([c[0] for c in calls]), "x": np.array([c[1] for c in calls]),
                        "psd": np.array([c[2] for c in calls]), "nis": res.nis.copy(),
                        "n_done": res.n_done, "diverged": res.diverged, "reason": res.divergence_reason})
        return res

    def pc(layout, cfg_):
        P = orig_pc(layout, cfg_).copy()
        P[ss.N_NEURAL:, ss.N_NEURAL:] *= 1e-12
        return P

    ukf._first_divergence, ukf.run_filter = fd, rf
    if variant == "V1":
        ss.prior_cov = pc
    try:
        with threadpool_limits(limits=1):
            t0 = time.perf_counter()
            res = passes.run_pass1(segments, starts, cfg, q, forward_only=True)
            wall = time.perf_counter() - t0
    finally:
        ukf._first_divergence, ukf.run_filter, ss.prior_cov = orig_fd, orig_rf, orig_pc

    layout = res.layout
    prior_layout_mean = ss.prior_mean(layout, cfg)
    pm = layout.params(prior_layout_mean[None])
    psd_prior = dict(ss.parameter_prior_sd(cfg)); psd_prior["m"] = cfg["priors"]["m_sd"]
    lo, hi = cfg["priors"]["m_truncate"]
    keep_dev, keep_nis, xs, per_seg, failures = [], [], [], [], []
    for seg, rec in zip(res.segments, segrecs):
        n = len(rec["dev"]) if rec["diverged"] else rec["n_done"]
        if rec["diverged"]:
            failures.append(f"seg {seg.index}: {rec['reason']} at step {seg.step}")
        xs.append(rec["x"][:n])
        if seg.nis_keep is None or seg.diverged:
            continue
        k = seg.nis_keep
        d, p, nis = rec["dev"][k], rec["psd"][k], rec["nis"][k]
        keep_dev.append(d)
        keep_nis.append(nis)
        if variant == "V0":
            dy = d[:, list(Y1)]                                    # SD units of the prior neural SD
            post_units = dy * sd1 / p                              # |x - ref| / posterior SD
            f_prior, f_post = dy > 10.0, post_units > 10.0
            per_seg.append({"segment": seg.index, "n": int(k.sum()),
                            "post_sd_over_prior_sd_median": float(np.median(p / sd1)),
                            "post_sd_over_prior_sd_max": float(np.max(p / sd1)),
                            "max_dev_prior_sd": float(dy.max()), "max_dev_post_sd": float(post_units.max()),
                            "n_prior10": int(f_prior.sum()), "n_post10": int(f_post.sum()),
                            "n_both": int((f_prior & f_post).sum()), "n_pairs": int(f_prior.size)})
    out = {"label": label, "variant": variant, "q": q, "wall_s": wall, "failures": failures,
           "recording_diverged": res.recording_diverged, "per_segment": per_seg}
    if keep_dev:
        D, N = np.concatenate(keep_dev), np.concatenate(keep_nis)
        Dy = D[:, list(Y1)]
        out["n_kept"] = int(D.shape[0])
        out["frac_y1"] = {str(k): float(np.mean(Dy > k)) for k in THRESHOLDS}
        out["max_y1"] = float(Dy.max())
        out["any10"] = float(np.mean(np.any(D > 10.0, axis=1)))
        out["mean_nis"], out["frac_nis_gt10"] = float(np.mean(N)), float(np.mean(N > 10.0))
    if any(len(x) for x in xs):
        X = np.concatenate([x for x in xs if len(x)])
        par = layout.params(X)
        par["m"] = np.clip(par["m"], lo, hi)
        out["excursion_prior_sd"] = {k: float(np.max(np.abs(par[k] - float(pm[k][0]))) / psd_prior[k]) for k in ss.PARAM_NAMES_FULL}
        out["median_params"] = {k: float(np.median(par[k])) for k in ss.PARAM_NAMES_FULL}
    return out


def _real_jobs(cfg):
    from src import preprocess as pp
    from src import tuning
    pilot = pp.load_pilot_ids(cfg, ROOT)
    loader = tuning.make_real_loader(cfg, ROOT, pilot, allow_all=False)   # the development guard stays on
    jobs, skipped, data = [], [], {}
    for sid in sorted(pilot):
        r = loader(sid)
        if r["reason"] is not None:
            skipped.append((sid, r["reason"]))
            continue
        data[sid] = (r["segments"], r["starts"])
        jobs += [(sid, r["segments"], r["starts"], q, cfg, v) for v in VARIANTS for q in Q_VALUES]
    return jobs, skipped, data


# ---- spectra (item 5) --------------------------------------------------------------------------------------

def _welch(x, fs):
    from scipy.signal import welch
    f, p = welch(x, fs=fs, window="hann", nperseg=int(round(2 * fs)), noverlap=int(round(fs)), detrend="constant")
    return f, p


def _observed_psd(segments, fs):
    tot, n = 0.0, 0
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float64)
        if seg.shape[1] < int(round(2 * fs)):
            continue
        nwin = (seg.shape[1] - int(round(2 * fs))) // int(round(fs)) + 1
        for ch in range(seg.shape[0]):
            f, p = _welch(seg[ch], fs)
            tot = tot + p * nwin
            n += nwin
    return f, tot / n


def _model_psd(cfg, params, with_noise):
    import math
    from src import model
    from src import state_space as ss
    fs_sim = cfg["rescaling"]["reference_simulation"]["sim_fs_hz"]
    fs = cfg["preprocessing"]["observation_fs_hz"]
    step = int(round(fs_sim / fs))
    mu, sigma = cfg["rescaling"]["mu_ref"], cfg["rescaling"]["sigma_ref"]
    A1, B1 = (float(v) for v in ss.rho_to_AB(np.exp(params["log_rho1"]), cfg))
    A2, B2 = (float(v) for v in ss.rho_to_AB(np.exp(params["log_rho2"]), cfg))
    m = params["m"]
    M = np.array([[1.0, m], [m, 1.0]])
    burn = int(round(SIM_BURN_S * fs))
    tot = 0.0
    for i in range(N_REALIZATIONS):
        res = model.simulate(cfg, int((SIM_BURN_S + SIM_SECONDS) * fs_sim), seed=SIM_SEED + i, n_nodes=2,
                             p=(params["p1"], params["p2"]), A=(A1, A2), B=(B1, B2), g12=params["g12"], g21=params["g21"])
        st = res.states[step::step]
        y = st[:, :, 1] - st[:, :, 2]
        z = mu + (y - mu) @ M.T
        if with_noise:
            rng = np.random.default_rng(SIM_SEED + 500 + i)
            z = z + rng.normal(size=z.shape) * math.sqrt(cfg["ukf"]["observation_noise"]["R_fraction_of_rescaled_variance"]) * sigma
        z = z[burn:]
        if not np.all(np.isfinite(z)):
            raise ValueError("non-finite model output")
        for ch in range(2):
            f, p = _welch(z[:, ch], fs)
            tot = tot + p
    return f, tot / (2 * N_REALIZATIONS)


def _spectral_metrics(f, p):
    band = (f >= 1.0) & (f <= 45.0)
    fb, pb = f[band], p[band]
    lp, lf = np.log10(pb), np.log10(fb)
    fit_mask = ~((fb >= SLOPE_EXCLUDE_HZ[0]) & (fb <= SLOPE_EXCLUDE_HZ[1]))
    slope, icpt = np.polyfit(lf[fit_mask], lp[fit_mask], 1)
    resid = lp - (slope * lf + icpt)
    alpha = (fb >= 8.0) & (fb <= 13.0)
    return {"peak_hz": float(fb[int(np.argmax(resid))]), "peak_over_aperiodic_log10": float(resid.max()),
            "alpha_share": float(pb[alpha].sum() / pb.sum()), "slope": float(slope)}


# ---- printing ------------------------------------------------------------------------------------------------

def _med_max(v):
    v = np.asarray(v, dtype=float)
    return f"{np.median(v):6.3f} [{v.max():6.3f}]"


def main():
    from src.config import load_config
    from src import state_space as ss
    out_path = Path(sys.argv[1])
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfg = load_config()
    from joblib import Parallel, delayed
    jobs, skipped, data = _real_jobs(cfg)
    print(f"{len(jobs)} runs ({len(jobs) // (len(VARIANTS) * len(Q_VALUES))} pilot ses-t1 recordings x {len(VARIANTS)} variants x "
          f"{len(Q_VALUES)} q), skipped {skipped}", flush=True)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_worker)(j) for j in jobs)
    print(f"wall {time.perf_counter() - t0:.0f} s", flush=True)
    out_path.write_text(json.dumps({"runs": res, "skipped": skipped}))
    key = lambda r: (r["variant"], r["q"])                                                       # noqa: E731
    keys = [(v, q) for v in VARIANTS for q in Q_VALUES]
    rows = {k: [r for r in res if key(r) == k] for k in keys}
    labels = sorted({r["label"] for r in res})
    fac = cfg["ukf"]["process_noise"]["parameter_random_walk_factor"]
    print(f"\nrandom-walk factor: V0 {fac:g}, V1 0 (and parameter covariance 1e-12 x prior), V2 {fac * 0.1:g}, V3 {fac * 10:g}")

    print("\n=== 1. y1 exceedance (both nodes pooled), fraction of post-burn-in samples beyond k SD; median [max] over recordings ===")
    print(f"  {'variant':<8}{'q':>7} " + "".join(f"{'>' + str(k) + ' SD':>17}" for k in THRESHOLDS) + f"{'max SD reached':>18}{'any state >10':>17}")
    for k in keys:
        rr = [r for r in rows[k] if "frac_y1" in r]
        print(f"  {k[0]:<8}{k[1]:>7g} " + "".join(f"{_med_max([r['frac_y1'][str(t)] for r in rr]):>17}" for t in THRESHOLDS)
              + f"{np.median([r['max_y1'] for r in rr]):>9.1f} [{max(r['max_y1'] for r in rr):5.1f}]".rjust(18)
              + f"{_med_max([r['any10'] for r in rr]):>17}")

    print("\n=== 2. parameter excursion from the prior mean, max over each recording, in prior SD; median [max] over recordings ===")
    pk = ss.PARAM_NAMES_FULL
    print(f"  {'variant':<8}{'q':>7} " + "".join(f"{p:>16}" for p in pk))
    for k in keys:
        rr = [r for r in rows[k] if "excursion_prior_sd" in r]
        print(f"  {k[0]:<8}{k[1]:>7g} " + "".join(f"{np.median([r['excursion_prior_sd'][p] for r in rr]):7.2f} [{max(r['excursion_prior_sd'][p] for r in rr):5.1f}]".rjust(16) for p in pk))

    print("\n=== 3. mean NIS per recording [fraction of NIS samples above 10] ===")
    for q in Q_VALUES:
        print(f"  q = {q:g}")
        print(f"    {'recording':<10}" + "".join(f"{v:>16}" for v in VARIANTS))
        for lab in labels:
            cells = []
            for v in VARIANTS:
                r = next(x for x in rows[(v, q)] if x["label"] == lab)
                cells.append((f"{r['mean_nis']:.2f} [{r['frac_nis_gt10']:.3f}]" if "mean_nis" in r else "none").rjust(16))
            print(f"    {lab:<10}" + "".join(cells))
        print(f"    {'median':<10}" + "".join(f"{np.median([x['mean_nis'] for x in rows[(v, q)] if 'mean_nis' in x]):>16.2f}" for v in VARIANTS))

    print("\n=== 4. V0: y1 posterior SD versus the observed excursion, per segment (q = 1e-2) ===")
    print("  post/prior = median posterior SD of y1 / prior neural SD (0.25 mV); max dev in prior-SD and in posterior-SD units;")
    print("  flags counted per (sample, node) after both burn-ins: rule = beyond 10 prior SD, post = beyond 10 posterior SD")
    print(f"    {'rec/seg':<11}{'n':>6}{'post/prior':>11}{'max dev prior-SD':>18}{'max dev post-SD':>17}{'rule flags':>11}{'post flags':>11}{'both':>6}")
    tot = {"rule": 0, "post": 0, "both": 0, "pairs": 0}
    for r in rows[("V0", 1e-2)]:
        for s in r["per_segment"]:
            print(f"    {r['label'] + '/' + str(s['segment']):<11}{s['n']:>6}{s['post_sd_over_prior_sd_median']:>11.2f}{s['max_dev_prior_sd']:>18.1f}"
                  f"{s['max_dev_post_sd']:>17.1f}{s['n_prior10']:>11}{s['n_post10']:>11}{s['n_both']:>6}")
            tot["rule"] += s["n_prior10"]; tot["post"] += s["n_post10"]; tot["both"] += s["n_both"]; tot["pairs"] += s["n_pairs"]
    print(f"  total (sample, node) pairs {tot['pairs']}: rule flags {tot['rule']} ({tot['rule'] / tot['pairs']:.4f}), "
          f"posterior-SD flags {tot['post']} ({tot['post'] / tot['pairs']:.4f}), both {tot['both']}; "
          f"posterior flags that the rule also flags: {tot['both'] / max(tot['post'], 1):.2f}, "
          f"rule flags that the posterior test also flags: {tot['both'] / max(tot['rule'], 1):.2f}")

    print("\n=== 5. spectra, V0 q = 1e-2: observed channel vs model output at the recording-level MEDIAN filtered parameters ===")
    print(f"  {N_REALIZATIONS} stochastic realizations of {SIM_SECONDS:g} s (seeds {SIM_SEED}..), 2-s Hann Welch, 0.5 Hz grid, 1-45 Hz; "
          f"slope = log10 PSD on log10 f excluding {SLOPE_EXCLUDE_HZ[0]:g}-{SLOPE_EXCLUDE_HZ[1]:g} Hz; peak = frequency of the largest log10 residual above that line; alpha = 8-13 Hz share")
    fs = cfg["preprocessing"]["observation_fs_hz"]
    print(f"  {'recording':<9}{'series':<22}{'peak Hz':>8}{'peak resid':>11}{'alpha share':>12}{'slope':>8}    median params")
    for lab in SPECTRUM_RECORDINGS:
        r = next((x for x in rows[("V0", 1e-2)] if x["label"] == lab), None)
        if r is None or "median_params" not in r:
            print(f"  {lab}: not available")
            continue
        f, p_obs = _observed_psd(data[lab][0], fs)
        mp = r["median_params"]
        pstr = " ".join(f"{k}={mp[k]:.3g}" for k in pk)
        m = _spectral_metrics(f, p_obs)
        print(f"  {lab:<9}{'observed':<22}{m['peak_hz']:>8.1f}{m['peak_over_aperiodic_log10']:>11.2f}{m['alpha_share']:>12.3f}{m['slope']:>8.2f}    {pstr}")
        for name, noise in (("model output", False), ("model + R noise", True)):
            try:
                f2, p_mod = _model_psd(cfg, mp, noise)
                m = _spectral_metrics(f2, p_mod)
                print(f"  {'':<9}{name:<22}{m['peak_hz']:>8.1f}{m['peak_over_aperiodic_log10']:>11.2f}{m['alpha_share']:>12.3f}{m['slope']:>8.2f}")
            except Exception as exc:                                  # a failed simulation is itself a finding
                print(f"  {'':<9}{name:<22} simulation failed: {type(exc).__name__}: {exc}")

    print("\n=== NaN/Inf or covariance-not-PD failures (state flag disabled) ===")
    bad = [(r["label"], r["variant"], r["q"], r["failures"]) for r in res if r["failures"]]
    print("  none" if not bad else "\n".join(f"  {a} {v} q={q:g}: {c}" for a, v, q, c in bad))
    print(f"\nrunning time per run (s): median {np.median([r['wall_s'] for r in res]):.0f}, max {max(r['wall_s'] for r in res):.0f}")


if __name__ == "__main__":
    main()
