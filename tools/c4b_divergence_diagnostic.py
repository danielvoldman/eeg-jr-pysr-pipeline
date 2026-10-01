"""C4b: read-only diagnostic of the §7.5 state-beyond-10-SD divergence rule (PLAN.md C4b).

Pilot subjects only (ses-t1). Changes no rule, threshold or config value and writes nothing under outputs/.
The state-SD flag is disabled DIAGNOSTICALLY by replacing ukf._first_divergence inside this script's own
processes with a wrapper that (a) records the per-step deviation of the 12 neural states from the reference
in SD units and the full filtered state, and (b) calls the original with check_state=False, so NaN/Inf and
covariance-PD checks stay on. Nothing in src/ is edited. Pass-1 style forward filtering (passes.run_pass1,
forward_only) at q = 1e-4, 1e-3, 1e-2, 1e-1. Synthetic comparison from tests/sim_data at the model's own
amplitude. Raw results go to a JSON path given on the command line (keep it outside the repo).

    python tools/c4b_divergence_diagnostic.py <raw_results.json> [n_jobs]
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

Q_VALUES = (1e-4, 1e-3, 1e-2, 1e-1)
THRESHOLDS = (3, 5, 10, 20)
SYNTH = [  # (label, seed, g12, g21, p)
    ("g0 p=220,220", 101, 0.0, 0.0, None),
    ("g0 p=180,250", 102, 0.0, 0.0, (180.0, 250.0)),
    ("g0 p=260,150", 103, 0.0, 0.0, (260.0, 150.0)),
    ("g=5,5 p=240,200", 104, 5.0, 5.0, (240.0, 200.0)),
    ("g12=12 p=220,260", 105, 12.0, 0.0, (220.0, 260.0)),
    ("g=8,8 p=220,220", 106, 8.0, 8.0, None),
]


def _worker(job):
    """One (recording, q). Top level for the spawn start method."""
    from threadpoolctl import threadpool_limits
    from src import passes, ukf
    from src import state_space as ss

    label, segments, starts, q, cfg = job
    fs = cfg["preprocessing"]["observation_fs_hz"]
    n_exempt = int(round(cfg["ukf"]["divergence"]["startup_exempt_s"] * fs))
    calls = []
    orig_fd, orig_rf = ukf._first_divergence, ukf.run_filter

    def fd(x_post, P_post, cfg_, center, sd, min_eig, check_state=True):
        if np.all(np.isfinite(x_post)):
            calls.append((np.abs(x_post[:ss.N_NEURAL] - center) / sd, np.array(x_post)))
        else:
            calls.append((np.full(ss.N_NEURAL, np.nan), np.array(x_post)))
        return orig_fd(x_post, P_post, cfg_, center, sd, min_eig, check_state=False)

    segrecs = []

    def rf(*a, **k):
        calls.clear()
        res = orig_rf(*a, **k)
        segrecs.append({"dev": np.array([c[0] for c in calls]), "x": np.array([c[1] for c in calls]),
                        "nis": res.nis.copy(), "n_done": res.n_done, "diverged": res.diverged,
                        "reason": res.divergence_reason})
        return res

    ukf._first_divergence, ukf.run_filter = fd, rf
    try:
        with threadpool_limits(limits=1):
            t0 = time.perf_counter()
            res = passes.run_pass1(segments, starts, cfg, q, forward_only=True, filter_name="19D")
            wall = time.perf_counter() - t0
    finally:
        ukf._first_divergence, ukf.run_filter = orig_fd, orig_rf

    layout = res.layout
    keep_dev, keep_nis = [], []
    first10 = None
    xs = []
    failures = []
    for seg, rec in zip(res.segments, segrecs):
        n = rec["n_done"] if not rec["diverged"] else len(rec["dev"])
        if rec["diverged"]:
            failures.append(f"seg {seg.index}: {rec['reason']} at step {seg.step}")
        if seg.nis_keep is not None and not seg.diverged:
            keep_dev.append(rec["dev"][seg.nis_keep])
            keep_nis.append(rec["nis"][seg.nis_keep])
        dev = rec["dev"][:n]
        if dev.shape[0] > n_exempt:
            over = np.argwhere(dev[n_exempt:] > 10.0)
            if over.size:
                t, s = int(over[0][0]) + n_exempt, int(over[0][1])
                cand = ((seg.start + t) / fs, s, seg.index)
                if first10 is None or cand[0] < first10[0]:
                    first10 = cand
        xs.append(rec["x"][:n])
    out = {"label": label, "q": q, "wall_s": wall, "failures": failures, "recording_diverged": res.recording_diverged,
           "first10": first10, "n_segments": len(res.segments)}
    if keep_dev:
        D, N = np.concatenate(keep_dev), np.concatenate(keep_nis)
        out["n_kept"] = int(D.shape[0])
        out["frac"] = {str(k): [float(np.mean(D[:, s] > k)) for s in range(ss.N_NEURAL)] for k in THRESHOLDS}
        out["any10"] = float(np.mean(np.any(D > 10.0, axis=1)))
        out["max_dev"] = [float(v) for v in D.max(axis=0)]
        out["mean_nis"], out["frac_nis_gt10"] = float(np.mean(N)), float(np.mean(N > 10.0))
    X = np.concatenate([x for x in xs if len(x)]) if any(len(x) for x in xs) else None
    if X is not None:
        par = layout.params(X)
        out["params"] = {k: [float(np.min(par[k])), float(np.max(par[k])), float(par[k][-1])] for k in ss.PARAM_NAMES_FULL}
    return out


def _real_jobs(cfg):
    import main as main_mod
    from src import preprocess as pp
    from src import tuning
    pilot = pp.load_pilot_ids(cfg, ROOT)
    loader = tuning.make_real_loader(cfg, ROOT, pilot, allow_all=False)   # the development guard stays on
    jobs, skipped = [], []
    for sid in sorted(pilot):
        r = loader(sid)
        if r["reason"] is not None:
            skipped.append((sid, r["reason"]))
            continue
        jobs += [(sid, r["segments"], r["starts"], q, cfg) for q in Q_VALUES]
    return jobs, skipped


def _synth_jobs(cfg):
    from tests import sim_data
    jobs, sds = [], {}
    for label, seed, g12, g21, p in SYNTH:
        r = sim_data.make_recording(cfg, [30.0, 30.0], seed, g12=g12, g21=g21, p=p)
        z = np.concatenate(r["segments"], axis=1)
        sds[label] = (float(z.mean()), float(z.std(ddof=1)))
        jobs += [(label, r["segments"], r["starts"], q, cfg) for q in Q_VALUES]
    return jobs, sds


def _fmt_table(results, title, labels):
    from src import state_space as ss
    print(f"\n{title}")
    for q in Q_VALUES:
        rows = [r for r in results if r["q"] == q and "frac" in r]
        if not rows:
            print(f"  q={q:g}: no samples")
            continue
        print(f"  q = {q:g}: fraction of post-burn-in samples beyond k x sqrt(neural_variance) from the reference; "
              f"median [max] over {len(rows)} recordings")
        print(f"    {'state':>7} " + " ".join(f"{'>' + str(k) + ' SD':>17}" for k in THRESHOLDS))
        for s in range(ss.N_NEURAL):
            cells = []
            for k in THRESHOLDS:
                v = np.array([r["frac"][str(k)][s] for r in rows])
                cells.append(f"{np.median(v):6.3f} [{v.max():6.3f}]".rjust(17))
            print(f"    {labels[s]:>7} " + " ".join(cells))
        a = np.array([r["any10"] for r in rows])
        print(f"    any state beyond 10 SD (the rule's trigger): median {np.median(a):.3f}, max {a.max():.3f}, "
              f"recordings with any: {int(np.sum(a > 0))} of {len(rows)}")


def main():
    from src.config import load_config
    from src import state_space as ss
    out_path = Path(sys.argv[1])
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cfg = load_config()
    from joblib import Parallel, delayed
    real_jobs, skipped = _real_jobs(cfg)
    synth_jobs, sds = _synth_jobs(cfg)
    print(f"real: {len(real_jobs) // len(Q_VALUES)} pilot ses-t1 recordings, skipped {skipped}; synthetic: {len(SYNTH)}", flush=True)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_worker)(j) for j in real_jobs + synth_jobs)
    print(f"wall {time.perf_counter() - t0:.0f} s", flush=True)
    real, synth = res[:len(real_jobs)], res[len(real_jobs):]
    out_path.write_text(json.dumps({"real": real, "synthetic": synth, "synthetic_mean_sd": sds, "skipped": skipped}))
    names = [f"n{n}.y{i}" for n in (1, 2) for i in range(6)]
    prior_layout = ss.make_layout(cfg)
    pm = prior_layout.params(ss.prior_mean(prior_layout, cfg)[None])
    psd = dict(ss.parameter_prior_sd(cfg)); psd["m"] = cfg["priors"]["m_sd"] if "m_sd" in cfg["priors"] else 0.15

    print("\n=== 1. per-state exceedance ===")
    _fmt_table(real, "REAL pilot data (state-SD flag disabled)", names)
    _fmt_table(synth, "SYNTHETIC (tests/sim_data, 6 series of 60 s, model amplitude)", names)
    print("\nsynthetic observation mean and SD (mV) per series (real data is rescaled to mu_ref = "
          f"{cfg['rescaling']['mu_ref']:.3f}, sigma_ref = {cfg['rescaling']['sigma_ref']:.3f}):")
    for k, (m, s) in sds.items():
        print(f"  {k:<20} mean {m:.3f}  SD {s:.3f}")
    print("\nfirst state beyond 10 SD (after the 0.5 s start-up), real: recording, q -> time s, state, segment")
    for r in real:
        if r["first10"]:
            t, s, k = r["first10"]
            print(f"  {r['label']} q={r['q']:g}: {t:8.2f} s  {names[s]}  seg {k}")
        else:
            print(f"  {r['label']} q={r['q']:g}: never")
    print("\nfirst state beyond 10 SD, synthetic:")
    for r in synth:
        f = r["first10"]
        print(f"  {r['label']:<18} q={r['q']:g}: " + ("never" if not f else f"{f[0]:8.2f} s  {names[f[1]]}  seg {f[2]}"))
    print("\nmax deviation (SD units) per state, real, q=1e-2, max over recordings:")
    rows = [r for r in real if r["q"] == 1e-2 and "max_dev" in r]
    print("  " + " ".join(f"{names[s]}:{max(r['max_dev'][s] for r in rows):.1f}" for s in range(12)))
    rows = [r for r in synth if r["q"] == 1e-2 and "max_dev" in r]
    print("max deviation, synthetic, q=1e-2, max over series:")
    print("  " + " ".join(f"{names[s]}:{max(r['max_dev'][s] for r in rows):.1f}" for s in range(12)))

    print("\n=== 2. filtered parameter trajectories: min / max / final (prior mean, prior SD in header) ===")
    keys = ss.PARAM_NAMES_FULL
    print("  prior mean: " + "  ".join(f"{k} {float(pm[k][0]):.3g}" for k in keys))
    print("  prior SD:   " + "  ".join(f"{k} {psd[k]:.3g}" for k in keys))
    for q in (1e-2,):
        print(f"  real, q = {q:g}")
        print(f"    {'recording':<10} " + " ".join(f"{k:>24}" for k in keys))
        for r in real:
            if r["q"] == q and "params" in r:
                print(f"    {r['label']:<10} " + " ".join(
                    f"{r['params'][k][0]:7.3g}/{r['params'][k][1]:7.3g}/{r['params'][k][2]:7.3g}".rjust(24) for k in keys))
    print("  worst excursion from the prior mean over recordings, in prior SD (max over min/max/final), per q")
    print(f"    {'q':>8} " + " ".join(f"{k:>9}" for k in keys))
    for q in Q_VALUES:
        rows = [r for r in real if r["q"] == q and "params" in r]
        cells = []
        for k in keys:
            ex = max(max(abs(v - float(pm[k][0])) for v in r["params"][k]) / psd[k] for r in rows)
            cells.append(f"{ex:9.2f}")
        print(f"    {q:>8g} " + " ".join(cells))
    print("  synthetic, same measure")
    for q in Q_VALUES:
        rows = [r for r in synth if r["q"] == q and "params" in r]
        cells = []
        for k in keys:
            ex = max(max(abs(v - float(pm[k][0])) for v in r["params"][k]) / psd[k] for r in rows)
            cells.append(f"{ex:9.2f}")
        print(f"    {q:>8g} " + " ".join(cells))

    print("\n=== 3. mean NIS (flag disabled) per recording per q; [fraction of NIS samples above 10] ===")
    print(f"    {'recording':<18} " + " ".join(f"{'q=' + format(q, 'g'):>16}" for q in Q_VALUES))
    for label in sorted({r["label"] for r in real}):
        cells = []
        for q in Q_VALUES:
            r = next(x for x in real if x["label"] == label and x["q"] == q)
            cells.append((f"{r['mean_nis']:.2f} [{r['frac_nis_gt10']:.3f}]" if "mean_nis" in r else "none").rjust(16))
        print(f"    {label:<18} " + " ".join(cells))
    print("  synthetic")
    for label in [s[0] for s in SYNTH]:
        cells = []
        for q in Q_VALUES:
            r = next(x for x in synth if x["label"] == label and x["q"] == q)
            cells.append((f"{r['mean_nis']:.2f} [{r['frac_nis_gt10']:.3f}]" if "mean_nis" in r else "none").rjust(16))
        print(f"    {label:<18} " + " ".join(cells))

    print("\n=== 4. NaN/Inf or covariance-not-PD failures with the state flag disabled ===")
    bad = [(r["label"], r["q"], r["failures"]) for r in real + synth if r["failures"]]
    print("  none" if not bad else "\n".join(f"  {a} q={b:g}: {c}" for a, b, c in bad))
    print(f"\nrunning time per job (s): real median {np.median([r['wall_s'] for r in real]):.0f}, "
          f"max {max(r['wall_s'] for r in real):.0f}")


if __name__ == "__main__":
    main()
