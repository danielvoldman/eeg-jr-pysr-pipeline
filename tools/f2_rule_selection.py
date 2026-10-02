"""F2 selection test for the DEV-007 divergence rule (PLAN F2; DEV-007, IMP-091). SYNTHETIC data, pilot feature table only.

Pre-declared before any run (IMP-091, config g0.rule_selection, g0.seeds.rule_selection = 96000): twenty series per
configuration, healthy filters (filter A at q_fixed on matched positives, Null A, Null B and a constructed limit-cycle worst
case) and runaway filters (q 0.3 and 1.0, a displaced p prior, data x3, the dropped 19-D filter). Every (series,
configuration) is run THREE times in pass 1: a reference run with both clauses off (it gives the trajectories for the
truth-based label and never stops on the state clause), the legacy rule and the new rule. A run is labelled RUNAWAY from
truth only (non-finite; centred RMS error of a PSP state above e_max times the true SD; time-mean parameter farther than the
multiple of prior SD from the TRUE value), otherwise HEALTHY. A run is FLAGGED if the rule leaves the recording diverged.
Acceptance: sensitivity >= sensitivity_min and false-positive rate <= false_positive_max, each pooled, point estimates, with at
least min_labelled_each runs of each label (else INCONCLUSIVE). On failure the tool reports and nothing is re-picked.

    .venv\\Scripts\\python.exe tools\\f2_rule_selection.py            # the real test (write-once results/pilot/rule_selection.json)
    .venv\\Scripts\\python.exe tools\\f2_rule_selection.py --smoke    # 2 series per set on a THROWAWAY block, nothing written

The pilot real recordings are not part of this test. No test subject is ever read.
"""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

PSP = (0, 1, 2, 6, 7, 8)                     # y0, y1, y2 of both nodes among the 12 neural states
PARAM_KEYS = ("p1", "p2", "log_rho1", "log_rho2", "g12", "g21", "m")
SMOKE_BLOCK = 99500                           # throwaway block of --smoke (in memory only)


# ---------------------------------------------------------------- pure helpers (tested)

def centred_state_ratios(x_filt, x_true):
    """Per column: RMS of ((x_f - mean x_f) - (x_t - mean x_t)) over the SD of x_t. A constant predictor scores 1."""
    xf, xt = np.asarray(x_filt, dtype=np.float64), np.asarray(x_true, dtype=np.float64)
    err = (xf - xf.mean(axis=0)) - (xt - xt.mean(axis=0))
    return np.sqrt(np.mean(err ** 2, axis=0)) / xt.std(axis=0)


def runaway_label(sel, finite, ratios, par_dev):
    """(label, reasons) with label 'runaway' or 'healthy' from truth only. finite: no non-finite state or covariance;
    ratios: centred RMS error ratios of the six PSP states; par_dev: time-mean parameter distance from the TRUE value in
    prior SD, a dict by parameter name. The candidate rules never enter."""
    reasons = []
    if not finite:
        reasons.append("non_finite")
    if ratios is not None and float(np.max(ratios)) > sel["e_max"]:
        reasons.append("state_error")
    if par_dev is not None and max(par_dev.values()) > sel["parameter_label_sd_multiple"]:
        reasons.append("parameter_error")
    return ("runaway" if reasons else "healthy"), reasons


def clopper_pearson(k, n, level=0.95):
    from scipy.stats import beta
    if n == 0:
        return None
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return [lo, hi]


def summarize(runs, sel):
    """Pooled and per-configuration counts. runs: dicts with config, label, flagged_new, flagged_legacy."""
    def block(subset, key):
        run_ = [r for r in subset if r["label"] == "runaway"]
        heal = [r for r in subset if r["label"] == "healthy"]
        k_r, k_h = sum(bool(r[key]) for r in run_), sum(bool(r[key]) for r in heal)
        return {"n_runaway": len(run_), "n_healthy": len(heal), "flagged_runaway": k_r, "flagged_healthy": k_h,
                "sensitivity": (k_r / len(run_)) if run_ else None, "sensitivity_ci95": clopper_pearson(k_r, len(run_)),
                "false_positive_rate": (k_h / len(heal)) if heal else None, "false_positive_ci95": clopper_pearson(k_h, len(heal))}
    labelled = [r for r in runs if r["label"] in ("runaway", "healthy")]
    out = {"pooled": {"new": block(labelled, "flagged_new"), "legacy": block(labelled, "flagged_legacy")},
           "per_config": {}, "per_reason": {}, "n_runs": len(runs), "n_unlabelled": len(runs) - len(labelled)}
    for c in sorted({r["config"] for r in labelled}):
        sub = [r for r in labelled if r["config"] == c]
        out["per_config"][c] = {"new": block(sub, "flagged_new"), "legacy": block(sub, "flagged_legacy")}
    for reason in ("non_finite", "state_error", "parameter_error"):
        sub = [r for r in labelled if r["label"] == "runaway" and reason in r["reasons"]]
        out["per_reason"][reason] = {"n": len(sub), "flagged_new": sum(bool(r["flagged_new"]) for r in sub),
                                     "flagged_legacy": sum(bool(r["flagged_legacy"]) for r in sub)}
    return out


def verdict(summary, sel):
    """PASS, FAIL or INCONCLUSIVE for the NEW rule (the legacy numbers carry no weight)."""
    p = summary["pooled"]["new"]
    if p["n_runaway"] < sel["min_labelled_each"] or p["n_healthy"] < sel["min_labelled_each"]:
        return "INCONCLUSIVE"
    ok = p["sensitivity"] >= sel["sensitivity_min"] and p["false_positive_rate"] <= sel["false_positive_max"]
    return "PASS" if ok else "FAIL"


def check_seed_block(cfg, block=None):
    """The selection block must not be any G0 block in any fresh-seed round (up to round 20), nor the burnt 99000."""
    sd = cfg["g0"]["seeds"]
    block = sd["rule_selection"] if block is None else block
    taken = {sd[k] + r * sd["fresh_round_stride"] for k in ("pilot", "full", "tuning", "preprocessing_gate", "grid") for r in range(21)}
    taken.add(99000)
    if block in taken:
        raise ValueError(f"seed block {block} is a G0 block or the burnt exploration block")
    return block


# ---------------------------------------------------------------- the worker

def _capture_run(segments, starts, cfg, q, filter_name, spec, passes, ukf, ukf_ext):
    """pass 1 forward-only with the filter results of every segment captured (for the label)."""
    captured = []
    orig_rf, orig_mk = ukf.run_filter, ukf_ext.make_runners

    def rf(*a, **k):
        r = orig_rf(*a, **k)
        captured.append(r)
        return r

    def mk(spec_, *a, **k):
        f, sm = orig_mk(spec_, *a, **k)

        def g(*aa, **kk):
            r = f(*aa, **kk)
            captured.append(r)
            return r
        return g, sm

    ukf.run_filter, ukf_ext.make_runners = rf, mk
    try:
        res = passes.run_pass1(segments, starts, cfg, q, forward_only=True, filter_name=filter_name, spec=spec)
    finally:
        ukf.run_filter, ukf_ext.make_runners = orig_rf, orig_mk
    return res, captured


def _task(payload):
    """One (series, configuration): reference, legacy and new pass-1 runs and the truth-based label. Top level (spawn)."""
    from threadpoolctl import threadpool_limits
    from legacy_rule import legacy
    from src import passes, ukf, ukf_ext
    from src import state_space as ss
    from src import synthetic_gate as sg
    cfg_nom, sel, spec_cfg, series, name, sset = payload
    with threadpool_limits(limits=1):
        cfg_run = sg.g0_cfg(spec_cfg["cfg"])
        filter_name, q = spec_cfg["filter"], spec_cfg["q"]
        segments = [np.asarray(s, dtype=np.float64) for s in series.segments]
        if spec_cfg.get("amplitude_factor"):
            mu = cfg_run["rescaling"]["mu_ref"]
            segments = [mu + spec_cfg["amplitude_factor"] * (s - mu) for s in segments]
        spec = passes.make_spec(cfg_run, filter_name, segments)
        t0 = time.perf_counter()
        ref_cfg = copy.deepcopy(cfg_run)
        ref_cfg["ukf"]["divergence"]["state_sd_multiple"] = float("inf")
        ref_cfg["ukf"]["divergence"]["parameter_sd_multiple"] = None
        ref, cap = _capture_run(segments, series.starts, ref_cfg, q, filter_name, spec, passes, ukf, ukf_ext)
        leg = passes.run_pass1(segments, series.starts, legacy(cfg_run), q, forward_only=True, filter_name=filter_name, spec=spec)
        new = passes.run_pass1(segments, series.starts, cfg_run, q, forward_only=True, filter_name=filter_name, spec=spec)
        rec = {"config": name, "set": sset, "index": int(series.index), "arm": series.arm, "level_index": series.level_index,
               "p_true": float(series.operating_point["p"]), "flagged_new": bool(new.recording_diverged),
               "flagged_legacy": bool(leg.recording_diverged), "new_reasons": sorted({str(s.reason) for s in new.segments if s.diverged}),
               "legacy_reasons": sorted({str(s.reason) for s in leg.segments if s.diverged}),
               "new_steps": [s.step for s in new.segments if s.diverged][:3], "legacy_steps": [s.step for s in leg.segments if s.diverged][:3]}
        layout = ref.layout
        finite = not any(s.diverged and str(s.reason).split(":")[0] in ("nan_inf", "covariance_not_pd", "linalg_error") for s in ref.segments)
        xf, xt, xs = [], [], []
        for seg, r, start in zip(ref.segments, cap, series.starts):
            n = int(r.n_done)
            if n == 0 or seg.diverged or seg.nis_keep is None:
                continue
            keep = np.asarray(seg.nis_keep)
            idx = np.flatnonzero(keep) + int(start)
            if idx.max() >= series.states.shape[0]:
                raise RuntimeError("truth rows do not cover the segment")
            X = np.asarray(r.x[:n])[keep]
            xs.append(X)
            xf.append(X[:, :ss.N_NEURAL])
            xt.append(series.states[idx].reshape(len(idx), ss.N_NEURAL))
        ratios = par_dev = None
        if xf:
            Xf, Xt = np.concatenate(xf), np.concatenate(xt)
            ratios = centred_state_ratios(Xf[:, PSP], Xt[:, PSP])
            par = layout.params(np.concatenate(xs))
            sd = dict(ss.parameter_prior_sd(cfg_nom))
            sd["m"] = cfg_nom["priors"]["m_sd"]
            g = float(series.gains[0])
            true = {"p1": float(series.operating_point["p"]), "p2": float(series.operating_point["p"]),
                    "log_rho1": ss.log_rho_prior_mean(cfg_nom), "log_rho2": ss.log_rho_prior_mean(cfg_nom),
                    "g12": g, "g21": g, "m": float(series.m)}
            par_dev = {k: float(abs(np.mean(par[k]) - true[k]) / sd[k]) for k in PARAM_KEYS}
        label, reasons = runaway_label(sel, finite, ratios, par_dev) if (xf or not finite) else ("unlabelled", [])
        rec.update({"label": label, "reasons": reasons, "finite": finite,
                    "state_ratio_max": None if ratios is None else float(np.max(ratios)),
                    "parameter_dev_max": None if par_dev is None else max(par_dev.values()),
                    "parameter_dev_argmax": None if par_dev is None else max(par_dev, key=par_dev.get),
                    "wall_s": time.perf_counter() - t0})
    return rec


# ---------------------------------------------------------------- orchestration

def configurations(cfg, sel):
    """(name, set, spec dict) in the order of IMP-091: healthy set first, then the runaway set."""
    q_fixed = cfg["ukf"]["process_noise"]["q_fixed"]
    base = {"cfg": cfg, "filter": "A", "q": q_fixed}
    out = [("healthy_matched_positive", "H1", dict(base)), ("healthy_null_A", "H2", dict(base)),
           ("healthy_null_B", "H3", dict(base)), ("healthy_limit_cycle_worst_case", "H4", dict(base))]
    for q in sel["runaway_q"]:
        out.append((f"runaway_q_{q:g}", "H1", dict(base, q=float(q))))
    shifted = copy.deepcopy(cfg)
    shifted["priors"]["p_mean"] = cfg["priors"]["p_mean"] + sel["runaway_prior_p_displacement_sd"] * cfg["priors"]["p_sd"]
    out.append(("runaway_prior_p_displaced", "H1", dict(base, cfg=shifted)))
    out.append(("runaway_data_amplitude", "H1", dict(base, amplitude_factor=float(sel["runaway_data_amplitude_factor"]))))
    out.append(("runaway_filter_19D", "H1", dict(base, filter="19D", q=float(sel["runaway_filter_19d_q"]))))
    return out


def make_series(cfg, sel, n, block_cfg, grid, table):
    from src import synthetic_gate as sg
    gc = sg.g0_cfg(block_cfg)
    ps = np.linspace(*cfg["g0"]["regime"]["p_range"], 12)
    wc = sel["healthy_worst_case"]
    sets = {"H1": [sg.generate_series(gc, "positive", i, grid, table, "rule_selection", keep_states=True) for i in range(n)],
            "H2": [sg.generate_series(gc, "null_A", i, grid, table, "rule_selection", keep_states=True) for i in range(n)],
            "H3": [sg.generate_series(gc, "null_B", i, grid, table, "rule_selection", keep_states=True) for i in range(n)]}
    h4 = []
    for i in range(n):
        p = float(ps[wc["p_grid_indices"][i % len(wc["p_grid_indices"])]])
        op = {"index": -1, "p": p, "input_sd_factor": float(wc["input_sd_factor"]), "noise_share": float(wc["noise_share"]),
              "regime": sg.LIMIT_CYCLE, "recording": "constructed", "target": [], "distance": 0.0}
        h4.append(sg.generate_series(gc, "positive", 100 + i, grid, table, "rule_selection", level_index=int(wc["level_index"]),
                                     operating_point=op, keep_states=True))
    sets["H4"] = h4
    return sets


def run(smoke=False, n_jobs=None):
    from joblib import Parallel, delayed
    from src import synthetic_gate as sg
    from src import tuning
    from src.config import load_config
    cfg = load_config()
    sel = cfg["g0"]["rule_selection"]
    block_cfg = copy.deepcopy(cfg)
    if smoke:
        block_cfg["g0"]["seeds"]["rule_selection"] = SMOKE_BLOCK
    else:
        check_seed_block(cfg)
    n = 2 if smoke else int(sel["n_per_config"])
    out_path = ROOT / cfg["paths"]["pilot_results_dir"] / sel["output_file"]
    if not smoke and out_path.exists():
        raise SystemExit(f"{out_path} exists: the selection test is write-once and is not retried on the same seeds")
    t0 = time.perf_counter()
    table, grid = sg.pilot_inputs(cfg, ROOT)
    sets = make_series(cfg, sel, n, block_cfg, grid, table)
    for lst in sets.values():                                  # the true states are needed for every arm
        for s in lst:
            if s.states is None:
                raise RuntimeError("series without truth states; generate_series(keep_states=True) is required")
    gen_s = time.perf_counter() - t0
    sg.emit(f"== F2 selection test{' (SMOKE, throwaway block, nothing written)' if smoke else ''}: {sum(len(v) for v in sets.values())} series "
            f"in {gen_s / 60:.1f} min; block {block_cfg['g0']['seeds']['rule_selection']} ==")
    jobs = []
    for name, sset, spec_cfg in configurations(cfg, sel):
        for ser in sets[sset]:
            jobs.append((cfg, sel, spec_cfg, ser, name, sset))
    workers = sg.pool_size(cfg, n_jobs)
    t1 = time.perf_counter()
    runs = Parallel(n_jobs=workers, backend=cfg["compute"]["joblib_backend"])(delayed(_task)(j) for j in jobs)
    eval_s = time.perf_counter() - t1
    summ = summarize(runs, sel)
    res = verdict(summ, sel)
    head, dirty = tuning._git_state(ROOT)
    doc = {"stage": "F2 rule selection (DEV-007, IMP-091); synthetic data, pilot feature table only", "smoke": bool(smoke),
           "verdict": res, "summary": summ, "thresholds": {k: sel[k] for k in ("e_max", "parameter_label_sd_multiple", "sensitivity_min",
                                                                             "false_positive_max", "n_per_config", "min_labelled_each")},
           "rule": {k: cfg["ukf"]["divergence"][k] for k in ("state_sd_multiple", "state_dwell_s", "parameter_sd_multiple", "parameter_dwell_s")},
           "seed_block": block_cfg["g0"]["seeds"]["rule_selection"], "runs": runs, "workers": workers,
           "timing_s": {"generation": gen_s, "evaluation": eval_s},
           "provenance": {"git_commit": head, "git_dirty": dirty,
                          "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()}}
    for line in report_lines(doc):
        sg.emit(line)
    if not smoke:
        sg.write_pilot_json(cfg, ROOT, doc, out_path)
        sg.emit(f"written: {out_path}")
    return doc


def report_lines(doc):
    s = doc["summary"]
    f = lambda v: "-" if v is None else f"{v:.3f}"          # noqa: E731
    L = [f"#### verdict (NEW rule): {doc['verdict']}   thresholds {doc['thresholds']}   rule {doc['rule']}"]
    for key in ("new", "legacy"):
        p = s["pooled"][key]
        L.append(f"pooled {key:<7} labelled runaway {p['n_runaway']:3d} (flagged {p['flagged_runaway']:3d}, sensitivity {f(p['sensitivity'])} "
                 f"CI {p['sensitivity_ci95']})   healthy {p['n_healthy']:3d} (flagged {p['flagged_healthy']:3d}, FPR {f(p['false_positive_rate'])} CI {p['false_positive_ci95']})")
    L.append(f"{'configuration':<34}{'runaway':>8}{'new flag':>9}{'old flag':>9}{'healthy':>9}{'new flag':>9}{'old flag':>9}")
    for c, v in s["per_config"].items():
        n_, o_ = v["new"], v["legacy"]
        L.append(f"{c:<34}{n_['n_runaway']:>8}{n_['flagged_runaway']:>9}{o_['flagged_runaway']:>9}{n_['n_healthy']:>9}{n_['flagged_healthy']:>9}{o_['flagged_healthy']:>9}")
    L.append("per label reason (a run can carry several): " + "; ".join(f"{k} n={v['n']} new {v['flagged_new']} legacy {v['flagged_legacy']}" for k, v in s["per_reason"].items()))
    L.append(f"unlabelled runs: {s['n_unlabelled']}   timing: generation {doc['timing_s']['generation'] / 60:.1f} min, evaluation {doc['timing_s']['evaluation'] / 60:.1f} min on {doc['workers']} workers")
    return L


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    a = ap.parse_args(argv)
    doc = run(smoke=a.smoke, n_jobs=a.n_jobs)
    return 0 if doc["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
