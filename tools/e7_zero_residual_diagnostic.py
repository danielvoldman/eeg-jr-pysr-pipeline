"""E7: DIAGNOSTIC ONLY, synthetic, pilot mode (PLAN.md E7; §9.1, §17, DEV-005, IMP-070, IMP-071). Does the planted product
residual explain the remaining gain error of E6?

Changes no src rule, threshold or config-file value and writes nothing under outputs/ (only results/pilot/e7_*.json). The
residual is switched off INSIDE this script (a deep-copied config with g0.planted_residual_rms_fraction_of_base = 0, so the
generator simulates once without the term); the operating-point override is the one of tools/e6_operating_point_diagnostic.py.
No PySR. Not a G0 verdict.

Filters: A and B ONLY. 19D is dropped from this diagnostic (it failed everywhere in E5 and E6 and the question is about the
candidates that survive the nulls at the easy operating point).

Arms (same seeds, levels and cells as E6; 20 positive = 5 per level, 20 Null A, 20 Null B, the 20-series tuning set, which is
generated like the positives and therefore also without the term):
  (i)  the E6 fixed C7-style operating point (real chain), residual c = 0      reference: E6 arm (c), residual at 50%
  (ii) the E5 matched operating points (real chain), residual c = 0            reference: E5 arm (a), residual at 50%
The null series have no residual in any arm, so their DATA equal the E6 / E5 nulls; they are scored again because q is tuned
on the (now residual-free) tuning set. Q/R rule and diagnostic fallback as in E5 / E6 (IMP-070), same criteria and table;
standard rule ON and flag disabled (DIAGNOSTIC ONLY) side by side. For reference the realised planted / base RMS ratio of the
E6 positives (residual at 50%, fixed point) and of the E5 positives (matched points) is reported per level.

    python tools/e7_zero_residual_diagnostic.py [--n-jobs 4] [--report-only]
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import e6_operating_point_diagnostic as e6  # noqa: E402
from src import synthetic_gate as sg  # noqa: E402
from src.config import load_config  # noqa: E402

OPTIONS = ("A", "B")                      # 19D is dropped from this diagnostic
SETS = e6.SETS
ARM_TEXT = {"i": "(i) E6 fixed C7-style operating point, real chain, residual c = 0",
            "ii": "(ii) E5 matched operating points, real chain, residual c = 0"}


def zero_residual_cfg(cfg):
    """Config copy with the planted residual switched off (c = 0); the config file and the original dict are untouched."""
    c = copy.deepcopy(cfg)
    c["g0"]["planted_residual_rms_fraction_of_base"] = 0.0
    return c


def build_sets_fixed(cfg, grid, table, regime, features):
    """Arm (i): the E6 fixed operating point with the real chain and no residual."""
    return e6.build_sets(zero_residual_cfg(cfg), grid, table, False, regime, features)


def build_sets_matched(cfg, grid, table):
    """Arm (ii): the E5 matched operating points with the real chain and no residual (same streams as E5)."""
    c = sg.g0_cfg(zero_residual_cfg(cfg))
    n = c["g0"]
    return {"positive": [sg.generate_series(c, "positive", i, grid, table, "pilot") for i in range(n["n_pilot_positive"])],
            "null_A": [sg.generate_series(c, "null_A", i, grid, table, "pilot") for i in range(n["n_pilot_null_A"])],
            "null_B": [sg.generate_series(c, "null_B", i, grid, table, "pilot") for i in range(n["n_pilot_null_B"])],
            "tuning": sg.tuning_set(c, grid, table)}


def ratio_summary(series):
    """Realised planted / base RMS ratio per coupling level (both nodes pooled): n values, median, min, max."""
    out = {}
    for lv in sorted({s.level_index for s in series}):
        vals = [v for s in series if s.level_index == lv for v in s.meta["rms_ratio"]]
        out[str(lv)] = {"n": len(vals), "median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    return out


def reference_ratios(cfg, grid, table, regime, features):
    """The realised ratios of the residual-carrying positives: E6 (fixed point, real chain) and E5 (matched points). The
    positives are regenerated from their seeds (about a minute); nothing is filtered."""
    c, g = e6.fixed_cfg_and_grid(sg.g0_cfg(cfg), grid)
    n = c["g0"]["n_pilot_positive"]
    with e6.override(c, features, table.scale, regime, False):
        fixed = [sg.generate_series(c, "positive", i, g, table, "pilot", with_truth=False) for i in range(n)]
    c5 = sg.g0_cfg(cfg)
    matched = [sg.generate_series(c5, "positive", i, grid, table, "pilot", with_truth=False) for i in range(n)]
    return {"e6_fixed_point": ratio_summary(fixed), "e5_matched": ratio_summary(matched)}


def compare_lines(ref, zero, delta):
    """Compact residual-on (c > 0, reference) vs residual-off (c = 0) rows per option and level, rule ON and flag disabled."""
    reports = {"a": ref, "b": zero, "c": {"options": {n: {"levels": {}, "nulls": {}} for n in zero["options"]}}}
    rows = e6.comparison_rows(reports)
    lines = e6.format_rows(rows)
    lines[0] = lines[0].replace("(a) drop err off", "c>0 drop err off").replace("(b) drop err off", "c=0 drop err off").replace(
        "(c) drop err off", "").replace("(a | b | c)", "(c>0 | c=0)")
    return [ln.replace(" | -", "").rstrip() for ln in lines], rows


def print_report(doc, emit=None):
    emit = emit or sg.emit
    emit("#### E7 (diagnostic only): filters A and B; 19D is dropped from this diagnostic")
    for key in ("i", "ii"):
        emit(f"#### {ARM_TEXT[key]}")
        a = doc["arms"][key]
        for line in sg.format_comparison({"delta": a["delta"], "options": a["options"]}):
            emit(line)
        emit(f"#### side by side, {key}: dropped by the standard rule / median smoothed gain error rule ON / same with the flag disabled "
             f"(DIAGNOSTIC ONLY); c>0 = the E6 / E5 reference with the residual, c=0 = this run")
        for line in doc["compare"][key]:
            emit(line)
    emit("#### realised planted / base RMS ratio of the residual-carrying positives, per level (both nodes pooled):")
    for name, tab in doc["rms_ratio"].items():
        emit(f"  {name}: " + "; ".join(f"L{int(lv) + 1} median {v['median']:.2f} (min {v['min']:.2f}, max {v['max']:.2f}, n {v['n']})"
                                       for lv, v in tab.items()))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--report-only", action="store_true", help="print the tables of the written results/pilot/e7_zero_residual.json")
    args = ap.parse_args(argv)
    cfg = load_config()
    emit = sg.emit
    path = ROOT / cfg["paths"]["pilot_results_dir"] / "e7_zero_residual.json"
    if args.report_only:
        print_report(e6.unjson(json.loads(path.read_text(encoding="utf-8"))))
        return 0
    n_workers = sg.pool_size(cfg, args.n_jobs)
    table, grid = sg.pilot_inputs(cfg, ROOT)
    fp = e6.fixed_point(cfg)
    regime = sg.classify_regime(cfg, fp["p"])
    features = e6.fixed_features(cfg, grid, len(grid.features))
    e6_doc = e6.unjson(json.loads((ROOT / cfg["paths"]["pilot_results_dir"] / "e6_operating_point.json").read_text(encoding="utf-8")))
    refs = {"i": {"options": {n: e6_doc["arms"]["c"]["options"][n] for n in OPTIONS}},
            "ii": {"options": {n: e6.reference_arm(ROOT, cfg)["options"][n] for n in OPTIONS}}}
    emit(f"== E7 (diagnostic only): options {list(OPTIONS)} (19D dropped); fixed point {fp}; regime {regime} ==")
    cache = ROOT / cfg["paths"]["cache_dir"] / cfg["g0"]["cache_subdir"] / "e7"
    zero_cfg = zero_residual_cfg(cfg)
    timing, arms, compare = {}, {}, {}
    for key in ("i", "ii"):
        t = time.perf_counter()
        sets = build_sets_fixed(cfg, grid, table, regime, features) if key == "i" else build_sets_matched(cfg, grid, table)
        timing[f"{key}_generation_s"] = time.perf_counter() - t
        peak = max(float(np.max(np.abs(s.truth["planted"]))) for s in sets["positive"] if s.truth is not None)
        emit(f"arm ({key}): {sum(len(v) for v in sets.values())} series in {timing[f'{key}_generation_s'] / 60:.1f} min; "
             f"largest |planted term| over the positives {peak:.3g} (must be 0)")
        t = time.perf_counter()
        arms[key] = {"options": e6.run_arm(zero_cfg, sets, list(OPTIONS), n_workers, cache / key), "delta": sg.delta(cfg),
                     "max_abs_planted": peak}
        timing[f"{key}_run_s"] = time.perf_counter() - t
        emit(f"arm ({key}): done in {timing[f'{key}_run_s'] / 60:.1f} min")
        compare[key], _ = compare_lines(refs[key], arms[key], sg.delta(cfg))
    doc = {"stage": "E7 diagnostic only (synthetic, no PySR, no verdict); 19D dropped", "dev005_decision": "not decided; evidence only",
           "options": list(OPTIONS), "fixed_point": fp, "regime_at_fixed_p": regime, "arms": arms, "references": refs, "compare": compare,
           "rms_ratio": reference_ratios(cfg, grid, table, regime, features), "timing": timing, "n_workers": n_workers}
    sg.write_pilot_json(cfg, ROOT, doc, path)
    print_report(e6.unjson(json.loads(json.dumps(sg._jsonable(doc)))))
    emit(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
