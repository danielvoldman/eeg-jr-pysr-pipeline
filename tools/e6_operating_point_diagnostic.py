"""E6: DIAGNOSTIC ONLY, synthetic, pilot mode (PLAN.md E6; §9.1, §17, DEV-005, IMP-070). Is the E5 failure caused by the
matched operating points of the G0 generator (p = 120 / 138, input SD up to 2x, limit cycle) or by the pipeline?

Changes no src rule, threshold or config value and writes nothing under outputs/ (only results/pilot/e6_*.json). The
operating point and the chain are overridden INSIDE this script (module attributes of src.synthetic_gate, restored on exit;
a deep-copied config and a copied grid for the noise split and exponent). No PySR. Not a G0 verdict.

Arms (the same G0 generator, seeds, levels and cells as the E5 pilot series; 20 positive = 5 per level, 20 Null A, 20 Null B,
the 20-series tuning set):
  (a) the E5 pilot series, matched operating points, real chain: REUSED from results/pilot/g0_filter_comparison.json
  (b) operating point fixed at the C7-style default, LIGHT chain (band-pass, edge trim, downsampling, rescaling on one
      continuous series; no notch, no blink correction, no segment rejection), as the C6 / C7 data had
  (c) the same fixed operating point with the REAL preprocessing and rescaling chain, unchanged
(a) vs (c) isolates the operating point; (c) vs (b) isolates the chain. The C7 data had no Null B, no planted product
residual and no artifacts; the generator keeps them in (b) and (c).

C7-style default: p = mean of the reference input uniform (220), input half-width x1, additive noise share 0.5 with 1/f
power exponent 1.6 (slope -1.6 of C4d / C6 / C7). C7 added 1/f at 50% of the share of (signal + white R) on top of a white
observation noise at R = 0.25 of the rescaled variance, i.e. the additive noise was 0.25 white : 1.25 1/f, so the 1/f part
of the noise is 1.25 / 1.5 = 5/6 (the generator's own default is 0.8). Both are derived below from config values.

    python tools/e6_operating_point_diagnostic.py [--n-jobs 4]
"""
import contextlib
import copy
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import synthetic_gate as sg  # noqa: E402
from src.config import load_config  # noqa: E402

NOISE_SHARE = 0.5          # the user's C7-style default
SLOPE = -1.6               # C4d / C6 / C7 power-law slope; the generator's exponent is its negative
SETS = ("positive", "null_A", "null_B")
ARMS = ("b", "c")
ARM_TEXT = {"a": "(a) E5 pilot series: matched operating points, real chain (reused)",
            "b": "(b) fixed C7-style operating point, light chain",
            "c": "(c) fixed C7-style operating point, real chain"}


# ---------------------------------------------------------------- the fixed operating point

def fixed_point(cfg):
    """The C7-style default derived from config values: p, input factor, noise share, 1/f exponent and the 1/f part of the
    additive noise."""
    ref = cfg["rescaling"]["reference_simulation"]
    r = cfg["ukf"]["observation_noise"]["R_fraction_of_rescaled_variance"]
    one_over_f = NOISE_SHARE / (1.0 - NOISE_SHARE) * (1.0 + r)       # 1/f power relative to the rescaled variance
    return {"p": 0.5 * (ref["input_noise_uniform_low_s_inv"] + ref["input_noise_uniform_high_s_inv"]), "input_sd_factor": 1.0,
            "noise_share": NOISE_SHARE, "exponent": -SLOPE, "one_over_f_fraction": one_over_f / (one_over_f + r)}


def fixed_cfg_and_grid(cfg, grid):
    """Config copy with the 1/f part of the noise set and a grid copy with the exponent set (what observe_noisy reads)."""
    fp = fixed_point(cfg)
    c = copy.deepcopy(cfg)
    c["g0"]["regime"]["one_over_f_fraction_of_noise"] = fp["one_over_f_fraction"]
    return c, dataclasses.replace(grid, exponent=fp["exponent"])


def fixed_features(cfg, grid, feature_index):
    """Scale-free features (alpha peak, relative alpha power, exponent) of the fixed operating point, measured like a grid
    point (2 nodes, 60 s) on an otherwise unused seed index."""
    fp = fixed_point(cfg)
    c, g = fixed_cfg_and_grid(cfg, grid)
    v = np.asarray(sg.grid_point_features(c, feature_index, fp["p"], fp["input_sd_factor"], fp["noise_share"], g.exponent), dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise sg.GateError("the features of the fixed operating point could not be fitted")
    return v


def z_distance(features, target, scale):
    """Distance used by the grid matching: Euclidean in features divided by the training scale."""
    return float(np.sqrt((((np.asarray(features) - np.asarray(target)) / np.asarray(scale)) ** 2).sum()))


def light_preprocess(cfg, x_uv, fs_hz, strict=False):
    """Light chain of arm (b): one continuous series, no notch, no blink correction, no segment rejection. Returned in the
    shape generate_series reads from the real chain's result."""
    x = sg.lite_chain(cfg, x_uv, fs_hz)
    clean_s = x.shape[1] / cfg["preprocessing"]["observation_fs_hz"]
    return SimpleNamespace(segments=[np.ascontiguousarray(x)], starts=[0],
                           meta={"b5": {"log": {"clean_s": float(clean_s), "light_chain": True}}})


@contextlib.contextmanager
def override(cfg, features, table_scale, regime, light, original_op=None):
    """Replace series_operating_point (and, for the light chain, preprocess_series) of src.synthetic_gate for the duration
    of the block; both are restored on exit. The replaced operating point keeps the drawn target recording of the original
    (so the series share their draw with E5) and reports the z-distance of the FIXED point to that target."""
    fp = fixed_point(cfg)
    orig_op, orig_pre = sg.series_operating_point, sg.preprocess_series
    base = orig_op if original_op is None else original_op

    def op(cfg_, grid, table, stream, i, round_=0):
        o = dict(base(cfg_, grid, table, stream, i, round_))
        o.update({"p": fp["p"], "input_sd_factor": fp["input_sd_factor"], "noise_share": fp["noise_share"], "regime": regime,
                  "matched_distance_e5": o["distance"], "distance": z_distance(features, o["target"], table_scale),
                  "fixed_operating_point": True})
        return o

    sg.series_operating_point = op
    if light:
        sg.preprocess_series = light_preprocess
    try:
        yield
    finally:
        sg.series_operating_point, sg.preprocess_series = orig_op, orig_pre


def build_sets(cfg, grid, table, light, regime, features):
    """The 80 series of one overridden arm: 20 positive, 20 Null A, 20 Null B (stream 'pilot') and the tuning set."""
    c, g = fixed_cfg_and_grid(sg.g0_cfg(cfg), grid)
    with override(c, features, table.scale, regime, light):
        n = c["g0"]
        sets = {"positive": [sg.generate_series(c, "positive", i, g, table, "pilot") for i in range(n["n_pilot_positive"])],
                "null_A": [sg.generate_series(c, "null_A", i, g, table, "pilot") for i in range(n["n_pilot_null_A"])],
                "null_B": [sg.generate_series(c, "null_B", i, g, table, "pilot") for i in range(n["n_pilot_null_B"])],
                "tuning": sg.tuning_set(c, g, table)}
    return sets


# ---------------------------------------------------------------- scoring (the E5 schema, three evaluated sets)

def arm_option_summary(cfg, name, qr, recs, runtime, std_refusal):
    """The E5 per option summary for the sets of E6 (positive, Null A, Null B); same fields as sg.option_summary."""
    pos = recs["positive"]
    pv = sg.positive_verdict(cfg, pos)
    off = sg.positive_verdict(cfg, [sg._flag_off_record(r) for r in pos])
    gp = sg.gain_profile(cfg, pos)
    levels = {}
    for lv, v in pv["levels"].items():
        rs = [r for r in pos if r["level_index"] == lv]
        levels[str(lv)] = {"g_true": rs[0]["g_true"], "n": v["n"], "n_dropped_by_standard_rule": v["n_diverged"],
                           "gain_error_median_smoothed": v["gain_error_median"], "gain_error_median_filtered": v["gain_error_median_filtered"],
                           "diagnostic_flag_off_gain_error_median_smoothed": off["levels"][lv]["gain_error_median"],
                           "diagnostic_flag_off_gain_error_median_filtered": off["levels"][lv]["gain_error_median_filtered"],
                           "contraction_g_median": sg._median_contraction(cfg, rs, ("g12", "g21")),
                           "contraction_m_median": sg._median_contraction(cfg, rs, ("m",)),
                           "posterior_sd_g_median": gp[lv]["posterior_sd_median"], "z_distance": sg._z_summary(rs),
                           "detection_floor": v["detection_floor"]}
    nulls = {}
    for arm in ("null_A", "null_B"):
        c = sg._null_counts(cfg, recs[arm])
        nulls[arm] = {**c, "diagnostic_flag_off": {"diagnostic_only": True, **sg._null_counts(cfg, recs[arm], True)},
                      "z_distance": sg._z_summary(recs[arm])}
    allrecs = [r for s in SETS for r in recs[s]]
    row = qr.table[qr.q_index]
    return {"filter": name, "levels": levels, "nulls": nulls, "stability_all_runs": sg.stability_verdict(allrecs),
            "n_dropped_by_standard_rule": {s: int(sum(r["diverged"] for r in recs[s])) for s in SETS}, "runtime": runtime,
            "tuning": {"q": qr.q, "refused": False, "refusal_reason": None, "mean_nis_at_q": row["mean_nis"], "nis_target": qr.target,
                       "nis_band": qr.band, "in_band": bool(qr.in_band), "at_grid_edge": bool(qr.at_grid_edge),
                       "n_matched": qr.n_matched, "n_recordings": qr.n_recordings, "table": qr.table,
                       "diagnostic_flag_off_tuning": std_refusal is not None, "standard_rule_refusal": std_refusal}}


def run_arm(cfg, sets, options, n_workers, cache_dir):
    """Tune q (the E5 rule: standard rule first, diagnostic flag-off fallback, both labelled) and evaluate the 60 series of
    every option. Returns {option: summary}."""
    from joblib import Parallel, delayed
    out = {}
    for name in options:
        t = time.perf_counter()
        qr = sg.tune_g0_q(cfg, sets["tuning"], name, cache_dir=cache_dir, n_jobs=n_workers)
        std_refusal = None
        if qr.q is None:
            std_refusal = {"reason": qr.refusal_reason, "n_matched": qr.n_matched, "n_recordings": qr.n_recordings,
                           "n_diverged_per_q": [{"q": r["q"], "n_diverged": r["n_diverged"]} for r in qr.table]}
            if cfg["g0"]["pilot"]["tuning_fallback"] == "flag_off_nis":
                off = copy.deepcopy(cfg)
                off["ukf"]["divergence"]["state_sd_multiple"] = float("inf")
                qr = sg.tune_g0_q(off, sets["tuning"], name, cache_dir=cache_dir, n_jobs=n_workers)
        tune_s = time.perf_counter() - t
        if qr.q is None:
            out[name] = {"filter": name, "tuning": {"q": None, "refused": True, "refusal_reason": qr.refusal_reason, "standard_rule": std_refusal}}
            continue
        jobs = [(s, ser) for s in SETS for ser in sets[s]]
        t = time.perf_counter()
        results = Parallel(n_jobs=n_workers, backend=cfg["compute"]["joblib_backend"])(
            delayed(sg._series_worker)((cfg, ser, name, float(qr.q))) for _, ser in jobs)
        eval_s = time.perf_counter() - t
        recs = {s: [] for s in SETS}
        for (s, _), r in zip(jobs, results):
            recs[s].append(r)
        runtime = {"tuning_s": tune_s, "evaluation_s": eval_s, "series_busy_s": float(sum(r["runtime_s"] for r in results)),
                   "n_series": len(results)}
        out[name] = arm_option_summary(cfg, name, qr, recs, runtime, std_refusal)
        out[name]["per_series"] = {s: [{"index": r["index"], "level_index": r["level_index"], "diverged": r["diverged"],
                                        "z_distance": r["z_distance"], "regime": r["regime"]} for r in recs[s]] for s in SETS}
    return out


# ---------------------------------------------------------------- the reference arm (a) and the comparison

def unjson(obj):
    """Undo sg._jsonable for reading: the strings 'inf', '-inf' and 'nan' become floats again, recursively."""
    if isinstance(obj, dict):
        return {k: unjson(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unjson(v) for v in obj]
    if isinstance(obj, str) and obj in ("inf", "-inf", "nan"):
        return float(obj)
    return obj


def reference_arm(root, cfg):
    """Arm (a) from the E5 files: the option summaries without the gate sets, and the per-series matched z-distance."""
    d = Path(root) / cfg["paths"]["pilot_results_dir"]
    comp = unjson(json.loads((d / cfg["g0"]["pilot"]["comparison_file"]).read_text(encoding="utf-8")))
    options = {}
    for name, o in comp["options"].items():
        o = copy.deepcopy(o)
        o["nulls"] = {k: v for k, v in o["nulls"].items() if k in ("null_A", "null_B")}
        options[name] = o
    gate = json.loads((d / "gate_19D.json").read_text(encoding="utf-8"))
    z = {arm: gate["report"]["arms"][arm]["z_distance"]["per_series"] for arm in SETS}
    return {"options": options, "z_distance_per_series": z, "delta": comp["delta"]}


def comparison_rows(reports):
    """Compact (a) / (b) / (c) rows per option and level: dropped and the smoothed gain error with the rule ON and with the
    flag disabled (diagnostic), then the null counts."""
    rows = []
    for name in reports["a"]["options"]:
        for lv in reports["a"]["options"][name].get("levels", {}):
            row = {"option": name, "row": f"L{int(lv) + 1}"}
            for arm in ("a", "b", "c"):
                v = reports[arm]["options"][name].get("levels", {}).get(lv)
                row[arm] = None if v is None else {"dropped": v["n_dropped_by_standard_rule"], "err_sm": v["gain_error_median_smoothed"],
                                                   "off_sm": v["diagnostic_flag_off_gain_error_median_smoothed"],
                                                   "c_g": v["contraction_g_median"], "c_m": v["contraction_m_median"]}
            rows.append(row)
        for nm in ("null_A", "null_B"):
            row = {"option": name, "row": nm}
            for arm in ("a", "b", "c"):
                v = reports[arm]["options"][name].get("nulls", {}).get(nm)
                row[arm] = None if v is None else {"out": v["n_out"], "dropped": v["n_dropped"], "inside": v["n_inside"],
                                                   "off_out": v["diagnostic_flag_off"]["n_out"], "n": v["n"]}
            rows.append(row)
    return rows


def format_rows(rows):
    def pc(v):
        return "inf" if v is None or not np.isfinite(v) else f"{100 * v:.0f}%"
    lines = [f"{'option':<7}{'row':<8}  {'(a) drop err off':<20}{'(b) drop err off':<20}{'(c) drop err off':<20}   contraction g/m (a | b | c)"]
    for r in rows:
        cells = []
        for arm in "abc":
            v = r[arm]
            if v is None:
                cells.append(f"{'-':<20}")
            elif "err_sm" in v:
                cells.append(f"{v['dropped']:>2}  {pc(v['err_sm']):>6} {pc(v['off_sm']):>6}".ljust(20))
            else:
                cells.append(f"out {v['out']:>2} drop {v['dropped']:>2} off {v['off_out']:>2}".ljust(20))
        tail = ""
        if r["a"] is not None and "c_g" in r["a"]:
            tail = "   " + " | ".join("-" if r[a] is None else f"{r[a]['c_g']:.2f}/{r[a]['c_m']:.2f}" for a in "abc")
        lines.append(f"{r['option']:<7}{r['row']:<8}  " + "".join(cells) + tail)
    return lines


def write_json(cfg, root, doc, name):
    path = Path(root) / cfg["paths"]["pilot_results_dir"] / name
    return sg.write_pilot_json(cfg, root, doc, path)


def print_report(doc, emit=None):
    """The E6 tables from the written document (also used by --report-only)."""
    emit = emit or sg.emit
    reports = doc["arms"]
    for arm in "abc":
        emit(f"#### {ARM_TEXT[arm]}")
        for line in sg.format_comparison({"delta": reports[arm]["delta"], "options": reports[arm]["options"]}):
            emit(line)
    emit("#### side by side: dropped by the standard rule / median smoothed gain error rule ON / same with the flag disabled (DIAGNOSTIC ONLY)")
    for line in format_rows(doc["comparison_rows"]):
        emit(line)
    emit(f"#### (a) matched z-distance per series (series 0-19), the same for every option: "
         f"{np.round(reports['a']['z_distance_per_series']['positive'], 2).tolist()}")
    emit(f"#### z-distance of the fixed point per series in (c): {np.round(reports['c']['z_distance_per_series']['positive'], 2).tolist()}")
    emit(f"#### regimes in (b) and (c): {reports['b']['regimes']} / {reports['c']['regimes']}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--report-only", action="store_true", help="print the tables of the written results/pilot/e6_operating_point.json")
    args = ap.parse_args(argv)
    cfg = load_config()
    emit = sg.emit
    if args.report_only:
        doc = unjson(json.loads((ROOT / cfg["paths"]["pilot_results_dir"] / "e6_operating_point.json").read_text(encoding="utf-8")))
        doc["arms"]["a"] = reference_arm(ROOT, cfg)
        print_report(doc)
        return 0
    n_workers = sg.pool_size(cfg, args.n_jobs)
    options = sg.gate_filters(cfg, True)
    table, grid = sg.pilot_inputs(cfg, ROOT)
    fp = fixed_point(cfg)
    regime = sg.classify_regime(cfg, fp["p"])
    features = fixed_features(cfg, grid, len(grid.features))
    emit(f"== E6 (diagnostic only): fixed point {fp}; regime of the noise-free run at p = {fp['p']:.0f}: {regime}; "
         f"features {np.round(features, 3).tolist()} ==")
    cache = ROOT / cfg["paths"]["cache_dir"] / cfg["g0"]["cache_subdir"] / "e6"
    reports = {"a": reference_arm(ROOT, cfg)}
    timing = {}
    for arm in ARMS:
        t = time.perf_counter()
        sets = build_sets(cfg, grid, table, light=(arm == "b"), regime=regime, features=features)
        timing[f"{arm}_generation_s"] = time.perf_counter() - t
        emit(f"arm ({arm}): {sum(len(v) for v in sets.values())} series generated in {timing[f'{arm}_generation_s'] / 60:.1f} min")
        t = time.perf_counter()
        reports[arm] = {"options": run_arm(cfg, sets, options, n_workers, cache / arm), "delta": sg.delta(cfg),
                        "z_distance_per_series": {s: [x.operating_point["distance"] for x in sets[s]] for s in SETS},
                        "regimes": {s: sorted({x.operating_point["regime"] for x in sets[s]}) for s in SETS}}
        timing[f"{arm}_run_s"] = time.perf_counter() - t
        emit(f"arm ({arm}): done in {timing[f'{arm}_run_s'] / 60:.1f} min")
    rows = comparison_rows(reports)
    doc = {"stage": "E6 diagnostic only (synthetic, no PySR, no verdict)", "dev005_decision": "not decided; evidence only",
           "fixed_point": fp, "regime_at_fixed_p": regime, "fixed_point_features": features.tolist(), "arms": reports,
           "comparison_rows": rows, "timing": timing, "n_workers": n_workers}
    path = write_json(cfg, ROOT, doc, "e6_operating_point.json")
    print_report(unjson(json.loads(json.dumps(sg._jsonable(doc)))))
    emit(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
