"""C6: read-only diagnostic on SYNTHETIC data: are the coupling gains and m still recovered at recording level
with a 1/f aperiodic background present, and do the nulls stay quiet? (PLAN.md C6; §7.5, §9.1 to §9.3)

Changes no src rule, threshold or config value; writes nothing under outputs/. Everything is applied inside
this script: the Numba backend is selected through the backend argument (ukf.run_filter / ukf.run_smoother are
wrapped with backend="numba" inside each worker, undone afterwards), and the state-SD flag is disabled the way
C4b to C4d disable it (here: a config COPY with ukf.divergence.state_sd_multiple = 1e30; the NaN/Inf and
covariance-PD checks stay on). No real recording is read.

Series (per series 234 s = 4 s burn + clean segments of 70, 50, 40, 60 s + 3 gaps of 3 s + 1 s tail):
  S0 plain (model units), S1 + 1/f noise (slope -1.6, 50% of the 1-45 Hz variance), S2 (80%), the noise added
  BEFORE the 0.5-45 Hz band-pass and the rescaling to mu_ref / sigma_ref, as in tools/c4d_aperiodic_diagnostic.py.
Cells: g12 = g21 = level * C2 for the four levels of g0.coupling_levels_x_C2, plus the null (g = 0); 6 series per
cell; m drawn per series from U(m_range of g0.mixing_m_range) (§9.1). p stays at the simulator default.
Runs per series: S0 q = 1e-2, S1 q = 1e-2, S2 q = 1e-2, S1 q = 1e-3; each (a) with the state-SD flag disabled,
full pass 1 (forward + smoother) for the estimates, and (b) with the standard rule ON, forward-only pass 1
(the divergence decision depends on the forward filter only) for the divergence counts.

    python tools/c6_gain_recovery_diagnostic.py <raw_results.json> [n_jobs]
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
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
os.environ["PYTHONPATH"] = os.pathsep.join([str(TOOLS), str(ROOT), os.environ.get("PYTHONPATH", "")])

import numpy as np  # noqa: E402

import c4d_aperiodic_diagnostic as D  # noqa: E402

SEG_SECONDS = [70.0, 50.0, 40.0, 60.0]
N_SERIES = 6
CELLS = ("L1", "L2", "L3", "L4", "null")          # the four coupling levels, then the null
RUNS = (("S0", None, 1e-2), ("S1", 0.5, 1e-2), ("S2", 0.8, 1e-2), ("S1", 0.5, 1e-3))   # (variant, 1/f share, q)
SEED_SIM = 30000          # sim seed = SEED_SIM + 100 * cell index + series index (C4b to C4d: 101 to 106)
SEED_1F = 40000           # 1/f noise seed = SEED_1F + sim seed * 10 + channel (C4d: 2000 + ...)
SEED_M = 50000            # m draw: default_rng(SEED_M + sim seed)
STATE_FLAG_OFF = 1e30     # state_sd_multiple used for the flag-disabled runs (config copy, inside this script)
CRITERION = 0.15          # g0.pass.median_gain_rel_error_max
CONTRACTION_MIN = 0.5     # g0.pass.contraction_min
_warm = {"done": False}


def _sim_data():
    """tests/sim_data.py loaded by path (tests/ has no __init__.py and a site-packages 'tests' package shadows it)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("c6_sim_data", ROOT / "tests" / "sim_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell_truth(cfg, cell):
    from src import model
    if cell == "null":
        return 0.0
    level = cfg["g0"]["coupling_levels_x_C2"][CELLS.index(cell)]
    return level * model.constants(cfg)["C2"]


def make_series(cfg, r, seed, share, seed_1f):
    """Segments and starts of one series from the simulated recording r. share None = plain (S0)."""
    from src import preprocess
    if share is None:
        return r["segments"], r["starts"]
    fs = cfg["preprocessing"]["observation_fs_hz"]
    z = r["z_all"].T.copy()
    out = np.empty_like(z)
    for ch in range(2):
        rng = np.random.default_rng(seed_1f + seed * 10 + ch)
        nz = D.shaped_noise(rng, z.shape[1], D.SLOPE)
        c = np.sqrt(share / (1.0 - share) * D.band_power(z[ch], fs) / D.band_power(nz, fs))
        out[ch] = z[ch] + c * nz
    filt = preprocess.bandpass(cfg, out, fs)
    lens = [s.shape[1] for s in r["segments"]]
    segs = [filt[:, st:st + n] for st, n in zip(r["starts"], lens)]
    allx = np.concatenate(segs, axis=1)
    mu, sg = cfg["rescaling"]["mu_ref"], cfg["rescaling"]["sigma_ref"]
    mean, sd = allx.mean(axis=1, keepdims=True), allx.std(ddof=1, axis=1, keepdims=True)
    return [np.ascontiguousarray(mu + sg * (s - mean) / sd) for s in segs], r["starts"]


def _pass1_numba(segments, starts, cfg, q, forward_only):
    """passes.run_pass1 with the Numba backend chosen through the backend argument (wrappers undone afterwards)."""
    from functools import partial
    from src import passes, ukf
    orig_rf, orig_rs = ukf.run_filter, ukf.run_smoother
    ukf.run_filter, ukf.run_smoother = partial(orig_rf, backend="numba"), partial(orig_rs, backend="numba")
    try:
        return passes.run_pass1(segments, starts, cfg, q, forward_only=forward_only)
    finally:
        ukf.run_filter, ukf.run_smoother = orig_rf, orig_rs


def _worker(job):
    """One (cell, series): simulate once, then every variant / q, flag disabled (estimates) and rule ON (divergence)."""
    from threadpoolctl import threadpool_limits
    from src import model
    cell, si, cfg = job
    ci = CELLS.index(cell)
    seed = SEED_SIM + 100 * ci + si
    g = _cell_truth(cfg, cell)
    m = float(np.random.default_rng(SEED_M + seed).uniform(*cfg["g0"]["mixing_m_range"]))
    cfg_off = copy.deepcopy(cfg)
    cfg_off["ukf"]["divergence"]["state_sd_multiple"] = STATE_FLAG_OFF
    prior_sd = {"g12": cfg["coupling"]["gain_prior_sd_factor_of_C2"] * model.constants(cfg)["C2"],
                "g21": cfg["coupling"]["gain_prior_sd_factor_of_C2"] * model.constants(cfg)["C2"],
                "m": cfg["priors"]["m_sd"]}
    out = []
    with threadpool_limits(limits=1):
        if not _warm["done"]:                                  # compile / cache load stays out of every timing
            w = _sim_data().make_recording(cfg, [8.0], 1, g12=1.0, g21=1.0)
            _pass1_numba(w["segments"], w["starts"], cfg_off, 1e-2, False)
            _warm["done"] = True
        t_sim = time.perf_counter()
        rec0 = _sim_data().make_recording(cfg, SEG_SECONDS, seed, g12=g, g21=g, m=m)
        base = {(v,): make_series(cfg, rec0, seed, sh, SEED_1F) for v, sh in (("S0", None), ("S1", 0.5), ("S2", 0.8))}
        t_sim = time.perf_counter() - t_sim
        for vname, share, q in RUNS:
            segs, starts = base[(vname,)]
            t0 = time.perf_counter()
            res = _pass1_numba(segs, starts, cfg_off, q, forward_only=False)
            wall = time.perf_counter() - t0
            on = _pass1_numba(segs, starts, cfg, q, forward_only=True)
            rec = {"cell": cell, "series": si, "seed": seed, "variant": vname, "q": q, "g_true": g, "m_true": m,
                   "wall_s": wall, "sim_s": t_sim,
                   "off_failures": [f"seg {s.index}: {s.reason} at step {s.step}" for s in res.segments if s.diverged],
                   "on_segments": len(on.segments), "on_segments_dropped": int(sum(s.diverged for s in on.segments)),
                   "on_recording_diverged": bool(on.recording_diverged),
                   "on_diverged_fraction": float(on.diverged_fraction)}
            if res.params is not None:
                p = res.params
                rec.update({"g12": p.g12, "g21": p.g21, "m": p.m, "m_raw": p.m_raw,
                            "m_clipped_fraction": p.m_clipped_fraction,
                            "contraction": {k: 1.0 - p.posterior_sd[k] / prior_sd[k] for k in ("g12", "g21", "m")},
                            "post_sd": {k: p.posterior_sd[k] for k in ("g12", "g21", "m")}})
            if res.gain_estimate is not None:
                rec.update({"g12_filt": res.gain_estimate["g12"], "g21_filt": res.gain_estimate["g21"]})
            out.append(rec)
    return out


# ---- reporting -------------------------------------------------------------------------------------

def _mm(v, fmt="{:.2f}"):
    v = np.asarray(v, dtype=np.float64)
    return f"{fmt.format(np.median(v))} [{fmt.format(v.min())},{fmt.format(v.max())}]"


def _rel(rec, key):
    return abs(rec[key] - rec["g_true"]) / rec["g_true"]


def main():
    from src.config import load_config
    from joblib import Parallel, delayed
    out_path = Path(sys.argv[1])
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cfg = load_config()
    jobs = [(c, s, cfg) for c in CELLS for s in range(N_SERIES)]
    print(f"{len(jobs)} (cell, series) jobs x {len(RUNS)} runs, {n_jobs} workers", flush=True)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_worker)(j) for j in jobs)
    print(f"wall {time.perf_counter() - t0:.0f} s", flush=True)
    rows = [r for rr in res for r in rr]
    out_path.write_text(json.dumps(rows))
    delta = cfg["g0"]["pass"]["null_delta_fraction_of_weakest_level"] * _cell_truth(cfg, "L1")
    keys = [(v, q) for v, _, q in RUNS]
    truth = {c: _cell_truth(cfg, c) for c in CELLS}
    print(f"\ntruth g per cell: " + ", ".join(f"{c} {truth[c]:.2f}" for c in CELLS) + f"; delta = {delta:.2f}; "
          f"6 series per cell; m truth U(0.1, 0.4) per series")

    def sel(v, q, c):
        return [r for r in rows if (r["variant"], r["q"], r["cell"]) == (v, q, c)]

    print("\n=== 1/2. gain recovery. g>0: relative error %, median [min,max] over series (smoothed = pass-1 smoothed mean; "
          "filt = filtered-gain estimator, 6 s burn-in). null: |g|. ok = fraction of series with BOTH gains within 15% "
          "(null: both |g| < delta); lvl = level median of the pooled 12 gain errors <= 15% ===")
    print(f"  {'run':<9}{'cell':<6}{'g12 smooth':>20}{'g21 smooth':>20}{'g12 filt':>20}{'g21 filt':>20}{'ok sm':>7}{'ok filt':>8}{'lvl sm':>7}{'lvl filt':>9}")
    for v, q in keys:
        for c in CELLS:
            rr = [r for r in sel(v, q, c) if "g12" in r and "g12_filt" in r]
            if not rr:
                print(f"  {v + ' q=' + format(q, 'g'):<9}{c:<6}  no estimate"); continue
            if c == "null":
                f = lambda k: _mm([abs(r[k]) for r in rr])
                ok_s = np.mean([abs(r["g12"]) < delta and abs(r["g21"]) < delta for r in rr])
                ok_f = np.mean([abs(r["g12_filt"]) < delta and abs(r["g21_filt"]) < delta for r in rr])
                lv_s = lv_f = float("nan")
            else:
                f = lambda k: _mm([100 * _rel(r, k) for r in rr], "{:.0f}")
                ok_s = np.mean([_rel(r, "g12") <= CRITERION and _rel(r, "g21") <= CRITERION for r in rr])
                ok_f = np.mean([_rel(r, "g12_filt") <= CRITERION and _rel(r, "g21_filt") <= CRITERION for r in rr])
                lv_s = np.median([_rel(r, k) for r in rr for k in ("g12", "g21")])
                lv_f = np.median([_rel(r, k) for r in rr for k in ("g12_filt", "g21_filt")])
            print(f"  {v + ' q=' + format(q, 'g'):<9}{c:<6}{f('g12'):>20}{f('g21'):>20}{f('g12_filt'):>20}{f('g21_filt'):>20}"
                  f"{ok_s:>7.2f}{ok_f:>8.2f}" + (f"{'-':>7}{'-':>9}" if c == "null" else f"{lv_s:>7.2f}{lv_f:>9.2f}"))
    print(f"  ({n_series_note(rows)})")

    print("\n=== 3. m recovery (per-series clipped mean minus truth) and posterior-SD contraction 1 - post/prior (g: prior SD 0.1 C2; m: 0.15); "
          "median [min,max]; * = level median < 0.5 ===")
    print(f"  {'run':<9}{'cell':<6}{'m err':>20}{'contr g12':>22}{'contr g21':>22}{'contr m':>22}")
    for v, q in keys:
        for c in CELLS:
            rr = [r for r in sel(v, q, c) if "g12" in r]
            if not rr:
                continue
            cells = []
            for k in ("g12", "g21", "m"):
                vals = [r["contraction"][k] for r in rr]
                cells.append((_mm(vals) + ("*" if np.median(vals) < CONTRACTION_MIN else " ")).rjust(22))
            print(f"  {v + ' q=' + format(q, 'g'):<9}{c:<6}{_mm([r['m'] - r['m_true'] for r in rr], '{:+.3f}'):>20}" + "".join(cells))

    print("\n=== 4/5. divergence with the standard rule ON (fraction of series with recording_diverged; fraction with >= 1 segment dropped; "
          "segments dropped of 4, total over the 6 series) and NaN/Inf/PD failures with the flag disabled; runtime per series "
          "(full pass 1, warm, 4 workers in parallel), median [min,max] s ===")
    print(f"  {'run':<9}{'cell':<6}{'rec div':>8}{'any seg':>8}{'segs':>6}{'off fail':>9}{'runtime s':>20}")
    for v, q in keys:
        for c in CELLS:
            rr = sel(v, q, c)
            print(f"  {v + ' q=' + format(q, 'g'):<9}{c:<6}{np.mean([r['on_recording_diverged'] for r in rr]):>8.2f}"
                  f"{np.mean([r['on_segments_dropped'] > 0 for r in rr]):>8.2f}{sum(r['on_segments_dropped'] for r in rr):>6d}"
                  f"{sum(bool(r['off_failures']) for r in rr):>9d}{_mm([r['wall_s'] for r in rr], '{:.1f}'):>20}")
    fails = [(r["variant"], r["q"], r["cell"], r["series"], r["off_failures"]) for r in rows if r["off_failures"]]
    print("\nNaN/Inf or covariance-not-PD failures (flag disabled): " + ("none" if not fails else str(fails)))
    print(f"simulation + 1/f + filtering setup per series (all variants): {_mm([r['sim_s'] for r in rows if r['variant'] == 'S0' and r['q'] == 1e-2], '{:.1f}')} s")


def n_series_note(rows):
    return f"{len(rows)} runs in total"


if __name__ == "__main__":
    main()
