"""C7: read-only comparison of three ways to absorb a 1/f background, on SYNTHETIC data only (PLAN.md C7; §7.4, §7.5,
§9.1 to §9.3, §20). Changes no src rule, threshold or config value; writes nothing under outputs/. The candidates
live in tools/c7/ext_ukf.py (edited copies of the Numba kernels) and are run through the unchanged src/passes.py.

Arms (Spec kinds): control N = the 19-D filter as in src (the extended kernel with no extra state and no adaptation,
tested equal to src.ukf_numba), A = per-channel OU coloured observation noise (21-D), B = per-channel random-walk
offset (21-D), C = Sage-Husa adaptive R (19-D). A and B take (s2, tau) per channel from the recording's own spectrum
(tools/c7/aperiodic_prior.py), C takes none; nothing is tuned on a gain result.

Data: as tools/c6_gain_recovery_diagnostic.py (234 s = 4 segments of 70, 50, 40 and 60 s; L1-L4 x C2 and a null; 6 series
per cell; S0 plain, S1 = 50% and S2 = 80% 1/f before the band-pass and the rescaling; m ~ U(0.1, 0.4) per series) with
FRESH seeds. Every arm runs (a) full pass 1 with the state-SD flag disabled (config copy) for the estimates and the
stability monitors and (b) forward-only pass 1 with the standard rule ON for the divergence counts.

Fixed pass criteria (agreed before any run), per arm and per (S1|S2, q):
  c1 median relative gain error <= 15% at L2, L3, L4 (pooled g12 and g21), for the smoother AND the filtered-gain estimator
  c2 every null series has both |g| < delta = 1.08, for both estimators
  c3 median m contraction >= 0.5 at L2, L3, L4
  c4 with the rule ON at least 5 of 6 series stay undiverged (recording-level flag), in every cell
  c5 no NaN/Inf or covariance-not-PD failure with the flag disabled
S0 (same q): per level L1-L4 median error <= control + 2 percentage points, all nulls inside delta, undiverged count per
cell >= control's. Viable = some q with c1-c5 for S1 and S2 and the S0 condition at that q.

    python tools/c7_candidate_comparison.py estimator <out.json>     # step 2: the estimator on S0 / S1 / S2, no filtering
    python tools/c7_candidate_comparison.py run <out.json> [n_jobs]  # step 3: everything
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
sys.path.insert(0, str(TOOLS / "c7"))
os.environ["PYTHONPATH"] = os.pathsep.join([str(TOOLS / "c7"), str(TOOLS), str(ROOT), os.environ.get("PYTHONPATH", "")])

import numpy as np  # noqa: E402

import aperiodic_prior as ap  # noqa: E402
import c6_gain_recovery_diagnostic as C6  # noqa: E402

CELLS = C6.CELLS
VARIANTS = (("S0", None), ("S1", 0.5), ("S2", 0.8))
QS = (1e-2, 1e-3)
ARMS = ("N", "A", "B", "C")
ARM_NAME = {"N": "control", "A": "A_ou", "B": "B_walk", "C": "C_adaptR"}
SEED_SIM = 60000          # sim seed = SEED_SIM + 100 * cell index + series index (C6: 30000; C4b to C4d: 101 to 106)
SEED_1F = 70000           # 1/f noise seed = SEED_1F + sim seed * 10 + channel (C6: 40000)
SEED_M = 80000            # m draw: default_rng(SEED_M + sim seed) (C6: 50000)
N_SERIES = 6
S2_FLOOR_FRACTION = 1e-6  # a fit with no positive bin gets s2 = this fraction of the channel variance (keeps P positive definite)
CRITERION, CONTRACTION_MIN = 0.15, 0.5
S0_MARGIN = 0.02          # S0: candidate level error <= control + 2 percentage points
_warm = {"done": False}


def _series_for(cfg, cell, si):
    """Simulate once, then the three variants' segments."""
    ci = CELLS.index(cell)
    seed = SEED_SIM + 100 * ci + si
    g = C6._cell_truth(cfg, cell)
    m = float(np.random.default_rng(SEED_M + seed).uniform(*cfg["g0"]["mixing_m_range"]))
    rec0 = C6._sim_data().make_recording(cfg, C6.SEG_SECONDS, seed, g12=g, g21=g, m=m)
    data = {v: C6.make_series(cfg, rec0, seed, sh, SEED_1F) for v, sh in VARIANTS}
    return seed, g, m, data


def _estimate(cfg, segs):
    from src import ukf
    fs = cfg["preprocessing"]["observation_fs_hz"]
    return ap.estimate(segs, fs, float(ukf.obs_noise(cfg)[0, 0]))


def _spec(kind, est):
    from ext_ukf import Spec
    if kind in ("N", "C"):
        return Spec(kind)
    s2 = np.maximum(est["s2"], S2_FLOOR_FRACTION * est["var"])
    return Spec(kind, s2=tuple(float(v) for v in s2), tau=tuple(float(v) for v in est["tau"]))


def _pass1(spec, segs, starts, cfg, q, forward_only):
    from ext_ukf import patched_filters
    from src import passes
    with patched_filters(spec) as pf:
        res = passes.run_pass1(segs, starts, cfg, q, forward_only=forward_only, filter_name="19D")
    return res, pf


def _worker(job):
    """One (cell, series): all variants, arms and q."""
    from threadpoolctl import threadpool_limits
    from src import model
    cell, si, cfg = job
    cfg_off = copy.deepcopy(cfg)
    cfg_off["ukf"]["divergence"]["state_sd_multiple"] = C6.STATE_FLAG_OFF
    c2 = model.constants(cfg)["C2"]
    prior_sd = {"g12": cfg["coupling"]["gain_prior_sd_factor_of_C2"] * c2, "g21": cfg["coupling"]["gain_prior_sd_factor_of_C2"] * c2,
                "m": cfg["priors"]["m_sd"]}
    out = []
    with threadpool_limits(limits=1):
        seed, g, m, data = _series_for(cfg, cell, si)
        if not _warm["done"]:                               # compile / cache load stays out of every timing
            w = C6._sim_data().make_recording(cfg, [8.0], 1, g12=1.0, g21=1.0)
            for kind in ARMS:
                from ext_ukf import Spec
                _pass1(Spec(kind, s2=(1.0, 1.0), tau=(0.1, 0.1)) if kind in ("A", "B") else Spec(kind), w["segments"], w["starts"], cfg_off, 1e-2, False)
            _warm["done"] = True
        for vname, _ in VARIANTS:
            segs, starts = data[vname]
            est = _estimate(cfg, segs)
            for arm in ARMS:
                spec = _spec(arm, est)
                for q in QS:
                    t0 = time.perf_counter()
                    res, pf = _pass1(spec, segs, starts, cfg_off, q, False)
                    wall = time.perf_counter() - t0
                    on, _ = _pass1(spec, segs, starts, cfg, q, True)
                    mons = pf.monitors
                    nis_n = sum(c for _, c in pf.nis)
                    rec = {"cell": cell, "series": si, "seed": seed, "variant": vname, "q": q, "arm": arm,
                           "g_true": g, "m_true": m, "wall_s": wall,
                           "est_s2": [float(v) for v in est["s2"]], "est_tau": [float(v) for v in est["tau"]],
                           "chan_var": [float(v) for v in est["var"]],
                           "off_failures": [f"seg {s.index}: {s.reason} at step {s.step}" for s in res.segments if s.diverged],
                           "min_eig_overall": min(mm["min_eig_overall"] for mm in mons),
                           "neg_eig_steps": int(sum(mm["n_negative_eig_steps"] for mm in mons)),
                           "jitter_fallbacks": int(sum(mm["n_jitter_fallbacks"] for mm in mons)),
                           "nan_inf": bool(any(mm["nan_inf_seen"] for mm in mons)),
                           "nis_mean": (sum(s for s, _ in pf.nis) / nis_n) if nis_n else None,
                           "on_segments": len(on.segments), "on_segments_dropped": int(sum(s.diverged for s in on.segments)),
                           "on_recording_diverged": bool(on.recording_diverged),
                           "r_used": [[float(v) for v in x] for x in pf.r_used]}
                    if res.params is not None:
                        p = res.params
                        rec.update({"g12": p.g12, "g21": p.g21, "m": p.m, "m_clipped_fraction": p.m_clipped_fraction,
                                    "contraction": {k: 1.0 - p.posterior_sd[k] / prior_sd[k] for k in ("g12", "g21", "m")}})
                    if res.gain_estimate is not None:
                        rec.update({"g12_filt": res.gain_estimate["g12"], "g21_filt": res.gain_estimate["g21"]})
                    out.append(rec)
    return out


def _estimator_worker(job):
    from threadpoolctl import threadpool_limits
    cell, si, cfg = job
    with threadpool_limits(limits=1):
        seed, g, m, data = _series_for(cfg, cell, si)
        rows = []
        for vname, _ in VARIANTS:
            est = _estimate(cfg, data[vname][0])
            rows.append({"cell": cell, "series": si, "variant": vname, "s2": [float(v) for v in est["s2"]],
                         "tau": [float(v) for v in est["tau"]], "var": [float(v) for v in est["var"]]})
    return rows


# ---- reporting -------------------------------------------------------------------------------------------------

def _mm(v, fmt="{:.2f}"):
    v = np.asarray(v, dtype=np.float64)
    return f"{fmt.format(np.median(v))} [{fmt.format(v.min())},{fmt.format(v.max())}]"


def _rel(r, key):
    return abs(r[key] - r["g_true"]) / r["g_true"]


def print_estimator(rows, cfg):
    from src import ukf
    print(f"\nestimator output (bins {ap.FIT_HZ} Hz excluding {ap.EXCLUDE_HZ}, tau in {ap.TAU_BOUNDS_S} s, white floor of R = "
          f"{ukf.obs_noise(cfg)[0, 0]:.4g} subtracted); 30 series x 2 channels per variant, median [min,max]")
    print(f"  {'variant':<8}{'s2 / channel variance':>26}{'tau s':>22}{'share > 0.10':>14}   (planted 1/f share of the 1-45 Hz variance: S0 0, S1 0.5, S2 0.8)")
    stop = False
    for v, _ in VARIANTS:
        rr = [r for r in rows if r["variant"] == v]
        share = np.array([s / var for r in rr for s, var in zip(r["s2"], r["var"])])
        tau = np.array([t for r in rr for t in r["tau"]])
        frac = float(np.mean(share > 0.10))
        print(f"  {v:<8}{_mm(share, '{:.3f}'):>26}{_mm(tau, '{:.3f}'):>22}{frac:>14.2f}")
        if v == "S0" and np.any(share > 0.10):
            stop = True
    print("  S0 rule: STOP if a fitted s2 exceeds 10% of the channel variance: " + ("TRIGGERED" if stop else "not triggered"))
    return stop


def _cell_rows(rows, arm, v, q, c):
    return [r for r in rows if (r["arm"], r["variant"], r["q"], r["cell"]) == (arm, v, q, c)]


def _level_err(rr, keys):
    return float(np.median([_rel(r, k) for r in rr for k in keys]))


def _null_ok(rr, keys):
    return bool(all(abs(r[k]) < C6.DELTA for r in rr for k in keys))


def report(rows, cfg):
    delta = cfg["g0"]["pass"]["null_delta_fraction_of_weakest_level"] * C6._cell_truth(cfg, "L1")
    C6.DELTA = delta
    combos = [(v, q) for v, _ in VARIANTS for q in QS]
    lab = lambda arm, v, q: f"{ARM_NAME[arm]:<9}{v} q={q:g}"
    print(f"\ntruth g per cell: " + ", ".join(f"{c} {C6._cell_truth(cfg, c):.2f}" for c in CELLS) + f"; delta = {delta:.2f}; 6 series per cell")

    print("\n=== 1. gain recovery: relative error %, median [min,max] over series (null: |g|); smooth = pass-1 smoothed mean, "
          "filt = filtered-gain estimator; ok = fraction of series with both gains within 15% (null: both |g| < delta); "
          "lvl = level median of the pooled 12 errors ===")
    print(f"  {'arm / run':<19}{'cell':<6}{'g12 smooth':>20}{'g21 smooth':>20}{'g12 filt':>20}{'g21 filt':>20}{'ok sm':>7}{'ok f':>6}{'lvl sm':>7}{'lvl f':>7}")
    for arm in ARMS:
        for v, q in combos:
            for c in CELLS:
                rr = [r for r in _cell_rows(rows, arm, v, q, c) if "g12" in r and "g12_filt" in r]
                if not rr:
                    print(f"  {lab(arm, v, q):<19}{c:<6}  no estimate"); continue
                if c == "null":
                    f = lambda k: _mm([abs(r[k]) for r in rr])
                    ok_s = np.mean([abs(r["g12"]) < delta and abs(r["g21"]) < delta for r in rr])
                    ok_f = np.mean([abs(r["g12_filt"]) < delta and abs(r["g21_filt"]) < delta for r in rr])
                    tail = f"{'-':>7}{'-':>7}"
                else:
                    f = lambda k: _mm([100 * _rel(r, k) for r in rr], "{:.0f}")
                    ok_s = np.mean([_rel(r, "g12") <= CRITERION and _rel(r, "g21") <= CRITERION for r in rr])
                    ok_f = np.mean([_rel(r, "g12_filt") <= CRITERION and _rel(r, "g21_filt") <= CRITERION for r in rr])
                    tail = f"{_level_err(rr, ('g12', 'g21')):>7.2f}{_level_err(rr, ('g12_filt', 'g21_filt')):>7.2f}"
                print(f"  {lab(arm, v, q):<19}{c:<6}{f('g12'):>20}{f('g21'):>20}{f('g12_filt'):>20}{f('g21_filt'):>20}{ok_s:>7.2f}{ok_f:>6.2f}{tail}")

    print("\n=== 2. m error (clipped smoothed mean minus truth) and contraction 1 - posterior SD / prior SD (g prior SD 0.1 C2, m 0.15), "
          "median [min,max]; * = median < 0.5 ===")
    print(f"  {'arm / run':<19}{'cell':<6}{'m err':>20}{'contr g12':>22}{'contr g21':>22}{'contr m':>22}")
    for arm in ARMS:
        for v, q in combos:
            for c in CELLS:
                rr = [r for r in _cell_rows(rows, arm, v, q, c) if "g12" in r]
                if not rr:
                    continue
                cells = []
                for k in ("g12", "g21", "m"):
                    vals = [r["contraction"][k] for r in rr]
                    cells.append((_mm(vals) + ("*" if np.median(vals) < CONTRACTION_MIN else " ")).rjust(22))
                print(f"  {lab(arm, v, q):<19}{c:<6}{_mm([r['m'] - r['m_true'] for r in rr], '{:+.3f}'):>20}" + "".join(cells))

    print("\n=== 3. divergence with the standard rule ON (rec div = fraction of series with the recording-level flag; any seg = fraction "
          "with >= 1 segment dropped; segs = dropped of 24), flag-disabled monitors (off fail = series with a NaN/Inf/PD failure; "
          "neg eig = steps with a negative eigenvalue, total; jit = jitter fallbacks; min eig = smallest over series), mean NIS after "
          "the 6-s burn-in (median [min,max]) and runtime per series (4 workers in parallel) ===")
    print(f"  {'arm / run':<19}{'cell':<6}{'rec div':>8}{'any seg':>8}{'segs':>5}{'off fail':>9}{'neg eig':>8}{'jit':>5}{'min eig':>10}{'NIS':>20}{'runtime s':>19}")
    for arm in ARMS:
        for v, q in combos:
            for c in CELLS:
                rr = _cell_rows(rows, arm, v, q, c)
                nis = [r["nis_mean"] for r in rr if r["nis_mean"] is not None]
                print(f"  {lab(arm, v, q):<19}{c:<6}{np.mean([r['on_recording_diverged'] for r in rr]):>8.2f}"
                      f"{np.mean([r['on_segments_dropped'] > 0 for r in rr]):>8.2f}{sum(r['on_segments_dropped'] for r in rr):>5d}"
                      f"{sum(bool(r['off_failures']) for r in rr):>9d}{sum(r['neg_eig_steps'] for r in rr):>8d}"
                      f"{sum(r['jitter_fallbacks'] for r in rr):>5d}{min(r['min_eig_overall'] for r in rr):>10.1e}"
                      f"{(_mm(nis) if nis else 'none'):>20}{_mm([r['wall_s'] for r in rr], '{:.1f}'):>19}")

    print("\n=== 4. fitted noise parameters (A and B share them): s2 / channel variance and tau, median [min,max] over the 60 channel fits per variant ===")
    for v, _ in VARIANTS:
        rr = [r for r in rows if r["arm"] == "A" and r["variant"] == v and r["q"] == QS[0]]
        share = [s / var for r in rr for s, var in zip(r["est_s2"], r["chan_var"])]
        tau = [t for r in rr for t in r["est_tau"]]
        print(f"  {v:<4} share {_mm(share, '{:.3f}')}   tau {_mm(tau, '{:.3f}')} s")

    print("\n=== 5. pass / fail against the fixed criteria ===")
    print("  c1 gain <= 15% at L2-L4 (smooth AND filt) | c2 nulls 6/6 within delta (both estimators) | c3 m contraction >= 0.5 at L2-L4 | "
          "c4 rule ON: >= 5 of 6 undiverged in every cell | c5 no NaN/PD failure (flag off)")
    print(f"  {'arm':<9}{'run':<10}{'c1':>4}{'c2':>4}{'c3':>4}{'c4':>4}{'c5':>4}   {'all':>4}")
    verdict = {}
    for arm in ARMS:
        for v, q in combos:
            if v == "S0":
                continue
            c1 = all(_level_err([r for r in _cell_rows(rows, arm, v, q, c) if "g12_filt" in r], k) <= CRITERION
                     for c in CELLS[1:4] for k in (("g12", "g21"), ("g12_filt", "g21_filt"))
                     if [r for r in _cell_rows(rows, arm, v, q, c) if "g12_filt" in r])
            n_est = all(len([r for r in _cell_rows(rows, arm, v, q, c) if "g12_filt" in r]) == N_SERIES for c in CELLS)
            c2 = all(_null_ok([r for r in _cell_rows(rows, arm, v, q, "null") if "g12_filt" in r], ks) for ks in (("g12", "g21"), ("g12_filt", "g21_filt"))) \
                and len([r for r in _cell_rows(rows, arm, v, q, "null") if "g12_filt" in r]) == N_SERIES
            c3 = all(np.median([r["contraction"]["m"] for r in _cell_rows(rows, arm, v, q, c) if "g12" in r] or [0.0]) >= CONTRACTION_MIN for c in CELLS[1:4])
            c4 = all(sum(not r["on_recording_diverged"] for r in _cell_rows(rows, arm, v, q, c)) >= 5 for c in CELLS)
            c5 = not any(r["off_failures"] or r["nan_inf"] for c in CELLS for r in _cell_rows(rows, arm, v, q, c))
            c1 = c1 and n_est
            verdict[(arm, v, q)] = c1 and c2 and c3 and c4 and c5
            print(f"  {ARM_NAME[arm]:<9}{v + ' q=' + format(q, 'g'):<10}" + "".join(f"{'Y' if x else 'n':>4}" for x in (c1, c2, c3, c4, c5)) + f"   {'PASS' if verdict[(arm, v, q)] else 'fail':>4}")

    print("\n  S0 (candidate vs the same-seed control, same q): level error <= control + 2 pp at L1-L4 (smooth and filt), nulls inside delta, undiverged count per cell >= control's")
    s0 = {}
    for arm in ARMS[1:]:
        for q in QS:
            ok = True
            for c in CELLS[:4]:
                for ks in (("g12", "g21"), ("g12_filt", "g21_filt")):
                    a = [r for r in _cell_rows(rows, arm, "S0", q, c) if "g12_filt" in r]
                    b = [r for r in _cell_rows(rows, "N", "S0", q, c) if "g12_filt" in r]
                    if not a or not b or _level_err(a, ks) > _level_err(b, ks) + S0_MARGIN:
                        ok = False
            for ks in (("g12", "g21"), ("g12_filt", "g21_filt")):
                a = [r for r in _cell_rows(rows, arm, "S0", q, "null") if "g12_filt" in r]
                if len(a) != N_SERIES or not _null_ok(a, ks):
                    ok = False
            for c in CELLS:
                nu = lambda ar: sum(not r["on_recording_diverged"] for r in _cell_rows(rows, ar, "S0", q, c))
                if nu(arm) < nu("N"):
                    ok = False
            s0[(arm, q)] = ok
            print(f"    {ARM_NAME[arm]:<9}S0 q={q:g}: {'ok' if ok else 'WORSE'}")

    print("\n=== 6. viability (some q with c1-c5 for S1 and S2 and the S0 condition at that q) ===")
    for arm in ARMS[1:]:
        qs_ok = [q for q in QS if verdict.get((arm, "S1", q)) and verdict.get((arm, "S2", q)) and s0[(arm, q)]]
        print(f"  {ARM_NAME[arm]:<9}" + (f"VIABLE at q = {qs_ok}" if qs_ok else "not viable"))
    print("  control (19-D) for reference: " + ", ".join(f"{v} q={q:g} {'PASS' if verdict[('N', v, q)] else 'fail'}" for v in ("S1", "S2") for q in QS))
    fails = [(r["arm"], r["variant"], r["q"], r["cell"], r["series"], r["off_failures"]) for r in rows if r["off_failures"]]
    print("\nNaN/Inf or covariance-not-PD failures (flag disabled): " + ("none" if not fails else f"{len(fails)} runs, first: {fails[0]}"))


def main():
    from src.config import load_config
    from joblib import Parallel, delayed
    mode, out_path = sys.argv[1], Path(sys.argv[2])
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    cfg = load_config()
    jobs = [(c, s, cfg) for c in CELLS for s in range(N_SERIES)]
    t0 = time.perf_counter()
    if mode == "estimator":
        res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_estimator_worker)(j) for j in jobs)
        rows = [r for rr in res for r in rr]
        out_path.write_text(json.dumps(rows))
        print(f"wall {time.perf_counter() - t0:.0f} s")
        print_estimator(rows, cfg)
        return
    print(f"{len(jobs)} (cell, series) jobs x {len(VARIANTS)} variants x {len(ARMS)} arms x {len(QS)} q, {n_jobs} workers", flush=True)
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_worker)(j) for j in jobs)
    rows = [r for rr in res for r in rr]
    out_path.write_text(json.dumps(rows))
    print(f"wall {time.perf_counter() - t0:.0f} s, {len(rows)} runs", flush=True)
    report(rows, cfg)


if __name__ == "__main__":
    main()
