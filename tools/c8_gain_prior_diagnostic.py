"""C8: read-only diagnostic on SYNTHETIC data only (PLAN.md C8; §7.3, §7.5, §7.6, §9.1 to §9.3). Is the residual gain attenuation
at q = 1e-3 (arms control 19-D, A_ou and B_walk 21-D; variants S0, S1, S2) caused by the gain prior N(0, (0.1 C2)^2) rather than
by the 1/f background? Changes no src rule, threshold or config value; writes nothing under outputs/. Reuses the C7 series,
seeds (C7: SEED_SIM 60000, SEED_1F 70000, SEED_M 80000), estimator, kernels and the unchanged src/passes.py.

The gain prior SD (coupling.gain_prior_sd_factor_of_C2) enters four places: the initial covariance P0 and the per-segment carry
reset (ss.prior_cov), the gain random-walk Q (ukf.process_noise: factor x prior variance x dt), the gap inflation in
passes.run_pass1 (factor x prior variance x gap) and the reference-refresh threshold (ss.parameter_prior_sd). Two ways to
widen it, both inside this script only:
  K  config copy with the factor x k: all four places move together (what a real config change would do).
  P  prior only: ss.prior_cov is wrapped (gain entries x k^2, so P0, the carry reset and the gap inflation widen) and
     ukf.process_noise is wrapped to use the UNWIDENED prior (the gain random-walk Q stays at the reference); the refresh
     threshold (ss.parameter_prior_sd) is untouched. The prior mean stays 0.
Settings: x1 (reference), K3, K10, P3, P10.

Fixed criteria (agreed before any run), per (arm, S1|S2, setting) at q = 1e-3:
  c1 median relative gain error <= 15% at L2, L3, L4 (pooled g12 and g21), smoothed AND filtered-gain estimator
  c2 all 6 null series have both |g| < delta = 1.08, both estimators
  c3 median m contraction >= 0.5 at L2, L3, L4
  c4 with the standard rule ON at least 5 of 6 series undiverged (recording-level flag), in every cell
  c5 no NaN/Inf or covariance-not-PD failure with the flag disabled
Reading rule: if c1 passes only where c2 fails, the prior is a trade-off, not a fix.

    python tools/c8_gain_prior_diagnostic.py time <out.json>         # one series (L3, series 0) per arm, all variants/settings
    python tools/c8_gain_prior_diagnostic.py run <out.json> [n_jobs] # everything
    python tools/c8_gain_prior_diagnostic.py report <out.json>       # tables again from a saved run
"""
import contextlib
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

import c6_gain_recovery_diagnostic as C6  # noqa: E402
import c7_candidate_comparison as C7  # noqa: E402

CELLS = C6.CELLS
VARIANTS = C7.VARIANTS
ARMS = ("N", "A", "B")
ARM_NAME = C7.ARM_NAME
Q = 1e-3
SETTINGS = (("x1", "K", 1.0), ("K3", "K", 3.0), ("K10", "K", 10.0), ("P3", "P", 3.0), ("P10", "P", 10.0))
N_SERIES = C7.N_SERIES
CRITERION, CONTRACTION_MIN = C7.CRITERION, C7.CONTRACTION_MIN
X1_TOL = 1e-6             # x1 against C7's q = 1e-3 rows: largest allowed absolute difference in any estimate
_warm = {"done": False}


def scaled_cfg(cfg, mode, k):
    """Config copy for setting (mode, k): K scales the factor; P keeps it (the widening is applied by widen_prior_only)."""
    out = copy.deepcopy(cfg)
    if mode == "K":
        out["coupling"]["gain_prior_sd_factor_of_C2"] = cfg["coupling"]["gain_prior_sd_factor_of_C2"] * k
    return out


@contextlib.contextmanager
def widen_prior_only(k):
    """Variant P: ss.prior_cov gain entries x k^2 (P0, carry reset, gap inflation); ukf.process_noise sees the original prior
    (gain random-walk Q unchanged); ss.parameter_prior_sd (refresh threshold) untouched."""
    from src import state_space as ss
    from src import ukf
    orig_cov, orig_pn = ss.prior_cov, ukf.process_noise

    def wide(layout, cfg):
        P = orig_cov(layout, cfg)
        for i, name in enumerate(layout.names):
            if name in ("g12", "g21"):
                P[i, i] *= k ** 2
        return P

    def pn(layout, cfg, q):
        ss.prior_cov = orig_cov
        try:
            return orig_pn(layout, cfg, q)
        finally:
            ss.prior_cov = wide

    ss.prior_cov, ukf.process_noise = wide, pn
    try:
        yield
    finally:
        ss.prior_cov, ukf.process_noise = orig_cov, orig_pn


def setting_context(mode, k):
    return widen_prior_only(k) if mode == "P" else contextlib.nullcontext()


def _worker(job):
    """One (cell, series, arms): all variants and settings at q = 1e-3."""
    from threadpoolctl import threadpool_limits
    from ext_ukf import Spec
    from src import model
    cell, si, cfg, arms = job
    c2 = model.constants(cfg)["C2"]
    prior_sd = {"g12": cfg["coupling"]["gain_prior_sd_factor_of_C2"] * c2, "g21": cfg["coupling"]["gain_prior_sd_factor_of_C2"] * c2,
                "m": cfg["priors"]["m_sd"]}
    out = []
    with threadpool_limits(limits=1):
        seed, g, m, data = C7._series_for(cfg, cell, si)
        cfg_off0 = copy.deepcopy(cfg)
        cfg_off0["ukf"]["divergence"]["state_sd_multiple"] = C6.STATE_FLAG_OFF
        if not _warm["done"]:
            w = C6._sim_data().make_recording(cfg, [8.0], 1, g12=1.0, g21=1.0)
            for kind in arms:
                C7._pass1(Spec(kind, s2=(1.0, 1.0), tau=(0.1, 0.1)) if kind in ("A", "B") else Spec(kind), w["segments"], w["starts"], cfg_off0, Q, False)
            _warm["done"] = True
        for vname, _ in VARIANTS:
            segs, starts = data[vname]
            est = C7._estimate(cfg, segs)
            for arm in arms:
                spec = C7._spec(arm, est)
                for sname, mode, k in SETTINGS:
                    cfg_s = scaled_cfg(cfg, mode, k)
                    cfg_off = copy.deepcopy(cfg_s)
                    cfg_off["ukf"]["divergence"]["state_sd_multiple"] = C6.STATE_FLAG_OFF
                    with setting_context(mode, k):
                        t0 = time.perf_counter()
                        res, pf = C7._pass1(spec, segs, starts, cfg_off, Q, False)
                        wall = time.perf_counter() - t0
                        on, _ = C7._pass1(spec, segs, starts, cfg_s, Q, True)
                    mons = pf.monitors
                    nis_n = sum(c for _, c in pf.nis)
                    rec = {"cell": cell, "series": si, "seed": seed, "variant": vname, "q": Q, "arm": arm, "setting": sname,
                           "g_true": g, "m_true": m, "wall_s": wall,
                           "off_failures": [f"seg {s.index}: {s.reason} at step {s.step}" for s in res.segments if s.diverged],
                           "min_eig_overall": min(mm["min_eig_overall"] for mm in mons),
                           "neg_eig_steps": int(sum(mm["n_negative_eig_steps"] for mm in mons)),
                           "jitter_fallbacks": int(sum(mm["n_jitter_fallbacks"] for mm in mons)),
                           "nan_inf": bool(any(mm["nan_inf_seen"] for mm in mons)),
                           "nis_mean": (sum(s for s, _ in pf.nis) / nis_n) if nis_n else None,
                           "on_segments_dropped": int(sum(s.diverged for s in on.segments)),
                           "on_recording_diverged": bool(on.recording_diverged)}
                    if res.params is not None:
                        p = res.params
                        rec.update({"g12": p.g12, "g21": p.g21, "m": p.m,
                                    "post_sd_g12": p.posterior_sd["g12"], "post_sd_g21": p.posterior_sd["g21"],
                                    "contraction": {kk: 1.0 - p.posterior_sd[kk] / prior_sd[kk] for kk in ("g12", "g21", "m")}})
                    if res.gain_estimate is not None:
                        rec.update({"g12_filt": res.gain_estimate["g12"], "g21_filt": res.gain_estimate["g21"]})
                    out.append(rec)
    return out


# ---- reporting ---------------------------------------------------------------------------------------------------

def _rows(rows, arm, v, s, c):
    return [r for r in rows if (r["arm"], r["variant"], r["setting"], r["cell"]) == (arm, v, s, c)]


def _est(rr):
    return [r for r in rr if "g12_filt" in r]


def _lvl(rr, keys):
    return float(np.median([abs(r[k] - r["g_true"]) / r["g_true"] for r in rr for k in keys]))


def _nullok(rr, keys):
    return all(abs(r[k]) < C6.DELTA for r in rr for k in keys)


def criteria(rows, arm, v, s):
    """(c1..c5 booleans, first failing name or None)."""
    c1 = True
    for c in CELLS[1:4]:
        rr = _est(_rows(rows, arm, v, s, c))
        if len(rr) != N_SERIES or _lvl(rr, ("g12", "g21")) > CRITERION or _lvl(rr, ("g12_filt", "g21_filt")) > CRITERION:
            c1 = False
    nu = _est(_rows(rows, arm, v, s, "null"))
    c2 = len(nu) == N_SERIES and _nullok(nu, ("g12", "g21")) and _nullok(nu, ("g12_filt", "g21_filt"))
    c3 = all(np.median([r["contraction"]["m"] for r in _est(_rows(rows, arm, v, s, c))] or [0.0]) >= CONTRACTION_MIN for c in CELLS[1:4])
    c4 = all(sum(not r["on_recording_diverged"] for r in _rows(rows, arm, v, s, c)) >= 5 for c in CELLS)
    c5 = not any(r["off_failures"] or r["nan_inf"] for c in CELLS for r in _rows(rows, arm, v, s, c))
    flags = (c1, c2, c3, c4, c5)
    first = next((f"c{i + 1}" for i, f in enumerate(flags) if not f), None)
    return flags, first


def x1_check(rows, c7_path):
    """x1 rows against C7's q = 1e-3 rows (same seeds): largest absolute difference per field, and count of differing flags."""
    c7 = {(r["arm"], r["variant"], r["cell"], r["series"]): r for r in json.loads(Path(c7_path).read_text()) if r["q"] == 1e-3}
    worst, n, flagdiff = 0.0, 0, 0
    for r in rows:
        if r["setting"] != "x1":
            continue
        o = c7.get((r["arm"], r["variant"], r["cell"], r["series"]))
        if o is None or "g12" not in r or "g12" not in o:
            continue
        n += 1
        worst = max(worst, *(abs(r[k] - o[k]) for k in ("g12", "g21", "m", "g12_filt", "g21_filt")))
        flagdiff += int(r["on_recording_diverged"] != o["on_recording_diverged"])
    return n, worst, flagdiff


def report(rows, cfg, c7_path=None):
    C6.DELTA = cfg["g0"]["pass"]["null_delta_fraction_of_weakest_level"] * C6._cell_truth(cfg, "L1")
    cells = sorted({r["cell"] for r in rows}, key=CELLS.index)
    full = len(cells) == len(CELLS)
    print(f"\nC8 gain-prior diagnostic, q = {Q:g}, truth g: " + ", ".join(f"{c} {C6._cell_truth(cfg, c):.2f}" for c in CELLS) +
          f"; delta = {C6.DELTA:.2f}; gain prior SD x1 = {cfg['coupling']['gain_prior_sd_factor_of_C2'] * 108.0:.1f}; settings "
          + ", ".join(s for s, _, _ in SETTINGS))
    if c7_path and Path(c7_path).exists():
        n, worst, fd = x1_check(rows, c7_path)
        print(f"x1 against C7 q=1e-3 rows (same seeds): {n} runs compared, largest |difference| in g12/g21/m/filtered gains {worst:.2e}, "
              f"recording-divergence flag differs in {fd}")
    print("\nper (arm, variant, setting):  err% = level median relative error, smoothed/filtered, L1 L2 L3 L4;  null = series with both"
          " |g| < delta (both estimators), of 6;  ratio = median estimated/true gain, smoothed, L2 L3 L4;  sdg = median posterior SD"
          " of g (1/s; prior SD in the row label), L2 L3 L4;  contr m = median m contraction L2 L3 L4;  undiv = min over cells of"
          " undiverged series (rule ON);  NIS = median of the mean NIS")
    print(f"  {'arm':<8}{'var':<4}{'set':<5}{'err% smoothed L1-L4':>26}{'err% filtered L1-L4':>26}{'null':>6}{'ratio L2-L4':>16}"
          f"{'sdg L2-L4':>17}{'contr m L2-L4':>17}{'undiv':>6}{'NIS':>6}")
    for arm in ARMS:
        for v, _ in VARIANTS:
            for s, _, _ in SETTINGS:
                lv_s, lv_f, ratio, sdg, cm, nis = [], [], [], [], [], []
                for c in CELLS[:4]:
                    rr = _est(_rows(rows, arm, v, s, c))
                    lv_s.append(f"{100 * _lvl(rr, ('g12', 'g21')):.0f}" if rr else "-")
                    lv_f.append(f"{100 * _lvl(rr, ('g12_filt', 'g21_filt')):.0f}" if rr else "-")
                for c in CELLS[1:4]:
                    rr = _est(_rows(rows, arm, v, s, c))
                    if rr:
                        ratio.append(f"{np.median([r[k] / r['g_true'] for r in rr for k in ('g12', 'g21')]):.2f}")
                        sdg.append(f"{np.median([r[k] for r in rr for k in ('post_sd_g12', 'post_sd_g21')]):.1f}")
                        cm.append(f"{np.median([r['contraction']['m'] for r in rr]):.2f}")
                    else:
                        ratio.append("-"); sdg.append("-"); cm.append("-")
                nu = _est(_rows(rows, arm, v, s, "null"))
                nok = sum(abs(r["g12"]) < C6.DELTA and abs(r["g21"]) < C6.DELTA and abs(r["g12_filt"]) < C6.DELTA and abs(r["g21_filt"]) < C6.DELTA
                          for r in nu)
                und = min(sum(not r["on_recording_diverged"] for r in _rows(rows, arm, v, s, c)) for c in cells)
                nis = [r["nis_mean"] for c in cells for r in _rows(rows, arm, v, s, c) if r["nis_mean"] is not None]
                print(f"  {ARM_NAME[arm]:<8}{v:<4}{s:<5}{' '.join(lv_s):>26}{' '.join(lv_f):>26}{nok:>4}/{len(nu)}{' '.join(ratio):>16}"
                      f"{' '.join(sdg):>17}{' '.join(cm):>17}{und:>6}{(np.median(nis) if nis else float('nan')):>6.2f}")
    if not full:
        print("\n(partial run: criteria not evaluated)")
        return
    print("\ncriteria: c1 gain <=15% at L2-L4 (smoothed AND filtered) | c2 nulls 6/6 (both estimators) | c3 m contraction >= 0.5 at L2-L4 | "
          "c4 rule ON >= 5/6 undiverged in every cell | c5 no NaN/PD failure (flag off)")
    print(f"  {'arm':<8}{'var':<4}{'set':<5}{'c1':>4}{'c2':>4}{'c3':>4}{'c4':>4}{'c5':>4}   first failing")
    trade = []
    for arm in ARMS:
        for v in ("S1", "S2"):
            for s, _, _ in SETTINGS:
                fl, first = criteria(rows, arm, v, s)
                print(f"  {ARM_NAME[arm]:<8}{v:<4}{s:<5}" + "".join(f"{'Y' if x else 'n':>4}" for x in fl) + f"   {first or 'PASS'}")
                if fl[0] and not fl[1]:
                    trade.append((ARM_NAME[arm], v, s))
    print("\nc1 passes only where c2 fails ('the prior is a trade-off, not a fix'): " + (", ".join("/".join(t) for t in trade) if trade else "no setting"))
    print("\nstability (flag disabled, all settings): " + ", ".join(
        f"{ARM_NAME[a]} {s}: negative-eig steps {sum(r['neg_eig_steps'] for r in rows if r['arm'] == a and r['setting'] == s)}, "
        f"jitter {sum(r['jitter_fallbacks'] for r in rows if r['arm'] == a and r['setting'] == s)}, "
        f"failures {sum(bool(r['off_failures']) or r['nan_inf'] for r in rows if r['arm'] == a and r['setting'] == s)}, "
        f"min eig {min(r['min_eig_overall'] for r in rows if r['arm'] == a and r['setting'] == s):.1e}"
        for a in ARMS for s, _, _ in SETTINGS))


def main():
    from src.config import load_config
    from joblib import Parallel, delayed
    mode, out_path = sys.argv[1], Path(sys.argv[2])
    cfg = load_config()
    c7_path = os.environ.get("C8_C7_JSON")
    if mode == "report":
        report(json.loads(out_path.read_text()), cfg, c7_path)
        return
    t0 = time.perf_counter()
    if mode == "time":
        jobs = [("L3", 0, cfg, (a,)) for a in ARMS]
        n_jobs = len(jobs)
    else:
        n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        jobs = [(c, s, cfg, ARMS) for c in CELLS for s in range(N_SERIES)]
    print(f"{len(jobs)} jobs, {n_jobs} workers, settings {[s for s, _, _ in SETTINGS]}", flush=True)
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_worker)(j) for j in jobs)
    rows = [r for rr in res for r in rr]
    out_path.write_text(json.dumps(rows))
    wall = time.perf_counter() - t0
    print(f"wall {wall:.0f} s, {len(rows)} runs", flush=True)
    if mode == "time":
        for a in ARMS:
            ra = [r for r in rows if r["arm"] == a]
            print(f"  {ARM_NAME[a]}: sum of flag-off pass-1 times {sum(r['wall_s'] for r in ra):.1f} s over {len(ra)} runs "
                  f"({np.mean([r['wall_s'] for r in ra]):.1f} s each); job wall incl. simulation and rule-ON pass is the batch wall above")
        print(f"  estimate for 30 series x 3 arms on 4 workers: {wall * 90 / 4 / 60:.0f} min (this batch ran 3 jobs in parallel, each 1 arm, "
              f"so 90 (series, arm) units / 4 workers)")
        print(f"  x1 check: {x1_check(rows, c7_path) if c7_path else 'no C7 json'}")
        return
    report(rows, cfg, c7_path)


if __name__ == "__main__":
    main()
