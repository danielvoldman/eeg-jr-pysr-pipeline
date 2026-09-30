"""Stage D1 read-only report: literal (L) versus matched (M) subtraction, and PySR-free recovery (§8.3, §9.1).

Nothing here is a result or a decision. Synthetic only (tests/planted_sim.py: true states, no filter, no
observation noise, the planted product residual of §9.1 at 50% of the base coupling RMS); no real recording,
no test subject. Run with .venv\\Scripts\\python.exe tools\\d1_estimator_report.py.

Part A: noise-free analytic signals. y4 = sin(2 pi f t + 0.3) with the EXACT derivative as the 'base
prediction' (the true residual is zero), so each row is the spurious residual an estimator leaves, in units of
the RMS of dy4/dt (L: pointwise subtraction; M: matched).
Part B: the simulator. NRMSE of the recovered planted term (§9.1 definition; recovery by OLS of the residual on
the planted basis, both evaluated on the true inputs), per support and form, and for the TV grid (literal).
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from src.config import load_config  # noqa: E402
from src import regression as R  # noqa: E402
import planted_sim as ps  # noqa: E402

FREQS = (2.0, 4.0, 10.0, 20.0)
LEVELS = (0.02, 0.05, 0.10, 0.25)             # x C2, section 9.1
SEEDS_PER_LEVEL = 3
SIM_SECONDS = 60.0
SEED_BASE = 60000


def part_a(cfg):
    fs = cfg["preprocessing"]["observation_fs_hz"]
    n = int(cfg["windows"]["training_window_s"] * fs) - int(cfg["windows"]["training_burn_in_s"] * fs)
    t = np.arange(n) / fs
    print("A. spurious residual / RMS(dy4/dt), analytic sinusoids, true residual = 0")
    print("support  f(Hz)   L (literal)   M (matched)")
    for sup in cfg["residual"]["weak_support_ms_pilot_compare"]:
        k = R.weak_kernel(sup, cfg)
        for f in FREQS:
            y = np.sin(2 * np.pi * f * t + 0.3)
            yd = 2 * np.pi * f * np.cos(2 * np.pi * f * t + 0.3)
            est = R.weak_derivative(y, k)
            lit = est - yd[k.K:n - k.K]
            mat = est - R.weak_smooth(yd, k)
            rms = np.sqrt(np.mean(yd ** 2))
            print(f"{sup:5d} ms {f:5.0f}   {np.sqrt(np.mean(lit ** 2)) / rms:11.4f}   "
                  f"{np.sqrt(np.mean(mat ** 2)) / rms:11.5f}")


def collect(cfg, sim, estimator, margin, **kw):
    fs = cfg["preprocessing"]["observation_fs_hz"]
    burn = int(round(cfg["windows"]["training_burn_in_s"] * fs))
    parts, bases, truths = [], [], []
    for start, xs, sd_ in ps.windows(sim, cfg):
        rows = R.window_rows(xs, sd_, sim.params, cfg, estimator=estimator, margin=margin,
                             window_start=start, subject="sim", **kw)
        b, tr = ps.window_truth(sim, start, xs.shape[0], burn, margin)
        parts.append(rows)
        bases.append(b)
        truths.append(tr)
    return R.Rows.concat(parts), np.concatenate(bases), np.concatenate(truths)


def part_b(cfg):
    jr = cfg["jansen_rit"]
    C2 = jr["C"] * jr["C2_multiplier"]
    common = max(R.weak_kernel(s, cfg).K for s in cfg["residual"]["weak_support_ms_pilot_compare"])
    supports = cfg["residual"]["weak_support_ms_pilot_compare"]
    grid = cfg["residual"]["tv_weight_grid"]
    res = {}
    t_tv = 0.0
    for li, lev in enumerate(LEVELS):
        g = lev * C2
        for s_i in range(SEEDS_PER_LEVEL):
            seed = SEED_BASE + 10 * li + s_i
            sim = ps.simulate_planted(cfg, seed, g, g, SIM_SECONDS)
            for sup in supports:
                for form in ("matched", "literal"):
                    rows, b, tr = collect(cfg, sim, "weak_form", common, support_ms=sup, form=form)
                    nr, chat = R.recovery_nrmse(rows.y, b, tr)
                    res.setdefault(("weak", sup, form, lev), []).append((nr, chat / (sim.planted[1100, 0] / sim.basis[1100, 0])))
            if s_i == 0:                                          # TV grid on one seed per level (cost)
                for a in grid:
                    t0 = time.time()
                    rows, b, tr = collect(cfg, sim, "total_variation", common, tv_alpha_rel=a)
                    t_tv += time.time() - t0
                    nr, chat = R.recovery_nrmse(rows.y, b, tr)
                    res.setdefault(("tv", a, "literal", lev), []).append((nr, chat / (sim.planted[1100, 0] / sim.basis[1100, 0])))
    print(f"\nB. PySR-free recovery NRMSE (median over series; coefficient ratio c_hat/c_true), "
          f"common margin {common} samples, {SIM_SECONDS:.0f}-s series, noise-free states, input noise present")
    print("estimator              " + "".join(f"  L{li + 1} g={lv * C2:5.1f}" for li, lv in enumerate(LEVELS)) + "   all-level median")
    keys = sorted({k[:3] for k in res}, key=lambda k: (k[0], k[2] != "matched", float(k[1])))
    for k in keys:
        cells = [res.get(k + (lev,), []) for lev in LEVELS]
        med = [np.median([c[0] for c in cell]) if cell else np.nan for cell in cells]
        allm = np.median([c[0] for cell in cells for c in cell])
        label = f"{k[0]} {k[1]} {k[2]}"
        print(f"{label:22s}" + "".join(f"  {m:11.3f}" for m in med) + f"   {allm:8.3f}")
    print("\ncoefficient ratio c_hat / c_true (median over all series)")
    for k in keys:
        allc = np.median([c[1] for lev in LEVELS for c in res.get(k + (lev,), [])])
        print(f"{k[0]} {k[1]} {k[2]:8s} {allc:8.3f}")
    print(f"\nTV time (rows of {len(LEVELS)} series x {len(grid)} alpha): {t_tv:.0f} s")


if __name__ == "__main__":
    cfg = load_config()
    part_a(cfg)
    t0 = time.time()
    part_b(cfg)
    print(f"total {time.time() - t0:.0f} s")
