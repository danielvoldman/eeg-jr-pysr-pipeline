"""Stage D2 smoke checks for PySR on this machine (PLAN.md D3; §8.2, §16.2, §16.5). Synthetic rows only.

One check per process, so each one starts from a cold interpreter:
    .venv\\Scripts\\python.exe tools\\d2_smoke.py <check> [args]
Checks: order_a (numpy and numba imported BEFORE PySR), order_b (PySR before numpy and numba), timeout
(30 s timeout overshoot), smoke (500 rows, y = x0 x1 + noise), precision (float32 vs float64 speed),
affinity_worker (run under `cmd /c start /affinity MASK /wait`, prints the Julia thread timing),
determinism (serial, fixed iterations, turbo off twice and on/off on three seeds).

`import main` comes first: it sets PYTHON_JULIACALL_THREADS from config before NumPy, MNE or PySR load
(§16.2). Nothing here is a result; no real recording, no test subject.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
T_START = time.time()
import main  # noqa: E402,F401  (sets the thread variables first)
from src.config import load_config  # noqa: E402

CFG = load_config()
NAMES3 = ["u_tgt", "u_src", "S_src"]


def product_data(n, seed, noise=0.05, tanh_term=False):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * X[:, 1] + (0.7 * np.tanh(X[:, 2]) if tanh_term else 0.0)
    return X, y + noise * rng.normal(size=n)


def tiny_fit(R, n=300, niterations=3, timeout_s=3600.0):
    X, y = product_data(n, 0)
    return R.fit_model(CFG, X, y, "synthetic", random_state=1, names=NAMES3, niterations=niterations,
                       timeout_s=timeout_s)


def report(d):
    print("RESULT " + json.dumps(d))


def check_order_a():
    t0 = time.time()
    import numpy as np
    import numba
    from numba import njit

    @njit(fastmath=False)
    def f(x):
        return np.sum(np.sin(x))
    f(np.arange(5.0))
    t1 = time.time()
    from src import regression as R
    R.load_pysr(CFG)
    t2 = time.time()
    model_, secs = tiny_fit(R)
    report({"check": "order_a_numpy_numba_first", "ok": True, "numpy_numba_s": round(t1 - t0, 1),
            "pysr_load_s": round(t2 - t1, 1), "tiny_fit_s": round(secs, 1), "n_equations": len(model_.equations_)})


def check_order_b():
    os.environ.setdefault("JULIA_DEPOT_PATH", str(REPO / CFG["pysr"]["julia_depot_dir"]))
    t0 = time.time()
    import pysr  # noqa: F401
    from juliacall import Main as jl
    t1 = time.time()
    import numpy as np
    from numba import njit

    @njit(fastmath=False)
    def f(x):
        return np.sum(np.sin(x))
    f(np.arange(5.0))
    from src import regression as R
    R.load_pysr(CFG)
    model_, secs = tiny_fit(R)
    report({"check": "order_b_pysr_first", "ok": True, "pysr_load_s": round(t1 - t0, 1),
            "threads": int(jl.seval("Threads.nthreads()")), "tiny_fit_s": round(secs, 1),
            "n_equations": len(model_.equations_)})


def check_timeout():
    from src import regression as R
    R.load_pysr(CFG)
    _, warm = tiny_fit(R)                                   # JIT of the first fit, reported separately
    X, y = product_data(5000, 3, tanh_term=True)
    out = []
    for t_s in (30.0,):
        model_, secs = R.fit_model(CFG, X, y, "synthetic", random_state=2, names=NAMES3, timeout_s=t_s)
        out.append({"timeout_s": t_s, "wall_s": round(secs, 2), "overshoot_s": round(secs - t_s, 2)})
    report({"check": "timeout_overshoot", "first_fit_s": round(warm, 1), "runs": out})


def check_smoke():
    import numpy as np
    from src import regression as R
    pysr = R.load_pysr(CFG)
    _, warm = tiny_fit(R)
    X, y = product_data(500, 5, noise=0.05)
    kw = R.model_kwargs(CFG, "synthetic", random_state=3, timeout_s=60.0)
    m = pysr.PySRRegressor(**kw)
    populations = m.get_params()["populations"]
    t0 = time.time()
    m.fit(X, y, variable_names=NAMES3)
    secs = time.time() - t0
    eq = m.equations_
    front = R.pareto_filter([R.FrontEntry(int(r.complexity), str(r.equation), float(r.loss)) for r in eq.itertuples()])
    sigs = {e.complexity: sorted(R.term_signatures(e.equation)) for e in front}
    product = any(s == ["u_src*u_tgt"] for s in sigs.values())
    # the lambdify evaluation must reproduce PySR's own loss on the fit data
    e0 = eq.iloc[-1]
    mine = float(np.mean((R.evaluate_equation(e0.equation, X, NAMES3) - y) ** 2))
    report({"check": "smoke_fit", "first_fit_s": round(warm, 1), "fit_s": round(secs, 1),
            "populations_default": populations, "populations_ge_16": bool(populations >= 16),
            "precision": kw["precision"], "batching": kw["batching"], "product_on_front": bool(product),
            "front": [[e.complexity, e.equation, round(e.loss, 5)] for e in front],
            "pysr_loss_last": float(e0.loss), "lambdify_loss_last": mine})


def check_precision():
    from src import regression as R
    R.load_pysr(CFG)
    X, y = product_data(5000, 7, tanh_term=True)
    res = {}
    for prec in (64, 32):
        kw = dict(random_state=4, names=NAMES3, niterations=20)
        import copy
        c = copy.deepcopy(CFG)
        c["pysr"]["precision"] = prec
        _, first = R.fit_model(c, X, y, "synthetic", **kw)
        warm = [R.fit_model(c, X, y, "synthetic", **kw)[1] for _ in range(2)]
        res[prec] = {"first_fit_s": round(first, 1), "warm_fit_s": [round(w, 2) for w in warm]}
    w64, w32 = min(res[64]["warm_fit_s"]), min(res[32]["warm_fit_s"])
    report({"check": "precision_cost", "niterations": 20, "rows": 5000, "runs": res,
            "float64_over_float32": round(w64 / w32, 2)})


def check_affinity_worker(mask, out):
    import psutil
    os.environ.setdefault("JULIA_DEPOT_PATH", str(REPO / CFG["pysr"]["julia_depot_dir"]))
    import pysr  # noqa: F401  (before juliacall: PySR sets the juliacall defaults)
    from juliacall import Main as jl
    jl.seval("""
    function busy(n)
        s = 0.0
        for i in 1:n
            s += sin(i * 1e-3)
        end
        return s
    end
    """)
    jl.seval("busy(1000000)")
    t = float(jl.seval("@elapsed Threads.@threads for k in 1:8; busy(40_000_000); end"))
    d = {"check": "affinity", "mask": mask, "python_affinity": sorted(psutil.Process().cpu_affinity()),
         "julia_threads": int(jl.seval("Threads.nthreads()")), "eight_thread_wall_s": round(t, 2)}
    Path(out).write_text(json.dumps(d), encoding="utf-8")
    report(d)


def check_determinism():
    from src import regression as R
    R.load_pysr(CFG)
    dc = CFG["pysr"]["determinism_check"]
    X, y = product_data(dc["n_rows"], 21, noise=0.05, tanh_term=True)
    names = NAMES3

    def run(seed, turbo):
        return R.fit_model(CFG, X, y, "synthetic", random_state=seed, names=names, serial=True, turbo=turbo,
                           niterations=dc["niterations"])
    out = {"check": "determinism"}
    _, out["first_fit_turbo_off_s"] = run(dc["seeds"][0], False)
    try:
        _, out["first_fit_turbo_on_s"] = run(dc["seeds"][0], True)
    except Exception as exc:                                 # turbo needs LoopVectorization in Julia
        out["turbo_error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        report(out)
        return 2
    a, ta = run(dc["seeds"][0], False)
    b, tb = run(dc["seeds"][0], False)
    out["off_twice_bitwise_identical"] = bool(R.fronts_identical(a.equations_, b.equations_))
    out["off_times_s"] = [round(ta, 2), round(tb, 2)]
    if not out["off_twice_bitwise_identical"]:
        report(out)
        return 3
    rows = []
    for seed in dc["seeds"]:
        off, t_off = run(seed, False)
        on, t_on = run(seed, True)
        rows.append({"seed": seed, "t_off_s": round(t_off, 2), "t_on_s": round(t_on, 2),
                     "speedup": round(t_off / t_on, 3),
                     "equivalent": bool(R.fronts_equivalent(off.equations_, on.equations_, CFG)),
                     "bitwise": bool(R.fronts_identical(off.equations_, on.equations_)),
                     "n_entries": [len(off.equations_), len(on.equations_)],
                     "best_off": str(off.equations_.iloc[-1].equation), "best_on": str(on.equations_.iloc[-1].equation)})
    out["turbo_vs_off"] = rows
    out["all_equivalent"] = all(r["equivalent"] for r in rows)
    report(out)
    return 0


if __name__ == "__main__":
    which = sys.argv[1]
    rc = 0
    if which == "order_a":
        check_order_a()
    elif which == "order_b":
        check_order_b()
    elif which == "timeout":
        check_timeout()
    elif which == "smoke":
        check_smoke()
    elif which == "precision":
        check_precision()
    elif which == "affinity_worker":
        check_affinity_worker(sys.argv[2], sys.argv[3])
    elif which == "determinism":
        rc = check_determinism()
    else:
        raise SystemExit(f"unknown check {which}")
    print(f"process_seconds {time.time() - T_START:.1f}")
    sys.exit(rc)
