"""Stage D2 read-only reports with real PySR fits on SYNTHETIC rows (PLAN.md D3, D5; §8.2, §17).

    .venv\\Scripts\\python.exe tools\\d2_pysr_reports.py recovery
    .venv\\Scripts\\python.exe tools\\d2_pysr_reports.py ensemble [timeout_s] [out.json]

`recovery`: the planted product on clean rows (centred, and the uncentred section 9.1 basis), selected with
the section 8.2 rule and scored on fresh held-out rows (the same code as the pytest marker test).
`ensemble`: the pilot's 3-refit mechanics (draw_ensemble with n_refits = 3, each refit on a random half of
the training subjects, same selection rule, signature recurrence). The subject IDs are the real training
IDs of outputs/split_42.json (IDs only; every row is synthetic) and the split's test IDs are present in the
row table, to show that the guard keeps them out. Nothing here is a result; no real recording is read.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
import main  # noqa: E402,F401  (thread variables before anything heavy)
from src import regression as R  # noqa: E402
from src.config import load_config  # noqa: E402
import test_regression_pysr as T  # noqa: E402

CFG = load_config()


def recovery():
    out = []
    for centred in (True, False):
        t0 = time.time()
        res, nrmse = T.recovery(centred)
        out.append({"basis": "centred product" if centred else "uncentred section 9.1 basis",
                    "equation": res.record["equation"], "complexity": res.record["complexity"],
                    "signatures": res.record["signatures"], "no_term": res.record["no_term"],
                    "heldout_nrmse": round(nrmse, 5), "val_loss": res.record["val_loss"],
                    "min_val_loss": res.record["min_val_loss"], "fit_s": round(res.seconds, 1),
                    "wall_s": round(time.time() - t0, 1), "n_fit_rows": res.n_fit_rows,
                    "n_val_rows": res.n_val_rows,
                    "eligible_complexities": res.selection.eligible,
                    "front_size": len(res.record["front"])})
    print("RESULT " + json.dumps(out, indent=1))


def ensemble(timeout_s, out_path):
    split = json.loads((REPO / "outputs" / "split_42.json").read_text(encoding="utf-8"))
    train, test = split["train"], split["test"]
    rng = np.random.default_rng(777)
    by = {}
    for s in train + test:                                  # synthetic rows for every ID; test IDs stay unused
        n = 1200
        X = rng.normal(size=(n, 3))
        scale = rng.uniform(0.8, 1.2)
        truth = scale * 1.5 * X[:, 0] * X[:, 1] + 0.8 * np.tanh(X[:, 2])
        y = truth + 0.1 * np.std(truth) * rng.normal(size=n)
        by[s] = R.Rows(X, y, np.zeros(n, int), np.zeros(n, np.int64), np.arange(n), np.full(n, s))
    t0 = time.time()
    out = R.run_ensemble(by, train, split, 42, CFG, n_refits=3, timeout_s=timeout_s)
    used = set()
    for r in out["refits"]:
        used |= set(r["fit_subjects"]) | set(r["val_subjects"])
    out["wall_s"] = round(time.time() - t0, 1)
    out["subjects_used_in_test_side"] = sorted(used & set(test))
    out["n_training_subjects"] = len(train)
    Path(out_path).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("RESULT " + json.dumps({
        "wall_s": out["wall_s"], "recurrence": out["recurrence"],
        "refits": [{k: r[k] for k in ("k", "equation", "complexity", "no_term", "signatures", "val_loss",
                                      "fit_seconds", "n_fit_rows", "n_val_rows")} |
                   {"half": len(r["half_subjects"]), "fit": len(r["fit_subjects"]), "val": len(r["val_subjects"])}
                   for r in out["refits"]],
        "test_subjects_used": out["subjects_used_in_test_side"]}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "recovery":
        recovery()
    else:
        ensemble(float(sys.argv[2]) if len(sys.argv) > 2 else CFG["pysr"]["timeout_refit_s"],
                 sys.argv[3] if len(sys.argv) > 3 else "ensemble_pilot.json")
