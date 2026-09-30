"""Stage D2 tests: the PySR wrapper (PLAN.md D3, D5; §8.2, §16.5; IMP-059, IMP-060).

Tests without the `pysr` marker need no Julia (keyword arguments, the equation evaluator, front scoring,
front comparison, the guards). Tests marked `pysr` load Julia and run real fits (minutes); pytest.ini
excludes them by default, run them with `-m pysr`. They skip with a reason if PySR or the pinned Julia is
not available. Synthetic rows only: no real recording, no test subject.
"""
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402,F401  (sets PYTHON_JULIACALL_THREADS from config before PySR can be imported)
from src import regression as R  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()


def pysr_or_skip():
    try:
        return R.load_pysr(CFG)
    except Exception as exc:                                # not installed, wrong pin, no Julia
        pytest.skip(f"PySR unavailable: {type(exc).__name__}: {str(exc)[:200]}")


# ---- keyword arguments --------------------------------------------------------------------------------------

def test_model_kwargs_follow_config_and_leave_the_search_at_default():
    kw = R.model_kwargs(CFG, "synthetic", random_state=7)
    assert kw["binary_operators"] == ["+", "-", "*", "/"] and kw["unary_operators"] == ["tanh", "exp"]
    assert kw["precision"] == 64 and kw["batching"] is False and kw["turbo"] is False
    assert kw["parallelism"] == "multithreading" and kw["timeout_in_seconds"] == 120.0
    assert kw["parsimony"] == 0.01 and kw["random_state"] == 7 and kw["niterations"] == 1000000
    for forbidden in ("populations", "ncycles_per_iteration", "fast_cycle", "batch_size", "procs",
                      "deterministic"):
        assert forbidden not in kw                          # left at PySR's defaults (§16.5.3)
    assert R.model_kwargs(CFG, "primary", random_state=1)["timeout_in_seconds"] == 1800.0
    assert R.model_kwargs(CFG, "refit", random_state=1)["timeout_in_seconds"] == 600.0
    assert R.model_kwargs(CFG, "pilot_primary", random_state=1)["timeout_in_seconds"] == 600.0
    assert R.model_kwargs(CFG, "refit", random_state=1, timeout_s=25)["timeout_in_seconds"] == 25.0
    assert R.model_kwargs(CFG, "refit", random_state=1, parsimony=0.03)["parsimony"] == 0.03


def test_serial_mode_is_deterministic_fixed_iterations_no_timeout():
    kw = R.model_kwargs(CFG, "synthetic", random_state=3, serial=True, niterations=8)
    assert kw["parallelism"] == "serial" and kw["deterministic"] is True and kw["niterations"] == 8
    assert kw["timeout_in_seconds"] is None and kw["random_state"] == 3
    with pytest.raises(R.RegressionError):
        R.model_kwargs(CFG, "synthetic", random_state=3, serial=True)


def test_multiprocessing_mode_passes_procs_only_there():
    cfg = copy.deepcopy(CFG)
    cfg["pysr"]["parallelism"] = "multiprocessing"
    assert R.model_kwargs(cfg, "refit", random_state=1)["procs"] == 8


def test_turbo_is_off_in_config_until_the_check_passes():
    assert CFG["pysr"]["turbo"] is False


def test_load_pysr_refuses_without_the_thread_variable(monkeypatch):
    if "juliacall" in sys.modules:
        pytest.skip("juliacall already imported in this process; the thread count is fixed")
    monkeypatch.delenv("PYTHON_JULIACALL_THREADS", raising=False)
    with pytest.raises(R.RegressionError, match="PYTHON_JULIACALL_THREADS"):
        R.load_pysr(CFG)


def test_config_pins_the_versions_setup_found():
    v = CFG["pysr"]["version"]
    assert v["pysr"] == "2.6.0" and v["symbolic_regression_jl"] == "2.5.1" and v["julia"] == "1.11.9"


# ---- equation evaluation and front scoring ----------------------------------------------------------------------

def test_evaluate_equation_matches_hand_values_and_flags_overflow():
    X = np.array([[0.5, 2.0, -1.0], [1.5, -1.0, 0.3]])
    names = ["u_tgt", "u_src", "S_src"]
    got = R.evaluate_equation("u_tgt * u_src - 0.25 * tanh(S_src) + 1.5", X, names)
    want = np.array([0.5 * 2.0 - 0.25 * math.tanh(-1.0) + 1.5, 1.5 * -1.0 - 0.25 * math.tanh(0.3) + 1.5])
    np.testing.assert_allclose(got, want, rtol=1e-14)
    assert np.all(R.evaluate_equation("0.37", X, names) == 0.37) and R.evaluate_equation("0.37", X, names).shape == (2,)
    big = R.evaluate_equation("exp(exp(u_src + 10))", X, names)
    assert np.all(np.isinf(big))                                    # exp(exp(12)) and exp(exp(9)) overflow float64


def test_front_entries_score_the_validation_rows_and_overflow_becomes_inf():
    names = ["u_tgt", "u_src", "S_src"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] * X[:, 1]
    table = pd.DataFrame({"complexity": [1, 3, 6], "loss": [1.0, 0.02, 0.01],
                          "equation": ["0.1", "u_tgt * u_src", "exp(exp(exp(u_src + 30)))"]})
    es = R.front_entries(table, X, y, names)
    assert es[0].val_loss == pytest.approx(np.mean((0.1 - y) ** 2), rel=1e-12)
    assert es[1].val_loss == pytest.approx(0.0, abs=1e-24)
    assert es[2].val_loss == math.inf and es[2].loss == 0.01
    sel = R.select_equation(es, CFG)
    assert sel.entry.complexity == 3 and not sel.no_term


# ---- front comparison (the turbo and determinism definitions) ---------------------------------------------------------

def front(rows):
    return pd.DataFrame(rows, columns=["complexity", "loss", "equation"])


def test_fronts_identical_is_bitwise():
    a = front([(1, 1.0, "0.5"), (3, 0.1, "u_tgt * 0.3123456789")])
    assert R.fronts_identical(a, a.copy())
    assert not R.fronts_identical(a, front([(1, 1.0, "0.5"), (3, 0.1 + 1e-16, "u_tgt * 0.3123456789")]))
    assert not R.fronts_identical(a, front([(1, 1.0, "0.5")]))


def test_fronts_equivalent_rounds_constants_and_checks_loss_rtol():
    a = front([(1, 1.0, "0.5"), (3, 0.1, "u_tgt * 0.31234"), (5, 0.05, "tanh(u_src * 1.00004) + 0.2")])
    b = front([(1, 1.0000000001, "0.50000001"), (3, 0.1000000001, "u_tgt * 0.312349"),
               (5, 0.0500000001, "tanh(u_src * 1.00001) + 0.2000001")])
    assert R.fronts_equivalent(a, b, CFG)                           # constants agree to 4 figures, losses to 1e-9
    assert not R.fronts_equivalent(a, front([(1, 1.0, "0.5"), (3, 0.1, "u_tgt * 0.4"),
                                             (5, 0.05, "tanh(u_src) + 0.2")]), CFG)       # a constant differs
    assert not R.fronts_equivalent(a, front([(1, 1.0, "0.5"), (3, 0.1, "u_src * 0.31234"),
                                             (5, 0.05, "tanh(u_src * 1.00004) + 0.2")]), CFG)   # structure differs
    assert not R.fronts_equivalent(a, front([(1, 1.0, "0.5"), (3, 0.1002, "u_tgt * 0.31234"),
                                             (5, 0.05, "tanh(u_src * 1.00004) + 0.2")]), CFG)   # loss off by 2e-3
    assert not R.fronts_equivalent(a, front([(1, 1.0, "0.5"), (2, 0.1, "u_tgt * 0.31234"),
                                             (5, 0.05, "tanh(u_src * 1.00004) + 0.2")]), CFG)   # complexity differs


# ---- guards before any Julia is touched -----------------------------------------------------------------------------

def make_split(n_train=40, n_test=10):
    return {"train": [f"sub-{i:03d}" for i in range(1, n_train + 1)],
            "test": [f"sub-{i:03d}" for i in range(500, 500 + n_test)]}


def synth_rows(rng, n, subject, centred=True):
    if centred:
        X = rng.normal(size=(n, 3))
        y = 2.0 * X[:, 0] * X[:, 1] + 0.01 * rng.normal(size=n)
        truth = 2.0 * X[:, 0] * X[:, 1]
    else:                                                       # the section 9.1 planted basis: uncentred, / SD
        X = np.column_stack([rng.normal(6.0, 1.7, n), rng.normal(6.0, 1.7, n), rng.normal(2.5, 1.0, n)])
        truth = 1.0 * (X[:, 1] / 1.7) * (X[:, 0] / 1.7)
        y = truth + 0.005 * np.std(truth) * rng.normal(size=n)
    rows = R.Rows(X, y, np.zeros(n, int), np.zeros(n, np.int64), np.arange(n), np.full(n, subject))
    return rows, truth


def test_run_fit_and_run_ensemble_refuse_a_test_subject_before_touching_julia():
    split = make_split()
    rng = np.random.default_rng(0)
    by = {s: synth_rows(rng, 50, s)[0] for s in split["train"] + split["test"]}
    with pytest.raises(R.RegressionError, match="not training subjects"):
        R.run_fit(by, split["train"][:5], [split["test"][0]], split, 42, CFG, role="synthetic")
    with pytest.raises(R.RegressionError, match="not training subjects"):
        R.run_ensemble(by, split["train"] + [split["test"][1]], split, 42, CFG, n_refits=3)


# ---- real fits (marker pysr) --------------------------------------------------------------------------------------------

def recovery(centred, timeout_s=None):
    pysr_or_skip()
    split = make_split(n_train=12, n_test=4)
    rng = np.random.default_rng(10 if centred else 11)
    by, truths = {}, {}
    for s in split["train"]:
        by[s], truths[s] = synth_rows(rng, 6000, s, centred)
    fit, val = R.split_subjects_fit_val(split["train"], split, 2042, CFG)
    res = R.run_fit(by, fit, val, split, 42, CFG, role="synthetic", timeout_s=timeout_s)
    held, truth = synth_rows(np.random.default_rng(99), 20000, "held", centred)          # fresh held-out rows
    Xh, _ = R.design(held, res.zscore)
    assert not res.selection.no_term, res.record["reason"]
    pred_z = R.evaluate_equation(res.selection.entry.equation, Xh)
    pred = pred_z * res.zscore.sd_y + res.zscore.mean_y
    nrmse = float(np.sqrt(np.mean((pred - truth) ** 2)) / np.std(truth))
    return res, nrmse


@pytest.mark.pysr
@pytest.mark.parametrize("centred", [True, False], ids=["centred_product", "uncentred_section_9_1_basis"])
def test_planted_product_is_selected_and_recovered_on_held_out_rows(centred):
    res, nrmse = recovery(centred)
    assert "u_src*u_tgt" in res.record["signatures"], res.record["equation"]
    assert nrmse <= 0.05, (nrmse, res.record["equation"])
    assert res.n_fit_rows == CFG["pysr"]["subsample"]["rows_per_fit"]                     # the fixed 50,000-row subsample
    assert res.record["precision"] == 64 and res.record["turbo"] is False


@pytest.mark.pysr
def test_ensemble_mechanics_with_short_timeouts():
    pysr_or_skip()
    split = make_split(n_train=40, n_test=10)
    rng = np.random.default_rng(5)
    by = {s: synth_rows(rng, 1500, s)[0] for s in split["train"] + split["test"]}
    cfg = copy.deepcopy(CFG)
    out = R.run_ensemble(by, split["train"], split, 42, cfg, n_refits=3, timeout_s=25.0)
    assert len(out["refits"]) == 3
    halves = [set(r["half_subjects"]) for r in out["refits"]]
    assert all(len(h) == 20 for h in halves) and len({tuple(sorted(h)) for h in halves}) == 3
    used = set().union(*[set(r["fit_subjects"]) | set(r["val_subjects"]) for r in out["refits"]])
    assert used <= set(split["train"]) and not used & set(split["test"])
    assert all((len(r["fit_subjects"]), len(r["val_subjects"])) == (16, 4) for r in out["refits"])
    rec = out["recurrence"]
    assert rec["n_refits"] == 3 and rec["needed"] == 3
    assert "u_src*u_tgt" in rec["stable"], rec["counts"]
