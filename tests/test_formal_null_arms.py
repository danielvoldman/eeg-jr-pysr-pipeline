"""K0 tests (IMP-094): the formal UKF-only null arms of src/synthetic_gate.py, the seed ledger, the not_run state and the
main.py / tuning.py gate fixes. The filters and the series are stubbed (hand-built records); expected values are hand tables,
a closed form, or a binomial root-find that is independent of the beta quantile the code uses. Nothing reads real data."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import binom

import main
from src import synthetic_gate as sg, tuning
from src.config import load_config

CFG = load_config()
STAB = {"n_runs": 2, "n_negative_eig_steps": 0, "n_nan_inf": 0, "n_linalg_divergences": 0, "n_jitter_fallbacks": 0,
        "min_eig_overall": 1e-3}
DELTA = 1.08


def cp_upper(k, n):
    """One-sided 95% Clopper-Pearson upper bound by root-finding on the binomial CDF (no beta quantile)."""
    return brentq(lambda u: binom.cdf(k, n, u) - 0.05, 1e-9, 1 - 1e-9)


def rec(index, g12=0.0, g21=0.0, *, dropped=False, off="same", has_term=None):
    """A null record. off: 'same' = flag-off estimates equal the rule-on ones, None = flag-off dropped, or (g12, g21)."""
    est = None if dropped else {"g12": g12, "g21": g21, "g12_filt": g12, "g21_filt": g21}
    if off == "same":
        off_est = est
    elif off is None:
        off_est = None
    else:
        off_est = {"g12_filt": off[0], "g21_filt": off[1]}
    return {"arm": "null_A", "index": index, "stream": "full", "filter": "A", "q": 1e-2, "level_index": None, "g_true": 0.0,
            "z_distance": 0.5, "regime": "limit_cycle", "diverged": dropped, "diverged_pass1": dropped,
            "diverged_fraction_pass1": 1.0 if dropped else 0.0, "n_segments": 1, "n_segments_dropped_pass1": int(dropped),
            "estimates": est, "stability": dict(STAB), "has_term": has_term, "nrmse": None, "linear_floor": None,
            "diagnostic_flag_off": {"diagnostic_only": True, "estimates": off_est}, "runtime_s": 1.0, "data_sha256": "ab" * 32}


# ---- configuration -------------------------------------------------------------------------------------------------

def test_formal_seeds_delta_and_rule_in_the_shipped_config():
    assert sg.formal_pairs(CFG) == [(92000, "null_A"), (92000, "null_B")]
    fresh = copy.deepcopy(CFG)
    fresh["g0"]["formal"]["round"] = 1
    assert sg.formal_pairs(fresh) == [(102000, "null_A"), (102000, "null_B")]          # block + 1 x stride 10000
    assert sg.delta(CFG) == pytest.approx(0.5 * 0.02 * 108) == pytest.approx(CFG["g0"]["formal"]["delta_expected"])
    assert CFG["ukf"]["divergence"]["state_dwell_s"] == 0 and CFG["ukf"]["divergence"]["parameter_sd_multiple"] is None
    assert CFG["g0"]["n_null_A_full"] == CFG["g0"]["n_null_B_full"] == 60


def test_burnt_blocks_are_the_pilot_tuning_gate_and_f2_blocks_and_not_92000():
    blocks = {b["block"] for b in CFG["g0"]["formal"]["burnt_blocks"]}
    assert blocks == {91000, 93000, 94000, 96000, 99000, 99500}
    assert 92000 not in blocks


# ---- the series status and the false-positive definition -----------------------------------------------------------

def test_status_hand_table_including_the_boundary_and_flag_off():
    cases = [(rec(0, 0.5, -0.2), "inside_delta"), (rec(1, 1.08, 0.0), "outside_delta"), (rec(2, 0.0, -1.08), "outside_delta"),
             (rec(3, 1.0799, 0.0), "inside_delta"), (rec(4, dropped=True), "dropped"), (rec(5, 0.1, -2.0), "outside_delta")]
    assert [sg.formal_status(CFG, r) for r, _ in cases] == [s for _, s in cases]
    r = rec(6, 0.1, 0.1, off=(3.0, 0.0))
    assert sg.formal_status(CFG, r) == "inside_delta" and sg.formal_status(CFG, r, flag_off=True) == "outside_delta"
    assert sg.formal_status(CFG, rec(7, 0.1, 0.1, off=None), flag_off=True) == "dropped"
    # a record whose rule-ON estimate exists but whose recording diverged is dropped (IMP-069)
    d = dict(rec(8, 0.1, 0.1), diverged=True)
    assert sg.formal_status(CFG, d) == "dropped"


def test_arm_report_hand_table_counts_false_positives_bound_and_cross_tab():
    recs = [rec(0, 0.2, 0.1), rec(1, 0.5, 0.5, off=(2.0, 0.0)), rec(2, 0.0, 0.0, off=None), rec(3, 1.08, 0.0),
            rec(4, -3.0, 0.0, off=(0.1, 0.1)), rec(5, dropped=True, off=(0.2, 0.2)), rec(6, dropped=True, off=None),
            rec(7, dropped=True, off=(5.0, 0.0)), rec(8, 0.3, 0.3), rec(9, 0.0, 0.9)]
    a = sg.formal_arm_report(CFG, "null_A", recs)
    assert a["n"] == 10 and a["rule_on"] == {"dropped": 3, "outside_delta": 2, "inside_delta": 5}
    assert a["false_positives"] == 5 and a["verdict"] is False and a["n_pending"] == 5
    assert a["upper_bound_95"] == pytest.approx(cp_upper(5, 10), abs=1e-6)
    fo = a["flag_off_diagnostic"]
    assert fo["diagnostic_only"] is True and fo["never_decides_a_verdict"] is True
    # flag-off statuses by series: in, out, dropped, out, in, in, dropped, out, in, in
    assert fo["counts"] == {"dropped": 2, "outside_delta": 3, "inside_delta": 5}
    x = fo["cross_tab_rule_on_by_flag_off"]
    assert x["dropped|inside_delta"] == 1 and x["dropped|dropped"] == 1 and x["dropped|outside_delta"] == 1
    assert x["outside_delta|outside_delta"] == 1 and x["outside_delta|inside_delta"] == 1
    assert x["inside_delta|dropped"] == 1 and x["inside_delta|outside_delta"] == 1 and x["inside_delta|inside_delta"] == 3
    assert sum(x.values()) == 10
    assert [r["status_rule_on"] for r in a["per_series"]][:4] == ["inside_delta", "inside_delta", "inside_delta", "outside_delta"]
    assert a["per_series"][3]["max_abs_g_rule_on"] == pytest.approx(1.08) and a["per_series"][5]["max_abs_g_rule_on"] is None


def test_arm_without_a_gain_half_failure_is_pending_never_a_pass():
    a = sg.formal_arm_report(CFG, "null_A", [rec(i, 0.3, -0.3) for i in range(60)])
    assert a["false_positives"] == 0 and a["verdict"] is None and a["n_pending"] == 60
    assert a["upper_bound_95"] == pytest.approx(1 - 0.05 ** (1 / 60)) == pytest.approx(cp_upper(0, 60), abs=1e-6)
    assert a["upper_bound_95"] == pytest.approx(0.0487, abs=5e-5)


def test_bounds_of_the_criterion_zero_of_sixty_passes_one_of_sixty_does_not():
    assert sg.upper_bound_95(0, 60) <= 0.05 < sg.upper_bound_95(1, 60)
    assert sg.upper_bound_95(1, 60) == pytest.approx(cp_upper(1, 60), abs=1e-6) == pytest.approx(0.0766, abs=5e-4)
    assert sg.upper_bound_95(0, 20) == pytest.approx(cp_upper(0, 20), abs=1e-6) == pytest.approx(0.139, abs=5e-4)


def test_selected_term_is_a_false_positive_too():
    a = sg.formal_arm_report(CFG, "null_B", [rec(0, 0.1, 0.1, has_term=True), rec(1, 0.1, 0.1, has_term=False)])
    assert a["false_positives"] == 1 and a["verdict"] is False and a["rule_on"]["inside_delta"] == 2


# ---- verdicts, flags, the not_run state ----------------------------------------------------------------------------

def test_verdicts_flags_and_not_run_for_a_null_failure_and_for_a_gain_half_clear_run():
    fail = {"null_A": {"verdict": False}, "null_B": {"verdict": None}}
    v = sg.formal_verdicts(fail)
    assert v == {"null_A": False, "null_B": None, "preproc_null": "cited_pilot", "positive": "not_run", "contraction": "not_run",
                 "stability": "not_run", "preproc_bias": "not_run"}
    f = sg.gate_flags(False, v)
    assert f["would_hard_stop"] is True and f["hard_stop"] is True and f["low_confidence"] is False
    assert f["not_run"] == ["contraction", "positive", "preproc_bias", "stability"] and f["complete"] is False
    assert f["pending"] == ["null_B"]
    clear = sg.gate_flags(False, sg.formal_verdicts({"null_A": {"verdict": None}, "null_B": {"verdict": None}}))
    assert clear["would_hard_stop"] is False and clear["hard_stop"] is False and clear["complete"] is False
    # not_run never counts as False: it must not set low_confidence nor a hard stop
    only_not_run = sg.gate_flags(False, {"null_A": True, "null_B": True, "positive": "not_run"})
    assert only_not_run["low_confidence"] is False and only_not_run["pending"] == [] and only_not_run["complete"] is False
    # complete needs neither pending nor not_run
    assert sg.gate_flags(False, {"null_A": True, "null_B": True})["complete"] is True


# ---- the ledger ----------------------------------------------------------------------------------------------------

def test_ledger_claim_burns_before_work_and_refuses_any_second_claim(tmp_path):
    path = tmp_path / "ledger.json"
    pairs = sg.formal_pairs(CFG)
    sg.ledger_claim(CFG, path, pairs, {"commit": "abc"})
    e = json.loads(path.read_text())["entries"]
    assert e["92000:null_A"]["status"] == "started" and e["92000:null_B"]["status"] == "started" and e["92000:null_A"]["commit"] == "abc"
    assert e["94000:*"]["status"] == "burnt" and e["91000:*"]["status"] == "burnt"
    before = path.read_bytes()
    with pytest.raises(sg.GateError, match="already started"):
        sg.ledger_claim(CFG, path, pairs, {"commit": "def"})
    assert path.read_bytes() == before
    sg.ledger_complete(CFG, path, pairs, {"result_sha256": "x"})
    assert json.loads(path.read_text())["entries"]["92000:null_A"]["status"] == "completed"
    with pytest.raises(sg.GateError, match="already completed"):                     # any status refuses
        sg.ledger_claim(CFG, path, pairs, {})


def test_ledger_wildcard_burnt_blocks_refuse_every_arm_and_92000_positive_stays_free(tmp_path):
    path = tmp_path / "ledger.json"
    for block, arm in [(94000, "artifact_null"), (94000, "positive"), (91000, "null_A"), (93000, "positive"), (96000, "null_B")]:
        with pytest.raises(sg.GateError, match="burnt"):
            sg.ledger_claim(CFG, path, [(block, arm)], {})
    assert not path.exists()                                                         # a refused claim writes nothing
    sg.ledger_claim(CFG, path, sg.formal_pairs(CFG), {})
    sg.ledger_claim(CFG, path, [(92000, "positive")], {})                          # the positive arm is a different key
    with pytest.raises(sg.GateError):
        sg.ledger_claim(CFG, path, [(92000, "positive")], {})


def test_ledger_claim_is_all_or_nothing(tmp_path):
    path = tmp_path / "ledger.json"
    sg.ledger_claim(CFG, path, [(92000, "null_B")], {})
    before = path.read_bytes()
    with pytest.raises(sg.GateError):
        sg.ledger_claim(CFG, path, [(92000, "null_A"), (92000, "null_B")], {})
    assert path.read_bytes() == before and "92000:null_A" not in json.loads(before)["entries"]


def test_ledger_complete_only_from_started(tmp_path):
    path = tmp_path / "ledger.json"
    with pytest.raises(sg.GateError, match="not in status started"):
        sg.ledger_complete(CFG, path, [(92000, "null_A")], {})
    sg.ledger_claim(CFG, path, [(92000, "null_A")], {})
    sg.ledger_complete(CFG, path, [(92000, "null_A")], {})
    with pytest.raises(sg.GateError, match="not in status started"):
        sg.ledger_complete(CFG, path, [(92000, "null_A")], {})


def test_write_once_never_replaces_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "r.json"
    sg.write_once(p, {"a": 1, "x": float("inf")})
    first = p.read_bytes()
    assert json.loads(first) == {"a": 1, "x": "inf"} and not list(tmp_path.glob("*.tmp"))
    with pytest.raises(sg.GateError, match="written once"):
        sg.write_once(p, {"a": 2})
    assert p.read_bytes() == first and not list(tmp_path.glob("*.tmp"))


# ---- preconditions ---------------------------------------------------------------------------------------------------

@pytest.fixture
def clean_git(monkeypatch):
    monkeypatch.setattr(tuning, "_git_state", lambda root: ("deadbeef", False))


def test_preconditions_pass_on_the_shipped_config_and_refuse_each_violation(temp_root, clean_git, monkeypatch):
    sg.formal_preconditions(CFG, temp_root)

    def broken(mutate):
        c = copy.deepcopy(CFG)
        mutate(c)
        return c

    cases = [("filter", broken(lambda c: c["g0"].__setitem__("filter", "B")), "g0.filter is not A"),
             ("dwell", broken(lambda c: c["ukf"]["divergence"].__setitem__("state_dwell_s", 0.5)), "legacy"),
             ("parameter clause", broken(lambda c: c["ukf"]["divergence"].__setitem__("parameter_sd_multiple", 8.0)), "legacy"),
             ("delta", broken(lambda c: c["g0"]["pass"].__setitem__("null_delta_fraction_of_weakest_level", 0.6)), "delta"),
             ("n", broken(lambda c: c["g0"].__setitem__("n_null_B_full", 61)), "60 series"),
             ("backend", broken(lambda c: c["g0"].__setitem__("backend", "numpy")), "numba")]
    for name, c, msg in cases:
        with pytest.raises(sg.GateError, match=msg):
            sg.formal_preconditions(c, temp_root)
    monkeypatch.setattr(sg, "g0_q", lambda cfg, name, tuned=None: 0.1)
    with pytest.raises(sg.GateError, match="q_fixed"):
        sg.formal_preconditions(CFG, temp_root)


def test_preconditions_refuse_dirty_tree_existing_files_burnt_seeds_and_pysr(temp_root, clean_git, monkeypatch):
    monkeypatch.setattr(tuning, "_git_state", lambda root: ("deadbeef", True))
    with pytest.raises(sg.GateError, match="not clean"):
        sg.formal_preconditions(CFG, temp_root)
    sg.formal_preconditions(CFG, temp_root, estimate_only=True)                       # the estimate needs no clean tree
    monkeypatch.setattr(tuning, "_git_state", lambda root: (None, None))
    with pytest.raises(sg.GateError, match="not clean"):
        sg.formal_preconditions(CFG, temp_root)
    monkeypatch.setattr(tuning, "_git_state", lambda root: ("deadbeef", False))
    paths = sg.formal_paths(CFG, temp_root)
    paths["results"].parent.mkdir(parents=True)
    paths["results"].write_text("{}")
    with pytest.raises(sg.GateError, match="g0_null_arms_r0.json exists"):
        sg.formal_preconditions(CFG, temp_root)
    paths["results"].unlink()
    paths["gate"].parent.mkdir(parents=True)
    paths["gate"].write_text("{}")
    with pytest.raises(sg.GateError, match="gate.json exists"):
        sg.formal_preconditions(CFG, temp_root)
    paths["gate"].unlink()
    sg.ledger_claim(CFG, paths["ledger"], sg.formal_pairs(CFG), {})
    with pytest.raises(sg.GateError, match="already started"):
        sg.formal_preconditions(CFG, temp_root)
    paths["ledger"].unlink()
    monkeypatch.setitem(sys.modules, "pysr", SimpleNamespace())
    with pytest.raises(sg.GateError, match="pysr is imported"):
        sg.formal_preconditions(CFG, temp_root, estimate_only=True)


def test_importing_the_gate_module_does_not_import_pysr():
    code = "import sys; import src.synthetic_gate; assert 'pysr' not in sys.modules and 'juliacall' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], cwd=Path(sg.__file__).resolve().parent.parent).returncode == 0


# ---- the cited artifact-only null ------------------------------------------------------------------------------------

def _pilot_gate(root, **over):
    doc = {"pilot": True, "filter": "A", "q": {"source": "q_fixed"},
           "preprocessing": {"null": {"n": 20, "n_fail": 20, "n_pending": 0, "delta": 1.08, "upper_bound_95": 1.0, "pass": False}}}
    doc.update(over)
    p = Path(root) / "results" / "pilot" / "gate_A.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def test_cited_pilot_null_carries_hash_counts_and_is_not_a_verdict(temp_root):
    p = _pilot_gate(temp_root)
    c = sg.cited_pilot_null(CFG, temp_root)
    assert c["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest() and (c["n"], c["n_fail"]) == (20, 20)
    assert c["seed_block"] == 94000 and "burnt" in c["seed_block_status"]
    assert c["is_verdict_of_this_run"] is False and c["counts_toward_hard_stop"] is False and "NOT a 5%" in c["rule"]


def test_cited_pilot_null_refuses_missing_wrong_q_or_wrong_size(temp_root):
    with pytest.raises(sg.GateError, match="not found"):
        sg.cited_pilot_null(CFG, temp_root)
    _pilot_gate(temp_root, q={"source": "tuned"})
    with pytest.raises(sg.GateError, match="q_fixed"):
        sg.cited_pilot_null(CFG, temp_root)
    _pilot_gate(temp_root, preprocessing={"null": {"n": 19, "n_fail": 19}})
    with pytest.raises(sg.GateError, match="q_fixed"):
        sg.cited_pilot_null(CFG, temp_root)


# ---- training-only inputs --------------------------------------------------------------------------------------------

def test_formal_inputs_refuse_a_training_list_that_overlaps_the_test_ids(temp_root, monkeypatch):
    split = {"train": ["sub-001", "sub-002"], "test": ["sub-002", "sub-003"], "pilot": ["sub-001"]}
    p = temp_root / "outputs" / "split_42.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(split))
    with pytest.raises(sg.GateError, match="overlaps the test list"):
        sg.formal_inputs(CFG, temp_root, 1)
    with pytest.raises(sg.GateError, match="training side"):
        sg.check_training_side(["sub-003"], split)


# ---- orchestration with stubbed series and filters -------------------------------------------------------------------

def _series(arm, specs):
    return [SimpleNamespace(arm=arm, stream="full", index=i, rec=r) for i, r in enumerate(specs)]


def _stub_run(monkeypatch, temp_root, a_specs, b_specs, seen=None):
    monkeypatch.setattr(tuning, "_git_state", lambda root: ("deadbeef", False))
    monkeypatch.setattr(sg, "formal_inputs", lambda cfg, root, n: (
        None, None, {"n_recordings": 5, "n_training_subjects": 3, "n_skipped": 0, "exponent": 1.0, "grid_key": "k" * 64,
                     "grid_from_cache": True}, {"preprocess": 0.0, "feature_table": 0.0, "grid": 0.0}))
    monkeypatch.setattr(sg, "formal_series", lambda cfg, grid, table, round_: {"null_A": _series("null_A", a_specs),
                                                                              "null_B": _series("null_B", b_specs)})

    def worker(payload):
        _, ser, name, q = payload
        if seen is not None:
            seen.append({k: v["status"] for k, v in sg.ledger_entries(CFG, sg.formal_paths(CFG, temp_root)["ledger"]).items()
                         if k.startswith("92000")})
        assert (name, q) == ("A", CFG["ukf"]["process_noise"]["q_fixed"])
        return dict(ser.rec, arm=ser.arm)
    monkeypatch.setattr(sg, "_series_worker", worker)
    _pilot_gate(temp_root)


def _inside(n):
    return [rec(i, 0.2, -0.2) for i in range(n)]


def test_formal_run_with_no_gain_half_failure_writes_results_only_and_unlocks_nothing(temp_root, monkeypatch):
    seen = []
    _stub_run(monkeypatch, temp_root, _inside(60), _inside(60), seen)
    doc = sg.run_formal_null_arms(CFG, temp_root, n_jobs=1)
    paths = sg.formal_paths(CFG, temp_root)
    assert doc["outcome"] == "gain_half_clear_pending_pysr" and doc["verdicts"]["null_A"] is None and doc["verdicts"]["null_B"] is None
    assert paths["results"].is_file() and not paths["gate"].exists()
    assert doc["files"]["gate"] is None and doc["files"]["results"] == "results/g0_null_arms_r0.json"
    assert doc["flags"]["hard_stop"] is False and doc["flags"]["complete"] is False and doc["not_run"] == list(sg.FORMAL_NOT_RUN)
    assert doc["cited_artifact_only_null"]["n_fail"] == 20 and doc["arms"]["null_A"]["n"] == 60
    assert seen and all(s == {"92000:null_A": "started", "92000:null_B": "started"} for s in seen)    # claimed before the work
    led = json.loads(paths["ledger"].read_text())["entries"]
    assert led["92000:null_A"]["status"] == "completed" and led["92000:null_A"]["outcome"] == "gain_half_clear_pending_pysr"
    assert led["92000:null_A"]["result_sha256"] == hashlib.sha256(paths["results"].read_bytes()).hexdigest()
    with pytest.raises(sg.GateError, match="refused"):                                 # the run cannot be repeated
        sg.run_formal_null_arms(CFG, temp_root, n_jobs=1)


def test_formal_run_with_a_null_failure_writes_a_hard_stop_gate_that_every_reader_refuses(temp_root, monkeypatch):
    b = _inside(59) + [rec(59, 2.5, 0.0)]
    _stub_run(monkeypatch, temp_root, _inside(60), b)
    doc = sg.run_formal_null_arms(CFG, temp_root, n_jobs=1)
    paths = sg.formal_paths(CFG, temp_root)
    assert doc["outcome"] == "null_failure_hard_stop" and doc["arms"]["null_B"]["false_positives"] == 1
    assert doc["arms"]["null_B"]["upper_bound_95"] == pytest.approx(cp_upper(1, 60), abs=1e-6)
    g = json.loads(paths["gate"].read_text())
    assert g["hard_stop"] is True and g["complete"] is False and g["pilot"] is False and g["would_hard_stop"] is True
    assert g["verdicts"]["null_B"] is False and g["verdicts"]["null_A"] is None and g["verdicts"]["preproc_null"] == "cited_pilot"
    assert all(g["verdicts"][k] == "not_run" for k in sg.FORMAL_NOT_RUN) and g["not_run"] == sorted(sg.FORMAL_NOT_RUN)
    assert g["formal_results_sha256"] == hashlib.sha256(paths["results"].read_bytes()).hexdigest()
    assert g["low_confidence"] is False
    with pytest.raises(tuning.GateError, match="hard stop"):
        tuning.check_gate(CFG, temp_root)
    reason, _ = main.check_phase_gate(CFG, temp_root, 2, False)
    assert reason and "hard stop" in reason


def test_a_crash_during_evaluation_leaves_the_seeds_burnt_and_no_result(temp_root, monkeypatch):
    _stub_run(monkeypatch, temp_root, _inside(60), _inside(60))

    def boom(payload):
        raise RuntimeError("worker died")
    monkeypatch.setattr(sg, "_series_worker", boom)
    with pytest.raises(RuntimeError):
        sg.run_formal_null_arms(CFG, temp_root, n_jobs=1)
    paths = sg.formal_paths(CFG, temp_root)
    assert not paths["results"].exists() and not paths["gate"].exists()
    assert json.loads(paths["ledger"].read_text())["entries"]["92000:null_A"]["status"] == "started"
    with pytest.raises(sg.GateError, match="already started"):
        sg.run_formal_null_arms(CFG, temp_root, n_jobs=1)


def test_a_failure_before_the_claim_burns_nothing(temp_root, monkeypatch):
    _stub_run(monkeypatch, temp_root, _inside(60), _inside(60))
    monkeypatch.setattr(sg, "formal_series", lambda *a: (_ for _ in ()).throw(RuntimeError("generation failed")))
    with pytest.raises(RuntimeError):
        sg.run_formal_null_arms(CFG, temp_root, n_jobs=1)
    assert not sg.formal_paths(CFG, temp_root)["ledger"].exists()


def test_the_driver_uses_only_training_inputs_and_null_arms_of_stream_full():
    assert sg.FORMAL_STREAM == "full" and sg.FORMAL_ARMS == ("null_A", "null_B")
    bad = SimpleNamespace(arm="positive", stream="full")
    with pytest.raises(sg.GateError, match="another arm or stream"):
        orig = sg.generate_series
        try:
            sg.generate_series = lambda *a, **k: bad
            sg.formal_series(CFG, None, None, 0)
        finally:
            sg.generate_series = orig


def test_cli_refuses_the_formal_run_together_with_pilot():
    with pytest.raises(SystemExit):
        sg.main(["--g0-null-arms", "--pilot"])


# ---- main.py and tuning.py ---------------------------------------------------------------------------------------------

def _gate(root, **doc):
    p = Path(root) / "outputs" / "gate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))


def _g0_step(root):
    return main.phase1_steps(CFG, root, False)[-1]


@pytest.mark.parametrize("doc,fails", [({"hard_stop": True, "complete": False}, True), ({"hard_stop": True, "complete": True}, True),
                                       ({"hard_stop": False, "complete": False}, True), ({"low_confidence": False}, True),
                                       ({"hard_stop": False, "complete": True}, False), ({"hard_stop": False}, False)])
def test_g0_step_fails_on_hard_stop_or_incomplete_even_when_skipped(temp_root, doc, fails):
    _gate(temp_root, **doc)
    step = _g0_step(temp_root)
    main.RUN_STATE.update(force=False, steps={})
    if fails:
        with pytest.raises(main.PhaseError, match="hard stop|incomplete"):
            main.run_steps([step], CFG, temp_root, False, False)
    else:
        main.run_steps([step], CFG, temp_root, False, False)
    assert main.RUN_STATE["steps"]["g0"]["status"] == "skipped"                       # the file existed, the step did not run


def test_phase1_writes_no_flag_on_a_hard_stop_gate_and_does_write_one_on_a_clean_gate(temp_root, monkeypatch):
    step = _g0_step(temp_root)
    monkeypatch.setitem(main.PHASE_STEPS, 1, lambda cfg, root, pilot: [step])
    monkeypatch.setattr(main, "check_phase_gate", lambda *a: (None, None))
    flag = main.flag_path(CFG, temp_root, 1, False)
    _gate(temp_root, hard_stop=True, complete=False, low_confidence=False)
    assert main.run_phase(CFG, temp_root, 1, False) == main.EXIT_FAILED and not flag.exists()
    _gate(temp_root, hard_stop=False, complete=True, low_confidence=False)
    assert main.run_phase(CFG, temp_root, 1, False) == main.EXIT_OK and flag.exists()


def test_g0_step_with_no_gate_is_still_not_built(temp_root):
    with pytest.raises(main.NotBuilt, match="g0-null-arms"):
        _g0_step(temp_root).run(False)


def test_phase_gate_and_check_gate_refuse_complete_false_but_accept_a_missing_key(temp_root):
    _gate(temp_root, hard_stop=False, complete=False, low_confidence=False)
    reason, low = main.check_phase_gate(CFG, temp_root, 2, False)
    assert reason and "incomplete" in reason and low is False
    with pytest.raises(tuning.GateError, match="incomplete"):
        tuning.check_gate(CFG, temp_root)
    for ok in ({"hard_stop": False, "complete": True, "low_confidence": False}, {"hard_stop": False, "low_confidence": False}):
        _gate(temp_root, **ok)
        assert main.check_phase_gate(CFG, temp_root, 2, False)[0] is None
        assert tuning.check_gate(CFG, temp_root)["hard_stop"] is False
    # a pilot is never refused
    p = temp_root / "results" / "pilot" / "gate_A.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"hard_stop": False, "complete": False, "low_confidence": True}))
    assert main.check_phase_gate(CFG, temp_root, 2, True) == (None, True)


def test_gate_document_lists_the_not_run_checks(temp_root, clean_git):
    v = sg.formal_verdicts({"null_A": {"verdict": False}, "null_B": {"verdict": None}})
    doc = sg.build_gate_document(CFG, temp_root, pilot=False, filter_name="A", verdicts=v, sections={})
    assert doc["not_run"] == sorted(sg.FORMAL_NOT_RUN) and doc["complete"] is False and doc["hard_stop"] is True
    assert doc["reasons"] == ["null_A failed"] and doc["pending"] == ["null_B"]
