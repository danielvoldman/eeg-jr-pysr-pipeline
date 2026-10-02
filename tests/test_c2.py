"""G3 tests: C2 aggregation (src/robustness.py c2_*, run_c2; src/regression.py ensemble persistence and crash handling;
§8.2, §8.4, §15.1; IMP-085). Expected values are hand arithmetic (ceil(0.7 * 25) = 18, ceil(0.7 * 3) = 3, 4 of 5 seeds),
hand-built ensemble and c1 documents; never read back from the code under test."""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import regression as R
from src import robustness as rb
from src import tuning
from src.config import load_config

CFG = load_config()
REPO = Path(__file__).resolve().parent.parent
TANH = "tanh(u_tgt)"
OTHER = "u_src + S_src"          # signatures {u_src, S_src}


def refit(k, eq=TANH, no_term=False, failed=False):
    sig = [] if (no_term or failed) else (["tanh(u_tgt)"] if eq == TANH else sorted(["S_src", "u_src"]))
    return {"k": k, "no_term": no_term or failed, "failed": failed, "equation": None if failed else eq,
            "signatures": sig, "half_subjects": ["sub-001", "sub-002"], "front": []}


def ens_doc(refits, pilot=False, n_exp=None):
    n = n_exp if n_exp is not None else (3 if pilot else 25)
    return {"schema": 1, "split_seed": 42, "pilot": pilot, "n_refits_expected": n, "refits": refits}


def frozen_doc(eq=TANH, no_term=False):
    return {"no_term": no_term, "equation": eq, "signatures": [] if no_term else ["tanh(u_tgt)"]}


def full_ensemble(n_tanh, n_other=0, n_no_term=0, n_failed=0):
    refits = [refit(k + 1) for k in range(n_tanh)]
    refits += [refit(n_tanh + k + 1, eq=OTHER) for k in range(n_other)]
    refits += [refit(n_tanh + n_other + k + 1, no_term=True) for k in range(n_no_term)]
    refits += [refit(n_tanh + n_other + n_no_term + k + 1, failed=True) for k in range(n_failed)]
    assert len(refits) == 25
    return ens_doc(refits)


# ---- C2(a): recurrence ------------------------------------------------------------------------------------------------

def test_signature_in_18_of_25_refits_is_stable_and_17_is_not():
    a = rb.c2_recurrence(full_ensemble(18, n_no_term=7), frozen_doc(), CFG)
    assert a["needed"] == 18 and a["stable"] == ["tanh(u_tgt)"] and a["a_passed"] is True
    b = rb.c2_recurrence(full_ensemble(17, n_no_term=8), frozen_doc(), CFG)
    assert b["stable"] == [] and b["a_passed"] is False


def test_no_term_and_crashed_refits_stay_in_the_denominator_and_are_counted():
    a = rb.c2_recurrence(full_ensemble(18, n_no_term=4, n_failed=3), frozen_doc(), CFG)
    assert a["n_refits"] == 25 and a["n_failed"] == 3 and a["n_no_term"] == 7 and a["counts"] == {"tanh(u_tgt)": 18}
    assert rb.c2_recurrence(full_ensemble(17, n_no_term=4, n_failed=4), frozen_doc(), CFG)["a_passed"] is False


def test_a_stable_signature_must_also_be_in_the_primary_equation():
    ens = full_ensemble(18, n_no_term=7)
    other = {"no_term": False, "equation": OTHER, "signatures": sorted(["S_src", "u_src"])}
    assert rb.c2_recurrence(ens, other, CFG)["a_passed"] is False
    assert rb.c2_recurrence(ens, frozen_doc(no_term=True, eq="0.37"), CFG)["a_passed"] is False
    assert rb.c2_recurrence(ens, None, CFG)["a_passed"] is None
    two = full_ensemble(19, n_other=6)                                   # both pass 18 only if each >= 18: tanh 19, other 6
    assert rb.c2_recurrence(two, frozen_doc(), CFG)["stable_in_primary"] == ["tanh(u_tgt)"]


def test_a_partial_or_mismatched_ensemble_is_refused():
    ens = full_ensemble(18, n_no_term=7)
    short = copy.deepcopy(ens)
    short["refits"].pop()
    with pytest.raises(rb.RobustnessError, match="partial"):
        rb.c2_recurrence(short, frozen_doc(), CFG)
    wrong_n = copy.deepcopy(ens)
    wrong_n["n_refits_expected"] = 24
    with pytest.raises(rb.RobustnessError, match="configuration"):
        rb.c2_recurrence(wrong_n, frozen_doc(), CFG)
    bad_sig = copy.deepcopy(ens)
    bad_sig["refits"][0]["signatures"] = ["tanh(u_src)"]
    with pytest.raises(rb.RobustnessError, match="stored signatures"):
        rb.c2_recurrence(bad_sig, frozen_doc(), CFG)
    dup = copy.deepcopy(ens)
    dup["refits"][1]["k"] = 1
    with pytest.raises(rb.RobustnessError):
        rb.c2_recurrence(dup, frozen_doc(), CFG)


def test_pilot_ensemble_of_three_needs_all_three():
    ens = ens_doc([refit(1), refit(2), refit(3, no_term=True)], pilot=True)
    assert rb.c2_recurrence(ens, frozen_doc(), CFG)["needed"] == 3
    assert rb.c2_recurrence(ens, frozen_doc(), CFG)["a_passed"] is False
    assert rb.c2_recurrence(ens_doc([refit(1), refit(2), refit(3)], pilot=True), frozen_doc(), CFG)["a_passed"] is True


def test_signatures_are_recomputed_from_the_equation_text():
    eq = "tanh(1.02*u_tgt + 0.01)"                                       # shares the signature tanh(u_tgt) (§8.4)
    ens = full_ensemble(18, n_no_term=7)
    for r in ens["refits"][:18]:
        r["equation"] = eq
    assert rb.c2_recurrence(ens, frozen_doc(eq=eq), CFG)["a_passed"] is True


# ---- C2(b) and the seed members ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("v,expected", [
    ((True, True, True, True, None), True), ((True, True, True, True, False), True), ((True, True, True, None, None), None),
    ((True, True, True, False, None), None), ((True, True, True, False, False), False), ((True, True, None, None, None), None),
    ((False, False, None, None, None), False), ((True, False, True, True, None), None), ((True, False, True, True, True), True),
    ((None, None, None, None, None), None)])
def test_seed_rule_is_three_valued_with_four_of_five(v, expected):
    out = rb.c2_seed_rule(dict(zip([42, 43, 44, 45, 46], v)), CFG)
    assert out["passed"] is expected and out["needed"] == 4 and out["of"] == 5


def test_seed_rule_counts_and_needs_five_seeds():
    out = rb.c2_seed_rule({42: True, 43: False, 44: None, 45: True, 46: True}, CFG)
    assert (out["n_passed"], out["n_failed"], out["n_undetermined"]) == (3, 1, 1)
    with pytest.raises(rb.RobustnessError):
        rb.c2_seed_rule({42: True, 43: True}, CFG)


@pytest.mark.parametrize("a,b,expected", [(True, True, True), (True, False, False), (False, None, False), (None, False, False),
                                          (True, None, None), (None, True, None), (None, None, None)])
def test_c2_is_a_and_b_three_valued(a, b, expected):
    assert rb.c2_verdict(a, b) is expected


def c1_doc(seed, p_m2, p_m0, passed, mode="scored", sha="x" * 64, pilot=True):
    comps = {} if mode == "absent" else {"M3_vs_M2": {"p": p_m2}, "M3_vs_M0": {"p": p_m0}}
    return {"split_seed": seed, "pilot": pilot, "mechanics_only": pilot, "m3": {"mode": mode, "frozen_equation_sha256": sha},
            "primary": {"comparisons": comps}, "c1_verdict": {"passed": None if mode == "absent" else passed}}


def test_c1_seed_p_is_the_larger_of_the_two_p_values_and_none_when_not_run():
    assert rb.c1_seed_p(c1_doc(43, 0.02, 0.07, True)) == 0.07
    assert rb.c1_seed_p(c1_doc(43, 0.30, 0.01, False)) == 0.30
    assert rb.c1_seed_p(c1_doc(43, 1.0, 0.01, False, mode="no_term")) == 1.0
    assert rb.c1_seed_p(c1_doc(43, 0, 0, None, mode="absent")) is None
    assert rb.c1_seed_p(None) is None


# ---- ensemble persistence and crash handling (regression.py) ------------------------------------------------------

def test_write_ensemble_is_write_once_and_load_checks_the_document(tmp_path):
    out = {"refits": [refit(1), refit(2), refit(3)], "recurrence": {"n_refits": 3}}
    doc = R.build_ensemble_document(out, CFG, tmp_path, 42, True)
    path = R.ensemble_path(CFG, tmp_path, 42, True)
    assert path.as_posix().endswith("results/pilot/ensemble_42.json")
    assert R.ensemble_path(CFG, tmp_path, 42, False).as_posix().endswith("outputs/ensemble_42.json")
    assert R.write_ensemble(path, doc) == "written" and R.write_ensemble(path, doc) == "unchanged"
    other = copy.deepcopy(doc)
    other["refits"][0]["equation"] = "tanh(u_src)"
    before = path.read_bytes()
    with pytest.raises(R.RegressionError):
        R.write_ensemble(path, other)
    assert path.read_bytes() == before
    loaded = R.load_ensemble(path)
    assert loaded["n_refits_expected"] == 3 and len(loaded["sha256"]) == 64
    broken = json.loads(path.read_text(encoding="utf-8"))
    del broken["refits"][0]["failed"]
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(R.RegressionError, match="failed"):
        R.load_ensemble(path)
    with pytest.raises(R.RegressionError):
        R.load_ensemble(tmp_path / "nope.json")


class _Fit:
    def __init__(self, record):
        self.record, self.seconds = record, 0.0


def test_a_crashed_refit_counts_as_no_term_and_is_reported(monkeypatch):
    split = {"train": [f"sub-{i:03d}" for i in range(1, 21)], "test": ["sub-099"]}
    calls = []

    def fake_fit(rows, fit_ids, val_ids, split_, seed, cfg, *, role, k=0, **kw):
        if k == 2:
            raise RuntimeError("julia exploded")
        return _Fit({"role": role, "k": k, "no_term": k == 3, "equation": TANH, "complexity": 2,
                     "signatures": [] if k == 3 else ["tanh(u_tgt)"], "seeds": {}, "front": []})

    monkeypatch.setattr(R, "run_fit", fake_fit)
    seen = []
    out = R.run_ensemble({}, split["train"], split, 42, CFG, n_refits=3, on_refit=seen.append)
    recs = out["refits"]
    assert [r["failed"] for r in recs] == [False, True, False] and recs[1]["no_term"] and recs[1]["signatures"] == []
    assert "julia exploded" in recs[1]["error"]
    assert out["recurrence"]["n_failed"] == 1 and out["recurrence"]["n_no_term"] == 2 and out["recurrence"]["n_refits"] == 3
    assert [r["k"] for r in seen] == [1, 2, 3] and all("half_subjects" in r for r in recs)
    assert out["recurrence"]["stable"] == []                                    # 1 of 3 refits < ceil(0.7 * 3) = 3
    assert calls == []


def test_our_own_refusals_are_not_swallowed_as_crashes(monkeypatch):
    split = {"train": [f"sub-{i:03d}" for i in range(1, 21)], "test": ["sub-099"]}

    def fake_fit(*a, **k):
        raise R.RegressionError("not training subjects")

    monkeypatch.setattr(R, "run_fit", fake_fit)
    with pytest.raises(R.RegressionError):
        R.run_ensemble({}, split["train"], split, 42, CFG, n_refits=3)


# ---- run_c2 on files (temp root, hand-built inputs) ----------------------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    import main as main_mod
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    split = {"train": ["sub-001", "sub-002", "sub-003"], "test": ["sub-050"]}
    monkeypatch.setattr(main_mod, "load_split", lambda seed, root=None, cfg=None: split)
    return tmp_path


def put_frozen(root, seed, pilot=True, eq=TANH, no_term=False):
    z = R.ZScore(mean_X=np.zeros(3), sd_X=np.ones(3), mean_y=0.0, sd_y=1.0, n=10)
    rec = {"role": "primary", "no_term": no_term, "equation": eq, "zscore": z.to_dict(), "front": [],
           "fit_subjects": ["sub-001"], "val_subjects": ["sub-002"], "complexity": 2, "val_loss": 1.0,
           "signatures": [] if no_term else ["tanh(u_tgt)"]}
    path = R.frozen_equation_path(CFG, root, seed, pilot)
    R.write_frozen_equation(path, R.build_frozen_document(rec, CFG, REPO, seed, pilot))
    return R.load_frozen_equation(path).sha256


def put_c1(root, seed, doc, pilot=True):
    path = rb.output_path(CFG, root, seed, pilot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def put_ensemble(root, refits, pilot=True):
    out = {"refits": refits, "recurrence": {}}
    R.write_ensemble(R.ensemble_path(CFG, root, 42, pilot), R.build_ensemble_document(out, CFG, root, 42, pilot))


def test_run_c2_pilot_aggregates_five_seeds_and_reports_the_holm_members(world):
    for s, (p2, p0, ok) in {42: (0.01, 0.02, True), 43: (0.02, 0.05, True), 44: (0.5, 0.01, False), 45: (0.03, 0.04, True),
                            46: (0.001, 0.002, True)}.items():
        put_c1(world, s, c1_doc(s, p2, p0, ok, sha=put_frozen(world, s)))
    put_ensemble(world, [refit(1), refit(2), refit(3)])
    out = rb.run_c2(CFG, world, pilot=True)
    d = out["doc"]
    assert out["path"].as_posix().endswith("results/pilot/c2_42.json") and d["mechanics_only"] and d["mode"] == "pilot"
    assert d["a_signature_recurrence"]["a_passed"] is True and d["a_signature_recurrence"]["needed"] == 3
    assert d["b_seed_rule"]["passed"] is True and d["b_seed_rule"]["n_passed"] == 4 and d["b_seed_rule"]["n_failed"] == 1
    assert d["c2_verdict"]["passed"] is True and d["missing_inputs"] == []
    assert d["holm_raw_p"] == {"c1_seed:43": 0.05, "c1_seed:44": 0.5, "c1_seed:45": 0.04, "c1_seed:46": 0.002}
    assert rb.run_c2(CFG, world, pilot=True)["status"] == "unchanged"


def test_run_c2_missing_seed_leaves_b_and_the_member_undetermined_never_p_one(world):
    for s, ok in {42: True, 43: True, 44: True}.items():
        put_c1(world, s, c1_doc(s, 0.01, 0.01, ok, sha=put_frozen(world, s)))
    put_ensemble(world, [refit(1), refit(2), refit(3)])
    d = rb.run_c2(CFG, world, pilot=True)["doc"]
    assert d["holm_raw_p"]["c1_seed:45"] is None and d["holm_raw_p"]["c1_seed:46"] is None
    assert d["b_seed_rule"]["passed"] is None and d["b_seed_rule"]["n_undetermined"] == 2
    assert d["c2_verdict"]["passed"] is None and len(d["missing_inputs"]) == 2


def test_run_c2_without_any_frozen_equation_makes_no_verdict(world):
    for s in (42, 43, 44, 45, 46):
        put_c1(world, s, c1_doc(s, 0, 0, None, mode="absent", sha=None))
    put_ensemble(world, [refit(1), refit(2), refit(3)])
    d = rb.run_c2(CFG, world, pilot=True)["doc"]
    assert d["a_signature_recurrence"]["a_passed"] is None and d["b_seed_rule"]["passed"] is None
    assert d["c2_verdict"]["passed"] is None and set(d["holm_raw_p"].values()) == {None}


def test_run_c2_a_no_term_seed_gives_p_one_and_fails_that_seed(world):
    for s in (42, 43, 44, 45, 46):
        sha = put_frozen(world, s, eq="0.37", no_term=True)
        put_c1(world, s, c1_doc(s, 1.0, 0.2, False, mode="no_term", sha=sha))
    put_ensemble(world, [refit(1), refit(2), refit(3)])
    d = rb.run_c2(CFG, world, pilot=True)["doc"]
    assert set(d["holm_raw_p"].values()) == {1.0} and d["b_seed_rule"]["passed"] is False
    assert d["a_signature_recurrence"]["a_passed"] is False and d["c2_verdict"]["passed"] is False


def test_run_c2_guard_refuses_mixed_modes_wrong_seeds_wrong_sha_and_test_subjects(world, monkeypatch):
    put_ensemble(world, [refit(1), refit(2), refit(3)])
    sha = put_frozen(world, 42)
    for s in (43, 44, 45, 46):
        put_c1(world, s, c1_doc(s, 0.1, 0.1, True, sha=put_frozen(world, s)))
    put_c1(world, 42, c1_doc(42, 0.1, 0.1, True, sha="0" * 64))
    with pytest.raises(rb.GuardError, match="sha256"):
        rb.run_c2(CFG, world, pilot=True)
    put_c1(world, 42, c1_doc(42, 0.1, 0.1, True, sha=sha, pilot=False))
    with pytest.raises(rb.GuardError, match="full file in a pilot run"):
        rb.run_c2(CFG, world, pilot=True)
    put_c1(world, 42, c1_doc(42, 0.1, 0.1, True, sha=sha))
    wrong = c1_doc(43, 0.1, 0.1, True, sha=put_frozen(world, 43))
    wrong["split_seed"] = 44
    put_c1(world, 43, wrong)
    with pytest.raises(rb.GuardError, match="split seed"):
        rb.run_c2(CFG, world, pilot=True)
    put_c1(world, 43, c1_doc(43, 0.1, 0.1, True, sha=put_frozen(world, 43)))
    bad = [refit(1), refit(2), refit(3)]
    bad[0]["half_subjects"] = ["sub-001", "sub-050"]
    R.write_ensemble(R.ensemble_path(CFG, world, 42, True),
                     R.build_ensemble_document({"refits": bad, "recurrence": {}}, CFG, world, 42, True), force=True)
    with pytest.raises(R.RegressionError, match="not training subjects"):
        rb.run_c2(CFG, world, pilot=True)
    with pytest.raises(rb.GuardError):
        rb.run_c2(CFG, world, pilot=True, confirmatory=True)
    with pytest.raises(rb.GuardError, match="confirmatory=True"):
        rb.run_c2(CFG, world, pilot=False)


def test_run_c2_confirmatory_needs_the_gate_and_refuses_mechanics_only_inputs(world, monkeypatch):
    monkeypatch.setattr(tuning, "check_gate", lambda cfg, root: (_ for _ in ()).throw(tuning.GateError("no gate")))
    with pytest.raises(tuning.GateError):
        rb.run_c2(CFG, world, pilot=False, confirmatory=True)
    monkeypatch.setattr(tuning, "check_gate", lambda cfg, root: {"low_confidence": True})
    put_c1(world, 42, dict(c1_doc(42, 0.1, 0.1, True, sha=None), pilot=False, mechanics_only=True), pilot=False)
    with pytest.raises(rb.GuardError, match="mechanics-only"):
        rb.run_c2(CFG, world, pilot=False, confirmatory=True)
