"""G8 tests: summary.json, equations.tex and the Holm adjustment (src/robustness.py G8 block; §18.2, §14, §15.2, §18.1;
IMP-084, IMP-088). Expected Holm values are hand arithmetic (rank factors m, m-1, ..., 1 and a running maximum); the
hand-built result files are written to tmp_path, never read from results/."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import regression as R
from src import robustness as rb
from src.config import load_config

CFG = load_config()
REPO = Path(__file__).resolve().parent.parent

ZSCORE = {"mean_X": [1.5, -2.0, 0.25], "sd_X": [2.0, 4.0, 0.5], "mean_y": 0.125, "sd_y": 8.0, "n": 1000}


def put(root, rel, doc):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def frozen_doc(eq="tanh(u_tgt) + u_src_S_src", no_term=False, pilot=True):
    return {"schema": 1, "split_seed": 42, "pilot": pilot, "variable_names": list(R.VARIABLE_NAMES), "equation": eq,
            "no_term": no_term, "reason": "selected equation is a bare constant" if no_term else "selected", "complexity": 4,
            "val_loss": 0.5, "min_val_loss": 0.48, "zscore": ZSCORE, "signatures": [] if no_term else ["tanh(u_tgt)"],
            "fit_subjects": ["sub-001"], "val_subjects": ["sub-002"]}


def gate_doc(low=True, hard=False):
    return {"pilot": True, "filter": "A", "hard_stop": hard, "would_hard_stop": hard, "low_confidence": low, "complete": True,
            "delta": 1.08, "reasons": [], "verdicts": {},
            "positive": {"pass": False, "levels": {}}, "null_A": {"pass": True, "n": 20, "n_fail": 0, "upper_bound_95": 0.14},
            "null_B": {"pass": True, "n": 20, "n_fail": 0, "upper_bound_95": 0.14}, "contraction": {"pass": True}, "stability": {"pass": True},
            "preprocessing": {"note": "x"}}


def c1_doc(seed=42, pilot=True, raw_p=None, passed=None):
    return {"split_seed": seed, "pilot": pilot, "mechanics_only": pilot, "m3": {"mode": "absent"},
            "c1_verdict": {"passed": passed, "reason": "r"},
            "primary": {"means": {"M0": 0.1}, "n_subjects": 3, "comparisons": {}, "bootstrap": {"B": 100}},
            "divergence": {}, "holm": {"raw_p": raw_p if raw_p is not None else {}}}


def root_with(tmp_path, pilot=True, **docs):
    """A tmp root holding the named result files. docs: gate, frozen, c1, c2, c3, c4, ..."""
    base = "results/pilot" if pilot else "results"
    names = {"c1": f"{base}/c1_42.json", "c2": f"{base}/c2_42.json", "c3": f"{base}/c3_42.json", "c4": f"{base}/c4_42.json",
             "diagnostics": f"{base}/diagnostics_42.json", "sensitivity": f"{base}/sensitivity_42.json",
             "gate": "results/pilot/gate_A.json" if pilot else "outputs/gate.json",
             "frozen": f"{'results/pilot' if pilot else 'outputs'}/frozen_equation_42.json"}
    for k, doc in docs.items():
        put(tmp_path, names[k], doc)
    return tmp_path


# ---- Holm family -----------------------------------------------------------------------------------------------------

P11 = {"c1_step:M1_vs_M0": 0.003, "c1_step:M2_vs_M1": 0.001, "freerun:2s": 0.02, "freerun:5s": 0.03, "freerun:10s": 0.04,
       "c3_test_icc:g12": 0.2, "c3_test_icc:g21": 0.3, "c1_seed:43": 0.4, "c1_seed:44": 0.5, "c1_seed:45": 0.01, "c1_seed:46": 0.002}


def split_docs(p):
    c1 = c1_doc(raw_p={k.split(":", 1)[1]: v for k, v in p.items() if k.startswith("c1_step:")})
    c2 = {"holm_raw_p": {k: v for k, v in p.items() if k.startswith("c1_seed:")}}
    c3 = {"holm_raw_p": {k: v for k, v in p.items() if k.startswith("c3_test_icc:")}}
    c4 = {"status": "attempted", "holm_raw_p": {k: v for k, v in p.items() if k.startswith("freerun:")}}
    return c1, c2, c3, c4


def test_holm_family_of_11_hand_values():
    # sorted p: .001 .002 .003 .01 .02 .03 .04 .2 .3 .4 .5 -> x11 x10 x9 x8 x7 x6 x5 x4 x3 x2 x1, running max
    rep = rb.holm_from_files(CFG, *split_docs(P11), pilot=False)
    assert rep["status"] == "complete" and rep["family_size"] == 11 and rep["c4_attempted"] is True
    adj = {m["name"]: m["adjusted_p"] for m in rep["members"]}
    want = {"c1_step:M2_vs_M1": 0.011, "c1_seed:46": 0.020, "c1_step:M1_vs_M0": 0.027, "c1_seed:45": 0.080, "freerun:2s": 0.14,
            "freerun:5s": 0.18, "freerun:10s": 0.20, "c3_test_icc:g12": 0.80, "c3_test_icc:g21": 0.90, "c1_seed:43": 0.90,
            "c1_seed:44": 0.90}
    for k, v in want.items():
        assert adj[k] == pytest.approx(v, abs=1e-12), k


def test_holm_family_of_8_when_c4_not_attempted_and_freerun_p_ignored():
    p = {k: v for k, v in P11.items() if not k.startswith("freerun:")}
    c1, c2, c3, _ = split_docs(p)
    c4 = {"status": "not_attempted", "holm_raw_p": {"freerun:2s": 0.0001}}      # must not enter the family
    rep = rb.holm_from_files(CFG, c1, c2, c3, c4, pilot=False)
    assert rep["family_size"] == 8 and rep["status"] == "complete" and rep["c4_attempted"] is False
    adj = {m["name"]: m["adjusted_p"] for m in rep["members"]}
    # sorted p: .001 .002 .003 .01 .2 .3 .4 .5 (c1_step x2, c1_seed 45/46, rest) -> x8 x7 x6 x5 x4 x3 x2 x1
    assert adj["c1_step:M2_vs_M1"] == pytest.approx(0.008)
    assert adj["c1_seed:46"] == pytest.approx(0.014)
    assert adj["c1_step:M1_vs_M0"] == pytest.approx(0.018)
    assert adj["c1_seed:45"] == pytest.approx(0.05)
    assert adj["c3_test_icc:g12"] == pytest.approx(0.8)
    assert adj["c1_seed:44"] == pytest.approx(0.9)
    assert not any(n.startswith("freerun:") for n in adj)


def test_holm_partial_family_gives_no_adjusted_value_and_names_the_missing():
    c1, c2, c3, c4 = split_docs(P11)
    c2 = {"holm_raw_p": {"c1_seed:43": 0.4}}                                     # 44-46 missing
    rep = rb.holm_from_files(CFG, c1, c2, c3, c4, pilot=False)
    assert rep["status"] == "partial_family"
    assert rep["missing"] == ["c1_seed:44", "c1_seed:45", "c1_seed:46"]
    assert all(m["adjusted_p"] is None for m in rep["members"])


def test_holm_with_no_files_is_partial_not_a_crash():
    rep = rb.holm_from_files(CFG, None, None, None, None, pilot=False)
    assert rep["status"] == "partial_family" and rep["family_size"] == 8 and len(rep["missing"]) == 8


def test_pilot_never_reports_an_adjusted_value_even_for_a_complete_family():
    rep = rb.holm_from_files(CFG, *split_docs(P11), pilot=True)
    assert rep["status"] == "pilot_no_adjustment"
    assert all(m["adjusted_p"] is None for m in rep["members"])
    assert all(m["raw_p"] is not None for m in rep["members"])                  # raw p stay visible


def test_holm_none_p_in_a_member_is_missing():
    p = dict(P11, **{"c3_test_icc:g12": None})
    rep = rb.holm_from_files(CFG, *split_docs(p), pilot=False)
    assert rep["status"] == "partial_family" and rep["missing"] == ["c3_test_icc:g12"]


# ---- paths and inputs ------------------------------------------------------------------------------------------------

def test_summary_paths_by_mode(tmp_path):
    full = rb.summary_paths(CFG, tmp_path, pilot=False)
    pilot = rb.summary_paths(CFG, tmp_path, pilot=True)
    assert full == (tmp_path / "results" / "summary.json", tmp_path / "results" / "equations.tex")
    assert pilot == (tmp_path / "results" / "pilot" / "summary.json", tmp_path / "results" / "pilot" / "equations.tex")


def test_inputs_use_the_gate_path_of_the_mode(tmp_path):
    assert rb.summary_inputs(CFG, tmp_path, 42, True)["gate"] == tmp_path / "results/pilot/gate_A.json"
    assert rb.summary_inputs(CFG, tmp_path, 42, False)["gate"] == tmp_path / "outputs/gate.json"


# ---- summary.json ----------------------------------------------------------------------------------------------------

def test_everything_absent_writes_explicit_absent_forms(tmp_path):
    out = rb.run_summary(CFG, tmp_path, 42, pilot=True)
    s = json.loads(out["summary_path"].read_text(encoding="utf-8"))
    assert s["residual"]["status"] == "absent" and "reason" in s["residual"]
    assert s["gate"]["status"] == "absent" and s["low_confidence"] is None
    for c in ("C1", "C2", "C3", "C4"):
        assert s["claims"][c]["status"] == "absent" and s["claims"][c]["verdict"]["passed"] is None
    assert s["holm"]["status"] == "partial_family"
    assert "frozen_equation" in s["missing_inputs"] and "gate" in s["missing_inputs"]
    tex = out["equations_path"].read_text(encoding="utf-8")
    assert "Residual absent" in tex and "mechanics only" in tex


def test_low_confidence_reaches_the_residual_and_every_claim(tmp_path):
    root_with(tmp_path, gate=gate_doc(low=True), frozen=frozen_doc(), c1=c1_doc())
    s = rb.build_summary(CFG, tmp_path, 42, pilot=True)
    assert s["low_confidence"] is True and s["residual"]["low_confidence"] is True
    assert all(s["claims"][c]["low_confidence"] is True for c in ("C1", "C2", "C3", "C4"))
    assert s["gate"]["null_A"]["upper_bound_95"] == 0.14 and s["gate"]["hard_stop"] is False


def test_gate_not_low_confidence_is_false_not_none(tmp_path):
    root_with(tmp_path, gate=gate_doc(low=False))
    s = rb.build_summary(CFG, tmp_path, 42, pilot=True)
    assert s["low_confidence"] is False and s["claims"]["C1"]["low_confidence"] is False


def test_no_verdict_is_invented_when_the_input_has_none(tmp_path):
    root_with(tmp_path, c1=c1_doc(passed=None))
    s = rb.build_summary(CFG, tmp_path, 42, pilot=True)
    assert s["claims"]["C1"]["verdict"]["passed"] is None and s["claims"]["C1"]["status"] == "present"


def test_c1_seed_table_comes_from_the_seed_files(tmp_path):
    root_with(tmp_path, c1=c1_doc())
    put(tmp_path, "results/pilot/c1_43.json", c1_doc(seed=43, passed=True))
    put(tmp_path, "results/pilot/c1_44.json", c1_doc(seed=44, passed=False))
    s = rb.build_summary(CFG, tmp_path, 42, pilot=True)
    seeds = s["claims"]["C1"]["other_seeds"]
    assert seeds["43"]["passed"] is True and seeds["44"]["passed"] is False and seeds["45"] is None and seeds["46"] is None


def test_term_residual_has_sympy_latex_signature_and_hash(tmp_path):
    root_with(tmp_path, frozen=frozen_doc())
    s = rb.build_summary(CFG, tmp_path, 42, pilot=True)
    r = s["residual"]
    assert r["status"] == "term" and r["signatures"] == ["tanh(u_tgt)"]
    assert "tanh(u_tgt)" in r["equation_sympy"] and "u_src_S_src" in r["equation_sympy"]
    assert r["equation_latex"].count(r"\tanh") == 1 and r"\tilde{u}_{\mathrm{tgt}}" in r["equation_latex"]
    fp = tmp_path / "results/pilot/frozen_equation_42.json"
    assert r["frozen_equation_sha256"] == hashlib.sha256(fp.read_bytes()).hexdigest()


def test_no_term_residual_is_recorded_not_omitted(tmp_path):
    root_with(tmp_path, frozen=frozen_doc(eq="0.7", no_term=True))
    out = rb.run_summary(CFG, tmp_path, 42, pilot=True)
    r = out["summary"]["residual"]
    assert r["status"] == "no_term" and r["equation_latex"] is None and r["reason"]
    assert "No residual term" in out["equations_path"].read_text(encoding="utf-8")


def test_gains_block_is_median_and_quartiles_of_the_c3_sessions(tmp_path):
    c3 = {"estimator": {"mode": "M2_stand_in"}, "all_pairs": {"directions": {"g12": {}, "g21": {}}},
          "gains_per_session": {"a": {"ses-t1": {"g12": 1.0, "g21": 10.0}, "ses-t2": {"g12": 3.0, "g21": 20.0}},
                                "b": {"ses-t1": {"g12": 11.0, "g21": 30.0}, "ses-t2": None}}, "holm_raw_p": {}}
    root_with(tmp_path, c3=c3)
    g = rb.build_summary(CFG, tmp_path, 42, pilot=True)["coupling_gains"]
    assert g["directions"]["g12"] == {"n_sessions": 3, "median": 3.0, "q25": 2.0, "q75": 7.0}
    assert g["directions"]["g21"]["median"] == 20.0 and g["directions"]["g21"]["n_sessions"] == 3


def test_compute_time_from_json_flags_legacy_flags_and_explicit_timings(tmp_path):
    put(tmp_path, "results/pilot/phase1.done", {"wall_seconds": 12.5})
    (tmp_path / "results/pilot/phase2.done").write_text("2026-10-02T00:00:00+00:00\n", encoding="utf-8")   # legacy timestamp flag
    s = rb.build_summary(CFG, tmp_path, 42, pilot=True, timings={"phase3": 7.0, "phase3_steps": {"c1": 3.0}})
    t = s["compute_time_s"]
    assert t["phase1"] == 12.5 and t["phase2"] is None and t["phase3"] == 7.0 and t["phase4"] is None
    assert t["phase3_steps"] == {"c1": 3.0}


def test_full_mode_reads_outputs_gate_and_never_the_pilot_files(tmp_path):
    root_with(tmp_path, pilot=False, gate=gate_doc(low=False), frozen=frozen_doc(pilot=False), c1=c1_doc(pilot=False))
    put(tmp_path, "results/pilot/c1_42.json", c1_doc(passed=True))                 # a pilot file that must stay unread
    out = rb.run_summary(CFG, tmp_path, 42, pilot=False)
    s = out["summary"]
    assert out["summary_path"] == tmp_path / "results" / "summary.json" and s["mode"] == "confirmatory"
    assert s["claims"]["C1"]["verdict"]["passed"] is None and s["mechanics_only"] is False
    assert s["gate"]["status"] == "present" and s["residual"]["status"] == "term"


def test_hard_stop_is_reported_in_the_gate_block(tmp_path):
    root_with(tmp_path, pilot=False, gate=gate_doc(hard=True))
    s = rb.build_summary(CFG, tmp_path, 42, pilot=False)
    assert s["gate"]["hard_stop"] is True


def test_run_is_deterministic_and_leaves_inputs_untouched(tmp_path):
    root_with(tmp_path, gate=gate_doc(), frozen=frozen_doc(), c1=c1_doc())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "results/pilot").glob("*.json")}
    a = rb.run_summary(CFG, tmp_path, 42, pilot=True)
    ba = (a["summary_path"].read_bytes(), a["equations_path"].read_bytes())
    b = rb.run_summary(CFG, tmp_path, 42, pilot=True)
    assert ba == (b["summary_path"].read_bytes(), b["equations_path"].read_bytes())
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert before == after


def test_summary_json_is_strict_json(tmp_path):
    root_with(tmp_path, gate=gate_doc(), frozen=frozen_doc())
    out = rb.run_summary(CFG, tmp_path, 42, pilot=True)
    json.loads(out["summary_path"].read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


# ---- equations.tex ---------------------------------------------------------------------------------------------------

def test_equations_tex_term_form(tmp_path):
    root_with(tmp_path, gate=gate_doc(low=False), frozen=frozen_doc())
    out = rb.run_summary(CFG, tmp_path, 42, pilot=True)
    tex = out["equations_path"].read_text(encoding="utf-8")
    assert r"\underbrace" in tex and r"\text{frozen residual } r" in tex
    assert r"\tanh" in tex and r"\tilde{u}_{\mathrm{tgt}}" in tex
    # the stored z-scoring constants appear verbatim at 6 significant digits
    for token in ("1.5", "-2", "0.25", "0.125", "8"):
        assert token in tex
    assert r"\begin{equation}" in tex and tex.count(r"\begin{equation}") == tex.count(r"\end{equation}") == 2
    assert tex.count(r"\begin{align}") == tex.count(r"\end{align}") == 1
    assert "mechanics only" in tex                                                   # a pilot is stamped
    assert "Low confidence" not in tex                                              # gate not low_confidence


def test_equations_tex_low_confidence_is_written(tmp_path):
    root_with(tmp_path, gate=gate_doc(low=True), frozen=frozen_doc())
    tex = rb.run_summary(CFG, tmp_path, 42, pilot=True)["equations_path"].read_text(encoding="utf-8")
    assert "Low confidence" in tex and r"\underbrace" in tex


def test_equations_tex_full_run_has_no_pilot_stamp(tmp_path):
    root_with(tmp_path, pilot=False, gate=gate_doc(low=False), frozen=frozen_doc(pilot=False))
    tex = rb.run_summary(CFG, tmp_path, 42, pilot=False)["equations_path"].read_text(encoding="utf-8")
    assert "mechanics only" not in tex.split("\n\n")[1] if len(tex.split("\n\n")) > 1 else True
    assert "% residual status: term; low_confidence: False; mechanics_only: False" in tex


# ---- ownership -------------------------------------------------------------------------------------------------------

def test_summary_does_not_import_pysr(tmp_path):
    root_with(tmp_path, gate=gate_doc(), frozen=frozen_doc())
    code = ("import sys\nfrom pathlib import Path\nfrom src import robustness as rb\nfrom src.config import load_config\n"
            f"rb.run_summary(load_config(), Path(r'{tmp_path}'), 42, pilot=True)\n"
            "assert not any(m == 'pysr' or m.startswith('pysr.') or m.startswith('juliacall') for m in sys.modules), 'pysr imported'\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_real_pilot_files_give_an_honest_summary():
    """The committed-by-hand pilot outputs: no frozen equation, no verdicts, no adjusted values (a read-only smoke test;
    skipped when results/pilot is absent, as on a fresh clone)."""
    if not (REPO / "results/pilot/c1_42.json").is_file():
        pytest.skip("no pilot results")
    s = rb.build_summary(CFG, REPO, 42, pilot=True)
    assert s["residual"]["status"] == "absent" and s["holm"]["status"] in ("partial_family", "pilot_no_adjustment")
    assert all(m["adjusted_p"] is None for m in s["holm"]["members"])
    assert all(s["claims"][c]["verdict"]["passed"] is None for c in ("C1", "C3", "C4"))
