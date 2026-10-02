"""H0 tests: phase wiring in main.py (gate and low_confidence carry-forward, JSON flags, resumability with provenance, step
plans, force) and the phase-1 pieces of src/preprocess.py (exclusion file names, allow_all scope, impulse responses, the 5%
halt). Orchestration is tested with hand-built steps and hand-written files in tmp_path; nothing reads real recordings or the
network, and no real output is touched (IMP-089)."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

import main
from src import preprocess as pp
from src.config import load_config

CFG = load_config()
CFG1 = json.loads(json.dumps(CFG))
CFG1["compute"]["joblib_n_jobs"] = 1                     # inline pool: the monkeypatched worker must run in this process


def put(root, rel, doc):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return p


def gate(root, pilot, *, hard=False, low=False):
    rel = "results/pilot/gate_A.json" if pilot else "outputs/gate.json"
    return put(root, rel, {"hard_stop": hard, "low_confidence": low})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- gate state and the prerequisite ---------------------------------------------------------------------------------

def test_phase1_needs_no_gate(temp_root):
    assert main.check_phase_gate(CFG, temp_root, 1, False) == (None, None)


def test_full_phase2_refused_without_a_gate_file(temp_root):
    reason, low = main.check_phase_gate(CFG, temp_root, 2, False)
    assert reason and "not found" in reason and low is None


def test_full_phase_refused_on_hard_stop_and_when_the_flag_is_missing(temp_root):
    gate(temp_root, False, hard=True, low=True)
    reason, low = main.check_phase_gate(CFG, temp_root, 3, False)
    assert reason and "hard stop" in reason and low is True
    put(temp_root, "outputs/gate.json", {"low_confidence": False})                 # no hard_stop key at all
    reason, _ = main.check_phase_gate(CFG, temp_root, 3, False)
    assert reason and "hard stop" in reason


def test_full_phase_allowed_and_low_confidence_carried(temp_root):
    gate(temp_root, False, hard=False, low=True)
    assert main.check_phase_gate(CFG, temp_root, 2, False) == (None, True)
    gate(temp_root, False, hard=False, low=False)
    assert main.check_phase_gate(CFG, temp_root, 4, False) == (None, False)


def test_pilot_never_refused_even_on_hard_stop_and_carries_low_confidence(temp_root):
    assert main.check_phase_gate(CFG, temp_root, 2, True) == (None, None)          # no pilot gate yet
    gate(temp_root, True, hard=True, low=True)
    assert main.check_phase_gate(CFG, temp_root, 2, True) == (None, True)


def test_pilot_reads_the_pilot_gate_and_full_reads_outputs_gate(temp_root):
    gate(temp_root, True, hard=True)                                                # a pilot hard stop must not block a full run
    reason, _ = main.check_phase_gate(CFG, temp_root, 2, False)
    assert reason and "not found" in reason
    gate(temp_root, False, hard=False)
    assert main.check_phase_gate(CFG, temp_root, 2, False)[0] is None


def _flag(root, phase, pilot=False):
    return main.flag_path(CFG, root, phase, pilot)


def _prior_flag(root, phase, pilot):
    f = _flag(root, phase, pilot)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x", encoding="utf-8")


@pytest.mark.parametrize("pilot", [False, True])
def test_cli_refuses_phase_on_hard_stop_only_in_full_mode(temp_root, monkeypatch, pilot):
    _prior_flag(temp_root, 1, pilot)
    gate(temp_root, pilot, hard=True)
    monkeypatch.setitem(main.PHASE_RUNNERS, 2, lambda cfg, root, p: True)
    rc = main.main(["--phase", "2"] + (["--pilot"] if pilot else []), root=temp_root)
    assert rc == (main.EXIT_OK if pilot else main.EXIT_PREREQ)
    if not pilot:
        assert "hard stop" in (temp_root / "logs" / "phase2.log").read_text(encoding="utf-8")
        assert not _flag(temp_root, 2).exists()


def test_flag_is_json_with_commit_hash_low_confidence_and_steps(temp_root, monkeypatch):
    gate(temp_root, False, hard=False, low=True)
    _prior_flag(temp_root, 1, False)

    def runner(cfg, root, pilot):
        main.RUN_STATE["steps"]["x"] = {"status": "ran", "seconds": 1.5, "drift": []}
        return True
    monkeypatch.setitem(main.PHASE_RUNNERS, 2, runner)
    assert main.main(["--phase", "2"], root=temp_root) == main.EXIT_OK
    doc = main.read_flag(_flag(temp_root, 2))
    assert doc["phase"] == 2 and doc["pilot"] is False and doc["mechanics_only"] is False
    assert doc["low_confidence"] is True and doc["wall_seconds"] >= 0.0
    assert doc["config_sha256"] == sha(temp_root / "config.yml")
    assert doc["steps"] == {"x": {"status": "ran", "seconds": 1.5, "drift": []}}
    assert "T" in doc["timestamp"] and "git_commit" in doc


def test_pilot_flag_is_mechanics_only_and_low_confidence_unknown_without_gate(temp_root, monkeypatch):
    monkeypatch.setitem(main.PHASE_RUNNERS, 1, lambda cfg, root, pilot: True)
    assert main.main(["--phase", "1", "--pilot"], root=temp_root) == main.EXIT_OK
    doc = main.read_flag(_flag(temp_root, 1, True))
    assert doc["pilot"] is True and doc["mechanics_only"] is True and doc["low_confidence"] is None


def test_read_flag_legacy_and_missing(temp_root):
    p = put(temp_root, "outputs/phase1.done", "2026-10-01T00:00:00+00:00\n")
    assert main.read_flag(p) == {"legacy": True, "text": "2026-10-01T00:00:00+00:00"}
    assert main.read_flag(temp_root / "nope") is None


def test_failed_runner_writes_no_flag(temp_root, monkeypatch):
    monkeypatch.setitem(main.PHASE_RUNNERS, 1, lambda cfg, root, pilot: False)
    assert main.main(["--phase", "1"], root=temp_root) == main.EXIT_FAILED
    assert not _flag(temp_root, 1).exists()


def test_force_is_accepted_with_a_phase_and_still_needs_make_split_for_dry_run():
    assert main.parse_args(["--phase", "3", "--force"]).force is True
    with pytest.raises(SystemExit):
        main.parse_args(["--phase", "3", "--dry-run"])


def test_force_on_a_full_run_warns_about_the_deviation_note(temp_root, monkeypatch, caplog):
    gate(temp_root, False)
    _prior_flag(temp_root, 1, False)
    monkeypatch.setitem(main.PHASE_RUNNERS, 2, lambda cfg, root, pilot: True)
    main.main(["--phase", "2", "--force"], root=temp_root)
    log = (temp_root / "logs" / "phase2.log").read_text(encoding="utf-8")
    assert "DEVIATIONS.md" in log and "--force" in log
    assert main.RUN_STATE["force"] is True


# ---- steps and resumability ------------------------------------------------------------------------------------------

def _run(steps, root, pilot=True, force=False):
    main.RUN_STATE.update(force=force, steps={})
    main.run_steps(steps, CFG, root, pilot, force)
    return main.RUN_STATE["steps"]


def test_missing_output_runs_existing_fresh_output_is_skipped(tmp_path):
    out = tmp_path / "a.json"
    calls = []

    def run(force):
        calls.append(force)
        out.write_text(json.dumps({"x": 1}), encoding="utf-8")
    steps = [main.Step("a", [out], run)]
    assert _run(steps, tmp_path)["a"]["status"] == "ran" and calls == [False]
    assert _run(steps, tmp_path)["a"]["status"] == "skipped" and calls == [False]


def test_partial_outputs_run_the_step(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("1", encoding="utf-8")
    calls = []
    _run([main.Step("s", [a, b], lambda f: calls.append(f))], tmp_path)
    assert calls == [False]


def test_force_reruns_and_passes_force_to_the_step(tmp_path):
    out = tmp_path / "a.txt"
    out.write_text("1", encoding="utf-8")
    calls = []
    st = _run([main.Step("a", [out], lambda f: calls.append(f))], tmp_path, force=True)
    assert calls == [True] and st["a"]["status"] == "ran"


def test_always_step_reruns_every_time(tmp_path):
    out = tmp_path / "s.json"
    out.write_text("{}", encoding="utf-8")
    calls = []
    _run([main.Step("s", [out], lambda f: calls.append(1), always=True)], tmp_path)
    _run([main.Step("s", [out], lambda f: calls.append(1), always=True)], tmp_path)
    assert calls == [1, 1]


def _doc(seed=42, prov=None, pilot=True):
    return {"split_seed": seed, "pilot": pilot, "provenance": prov}


def test_stale_when_the_split_file_changed(temp_root):
    split = put(temp_root, "outputs/split_42.json", {"a": 1})
    out = put(temp_root, "results/pilot/c1_42.json", _doc(prov={"split_file_sha256": sha(split)}))
    assert main.stale_reason(out, CFG, temp_root) == (None, [])
    split.write_text(json.dumps({"a": 2}), encoding="utf-8")
    reason, _ = main.stale_reason(out, CFG, temp_root)
    assert reason and "split file" in reason
    with pytest.raises(main.StepRefused, match="--force"):
        _run([main.Step("c1", [out], lambda f: None)], temp_root)


def test_stale_when_the_frozen_equation_appears_changes_or_vanishes(temp_root):
    split = put(temp_root, "outputs/split_42.json", {"a": 1})
    prov = {"split_file_sha256": sha(split), "frozen_equation_sha256": None}
    out = put(temp_root, "results/pilot/c1_42.json", _doc(prov=prov))
    assert main.stale_reason(out, CFG, temp_root)[0] is None                       # made without one, none exists: fine
    fz = put(temp_root, "results/pilot/frozen_equation_42.json", {"e": 1})
    assert "frozen equation" in main.stale_reason(out, CFG, temp_root)[0]           # one has appeared
    prov["frozen_equation_sha256"] = sha(fz)
    out = put(temp_root, "results/pilot/c1_42.json", _doc(prov=prov))
    assert main.stale_reason(out, CFG, temp_root)[0] is None
    fz.write_text(json.dumps({"e": 2}), encoding="utf-8")
    assert "frozen equation" in main.stale_reason(out, CFG, temp_root)[0]
    fz.unlink()
    assert "frozen equation" in main.stale_reason(out, CFG, temp_root)[0]           # it vanished


def test_full_run_checks_the_full_frozen_path_not_the_pilot_one(temp_root):
    split = put(temp_root, "outputs/split_42.json", {"a": 1})
    out = put(temp_root, "results/c1_42.json", _doc(pilot=False, prov={"split_file_sha256": sha(split),
                                                                        "frozen_equation_sha256": None}))
    put(temp_root, "results/pilot/frozen_equation_42.json", {"e": 1})              # a pilot equation must not matter
    assert main.stale_reason(out, CFG, temp_root)[0] is None
    put(temp_root, "outputs/frozen_equation_42.json", {"e": 1})
    assert main.stale_reason(out, CFG, temp_root)[0] is not None


def test_config_drift_is_logged_not_refused(temp_root, caplog):
    split = put(temp_root, "outputs/split_42.json", {"a": 1})
    out = put(temp_root, "results/pilot/c1_42.json",
              _doc(prov={"split_file_sha256": sha(split), "config_yml_sha256": "0" * 64}))
    reason, drift = main.stale_reason(out, CFG, temp_root)
    assert reason is None and drift == ["config.yml changed since the output was written"]
    st = _run([main.Step("c1", [out], lambda f: pytest.fail("must be reused"))], temp_root)
    assert st["c1"]["status"] == "skipped" and st["c1"]["drift"] == drift


def test_output_without_provenance_or_non_json_is_reused(tmp_path):
    a = tmp_path / "a.json"
    a.write_text("{}", encoding="utf-8")
    b = tmp_path / "b.npz"
    b.write_bytes(b"x")
    assert main.stale_reason(a, CFG, tmp_path) == (None, [])
    assert main.stale_reason(b, CFG, tmp_path) == (None, [])


def test_invalid_json_output_is_stale(tmp_path):
    a = tmp_path / "a.json"
    a.write_text("{not json", encoding="utf-8")
    assert "not valid JSON" in main.stale_reason(a, CFG, tmp_path)[0]


def test_pilot_steps_do_not_enter_the_all_subjects_scope_full_steps_do(tmp_path):
    seen = []
    step = main.Step("s", [tmp_path / "missing"], lambda f: seen.append(pp._PERMIT["depth"]))
    _run([step], tmp_path, pilot=True)
    _run([step], tmp_path, pilot=False)
    assert seen == [0, 1] and pp._PERMIT["depth"] == 0


def test_a_step_failure_returns_false_from_the_runner_and_names_the_step(temp_root, monkeypatch, caplog):
    def boom(cfg, root, pilot):
        return [main.Step("boom", [Path(root) / "x"], lambda f: (_ for _ in ()).throw(RuntimeError("kaboom")))]
    monkeypatch.setitem(main.PHASE_STEPS, 3, boom)
    runner = main._make_runner(3)
    main.RUN_STATE.update(force=False, steps={})
    with caplog.at_level(logging.ERROR, logger=main.LOGGER_NAME):
        assert runner(CFG, temp_root, True) is False
    assert "kaboom" in caplog.text


def test_not_built_steps_fail_the_phase_with_the_stage_name(temp_root, caplog):
    main.RUN_STATE.update(force=False, steps={})
    with caplog.at_level(logging.ERROR, logger=main.LOGGER_NAME):
        assert main._make_runner(4)(CFG, temp_root, True) is False
    assert "NotBuilt" in caplog.text and "H1" in caplog.text


# ---- step plans ------------------------------------------------------------------------------------------------------

def _split_file(root, pilot_ids=("sub-005", "sub-008")):
    put(root, "outputs/split_42.json", {"pilot": list(pilot_ids)})


def test_phase1_pilot_plan(temp_root):
    _split_file(temp_root)
    names = [s.name for s in main.phase1_steps(CFG, temp_root, True)]
    assert names == ["preprocess:hp0.5_default", "preprocess:hp0.5_strict", "preprocess:hp0.1_default", "preprocess:hp0.1_strict",
                     "impulse_response", "g0_pilot"]
    outs = [s.outputs[0] for s in main.phase1_steps(CFG, temp_root, True)]
    assert outs[0] == temp_root / "results" / "pilot" / "exclusions.json"
    assert outs[2] == temp_root / "results" / "pilot" / "exclusions_hp0.1_default.json"
    assert outs[4] == temp_root / "results" / "pilot" / "bandpass_impulse_response.npz"
    assert outs[5] == temp_root / "results" / "pilot" / "gate_A.json"


def test_phase1_full_plan_downloads_first_and_stops_at_the_unbuilt_full_g0(temp_root):
    steps = main.phase1_steps(CFG, temp_root, False)
    assert steps[0].name == "download" and steps[-1].name == "g0"
    assert steps[0].outputs == [temp_root / "data" / "MANIFEST.sha256"]
    assert steps[-1].outputs == [temp_root / "outputs" / "gate.json"]
    assert steps[1].outputs == [temp_root / "outputs" / "exclusions.json"]
    assert steps[3].outputs == [temp_root / "outputs" / "exclusions_hp0.1_default.json"]
    with pytest.raises(main.NotBuilt, match="Stage K"):
        steps[-1].run(False)


def test_phase2_plan_pilot_and_full(temp_root):
    pil = main.phase2_steps(CFG, temp_root, True)
    assert [s.name for s in pil] == ["baseline:42", "qr_report:42", "primary_fit"]
    assert pil[0].outputs[0] == temp_root / "results" / "pilot" / "baseline_var_42.json"
    assert pil[2].outputs == [temp_root / "results" / "pilot" / "frozen_equation_42.json"]
    full = main.phase2_steps(CFG, temp_root, False)
    assert [s.name for s in full][:5] == [f"baseline:{s}" for s in (42, 43, 44, 45, 46)]
    assert len(full[-1].outputs) == 5
    with pytest.raises(main.NotBuilt):
        pil[2].run(False)


def test_phase3_plan_pilot_and_full(temp_root):
    pil = main.phase3_steps(CFG, temp_root, True)
    assert [s.name for s in pil] == ["c1:42", "c2", "c3", "c4", "diagnostics", "sensitivity", "summary"]
    assert pil[-1].always and not any(s.always for s in pil[:-1])
    assert pil[-1].outputs == [temp_root / "results" / "pilot" / "summary.json", temp_root / "results" / "pilot" / "equations.tex"]
    full = main.phase3_steps(CFG, temp_root, False)
    assert [s.name for s in full][:5] == [f"c1:{s}" for s in (42, 43, 44, 45, 46)]
    assert full[0].outputs == [temp_root / "results" / "c1_42.json"]


def test_phase3_runs_confirmatory_only_in_full_mode(temp_root, monkeypatch):
    from src import robustness as rb
    seen = {}
    monkeypatch.setattr(rb, "run_c1", lambda cfg, root, seed, **kw: seen.setdefault((seed, kw["pilot"]), kw["confirmatory"]))
    for pilot in (True, False):
        steps = main.phase3_steps(CFG, temp_root, pilot)
        steps[0].run(False)
    assert seen == {(42, True): False, (42, False): True}


def test_phase_runners_are_real_now():
    assert all(r.__name__ == "runner" for r in main.PHASE_RUNNERS.values())


# ---- preprocess: exclusions files, scope, impulse responses, halt -------------------------------------------------------

def test_exclusion_file_names_per_variant(tmp_path):
    ref = CFG["statistics"]["sensitivity"]["reference_variant"]
    assert pp.exclusions_path(CFG, tmp_path, ref, False) == tmp_path / "outputs" / "exclusions.json"
    assert pp.exclusions_path(CFG, tmp_path, "hp0.5_strict", False) == tmp_path / "outputs" / "exclusions_hp0.5_strict.json"
    assert pp.exclusions_path(CFG, tmp_path, "hp0.5_strict", True) == tmp_path / "results" / "pilot" / "exclusions_hp0.5_strict.json"


def test_allow_all_only_inside_the_scope():
    with pytest.raises(pp.DevelopmentGuardError, match="all_subjects_permitted"):
        pp.assert_dev_subject("sub-099", frozenset(), allow_all=True)
    with pp.all_subjects_permitted():
        pp.assert_dev_subject("sub-099", frozenset(), allow_all=True)
        with pp.all_subjects_permitted():
            pass
        pp.assert_dev_subject("sub-099", frozenset(), allow_all=True)             # still inside the outer scope
    with pytest.raises(pp.DevelopmentGuardError):
        pp.assert_dev_subject("sub-099", frozenset(), allow_all=True)


def test_scope_is_released_when_the_body_raises():
    with pytest.raises(ValueError):
        with pp.all_subjects_permitted():
            raise ValueError("x")
    assert pp._PERMIT["depth"] == 0


def test_dev_guard_still_refuses_non_pilot_without_allow_all():
    with pytest.raises(pp.DevelopmentGuardError):
        pp.assert_dev_subject("sub-099", frozenset({"sub-005"}), allow_all=False)
    pp.assert_dev_subject("sub-005", frozenset({"sub-005"}), allow_all=False)


def test_impulse_responses_saved_for_both_highpass_variants(tmp_path):
    path = pp.save_impulse_responses(CFG, tmp_path / "ir.npz")
    z = np.load(path)
    fs = float(CFG["dataset"]["native_fs_hz"])
    half = int(round(CFG["preprocessing"]["bandpass"]["impulse_response_duration_s"] * fs * 0.5))
    assert float(z["fs_hz"]) == fs and z["highpass_hz"].tolist() == [0.5, 0.1]
    for key, hp in (("response_primary", 0.5), ("response_sensitivity", 0.1)):
        r = z[key]
        assert r.shape == (2 * half + 1,) and r.dtype == np.float64
        np.testing.assert_array_equal(r, pp.bandpass_impulse_response(CFG, fs, hp))
    assert not np.array_equal(z["response_primary"], z["response_sensitivity"])
    assert "section 5.1" in str(z["description"])


def test_impulse_response_path_by_mode(tmp_path):
    assert pp.impulse_response_path(CFG, tmp_path, False) == tmp_path / "outputs" / "bandpass_impulse_response.npz"
    assert pp.impulse_response_path(CFG, tmp_path, True) == tmp_path / "results" / "pilot" / "bandpass_impulse_response.npz"


def _fake_decisions(n_total, n_units_fail):
    out = {}
    for i in range(n_total):
        failed = i < n_units_fail
        out[(f"sub-{i:03d}", "ses-t1")] = {"status": "excluded" if failed else "kept", "reason": "units_check" if failed else None,
                                          "detail": {}, "clean_s": None if failed else 200.0}
    return out


@pytest.mark.parametrize("n_fail,halt", [(7, False), (8, True)])
def test_preprocess_variant_reports_the_five_percent_halt_on_153_recordings(tmp_path, monkeypatch, n_fail, halt):
    """5% of 153 is 7.65: 7 failures do not halt, 8 do (exact arithmetic, §4.2)."""
    dec = _fake_decisions(153, n_fail)
    files = [tmp_path / "data" / f"sub-{i:03d}" / "ses-t1" / "eeg" / "x.edf" for i in range(153)]
    monkeypatch.setattr(pp, "recording_files", lambda cfg, root, subjects=None: files)
    keys = iter(sorted(dec))
    monkeypatch.setattr(pp, "_variant_worker", lambda payload: (lambda k: (k, dec[k]))(next(keys)))
    s = pp.preprocess_variant(CFG, tmp_path, {"name": "v", "highpass": False, "strict": False}, None,
                              allow_all=False, pilot_ids=frozenset(), n_jobs=1)
    assert s["units_check_halt"] == {"n_failed": n_fail, "n_total": 153, "fraction": n_fail / 153, "halt": halt}
    assert len(s["recordings"]) == 153


def test_variant_step_refuses_to_complete_on_a_halt_but_writes_the_file(temp_root, monkeypatch):
    _split_file(temp_root)
    dec = _fake_decisions(10, 3)
    monkeypatch.setattr(pp, "recording_files", lambda cfg, root, subjects=None: [temp_root / f"f{i}" for i in range(10)])
    keys = iter(sorted(dec))
    monkeypatch.setattr(pp, "_variant_worker", lambda payload: (lambda k: (k, dec[k]))(next(keys)))
    step = main.phase1_steps(CFG1, temp_root, True)[0]
    with pytest.raises(main.PhaseError, match="units-check halt"):
        step.run(False)
    doc = json.loads(step.outputs[0].read_text(encoding="utf-8"))
    assert doc["units_check_halt"]["halt"] is True and doc["variant"] == "hp0.5_default"
    assert doc["config_sha256"] == sha(temp_root / "config.yml") and "manifest_summary" in doc


def test_variant_step_writes_exclusions_with_variant_hash_and_manifest_line(temp_root, monkeypatch):
    _split_file(temp_root)
    put(temp_root, "data/MANIFEST.sha256", "# files=632 manifest_sha256=abc dataset=ds003775\n")
    dec = _fake_decisions(10, 0)
    monkeypatch.setattr(pp, "recording_files", lambda cfg, root, subjects=None: [temp_root / f"f{i}" for i in range(10)])
    keys = iter(sorted(dec))
    monkeypatch.setattr(pp, "_variant_worker", lambda payload: (lambda k: (k, dec[k]))(next(keys)))
    step = main.phase1_steps(CFG1, temp_root, True)[2]                                # hp0.1_default
    step.run(False)
    doc = json.loads(step.outputs[0].read_text(encoding="utf-8"))
    assert doc["variant"] == "hp0.1_default" and doc["manifest_summary"].startswith("# files=632")
    assert doc["units_check_halt"]["halt"] is False and len(doc["subjects"]) == 10
    assert step.outputs[0].name == "exclusions_hp0.1_default.json"


def test_workers_receive_the_variant_flags_and_the_guard_inputs(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(pp, "recording_files", lambda cfg, root, subjects=None: [tmp_path / "a", tmp_path / "b"])
    monkeypatch.setattr(pp, "_variant_worker", lambda p: (seen.append(p), (("s", "ses-t1"), _fake_decisions(1, 0)[("sub-000", "ses-t1")]))[1])
    pp.preprocess_variant(CFG, tmp_path, {"name": "v", "highpass": True, "strict": False}, ["sub-005"], allow_all=True,
                          pilot_ids=frozenset({"sub-005"}), n_jobs=1)
    cfg, edf, data_root, manifest, pilot_ids, allow_all, highpass, strict, cache_root = seen[0]
    assert (highpass, strict, allow_all) == (True, False, True) and pilot_ids == frozenset({"sub-005"})
    assert cache_root == tmp_path / "cache" and data_root == tmp_path / "data"


# ---- the pilot G0 driver restricted to filter A (phase 1 --pilot) --------------------------------------------------------

def test_restricted_g0_pilot_never_overwrites_the_all_option_evidence_files(tmp_path):
    from src import synthetic_gate as sg
    allo = sg.comparison_paths(CFG, tmp_path)
    assert sg.comparison_paths(CFG, tmp_path, list(CFG["g0"]["filter_options"])) == allo
    only_a = sg.comparison_paths(CFG, tmp_path, ["A"])
    assert only_a[0].name == "g0_filter_comparison_A.json" and only_a[1].name == "g0_pilot_timing_A.json"
    assert only_a[0] != allo[0] and only_a[1] != allo[1] and only_a[0].parent == allo[0].parent


def test_restricted_g0_pilot_refuses_an_unknown_option(tmp_path):
    from src import synthetic_gate as sg
    with pytest.raises(sg.GateError, match="not G0 filter options"):
        sg.run_pilot(CFG, tmp_path, options=["Z"])
