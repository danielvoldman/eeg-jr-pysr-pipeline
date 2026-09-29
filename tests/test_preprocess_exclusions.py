"""B6 tests: the §12 exclusion rules applied per recording, subject consequences, the halt rule, the
exclusion file (§12, §11.1; IMP-012). Pure functions over hand-made results; nothing is read from data/
and nothing is written under the real outputs/.
"""
import json
from pathlib import Path

import pytest

from src import preprocess
from src.config import load_config

CFG = load_config()
S1, S2 = CFG["dataset"]["session_first"], CFG["dataset"]["session_second"]
MIN_CLEAN_S = 60.0     # §12 placeholder minimum, written out here on purpose (independent of config)
MAX_CORRECTED = 0.10   # IMP-013 placeholder: largest allowed corrected fraction of the trimmed recording
TRIMMED_S = 100.0      # hand-made trimmed recording length: the limit is then exactly 10.0 s
STEP_S = 1 / 1024      # one sample at 1024 Hz; corrected times are multiples of it


def meta(subject="sub-001", session=S1, units_ok=True, bad=(), sd=5.0):
    return {"rel_path": f"{subject}/{session}/eeg/x_eeg.edf", "subject": subject, "session": session,
            "units_passed": units_ok, "units_median_sd_uv": sd, "bad_electrodes": list(bad),
            "electrodes": {n: {"flags": ["rms_above_max"]} for n in bad}}


def b5(clean_s, corrected=(0.0, 0.0), trimmed_s=TRIMMED_S):
    return {"b5": {"log": {"clean_s": clean_s, "trimmed_s": trimmed_s,
                           "blinks": [{"corrected_s": c} for c in corrected]}}}


def kept():
    return {"status": "kept", "reason": None, "detail": {}, "clean_s": 200.0}


def excluded(reason="bad_electrode"):
    return {"status": "excluded", "reason": reason, "detail": {}, "clean_s": None}


# ---------------------------------------------------------------- per-recording rules, one test each

def test_units_check_failure_excludes():
    d = preprocess.recording_decision(CFG, meta(units_ok=False, sd=0.5), None)
    assert d["status"] == "excluded" and d["reason"] == "units_check" and d["detail"]["median_sd_uv"] == 0.5


def test_bad_electrode_excludes_and_names_it():
    d = preprocess.recording_decision(CFG, meta(bad=["PO3"]), None)
    assert d["status"] == "excluded" and d["reason"] == "bad_electrode"
    assert d["detail"]["electrodes"] == {"PO3": ["rms_above_max"]}


def test_clean_data_minimum_at_59_9_and_60_0_seconds():
    assert preprocess.recording_decision(CFG, meta(), b5(59.9))["reason"] == "insufficient_clean_data"
    ok = preprocess.recording_decision(CFG, meta(), b5(MIN_CLEAN_S))
    assert ok["status"] == "kept" and ok["reason"] is None and ok["clean_s"] == MIN_CLEAN_S
    d = preprocess.recording_decision(CFG, meta(), b5(59.9))
    assert d["detail"] == {"clean_s": 59.9, "minimum_s": MIN_CLEAN_S}


def test_rule_order_units_then_electrode_then_clean_data():
    both = preprocess.recording_decision(CFG, meta(units_ok=False, bad=["P3"]), None)
    assert both["reason"] == "units_check"
    assert preprocess.recording_decision(CFG, meta(bad=["P3"]), b5(10.0))["reason"] == "bad_electrode"


def test_clean_data_rule_needs_b5_results():
    with pytest.raises(preprocess.PreprocessError, match="B5"):
        preprocess.recording_decision(CFG, meta(), None)
    with pytest.raises(preprocess.PreprocessError, match="B5"):
        preprocess.recording_decision(CFG, meta(), {"b5": None})


# ---------------------------------------------------------------- corrected-time guard (IMP-013)

def test_corrected_time_exactly_at_the_fraction_is_kept_and_just_above_is_excluded():
    limit_s = MAX_CORRECTED * TRIMMED_S                                  # 10.0 s
    at = preprocess.recording_decision(CFG, meta(), b5(200.0, (limit_s, 0.0)))
    assert at["status"] == "kept" and at["reason"] is None
    above = preprocess.recording_decision(CFG, meta(), b5(200.0, (limit_s + STEP_S, 0.0)))
    assert above["status"] == "excluded" and above["reason"] == "excessive_blink_correction"
    assert above["detail"]["maximum_fraction"] == MAX_CORRECTED
    assert above["detail"]["corrected_fraction"] == pytest.approx([(limit_s + STEP_S) / TRIMMED_S, 0.0])
    assert above["clean_s"] == 200.0


def test_corrected_time_boundary_in_samples_for_a_non_round_recording_length():
    trimmed_s = 282112 * STEP_S                                          # 275.5 s; limit 27.55 s = 28211.2 samples
    ok = preprocess.recording_decision(CFG, meta(), b5(200.0, (28211 * STEP_S, 0.0), trimmed_s))
    bad = preprocess.recording_decision(CFG, meta(), b5(200.0, (28212 * STEP_S, 0.0), trimmed_s))
    assert ok["status"] == "kept" and bad["reason"] == "excessive_blink_correction"


def test_either_channel_over_the_fraction_excludes():
    over = MAX_CORRECTED * TRIMMED_S + 1.0
    for corrected in ((over, 0.0), (0.0, over), (over, over)):
        assert preprocess.recording_decision(CFG, meta(), b5(200.0, corrected))["reason"] == "excessive_blink_correction"
    assert preprocess.recording_decision(CFG, meta(), b5(200.0, (9.0, 9.0)))["status"] == "kept"


def test_blink_guard_comes_after_bad_electrode_and_before_clean_data_minimum():
    over = MAX_CORRECTED * TRIMMED_S + 1.0
    assert preprocess.recording_decision(CFG, meta(bad=["P3"]), None)["reason"] == "bad_electrode"
    assert preprocess.recording_decision(CFG, meta(units_ok=False), b5(200.0, (over, over)))["reason"] == "units_check"
    both = preprocess.recording_decision(CFG, meta(), b5(10.0, (over, over)))          # also below the clean-data minimum
    assert both["reason"] == "excessive_blink_correction"
    assert preprocess.recording_decision(CFG, meta(), b5(10.0, (1.0, 1.0)))["reason"] == "insufficient_clean_data"


def test_blink_guard_subject_consequences_are_those_of_the_other_recording_rules():
    over = MAX_CORRECTED * TRIMMED_S + 1.0
    bad = preprocess.recording_decision(CFG, meta(), b5(200.0, (over, 0.0)))
    good = preprocess.recording_decision(CFG, meta(), b5(200.0))
    s = status(good, bad)                                                # t2-only failure
    assert s["in_c1"] and not s["in_c3"] and s["t2"] == {"status": "excluded", "reason": "excessive_blink_correction"}
    s = status(bad, good)                                                # t1 failure: t2 unused
    assert not s["in_c1"] and not s["in_c3"]
    assert s["t1"]["reason"] == "excessive_blink_correction"
    assert s["t2"] == {"status": "excluded", "reason": "t1_excluded_t2_unused"}
    s = status(bad, None)
    assert not s["in_c1"] and not s["in_c3"] and s["t2"] is None


def test_blink_guard_is_logged(caplog):
    with caplog.at_level("WARNING", logger="pipeline.preprocess"):
        preprocess.recording_decision(CFG, meta(), b5(200.0, (50.0, 0.0)))
    assert len(caplog.records) == 1
    assert "excessive_blink_correction" in caplog.records[0].getMessage()
    assert "sub-001/ses-t1" in caplog.records[0].getMessage()


def test_corrected_fractions_helper():
    assert preprocess.corrected_fractions(b5(1.0, (5.0, 20.0))["b5"]["log"]) == [0.05, 0.2]


def test_blink_guard_config_leaf_is_a_placeholder_of_ten_percent():
    raw = __import__("yaml").safe_load((Path(__file__).resolve().parent.parent / "config.yml").read_text(encoding="utf-8"))
    leaf = raw["preprocessing"]["ocular"]["max_corrected_fraction"]
    assert leaf["value"] == MAX_CORRECTED and leaf["prov"] == "placeholder"
    assert CFG["preprocessing"]["ocular"]["max_corrected_fraction"] == MAX_CORRECTED


def test_every_exclusion_is_logged(caplog):
    with caplog.at_level("WARNING", logger="pipeline.preprocess"):
        preprocess.recording_decision(CFG, meta(units_ok=False), None)
        preprocess.recording_decision(CFG, meta(bad=["P4"]), None)
        preprocess.recording_decision(CFG, meta(), b5(1.0))
        preprocess.recording_decision(CFG, meta(), b5(100.0))       # kept: no warning
    text = " ".join(r.getMessage() for r in caplog.records)
    assert len(caplog.records) == 3
    for reason in ("units_check", "bad_electrode", "insufficient_clean_data"):
        assert reason in text and "sub-001/ses-t1" in text


# ---------------------------------------------------------------- subject-level consequences

def status(t1, t2):
    dec = {("sub-001", S1): t1}
    if t2 is not None:
        dec[("sub-001", S2)] = t2
    return preprocess.subject_status(CFG, dec)["sub-001"]


def test_both_kept_is_in_c1_and_c3():
    s = status(kept(), kept())
    assert s["in_c1"] and s["in_c3"] and s["t2"]["status"] == "kept"


def test_t2_only_failure_keeps_c1_and_drops_c3():
    s = status(kept(), excluded())
    assert s["in_c1"] and not s["in_c3"]
    assert s["t1"]["status"] == "kept" and s["t2"] == {"status": "excluded", "reason": "bad_electrode"}


def test_t1_failure_removes_c1_and_c3_and_the_t2_recording_is_unused():
    s = status(excluded("units_check"), kept())
    assert not s["in_c1"] and not s["in_c3"]
    assert s["t2"] == {"status": "excluded", "reason": "t1_excluded_t2_unused"}


def test_no_t2_recording_stays_in_c1_never_in_c3():
    s = status(kept(), None)
    assert s["in_c1"] and not s["in_c3"] and s["t2"] is None
    s = status(excluded(), None)
    assert not s["in_c1"] and not s["in_c3"] and s["t2"] is None


def test_subject_without_t1_is_an_error():
    with pytest.raises(preprocess.PreprocessError, match="t1"):
        preprocess.subject_status(CFG, {("sub-001", S2): kept()})


# ---------------------------------------------------------------- halt rule and structure

def cohort(n_units_fail):
    """153 recordings: 111 t1 and 42 t2 (as ds003775); the first n_units_fail t1 recordings fail the units check."""
    dec = {}
    for i in range(111):
        dec[(f"sub-{i:03d}", S1)] = excluded("units_check") if i < n_units_fail else kept()
    for i in range(42):
        dec[(f"sub-{i + 60:03d}", S2)] = kept()
    assert len(dec) == 153
    return dec


def test_halt_rule_at_its_boundary_8_of_153_halts_7_does_not():
    assert preprocess.apply_exclusions(CFG, cohort(8))["units_check_halt"]["halt"] is True
    seven = preprocess.apply_exclusions(CFG, cohort(7))["units_check_halt"]
    assert seven["halt"] is False and seven["n_failed"] == 7 and seven["n_total"] == 153


def test_apply_exclusions_structure_and_counts():
    dec = cohort(0)
    dec[("sub-100", S1)] = excluded("bad_electrode")
    dec[("sub-101", S2)] = excluded("insufficient_clean_data")
    out = preprocess.apply_exclusions(CFG, dec)
    assert out["schema"] == 1 == CFG["preprocessing"]["exclusions"]["schema_version"] and len(out["recordings"]) == 153
    assert out["recordings"]["sub-100/ses-t1"]["reason"] == "bad_electrode"
    assert out["subjects"]["sub-100"]["in_c1"] is False
    assert out["subjects"]["sub-101"]["in_c1"] is True and out["subjects"]["sub-101"]["in_c3"] is False
    assert len(out["subjects"]) == 111


def test_write_exclusions_layout_utf8_lf_sorted(tmp_path):
    out = preprocess.apply_exclusions(CFG, cohort(0))
    path = tmp_path / "outputs" / "exclusions.json"
    preprocess.write_exclusions(path, out, variant={"strict": False, "sensitivity_highpass": False},
                                config_sha256="c" * 64, manifest_summary="# files=632")
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    doc = json.loads(raw.decode("utf-8"))
    assert list(doc) == sorted(doc)
    assert doc["variant"] == {"strict": False, "sensitivity_highpass": False}
    assert doc["config_sha256"] == "c" * 64 and doc["units_check_halt"]["halt"] is False
    assert set(doc["subjects"]["sub-000"]) == {"t1", "t2", "in_c1", "in_c3"}


def test_exclusions_path_is_a_config_leaf_and_no_test_writes_to_real_outputs():
    assert CFG["paths"]["exclusions_file"] == "outputs/exclusions.json"
