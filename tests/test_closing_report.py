"""Closing report checks (IMP-099): docs/G0_HARD_STOP_REPORT.md against its sources.

Two marker kinds in the report:
  <!--src FILE PATH=VALUE-->   a field of a result file (results/ and outputs/ are not committed, so these tests skip
                               with an explicit reason when the file is absent)
  <!--doc FILE::TEXT-->        text that must occur in a tracked document (always checked)
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "G0_HARD_STOP_REPORT.md"

SRC = re.compile(r"<!--src (\S+) (\S+)=(\S+)-->")
DOC = re.compile(r"<!--doc (\S+?)::(.+?)-->")
SHA_ROW = re.compile(r"^\| `([^`]+)`[^|]*\| `([0-9a-f]{64})` \|$", re.M)
FORBIDDEN = [r"\binvents?\b", r"\bcreates? (physical )?coupling", r"\bhallucinat", r"\bproves? (that )?real EEG"]


def report_text():
    return REPORT.read_text(encoding="utf-8")


def lookup(doc, path):
    cur = doc
    for part in path.split("."):
        cur = cur[part]
    return cur


def matches(actual, expected):
    if isinstance(actual, bool):
        return str(actual) == expected
    if isinstance(actual, (int, float)):
        decimals = len(expected.split(".")[1]) if "." in expected else 0
        return round(float(actual), decimals) == round(float(expected), decimals)
    return str(actual) == expected


def test_report_exists_and_has_markers():
    text = report_text()
    assert len(SRC.findall(text)) >= 25
    assert len(DOC.findall(text)) >= 12


def test_doc_markers_occur_in_tracked_documents():
    missing = []
    for fname, needle in DOC.findall(report_text()):
        path = ROOT / fname
        assert path.is_file(), fname
        if needle not in path.read_text(encoding="utf-8"):
            missing.append((fname, needle))
    assert not missing


def test_src_markers_match_result_files():
    entries = SRC.findall(report_text())
    files = sorted({e[0] for e in entries})
    absent = [f for f in files if not (ROOT / f).is_file()]
    if absent:
        pytest.skip(f"result files absent (results/ and outputs/ are not committed): {absent}")
    bad = []
    for fname, path, expected in entries:
        doc = json.loads((ROOT / fname).read_text(encoding="utf-8"))
        actual = lookup(doc, path)
        if not matches(actual, expected):
            bad.append((fname, path, expected, actual))
    assert not bad


def test_sha256_table_matches_files():
    rows = SHA_ROW.findall(report_text())
    assert len(rows) >= 4
    absent = [f for f, _ in rows if not (ROOT / f).is_file()]
    if absent:
        pytest.skip(f"result files absent (results/ and outputs/ are not committed): {absent}")
    for fname, digest in rows:
        assert hashlib.sha256((ROOT / fname).read_bytes()).hexdigest() == digest, fname


def test_every_file_with_a_src_marker_has_a_hash_row():
    text = report_text()
    hashed = {f for f, _ in SHA_ROW.findall(text)}
    used = {e[0] for e in SRC.findall(text)}
    assert used <= hashed


def test_every_claim_has_its_verdict_and_reason():
    text = report_text()
    for claim in ("C1", "C2", "C3", "C4"):
        row = next(l for l in text.splitlines() if l.startswith(f"| **{claim}**"))
        assert "**Not attempted**" in row
    g0 = next(l for l in text.splitlines() if l.startswith("| **G0**"))
    assert "FAILED, hard stop" in g0
    assert "not run" in g0 and "cited from the pilot" in g0


def test_wording_rule():
    text = report_text()
    assert "cannot separate coupling from zero" in text
    for pat in FORBIDDEN:
        assert not re.search(pat, text, re.I), pat


def test_pending_is_not_pass_and_unknowns_listed():
    text = report_text()
    assert "pending, not passed" in text
    assert "**Findings" in text and "**Unknown" in text


def test_hash_of_gate_matches_formal_result_hash():
    gate = ROOT / "outputs" / "gate.json"
    res = ROOT / "results" / "g0_null_arms_r1.json"
    if not (gate.is_file() and res.is_file()):
        pytest.skip("outputs/gate.json or results/g0_null_arms_r1.json absent (not committed)")
    doc = json.loads(gate.read_text(encoding="utf-8"))
    assert doc["formal_results_sha256"] == hashlib.sha256(res.read_bytes()).hexdigest()


IMP101_NUMBERS = [
    "371.66", "88 of 100", "0.252", "10.4 SD", "87 of 88", "120 or 138.2",
    "0.3 to 0.7 ruler SD", "1 to 9", "141 ms", "sub-019 20", "sub-074 141", "sub-081 31", "sub-086 31",
    "sub-088 20", "sub-107 129", "6 of 12", "12 fits", "120 s", "6/6", "0.0001 to 0.0011", "0.0016 to 0.0067",
    "3/3", "0/3", "0.19 to 0.22", "0.28 to 0.29", "0.28, 0.71, 0.70", "0.22 to 0.23", "0.08 to 0.18",
]


def imp101_row():
    text = (ROOT / "DEVIATIONS.md").read_text(encoding="utf-8")
    rows = [l for l in text.splitlines() if l.startswith("| IMP-101 |")]
    assert len(rows) == 1
    return rows[0]


def test_imp101_numbers_in_report_and_deviations():
    row = imp101_row()
    report = report_text()
    d_rows = [l for l in report.splitlines() if l.rstrip().endswith("| IMP-101 |")]
    assert len(d_rows) == 3
    d_text = " ".join(d_rows)
    # the finding in section 7 repeats the oracle result, so only the derivative range and counts are compared there
    for num in IMP101_NUMBERS:
        assert num in row, ("DEVIATIONS IMP-101", num)
    for num in IMP101_NUMBERS:
        assert num in d_text or num in report, ("report", num)
    for num in ("88 of 100", "0.252", "10.4 SD", "87 of 88", "371.66", "141 ms", "129 ms", "6 of 12", "0.0016 to 0.0067", "0.28, 0.71, 0.70"):
        assert num in d_text, ("section 4 rows", num)


def test_imp101_wording_and_unknown():
    report = report_text()
    assert "best case on clean true states and says nothing about filtered states" in report
    assert "Whether PySR recovers filtered-state residuals" in report
    assert "noise-driven regime while the data are limit-cycle" not in report
    assert "D2 smoke (IMP-060)" in report
    assert "this docs commit" in imp101_row() or re.search(r"\b[0-9a-f]{7,40}\b", imp101_row().split("|")[-2])
