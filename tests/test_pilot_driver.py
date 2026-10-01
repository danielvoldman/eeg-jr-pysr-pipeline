"""E5 tests: the pilot driver and the DEV-005 comparison report of src/synthetic_gate.py (UKF-only stage). The filters are
stubbed, so every expected value is a hand table or a closed form; the real chain is exercised by the E2 to E4 tests and by
the pilot run itself."""
import copy
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import passes, synthetic_gate as sg, tuning
from src.config import load_config

CFG = load_config()
SRC = Path(sg.__file__)
G = (2.16, 5.4, 10.8, 27.0)
PRIOR = {"p1": 50.0, "p2": 50.0, "log_rho1": 0.2, "log_rho2": 0.2, "g12": 10.8, "g21": 10.8, "m": 0.15}
COUNTS = {"positive": 20, "null_A": 20, "null_B": 20, "gate_positive": 20, "gate_null": 20}


# ---- small pieces -------------------------------------------------------------------------------------

def _rec(level, g_true, e12=0.0, e21=0.0, *, diverged=False, f12=None, f21=None, sd_frac=0.2, m_sd_frac=0.2):
    sd = {k: sd_frac * v for k, v in PRIOR.items()}
    sd["m"] = m_sd_frac * PRIOR["m"]
    est = {"g12": g_true * (1 + e12), "g21": g_true * (1 + e21), "m": 0.25, "posterior_sd": sd,
           "g12_filt": g_true * (1 + (e12 if f12 is None else f12)), "g21_filt": g_true * (1 + (e21 if f21 is None else f21))}
    return {"level_index": level, "g_true": g_true, "diverged": diverged, "estimates": None if diverged else est}


def _null(g12f, g21f, diverged=False, off=None):
    r = {"diverged": diverged, "has_term": None, "level_index": None,
         "estimates": None if diverged else {"g12_filt": g12f, "g21_filt": g21f}}
    if off is not None:
        r["diagnostic_flag_off"] = {"diagnostic_only": True, "estimates": off}
    return r


def test_flag_off_record_reads_the_diagnostic_and_a_missing_estimate_is_diverged():
    r = dict(_rec(1, 5.4), diagnostic_flag_off={"diagnostic_only": True, "estimates": {"g12": 9.0, "g21": 9.5}})
    f = sg._flag_off_record(r)
    assert f["estimates"] == {"g12": 9.0, "g21": 9.5} and f["diverged"] is False and r["estimates"]["g12"] == pytest.approx(5.4)
    for bad in (dict(_rec(1, 5.4)), dict(_rec(1, 5.4), diagnostic_flag_off={"estimates": None})):
        g = sg._flag_off_record(bad)
        assert g["diverged"] is True and g["estimates"] is None
    assert sg.pooled_gain_errors(sg._flag_off_record(r)) == pytest.approx([(9.0 - 5.4) / 5.4, (9.5 - 5.4) / 5.4])


def test_null_counts_hand_table_rule_on_and_flag_off():
    recs = [_null(0.5, -0.2), _null(1.08, 0.0), _null(0.1, -2.0), _null(0, 0, diverged=True), _null(0.3, 0.3)]
    c = sg._null_counts(CFG, recs)
    # larger |g|: 0.5, 1.08 (= delta, out), 2.0 (out), none, 0.3 -> median of [0.3, 0.5, 1.08, 2.0] = 0.79
    assert (c["n"], c["n_out"], c["n_dropped"], c["n_inside"]) == (5, 2, 1, 2)
    assert c["median_max_abs_g"] == pytest.approx(0.79)
    offs = [_null(0, 0, off={"g12_filt": 5.0, "g21_filt": 0.0}), _null(0, 0, off={"g12_filt": 0.1, "g21_filt": 0.1}),
            _null(0, 0, diverged=True, off={"g12_filt": 0.2, "g21_filt": 1.2}), _null(0, 0), _null(0, 0, off={"g12": 1.0})]
    f = sg._null_counts(CFG, offs, flag_off=True)
    assert (f["n_out"], f["n_dropped"], f["n_inside"]) == (2, 2, 1)       # 5.0 and 1.2 out; no filtered gain at all: dropped
    assert sg._null_counts(CFG, [_null(0, 0, diverged=True)])["median_max_abs_g"] is None


def test_median_contraction_by_hand():
    rs = [_rec(1, 5.4, sd_frac=0.2), _rec(1, 5.4, sd_frac=0.5), _rec(1, 5.4, diverged=True)]
    # g (g12 and g21 pooled): 0.8, 0.8, 0.5, 0.5, 0, 0 -> median (0.5 + 0.5) / 2 = 0.5
    assert sg._median_contraction(CFG, rs, ("g12", "g21")) == pytest.approx(0.5)
    rs2 = [_rec(1, 5.4, m_sd_frac=0.4), _rec(1, 5.4, m_sd_frac=0.4), _rec(1, 5.4, diverged=True)]
    assert sg._median_contraction(CFG, rs2, ("m",)) == pytest.approx(0.6)       # 0.6, 0.6, 0
    assert sg._median_contraction(CFG, [_rec(1, 5.4, diverged=True)], ("m",)) == 0.0


def test_series_digest_is_the_sha_of_the_bytes_and_sees_every_sample():
    a = SimpleNamespace(segments=[np.arange(8.0).reshape(2, 4), np.ones((2, 3))], starts=[0, 10])
    h = hashlib.sha256()
    h.update(np.arange(8.0).reshape(2, 4).tobytes())
    h.update(np.ones((2, 3)).tobytes())
    h.update(np.array([0, 10], dtype=np.int64).tobytes())
    assert sg.series_digest(a) == h.hexdigest()
    b = copy.deepcopy(a)
    b.segments[1][1, 2] += 1e-12
    assert sg.series_digest(b) != sg.series_digest(a)
    c = copy.deepcopy(a)
    c.starts = [0, 11]
    assert sg.series_digest(c) != sg.series_digest(a)
    f32 = SimpleNamespace(segments=[np.arange(8.0, dtype=np.float32).reshape(2, 4), np.ones((2, 3))], starts=[0, 10])
    assert sg.series_digest(f32) == sg.series_digest(a)                         # cast to float64 first


def _paired(digest_for):
    return {name: {s: [{"index": i, "data_sha256": digest_for(name, s, i)} for i in range(2)] for s in sg.KIND_SETS}
            for name in ("19D", "A")}


def test_check_pairing_accepts_identical_digests_and_rejects_one_difference():
    ok = sg.check_pairing(_paired(lambda n, s, i: f"{s}{i}"))
    assert ok["paired"] and ok["n_series"] == 10 and ok["n_options"] == 2
    with pytest.raises(sg.GateError, match="same series"):
        sg.check_pairing(_paired(lambda n, s, i: "x" if (n, s, i) == ("A", "null_B", 1) else f"{s}{i}"))
    assert sg.check_pairing({}) is None


def test_estimate_runtime_by_hand():
    one = {k: {"expected_s": 10.0, "bound_s": 20.0} for k in sg.KIND_SETS}
    one["tuning"] = {"forward_one_q_s": 5.0}
    est = sg.estimate_runtime(CFG, {"19D": one, "A": one}, 360.0, 4)
    # series: 100 x 10 s = 1000 s (bound 2000 s); tuning: 20 series x 8 q x 5 s = 800 s; 4 workers; 3600 s per hour
    assert est["per_option"]["19D"]["expected_h"] == pytest.approx(1800 / 4 / 3600)
    assert est["per_option"]["19D"]["bound_h"] == pytest.approx(2800 / 4 / 3600)
    assert est["per_option"]["A"]["tuning_h"] == pytest.approx(800 / 4 / 3600)
    assert est["generation_h"] == pytest.approx(0.1)
    assert est["expected_h"] == pytest.approx(0.1 + 2 * 0.125) and est["bound_h"] == pytest.approx(0.1 + 2 * 2800 / 14400)


def _clock(monkeypatch):
    t = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: t[0])
    return t


def test_time_series_passes_expected_and_conservative_bound(monkeypatch):
    t = _clock(monkeypatch)
    calls = []
    series = SimpleNamespace(segments=[np.zeros((2, 8))], starts=[0])

    def p1(segments, starts, cfg, q, **kw):
        off = cfg["ukf"]["divergence"]["state_sd_multiple"] == float("inf")
        t[0] += 2.0 if off else 3.0
        calls.append(("p1", "off" if off else "on"))
        return SimpleNamespace(params="P", recording_diverged=(not off) and state["diverge_on"], segments=[])

    def p2(segments, starts, params, cfg, q, **kw):
        t[0] += 7.0
        calls.append(("p2", "off" if cfg["ukf"]["divergence"]["state_sd_multiple"] == float("inf") else "on"))

    monkeypatch.setattr(passes, "run_pass1", p1)
    monkeypatch.setattr(passes, "run_pass2", p2)
    state = {"diverge_on": False}
    a = sg.time_series_passes(CFG, series, "19D", 1e-2)
    assert (a["p1_on_s"], a["p2_on_s"], a["p1_off_s"], a["p2_bound_s"]) == (3.0, 7.0, 2.0, None)
    assert a["expected_s"] == 12.0 and a["bound_s"] == 12.0 and calls == [("p1", "on"), ("p2", "on"), ("p1", "off")]
    calls.clear()
    state["diverge_on"] = True                       # pass 2 skipped under the standard rule: timed on the flag-off parameters
    b = sg.time_series_passes(CFG, series, "19D", 1e-2)
    assert b["p2_on_s"] is None and b["p2_bound_s"] == 7.0 and b["diverged_pass1"] is True
    assert b["expected_s"] == 5.0 and b["bound_s"] == 12.0 and calls == [("p1", "on"), ("p1", "off"), ("p2", "off")]


def test_pool_size_never_exceeds_the_affinity_mask(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil.Process, "cpu_affinity", lambda self: [0, 1])
    assert sg.pool_size(CFG) == 2 and sg.pool_size(CFG, 1) == 1 and sg.pool_size(CFG, 8) == 2
    monkeypatch.setattr(psutil.Process, "cpu_affinity", lambda self: list(range(8)))
    assert sg.pool_size(CFG) == CFG["compute"]["joblib_n_jobs"] == 4 and sg.pool_size(CFG, 6) == 6


def test_pilot_files_are_written_under_results_pilot_only(tmp_path):
    p = sg.write_pilot_json(CFG, tmp_path, {"a": float("inf"), "b": np.float64(1.5)}, tmp_path / "results" / "pilot" / "x.json")
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": "inf", "b": 1.5}
    for bad in (tmp_path / "outputs" / "gate.json", tmp_path / "x.json", tmp_path / "results" / "x.json"):
        with pytest.raises(sg.GateError):
            sg.write_pilot_json(CFG, tmp_path, {}, bad)
    assert not (tmp_path / "outputs").exists()


# ---- the driver with stubbed series and filters ------------------------------------------------------------

def _fake_sets():
    sets = {}
    for s, n in COUNTS.items():
        sets[s] = [SimpleNamespace(arm=s, index=i, level_index=i % 4 if "positive" in s else None, stream="pilot",
                                   segments=[np.full((2, 4), float(i))], starts=[0]) for i in range(n)]
    sets["tuning"] = [SimpleNamespace(arm="positive", index=i, segments=[np.zeros((2, 4))], starts=[0]) for i in range(20)]
    return sets


def _stub_record(payload):
    """What the worker would return, from a hand rule per option: option 19D has 10% / 5% positive errors (30% flag off),
    option A has 2% / 1% (4% flag off); Null A of 19D has 2 series out and 1 dropped, flag off 5 out."""
    cfg, series, name, q = payload
    pos = "positive" in series.arm
    base = {"arm": series.arm, "index": series.index, "stream": series.stream, "filter": name, "q": q,
            "level_index": series.level_index, "z_distance": 0.5 + 0.1 * series.index, "regime": "limit_cycle",
            "diverged_pass1": False, "stability": {"n_runs": 2, "n_negative_eig_steps": 0, "n_nan_inf": 0, "n_linalg_divergences": 0,
                                                    "n_jitter_fallbacks": 1, "min_eig_overall": 1e-6},
            "has_term": None, "nrmse": None, "linear_floor": 0.3 if pos else None, "runtime_s": 2.0,
            "data_sha256": sg.series_digest(series), "n_segments": 4, "n_segments_dropped_pass1": 0, "diverged_fraction_pass1": 0.0}
    e_sm, e_ft, e_off = (0.10, 0.05, 0.30) if name == "19D" else (0.02, 0.01, 0.04)
    if pos:
        g = G[series.level_index]
        r = dict(_rec(series.level_index, g, e_sm, -e_sm, f12=e_ft, f21=e_ft, m_sd_frac=0.6), **base)
        r["estimates"]["rho1"] = r["estimates"]["rho2"] = 0.148
        off = {k: g * (1 + e_off) for k in ("g12", "g21", "g12_filt", "g21_filt")}
        off["posterior_sd"] = r["estimates"]["posterior_sd"]
        r["diagnostic_flag_off"] = {"diagnostic_only": True, "estimates": off}
        return r
    out = name == "19D" and series.arm == "null_A" and series.index < 2
    dead = name == "19D" and series.arm == "null_A" and series.index == 2
    off = {"g12_filt": 5.0 if series.index < 5 and series.arm == "null_A" else 0.2, "g21_filt": 0.1}
    r = dict(_null(2.0 if out else 0.3, 0.1, diverged=dead, off=off), **base)
    r["g_true"] = 0.0
    return r


def _qr(q, refused=False):
    table = [{"q": float(x), "mean_nis": 1.0 + 0.3 * j, "n_diverged": j} for j, x in enumerate(tuning.q_grid(CFG))]
    return SimpleNamespace(q=q, q_index=None if q is None else 3, refused=refused, refusal_reason="stub refusal" if refused else None,
                           target=2.0, band=0.5, in_band=True, at_grid_edge=False, nis_spread=0.1, n_recordings=20, n_matched=20,
                           table=table)


def _fake_timing(expected_s, bound_s=None):
    one = {k: {"p1_on_s": 1.0, "p2_on_s": 1.0, "p1_off_s": 1.0, "p2_bound_s": None, "expected_s": expected_s,
               "bound_s": bound_s or expected_s} for k in sg.KIND_SETS}
    one["tuning"] = {"forward_one_q_s": 1.0}
    return one


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["g0"]["filter_options"] = ["19D", "A", "B"]
    monkeypatch.setattr(sg, "pilot_inputs", lambda c, r: (None, None))
    monkeypatch.setattr(sg, "pilot_series_sets", lambda c, g, t: _fake_sets())
    monkeypatch.setattr(sg, "time_option", lambda c, name, sets: _fake_timing(10.0))
    monkeypatch.setattr(sg, "_series_worker", _stub_record)
    seen = []

    def tune(c, series, name, cache_dir=None, n_jobs=1, **kw):
        flag_off = c["ukf"]["divergence"]["state_sd_multiple"] == float("inf")
        seen.append((name, flag_off))
        if name == "B" or (name == "A" and not flag_off):         # B refuses either way; A only under the standard rule
            return _qr(None, True)
        return _qr(1e-2 if name == "19D" else 1e-3)

    monkeypatch.setattr(sg, "tune_g0_q", tune)
    cfg["_seen_tuning_calls"] = seen
    return cfg, tmp_path


def test_driver_stops_before_any_evaluation_when_the_estimate_is_above_the_limit(stubbed, monkeypatch):
    cfg, root = stubbed
    monkeypatch.setattr(sg, "time_option", lambda c, name, sets: _fake_timing(10.0, bound_s=1.0e5))
    called = []
    monkeypatch.setattr(sg, "tune_g0_q", lambda *a, **k: called.append(1))
    with pytest.raises(sg.PilotStop, match="exceeds"):
        sg.run_pilot(cfg, root, n_jobs=1)
    assert not called
    cmp_path, timing_path = sg.comparison_paths(cfg, root)
    assert timing_path.is_file() and not cmp_path.exists() and not (root / "outputs").exists()
    assert json.loads(timing_path.read_text(encoding="utf-8"))["estimate"]["bound_h"] > cfg["g0"]["pilot"]["max_estimated_hours"]


def test_driver_estimate_only_runs_nothing_after_the_estimate(stubbed, monkeypatch):
    cfg, root = stubbed
    monkeypatch.setattr(sg, "tune_g0_q", lambda *a, **k: pytest.fail("tuning must not run"))
    out = sg.run_pilot(cfg, root, n_jobs=1, estimate_only=True)
    assert set(out) == {"estimate"} and out["estimate"]["bound_h"] < cfg["g0"]["pilot"]["max_estimated_hours"]
    assert not sg.comparison_paths(cfg, root)[0].exists()


def test_driver_report_values_files_and_the_printed_table(stubbed, capsys):
    cfg, root = stubbed
    rep = sg.run_pilot(cfg, root, n_jobs=1)
    cmp_path, _ = sg.comparison_paths(cfg, root)
    doc = json.loads(cmp_path.read_text(encoding="utf-8"))
    assert doc["pilot"] is True and doc["stage"] == "UKF-only (no PySR)" and "not decided" in doc["dev005_decision"]
    assert doc["pairing"]["paired"] and doc["pairing"]["n_options"] == 2 and doc["pairing"]["n_series"] == 100
    o = doc["options"]["19D"]
    assert o["tuning"]["q"] == pytest.approx(1e-2) and o["tuning"]["mean_nis_at_q"] == pytest.approx(1.9) and o["tuning"]["in_band"]
    for lv in "0123":
        v = o["levels"][lv]
        assert v["g_true"] == G[int(lv)] and v["n"] == 5 and v["n_dropped_by_standard_rule"] == 0
        assert v["gain_error_median_smoothed"] == pytest.approx(0.10) and v["gain_error_median_filtered"] == pytest.approx(0.05)
        assert v["diagnostic_flag_off_gain_error_median_smoothed"] == pytest.approx(0.30)
        assert v["contraction_g_median"] == pytest.approx(0.8) and v["contraction_m_median"] == pytest.approx(0.4)
        assert v["detection_floor"] is (lv == "0")
    # level 0 holds series 0, 4, 8, 12, 16: z = 0.5 + 0.1 i -> median 1.3, max 2.1
    assert o["levels"]["0"]["z_distance"] == {"median": pytest.approx(1.3), "max": pytest.approx(2.1)}
    na = o["nulls"]["null_A"]
    assert (na["n_out"], na["n_dropped"], na["n_inside"]) == (2, 1, 17) and na["median_max_abs_g"] == pytest.approx(0.3)
    assert na["diagnostic_flag_off"]["n_out"] == 5 and na["diagnostic_flag_off"]["diagnostic_only"] is True
    assert o["n_dropped_by_standard_rule"] == {"positive": 0, "null_A": 1, "null_B": 0, "gate_positive": 0, "gate_null": 0}
    assert o["nulls"]["null_B"]["n_out"] == 0 and o["nulls"]["gate_null"]["n"] == 20
    assert o["stability_all_runs"]["n_runs"] == 200 and o["stability_all_runs"]["n_jitter_fallbacks"] == 100
    assert o["runtime"]["n_series"] == 100 and o["runtime"]["series_busy_s"] == pytest.approx(200.0)
    a = doc["options"]["A"]
    assert a["levels"]["1"]["gain_error_median_smoothed"] == pytest.approx(0.02) and a["nulls"]["null_A"]["n_out"] == 0
    # option B: q refused, so no evaluation and no gate document
    assert doc["options"]["B"]["tuning"]["refused"] is True and "levels" not in doc["options"]["B"]
    assert doc["options"]["B"]["tuning"]["standard_rule"]["n_diverged_per_q"][7] == {"q": pytest.approx(0.1), "n_diverged": 7}
    # tuning calls: 19D once (standard rule); A standard rule then the flag-off diagnostic; B both, refused both times
    assert cfg["_seen_tuning_calls"] == [("19D", False), ("A", False), ("A", True), ("B", False), ("B", True)]
    assert o["tuning"]["diagnostic_flag_off_tuning"] is False and o["tuning"]["standard_rule_refusal"] is None
    ta = a["tuning"]
    assert ta["diagnostic_flag_off_tuning"] is True and ta["q"] == pytest.approx(1e-3)
    assert [d["n_diverged"] for d in ta["standard_rule_refusal"]["n_diverged_per_q"]] == list(range(8))
    assert set(doc["gate_documents"]) == {"19D", "A"}
    # gate documents: pilot only, results/pilot only, never outputs/gate.json, and they cannot unlock real fitting
    for name, path in doc["gate_documents"].items():
        assert Path(path) == root / "results" / "pilot" / f"gate_{name}.json"
        g = json.loads(Path(path).read_text(encoding="utf-8"))
        assert g["pilot"] is True and g["hard_stop"] is False and g["filter"] == name and not g["complete"]
    assert not (root / "outputs").exists()
    with pytest.raises(tuning.GateError):
        tuning.check_gate(cfg, root)
    g19 = json.loads((root / "results" / "pilot" / "gate_19D.json").read_text(encoding="utf-8"))
    # 19D: null A has 3 failures in 20 (a hard stop under a full run); the positive gain criterion holds (10%) but NRMSE is pending
    assert g19["would_hard_stop"] is True and g19["verdicts"]["null_A"] is False and g19["verdicts"]["positive"] is None
    assert g19["verdicts"]["contraction"] is False                                  # median m contraction 0.4 < 0.5
    ga = json.loads((root / "results" / "pilot" / "gate_A.json").read_text(encoding="utf-8"))
    assert ga["q"]["diagnostic_flag_off_tuning"] is True and g19["q"]["diagnostic_flag_off_tuning"] is False
    assert ga["would_hard_stop"] is False and ga["verdicts"]["null_A"] is None and ga["verdicts"]["positive"] is None   # PySR pending
    text = capsys.readouterr().out
    assert text.count("== option") == 3 and "q refused" in text and "DIAGNOSTIC ONLY" in text
    assert text.count("q is DIAGNOSTIC (tuned with the state-SD flag disabled)") == 1
    assert len(re.findall(r"tuned q \S+ \(refused \w+\) in 0\.0 min", text)) == 3          # minutes, not a precedence slip
    assert "series diverged per q [0, 1, 2, 3, 4, 5, 6, 7] of 20" in text
    row = re.search(r"^\s+L2\s+5\.4\s+5\s+0\s+10%\s+5%\s+30%\s+30%\s+0\.80\s+0\.40\s+\S+/\S+$", text, re.M)
    assert row, text
    assert re.search(r"null_A\s+2/20\s+1\s+17\s+0\.30\s+5/20\s+0", text)
    assert re.search(r"L1\*", text) and rep["options"]["19D"]["levels"]["1"]["g_true"] == G[1]


def test_format_comparison_marks_the_diagnostic_columns_and_the_refusal():
    rep = {"delta": 1.08, "options": {"B": {"tuning": {"q": None, "refusal_reason": "why"}}}}
    lines = sg.format_comparison(rep)
    assert lines == ["== option B: q refused, no evaluation (why) =="]


def test_e5_source_and_config_rules():
    src = SRC.read_text(encoding="utf-8")
    e5 = src.split("# ---------------------------------------------------------------- E5")[1].split("# ---------------------------------------------------------------- the E1 report")[0]
    assert not re.search(r"np\.random\.(?!default_rng)", src) and "gate_file" not in e5 and "check_gate" not in e5
    p = CFG["g0"]["pilot"]
    assert p["tuning_fallback"] == "flag_off_nis" and p["max_estimated_hours"] == 10 and p["timing_series_index"] == 2 and p["timing_q"] == 1.0e-2
    raw = (SRC.parent.parent / "config.yml").read_text(encoding="utf-8")
    block = raw.split("  pilot:\n", 1)[1].split("\n\n", 1)[0]
    assert block.count("prov: placeholder") == 9 and "unset" not in block
