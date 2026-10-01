"""E6 tests: the operating-point and chain override of tools/e6_operating_point_diagnostic.py (diagnostic only). Expected
values are hand-derived from the stated C7-style default; the filters are not run here."""
import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

import e6_operating_point_diagnostic as e6  # noqa: E402
import test_pilot_driver as pd  # noqa: E402
from src import synthetic_gate as sg  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()
TOOL = Path(e6.__file__)


def _op_stub(cfg, grid, table, stream, i, round_=0):
    return {"series": i, "recording": "sub-x", "target": [11.0, 0.4, 1.6], "distance": 0.25, "p": 120.0, "input_sd_factor": 2.0,
            "noise_share": 0.2, "regime": "limit_cycle", "index": 7}


def _grid(exponent=9.9):
    return sg.Grid(p=np.zeros(1), sd_factor=np.zeros(1), share=np.zeros(1), features=np.zeros((1, 3)), regime=["x"],
                   exponent=exponent, key="k")


def test_fixed_point_is_the_c7_style_default_derived_from_config():
    fp = e6.fixed_point(CFG)
    assert fp["p"] == 220.0 and fp["input_sd_factor"] == 1.0 and fp["noise_share"] == 0.5
    assert fp["exponent"] == pytest.approx(1.6)                         # power exponent for the slope -1.6
    # C7: additive noise = 0.25 white (R) + 1/f at share 0.5 of (1 + 0.25) = 1.25 -> 1/f part 1.25 / 1.5
    assert fp["one_over_f_fraction"] == pytest.approx(1.25 / 1.5)
    assert CFG["ukf"]["observation_noise"]["R_fraction_of_rescaled_variance"] == 0.25


def test_noise_split_and_exponent_overrides_do_not_touch_the_original_config_or_grid():
    g = _grid(1.2)
    c, g2 = e6.fixed_cfg_and_grid(CFG, g)
    assert c["g0"]["regime"]["one_over_f_fraction_of_noise"] == pytest.approx(5 / 6) and g2.exponent == pytest.approx(1.6)
    assert CFG["g0"]["regime"]["one_over_f_fraction_of_noise"] == 0.8 and g.exponent == 1.2


def test_z_distance_by_hand():
    assert e6.z_distance([10.0, 0.5, 1.6], [11.0, 0.4, 1.6], [2.0, 0.2, 0.1]) == pytest.approx(np.sqrt(0.5 ** 2 + 0.5 ** 2))
    assert e6.z_distance([1, 1, 1], [1, 1, 1], [1, 1, 1]) == 0.0


def test_override_sets_the_fixed_point_keeps_the_draw_and_restores_both_hooks():
    orig_op, orig_pre = sg.series_operating_point, sg.preprocess_series
    feats = np.array([10.0, 0.5, 1.6])
    with e6.override(CFG, feats, np.array([2.0, 0.2, 0.1]), "noise_driven", light=False, original_op=_op_stub):
        assert sg.series_operating_point is not orig_op and sg.preprocess_series is orig_pre
        o = sg.series_operating_point(CFG, None, None, "pilot", 3)
    assert (o["p"], o["input_sd_factor"], o["noise_share"], o["regime"]) == (220.0, 1.0, 0.5, "noise_driven")
    assert o["recording"] == "sub-x" and o["target"] == [11.0, 0.4, 1.6] and o["matched_distance_e5"] == 0.25
    assert o["distance"] == pytest.approx(np.sqrt(0.5 ** 2 + 0.5 ** 2)) and o["fixed_operating_point"] is True
    assert sg.series_operating_point is orig_op and sg.preprocess_series is orig_pre
    with pytest.raises(RuntimeError):
        with e6.override(CFG, feats, np.ones(3), "limit_cycle", light=True, original_op=_op_stub):
            assert sg.preprocess_series is e6.light_preprocess
            raise RuntimeError
    assert sg.series_operating_point is orig_op and sg.preprocess_series is orig_pre        # restored after an error


def _generate(light, monkeypatch):
    """One short positive series through generate_series under the override, with the noise and simulation inputs captured."""
    seen = {}
    real_obs, real_run = sg.observe_noisy, sg.run_planted

    def obs(cfg, y, fs, m, share, exponent, rng):
        seen.update(share=share, exponent=exponent, frac=cfg["g0"]["regime"]["one_over_f_fraction_of_noise"], m=m)
        return real_obs(cfg, y, fs, m, share, exponent, rng)

    def run(cfg, u, g12, g21, p, c_rel, burn):
        seen.update(p=np.asarray(p).tolist())
        return real_run(cfg, u, g12, g21, p, c_rel, burn)

    monkeypatch.setattr(sg, "observe_noisy", obs)
    monkeypatch.setattr(sg, "run_planted", run)
    c = sg.g0_cfg(CFG)
    c["g0"]["series_duration_s"] = 12
    table = SimpleNamespace(target_sd_uv=10.0, scale=np.ones(3))
    cc, gg = e6.fixed_cfg_and_grid(c, _grid())
    with e6.override(cc, np.array([10.0, 0.5, 1.6]), table.scale, "limit_cycle", light, original_op=_op_stub):
        s = sg.generate_series(cc, "positive", 2, gg, table, "pilot")
    return s, seen


def test_generation_under_the_override_uses_p_220_share_half_exponent_and_split(monkeypatch):
    s, seen = _generate(False, monkeypatch)
    assert seen["p"] == [220.0, 220.0] and seen["share"] == 0.5 and seen["exponent"] == pytest.approx(1.6)
    assert seen["frac"] == pytest.approx(5 / 6) and 0.1 <= seen["m"] <= 0.4
    assert s.operating_point["p"] == 220.0 and s.operating_point["fixed_operating_point"] is True
    assert s.level_index == 2                                                  # the level cell is unchanged (i mod 4)
    assert "rejected_segments" in s.meta["preprocessing_log"]                  # the real chain ran


def test_the_light_chain_is_one_continuous_segment_at_the_observation_rate(monkeypatch):
    s, _ = _generate(True, monkeypatch)
    assert len(s.segments) == 1 and s.starts == [0] and s.segments[0].shape[0] == 2
    n = s.segments[0].shape[1]
    assert s.meta["preprocessing_log"].get("light_chain") is True and s.meta["clean_s"] == pytest.approx(n / 256)
    assert abs(n / 256 - 12.0) < 1.0 and np.isfinite(s.segments[0]).all()                       # 12 s kept after the trims
    mu, sd = CFG["rescaling"]["mu_ref"], CFG["rescaling"]["sigma_ref"]
    assert s.segments[0].mean(axis=1) == pytest.approx([mu, mu], rel=1e-9) and s.segments[0].std(axis=1, ddof=1) == pytest.approx([sd, sd], rel=1e-9)


def test_build_sets_counts_and_scopes_the_override_to_the_call(monkeypatch):
    calls = []
    orig_op, orig_pre = sg.series_operating_point, sg.preprocess_series

    def fake_generate(cfg, arm, i, grid, table, stream, *a, **k):
        calls.append((arm, i, stream, sg.preprocess_series is e6.light_preprocess, grid.exponent))
        return SimpleNamespace(arm=arm, index=i)

    monkeypatch.setattr(sg, "generate_series", fake_generate)
    monkeypatch.setattr(sg, "tuning_set", lambda c, g, t: [SimpleNamespace(arm="positive", index=i) for i in range(20)])
    table = SimpleNamespace(scale=np.ones(3))
    sets = e6.build_sets(CFG, _grid(), table, True, "limit_cycle", np.zeros(3))
    assert {k: len(v) for k, v in sets.items()} == {"positive": 20, "null_A": 20, "null_B": 20, "tuning": 20}
    assert [c[:3] for c in calls[:3]] == [("positive", 0, "pilot"), ("positive", 1, "pilot"), ("positive", 2, "pilot")]
    assert [c[0] for c in calls].count("null_B") == 20
    assert all(c[3] for c in calls) and all(c[4] == pytest.approx(1.6) for c in calls)
    assert sg.series_operating_point is orig_op and sg.preprocess_series is orig_pre
    calls.clear()
    e6.build_sets(CFG, _grid(), table, False, "limit_cycle", np.zeros(3))
    assert not any(c[3] for c in calls)


def test_arm_option_summary_hand_values_and_only_the_three_sets():
    sets = pd._fake_sets()
    recs = {s: [pd._stub_record((CFG, ser, "19D", 1e-2)) for ser in sets[s]] for s in e6.SETS}
    out = e6.arm_option_summary(CFG, "19D", pd._qr(1e-2), recs,
                                {"tuning_s": 1.0, "evaluation_s": 2.0, "series_busy_s": 3.0, "n_series": 60}, None)
    assert set(out["nulls"]) == {"null_A", "null_B"} and set(out["n_dropped_by_standard_rule"]) == set(e6.SETS)
    v = out["levels"]["1"]
    assert v["g_true"] == 5.4 and v["gain_error_median_smoothed"] == pytest.approx(0.10) and v["gain_error_median_filtered"] == pytest.approx(0.05)
    assert v["diagnostic_flag_off_gain_error_median_smoothed"] == pytest.approx(0.30)
    assert v["contraction_g_median"] == pytest.approx(0.8) and v["contraction_m_median"] == pytest.approx(0.4)
    na = out["nulls"]["null_A"]
    assert (na["n_out"], na["n_dropped"], na["n_inside"]) == (2, 1, 17)
    assert out["stability_all_runs"]["n_runs"] == 120 and out["tuning"]["mean_nis_at_q"] == pytest.approx(1.9)
    assert out["tuning"]["diagnostic_flag_off_tuning"] is False
    text = "\n".join(sg.format_comparison({"delta": 1.08, "options": {"19D": out}}))
    assert "null_A" in text and "gate_null" not in text


def _arms():
    a = {"levels": {"0": {"n_dropped_by_standard_rule": 5, "gain_error_median_smoothed": 0.5,
                          "diagnostic_flag_off_gain_error_median_smoothed": 1.5, "contraction_g_median": 0.1, "contraction_m_median": 0.2}},
         "nulls": {"null_A": {"n_out": 3, "n_dropped": 4, "n_inside": 13, "n": 20, "diagnostic_flag_off": {"n_out": 7}},
                   "null_B": {"n_out": 0, "n_dropped": 0, "n_inside": 20, "n": 20, "diagnostic_flag_off": {"n_out": 1}}}}
    b = copy.deepcopy(a)
    b["levels"]["0"]["n_dropped_by_standard_rule"] = 0
    b["levels"]["0"]["gain_error_median_smoothed"] = 0.12
    return {"a": {"options": {"A": a}}, "b": {"options": {"A": b}}, "c": {"options": {"A": {"levels": {}, "nulls": {}}}}}


def test_comparison_rows_and_their_text():
    rows = e6.comparison_rows(_arms())
    assert [r["row"] for r in rows] == ["L1", "null_A", "null_B"]
    assert rows[0]["a"] == {"dropped": 5, "err_sm": 0.5, "off_sm": 1.5, "c_g": 0.1, "c_m": 0.2} and rows[0]["b"]["err_sm"] == 0.12
    assert rows[0]["c"] is None and rows[1]["a"] == {"out": 3, "dropped": 4, "inside": 13, "off_out": 7, "n": 20}
    text = "\n".join(e6.format_rows(rows))
    assert re.search(r"A\s+L1\s+5\s+50%\s+150%\s+0\s+12%", text) and "out  3 drop  4 off  7" in text and "0.10/0.20" in text


def test_reference_arm_reads_the_e5_files_and_drops_the_gate_sets(tmp_path):
    d = tmp_path / "results" / "pilot"
    d.mkdir(parents=True)
    opt = {"levels": {"0": {"gain_error_median_smoothed": "inf", "x": ["-inf", "nan", "text"]}}, "nulls": {"null_A": {"n_out": 1}, "null_B": {"n_out": 2}, "gate_null": {"n_out": 3}}}
    (d / CFG["g0"]["pilot"]["comparison_file"]).write_text(json.dumps({"delta": 1.08, "options": {"19D": opt}}), encoding="utf-8")
    z = {"z_distance": {"per_series": [0.5, 1.5]}}
    (d / "gate_19D.json").write_text(json.dumps({"report": {"arms": {s: z for s in e6.SETS}}}), encoding="utf-8")
    ref = e6.reference_arm(tmp_path, CFG)
    assert set(ref["options"]["19D"]["nulls"]) == {"null_A", "null_B"} and ref["z_distance_per_series"]["positive"] == [0.5, 1.5]
    assert ref["delta"] == 1.08
    lv = ref["options"]["19D"]["levels"]["0"]
    assert lv["gain_error_median_smoothed"] == float("inf") and lv["x"][0] == float("-inf") and np.isnan(lv["x"][1]) and lv["x"][2] == "text"
    assert e6.unjson({"a": [{"b": "inf"}], "c": 3, "d": "infinity"}) == {"a": [{"b": float("inf")}], "c": 3, "d": "infinity"}


def test_outputs_stay_under_results_pilot_and_the_tool_has_no_forbidden_calls(tmp_path):
    p = e6.write_json(CFG, tmp_path, {"x": 1}, "e6_x.json")
    assert p == tmp_path / "results" / "pilot" / "e6_x.json" and json.loads(p.read_text(encoding="utf-8")) == {"x": 1}
    src = TOOL.read_text(encoding="utf-8")
    assert not re.search(r"np\.random\.(?!default_rng)", src) and "gate_file" not in src and "check_gate" not in src
    with pytest.raises(sg.GateError):
        sg.write_pilot_json(CFG, tmp_path, {}, tmp_path / "outputs" / "gate.json")
