"""E7 tests: the c = 0 (zero planted residual) override of tools/e7_zero_residual_diagnostic.py (diagnostic only). Expected
values are closed forms or hand tables; the filters are not run here."""
import copy
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
import e7_zero_residual_diagnostic as e7  # noqa: E402
import test_e6_operating_point as t6  # noqa: E402
from src import synthetic_gate as sg  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()
TOOL = Path(e7.__file__)


def _short(cfg):
    c = sg.g0_cfg(cfg)
    c["g0"]["series_duration_s"] = 12
    return c


def _table():
    return SimpleNamespace(target_sd_uv=10.0, scale=np.ones(3))


def test_zero_residual_cfg_changes_only_the_fraction_and_leaves_the_original_alone():
    z = e7.zero_residual_cfg(CFG)
    assert z["g0"]["planted_residual_rms_fraction_of_base"] == 0.0 and CFG["g0"]["planted_residual_rms_fraction_of_base"] == 0.5
    a, b = copy.deepcopy(z), copy.deepcopy(CFG)
    a["g0"]["planted_residual_rms_fraction_of_base"] = b["g0"]["planted_residual_rms_fraction_of_base"] = None
    assert a == b


def _positive(cfg, level_i=2):
    c, g = e6.fixed_cfg_and_grid(_short(cfg), t6._grid())
    with e6.override(c, np.array([10.0, 0.5, 1.6]), np.ones(3), "limit_cycle", False, original_op=t6._op_stub):
        return sg.generate_series(c, "positive", level_i, g, _table(), "pilot")


def test_a_zero_residual_positive_has_no_planted_term_and_a_nominal_one_has():
    s0 = _positive(e7.zero_residual_cfg(CFG))
    assert s0.meta["c"] == [0.0, 0.0] and s0.meta["rms_ratio"] == [0.0, 0.0]
    assert np.all(s0.truth["planted"] == 0.0)
    s1 = _positive(CFG)
    assert all(c > 0 for c in s1.meta["c"]) and np.any(s1.truth["planted"] != 0.0)
    # the residual is a 50% RMS term (E2: realised 0.46 to 0.53 at L1 to L3): a loose independent band
    assert all(0.3 < r < 0.7 for r in s1.meta["rms_ratio"])
    assert s0.gains == s1.gains == (10.8, 10.8) and s0.level_index == s1.level_index == 2     # same level cell


def test_the_zero_residual_series_differs_from_the_nominal_one_only_through_the_term():
    s0, s1 = _positive(e7.zero_residual_cfg(CFG)), _positive(CFG)
    n = min(s0.segments[0].shape[1], s1.segments[0].shape[1])
    assert not np.allclose(s0.segments[0][:, :n], s1.segments[0][:, :n])


def test_null_series_do_not_depend_on_the_residual_setting():
    outs = []
    for cfg in (CFG, e7.zero_residual_cfg(CFG)):
        c, g = e6.fixed_cfg_and_grid(_short(cfg), t6._grid())
        with e6.override(c, np.array([10.0, 0.5, 1.6]), np.ones(3), "limit_cycle", False, original_op=t6._op_stub):
            outs.append(sg.generate_series(c, "null_A", 1, g, _table(), "pilot"))
    assert len(outs[0].segments) == len(outs[1].segments)
    for a, b in zip(outs[0].segments, outs[1].segments):
        assert np.array_equal(a, b)


def test_build_sets_use_the_zero_residual_config_and_the_right_operating_points(monkeypatch):
    seen = []

    def fake(cfg, arm, i, grid, table, stream, *a, **k):
        seen.append((arm, i, cfg["g0"]["planted_residual_rms_fraction_of_base"], sg.series_operating_point is orig_op,
                     sg.preprocess_series is orig_pre, grid.exponent))
        return SimpleNamespace(arm=arm, index=i)

    orig_op, orig_pre = sg.series_operating_point, sg.preprocess_series
    monkeypatch.setattr(sg, "generate_series", fake)
    monkeypatch.setattr(sg, "tuning_set", lambda c, g, t: [SimpleNamespace(arm="positive", index=i) for i in range(20)])
    grid, table = t6._grid(), _table()
    fixed = e7.build_sets_fixed(CFG, grid, table, "limit_cycle", np.zeros(3))
    assert {k: len(v) for k, v in fixed.items()} == {"positive": 20, "null_A": 20, "null_B": 20, "tuning": 20}
    assert all(s[2] == 0.0 and not s[3] and s[4] and s[5] == pytest.approx(1.6) for s in seen)       # fixed point, real chain
    seen.clear()
    matched = e7.build_sets_matched(CFG, grid, table)
    assert {k: len(v) for k, v in matched.items()} == {"positive": 20, "null_A": 20, "null_B": 20, "tuning": 20}
    assert all(s[2] == 0.0 and s[3] and s[4] and s[5] == 9.9 for s in seen)                          # E5 hooks, E5 grid
    assert [s[0] for s in seen].count("null_B") == 20 and sg.series_operating_point is orig_op


def test_ratio_summary_by_hand():
    ser = [SimpleNamespace(level_index=0, meta={"rms_ratio": [0.4, 0.6]}), SimpleNamespace(level_index=0, meta={"rms_ratio": [0.5, 0.5]}),
           SimpleNamespace(level_index=3, meta={"rms_ratio": [0.8, 0.9]}), SimpleNamespace(level_index=1, meta={"rms_ratio": [0.1, 0.2]}),
           SimpleNamespace(level_index=1, meta={"rms_ratio": [0.9, 0.3]})]
    out = e7.ratio_summary(ser)
    assert out["0"] == {"n": 4, "median": 0.5, "min": 0.4, "max": 0.6}
    assert out["1"] == {"n": 4, "median": 0.25, "min": 0.1, "max": 0.9}          # median 0.25, not the midrange 0.5
    assert out["3"]["n"] == 2 and out["3"]["median"] == pytest.approx(0.85) and out["3"]["max"] == 0.9


def test_only_a_and_b_are_run_and_the_comparison_lines_label_the_two_residual_settings():
    assert e7.OPTIONS == ("A", "B") and "19D" not in e7.OPTIONS
    ref, zero = t6._arms()["a"], t6._arms()["b"]
    lines, rows = e7.compare_lines({"options": {"A": ref["options"]["A"]}}, {"options": {"A": zero["options"]["A"]}}, 1.08)
    assert "c>0" in lines[0] and "c=0" in lines[0] and "(c)" not in lines[0] and "(c>0 | c=0)" in lines[0]
    assert re.search(r"A\s+L1\s+5\s+50%\s+150%\s+0\s+12%", "\n".join(lines)) and not any(ln.endswith("| -") for ln in lines)
    assert [r["row"] for r in rows] == ["L1", "null_A", "null_B"]


def test_source_rules():
    src = TOOL.read_text(encoding="utf-8")
    assert not re.search(r"np\.random\.(?!default_rng)", src) and "gate_file" not in src and "check_gate" not in src
    assert "19D is dropped" in src and "planted_residual_rms_fraction_of_base" in src
