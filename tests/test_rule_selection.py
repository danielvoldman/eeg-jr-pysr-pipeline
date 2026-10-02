"""F2 tests: the helpers and the worker of tools/f2_rule_selection.py (DEV-007, IMP-091). Expected values are closed forms or
hand tables; the worker runs on a short SYNTHETIC recording (no real data, no test subject)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

import f2_rule_selection as f2  # noqa: E402
import sim_data  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()
SEL = CFG["g0"]["rule_selection"]


def test_thresholds_are_the_predeclared_values():
    assert (SEL["e_max"], SEL["sensitivity_min"], SEL["false_positive_max"], SEL["n_per_config"], SEL["min_labelled_each"]) == (3, 0.9, 0.05, 20, 20)
    assert SEL["parameter_label_sd_multiple"] == 8 and CFG["g0"]["seeds"]["rule_selection"] == 96000
    assert SEL["runaway_q"] == [0.3, 1.0] and SEL["runaway_prior_p_displacement_sd"] == 7 and SEL["runaway_data_amplitude_factor"] == 3


def test_centred_ratio_closed_forms():
    rng = np.random.default_rng(0)
    xt = rng.normal(size=(5000, 2)) * [1.0, 4.0]
    assert np.allclose(f2.centred_state_ratios(np.zeros_like(xt) + 3.0, xt), 1.0, atol=1e-12)        # a constant predictor scores 1
    assert np.allclose(f2.centred_state_ratios(xt + 5.0, xt), 0.0, atol=1e-12)                        # a pure offset is ignored
    assert np.allclose(f2.centred_state_ratios(2.0 * xt, xt), 1.0, atol=1e-12)
    assert np.allclose(f2.centred_state_ratios(4.0 * xt, xt), 3.0, atol=1e-12)


def test_label_boundaries_and_reasons():
    ok_par = {k: 0.0 for k in f2.PARAM_KEYS}
    assert f2.runaway_label(SEL, True, np.array([2.99, 1.0]), ok_par) == ("healthy", [])
    assert f2.runaway_label(SEL, True, np.array([3.01, 1.0]), ok_par) == ("runaway", ["state_error"])
    assert f2.runaway_label(SEL, True, np.array([1.0]), dict(ok_par, p1=8.0))[0] == "healthy"
    assert f2.runaway_label(SEL, True, np.array([1.0]), dict(ok_par, g12=8.01)) == ("runaway", ["parameter_error"])
    lab, why = f2.runaway_label(SEL, False, None, None)
    assert lab == "runaway" and why == ["non_finite"]
    lab, why = f2.runaway_label(SEL, False, np.array([4.0]), dict(ok_par, m=9.0))
    assert lab == "runaway" and why == ["non_finite", "state_error", "parameter_error"]


def test_clopper_pearson_closed_forms():
    lo, hi = f2.clopper_pearson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-9)
    assert f2.clopper_pearson(0, 60)[1] == pytest.approx(1 - 0.025 ** (1 / 60), abs=1e-9)            # two-sided 95% (IMP-069 quotes the one-sided 0.0487)
    lo, hi = f2.clopper_pearson(20, 20)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 20), abs=1e-9)
    assert f2.clopper_pearson(0, 0) is None


def runs(n_run, k_run_new, n_heal, k_heal_new, k_run_old=0, k_heal_old=0):
    out = []
    for i in range(n_run):
        out.append({"config": "r", "label": "runaway", "reasons": ["state_error"], "flagged_new": i < k_run_new, "flagged_legacy": i < k_run_old})
    for i in range(n_heal):
        out.append({"config": "h", "label": "healthy", "reasons": [], "flagged_new": i < k_heal_new, "flagged_legacy": i < k_heal_old})
    return out


def test_verdict_boundaries_inclusive_and_inconclusive():
    v = lambda r: f2.verdict(f2.summarize(r, SEL), SEL)                                              # noqa: E731
    assert v(runs(20, 18, 20, 1)) == "PASS"                                                           # 0.9 and 0.05 exactly
    assert v(runs(20, 17, 20, 1)) == "FAIL"                                                           # sensitivity 0.85
    assert v(runs(20, 18, 20, 2)) == "FAIL"                                                           # FPR 0.10
    assert v(runs(40, 40, 40, 0)) == "PASS"
    assert v(runs(19, 19, 40, 0)) == "INCONCLUSIVE" and v(runs(40, 40, 19, 0)) == "INCONCLUSIVE"
    assert v(runs(20, 20, 20, 0, k_run_old=0, k_heal_old=20)) == "PASS"                                # the legacy numbers carry no weight


def test_summary_counts_unlabelled_and_reasons():
    r = runs(4, 3, 6, 1, k_run_old=1, k_heal_old=2) + [{"config": "x", "label": "unlabelled", "reasons": [], "flagged_new": True, "flagged_legacy": True}]
    s = f2.summarize(r, SEL)
    assert s["n_unlabelled"] == 1 and s["pooled"]["new"]["flagged_runaway"] == 3 and s["pooled"]["legacy"]["flagged_healthy"] == 2
    assert s["per_reason"]["state_error"] == {"n": 4, "flagged_new": 3, "flagged_legacy": 1}
    assert s["per_config"]["h"]["new"]["false_positive_rate"] == pytest.approx(1 / 6)


def test_seed_block_guard_refuses_every_g0_block_and_the_burnt_one():
    assert f2.check_seed_block(CFG) == 96000
    sd = CFG["g0"]["seeds"]
    for bad in (sd["pilot"], sd["full"], sd["tuning"], sd["preprocessing_gate"], sd["grid"], sd["pilot"] + 10000, sd["full"] + 10000,
                sd["tuning"] + 20000, sd["preprocessing_gate"] + 10000, 99000):
        with pytest.raises(ValueError):
            f2.check_seed_block(CFG, bad)
    assert f2.check_seed_block(CFG, f2.SMOKE_BLOCK) == f2.SMOKE_BLOCK


def test_configurations_follow_imp_091():
    names = [c[0] for c in f2.configurations(CFG, SEL)]
    assert names == ["healthy_matched_positive", "healthy_null_A", "healthy_null_B", "healthy_limit_cycle_worst_case",
                     "runaway_q_0.3", "runaway_q_1", "runaway_prior_p_displaced", "runaway_data_amplitude", "runaway_filter_19D"]
    spec = {n: s for n, _, s in f2.configurations(CFG, SEL)}
    assert spec["runaway_q_0.3"]["q"] == 0.3 and spec["healthy_null_A"]["q"] == CFG["ukf"]["process_noise"]["q_fixed"]
    assert spec["runaway_prior_p_displaced"]["cfg"]["priors"]["p_mean"] == pytest.approx(CFG["priors"]["p_mean"] + 7 * CFG["priors"]["p_sd"])
    assert CFG["priors"]["p_mean"] == 220.0                                                           # the nominal config is untouched
    assert spec["runaway_filter_19D"]["filter"] == "19D" and spec["runaway_data_amplitude"]["amplitude_factor"] == 3.0


def fake_series(seconds=8, seed=5, g=0.0, p=220.0):
    states, z = sim_data.simulate_stream(CFG, seconds, seed, g12=g, g21=g, m=0.2, p=p)
    return SimpleNamespace(segments=[z.T.copy()], starts=[0], states=states.reshape(len(states), 2, 6), gains=(g, g), m=0.2,
                           operating_point={"p": p}, index=0, arm="null_A", level_index=None)


def test_worker_labels_a_healthy_run_and_flags_a_blown_up_one():
    ser = fake_series()
    spec = {"cfg": CFG, "filter": "A", "q": CFG["ukf"]["process_noise"]["q_fixed"]}
    ok = f2._task((CFG, SEL, spec, ser, "healthy", "H1"))
    assert ok["label"] in ("healthy", "runaway") and ok["flagged_new"] is False and ok["flagged_legacy"] is False
    assert ok["config"] == "healthy" and ok["wall_s"] > 0
    bad = f2._task((CFG, SEL, dict(spec, amplitude_factor=40.0), ser, "blown", "H1"))
    assert bad["flagged_new"] and bad["flagged_legacy"] and bad["label"] == "runaway"
    assert bad["state_ratio_max"] > SEL["e_max"] or bad["parameter_dev_max"] > SEL["parameter_label_sd_multiple"]
    assert bad["new_steps"][0] >= bad["legacy_steps"][0]                                              # the new rule never stops earlier
