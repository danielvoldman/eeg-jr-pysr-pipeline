"""E8 tests: the helpers of tools/e8_real_pilot_comparison.py (diagnostic, real pilot data). Expected values are hand tables or
closed forms; the worker is exercised on a short SYNTHETIC recording (no real data is read by the tests)."""
import contextlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

import e8_real_pilot_comparison as e8  # noqa: E402
import sim_data  # noqa: E402
from src import state_space as ss  # noqa: E402
from src import synthetic_gate as sg  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()
TOOL = Path(e8.__file__)
LAYOUT = ss.make_layout(CFG)
PRIOR = ss.prior_mean(LAYOUT, CFG)
PSD = dict(ss.parameter_prior_sd(CFG))
PSD["m"] = CFG["priors"]["m_sd"]


def _col(name):
    return list(LAYOUT.names).index(name)


def test_med_max_by_hand():
    assert e8.med_max([1.0, 5.0, 2.0]) == (2.0, 5.0) and e8.med_max([4.0, 2.0]) == (3.0, 4.0)
    assert e8.med_max([]) == (None, None) and e8.med_max([None, 7.0]) == (7.0, 7.0)


def test_param_excursions_in_prior_sd_with_the_m_clip_and_the_group_maximum():
    X = np.tile(PRIOR, (3, 1))
    X[1, _col("p1")] += 2.0 * PSD["p1"]
    X[1, _col("p2")] -= 1.0 * PSD["p2"]                      # smaller than p1: the group takes the larger
    X[2, _col("g12")] -= 3.0 * PSD["g12"]
    X[2, _col("g21")] += 4.0 * PSD["g21"]                     # the larger member decides the group
    X[2, _col("log_rho2")] += 1.5 * PSD["log_rho2"]
    X[2, _col("m")] += 10.0 * PSD["m"]                       # far outside the truncation range: clipped first
    out = e8.param_excursions(LAYOUT, X, CFG)
    lo, hi = CFG["priors"]["m_truncate"]
    assert out["p"] == pytest.approx(2.0) and out["g"] == pytest.approx(4.0) and out["log_rho"] == pytest.approx(1.5)
    assert out["m"] == pytest.approx((hi - PRIOR[_col("m")]) / PSD["m"])
    assert set(out) == {"p", "log_rho", "g", "m"}


def _states(neural, params=None):
    x = PRIOR.copy()
    x[:ss.N_NEURAL] = neural
    if params:
        for k, v in params.items():
            x[_col(k)] = v
    return x


def test_reference_deviations_at_the_prior_reference_are_in_prior_sd_units():
    sd = np.sqrt(np.tile(ss.neural_variance(CFG), ss.N_NODES))
    c0 = np.asarray(ss.initial_neural_state(CFG))
    shifted = c0.copy()
    shifted[1] += 3.0 * sd[1]
    shifted[7] -= 4.5 * sd[7]
    X = np.vstack([_states(c0), _states(shifted)])
    dev = e8.reference_deviations(LAYOUT, X, CFG)
    assert np.allclose(dev[0], 0.0) and dev[1, 1] == pytest.approx(3.0) and dev[1, 7] == pytest.approx(4.5)
    assert np.count_nonzero(dev[1] > 1e-9) == 2


def test_reference_deviations_follow_the_reference_when_the_parameters_move():
    p = {"p1": 250.0, "p2": 180.0, "g12": 12.0, "g21": 7.0}
    keys = ("p1", "p2", "log_rho1", "log_rho2", "g12", "g21")
    cur = {k: float(PRIOR[_col(k)]) for k in keys}
    cur.update(p)
    center = np.asarray(ss.coupled_steady_state(CFG, *(cur[k] for k in keys)))
    sd = np.sqrt(np.tile(ss.neural_variance(CFG), ss.N_NODES))
    x = center.copy()
    x[2] += 2.0 * sd[2]
    dev = e8.reference_deviations(LAYOUT, np.vstack([_states(x, p)]), CFG)
    assert dev[0, 2] == pytest.approx(2.0) and np.allclose(np.delete(dev[0], 2), 0.0, atol=1e-9)


def test_y1_exceedance_fraction_and_maximum_over_the_kept_samples_only():
    dev = np.zeros((5, 12))
    dev[:, 1] = [1, 11, 3, 25, 2]
    dev[:, 7] = [0, 0, 12, 0, 4]
    dev[:, 3] = 99.0                                            # not a y1 state: ignored
    keep = np.array([False, True, True, True, False])
    frac, mx = e8.y1_exceedance(dev, keep)
    # kept pairs: (11, 0), (3, 12), (25, 0) -> beyond 10: 11, 12, 25 = 3 of 6
    assert frac == pytest.approx(0.5) and mx == 25.0
    assert e8.y1_exceedance(dev, np.zeros(5, bool)) == (None, None)


def _grid():
    f = np.arange(0.0, 40.5, 0.5)
    base = np.ones((2, f.size))
    obs = base.copy()
    obs[:, (f >= 8) & (f <= 13)] = 5.0
    return f, base, obs


def test_alpha_metrics_prominence_and_share_by_hand():
    f, flat, obs = _grid()
    out = e8.alpha_metrics(f, 0.5 * obs, obs, [8, 13])
    assert out[0]["prominence_observed"] == pytest.approx(5.0) and out[0]["prominence_noise"] == pytest.approx(5.0)
    assert out[1]["alpha_share_of_observed"] == pytest.approx(0.5)
    out = e8.alpha_metrics(f, flat, obs, [8, 13])                 # a flat noise state: no peak, one fifth of the alpha power
    assert out[0]["prominence_noise"] == pytest.approx(1.0) and out[0]["alpha_share_of_observed"] == pytest.approx(0.2)
    assert len(out) == 2


def _run(rec, option, q, mode, **kw):
    r = {"recording": rec, "option": option, "q": q, "mode": mode, "n_segments": 4, "n_segments_dropped": 0, "n_samples": 1000,
         "n_samples_dropped": 0, "stability": {"n_runs": 4, "n_negative_eig_steps": 0, "n_nan_inf": 0, "n_linalg_divergences": 0,
                                                "n_jitter_fallbacks": 0, "min_eig_overall": 1e-6}}
    r.update(kw)
    return r


def _doc():
    runs = []
    for i, (sid, dropped, nis, y1) in enumerate((("sub-a", 1, 1.0, 0.01), ("sub-b", 4, 2.0, 0.03), ("sub-c", 2, 3.0, 0.05))):
        runs.append(_run(sid, "19D", 0.01, "on", n_segments_dropped=dropped, n_samples_dropped=250 * dropped))
        runs.append(_run(sid, "19D", 0.01, "off", nis_mean=nis, nis_frac_gt10=0.1 * (i + 1), y1_gt10=y1, y1_max_sd=10.0 * (i + 1),
                         excursion={"p": 5.0 * (i + 1), "log_rho": 1.0, "g": 2.0, "m": 2.0},
                         gains={"g12": {"mean": float(i)}, "g21": {"mean": 2.0 * i}}))
    return {"runs": runs, "aperiodic": [{"recording": "sub-a", "s2_over_var": [0.1, 0.2], "tau": [0.3, 0.32], "tau_at_upper": [False, True]}],
            "reference": e8.REFERENCE}


def test_report_tables_median_and_max_over_the_recordings():
    text = "\n".join(e8.report_lines(_doc()))
    # rule ON: dropped segments 1, 4, 2 -> median 2 [4]; samples 25%, 100%, 50% -> median 50.0 [100.0]; one recording lost every segment
    assert re.search(r"19D\s+0\.01\s+2 \[4\]\s+50\.0 \[100\.0\]\s+1 of 3", text)
    # flag off: p excursion 5, 10, 15 -> 10.0 [15.0]; NIS 1, 2, 3 -> 2.00 [3.00]; y1 > 10 SD 0.01, 0.03, 0.05 -> 0.030 [0.050]; max y1 30.0
    assert re.search(r"19D\s+0\.01\s+10\.0 \[15\.0\]\s+1\.0 \[1\.0\]\s+2\.0 \[2\.0\]\s+2\.0 \[2\.0\]\s+2\.00 \[3\.00\]\s+0\.200 \[0\.300\]\s+0\.030 \[0\.050\]\s+20\.0 \[30\.0\]", text)
    assert re.search(r"g12 median.*\n19D\s+0\.01\s+1\.00 \( 0\.50,  1\.50\)\s+2\.00 \( 1\.00,  3\.00\)", text)       # gains across recordings
    assert "s2/var 0.100 / 0.200" in text and "at the tau upper bound: False / True" in text
    assert "0.021" in text and "39.5" in text and "2.06" in text                                      # the C4b to C4d reference line


def test_aggregate_groups_by_option_q_and_mode():
    agg = e8.aggregate(_doc()["runs"])
    assert set(agg) == {("19D", 0.01, "on"), ("19D", 0.01, "off")} and len(agg[("19D", 0.01, "on")]) == 3


# ---- the worker on a SHORT SYNTHETIC recording ----------------------------------------------------------------

def _synthetic():
    r = sim_data.make_recording(CFG, [24.0], 5, g12=8.0, g21=8.0, p=(220.0, 250.0))
    return r["segments"], r["starts"]


def test_worker_modes_set_the_state_flag_and_return_the_expected_fields(monkeypatch):
    seen = []
    real = sg.filter_context

    def spy(cfg, name, segments):
        seen.append((name, cfg["ukf"]["divergence"]["state_sd_multiple"], cfg["ukf"]["numba"]["enabled"]))
        return real(cfg, name, segments)

    monkeypatch.setattr(sg, "filter_context", spy)
    seg, st = _synthetic()
    on = e8._worker(("sub-x", seg, st, "19D", 1e-2, "on", CFG))
    off = e8._worker(("sub-x", seg, st, "A", 1e-2, "off", CFG))
    assert seen[0] == ("19D", CFG["ukf"]["divergence"]["state_sd_multiple"], True) and seen[1] == ("A", float("inf"), True)
    for r in (on, off):
        assert r["n_segments"] == 1 and r["n_samples"] == 24 * 256 and r["stability"]["n_negative_eig_steps"] == 0
        assert set(r["excursion"]) == {"p", "log_rho", "g", "m"}
    assert "y1_gt10" not in on and "alpha" not in on                              # only the flag-off runs measure these
    assert 0.0 <= off["y1_gt10"] <= 1.0 and off["y1_max_sd"] >= 0.0 and off["nis_mean"] > 0
    assert len(off["alpha"]) == 2 and off["gains"]["g12"]["q25"] <= off["gains"]["g12"]["q75"]
    # the kept samples are the ones after both burn-ins: fewer than the recording, at least the recording minus both burn-ins
    n_keep = off["n_kept"]
    burn = (CFG["windows"]["training_burn_in_s"] + CFG["passes"]["estimator_burn_in_s"]) * 256
    assert 24 * 256 - burn <= n_keep < 24 * 256


def test_a_19d_flag_off_run_equals_the_plain_pass_1_gain_estimate():
    from src import passes
    seg, st = _synthetic()
    r = e8._worker(("sub-x", seg, st, "19D", 1e-2, "off", CFG))
    c = sg.g0_cfg(CFG)
    c["ukf"]["divergence"]["state_sd_multiple"] = float("inf")
    plain = passes.run_pass1(seg, st, c, 1e-2, forward_only=True)
    # the forward-only gain estimate is the post-estimator-burn-in mean of the filtered gain; the worker uses the NIS-kept samples
    assert abs(r["gains"]["g12"]["mean"] - plain.gain_estimate["g12"]) < 3.0 and r["n_samples"] == plain.n_clean


def test_source_rules():
    src = TOOL.read_text(encoding="utf-8")
    assert not re.search(r"np\.random\.(?!default_rng)", src) and "gate_file" not in src and "check_gate" not in src
    assert "allow_all=False" in src and "write_pilot_json" in src and "ses-t1" in src
    assert e8.NIS_HIGH == 10.0 and e8.EXCEED_SD == 10.0 and e8.Q_VALUES == (1e-2, 1e-3) and e8.OPTIONS == ("19D", "A", "B") and e8.Y1 == (1, 7)
