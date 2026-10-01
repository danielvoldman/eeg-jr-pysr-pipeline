"""E3 tests: the artifacts of the preprocessing gate and its bias scoring (§5.2) in src/synthetic_gate.py.
Expected values are closed forms, hand-built tables or independent numpy computations."""
import copy
import re
from pathlib import Path

import numpy as np
import pytest

from src import synthetic_gate as sg
from src.config import load_config

CFG = load_config()
SRC = Path(sg.__file__)
FS = 2048


def _cfg(blink_uv=30.0, emg_rms=0.30, drift_uv=50.0, emg_rate=3.0, blink_rate=0.0667):
    c = copy.deepcopy(CFG)
    pg = c["g0"]["preprocessing_gate"]
    pg["blink"] = dict(pg["blink"], amplitude_uv=blink_uv, rate_per_s=blink_rate)
    pg["emg"] = dict(pg["emg"], rms_fraction_of_channel_sd=emg_rms, rate_per_min=emg_rate)
    pg["drift"] = dict(pg["drift"], amplitude_uv=drift_uv)
    return c


def _x(T, sd=6.6, seed=0):
    return np.random.default_rng(seed).standard_normal((2, int(T * FS))) * sd


# ---- events, blinks, EMG ----------------------------------------------------------------------------

def test_event_times_are_poisson_with_the_given_rate_and_seeded():
    rate, T = 0.0667, 200_000.0
    ev = sg._event_times(np.random.default_rng(1), rate, T)
    assert len(ev) == pytest.approx(rate * T, rel=0.03)
    assert np.mean(np.diff(ev)) == pytest.approx(1.0 / rate, rel=0.03)
    assert ev == sg._event_times(np.random.default_rng(1), rate, T)
    assert all(0.0 <= t < T for t in ev)


def test_blink_wave_is_a_raised_cosine_of_30_uv_and_0_3_s():
    w = sg.blink_wave(CFG, FS)
    n = int(round(0.3 * FS))
    assert n == 614 and w.shape == (n,)
    assert w[0] == pytest.approx(0.0, abs=1e-12) and w[-1] == pytest.approx(0.0, abs=1e-12)
    assert 29.99 < w.max() <= 30.0
    for k in (50, 150, 307, 500):
        assert w[k] == pytest.approx(15.0 * (1.0 - np.cos(2.0 * np.pi * k / (n - 1))), rel=1e-12)


def test_emg_carrier_is_unit_rms_band_limited_and_ramped():
    c = _cfg()
    n = 3 * FS
    car = sg.emg_carrier(c, np.random.default_rng(2), n, FS)
    mid = car[int(0.2 * FS):-int(0.2 * FS)]
    assert np.sqrt(np.mean(mid ** 2)) == pytest.approx(1.0, rel=0.05)
    spec = np.abs(np.fft.rfft(car)) ** 2
    f = np.fft.rfftfreq(n, 1 / FS)
    assert spec[(f >= 15) & (f <= 160)].sum() / spec.sum() > 0.95
    ramp = int(round(0.05 * FS))
    assert car[0] == 0.0 and abs(car[ramp // 2]) < 3.0 * np.abs(car).max()
    assert np.sqrt(np.mean(car[:ramp // 4] ** 2)) < 0.2 * np.sqrt(np.mean(mid ** 2))


# ---- the artifact signal ----------------------------------------------------------------------------

def test_independent_drift_has_the_stated_frequency_amplitude_and_channel_phases():
    c = _cfg(blink_uv=0.0, emg_rate=1e-9, blink_rate=1e-9)
    x = _x(400)
    art, meta = sg.artifact_signal(c, np.random.default_rng(3), x, FS, "independent")
    t = np.arange(x.shape[1]) / FS
    for ch in (0, 1):
        a = 2 * np.mean(art[ch] * np.sin(2 * np.pi * 0.05 * t))
        b = 2 * np.mean(art[ch] * np.cos(2 * np.pi * 0.05 * t))
        assert np.hypot(a, b) == pytest.approx(50.0, rel=1e-6)
    ph = [np.arctan2(2 * np.mean(art[ch] * np.cos(2 * np.pi * 0.05 * t)),
                     2 * np.mean(art[ch] * np.sin(2 * np.pi * 0.05 * t))) for ch in (0, 1)]
    assert abs(ph[0] - ph[1]) > 1e-3
    assert meta["mode"] == "independent" and meta["n_blinks"] == 0


def test_independent_blink_count_and_amplitude():
    c = _cfg(emg_rate=1e-9, drift_uv=0.0)
    T = 1500.0
    art, meta = sg.artifact_signal(c, np.random.default_rng(4), _x(T), FS, "independent")
    mean = 2 * 0.0667 * T                                                   # both channels
    assert abs(meta["n_blinks"] - mean) < 4 * np.sqrt(mean)
    assert 29.0 < art.max() <= 60.0                                         # a lone blink peaks at 30 (overlaps add)
    assert art.min() >= 0.0


def test_emg_rms_burst_rate_and_band():
    c = _cfg(blink_uv=0.0, drift_uv=0.0)
    T, sd = 3000.0, 6.6
    art, meta = sg.artifact_signal(c, np.random.default_rng(5), _x(T, sd), FS, "independent")
    mean_sq_expected = 0.30 ** 2 * sd ** 2 * (3.0 / 60.0) * 2.0              # rate x mean duration (2 s) x RMS^2
    got = np.mean(art ** 2, axis=1)
    assert got == pytest.approx([mean_sq_expected] * 2, rel=0.2)             # the ramps remove a little
    assert abs(meta["n_emg"] - 2 * 3.0 / 60.0 * T) < 4 * np.sqrt(2 * 3.0 / 60.0 * T)
    spec = np.abs(np.fft.rfft(art[0])) ** 2
    f = np.fft.rfftfreq(art.shape[1], 1 / FS)
    assert spec[(f >= 15) & (f <= 160)].sum() / spec.sum() > 0.95


def test_bilateral_blinks_are_one_event_set_delayed_and_scaled():
    c = _cfg(emg_rate=1e-9, drift_uv=0.0)
    art, meta = sg.artifact_signal(c, np.random.default_rng(6), _x(300), FS, "bilateral")
    lag, f0, f1 = meta["lag_steps"], *meta["factors"]
    assert 0 <= lag <= 8 and 0.7 <= f0 <= 1.3 and 0.7 <= f1 <= 1.3
    assert meta["n_blinks"] > 5
    ref = art[0] / f0
    assert np.allclose(art[1][lag:] / f1, ref[:ref.shape[0] - lag], atol=1e-9)


def test_bilateral_emg_shares_the_carrier_but_independent_emg_does_not():
    c = _cfg(blink_uv=0.0, drift_uv=0.0)
    art, meta = sg.artifact_signal(c, np.random.default_rng(7), _x(600), FS, "bilateral")
    lag = meta["lag_steps"]
    a, b = art[0][:art.shape[1] - lag], art[1][lag:]
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(1.0, abs=1e-9)
    ind, _ = sg.artifact_signal(c, np.random.default_rng(7), _x(600), FS, "independent")
    assert abs(np.corrcoef(ind[0], ind[1])[0, 1]) < 0.05


def test_bilateral_emg_amplitude_is_30_percent_of_each_channels_own_sd_times_its_factor():
    c = _cfg(blink_uv=0.0, drift_uv=0.0)
    sd = np.array([6.6, 3.3])
    x = _x(3000) * (sd / 6.6)[:, None]
    art, meta = sg.artifact_signal(c, np.random.default_rng(11), x, FS, "bilateral")
    expected = (0.30 * sd * np.array(meta["factors"])) ** 2 * (3.0 / 60.0) * 2.0
    assert np.mean(art ** 2, axis=1) == pytest.approx(expected, rel=0.25)


def test_bilateral_drift_has_one_phase():
    c = _cfg(blink_uv=0.0, emg_rate=1e-9)
    art, meta = sg.artifact_signal(c, np.random.default_rng(8), _x(300), FS, "bilateral")
    lag, (f0, f1) = meta["lag_steps"], meta["factors"]
    a0 = art[0] / f0
    # same sine, the second channel lag samples later: art1(t) = f1 / f0 art0(t - lag)
    assert np.allclose(art[1][lag:] / f1, a0[:a0.shape[0] - lag], atol=1e-9)


def test_artifact_signal_is_seeded_and_rejects_unknown_modes():
    x = _x(60)
    a, _ = sg.artifact_signal(CFG, np.random.default_rng(9), x, FS, "independent")
    b, _ = sg.artifact_signal(CFG, np.random.default_rng(9), x, FS, "independent")
    c, _ = sg.artifact_signal(CFG, np.random.default_rng(10), x, FS, "independent")
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    with pytest.raises(sg.GateError):
        sg.artifact_signal(CFG, np.random.default_rng(1), x, FS, "bogus")


# ---- series with artifacts --------------------------------------------------------------------------

def _short():
    c = copy.deepcopy(CFG)
    c["g0"]["regime_grid_shape"] = [2, 2, 2]
    c["g0"]["regime"]["grid_sim_duration_s"] = 24
    c["g0"]["series_duration_s"] = 30
    return c


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    c = _short()
    table = sg.FeatureTable(["r0", "r1"], np.array([[10.0, 0.50, 1.4], [9.5, 0.40, 1.5]]),
                            np.array([[6.0, 7.0], [6.5, 6.6]]))
    return c, table, sg.build_grid(c, table.exponent, tmp_path_factory.mktemp("grid"), n_jobs=1)


def test_artifacts_change_only_the_observation_not_the_truth(world):
    c, table, grid = world
    plain = sg.generate_series(c, "positive", 2, grid, table, "preprocessing_gate")
    art = sg.generate_series(c, "positive", 2, grid, table, "preprocessing_gate", artifacts="independent")
    assert "artifacts" not in plain.meta and art.meta["artifacts"]["mode"] == "independent"
    assert all(np.array_equal(plain.truth[k], art.truth[k]) for k in plain.truth)
    assert plain.m == art.m and plain.gains == art.gains
    assert not np.array_equal(np.concatenate(plain.segments, axis=1)[:, :100], np.concatenate(art.segments, axis=1)[:, :100]) \
        or sum(s.shape[1] for s in art.segments) != sum(s.shape[1] for s in plain.segments)


def test_artifact_null_series_has_zero_coupling_bilateral_artifacts_and_no_truth(world):
    c, table, grid = world
    s = sg.generate_series(c, "artifact_null", 1, grid, table, "preprocessing_gate")
    a = s.meta["artifacts"]
    assert s.gains == (0.0, 0.0) and s.level_index is None and s.truth is None and s.meta["c"] == [0.0, 0.0]
    assert a["mode"] == "bilateral" and 0 <= a["lag_steps"] <= 8 and all(0.7 <= v <= 1.3 for v in a["factors"])
    assert 0.1 <= s.m <= 0.4
    with pytest.raises(sg.GateError):
        sg.generate_series(c, "positive", 0, grid, table, "pilot", artifacts="bogus")


def test_gate_set_levels_and_sizes(world):
    c, table, grid = world
    cc = copy.deepcopy(c)
    cc["g0"]["n_preprocessing_gate_positive"], cc["g0"]["n_preprocessing_gate_null"] = 8, 3
    pos, nul = sg.preprocessing_gate_set(cc, grid, table)
    assert [s.level_index for s in pos] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert all(s.stream == "preprocessing_gate" and s.meta["artifacts"]["mode"] == "independent" for s in pos)
    assert [s.arm for s in nul] == ["artifact_null"] * 3
    assert CFG["g0"]["n_preprocessing_gate_positive"] == 20 and CFG["g0"]["n_preprocessing_gate_null"] == 20
    assert sorted(i % 4 for i in range(20)).count(0) == 5                    # 5 per level at the full size


# ---- scoring ----------------------------------------------------------------------------------------

def test_null_pass_rule_is_strict_and_needs_no_term():
    d = 1.08
    assert sg.delta(CFG) == pytest.approx(d)
    assert sg.null_pass(CFG, 1.0, -1.0, False)
    assert not sg.null_pass(CFG, 1.08, 0.0, False)                           # |g| = delta is not inside
    assert not sg.null_pass(CFG, 0.2, -1.2, False)
    assert not sg.null_pass(CFG, 0.2, 0.2, True)
    assert not sg.null_pass(CFG, None, None, False)


RHO = 3.25 / 22.0


def _rec(level, g_true, e12, e21, er1=0.0, er2=0.0, nrmse=None):
    r = {"level_index": level, "g_true": g_true, "g12": g_true * (1 + e12), "g21": g_true * (1 + e21),
         "rho1": RHO * (1 + er1), "rho2": RHO * (1 + er2)}
    if nrmse is not None:
        r["nrmse"] = nrmse
    return r


def _levels(e12, e21, er1=0.0, er2=0.0, nrmse=None, per_level=2):
    out = []
    for lv, g in enumerate((2.16, 5.4, 10.8, 27.0)):
        out += [_rec(lv, g, e12, e21, er1, er2, nrmse) for _ in range(per_level)]
    return out


def test_bias_verdict_uses_signed_median_pooled_over_levels_two_and_up():
    recs = _levels(0.10, 0.10, 0.05, 0.05, nrmse=0.2)
    recs[0] = _rec(0, 2.16, 9.0, 9.0, 3.0, 3.0, 9.0)                         # level 1: a wild outlier, ignored
    v = sg.preprocessing_bias_verdict(CFG, recs)
    assert v["n_series"] == 6 and v["gain_bias"] == pytest.approx(0.10) and v["ei_bias"] == pytest.approx(0.05)
    assert v["gain_ok"] and v["ei_ok"] and v["nrmse_median"] == 0.2 and v["nrmse_ok"] and v["pass"] is True
    assert v["rho_true"] == pytest.approx(0.14772727272727273) and v["p_excluded"] is True
    assert set(v["per_level"]) == {1, 2, 3} and "literature" in v["ei_note"]


def test_bias_is_signed_not_absolute():
    # errors +0.3 and -0.3 balance out: signed median 0 (an absolute-error rule would fail)
    recs = [_rec(1, 5.4, 0.3, -0.3), _rec(2, 10.8, 0.3, -0.3), _rec(3, 27.0, -0.3, 0.3)]
    v = sg.preprocessing_bias_verdict(CFG, recs)
    assert v["gain_bias"] == pytest.approx(0.0, abs=1e-12) and v["gain_ok"]


def test_bias_tolerance_nrmse_and_pending():
    ok = sg.preprocessing_bias_verdict(CFG, _levels(0.14, 0.14))
    bad = sg.preprocessing_bias_verdict(CFG, _levels(0.16, 0.16))
    ei_bad = sg.preprocessing_bias_verdict(CFG, _levels(0.0, 0.0, 0.2, 0.2))
    assert ok["gain_ok"] and ok["pass"] is None and ok["nrmse_ok"] is None        # PySR part pending
    assert not bad["gain_ok"] and bad["pass"] is False
    assert not ei_bad["ei_ok"] and ei_bad["pass"] is False
    nr = sg.preprocessing_bias_verdict(CFG, _levels(0.0, 0.0, nrmse=0.3))
    assert nr["nrmse_ok"] is False and nr["pass"] is False
    assert sg.preprocessing_bias_verdict(CFG, _levels(0.0, 0.0, nrmse=0.25))["pass"] is True    # <= 0.25


def test_bias_diverged_series_count_as_infinite_error():
    recs = _levels(0.0, 0.0)
    elig = [r for r in recs if r["level_index"] >= 1]
    for r in elig[:2]:                                                          # 2 of 6 diverged
        r.update(g12=None, g21=None, rho1=None, rho2=None)
    v = sg.preprocessing_bias_verdict(CFG, recs)
    assert v["n_diverged"] == 2 and v["gain_ok"]                               # the median is still 0 with 4 good of 6
    for r in elig[2:4]:                                                         # 4 of 6 diverged: the median is inf
        r.update(g12=None, g21=None, rho1=None, rho2=None)
    v = sg.preprocessing_bias_verdict(CFG, recs)
    assert v["n_diverged"] == 4 and not v["gain_ok"] and not v["ei_ok"] and v["gain_bias"] == float("inf")
    with pytest.raises(sg.GateError):
        sg.preprocessing_bias_verdict(CFG, [_rec(0, 2.16, 0, 0)])


# ---- hygiene ----------------------------------------------------------------------------------------

def test_source_and_config_for_e3():
    text = SRC.read_text(encoding="utf-8")
    assert not re.search(r"(^|\s)print\(", text)
    assert not re.search(r"np\.random\.(seed|rand|randn|normal|uniform|randint|choice|shuffle|permutation)\b", text)
    pg = CFG["g0"]["preprocessing_gate"]
    assert pg["blink"] == {"duration_s": 0.3, "amplitude_uv": 30, "rate_per_s": 0.0667}
    assert pg["emg"]["band_hz"] == [20, 150] and pg["emg"]["rms_fraction_of_channel_sd"] == 0.30
    assert pg["drift"] == {"freq_hz": 0.05, "amplitude_uv": 50}
    assert pg["bilateral_lag_ms"] == [0, 4] and pg["bias_tolerance"] == 0.15 and pg["bias_from_level_index"] == 1
