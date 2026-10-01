"""E1 tests: the operating-regime grid of src/synthetic_gate.py (§9.1). Expected values are computed here, by
independent routes (closed forms, brute force, hand tables), never by calling the function under test twice."""
import copy
import re
from pathlib import Path

import numpy as np
import pytest

from src import model, synthetic_gate as sg
from src.config import load_config

CFG = load_config()
SRC = Path(sg.__file__)


def _cfg(**g0_regime):
    c = copy.deepcopy(CFG)
    c["g0"]["regime"].update(g0_regime)
    return c


# ---- axes ------------------------------------------------------------------------------------------

def test_axes_shape_endpoints_and_spacing():
    p, sdf, share = sg.axes(CFG)
    assert (len(p), len(sdf), len(share)) == (12, 8, 5)
    assert p[0] == 120.0 and p[-1] == 320.0
    assert np.allclose(np.diff(p), 200.0 / 11.0)                      # linear
    assert sdf[0] == pytest.approx(0.25) and sdf[-1] == pytest.approx(2.0)
    assert np.allclose(sdf[1:] / sdf[:-1], 8.0 ** (1.0 / 7.0))         # geometric: (2 / 0.25)^(1/7)
    assert share[0] == pytest.approx(0.2) and share[-1] == pytest.approx(0.8)
    assert np.allclose(np.diff(share), 0.15)


def test_point_index_is_row_major_with_share_fastest():
    assert sg.point_index(CFG, 0, 0, 0) == 0
    assert sg.point_index(CFG, 0, 0, 1) == 1
    assert sg.point_index(CFG, 0, 1, 0) == 5
    assert sg.point_index(CFG, 1, 0, 0) == 40
    assert sg.point_index(CFG, 11, 7, 4) == 479


# ---- noise and features ----------------------------------------------------------------------------

def test_inband_power_of_a_sinusoid_is_closed_form():
    fs, n, a = 2048, 2048 * 8, 2.0
    t = np.arange(n) / fs
    x = a * np.sin(2 * np.pi * 10.0 * t)
    assert sg.inband_power(x, fs, (1, 45)) == pytest.approx((a * n / 2) ** 2, rel=1e-9)   # |X_k|^2 = (A N / 2)^2
    assert sg.inband_power(x, fs, (50, 100)) < 1e-12 * (a * n / 2) ** 2


def _independent_inband(x, fs, lo, hi):
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.arange(len(X)) * fs / len(x)
    return X[(f >= lo) & (f <= hi)].sum()


def test_observe_noisy_mixes_deviations_and_hits_the_noise_share():
    fs, n, m, share = 2048, 2048 * 20, 0.3, 0.5
    rng0 = np.random.default_rng(7)
    y = CFG["rescaling"]["mu_ref"] + rng0.standard_normal((n, 2))
    z = sg.observe_noisy(CFG, y, fs, m, share, 1.4, np.random.default_rng(1))
    mu = CFG["rescaling"]["mu_ref"]
    mixed = np.array([[1, m], [m, 1]]) @ (y - mu).T
    noise = z - mu - mixed
    for ch in range(2):
        ps, pn = _independent_inband(mixed[ch], fs, 1, 45), _independent_inband(noise[ch], fs, 1, 45)
        assert pn / (ps + pn) == pytest.approx(share, abs=0.02)        # cross term is small
        assert pn / ps == pytest.approx(share / (1 - share), rel=0.05)


def test_noise_exponent_and_white_fraction():
    fs, n = 2048, 2048 * 80
    y = np.random.default_rng(3).standard_normal((n, 2)) * 0.0 + 1.0
    y[:, 0] += np.sin(2 * np.pi * 10 * np.arange(n) / fs)               # a signal so the in-band power is positive
    y[:, 1] += np.sin(2 * np.pi * 10 * np.arange(n) / fs)

    def slope(c):
        out = []
        for seed in (5, 6, 7):                                          # a single realisation is noisy at 3-6 Hz
            z = sg.observe_noisy(c, y, fs, 0.0, 0.8, 1.5, np.random.default_rng(seed))
            noise = z[0] - CFG["rescaling"]["mu_ref"] - (y[:, 0] - CFG["rescaling"]["mu_ref"])
            f = np.fft.rfftfreq(n, 1 / fs)
            P = np.abs(np.fft.rfft(noise)) ** 2
            edges = np.geomspace(3, 45, 13)
            fc = [np.sqrt(lo * hi) for lo, hi in zip(edges[:-1], edges[1:])]
            pc = [P[(f >= lo) & (f < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
            out.append(np.polyfit(np.log10(fc), np.log10(pc), 1)[0])
        return float(np.mean(out))
    assert slope(_cfg(one_over_f_fraction_of_noise=1.0)) == pytest.approx(-1.5, abs=0.15)
    assert abs(slope(_cfg(one_over_f_fraction_of_noise=0.0))) < 0.15


def test_spectral_features_recover_a_known_spectrum():
    f = np.arange(0, 128.5, 0.5)
    f[0] = 0.0
    psd = np.empty_like(f)
    psd[1:] = 10.0 ** (1.0 - 1.6 * np.log10(f[1:])) + 3.0 * np.exp(-((f[1:] - 10.0) ** 2) / (2 * 1.0 ** 2))
    psd[0] = psd[1]
    v = sg.spectral_features(CFG, f, psd)
    assert v[0] == pytest.approx(10.0, abs=0.5)
    assert v[2] == pytest.approx(1.6, abs=0.15)
    a, t = (f >= 8) & (f <= 13), (f >= 1) & (f <= 45)
    expected = np.sum((psd[a][1:] + psd[a][:-1]) / 2 * 0.5) / np.sum((psd[t][1:] + psd[t][:-1]) / 2 * 0.5)
    assert v[1] == pytest.approx(expected, rel=1e-9)


def test_mean_psd_white_noise_level_and_block_weighting():
    fs = 256
    rng = np.random.default_rng(11)
    a = rng.standard_normal((2, fs * 40))
    b = rng.standard_normal((2, fs * 10)) * 3.0
    f, p = sg.mean_psd([a, b], fs, 2)
    # one-sided density 2 sigma^2 / fs, blocks weighted 20 : 5 (per channel the same weights)
    expected = (20 * 2 * 1.0 / fs + 5 * 2 * 9.0 / fs) / 25
    assert np.mean(p[5:-5]) == pytest.approx(expected, rel=0.08)
    with pytest.raises(sg.GateError):
        sg.mean_psd([np.zeros((2, 10))], fs, 2)


# ---- matching --------------------------------------------------------------------------------------

def test_match_is_scaled_brute_force_first_on_ties_and_skips_invalid():
    grid = np.array([[10.0, 0.50, 1.5], [10.0, 0.50, 1.5], [12.0, 0.30, 1.0], [9.0, 0.45, 1.4], [10.1, 0.50, 1.5]])
    valid = np.array([True, True, True, True, True])
    scale = np.array([1.0, 0.1, 0.5])
    target = np.array([10.0, 0.50, 1.5])
    i, d = sg.match_grid_point(grid, valid, target, scale)
    brute = [np.sqrt(sum(((grid[k, j] - target[j]) / scale[j]) ** 2 for j in range(3))) for k in range(5)]
    assert i == int(np.argmin(brute)) and d == pytest.approx(min(brute))
    assert i == 0                                                       # rows 0, 1 tie exactly: first wins
    i2, _ = sg.match_grid_point(grid, np.array([False, False, True, True, True]), target, scale)
    assert i2 == 4
    with pytest.raises(sg.GateError):
        sg.match_grid_point(grid, np.zeros(5, dtype=bool), target, scale)


def test_scaling_changes_the_winner():
    grid = np.array([[10.0, 0.9, 1.5], [10.6, 0.5, 1.5]])
    target = np.array([10.0, 0.5, 1.5])
    u, _ = sg.match_grid_point(grid, np.ones(2, bool), target, np.ones(3))                    # 0.4 vs 0.6 -> row 0
    s, _ = sg.match_grid_point(grid, np.ones(2, bool), target, np.array([1.0, 0.05, 1.0]))    # 8.0 vs 0.6 -> row 1
    assert (u, s) == (0, 1)


def test_draw_is_seeded():
    t = sg.FeatureTable([f"r{i}" for i in range(15)], np.zeros((15, 3)), np.ones((15, 2)))
    a = [sg.draw_feature_vector(t, np.random.default_rng([91000, 0, i]))[0] for i in range(20)]
    b = [sg.draw_feature_vector(t, np.random.default_rng([91000, 0, i]))[0] for i in range(20)]
    c = [sg.draw_feature_vector(t, np.random.default_rng([92000, 0, i]))[0] for i in range(20)]
    assert a == b and a != c and len(set(a)) > 3


# ---- regime ----------------------------------------------------------------------------------------

def _stub_sim(output):
    def sim(cfg, n, **kw):
        class R:
            pass
        r = R()
        r.output = output(n)
        return r
    return sim


def test_classifier_with_stub_simulators(monkeypatch):
    fs = CFG["rescaling"]["reference_simulation"]["sim_fs_hz"]
    monkeypatch.setattr(model, "simulate", _stub_sim(lambda n: np.full((n, 1), 3.0)))
    assert sg.classify_regime(CFG, 220.0) == sg.NOISE_DRIVEN
    monkeypatch.setattr(model, "simulate", _stub_sim(
        lambda n: (1.0 * np.sin(2 * np.pi * 10 * np.arange(n) / fs))[:, None]))
    assert sg.classify_regime(CFG, 220.0) == sg.LIMIT_CYCLE
    monkeypatch.setattr(model, "simulate", _stub_sim(
        lambda n: (0.04 * np.sin(2 * np.pi * 10 * np.arange(n) / fs))[:, None]))     # pp 0.08 > 0.05
    assert sg.classify_regime(CFG, 220.0) == sg.LIMIT_CYCLE
    monkeypatch.setattr(model, "simulate", _stub_sim(
        lambda n: (0.02 * np.sin(2 * np.pi * 10 * np.arange(n) / fs))[:, None]))     # pp 0.04 < 0.05
    assert sg.classify_regime(CFG, 220.0) == sg.NOISE_DRIVEN


def test_classifier_real_runs():
    assert sg.classify_regime(CFG, 400.0) == sg.NOISE_DRIVEN             # above the upper Hopf point: stable fixed point
    assert sg.classify_regime(CFG, 220.0) == sg.LIMIT_CYCLE              # the classical alpha limit cycle


def test_steady_state_is_not_unique_below_the_grid():
    """Why the p axis starts at 120: below it the fixed-point residual has 3 roots and model.steady_state raises."""
    with pytest.raises(model.ModelError):
        sg.classify_regime(CFG, 60.0)


# ---- training-only table ---------------------------------------------------------------------------

def _loader_factory(calls):
    rng = np.random.default_rng(2)

    def load(subject):
        calls.append(subject)
        x = rng.standard_normal((2, 256 * 30)) * 1.0
        return [{"name": f"{subject}_ses-t1", "reason": None, "segments": [x], "sd_uv": [5.0, 7.0]},
                {"name": f"{subject}_ses-t2", "reason": "excluded: insufficient_clean_data"}]
    return load


def test_table_refuses_test_subjects_before_loading_anything():
    split = {"train": ["sub-001", "sub-002"], "test": ["sub-003"]}
    calls = []
    with pytest.raises(sg.GateError, match="sub-003"):
        sg.build_feature_table(CFG, ["sub-001", "sub-003"], split, _loader_factory(calls))
    assert calls == []


def test_table_contents_and_skips():
    split = {"train": ["sub-001", "sub-002"], "test": []}
    calls = []
    t = sg.build_feature_table(CFG, ["sub-002", "sub-001"], split, _loader_factory(calls))
    assert calls == ["sub-001", "sub-002"]                               # sorted
    assert t.names == ["sub-001_ses-t1", "sub-002_ses-t1"]
    assert t.features.shape == (2, 3) and np.all(np.isfinite(t.features))
    assert [s[0] for s in t.skipped] == ["sub-001_ses-t2", "sub-002_ses-t2"]
    assert t.target_sd_uv == 6.0                                         # median of [5, 7, 5, 7]


def test_real_split_pilot_is_on_the_training_side():
    root = Path(__file__).resolve().parent.parent
    split = sg.load_split(CFG, root)
    sg.check_training_side(split["pilot"], split)
    with pytest.raises(sg.GateError):
        sg.check_training_side([split["test"][0]], split)


# ---- the grid --------------------------------------------------------------------------------------

def _tiny():
    c = copy.deepcopy(CFG)
    c["g0"]["regime_grid_shape"] = [2, 2, 2]
    c["g0"]["regime"]["grid_sim_duration_s"] = 24
    return c


def test_build_grid_is_deterministic_cached_and_keyed(tmp_path):
    c = _tiny()
    g1 = sg.build_grid(c, 1.4, tmp_path, n_jobs=1)
    assert g1.features.shape == (8, 3) and g1.valid.all() and not g1.from_cache
    assert len(g1.regime) == 8
    g2 = sg.build_grid(c, 1.4, tmp_path, n_jobs=1)
    assert g2.from_cache and np.array_equal(g1.features, g2.features) and g1.regime == g2.regime
    g3 = sg.build_grid(c, 1.5, tmp_path, n_jobs=1)                      # a different exponent is a different key
    assert not g3.from_cache and g3.key != g1.key
    other = copy.deepcopy(c)
    other["g0"]["regime"]["grid_mixing_m"] = 0.3
    assert sg.grid_key(other, 1.4) != g1.key
    # an independent recomputation of one point reproduces the stored features exactly
    p, sdf, share = sg.axes(c)
    again = sg.grid_point_features(c, 5, p[1], sdf[0], share[1], 1.4)
    assert np.array_equal(again, g1.features[5])
    assert g1.point(5) == {"index": 5, "p": float(p[1]), "input_sd_factor": float(sdf[0]),
                           "noise_share": float(share[1]), "regime": g1.regime[5]}


def test_noise_share_moves_the_exponent_and_alpha_share(tmp_path):
    c = _tiny()
    c["g0"]["regime_grid_shape"] = [1, 1, 2]
    g = sg.build_grid(c, 1.4, tmp_path, n_jobs=1)
    assert g.features[1, 1] < g.features[0, 1]                           # more noise, smaller alpha share


# ---- hygiene ---------------------------------------------------------------------------------------

def test_source_has_no_print_global_random_or_unset_seed():
    text = SRC.read_text(encoding="utf-8")
    assert not re.search(r"(^|\s)print\(", text)
    assert not re.search(r"np\.random\.(seed|rand|randn|normal|uniform|randint|choice|shuffle|permutation)\b", text)
    assert "default_rng" in text
    assert "preprocess" in text and "import preprocess as pp" in text


def test_config_leaves_for_g0():
    for k in ("pilot", "full", "tuning", "preprocessing_gate", "grid", "fresh_round_stride"):
        assert isinstance(CFG["g0"]["seeds"][k], int)
    assert (CFG["g0"]["seeds"]["pilot"], CFG["g0"]["seeds"]["full"], CFG["g0"]["seeds"]["tuning"],
            CFG["g0"]["seeds"]["preprocessing_gate"]) == (91000, 92000, 93000, 94000)
    assert CFG["g0"]["filter"] is None and CFG["g0"]["filter_options"] == ["19D", "A", "B"]
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yml").read_text(encoding="utf-8"))
    assert raw["g0"]["filter"]["prov"] == "unset"
    assert raw["g0"]["seeds"]["pilot"]["prov"] == "placeholder"
