"""E2 tests: the series generator, Null A, Null B and the G0 tuning set of src/synthetic_gate.py (§9.1, §9.2, §7.5).
Expected values come from independent routes: closed forms, the test-only D1 simulator, brute force."""
import copy
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import model, synthetic_gate as sg  # noqa: E402
from src import preprocess as pp  # noqa: E402
from src.config import load_config  # noqa: E402
import planted_sim as ps  # noqa: E402

CFG = load_config()
SRC = Path(sg.__file__)


def _short():
    c = copy.deepcopy(CFG)
    c["g0"]["regime_grid_shape"] = [2, 2, 2]
    c["g0"]["regime"]["grid_sim_duration_s"] = 24
    c["g0"]["series_duration_s"] = 20
    return c


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    c = _short()
    table = sg.FeatureTable(["r0", "r1"], np.array([[10.0, 0.50, 1.4], [9.5, 0.40, 1.5]]),
                            np.array([[6.0, 7.0], [6.5, 6.6]]))
    grid = sg.build_grid(c, table.exponent, tmp_path_factory.mktemp("grid"), n_jobs=1)
    return c, table, grid


# ---- levels, delta ----------------------------------------------------------------------------------

def test_levels_and_delta_by_hand():
    C2 = 0.8 * 135.0
    assert C2 == 108.0
    assert [sg.level_gain(CFG, k) for k in range(4)] == pytest.approx([2.16, 5.4, 10.8, 27.0])
    assert sg.delta(CFG) == pytest.approx(0.5 * 2.16) == pytest.approx(1.08)


# ---- the kernel -------------------------------------------------------------------------------------

def _input(n=4096, p=(220.0, 260.0), seed=1):
    return model.draw_input(np.random.default_rng(seed), n, 2, np.array(p), CFG)


def test_zero_coefficient_is_bit_identical_to_model_simulate():
    u = _input()
    ref = model.simulate(CFG, 4096, input=u, n_nodes=2, p=[220.0, 260.0], g12=12.0, g21=5.0)
    (states, drive, res, _, _), _, _ = sg._run_kernel(CFG, u, 12.0, 5.0, [220.0, 260.0], [0.0, 0.0], [1.0, 1.0])
    assert np.array_equal(states, ref.states)
    assert np.array_equal(drive, ref.drive)
    assert not res.any()


def test_planted_kernel_equals_the_d1_test_simulator_with_a_planted_term():
    u = _input(seed=2)
    c, sd = np.array([1.3, 0.9]), np.array([2.0, 3.5])
    mine, _, _ = sg._run_kernel(CFG, u, 12.0, 5.0, [220.0, 260.0], c, sd)
    cfg_ps = copy.deepcopy(CFG)
    cfg_ps["rescaling"]["reference_simulation"]["sim_fs_hz"] = CFG["g0"]["generation_fs_hz"]
    theirs, _, _ = ps._run(cfg_ps, u, 12.0, 5.0, np.array([220.0, 260.0]), c, sd)
    n = u.shape[0]
    for a, b in zip(mine[:3], theirs[:3]):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12)
    for a, b in zip(mine[3:], theirs[3:]):                                  # the last buffer row is never written
        assert np.allclose(a[:n + 20], b[:n + 20], rtol=1e-12, atol=1e-12)
    assert np.abs(mine[2]).max() > 0.1                                    # the planted term is really present


def test_planted_term_reads_the_source_exactly_delay_steps_back():
    """res[k, j] = c_j u_src(k - delay) u_tgt(k) from the returned STATES, not from the kernel's own buffers."""
    u = _input(n=3000, seed=3)
    c, sd = np.array([1.1, 0.7]), np.array([2.5, 4.0])
    (states, _, res, _, _), _, d = sg._run_kernel(CFG, u, 12.0, 5.0, [220.0, 260.0], c, sd)
    assert d == 20
    pot = states[:, :, 1] - states[:, :, 2]
    for j in (0, 1):
        src = 1 - j
        k = np.arange(d, 3000)
        expected = c[j] * (pot[k - d, src] / sd[src]) * (pot[k, j] / sd[j])
        assert np.allclose(res[k, j], expected, rtol=1e-12, atol=1e-12)
        assert not np.allclose(res[k, j], c[j] * (pot[k - d + 1, src] / sd[src]) * (pot[k, j] / sd[j]), atol=1e-6)


def test_c_gives_half_the_rms_of_the_base_drive_and_is_defined_by_the_first_pass():
    fs = CFG["g0"]["generation_fs_hz"]
    n, burn, d = 10 * fs, 2 * fs, 20
    u = model.draw_input(np.random.default_rng(4), n, 2, np.array([220.0, 220.0]), CFG)
    g = 10.8
    out = sg.run_planted(CFG, u, g, g, [220.0, 220.0], 0.5, burn)
    ref = model.simulate(CFG, n, input=u, n_nodes=2, p=[220.0, 220.0], g12=g, g21=g)     # first pass, c = 0
    pot = ref.states[:, :, 1] - ref.states[:, :, 2]
    sd = pot[burn:n].std(axis=0)
    for j in (0, 1):
        src = 1 - j
        k = np.arange(burn, n)
        basis = (pot[k - d, src] / sd[src]) * (pot[k, j] / sd[j])
        expected_c = 0.5 * np.sqrt(np.mean(ref.drive[burn:, j] ** 2)) / np.sqrt(np.mean(basis ** 2))
        assert out["c"][j] == pytest.approx(expected_c, rel=1e-9)
        assert out["rms_ratio"][j] == pytest.approx(0.5, rel=0.15)                 # realised ratio of the final run


def test_zero_planted_fraction_is_the_plain_model():
    u = _input(seed=5)
    out = sg.run_planted(CFG, u, 0.0, 0.0, [220.0, 220.0], 0.0, 1000)
    ref = model.simulate(CFG, 4096, input=u, n_nodes=2, p=[220.0, 220.0])
    assert np.array_equal(out["states"], ref.states) and not out["c"].any()


# ---- Null B input -----------------------------------------------------------------------------------

def _null_b(i):
    n, p, hw = 400_000, np.array([220.0, 220.0]), model.default_half_width(CFG)
    rng_in = sg.series_rng(CFG, "pilot", "null_B", "input", i)
    rng_c = sg.series_rng(CFG, "pilot", "null_B", "common_input", i)
    u, meta = sg.null_b_input(CFG, rng_in, rng_c, n, p, hw)
    return (u - p) / hw, meta


def _a_lagged_series():
    for i in range(40):
        x, meta = _null_b(i)
        if meta["lag_steps"] >= 10:
            return x, meta
    raise AssertionError("no lagged series in 40 draws")


def test_null_b_variance_correlation_and_lag():
    x, meta = _a_lagged_series()
    lag, share, lead = meta["lag_steps"], meta["share"], meta["leading_node"]
    assert 0.30 <= share <= 0.50 and 0.0 <= meta["lag_ms_drawn"] <= 20.0
    assert lag == int(round(meta["lag_ms_drawn"] * 1e-3 * 2048))
    assert np.allclose(x.var(axis=0), 1.0 / 3.0, rtol=0.02)                # each node's input variance is unchanged
    other, leader = x[:, 1 - lead], x[:, lead]
    n = len(x)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])                      # noqa: E731
    assert corr(other[lag:], leader[:n - lag]) == pytest.approx(share, abs=0.01)
    assert abs(corr(x[:, 0], x[:, 1])) < 0.01                               # nothing at lag 0
    best = max(range(0, 45), key=lambda s: corr(other[s:], leader[:n - s]))
    assert best == lag


def test_null_b_lag_and_share_vary_across_series_and_are_deterministic():
    m1 = [_null_b(i)[1] for i in range(4)]
    m2 = [_null_b(i)[1] for i in range(4)]
    assert m1 == m2
    assert len({m["lag_steps"] for m in m1}) > 1 and len({m["share"] for m in m1}) > 1


# ---- seeds ------------------------------------------------------------------------------------------

def test_seed_streams_are_disjoint_and_deterministic():
    keys = {(s, a, sub): sg.series_rng(CFG, s, a, sub, 0).integers(1 << 62)
            for s in ("pilot", "full", "tuning", "preprocessing_gate")
            for a in ("positive", "null_A", "null_B")
            for sub in ("input", "mixing", "noise")}
    assert len(set(keys.values())) == len(keys)
    assert sg.series_rng(CFG, "pilot", "positive", "input", 3).integers(1 << 62) == \
        sg.series_rng(CFG, "pilot", "positive", "input", 3).integers(1 << 62)
    assert sg.series_rng(CFG, "pilot", "positive", "input", 3, round_=1).integers(1 << 62) != \
        sg.series_rng(CFG, "pilot", "positive", "input", 3).integers(1 << 62)
    assert sg.stream_base(CFG, "pilot", 2) == 91000 + 2 * 10000
    assert (sg.stream_base(CFG, "full"), sg.stream_base(CFG, "tuning"),
            sg.stream_base(CFG, "preprocessing_gate")) == (92000, 93000, 94000)


# ---- observation chain ------------------------------------------------------------------------------

def test_observation_rows_and_alignment_through_the_real_chain():
    fs, step = 2048, 8
    n_obs = 20 * 256
    rows = sg.observation_rows(CFG, n_obs)
    n_burn, trim = 4 * fs, 5 * fs
    assert np.array_equal(rows - 1 - n_burn, trim + step * np.arange(n_obs))
    n_keep = (20 + 10) * fs
    t = np.arange(n_keep) / fs
    x = np.vstack([np.sin(2 * np.pi * 3.0 * t), np.sin(2 * np.pi * 3.0 * t)])
    bp = pp.trim_edges(pp.bandpass(CFG, x, fs), fs, 5.0)
    out = pp.downsample(CFG, bp, fs)
    expected = np.sin(2 * np.pi * 3.0 * (trim + step * np.arange(out.shape[1])) / fs)
    assert out.shape[1] == n_obs
    assert np.max(np.abs(out[0, 100:-100] - expected[100:-100])) < 0.02       # zero phase: no lag after the chain


def test_downsample_is_anti_aliased_and_decimation_is_not():
    fs = 2048
    t = np.arange(fs * 30) / fs
    tone = np.sin(2 * np.pi * 300.0 * t)
    real = pp.downsample(CFG, tone[None, :], fs)[0]
    assert np.sqrt(np.mean(real[300:-300] ** 2)) < 1e-3
    assert np.sqrt(np.mean(tone[::8] ** 2)) == pytest.approx(np.sqrt(0.5), rel=0.01)    # aliasing to 44 Hz


def test_scale_to_uv_sets_the_post_bandpass_sd(world):
    c, _, _ = world
    fs = 2048
    z = np.random.default_rng(6).standard_normal((2, fs * 30)) * np.array([[0.3], [2.0]]) + 7.5
    x, factor = sg.scale_to_uv(c, z, fs, 6.6)
    bp = pp.trim_edges(pp.bandpass(c, x, fs), fs, 5.0)
    assert np.allclose(bp.std(axis=1, ddof=1), 6.6, rtol=1e-4)
    assert factor[0] > factor[1]


# ---- whole series -----------------------------------------------------------------------------------

def test_positive_series_chain_scaling_truth_and_summary(world, monkeypatch):
    c, table, grid = world
    calls = []
    for name in ("notch", "bandpass", "segment_signal", "downsample"):
        orig = getattr(pp, name)
        monkeypatch.setattr(pp, name, (lambda o, nm: lambda *a, **k: (calls.append(nm), o(*a, **k))[1])(orig, name))
    s = sg.generate_series(c, "positive", 3, grid, table, "pilot")
    monkeypatch.undo()
    assert {"notch", "bandpass", "segment_signal", "downsample"} <= set(calls)
    assert s.level_index == 3 and s.gains == (27.0, 27.0)
    assert 0.1 <= s.m <= 0.4
    allx = np.concatenate(s.segments, axis=1)
    assert np.allclose(allx.mean(axis=1), CFG["rescaling"]["mu_ref"], atol=1e-9)
    assert np.allclose(allx.std(axis=1, ddof=1), CFG["rescaling"]["sigma_ref"], atol=1e-9)
    assert allx.shape[1] == 20 * 256 and s.starts == [0]
    # c comes from the c = 0 pass (tested above at 0.5 +- 15%); the planted term feeds back, so at the strongest level
    # the realised ratio of the final run drifts upward. It is reported, not retuned (approved plan).
    assert all(0.3 < r < 1.0 for r in s.meta["rms_ratio"])
    t = s.truth
    n = allx.shape[1]
    assert all(t[k].shape == (n, 2) for k in ("u_tgt", "u_src", "s_src", "basis", "planted"))
    a = model.constants(CFG)["a"]
    assert np.allclose(t["planted"], t["basis"] * 3.25 * a * np.array(s.meta["c"])[None, :], rtol=1e-12)
    # u_src is the source potential 20 steps = 2.5 samples earlier: linear interpolation of the source's own u_tgt
    j = 0
    k = np.arange(10, n)
    interp = 0.5 * (t["u_tgt"][k - 3, 1 - j] + t["u_tgt"][k - 2, 1 - j])
    assert np.max(np.abs(t["u_src"][k, j] - interp)) < 0.05 * np.std(t["u_tgt"][:, 1 - j])
    summ = s.summary()
    i_best, d_best = sg.match_grid_point(grid.features, grid.valid, np.array(s.operating_point["target"]), table.scale)
    brute = np.sqrt((((grid.features - np.array(s.operating_point["target"])) / table.scale) ** 2).sum(axis=1))
    assert summ["z_distance"] == pytest.approx(brute.min()) and i_best == int(np.argmin(brute))
    assert summ["regime"] == grid.regime[i_best] and summ["matched_recording"] in table.names
    assert summ["clean_s"] == 20.0 and summ["g12"] == 27.0


def test_generation_is_deterministic_and_seeded(world):
    c, table, grid = world
    a = sg.generate_series(c, "positive", 1, grid, table, "pilot", with_truth=False)
    b = sg.generate_series(c, "positive", 1, grid, table, "pilot", with_truth=False)
    d = sg.generate_series(c, "positive", 1, grid, table, "pilot", round_=1, with_truth=False)
    e = sg.generate_series(c, "positive", 5, grid, table, "pilot", with_truth=False)
    assert all(np.array_equal(x, y) for x, y in zip(a.segments, b.segments)) and a.m == b.m
    assert not np.array_equal(a.segments[0], d.segments[0]) and not np.array_equal(a.segments[0], e.segments[0])
    assert d.meta["round"] == 1


def test_levels_cycle_over_series_index(world):
    c, table, grid = world
    got = [sg.generate_series(c, "positive", i, grid, table, "pilot", with_truth=False).level_index for i in range(5)]
    assert got == [0, 1, 2, 3, 0]


def test_null_arms_have_zero_gain_and_no_planted_term(world):
    c, table, grid = world
    a = sg.generate_series(c, "null_A", 2, grid, table, "pilot")
    b = sg.generate_series(c, "null_B", 2, grid, table, "pilot")
    for s in (a, b):
        assert s.gains == (0.0, 0.0) and s.level_index is None and s.truth is None
        assert s.meta["c"] == [0.0, 0.0]
        assert 0.1 <= s.m <= 0.4
    assert a.operating_point == b.operating_point                              # paired operating point
    assert "null_B" not in a.meta and 0.3 <= b.meta["null_B"]["share"] <= 0.5
    assert not np.array_equal(a.segments[0], b.segments[0])
    with pytest.raises(sg.GateError):
        sg.generate_series(c, "bogus", 0, grid, table, "pilot")


# ---- the tuning set ---------------------------------------------------------------------------------

def test_tuning_set_is_twenty_mid_level_series_from_its_own_block(world):
    c, table, grid = world
    ts = sg.tuning_set(c, grid, table)
    assert len(ts) == 20 and {s.stream for s in ts} == {"tuning"}
    assert {s.level_index for s in ts} == {2} and {s.gains for s in ts} == {(10.8, 10.8)}
    assert all(s.truth is None for s in ts)
    pilot0 = sg.generate_series(c, "positive", 0, grid, table, "pilot", level_index=2, with_truth=False)
    assert not np.array_equal(ts[0].segments[0], pilot0.segments[0])


def test_tune_g0_q_uses_the_numba_copy_and_routes_other_filters(world, monkeypatch, tmp_path):
    c, table, grid = world
    seen = {}
    from src import tuning

    def fake(recs, cfg, cache_dir=None, n_jobs=1, min_recordings=None):
        seen.update(ids=[r["id"] for r in recs], numba=cfg["ukf"]["numba"]["enabled"], n=len(recs),
                    mr=min_recordings, cache=cache_dir)
        return "result"
    monkeypatch.setattr(tuning, "tune_q", fake)
    series = sg.tuning_set(c, grid, table, n=3)
    assert sg.tune_g0_q(c, series, "19D", n_jobs=1, min_recordings=3) == "result"
    assert seen["ids"] == ["tuning_0", "tuning_1", "tuning_2"] and seen["numba"] is True and seen["mr"] == 3
    assert CFG["ukf"]["numba"]["enabled"] is True and c["ukf"]["numba"]["enabled"] is True     # global switch untouched (true since DEV-005)
    called = []
    monkeypatch.setattr(sg, "tune_g0_q_option", lambda cfg, ser, name, *a: called.append(name) or "option result")
    for other in ("A", "B"):                                                   # E4: A and B go through the gate's own worker
        assert sg.tune_g0_q(c, series, other) == "option result"
    assert called == ["A", "B"]


def test_tune_g0_q_real_run_returns_a_result_and_writes_no_qr_file(world, tmp_path, monkeypatch):
    c, table, grid = world
    monkeypatch.chdir(tmp_path)
    series = sg.tuning_set(c, grid, table, n=2)
    res = sg.tune_g0_q(c, series, "19D", min_recordings=2)
    from src import tuning
    grid_q = tuning.q_grid(c)
    assert res.q is None or any(np.isclose(res.q, g) for g in grid_q)
    assert list(tmp_path.iterdir()) == []


# ---- hygiene ----------------------------------------------------------------------------------------

def test_source_has_no_decimation_print_or_global_random():
    text = SRC.read_text(encoding="utf-8")
    assert not re.search(r"\[::\s*(step|8)\]", text)
    assert not re.search(r"(^|\s)print\(", text)
    assert not re.search(r"np\.random\.(seed|rand|randn|normal|uniform|randint|choice|shuffle|permutation)\b", text)
    assert "pp.downsample" in text or "preprocess" in text


def test_config_leaves_for_e2():
    g0 = CFG["g0"]
    assert g0["seeds"]["arm_codes"] == {"positive": 1, "null_A": 2, "null_B": 3, "artifact_null": 4}
    assert g0["series_duration_s"] == 240 and g0["generation_burn_in_s"] == 4
    assert g0["backend"] == "numba" and g0["tuning_level_index"] == 2 and g0["n_tuning_series"] == 20
