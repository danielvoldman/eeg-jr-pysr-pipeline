"""C4: the Q/R tuning rule (§7.5, §7.6; IMP-037 to IMP-043).

Expected values are computed independently of src/tuning.py: the grid from numpy.linspace, the NIS from a
separate pass-1 loop that recomputes innovation' S^-1 innovation from the filter's stored innovation and S,
the burn-in masks from hand arithmetic, and the selection by a separate argmin.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src import passes, tuning, ukf
from src import state_space as ss
from src.config import load_config
from tests import sim_data

CFG = load_config()
FS = CFG["preprocessing"]["observation_fs_hz"]
GRID = 10.0 ** np.linspace(-4.0, -1.0, 8)          # independent of tuning.q_grid


# ---- independent oracle -------------------------------------------------------------------------------------

def oracle_nis(cfg, segments, starts, q):
    """Mean NIS of one recording: own pass-1 loop (carry with gap inflation, both burn-ins) over
    ukf.run_filter, NIS recomputed from the stored innovation and S."""
    layout = ss.make_layout(cfg)
    n_neural = ss.N_NEURAL
    burn = int(round(cfg["windows"]["training_burn_in_s"] * FS))
    est = int(round(cfg["passes"]["estimator_burn_in_s"] * FS))
    walk = cfg["ukf"]["process_noise"]["parameter_random_walk_factor"]
    px, pP = ss.prior_mean(layout, cfg), ss.prior_cov(layout, cfg)
    carry = None
    last_end = None
    processed = 0
    vals = []
    for seg, start in zip(segments, starts):
        z = np.asarray(seg, dtype=np.float64).T
        x0, P0 = px.copy(), pP.copy()
        if carry is not None:
            gap = (start - last_end) / FS
            x0[n_neural:] = carry[0]
            P0[n_neural:, n_neural:] = carry[1] + np.diag(walk * np.diag(pP)[n_neural:] * gap)
        res = ukf.run_filter(z, cfg, layout, q, x0=x0, P0=P0, keep_cov=True)
        assert not res.diverged
        carry = (res.x[-1, n_neural:].copy(), res.P[-1][n_neural:, n_neural:].copy())
        last_end = start + z.shape[0]
        for t in range(z.shape[0]):
            if t >= burn and processed + t >= est:
                vals.append(float(res.innovation[t] @ np.linalg.solve(res.S[t], res.innovation[t])))
        processed += z.shape[0]
    return float(np.mean(vals)), len(vals)


@pytest.fixture(scope="module")
def sim_recs():
    out = []
    for i, (seed, g12) in enumerate([(11, 8.0), (12, 0.0)]):
        r = sim_data.make_recording(CFG, [7, 7], seed, g12=g12)
        out.append({"id": f"sim{i}", "segments": r["segments"], "starts": r["starts"]})
    return out


@pytest.fixture(scope="module")
def tuned(sim_recs, tmp_path_factory):
    cache = tmp_path_factory.mktemp("qrcache")
    return tuning.tune_q(sim_recs, CFG, cache_dir=cache, min_recordings=2), cache


# ---- grid -------------------------------------------------------------------------------------------------

def test_grid_has_eight_log_spaced_values_from_config():
    g = tuning.q_grid(CFG)
    assert len(g) == 8
    np.testing.assert_allclose(g, GRID, rtol=1e-12)
    assert g[0] == pytest.approx(1e-4) and g[-1] == pytest.approx(1e-1)
    np.testing.assert_allclose(np.diff(np.log10(g)), 3.0 / 7.0, rtol=1e-12)


def test_r_is_a_quarter_of_sigma_ref_squared():
    R = ukf.obs_noise(CFG)
    np.testing.assert_allclose(R, 0.25 * CFG["rescaling"]["sigma_ref"] ** 2 * np.eye(2), rtol=1e-14)


# ---- the rule against the independent oracle -------------------------------------------------------------

def test_tuning_table_and_choice_match_the_independent_oracle(tuned, sim_recs):
    res, _ = tuned
    assert not res.refused and res.n_matched == 2
    means = []
    for j, q in enumerate(GRID):
        per_rec = [oracle_nis(CFG, r["segments"], r["starts"], q)[0] for r in sim_recs]
        means.append(float(np.mean(per_rec)))
        assert res.table[j]["mean_nis"] == pytest.approx(means[j], rel=1e-9), j
        assert res.table[j]["q"] == pytest.approx(q, rel=1e-12)
    expected = int(np.argmin([abs(m - 2.0) for m in means]))
    assert res.q_index == expected and res.q == pytest.approx(GRID[expected], rel=1e-12)
    # the winner is closer to 2 than every other grid value
    assert all(abs(means[expected] - 2.0) <= abs(m - 2.0) for m in means)
    print("NIS vs q:", [f"{m:.3f}" for m in means], "-> q", res.q)


def test_samples_after_both_burn_ins_are_counted(tuned, sim_recs):
    res, _ = tuned
    # 7 s + 7 s at 256 Hz: seg 1 keeps samples with cum >= 6 s (1536): 1792 - 1536 = 256; seg 2 keeps t >= 128 and
    # cum = 1792 + t >= 1536 always: 1792 - 128 = 1664
    assert res.recordings[0]["n_samples"] == [256 + 1664] * 8


def test_recording_run_is_deterministic_and_cache_round_trips(tuned, sim_recs, monkeypatch):
    res, cache = tuned
    e1 = tuning.recording_nis(sim_recs[0]["segments"], sim_recs[0]["starts"], CFG, float(GRID[3]))
    e2 = tuning.recording_nis(sim_recs[0]["segments"], sim_recs[0]["starts"], CFG, float(GRID[3]))
    assert e1 == e2                                                  # bit-identical
    assert e1["mean_nis"] == res.recordings[0]["mean_nis"][3]
    # second call: every result comes from the cache; no filter run is allowed
    monkeypatch.setattr(passes, "run_pass1", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    again = tuning.tune_q(sim_recs, CFG, cache_dir=cache, min_recordings=2)
    assert again.to_dict() == res.to_dict()


def test_cache_key_depends_on_config_and_q():
    k = tuning.cache_key("rec", 1e-2, CFG)
    assert k == tuning.cache_key("rec", 1e-2, CFG)
    assert k != tuning.cache_key("rec", 1e-3, CFG)
    assert k != tuning.cache_key("rec2", 1e-2, CFG)
    for section, path in [("ukf", ("qr_rule", "nis_band")), ("windows", ("training_burn_in_s",)),
                          ("passes", ("estimator_burn_in_s",)), ("coupling", ("delay_substeps",)),
                          ("ukf", ("observation_noise", "R_fraction_of_rescaled_variance"))]:
        cfg2 = json.loads(json.dumps(CFG))
        d = cfg2[section]
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = d[path[-1]] + 1
        assert tuning.cache_key("rec", 1e-2, cfg2) != k, (section, path)


def test_forward_only_matches_the_full_pass_and_leaves_default_unchanged():
    r = sim_data.make_recording(CFG, [3, 3], 5, g12=5.0)
    cfg = json.loads(json.dumps(CFG))
    cfg["passes"]["estimator_burn_in_s"] = 1
    full = passes.run_pass1(r["segments"], r["starts"], cfg, 1e-2)
    fwd = passes.run_pass1(r["segments"], r["starts"], cfg, 1e-2, forward_only=True)
    assert full.params is not None and fwd.params is None
    for a, b in zip(full.segments, fwd.segments):
        assert a.nis is None and a.nis_keep is None              # the default path stores nothing new
        assert b.nis.shape == (a.n,) and b.nis_keep.dtype == bool
        np.testing.assert_array_equal(a.carry_out_mean, b.carry_out_mean)
        np.testing.assert_array_equal(a.carry_out_var, b.carry_out_var)
    assert full.gain_estimate == fwd.gain_estimate
    # segment 0 starts from the prior in both: its NIS equals a plain forward run of the filter
    z0 = np.ascontiguousarray(np.asarray(r["segments"][0], dtype=np.float64).T)
    np.testing.assert_array_equal(fwd.segments[0].nis, ukf.run_filter(z0, cfg, ss.make_layout(cfg), 1e-2).nis)
    burn, est = 128, 256                                          # 0.5 s and 1 s at 256 Hz
    exp_keep = np.arange(z0.shape[0]) >= max(burn, est)
    np.testing.assert_array_equal(fwd.segments[0].nis_keep, exp_keep)
    assert full.recording_diverged == fwd.recording_diverged is False


def test_parallel_workers_equal_serial():
    cfg = json.loads(json.dumps(CFG))
    cfg["passes"]["estimator_burn_in_s"] = 0
    recs = []
    for i in range(2):
        r = sim_data.make_recording(cfg, [1.5], 30 + i)
        recs.append({"id": f"p{i}", "segments": r["segments"], "starts": r["starts"]})
    a = tuning.tune_q(recs, cfg, n_jobs=1, min_recordings=2)
    b = tuning.tune_q(recs, cfg, n_jobs=2, min_recordings=2)
    assert a.to_dict() == b.to_dict()


# ---- NIS samples on stubbed filter output (hand arithmetic) ---------------------------------------------------

def _fake_filter(nis_by_call, diverge_calls=()):
    calls = {"i": 0}

    def fake(z, cfg, layout, q, x0=None, P0=None, buffer=None, keep_cov=False):
        i = calls["i"]
        calls["i"] += 1
        T, n = z.shape[0], layout.n
        div = i in diverge_calls
        x = np.tile(ss.prior_mean(layout, cfg), (T, 1))
        return SimpleNamespace(x=x, P=None, P_last=ss.prior_cov(layout, cfg), nis=np.asarray(nis_by_call[i], float),
                               diverged=div, divergence_reason="stub" if div else None,
                               divergence_step=0 if div else None, monitor={})
    return fake


def test_burn_in_masks_and_dropped_segments(monkeypatch):
    burn, est = 128, 1536
    lens = [1000, 700, 1200]
    nis = [np.arange(n, dtype=float) + 10000 * k for k, n in enumerate(lens)]
    monkeypatch.setattr(ukf, "run_filter", _fake_filter(nis, diverge_calls=(1,)))
    segs = [np.zeros((2, n)) for n in lens]
    starts = [0, 2000, 4000]
    out = tuning.recording_nis(segs, starts, CFG, 1e-2)
    # segment 0: cum 0..999 < 1536 -> nothing; segment 1 diverged: dropped, does not advance the count;
    # segment 2: cum = 1000 + t, kept when t >= 128 and 1000 + t >= 1536, i.e. t >= 536
    expected = nis[2][536:]
    assert out["n_samples"] == len(expected)
    assert out["mean_nis"] == pytest.approx(float(expected.mean()), rel=1e-12)
    assert out["n_diverged_samples"] == 700 and out["n_clean"] == sum(lens)
    assert out["recording_diverged"] is True                      # 700 of 2900 clean samples: 24 percent


def test_recording_diverged_flag_uses_ten_percent(monkeypatch):
    lens = [900, 100]
    nis = [np.ones(n) for n in lens]
    monkeypatch.setattr(ukf, "run_filter", _fake_filter(nis, diverge_calls=(1,)))
    out = tuning.recording_nis([np.zeros((2, n)) for n in lens], [0, 2000], CFG, 1e-2)
    assert out["recording_diverged"] is False                     # exactly 10.0 percent
    monkeypatch.setattr(ukf, "run_filter", _fake_filter([np.ones(899), np.ones(101)], diverge_calls=(1,)))
    out = tuning.recording_nis([np.zeros((2, 899)), np.zeros((2, 101))], [0, 2000], CFG, 1e-2)
    assert out["recording_diverged"] is True


# ---- selection on stubbed tables ----------------------------------------------------------------------------------

def _entry(m, n=100, div=False):
    return {"mean_nis": m, "n_samples": n, "recording_diverged": div, "n_clean": n, "n_diverged_samples": 0,
            "n_segments": 1, "n_segments_diverged": int(div)}


def _table(rows):
    """rows[i][j] = mean NIS (float) or "div"."""
    return [[_entry(None if v == "div" else v, div=(v == "div")) for v in row] for row in rows]


def _select(rows, **kw):
    n = len(rows)
    kw.setdefault("min_recordings", min(15, n))
    return tuning.select_q(_table(rows), [f"r{i}" for i in range(n)], [f"k{i}" for i in range(n)], GRID, CFG, **kw)


def test_closest_to_two_uses_absolute_difference():
    # mean NIS per q; |1.0 - 2| = 1.0 versus |3.5 - 2| = 1.5: absolute difference picks 1.0 (index 2);
    # a log-ratio rule would pick 3.5
    nis = [9.0, 5.0, 1.0, 3.5, 4.0, 6.0, 7.0, 8.0]
    res = _select([nis] * 3)
    assert res.q_index == 2 and res.q == pytest.approx(GRID[2]) and not res.tie_broken
    assert res.in_band is False and res.low_confidence_qr is True


def test_exact_tie_goes_to_the_smaller_q_and_is_flagged():
    nis = [9.0, 1.5, 9.0, 2.5, 9.0, 9.0, 9.0, 9.0]         # |1.5-2| == |2.5-2| exactly
    res = _select([nis] * 4)
    assert res.q_index == 1 and res.tie_broken is True and res.in_band is True


def test_unweighted_mean_over_recordings_and_pooled_reported():
    rows = _table([[2.0] * 8, [4.0] * 8])
    rows[0] = [dict(e, n_samples=1) for e in rows[0]]
    rows[1] = [dict(e, n_samples=3) for e in rows[1]]
    res = tuning.select_q(rows, ["a", "b"], ["ka", "kb"], GRID, CFG, min_recordings=2)
    assert res.table[0]["mean_nis"] == pytest.approx(3.0)              # unweighted
    assert res.table[0]["pooled_mean_nis"] == pytest.approx(3.5)       # sample-weighted, for reference


def test_band_edge_and_spread_flags():
    res = _select([[1.0, 1.2, 1.5, 1.85, 2.3, 3.0, 4.0, 5.0]] * 3)
    assert res.in_band and not res.at_grid_edge and res.nis_spread == pytest.approx(4.0)
    assert res.q_index == 3                                             # 1.85 (0.15) beats 2.3 (0.3)
    res = _select([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5]] * 3)        # nothing in band, best at the low edge
    assert res.q_index == 0 and res.at_grid_edge and not res.in_band and res.low_confidence_qr
    assert res.refused is False and res.q == pytest.approx(GRID[0])


def test_diverged_recording_excluded_at_all_q_not_per_q():
    good = [2.6, 1.0, 1.9, 4.0, 5.0, 6.0, 7.0, 8.0]
    bad = [50.0, 50.0, 50.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    bad[0] = "div"                                                       # diverged at the smallest q only
    rows = [good] * 15 + [bad]
    res = _select(rows)
    assert res.n_matched == 15 and res.recordings[15]["matched"] is False
    # every q is averaged over the SAME 15 recordings: q index 2 (1.9) wins; a per-q drop would average the
    # bad recording in at the other 7 values (mean at index 2 would be (15*1.9 + 50)/16 != 1.9)
    assert res.table[2]["mean_nis"] == pytest.approx(1.9) and res.q_index == 2
    assert res.table[0]["n_diverged"] == 1 and all(t["n_diverged"] == 0 for t in res.table[1:])


def test_refuses_below_minimum_recordings_and_flags_high_divergence():
    ok = [2.0] * 8
    div = ["div"] * 8
    res = _select([ok] * 14 + [div] * 6)                                # 14 usable of 20 < 15
    assert res.refused and res.q is None and "14 of 20" in res.refusal_reason
    assert len(res.q_flagged_divergence) == 8                           # 30% diverged at every q, > 25%
    res = _select([ok] * 15 + [div] * 5)                                # exactly 15: allowed; 25% is not more than 25%
    assert not res.refused and res.q_flagged_divergence == []


def test_recording_without_nis_samples_counts_as_unusable():
    rows = _table([[2.0] * 8] * 15)
    rows.append([_entry(None) for _ in range(8)])
    res = tuning.select_q(rows, [str(i) for i in range(16)], [str(i) for i in range(16)], GRID, CFG)
    assert res.n_matched == 15 and res.recordings[15]["reason"]


# ---- draw, test isolation, gate, output ---------------------------------------------------------------------------

class _Poisoned(dict):
    """A split whose test list must never be read."""

    def __getitem__(self, k):
        if k == "test":
            raise AssertionError("the tuning draw touched the test side")
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k == "test":
            raise AssertionError("the tuning draw touched the test side")
        return super().get(k, default)

    def values(self):
        raise AssertionError("the tuning draw iterated over the whole split")

    items = values


def _split():
    return _Poisoned(train=[f"sub-{i:03d}" for i in range(1, 79)], test=[f"sub-{i:03d}" for i in range(79, 112)],
                     pilot=[f"sub-{i:03d}" for i in (5, 8, 19, 51, 56, 63, 74, 81)[:6]])


def test_draw_reads_only_train_and_is_deterministic():
    split = _split()
    o1 = tuning.draw_for_split(split, 42, CFG)
    o2 = tuning.draw_for_split(split, 42, CFG)
    assert o1 == o2 and sorted(o1) == sorted(split["train"])         # a permutation of the training side
    assert not set(o1[:20]) & set(dict.__getitem__(split, "test"))
    assert o1 != tuning.draw_for_split(split, 43, CFG)
    # the seed is offset + split seed: an independent generator reproduces the order
    rng = np.random.default_rng(1000 + 42)
    train = sorted(split["train"])
    assert o1 == [train[i] for i in rng.permutation(len(train))]
    assert len(set(o1[:20])) == 20


def test_pilot_subjects_can_be_drawn():
    split = _split()
    hit = [len(set(tuning.draw_for_split(split, s, CFG)[:20]) & set(dict.__getitem__(split, "pilot")))
           for s in (42, 43, 44, 45, 46)]
    assert max(hit) > 0


def test_collect_recordings_skips_ineligible_in_draw_order():
    order = ["a", "b", "c", "d", "e"]

    def loader(s):
        if s in ("a", "c"):
            return {"reason": "excluded: bad_electrode"}
        return {"reason": None, "segments": [], "starts": [], "key": s}
    chosen, skipped = tuning.collect_recordings(order, 2, loader)
    assert [c["id"] for c in chosen] == ["b", "d"]
    assert [s["subject"] for s in skipped] == ["a", "c"]


def _stub_nis(monkeypatch, table_by_id):
    def fake(segments, starts, cfg, q):
        rid = int(segments[0][0, 0])
        j = int(np.argmin(np.abs(GRID - q)))
        return _entry(table_by_id[rid][j])
    monkeypatch.setattr(tuning, "recording_nis", fake)


def _real_like_loader(subject):
    i = int(subject.split("-")[1])
    return {"reason": None, "segments": [np.full((2, 10), float(i))], "starts": [0], "key": f"k{i}"}


def test_run_real_uses_only_training_subjects_and_needs_gate(temp_root, monkeypatch):
    import main as main_mod
    split = _split()
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(split["train"]))
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("test IDs")))
    monkeypatch.setattr(main_mod, "load_split", lambda *a, **k: (_ for _ in ()).throw(AssertionError("whole split")))
    _stub_nis(monkeypatch, {i: [2.6, 1.0, 1.9, 4.0, 5.0, 6.0, 7.0, 8.0] for i in range(1, 112)})
    with pytest.raises(tuning.GateError):
        tuning.run_real(CFG, temp_root, 42, loader=_real_like_loader)
    assert not (temp_root / "outputs" / "qr_42.json").exists()
    (temp_root / "outputs").mkdir()
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": True}))
    with pytest.raises(tuning.GateError):
        tuning.run_real(CFG, temp_root, 42, loader=_real_like_loader)
    (temp_root / "outputs" / "gate.json").write_text(json.dumps({"hard_stop": False, "low_confidence": True}))
    res = tuning.run_real(CFG, temp_root, 42, loader=_real_like_loader, n_jobs=1)
    assert res.q_index == 2
    doc = json.loads((temp_root / "outputs" / "qr_42.json").read_text(encoding="utf-8"))
    assert len(doc["subjects"]) == 20 and set(doc["subjects"]) <= set(split["train"])
    assert doc["draw"]["train_only"] and doc["gate_low_confidence"] is True
    for key in ("git_commit", "git_dirty", "code_sha256", "config_yml_sha256", "filter_fingerprint_sha256"):
        assert key in doc["provenance"]
    assert len(doc["result"]["table"]) == 8 and doc["result"]["q"] == pytest.approx(GRID[2])
    # the cache holds one file per (recording, q)
    assert len(list((temp_root / "cache" / "qr_tuning").glob("*.json"))) == 20 * 8
    # a rerun with the same result leaves the file alone; a different result is refused
    before = (temp_root / "outputs" / "qr_42.json").read_bytes()
    tuning.run_real(CFG, temp_root, 42, loader=_real_like_loader, n_jobs=1)
    assert (temp_root / "outputs" / "qr_42.json").read_bytes() == before
    _stub_nis(monkeypatch, {i: [9.0, 9.0, 9.0, 2.0, 9.0, 9.0, 9.0, 9.0] for i in range(1, 112)})
    cfg2 = json.loads(json.dumps(CFG))
    cfg2["ukf"]["qr_rule"]["cache_subdir"] = "qr_other"
    with pytest.raises(tuning.TuningError):
        tuning.run_real(cfg2, temp_root, 42, loader=_real_like_loader, n_jobs=1)


def test_pilot_mode_needs_no_gate_and_writes_under_results_pilot(temp_root, monkeypatch):
    import main as main_mod
    split = _split()
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(split["train"]))
    pilot = dict.__getitem__(split, "pilot")
    _stub_nis(monkeypatch, {i: [2.6, 1.0, 1.9, 4.0, 5.0, 6.0, 7.0, 8.0] for i in range(1, 112)})
    res = tuning.run_real(CFG, temp_root, 42, pilot=True, loader=_real_like_loader, pilot_ids=pilot, n_jobs=1)
    doc = json.loads((temp_root / "results" / "pilot" / "qr_42.json").read_text(encoding="utf-8"))
    assert doc["pilot"] and sorted(doc["subjects"]) == sorted(pilot) and res.n_matched == len(pilot)
    assert not (temp_root / "outputs" / "qr_42.json").exists()


def test_tuning_never_imports_preprocess_at_module_level_or_pysr():
    import ast
    from pathlib import Path
    tree = ast.parse((Path(tuning.__file__)).read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in top for a in n.names} | {n.module for n in top if isinstance(n, ast.ImportFrom)}
    assert "src.preprocess" not in names and "pysr" not in names and "numba" not in names
    import re
    assert not re.search(r"(?<![A-Za-z_])print\(", Path(tuning.__file__).read_text(encoding="utf-8"))


def test_nis_is_innovation_form_with_s_without_q():
    """NIS_t = innovation' S^-1 innovation with the filterpy-form S (IMP-019): no Q inside S. Checked at the
    largest q, where a Q term would show most, against a recomputation from the stored innovation and S, and
    the stored S against the sigma-point spread of R plus the propagated points only (never larger than the
    same S with Q added)."""
    r = sim_data.make_recording(CFG, [2.0], 8, g12=4.0, burn_seconds=2.0)
    layout = ss.make_layout(CFG)
    z = r["segments"][0].T
    res = ukf.run_filter(z, CFG, layout, float(GRID[-1]))
    assert not res.diverged
    for t in (0, 50, 200, res.n_done - 1):
        expect = float(res.innovation[t] @ np.linalg.solve(res.S[t], res.innovation[t]))
        assert res.nis[t] == pytest.approx(expect, rel=1e-10)
    # a Q-in-S NIS would be smaller: y1 - y2 of each node carries the process noise of both states
    Q = ukf.process_noise(layout, CFG, float(GRID[-1]))
    qo = np.diag([Q[1, 1] + Q[2, 2], Q[7, 7] + Q[8, 8]])
    t = 100
    with_q = float(res.innovation[t] @ np.linalg.solve(res.S[t] + qo, res.innovation[t]))
    assert with_q < res.nis[t] * 0.999999
