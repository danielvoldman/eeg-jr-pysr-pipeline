"""G7 tests: the §5.2 real-data preprocessing-sensitivity analysis (src/robustness.py variant_loader, resolve_sensitivity_subjects,
sens_matched, sens_summary, run_sensitivity; §5.2; IMP-087). Expected values are hand arithmetic (|85 - 100| / 100 = 0.15,
Spearman 1 - 6 * 2 / 60 = 0.8), the independent Q/R draw, and stub recordings; never read back from the code under test."""
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import preprocess as pp
from src import robustness as rb
from src import tuning
from src.config import load_config
from sim_data import make_recording

CFG = load_config()
REPO = Path(__file__).resolve().parent.parent
VARIANTS = [dict(v) for v in CFG["statistics"]["sensitivity"]["variants"]]
NAMES = [v["name"] for v in VARIANTS]


def row(g12=1.0, g21=1.0, lr1=-2.0, lr2=-2.0, m=0.2, p1=100.0, p2=100.0, diverged=False, params=True):
    return {"recording_diverged": diverged, "diverged_fraction": 1.0 if diverged else 0.0, "n_clean": 1, "clean_s": 1.0,
            "n_diverged": 0, "n_segments": 1, "n_segments_diverged": 0,
            "params": {"g12": g12, "g21": g21, "log_rho1": lr1, "log_rho2": lr2, "m": m, "p1": p1, "p2": p2} if params else None}


def table_of(*per_subject):
    """per_subject: {variant name: row or None} per subject."""
    return {f"s{i}": d for i, d in enumerate(per_subject)}


# ---- configuration ----------------------------------------------------------------------------------------------------------------

def test_the_four_variants_are_the_two_by_two_of_section_5_2_and_the_reference_is_the_primary_pipeline():
    got = {(v["highpass"], v["strict"]) for v in VARIANTS}
    assert got == {(False, False), (False, True), (True, False), (True, True)} and len(VARIANTS) == 4
    assert CFG["statistics"]["sensitivity"]["reference_variant"] == "hp0.5_default" == NAMES[0]
    assert CFG["g0"]["preprocessing_gate"]["sensitivity_subjects"] == 20 and CFG["preprocessing"]["sensitivity_highpass_hz"] == 0.1
    assert CFG["statistics"]["sensitivity"]["judged_parameters"] == ["log_rho1", "log_rho2", "g12", "g21"]


# ---- matched subjects and the summary -----------------------------------------------------------------------------------------

def test_only_subjects_usable_in_every_variant_are_matched():
    ok = {n: row() for n in NAMES}
    excluded_in_strict = {n: (None if n == "hp0.5_strict" else row()) for n in NAMES}
    diverged_in_slow = {n: row(diverged=(n == "hp0.1_strict")) for n in NAMES}
    no_params = {n: row(params=(n != "hp0.1_default")) for n in NAMES}
    t = table_of(ok, excluded_in_strict, diverged_in_slow, no_params, ok)
    assert rb.sens_matched(t, VARIANTS) == ["s0", "s4"]


def test_median_absolute_relative_change_and_the_robust_flag_with_the_limit_included():
    t = table_of({"hp0.5_default": row(g12=100.0), "hp0.5_strict": row(g12=85.0), "hp0.1_default": row(g12=60.0),
                  "hp0.1_strict": row(g12=100.0)},
                 {"hp0.5_default": row(g12=4.0), "hp0.5_strict": row(g12=4.4), "hp0.1_default": row(g12=5.0),
                  "hp0.1_strict": row(g12=4.0)},
                 {"hp0.5_default": row(g12=-5.0), "hp0.5_strict": row(g12=-5.5), "hp0.1_default": row(g12=-4.0),
                  "hp0.1_strict": row(g12=-5.0)})
    out = rb.sens_summary(t, ["s0", "s1", "s2"], VARIANTS, CFG)
    assert "hp0.5_default" not in out                                              # the reference is not compared with itself
    strict = out["hp0.5_strict"]["g12"]
    assert strict["median_abs_relative_change"] == pytest.approx(0.1) and strict["robust"] is True and strict["n"] == 3
    slow = out["hp0.1_default"]["g12"]                                              # 0.4, 0.25, 0.2 -> median 0.25
    assert slow["median_abs_relative_change"] == pytest.approx(0.25) and slow["robust"] is False
    assert out["hp0.1_strict"]["g12"]["median_abs_relative_change"] == 0.0
    assert slow["median_abs_reference"] == pytest.approx(5.0)
    edge = rb.sens_summary(table_of(*[{n: row(g12=(100.0 if n == "hp0.5_default" else 85.0)) for n in NAMES}] * 3),
                           ["s0", "s1", "s2"], VARIANTS, CFG)
    assert edge["hp0.5_strict"]["g12"]["median_abs_relative_change"] == 0.15 and edge["hp0.5_strict"]["g12"]["robust"] is True


def test_p_is_never_judged_and_m_and_p_are_reported_without_a_robust_flag():
    t = table_of({n: row(p1=100.0 if n == "hp0.5_default" else 50.0) for n in NAMES})
    out = rb.sens_summary(t, ["s0"], VARIANTS, CFG)["hp0.5_strict"]
    assert out["p1"]["judged"] is False and out["p1"]["robust"] is None and out["p1"]["median_abs_relative_change"] == 0.5
    assert out["m"]["judged"] is False and out["g12"]["judged"] is True and out["log_rho1"]["judged"] is True


def test_spearman_is_reported_without_a_threshold_and_needs_three_subjects_with_spread():
    vals = {"hp0.5_default": [1.0, 2.0, 3.0, 4.0], "hp0.5_strict": [1.0, 3.0, 2.0, 4.0]}
    t = table_of(*[{n: row(g12=vals.get(n, [1.0] * 4)[i]) for n in NAMES} for i in range(4)])
    e = rb.sens_summary(t, ["s0", "s1", "s2", "s3"], VARIANTS, CFG)
    assert e["hp0.5_strict"]["g12"]["spearman"] == pytest.approx(0.8)                    # 1 - 6 * (0 + 1 + 1 + 0) / (4 * 15)
    assert e["hp0.1_default"]["g12"]["spearman"] is None                                   # constant values: no rank correlation
    two = rb.sens_summary(table_of({n: row(g12=1.0) for n in NAMES}, {n: row(g12=2.0) for n in NAMES}), ["s0", "s1"], VARIANTS, CFG)
    assert two["hp0.5_strict"]["g12"]["spearman"] is None
    assert rb.sens_summary({}, [], VARIANTS, CFG)["hp0.5_strict"]["g12"]["median_abs_relative_change"] is None


# ---- guard ---------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def split_world(monkeypatch):
    import main as main_mod
    train = [f"sub-{i:03d}" for i in range(1, 31)]
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(train))
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the test list was loaded")))
    return train


def test_the_pilot_plan_is_the_pilot_subjects_on_the_training_side_and_never_the_test_list(split_world):
    plan = rb.resolve_sensitivity_subjects(CFG, "/r", 42, pilot=True, pilot_ids=split_world[:5])
    assert plan.mode == "pilot" and plan.ids == tuple(split_world[:5]) and plan.allow_all is False
    with pytest.raises(rb.GuardError, match="training side"):
        rb.resolve_sensitivity_subjects(CFG, "/r", 42, pilot=True, pilot_ids=["sub-099"])
    with pytest.raises(rb.GuardError):
        rb.resolve_sensitivity_subjects(CFG, "/r", 42, pilot=True, confirmatory=True)
    with pytest.raises(rb.GuardError, match="confirmatory=True"):
        rb.resolve_sensitivity_subjects(CFG, "/r", 42, pilot=False)


def test_the_confirmatory_plan_needs_the_gate_and_is_the_q_r_draw_of_training_subjects(split_world, monkeypatch, tmp_path):
    monkeypatch.setattr(tuning, "check_gate", lambda cfg, root: (_ for _ in ()).throw(tuning.GateError("no gate")))
    with pytest.raises(tuning.GateError):
        rb.resolve_sensitivity_subjects(CFG, tmp_path, 42, pilot=False, confirmatory=True)
    monkeypatch.setattr(tuning, "check_gate", lambda cfg, root: {})
    plan = rb.resolve_sensitivity_subjects(CFG, tmp_path, 42, pilot=False, confirmatory=True)
    rng = np.random.default_rng(1000 + 42)                                         # qr_rule.draw_seed_offset + split seed
    perm = rng.permutation(len(split_world))
    assert list(plan.ids) == [sorted(split_world)[i] for i in perm] and plan.allow_all is True and set(plan.ids) == set(split_world)


# ---- the variant loader ---------------------------------------------------------------------------------------------------------------

def test_the_variant_loader_passes_each_variants_flags_to_the_preprocessing_and_keeps_the_dev_guard(tmp_path, monkeypatch):
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    edf = tmp_path / "data" / "sub-001" / "ses-t1" / "eeg" / "sub-001_ses-t1_task-x_eeg.edf"
    edf.parent.mkdir(parents=True)
    edf.write_bytes(b"")
    calls = []

    def seg(cfg, path, data_root, manifest, pilot_ids, *, allow_all=False, sensitivity_highpass=False, strict=False, cache_root=None):
        calls.append((sensitivity_highpass, strict, allow_all))
        return SimpleNamespace(segments=["s"], starts=[0], meta={"sha256": "abc"})

    monkeypatch.setattr(pp, "load_manifest", lambda path: {})
    monkeypatch.setattr(pp, "segment_recording", seg)
    monkeypatch.setattr(pp, "recording_decision", lambda cfg, meta, seg_meta=None: {"status": "kept", "reason": None})
    seen_keys = []
    monkeypatch.setattr(pp, "cache_key_b5", lambda cfg, sha, hp, strict: seen_keys.append((hp, strict)) or f"{hp}{strict}")
    load = rb.variant_loader(CFG, tmp_path, frozenset({"sub-001"}), allow_all=False)
    for v in VARIANTS:
        out = load("sub-001", v)
        assert out["reason"] is None and out["key"] == f"b5:{v['highpass']}{v['strict']}"
    assert calls == [(v["highpass"], v["strict"], False) for v in VARIANTS] and seen_keys == [(v["highpass"], v["strict"]) for v in VARIANTS]
    assert load("sub-002", VARIANTS[0])["reason"].startswith("0 ses-t1 EDF files")
    monkeypatch.setattr(pp, "recording_decision", lambda cfg, meta, seg_meta=None: {"status": "excluded", "reason": "insufficient_clean_data"})
    assert load("sub-001", VARIANTS[2])["reason"] == "excluded: insufficient_clean_data"


# ---- run_sensitivity end to end (stub loader, temp root) ---------------------------------------------------------------------------------

def _stub_rec(seed, blow=False, seconds=(8.0, 8.0)):
    rec = make_recording(CFG, seconds, seed, g12=2.0, g21=1.0, m=0.2, p=np.array([200.0, 210.0]), gap_seconds=1.0, burn_seconds=2.0)
    segs = [np.ascontiguousarray(s * 40.0 + 25.0) for s in rec["segments"]] if blow else rec["segments"]
    return {"reason": None, "segments": segs, "starts": rec["starts"], "key": f"k{seed}{blow}"}


def _world(monkeypatch, n=8):
    import main as main_mod
    train = [f"sub-{i:03d}" for i in range(1, n + 1)]
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(train))
    monkeypatch.setattr(main_mod, "test_subjects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the test list was loaded")))
    asked = []

    def loader(subject, variant):
        asked.append((subject, variant["name"]))
        k = train.index(subject)
        if subject == train[0] and variant["highpass"]:
            return _stub_rec(900 + k, blow=True)                                  # diverges only under the slow high-pass
        if subject == train[1] and variant["strict"]:
            return {"reason": "excluded: insufficient_clean_data"}
        return _stub_rec(700 + k + 50 * variant["highpass"])
    return train, loader, asked


@pytest.fixture(scope="module")
def pilot_run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("sensroot")
    shutil.copy(REPO / "config.yml", root / "config.yml")
    train, loader, asked = _world(mp)
    out = rb.run_sensitivity(CFG, root, 42, pilot=True, loader=loader, pilot_ids=train[:5], n_jobs=2)
    yield {"root": root, "out": out, "train": train, "loader": loader, "asked": asked}
    mp.undo()


def test_pilot_run_is_descriptive_mechanics_only_and_reads_only_pilot_subjects(pilot_run):
    d = pilot_run["out"]["doc"]
    pilot = set(pilot_run["train"][:5])
    assert {s for s, _ in pilot_run["asked"]} <= pilot and set(d["subjects"]) == pilot
    assert d["pilot"] and d["mechanics_only"] and d["descriptive_only"] and d["mode"] == "pilot"
    assert d["n_subjects_stand_in"] is True and d["reference_variant"] == "hp0.5_default"
    assert pilot_run["out"]["path"].parent == pilot_run["root"] / "results" / "pilot"
    assert not (pilot_run["root"] / "results" / "sensitivity_42.json").exists()
    assert "p_value" not in json.dumps(d) and set(d["variants"][0]) == {"name", "highpass", "strict"}


def test_pilot_run_reports_per_variant_exclusions_and_divergences_and_uses_matched_subjects_only(pilot_run):
    d = pilot_run["out"]["doc"]
    train = pilot_run["train"]
    pv = d["per_variant"]
    assert pv["hp0.5_strict"]["n_excluded"] == 1 and pv["hp0.1_strict"]["n_excluded"] == 1 and pv["hp0.5_default"]["n_excluded"] == 0
    assert pv["hp0.1_default"]["n_recordings_diverged"] == 1 and pv["hp0.5_default"]["n_recordings_diverged"] == 0
    assert d["excluded_recordings"] == [{"subject": train[1], "variant": "hp0.1_strict", "reason": "excluded: insufficient_clean_data"},
                                        {"subject": train[1], "variant": "hp0.5_strict", "reason": "excluded: insufficient_clean_data"}]
    assert d["matched_subjects"] == train[2:5] and d["n_matched"] == 3 and train[0] not in d["matched_subjects"]
    for name, per in d["summary"].items():
        assert name != "hp0.5_default" and all(e["n"] == 3 for e in per.values())
    assert set(d["summary"]["hp0.1_default"]) == {"log_rho1", "log_rho2", "g12", "g21", "m", "p1", "p2"}
    assert train[0] in d["parameters"] and d["parameters"][train[1]]["hp0.5_strict"] is None


def test_pilot_run_parameters_equal_an_independent_pass_1_of_the_same_recording(pilot_run):
    from src import passes
    d = pilot_run["out"]["doc"]
    sid = pilot_run["train"][2]
    rec = _stub_rec(700 + 2)
    p1 = passes.run_pass1(rec["segments"], rec["starts"], CFG, forward_only=False, spec=passes.make_spec(CFG, "A", rec["segments"]))
    got = d["parameters"][sid]["hp0.5_default"]
    assert got["g12"] == pytest.approx(p1.params.g12, rel=1e-12) and got["log_rho1"] == pytest.approx(p1.params.log_rho1, rel=1e-12)
    slow = _stub_rec(700 + 2 + 50)                                                       # the slow variant got its own data
    assert d["parameters"][sid]["hp0.1_default"]["g12"] != got["g12"] and slow["key"] != rec["key"]


def test_pilot_run_rerun_is_unchanged_whatever_the_worker_count(pilot_run):
    again = rb.run_sensitivity(CFG, pilot_run["root"], 42, pilot=True, loader=pilot_run["loader"], pilot_ids=pilot_run["train"][:5],
                               n_jobs=1, use_cache=False)
    assert again["status"] == "unchanged"


def test_the_loader_is_guarded_and_the_variant_count_is_checked(pilot_run):
    plan = rb.resolve_sensitivity_subjects(CFG, pilot_run["root"], 42, pilot=True, pilot_ids=pilot_run["train"][:5])
    with pytest.raises(rb.GuardError):
        rb.guarded_loader(pilot_run["loader"], plan)(pilot_run["train"][6], VARIANTS[0])
    cfg = copy.deepcopy(CFG)
    cfg["statistics"]["sensitivity"]["variants"] = VARIANTS[:3]
    with pytest.raises(rb.RobustnessError, match="4 of"):
        rb.run_sensitivity(cfg, pilot_run["root"], 42, pilot=True, loader=pilot_run["loader"], pilot_ids=pilot_run["train"][:5])


def test_confirmatory_run_takes_the_first_eligible_subjects_of_the_q_r_draw_and_needs_the_gate(tmp_path, monkeypatch):
    shutil.copy(REPO / "config.yml", tmp_path / "config.yml")
    train, loader, asked = _world(monkeypatch, n=9)
    cfg = copy.deepcopy(CFG)
    cfg["g0"]["preprocessing_gate"]["sensitivity_subjects"] = 4
    monkeypatch.setattr(tuning, "check_gate", lambda c, r: (_ for _ in ()).throw(tuning.GateError("no gate")))
    with pytest.raises(tuning.GateError):
        rb.run_sensitivity(cfg, tmp_path, 42, pilot=False, confirmatory=True, loader=loader)
    assert asked == []
    monkeypatch.setattr(tuning, "check_gate", lambda c, r: {})
    out = rb.run_sensitivity(cfg, tmp_path, 42, pilot=False, confirmatory=True, loader=loader, n_jobs=2)
    d = out["doc"]
    order = [sorted(train)[i] for i in np.random.default_rng(1000 + 42).permutation(len(train))]
    assert d["subjects"] == order[:4] and d["mode"] == "confirmatory" and not d["mechanics_only"] and d["n_subjects_stand_in"] is False
    assert out["path"] == tmp_path / "results" / "sensitivity_42.json"
    ref = [a for a in asked if a[1] == "hp0.5_default"]
    assert [s for s, _ in ref] == order[:4]                                              # stops after n_want, no further subject is read
    assert not any(s in order[4:] for s, _ in asked)
