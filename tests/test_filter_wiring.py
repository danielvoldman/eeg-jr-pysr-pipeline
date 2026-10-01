"""F0 tests: real-data runs use filter A at q_fixed by default (DEV-005, DEV-006, IMP-074).

Expected values are literals written here (12 neural + 7 parameters + 2 noise states = 21; windows 12 + 2 = 14;
q_fixed 1e-2 from DEV-005), numbers pinned from HEAD c25886b (tests/test_filter_golden.py), arithmetic done in the
test, or a plain NumPy-backend filter as the oracle. Nothing touches config.yml except reading it.
"""
import copy
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import passes, state_space as ss, synthetic_gate as sg, tuning, ukf, ukf_ext
from src.config import load_config
from sim_data import make_recording

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
Q_DECLARED = 1.0e-2             # DEV-005: written here, not read from config


@pytest.fixture(scope="module")
def rec():
    return make_recording(CFG, (8, 6), seed=21, g12=8.0, g21=4.0, m=0.3, p=np.array([220.0, 250.0]))


@pytest.fixture()
def spy(monkeypatch):
    """Records every run_filter_ext call as (base n, extra n) and every plain ukf.run_filter call."""
    calls = {"ext": [], "plain": 0}
    real_ext, real_plain = ukf_ext.run_filter_ext, ukf.run_filter

    def ext(z, cfg, layout, q, sp, **kw):
        calls["ext"].append((layout.n, sp.nx, q, sp))
        return real_ext(z, cfg, layout, q, sp, **kw)

    def plain(*a, **k):
        calls["plain"] += 1
        return real_plain(*a, **k)
    monkeypatch.setattr(ukf_ext, "run_filter_ext", ext)
    monkeypatch.setattr(ukf, "run_filter", plain)
    return calls


# ---- defaults -----------------------------------------------------------------------------------------

def test_default_real_data_run_is_filter_A_21d_pass1_and_14d_windows(rec, spy):
    out = passes.run_recording(rec["segments"], rec["starts"], CFG)
    assert out.pass1.filter_name == "A" and out.pass2.filter_name == "A"
    assert out.pass1.state_dim == 12 + 7 + 2 == 21 and out.pass2.state_dim == 12 + 2 == 14
    assert spy["plain"] == 0                                   # the plain 19-D filter is never called
    n_windows = len(out.pass2.windows)
    assert [(a, b) for a, b, _, _ in spy["ext"]][:2] == [(19, 2), (19, 2)]       # 19 base states + 2 noise states
    assert all(v == (12, 2) for v in [(a, b) for a, b, _, _ in spy["ext"]][2:]) and len(spy["ext"]) == 2 + n_windows


def test_default_q_equals_the_declared_value_and_is_read_from_config(rec):
    seg, st = rec["segments"][:1], rec["starts"][:1]
    assert passes.run_pass1(seg, st, CFG, forward_only=True).q == Q_DECLARED
    cfg2 = copy.deepcopy(CFG)
    cfg2["ukf"]["process_noise"]["q_fixed"] = 3.0e-3
    assert passes.run_pass1(seg, st, cfg2, forward_only=True).q == 3.0e-3         # read, not hard-coded
    assert passes.run_pass1(seg, st, cfg2, 5.0e-3, forward_only=True).q == 5.0e-3  # an explicit q wins


def test_explicit_19d_is_the_plain_filter_and_needs_an_explicit_q(rec, spy):
    seg, st = rec["segments"][:1], rec["starts"][:1]
    res = passes.run_pass1(seg, st, CFG, 1.0e-2, forward_only=True, filter_name="19D")
    assert res.filter_name == "19D" and res.state_dim == 19 and res.spec is None
    assert spy["plain"] == 1 and spy["ext"] == []
    with pytest.raises(passes.PassError, match="explicit q"):
        passes.run_pass1(seg, st, CFG, forward_only=True, filter_name="19D")
    with pytest.raises(passes.PassError, match="explicit q"):
        passes.run_recording(seg, st, CFG, filter_name="19D")


def test_default_equals_the_old_patched_context_path_bit_for_bit(rec):
    seg, st = rec["segments"], rec["starts"]
    new = passes.run_pass1(seg, st, CFG)
    spec = ukf_ext.spec_for("A", seg, CFG)
    with ukf_ext.patched_filters(spec):                           # the F0-era wiring: ukf.run_filter replaced
        old = passes.run_pass1(seg, st, CFG, Q_DECLARED, filter_name="19D")
    for k in ("g12", "g21", "m", "p1", "p2", "log_rho1", "log_rho2"):
        assert getattr(new.params, k) == getattr(old.params, k), k
    assert new.gain_estimate == old.gain_estimate


def test_no_global_patch_is_left_behind(rec):
    before = (ukf.run_filter, ukf.run_smoother)
    passes.run_recording(rec["segments"][:1], rec["starts"][:1], CFG)
    assert (ukf.run_filter, ukf.run_smoother) == before


def test_pass1_and_pass2_share_one_spec_estimated_once(rec, spy, monkeypatch):
    n = []
    real = ukf_ext.spec_for
    monkeypatch.setattr(ukf_ext, "spec_for", lambda *a, **k: (n.append(1), real(*a, **k))[1])
    passes.run_recording(rec["segments"][:1], rec["starts"][:1], CFG)
    assert len(n) == 1
    specs = {id(sp) for _, _, _, sp in spy["ext"]}
    assert len(specs) == 1 and len(spy["ext"]) > 2                # pass 1 and the windows all use the same spec


# ---- one config source for the filter -------------------------------------------------------------------------

@pytest.mark.parametrize("name, expect", [("A", "A"), ("B", "B"), ("19D", "19D")])
def test_g0_filter_in_config_decides_the_default_and_the_gate_agrees(name, expect):
    cfg2 = copy.deepcopy(CFG)
    cfg2["g0"]["filter"] = name
    assert passes.resolve_filter(cfg2) == expect
    assert sg.gate_filters(cfg2, pilot=False) == [expect]
    assert passes.resolve_filter(cfg2, "B") == "B"                 # an explicit name wins


def test_unset_or_unknown_filter_is_refused_everywhere():
    cfg2 = copy.deepcopy(CFG)
    cfg2["g0"]["filter"] = None
    with pytest.raises(passes.PassError, match="unset"):
        passes.resolve_filter(cfg2)
    with pytest.raises(sg.GateError, match="unset"):
        sg.gate_filters(cfg2, pilot=False)
    with pytest.raises(passes.PassError):
        passes.resolve_filter(CFG, "Z")


def test_only_passes_py_reads_the_g0_filter_leaf():
    pat = re.compile(r"""\[["']g0["']\]\s*\[["']filter["']\]""")
    hits = [p.name for p in (REPO_ROOT / "src").glob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert hits == ["passes.py"]


def test_resolve_q_ignores_any_qr_file(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "qr_42.json").write_text(json.dumps({"result": {"q": 1e-4}}))
    assert passes.resolve_q(CFG) == Q_DECLARED


# ---- the section 7.5 rule on the real-data path --------------------------------------------------------------------

GRID = tuning.q_grid(CFG)


def _split():
    return json.loads((REPO_ROOT / "outputs" / "split_42.json").read_text(encoding="utf-8"))


def test_run_real_reports_the_rule_but_declares_q_fixed(temp_root, monkeypatch):
    import main as main_mod
    split = _split()
    pilot = split["pilot"]
    monkeypatch.setattr(main_mod, "training_subjects", lambda seed, root=None, cfg=None: list(split["train"]))
    seen = []

    def fake(segments, starts, cfg, q, filter_name=None):
        seen.append(filter_name)
        j = int(np.argmin(np.abs(GRID - q)))
        return {"mean_nis": [2.6, 1.0, 1.9, 4.0, 5.0, 6.0, 7.0, 8.0][j], "n_samples": 100, "recording_diverged": False,
                "n_clean": 100, "n_diverged_samples": 0, "n_segments": 1, "n_segments_diverged": 0}
    monkeypatch.setattr(tuning, "recording_nis", fake)
    loader = lambda s: {"reason": None, "segments": [np.full((2, 10), float(s.split("-")[1]))], "starts": [0],
                        "key": "k" + s}                                           # noqa: E731
    res = tuning.run_real(CFG, temp_root, 42, pilot=True, loader=loader, pilot_ids=pilot, n_jobs=1)
    doc = json.loads((temp_root / "results" / "pilot" / "qr_42.json").read_text(encoding="utf-8"))
    assert set(seen) == {"A"}                                      # default filter on the real-data path
    assert doc["filter"] == "A" and doc["rule_role"] == "reported_not_used"
    assert res.q == pytest.approx(GRID[2]) and doc["q_used"] == Q_DECLARED != doc["result"]["q"]


def test_cache_key_depends_on_the_filter():
    assert tuning.cache_key("r", 1e-2, CFG, "A") != tuning.cache_key("r", 1e-2, CFG, "19D")
    assert tuning.cache_key("r", 1e-2, CFG, "A") == tuning.cache_key("r", 1e-2, CFG, "A")


def test_a_change_in_ukf_ext_invalidates_the_tuning_cache(tmp_path, monkeypatch):
    assert "ukf_ext.py" in tuning._CODE_FILES
    for name in tuning._CODE_FILES:
        shutil.copy(REPO_ROOT / "src" / name, tmp_path / name)
    monkeypatch.setattr(tuning, "__file__", str(tmp_path / "tuning.py"))
    h0, k0 = tuning.code_hash(), tuning.cache_key("r", 1e-2, CFG, "A")
    (tmp_path / "other.py").write_text("x = 1\n")                  # an unrelated file changes nothing
    assert tuning.code_hash() == h0
    with open(tmp_path / "ukf_ext.py", "a", encoding="utf-8") as fh:
        fh.write("\n# changed\n")
    assert tuning.code_hash() != h0 and tuning.cache_key("r", 1e-2, CFG, "A") != k0


def test_recording_nis_default_is_A_and_differs_from_19d(rec):
    seg, st = rec["segments"][:1], rec["starts"][:1]
    a = tuning.recording_nis(seg, st, CFG, 1.0e-2)
    d = tuning.recording_nis(seg, st, CFG, 1.0e-2, filter_name="19D")
    assert a["mean_nis"] != d["mean_nis"]
    assert a["mean_nis"] == tuning.recording_nis(seg, st, CFG, 1.0e-2, filter_name="A")["mean_nis"]


# ---- M1 layout through the ext kernel; z_pred and per-sample squared error -------------------------------------------

def test_ext_kernel_runs_a_no_gain_m1_layout(rec):
    """17 + 2 noise states = 19-D (DEV-006). The sigma-point set depends on the state count, so a 19-D filter is a
    different filter from the plain 17-D one; the no-gain layout is therefore checked with no extra state (the kernel
    then equals the plain filter; the plain NumPy 17-D filter is the oracle) and with the two noise states (runs, finite)."""
    m1 = ss.make_layout(CFG, include_gains=False)
    assert m1.n == 17
    z = np.ascontiguousarray(rec["segments"][0][:, :600].T)
    zero = ukf_ext.run_filter_ext(z, CFG, m1, 1.0e-2, ukf_ext.Spec("N"))
    oracle = ukf.run_filter(z, CFG, m1, 1.0e-2, backend="numpy")   # plain 17-D NumPy filter, independent code
    np.testing.assert_allclose(zero.x, oracle.x, rtol=1e-8, atol=1e-10)
    full = ukf_ext.run_filter_ext(z, CFG, m1, 1.0e-2, ukf_ext.spec_for("A", [z.T], CFG))
    assert not full.diverged and full.x.shape == (600, 17 + 2) and np.all(np.isfinite(full.x))
    res = passes.run_pass1(rec["segments"][:1], rec["starts"][:1], CFG, forward_only=True, layout=m1)
    assert res.state_dim == 17 + 2 == 19 and not res.recording_diverged and res.gain_estimate is None


@pytest.mark.parametrize("name", ["19D", "A"])
def test_z_pred_and_squared_error_are_exposed_and_correct(rec, name):
    seg = rec["segments"][0][:, :700]
    res = passes.run_pass1([seg], [0], CFG, 1.0e-2, forward_only=True, filter_name=name)
    s = res.segments[0]
    assert s.z_pred.shape == (700, 2) and s.sq_err.shape == (700,) and np.all(np.isfinite(s.z_pred))
    expect = ((seg.T - s.z_pred) ** 2).mean(axis=1)                # arithmetic done here: mean over both channels
    np.testing.assert_allclose(s.sq_err, expect, rtol=0, atol=1e-12)
    if name == "19D":                                              # one-step prediction of the plain NumPy filter
        layout = ss.make_layout(CFG)
        oracle = ukf.run_filter(np.ascontiguousarray(seg.T), CFG, layout, 1.0e-2, backend="numpy")
        np.testing.assert_allclose(s.z_pred, oracle.z_pred, rtol=1e-6, atol=1e-8)
        # the innovation is observation minus prediction
        np.testing.assert_allclose(seg.T - s.z_pred, oracle.innovation, rtol=1e-6, atol=1e-8)


def test_diverged_segment_has_no_prediction(rec):
    bad = np.full((2, 400), np.nan)
    res = passes.run_pass1([bad], [0], CFG, 1.0e-2, forward_only=True, filter_name="19D")
    s = res.segments[0]
    assert s.diverged and s.z_pred is None and s.sq_err is None
