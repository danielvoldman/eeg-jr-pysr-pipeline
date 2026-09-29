"""A4 tests: reference constants mu_ref and sigma_ref (§5.1, §7.6; IMP-005).

Expected values come from closed-form arithmetic with plain `math`, never from
the code under test. Nothing here touches the real config.yml, outputs/,
results/ or logs/ (a test asserts the real config.yml bytes are unchanged).
"""
import copy
import hashlib
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import model
from src.config import DEFAULT_CONFIG_PATH, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
REF = CFG["rescaling"]["reference_simulation"]
BP = CFG["preprocessing"]["bandpass"]
FS = REF["sim_fs_hz"]
FS_OBS = CFG["preprocessing"]["observation_fs_hz"]
ORDER = BP["filter_order"]
LP = BP["lowpass_hz"]
HP = BP["highpass_hz"]
REAL_CONFIG_SHA = hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()


# ---- closed form (independent of preprocess.py; same formula as test_preprocess.py) ---
# One Butterworth pass, bilinear transform: |H| = 1/sqrt(1 + r^(2n)); forward-backward
# squares it: 1/(1 + r^(2n)). low-pass r = tan(pi f/fs)/tan(pi fc/fs); high-pass inverted.

def ref_gain_warped(f, hp, fs):
    lp_part = 1.0 / (1.0 + (math.tan(math.pi * f / fs) / math.tan(math.pi * LP / fs)) ** (2 * ORDER))
    hp_part = 1.0 / (1.0 + (math.tan(math.pi * hp / fs) / math.tan(math.pi * f / fs)) ** (2 * ORDER))
    return hp_part * lp_part


def short_cfg(duration_s, burn_in_s):
    cfg = copy.deepcopy(CFG)
    cfg["rescaling"]["reference_simulation"]["duration_s"] = duration_s
    cfg["rescaling"]["reference_simulation"]["burn_in_s"] = burn_in_s
    return cfg


def known_signal(offset, amp_fast, amp_slow, seconds):
    t = np.arange(int(round(seconds * FS))) / FS
    return (offset + amp_fast * np.sin(2.0 * math.pi * 10.0 * t + 0.3)
            + amp_slow * np.sin(2.0 * math.pi * 0.05 * t + 1.1))


# ---- reference_statistics on a known signal ------------------------------------------

def test_reference_statistics_known_signal(capsys):
    # 120 s = 6 cycles of the 0.05 Hz sine (period 20 s) and 1200 cycles of the 10 Hz sine,
    # so the unfiltered mean is exactly the offset up to round-off (~1e-13); tolerance 1e-9.
    offset, a_fast, a_slow, seconds = 7.5, 1.0, 50.0, 120
    y = known_signal(offset, a_fast, a_slow, seconds)
    mu, sigma = model.reference_statistics(CFG, y, FS)
    assert abs(mu - offset) < 1e-9
    # Expected sigma: the offset and the 0.05 Hz sine are removed by the high-pass (the
    # 0.05 Hz gain is 1/(1 + r^8) with r ~ 10, ~1e-8, so 50 mV leaves ~5e-7 mV: 5e-7 of
    # the ~0.7 mV result); the 10 Hz sine keeps A/sqrt(2) * g(10 Hz). The trimmed segment
    # holds whole 10 Hz cycles (trim 5 s, 0.1 s period). ddof=1 vs the population SD
    # changes the result by a factor sqrt(N/(N-1)) ~ 1 + 1/(2N) ~ 1.8e-5 at N ~ 28,000.
    expected = a_fast / math.sqrt(2.0) * ref_gain_warped(10.0, HP, FS)
    rel = abs(sigma - expected) / expected
    with capsys.disabled():
        print(f"\n[known signal] mu={mu:.12g} (offset {offset}); sigma={sigma:.12g} "
              f"expected {expected:.12g}; relative error {rel:.3e}")
    assert rel < 1e-4


# ---- simulated series -------------------------------------------------------------------

def test_kept_duration_exact_and_burn_in_discarded():
    cfg = short_cfg(20, 2)
    y = model.simulate_reference(cfg, seed=7)
    assert y.size == 20 * FS
    assert y.dtype == np.float64
    n_burn = 2 * FS
    full = model.simulate(cfg, n_burn + 20 * FS, seed=7, n_nodes=1, p=REF["p_s_inv"]).output[:, 0]
    # output has n_steps + 1 rows (row 0 = initial state); kept rows are 1 + n_burn ... end
    np.testing.assert_array_equal(y, full[1 + n_burn:])
    assert full.size == n_burn + 20 * FS + 1


def test_real_config_keeps_600_s():
    y = model.simulate_reference(CFG)
    assert y.size == 600 * FS
    assert np.all(np.isfinite(y))


def test_same_seed_bit_identical_different_seed_differs():
    cfg = short_cfg(30, 2)
    a = model.compute_reference_constants(cfg, seed=11)
    b = model.compute_reference_constants(cfg, seed=11)
    c = model.compute_reference_constants(cfg, seed=12)
    assert (a["mu_ref"], a["sigma_ref"]) == (b["mu_ref"], b["sigma_ref"])
    assert a["mu_ref"] != c["mu_ref"] and a["sigma_ref"] != c["sigma_ref"]
    assert a["n_mu_samples"] == 30 * FS
    assert a["n_sigma_samples"] == 30 * FS_OBS - 2 * REF["edge_trim_s"] * FS_OBS


def test_default_seed_is_config_seed():
    cfg = short_cfg(30, 2)
    a = model.compute_reference_constants(cfg)
    b = model.compute_reference_constants(cfg, seed=REF["seed"])
    assert (a["mu_ref"], a["sigma_ref"]) == (b["mu_ref"], b["sigma_ref"])


# ---- trim at both ends --------------------------------------------------------------------

@pytest.mark.parametrize("end", ("start", "end"))
def test_edge_transient_excluded_by_trim(end, capsys):
    seconds = 60
    y = known_signal(0.0, 1.0, 0.0, seconds)
    _, clean = model.reference_statistics(CFG, y, FS)
    burst = y.copy()
    n = int(round(1.0 * FS))          # a 1-s burst, 100x the signal, inside the trimmed 5 s
    seg = slice(FS, FS + n) if end == "start" else slice(y.size - FS - n, y.size - FS)
    burst[seg] += 100.0 * np.sin(2.0 * math.pi * 3.0 * np.arange(n) / FS)
    _, trimmed = model.reference_statistics(CFG, burst, FS)
    cfg0 = copy.deepcopy(CFG)
    cfg0["rescaling"]["reference_simulation"]["edge_trim_s"] = 0
    _, untrimmed = model.reference_statistics(cfg0, burst, FS)
    with capsys.disabled():
        print(f"\n[trim, burst at {end}] clean {clean:.9g}, trimmed {trimmed:.9g}, "
              f"no trim {untrimmed:.9g}")
    # The 3 Hz burst is 5 s or more from the kept interval, and the 0.5-45 Hz filter's
    # tails 4 s past it are below 1e-3 of the burst; without trim the SD is many times larger.
    assert abs(trimmed - clean) < 1e-3 * clean
    assert untrimmed > 10.0 * clean


# ---- git guard --------------------------------------------------------------------------------

def fake_git(monkeypatch, head, porcelain):
    def _git(args):
        return {"rev-parse": head + "\n", "status": porcelain}[args[0]]
    monkeypatch.setattr(model, "_git", _git)


def test_check_git_state(monkeypatch):
    fake_git(monkeypatch, "abc123", "")
    assert model.check_git_state("abc123") == "abc123"
    with pytest.raises(model.ModelError, match="HEAD is abc123"):
        model.check_git_state("def456")
    fake_git(monkeypatch, "abc123", " M src/model.py\n")
    with pytest.raises(model.ModelError, match="porcelain"):
        model.check_git_state("abc123")
    fake_git(monkeypatch, "abc123", "?? new.txt\n")
    with pytest.raises(model.ModelError, match="porcelain"):
        model.check_git_state("abc123")


def test_real_write_refused_on_dirty_tree_before_any_computation(monkeypatch, tmp_path):
    cfg_copy = tmp_path / "config.yml"
    shutil.copy(DEFAULT_CONFIG_PATH, cfg_copy)
    before = cfg_copy.read_bytes()
    fake_git(monkeypatch, "abc123", " M x\n")

    def boom(*a, **k):
        raise AssertionError("computation must not start")
    monkeypatch.setattr(model, "compute_reference_constants", boom)
    assert model.main(["--compute-reference", "--expect-commit", "abc123"], config_path=cfg_copy) == 1
    fake_git(monkeypatch, "abc123", "")
    assert model.main(["--compute-reference", "--expect-commit", "zzz"], config_path=cfg_copy) == 1
    assert cfg_copy.read_bytes() == before


def test_real_write_needs_expect_commit(tmp_path):
    with pytest.raises(SystemExit):
        model.main(["--compute-reference"], config_path=tmp_path / "config.yml")


# ---- config writer (on a temp copy) ---------------------------------------------------------

PROV = {"seed": 42, "units": model.UNITS, "git_commit": "abc123", "git_dirty": False,
        "kept_duration_s": 600}


@pytest.fixture
def cfg_copy(tmp_path):
    p = tmp_path / "config.yml"
    shutil.copy(DEFAULT_CONFIG_PATH, p)
    return p


def test_writer_changes_only_intended_lines(cfg_copy):
    mu, sigma = 7.123456789012345, 1.0987654321098765
    before = cfg_copy.read_bytes().decode("utf-8").splitlines(keepends=True)
    model.write_reference_constants(cfg_copy, mu, sigma, PROV)
    after = cfg_copy.read_bytes().decode("utf-8").splitlines(keepends=True)
    assert len(before) == len(after)
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(changed) == 3
    keys = []
    for i in changed:
        assert before[i].startswith("  ") and "{value: null, prov: computed" in before[i]
        # everything except the value text is byte-identical (including the line ending)
        assert after[i].replace(after[i].split("{value: ")[1].split(", prov:")[0], "null", 1) == before[i]
        keys.append(before[i].split(":")[0].strip())
    assert keys == ["mu_ref", "sigma_ref", "computed_provenance"]
    assert [l for l in before if l.lstrip().startswith("#")] == [l for l in after if l.lstrip().startswith("#")]
    # line endings survive: the CRLF count is unchanged
    assert cfg_copy.read_bytes().count(b"\r\n") == DEFAULT_CONFIG_PATH.read_bytes().count(b"\r\n")


def test_writer_floats_round_trip_and_are_floats(cfg_copy):
    mu, sigma = 7.123456789012345, 1.0987654321098765
    model.write_reference_constants(cfg_copy, mu, sigma, PROV)
    cfg = load_config(cfg_copy)
    assert type(cfg["rescaling"]["mu_ref"]) is float and type(cfg["rescaling"]["sigma_ref"]) is float
    assert cfg["rescaling"]["mu_ref"] == mu
    assert cfg["rescaling"]["sigma_ref"] == sigma
    assert cfg["rescaling"]["computed_provenance"] == PROV


@pytest.mark.parametrize("x", (1e-05, 1e16, 1.5e-05, -2e-07, 3.0, 0.1 + 0.2))
def test_yaml_float_is_read_back_as_the_same_float(x):
    import yaml
    text = model.yaml_float(x)
    if "e" in text:
        assert "." in text.split("e")[0]          # 1e-05 would be read by PyYAML as a string
    back = yaml.safe_load(f"v: {text}")["v"]
    assert type(back) is float and back == x


def test_writer_small_sigma_written_with_dot(cfg_copy):
    model.write_reference_constants(cfg_copy, 1e-05, 2e-05, PROV)
    text = cfg_copy.read_text(encoding="utf-8")
    assert "value: 1.0e-05" in text and "1e-05" not in text
    cfg = load_config(cfg_copy)
    assert cfg["rescaling"]["mu_ref"] == 1e-05 and type(cfg["rescaling"]["mu_ref"]) is float


def test_writer_second_write_refuses_without_force(cfg_copy):
    model.write_reference_constants(cfg_copy, 7.0, 1.0, PROV)
    after_first = cfg_copy.read_bytes()
    with pytest.raises(model.ModelError, match="DEVIATIONS.md"):
        model.write_reference_constants(cfg_copy, 8.0, 2.0, PROV)
    assert cfg_copy.read_bytes() == after_first
    model.write_reference_constants(cfg_copy, 8.0, 2.0, PROV, force=True)
    assert load_config(cfg_copy)["rescaling"]["mu_ref"] == 8.0


def test_writer_rejects_non_finite(cfg_copy):
    before = cfg_copy.read_bytes()
    with pytest.raises(model.ModelError):
        model.write_reference_constants(cfg_copy, float("nan"), 1.0, PROV)
    assert cfg_copy.read_bytes() == before


def test_provenance_record_fields():
    result = {"seed": 42, "n_mu_samples": 600 * FS, "n_sigma_samples": 590 * FS_OBS}
    rec = model.provenance_record(CFG, result, "abc123", False)
    for key in ("seed", "kept_duration_s", "burn_in_s", "sim_fs_hz", "observation_fs_hz",
                "edge_trim_s", "ddof", "units", "git_commit", "git_dirty", "numpy", "scipy", "numba"):
        assert key in rec
    assert "mV" in rec["units"]
    assert rec["kept_duration_s"] == 600 and rec["burn_in_s"] == 10


# ---- config leaves and hygiene ------------------------------------------------------------------

def test_new_config_leaves_are_placeholders():
    import yaml
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    refsim = raw["rescaling"]["reference_simulation"]
    for key in ("edge_trim_s", "ddof", "diagnostic_seeds"):
        assert refsim[key]["prov"] == "placeholder", key
    assert refsim["edge_trim_s"]["ref"] == "§5.1"
    assert raw["rescaling"]["computed_provenance"]["prov"] == "computed"


def test_real_config_untouched_by_this_module():
    # runs last in this file; the hash was taken at import
    assert hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest() == REAL_CONFIG_SHA
