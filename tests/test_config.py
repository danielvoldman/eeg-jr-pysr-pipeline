import subprocess
import sys

import pytest
import yaml

from src.config import ConfigError, load_config
from conftest import REPO_ROOT


def _write(tmp_path, text):
    p = tmp_path / "c.yml"
    p.write_text(text, encoding="utf-8")
    return p


def test_real_config_loads():
    cfg = load_config()
    assert cfg["compute"]["julia_threads"] == 8
    assert cfg["compute"]["env_vars_before_numpy_import"]["OMP_NUM_THREADS"] == "1"
    assert cfg["paths"]["pilot_phase_flag_pattern"] == "results/pilot/phase{n}.done"


def test_meta_passes_through_untouched():
    cfg = load_config()
    raw = yaml.safe_load((REPO_ROOT / "config.yml").read_text(encoding="utf-8"))
    assert cfg["meta"] == raw["meta"]
    assert cfg["meta"]["document_version"] == "v0.6"


def test_value_without_prov_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, 'a:\n  x: {value: 1, ref: "§1"}\n'))


def test_prov_without_value_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, 'a:\n  x: {prov: locked, ref: "§1"}\n'))


def test_unknown_prov_and_bad_ref_raise(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, 'a:\n  x: {value: 1, prov: guess, ref: "§1"}\n'))
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, 'a:\n  x: {value: 1, prov: locked}\n'))


def test_null_allowed_only_for_unset_and_computed(tmp_path):
    ok = _write(tmp_path,
                'a:\n  u: {value: null, prov: unset, ref: "§1"}\n'
                '  c: {value: null, prov: computed, ref: "§1"}\n')
    cfg = load_config(ok)
    assert cfg["a"]["u"] is None and cfg["a"]["c"] is None
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, 'a:\n  x: {value: null, prov: locked, ref: "§1"}\n'))


def test_config_does_not_import_numpy():
    code = "import sys; import src.config; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
