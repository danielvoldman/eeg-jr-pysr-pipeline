import ast
import multiprocessing
import os
import subprocess
import sys

import pytest

import main
from conftest import REPO_ROOT
from src.config import load_config

ENV_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "PYTHON_JULIACALL_THREADS")
PHASES = (1, 2, 3, 4)


def _flag(root, phase, pilot=False):
    cfg = load_config(root / "config.yml")
    return main.flag_path(cfg, root, phase, pilot)


# ---- thread variables before numpy ------------------------------------------

def test_env_set_on_import_before_numpy_runtime():
    clean = {k: v for k, v in os.environ.items() if k not in ENV_NAMES}
    code = ("import sys, os; import main; "
            "print('numpy' in sys.modules); "
            "print(*[os.environ.get(n) for n in "
            f"{ENV_NAMES!r}])")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=clean,
                         capture_output=True, text=True, check=True)
    numpy_seen, values = out.stdout.strip().splitlines()
    assert numpy_seen == "False"
    assert values.split() == ["1", "1", "1", "8"]


def test_bootstrap_overrides_inherited_values(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    main._bootstrap_env()
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_only_allowed_imports_precede_bootstrap():
    tree = ast.parse((REPO_ROOT / "main.py").read_text(encoding="utf-8"))
    stdlib = set(sys.stdlib_module_names)
    allowed = stdlib | {"yaml"}
    seen_call = False
    checked = 0
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "_bootstrap_env"):
            seen_call = True
            break
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module] if node.module == "src.config" else [node.module.split(".")[0]]
        else:
            continue
        for m in mods:
            assert m in allowed or m == "src.config", f"import {m} before bootstrap"
            checked += 1
    assert seen_call and checked > 0


# ---- spawned worker sees 1 BLAS thread --------------------------------------

def test_spawned_worker_single_blas_thread(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    main._bootstrap_env()  # parent sets env before the pool starts
    import _worker
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        info = pool.apply(_worker.blas_report)
    assert info, "threadpool_info() is empty"
    assert any(p["user_api"] == "blas" for p in info), f"no BLAS library found: {info}"
    for pool_info in info:
        assert pool_info["num_threads"] == 1, pool_info


# ---- phase gating ------------------------------------------------------------

@pytest.mark.parametrize("phase", [2, 3, 4])
def test_phase_refuses_without_previous_flag(temp_root, phase):
    assert main.main(["--phase", str(phase)], root=temp_root) == main.EXIT_PREREQ
    log = (temp_root / "logs" / f"phase{phase}.log").read_text(encoding="utf-8")
    assert "refused" in log and "not implemented" not in log


def test_phase2_starts_with_flag(temp_root):
    f = _flag(temp_root, 1)
    f.parent.mkdir(parents=True)
    f.write_text("x", encoding="utf-8")
    rc = main.main(["--phase", "2"], root=temp_root)
    assert rc == main.EXIT_FAILED  # got past the gate and ran the stub
    log = (temp_root / "logs" / "phase2.log").read_text(encoding="utf-8")
    assert "not implemented" in log


def test_phase1_has_no_predecessor(temp_root):
    assert main.main(["--phase", "1"], root=temp_root) == main.EXIT_FAILED
    log = (temp_root / "logs" / "phase1.log").read_text(encoding="utf-8")
    assert "not implemented" in log and "refused" not in log


def test_invalid_phase_rejected(temp_root):
    for bad in ("0", "5"):
        with pytest.raises(SystemExit) as e:
            main.main(["--phase", bad], root=temp_root)
        assert e.value.code == 2


# ---- flags -------------------------------------------------------------------

@pytest.mark.parametrize("pilot", [False, True])
@pytest.mark.parametrize("phase", PHASES)
def test_stub_never_writes_done_flag(temp_root, phase, pilot):
    for p in range(1, phase):  # satisfy the prerequisite
        f = _flag(temp_root, p, pilot)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    rc = main.main(["--phase", str(phase)] + (["--pilot"] if pilot else []),
                   root=temp_root)
    assert rc == main.EXIT_FAILED
    assert not _flag(temp_root, phase, pilot).exists()


def test_flag_written_only_on_runner_success(temp_root, monkeypatch):
    # Test double for the flag logic only; not a pipeline result.
    monkeypatch.setitem(main.PHASE_RUNNERS, 1, lambda cfg, root, pilot: True)
    assert main.main(["--phase", "1"], root=temp_root) == main.EXIT_OK
    assert _flag(temp_root, 1).is_file()
    monkeypatch.setitem(main.PHASE_RUNNERS, 1, lambda cfg, root, pilot: None)
    _flag(temp_root, 1).unlink()
    assert main.main(["--phase", "1"], root=temp_root) == main.EXIT_FAILED
    assert not _flag(temp_root, 1).exists()


# ---- pilot isolation (IMP-002) ----------------------------------------------

def test_pilot_creates_results_pilot_and_pilot_log(temp_root):
    main.main(["--phase", "1", "--pilot"], root=temp_root)
    cfg = load_config(temp_root / "config.yml")
    assert (temp_root / cfg["paths"]["pilot_results_dir"]).is_dir()
    assert (temp_root / "logs" / "phase1_pilot.log").is_file()
    assert not (temp_root / "logs" / "phase1.log").exists()


def test_pilot_flags_live_under_results_pilot(temp_root, monkeypatch):
    monkeypatch.setitem(main.PHASE_RUNNERS, 1, lambda cfg, root, pilot: True)
    assert main.main(["--phase", "1", "--pilot"], root=temp_root) == main.EXIT_OK
    assert (temp_root / "results" / "pilot" / "phase1.done").is_file()
    assert not (temp_root / "outputs" / "phase1.done").exists()


def test_pilot_ignores_full_flags_and_full_ignores_pilot_flags(temp_root):
    full = _flag(temp_root, 1, False)
    full.parent.mkdir(parents=True)
    full.write_text("x", encoding="utf-8")
    assert main.main(["--phase", "2", "--pilot"], root=temp_root) == main.EXIT_PREREQ
    full.unlink()
    pilot = _flag(temp_root, 1, True)
    pilot.parent.mkdir(parents=True, exist_ok=True)
    pilot.write_text("x", encoding="utf-8")
    assert main.main(["--phase", "2"], root=temp_root) == main.EXIT_PREREQ
    assert main.main(["--phase", "2", "--pilot"], root=temp_root) == main.EXIT_FAILED


# ---- run-time folders --------------------------------------------------------

def test_runtime_dirs_created_by_main(temp_root):
    main.main(["--phase", "1"], root=temp_root)
    for d in ("cache", "outputs", "results", "logs"):
        assert (temp_root / d).is_dir()


def test_runtime_dirs_ignored_by_git():
    for d in ("cache/", "outputs/", "results/", "logs/", "results/pilot/"):
        r = subprocess.run(["git", "check-ignore", "-q", d + "x"], cwd=REPO_ROOT)
        assert r.returncode == 0, f"{d} is not git-ignored"


def test_end_to_end_cli_stub_exit_code(temp_root):
    """The real entry point, run as a script, exits non-zero for a stub."""
    r = subprocess.run([sys.executable, str(REPO_ROOT / "main.py"), "--phase", "3"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == main.EXIT_PREREQ  # no outputs/phase2.done in the real repo
