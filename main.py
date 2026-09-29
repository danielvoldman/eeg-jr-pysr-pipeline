"""Pipeline entry point: `python main.py --phase N [--pilot]` (§16.2, §18, §18.1).

The thread variables must be set before NumPy, MNE or PySR are imported, so
only the standard library, PyYAML and src.config are imported above the
bootstrap call. Julia's thread count is fixed at startup (§16.2).
"""
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import yaml  # noqa: F401  (allowed before the env bootstrap)

from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parent

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_PREREQ = 2

LOGGER_NAME = "pipeline"


def _bootstrap_env(config_path=None):
    """Set the thread variables from config.yml (§16.2). Overrides inherited values."""
    cfg = load_config(config_path or REPO_ROOT / "config.yml")
    for name, value in cfg["compute"]["env_vars_before_numpy_import"].items():
        os.environ[name] = str(value)


_bootstrap_env()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="EEG Jansen-Rit PySR pipeline")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--pilot", action="store_true",
                        help="reduced §17 settings; output under results/pilot/")
    return parser.parse_args(argv)


def flag_path(cfg, root, phase, pilot):
    key = "pilot_phase_flag_pattern" if pilot else "phase_flag_pattern"  # IMP-002
    return Path(root) / cfg["paths"][key].format(n=phase)


def ensure_dirs(cfg, root, pilot):
    paths = cfg["paths"]
    names = ["cache_dir", "outputs_dir", "results_dir", "logs_dir"]
    if pilot:
        names.append("pilot_results_dir")
    for name in names:
        (Path(root) / paths[name]).mkdir(parents=True, exist_ok=True)


def setup_logging(cfg, root, phase, pilot):
    suffix = "_pilot" if pilot else ""
    log_file = Path(root) / cfg["paths"]["logs_dir"] / f"phase{phase}{suffix}.log"
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_file, encoding="utf-8"),
                    logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def check_prerequisite(cfg, root, phase, pilot):
    """Return None if phase may start, else the missing flag path (§18.1)."""
    if phase == 1:
        return None
    prev = flag_path(cfg, root, phase - 1, pilot)
    return None if prev.is_file() else prev


def _make_stub(phase):
    def runner(cfg, root, pilot):
        logging.getLogger(LOGGER_NAME).error(
            "phase %d: not implemented", phase)
        return False
    return runner


# Real runners replace these stubs as their scripts are built. A runner returns
# True only on success; main() writes the .done flag on that basis alone.
PHASE_RUNNERS = {n: _make_stub(n) for n in (1, 2, 3, 4)}


def run_phase(cfg, root, phase, pilot):
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("phase %d start (pilot=%s)", phase, pilot)
    if PHASE_RUNNERS[phase](cfg, root, pilot) is not True:
        logger.error("phase %d did not succeed; no flag written", phase)
        return EXIT_FAILED
    flag = flag_path(cfg, root, phase, pilot)
    flag.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    flag.write_text(f"{stamp}\n", encoding="utf-8")
    logger.info("phase %d done; flag %s", phase, flag)
    return EXIT_OK


def main(argv=None, root=None):
    args = parse_args(argv)
    root = Path(root) if root is not None else REPO_ROOT
    cfg = load_config(root / "config.yml")
    ensure_dirs(cfg, root, args.pilot)
    logger = setup_logging(cfg, root, args.phase, args.pilot)
    missing = check_prerequisite(cfg, root, args.phase, args.pilot)
    if missing is not None:
        logger.error("phase %d refused: required flag %s not found",
                     args.phase, missing)
        return EXIT_PREREQ
    return run_phase(cfg, root, args.phase, args.pilot)


if __name__ == "__main__":
    sys.exit(main())
