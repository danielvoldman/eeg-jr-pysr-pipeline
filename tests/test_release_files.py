"""Release files (PLAN Stage L, §18.4, IMP-099): requirements.txt, README.md, LICENSE."""
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()


def pins():
    out = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        assert m, f"not an exact pin: {line!r}"
        out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def test_every_requirement_is_an_exact_pin():
    assert len(pins()) >= 40


def test_named_pins_present_and_consistent_with_config():
    p = pins()
    for name in ("pysr", "filterpy", "joblib", "specparam", "numpy", "scipy", "numba", "mne", "sympy", "pyyaml", "psutil"):
        assert name in p, name
    assert p["pysr"] == CFG["pysr"]["version"]["pysr"]
    assert p["juliacall"] == CFG["pysr"]["version"]["juliacall"]
    assert p["juliapkg"] == CFG["pysr"]["version"]["juliapkg"]
    assert (p["joblib"], p["specparam"], p["filterpy"]) == ("1.6.0", "2.0.0rc7", "1.4.5")


def test_pins_equal_the_installed_versions():
    bad = []
    for name, ver in pins().items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            bad.append((name, ver, "not installed"))
            continue
        if installed != ver:
            bad.append((name, ver, installed))
    assert not bad


def test_requirements_header_records_the_julia_pins():
    head = "\n".join(l for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if l.startswith("#"))
    assert CFG["pysr"]["version"]["julia"] in head and CFG["pysr"]["version"]["symbolic_regression_jl"] in head


def test_pip_check_is_clean():
    r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_license_is_mit_with_the_owner_and_year():
    t = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert t.startswith("MIT License")
    assert "Copyright (c) 2026 Daniel Voldman" in t
    assert "Permission is hereby granted, free of charge" in t


def test_readme_banner_and_required_sections():
    t = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Status: this repository ends at a preregistered hard stop" in t.split("\n")[2]
    assert "docs/G0_HARD_STOP_REPORT.md" in t
    for heading in ("## What it does", "## Install", "## How to run", "## What the hard stop means", "## Where results are",
                    "## Wall-clock cost"):
        assert heading in t, heading
    for phase in range(1, 5):
        assert f"main.py --phase {phase}" in t
    assert "--pilot" in t
    assert "ds003775 has its own terms" in t
    assert "not measured" in t  # the §16.3 estimates are labelled unmeasured


def test_readme_commands_exist_in_main():
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    for flag in ("--phase", "--pilot", "--force", "--dry-run", "--make-split"):
        assert flag in main
    fig = (ROOT / "src" / "figures.py").read_text(encoding="utf-8")
    assert "--figure" in fig and "--all" in fig


def test_gitignore_still_excludes_generated_folders():
    g = (ROOT / ".gitignore").read_text(encoding="utf-8").split("\n")
    for line in ("data/", "cache/", "outputs/", "results/", "logs/"):
        assert line in g
