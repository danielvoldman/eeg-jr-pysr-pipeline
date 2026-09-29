import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_root(tmp_path):
    """A scratch repo root holding a copy of config.yml, so tests never touch real folders."""
    shutil.copy(REPO_ROOT / "config.yml", tmp_path / "config.yml")
    return tmp_path
