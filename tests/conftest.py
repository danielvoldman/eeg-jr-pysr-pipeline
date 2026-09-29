import logging
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_root(tmp_path):
    """A scratch repo root holding a copy of config.yml, so tests never touch real folders."""
    shutil.copy(REPO_ROOT / "config.yml", tmp_path / "config.yml")
    return tmp_path


@pytest.fixture(autouse=True)
def _close_pipeline_log_handlers():
    """main.main() attaches FileHandlers to the 'pipeline' logger; close and remove them."""
    yield
    logger = logging.getLogger("pipeline")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
