import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-clone",
        action="store_true",
        default=False,
        help="Run the slow GPU voice-cloning test (downloads the model on first run).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: heavy GPU test, opt in with --run-clone")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-clone"):
        return
    skip = pytest.mark.skip(reason="needs --run-clone (slow GPU + model download)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
