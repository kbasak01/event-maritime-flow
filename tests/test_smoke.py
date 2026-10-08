"""Phase 0 smoke test: the package and every planned subpackage import; markers are registered."""

import importlib
import importlib.metadata

import pytest

import evflow

SUBPACKAGES = [
    "io", "events", "flow", "sim", "models", "losses", "train", "eval", "teacher", "maritime",
    "export",
]


def test_version_matches_installed_metadata():
    assert evflow.__version__ == importlib.metadata.version("evflow")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    mod = importlib.import_module(f"evflow.{name}")
    assert mod.__doc__, f"evflow.{name} needs a one-line docstring"


def test_markers_registered(pytestconfig):
    markers = " ".join(pytestconfig.getini("markers"))
    for m in ("gpu:", "dataset:", "slow:"):
        assert m in markers
