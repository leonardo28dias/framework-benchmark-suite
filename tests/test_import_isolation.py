"""
Tests proving that importing a pipeline stage does not trigger any work.

Each module must be importable without touching the filesystem or network.
Side-effecting logic belongs behind main() guards or explicit function calls.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_importing_download_collection_does_not_call_downstream_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing download_collection must not trigger build_catalog or validate_datasets."""
    monkeypatch.setattr(sys, "modules", {})

    build_catalog_stub: Any = types.ModuleType("build_catalog")
    build_catalog_stub.main = MagicMock()
    validate_stub: Any = types.ModuleType("validate_datasets")
    validate_stub.main = MagicMock()

    monkeypatch.setitem(sys.modules, "build_catalog", build_catalog_stub)
    monkeypatch.setitem(sys.modules, "validate_datasets", validate_stub)

    importlib.invalidate_caches()
    importlib.import_module("download_collection")

    build_catalog_stub.main.assert_not_called()
    validate_stub.main.assert_not_called()


@pytest.mark.parametrize(
    "module_name",
    ["build_catalog", "export_collections", "validate_datasets"],
    ids=["build_catalog", "export_collections", "validate_datasets"],
)
def test_importing_stage_module_does_not_run_main(module_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing any stage module must not call its own main()."""
    monkeypatch.setattr(sys, "modules", {})
    importlib.invalidate_caches()

    module = importlib.import_module(module_name)

    assert module is sys.modules[module_name]


def test_importing_define_collection_does_not_write_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Importing define_collection must not create any collection YAML files."""
    collections_dir = tmp_path / "collections"
    collections_dir.mkdir()
    monkeypatch.setattr("define_collection.COLLECTIONS_DIR", collections_dir)

    importlib.invalidate_caches()
    importlib.import_module("define_collection")

    yaml_files = list(collections_dir.glob("*.yaml"))
    assert len(yaml_files) == 0
