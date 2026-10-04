"""Offline demo that exercises the pipeline on deterministic synthetic data.

Run with::

    python -m pipeline.demo

No network access or API credentials are required.  The demo creates a
temporary project layout, generates two small synthetic datasets, catalogs
them, validates/cleans them, and exports a folder archive.  Everything is
written to a temporary directory that is left behind for inspection only when
``--keep`` is passed (default: cleaned up automatically).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from dataset_factories import classification_frame, regression_frame  # noqa: E402


def _create_temp_project() -> tuple[Path, Path]:
    """Creates a temporary project tree and returns (root, scripts_dir)."""
    tmp_root = Path(tempfile.mkdtemp(prefix="benchmark_demo_"))
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))

    for sub in (
        tmp_root / "data" / "raw" / "collections",
        tmp_root / "data" / "processed" / "collections",
        tmp_root / "catalog" / "datasets",
        tmp_root / "manifests",
        tmp_root / "collections",
        tmp_root / "exports",
    ):
        sub.mkdir(parents=True, exist_ok=True)
    return tmp_root, scripts


def _patch_config(root: Path) -> None:
    """Points the pipeline modules at the temporary project layout."""
    import config

    base = root / "data" / "raw" / "collections"
    for attr, value in (
        ("ROOT_DIR", root),
        ("DATA_DIR", root / "data"),
        ("RAW_DIR", base),
        ("PROCESSED_DIR", root / "data" / "processed" / "collections"),
        ("CATALOG_DIR", root / "catalog"),
        ("DATASETS_CATALOG_DIR", root / "catalog" / "datasets"),
        ("CATALOG_CSV_PATH", root / "catalog" / "catalog.csv"),
        ("MANIFEST_DIR", root / "manifests"),
        ("COLLECTIONS_DIR", root / "collections"),
        ("EXPORTS_DIR", root / "exports"),
    ):
        setattr(config, attr, value)

    import build_catalog
    import download_collection
    import export_collections
    import validate_datasets

    for module in (build_catalog, download_collection, export_collections, validate_datasets):
        for attr, value in (
            ("ROOT_DIR", root),
            ("DATA_DIR", root / "data"),
            ("RAW_DIR", base),
            ("PROCESSED_DIR", root / "data" / "processed" / "collections"),
            ("CATALOG_DIR", root / "catalog"),
            ("DATASETS_CATALOG_DIR", root / "catalog" / "datasets"),
            ("CATALOG_CSV_PATH", root / "catalog" / "catalog.csv"),
            ("MANIFEST_DIR", root / "manifests"),
            ("COLLECTIONS_DIR", root / "collections"),
            ("EXPORTS_DIR", root / "exports"),
        ):
            if hasattr(module, attr):
                setattr(module, attr, value)


def _seed_raw_collection(root: Path, collection_id: str) -> list[str]:
    """Writes two raw datasets for *collection_id* and returns their folder names."""
    import utils

    base = root / "data" / "raw" / "collections" / collection_id
    base.mkdir(parents=True, exist_ok=True)

    cls_name = "demo_classifier"
    cls_id = utils.dataset_storage_name(cls_name, 1001)
    cls_dir = base / cls_id
    cls_dir.mkdir(parents=True, exist_ok=True)
    classification_frame().to_csv(cls_dir / "data.csv", index=False)

    reg_name = "demo_regressor"
    reg_id = utils.dataset_storage_name(reg_name, 1002)
    reg_dir = base / reg_id
    reg_dir.mkdir(parents=True, exist_ok=True)
    regression_frame().to_csv(reg_dir / "data.csv", index=False)

    return [cls_id, reg_id]


def _mock_openml() -> None:
    """Stubs out OpenML so cataloging works without network access."""
    import openml

    class _FakeDataset:
        def __init__(self, did: int) -> None:
            self.id = did
            self.name = "demo"
            self.licence = "CC0"
            self.license = "CC0"
            self.citation = "Demo citation."
            self.paper_url = "https://example.org"

    def _fake_get_dataset(did: int, download_data: bool = False, **kwargs: object) -> _FakeDataset:
        return _FakeDataset(int(did))

    def _fake_list_datasets(**kwargs: object):
        import pandas as pd

        return pd.DataFrame()

    openml.datasets.get_dataset = _fake_get_dataset  # type: ignore[assignment]
    openml.datasets.list_datasets = _fake_list_datasets  # type: ignore[assignment]


def run_demo(keep: bool = False) -> int:
    """End-to-end demo: synthesize data, catalog, validate, export."""
    root, _ = _create_temp_project()
    _patch_config(root)
    _mock_openml()

    collection_id = "demo_collection"
    dataset_ids = _seed_raw_collection(root, collection_id)

    print(f"\n{'='*56}")
    print("  Benchmark Suite – Offline Demo")
    print(f"{'='*56}\n")

    print(f"[1/4] Seeded collection '{collection_id}' with {len(dataset_ids)} datasets:")
    for did in dataset_ids:
        print(f"      • {did}")

    print("\n[2/4] Cataloging …")
    import build_catalog

    rc = build_catalog.main([])
    if rc != 0:
        print(f"[FAIL] Catalog stage returned exit code {rc}")
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
        return rc

    import pandas as pd

    catalog = pd.read_csv(root / "catalog" / "catalog.csv")
    print(f"      Catalog entries: {len(catalog)}")
    print(f"      Columns: {list(catalog.columns)}")

    print("\n[3/4] Validating / cleaning …")
    import validate_datasets

    rc = validate_datasets.main([])
    if rc != 0:
        print(f"[FAIL] Validation stage returned exit code {rc}")
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
        return rc

    print("\n[4/4] Exporting …")
    import export_collections

    rc = export_collections.main(["--collection", collection_id, "--format", "folder"])
    if rc != 0:
        print(f"[FAIL] Export stage returned exit code {rc}")
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
        return rc

    export_root = root / "exports" / collection_id
    exported = list(export_root.rglob("data.csv"))
    manifests = list((root / "manifests").glob("*.resolved.yaml"))

    print(f"\n{'='*56}")
    print("  Demo complete — summary")
    print(f"{'='*56}")
    print(f"  Temporary root:  {root}")
    print(f"  Catalog entries: {len(catalog)}")
    print(f"  Cleaned CSVs:    {len(exported)}")
    print(f"  Manifests:       {len(manifests)}")
    for m in manifests:
        print(f"    • {m.name}")
    print()

    if not keep:
        shutil.rmtree(root, ignore_errors=True)
        print("[INFO] Temporary files cleaned up. Re-run with --keep to inspect.")
    else:
        print(f"[INFO] Temporary files kept at: {root}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline.demo",
        description="Run an offline demo of the Benchmark Suite pipeline.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the temporary working directory after the demo (default: clean up).",
    )
    args = parser.parse_args(argv)
    return run_demo(keep=args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
