#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from config import PROCESSED_DIR, MANIFEST_DIR, EXPORTS_DIR
from logging_utils import configure_logging, log_event, stage_context

# Expected failures while reading a resolved manifest or copying its datasets.
# They are scoped to one collection: the collection is reported as failed and
# the export run continues (main() exits non-zero when nothing was exported).
EXPORT_ERRORS = (OSError, yaml.YAMLError, shutil.Error, ValueError)

class ParquetUnavailable(RuntimeError):
    """Raised when Parquet conversion is requested but pyarrow is missing."""


def write_parquet(source_csv: Path, target_parquet: Path) -> None:
    """Converts an already-exported CSV matrix into a Parquet file.

    CSV remains the default; Parquet is opt-in and needs pyarrow. A missing
    engine is a clear, actionable failure instead of a silent skip.
    """
    try:
        frame = pd.read_csv(source_csv)
        frame.to_parquet(target_parquet, index=False)
    except ImportError as e:
        raise ParquetUnavailable(
            "Parquet export requires the optional 'pyarrow' package. Install it with: pip install pyarrow"
        ) from e


def get_available_collections() -> list[str]:
    """Discovers all collections that have a resolved manifest blueprint."""
    if not MANIFEST_DIR.exists():
        return []
    return [f.stem.replace(".resolved", "") for f in MANIFEST_DIR.glob("*.resolved.yaml")]

def export_single_collection(collection_id: str, target_dir: Path, *, parquet: bool = False) -> bool:
    """Gathers all validated datasets belonging to a collection into a clean target structure.

    Every dataset keeps its CSV matrix; when ``parquet`` is set an additional
    ``data.parquet`` copy is written next to it.
    """
    manifest_path = MANIFEST_DIR / f"{collection_id}.resolved.yaml"
    if not manifest_path.exists():
        print(f"[SKIPPED] No resolved manifest found for '{collection_id}'. Run validation first.")
        log_event(
            "collection_skipped", level=30, collection_id=collection_id,
            operation="export", outcome="skipped", reason="no resolved manifest",
        )
        return False

    print(f"[INFO] Packaging Collection: {collection_id.upper()}")
    log_event("collection_export_started", collection_id=collection_id, operation="export", parquet=parquet)

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = yaml.safe_load(f)

        if not isinstance(manifest_data, dict):
            raise ValueError("manifest is not a YAML mapping")

        selected_datasets = manifest_data.get("selected_datasets", [])
        if not selected_datasets:
            print(f"    [WARNING] Manifest for '{collection_id}' contains zero validated datasets.")
            return False

        collection_export_path = target_dir / collection_id
        collection_export_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(manifest_path, collection_export_path / "manifest.yaml")

        copied_count = 0
        missing_datasets = []
        for ds in selected_datasets:
            if not isinstance(ds, dict) or "dataset_id" not in ds:
                raise ValueError(f"manifest lists an invalid dataset entry: {ds!r}")

            dataset_id = ds["dataset_id"]
            source_csv = PROCESSED_DIR / collection_id / dataset_id / "data.csv"

            if not source_csv.exists():
                print(f"    [ERROR] Missing processed CSV matrix for dataset: {dataset_id}")
                missing_datasets.append(dataset_id)
                continue

            ds_export_dir = collection_export_path / dataset_id
            ds_export_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_csv, ds_export_dir / "data.csv")
            if parquet:
                write_parquet(source_csv, ds_export_dir / "data.parquet")
            copied_count += 1

        print(f"    [SUCCESS] Bundled {copied_count} datasets into export path.")

        if missing_datasets:
            print(f"    [FAILED] Export of '{collection_id}' is incomplete, missing: "
                  f"{', '.join(str(d) for d in missing_datasets)}")
            log_event(
                "collection_export_failed", level=40, collection_id=collection_id,
                operation="export", outcome="failure", missing=missing_datasets,
            )
            return False

        log_event(
            "collection_exported", collection_id=collection_id, operation="export",
            outcome="success", datasets=copied_count, parquet=parquet,
        )
        return True

    except ParquetUnavailable as e:
        print(f"    [CRITICAL] {e}")
        log_event(
            "collection_export_failed", level=40, collection_id=collection_id,
            operation="export", outcome="failure", error=str(e),
        )
        return False

    except EXPORT_ERRORS as e:
        print(f"    [CRITICAL] Breakdown partitioning collection '{collection_id}': {type(e).__name__}: {e}")
        log_event(
            "collection_export_failed", level=40, collection_id=collection_id,
            operation="export", outcome="failure", error=f"{type(e).__name__}: {e}",
        )
        return False

def main(argv: Sequence[str] | None = None) -> int:
    """Exports validated collections. Returns an exit code (0 = success)."""
    configure_logging()
    with stage_context("export") as stage:
        parser = argparse.ArgumentParser(description="Export validated ML datasets for external environments.")
        parser.add_argument("--collection", type=str, default="all", help="Target collection ID or 'all'.")
        parser.add_argument("--format", type=str, choices=["folder", "zip"], default="zip", help="Output architecture.")
        parser.add_argument("--output", type=str, default=None, help="Custom destination directory path.")
        parser.add_argument(
            "--parquet", action="store_true", default=False,
            help="Also write data.parquet next to each exported data.csv (requires pyarrow).",
        )
        args = parser.parse_args(argv)

        export_base = Path(args.output).resolve() if args.output else EXPORTS_DIR
        export_base.mkdir(parents=True, exist_ok=True)
        available = get_available_collections()

        if not available:
            print(f"[ERROR] No resolved collection manifests found at: {MANIFEST_DIR}")
            stage.fail("no resolved manifests", reason=str(MANIFEST_DIR))
            return 1

        targets = available if args.collection.lower() == "all" else [args.collection.replace(".yaml", "").replace(".resolved", "")]
        if not all(t in available for t in targets):
            print(f"[ERROR] Invalid target. Available collections: {available}")
            stage.fail("invalid export target", requested=args.collection, available=available)
            return 1

        workspace_name = "export_bundle_all" if args.collection == "all" else f"export_bundle_{targets[0]}"
        temp_workspace = export_base / workspace_name

        if temp_workspace.exists():
            shutil.rmtree(temp_workspace)
        temp_workspace.mkdir(parents=True, exist_ok=True)

        successful_exports = sum(
            1 for col_id in targets
            if export_single_collection(col_id, temp_workspace, parquet=args.parquet)
        )

        if successful_exports == 0:
            print("\n[ERROR] Pipeline terminated. No collection components gathered.")
            shutil.rmtree(temp_workspace)
            stage.fail("no collection could be exported", targets=targets)
            return 1

        if args.format == "folder":
            for item in temp_workspace.iterdir():
                target_item = export_base / item.name
                if target_item.exists():
                    shutil.rmtree(target_item) if target_item.is_dir() else target_item.unlink()
                shutil.move(str(item), str(target_item))
            shutil.rmtree(temp_workspace)
            print(f"[FINISHED] Directory tree expanded at: {export_base}")
            log_event(
                "export_completed", operation="export", outcome="success",
                collections=successful_exports, format="folder", parquet=args.parquet,
            )

        elif args.format == "zip":
            zip_archive_path = export_base / f"{workspace_name.replace('export_bundle_', '')}.zip"
            with zipfile.ZipFile(zip_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_workspace):
                    for file in files:
                        file_path = Path(root) / file
                        zipf.write(file_path, file_path.relative_to(temp_workspace))
            shutil.rmtree(temp_workspace)
            print(f"[FINISHED] Archive generated at: {zip_archive_path}")
            log_event(
                "export_completed", operation="export", outcome="success",
                collections=successful_exports, format="zip", parquet=args.parquet,
            )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
