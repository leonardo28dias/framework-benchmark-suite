#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
import yaml

# Configuration of structured directories
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_BASE = ROOT / "data" / "processed" / "collections"
MANIFEST_DIR = ROOT / "manifests"
DEFAULT_EXPORT_BASE = ROOT / "exports"


def get_available_collections() -> list[str]:
    """Discovers all collections that have a resolved manifest blueprint."""
    if not MANIFEST_DIR.exists():
        return []
    return [f.stem.replace(".resolved", "") for f in MANIFEST_DIR.glob("*.resolved.yaml")]


def export_single_collection(collection_id: str, target_dir: Path) -> bool:
    """
    Gathers all validated datasets belonging to a collection using its 
    resolved manifest contract, copying them into a clean target structure.
    """
    manifest_path = MANIFEST_DIR / f"{collection_id}.resolved.yaml"
    if not manifest_path.exists():
        print(f"⚠️  [Skipped] No resolved manifest found for collection '{collection_id}'. Run validate_datasets.py first.")
        return False

    print(f"📦 Packaging Collection: {collection_id.upper()}")

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = yaml.safe_load(f)
        
        selected_datasets = manifest_data.get("selected_datasets", [])
        if not selected_datasets:
            print(f"    ⚠️  [Warning] Manifest for '{collection_id}' contains zero validated datasets.")
            return False

        # Create localized workspace inside the export destination
        collection_export_path = target_dir / collection_id
        collection_export_path.mkdir(parents=True, exist_ok=True)

        # Copy the master manifest blueprint along with the data for notebook context
        shutil.copy(manifest_path, collection_export_path / "manifest.yaml")

        copied_count = 0
        for ds in selected_datasets:
            dataset_id = ds.get("dataset_id")
            source_csv = PROCESSED_BASE / collection_id / dataset_id / "data.csv"

            if not source_csv.exists():
                print(f"    ❌ Missing processed CSV matrix for dataset: {dataset_id}")
                continue

            # Isolate each dataset into its own clear folder within the export block
            ds_export_dir = collection_export_path / dataset_id
            ds_export_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(source_csv, ds_export_dir / "data.csv")
            copied_count += 1
            print(f"    -> Aggregated dataset asset: '{dataset_id}'")

        print(f"    ✅ Successfully bundled {copied_count} datasets into export path.")
        return True

    except Exception as e:
        print(f"    💥 Critical breakdown partitioning collection '{collection_id}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export validated machine learning dataset collections for external notebook environments."
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="all", 
        help="Target collection ID to bundle. Pass 'all' to package every available collection profile."
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["folder", "zip"], 
        default="zip", 
        help="Output architecture strategy. 'folder' creates structured trees; 'zip' generates a highly portable archive."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom destination directory path. Supports relative paths or absolute systems. Defaults to internal 'exports/' folder."
    )
    args = parser.parse_args()

    # Determine dynamic export destination
    if args.output:
        export_base = Path(args.output).resolve()
    else:
        export_base = DEFAULT_EXPORT_BASE

    export_base.mkdir(parents=True, exist_ok=True)
    available = get_available_collections()

    if not available:
        print(f"[Error] No resolved collection manifests found at: {MANIFEST_DIR}")
        print("Please ensure you have run your automated ingestion pipeline or 'validate_datasets.py' first.")
        sys.exit(1)

    # Resolve targeting list
    if args.collection.lower() == "all":
        targets = available
        print(f"Found {len(targets)} active validated collection manifests ready for deployment.")
    else:
        clean_id = args.collection.replace(".yaml", "").replace(".resolved", "")
        if clean_id not in available:
            print(f"[Error] Requested collection '{clean_id}' is not currently available or validated.")
            print(f"Available collection options: {available}")
            sys.exit(1)
        targets = [clean_id]

    # Establish an isolated workspace directory inside our export base
    workspace_name = "export_bundle_all" if args.collection == "all" else f"export_bundle_{targets[0]}"
    temp_workspace = export_base / workspace_name
    
    if temp_workspace.exists():
        shutil.rmtree(temp_workspace)
    temp_workspace.mkdir(parents=True, exist_ok=True)

    # Process all selected targets into the staging area
    successful_exports = 0
    for col_id in targets:
        if export_single_collection(col_id, temp_workspace):
            successful_exports += 1

    if successful_exports == 0:
        print("\n❌ Pipeline terminated. No collection components were successfully gathered.")
        if temp_workspace.exists():
            shutil.rmtree(temp_workspace)
        sys.exit(1)

    # Route formatting layout configurations
    if args.format == "folder":
        print(f"\nMoving generated collection folders directly into target destination...")
        # FIX: Move contents of temp_workspace into export_base to eliminate double-nesting issues
        for item in temp_workspace.iterdir():
            target_item = export_base / item.name
            if target_item.exists():
                if target_item.is_dir():
                    shutil.rmtree(target_item)
                else:
                    target_item.unlink()
            shutil.move(str(item), str(target_item))
            
        shutil.rmtree(temp_workspace)
        print(f"🏁 [Finished] Clean collection directory tree expanded at: {export_base}")
    
    elif args.format == "zip":
        zip_archive_path = export_base / f"{workspace_name.replace('export_bundle_', '')}.zip"
        print(f"\n🤐 Compressing export assets into standalone distribution archive...")
        
        with zipfile.ZipFile(zip_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_workspace):
                for file in files:
                    file_path = Path(root) / file
                    archive_name = file_path.relative_to(temp_workspace)
                    zipf.write(file_path, archive_name)
                    
        shutil.rmtree(temp_workspace)
        print(f"\n🏁 [Finished] Standalone portability archive generated at: {zip_archive_path}")


if __name__ == "__main__":
    main()