#!/usr/bin/env python3
"""
Pipeline orchestrator for the Benchmark Suite.

Explicitly controls execution order so each stage can be imported independently
without triggering work, run on its own from the command line, or run as part
of the full pipeline here.

    define_collection
        -> download_collection
        -> build_catalog
        -> validate_datasets
        -> export

Usage:
    python scripts/run_pipeline.py define --filter "task == classification" --id my_col --name "My Collection"
    python scripts/run_pipeline.py download --collection my_col --limit 5
    python scripts/run_pipeline.py catalog
    python scripts/run_pipeline.py validate
    python scripts/run_pipeline.py export --collection all --format zip
    python scripts/run_pipeline.py full --define-args '...' --download-args '...' --export-format zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from config import COLLECTIONS_DIR
from logging_utils import configure_logging, log_event


def run_define(args: argparse.Namespace) -> int:
    import define_collection
    argv = []
    if args.filter:
        for f in args.filter:
            argv.extend(["--filter", f])
    if args.id:
        argv.extend(["--id", args.id])
    if args.name:
        argv.extend(["--name", args.name])
    output_dir = args.output_dir or COLLECTIONS_DIR
    argv.extend(["--output-dir", str(output_dir)])
    return define_collection.main(argv)


def run_download(args: argparse.Namespace) -> int:
    import download_collection
    argv = []
    if args.collection:
        argv.extend(["--collection", args.collection])
    if args.limit:
        argv.extend(["--limit", str(args.limit)])
    return download_collection.main(argv)


def run_catalog(args: argparse.Namespace) -> int:
    import build_catalog
    return build_catalog.main([])


def run_validate(args: argparse.Namespace) -> int:
    import validate_datasets
    argv = getattr(args, "validate_argv", None) or getattr(args, "argv", None) or []
    return validate_datasets.main(argv)


def run_export(args: argparse.Namespace) -> int:
    import export_collections
    argv = []
    if args.collection:
        argv.extend(["--collection", args.collection])
    if args.format:
        argv.extend(["--format", args.format])
    if args.output:
        argv.extend(["--output", args.output])
    if args.parquet:
        argv.append("--parquet")
    return export_collections.main(argv)


def run_full(args: argparse.Namespace) -> int:
    """Runs define -> download -> catalog -> validate -> export in sequence."""
    run_id = configure_logging()

    steps = [
        ("define", lambda: run_define(args)),
        ("download", lambda: run_download(args)),
        ("catalog", lambda: run_catalog(args)),
        ("validate", lambda: run_validate(args)),
        ("export", lambda: run_export(args)),
    ]

    results: dict[str, int] = {}
    for name, step_fn in steps:
        code = step_fn()
        results[name] = code
        if code != 0:
            log_event("pipeline_aborted", stage=name, run_id=run_id, reason=f"{name} exited with code {code}")
            return code

    log_event("pipeline_completed", run_id=run_id, stages=results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Suite pipeline orchestrator.")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_define = sub.add_parser("define", help="Define a collection blueprint.")
    p_define.add_argument("--filter", action="append", default=[], help='Criteria expression, e.g. "task == classification"')
    p_define.add_argument("--id", default=None, help="Explicit collection ID.")
    p_define.add_argument("--name", default=None, help="Explicit collection name.")
    p_define.add_argument("--output-dir", type=Path, default=None, help="Output directory for the blueprint.")
    p_define.set_defaults(func=run_define)

    p_download = sub.add_parser("download", help="Download datasets from OpenML.")
    p_download.add_argument("--collection", type=str, default=None, help="Target collection name.")
    p_download.add_argument("--limit", type=int, default=30, help="Max datasets to pull.")
    p_download.set_defaults(func=run_download)

    p_catalog = sub.add_parser("catalog", help="Build the dataset catalog.")
    p_catalog.set_defaults(func=run_catalog)

    p_validate = sub.add_parser("validate", help="Validate and clean datasets.")
    p_validate.add_argument("--argv", nargs="*", default=[], help="Extra arguments for validate_datasets (pass-through).")
    p_validate.set_defaults(func=run_validate)

    p_export = sub.add_parser("export", help="Export validated collections.")
    p_export.add_argument("--collection", type=str, default="all", help="Target collection or 'all'.")
    p_export.add_argument("--format", type=str, choices=["folder", "zip"], default="zip", help="Output format.")
    p_export.add_argument("--output", type=str, default=None, help="Custom destination directory.")
    p_export.add_argument("--parquet", action="store_true", help="Also write Parquet files.")
    p_export.set_defaults(func=run_export)

    p_full = sub.add_parser("full", help="Run the complete pipeline with shared run ID.")
    p_full.add_argument("--filter", action="append", default=[], help='Criteria expression')
    p_full.add_argument("--id", default=None, help="Explicit collection ID.")
    p_full.add_argument("--name", default=None, help="Explicit collection name.")
    p_full.add_argument("--output-dir", type=Path, default=None, help="Output directory for the blueprint.")
    p_full.add_argument("--collection", type=str, default=None, help="Collection to download.")
    p_full.add_argument("--limit", type=int, default=30, help="Download limit.")
    p_full.add_argument("--export-format", "--format", dest="format", type=str, choices=["folder", "zip"], default="zip")
    p_full.add_argument("--output", type=str, default=None, help="Custom export destination directory.")
    p_full.add_argument("--parquet", action="store_true", help="Also export Parquet.")
    p_full.add_argument("--validate-argv", nargs="*", default=[], help="Extra validate args.")
    p_full.set_defaults(func=run_full)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
