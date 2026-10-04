"""Console entry point for the Benchmark Suite pipeline.

Provides a lightweight argparse-based CLI that delegates to the existing
flat scripts under ``scripts/``::

    benchmark-framework define   --filter "task == classification" --id my_col
    benchmark-framework download --collection my_col --limit 5
    benchmark-framework catalog
    benchmark-framework validate
    benchmark-framework export   --collection my_col --format zip
    benchmark-framework run      --filter "task == classification" --id my_col
    benchmark-framework demo

Each command forwards to the corresponding stage's ``main()`` function so
there is no duplicated business logic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from logging_utils import configure_logging  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-framework",
        description="Declarative, validation-first ETL pipeline for ML benchmark datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_define = sub.add_parser("define", help="Define a collection blueprint from filter expressions.")
    p_define.add_argument("--filter", action="append", default=[], help='Criteria expression, e.g. "task == classification"')
    p_define.add_argument("--id", default=None, help="Explicit collection ID.")
    p_define.add_argument("--name", default=None, help="Explicit collection name.")
    p_define.add_argument("--output-dir", type=Path, default=None, help="Output directory for the blueprint.")

    p_download = sub.add_parser("download", help="Download datasets from OpenML for a collection.")
    p_download.add_argument("--collection", type=str, default=None, help="Target collection name.")
    p_download.add_argument("--limit", type=int, default=30, help="Max datasets to pull per collection.")

    sub.add_parser("catalog", help="Build the dataset catalog from downloaded raw CSVs.")

    p_validate = sub.add_parser("validate", help="Validate and clean cataloged datasets.")
    p_validate.add_argument("--argv", nargs="*", default=[], help="Extra arguments passed through to the validation stage.")

    p_export = sub.add_parser("export", help="Export validated collections.")
    p_export.add_argument("--collection", type=str, default="all", help="Target collection or 'all'.")
    p_export.add_argument("--format", type=str, choices=["folder", "zip"], default="zip", help="Output format.")
    p_export.add_argument("--output", type=str, default=None, help="Custom destination directory.")
    p_export.add_argument("--parquet", action="store_true", help="Also write Parquet files.")

    p_run = sub.add_parser("run", help="Run the full pipeline (define → download → catalog → validate → export).")
    p_run.add_argument("--filter", action="append", default=[], help='Criteria expression')
    p_run.add_argument("--id", default=None, help="Explicit collection ID.")
    p_run.add_argument("--name", default=None, help="Explicit collection name.")
    p_run.add_argument("--output-dir", type=Path, default=None, help="Output directory for the blueprint.")
    p_run.add_argument("--collection", type=str, default=None, help="Collection to download.")
    p_run.add_argument("--limit", type=int, default=30, help="Download limit.")
    p_run.add_argument("--export-format", "--format", dest="format", type=str, choices=["folder", "zip"], default="zip")
    p_run.add_argument("--output", type=str, default=None, help="Custom export destination directory.")
    p_run.add_argument("--parquet", action="store_true", help="Also export Parquet.")
    p_run.add_argument("--validate-argv", nargs="*", default=[], help="Extra validate args.")

    p_demo = sub.add_parser("demo", help="Run the offline demo (no network/API needed).")
    p_demo.add_argument("--keep", action="store_true", help="Keep temp files after demo (default: clean up).")

    return parser


def _run_define(args: argparse.Namespace) -> int:
    import define_collection

    argv: list[str] = []
    if args.filter:
        for f in args.filter:
            argv.extend(["--filter", f])
    if args.id:
        argv.extend(["--id", args.id])
    if args.name:
        argv.extend(["--name", args.name])
    if args.output_dir:
        argv.extend(["--output-dir", str(args.output_dir)])
    return define_collection.main(argv)


def _run_download(args: argparse.Namespace) -> int:
    import download_collection

    argv: list[str] = []
    if args.collection:
        argv.extend(["--collection", args.collection])
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    return download_collection.main(argv)


def _run_catalog(args: argparse.Namespace) -> int:
    import build_catalog

    return build_catalog.main([])


def _run_validate(args: argparse.Namespace) -> int:
    import validate_datasets

    return validate_datasets.main(args.argv or [])


def _run_export(args: argparse.Namespace) -> int:
    import export_collections

    argv: list[str] = []
    if args.collection:
        argv.extend(["--collection", args.collection])
    if args.format:
        argv.extend(["--format", args.format])
    if args.output:
        argv.extend(["--output", args.output])
    if args.parquet:
        argv.append("--parquet")
    return export_collections.main(argv)


def _run_full(args: argparse.Namespace) -> int:
    from run_pipeline import main as pipeline_main

    pipeline_argv: list[str] = ["full"]
    if args.filter:
        for f in args.filter:
            pipeline_argv.extend(["--filter", f])
    if args.id:
        pipeline_argv.extend(["--id", args.id])
    if args.name:
        pipeline_argv.extend(["--name", args.name])
    if args.output_dir:
        pipeline_argv.extend(["--output-dir", str(args.output_dir)])
    if args.collection:
        pipeline_argv.extend(["--collection", args.collection])
    if args.limit:
        pipeline_argv.extend(["--limit", str(args.limit)])
    pipeline_argv.extend(["--export-format", args.format])
    if args.output:
        pipeline_argv.extend(["--output", args.output])
    if args.parquet:
        pipeline_argv.append("--parquet")
    if args.validate_argv:
        pipeline_argv.extend(["--validate-argv", *args.validate_argv])
    return pipeline_main(pipeline_argv)


def _run_demo(args: argparse.Namespace) -> int:
    from pipeline.demo import run_demo

    return run_demo(keep=getattr(args, "keep", False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging()

    dispatch = {
        "define": _run_define,
        "download": _run_download,
        "catalog": _run_catalog,
        "validate": _run_validate,
        "export": _run_export,
        "run": _run_full,
        "demo": _run_demo,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 2

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
