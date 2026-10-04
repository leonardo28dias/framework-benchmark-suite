# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Editable install support via `pip install -e ".[dev]"` with console entry point `benchmark-framework`
- `Makefile` with `install`, `dev-install`, `test`, `lint`, `typecheck`, `ci`, `demo`, `pipeline`, and `clean` targets
- Offline demo module (`pipeline/demo.py`) that runs the full pipeline on deterministic synthetic data without API credentials
- `pipeline/cli.py` — argparse-based CLI with `define`, `download`, `catalog`, `validate`, `export`, `run`, and `demo` subcommands
- Local OpenML type stubs (`stubs/openml/`) enabling `mypy .` to pass without broad `Any` workarounds
- Benchmarking notebook (`notebooks/benchmark_collection.ipynb`) running scikit-learn models across all collections
- Test suite for CLI (`tests/test_cli.py`) and demo (`tests/test_demo.py`)
- `pipeline` package with `__init__.py`
- README sections: Quick Start, Demo, CLI, Testing, Makefile Commands

### Changed
- **Logging fix:** Set `logger.propagate = False` in `logging_utils.py` to eliminate duplicate log lines from root logger propagation
- **Orchestrator fix:** Removed nested `stage_context` wrappers in `run_pipeline.py` that caused duplicate `stage_started`/`stage_completed` lifecycle events
- **Datetime fix:** Replaced deprecated `datetime.datetime.utcnow()` with `datetime.datetime.now(datetime.timezone.utc)` in `validate_datasets.py`
- **CLI fix:** `--validate-argv` is now properly forwarded through `benchmark-framework run` and `benchmark-framework validate`
- **Redundant call fix:** Removed duplicate `configure_logging()` call in `run_pipeline.py` catalog stage
- Updated `pyproject.toml` with `[build-system]`, `[project]`, `[project.scripts]`, and `[project.optional-dependencies]`
- Updated `.gitignore` to exclude `stubs/` and notebook output artifacts
- Ruff now excludes `notebooks/` from linting (notebook cell imports must precede path manipulation)

### Removed
- No breaking changes to existing pipeline behavior

## [0.1.0] - Initial Public Release

### Added
- Collection definition DSL with custom AST parser (`define_collection.py`)
- OpenML ingestion with license/citation enforcement (`download_collection.py`)
- Catalog generation with structural diagnostics (`build_catalog.py`)
- Dataset validation and cleaning with 11 quality gates (`validate_datasets.py`)
- Collection export to ZIP or folder format (`export_collections.py`)
- Pydantic blueprint validation (`schemas.py`)
- Structured JSON logging with run/stage context (`logging_utils.py`)
- xxhash-based incremental cataloging (`utils.py`)
- DuckDB analytics demo (`duckdb_demo.py`)
