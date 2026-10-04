# Architecture Document

## Overview

The Benchmark Suite is a validation-first ETL pipeline designed to ingest, audit, and clean public data repositories into standardized, legally clear, and structurally resilient modeling assets for machine learning benchmarking.

Instead of downloading datasets and cleaning them manually during the modeling phase, the pipeline follows a fixed sequence to guarantee structural integrity before an engineer ever touches the data.

## Pipeline Sequence

```
Definition → Ingestion → Cataloging → Sanitization → Export
```

Each stage is a standalone script that can be run independently, combined with other stages, or orchestrated together via `scripts/run_pipeline.py` or the `benchmark-framework` CLI.

## Component Breakdown

### 1. Definition (`define_collection.py`)

A custom recursive-descent AST parser that translates command-line filter expressions (e.g., `task == classification AND n_instances >= 500`) into structured YAML condition mappings. The generated blueprint is validated against a Pydantic schema (`schemas.py`) before it is ever written to disk.

**Inputs:** CLI `--filter` expressions, `--id`, `--name`  
**Output:** `collections/<id>.yaml` blueprint files

### 2. Ingestion (`download_collection.py`)

The ingestion layer. Converts collection blueprint bounds into OpenML API query parameters, streams tabular data, and enforces open-source licensing and citation requirements. Each downloaded dataset is stored with a collision-free name (`<sanitized_name>__<openml_id>`) to disambiguate datasets that share sanitized names.

**Dependencies:** OpenML API, `pandas`  
**Output:** `data/raw/collections/<collection_id>/<dataset_id>/data.csv`

### 3. Cataloging (`build_catalog.py`)

The metadata engine. Performs structural x-ray of downloaded CSVs to infer task types, feature types, data quality metrics (missing rates, constant columns, high-cardinality columns), and academic metadata (citation, license). Uses incremental checksum-based caching via `xxhash` to skip reprocessing unchanged datasets.

**Key feature:** Incremental — re-runs are fast because only changed datasets are reprocessed  
**Output:** `catalog/catalog.csv` (master ledger), `catalog/datasets/<id>.yaml` (per-dataset metadata)

### 4. Sanitization (`validate_datasets.py`)

The cleaning gate. Safely drops constant features, removes high-cardinality leakage IDs, drops rows with missing targets, removes duplicates, enforces statistical minimums (instance count, feature count, minority class size), and materializes frozen `.resolved.yaml` manifests that pin the exact validated dataset composition.

**Validation rules:** 11 quality gates including missing target handling, duplicate removal, constant column detection, high-cardinality leakage elimination, and instance/feature bounds  
**Output:** `data/processed/collections/<collection_id>/<dataset_id>/data.csv`, `manifests/<collection_id>.resolved.yaml`

### 5. Export (`export_collections.py`)

The deployment step. Gathers all validated assets referenced in resolved manifests and builds structured workspaces for end users. Supports ZIP archives, folder exports, and optional Parquet conversion via `pyarrow`.

**Output:** `exports/<collection_id>/` containing manifest and `data.csv` (and `data.parquet` if `--parquet`)

## Data Flow

```
collections/*.yaml          →  Declarative specifications (input)
     ↓
data/raw/collections/       →  Unaltered tabular CSV extractions
     ↓
catalog/                    →  Generated metadata indices and dataset properties
     ↓
data/processed/             →  Pristine matrices, cleared of constants, noise, and singletons
     ↓
manifests/*.resolved.yaml   →  Frozen execution ledgers tracking validated inclusions
     ↓
exports/                    →  Formatted distribution packages (ZIP/folder)
```

## Orchestration

### `run_pipeline.py` (orchestrator)

Controls execution order across stages. Supports running individual stages (`define`, `download`, `catalog`, `validate`, `export`) or the full pipeline (`full`). Each stage reports structured JSON logs with a shared `run_id` for traceability.

### CLI: `benchmark-framework`

Console entry point installed via `pyproject.toml`. Maps each subcommand directly to the corresponding stage's `main()` function — no business logic duplication.

## Cross-Cutting Concerns

### Structured Logging (`logging_utils.py`)

- One compact JSON object per line on stderr
- Each record carries `run_id`, `stage`, `collection_id`, `event`, `outcome`
- Stage lifecycle events: `stage_started`, `stage_completed`, `stage_failed`
- Failures include `failure_reason` and exception details via `StageOutcome`

### Pydantic Validation (`schemas.py`)

- `CollectionBlueprint` model validates YAML blueprints before use
- Field-level validation for operators, numeric fields, and boolean fields
- `extra="forbid"` prevents silent acceptance of typos in blueprint keys

### Checksum-Based Incrementality (`utils.py`)

- `file_checksum()` uses `xxhash.xxh64` for fast, deterministic file hashing
- `dataset_storage_name()` combines sanitized name + OpenML ID to prevent folder collisions
- Catalog stage reuses per-dataset YAML metadata when raw CSV checksum is unchanged

## External Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | DataFrame operations and CSV I/O |
| `numpy` | Numerical computing |
| `pyyaml` | YAML blueprint parsing/serialization |
| `openml` | Dataset discovery and download |
| `pydantic` | Blueprint validation |
| `xxhash` | Fast file checksums for incremental cataloging |
| `duckdb` | Analytics queries on catalog data |
| `pyarrow` | Parquet export (optional) |
| `scikit-learn` | Modeling algorithms in the benchmark notebook |

## Directory Layout

```
.
├── scripts/              # Pipeline stage scripts (flat modules)
│   ├── config.py         # Central path and threshold definitions
│   ├── define_collection.py
│   ├── download_collection.py
│   ├── build_catalog.py
│   ├── validate_datasets.py
│   ├── export_collections.py
│   ├── run_pipeline.py   # Orchestrator
│   ├── logging_utils.py  # JSON structured logging
│   ├── schemas.py        # Pydantic models
│   ├── utils.py          # Hash and naming utilities
│   └── duckdb_demo.py    # DuckDB analytics demo
├── pipeline/             # High-level package (CLI + demo)
│   ├── __init__.py
│   ├── cli.py            # Console entry point
│   └── demo.py           # Offline demo
├── tests/                # Test suite (186 tests)
├── notebooks/            # Jupyter benchmarking notebook
├── stubs/openml/         # Local mypy stubs for untyped openml package
├── collections/          # Collection blueprints
├── data/                 # Raw and processed datasets
├── catalog/              # Generated catalog metadata
├── manifests/            # Resolved collection manifests
└── exports/              # Exported collection archives
```

## Design Principles

1. **Import isolation:** No script executes work on import. All logic is guarded behind `main()` functions.
2. **No hidden side effects:** Each stage can be imported, tested, and executed independently.
3. **Validation-first:** Invalid configurations fail before any expensive work starts.
4. **Deterministic:** Checksums and seeded synthetic data ensure reproducible runs.
5. **Offline-capable:** The demo and all tests work without network access.
```
