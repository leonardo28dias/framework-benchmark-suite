# User Guide

Benchmark Suite is designed to be simple in day-to-day use:

1. **Define** a collection
2. **Download** the datasets that match it
3. **Export** the collection to your working folder

That is the normal path. Additional scripts exist for manual inspection, debugging, and double verification.

## 1. What this framework is for

This project helps you build reproducible benchmark collections from tabular datasets without hand-picking files every time. It enforces strict data quality standards (minimum sample counts, acceptable missing rates, no leakage IDs) before data reaches your modeling code.

The pipeline flow is:

```
define collection → download → catalog → validate → export
```

For most users, only the first and last steps are required — the cataloging and validation steps run automatically as part of the full pipeline.

## 2. Quick Start

### Install

```bash
git clone https://github.com/your-username/dataset-benchmark-framework.git
cd dataset-benchmark-framework

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install with all dev dependencies
pip install -e ".[dev]"
```

### Run the offline demo (no API credentials needed)

```bash
python -m pipeline.demo
```

This generates deterministic synthetic datasets, catalogs them, validates/cleans them, and exports a folder archive — all in a temporary directory that is cleaned up automatically.

### Run the full pipeline (requires OpenML API access)

```bash
benchmark-framework run \
  --filter "task == classification AND task_subtype == binary" \
  --id my_collection \
  --name "My Collection" \
  --limit 10
```

## 3. CLI Commands

| Command | Purpose |
|---------|---------|
| `benchmark-framework define` | Create a collection blueprint from filter expressions |
| `benchmark-framework download` | Download datasets from OpenML for a collection |
| `benchmark-framework catalog` | Build the dataset catalog from raw CSVs |
| `benchmark-framework validate` | Validate and clean cataloged datasets |
| `benchmark-framework export` | Export validated collections (ZIP or folder, optional Parquet) |
| `benchmark-framework run` | Run the full pipeline: define → download → catalog → validate → export |
| `benchmark-framework demo` | Run the offline demo |

Each command supports `--help` for detailed usage.

### CLI examples

```bash
# Define a binary classification collection
benchmark-framework define \
  --filter "task == classification AND task_subtype == binary" \
  --filter "missing_rate <= 0.05" \
  --id binary_clean \
  --name "Binary Clean Classification"

# Download matching datasets
benchmark-framework download --collection binary_clean --limit 10

# Full pipeline with Parquet export
benchmark-framework run \
  --filter "task == regression" \
  --id reg_collection \
  --limit 5 \
  --parquet
```

## 4. Manual Stage Execution

You can also run stage scripts directly. This is useful for debugging and manual verification.

### Define a collection

```bash
python scripts/define_collection.py \
  --filter "(task == classification AND task_subtype == binary) AND (missing_rate <= 0.05) AND (n_features <= 50)" \
  --filter "temporal == False" \
  --name "Binary Clean Classification" \
  --id "binary_clean"
```

This creates a YAML blueprint in `collections/`.

### Download datasets

```bash
python scripts/download_collection.py --collection binary_clean --limit 10
```

This fills `data/raw/collections/binary_clean/` with matching datasets.

### Build the catalog manually

```bash
python scripts/build_catalog.py
```

This scans raw data and creates `catalog/catalog.csv` with metadata for each dataset.

### Validate datasets

```bash
python scripts/validate_datasets.py
```

This runs the cleaning gate, produces `data/processed/` matrices, and generates `.resolved.yaml` manifests.

### Export

```bash
python scripts/export_collections.py \
  --collection binary_clean \
  --format folder \
  --parquet
```

## 5. Filter Expression Language

The framework supports a custom query language for defining collection criteria:

| Operator | Meaning |
|----------|---------|
| `==` | Equals |
| `!=` | Not equals |
| `>=` | Greater than or equal |
| `>` | Greater than |
| `<=` | Less than or equal |
| `<` | Less than |
| `AND` | Logical conjunction |
| `OR` | Logical disjunction |

### Available fields

| Field | Type | Description |
|-------|------|-------------|
| `task` | string | `classification` or `regression` |
| `task_subtype` | string | `binary`, `multiclass`, `continuous` |
| `n_instances` | integer | Number of rows |
| `n_features` | integer | Number of feature columns |
| `n_classes` | integer | Number of target classes |
| `missing_rate` | float | Fraction of missing values (0–1) |
| `imbalance_ratio` | float | Max/min class size ratio |
| `p_to_n_ratio` | float | Features-to-instances ratio |
| `temporal` | boolean | Whether data has a time component |
| `grouped` | boolean | Whether data is grouped |
| `academic_usage` | boolean | Whether dataset allows academic usage |
| `iid_assumed` | boolean | Whether i.i.d. assumption holds |

### Examples

```bash
# Binary classification with strict quality constraints
benchmark-framework define \
  --filter "(task == classification AND task_subtype == binary) AND (missing_rate <= 0.05) AND (n_features <= 50)" \
  --id "binary_clean"

# Small regression datasets
benchmark-framework define \
  --filter "task == regression" \
  --filter "n_instances <= 2000" \
  --filter "n_features <= 30" \
  --id "small_regression"

# Large imbalanced classification
benchmark-framework define \
  --filter "task == classification AND imbalance_ratio >= 3 AND n_instances >= 1000" \
  --id "large_imbalanced"
```

## 6. Directory Layout

```
.
├── scripts/              # Pipeline stage scripts
├── pipeline/             # CLI + offline demo package
├── tests/                # Test suite (186 tests, all offline)
├── notebooks/            # Benchmarking Jupyter notebook
├── stubs/openml/         # mypy type stubs for OpenML
├── collections/          # Collection blueprints (input)
├── data/raw/             # Unaltered CSV extractions (input/output)
├── data/processed/       # Cleaned, validated matrices (output)
├── catalog/              # Generated metadata indices (output)
├── manifests/            # Frozen resolved manifests (output)
├── exports/              # Exported collection archives (output)
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Development dependencies (legacy)
└── pyproject.toml        # Project metadata, packaging, tool config
```

## 7. Benchmarking Notebook

The `notebooks/benchmark_collection.ipynb` notebook runs scikit-learn models across all validated collections.

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Launch the notebook
jupyter notebook notebooks/benchmark_collection.ipynb
```

Set `COLLECTIONS` in the first code cell:
- `None` — auto-discover all collections in `data/processed/collections/`
- `['smoke_test']` — benchmark a specific collection
- `['smoke_test', 'binary_clean']` — benchmark multiple collections

**Algorithms benchmarked:**
- **Classification:** LogisticRegression, RandomForest, GradientBoosting (5-fold stratified CV)
- **Regression:** Ridge, RandomForest, GradientBoosting (5-fold CV)

Results are saved to `exports/all_collections_benchmark_results.csv`.

## 8. Testing

The test suite runs entirely offline — no network access or API credentials required. OpenML is fully mocked in all tests.

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_demo.py -v

# Run with coverage
pytest --cov

# Via Makefile
make test
make ci  # runs lint + typecheck + test
```

## 9. Development

```bash
# Install with dev dependencies (includes ruff, mypy, pytest, jupyter)
pip install -e ".[dev]"

# Lint
make lint       # ruff check .

# Type check
make typecheck  # mypy .

# Run the demo
make demo       # python -m pipeline.demo --keep
```

## 10. Troubleshooting

**No processed data found:** Run `benchmark-framework validate` first to produce cleaned datasets in `data/processed/`.

**OpenML API errors:** Ensure you have network access and a valid OpenML API key. Set it via `OPENML_API_KEY` environment variable or configure in `~/.openml/config`.

**Catalog incremental mode skips datasets:** This is expected behavior. Re-run `build_catalog.main()` after modifying raw data to force reprocessing.
