# Benchmark Suite

A declarative, validation-first pipeline designed to ingest, audit, and clean public data repositories into standardized, legally clear, and highly resilient modeling assets for automated machine learning benchmarking.

This framework replaces haphazard manual data gathering with an automated refinery pipeline inspired by the rigorous data-integrity standards of the OpenML CC-18 benchmark.

---

## Core Objectives & Integrity Gates

In production and benchmarking environments, machine learning pipelines frequently crash due to unexpected schema shifts, missing target spaces, or illegal data licenses. This repository addresses these challenges by enforcing a strict definition of a "Good Dataset" through multi-layer verification gates before any modeling code runs.

### 1. Provenance & Academic Sourcing (OpenML Exclusive)

To guarantee legal compliance and scientific reproducibility, this framework sources its data exclusively via the **OpenML API**.

- Strict Licensing: The ingestion engine drops datasets without an explicit open-source license.
- Academic Verification: Datasets must include a valid citation URL or paper reference.

### 2. Operational Sizing Limits

- Instances: 500–10,000 samples
- Features: 5–100 non-target attributes

### 3. Statistical Integrity (No-Noise Policy)

- Row deduplication
- Drop columns with >50% missing values
- Remove zero-variance features
- Remove identifier / leakage columns

### 4. Structural Machine Learning Safety

- Enforces classification/regression only
- Removes rows with missing targets
- Rejects constant targets
- Classification max: 20 classes
- Minimum 5 samples per class
- No singleton datasets

---

## End-to-End Pipeline Architecture

[1. DEFINE] → [2. DOWNLOAD] → [3. CATALOG] → [4. SANITIZE]

define_collection.py → download_collection.py → build_catalog.py → validate_datasets.py

(AST YAML Blueprint) → (In-Memory Stream) → (Metadata Ledger) → (Clean Data & Manifest)

### Modular Component Breakdown
| Operational Phase | Module                   | Primary Responsibility                                     | Input / Output                                 |
| ----------------- | ------------------------ | ---------------------------------------------------------- | ---------------------------------------------- |
| Declaration       | `define_collection.py`   | Converts filter queries into AST-based YAML specifications | CLI filters → `collections/<id>.yaml`          |
| Ingestion         | `download_collection.py` | Downloads datasets, validates licensing, streams data      | YAML → `data/raw/<collection>/data.csv`        |
| Cataloging        | `build_catalog.py`       | Builds dataset metadata and structural diagnostics         | CSV → `catalog/catalog.csv` + YAMLs            |
| Sanitization      | `validate_datasets.py`   | Cleans datasets and produces final ML-ready outputs        | Catalog → `data/processed/` + `.resolved.yaml` |

### Quick Start

Pipeline is fully chained:

download_collection.py → build_catalog.py → validate_datasets.py
#### Define a Collection
python scripts/define_collection.py \
  --filter "task == classification" \
  --filter "n_instances >= 1000" \
  --filter "missing_rate <= 0.05" \
  --id binary_clean_core \
  --name "Binary Clean Core Classification"
#### Run Pipeline
python scripts/download_collection.py --collection binary_clean_core
#### Manual Overrides
python scripts/build_catalog.py
python scripts/validate_datasets.py

### Benchmark Results
The following tables show the baseline evaluation performance achieved out-of-the-box by the framework after streaming, validating, and cleaning core benchmark suites. Performance metrics are recorded using an un-tuned Random Forest baseline (100 estimators, 80/20 train/test partition).
#### Core Clean Binary Classification Suite
| Dataset ID          | Instances | Features | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC AUC |
| ------------------- | --------: | -------: | -------: | --------------: | -----------: | -------: | ------: |
| pc2                 |     1,406 |       36 |   0.9787 |          0.4911 |       0.4982 |   0.4946 |  0.6917 |
| mc1                 |     2,016 |       38 |   0.9703 |          0.4888 |       0.4962 |   0.4925 |  0.8440 |
| spambase            |     4,210 |       57 |   0.9454 |          0.9439 |       0.9420 |   0.9429 |  0.9895 |
| spambase_reproduced |     4,210 |       57 |   0.9454 |          0.9439 |       0.9420 |   0.9429 |  0.9895 |
| pc1                 |       954 |       21 |   0.9215 |          0.6947 |       0.6287 |   0.6530 |  0.8666 |
| pc3                 |     1,439 |       37 |   0.8924 |          0.7018 |       0.5567 |   0.5737 |  0.7550 |
| pc4                 |     1,344 |       37 |   0.8885 |          0.7698 |       0.6565 |   0.6912 |  0.9399 |
| wine                |     2,231 |       11 |   0.7964 |          0.7949 |       0.7934 |   0.7940 |  0.8685 |
| kc1                 |     1,212 |       21 |   0.6996 |          0.5814 |       0.5651 |   0.5683 |  0.6149 |
| eye_movements       |     7,608 |       20 |   0.6176 |          0.6177 |       0.6176 |   0.6176 |  0.6748 |

#### Core Imbalanced Multiclass Suite
| Dataset ID             | Instances | Features | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC AUC |
| ---------------------- | --------: | -------: | -------: | --------------: | -----------: | -------: | ------: |
| page_blocks            |     5,406 |       10 |   0.9713 |          0.8244 |       0.8765 |   0.8403 |  0.9955 |
| baseball               |     1,340 |       16 |   0.9440 |          0.7951 |       0.6104 |   0.6752 |  0.9548 |
| analcatdata_halloffame |     1,340 |       16 |   0.9440 |          0.7951 |       0.6104 |   0.6752 |  0.9548 |
| wine_quality           |     5,318 |       11 |   0.5733 |          0.4606 |       0.2576 |   0.2693 |  0.7392 |
| wine_quality_white     |     3,961 |       11 |   0.5561 |          0.3697 |       0.2623 |   0.2793 |  0.7138 |

#### Core Regression Suite
 | Dataset ID                | Instances | Features | R2 Score |   RMSE |    MAE |
| ------------------------- | --------: | -------: | -------: | -----: | -----: |
| parkinsonspeechdataset... |     1,039 |       28 |   1.0000 | 0.0000 | 0.0000 |
| rf1                       |     9,125 |       71 |   0.9995 | 0.4943 | 0.2379 |
| blocks                    |     5,416 |       14 |   0.8989 | 0.1246 | 0.0248 |
| authorship                |     8,417 |       30 |   0.8352 | 0.3190 | 0.2180 |
| turkiyestudentevaluation  |     3,977 |       32 |   0.7747 | 0.5901 | 0.3172 |
| waveformdatabasegenerator |     5,000 |       21 |   0.6386 | 0.4830 | 0.3348 |
| emotions                  |     5,937 |       71 |   0.6031 | 0.2994 | 0.1935 |
| wq                        |     1,060 |       29 |   0.0946 | 1.3876 | 0.9236 |

⚠️ Residual Operational Risks: > While the validation engine enforces structural parity and extracts verifiable academic URLs, it cannot detect semantic data risks such as mathematical feature proxies (data leakage) or severe class imbalance anomalies. For example, the perfect R2 scores in the Regression Suite indicate the model is exploiting a direct target proxy left in the feature space. Review individual dataset profiles in catalog/ to determine if custom loss weighting, resampling steps, or domain-specific manual exclusions are required.

Repository Structure


AutoML/
├── catalog/                   # Generated database indices and diagnostics
│   ├── catalog.csv            # Central flat matrix ledger for fast querying
│   └── datasets/              # Detailed structural/academic YAML profiles
├── collections/               # Input declarative specification blueprints
│   └── binary_clean_core.yaml
├── data/
│   ├── processed/             # Pristine matrices (cleared of constants, noise, singletons)
│   └── raw/                   # Unaltered tabular CSV extractions grouped by collection
├── manifests/                 # Frozen execution ledgers and deployment matrices
│   └── binary_clean_core.resolved.yaml
└── scripts/                   # Core automated execution modules
    ├── build_catalog.py
    ├── define_collection.py
    ├── download_collection.py
    ├── export_collections.py
    ├── validate_collections.py
    └── validate_datasets.py

Future Work

    AMLB Integration: Standardize data processing configurations to directly export task files compatible with the automated machine learning benchmark framework.

    OpenML Task Generation: Programmatically output official execution task configurations to enable automatic tracking on public evaluation platforms.

    AutoML Evaluation Backend: Connect the processed data tracks directly to automated training loops for frameworks like AutoGluon, FLAML, and H2O.

    Automated Dashboard Reporting: Generate deployment-ready diagnostic summaries, comparing training speeds, column-type variances, and score matrices across different engine releases.

License

This project is distributed under the MIT License. See the LICENSE file for details.



