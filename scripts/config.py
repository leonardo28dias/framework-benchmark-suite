"""
Central configuration module for the Benchmark Suite framework.
Defines absolute paths and operational thresholds to eliminate "magic" numbers.
"""
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data Paths
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "collections"
PROCESSED_DIR = DATA_DIR / "processed" / "collections"

# Catalog & Manifest Paths
CATALOG_DIR = ROOT_DIR / "catalog"
DATASETS_CATALOG_DIR = CATALOG_DIR / "datasets"
CATALOG_CSV_PATH = CATALOG_DIR / "catalog.csv"
MANIFEST_DIR = ROOT_DIR / "manifests"
COLLECTIONS_DIR = ROOT_DIR / "collections"
EXPORTS_DIR = ROOT_DIR / "exports"

# Dataset Structural Constraints
MIN_INSTANCES = 500
MAX_INSTANCES = 10000
MIN_FEATURES = 5
MAX_FEATURES = 100

# Data Quality Thresholds
MAX_MISSING_RATE = 0.50
MIN_MINORITY_CLASS_COUNT = 5
MAX_CLASSIFICATION_CLASSES = 20
GLOBAL_MIN_CLASS_SIZE = 2
