# Architecture Document

## 1. Overview

Benchmark Suite is a small framework for building reproducible tabular benchmark collections.

The goal is straightforward: define what you want, download matching datasets, validate them, and export the final collection into the environment you will actually use.

The project is organized as a pipeline, not as a single script. Each step has a clear job and produces files for the next step.

## 2. Core idea

The framework is built around a validation-first workflow.

Instead of downloading datasets and cleaning them later by hand, the project moves through a fixed sequence:

```text
collection definition
→ download
→ catalog build
→ dataset validation
→ manifest materialization
→ export

That makes the pipeline easier to inspect, easier to repeat, and easier to debug.

3. Main components
define_collection.py

Creates a collection YAML from filter rules.

Its job is to describe which datasets should be considered. It does not download anything.

download_collection.py

Reads the collection definition and downloads the matching datasets.

It is the ingestion layer of the project.

build_catalog.py

Scans the downloaded raw datasets and builds metadata files.

It records structural information such as task type, number of instances, number of features, and other catalog fields.

validate_collections.py

Checks whether candidate datasets match the collection rules.

This is the selection gate.

validate_datasets.py

Cleans and validates the downloaded datasets.

This step removes bad rows or columns, checks target integrity, applies size constraints, and saves the processed datasets.

materialize_collection.py

Freezes the final collection into a resolved manifest.

The manifest is the final record of what was accepted and why.

export_collections.py

Packages validated collections into a notebook-friendly or working-directory-friendly output.

This is the handoff step for downstream work.

4. Data flow

The folder structure follows the same logic as the scripts.

collections/
    collection definitions

data/raw/
    downloaded or manually placed raw datasets

catalog/
    dataset metadata and catalog outputs

data/processed/
    cleaned and validated datasets

manifests/
    resolved collection manifests

exports/
    packaged collections for notebook or external use

The data moves forward in one direction.

Raw data should not be edited directly once it has been downloaded. Any correction or cleanup should happen through the validation step.

5. Collection lifecycle

A collection goes through a few stages.

Stage 1 — Definition

A collection starts as a YAML file in collections/.

This file describes the selection logic.

Stage 2 — Download

The framework uses the collection rules to fetch matching datasets.

The output goes into data/raw/.

Stage 3 — Catalog

The framework scans the downloaded files and writes metadata.

This makes the datasets searchable and traceable.

Stage 4 — Validation

The framework checks the datasets against its rules.

Only datasets that pass move forward into data/processed/.

Stage 5 — Materialization

The final accepted dataset list is frozen into a manifest.

This locks the collection state so it can be reused later.

Stage 6 — Export

The collection is packaged for use in a notebook or external working folder.

6. Validation philosophy

The framework is intentionally strict.

A dataset should not pass just because it exists. It should also be structurally usable.

The validation logic focuses on things like:

task type
target presence
target stability
number of rows
number of features
missingness
constant columns
high-cardinality leakage columns
class imbalance problems

This keeps the benchmark collection stable enough for repeatable experiments.

7. Manual vs automatic use

The framework supports two ways of working.

Automatic use

This is the normal path.

You define the collection, download it, validate it, and export it.

Manual use

You can also run individual stages by hand.

That is useful when:

you want to inspect the raw catalog
you want to verify validation output
you are adding manual datasets
you want to debug one broken step
you want to check results before exporting

The manual path is there for control, not because it is required.

8. Export behavior

The export script is designed to bundle validated collections into a clean workspace.

By default it exports all available validated collections into the exports/ folder.

It can also:

export one specific collection
export as a folder tree
export as a zip archive
write to a custom output path

The output is meant to be easy to move into a notebook, local experiment folder, or evaluation environment.

9. Design choices

A few design decisions shape the project.

YAML first

Collections and metadata are stored in YAML because it is readable, version-friendly, and easy to edit by hand.

Validation before modeling

The framework rejects bad structural inputs early, before they become training-time problems.

Small modules

Each script has one main responsibility. That makes the pipeline easier to debug and replace later.

Reproducible outputs

Resolved manifests and processed datasets are frozen outputs. They should not change unless the pipeline is run again.

10. What this project is not

Benchmark Suite is not trying to be a full AutoML platform.

It is the layer before that.

It helps with:

choosing datasets
checking them
cleaning them
organizing them
exporting them

It does not try to replace the actual benchmark runner or model training environment.

11. Future direction

The next natural step is integration with an AutoML evaluation system such as AMLB.

That would let the framework move from curated dataset collections to full benchmark execution without changing the collection logic.

12. Summary

Benchmark Suite is built to make benchmark data easier to manage.

The framework starts with a collection definition, checks the data, freezes the result, and exports it in a form that is easy to reuse later.

The main idea is simple: keep the data pipeline explicit, reproducible, and easy to inspect.