# User Guide

Benchmark Suite is meant to stay simple in day-to-day use:

1. define a collection,
2. download the datasets that match it,
3. export the collection to the folder or notebook environment you want to work in.

That is the normal path. The other scripts are there for manual checks, debugging, and double verification when needed.

## 1. What this framework is for

This project helps you build reproducible benchmark collections from tabular datasets without hand-picking files every time.

The usual flow is:

- define a collection in YAML
- download the datasets that match it
- export the collection to your working folder

For most users, that is all they need.

## 2. Typical workflow

### Step 1 — Define a collection

Create a collection with the criteria you want.

A stronger example looks like this:

```bash
python scripts/define_collection.py \
  --filter "(task == classification AND task_subtype == binary) AND (missing_rate <= 0.05) AND (n_features <= 50)" \
  --filter "temporal == False" \
  --name "Binary Clean Classification" \
  --id "binary_clean"

This creates the collection definition file in collections/.

A collection file describes what should be included. It does not download anything yet.
Step 2 — Download the collection

Once the collection is defined, download the matching datasets:

python scripts/download_collection.py --collection binary_clean

This fills the raw data folder with the datasets that match the collection rules.

The structural rules that are non-negotiable in the pipeline are enforced by the framework itself during download and validation. You do not need to repeat those constraints in the collection definition.
Step 3 — Export the collection

After the datasets are downloaded and validated, export them to the folder you want to use.

If you want the default export location, just run:

python scripts/export_collections.py

That exports all available validated collections into the default exports/ folder.

To export one specific collection into a folder of your choice:

python scripts/export_collections.py \
  --collection binary_clean \
  --format folder \
  --output /path/to/your/notebook/folder

To export one collection as a zip file:

python scripts/export_collections.py \
  --collection binary_clean \
  --format zip \
  --output /path/to/your/notebook/folder

If you do not pass --output, the export goes into the project’s default exports/ folder.
3. The three main steps

If you only want to use the framework in the normal way, these are the only steps that matter:

    define the collection

    download the collection

    export it to the folder you want to work in

That is the standard workflow.
4. Manual checks

The framework also lets you verify parts of the pipeline by hand. These are optional, but useful when you want to double-check a step.
Build the catalog manually

If you want to inspect the raw datasets and build the metadata catalog yourself:

python scripts/build_catalog.py

This scans the raw data and creates the catalog files.

Use this when you want to confirm that the metadata was detected correctly.
Validate datasets manually

If you want to verify the datasets before using them:

python scripts/validate_datasets.py

This checks dataset structure, target consistency, feature counts, missingness, and other validation rules.

Use this when you want to double-check the cleaning and validation stage.
5. When to use the manual steps

The manual steps are not required for normal use.

They are useful when:

    you want to inspect the catalog before moving on

    you want to validate the datasets a second time

    you are working with manually added datasets

    you want to debug a dataset that failed later in the pipeline

    you want to confirm the framework output before exporting it

If everything is working normally, you can skip them.
6. Working with manual datasets

The framework is not limited to downloaded datasets. You can also add datasets manually, then run the catalog and validation steps on them.

A normal manual flow looks like this:

    place the dataset files in the expected raw data folder

    build the catalog

    validate the datasets

    define or update the collection

    download or materialize the collection

    export it to the folder you want to use

That makes it possible to use the framework even when a dataset did not come from the normal OpenML automated download path.
7. Where files go

The project uses a simple layout:

    collections/ for collection definitions

    data/raw/ for downloaded or manually placed raw datasets

    catalog/ for generated metadata

    data/processed/ for cleaned and validated datasets

    manifests/ for resolved collection outputs

    exports/ for exported collections

This keeps the pipeline easy to trace.
8. Recommended way to use the project

For normal use, stick to this order:

define collection
→ download collection
→ export collection to your chosen folder

Only use the catalog and validation scripts directly when you want to inspect, verify, or troubleshoot a step.
9. Common examples
A binary classification collection with stricter filtering

python scripts/define_collection.py \
  --filter "(task == classification AND task_subtype == binary) AND (missing_rate <= 0.05) AND (academic_usage == True)" \
  --filter "grouped == False"

A smaller collection that is easier to test

python scripts/define_collection.py \
  --filter "task == regression" \
  --filter "n_instances <= 2000" \
  --filter "n_features <= 30"

Build and check everything manually

python scripts/build_catalog.py
python scripts/validate_datasets.py

10. Export behavior

By default, export_collections.py exports all validated collections.

If you run:

python scripts/export_collections.py

the script writes the export bundle into exports/.

If you want a different destination, use --output.

If you want folders instead of a zip archive, use --format folder.

If you want a single archive, use --format zip.
11. Notes

The framework is designed to stay practical.

For normal use, only the three main steps matter. The other scripts are there to give you more control when you need it.

If you are preparing a notebook or benchmark environment, the cleanest path is:

    define the collection

    download the collection

    export it to the folder you want to work in

That is the workflow this project is built around.

