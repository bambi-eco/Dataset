# Downloading the dataset

Selecting flights and fetching them from Zenodo. For what the versions are, see
[dataset-versions.md](dataset-versions.md).

## Installation

```bash
pip install -r requirements.txt
```

The scripts are tested with Python 3.10+.

## Automatic Download

Selectively download flight ZIPs from the BAMBI dataset hosted on Zenodo. Uses the zenodo_upload_summary_*.json files to resolve which depositions contain which flights, so you can grab exactly what you need without fetching entire multi-GB depositions. Use `filter_flights.py` to get a list of flight IDs for the data that you are looking for (e.g. filtered for species).


```bash
# List all available flights
python download_from_zenodo.py --list

# Download flights 0, 5, and 12
python download_from_zenodo.py -f 0 5 12

# Download flights 10 through 25
python download_from_zenodo.py --range 10 25

# Download all flights from parts 1 and 3
python download_from_zenodo.py --parts 1 3

# Download all flights of a dataset split (train / val / test)
python download_from_zenodo.py --split train
python download_from_zenodo.py --split val
python download_from_zenodo.py --split test

# Download everything, extract, and clean up ZIPs
python download_from_zenodo.py --unzip

# Preview what a full download would do
python download_from_zenodo.py --dry-run

# Download a different dataset version instead of the pre-processed videos (compatible with all other flags like -f, --range, --split, etc.)
python download_from_zenodo.py --version raw
python download_from_zenodo.py --version matched
python download_from_zenodo.py --version orthographic

# Recordings plus the OWL-transferred RGB annotations, side by side in one folder
python download_from_zenodo.py --version owl-transferred -f 119 --unzip
```

> **Note:** `--version` selects which dataset version to download and defaults to `base` (the pre-processed videos). The available versions are `base`, `raw`, `matched`, `orthographic`, and `owl-transferred`; each is described by its own `flight_metadata/zenodo_upload_summary_*.json`. A summary file from a custom location can be supplied with `-s <path>`, which overrides `--version`.

> **Note:** `owl-transferred` is a **layer on top of `base`**, not a standalone release: it ships only annotation files. Selecting it downloads the `base` recordings *and* the transferred annotations into the same output directory, so a single command gives a complete, usable flight. Its archives are named `owl_labels_<id>.zip` so they cannot collide with the `flight_<id>.zip` of the base layer. Flights that the base release ships without thermal labels have nothing to transfer and are reported as a coverage gap at the end of the run.

> **Note:** `--split` reads flight IDs from `flight_metadata/splits.json`. A custom path can be supplied with `--splits-file <path>`. The flag is silently ignored when `-f`, `--range`, or `--parts` is also specified.

## Flight filter

Filter flights based on metadata JSON files by species, occlusion, sex, age, weather, date range, and drone name.

All filters combine with **AND** logic between each other. Within list filters (`--species`, `--drone`, `--sex`, `--age`) values combine with **OR** logic. Weather flags combine with **AND** (all specified conditions must be present).


```bash
# Multiple species (OR: flights containing either)
python filter_flights.py --species "Roe deer" "Homo sapiens" "Q122069"

# Only flights with occluded frames
python filter_flights.py --occlusion true

# Flights with male or female subjects
python filter_flights.py --sex male female

# Flights with juvenile or adult animals
python filter_flights.py --age juvenile adult

# Flights that are both cloudy AND windy
python filter_flights.py --weather cloudy windy

# Flights in October 2024
python filter_flights.py --min-date 2024-10-01 --max-date 2024-10-31

# Visible roe deer in sunny weather during October 2024
python filter_flights.py ./metadata \
    --species "Roe deer" \
    --occlusion false \
    --weather sunny \
    --min-date 2024-10-01 \
    --max-date 2024-10-31 \
    -v
```
