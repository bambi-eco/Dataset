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

# Annotations, poses and metadata only, never the video: a few MB per flight
python download_from_zenodo.py --annotations-only -f 146
python download_from_zenodo.py --version environment-all --annotations-only

# Exactly the files you name out of a flight's archive, in any version
python download_from_zenodo.py -f 146 --include '*_gt.txt' '*_matched_poses.json'
python download_from_zenodo.py --version raw -f 1 --include '*.srt' air_data.csv
python download_from_zenodo.py --version matched -f 1 --include 'labels/*'
python download_from_zenodo.py --version raw -f 1 --exclude '*_V_*.mp4'

# What does a flight's archive hold? Sizes, and what a selection would pick
python download_from_zenodo.py --version orthographic -f 1 --list-files
python download_from_zenodo.py --version orthographic -f 1 --list-files --annotations-only
```

> **Note:** `--version` selects which dataset version to download and defaults to `base` (the pre-processed videos). The available versions are `base`, `raw`, `matched`, `orthographic`, and `owl-transferred`; each is described by its own `flight_metadata/zenodo_upload_summary_*.json`. A summary file from a custom location can be supplied with `-s <path>`, which overrides `--version`.

> **Note:** `owl-transferred` is a **layer on top of `base`**, not a standalone release: it ships only annotation files. Selecting it downloads the `base` recordings *and* the transferred annotations into the same output directory, so a single command gives a complete, usable flight. Its archives are named `owl_labels_<id>.zip` so they cannot collide with the `flight_<id>.zip` of the base layer. Flights that the base release ships without thermal labels have nothing to transfer and are reported as a coverage gap at the end of the run.

> **Note:** `--annotations-only` (`-a`) leaves the video and the mask images on Zenodo and fetches everything else in the archive: `<id>_gt.txt`, `<id>_matched_poses.json`, `<id>_metadata.json`, `<id>_correction.json`, `<id>_track_mapping.json`, and whatever an annotation layer adds. It works because a zip archive keeps its table of contents at the end and every member at a known offset, and Zenodo honours HTTP range requests, so the script opens the archive in place and reads only the members it wants: three or four small requests and about 3 MB for a base flight whose archive is 1.5 GB. A study that joins the animal boxes with the environment layers over all 301 flights needs about 1.3 GB this way, against roughly 400 GB with the videos. The files land extracted, and a flight counts as present once its `<id>_gt.txt`, poses and metadata are there, so a rerun skips it. `--annotations-only` implies `--unzip`.

> **Note:** `--include` / `--exclude` (`-i` / `-x`) generalise `--annotations-only` to any selection, in every version: each takes glob patterns that match either the path inside the archive (`labels/*`, `thermal/*_utm.txt`) or the bare file name (`*_gt.txt`), case-insensitively, with `<id>` standing for the flight id. They combine with each other and with `--annotations-only` (which then drops imagery from whatever is included), use the same range requests, and imply `--unzip`. Paths inside the archive are kept, so a selection unpacks exactly as the full archive would, just with fewer files; a file already on disk with the right size is skipped, so you can add files to an earlier download with a second call. `--list-files` prints each requested flight's archive contents (the thousands of frames of `matched` and `orthographic` collapsed per directory), marks what the current selection would fetch, and downloads nothing.

> **Note:** The `raw`, `matched` and `orthographic` archives do not carry the flight id in their file names (`air_data.csv`, `manifest.json`, `rgb/`, `thermal/`). Unpacked, one flight goes straight into the output directory as before; with more than one flight each goes into `<output>/<id>/`, so two flights never overwrite each other's files. This applies to `--unzip` and to every selective mode alike.

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
