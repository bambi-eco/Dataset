# 🦌 BAMBI Dataset

**Multimodal Nadir UAV-Recordings of Forest Wildlife**

The BAMBI dataset is a large-scale airborne multispectral wildlife dataset comprising 389 paired RGB and thermal aerial video sequences recorded across diverse forest and forest-adjacent habitats in Austria. Each frame is geo-referenced with precise global coordinates (longitude, latitude, and altitude), enabling learning and evaluation in both image space and geographic space.

This repository provides sample scripts for downloading, processing, and visualizing the dataset.

> **Citation:** If you use this dataset in your research, please cite our paper (see [Citation](#citation)).

---

## Dataset Overview

| Property | Value |
|---|---|
| Video sequences | 389 paired RGB + thermal |
| Total flight time | ~45 hours |
| Annotated tracks | 5,100 |
| Key frames | 92,701 |
| Interpolated bounding boxes | 1,218,903 |
| Species classes | 12 |
| Recording period | January 2023 – November 2024 |
| Location | Austria (Tyrol, Upper Austria, Lower Austria, Salzburg, Styria, Carinthia) |

### Species

The dataset covers the following 12 classes:

| # | Species | Common Name | Wikidata                                                                                                  | Tracks | Key Frames |
|---|---|---|-----------------------------------------------------------------------------------------------------------|---:|---:|
| 1 | *Sus scrofa* | Wild boar | [Q58697](https://www.wikidata.org/wiki/Q58697)                                                            | 1,770 | 26,132 |
| 2 | *Cervus elaphus* | Red deer | [Q1219579](https://www.wikidata.org/wiki/Q1219579)                                                        | 1,613 | 26,674 |
| 3 | *Capreolus capreolus* | Roe deer | [Q122069](https://www.wikidata.org/wiki/Q122069)                                                          | 682 | 9,404 |
| 4 | *Dama dama* | Fallow deer | [Q20908334](https://www.wikidata.org/wiki/Q20908334)                                                      | 297 | 20,003 |
| 5 | *Capra ibex* | Alpine ibex | [Q168327](https://www.wikidata.org/wiki/Q168327)                                                          | 100 | 3,005 |
| 6 | *Rupicapra rupicapra* | Chamois | [Q131340](https://www.wikidata.org/wiki/Q131340)                                                          | 15 | 747 |
| 7 | *Aves* | Bird | [Q5113](https://www.wikidata.org/wiki/Q5113)                                                              | 75 | 942 |
| 8 | *Homo sapiens* | Human | [Q15978631](https://www.wikidata.org/wiki/Q15978631)                                                      | 93 | 1,158 |
| 9 | *Canis lupus familiaris* | Dog | [Q26972265](https://www.wikidata.org/wiki/Q26972265)                                                      | 7 | 95 |
| 10 | *Sus scrofa × Sus domesticus* | Hybrid pig | [Q602666](https://www.wikidata.org/wiki/Q602666) (no matching wikidata id, so workaround with forest hog) | 44 | 1,484 |
| 11 | — | No-animal | [Q10738](https://www.wikidata.org/wiki/Q10738) (it is not an animal, it is a rock, THE ROCK)              | 60 | 521 |
| 12 | — | Unknown | [Q24238356](https://www.wikidata.org/wiki/Q24238356)                                                      | 344 | 2,536 |


### Annotation Format (MOT)

Annotations are stored as CSV files (no header) following a custom MOT format:

```
frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility, species, gender, age, is_propagated
```

| Column | Type | Description |
|---|---|---|
| `frame` | int | Frame index |
| `track_id` | int | Unique track identifier |
| `bb_left` | int | Bounding box left coordinate (px) |
| `bb_top` | int | Bounding box top coordinate (px) |
| `bb_width` | int | Bounding box width (px) |
| `bb_height` | int | Bounding box height (px) |
| `conf` | float | Confidence score |
| `class_id` | int | Species class identifier |
| `visibility` | float | Visibility (1.0 = fully visible, 0.0 = fully occluded) |
| `species` | str | Species name |
| `gender` | int | Gender (0 = unknown, 1 = male, 2 = female) |
| `age` | int | Age (0 = unknown, 1 = juvenile, 2 = adult) |
| `is_propagated` | int | 0 = annotated key frame, 1 = interpolated |

Class_ids are defined as `<wikidata_id>-<gender>-<age>-<visibility>` with the following mapping:
```json
{
  "class_mapping": {
    "Q10738-0-0-0": 0,
    "Q10738-0-0-1": 1,
    "Q1219579-0-0-0": 2,
    "Q1219579-0-0-1": 3,
    "Q1219579-0-1-0": 4,
    "Q1219579-0-1-1": 5,
    "Q1219579-0-2-0": 6,
    "Q1219579-0-2-1": 7,
    "Q1219579-1-0-0": 8,
    "Q1219579-1-0-1": 9,
    "Q1219579-1-2-0": 10,
    "Q1219579-1-2-1": 11,
    "Q1219579-2-0-0": 12,
    "Q1219579-2-0-1": 13,
    "Q1219579-2-1-0": 14,
    "Q1219579-2-2-0": 15,
    "Q1219579-2-2-1": 16,
    "Q122069-0-0-0": 17,
    "Q122069-0-0-1": 18,
    "Q122069-0-1-0": 19,
    "Q122069-0-1-1": 20,
    "Q122069-0-2-0": 21,
    "Q122069-0-2-1": 22,
    "Q131340-0-0-0": 23,
    "Q131340-0-0-1": 24,
    "Q131340-0-2-0": 25,
    "Q131340-0-2-1": 26,
    "Q15978631-0-0-0": 27,
    "Q15978631-0-0-1": 28,
    "Q15978631-0-2-0": 29,
    "Q15978631-0-2-1": 30,
    "Q15978631-2-0-0": 31,
    "Q15978631-2-2-0": 32,
    "Q15978631-2-2-1": 33,
    "Q168327-0-0-0": 34,
    "Q168327-0-0-1": 35,
    "Q168327-0-1-0": 36,
    "Q168327-0-1-1": 37,
    "Q168327-0-2-0": 38,
    "Q168327-0-2-1": 39,
    "Q168327-2-2-0": 40,
    "Q168327-2-2-1": 41,
    "Q20908334-0-0-0": 42,
    "Q20908334-0-1-0": 43,
    "Q20908334-1-0-0": 44,
    "Q20908334-1-1-0": 45,
    "Q20908334-1-2-0": 46,
    "Q20908334-2-2-0": 47,
    "Q20908334-2-2-1": 48,
    "Q24238356-0-0-0": 49,
    "Q24238356-0-0-1": 50,
    "Q24238356-0-2-0": 51,
    "Q24238356-0-2-1": 52,
    "Q26972265-0-0-0": 53,
    "Q26972265-0-2-0": 54,
    "Q26972265-0-2-1": 55,
    "Q5113-0-0-0": 56,
    "Q5113-0-0-1": 57,
    "Q5113-0-2-0": 58,
    "Q5113-0-2-1": 59,
    "Q58697-0-0-0": 60,
    "Q58697-0-0-1": 61,
    "Q58697-0-1-0": 62,
    "Q58697-0-1-1": 63,
    "Q58697-0-2-0": 64,
    "Q58697-0-2-1": 65,
    "Q58697-1-2-0": 66,
    "Q58697-1-2-1": 67,
    "Q58697-2-2-0": 68,
    "Q58697-2-2-1": 69,
    "Q602666-0-0-0": 70,
    "Q602666-0-0-1": 71,
    "Q602666-0-1-0": 72,
    "Q602666-0-2-0": 73,
    "Q602666-0-2-1": 74
  }
}
```

---

## Availability

The dataset is publicly available on Zenodo:

- [10.5281/zenodo.18692354](https://doi.org/10.5281/zenodo.18692354)
- [10.5281/zenodo.18698508](https://doi.org/10.5281/zenodo.18698508)
- [10.5281/zenodo.18703312](https://doi.org/10.5281/zenodo.18703312)
- [10.5281/zenodo.18705705](https://doi.org/10.5281/zenodo.18705705)
- [10.5281/zenodo.18707610](https://doi.org/10.5281/zenodo.18707610)
- [10.5281/zenodo.18711217](https://doi.org/10.5281/zenodo.18711217)
- [10.5281/zenodo.18715162](https://doi.org/10.5281/zenodo.18715162)
- [10.5281/zenodo.18717601](https://doi.org/10.5281/zenodo.18717601)

---

## Scripts

### Installation

```bash
pip install -r requirements.txt
```

The scripts are tested with Python 3.10+.

### Automatic Download

Selectively download flight ZIPs from the BAMBI dataset hosted on Zenodo. Uses the zenodo_upload_summary.json to resolve which depositions contain which flights, so you can grab exactly what you need without fetching entire multi-GB depositions. Use `filter_flights.py` to get a list of flight IDs for the data that you are looking for (e.g. filtered for species).


```bash
# List all available flights
python download_from_zenodo.py -s zenodo_upload_summary.json --list

# Download flights 0, 5, and 12
python download_from_zenodo.py -f 0 5 12

# Download flights 10 through 25
python download_from_zenodo.py --range 10 25

# Download all flights from parts 1 and 3
python download_from_zenodo.py --parts 1 3

# Download everything, extract, and clean up ZIPs
python download_from_zenodo.py --unzip

# Preview what a full download would do
python download_from_zenodo.py --dry-run
```

### Flight filter

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

### Frame Extraction

Extract thermal and RGB frames from the side-by-side video files. The left half of each frame contains the thermal channel, the right half the RGB channel.

```bash
# Extract every frame
python frame_extraction.py video.mp4 -o ./frames

# Extract every 10th frame, RGB only
python frame_extraction.py video.mp4 -o ./frames --sample-rate 10 --thermal false

# Extract a specific frame range
python frame_extraction.py video.mp4 -o ./frames --start 1000 --end 2000
```

Output structure:
```
frames/
├── thermal/
│   ├── video_00001000.png
│   ├── video_00001001.png
│   └── ...
└── rgb/
    ├── video_00001000.png
    ├── video_00001001.png
    └── ...
```

### MOT Interpolation

Interpolate missing frames between annotated key frames using linear interpolation of bounding box coordinates. Interpolated entries are marked with `is_propagated=1`.

```bash
# Interpolate a single annotation file
python mot_interpolation.py annotations.txt ./output

# Interpolate all files in a folder with a custom frame step
python mot_interpolation.py ./annotations_folder ./output --step 2
```

### MOT Filter

Filter annotation files by species, class ID, visibility, bounding box size, gender, or age.

```bash
# Keep only wild boar annotations
python mot_filter.py annotations.txt -o ./filtered --species "Sus scrofa (Wild boar)"

# Filter by class ID and minimum bounding box width
python mot_filter.py ./annotations/ -o ./filtered --class-id 50 51 --min-width 25

# Combine multiple filters
python mot_filter.py annotations.txt -o ./filtered --species "Cervus elaphus (Red deer)" --visibility 1.0 --min-width 30
```

### MOT to YOLO Conversion

Convert MOT annotations to YOLO label format for training object detection models.

```bash
# Convert with default label (class_id only)
python mot_to_yolo.py annotations.txt -o ./yolo_labels

# Include species, gender, and age in the label
python mot_to_yolo.py ./annotations/ -o ./yolo_labels --img-width 640 --img-height 512 --labels class_id gender age
```

### Visualization

#### On Extracted Frames

Overlay bounding boxes on individual extracted frames:

```bash
# Visualize key frame annotations
python mot_frame_viewer.py frame_image.png annotations.txt --show

# With interpolation of in-between frames
python mot_frame_viewer.py frame_image.png annotations.txt --interpolate --show

# Save the visualization to a file
python mot_frame_viewer.py frame_image.png annotations.txt -o output.png
```

#### On Video

Overlay bounding box tracks directly on video:

```bash
# Live preview
python mot_video_viewer.py video.mp4 annotations.txt --show

# Save annotated video
python mot_video_viewer.py video.mp4 annotations.txt -o annotated_output.mp4

# With interpolated tracks
python mot_video_viewer.py video.mp4 annotations.txt -o output.mp4 --interpolate
```

### DEM from Poses

Download, merge, and clip Digital Elevation Model (DGM) data from the Austrian Federal Office of Metrology and Surveying (BEV) based on GPS coordinates from pose files. Outputs a GeoTIFF clipped to the flight area, with optional GLB mesh generation.

```bash
# From a single poses file
python dem_from_poses.py --file recordings/0_matched_poses.json

# From a folder (finds all *_poses.json recursively)
python dem_from_poses.py --folder recordings/

# Custom padding and output directory
python dem_from_poses.py --folder recordings/ --padding 100 --output-dir /tmp/dems

# Custom simplification factor and output CRS
python dem_from_poses.py --file recording/0_matched_poses.json --simplify 1 --output-crs EPSG:32633

# Skip mesh generation
python dem_from_poses.py --file recording/0_matched_poses.json --no-mesh
```

| Option | Default | Description |
|---|---|---|
| `--file` / `--folder` | *(required)* | Single `*_poses.json` file or folder to scan recursively |
| `--padding` | `50` | Padding around bounding box in metres |
| `--output-dir` | `DEM/` subfolder | Output directory for GeoTIFF and mesh files |
| `--output-crs` | `EPSG:32633` | Output coordinate reference system |
| `--cache-dir` | `~/.cache/austria_dem` | Tile cache directory |
| `--force-download` | `false` | Re-download tiles even if cached |
| `--no-mesh` | `false` | Skip GLTF (`.glb`) mesh generation |
| `--simplify` | `2` | Mesh simplification factor |

### Add Relative DEM Position to Poses

Add relative location offsets to drone pose files based on DEM origin metadata created with `dem_from_poses.py`. Converts WGS84 (lat/lng/alt) coordinates to the CRS defined in the DEM metadata, then computes relative `[x, y, z]` offsets from the DEM origin. Also adds a `rotation` field (`[pitch, roll, yaw]`) to each image entry.

```bash
# Single file pair
python add_relative_dem_position_to_poses.py --poses 0_matched_poses.json --dem 0_matched_dem.json --output ./output

# Folder mode (matches files by filename prefix)
python add_relative_dem_position_to_poses.py --poses ./poses/ --dem ./dem_metadata/ --output ./output

# Modify pose files in place
python add_relative_dem_position_to_poses.py --poses ./poses/ --dem ./dem_metadata/ --inplace
```

| Option | Default | Description |
|---|---|---|
| `--poses` | *(required)* | Single pose JSON file or folder of `*_matched_poses.json` files |
| `--dem` | *(required)* | Single DEM metadata JSON file or folder of `*_matched_dem.json` files |
| `--output` | — | Output folder for modified pose files |
| `--inplace` | `false` | Modify pose files in place (requires no `--output`) |

In folder mode, files are matched by the prefix before the first underscore (e.g., `0_matched_poses.json` matches `0_matched_dem.json`).

---

## RGB–Thermal Frame Matching

Although the drones record RGB and thermal imagery simultaneously, the two modalities are **not temporally synchronized**. Temporal offsets vary with flight dynamics, so annotations from thermal frames cannot be directly transferred to RGB images — correspondence must be established at sequence or frame level.

The dataset includes a **local patch-based matching strategy** to align thermal annotations with their RGB counterparts:

1. For each thermal detection, the annotated crop is extracted.
2. A larger search region is generated in the corresponding RGB frame by expanding the bounding box.
3. The RGB region is converted to grayscale and processed using multiple transformations (e.g., CLAHE and edge-based filtering), producing five different variants.
4. Each variant is matched to the thermal crop via a sliding-window approach, selecting the position with the highest matching confidence.
5. For frames with **multiple detections**, alignment is accepted if pixel shift estimates across methods agree within ±10 pixels, and the most consistent candidate is selected.
6. For frames with a **single detection**, a stricter majority consensus is required; otherwise the sample is discarded.

The implementation is available in a separate repository:

🔗 **[BAMBI BBox Corrections](https://github.com/HugoMarkoff/BAMBI_BBox_Corrections)**

## Additional related repositories:

- [AlfsPY](https://github.com/bambi-eco/alfs_py): Framework for orthographic projections and light field renderings based on the drone recordings.
- [Detection](https://github.com/bambi-eco/bambi_detection): Examples on using AlfsPY for different tasks like geo-tiff generation.
- [Geo-Referenced Tracking](https://github.com/bambi-eco/Geo-Referenced-Tracking): Implementation of tracking algorithms based on local image as well as global world coordinates.
- [Bambi-QGIS](https://github.com/bambi-eco/Bambi-QGIS): Plugin for integrating drone video processing to the geo-information system QIGS.


---

## Citation

```bibtex
@inproceedings{bambi2026,
  title     = {The BAMBI Dataset: Multimodal Nadir UAV-Recordings of Forest Wildlife},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```

---

## License

This repository is licensed under the [MIT License](LICENSE).