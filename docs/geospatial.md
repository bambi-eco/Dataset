# Geospatial tools

Deriving terrain models from the flight poses, and relating the drone position
to that terrain.

## DEM from Poses

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

## Add Relative DEM Position to Poses

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

## Overview map of all flights

`flight_overview_map.py` draws where the dataset was recorded. Many flights sit
on top of each other -- 130 of the 386 are from one forest near Purkersdorf --
so flights recorded within a given distance of each other are aggregated into
**sites** and drawn as one proportional symbol.

Positions come from the flight logs, never from a video: `<id>_matched_poses.json`
of the base release, or `air_data.csv` of the raw release. Both are reachable
with `--annotations-only`, so the whole dataset can be summarised from a few
hundred MB.

```bash
# the pose files of every flight (no videos), then the map
python download_from_zenodo.py --annotations-only --range 0 400 -o poses/
python flight_overview_map.py --poses poses/ -o figures/flight_overview.png \
    --geojson sites.geojson --csv sites.csv --cache flights.csv

# raw flight logs work as well, one folder per flight
python flight_overview_map.py --raw raw/ -o overview.png

# coarser aggregation, zoomed to the sites, no colour encoding
python flight_overview_map.py --poses poses/ --radius 5 --extent data --color-by none -o overview.png
```

| Option | Default | Description |
|---|---|---|
| `--poses` / `--raw` | *(one required)* | Folders searched recursively for `*_poses.json` or per-flight `air_data.csv` |
| `--metadata` | `flight_metadata` | Split, species, drone and recording site of every flight |
| `--cache` | — | Per-flight summary CSV; written on the first run and reused afterwards |
| `--radius` | `1` | Flights closer than this many km share a site (single linkage) |
| `--color-by` | `species` | Colour ramp over `species`, `dates`, or `none` for one flat hue |
| `--extent` | `country` | Show all of Austria, or `data` to zoom to the sites |
| `--theme` | `light` | `light` or `dark` |
| `--labels`, `--bars` | `12` | How many sites get a label on the map and a bar in the ranked panel |
| `-o`, `--geojson`, `--csv` | | The figure (`.png`/`.pdf`/`.svg`), one point per site for QGIS, and the site table |

Circle **area** is proportional to the number of flights, the colour ramp is the
number of species recorded there, and the panel on the right ranks the largest
sites. State boundaries come from Natural Earth (public domain) and are
downloaded once into `~/.cache/bambi_overview`; `--offline` skips them.

Flights whose log is not at hand are still counted: `flight_metadata/` names the
recording campaign of every flight, so a flight without coordinates joins the
site its campaign-mates were placed at. The subtitle says how many were placed
each way, and the run prints how well the geographic clustering agrees with
those names -- how many sites merge several campaign names, and how many names
are split over several sites. Both should be zero at a sensible radius.

---
