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
