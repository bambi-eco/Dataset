#!/usr/bin/env python3
"""
DEM Downloader from Poses JSON Files

Downloads, merges, and clips Digital Elevation Model (DGM) data from the
Austrian Federal Office of Metrology and Surveying (BEV) ATOM service,
using GPS coordinates extracted from *_poses.json files.

Two modes of operation:
  1. Single file:  --file path/to/0_matched_poses.json
  2. Folder scan:  --folder path/to/data/   (finds all *_poses.json recursively)

The output GeoTIFF is saved to a DEM/ subfolder next to the input JSON file(s)
by default, or to a custom location via --output-dir.

Data source: https://data.bev.gv.at
License: CC-BY-4.0 (Austrian Federal Office BEV 1m ALS-DTM)

Dependencies:
    pip install numpy rasterio pyproj requests shapely

Usage examples:
    # From a single poses file
    python dem_from_poses.py --file recordings/0_matched_poses.json

    # From a folder (finds all *_poses.json)
    python dem_from_poses.py --folder recordings/

    # Custom padding and output directory
    python dem_from_poses.py --folder recordings/ --padding 100 --output-dir /tmp/dems

    # With custom simplification and CRS
    python dem_from_poses.py --file recording/0_matched_poses.json --simplify 1 --output-crs EPSG:32633

    # Skip mesh generation
    python dem_from_poses.py --file recording/0_matched_poses.json --no-mesh
"""

import argparse
import json
import logging
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except ImportError:
    print("Error: rasterio is required. Install with: pip install rasterio")
    sys.exit(1)

try:
    from pyproj import Transformer
except ImportError:
    print("Error: pyproj is required. Install with: pip install pyproj")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests is required. Install with: pip install requests")
    sys.exit(1)

try:
    from shapely.geometry import box, mapping
except ImportError:
    print("Error: shapely is required. Install with: pip install shapely")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WGS84_CRS = "EPSG:4326"
BEV_CRS = "EPSG:3035"
DEFAULT_OUTPUT_CRS = "EPSG:32633"  # UTM zone 33N

BEV_URL_PATTERNS = [
    ("20230915", "ALS_DTM_CRS3035RES50000mN{north}E{east}.tif"),
    ("20190915", "CRS3035RES50000mN{north}E{east}.tif"),
    ("20210401", "ALS_DTM_CRS3035RES50000mN{north}E{east}.tif"),
]
BEV_DOWNLOAD_BASE = "https://data.bev.gv.at/download/ALS/DTM/"
TILE_SIZE = 50000  # 50 km in metres (EPSG:3035)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "austria_dem"


# ===================================================================
# Bounding-box helpers
# ===================================================================
@dataclass
class BoundingBox:
    """Geographic bounding box in WGS-84."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @classmethod
    def from_coords(
        cls,
        lats: List[float],
        lons: List[float],
        padding_meters: float = 0,
    ) -> "BoundingBox":
        """Create a bounding box from lists of lat/lon with optional padding."""
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        if padding_meters > 0:
            lat_center = (min_lat + max_lat) / 2
            m_per_deg_lat = 111320
            m_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
            lat_pad = padding_meters / m_per_deg_lat
            lon_pad = padding_meters / m_per_deg_lon
            min_lat -= lat_pad
            max_lat += lat_pad
            min_lon -= lon_pad
            max_lon += lon_pad

        return cls(min_lat, min_lon, max_lat, max_lon)

    def to_crs(self, target_crs: str) -> Tuple[float, float, float, float]:
        """Convert to projected coordinates (min_x, min_y, max_x, max_y)."""
        transformer = Transformer.from_crs(WGS84_CRS, target_crs, always_xy=True)
        corners = [
            (self.min_lon, self.min_lat),
            (self.max_lon, self.min_lat),
            (self.min_lon, self.max_lat),
            (self.max_lon, self.max_lat),
        ]
        xs, ys = [], []
        for lon, lat in corners:
            x, y = transformer.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        return min(xs), min(ys), max(xs), max(ys)


# ===================================================================
# Poses JSON reader
# ===================================================================
def read_poses_json(path: Path) -> Tuple[List[float], List[float]]:
    """
    Read a *_poses.json and return (lats, lons).

    Expected structure::

        { "images": [ {"lat": ..., "lng": ..., ...}, ... ] }
    """
    with open(path) as f:
        data = json.load(f)

    images = data.get("images", [])
    if not images:
        raise ValueError(f"No 'images' array found in {path}")

    lats, lons = [], []
    for img in images:
        lat = img.get("lat")
        lon = img.get("lng") or img.get("lon") or img.get("longitude")
        if lat is None or lon is None:
            continue
        lats.append(float(lat))
        lons.append(float(lon))

    if not lats:
        raise ValueError(f"No valid lat/lng entries in {path}")

    return lats, lons


def find_poses_files(folder: Path) -> List[Path]:
    """Recursively find all *_poses.json files in *folder*."""
    pattern = str(folder / "**" / "*_poses.json")
    files = sorted(Path(p) for p in glob(pattern, recursive=True))
    return files


# ===================================================================
# Tile calculation & downloading  (mirrors the original script)
# ===================================================================
class BEVTileCalculator:
    """Work out which 50 km × 50 km BEV tiles cover a bounding box."""

    def get_required_tiles(self, bbox: BoundingBox) -> List[str]:
        min_x, min_y, max_x, max_y = bbox.to_crs(BEV_CRS)
        logger.debug(
            "Bbox EPSG:3035  %.0f, %.0f → %.0f, %.0f", min_x, min_y, max_x, max_y
        )
        start_e = int(min_x // TILE_SIZE) * TILE_SIZE
        end_e = int(max_x // TILE_SIZE) * TILE_SIZE
        start_n = int(min_y // TILE_SIZE) * TILE_SIZE
        end_n = int(max_y // TILE_SIZE) * TILE_SIZE
        tiles = [
            f"N{n}E{e}"
            for n in range(start_n, end_n + TILE_SIZE, TILE_SIZE)
            for e in range(start_e, end_e + TILE_SIZE, TILE_SIZE)
        ]
        logger.info("Required BEV tiles: %d  (%s)", len(tiles), ", ".join(tiles))
        return tiles

    @staticmethod
    def get_download_urls(tile_name: str) -> List[str]:
        m = re.search(r"N(\d+)E(\d+)", tile_name)
        if not m:
            return []
        north, east = m.group(1), m.group(2)
        return [
            f"{BEV_DOWNLOAD_BASE}{date}/{pat.format(north=north, east=east)}"
            for date, pat in BEV_URL_PATTERNS
        ]


class BEVDownloader:
    """Download (and cache) BEV DEM tiles."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Austria-DEM-Processor/1.0 (Wildlife Research)"}
        )
        self.calc = BEVTileCalculator()

    def _cache_path(self, tile_name: str) -> Path:
        return self.cache_dir / f"{tile_name}.tif"

    def download_tile(self, tile_name: str, force: bool = False) -> Optional[Path]:
        cp = self._cache_path(tile_name)
        if cp.exists() and not force:
            logger.info("Using cached tile: %s", tile_name)
            return cp

        for url in self.calc.get_download_urls(tile_name):
            logger.info("Downloading %s …", tile_name)
            logger.debug("  URL: %s", url)
            try:
                resp = self.session.get(url, stream=True, timeout=600)
                if resp.status_code == 404:
                    logger.debug("  404 – trying next URL pattern")
                    continue
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(cp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            logger.debug("  %.1f %%", downloaded / total * 100)
                mb = cp.stat().st_size / 1024 / 1024
                logger.info("Downloaded %s (%.1f MB)", tile_name, mb)
                return cp
            except requests.RequestException as exc:
                logger.debug("  Failed: %s", exc)
                continue

        logger.warning("Tile not available from any URL pattern: %s", tile_name)
        if cp.exists():
            cp.unlink()
        return None

    def download_for_bbox(
        self, bbox: BoundingBox, force: bool = False
    ) -> List[Path]:
        tiles = self.calc.get_required_tiles(bbox)
        paths = []
        for t in tiles:
            p = self.download_tile(t, force=force)
            if p:
                paths.append(p)
        if not paths:
            logger.error("No tiles downloaded – is the area inside Austria?")
        return paths


# ===================================================================
# DEM processing  (merge → clip → reproject)
# ===================================================================
class DEMProcessor:
    def __init__(self, output_crs: str = DEFAULT_OUTPUT_CRS):
        self.output_crs = output_crs

    def merge_and_clip(
        self, input_files: List[Path], output_file: Path, bbox: BoundingBox
    ) -> Optional[Path]:
        if not input_files:
            logger.error("No input files to process")
            return None
        logger.info("Merging & clipping %d tile(s) …", len(input_files))
        try:
            srcs = [rasterio.open(f) for f in input_files]
            src_crs = srcs[0].crs
            min_x, min_y, max_x, max_y = bbox.to_crs(str(src_crs))
            clip_geom = box(min_x, min_y, max_x, max_y)
            logger.info(
                "Clip bounds (%s): %.1f, %.1f → %.1f, %.1f",
                src_crs, min_x, min_y, max_x, max_y,
            )

            if len(srcs) == 1:
                out_image, out_transform = mask(
                    srcs[0], [mapping(clip_geom)], crop=True, all_touched=True
                )
                out_meta = srcs[0].meta.copy()
            else:
                mosaic, mosaic_transform = merge(srcs)
                merged_meta = srcs[0].meta.copy()
                merged_meta.update(
                    height=mosaic.shape[1],
                    width=mosaic.shape[2],
                    transform=mosaic_transform,
                )
                tmp = output_file.parent / "temp_merged.tif"
                with rasterio.open(tmp, "w", **merged_meta) as dst:
                    dst.write(mosaic)
                with rasterio.open(tmp) as src:
                    out_image, out_transform = mask(
                        src, [mapping(clip_geom)], crop=True, all_touched=True
                    )
                    out_meta = src.meta.copy()
                tmp.unlink()

            out_meta.update(
                driver="GTiff",
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                compress="lzw",
            )
            with rasterio.open(output_file, "w", **out_meta) as dst:
                dst.write(out_image)
            for s in srcs:
                s.close()
            logger.info("Clipped DEM: %s", output_file)
            return output_file
        except Exception as exc:
            logger.error("merge_and_clip failed: %s", exc)
            return None

    def reproject(self, input_file: Path, output_file: Path) -> Optional[Path]:
        target = self.output_crs
        logger.info("Reprojecting to %s …", target)
        try:
            with rasterio.open(input_file) as src:
                if str(src.crs) == target:
                    shutil.copy2(input_file, output_file)
                    return output_file
                transform, width, height = calculate_default_transform(
                    src.crs, target, src.width, src.height, *src.bounds
                )
                kw = src.meta.copy()
                kw.update(
                    crs=target, transform=transform,
                    width=width, height=height, compress="lzw",
                )
                with rasterio.open(output_file, "w", **kw) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target,
                        resampling=Resampling.bilinear,
                    )
            logger.info("Reprojected: %s", output_file)
            return output_file
        except Exception as exc:
            logger.error("Reprojection failed: %s", exc)
            return None


# ===================================================================
# GLTF mesh generation (optional, mirrors original script)
# ===================================================================
class GLTFMeshGenerator:
    """Generate a GLTF (.glb) mesh from a GeoTIFF DEM."""

    def __init__(self, simplify_factor: int = 2):
        self.simplify_factor = max(1, simplify_factor)

    def generate(
        self,
        geotiff: Path,
        output_glb: Path,
        metadata_path: Optional[Path] = None,
    ) -> bool:
        logger.info("Generating GLTF mesh from %s …", geotiff.name)
        try:
            with rasterio.open(geotiff) as src:
                elev = src.read(1).astype(np.float64)
                nodata = src.nodata
                if nodata is not None:
                    elev[elev == nodata] = np.nan
                elev[elev == 0] = np.nan

                transform = src.transform
                crs = str(src.crs)
                bounds = src.bounds
                w, h = src.width, src.height

            valid = elev[~np.isnan(elev)]
            if valid.size == 0:
                logger.error("No valid elevation data")
                return False

            origin_x, origin_y = bounds.left, bounds.bottom
            origin_z = float(valid.min())
            tx = Transformer.from_crs(crs, WGS84_CRS, always_xy=True)
            origin_lon, origin_lat = tx.transform(origin_x, origin_y)

            sf = self.simplify_factor
            if sf > 1:
                elev = elev[::sf, ::sf]
                pw = transform.a * sf
                ph = transform.e * sf
            else:
                pw, ph = transform.a, transform.e
            mh, mw = elev.shape

            elev = np.nan_to_num(elev, nan=origin_z) - origin_z

            # vertices
            verts = np.empty((mh, mw, 3), dtype=np.float32)
            cols = np.arange(mw, dtype=np.float32) * pw
            rows = (np.arange(mh - 1, -1, -1, dtype=np.float32)) * abs(ph)
            verts[:, :, 0] = cols[np.newaxis, :]
            verts[:, :, 1] = rows[:, np.newaxis]
            verts[:, :, 2] = elev.astype(np.float32)
            verts_flat = verts.reshape(-1, 3)

            # indices
            r = np.arange(mh - 1)[:, None]
            c = np.arange(mw - 1)[None, :]
            i00 = (r * mw + c).ravel()
            i10 = (r * mw + c + 1).ravel()
            i01 = ((r + 1) * mw + c).ravel()
            i11 = ((r + 1) * mw + c + 1).ravel()
            indices = np.column_stack([i00, i01, i10, i10, i01, i11]).ravel().astype(np.uint32)

            # normals
            normals = self._normals(verts_flat, indices)

            self._write_glb(verts_flat, indices, normals, output_glb)

            # metadata
            if metadata_path is None:
                metadata_path = output_glb.with_suffix(".json")
            t9 = [transform.a, transform.b, transform.c,
                  transform.d, transform.e, transform.f, 0, 0, 1]
            meta = {
                "width": w, "height": h, "crs": crs, "transform": t9,
                "origin": [float(origin_x), float(origin_y), float(origin_z)],
                "origin_wgs84": {
                    "latitude": float(origin_lat),
                    "longitude": float(origin_lon),
                    "altitude": float(origin_z),
                },
            }
            with open(metadata_path, "w") as f:
                json.dump(meta, f, indent=4)
            logger.info("Mesh: %s  |  Metadata: %s", output_glb, metadata_path)
            return True
        except Exception as exc:
            logger.error("Mesh generation failed: %s", exc, exc_info=True)
            return False

    # --- helpers -------------------------------------------------------
    @staticmethod
    def _normals(verts: np.ndarray, indices: np.ndarray) -> np.ndarray:
        norms = np.zeros_like(verts)
        tris = indices.reshape(-1, 3)
        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            fn = np.cross(v1 - v0, v2 - v0)
            n = np.linalg.norm(fn)
            if n > 0:
                fn /= n
            norms[tri] += fn
        lens = np.linalg.norm(norms, axis=1, keepdims=True)
        lens[lens == 0] = 1
        return (norms / lens).astype(np.float32)

    @staticmethod
    def _write_glb(
        verts: np.ndarray, indices: np.ndarray,
        normals: np.ndarray, path: Path,
    ):
        def _pad4(b: bytes) -> bytes:
            return b + b"\x00" * ((4 - len(b) % 4) % 4)

        vb = _pad4(verts.astype(np.float32).tobytes())
        nb = _pad4(normals.astype(np.float32).tobytes())
        ib = _pad4(indices.astype(np.uint32).tobytes())
        buf = vb + nb + ib

        vr = verts.reshape(-1, 3)
        nr = normals.reshape(-1, 3)
        gltf = {
            "asset": {"version": "2.0", "generator": "DEM-from-poses"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(vr), "type": "VEC3",
                 "min": vr.min(0).tolist(), "max": vr.max(0).tolist()},
                {"bufferView": 1, "componentType": 5126, "count": len(nr), "type": "VEC3",
                 "min": nr.min(0).tolist(), "max": nr.max(0).tolist()},
                {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR",
                 "min": [int(indices.min())], "max": [int(indices.max())]},
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": verts.nbytes, "target": 34962},
                {"buffer": 0, "byteOffset": len(vb), "byteLength": normals.nbytes, "target": 34962},
                {"buffer": 0, "byteOffset": len(vb) + len(nb), "byteLength": indices.nbytes, "target": 34963},
            ],
            "buffers": [{"byteLength": len(buf)}],
        }
        jb = json.dumps(gltf, separators=(",", ":")).encode()
        jb += b" " * ((4 - len(jb) % 4) % 4)
        total = 12 + 8 + len(jb) + 8 + len(buf)
        with open(path, "wb") as f:
            f.write(b"glTF")
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<I", total))
            f.write(struct.pack("<I", len(jb)))
            f.write(b"JSON")
            f.write(jb)
            f.write(struct.pack("<I", len(buf)))
            f.write(b"BIN\x00")
            f.write(buf)
        logger.info(
            "GLB written: %s  (%.2f MB, %d verts, %d tris)",
            path, path.stat().st_size / 1e6, len(vr), len(indices) // 3,
        )


# ===================================================================
# Main pipeline
# ===================================================================
def process_dem(
    lats: List[float],
    lons: List[float],
    output_tif: Path,
    *,
    padding: float = 50,
    output_crs: str = DEFAULT_OUTPUT_CRS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    mesh: bool = False,
    simplify: int = 2,
) -> Optional[Path]:
    """
    Full pipeline: bbox → download → merge/clip → reproject → (optional mesh).

    Returns the final GeoTIFF path on success, None on failure.
    """
    bbox = BoundingBox.from_coords(lats, lons, padding_meters=padding)
    logger.info("=" * 60)
    logger.info("Bounding box (WGS-84):")
    logger.info("  SW: %.6f, %.6f", bbox.min_lat, bbox.min_lon)
    logger.info("  NE: %.6f, %.6f", bbox.max_lat, bbox.max_lon)
    logger.info("=" * 60)

    # 1. Download
    downloader = BEVDownloader(cache_dir=cache_dir)
    tile_paths = downloader.download_for_bbox(bbox, force=force_download)
    if not tile_paths:
        return None

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    processor = DEMProcessor(output_crs=output_crs)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # 2. Merge & clip
        clipped = td / "clipped.tif"
        if not processor.merge_and_clip(tile_paths, clipped, bbox):
            return None

        # 3. Reproject
        reproj = td / "reprojected.tif"
        if not processor.reproject(clipped, reproj):
            return None

        shutil.copy2(reproj, output_tif)
        logger.info("Output GeoTIFF: %s", output_tif)

    # 4. Optional mesh
    if mesh:
        glb = output_tif.with_suffix(".glb")
        GLTFMeshGenerator(simplify_factor=simplify).generate(output_tif, glb)

    return output_tif


def derive_output_dir(json_path: Path, custom_dir: Optional[Path]) -> Path:
    """Return the output directory – defaults to a DEM/ sibling of the JSON."""
    if custom_dir is not None:
        return Path(custom_dir)
    return json_path.parent / "DEM"


def stem_from_json(json_path: Path) -> str:
    """
    Derive a clean filename stem from the poses JSON path.

    '0_matched_poses.json' → '0_matched'
    """
    name = json_path.stem                       # '0_matched_poses'
    if name.endswith("_poses"):
        name = name[: -len("_poses")]           # '0_matched'
    return name or "dem"


# ===================================================================
# CLI
# ===================================================================
def main():
    p = argparse.ArgumentParser(
        description="Download Austrian DEM tiles for areas covered by poses JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python dem_from_poses.py --file recordings/0_matched_poses.json

  # All *_poses.json in a folder tree
  python dem_from_poses.py --folder recordings/

  # Custom padding & output directory
  python dem_from_poses.py --folder recordings/ --padding 100 --output-dir /tmp/dems

  # Generate mesh alongside GeoTIFF
  python dem_from_poses.py --file rec/0_matched_poses.json --simplify 2

  # Skip mesh generation
  python dem_from_poses.py --file rec/0_matched_poses.json --no-mesh
""",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--file", "-f", type=Path,
        help="Path to a single *_poses.json file.",
    )
    mode.add_argument(
        "--folder", "-d", type=Path,
        help="Folder to scan (recursively) for *_poses.json files.",
    )

    p.add_argument("--padding", type=float, default=50,
                    help="Padding around bounding box in metres (default: 50).")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory (default: DEM/ subfolder next to JSON).")
    p.add_argument("--output-crs", default=DEFAULT_OUTPUT_CRS,
                    help=f"Output CRS (default: {DEFAULT_OUTPUT_CRS}).")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help=f"Tile cache directory (default: {DEFAULT_CACHE_DIR}).")
    p.add_argument("--force-download", action="store_true",
                    help="Re-download tiles even if cached.")
    p.add_argument("--no-mesh", action="store_true",
                    help="Skip GLTF (.glb) mesh generation (enabled by default).")
    p.add_argument("--simplify", type=int, default=2,
                    help="Mesh simplification factor (default: 2).")
    p.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose / debug output.")

    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Collect JSON files -------------------------------------------------
    if args.file:
        if not args.file.exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        json_files = [args.file]
    else:
        if not args.folder.is_dir():
            logger.error("Not a directory: %s", args.folder)
            sys.exit(1)
        json_files = find_poses_files(args.folder)
        if not json_files:
            logger.error("No *_poses.json files found in %s", args.folder)
            sys.exit(1)
        logger.info("Found %d poses file(s):", len(json_files))
        for jf in json_files:
            logger.info("  %s", jf)

    # Process each file --------------------------------------------------
    results = []
    for jf in json_files:
        logger.info("─" * 60)
        logger.info("Processing %s", jf)
        try:
            lats, lons = read_poses_json(jf)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Skipping %s: %s", jf, exc)
            continue

        logger.info("  %d frames, lat [%.6f … %.6f], lon [%.6f … %.6f]",
                     len(lats), min(lats), max(lats), min(lons), max(lons))

        out_dir = derive_output_dir(jf, args.output_dir)
        stem = stem_from_json(jf)
        out_tif = out_dir / f"{stem}_dem.tif"

        result = process_dem(
            lats, lons, out_tif,
            padding=args.padding,
            output_crs=args.output_crs,
            cache_dir=args.cache_dir,
            force_download=args.force_download,
            mesh=not args.no_mesh,
            simplify=args.simplify,
        )
        if result:
            results.append(result)

    # Summary ------------------------------------------------------------
    logger.info("═" * 60)
    if results:
        logger.info("Done – %d / %d DEM(s) created:", len(results), len(json_files))
        for r in results:
            logger.info("  %s", r)
    else:
        logger.error("No DEMs were created.")
        sys.exit(1)


if __name__ == "__main__":
    main()