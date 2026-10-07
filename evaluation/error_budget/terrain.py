#!/usr/bin/env python3
"""
Height above ground of every published frame, from the BEV 1 m terrain model.

For every flight the nadir points of the corrected camera poses are looked up
in the Austrian ALS DTM (EPSG:3035, 50 km tiles).  The tiles are read by
window over HTTP (GDAL /vsicurl), so a flight costs a few seconds and no tile
is downloaded whole.  The correction of the dataset (`<id>_correction.json`,
translation per frame) is applied to the published pose before the lookup,
which is what the renderer does.

Writes <data>/agl/<id>_agl.csv with one row per frame: frame, x, y, alt,
ground, agl, and keeps a small clipped DEM per flight in <data>/agl/dem/.
Resumable; flights with an existing CSV are skipped.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import common as C   # noqa: E402
import frame_dem_animation as fa   # noqa: E402

BEV = "https://data.bev.gv.at/download/ALS/DTM/"
PATTERNS = [("20230915", "ALS_DTM_CRS3035RES50000mN{n}E{e}.tif"), ("20190915", "CRS3035RES50000mN{n}E{e}.tif"),
            ("20210401", "ALS_DTM_CRS3035RES50000mN{n}E{e}.tif")]
TILE = 50000
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
_open_cache = {}


def tile_dataset(n: int, e: int):
    import rasterio
    key = (n, e)
    if key not in _open_cache:
        last = None
        for date, pat in PATTERNS:
            url = f"/vsicurl/{BEV}{date}/{pat.format(n=n, e=e)}"
            try:
                _open_cache[key] = rasterio.open(url)
                break
            except Exception as exc:   # noqa: BLE001
                last = exc
        else:
            raise RuntimeError(f"no BEV tile N{n}E{e}: {last}")
    return _open_cache[key]


def ground_heights(x3035: np.ndarray, y3035: np.ndarray, dem_out: Path) -> np.ndarray:
    """Terrain height at the given EPSG:3035 points; also writes a clipped DEM of the extent."""
    import rasterio
    from rasterio.windows import Window
    margin = 60.0
    good = np.isfinite(x3035) & np.isfinite(y3035)
    x0, x1 = x3035[good].min() - margin, x3035[good].max() + margin
    y0, y1 = y3035[good].min() - margin, y3035[good].max() + margin
    z = np.full(len(x3035), np.nan)
    tiles = [(n, e) for n in range(int(y0 // TILE) * TILE, int(y1 // TILE) * TILE + 1, TILE)
             for e in range(int(x0 // TILE) * TILE, int(x1 // TILE) * TILE + 1, TILE)]
    clips = []
    for n, e in tiles:
        ds = tile_dataset(n, e)
        tr = ds.transform
        col0, row0 = ~tr * (max(x0, ds.bounds.left), min(y1, ds.bounds.top))
        col1, row1 = ~tr * (min(x1, ds.bounds.right), max(y0, ds.bounds.bottom))
        c0, r0 = int(np.floor(col0)), int(np.floor(row0))
        c1, r1 = int(np.ceil(col1)), int(np.ceil(row1))
        if c1 <= c0 or r1 <= r0:
            continue
        win = Window(c0, r0, c1 - c0, r1 - r0)
        arr = ds.read(1, window=win).astype(np.float32)
        nod = ds.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        wtr = ds.window_transform(win)
        clips.append((arr, wtr))
        cols, rows = ~wtr * (x3035, y3035)
        cols = np.asarray(cols); rows = np.asarray(rows)
        inside = good & (cols >= 0) & (cols < arr.shape[1] - 1) & (rows >= 0) & (rows < arr.shape[0] - 1)
        if inside.any():
            ci = cols[inside]; ri = rows[inside]
            c_ = np.floor(ci).astype(int); r_ = np.floor(ri).astype(int)
            fc = ci - c_; fr = ri - r_
            v = (arr[r_, c_] * (1 - fc) * (1 - fr) + arr[r_, c_ + 1] * fc * (1 - fr)
                 + arr[r_ + 1, c_] * (1 - fc) * fr + arr[r_ + 1, c_ + 1] * fc * fr)
            z[np.where(inside)[0]] = v
    if clips and dem_out is not None:
        arr, wtr = max(clips, key=lambda c: c[0].size)
        with rasterio.open(dem_out, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                           crs="EPSG:3035", transform=wtr, compress="deflate", nodata=np.nan) as dst:
            dst.write(arr, 1)
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"))
    ap.add_argument("--flights", nargs="*", default=None)
    args = ap.parse_args()
    out = args.data / "agl"
    (out / "dem").mkdir(parents=True, exist_ok=True)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    fls = C.flights(args.data, args.flights)
    done = 0
    for fl in fls:
        dest = out / f"{fl.fid}_agl.csv"
        if dest.exists():
            continue
        try:
            poses = C.load_poses(fl)
            corr = fl.path("correction.json")
            t_corr, _ = fa.load_corrections(corr, poses["n"]) if corr.exists() else (np.zeros((poses["n"], 3)), None)
            x, y = tr.transform(poses["lng"], poses["lat"])
            x = np.asarray(x) + t_corr[:, 0]; y = np.asarray(y) + t_corr[:, 1]     # the correction is in a local east/north frame
            alt = poses["alt"] + t_corr[:, 2]
            z = ground_heights(x, y, out / "dem" / f"{fl.fid}_dem3035.tif")
            agl = alt - z
            with open(dest, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["frame", "x3035", "y3035", "alt_corrected", "ground", "agl"])
                for i in range(poses["n"]):
                    wr.writerow([i, round(float(x[i]), 2), round(float(y[i]), 2), round(float(alt[i]), 2),
                                 round(float(z[i]), 2) if np.isfinite(z[i]) else "", round(float(agl[i]), 2) if np.isfinite(agl[i]) else ""])
            done += 1
            print(f"flight {fl.fid}: {poses['n']} frames, AGL median {np.nanmedian(agl):.1f} m (p5 {np.nanpercentile(agl, 5):.1f}, p95 {np.nanpercentile(agl, 95):.1f}), "
                  f"{np.isnan(agl).mean() * 100:.0f} % outside the DEM", flush=True)
        except Exception as exc:   # noqa: BLE001
            print(f"flight {fl.fid}: failed: {exc}", flush=True)
    print(f"done, {done} new flights")


if __name__ == "__main__":
    main()
