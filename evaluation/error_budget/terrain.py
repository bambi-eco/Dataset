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
With --takeoff, a second pass appends agl_takeoff: terrain at the take-off
point + barometric height above take-off - terrain at the nadir, a height that
depends neither on the published altitude nor on the pose correction.
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


def takeoff_reference(fl: C.Flight, poses: dict, ground: np.ndarray):
    """
    Height above ground per frame that depends neither on the published altitude nor on the correction:
    terrain at the take-off point + barometric height above take-off (flight log) - terrain at the nadir.
    Returns (heights, take-off terrain, DJI take-off altitude); heights are NaN outside the flight log.
    """
    from pyproj import Transformer
    n = poses["n"]
    if not fl.has_raw:
        return np.full(n, np.nan), np.nan, np.nan
    rows = [{(k or "").strip(): v for k, v in r.items()} for r in csv.DictReader(open(fl.raw / "air_data.csv", encoding="utf-8"))]
    try:
        hat = np.array([float(r["height_above_takeoff(feet)"]) * 0.3048 for r in rows])
        asl = np.array([float(r["altitude_above_seaLevel(feet)"]) * 0.3048 for r in rows])
        lat = np.array([float(r["latitude"]) for r in rows]); lon = np.array([float(r["longitude"]) for r in rows])
    except (KeyError, ValueError):
        return np.full(n, np.nan), np.nan, np.nan
    ok = np.where((np.abs(lat) > 1) & (np.abs(hat) < 0.5))[0]
    if not len(ok):
        return np.full(n, np.nan), np.nan, np.nan
    k0 = ok[0]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x0, y0 = tr.transform([lon[k0]], [lat[k0]])
    z0 = float(ground_heights(np.array(x0, float), np.array(y0, float), None)[0])
    ad = C.load_airdata(fl)
    t0 = ad["t"][0]
    ads = C.seconds(ad["t"], t0)
    ps = C.seconds(poses["t"], t0)
    h = z0 + np.interp(ps, ads, hat) - ground
    h[(ps < ads[0] - 1) | (ps > ads[-1] + 1)] = np.nan
    return h, z0, float(asl[k0] - hat[k0])


def add_takeoff_reference(data: Path, fls) -> None:
    """Second pass: append agl_takeoff to every <id>_agl.csv (and the take-off terrain/altitude to agl/takeoff.csv)."""
    out = data / "agl"
    summary = []
    for fl in fls:
        dest = out / f"{fl.fid}_agl.csv"
        if not dest.exists():
            continue
        rows = list(csv.DictReader(open(dest)))
        if rows and "agl_takeoff" in rows[0]:
            continue
        try:
            poses = C.load_poses(fl)
            ground = np.array([float(r["ground"]) if r["ground"] != "" else np.nan for r in rows])
            h, z0, dji0 = takeoff_reference(fl, poses, ground)
        except Exception as exc:   # noqa: BLE001
            print(f"flight {fl.fid}: take-off reference failed: {exc}", flush=True)
            h, z0, dji0 = np.full(len(rows), np.nan), np.nan, np.nan
        with open(dest, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["agl_takeoff"])
            wr.writeheader()
            for r, v in zip(rows, h):
                r["agl_takeoff"] = round(float(v), 2) if np.isfinite(v) else ""
                wr.writerow(r)
        agl = np.array([float(r["agl"]) if r["agl"] != "" else np.nan for r in rows])
        d = agl - h
        summary.append((fl.fid, z0, dji0, float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan))
        print(f"flight {fl.fid}: take-off terrain {z0:.1f} m, DJI take-off altitude {dji0:.1f} m, "
              f"corrected minus take-off-referenced height: median {summary[-1][3]:+.2f} m", flush=True)
    with open(out / "takeoff.csv", "a", newline="") as f:
        csv.writer(f).writerows(summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"))
    ap.add_argument("--flights", nargs="*", default=None)
    ap.add_argument("--takeoff", action="store_true", help="add the take-off-referenced height (agl_takeoff) to existing CSVs")
    args = ap.parse_args()
    if args.takeoff:
        add_takeoff_reference(args.data, C.flights(args.data, args.flights))
        return
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
