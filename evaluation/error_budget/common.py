"""
Shared loaders for the geometric error budget of the BAMBI projections.

Data layout (what `fetch_all.py` / `download_from_zenodo.py --annotations-only`
produce):

    <data>/base/<id>/<id>_matched_poses.json, <id>_gt.txt, <id>_metadata.json,
                     <id>_correction.json, <id>_mask_t.png, <id>_mask_w.png
    <data>/raw/<id>/DJI_*_T_*.SRT, DJI_*_V_*.SRT, air_data.csv, T_calib.json, W_calib.json

All timestamps are naive local time (Europe/Vienna), as in the SRT files.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import srt_airdata_sync_eval as se   # noqa: E402

W = H = 1024                 # released frame size
FPS = 30.0
MATCH_RADIUS = 1.37          # the paper's point-matching radius on the ground, metres
ALT_BINS = [(0, 30), (30, 42), (42, 52), (52, 62), (62, 1000)]   # the paper's altitude bins
ALT_LABELS = ["< 30", "30-42", "42-52", "52-62", ">= 62"]


@dataclass
class Flight:
    fid: str
    base: Path
    raw: Optional[Path]

    def path(self, suffix: str) -> Path:
        return self.base / f"{self.fid}_{suffix}"

    @property
    def has_raw(self) -> bool:
        return self.raw is not None and (self.raw / "air_data.csv").exists()


def flights(data: Path, only: Optional[list[str]] = None) -> list[Flight]:
    out = []
    for d in sorted((data / "base").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10 ** 6):
        fid = d.name
        if only and fid not in only:
            continue
        if not (d / f"{fid}_matched_poses.json").exists() or not (d / f"{fid}_gt.txt").exists():
            continue
        raw = data / "raw" / fid
        out.append(Flight(fid, d, raw if raw.exists() else None))
    return out


# ----------------------------------------------------------------------------- base files

def load_poses(fl: Flight) -> dict:
    """Published poses: arrays t (naive local datetimes), lat, lng, alt, tilt (deg from nadir), heading."""
    j = json.load(open(fl.path("matched_poses.json")))
    img = j["images"]
    t = np.array([dt.datetime.fromisoformat(e["timestamp"]).replace(tzinfo=None) for e in img])
    origin_alt = float((j.get("origin") or {}).get("altitude", 0.0))
    if img and "alt" in img[0]:                          # lat/lng/alt/pitch/yaw form
        pitch = np.array([float(e.get("pitch", 0.0)) for e in img])
        tilt = np.abs(((pitch + 180) % 360) - 180)        # 360.1 -> 0.1, 359.5 -> 0.5
        alt = np.array([float(e["alt"]) for e in img])
        heading = np.array([float(e.get("yaw", 0.0)) for e in img])
        fovy = np.full(len(img), np.nan)
    else:                                                # location/rotation/fovy form (alfs_py poses)
        loc = np.array([e["location"] for e in img], dtype=float)
        rot = np.array([e.get("rotation", [0, 0, 0]) for e in img], dtype=float)
        alt = origin_alt + loc[:, 2]
        tilt = np.abs(((rot[:, 0] + 180) % 360) - 180)
        heading = rot[:, 2]
        fovy = np.array([float(np.ravel(e.get("fovy", [np.nan]))[0]) for e in img])
    return dict(t=t, lat=np.array([float(e["lat"]) for e in img]), lng=np.array([float(e["lng"]) for e in img]),
                alt=alt, tilt=tilt, heading=heading, fovy=fovy, drone=j.get("drone", ""), n=len(img))


def load_gt(fl: Flight) -> np.ndarray:
    """Annotated boxes as a structured array: frame, tid, l, t, w, h, species."""
    rows = []
    for line in open(fl.path("gt.txt"), encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split(",")
        if len(f) < 6:
            continue
        try:
            rows.append((int(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]),
                         f[9].strip() if len(f) > 9 else ""))
        except ValueError:
            continue
    return np.array(rows, dtype=[("frame", "i4"), ("tid", "i4"), ("l", "f4"), ("t", "f4"), ("w", "f4"), ("h", "f4"), ("species", "U48")])


def load_metadata(fl: Flight) -> dict:
    p = fl.path("metadata.json")
    return json.load(open(p)) if p.exists() else {}


def load_mask(fl: Flight, cam: str) -> Optional[np.ndarray]:
    import cv2
    p = fl.path(f"mask_{cam.lower()}.png")
    if not p.exists():
        return None
    m = cv2.imread(str(p), 0)
    return m > 127


# ----------------------------------------------------------------------------- raw files

def load_calib(fl: Flight, cam: str) -> Optional[dict]:
    """cam 'T' or 'W': {'mtx': (3,3), 'dist': (5,)} or None."""
    if fl.raw is None:
        return None
    p = fl.raw / f"{cam.upper()}_calib.json"
    if not p.exists():
        return None
    c = json.load(open(p))
    return dict(mtx=np.asarray(c["mtx"], dtype=np.float64), dist=np.asarray(c["dist"], dtype=np.float64).reshape(-1))


def load_airdata(fl: Flight) -> Optional[dict]:
    if not fl.has_raw:
        return None
    rows = [{(k or "").strip(): v for k, v in r.items()} for r in csv.DictReader(open(fl.raw / "air_data.csv", encoding="utf-8"))]
    if not rows:
        return None
    try:
        ad = se.parse_airdata(fl.raw / "air_data.csv")
    except KeyError:                       # older AirData exports lack some columns: parse what is there
        try:
            t_ms = np.array([float(r["time(millisecond)"]) for r in rows])
            d0 = dt.datetime.strptime(rows[0]["datetime(utc)"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
            d0 = d0.astimezone(se.VIENNA).replace(tzinfo=None)
        except (KeyError, ValueError):
            return None

        def f(k):
            try:
                return np.array([float(r[k]) for r in rows])
            except (KeyError, ValueError):
                return np.full(len(rows), np.nan)
        ad = dict(t=np.array([d0 + dt.timedelta(milliseconds=m - t_ms[0]) for m in t_ms]), t_ms=t_ms, lat=f("latitude"), lon=f("longitude"),
                  alt=f("altitude_above_seaLevel(feet)") * 0.3048, heading=f("compass_heading(degrees)"), gimbal=f("gimbal_heading(degrees)"),
                  is_video=np.array([r.get("isVideo", "0") == "1" for r in rows]), speed_mph=f("speed(mph)"))
        if np.isnan(ad["lat"]).all():
            return None

    def col(name):
        try:
            return np.array([float(r[name]) if r.get(name, "") not in ("", None) else np.nan for r in rows])
        except (KeyError, ValueError):
            return np.full(len(rows), np.nan)
    ad["agl"] = col("height_above_ground_at_drone_location(feet)") * 0.3048
    ad["above_takeoff"] = col("height_above_takeoff(feet)") * 0.3048
    return ad


def load_srt(fl: Flight, cam: str = "T") -> list[dict]:
    """One parsed SRT per recording, sorted by start time, each with 'name'."""
    if fl.raw is None:
        return []
    files = sorted(p for p in fl.raw.iterdir() if p.suffix.upper() == ".SRT" and re.search(rf"_\d{{4}}_{cam}(\.|_)", p.name))
    parts = []
    for p in files:
        s = se.parse_srt(p)
        if len(s["t"]):
            s["name"] = p.name
            parts.append(s)
    return sorted(parts, key=lambda s: s["t"][0])


def seconds(times, t0):
    return se.seconds(times, t0)


def nearest_index(sorted_s: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    """Index of the nearest value in a sorted array for every query."""
    j = np.searchsorted(sorted_s, query_s).clip(1, len(sorted_s) - 1)
    return np.where(np.abs(sorted_s[j - 1] - query_s) < np.abs(sorted_s[j] - query_s), j - 1, j)


def alt_bin(agl: np.ndarray) -> np.ndarray:
    """Index of the paper's altitude bin for every value (nan -> -1)."""
    out = np.full(len(agl), -1, dtype=int)
    for i, (lo, hi) in enumerate(ALT_BINS):
        out[(agl >= lo) & (agl < hi)] = i
    return out


def pct(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")
