"""
Join the animal annotations with the environment layers, per box and per frame.

The environment layers (``<id>_environment.json`` and
``<id>_environment_nc.json``) cover exactly the key frames that carry animal
boxes, so every box can be read against the masks of its own frame with
nothing interpolated. This script does that reading once and writes three
tables that the rest of the analysis works from:

* ``boxes.csv``  -- one row per annotated box: species, sex, age, visibility,
  the box, and for every environment class the fraction of the box it covers,
  the fraction of a ring around the box, and the distance from the box centre
  to the nearest pixel of that class.
* ``frames.csv`` -- one row per key frame: coverage of every class over the
  imaged area, the release flags, canopy fragmentation, and how many animals
  of which species are in the frame.
* ``flights.csv`` -- one row per flight: recording time, drone, split, the
  estimated ground sampling distance and the letterbox extent.

Usage::

    python environment_features.py bambi_downloads/ --out features/

``bambi_downloads/`` holds one directory or one flat set of files per flight,
as ``download_from_zenodo.py --unzip`` leaves them: ``<id>_gt.txt``,
``<id>_metadata.json``, ``<id>_environment.json`` and, if the non-commercial
layer was fetched too, ``<id>_environment_nc.json``. Flights missing the
animal or the environment file are skipped with a note.

Two things to know before reading the numbers:

* The boxes are annotated on the **thermal** view and the masks are computed
  on the **RGB** view. The two are not perfectly registered: leaving thermal
  boxes unmoved gives a mean centre error of about 16 px against
  human-accepted RGB boxes (see docs/label-transfer.md). The per-box
  fractions therefore carry that much positional blur. With ``--rgb-boxes``
  the transferred ``<id>_rgb_gt.txt`` boxes are used instead where a flight
  has them, which is a useful robustness check but not ground truth either.
* The masks are undefined in the letterbox bands of the RGB frame. Boxes
  whose centre falls there get ``in_imaged_area = 0`` and NaN environment
  features; distances are measured to the nearest mask pixel and are
  right-censored by the imaged-area boundary, which ``dist_edge`` records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SAM_CLASSES = ["snow", "water", "road", "grass", "rock", "bare ground",
               "roof", "vehicle"]
NC_CLASSES = ["tree cover", "deadwood"]
ALL_CLASSES = SAM_CLASSES + NC_CLASSES

# Column key for the tables; "bare ground" -> "bare_ground".
def col(name: str) -> str:
    return name.replace(" ", "_")


# Approximate nose-to-tail body length in metres, used to estimate the ground
# sampling distance of a flight from the long side of its boxes. Humans, dogs,
# birds and the catch-all classes are left out: seen from above their extent
# is not a body length.
BODY_LENGTH_M = {
    "Sus scrofa (Wild boar)": 1.40,
    "Cervus elaphus (Red deer)": 2.00,
    "Capreolus capreolus (Roe deer)": 1.15,
    "Dama dama (Fallow deer)": 1.60,
    "Capra ibex (Alpine ibex)": 1.50,
    "Rupicapra rupicapra (Chamois)": 1.20,
    "Sus scrofa x Sus domesticus (Hybrid pig)": 1.40,
}

RING_PX = 32        # width of the ring around a box (about 1 m at 3.5 cm/px)
WINDOW_PX = 128     # half-size of the local window around the box centre


# --------------------------------------------------------------------------
# reading the released files
# --------------------------------------------------------------------------
def rle_decode(rle: dict) -> np.ndarray:
    """COCO column-major RLE -> uint8 mask. Vectorised; same result as
    environment_segmentation.rle_decode."""
    counts = np.asarray(rle["counts"], dtype=np.int64)
    if counts.size == 0:
        return np.zeros(rle["size"], np.uint8)
    values = (np.arange(counts.size) % 2).astype(np.uint8)
    return np.repeat(values, counts).reshape(rle["size"], order="F")


def read_gt(path: Path) -> dict[int, list[dict]]:
    """MOT file -> {frame: [box, ...]}. Only key frames are kept."""
    out: dict[int, list[dict]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 13:
                continue
            if int(row[12]) != 0:
                continue
            out[int(row[0])].append({
                "track_id": int(row[1]),
                "x": float(row[2]), "y": float(row[3]),
                "w": float(row[4]), "h": float(row[5]),
                "class_id": int(row[7]),
                "visibility": float(row[8]),
                "species": row[9].strip(),
                "gender": int(row[10]), "age": int(row[11]),
            })
    return out


def read_layer(path: Path) -> dict[int, dict]:
    """A published environment file -> {frame: record}."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = {int(f["frame"]): f for f in doc["frames"]}
    return {"doc": doc, "frames": frames}


def find(flight_dir: Path, flight: str, suffix: str) -> Path | None:
    for cand in (flight_dir / f"{flight}{suffix}", flight_dir / flight / f"{flight}{suffix}"):
        if cand.exists():
            return cand
    return None


def discover_flights(root: Path) -> list[str]:
    ids = set()
    for p in root.rglob("*_environment.json"):
        ids.add(p.name.split("_")[0])
    return sorted(ids, key=int)


# --------------------------------------------------------------------------
# per-flight geometry
# --------------------------------------------------------------------------
def imaged_rows(masks_by_frame: dict[int, dict[str, np.ndarray]]) -> tuple[int, int]:
    """Letterbox extent of the RGB frame, from the union of all masks.

    Every layer excludes the letterbox bands, so the first and last rows that
    any mask touches over the whole flight bound the imaged area to within a
    few pixels. Falls back to the typical 100-row bands when a flight has no
    mask at all.
    """
    rmin, rmax = 1024, -1
    for per in masks_by_frame.values():
        for m in per.values():
            rows = np.flatnonzero(m.any(axis=1))
            if rows.size:
                rmin = min(rmin, int(rows[0]))
                rmax = max(rmax, int(rows[-1]))
    if rmax < 0:
        return 100, 923
    return rmin, rmax


def estimate_gsd_cm(boxes: list[dict]) -> float | None:
    """Ground sampling distance from the long side of the boxes.

    Median over species with a known body length of
    ``body_length / long_side``. Boxes are drawn tight around the visible
    body, often with the head lowered, and an axis-aligned box around a body
    at a random heading is somewhat longer than the body, so the long side is
    taken as the body length with no further factor. Over the release this
    gives a median of about 3.3 cm/px against the 3.5 cm/px quoted in
    docs/environment.md; treat it as an estimate to within about 10%.
    """
    def collect(min_visibility: float) -> list[float]:
        ests = []
        for b in boxes:
            L = BODY_LENGTH_M.get(b["species"])
            if L is None or b["visibility"] < min_visibility:
                continue
            long_side = max(b["w"], b["h"])
            if long_side < 8:
                continue
            ests.append(100.0 * L / long_side)
        return ests

    # fully visible boxes first; a flight where every box is half occluded
    # falls back to all of them
    ests = collect(1.0)
    if len(ests) < 5:
        ests = collect(0.0)
    if len(ests) < 5:
        return None
    return float(np.median(ests))


def canopy_structure(mask: np.ndarray, rows: tuple[int, int]) -> dict:
    """How the canopy is broken up: number of blobs, edge density, mean blob size."""
    r0, r1 = rows
    area = max((r1 - r0 + 1) * mask.shape[1], 1)
    if not mask.any():
        return {"tree_blobs": 0, "tree_edge_density": 0.0, "tree_mean_blob_px": 0.0,
                "tree_largest_blob_frac": 0.0}
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    sizes = sizes[sizes >= 64]        # drop specks under ~8x8 px
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(len(c) for c in contours)
    return {
        "tree_blobs": int(sizes.size),
        "tree_edge_density": float(perim / area),
        "tree_mean_blob_px": float(sizes.mean()) if sizes.size else 0.0,
        "tree_largest_blob_frac": float(sizes.max() / max(mask.sum(), 1)) if sizes.size else 0.0,
    }


# --------------------------------------------------------------------------
# per-box features
# --------------------------------------------------------------------------
def box_features(box: dict, masks: dict[str, np.ndarray],
                 dists: dict[str, np.ndarray], rows: tuple[int, int],
                 H: int = 1024, W: int = 1024) -> dict:
    r0, r1 = rows
    x0 = int(round(box["x"])); y0 = int(round(box["y"]))
    x1 = int(round(box["x"] + box["w"])); y1 = int(round(box["y"] + box["h"]))
    cx = 0.5 * (x0 + x1); cy = 0.5 * (y0 + y1)
    cxi = min(max(int(cx), 0), W - 1); cyi = min(max(int(cy), 0), H - 1)

    out = {"cx": cx, "cy": cy}
    inside = r0 <= cy <= r1
    out["in_imaged_area"] = int(inside)
    # distance from the centre to the nearest edge of the imaged area
    out["dist_edge"] = float(min(cy - r0, r1 - cy, cx, W - 1 - cx))

    # clip the box to the imaged area
    bx0, bx1 = max(x0, 0), min(x1, W)
    by0, by1 = max(y0, r0), min(y1, r1 + 1)
    box_area = max((bx1 - bx0) * (by1 - by0), 0)
    # ring: the box dilated by RING_PX, minus the box
    rx0, rx1 = max(x0 - RING_PX, 0), min(x1 + RING_PX, W)
    ry0, ry1 = max(y0 - RING_PX, r0), min(y1 + RING_PX, r1 + 1)
    ring_area = max((rx1 - rx0) * (ry1 - ry0), 0) - box_area
    # window: WINDOW_PX around the centre
    wx0, wx1 = max(cxi - WINDOW_PX, 0), min(cxi + WINDOW_PX, W)
    wy0, wy1 = max(cyi - WINDOW_PX, r0), min(cyi + WINDOW_PX, r1 + 1)
    win_area = max((wx1 - wx0) * (wy1 - wy0), 0)

    for cls in ALL_CLASSES:
        k = col(cls)
        m = masks.get(cls)
        if not inside:
            out[f"{k}_box"] = out[f"{k}_ring"] = out[f"{k}_win"] = math.nan
            out[f"{k}_dist"] = math.nan
            continue
        if m is None:                       # class not detected in this frame
            out[f"{k}_box"] = out[f"{k}_ring"] = out[f"{k}_win"] = 0.0
            out[f"{k}_dist"] = math.inf
            continue
        in_box = int(m[by0:by1, bx0:bx1].sum()) if box_area > 0 else 0
        in_dil = int(m[ry0:ry1, rx0:rx1].sum())
        in_win = int(m[wy0:wy1, wx0:wx1].sum()) if win_area > 0 else 0
        out[f"{k}_box"] = in_box / box_area if box_area > 0 else math.nan
        out[f"{k}_ring"] = (in_dil - in_box) / ring_area if ring_area > 0 else math.nan
        out[f"{k}_win"] = in_win / win_area if win_area > 0 else math.nan
        out[f"{k}_dist"] = float(dists[cls][cyi, cxi])
    return out


# --------------------------------------------------------------------------
# one flight
# --------------------------------------------------------------------------
def process_flight(args: tuple) -> tuple[str, list[dict], list[dict], dict | None, str]:
    root, flight, rgb_boxes = args
    root = Path(root)
    gt_path = find(root, flight, "_rgb_gt.txt") if rgb_boxes else None
    if gt_path is None:
        gt_path = find(root, flight, "_gt.txt")
    env_path = find(root, flight, "_environment.json")
    nc_path = find(root, flight, "_environment_nc.json")
    meta_path = find(root, flight, "_metadata.json")
    if gt_path is None or env_path is None:
        return flight, [], [], None, "missing gt or environment file"

    gt = read_gt(gt_path)
    env = read_layer(env_path)
    nc = read_layer(nc_path) if nc_path else None
    meta = json.loads(meta_path.read_text()) if meta_path else {}
    info = meta.get("flight_info", {})

    # decode every mask once
    masks_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for fr, rec in env["frames"].items():
        masks_by_frame[fr] = {c: rle_decode(r) for c, r in rec["masks"].items()}
    if nc:
        for fr, rec in nc["frames"].items():
            masks_by_frame.setdefault(fr, {}).update(
                {c: rle_decode(r) for c, r in rec["masks"].items()})
    rows = imaged_rows(masks_by_frame)
    r0, r1 = rows
    imaged_px = (r1 - r0 + 1) * 1024

    all_boxes = [b for fr in gt.values() for b in fr]
    gsd = estimate_gsd_cm(all_boxes)
    unreliable = set(env["doc"].get("unreliable_classes", []))

    frame_rows, box_rows = [], []
    for fr in sorted(env["frames"]):
        rec = env["frames"][fr]
        masks = masks_by_frame.get(fr, {})
        boxes = gt.get(fr, [])
        # distance transforms, only for classes present
        dists = {}
        for c, m in masks.items():
            inv = (1 - m).astype(np.uint8)
            dists[c] = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

        frow = {"flight": flight, "frame": fr,
                "undetermined": int(bool(rec.get("undetermined", False))),
                "dynamic_range": rec.get("dynamic_range"),
                "n_unstable": len(rec.get("unstable", {}) or {}),
                "n_boxes": len(boxes),
                "n_species": len({b["species"] for b in boxes}),
                "n_occluded": sum(1 for b in boxes if b["visibility"] < 1.0)}
        for cls in SAM_CLASSES:
            k = col(cls)
            frow[f"{k}_cov"] = rec["coverage"].get(cls, 0.0)
            sm = (rec.get("smoothed") or {}).get(cls)
            frow[f"{k}_smooth"] = math.nan if sm is None else sm
            frow[f"{k}_unstable"] = int(cls in (rec.get("unstable") or {}))
            frow[f"{k}_unreliable"] = int(cls in unreliable)
        for cls in NC_CLASSES:
            k = col(cls)
            m = masks.get(cls)
            frow[f"{k}_cov"] = float(m[r0:r1 + 1].sum() / imaged_px) if m is not None else (
                0.0 if nc and fr in nc["frames"] else math.nan)
        frow.update(canopy_structure(masks["tree cover"], rows)
                    if "tree cover" in masks else canopy_structure(np.zeros((1024, 1024), np.uint8), rows))
        sp = Counter(b["species"] for b in boxes)
        frow["species"] = ";".join(f"{s}:{n}" for s, n in sorted(sp.items()))
        frame_rows.append(frow)

        for b in boxes:
            brow = {"flight": flight, "frame": fr, "track_id": b["track_id"],
                    "species": b["species"], "class_id": b["class_id"],
                    "gender": b["gender"], "age": b["age"],
                    "visibility": b["visibility"],
                    "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                    "n_boxes_in_frame": len(boxes),
                    "undetermined": frow["undetermined"]}
            brow.update(box_features(b, masks, dists, rows))
            box_rows.append(brow)

    flight_row = {"flight": flight,
                  "split": meta.get("split"),
                  "start_time": info.get("start_time"),
                  "end_time": info.get("end_time"),
                  "weather": info.get("weather"),
                  "drone": info.get("drone_name"),
                  "n_key_frames": len(env["frames"]),
                  "n_boxes": len(all_boxes),
                  "has_nc_layer": int(nc is not None),
                  "gsd_cm": gsd,
                  "letterbox_top": r0, "letterbox_bottom": r1,
                  "unreliable_classes": ";".join(sorted(unreliable)),
                  "boxes_source": "rgb_gt" if gt_path.name.endswith("_rgb_gt.txt") else "gt"}
    return flight, box_rows, frame_rows, flight_row, "ok"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="directory with the downloaded flights")
    ap.add_argument("--out", type=Path, default=Path("features"))
    ap.add_argument("--flights", nargs="*", help="only these flight ids")
    ap.add_argument("--rgb-boxes", action="store_true",
                    help="use the owl-transferred RGB boxes where a flight has them")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    flights = args.flights or discover_flights(args.root)
    if not flights:
        sys.exit(f"no *_environment.json under {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)

    boxes, frames, flights_out = [], [], []
    jobs = [(str(args.root), f, args.rgb_boxes) for f in flights]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (flight, b, fr, fl, status) in enumerate(ex.map(process_flight, jobs)):
            if status != "ok":
                print(f"  {flight}: {status}", flush=True)
                continue
            boxes.extend(b); frames.extend(fr); flights_out.append(fl)
            if i % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} flights, {len(boxes)} boxes", flush=True)

    write_csv(args.out / "boxes.csv", boxes)
    write_csv(args.out / "frames.csv", frames)
    write_csv(args.out / "flights.csv", flights_out)
    print(f"{len(flights_out)} flights, {len(frames)} frames, {len(boxes)} boxes -> {args.out}")


if __name__ == "__main__":
    main()
