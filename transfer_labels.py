#!/usr/bin/env python3
"""
Transfer thermal MOT annotations to the RGB view using OWL point detections.

RGB and thermal are recorded simultaneously but are not perfectly aligned, so a
bounding box annotated on a thermal frame does not sit on the animal in the
corresponding RGB frame. Measured over the 247,382 boxes of the `matched`
subset that carry both a thermal and an accepted RGB annotation, the
discrepancy is a pure 2-D translation:

  * bb_width and bb_height are identical in the two views for EVERY box,
  * the centre shift has a median magnitude of 10.3 px (p90 34, p99 54, max
    116) on 1024x1024 frames,
  * per-flight mean shifts range from -37 to +20 px in x and -20 to +27 px in
    y, and the shift also drifts WITHIN a flight (sd up to 22 px).

So transferring a label means estimating that translation. This script does it
by locating the animals in the RGB frame with OWL (Overhead Wildlife Locator,
https://github.com/microsoft/MegaDetector-Overhead), a point detector for
aerial wildlife imagery, and re-centring each thermal box on its matched point.

Four stages:

  1. MATCH      Per frame, thermal box centres are assigned to OWL points by
                rectangular Hungarian assignment on Euclidean distance, with a
                hard gate (--gate) so an isolated point cannot capture a box on
                the far side of the image. Each accepted assignment yields a
                raw shift, point - centre.
  2. CONSENSUS  Per frame, the median of that frame's raw shifts; then a
                rolling median of those over time within the flight
                (--cons-window). Misalignment is largely a whole-frame effect
                that drifts slowly, so this curve is far steadier than any
                single detection, and it is defined on frames where a
                particular animal was never detected.
  3. REFINE     The assignment is re-run with the consensus as a prior: costs
                are measured from the consensus-shifted centre, under a much
                tighter gate (--gate2). In a frame holding several animals the
                first pass can hand a box its neighbour's point; the prior
                breaks the tie.
  4. SMOOTH     Per track, a rolling median of the shift over time (--smooth).
                The true shift drifts slowly while per-frame detector noise
                does not. Boxes with no surviving match take the consensus, or
                are left untouched if the flight has no matches at all.

Every output row records, in a sidecar CSV, which stage produced it, so rows
that the detector never saw can be reviewed separately.

Input MOT format (CSV, no header) as documented in the repository README:

  frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id,
  visibility, species, gender, age, is_propagated

Only bb_left and bb_top are modified; every other column, however many there
are, is written back verbatim. The 10-column files of the `matched` subset are
handled the same way.

Detection CSV format, one row per detected animal:

  frame, x, y[, score][, sequence]

`x` and `y` are the point in IMAGE pixels. A raw `detections.csv` from
MegaDetector-Overhead's tools/test.py is also accepted directly (columns
images, labels, dscores, x, y, ...), but note that it stores points in HEATMAP
coordinates -- the shipped configs use down_ratio 2 with up=False, so a point
in a 1024x1024 frame comes back in 0..511. Pass --detection-scale 2 for those.

Examples
--------
  # single flight, detections already in image pixels
  python transfer_labels.py 119_thermal_mot.txt \\
      --detections owl_detections_flight119.csv \\
      --output 119_rgb_mot.txt

  # score the result against the accepted RGB annotations of `matched`
  python transfer_labels.py 119_thermal_mot.txt \\
      --detections owl_detections_flight119.csv \\
      --output 119_rgb_mot.txt --reference 119_accepted_rgb_mot.txt

  # a folder of flights, raw OWL output in heatmap coordinates
  python transfer_labels.py ./mot --detections ./detections \\
      --output ./mot_rgb --detection-scale 2
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


# Column positions shared by the 13-column dataset format and the 10-column
# `matched` format. Anything beyond these six is carried through untouched.
COL_FRAME, COL_TRACK, COL_LEFT, COL_TOP, COL_WIDTH, COL_HEIGHT = range(6)

SOURCES = ("owl", "smoothed", "consensus", "unchanged")


# --------------------------------------------------------------------------- #
#  I/O
# --------------------------------------------------------------------------- #

def read_mot(path: Path) -> list[dict]:
    """Read a MOT file, keeping every original field for round-tripping."""
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.reader(fh):
            if not raw or raw[0].strip().startswith("#"):
                continue
            if len(raw) < 6:
                raise ValueError(f"{path}: expected >=6 columns, got {len(raw)}")
            rows.append({
                "frame": int(float(raw[COL_FRAME])),
                "track_id": int(float(raw[COL_TRACK])),
                "left": float(raw[COL_LEFT]),
                "top": float(raw[COL_TOP]),
                "width": float(raw[COL_WIDTH]),
                "height": float(raw[COL_HEIGHT]),
                "raw": [f.strip() for f in raw],
            })
    return rows


def write_mot(path: Path, rows: list[dict], integer: bool = True) -> None:
    """Write rows back out, replacing only bb_left and bb_top."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for r in sorted(rows, key=lambda r: (r["frame"], r["track_id"])):
            out = list(r["raw"])
            left, top = r["left"], r["top"]
            out[COL_LEFT] = str(round(left)) if integer else f"{left:g}"
            out[COL_TOP] = str(round(top)) if integer else f"{top:g}"
            w.writerow(out)


def read_detections(path: Path, scale: float = 1.0,
                    sequence: str | None = None) -> dict[int, np.ndarray]:
    """Read a detection CSV into {frame: (n, 3) array of x, y, score}.

    Accepts both the compact `frame,x,y,score` layout and a raw
    MegaDetector-Overhead `detections.csv`, which lists a frame it found nothing
    in as a single row with an empty `x`. Such frames are kept as empty entries
    so that "scored, found nothing" stays distinguishable from "never
    processed" when reporting coverage; both fall back to the consensus curve.
    """
    per_frame: dict[int, list] = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        if "x" not in cols or "y" not in cols:
            raise ValueError(f"{path}: needs at least 'x' and 'y' columns, "
                             f"found {reader.fieldnames}")
        c_score = cols.get("score") or cols.get("dscores")
        c_frame = cols.get("frame")
        c_image = cols.get("image") or cols.get("images")
        c_seq = cols.get("sequence") or cols.get("seq")
        if not c_frame and not c_image:
            raise ValueError(f"{path}: needs a 'frame' or 'image' column")

        for row in reader:
            if c_seq and sequence is not None and row[c_seq] != sequence:
                continue
            if c_frame:
                frame = int(float(row[c_frame]))
            else:
                # `<anything>_<frame>.<ext>` -- the naming that both
                # frame_extraction.py and the OWL staging folders produce.
                stem = row[c_image].rsplit(".", 1)[0]
                head, _, tail = stem.rpartition("_")
                if sequence is not None and head and head != sequence:
                    continue
                frame = int(tail)
            per_frame.setdefault(frame, [])
            val = row[cols["x"]]
            if val is None or val == "" or val.lower() == "nan":
                continue                      # frame scored, nothing detected
            score = float(row[c_score]) if c_score and row[c_score] else 1.0
            per_frame[frame].append(
                (float(val) * scale, float(row[cols["y"]]) * scale, score))

    return {f: np.asarray(v, dtype=float).reshape(-1, 3)
            for f, v in per_frame.items()}


# --------------------------------------------------------------------------- #
#  Estimation
# --------------------------------------------------------------------------- #

def rolling_median(frames: np.ndarray, values: np.ndarray,
                   at: np.ndarray, half_window: float) -> np.ndarray:
    """Median of `values` whose frame lies within +/-half_window of each `at`.

    Indexed by frame rather than by sample, because annotations may be sparse
    or strided and what matters is temporal locality, not how many samples
    happen to exist. Falls back to the global median where a window is empty.
    """
    if len(frames) == 0:
        return np.zeros(len(at))
    order = np.argsort(frames)
    frames, values = frames[order], values[order]
    lo = np.searchsorted(frames, at - half_window, side="left")
    hi = np.searchsorted(frames, at + half_window, side="right")
    fallback = float(np.median(values))
    out = np.empty(len(at))
    for i, (a, b) in enumerate(zip(lo, hi)):
        out[i] = np.median(values[a:b]) if b > a else fallback
    return out


class ConsensusCurve:
    """Temporally smoothed whole-frame shift, built from the matched boxes."""

    def __init__(self, shifts: dict, half_window: float, min_support: int = 0):
        self.half_window = half_window
        self.min_support = min_support
        per_frame: dict[int, list] = defaultdict(list)
        for (frame, _track), (dx, dy, _d, _s) in shifts.items():
            per_frame[frame].append((dx, dy))
        pts = [(f, float(np.median([v[0] for v in v_])),
                float(np.median([v[1] for v in v_])))
               for f, v_ in per_frame.items()]
        self.points = np.asarray(sorted(pts)).reshape(-1, 3)
        self._cache: dict[int, np.ndarray | None] = {}

    def __len__(self) -> int:
        return len(self.points)

    def at(self, frame: int) -> np.ndarray | None:
        """The consensus shift at `frame`, or None if too little supports it."""
        if len(self.points) == 0:
            return None
        if frame in self._cache:
            return self._cache[frame]
        f = self.points[:, 0]
        support = int(np.searchsorted(f, frame + self.half_window, "right")
                      - np.searchsorted(f, frame - self.half_window, "left"))
        if support < self.min_support:
            value = None
        else:
            at = np.array([float(frame)])
            value = np.array([
                rolling_median(f, self.points[:, 1], at, self.half_window)[0],
                rolling_median(f, self.points[:, 2], at, self.half_window)[0]])
        self._cache[frame] = value
        return value


def match_frames(rows: list[dict], detections: dict[int, np.ndarray],
                 gate: float, prior: ConsensusCurve | None = None) -> dict:
    """Hungarian assignment per frame -> {(frame, track_id): (dx, dy, dist, score)}.

    With a `prior`, distances are measured from the shifted centre, so the gate
    sits around the predicted position instead of the unaligned one and can be
    much tighter.
    """
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)

    out = {}
    for frame, boxes in by_frame.items():
        points = detections.get(frame)
        if points is None or len(points) == 0:
            continue
        centres = np.array([[r["left"] + r["width"] / 2,
                             r["top"] + r["height"] / 2] for r in boxes])
        offset = np.zeros(2)
        if prior is not None:
            p = prior.at(frame)
            if p is not None:
                offset = p
        cost = np.linalg.norm((centres + offset)[:, None, :]
                              - points[None, :, :2], axis=2)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] > gate:
                continue
            out[(frame, boxes[i]["track_id"])] = (
                points[j, 0] - centres[i, 0],
                points[j, 1] - centres[i, 1],
                float(cost[i, j]), float(points[j, 2]))
    return out


def smooth_per_track(rows: list[dict], shifts: dict,
                     half_window: float) -> dict:
    """Rolling median of each track's shift -> {(frame, track): (dx, dy)}."""
    frames_of: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        frames_of[r["track_id"]].append(r["frame"])

    out = {}
    for track, frames in frames_of.items():
        have = np.array(sorted({f for f in frames if (f, track) in shifts}))
        if len(have) == 0:
            continue
        dx = np.array([shifts[(f, track)][0] for f in have])
        dy = np.array([shifts[(f, track)][1] for f in have])
        mx = rolling_median(have, dx, have, half_window)
        my = rolling_median(have, dy, have, half_window)
        for f, a, b in zip(have, mx, my):
            out[(int(f), track)] = (float(a), float(b))
    return out


def estimate_shifts(rows: list[dict], detections: dict[int, np.ndarray],
                    gate: float = 60.0, gate2: float = 45.0,
                    cons_window: float = 150.0, smooth: float = 15.0,
                    min_support: int = 0, refine: bool = True) -> dict:
    """Run the four stages. Returns {(frame, track_id): (dx, dy, source, info)}."""
    raw = match_frames(rows, detections, gate)
    consensus = ConsensusCurve(raw, cons_window, min_support)
    if refine and len(consensus):
        raw = match_frames(rows, detections, gate2, prior=consensus)
        consensus = ConsensusCurve(raw, cons_window, min_support)

    smoothed = smooth_per_track(rows, raw, smooth) if smooth > 0 else {}

    result = {}
    for r in rows:
        key = (r["frame"], r["track_id"])
        if key in smoothed:
            dx, dy = smoothed[key]
            source = "smoothed"
        elif key in raw:
            dx, dy = raw[key][0], raw[key][1]
            source = "owl"
        else:
            c = consensus.at(r["frame"])
            if c is None:
                dx = dy = 0.0
                source = "unchanged"
            else:
                dx, dy = float(c[0]), float(c[1])
                source = "consensus"
        info = raw.get(key)
        result[key] = (dx, dy, source,
                       {"dist": info[2], "score": info[3]} if info else {})
    return result


def apply_shifts(rows: list[dict], shifts: dict) -> list[dict]:
    """Return new rows with bb_left/bb_top moved by the estimated shift."""
    out = []
    for r in rows:
        dx, dy, source, info = shifts[(r["frame"], r["track_id"])]
        moved = dict(r)
        moved["left"] = r["left"] + dx
        moved["top"] = r["top"] + dy
        moved["shift"] = (dx, dy)
        moved["source"] = source
        moved["info"] = info
        out.append(moved)
    return out


def write_provenance(path: Path, rows: list[dict]) -> None:
    """Sidecar CSV: where each box's shift came from, and how confident it is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "track_id", "source", "dx", "dy",
                    "match_distance", "detection_score"])
        for r in sorted(rows, key=lambda r: (r["frame"], r["track_id"])):
            info = r.get("info") or {}
            w.writerow([r["frame"], r["track_id"], r["source"],
                        round(r["shift"][0], 2), round(r["shift"][1], 2),
                        round(info["dist"], 2) if "dist" in info else "",
                        round(info["score"], 4) if "score" in info else ""])


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def _centres(rows: list[dict]) -> dict:
    return {(r["frame"], r["track_id"]):
            (r["left"] + r["width"] / 2, r["top"] + r["height"] / 2)
            for r in rows}


def _iou_same_size(dx: float, dy: float, w: float, h: float) -> float:
    """IoU of two equally sized axis-aligned boxes offset by (dx, dy)."""
    inter = max(w - abs(dx), 0.0) * max(h - abs(dy), 0.0)
    return inter / (2 * w * h - inter) if inter else 0.0


def evaluate(predicted: list[dict], reference: list[dict],
             thermal: list[dict]) -> dict:
    """Score predictions against accepted RGB annotations.

    The comparison is per TRACK: a predicted box is scored against the
    reference box carrying the same track_id, never against whichever box
    happens to be nearest, so a label that lands on the neighbouring animal
    counts as a large error rather than a hit.

    `identity_ok` additionally reports, for frames holding at least two
    annotated animals, how often the nearest reference box to a prediction is
    its own -- the part of the error that is confusion between animals rather
    than imprecision. It can only see animals that were annotated, so it is a
    lower bound.
    """
    ref = {(r["frame"], r["track_id"]): r for r in reference}
    thm = _centres(thermal)
    pc = _centres(predicted)

    keys = [k for k in pc if k in ref and k in thm]
    if not keys:
        return {"n": 0}

    err_pred, err_id, iou_pred, iou_id = [], [], [], []
    for k in keys:
        r = ref[k]
        gx, gy = r["left"] + r["width"] / 2, r["top"] + r["height"] / 2
        w, h = r["width"], r["height"]
        px, py = pc[k]
        tx, ty = thm[k]
        err_pred.append(math.hypot(px - gx, py - gy))
        err_id.append(math.hypot(tx - gx, ty - gy))
        iou_pred.append(_iou_same_size(px - gx, py - gy, w, h))
        iou_id.append(_iou_same_size(tx - gx, ty - gy, w, h))

    e_p, e_i = np.array(err_pred), np.array(err_id)
    i_p, i_i = np.array(iou_pred), np.array(iou_id)

    # Identity: is the nearest reference box in the frame the box's own?
    by_frame: dict[int, list] = defaultdict(list)
    for k in keys:
        by_frame[k[0]].append(k)
    id_ok = id_tot = 0
    for frame, ks in by_frame.items():
        if len(ks) < 2:
            continue
        g = np.array([[ref[k]["left"] + ref[k]["width"] / 2,
                       ref[k]["top"] + ref[k]["height"] / 2] for k in ks])
        p = np.array([pc[k] for k in ks])
        nearest = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2).argmin(1)
        id_ok += int((nearest == np.arange(len(ks))).sum())
        id_tot += len(ks)

    def stats(err, iou):
        return {
            "mean_error_px": float(err.mean()),
            "median_error_px": float(np.median(err)),
            "p90_error_px": float(np.percentile(err, 90)),
            "mean_iou": float(iou.mean()),
            "iou_gt_50": float((iou > 0.5).mean()),
            "iou_gt_75": float((iou > 0.75).mean()),
        }

    counts: dict[str, int] = defaultdict(int)
    for r in predicted:
        counts[r.get("source", "?")] += 1

    return {
        "n": len(keys),
        "transferred": stats(e_p, i_p),
        "unchanged_baseline": stats(e_i, i_i),
        "improved_fraction": float((e_p < e_i - 1e-9).mean()),
        "worsened_fraction": float((e_p > e_i + 1e-9).mean()),
        "identity_ok": (id_ok / id_tot) if id_tot else None,
        "identity_n": id_tot,
        "sources": dict(counts),
    }


def format_report(name: str, m: dict) -> str:
    if not m.get("n"):
        return f"{name}: nothing to score (no shared frame/track_id)"
    t, b = m["transferred"], m["unchanged_baseline"]
    lines = [
        f"{name}: {m['n']} boxes with an accepted RGB reference",
        f"  {'':<22}{'mean':>8}{'median':>8}{'p90':>8}{'IoU':>8}"
        f"{'IoU>.5':>9}{'IoU>.75':>9}",
    ]
    for label, s in (("unchanged (thermal)", b), ("transferred", t)):
        lines.append(
            f"  {label:<22}{s['mean_error_px']:8.2f}{s['median_error_px']:8.2f}"
            f"{s['p90_error_px']:8.2f}{s['mean_iou']:8.3f}"
            f"{100 * s['iou_gt_50']:8.1f}%{100 * s['iou_gt_75']:8.1f}%")
    lines.append(f"  closer to the reference than leaving it alone: "
                 f"{100 * m['improved_fraction']:.1f}%  "
                 f"(worse: {100 * m['worsened_fraction']:.1f}%)")
    if m["identity_ok"] is not None:
        lines.append(f"  on the right animal, in multi-animal frames : "
                     f"{100 * m['identity_ok']:.2f}%  "
                     f"({m['identity_n']} boxes)")
    lines.append("  shift source: " + ", ".join(
        f"{k} {v}" for k, v in sorted(m["sources"].items(),
                                      key=lambda kv: -kv[1])))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def sequence_of(path: Path) -> str:
    """Sequence id = everything before the first underscore, as elsewhere."""
    return path.stem.split("_", 1)[0]


def find_for_sequence(seq: str, folder: Path, suffix: str) -> Path | None:
    """Locate the file in `folder` belonging to sequence `seq`.

    Matches on MAXIMAL runs of digits in the filename, so `119` is found in
    `119.txt`, `119_accepted_rgb_mot.txt`, `detections_119.csv` and
    `owl_detections_flight119.csv` alike, while flight 19 is not matched by
    `flight119` -- the run there is "119", not "19".
    """
    hits = [c for c in sorted(folder.glob(f"*{suffix}"))
            if seq in re.findall(r"\d+", c.stem)]
    if len(hits) > 1:
        print(f"  warning: {len(hits)} files in {folder} match sequence "
              f"{seq}, using {hits[0].name}")
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transfer thermal MOT annotations to the RGB view using "
                    "OWL point detections.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path,
                    help="Thermal MOT .txt file, or a folder of them.")
    ap.add_argument("-d", "--detections", type=Path, required=True,
                    help="Detection CSV, or a folder of them (matched to the "
                         "MOT files by sequence id).")
    ap.add_argument("-o", "--output", type=Path, required=True,
                    help="Output MOT file, or a folder in folder mode.")
    ap.add_argument("--reference", type=Path,
                    help="Accepted RGB MOT file (or folder) to score against.")
    ap.add_argument("--detection-scale", type=float, default=1.0,
                    help="Multiply detection x/y by this. Use 2 for a raw "
                         "MegaDetector-Overhead detections.csv, which stores "
                         "heatmap coordinates (down_ratio 2, up=False). "
                         "Default: 1.")
    ap.add_argument("--gate", type=float, default=60.0,
                    help="Max centre-to-point distance for a first-pass "
                         "assignment, px. Default: 60, just above the p99 of "
                         "the observed shift magnitude.")
    ap.add_argument("--gate2", type=float, default=45.0,
                    help="Gate for the consensus-primed second pass, px. "
                         "Default: 45.")
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip the consensus-primed second pass.")
    ap.add_argument("--cons-window", type=float, default=150.0,
                    help="Half-window, in frames, of the consensus curve. "
                         "Default: 150.")
    ap.add_argument("--smooth", type=float, default=15.0,
                    help="Half-window, in frames, of the per-track rolling "
                         "median. 0 disables it. Default: 15.")
    ap.add_argument("--min-support", type=int, default=0,
                    help="Matched frames required inside the consensus window "
                         "before the curve is trusted; below it a box is left "
                         "unchanged. Default: 0 (always trust it).")
    ap.add_argument("--float-coords", action="store_true",
                    help="Keep sub-pixel coordinates instead of rounding to "
                         "integers as the released annotations do.")
    ap.add_argument("--report", type=Path,
                    help="Write the evaluation metrics to this JSON file.")
    args = ap.parse_args()

    if args.input.is_dir():
        mot_files = sorted(args.input.glob("*.txt"))
        if not mot_files:
            print(f"No .txt files in {args.input}")
            return 1
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        mot_files = [args.input]

    metrics_all = {}
    for mot_path in mot_files:
        seq = sequence_of(mot_path)

        det_path = args.detections
        if det_path.is_dir():
            det_path = find_for_sequence(seq, det_path, ".csv")
            if det_path is None:
                print(f"  {mot_path.name}: no detection CSV for sequence {seq}")
                continue

        rows = read_mot(mot_path)
        detections = read_detections(det_path, args.detection_scale, seq)
        shifts = estimate_shifts(
            rows, detections,
            gate=args.gate, gate2=args.gate2, cons_window=args.cons_window,
            smooth=args.smooth, min_support=args.min_support,
            refine=not args.no_refine)
        moved = apply_shifts(rows, shifts)

        out_path = (args.output / f"{seq}_rgb_mot.txt"
                    if args.input.is_dir() else args.output)
        write_mot(out_path, moved, integer=not args.float_coords)
        write_provenance(out_path.with_name(out_path.stem + "_provenance.csv"),
                         moved)

        scored = sum(1 for f in detections)
        print(f"{mot_path.name}: {len(rows)} boxes, {scored} frames scored, "
              f"-> {out_path}")

        ref_path = args.reference
        if ref_path is not None and ref_path.is_dir():
            ref_path = find_for_sequence(seq, ref_path, ".txt")
        if ref_path is not None and ref_path.exists():
            m = evaluate(moved, read_mot(ref_path), rows)
            metrics_all[seq] = m
            print(format_report(f"  flight {seq}", m))

    if args.report and metrics_all:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metrics_all, indent=2))
        print(f"\nMetrics written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
