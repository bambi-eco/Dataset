#!/usr/bin/env python3
"""
Visualize bounding boxes from custom MOT ground truth files on extracted video frames.

Usage:
    python mot_frame_viewer.py <frame_image> <mot_file> [--output <output_path>] [--show]
    python mot_frame_viewer.py <frame_image> <mot_file> --interpolate [--show]

Frame images follow the naming: <video_name>_<frame_idx>.png
The frame_idx is extracted and used to filter matching detections from the MOT file.

With --interpolate, tracks that have keyframes bracketing the target frame will have
their bounding boxes linearly interpolated (shown with dashed lines).

MOT format (CSV, no header):
    frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility,
    species, gender, age, is_propagated
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# Distinct colors per track_id (BGR)
TRACK_COLORS = [
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 0, 255),     # red
    (255, 255, 0),   # cyan
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (128, 255, 0),   # spring green
    (255, 128, 0),   # light blue
    (0, 128, 255),   # orange
    (255, 0, 128),   # purple
    (128, 0, 255),   # violet
    (0, 255, 128),   # sea green
    (200, 200, 0),   # teal
    (0, 200, 200),   # gold
    (200, 0, 200),   # pink
]

MOT_COLUMNS = [
    "frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
    "conf", "class_id", "visibility", "species", "gender", "age", "is_propagated"
]


def extract_frame_idx(frame_path: Path) -> int:
    """Extract frame index from filename like <video_name>_<frame_idx>.png"""
    stem = frame_path.stem
    match = re.search(r'_(\d+)$', stem)
    if not match:
        raise ValueError(
            f"Cannot extract frame index from '{frame_path.name}'. "
            f"Expected format: <video_name>_<frame_idx>.png"
        )
    return int(match.group(1))


def parse_mot_file(mot_path: Path) -> list[dict]:
    """Parse the custom MOT CSV file into a list of detection dicts."""
    detections = []
    with open(mot_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith('#'):
                continue
            row = [x.strip() for x in row]
            if len(row) < len(MOT_COLUMNS):
                continue
            det = {
                "frame": int(row[0]),
                "track_id": int(row[1]),
                "bb_left": int(row[2]),
                "bb_top": int(row[3]),
                "bb_width": int(row[4]),
                "bb_height": int(row[5]),
                "conf": float(row[6]),
                "class_id": int(row[7]),
                "visibility": float(row[8]),
                "species": row[9] if len(row) > 9 else "",
                "gender": int(row[10]) if len(row) > 10 and row[10] else 0,
                "age": int(row[11]) if len(row) > 11 and row[11] else 0,
                "is_propagated": int(row[12]) if len(row) > 12 and row[12] else 0,
            }
            detections.append(det)
    return detections


def build_track_index(detections: list[dict]) -> dict[int, list[dict]]:
    """Group detections by track_id, sorted by frame."""
    tracks = defaultdict(list)
    for det in detections:
        tracks[det["track_id"]].append(det)
    for tid in tracks:
        tracks[tid].sort(key=lambda d: d["frame"])
    return dict(tracks)


def interpolate_tracks(all_dets: list[dict], target_frame: int) -> list[dict]:
    """
    Interpolate bounding boxes for a target frame using linear interpolation
    between the two nearest keyframes per track.

    Returns a list of detections for the target frame. Detections that already
    exist at the exact frame are returned as-is. For frames between two
    keyframes of the same track, bb_left/top/width/height are linearly
    interpolated and the detection is marked as interpolated.

    Only tracks whose keyframe range spans the target frame are included.
    """
    tracks = build_track_index(all_dets)
    result = []

    for tid, keyframes in tracks.items():
        # Check if exact frame exists
        exact = [d for d in keyframes if d["frame"] == target_frame]
        if exact:
            for d in exact:
                result.append({**d, "_interpolated": False})
            continue

        # Find bracketing keyframes (last before, first after)
        before = [d for d in keyframes if d["frame"] < target_frame]
        after = [d for d in keyframes if d["frame"] > target_frame]

        if not before or not after:
            # Target frame is outside this track's keyframe range — skip
            continue

        kf_a = before[-1]   # closest keyframe before
        kf_b = after[0]     # closest keyframe after

        # Linear interpolation factor
        span = kf_b["frame"] - kf_a["frame"]
        t = (target_frame - kf_a["frame"]) / span

        interp_det = {
            "frame": target_frame,
            "track_id": tid,
            "bb_left": int(round(kf_a["bb_left"] + t * (kf_b["bb_left"] - kf_a["bb_left"]))),
            "bb_top": int(round(kf_a["bb_top"] + t * (kf_b["bb_top"] - kf_a["bb_top"]))),
            "bb_width": int(round(kf_a["bb_width"] + t * (kf_b["bb_width"] - kf_a["bb_width"]))),
            "bb_height": int(round(kf_a["bb_height"] + t * (kf_b["bb_height"] - kf_a["bb_height"]))),
            "conf": kf_a["conf"] + t * (kf_b["conf"] - kf_a["conf"]),
            "class_id": kf_a["class_id"],
            "visibility": kf_a["visibility"] + t * (kf_b["visibility"] - kf_a["visibility"]),
            "species": kf_a["species"],
            "gender": kf_a["gender"],
            "age": kf_a["age"],
            "is_propagated": kf_a["is_propagated"],
            "_interpolated": True,
            "_kf_a_frame": kf_a["frame"],
            "_kf_b_frame": kf_b["frame"],
        }
        result.append(interp_det)

    return result


def get_track_color(track_id: int) -> tuple:
    """Get a consistent color for a given track_id."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_length=8, gap_length=5):
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Draw each edge as a dashed line
    edges = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (sx, sy), (ex, ey) in edges:
        dist = np.hypot(ex - sx, ey - sy)
        if dist == 0:
            continue
        dx, dy = (ex - sx) / dist, (ey - sy) / dist
        pos = 0.0
        drawing = True
        while pos < dist:
            seg = dash_length if drawing else gap_length
            seg = min(seg, dist - pos)
            px1 = int(round(sx + dx * pos))
            py1 = int(round(sy + dy * pos))
            px2 = int(round(sx + dx * (pos + seg)))
            py2 = int(round(sy + dy * (pos + seg)))
            if drawing:
                cv2.line(img, (px1, py1), (px2, py2), color, thickness, cv2.LINE_AA)
            pos += seg
            drawing = not drawing


def draw_detections(image: np.ndarray, detections: list[dict],
                    show_labels: bool = True, thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes and labels on the image.
    Interpolated detections are drawn with dashed outlines."""
    vis = image.copy()

    for det in detections:
        x, y = det["bb_left"], det["bb_top"]
        w, h = det["bb_width"], det["bb_height"]
        track_id = det["track_id"]
        is_interp = det.get("_interpolated", False)
        color = get_track_color(track_id)

        # Draw bounding box — dashed for interpolated
        if is_interp:
            draw_dashed_rect(vis, (x, y), (x + w, y + h), color, thickness)
        else:
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        if show_labels:
            # Build label
            species = det["species"]
            label_parts = [f"ID:{track_id}"]
            if species and species != "Unknown":
                paren = species.find("(")
                if paren != -1:
                    short = species[paren + 1:species.find(")")].strip()
                else:
                    short = species
                label_parts.append(short)
            else:
                label_parts.append("Unknown")

            label_parts.append(f"c:{det['conf']:.2f}")

            if det.get("is_propagated"):
                label_parts.append("prop")

            if is_interp:
                label_parts.append(
                    f"interp [{det['_kf_a_frame']}-{det['_kf_b_frame']}]"
                )

            label = " | ".join(label_parts)

            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            label_y = max(y - 4, th + 4)
            # Slightly translucent feel for interpolated: darken the color
            bg_color = tuple(int(c * 0.6) for c in color) if is_interp else color
            cv2.rectangle(vis, (x, label_y - th - 4), (x + tw + 4, label_y + 2), bg_color, -1)
            cv2.putText(vis, label, (x + 2, label_y - 2), font, font_scale,
                        (255, 255, 255) if is_interp else (0, 0, 0),
                        font_thickness, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MOT bounding boxes on an extracted video frame."
    )
    parser.add_argument("frame", type=Path, help="Path to the frame image (<name>_<frame_idx>.png)")
    parser.add_argument("mot_file", type=Path, help="Path to the MOT ground truth file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: <frame>_vis.png)")
    parser.add_argument("--show", action="store_true", help="Display the result in a window")
    parser.add_argument("--no-labels", action="store_true", help="Hide labels, show boxes only")
    parser.add_argument("--thickness", type=int, default=2, help="Box line thickness (default: 2)")
    parser.add_argument("--no-interpolate", action="store_true",
                        help="Disable interpolation. By default, bounding boxes are linearly "
                             "interpolated for frames between keyframes (shown with dashed lines).")
    args = parser.parse_args()

    # Validate inputs
    if not args.frame.exists():
        print(f"Error: Frame image not found: {args.frame}", file=sys.stderr)
        sys.exit(1)
    if not args.mot_file.exists():
        print(f"Error: MOT file not found: {args.mot_file}", file=sys.stderr)
        sys.exit(1)

    # Extract frame index
    frame_idx = extract_frame_idx(args.frame)
    print(f"Frame index: {frame_idx}")

    # Load image
    image = cv2.imread(str(args.frame))
    if image is None:
        print(f"Error: Could not read image: {args.frame}", file=sys.stderr)
        sys.exit(1)

    # Parse MOT and get detections for target frame
    all_dets = parse_mot_file(args.mot_file)

    if not args.no_interpolate:
        frame_dets = interpolate_tracks(all_dets, frame_idx)
        n_exact = sum(1 for d in frame_dets if not d.get("_interpolated", False))
        n_interp = sum(1 for d in frame_dets if d.get("_interpolated", False))
        print(f"Found {n_exact} exact + {n_interp} interpolated detections for frame {frame_idx} "
              f"(total {len(all_dets)} detections, "
              f"{len(build_track_index(all_dets))} tracks in file)")
    else:
        frame_dets = [d for d in all_dets if d["frame"] == frame_idx]
        print(f"Found {len(frame_dets)} detections for frame {frame_idx} "
              f"(total {len(all_dets)} detections in file)")

    if not frame_dets:
        # Show available frames for debugging
        frames = sorted(set(d["frame"] for d in all_dets))
        if frames:
            print(f"Available frames: {frames[:20]}{'...' if len(frames) > 20 else ''}")
        print("No detections to draw. Saving original image.")

    # Draw and save
    vis = draw_detections(image, frame_dets, show_labels=not args.no_labels,
                          thickness=args.thickness)

    output_path = args.output or args.frame.with_name(args.frame.stem + "_vis.png")
    cv2.imwrite(str(output_path), vis)
    print(f"Saved: {output_path}")

    if args.show:
        cv2.imshow(f"Frame {frame_idx} - {len(frame_dets)} detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()