#!/usr/bin/env python3
"""
Visualize custom MOT ground truth / tracking results on video.

Usage:
    python visualize_mot.py video.mp4 annotations.txt -o output.mp4
    python visualize_mot.py video.mp4 annotations.txt -o output.mp4 --interpolate
    python visualize_mot.py video.mp4 annotations.txt --show  # live preview only
"""

import argparse
import csv
import colorsys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Column definition ───────────────────────────────────────────────────────
COLUMNS = [
    "frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
    "conf", "class_id", "visibility", "species", "gender", "age", "is_propagated",
]

INT_COLS = {"frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
            "class_id", "gender", "age", "is_propagated"}
FLOAT_COLS = {"conf", "visibility"}


def parse_mot_file(path: str) -> list[dict]:
    """Parse the custom MOT CSV into a list of detection dicts."""
    detections = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            det = {}
            for i, col in enumerate(COLUMNS):
                val = row[i].strip() if i < len(row) else ""
                if col in INT_COLS:
                    det[col] = int(val) if val else 0
                elif col in FLOAT_COLS:
                    det[col] = float(val) if val else 0.0
                else:
                    det[col] = val
            detections.append(det)
    return detections


def build_tracks(detections: list[dict]) -> dict[int, list[dict]]:
    """Group detections by track_id, sorted by frame."""
    tracks: dict[int, list[dict]] = defaultdict(list)
    for det in detections:
        tracks[det["track_id"]].append(det)
    for tid in tracks:
        tracks[tid].sort(key=lambda d: d["frame"])
    return dict(tracks)


def interpolate_tracks(tracks: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """Linearly interpolate missing frames within each track."""
    interpolated = {}
    for tid, dets in tracks.items():
        if len(dets) < 2:
            interpolated[tid] = list(dets)
            continue

        new_dets = []
        for i in range(len(dets) - 1):
            d0 = dets[i]
            d1 = dets[i + 1]
            new_dets.append(d0)

            f0, f1 = d0["frame"], d1["frame"]
            gap = f1 - f0
            if gap <= 1:
                continue

            # Linearly interpolate bbox for each missing frame
            for f in range(f0 + 1, f1):
                t = (f - f0) / gap
                interp = {
                    "frame": f,
                    "track_id": tid,
                    "bb_left": int(round(d0["bb_left"] + t * (d1["bb_left"] - d0["bb_left"]))),
                    "bb_top": int(round(d0["bb_top"] + t * (d1["bb_top"] - d0["bb_top"]))),
                    "bb_width": int(round(d0["bb_width"] + t * (d1["bb_width"] - d0["bb_width"]))),
                    "bb_height": int(round(d0["bb_height"] + t * (d1["bb_height"] - d0["bb_height"]))),
                    "conf": d0["conf"] + t * (d1["conf"] - d0["conf"]),
                    "visibility": d0["visibility"] + t * (d1["visibility"] - d0["visibility"]),
                    "class_id": d0["class_id"],
                    "species": d0["species"],
                    "gender": d0["gender"],
                    "age": d0["age"],
                    "is_propagated": 1,  # mark interpolated as propagated
                    "_interpolated": True,
                }
                new_dets.append(interp)

        new_dets.append(dets[-1])
        interpolated[tid] = new_dets
    return interpolated


def index_by_frame(tracks: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """Re-index all detections by frame number for fast lookup."""
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for dets in tracks.values():
        for det in dets:
            by_frame[det["frame"]].append(det)
    return dict(by_frame)


# ── Color generation ────────────────────────────────────────────────────────

def track_color(track_id: int) -> tuple[int, int, int]:
    """Generate a distinct BGR color for a track id using golden-ratio hue spacing."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


SPECIES_COLORS = {
    "Homo sapiens (Human)": (0, 200, 255),       # orange-yellow
    "Capreolus capreolus (Roe deer)": (0, 255, 0),  # green
    "Unknown": (255, 180, 0),                     # blue-cyan
}


def get_color(det: dict, color_mode: str) -> tuple[int, int, int]:
    if color_mode == "species":
        return SPECIES_COLORS.get(det["species"], (200, 200, 200))
    else:  # "track"
        return track_color(det["track_id"])


# ── Drawing ─────────────────────────────────────────────────────────────────

def draw_detection(frame_img: np.ndarray, det: dict, color_mode: str,
                   show_labels: bool, trail: Optional[list[tuple]] = None):
    """Draw a single detection box + optional label and trail."""
    x, y, w, h = det["bb_left"], det["bb_top"], det["bb_width"], det["bb_height"]
    color = get_color(det, color_mode)
    is_interp = det.get("_interpolated", False)

    thickness = 2
    if is_interp:
        # Dashed-style: draw with thinner line and slightly transparent overlay
        thickness = 1

    cv2.rectangle(frame_img, (x, y), (x + w, y + h), color, thickness)

    if is_interp:
        # Small "I" marker in corner
        cv2.putText(frame_img, "I", (x + 2, y + h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    # Draw trail (last N center positions)
    if trail and len(trail) > 1:
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            t_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame_img, trail[i - 1], trail[i], t_color, 1, cv2.LINE_AA)

    if show_labels:
        tid = det["track_id"]
        species_short = det["species"].split("(")[-1].rstrip(")") if det["species"] else "?"
        vis = det["visibility"]
        if vis == 0:
            vis = "Occluded"
        else:
            vis = "Visible"

        age = det["age"]
        if age == 0:
            age = "Unknown"
        elif age == 1:
            age = "Juvenile"
        elif age == 2:
            age = "Adult"

        sex = det["gender"]
        if sex == 0:
            sex = "Unknown"
        elif sex == 1:
            sex = "Juvenile"
        elif sex == 2:
            sex = "Adult"

        label = f"#{tid} {species_short} ({vis}/{age}/{sex})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        # Background rectangle
        cv2.rectangle(frame_img, (x, y - th - baseline - 4), (x + tw + 2, y), color, -1)
        # Text (black on colored bg)
        cv2.putText(frame_img, label, (x + 1, y - baseline - 2),
                    font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame_img: np.ndarray, frame_idx: int, total_frames: int,
             n_dets: int, is_interpolating: bool):
    """Draw frame info overlay."""
    h, w = frame_img.shape[:2]
    lines = [
        f"Frame: {frame_idx}/{total_frames}",
        f"Detections: {n_dets}",
    ]
    if is_interpolating:
        lines.append("Interpolation: ON")

    for i, text in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(frame_img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize custom MOT annotations on video.")
    parser.add_argument("--video", default=r"Z:\bambi_dataset\videos\223_matched_processed.mp4", help="Path to input video")
    parser.add_argument("--mot_file", default=r"Z:\bambi_dataset\mot\mot\train\223_gt.txt", help="Path to MOT annotation txt/csv")
    parser.add_argument("-o", "--output", help="Path to output video (mp4)")
    parser.add_argument("--show", action="store_true",
                        help="Show live preview window")
    parser.add_argument("--interpolate", action="store_true",
                        help="Linearly interpolate missing frames in tracks")
    parser.add_argument("--color", choices=["track", "species"], default="track",
                        help="Coloring mode: by track ID or by species (default: track)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide text labels on boxes")
    parser.add_argument("--trail", type=int, default=0,
                        help="Number of past frames to show as trail (0=off)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First frame to process")
    parser.add_argument("--end-frame", type=int, default=-1,
                        help="Last frame to process (-1 = all)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Display scale factor for --show mode")
    parser.add_argument("--fps-override", type=float, default=0,
                        help="Override output FPS (0 = use source FPS)")
    args = parser.parse_args()

    if not args.output and not args.show:
        parser.error("Specify --output and/or --show")



    # Parse annotations
    print(f"Loading annotations from {args.mot_file} ...")
    detections = parse_mot_file(args.mot_file)
    tracks = build_tracks(detections)
    print(f"  {len(detections)} detections, {len(tracks)} tracks")

    if args.interpolate:
        tracks = interpolate_tracks(tracks)
        total_after = sum(len(d) for d in tracks.values())
        print(f"  After interpolation: {total_after} detections "
              f"(+{total_after - len(detections)} interpolated)")

    frame_dets = index_by_frame(tracks)

    # Track center history for trails
    trail_history: dict[int, list[tuple]] = defaultdict(list)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video {args.video}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = args.fps_override if args.fps_override > 0 else src_fps

    print(f"Video: {width}x{height} @ {src_fps:.1f} fps, {total_frames} frames")

    end_frame = args.end_frame if args.end_frame > 0 else total_frames

    # Writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, out_fps, (width, height))
        if not writer.isOpened():
            print(f"Error: cannot create output video {args.output}")
            return

    # Seek to start
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    frame_idx = args.start_frame
    paused = False

    while frame_idx < end_frame:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # In paused mode, just wait for key
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                paused = False
            elif key == ord("q") or key == 27:
                break
            continue

        # Draw detections for this frame
        dets = frame_dets.get(frame_idx, [])
        for det in dets:
            tid = det["track_id"]
            cx = det["bb_left"] + det["bb_width"] // 2
            cy = det["bb_top"] + det["bb_height"] // 2

            if args.trail > 0:
                trail_history[tid].append((cx, cy))
                if len(trail_history[tid]) > args.trail:
                    trail_history[tid] = trail_history[tid][-args.trail:]
                trail = trail_history[tid]
            else:
                trail = None

            draw_detection(frame, det, args.color, not args.no_labels, trail)

        draw_hud(frame, frame_idx, total_frames, len(dets), args.interpolate)

        if writer:
            writer.write(frame)

        if args.show:
            if args.scale != 1.0:
                disp = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            else:
                disp = frame
            cv2.imshow("MOT Visualizer", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = True
                continue

        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"  Processed {frame_idx}/{end_frame} frames ...")

    cap.release()
    if writer:
        writer.release()
        print(f"Output written to {args.output}")
    if args.show:
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()