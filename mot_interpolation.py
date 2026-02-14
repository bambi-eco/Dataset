#!/usr/bin/env python3
"""
Interpolate missing frames in custom MOT ground truth files.

For each track_id, finds gaps between existing detections and linearly
interpolates bb_left, bb_top, bb_width, bb_height for all missing frames.
Interpolated rows are marked with is_propagated=1.

Usage:
    python mot_interpolation.py <input_path> <output_folder> [--step STEP]

    input_path:     Path to a single .txt MOT file or a folder of .txt files.
    output_folder:  Where interpolated files are written (same filenames).
    --step STEP:    Frame step for interpolation (default: 1).
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict


COLUMNS = [
    "frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
    "conf", "class_id", "visibility", "species", "gender", "age", "is_propagated"
]

INT_COLS = {"frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "class_id", "gender", "age", "is_propagated"}
FLOAT_COLS = {"conf", "visibility"}
STR_COLS = {"species"}

# Columns to interpolate numerically
INTERP_NUMERIC = ["bb_left", "bb_top", "bb_width", "bb_height"]


def parse_line(line: str) -> dict:
    """Parse a single CSV line into a detection dict."""
    parts = line.strip().split(",")
    if len(parts) < len(COLUMNS):
        # Pad missing fields
        parts.extend([""] * (len(COLUMNS) - len(parts)))

    det = {}
    for i, col in enumerate(COLUMNS):
        val = parts[i].strip()
        if col in INT_COLS:
            det[col] = int(val) if val else 0
        elif col in FLOAT_COLS:
            det[col] = float(val) if val else 0.0
        elif col in STR_COLS:
            det[col] = val
        else:
            det[col] = val
    return det


def format_line(det: dict) -> str:
    """Format a detection dict back into a CSV line."""
    return (
        f"{det['frame']},"
        f"{det['track_id']},"
        f"{det['bb_left']},"
        f"{det['bb_top']},"
        f"{det['bb_width']},"
        f"{det['bb_height']},"
        f"{det['conf']:.2f},"
        f"{det['class_id']},"
        f"{det['visibility']:.2f},"
        f"{det.get('species', '')},"
        f"{det.get('gender', '')},"
        f"{det.get('age', '')},"
        f"{det['is_propagated']}\n"
    )


def lerp(a, b, t):
    """Linear interpolation between a and b at parameter t in [0, 1]."""
    return a + (b - a) * t


def interpolate_track(detections: list[dict], step: int = 1) -> list[dict]:
    """
    Given a list of detections for a single track (sorted by frame),
    interpolate missing frames between consecutive detections.

    Args:
        detections: Sorted list of detection dicts for one track_id.
        step: Frame step for interpolation (default 1 = every frame).

    Returns:
        List of all detections (original + interpolated), sorted by frame.
    """
    if len(detections) < 2:
        return detections

    result = []

    for i in range(len(detections) - 1):
        d_start = detections[i]
        d_end = detections[i + 1]
        result.append(d_start)

        frame_start = d_start["frame"]
        frame_end = d_end["frame"]
        gap = frame_end - frame_start

        if gap <= step:
            # No frames to interpolate
            continue

        # Generate interpolated frames
        for f in range(frame_start + step, frame_end, step):
            t = (f - frame_start) / gap
            interp_det = {
                "frame": f,
                "track_id": d_start["track_id"],
                "bb_left": round(lerp(d_start["bb_left"], d_end["bb_left"], t)),
                "bb_top": round(lerp(d_start["bb_top"], d_end["bb_top"], t)),
                "bb_width": round(lerp(d_start["bb_width"], d_end["bb_width"], t)),
                "bb_height": round(lerp(d_start["bb_height"], d_end["bb_height"], t)),
                "conf": lerp(d_start["conf"], d_end["conf"], t),
                "class_id": d_start["class_id"],
                "visibility": lerp(d_start["visibility"], d_end["visibility"], t),
                "species": d_start["species"],
                "gender": d_start["gender"],
                "age": d_start["age"],
                "is_propagated": 1,
            }
            result.append(interp_det)

    # Add the last detection
    result.append(detections[-1])
    return result


def process_file(input_path: Path, output_path: Path, step: int = 1):
    """Process a single MOT file: interpolate all tracks and write output."""
    # Read all detections
    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    detections = [parse_line(line) for line in lines]

    # Group by track_id
    tracks: dict[int, list[dict]] = defaultdict(list)
    for det in detections:
        tracks[det["track_id"]].append(det)

    # Sort each track by frame and interpolate
    all_output = []
    original_count = len(detections)
    interpolated_count = 0

    for track_id in sorted(tracks.keys()):
        track_dets = sorted(tracks[track_id], key=lambda d: d["frame"])
        interpolated = interpolate_track(track_dets, step=step)
        new_count = sum(1 for d in interpolated if d["is_propagated"] == 1)
        interpolated_count += new_count
        all_output.extend(interpolated)

    # Sort by frame, then track_id
    all_output.sort(key=lambda d: (d["frame"], d["track_id"]))

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for det in all_output:
            f.write(format_line(det))

    total = len(all_output)
    print(f"  {input_path.name}: {original_count} original + {interpolated_count} interpolated = {total} total detections")


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate bounding boxes in custom MOT files."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a single .txt MOT file or a folder containing .txt files.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Target folder for interpolated output files.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step for interpolation (default: 1, i.e. every frame).",
    )
    args = parser.parse_args()

    input_path: Path = args.input_path
    output_folder: Path = args.output_folder

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
        if not files:
            print(f"No .txt files found in {input_path}")
            return
    else:
        print(f"Error: {input_path} is neither a file nor a directory.")
        return

    print(f"Processing {len(files)} file(s), step={args.step}")
    print(f"Output folder: {output_folder}")
    print()

    for filepath in files:
        output_path = output_folder / filepath.name
        process_file(filepath, output_path, step=args.step)

    print("\nDone.")


if __name__ == "__main__":
    main()