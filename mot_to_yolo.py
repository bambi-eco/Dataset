#!/usr/bin/env python3
"""
Convert custom MOT ground-truth files to YOLO label format.

MOT columns (0-indexed):
  0: frame
  1: track_id
  2: bb_left
  3: bb_top
  4: bb_width
  5: bb_height
  6: conf
  7: class_id
  8: visibility
  9: species
 10: gender
 11: age
 12: is_propagated

YOLO output per line:
  <label> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

The <label> is built by concatenating the requested properties with '-'.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MOT_COLUMNS = [
    "frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
    "conf", "class_id", "visibility", "species", "gender", "age", "is_propagated",
]

LABEL_PROPERTIES = ["class_id", "gender", "age", "visibility", "species"]


def parse_mot_line(row: list[str]) -> dict:
    """Parse a single MOT CSV row into a dict."""
    d = {}
    for i, col in enumerate(MOT_COLUMNS):
        val = row[i].strip() if i < len(row) else ""
        # Numeric fields
        if col in ("frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
                    "class_id", "gender", "age", "is_propagated"):
            try:
                d[col] = int(val)
            except ValueError:
                d[col] = val  # keep as string if not parseable
        elif col in ("conf", "visibility"):
            try:
                d[col] = float(val)
            except ValueError:
                d[col] = val
        else:
            d[col] = val
    return d


def build_label(det: dict, properties: list[str]) -> str:
    """Build a YOLO label string from the requested properties."""
    parts = []
    for prop in properties:
        val = det.get(prop, "")
        # For visibility, format as two decimals
        if prop == "visibility" and isinstance(val, float):
            parts.append(f"{val:.2f}")
        else:
            parts.append(str(val))
    return "-".join(parts)


def convert_mot_file(
    mot_path: Path,
    output_dir: Path,
    img_width: int,
    img_height: int,
    label_properties: list[str],
):
    """Convert a single MOT file to per-frame YOLO label files."""
    mot_name = mot_path.stem  # filename without extension

    # Group detections by frame
    frames: dict[int, list[dict]] = defaultdict(list)

    with open(mot_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            det = parse_mot_line(row)
            frames[det["frame"]].append(det)

    written = 0
    for frame_id in sorted(frames.keys()):
        dets = frames[frame_id]
        out_path = output_dir / f"{mot_name}_{frame_id}.txt"

        lines = []
        for det in dets:
            label = build_label(det, label_properties)

            # Convert MOT bbox (left, top, w, h) -> YOLO (cx, cy, w, h) normalized
            bb_left = det["bb_left"]
            bb_top = det["bb_top"]
            bb_w = det["bb_width"]
            bb_h = det["bb_height"]

            cx = (bb_left + bb_w / 2.0) / img_width
            cy = (bb_top + bb_h / 2.0) / img_height
            w = bb_w / img_width
            h = bb_h / img_height

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            lines.append(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        with open(out_path, "w") as out_f:
            out_f.writelines(lines)
        written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom MOT ground-truth files to YOLO label format."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a single MOT .txt file or a folder containing MOT .txt files.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for YOLO label files.",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        required=True,
        default=1024,
        help="Image width in pixels (for normalization).",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=1024,
        help="Image height in pixels (for normalization).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["class_id"],
        choices=LABEL_PROPERTIES,
        help=(
            "Properties to concatenate as the YOLO label (joined with '-'). "
            "Default: class_id. "
            "Example: --labels class_id gender age → '2-0-0'"
        ),
    )

    args = parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect MOT files
    if input_path.is_file():
        mot_files = [input_path]
    elif input_path.is_dir():
        mot_files = sorted(input_path.glob("*.txt"))
        if not mot_files:
            print(f"No .txt files found in {input_path}")
            return
    else:
        print(f"Input path does not exist: {input_path}")
        return

    print(f"Label format: {' - '.join(args.labels)}")
    print(f"Image size: {args.img_width} x {args.img_height}")
    print(f"Output dir: {output_dir}")
    print(f"Processing {len(mot_files)} file(s)...\n")

    total_frames = 0
    for mot_file in mot_files:
        n = convert_mot_file(mot_file, output_dir, args.img_width, args.img_height, args.labels)
        print(f"  {mot_file.name} → {n} frame label files")
        total_frames += n

    print(f"\nDone. {total_frames} YOLO label files written to {output_dir}")


if __name__ == "__main__":
    main()