#!/usr/bin/env python3
"""
Filter custom MOT ground truth files based on various criteria.

Usage examples:
    # Filter a single file by species
    python mot_filter.py input.txt -o output_dir --species "Homo sapiens (Human)"

    # Filter a folder, keep only class_id 50 and 51, with min width 25
    python mot_filter.py input_folder/ -o output_dir --class-id 50 51 --min-width 25

    # Filter by multiple species and visibility values
    python mot_filter.py input_folder/ -o output_dir --species "Homo sapiens (Human)" "Capreolus capreolus (Roe deer)" --visibility 1.0 0.5

    # Combine filters: class, size range, and gender
    python mot_filter.py input.txt -o output_dir --class-id 2 --min-width 30 --max-height 40 --gender 0 1
"""

import argparse
import csv
import io
from pathlib import Path
from typing import Optional


COLUMNS = [
    "frame",        # int
    "track_id",     # int
    "bb_left",      # int
    "bb_top",       # int
    "bb_width",     # int
    "bb_height",    # int
    "conf",         # float
    "class_id",     # int
    "visibility",   # float
    "species",      # str
    "gender",       # int/str
    "age",          # int/str
    "is_propagated" # int (0/1)
]


def parse_line(line: str) -> Optional[dict]:
    """Parse a single MOT line into a dict."""
    line = line.strip()
    if not line:
        return None

    reader = csv.reader(io.StringIO(line))
    values = next(reader)

    if len(values) < len(COLUMNS):
        # Pad with empty strings if columns are missing
        values.extend([""] * (len(COLUMNS) - len(values)))

    record = {}
    for i, col in enumerate(COLUMNS):
        val = values[i].strip() if i < len(values) else ""
        if col in ("frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "class_id"):
            record[col] = int(val) if val else 0
        elif col in ("conf", "visibility"):
            record[col] = float(val) if val else 0.0
        elif col in ("gender", "age", "is_propagated"):
            # Keep as string for flexible matching (could be int or text)
            record[col] = val
        else:
            record[col] = val

    return record


def format_line(record: dict) -> str:
    """Format a record dict back to a MOT CSV line."""
    return (
        f"{record['frame']},"
        f"{record['track_id']},"
        f"{record['bb_left']},"
        f"{record['bb_top']},"
        f"{record['bb_width']},"
        f"{record['bb_height']},"
        f"{record['conf']:.2f},"
        f"{record['class_id']},"
        f"{record['visibility']:.2f},"
        f"{record['species']},"
        f"{record['gender']},"
        f"{record['age']},"
        f"{record['is_propagated']}"
    )


def matches_filter(record: dict, filters: dict) -> bool:
    """Check if a record passes all active filters."""

    # List filters: if the list is non-empty, the value must be in the list
    for key in ("class_id", "species", "gender", "age", "visibility"):
        filter_values = filters.get(key)
        if filter_values is not None and len(filter_values) > 0:
            record_val = record[key]
            # For numeric comparisons, try matching both as-is and converted
            if key == "class_id":
                if record_val not in [int(v) for v in filter_values]:
                    return False
            elif key == "visibility":
                if record_val not in [float(v) for v in filter_values]:
                    return False
            else:
                # String matching for species, gender, age
                str_values = [str(v) for v in filter_values]
                if str(record_val) not in str_values:
                    return False

    # Range filters for width
    if filters.get("min_width") is not None:
        if record["bb_width"] < filters["min_width"]:
            return False
    if filters.get("max_width") is not None:
        if record["bb_width"] > filters["max_width"]:
            return False

    # Range filters for height
    if filters.get("min_height") is not None:
        if record["bb_height"] < filters["min_height"]:
            return False
    if filters.get("max_height") is not None:
        if record["bb_height"] > filters["max_height"]:
            return False

    return True


def filter_file(input_path: Path, output_path: Path, filters: dict) -> dict:
    """Filter a single MOT file and write results. Returns stats."""
    total = 0
    kept = 0
    lines_out = []

    with open(input_path, "r") as f:
        for line in f:
            record = parse_line(line)
            if record is None:
                continue
            total += 1
            if matches_filter(record, filters):
                kept += 1
                lines_out.append(format_line(record))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")

    return {"total": total, "kept": kept, "file": input_path.name}


def collect_unique_values(input_path: Path) -> dict:
    """Scan file(s) and collect unique values per filterable column."""
    files = []
    if input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
    elif input_path.is_file():
        files = [input_path]

    unique = {col: set() for col in ("class_id", "species", "gender", "age", "visibility")}
    width_range = [float("inf"), float("-inf")]
    height_range = [float("inf"), float("-inf")]

    for fp in files:
        with open(fp, "r") as f:
            for line in f:
                record = parse_line(line)
                if record is None:
                    continue
                for col in unique:
                    unique[col].add(record[col])
                width_range[0] = min(width_range[0], record["bb_width"])
                width_range[1] = max(width_range[1], record["bb_width"])
                height_range[0] = min(height_range[0], record["bb_height"])
                height_range[1] = max(height_range[1], record["bb_height"])

    return {
        "unique": {k: sorted(v, key=str) for k, v in unique.items()},
        "width_range": width_range if width_range[0] != float("inf") else [0, 0],
        "height_range": height_range if height_range[0] != float("inf") else [0, 0],
        "file_count": len(files),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter custom MOT files by class, species, gender, age, visibility, and bounding box size.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Output folder for filtered MOT files.",
    )

    # List filters (nargs='*' means: if flag given with no args -> empty list; if flag not given -> None)
    parser.add_argument(
        "--class-id",
        nargs="+",
        type=int,
        default=None,
        help="Filter by class_id(s). E.g. --class-id 2 50 51",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        type=str,
        default=None,
        help='Filter by species name(s). E.g. --species "Homo sapiens (Human)" "Unknown"',
    )
    parser.add_argument(
        "--gender",
        nargs="+",
        type=str,
        default=None,
        help="Filter by gender value(s). E.g. --gender 0 1",
    )
    parser.add_argument(
        "--age",
        nargs="+",
        type=str,
        default=None,
        help="Filter by age value(s). E.g. --age 0 1",
    )
    parser.add_argument(
        "--visibility",
        nargs="+",
        type=float,
        default=None,
        help="Filter by visibility value(s). E.g. --visibility 1.0 0.5",
    )

    # Range filters
    parser.add_argument("--min-width", type=int, default=None, help="Minimum bounding box width (inclusive).")
    parser.add_argument("--max-width", type=int, default=None, help="Maximum bounding box width (inclusive).")
    parser.add_argument("--min-height", type=int, default=None, help="Minimum bounding box height (inclusive).")
    parser.add_argument("--max-height", type=int, default=None, help="Maximum bounding box height (inclusive).")

    # Utility
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print unique values found in the input file(s) and exit (useful for discovering filter values).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help='Optional suffix for output filenames. E.g. --suffix "_filtered" turns 346_gt.txt into 346_gt_filtered.txt',
    )

    args = parser.parse_args()
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    # --info mode: just print stats and exit
    if args.info:
        info = collect_unique_values(input_path)
        print(f"Scanned {info['file_count']} file(s):")
        print(f"  class_id values:    {info['unique']['class_id']}")
        print(f"  species values:     {info['unique']['species']}")
        print(f"  gender values:      {info['unique']['gender']}")
        print(f"  age values:         {info['unique']['age']}")
        print(f"  visibility values:  {info['unique']['visibility']}")
        print(f"  bb_width range:     {info['width_range'][0]} - {info['width_range'][1]}")
        print(f"  bb_height range:    {info['height_range'][0]} - {info['height_range'][1]}")
        return

    # Build filters dict
    filters = {
        "class_id": args.class_id,
        "species": args.species,
        "gender": args.gender,
        "age": args.age,
        "visibility": args.visibility,
        "min_width": args.min_width,
        "max_width": args.max_width,
        "min_height": args.min_height,
        "max_height": args.max_height,
    }

    # Check if any filter is active
    active = any(v is not None for v in filters.values())
    if not active:
        print("Warning: No filters specified. All detections will be kept (files will be copied as-is).")

    # Collect input files
    if input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
        if not files:
            print(f"No .txt files found in '{input_path}'.")
            return
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"Error: '{input_path}' is neither a file nor a directory.")
        return

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    total_files = len(files)
    total_kept = 0
    total_detections = 0

    for fp in files:
        stem = fp.stem
        out_name = f"{stem}{args.suffix}{fp.suffix}"
        out_path = output_dir / out_name

        stats = filter_file(fp, out_path, filters)
        total_kept += stats["kept"]
        total_detections += stats["total"]

        pct = (stats["kept"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {stats['file']}: {stats['kept']}/{stats['total']} detections kept ({pct:.1f}%) -> {out_path}")

    print(f"\nSummary: {total_kept}/{total_detections} total detections kept across {total_files} file(s).")


if __name__ == "__main__":
    main()