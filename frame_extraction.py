#!/usr/bin/env python3
"""
Extract frames from a split video where the left half is thermal and the right half is RGB.
Frames are saved to separate subfolders with configurable sampling rate.

Supports both individual video files and folders. When a folder is given,
all video files are discovered and matched with MOT annotation files by their
shared ID prefix (e.g., 0042_flight.mp4 matches 0042_annotations.txt).

The --mot flag accepts a single MOT file, a folder of MOT files, or is omitted
to auto-discover MOT files alongside the video(s).

When MOT filter arguments are provided (e.g. --species, --class-id, --min-width),
only frames containing at least one matching annotation are exported.
"""

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Optional

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".wmv", ".flv"}
MOT_EXTENSIONS = {".txt", ".csv"}

MOT_COLUMNS = [
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


# ---------------------------------------------------------------------------
# MOT parsing, formatting, and filtering
# ---------------------------------------------------------------------------

def parse_mot_line(line: str) -> Optional[dict]:
    """Parse a single MOT line into a dict."""
    line = line.strip()
    if not line:
        return None

    reader = csv.reader(io.StringIO(line))
    values = next(reader)

    if len(values) < 6:
        return None

    # Pad with empty strings if columns are missing
    if len(values) < len(MOT_COLUMNS):
        values.extend([""] * (len(MOT_COLUMNS) - len(values)))

    record = {}
    for i, col in enumerate(MOT_COLUMNS):
        val = values[i].strip() if i < len(values) else ""
        if col in ("frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "class_id"):
            record[col] = int(val) if val else 0
        elif col in ("conf", "visibility"):
            record[col] = float(val) if val else 0.0
        elif col in ("gender", "age", "is_propagated"):
            record[col] = val
        else:
            record[col] = val

    return record


def format_mot_line(record: dict) -> str:
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


def matches_mot_filter(record: dict, filters: dict) -> bool:
    """Check if a MOT record passes all active annotation filters."""

    # List filters
    for key in ("class_id", "species", "gender", "age", "visibility"):
        filter_values = filters.get(key)
        if filter_values is not None and len(filter_values) > 0:
            record_val = record[key]
            if key == "class_id":
                if record_val not in [int(v) for v in filter_values]:
                    return False
            elif key == "visibility":
                if record_val not in [float(v) for v in filter_values]:
                    return False
            else:
                for v in filter_values:
                    try:
                        if record_val.index(v):
                            return True
                    except ValueError:
                        pass
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


def has_active_mot_filters(mot_filters: dict) -> bool:
    """Check if any MOT annotation filter is active."""
    return any(v is not None for v in mot_filters.values())


def load_and_filter_mot(
    mot_path: Path,
    mot_filters: dict,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    sample_rate: int = 1,
) -> tuple[dict[int, list[dict]], list[dict]]:
    """
    Load a MOT file, apply frame range / sample rate constraints, then
    apply annotation-level filters.

    Returns:
        frame_records: dict mapping frame_idx -> list of matching records
        all_kept:      flat list of all kept records (for writing output)
    """
    frame_records: dict[int, list[dict]] = {}
    all_kept: list[dict] = []

    filtering = has_active_mot_filters(mot_filters)

    with open(mot_path, "r") as f:
        for line in f:
            record = parse_mot_line(line)
            if record is None:
                continue

            frame = record["frame"]

            # Frame range / sample rate
            if frame < start_frame:
                continue
            if end_frame is not None and frame >= end_frame:
                continue
            if (frame - start_frame) % sample_rate != 0:
                continue

            # Annotation filter
            if filtering and not matches_mot_filter(record, mot_filters):
                continue

            frame_records.setdefault(frame, []).append(record)
            all_kept.append(record)

    return frame_records, all_kept


def write_filtered_mot(records: list[dict], output_path: Path) -> int:
    """Write filtered MOT records to a file. Returns count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(format_mot_line(rec) + "\n")
    return len(records)


# ---------------------------------------------------------------------------
# Video / folder discovery
# ---------------------------------------------------------------------------

def extract_id(filename: str) -> Optional[str]:
    """Extract the ID prefix from a filename like <id>_<rest>.<ext>."""
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    if len(parts) >= 2:
        return parts[0]
    return None


def collect_mot_files(folder: Path) -> dict[str, Path]:
    """Scan a folder for MOT files and return a dict mapping ID -> path."""
    mot_files = {}
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in MOT_EXTENSIONS:
            continue
        file_id = extract_id(f.name)
        if file_id is not None:
            mot_files[file_id] = f
    return mot_files


def collect_video_files(folder: Path) -> dict[str, Path]:
    """Scan a folder for video files and return a dict mapping ID -> path."""
    videos = {}
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        file_id = extract_id(f.name)
        if file_id is not None:
            videos[file_id] = f
    return videos


def resolve_mot_source(
    mot_arg: Optional[str],
    fallback_folder: Path,
) -> dict[str, Path]:
    """
    Resolve the --mot argument into a dict mapping ID -> MOT file path.

    - mot_arg is None   -> auto-discover MOT files in fallback_folder
    - mot_arg is a file -> return single entry keyed by its ID
    - mot_arg is a dir  -> discover MOT files in that directory
    """
    if mot_arg is None:
        return collect_mot_files(fallback_folder)

    mot_path = Path(mot_arg)
    if mot_path.is_dir():
        return collect_mot_files(mot_path)
    elif mot_path.is_file():
        file_id = extract_id(mot_path.name)
        if file_id is not None:
            return {file_id: mot_path}
        # No parseable ID – return with the stem as key so it can still
        # be used explicitly in single-video mode.
        return {"__single__": mot_path}
    else:
        print(f"Warning: MOT path '{mot_arg}' does not exist, ignoring.")
        return {}


def build_pairs(
    video_source: dict[str, Path],
    mot_source: dict[str, Path],
) -> list[dict]:
    """
    Match video files with MOT files by ID.

    Returns a list of dicts with keys: 'id', 'video', 'mot' (mot may be None).
    """
    pairs = []
    for vid_id, video_path in sorted(video_source.items()):
        pairs.append({
            "id": vid_id,
            "video": video_path,
            "mot": mot_source.get(vid_id),
        })
    return pairs


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Path,
    output_dir: Path,
    video_id: Optional[str] = None,
    mot_path: Optional[Path] = None,
    mot_filters: Optional[dict] = None,
    sample_rate: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    export_thermal: bool = True,
    export_rgb: bool = True,
    use_id_naming: bool = True,
):
    """
    Extract frames from a single video file.

    When mot_path is given and mot_filters are active, only frames that contain
    at least one annotation passing the filter are exported.
    """
    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}")
        return 0

    if not export_thermal and not export_rgb:
        print("Error: At least one of --thermal or --rgb must be enabled.")
        return 0

    # Determine naming prefix
    if use_id_naming and video_id:
        name_prefix = video_id
    else:
        name_prefix = video_path.stem

    # --- Pre-filter MOT to determine allowed frames ---
    allowed_frames: Optional[set[int]] = None  # None means all frames allowed
    mot_kept_records: list[dict] = []

    if mot_filters is None:
        mot_filters = {}

    if mot_path and mot_path.is_file():
        filtering_by_mot = has_active_mot_filters(mot_filters)
        frame_records, mot_kept_records = load_and_filter_mot(
            mot_path, mot_filters, start_frame, end_frame, sample_rate,
        )
        if filtering_by_mot:
            allowed_frames = set(frame_records.keys())
            print(f"  MOT filter active: {len(allowed_frames)} frames with matching annotations")

    # Create output subdirectories
    thermal_dir = output_dir / "thermal"
    rgb_dir = output_dir / "rgb"

    if export_thermal:
        thermal_dir.mkdir(parents=True, exist_ok=True)
    if export_rgb:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_width = width // 2

    print(f"  Video: {video_path.name}")
    if video_id:
        print(f"  ID: {video_id}" + (f"  MOT: {mot_path.name}" if mot_path else "  MOT: (none)"))
    print(f"  Resolution: {width}x{height} (each half: {half_width}x{height})")
    print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"  Frame range: {start_frame} to {end_frame if end_frame is not None else 'end'} (exclusive)")
    print(f"  Sample rate: every {sample_rate} frame(s)")
    print(f"  Naming: {name_prefix}_<frame_idx>.png")

    # Seek to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    exported_count = 0

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % sample_rate == 0:
            # If MOT filtering is active, skip frames without matching annotations
            if allowed_frames is not None and frame_idx not in allowed_frames:
                frame_idx += 1
                continue

            filename = f"{name_prefix}_{frame_idx:08d}.png"

            if export_thermal:
                thermal_frame = frame[:, :half_width]
                cv2.imwrite(str(thermal_dir / filename), thermal_frame)

            if export_rgb:
                rgb_frame = frame[:, half_width:]
                cv2.imwrite(str(rgb_dir / filename), rgb_frame)

            exported_count += 1
            if exported_count % 100 == 0:
                print(f"    Exported {exported_count} frames (frame index {frame_idx})...")

        frame_idx += 1

    cap.release()

    # Write filtered MOT file
    if mot_path and mot_path.is_file():
        mot_output = output_dir / f"{name_prefix}_gt.txt"
        n_written = write_filtered_mot(mot_kept_records, mot_output)
        print(f"  MOT: {n_written} annotations -> {mot_output.name}")

    print(f"  Done: {exported_count} frames exported.\n")
    return exported_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_mot_filter_args(parser: argparse.ArgumentParser):
    """Add MOT annotation filter arguments to a parser."""
    group = parser.add_argument_group(
        "MOT annotation filters",
        "When a MOT file is available, these filters restrict which frames are "
        "exported to only those containing at least one matching annotation.",
    )
    group.add_argument(
        "--class-id",
        nargs="+",
        type=int,
        default=None,
        help="Filter by class_id(s). E.g. --class-id 2 50 51",
    )
    group.add_argument(
        "--species",
        nargs="+",
        type=str,
        default=None,
        help='Filter by species name(s). E.g. --species "Homo sapiens (Human)"',
    )
    group.add_argument(
        "--gender",
        nargs="+",
        type=str,
        default=None,
        help="Filter by gender value(s). E.g. --gender 0 1",
    )
    group.add_argument(
        "--age",
        nargs="+",
        type=str,
        default=None,
        help="Filter by age value(s). E.g. --age 0 1",
    )
    group.add_argument(
        "--visibility",
        nargs="+",
        type=float,
        default=None,
        help="Filter by visibility value(s). E.g. --visibility 1.0 0.5",
    )
    group.add_argument("--min-width", type=int, default=None, help="Min bounding box width (inclusive).")
    group.add_argument("--max-width", type=int, default=None, help="Max bounding box width (inclusive).")
    group.add_argument("--min-height", type=int, default=None, help="Min bounding box height (inclusive).")
    group.add_argument("--max-height", type=int, default=None, help="Max bounding box height (inclusive).")


def build_mot_filters(args: argparse.Namespace) -> dict:
    """Build a MOT filter dict from parsed CLI arguments."""
    return {
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from split video(s) (left=thermal, right=RGB).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Single video (auto-discovers MOT in same folder by ID)
  %(prog)s 0042_flight.mp4 output/

  # Single video with explicit MOT file
  %(prog)s 0042_flight.mp4 output/ --mot 0042_gt.txt

  # Folder with MOT files in the same folder (auto-discovered)
  %(prog)s data_folder/ output/

  # Videos in one folder, MOT files in another
  %(prog)s videos/ output/ --mot annotations/

  # Only export frames containing roe deer annotations
  %(prog)s data_folder/ output/ --species "Capreolus capreolus (Roe deer)"

  # Only frames with large detections of class 2
  %(prog)s data_folder/ output/ --class-id 2 --min-width 30

  # Folder, keep original video-name-based filenames
  %(prog)s data_folder/ output/ --no-id-naming
""",
    )
    parser.add_argument(
        "input",
        help="Path to a video file or a folder containing video files.",
    )
    parser.add_argument("output", help="Path to the output directory.")
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=1,
        help="Extract every N-th frame (default: 1 = every frame).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame index to consider (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last frame index (exclusive). Default: process until end of video.",
    )
    parser.add_argument(
        "--thermal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export thermal frames (left half). Use --no-thermal to disable.",
    )
    parser.add_argument(
        "--rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export RGB frames (right half). Use --no-rgb to disable.",
    )
    parser.add_argument(
        "--id-naming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Name output files as <id>_<frame>.png (default). "
             "Use --no-id-naming for <video_stem>_<frame>.png.",
    )
    parser.add_argument(
        "--mot",
        type=str,
        default=None,
        help="Path to a MOT annotation file or folder of MOT files. "
             "If omitted, MOT files are auto-discovered next to the video(s) "
             "by matching the <id>_ prefix.",
    )

    # Add MOT filter arguments
    add_mot_filter_args(parser)

    args = parser.parse_args()

    if not args.thermal and not args.rgb:
        parser.error("At least one of --thermal or --rgb must be enabled.")

    if args.gender is not None and len(args.gender) > 0:
        if "Unknown" in args.gender:
            args.gender.append("0")
        elif "Male" in args.gender:
            args.gender.append("1")
        elif "Female" in args.gender:
            args.gender.append("2")

    if args.age is not None and len(args.age) > 0:
        if "Unknown" in args.age:
            args.age.append("0")
        elif "Juvenile" in args.age:
            args.age.append("1")
        elif "Adult" in args.age:
            args.age.append("2")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    mot_filters = build_mot_filters(args)

    if has_active_mot_filters(mot_filters):
        active = [k for k, v in mot_filters.items() if v is not None]
        print(f"MOT filters active: {', '.join(active)}\n")

    if input_path.is_dir():
        # --- Folder mode ---
        video_source = collect_video_files(input_path)
        if not video_source:
            print(f"Error: No video files found in {input_path}")
            sys.exit(1)

        # Resolve MOT source: explicit path or auto-discover in video folder
        mot_source = resolve_mot_source(args.mot, fallback_folder=input_path)
        pairs = build_pairs(video_source, mot_source)

        print(f"Found {len(pairs)} video(s):\n")
        for p in pairs:
            mot_info = p['mot'].name if p['mot'] else '(none)'
            print(f"  [{p['id']}] {p['video'].name}  <->  {mot_info}")
        print()

        if has_active_mot_filters(mot_filters):
            missing_mot = [p for p in pairs if p['mot'] is None]
            if missing_mot:
                ids = ", ".join(p["id"] for p in missing_mot)
                print(f"Warning: MOT filters active but no MOT files for IDs: {ids}")
                print("         These videos will export ALL sampled frames.\n")

        total_exported = 0
        for p in pairs:
            vid_output = output_dir / p["id"]
            total_exported += extract_frames(
                video_path=p["video"],
                output_dir=vid_output,
                video_id=p["id"],
                mot_path=p["mot"],
                mot_filters=mot_filters,
                sample_rate=args.sample_rate,
                start_frame=args.start,
                end_frame=args.end,
                export_thermal=args.thermal,
                export_rgb=args.rgb,
                use_id_naming=args.id_naming,
            )

        print(f"All done. {total_exported} frames exported from {len(pairs)} video(s).")

    elif input_path.is_file():
        # --- Single video mode ---
        video_id = extract_id(input_path.name)

        # Resolve MOT: explicit path or auto-discover in video's parent folder
        mot_source = resolve_mot_source(args.mot, fallback_folder=input_path.parent)

        # Find matching MOT: by video ID, or use the single explicit file
        mot_path = None
        if video_id and video_id in mot_source:
            mot_path = mot_source[video_id]
        elif "__single__" in mot_source:
            mot_path = mot_source["__single__"]

        extract_frames(
            video_path=input_path,
            output_dir=output_dir,
            video_id=video_id,
            mot_path=mot_path,
            mot_filters=mot_filters,
            sample_rate=args.sample_rate,
            start_frame=args.start,
            end_frame=args.end,
            export_thermal=args.thermal,
            export_rgb=args.rgb,
            use_id_naming=args.id_naming,
        )
    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()