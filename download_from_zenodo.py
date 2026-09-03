#!/usr/bin/env python3
"""
Selective Zenodo Downloader for the BAMBI Dataset.

Uses the zenodo_upload_summary_*.json files produced by the uploader to download
specific flight ZIPs without fetching entire depositions.

Usage:
    # Download specific flights by prefix
    python download_from_zenodo.py -f 0 5 12 42

    # Download a range of flights
    python download_from_zenodo.py --range 10 25

    # List all available flights
    python download_from_zenodo.py --list

    # Download all flights from a specific part
    python download_from_zenodo.py --parts 1 3

    # Download all flights of a dataset split (train / val / test)
    python download_from_zenodo.py --split val

    # Download all flights (no filter)
    python download_from_zenodo.py

    # Download and automatically extract (deletes ZIPs after extraction)
    python download_from_zenodo.py --unzip

    # Download another dataset version
    # (base / raw / matched / orthographic / owl-transferred)
    python download_from_zenodo.py --version raw -f 0 5 12

    # Recordings plus the OWL-transferred RGB annotations, in one directory
    python download_from_zenodo.py --version owl-transferred -f 119 --unzip

    # Use a summary file from a custom location
    python download_from_zenodo.py -s /path/to/zenodo_upload_summary.json

Environment variable ZENODO_TOKEN can be used for restricted depositions.
"""

import argparse
import glob
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests

ZENODO_API = "https://zenodo.org/api"
ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"

# Resolved against this file, not the working directory: these summaries ship
# with the repository, and `--version` has to keep working when the script is
# called by path from somewhere else, as the notebooks do.
METADATA_DIR = Path(__file__).resolve().parent / "flight_metadata"

# Dataset versions and the summary file each one is described by.
VERSION_SUMMARIES = {
    "base": METADATA_DIR / "zenodo_upload_summary.json",
    "raw": METADATA_DIR / "zenodo_upload_summary_raw.json",
    "matched": METADATA_DIR / "zenodo_upload_summary_matched.json",
    "orthographic": METADATA_DIR / "zenodo_upload_summary_orthographic.json",
    "owl-transferred": METADATA_DIR / "zenodo_upload_summary_owl_transferred.json",
}

# Some versions are a LAYER on top of another rather than a standalone release.
# `owl-transferred` ships only the transferred RGB annotations, which are of no
# use without the recordings they annotate, so asking for it downloads the base
# recordings too and the two land side by side in one output directory.
# Layers are fetched in the order listed.
VERSION_LAYERS = {
    "owl-transferred": ["base", "owl-transferred"],
}

# Layers of the same flight share an output directory, so their archives must
# not collide. A summary may override this per part with a "zip_prefix" field.
DEFAULT_ZIP_PREFIX = "flight_"

# What each layer contributes, used with --unzip to tell "this layer is already
# extracted" from "some other layer of this flight is". `<id>` is the flight id.
LAYER_MARKERS = {
    "base": ["<id>_matched_processed.mp4"],
    "owl-transferred": ["<id>_rgb_gt.txt", "<id>_provenance.csv",
                        "<id>_owl_detections.csv"],
}


def layers_of(version: str) -> list[str]:
    """The summary keys that make up *version*, in download order."""
    return VERSION_LAYERS.get(version, [version])


def load_summary(path: Path) -> list[dict]:
    """Load and validate the upload summary JSON."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        sys.exit("Error: Summary file is empty or has unexpected format.")
    return data


def build_flight_index(summary: list[dict]) -> dict[str, dict]:
    """
    Build a lookup: flight_prefix -> {part, deposition_id, zip_name, files}.
    """
    index = {}
    for part in summary:
        dep_id = part["deposition_id"]
        part_num = part["part"]
        details = part.get("flight_details", {})
        zip_prefix = part.get("zip_prefix", DEFAULT_ZIP_PREFIX)
        for prefix in part["flights"]:
            index[prefix] = {
                "part": part_num,
                "deposition_id": dep_id,
                "zip_name": f"{zip_prefix}{prefix}.zip",
                "files": details.get(prefix, []),
            }
    return index


def flight_already_exists(
    prefix: str,
    output_dir: Path,
    unzip_mode: bool,
    zip_name: Optional[str] = None,
    marker_files: Optional[list[str]] = None,
) -> bool:
    """
    Check whether a flight has already been downloaded (or extracted).

    In normal mode:  check if the ZIP file exists.
    In unzip mode:   check if the files this layer contributes are already in
                     the output directory. `marker_files` names them (with
                     `<id>` standing in for the flight id); without it, any
                     file carrying the flight prefix counts, which is the right
                     test for a single-layer version but too loose for a
                     layered one, where the base layer would otherwise make the
                     annotation layer look present.
    """
    zip_path = output_dir / (zip_name or f"flight_{prefix}.zip")

    if zip_path.exists():
        return True

    if unzip_mode:
        if marker_files:
            return all((output_dir / m.replace("<id>", prefix)).exists()
                       for m in marker_files)
        # Check for any file starting with the flight prefix
        matches = glob.glob(str(output_dir / f"{prefix}_*")) + \
                  glob.glob(str(output_dir / f"{prefix}.*"))
        if matches:
            return True

    return False


def resolve_requested_flights(
    args: argparse.Namespace, index: dict[str, dict]
) -> list[str]:
    """Determine which flight prefixes the user wants to download."""
    requested: set[str] = set()

    if args.flights:
        requested.update(args.flights)

    if args.range:
        start, end = args.range
        for prefix in index:
            try:
                val = int(prefix)
                if start <= val <= end:
                    requested.add(prefix)
            except ValueError:
                pass

    if args.parts:
        for prefix, info in index.items():
            if info["part"] in args.parts:
                requested.add(prefix)

    # Validate
    missing = requested - set(index.keys())
    if missing:
        print(f"⚠  Unknown flight prefixes (skipping): {', '.join(sorted(missing, key=lambda x: int(x) if x.isdigit() else x))}")
        requested -= missing

    return sorted(requested, key=lambda x: int(x) if x.isdigit() else x)


def get_deposition_files(api_base: str, deposition_id: int, token: Optional[str]) -> dict[str, str]:
    """
    Fetch the file listing for a deposition.
    Returns {filename: download_url}.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(f"{api_base}/records/{deposition_id}", headers=headers)
    if r.status_code == 404:
        # Try draft endpoint (unpublished depositions need auth)
        r = requests.get(
            f"{api_base}/deposit/depositions/{deposition_id}",
            headers=headers,
        )
    r.raise_for_status()
    data = r.json()

    file_map = {}
    for f in data.get("files", []):
        name = f.get("filename") or f.get("key")
        url = f.get("links", {}).get("download") or f.get("links", {}).get("self")
        if name and url:
            file_map[name] = url

    return file_map


def download_file(url: str, dest: Path, token: Optional[str]) -> None:
    """Stream-download a file with progress indication."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 8 * 1024 * 1024  # 8 MB chunks

    # A carriage-return progress meter only redraws in place on a terminal.
    # Piped to a file, a log or a notebook cell it accumulates into one endless
    # line, so there it is reported at intervals on separate lines instead.
    interactive = sys.stdout.isatty()
    step = max(total // 10, 1) if total else 0
    next_report = step

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if not total:
                continue
            if interactive:
                print(f"\r     {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                      f"({downloaded / total * 100:.0f}%)", end="", flush=True)
            elif downloaded >= next_report:
                print(f"     {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                      f"({downloaded / total * 100:.0f}%)", flush=True)
                next_report += step
    if interactive:
        print()


def extract_and_remove_zip(zip_path: Path, output_dir: Path) -> int:
    """
    Extract a ZIP file into output_dir and delete the ZIP afterwards.
    Returns the number of extracted files.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        zf.extractall(output_dir)
    zip_path.unlink()
    return len(members)


def print_flight_table(index: dict[str, dict]) -> None:
    """Pretty-print all available flights grouped by part."""
    parts: dict[int, list[str]] = {}
    for prefix, info in index.items():
        parts.setdefault(info["part"], []).append(prefix)

    total_flights = len(index)
    print(f"\n📋 Available flights: {total_flights}\n")

    for part_num in sorted(parts):
        prefixes = parts[part_num]
        prefix_range = (
            f"{prefixes[0]}–{prefixes[-1]}" if len(prefixes) > 1 else prefixes[0]
        )
        print(f"  Part {part_num} ({len(prefixes)} flights: {prefix_range})")
        # Show file composition from first flight as example
        first = index[prefixes[0]]
        if first["files"]:
            suffixes = [
                f.replace(f"{prefixes[0]}", "<id>", 1) for f in first["files"]
            ]
            print(f"    Files per flight: {', '.join(suffixes)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Selectively download BAMBI flight ZIPs from Zenodo."
    )
    parser.add_argument(
        "--version", "-v",
        choices=sorted(VERSION_SUMMARIES),
        default="base",
        help="Dataset version to download (default: base). "
             "'owl-transferred' is a layer on the base release and pulls both: "
             "the recordings and the transferred RGB annotations.",
    )
    parser.add_argument(
        "--summary", "-s",
        type=Path,
        help="Path to a zenodo_upload_summary JSON file. Overrides --version.",
    )
    parser.add_argument(
        "--flights", "-f",
        nargs="+",
        type=str,
        help="Flight prefixes to download (e.g. 0 5 12 42)",
    )
    parser.add_argument(
        "--range", "-r",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Download flights in a numeric range (inclusive)",
    )
    parser.add_argument(
        "--parts", "-p",
        nargs="+",
        type=int,
        help="Download all flights from specific part numbers",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Download all flights belonging to a dataset split. "
             "Ignored when -f, --range, or --parts is also specified.",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=METADATA_DIR / "splits.json",
        help="Path to splits.json used by --split "
             "(default: the splits.json shipped next to this script)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available flights and exit",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path(r"./bambi_downloads"),
        help="Download destination (default: ./bambi_downloads)",
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=os.environ.get("ZENODO_TOKEN"),
        help="Zenodo token (only needed for draft/restricted depositions during testing)",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Zenodo Sandbox",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--unzip", "-u",
        action="store_true",
        help="Extract ZIPs after download and delete the ZIP files",
    )
    args = parser.parse_args()

    # ── Resolve the layer(s) this version is made of ─────────────────────────
    # An explicit --summary is always a single, self-contained layer.
    if args.summary:
        layer_names = ["custom"]
        layer_summaries = {"custom": args.summary}
    else:
        layer_names = layers_of(args.version)
        layer_summaries = {name: VERSION_SUMMARIES[name] for name in layer_names}

    layer_index: dict[str, dict[str, dict]] = {}
    for name in layer_names:
        path = layer_summaries[name]
        if not path.exists():
            sys.exit(f"Error: Summary file not found: {path}")
        layer_index[name] = build_flight_index(load_summary(path))

    # The flight universe is the union over layers: `owl-transferred` has no
    # annotations for a flight the base release ships without labels, and that
    # is a gap in coverage, not an unknown flight id.
    index: dict[str, dict] = {}
    for name in layer_names:
        for prefix, info in layer_index[name].items():
            index.setdefault(prefix, info)

    api_base = ZENODO_SANDBOX_API if args.sandbox else ZENODO_API

    if len(layer_names) > 1:
        print(f"ℹ  Version '{args.version}' is layered: "
              f"{' + '.join(layer_names)}")

    # ── List mode ────────────────────────────────────────────────────────────
    if args.list:
        for name in layer_names:
            if len(layer_names) > 1:
                print(f"\n=== layer: {name} ===")
            print_flight_table(layer_index[name])
        return

    # ── Resolve flights to download ──────────────────────────────────────────
    has_explicit_filter = bool(args.flights) or args.range is not None or bool(args.parts)

    if args.split and has_explicit_filter:
        print("WARNING: --split is ignored when -f, --range, or --parts is also specified.")

    if not has_explicit_filter and args.split:
        # Load the splits file and expand the split into individual flight IDs
        if not args.splits_file.exists():
            sys.exit(f"Error: Splits file not found: {args.splits_file}")
        with open(args.splits_file) as fh:
            splits_data = json.load(fh)
        if args.split not in splits_data:
            sys.exit(f"Error: Split '{args.split}' not found in {args.splits_file}. "
                     f"Available: {', '.join(splits_data.keys())}")
        split_fids = [str(fid) for fid in splits_data[args.split]]
        if not split_fids:
            sys.exit(f"Error: No flights found for split '{args.split}'.")
        print(f"[split] '{args.split}': {len(split_fids)} flights loaded "
              f"from {args.splits_file}")
        args.flights = split_fids
        prefixes = resolve_requested_flights(args, index)
    elif not has_explicit_filter:
        # No filter at all → download everything
        print("ℹ  No filter specified — downloading all flights.")
        prefixes = sorted(index.keys(), key=lambda x: int(x) if x.isdigit() else x)
    else:
        args.flights = [str(x) for x in (args.flights or [])]
        prefixes = resolve_requested_flights(args, index)

    if not prefixes:
        sys.exit("No valid flights to download.")

    os.makedirs(args.output_dir, exist_ok=True)

    totals = {"downloaded": 0, "extracted": 0, "skipped": 0}
    failed: list[str] = []
    missing_in_layer: dict[str, list[str]] = {}

    for name in layer_names:
        this_index = layer_index[name]
        markers = LAYER_MARKERS.get(name)

        # A layer need not carry every requested flight.
        wanted = [p for p in prefixes if p in this_index]
        absent = [p for p in prefixes if p not in this_index]
        if absent:
            missing_in_layer[name] = absent

        if len(layer_names) > 1:
            print(f"\n{'═' * 50}")
            print(f"  LAYER: {name}  ({len(wanted)} of {len(prefixes)} "
                  f"requested flight(s) available)")
            print(f"{'═' * 50}")

        if not wanted:
            print("  nothing to do for this layer")
            continue

        # ── Pre-filter: skip already downloaded / extracted flights ──────────
        to_download = []
        skipped_count = 0
        for prefix in wanted:
            if flight_already_exists(prefix, args.output_dir, args.unzip,
                                     this_index[prefix]["zip_name"], markers):
                skipped_count += 1
            else:
                to_download.append(prefix)

        totals["skipped"] += skipped_count
        if skipped_count:
            print(f"⏭  Skipping {skipped_count} flight(s) already present "
                  f"in {args.output_dir}")

        if not to_download:
            print("✅ All requested flights are already downloaded — nothing to do.")
            continue

        # Group by deposition for efficient API calls
        by_deposition: dict[int, list[str]] = {}
        for prefix in to_download:
            dep_id = this_index[prefix]["deposition_id"]
            by_deposition.setdefault(dep_id, []).append(prefix)

        print(f"\n📥 Downloading {len(to_download)} flight(s) from "
              f"{len(by_deposition)} deposition(s)")
        if args.unzip:
            print("📦 ZIPs will be extracted and removed after download")

        if args.dry_run:
            for dep_id, dep_prefixes in by_deposition.items():
                part_num = this_index[dep_prefixes[0]]["part"]
                print(f"\n  Part {part_num} (deposition {dep_id}):")
                for p in dep_prefixes:
                    print(f"    {this_index[p]['zip_name']}")
            print(f"\n✋ Dry run — nothing downloaded. "
                  f"({skipped_count} already present)")
            continue

        # ── Download ─────────────────────────────────────────────────────────
        for dep_id, dep_prefixes in by_deposition.items():
            part_num = this_index[dep_prefixes[0]]["part"]
            print(f"\n{'─' * 50}")
            print(f"  Part {part_num} (deposition {dep_id})")

            # Fetch file listing once per deposition
            try:
                file_map = get_deposition_files(api_base, dep_id, args.token)
            except requests.HTTPError as e:
                print(f"  ❌ Failed to fetch deposition {dep_id}: {e}")
                failed.extend(dep_prefixes)
                continue

            for prefix in dep_prefixes:
                zip_name = this_index[prefix]["zip_name"]
                dest_path = args.output_dir / zip_name

                if zip_name not in file_map:
                    print(f"  ❌ {zip_name} not found in deposition files")
                    failed.append(prefix)
                    continue

                print(f"  ⬇  {zip_name}")
                try:
                    download_file(file_map[zip_name], dest_path, args.token)
                    totals["downloaded"] += 1
                except requests.HTTPError as e:
                    print(f"     ❌ Download failed: {e}")
                    dest_path.unlink(missing_ok=True)
                    failed.append(prefix)
                    continue

                # Extract if requested
                if args.unzip:
                    try:
                        n_files = extract_and_remove_zip(dest_path, args.output_dir)
                        print(f"     📦 Extracted {n_files} file(s), ZIP removed")
                        totals["extracted"] += 1
                    except (zipfile.BadZipFile, OSError) as e:
                        print(f"     ⚠  Extraction failed: {e} (ZIP kept)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"✅ Done! Downloaded: {totals['downloaded']}, "
          f"Skipped: {totals['skipped']}", end="")
    if args.unzip:
        print(f", Extracted: {totals['extracted']}", end="")
    if failed:
        print(f", Failed: {len(failed)} ({', '.join(failed)})")
    else:
        print()

    for name, absent in missing_in_layer.items():
        print(f"ℹ  {len(absent)} requested flight(s) have no '{name}' data: "
              f"{', '.join(absent[:12])}"
              + (" …" if len(absent) > 12 else ""))

    print(f"   Files saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()