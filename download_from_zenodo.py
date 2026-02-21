#!/usr/bin/env python3
"""
Selective Zenodo Downloader for the BAMBI Dataset.

Uses the zenodo_upload_summary.json produced by the uploader to download
specific flight ZIPs without fetching entire depositions.

Usage:
    # Download specific flights by prefix
    python download_from_zenodo.py -s zenodo_upload_summary.json -f 0 5 12 42

    # Download a range of flights
    python download_from_zenodo.py -s zenodo_upload_summary.json --range 10 25

    # List all available flights
    python download_from_zenodo.py -s zenodo_upload_summary.json --list

    # Download all flights from a specific part
    python download_from_zenodo.py -s zenodo_upload_summary.json --parts 1 3

    # Download all flights (no filter)
    python download_from_zenodo.py -s zenodo_upload_summary.json

    # Download and automatically extract (deletes ZIPs after extraction)
    python download_from_zenodo.py -s zenodo_upload_summary.json --unzip

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
        for prefix in part["flights"]:
            index[prefix] = {
                "part": part_num,
                "deposition_id": dep_id,
                "zip_name": f"flight_{prefix}.zip",
                "files": details.get(prefix, []),
            }
    return index


def flight_already_exists(prefix: str, output_dir: Path, unzip_mode: bool) -> bool:
    """
    Check whether a flight has already been downloaded (or extracted).

    In normal mode:  check if the ZIP file exists.
    In unzip mode:   check if any files with the flight prefix exist in the
                     output directory (i.e. the flight was already extracted).
                     Falls back to checking the ZIP as well.
    """
    zip_path = output_dir / f"flight_{prefix}.zip"

    if zip_path.exists():
        return True

    if unzip_mode:
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

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1e6
                total_mb = total / 1e6
                print(
                    f"\r     {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)",
                    end="",
                    flush=True,
                )
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
        "--summary", "-s",
        type=Path,
        default=Path(r"./flight_metadata/zenodo_upload_summary.json"),
        help="Path to zenodo_upload_summary.json",
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

    summary = load_summary(args.summary)
    index = build_flight_index(summary)
    api_base = ZENODO_SANDBOX_API if args.sandbox else ZENODO_API

    # ── List mode ────────────────────────────────────────────────────────────
    if args.list:
        print_flight_table(index)
        return

    # ── Resolve flights to download ──────────────────────────────────────────
    if not args.flights and args.range is None and not args.parts:
        # No filter specified → download everything
        print("ℹ  No filter specified — downloading all flights.")
        prefixes = sorted(index.keys(), key=lambda x: int(x) if x.isdigit() else x)
    else:
        args.flights = [str(x) for x in (args.flights or [])]
        prefixes = resolve_requested_flights(args, index)

    if not prefixes:
        sys.exit("No valid flights to download.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Pre-filter: skip already downloaded / extracted flights ───────────────
    to_download = []
    skipped_count = 0
    for prefix in prefixes:
        if flight_already_exists(prefix, args.output_dir, args.unzip):
            skipped_count += 1
        else:
            to_download.append(prefix)

    if skipped_count:
        print(f"⏭  Skipping {skipped_count} flight(s) already present in {args.output_dir}")

    if not to_download:
        print("✅ All requested flights are already downloaded — nothing to do.")
        return

    # Group by deposition for efficient API calls
    by_deposition: dict[int, list[str]] = {}
    for prefix in to_download:
        dep_id = index[prefix]["deposition_id"]
        by_deposition.setdefault(dep_id, []).append(prefix)

    total_count = len(to_download)
    print(f"\n📥 Downloading {total_count} flight(s) from {len(by_deposition)} deposition(s)")
    if args.unzip:
        print("📦 ZIPs will be extracted and removed after download")

    if args.dry_run:
        for dep_id, dep_prefixes in by_deposition.items():
            part_num = index[dep_prefixes[0]]["part"]
            print(f"\n  Part {part_num} (deposition {dep_id}):")
            for p in dep_prefixes:
                print(f"    flight_{p}.zip")
        print(f"\n✋ Dry run — nothing downloaded. ({skipped_count} already present)")
        return

    # ── Download ─────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_count = 0
    extracted_count = 0
    failed: list[str] = []

    for dep_id, dep_prefixes in by_deposition.items():
        part_num = index[dep_prefixes[0]]["part"]
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
            zip_name = index[prefix]["zip_name"]
            dest_path = args.output_dir / zip_name

            if zip_name not in file_map:
                print(f"  ❌ {zip_name} not found in deposition files")
                failed.append(prefix)
                continue

            print(f"  ⬇  {zip_name}")
            try:
                download_file(file_map[zip_name], dest_path, args.token)
                downloaded_count += 1
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
                    extracted_count += 1
                except (zipfile.BadZipFile, OSError) as e:
                    print(f"     ⚠  Extraction failed: {e} (ZIP kept)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"✅ Done! Downloaded: {downloaded_count}, Skipped: {skipped_count}", end="")
    if args.unzip:
        print(f", Extracted: {extracted_count}", end="")
    if failed:
        print(f", Failed: {len(failed)} ({', '.join(failed)})")
    else:
        print()
    print(f"   Files saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()