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

    # Only the annotation, pose and metadata files, never the video: reads the
    # members straight out of the archive on Zenodo with HTTP range requests,
    # a few MB per flight instead of a gigabyte or more
    python download_from_zenodo.py --annotations-only -f 146

    # The same for a whole annotation study: every flight that has an
    # environment layer, with its boxes, but without a single video
    python download_from_zenodo.py --version environment-all --annotations-only

    # Pick single files out of a flight's archive, in any version: glob
    # patterns match the path inside the archive or just the file name
    python download_from_zenodo.py -f 146 --include '*_gt.txt' '*_poses.json'
    python download_from_zenodo.py --version raw -f 1 --include '*.SRT' air_data.csv
    python download_from_zenodo.py --version matched -f 1 --include 'labels/*'
    python download_from_zenodo.py --version raw -f 1 --exclude '*_V_*.MP4'

    # See what a flight's archive holds before choosing (sizes, no download)
    python download_from_zenodo.py --version orthographic -f 1 --list-files

    # Use a summary file from a custom location
    python download_from_zenodo.py -s /path/to/zenodo_upload_summary.json

Environment variable ZENODO_TOKEN can be used for restricted depositions.
"""

import argparse
import fnmatch
import glob
import io
import json
import os
import sys
import time
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
    "environment": METADATA_DIR / "zenodo_upload_summary_environment.json",
    "environment-nc": METADATA_DIR / "zenodo_upload_summary_environment_nc.json",
}

# The licence each layer is published under. The environment layers are split
# by licence rather than by subject: the models behind tree cover and deadwood
# both descend from NVIDIA's SegFormer, whose Source Code License permits
# research and evaluation use only, so anything derived from them is released
# non-commercially. Tree positions come from DeepForest (MIT) and the snow mask
# from a plain threshold, so neither carries that restriction and both stay
# under the same licence as the rest of the dataset.
LAYER_LICENCES = {
    "base": "CC-BY-4.0",
    "raw": "CC-BY-4.0",
    "matched": "CC-BY-4.0",
    "orthographic": "CC-BY-4.0",
    "owl-transferred": "CC-BY-4.0",
    "environment": "CC-BY-4.0",
    "environment-nc": "CC-BY-NC-4.0",
}
NON_COMMERCIAL = {"environment-nc"}

# Some versions are a LAYER on top of another rather than a standalone release.
# `owl-transferred` ships only the transferred RGB annotations, which are of no
# use without the recordings they annotate, so asking for it downloads the base
# recordings too and the two land side by side in one output directory.
# Layers are fetched in the order listed.
VERSION_LAYERS = {
    "owl-transferred": ["base", "owl-transferred"],
    "environment": ["base", "environment"],
    "environment-nc": ["base", "environment-nc"],
    # Everything, at the cost of a mixed licence -- see the warning printed
    # when this is selected.
    "environment-all": ["base", "environment", "environment-nc"],
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
    "environment": ["<id>_environment.json"],
    "environment-nc": ["<id>_environment_nc.json"],
}

# With --annotations-only the video and the masks are never fetched, so the
# base layer is "present" once its annotation files are.
ANNOTATION_MARKERS = dict(LAYER_MARKERS)
ANNOTATION_MARKERS["base"] = ["<id>_gt.txt", "<id>_matched_poses.json",
                              "<id>_metadata.json"]

# What --annotations-only leaves in the archive: anything that is imagery,
# plus the Windows thumbnail caches that slipped into some archives.
MEDIA_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".png", ".jpg", ".jpeg",
                  ".tif", ".tiff", ".bmp"}
JUNK_NAMES = {"thumbs.db", ".ds_store"}

# Layers whose archives do not carry the flight id in their file names (raw:
# `air_data.csv`, `T_calib.json`; matched / orthographic: `rgb/`, `thermal/`,
# `manifest.json`). One flight unpacks fine into its own directory, but two
# into the same one would overwrite each other, so with more than one flight
# each is unpacked into `<output>/<id>/` instead. The flight-prefixed layers
# stay flat, as their files are meant to sit side by side across layers.
NESTED_LAYERS = {"raw", "matched", "orthographic"}


def licence_of(version: str) -> str:
    """The effective licence of a version, given the layers it pulls."""
    layers = layers_of(version)
    if any(l in NON_COMMERCIAL for l in layers):
        return "CC-BY-NC-4.0 (mixed)"
    return "CC-BY-4.0"


def print_licence_notice(version: str, layers: list[str]) -> None:
    """Say plainly what may be done with what is about to be downloaded.

    A mixed download is the case worth being loud about: the files land in one
    directory and nothing about a mask file says which licence it came under,
    so the distinction has to be made here or it is lost.
    """
    nc = [l for l in layers if l in NON_COMMERCIAL]
    if not nc:
        return
    ok = [l for l in layers if l not in NON_COMMERCIAL]
    print()
    print("  " + "!" * 68)
    print("  !  This download is NOT uniformly licensed.")
    print("  !")
    for l in nc:
        print(f"  !  {l:<18} {LAYER_LICENCES.get(l, '?'):<16} "
              f"NON-COMMERCIAL USE ONLY")
    for l in ok:
        print(f"  !  {l:<18} {LAYER_LICENCES.get(l, '?'):<16} ")
    print("  !")
    print("  !  The non-commercial layers derive from models built on NVIDIA's")
    print("  !  SegFormer, licensed for research and evaluation only. Files from")
    print("  !  all layers land in the same directory, so if you redistribute or")
    print("  !  use this commercially, keep them apart.")
    print("  " + "!" * 68)


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
    nested: bool = False,
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

    if nested:
        # The flight has a directory of its own; anything in it means a
        # previous run got as far as extracting.
        return unzip_mode and zip_path.parent.is_dir() and \
            any(zip_path.parent.iterdir())

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


class RemoteFile(io.RawIOBase):
    """A file-like view of a URL, read with HTTP range requests.

    A zip archive keeps its table of contents at the very end and stores every
    member at a known offset, so `zipfile` only ever needs a handful of
    `seek` + `read` calls to pull one member out. Zenodo serves ranges, which
    turns "download 1.5 GB and keep 3 MB of it" into three or four small
    requests. Every read is one GET with a `Range` header; a transient error
    is retried with backoff.
    """

    def __init__(self, url: str, token: Optional[str] = None):
        self.url = url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.session = requests.Session()
        self.pos = 0
        probe = self.session.get(url, headers={**self.headers, "Range": "bytes=0-0"},
                                 stream=True)
        probe.raise_for_status()
        content_range = probe.headers.get("Content-Range", "")
        if probe.status_code != 206 or "/" not in content_range:
            raise RuntimeError("server does not honour HTTP range requests")
        self.size = int(content_range.rsplit("/", 1)[1])
        probe.close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        headers = {**self.headers, "Range": f"bytes={self.pos}-{end}"}
        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers=headers, timeout=120)
                r.raise_for_status()
                data = r.content
                self.pos += len(data)
                return data
            except (requests.RequestException, OSError) as e:
                last_error = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"range read failed after retries: {last_error}")


def member_selected(name: str, include: Optional[list[str]],
                    exclude: Optional[list[str]], annotations_only: bool) -> bool:
    """Whether the archive member *name* is wanted.

    A pattern matches either the full path inside the archive
    (`labels/1_rgb_mot.txt`) or just the file name (`1_rgb_mot.txt`), so
    `'*_gt.txt'` works whether or not the archive has directories.
    `<id>` in a pattern is not expanded here; the caller does that per flight.
    """
    base = name.rsplit("/", 1)[-1]

    def hit(patterns: list[str]) -> bool:
        # Case-insensitive: the raw archives say `.MP4` and `.SRT`, the
        # processed ones `.mp4`, and nobody should have to know which.
        return any(fnmatch.fnmatchcase(name.lower(), p.lower())
                   or fnmatch.fnmatchcase(base.lower(), p.lower())
                   for p in patterns)

    if annotations_only and (Path(base).suffix.lower() in MEDIA_SUFFIXES
                             or base.lower() in JUNK_NAMES):
        return False
    if include and not hit(include):
        return False
    if exclude and hit(exclude):
        return False
    return True


def list_members(url: str, token: Optional[str]) -> list[zipfile.ZipInfo]:
    """The table of contents of a remote archive, without downloading it."""
    with zipfile.ZipFile(RemoteFile(url, token)) as zf:
        return [i for i in zf.infolist() if not i.is_dir()]


def download_members(url: str, output_dir: Path, token: Optional[str],
                     select=lambda name: True) -> tuple[list[str], int]:
    """Extract the members of a remote archive that *select* accepts.

    Returns (names written, number skipped). Paths inside the archive are
    kept, exactly as a full download with --unzip would lay them out, so a
    selective download is always a subset of the full one. A member already on
    disk with the right size is skipped, which makes a rerun cheap and lets a
    second call add files to an earlier one.
    """
    remote = RemoteFile(url, token)
    written, skipped = [], 0
    root = output_dir.resolve()
    with zipfile.ZipFile(remote) as zf:
        for info in zf.infolist():
            if info.is_dir() or not select(info.filename):
                continue
            dest = (output_dir / info.filename).resolve()
            if root not in dest.parents:
                print(f"     ⚠  refusing to write outside {output_dir}: "
                      f"{info.filename}")
                continue
            if dest.exists() and dest.stat().st_size == info.file_size:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            written.append(info.filename)
    return written, skipped


def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def print_members(members: list[zipfile.ZipInfo], select) -> None:
    """Print an archive's contents, collapsing per-frame directories.

    The matched and orthographic archives hold thousands of frames each, so
    a directory with many files of one type is shown as a single line.
    """
    groups: dict[tuple[str, str], list[zipfile.ZipInfo]] = {}
    for m in members:
        d, _, base = m.filename.rpartition("/")
        groups.setdefault((d, Path(base).suffix), []).append(m)
    total = chosen = 0
    for (d, suffix), ms in sorted(groups.items()):
        size = sum(m.file_size for m in ms)
        picked = [m for m in ms if select(m.filename)]
        total += size
        chosen += sum(m.file_size for m in picked)
        mark = "✓" if len(picked) == len(ms) else ("~" if picked else " ")
        if len(ms) > 3:
            print(f"     {mark} {d + '/' if d else ''}*{suffix:<24} "
                  f"{len(ms):>6} files {format_size(size):>10}")
        else:
            for m in ms:
                print(f"     {mark} {m.filename:<40} "
                      f"{format_size(m.file_size):>10}")
    print(f"     selected {format_size(chosen)} of {format_size(total)}")


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
        # Some versions are composites with no summary of their own, so the
        # choices are the union of both tables.
        choices=sorted(set(VERSION_SUMMARIES) | set(VERSION_LAYERS)),
        default="base",
        help="Dataset version to download (default: base). "
             "'owl-transferred', 'environment' and 'environment-nc' are layers "
             "on the base release and pull the recordings too. "
             "'environment-all' pulls both environment layers and is therefore "
             "partly non-commercial -- see --licences.",
    )
    parser.add_argument(
        "--licences",
        action="store_true",
        help="Print the licence of every dataset version and exit.",
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
    parser.add_argument(
        "--annotations-only", "-a",
        action="store_true",
        help="Fetch only the annotation, pose and metadata files of each "
             "flight and never the video or the masks. The members are read "
             "straight out of the archive on Zenodo with HTTP range requests, "
             "a few MB per flight instead of the full archive, and land "
             "extracted in the output directory. Implies --unzip.",
    )
    parser.add_argument(
        "--include", "-i",
        nargs="+",
        metavar="PATTERN",
        help="Fetch only the archive members matching one of these glob "
             "patterns, e.g. '*_gt.txt' 'labels/*' '*.SRT'. A pattern "
             "matches the path inside the archive or the bare file name; "
             "'<id>' stands for the flight id. Works for every version, "
             "reads the members in place like --annotations-only, and "
             "implies --unzip. Quote the patterns so the shell leaves them "
             "alone.",
    )
    parser.add_argument(
        "--exclude", "-x",
        nargs="+",
        metavar="PATTERN",
        help="Skip the archive members matching one of these glob patterns "
             "(same rules as --include, and combinable with it and with "
             "--annotations-only).",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Show what each requested flight's archive contains, with sizes "
             "and which files --include / --exclude / --annotations-only would "
             "pick, without downloading anything. Needs a flight selection.",
    )
    args = parser.parse_args()
    # Anything that picks members out of an archive reads them in place with
    # range requests rather than downloading the whole ZIP.
    selective = bool(args.annotations_only or args.include or args.exclude)
    if selective:
        args.unzip = True

    if args.licences:
        print(f"\n{'version':<18}{'licence':<24}layers")
        for v in sorted(set(VERSION_SUMMARIES) | set(VERSION_LAYERS)):
            print(f"{v:<18}{licence_of(v):<24}{' + '.join(layers_of(v))}")
        print("\nNon-commercial layers: " + ", ".join(sorted(NON_COMMERCIAL)))
        print("They derive from models built on NVIDIA's SegFormer, which is\n"
              "licensed for research and evaluation use only.")
        return

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
    if not args.summary:
        print_licence_notice(args.version, layer_names)

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

    if args.list_files and not has_explicit_filter and not args.split:
        sys.exit("Error: --list-files needs a flight selection (-f, --range, "
                 "--parts or --split); listing every archive takes a while.")

    def selector(prefix: str):
        """The member filter for one flight, with `<id>` filled in."""
        inc = [p.replace("<id>", prefix) for p in args.include or []]
        exc = [p.replace("<id>", prefix) for p in args.exclude or []]
        return lambda name: member_selected(name, inc, exc,
                                            args.annotations_only)

    os.makedirs(args.output_dir, exist_ok=True)

    totals = {"downloaded": 0, "extracted": 0, "skipped": 0}
    failed: list[str] = []
    missing_in_layer: dict[str, list[str]] = {}

    for name in layer_names:
        this_index = layer_index[name]
        markers = (ANNOTATION_MARKERS if args.annotations_only
                   else LAYER_MARKERS).get(name)

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

        # Files without the flight id in their names need a directory per
        # flight as soon as there is more than one flight.
        nested = name in NESTED_LAYERS and len(prefixes) > 1

        def flight_dir(prefix: str) -> Path:
            return args.output_dir / prefix if nested else args.output_dir

        # ── List mode: show the archive contents, download nothing ───────────
        if args.list_files:
            file_maps: dict[int, dict[str, str]] = {}
            for prefix in wanted:
                info = this_index[prefix]
                dep_id = info["deposition_id"]
                try:
                    if dep_id not in file_maps:
                        file_maps[dep_id] = get_deposition_files(
                            api_base, dep_id, args.token)
                    url = file_maps[dep_id][info["zip_name"]]
                    print(f"\n  {info['zip_name']}  (part {info['part']}, "
                          f"deposition {dep_id})")
                    print_members(list_members(url, args.token),
                                  selector(prefix))
                except (requests.RequestException, KeyError, RuntimeError,
                        zipfile.BadZipFile) as e:
                    print(f"  ❌ {info['zip_name']}: cannot list ({e})")
                    failed.append(prefix)
            continue

        # ── Pre-filter: skip already downloaded / extracted flights ──────────
        # With --include / --exclude the flight's files are not known up
        # front, so every flight is opened and the members already on disk
        # are skipped one by one instead.
        to_download = []
        skipped_count = 0
        for prefix in wanted:
            if args.include or args.exclude:
                to_download.append(prefix)
                continue
            if args.annotations_only and not markers:
                to_download.append(prefix)
                continue
            if flight_already_exists(prefix, flight_dir(prefix), args.unzip,
                                     this_index[prefix]["zip_name"], markers,
                                     nested):
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
        if nested:
            print(f"📁 '{name}' archives do not carry the flight id in their "
                  f"file names: each flight goes to {args.output_dir}/<id>/")
        if selective:
            what = []
            if args.annotations_only:
                what.append("no video, masks or frames")
            if args.include:
                what.append("only " + " ".join(args.include))
            if args.exclude:
                what.append("not " + " ".join(args.exclude))
            print(f"✂  Selected files ({'; '.join(what)}) are read out of "
                  f"each archive in place, the rest stays on Zenodo")
        elif args.unzip:
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
                target = flight_dir(prefix)
                target.mkdir(parents=True, exist_ok=True)
                dest_path = target / zip_name

                if zip_name not in file_map:
                    print(f"  ❌ {zip_name} not found in deposition files")
                    failed.append(prefix)
                    continue

                print(f"  ⬇  {zip_name}")
                if selective:
                    try:
                        names, n_have = download_members(
                            file_map[zip_name], target, args.token,
                            selector(prefix))
                        shown = ", ".join(names[:8]) + \
                            (f" … (+{len(names) - 8})" if len(names) > 8 else "")
                        print(f"     ✂  {len(names)} file(s)"
                              + (f": {shown}" if names else "")
                              + (f", {n_have} already present" if n_have else ""))
                        if not names and not n_have:
                            print("     ⚠  no file in this archive matches "
                                  "the selection")
                        totals["downloaded"] += 1
                        totals["extracted"] += 1
                    except (requests.RequestException, zipfile.BadZipFile,
                            RuntimeError, OSError) as e:
                        print(f"     ❌ Download failed: {e}")
                        failed.append(prefix)
                    continue
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
                        n_files = extract_and_remove_zip(dest_path, target)
                        print(f"     📦 Extracted {n_files} file(s), ZIP removed")
                        totals["extracted"] += 1
                    except (zipfile.BadZipFile, OSError) as e:
                        print(f"     ⚠  Extraction failed: {e} (ZIP kept)")

    if args.list_files:
        for name, absent in missing_in_layer.items():
            print(f"\nℹ  no '{name}' archive for: {', '.join(absent)}")
        return

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
    # Repeated at the end as well as the start: on a long download the opening
    # notice has scrolled well out of sight by the time it finishes.
    if not args.summary:
        print_licence_notice(args.version, layer_names)


if __name__ == "__main__":
    main()