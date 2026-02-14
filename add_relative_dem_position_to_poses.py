"""
Add relative location offsets to drone pose files based on DEM origin metadata created with "dem_from_poses.py".

Converts WGS84 (lat/lng/alt) coordinates from pose files to the CRS defined in
the DEM metadata, then computes relative [x, y, z] offsets from the DEM origin.
"""

import argparse
import json
from pathlib import Path

from pyproj import Transformer


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def add_locations(poses: dict, dem_meta: dict) -> dict:
    """Add relative 'location' [x, y, z] to each image in poses."""
    crs = dem_meta["crs"]
    origin_x, origin_y, origin_z = dem_meta["origin"]

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    for image in poses["images"]:
        # pyproj with always_xy: (lng, lat) -> (easting, northing)
        easting, northing = transformer.transform(image["lng"], image["lat"])
        image["location"] = [
            round(easting - origin_x, 6),
            round(northing - origin_y, 6),
            round(image["alt"] - origin_z, 6),
        ]

    return poses


def process_pair(pose_path: Path, dem_path: Path, output_path: Path) -> None:
    poses = load_json(pose_path)
    dem_meta = load_json(dem_path)
    poses = add_locations(poses, dem_meta)
    save_json(output_path, poses)
    print(f"  {pose_path.name} -> {output_path}")


def match_files(pose_dir: Path, dem_dir: Path) -> list[tuple[Path, Path]]:
    """Match pose and DEM files by filename prefix (everything before the first '_')
    or by identical stems if prefixes don't match."""
    pose_files = sorted(pose_dir.glob("*_matched_poses.json"))
    dem_files = sorted(dem_dir.glob("*_matched_dem.json"))

    dem_lookup = {}
    for d in dem_files:
        # Use prefix before first underscore as key
        key = d.name.split("_")[0]
        dem_lookup[key] = d

    pairs = []
    for p in pose_files:
        key = p.name.split("_")[0]
        if key in dem_lookup:
            pairs.append((p, dem_lookup[key]))
        else:
            print(f"  WARNING: No matching DEM found for {p.name}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Add relative location offsets to drone pose files based on DEM origin."
    )
    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
        help="Single pose JSON file or folder of *_matched_poses.json files.",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        required=True,
        help="Single DEM metadata JSON file or folder of *_matched_dem.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder for modified pose files. If omitted, --inplace is required.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Modify pose files in place (only used when --output is not given).",
    )

    args = parser.parse_args()

    if args.output is None and not args.inplace:
        parser.error("Either --output or --inplace must be specified.")

    # Single file mode
    if args.poses.is_file():
        if not args.dem.is_file():
            parser.error("When --poses is a file, --dem must also be a file.")

        if args.inplace and args.output is None:
            out = args.poses
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            out = args.output / args.poses.name

        print("Processing single file:")
        process_pair(args.poses, args.dem, out)

    # Folder mode
    elif args.poses.is_dir():
        if not args.dem.is_dir():
            parser.error("When --poses is a folder, --dem must also be a folder.")

        pairs = match_files(args.poses, args.dem)
        if not pairs:
            print("No matching pose/DEM pairs found.")
            return

        print(f"Processing {len(pairs)} file pair(s):")
        for pose_path, dem_path in pairs:
            if args.inplace and args.output is None:
                out = pose_path
            else:
                args.output.mkdir(parents=True, exist_ok=True)
                out = args.output / pose_path.name

            process_pair(pose_path, dem_path, out)

    else:
        parser.error(f"--poses path does not exist: {args.poses}")

    print("Done.")


if __name__ == "__main__":
    main()