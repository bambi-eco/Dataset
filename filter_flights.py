#!/usr/bin/env python3
"""
Filter BAMBI flight metadata files by species, occlusion, sex, age, weather, date range, and drone name.

Usage examples:
    # Find flights containing Roe deer
    python filter_flights.py /path/to/metadata --species "Capreolus capreolus"

    # Find flights with occluded frames of any species
    python filter_flights.py /path/to/metadata --occlusion true

    # Find flights with visible (unoccluded) frames only
    python filter_flights.py /path/to/metadata --occlusion false

    # Find flights in sunny AND windy weather
    python filter_flights.py /path/to/metadata --weather sunny windy

    # Find flights between two dates with female animals
    python filter_flights.py /path/to/metadata --min-date 2024-10-01 --max-date 2024-11-01 --sex female

    # Find flights by drone name
    python filter_flights.py /path/to/metadata --drone "Matric 30 Thermal"

    # Combine multiple filters (AND logic)
    python filter_flights.py /path/to/metadata --species "Homo sapiens" --occlusion false --sex female --weather sunny
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

WEATHER_FLAGS = {
    "night":       0b000000,
    "sunny":       0b000001,
    "cloudy":      0b000010,
    "rainy":       0b000100,
    "snowy":       0b001000,
    "windy":       0b010000,
    "foggy":       0b100000,
}

WEATHER_NAMES = {v: k for k, v in WEATHER_FLAGS.items() if v != 0}


def decode_weather(value: int) -> list[str]:
    """Decode a weather integer into a list of flag names."""
    if value == 0:
        return ["night"]
    flags = []
    for bit_val, name in sorted(WEATHER_NAMES.items()):
        if value & bit_val:
            flags.append(name)
    return flags


def encode_weather(names: list[str]) -> int:
    """Encode a list of weather flag names into a bitmask."""
    mask = 0
    for name in names:
        key = name.lower()
        if key not in WEATHER_FLAGS:
            raise ValueError(f"Unknown weather flag: '{name}'. Valid: {list(WEATHER_FLAGS.keys())}")
        mask |= WEATHER_FLAGS[key]
    return mask


def load_metadata(folder: Path) -> list[dict]:
    """Load all *_metadata.json files from a folder."""
    files = sorted(folder.glob("*_metadata.json"))
    if not files:
        print(f"Warning: No *_metadata.json files found in {folder}", file=sys.stderr)
    records = []
    for f in files:
        try:
            with open(f, "r") as fh:
                records.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load {f}: {e}", file=sys.stderr)
    return records


def matches_species_filter(record: dict, species_query: str) -> list[dict]:
    """Return species entries whose name contains the query (case-insensitive)."""
    query = species_query.lower()
    matches = []
    for sp_data in record.get("species_present", {}).values():
        if query in sp_data.get("species_name", "").lower() or query == sp_data.get("wikidata_id", "").lower():
            matches.append(sp_data)
    return matches


def filter_records(records: list[dict], args) -> list[str]:
    """Apply all filters and return matching flight keys."""
    matching_keys = []

    for record in records:
        flight_key = record.get("flight_key", "?")
        flight_info = record.get("flight_info", {})
        species_present = record.get("species_present", {})

        # --- Flight-level filters ---

        # Weather filter: all requested flags must be set
        if args.weather:
            required_mask = encode_weather(args.weather)
            actual_weather = flight_info.get("weather", 0)
            if actual_weather is None or (actual_weather & required_mask) == 0:
                continue

        # Date range filter
        start_time_str = flight_info.get("start_time")
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
            except ValueError:
                start_time = None
        else:
            start_time = None

        if args.min_date:
            if start_time is None or start_time.date() < args.min_date:
                continue
        if args.max_date:
            if start_time is None or start_time.date() > args.max_date:
                continue

        # Drone name filter (OR logic, case-insensitive substring)
        if args.drone:
            drone_name = flight_info.get("drone_name", "")
            if not any(d.lower() in drone_name.lower() for d in args.drone):
                continue

        # --- Species-level filters ---
        # Determine candidate species entries (OR: any listed species matches)
        if args.species:
            candidates = []
            for sp_query in args.species:
                candidates.extend(matches_species_filter(record, sp_query))
            if not candidates:
                continue
        else:
            candidates = list(species_present.values())

        if not candidates:
            continue

        # Apply species-attribute filters: at least one candidate must satisfy ALL
        def species_matches(sp: dict) -> bool:
            if args.occlusion == "true" and not sp.get("contains_occluded_frames", False):
                return False
            if args.occlusion == "false" and not sp.get("contains_unoccluded_frames", False):
                return False
            if args.sex:
                sex_map = {
                    "unknown": "contains_unknown_sex",
                    "male": "contains_male",
                    "female": "contains_female",
                }
                if not any(sp.get(sex_map[s], False) for s in args.sex):
                    return False
            if args.age:
                age_map = {
                    "unknown": "contains_unknown_age",
                    "juvenile": "contains_juvenile",
                    "adult": "contains_adult",
                }
                if not any(sp.get(age_map[a], False) for a in args.age):
                    return False
            return True

        if any(species_matches(sp) for sp in candidates):
            matching_keys.append(flight_key)

    return matching_keys


def main():
    parser = argparse.ArgumentParser(
        description="Filter BAMBI flight metadata by species, attributes, weather, date, and drone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--folder", default=Path(r"./flight_metadata"), type=Path, help="Folder containing *_metadata.json files")

    # Species filter (OR logic: any listed species must be present)
    parser.add_argument("--species", nargs="+", type=str, default=None,
                        help="Filter by species name or Wikidata ID (OR logic, case-insensitive substring match)")

    # Occlusion filter
    parser.add_argument("--occlusion", type=str, default=None,
                        choices=["true", "false"],
                        help="Filter by occlusion: true = occluded frames, false = visible/unoccluded frames, omit = all")

    # Sex filter (OR logic within, i.e. --sex male female = male OR female)
    parser.add_argument("--sex", nargs="+", choices=["unknown", "male", "female"], default=None,
                        help="Filter by sex (OR logic: any listed sex must be present)")

    # Age filter (OR logic within)
    parser.add_argument("--age", nargs="+", choices=["unknown", "juvenile", "adult"], default=None,
                        help="Filter by age (OR logic: any listed age must be present)")

    # Weather filter (AND logic: all listed flags must be set)
    parser.add_argument("--weather", nargs="+",
                        choices=list(WEATHER_FLAGS.keys()),
                        default=None,
                        help="Filter by weather flags (AND logic: all listed flags must be set)")

    # Date range
    parser.add_argument("--min-date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                        default=None, help="Minimum start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--max-date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                        default=None, help="Maximum start date (YYYY-MM-DD, inclusive)")

    # Drone filter (OR logic: any listed drone must match)
    parser.add_argument("--drone", nargs="+", type=str, default=None,
                        help="Filter by drone name (OR logic, case-insensitive substring match)")

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed info for each match")
    parser.add_argument("--count", action="store_true",
                        help="Only print the count of matching flights")

    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: '{args.folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    records = load_metadata(args.folder)
    if not records:
        print("No metadata records loaded.", file=sys.stderr)
        sys.exit(1)

    # Print applied filters
    print("=" * 60)
    print("Applied filters:")
    print("-" * 60)
    active = False
    if args.species:
        print(f"  Species:    {', '.join(args.species)}  (OR, substring match)")
        active = True
    if args.occlusion is not None:
        label = "occluded" if args.occlusion == "true" else "visible (unoccluded)"
        print(f"  Occlusion:  {label}")
        active = True
    if args.sex:
        print(f"  Sex:        {', '.join(args.sex)}  (OR)")
        active = True
    if args.age:
        print(f"  Age:        {', '.join(args.age)}  (OR)")
        active = True
    if args.weather:
        mask = encode_weather(args.weather)
        print(f"  Weather:    {', '.join(args.weather)}  (AND, mask={mask:#010b})")
        active = True
    if args.min_date:
        print(f"  Min date:   {args.min_date}")
        active = True
    if args.max_date:
        print(f"  Max date:   {args.max_date}")
        active = True
    if args.drone:
        print(f"  Drone:      {', '.join(args.drone)}  (OR, substring match)")
        active = True
    if not active:
        print("  (none — returning all flights)")
        print("=" * 60)
        print([int(x["flight_key"]) for x in records])
        return

    print("=" * 60)
    print()

    matching_keys = filter_records(records, args)

    if args.count:
        print(len(matching_keys))
    elif args.verbose:
        # Re-iterate to print details
        records_by_key = {r["flight_key"]: r for r in records}
        for key in matching_keys:
            r = records_by_key[key]
            fi = r.get("flight_info", {})
            weather_val = fi.get("weather", 0)
            weather_str = ", ".join(decode_weather(weather_val))
            species_names = [sp["species_name"] for sp in r.get("species_present", {}).values()]
            print(f"Flight {key}:")
            print(f"  Drone:   {fi.get('drone_name', 'N/A')}")
            print(f"  Date:    {fi.get('start_time', 'N/A')}")
            print(f"  Weather: {weather_str} ({weather_val})")
            print(f"  Species: {', '.join(species_names)}")
            print(f"  Frames:  {r.get('frame_count', 'N/A')}")
            print()
    else:
        print(matching_keys)

    print(f"\n{len(matching_keys)} / {len(records)} flights matched.", file=sys.stderr)


if __name__ == "__main__":
    main()