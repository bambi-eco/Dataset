#!/usr/bin/env python3
"""
Extends flight_metadata/splits.json to cover all available flights.

Usage:
    python extend_splits.py [--eps KM] [--seed N] [--dry-run] [--verbose]

Strategy
--------
1. Load all *_metadata.json files  -> full flight inventory.
2. Load GPS centroids from *_matched_poses.json (mean lat/lng per flight).
3. Build spatial clusters in two phases to avoid single-linkage chain merging:
     Phase A  - Group flights by the named location extracted from their
                SharePoint URL (e.g. "NOe_Purkersdorf").  Flights recorded
                at the same named area on different dates land in one group.
     Phase B  - Compute each group's GPS centroid.  Groups whose centroids
                are within EPS km are merged.  Because we compare GROUP
                centroids (not individual pairs), this is average-linkage-like
                and does not create chains.
4. Existing split assignments (splits.json) are preserved exactly.
   If a cluster already contains assigned flights the whole cluster inherits
   that split (majority vote on conflict, with a warning).
5. Entirely new clusters are assigned using stratified greedy balancing:
     - Each cluster is tagged with a (season, dominant_species) stratum.
     - Clusters are interleaved round-robin across strata so every stratum
       gets coverage in each split.
     - Within each step the split with the largest frame-count deficit
       relative to TARGET receives the next cluster.
6. Write the updated splits.json (or preview with --dry-run).

Target ratio: 70% train / 15% val / 15% test  (by frame count).
"""
import argparse
import json
import math
import os
import re
import random
import sys
from collections import defaultdict
from datetime import datetime

# ── Constants -----------------------------------------------------------------

METADATA_DIR = "flight_metadata"
POSES_DIR    = "poses"
SPLITS_FILE  = "flight_metadata/splits.json"
SPLITS       = ("train", "val", "test")
TARGET       = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_EPS_KM = 5.0
DEFAULT_SEED   = 42


# ── Helpers -------------------------------------------------------------------

def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def get_season(dt_str):
    month = datetime.fromisoformat(dt_str).month
    if month in (3, 4, 5):  return "spring"
    if month in (6, 7, 8):  return "summer"
    if month in (9, 10, 11): return "autumn"
    return "winter"


def extract_location_name(sharepoint_link):
    """Return the region+location part of the date-segment in a SharePoint URL.

    .../2023_07_27_NOe_Purkersdorf/...    -> "NOe_Purkersdorf"
    .../2024_10_31_OOe_Ansfelden_ASP/...  -> "OOe_Ansfelden_ASP"
    Returns "" when the pattern is absent.
    """
    m = re.search(r"/\d{4}_\d{2}_\d{2}_([^/]+)/", sharepoint_link)
    return m.group(1) if m else ""


# ── Data loading --------------------------------------------------------------

def load_all_metadata(meta_dir):
    """Return {flight_id (int): metadata_dict}."""
    flights = {}
    for fname in os.listdir(meta_dir):
        m = re.match(r"^(\d+)_metadata\.json$", fname)
        if not m:
            continue
        fid = int(m.group(1))
        with open(os.path.join(meta_dir, fname), encoding="utf-8") as fh:
            flights[fid] = json.load(fh)
    return flights


def load_centroid(fid, poses_dir):
    """Return (lat, lng) centroid for *fid*, or None if unavailable."""
    path = os.path.join(poses_dir, f"{fid}_matched_poses.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    imgs = [img for img in data.get("images", [])
            if "lat" in img and "lng" in img]
    if not imgs:
        return None
    lat = sum(img["lat"] for img in imgs) / len(imgs)
    lng = sum(img["lng"] for img in imgs) / len(imgs)
    if not (math.isfinite(lat) and math.isfinite(lng)):
        return None
    return lat, lng


# ── Union-Find ----------------------------------------------------------------

class UnionFind:
    def __init__(self, keys):
        self._p = {k: k for k in keys}

    def find(self, x):
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[ra] = rb

    def groups(self):
        result = defaultdict(list)
        for k in self._p:
            result[self.find(k)].append(k)
        return dict(result)


# ── Clustering ----------------------------------------------------------------

def group_centroid(fids, centroids):
    """Mean (lat, lng) of all flights in *fids* that have GPS data."""
    pts = [centroids[fid] for fid in fids if fid in centroids]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def build_clusters(flights, centroids, eps_km):
    """
    Two-phase spatial clustering that avoids single-linkage chaining.

    Phase A: group flights by exact location name from their SharePoint URL.
             Flights with no parseable location form singleton groups.
    Phase B: compute each group's GPS centroid; merge groups whose centroids
             are within eps_km.  Groups without GPS are left as-is.

    Returns {cluster_root (int): [flight_ids]}.
    """
    # --- Phase A: location-name groups ---------------------------------------
    loc_to_fids = defaultdict(list)   # location_name -> [fid, ...]
    fid_to_loc  = {}

    for fid, meta in flights.items():
        link = meta.get("flight_info", {}).get("sharepoint_link", "")
        loc  = extract_location_name(link)
        fid_to_loc[fid] = loc if loc else f"__singleton_{fid}__"
        loc_to_fids[fid_to_loc[fid]].append(fid)

    # Assign a representative (smallest fid) per location group
    loc_rep = {loc: min(fids) for loc, fids in loc_to_fids.items()}

    # Initialise Union-Find over location representatives
    uf_locs = UnionFind(loc_rep.values())

    # --- Phase B: merge location groups by GPS centroid distance -------------
    loc_centroid = {}
    for loc, fids in loc_to_fids.items():
        c = group_centroid(fids, centroids)
        if c is not None:
            loc_centroid[loc] = c

    gps_locs = list(loc_centroid.keys())
    for i in range(len(gps_locs)):
        for j in range(i + 1, len(gps_locs)):
            la, lb = gps_locs[i], gps_locs[j]
            lat_a, lng_a = loc_centroid[la]
            lat_b, lng_b = loc_centroid[lb]
            if haversine_km(lat_a, lng_a, lat_b, lng_b) <= eps_km:
                uf_locs.union(loc_rep[la], loc_rep[lb])

    # Map every flight id to its final cluster root
    flight_uf = UnionFind(flights.keys())
    for fid in flights:
        loc  = fid_to_loc[fid]
        rep  = loc_rep[loc]
        root = uf_locs.find(rep)
        # Attach all flights of this location to the cluster root flight
        # We use the root *location representative* as the cluster key
        flight_uf.union(fid, root)

    return flight_uf.groups()


# ── Flight feature extraction -------------------------------------------------

def dominant_species(meta):
    """Common name of the species with the most labels, or 'none'."""
    sp = meta.get("species_present", {})
    if not sp:
        return "none"
    best = max(sp.values(), key=lambda v: v.get("num_labels", 0))
    name = best.get("species_name", "unknown")
    m = re.search(r"\(([^)]+)\)", name)
    return m.group(1) if m else name.split()[0]


def cluster_stratum(fids, flights):
    """Return (season, species) majority-vote stratum for a cluster."""
    seasons, species = [], []
    for fid in fids:
        meta  = flights[fid]
        start = meta.get("flight_info", {}).get("start_time")
        if start:
            seasons.append(get_season(start))
        species.append(dominant_species(meta))
    season = max(set(seasons), key=seasons.count) if seasons else "unknown"
    sp     = max(set(species), key=species.count) if species else "unknown"
    return season, sp


def cluster_frame_count(fids, flights):
    return sum(flights[fid].get("frame_count", 0) for fid in fids)


def flight_bbox_count(meta):
    """Total number of bounding box instances across all species in one flight."""
    return sum(sp.get("num_label_states", 0)
               for sp in meta.get("species_present", {}).values())


# ── Split assignment ----------------------------------------------------------

def assign_new_clusters(new_clusters, flights, frame_counts, rng):
    """
    Assign each new cluster to a split using stratified greedy balancing.
    Returns {cluster_root: split_name}.
    """
    # Tag each cluster
    strata = defaultdict(list)
    for root, fids in new_clusters.items():
        stratum = cluster_stratum(fids, flights)
        frames  = cluster_frame_count(fids, flights)
        strata[stratum].append((root, frames))

    # Shuffle within each stratum for reproducibility
    for v in strata.values():
        rng.shuffle(v)

    # Round-robin across strata in random order
    strata_order = sorted(strata.keys())
    rng.shuffle(strata_order)
    queues     = {k: list(v) for k, v in strata.items()}
    interleaved = []
    while any(queues[k] for k in strata_order):
        for k in strata_order:
            if queues[k]:
                interleaved.append(queues[k].pop(0))

    # Greedy: always feed the split with the largest frame-count deficit
    counts      = dict(frame_counts)
    assignments = {}
    for root, frames in interleaved:
        total      = max(1, sum(counts.values()))
        best_split = max(SPLITS, key=lambda s: TARGET[s] - counts[s] / total)
        assignments[root] = best_split
        counts[best_split] += max(frames, 1)

    return assignments


# ── Main ----------------------------------------------------------------------

def say(msg):
    """Print ensuring UTF-8 on all platforms."""
    print(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Extend flight_metadata/splits.json to cover all available flights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eps", type=float, default=DEFAULT_EPS_KM, metavar="KM",
        help="Cluster-centroid merge radius in km (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print assignments and statistics but do NOT write splits.json",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--meta-dir",    default=METADATA_DIR)
    parser.add_argument("--poses-dir",   default=POSES_DIR)
    parser.add_argument("--splits-file", default=SPLITS_FILE)
    args = parser.parse_args()
    # args.dry_run = True

    rng = random.Random(args.seed)

    # -- Load data -------------------------------------------------------------
    say("Loading metadata ...")
    flights    = load_all_metadata(args.meta_dir)
    all_fids   = sorted(flights.keys())
    say(f"  {len(all_fids)} flights found")

    say("Loading GPS centroids ...")
    centroids = {}
    for fid in all_fids:
        c = load_centroid(fid, args.poses_dir)
        if c is not None:
            centroids[fid] = c
    say(f"  {len(centroids)} / {len(all_fids)} flights have GPS data")

    with open(args.splits_file, encoding="utf-8") as fh:
        splits_json = json.load(fh)

    existing = {}
    for split_name, fids in splits_json.items():
        for fid in fids:
            existing[fid] = split_name

    missing_fids = [fid for fid in all_fids if fid not in existing]
    say(f"  {len(existing)} flights already in splits.json, "
        f"{len(missing_fids)} missing")

    # -- Cluster ---------------------------------------------------------------
    say(f"Building spatial clusters (eps = {args.eps} km) ...")
    clusters = build_clusters(flights, centroids, args.eps)
    say(f"  {len(clusters)} location clusters formed")

    if args.verbose:
        say("\n  Cluster overview:")
        for root, fids in sorted(clusters.items(), key=lambda x: -len(x[1])):
            locs = sorted({extract_location_name(
                               flights[f].get("flight_info", {}).get("sharepoint_link", ""))
                           for f in fids} - {""})
            say(f"    cluster {root:4d}: {len(fids):3d} flights | {locs}")

    # -- Determine existing split per cluster ----------------------------------
    cluster_split   = {}
    conflict_list   = []
    for root, fids in clusters.items():
        assigned = {existing[fid] for fid in fids if fid in existing}
        if not assigned:
            continue
        if len(assigned) == 1:
            cluster_split[root] = next(iter(assigned))
        else:
            votes    = [existing[fid] for fid in fids if fid in existing]
            majority = max(set(votes), key=votes.count)
            cluster_split[root] = majority
            conflict_list.append((root, sorted(fids), assigned))

    if conflict_list:
        say(f"\n  WARNING: {len(conflict_list)} cluster(s) span multiple "
            "existing splits (majority vote applied):")
        for root, fids, splits_seen in conflict_list:
            say(f"    cluster root={root}, flights={fids}, "
                f"existing splits={splits_seen}")

    # -- Separate new flights --------------------------------------------------
    partially_new   = []          # (fid, inherited_split)
    fully_new_clusters = {}       # root -> [fids]  (zero existing assignments)

    for root, fids in clusters.items():
        if root in cluster_split:
            for fid in fids:
                if fid not in existing:
                    partially_new.append((fid, cluster_split[root]))
        else:
            fully_new_clusters[root] = fids

    say(f"  {len(fully_new_clusters)} fully-new clusters to assign")
    say(f"  {len(partially_new)} new flights inheriting cluster split")

    # -- Greedy assignment for fully-new clusters ------------------------------
    frame_counts = {sp: 0 for sp in SPLITS}
    for sp in SPLITS:
        for fid in splits_json.get(sp, []):
            frame_counts[sp] += flights.get(fid, {}).get("frame_count", 0)

    new_assignments = assign_new_clusters(
        fully_new_clusters, flights, frame_counts, rng)

    # -- Build final splits ----------------------------------------------------
    final = {sp: set(splits_json.get(sp, [])) for sp in SPLITS}

    for fid, sp in partially_new:
        final[sp].add(fid)

    for root, sp in new_assignments.items():
        for fid in fully_new_clusters[root]:
            final[sp].add(fid)

    final_sorted = {sp: sorted(final[sp]) for sp in SPLITS}

    # -- Statistics ------------------------------------------------------------
    total_flights = len(all_fids)
    total_frames  = max(1, sum(f.get("frame_count", 0) for f in flights.values()))
    total_bboxes  = max(1, sum(flight_bbox_count(f) for f in flights.values()))

    say("\n=== Final split statistics ===")
    say(f"  {'':5s}  {'flights':>8s} {'':>6s}  {'frames':>8s} {'':>6s}  "
        f"{'bboxes':>8s} {'':>6s}")
    say(f"  {'':5s}  {'abs':>8s} {'rel':>6s}  {'abs':>8s} {'rel':>6s}  "
        f"{'abs':>8s} {'rel':>6s}")
    say("  " + "-" * 57)
    for sp in SPLITS:
        fids = final_sorted[sp]
        fc   = sum(flights.get(fid, {}).get("frame_count", 0) for fid in fids)
        bc   = sum(flight_bbox_count(flights.get(fid, {}))     for fid in fids)
        say(f"  {sp:5s}  {len(fids):>8d} {100*len(fids)/total_flights:>5.1f}%  "
            f"{fc:>8d} {100*fc/total_frames:>5.1f}%  "
            f"{bc:>8d} {100*bc/total_bboxes:>5.1f}%")
    say("  " + "-" * 57)
    say(f"  {'total':5s}  {total_flights:>8d} {'100.0%':>6s}  "
        f"{total_frames:>8d} {'100.0%':>6s}  "
        f"{total_bboxes:>8d} {'100.0%':>6s}")

    # Newly assigned summary
    newly_by_split = defaultdict(list)
    for fid in missing_fids:
        for sp in SPLITS:
            if fid in final[sp]:
                newly_by_split[sp].append(fid)
                break

    total_new = sum(len(v) for v in newly_by_split.values())
    say(f"\n=== Newly assigned: {total_new} flights ===")
    for sp in SPLITS:
        new_fids = sorted(newly_by_split[sp])
        if not new_fids:
            continue
        say(f"  {sp}: {new_fids}")
        if args.verbose:
            for fid in new_fids:
                meta   = flights[fid]
                link   = meta.get("flight_info", {}).get("sharepoint_link", "")
                loc    = extract_location_name(link)
                start  = meta.get("flight_info", {}).get("start_time", "?")
                season = get_season(start) if start != "?" else "?"
                sp_n   = dominant_species(meta)
                frames = meta.get("frame_count", 0)
                gps    = "GPS" if fid in centroids else "no-GPS"
                say(f"      [{fid:4d}] {loc:<32s} {season:<7s} "
                    f"{sp_n:<22s} {frames:4d} frames  {gps}")

    if args.verbose and new_assignments:
        say("\n=== Fully-new cluster assignments ===")
        for root, sp in sorted(new_assignments.items()):
            fids   = sorted(fully_new_clusters[root])
            season, sp_n = cluster_stratum(fids, flights)
            frames = cluster_frame_count(fids, flights)
            locs   = sorted({extract_location_name(
                                 flights[f].get("flight_info", {}).get("sharepoint_link", ""))
                             for f in fids} - {""})
            say(f"  -> {sp:5s} | root={root} flights={fids} | "
                f"{season}/{sp_n} | {frames} frames | locs={locs}")

    # -- Write output ----------------------------------------------------------
    if args.dry_run:
        say("\n[dry-run] splits.json NOT written.")
    else:
        with open(args.splits_file, "w", encoding="utf-8") as fh:
            json.dump(final_sorted, fh, indent=2)
        say(f"\nWrote {args.splits_file}")

    # Sanity check
    covered   = {fid for sp in SPLITS for fid in final_sorted[sp]}
    uncovered = [fid for fid in all_fids if fid not in covered]
    if uncovered:
        say(f"\nWARNING: {len(uncovered)} flights still unassigned: {uncovered}")
    else:
        say("All flights successfully assigned to a split.")


if __name__ == "__main__":
    main()
