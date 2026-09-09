#!/usr/bin/env python3
"""
Overview map of every BAMBI flight.

The dataset was recorded on a few dozen sites, and many flights sit on top of
each other: 130 of the 386 flights are from one forest near Purkersdorf. Drawing
one marker per flight therefore says nothing. This script aggregates flights
that were recorded within a given distance of each other into **sites** and
draws one proportional symbol per site.

Positions come from the flight logs, in this order:

* ``<id>_matched_poses.json`` from the base release -- the poses of the video
  frames, which is what the rest of the toolchain uses;
* ``air_data.csv`` from the raw release, read from a per-flight subfolder.

Neither carries a video, so the whole dataset can be summarised from a few
hundred MB::

    python download_from_zenodo.py --annotations-only --range 0 400 -o poses/
    python flight_overview_map.py --poses poses/ -o figures/flight_overview.png

Flights whose log is not at hand are still placed: ``flight_metadata/`` names
the recording site of every flight, so a flight without coordinates joins the
site its campaign-mates were located at. The figure says how many were placed
that way.

Outputs, all optional except the figure:

* the map (``-o`` -- .png, .pdf or .svg);
* ``--geojson`` one point per site with its properties, ready for QGIS;
* ``--csv`` the same table as text;
* ``--flight-csv`` the per-flight summary, which doubles as the cache.

The state boundaries are Natural Earth (public domain). The script downloads
them once into ``~/.cache/bambi_overview`` and works without them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "bambi_overview"
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
          "geojson/ne_10m_admin_1_states_provinces.geojson")
NE_RAW = "ne_10m_admin_1_states_provinces.geojson"
BOUNDARY_FILE = "austria_states.geojson"
METRIC_CRS = "EPSG:3416"          # ETRS89 / Austria Lambert, metres

# Palette: one hue for the marks, ink and surfaces for everything else.
THEME = {
    "light": dict(surface="#fcfcfb", panel="#f3f2ef", ink="#0b0b0b", ink2="#52514e", ink3="#8a8985",
                  land="#e7e6e1", border="#c9c8c2", ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
                  mark="#2a78d6", ring="#fcfcfb"),
    "dark": dict(surface="#1a1a19", panel="#242422", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8985",
                 land="#2f2f2c", border="#4a4a45", ramp=["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
                 mark="#3987e5", ring="#1a1a19"),
}

# Region tokens used in the recording folder names.
REGIONS = {"NOe": "Lower Austria", "OOe": "Upper Austria", "Ktn": "Carinthia", "Stm": "Styria",
           "Sbg": "Salzburg", "T": "Tyrol", "Tirol": "Tyrol", "Bgld": "Burgenland", "W": "Vienna"}


# ============================================================================
# Per-flight summary
# ============================================================================

@dataclass
class Flight:
    flight: str
    lat: float = float("nan")
    lon: float = float("nan")
    n_positions: int = 0
    extent_m: float = 0.0
    alt_min: float = float("nan")
    alt_max: float = float("nan")
    duration_s: float = float("nan")
    source: str = ""
    start: str = ""
    drone: str = ""
    split: str = ""
    place: str = ""
    folder: str = ""
    key_frames: int = 0
    species: list = field(default_factory=list)

    @property
    def located(self) -> bool:
        return not math.isnan(self.lat)


def _stats(lats, lons, alts, times, flight, source) -> Flight:
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    ok = np.isfinite(lats) & np.isfinite(lons) & (np.abs(lats) > 1e-6)
    lats, lons = lats[ok], lons[ok]
    if len(lats) == 0:
        return Flight(flight=flight, source=source)
    lat, lon = float(np.median(lats)), float(np.median(lons))
    m_lat, m_lon = 111320.0, 111320.0 * math.cos(math.radians(lat))
    extent = float(math.hypot((lats.max() - lats.min()) * m_lat, (lons.max() - lons.min()) * m_lon))
    alts = np.asarray([a for a in alts if a is not None and np.isfinite(a)], float)
    dur = float("nan")
    if len(times) > 1:
        try:
            t0, t1 = min(times), max(times)
            dur = (t1 - t0).total_seconds()
        except TypeError:
            pass
    return Flight(flight=flight, lat=lat, lon=lon, n_positions=int(len(lats)), extent_m=extent,
                  alt_min=float(alts.min()) if len(alts) else float("nan"),
                  alt_max=float(alts.max()) if len(alts) else float("nan"),
                  duration_s=dur, source=source,
                  start=min(times).isoformat() if times else "")


def read_poses(path: Path) -> Flight:
    """Summarise a ``<id>_matched_poses.json`` from the base release."""
    flight = path.name.split("_")[0]
    with open(path) as f:
        data = json.load(f)
    images = data.get("images", [])
    lats, lons, alts, times = [], [], [], []
    for im in images:
        lats.append(im.get("lat")); lons.append(im.get("lng", im.get("lon")))
        alts.append(im.get("alt"))
        ts = im.get("timestamp")
        if ts:
            try:
                times.append(datetime.fromisoformat(ts).replace(tzinfo=None))
            except ValueError:
                pass
    return _stats(lats, lons, alts, times, flight, "poses")


def read_airdata(path: Path, flight: str) -> Flight:
    """Summarise the recording part of an ``air_data.csv`` from the raw release."""
    rows = [{(k or "").strip(): v for k, v in r.items()}
            for r in csv.DictReader(open(path, encoding="utf-8", errors="replace"))]
    if not rows:
        return Flight(flight=flight, source="airdata")
    video = [r for r in rows if r.get("isVideo") == "1"] or rows
    lats = [float(r["latitude"]) for r in video if r.get("latitude")]
    lons = [float(r["longitude"]) for r in video if r.get("longitude")]
    alts = [float(r["altitude_above_seaLevel(feet)"]) * 0.3048 for r in video
            if r.get("altitude_above_seaLevel(feet)")]
    times = []
    try:
        t0 = datetime.strptime(rows[0]["datetime(utc)"], "%Y-%m-%d %H:%M:%S")
        ms = [float(r["time(millisecond)"]) for r in video]
        times = [t0, t0]
        f = Flight(flight=flight)
        f.duration_s = (max(ms) - min(ms)) / 1000.0
    except (KeyError, ValueError):
        pass
    out = _stats(lats, lons, alts, [], flight, "airdata")
    if times:
        out.duration_s = (max(ms) - min(ms)) / 1000.0
        out.start = t0.isoformat()
    return out


def read_metadata(folder: Path) -> dict:
    """flight id -> split, drone, place, campaign folder, key frames, species."""
    meta = {}
    for p in sorted(folder.glob("*_metadata.json")):
        try:
            d = json.load(open(p))
        except json.JSONDecodeError:
            continue
        info = d.get("flight_info", {})
        link = urllib.parse.unquote(info.get("sharepoint_link") or "")
        m = re.search(r"/Processed/[^/]+/([^/]+)/", link)
        campaign = m.group(1) if m else ""
        place = re.sub(r"^\d{4}_\d{2}_\d{2}_?", "", campaign)
        meta[str(d.get("flight_key", p.name.split("_")[0]))] = dict(
            split=d.get("split", ""), drone=info.get("drone_name", ""), place=place, folder=campaign,
            key_frames=int(d.get("frame_count") or 0), start=info.get("start_time", ""),
            species=sorted({v.get("species_name", "") for v in (d.get("species_present") or {}).values()}),
        )
    return meta


def collect(pose_dirs, raw_dirs, meta_dir: Path, cache: Optional[Path], refresh: bool) -> list[Flight]:
    known: dict[str, Flight] = {}
    if cache and cache.exists() and not refresh:
        for r in csv.DictReader(open(cache)):
            f = Flight(flight=r["flight"], source=r.get("source", ""))
            for k in ("lat", "lon", "extent_m", "alt_min", "alt_max", "duration_s"):
                setattr(f, k, float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan"))
            f.n_positions = int(r.get("n_positions") or 0)
            known[f.flight] = f
        print(f"cache: {len(known)} flights from {cache}")

    for d in pose_dirs:
        for p in sorted(Path(d).rglob("*_poses.json")):
            fid = p.name.split("_")[0]
            if fid in known and known[fid].located:
                continue
            known[fid] = read_poses(p)
    for d in raw_dirs:
        d = Path(d)
        cands = [(p.parent.name, p) for p in sorted(d.rglob("air_data.csv"))]
        for fid, p in cands:
            if not fid.isdigit():
                continue
            if fid in known and known[fid].located:
                continue
            known[fid] = read_airdata(p, fid)

    meta = read_metadata(meta_dir) if meta_dir and meta_dir.is_dir() else {}
    for fid, m in meta.items():
        f = known.setdefault(fid, Flight(flight=fid))
        f.split, f.drone, f.place, f.folder = m["split"], m["drone"], m["place"], m["folder"]
        f.key_frames, f.species = m["key_frames"], m["species"]
        f.start = f.start or m["start"]
    flights = sorted(known.values(), key=lambda f: int(f.flight) if f.flight.isdigit() else 0)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["flight", "source", "n_positions", "lat", "lon", "extent_m", "alt_min", "alt_max",
                        "duration_s", "start", "drone", "split", "place", "folder", "key_frames", "species"])
            for f in flights:
                w.writerow([f.flight, f.source, f.n_positions, f"{f.lat:.7f}", f"{f.lon:.7f}",
                            f"{f.extent_m:.1f}", f"{f.alt_min:.1f}", f"{f.alt_max:.1f}",
                            f"{f.duration_s:.1f}", f.start, f.drone, f.split, f.place, f.folder,
                            f.key_frames, "|".join(f.species)])
        print(f"wrote {cache}")
    return flights


# ============================================================================
# Aggregation into sites
# ============================================================================

@dataclass
class Site:
    sid: int
    lat: float
    lon: float
    flights: list = field(default_factory=list)
    inferred: list = field(default_factory=list)     # flights placed via their campaign, no own log

    @property
    def n(self) -> int:
        return len(self.flights) + len(self.inferred)

    @property
    def label(self) -> str:
        names = Counter(f.place for f in self.flights + self.inferred if f.place)
        return pretty_place(names.most_common(1)[0][0]) if names else f"site {self.sid}"

    @property
    def region(self) -> str:
        toks = Counter(f.place.split("_")[0] for f in self.flights + self.inferred if f.place)
        return REGIONS.get(toks.most_common(1)[0][0], "") if toks else ""

    @property
    def species(self) -> list:
        s = set()
        for f in self.flights + self.inferred:
            s.update(x for x in f.species if x)
        return sorted(s)

    @property
    def key_frames(self) -> int:
        return sum(f.key_frames for f in self.flights + self.inferred)

    @property
    def hours(self) -> float:
        d = [f.duration_s for f in self.flights if np.isfinite(f.duration_s)]
        return float(np.sum(d) / 3600.0) if d else float("nan")

    @property
    def dates(self) -> list:
        return sorted({f.start[:10] for f in self.flights + self.inferred if f.start})

    @property
    def spread_m(self) -> float:
        if len(self.flights) < 2:
            return 0.0
        m_lat, m_lon = 111320.0, 111320.0 * math.cos(math.radians(self.lat))
        d = [math.hypot((f.lat - self.lat) * m_lat, (f.lon - self.lon) * m_lon) for f in self.flights]
        return float(max(d))


def pretty_place(place: str) -> str:
    """'Stm_Feldbach_Lembach' -> 'Feldbach Lembach'; drop the region and owner tokens."""
    parts = [p for p in place.split("_") if p and p not in REGIONS and p not in ("ASP", "Gatter", "FH")]
    return " ".join(parts) if parts else place


def to_metric(lats, lons):
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
        x, y = tr.transform(np.asarray(lons, float), np.asarray(lats, float))
        return np.asarray(x), np.asarray(y)
    except ImportError:                                  # local equirectangular fallback
        lat0 = float(np.mean(lats))
        return (np.asarray(lons, float) * 111320.0 * math.cos(math.radians(lat0)),
                np.asarray(lats, float) * 111320.0)


def cluster(flights: list[Flight], radius_m: float) -> list[Site]:
    """Single-linkage clustering: flights closer than ``radius_m`` share a site."""
    located = [f for f in flights if f.located]
    if not located:
        return []
    x, y = to_metric([f.lat for f in located], [f.lon for f in located])
    n = len(located)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(x)                                # sweep so the O(n^2) stays cheap
    for a_i in range(n):
        a = order[a_i]
        for b_i in range(a_i + 1, n):
            b = order[b_i]
            if x[b] - x[a] > radius_m:
                break
            if math.hypot(x[b] - x[a], y[b] - y[a]) <= radius_m:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    groups = defaultdict(list)
    for i, f in enumerate(located):
        groups[find(i)].append(f)

    sites = []
    for sid, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), start=1):
        lat = float(np.mean([m.lat for m in members]))
        lon = float(np.mean([m.lon for m in members]))
        sites.append(Site(sid=sid, lat=lat, lon=lon, flights=members))

    # flights without a log join the site their campaign-mates were located at
    by_place = defaultdict(Counter)                  # place name -> how many located flights per site
    for i, s in enumerate(sites):
        for f in s.flights:
            if f.place:
                by_place[f.place][i] += 1
    for f in flights:
        if f.located or not f.place:
            continue
        c = by_place.get(f.place)
        if c:
            sites[c.most_common(1)[0][0]].inferred.append(f)
    sites.sort(key=lambda s: -s.n)
    for i, s in enumerate(sites, start=1):
        s.sid = i
    return sites


# ============================================================================
# Rendering
# ============================================================================

def load_boundaries(path: Optional[Path], offline: bool):
    """Austrian state polygons as lists of (lon, lat) rings."""
    p = Path(path) if path else CACHE_DIR / BOUNDARY_FILE
    if not p.exists() and not path and not offline:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw = CACHE_DIR / NE_RAW
        try:
            if not raw.exists():
                print(f"downloading state boundaries from Natural Earth ({NE_URL.rsplit('/', 1)[-1]}) ...")
                urllib.request.urlretrieve(NE_URL, raw)
            data = json.load(open(raw))
            feats = [f for f in data["features"]
                     if f["properties"].get("iso_a2") == "AT" or f["properties"].get("admin") == "Austria"]
            try:
                from shapely.geometry import shape, mapping
                feats = [{"type": "Feature", "properties": {"name": f["properties"].get("name")},
                          "geometry": mapping(shape(f["geometry"]).simplify(0.002, preserve_topology=True))}
                         for f in feats]
            except ImportError:
                pass
            json.dump({"type": "FeatureCollection", "features": feats,
                       "attribution": "Natural Earth, public domain"}, open(p, "w"))
        except Exception as exc:                                    # noqa: BLE001
            print(f"no boundaries ({exc}); drawing the map without them")
            return []
    if not p.exists():
        return []
    data = json.load(open(p))
    rings = []
    for f in data.get("features", []):
        g = f.get("geometry") or {}
        polys = g.get("coordinates", []) if g.get("type") == "Polygon" else \
            [r for poly in g.get("coordinates", []) for r in poly]
        if g.get("type") == "Polygon":
            polys = g["coordinates"]
        for ring in polys:
            if len(ring) > 2:
                rings.append(np.asarray(ring, float))
    return rings


def repel_labels(ax, xs, ys, texts, colour, fontsize, radii=None, iterations=400):
    """
    Put each label next to its mark, push overlapping labels apart, keep them
    inside the axes and off their own circle, and draw a leader line when a
    label had to move far.  Everything here is in display pixels.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = np.array([ax.transData.transform((x, y)) for x, y in zip(xs, ys)], float)
    pt2px = fig.dpi / 72.0
    rad = (np.asarray(radii, float) if radii is not None else np.full(len(px), 4.0)) * pt2px

    sizes = []
    for t in texts:
        tp = ax.text(0, 0, t, fontsize=fontsize, alpha=0)
        bb = tp.get_window_extent(renderer=renderer)
        sizes.append((bb.width, bb.height))
        tp.remove()
    sizes = np.asarray(sizes, float)

    bbox = ax.get_window_extent(renderer=renderer)
    lab = px + np.stack([rad + 6 * pt2px, np.zeros(len(px))], axis=1)     # anchor: left edge, centre line

    for _ in range(iterations):
        moved = False
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                dx, dy = lab[j] - lab[i]
                ox = (sizes[i, 0] + sizes[j, 0]) / 2 + 8 - abs(dx)
                oy = (sizes[i, 1] + sizes[j, 1]) / 2 + 4 - abs(dy)
                if ox > 0 and oy > 0:                       # separate along the cheaper axis
                    if oy <= ox / 3:
                        d = oy / 2 * (1 if dy >= 0 else -1)
                        lab[i, 1] -= d; lab[j, 1] += d
                    else:
                        d = min(ox, 40) / 2 * (1 if dx >= 0 else -1)
                        lab[i, 0] -= d; lab[j, 0] += d
                    moved = True
        for i in range(len(lab)):
            d = lab[i] - px[i]
            r = math.hypot(*d) or 1.0
            lo, hi = rad[i] + 6 * pt2px, rad[i] + 70 * pt2px
            if r < lo:
                lab[i] = px[i] + d / r * lo
            elif r > hi:
                lab[i] = px[i] + d / r * hi
            lab[i, 0] = min(max(lab[i, 0], bbox.x0 + 4), bbox.x1 - sizes[i, 0] - 10)
            lab[i, 1] = min(max(lab[i, 1], bbox.y0 + sizes[i, 1]), bbox.y1 - sizes[i, 1])
        if not moved:
            break

    inv = ax.transData.inverted()
    for i, t in enumerate(texts):
        lx, ly = lab[i]
        mx, my = px[i]
        # leader line from the label's near edge to the rim of the mark
        if math.hypot(lx - mx, ly - my) > rad[i] + 14 * pt2px:
            ex = lx if lx < mx else lx
            ax.annotate("", xy=inv.transform((mx, my)), xytext=inv.transform((ex, ly)),
                        arrowprops=dict(arrowstyle="-", color=colour, lw=0.6, alpha=0.5,
                                        shrinkA=1, shrinkB=rad[i] / pt2px + 2), zorder=5)
        ax.text(*inv.transform((lx, ly)), t, fontsize=fontsize, color=colour,
                va="center", ha="left", zorder=6)


def render(sites, flights, args, th):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

    rings = load_boundaries(args.boundaries, args.offline)
    lat0 = float(np.mean([s.lat for s in sites]))
    kx = math.cos(math.radians(lat0))                     # equirectangular, x scaled so shapes are true

    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=args.dpi, facecolor=th["surface"])
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1.0], height_ratios=[1, 0.16],
                          left=0.035, right=0.985, top=0.855, bottom=0.065, wspace=0.115, hspace=0.02)
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(th["surface"])
    bx = fig.add_subplot(gs[:, 1]); bx.set_facecolor(th["surface"])

    for ring in rings:
        ax.fill(ring[:, 0] * kx, ring[:, 1], facecolor=th["land"], edgecolor=th["border"], lw=0.7, zorder=0)

    # size: circle AREA proportional to the flight count (scatter's s is area in pt^2)
    counts = np.array([s.n for s in sites], float)
    area = counts / counts.max() * (args.max_marker ** 2)
    area = np.maximum(area, 18.0)

    # colour: sequential ramp over the chosen magnitude, or one flat hue
    cmap = LinearSegmentedColormap.from_list("bambi_blue", th["ramp"])
    if args.color_by == "species":
        vals = np.array([len(s.species) for s in sites], float)
        clabel = "species recorded"
    elif args.color_by == "dates":
        vals = np.array([len(s.dates) for s in sites], float)
        clabel = "recording days"
    else:
        vals, clabel = None, ""
    if vals is not None and np.nanmax(vals) > np.nanmin(vals):
        norm = Normalize(vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)))
        colours = np.array([cmap(norm(v)) for v in vals])
    else:
        vals, norm = None, None
        colours = np.array([to_rgba(th["mark"])] * len(sites))

    order = np.argsort(-counts)                            # big first, small drawn on top
    ax.scatter([sites[i].lon * kx for i in order], [sites[i].lat for i in order],
               s=area[order], c=colours[order], edgecolors=th["ring"], linewidths=1.6,
               alpha=0.95, zorder=3)

    xs = [s.lon * kx for s in sites]; ys = [s.lat for s in sites]
    if rings and args.extent == "country":
        bx0 = min(r[:, 0].min() for r in rings) * kx; bx1 = max(r[:, 0].max() for r in rings) * kx
        by0 = min(r[:, 1].min() for r in rings); by1 = max(r[:, 1].max() for r in rings)
        x0, x1 = min(bx0, min(xs)), max(bx1, max(xs))
        y0, y1 = min(by0, min(ys)), max(by1, max(ys))
        pad = 0.02
    else:
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        pad = 0.14
    xr = max(x1 - x0, 0.4); yr = max(y1 - y0, 0.3)
    ax.set_xlim(x0 - pad * xr - 0.06, x1 + pad * xr + 0.06)
    ax.set_ylim(y0 - pad * yr - 0.05, y1 + pad * yr + 0.16)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    big = list(np.argsort(-counts)[: args.labels])
    repel_labels(ax, [sites[i].lon * kx for i in big], [sites[i].lat for i in big],
                 [f"{sites[i].label}  {sites[i].n}" for i in big], th["ink2"], args.label_size,
                 radii=[math.sqrt(area[i]) / 2 for i in big])

    # legends live under the map, so they cannot collide with the marks
    lax = fig.add_subplot(gs[1, 0]); lax.set_facecolor(th["surface"])
    lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.set_xticks([]); lax.set_yticks([])
    for sp in lax.spines.values():
        sp.set_visible(False)
    mx = int(counts.max())
    mid = next((v for v in (5, 10, 25, 50, 100, 250, 500) if v >= mx / 4), mx)
    ref = sorted({int(counts.min()), min(mid, mx), mx})
    lax.text(0.0, 0.93, "flights per site", color=th["ink2"], fontsize=args.label_size, va="center")
    fig.canvas.draw()
    lbb = lax.get_window_extent(renderer=fig.canvas.get_renderer())
    pt2px = fig.dpi / 72.0
    base = 0.30                                            # baseline the circles sit on
    xpos = 0.012
    for v in ref:
        a = max(v / counts.max() * (args.max_marker ** 2), 18.0)
        r_px = math.sqrt(a / math.pi) * pt2px
        ry, rx = r_px / lbb.height, r_px / lbb.width
        lax.scatter([xpos + rx], [base + ry], s=a, facecolors="none", edgecolors=th["ink3"],
                    linewidths=1.0, clip_on=False, zorder=4)
        lax.text(xpos + rx, base - 0.12, str(v), color=th["ink3"], fontsize=args.label_size - 1,
                 ha="center", va="top")
        xpos += 2 * rx + 0.022
    if vals is not None:
        cx0 = 0.42
        ramp = np.linspace(0, 1, 256)[None, :]
        lax.imshow(ramp, aspect="auto", cmap=cmap, extent=(cx0, cx0 + 0.22, 0.34, 0.54), zorder=3)
        lax.add_patch(plt.Rectangle((cx0, 0.34), 0.22, 0.20, fill=False, ec=th["border"], lw=0.6, zorder=4))
        lax.text(cx0, 0.92, clabel, color=th["ink2"], fontsize=args.label_size, va="center")
        lax.text(cx0, 0.18, f"{int(np.nanmin(vals))}", color=th["ink3"], fontsize=args.label_size - 1, ha="left", va="top")
        lax.text(cx0 + 0.22, 0.18, f"{int(np.nanmax(vals))}", color=th["ink3"], fontsize=args.label_size - 1,
                 ha="right", va="top")

    # ranked bars
    top = sorted(sites, key=lambda s: -s.n)[: args.bars]
    ypos = np.arange(len(top))[::-1]
    bar_col = [cmap(norm(len(s.species) if args.color_by == "species" else len(s.dates)))
               if norm is not None else th["mark"] for s in top]
    bx.barh(ypos, [s.n for s in top], height=0.62, color=bar_col, zorder=3)
    for y, s in zip(ypos, top):
        bx.text(s.n + max(s.n for s in top) * 0.015, y, str(s.n), va="center", ha="left",
                fontsize=args.label_size, color=th["ink2"], zorder=4)
    bx.set_yticks(ypos)
    bx.set_yticklabels([f"{s.label}" for s in top], fontsize=args.label_size, color=th["ink"])
    bx.set_xlim(0, max(s.n for s in top) * 1.18)
    bx.set_xlabel("flights", fontsize=args.label_size, color=th["ink2"])
    bx.tick_params(axis="x", colors=th["ink3"], labelsize=args.label_size - 1, length=0)
    bx.tick_params(axis="y", length=0)
    bx.grid(axis="x", color=th["border"], lw=0.5, alpha=0.6, zorder=0)
    bx.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        bx.spines[side].set_visible(False)
    bx.set_title("sites by flights" if len(top) >= len(sites) else f"largest {len(top)} sites", fontsize=args.label_size + 1, color=th["ink"], loc="left", pad=8)

    n_loc = sum(1 for f in flights if f.located)
    n_inf = sum(len(s.inferred) for s in sites)
    key_frames = sum(s.key_frames for s in sites)
    species = sorted({sp for s in sites for sp in s.species})
    fig.text(0.035, 0.955, args.title, fontsize=args.label_size + 8, color=th["ink"], va="top", weight="bold")
    placed = sum(s.n for s in sites)
    sub = (f"{placed} of {len(flights)} flights on {len(sites)} sites, aggregated within {args.radius:g} km"
           f"   ·   {n_loc} placed from their own flight log, {n_inf} from their campaign"
           f"   ·   {len(species)} species   ·   {key_frames:,} annotated key frames")
    fig.text(0.035, 0.905, sub, fontsize=args.label_size + 1, color=th["ink2"], va="top")
    fig.text(0.985, 0.018, "boundaries: Natural Earth (public domain)", fontsize=args.label_size - 1,
             color=th["ink3"], ha="right", va="top")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=th["surface"])
    print(f"wrote {args.output}")


# ============================================================================
# CLI
# ============================================================================

def write_tables(sites, args):
    if args.geojson:
        fc = {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(s.lon, 6), round(s.lat, 6)]},
            "properties": {"site": s.sid, "label": s.label, "region": s.region, "flights": s.n,
                           "flights_located": len(s.flights), "flights_inferred": len(s.inferred),
                           "key_frames": s.key_frames, "hours": None if math.isnan(s.hours) else round(s.hours, 2),
                           "spread_m": round(s.spread_m, 1), "days": len(s.dates),
                           "first": s.dates[0] if s.dates else "", "last": s.dates[-1] if s.dates else "",
                           "species": "; ".join(s.species),
                           "flight_ids": ",".join(f.flight for f in s.flights + s.inferred)},
        } for s in sites]}
        args.geojson.parent.mkdir(parents=True, exist_ok=True)
        json.dump(fc, open(args.geojson, "w"), indent=1)
        print(f"wrote {args.geojson}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["site", "label", "region", "lat", "lon", "flights", "flights_located", "flights_inferred",
                        "key_frames", "hours", "spread_m", "days", "first", "last", "species", "flight_ids"])
            for s in sites:
                w.writerow([s.sid, s.label, s.region, f"{s.lat:.6f}", f"{s.lon:.6f}", s.n, len(s.flights),
                            len(s.inferred), s.key_frames, "" if math.isnan(s.hours) else f"{s.hours:.2f}",
                            f"{s.spread_m:.1f}", len(s.dates), s.dates[0] if s.dates else "",
                            s.dates[-1] if s.dates else "", "; ".join(s.species),
                            ",".join(f.flight for f in s.flights + s.inferred)])
        print(f"wrote {args.csv}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Overview map of all BAMBI flights, aggregated into sites.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--poses", type=Path, nargs="*", default=[],
                     help="folder(s) with <id>_matched_poses.json (searched recursively)")
    src.add_argument("--raw", type=Path, nargs="*", default=[],
                     help="folder(s) with per-flight subfolders containing air_data.csv")
    src.add_argument("--metadata", type=Path, default=Path("flight_metadata"),
                     help="flight_metadata folder: split, species, drone and recording site per flight")
    src.add_argument("--cache", type=Path, default=None, help="per-flight summary CSV, read and written")
    src.add_argument("--refresh", action="store_true", help="ignore the cached summaries")

    agg = p.add_argument_group("aggregation")
    agg.add_argument("--radius", type=float, default=1.0,
                     help="flights closer than this many km share a site (single linkage)")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output", type=Path, default=Path("flight_overview.png"))
    out.add_argument("--geojson", type=Path, default=None, help="one point per site, for QGIS")
    out.add_argument("--csv", type=Path, default=None, help="the site table as CSV")
    out.add_argument("--flight-csv", type=Path, default=None, help="the per-flight table as CSV")
    out.add_argument("--boundaries", type=Path, default=None, help="state boundary GeoJSON (default: cached)")
    out.add_argument("--offline", action="store_true", help="never download the boundaries")

    sty = p.add_argument_group("style")
    sty.add_argument("--theme", choices=["light", "dark"], default="light")
    sty.add_argument("--extent", choices=["country", "data"], default="country",
                     help="show the whole country, or zoom to the sites")
    sty.add_argument("--color-by", choices=["species", "dates", "none"], default="species")
    sty.add_argument("--title", default="BAMBI: where the flights were recorded")
    sty.add_argument("--width", type=int, default=1600)
    sty.add_argument("--height", type=int, default=850)
    sty.add_argument("--dpi", type=int, default=140)
    sty.add_argument("--labels", type=int, default=12, help="how many sites get a label on the map")
    sty.add_argument("--bars", type=int, default=12, help="how many sites in the ranked panel")
    sty.add_argument("--label-size", type=float, default=8.5)
    sty.add_argument("--max-marker", type=float, default=44.0,
                     help="side in points of the largest circle; area scales with the flight count")
    args = p.parse_args(argv)

    if not args.poses and not args.raw:
        p.error("give at least one --poses or --raw folder")

    flights = collect(args.poses, args.raw, args.metadata, args.cache or args.flight_csv, args.refresh)
    if args.flight_csv and args.flight_csv != args.cache:
        args.cache = args.flight_csv
    located = [f for f in flights if f.located]
    print(f"{len(flights)} flights, {len(located)} with a position "
          f"({Counter(f.source for f in located)})")
    if not located:
        sys.exit("no flight has coordinates; download poses or air_data first")

    sites = cluster(flights, args.radius * 1000.0)
    placed = sum(s.n for s in sites)
    print(f"{len(sites)} sites within {args.radius:g} km, holding {placed} of {len(flights)} flights")
    for s in sites[:15]:
        print(f"  site {s.sid:2d}  {s.n:4d} flights  spread {s.spread_m:6.0f} m  "
              f"{s.lat:.5f},{s.lon:.5f}  {s.label} ({s.region})")
    # how well does the geographic clustering agree with the recording campaigns?
    mixed = [s for s in sites if len({f.place for f in s.flights if f.place}) > 1]
    split_places = Counter()
    for s in sites:
        for pl in {f.place for f in s.flights if f.place}:
            split_places[pl] += 1
    torn = [k for k, v in split_places.items() if v > 1]
    print(f"agreement with the named recording sites: {len(mixed)} site(s) merge several names, "
          f"{len(torn)} name(s) split over several sites")

    render(sites, flights, args, THEME[args.theme])
    write_tables(sites, args)


if __name__ == "__main__":
    main()
