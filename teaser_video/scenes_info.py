"""The scenes that carry the dataset's facts: title, map, numbers, species, tasks, credits."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import (BG, FAINT, FG, GREEN, H, HOT, MUTED, PANEL, SKY, SPECIES, W, Scene, blank, blur, chip,
                    clamp, cover, darken, ease_in_out, ease_out, fmt_int, lerp, ramp, rgba, rounded_mask,
                    section_title, text, text_width, vignette)

# Numbers from the paper (Sections 3-5, Table 1).
FACTS = dict(sequences=386, hours=49, minutes=44, gigabytes=372, tracks=5100, keyframes=92701,
             boxes=1218903, classes=12, pairs=5305, box_pairs=13057, rgb_boxes=16873)

# Table 1: tracks and key frames per class.
TABLE1 = [
    ("Q58697", 1770, 26132), ("Q1219579", 1613, 26674), ("Q122069", 682, 9404),
    ("Q24238356", 344, 2536), ("Q20908334", 297, 20003), ("Q168327", 100, 3005),
    ("Q15978631", 93, 1158), ("Q5113", 75, 942), ("Q10738", 60, 521), ("Q602666", 44, 1484),
    ("Q131340", 15, 747), ("Q26972265", 7, 95),
]


# ── Title ──────────────────────────────────────────────────────────────────
class TitleScene(Scene):
    """Footage behind the title; a divider sweeps across, thermal on its left, RGB on its right."""
    name = "title"

    def __init__(self, flight, start: int, duration: float = 6.5, subtitle_lines: Sequence[str] = ()):
        self.f, self.start, self.duration = flight, start, duration
        self.subtitle_lines = subtitle_lines

    def render(self, t):
        th, rgb = self.f.frame(self.start + int(t * 29.97 * 0.8))
        th = cover(Image.fromarray(th[120:904, 120:904, ::-1]), (W, H))
        rgb = cover(Image.fromarray(rgb[120:904, 120:904, ::-1]), (W, H))
        split = int(lerp(-0.1, 0.55, ease_in_out(clamp(t / 4.5))) * W)
        img = rgb.copy()
        if split > 0:
            img.paste(th.crop((0, 0, split, H)), (0, 0))
        img = darken(img, 0.45)
        img = vignette(img, 0.8)
        d = ImageDraw.Draw(img, "RGBA")
        if 0 < split < W:
            d.line([(split, 0), (split, H)], fill=rgba(FG, 0.85), width=2)
            text(d, (split - 18, H - 60), "THERMAL", 18, 700, fill=HOT, anchor="rs", tracking=0.2)
            text(d, (split + 18, H - 60), "RGB", 18, 700, fill=GREEN, anchor="ls", tracking=0.2)
        a1 = ramp(t, 0.6, 1.6)
        a2 = ramp(t, 1.2, 2.2)
        a3 = ramp(t, 1.8, 2.8)
        dy = (1 - a1) * 24
        text(d, (120, 318 + dy), "THE", 34, 600, fill=HOT, alpha=a1, tracking=0.35)
        text(d, (112, 470 + dy), "BAMBI Dataset", 124, 800, fill=FG, alpha=a1, anchor="ls")
        text(d, (120, 540), "Multimodal Nadir UAV-Recordings of Forest Wildlife", 40, 500, fill=FG, alpha=a2,
             anchor="ls")
        x = 120
        for lab, col in (("386 paired flights", FG), ("RGB + thermal", HOT), ("geo-referenced", SKY),
                         ("tracked & annotated", GREEN)):
            x = chip(d, (x, 590), lab, col, size=20, alpha=a3) + 14
        for i, line in enumerate(self.subtitle_lines):
            text(d, (120, 690 + 34 * i), line, 22, 500, fill=MUTED, alpha=ramp(t, 2.2, 3.2))
        return img


# ── Map ────────────────────────────────────────────────────────────────────
def _austria_shapes(assets: Path):
    """Austria's outline, its states and its neighbours as lists of (N, 2) lon/lat rings.

    ``assets/austria.json`` is a simplified extract of Natural Earth (public domain).
    """
    data = json.loads((assets / "austria.json").read_text(encoding="utf-8"))
    as_arrays = lambda rings: [np.asarray(r, float) for r in rings]
    return as_arrays(data["country"]), as_arrays(data["states"]), as_arrays(data["neighbours"])


def flight_locations(ann_dir: Path, meta_dir: Path):
    """``[(date, lat, lon, drone, key)]`` for every flight whose poses are on disk, oldest first."""
    out = []
    for meta in sorted(Path(meta_dir).glob("*_metadata.json")):
        key = meta.name.split("_")[0]
        poses = Path(ann_dir) / f"{key}_matched_poses.json"
        if not poses.exists():
            continue
        m = json.loads(meta.read_text())
        fi = m.get("flight_info", {})
        if "start_time" not in fi:
            continue
        with open(poses) as fh:
            im = json.load(fh)["images"]
        lat = float(np.median([p["lat"] for p in im[::50]]))
        lon = float(np.median([p["lng"] for p in im[::50]]))
        out.append((datetime.fromisoformat(fi["start_time"]), lat, lon, fi.get("drone_name", ""), key))
    return sorted(out)


def flight_dates(meta_dir: Path):
    out = []
    for meta in Path(meta_dir).glob("*_metadata.json"):
        fi = json.loads(meta.read_text()).get("flight_info", {})
        if "start_time" in fi:
            out.append(datetime.fromisoformat(fi["start_time"]))
    return out


class MapScene(Scene):
    name = "map"
    LON = (9.4, 17.3)
    LAT = (46.3, 49.1)

    def __init__(self, assets: Path, locations, dates, duration: float = 9.0):
        self.duration = duration
        self.locations = locations
        self.dates = sorted(dates)
        self.country, self.states, self.neigh = _austria_shapes(assets)
        k = math.cos(math.radians(47.6))
        self.scale = 1180 / ((self.LON[1] - self.LON[0]) * k)
        self.k = k
        self.ox, self.oy = 70, 300
        self.base = self._base()
        self.t0 = self.dates[0]
        self.t1 = self.dates[-1]

    def xy(self, lon, lat):
        return (self.ox + (lon - self.LON[0]) * self.k * self.scale,
                self.oy + (self.LAT[1] - lat) * self.scale)

    def _poly(self, ring):
        return [self.xy(x, y) for x, y in ring]

    def _base(self):
        img = blank()
        d = ImageDraw.Draw(img, "RGBA")
        for r in self.neigh:
            d.line(self._poly(r), fill=rgba(FAINT, 0.35), width=1)
        for r in self.country:
            d.polygon(self._poly(r), fill=rgba(PANEL, 1.0))
        for r in self.states:
            d.line(self._poly(r), fill=rgba(FAINT, 0.9), width=1)
        for r in self.country:
            d.line(self._poly(r) + self._poly(r)[:1], fill=rgba(MUTED, 0.9), width=3, joint="curve")
        return img

    def render(self, t):
        img = self.base.copy()
        d = ImageDraw.Draw(img, "RGBA")
        a = ramp(t, 0, 0.6)
        section_title(d, "Where", "Temperate forests across six Austrian states", alpha=a)
        prog = ease_in_out(clamp((t - 0.8) / (self.duration - 2.4)))
        now = self.t0 + (self.t1 - self.t0) * prog
        shown = [l for l in self.locations if l[0] <= now]
        # flights at one site share a dot that grows with their number; a pulse marks a new one
        span = max(1.0, (self.t1 - self.t0).total_seconds())
        sites = {}
        for dt, lat, lon, drone, key in shown:
            k = (round(lat / 0.04), round(lon / 0.04))
            s_ = sites.setdefault(k, [0, lat, lon, drone, dt])
            s_[0] += 1
            s_[4] = dt
        for n_, lat, lon, drone, last in sorted(sites.values(), key=lambda v: -v[0]):
            x, y = self.xy(lon, lat)
            col = HOT if "30" in drone else SKY
            r = 5 + 2.2 * math.sqrt(n_)
            d.ellipse((x - r, y - r, x + r, y + r), fill=rgba(col, 0.55), outline=rgba(col, 0.95), width=2)
            age = (now - last).total_seconds() / span * (self.duration - 2.4)
            if age < 0.5:
                rr = r + 22 * age / 0.5
                d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=rgba(col, 1 - age / 0.5), width=2)
        # date ticker and counters
        rx = 1330
        text(d, (rx, 330), f"{now:%B %Y}", 52, 700, fill=FG, alpha=a, anchor="ls")
        n = sum(1 for x in self.dates if x <= now)
        text(d, (rx, 380), f"{n} of {len(self.dates)} flights", 26, 500, fill=MUTED, alpha=a, anchor="ls")
        # flights per month, like Figure 5 (d) of the paper
        keys = [(y, m) for y in (2023, 2024) for m in range(1, 13) if not (y == 2024 and m == 12)]
        allc, cur = {}, {}
        for x in self.dates:
            allc[(x.year, x.month)] = allc.get((x.year, x.month), 0) + 1
            if x <= now:
                cur[(x.year, x.month)] = cur.get((x.year, x.month), 0) + 1
        mx = max(allc.values())
        bx, by, bw, bh = rx, 700, 20, 230
        text(d, (bx, by - bh - 24), "FLIGHTS PER MONTH", 16, 700, fill=MUTED, alpha=a, tracking=0.16)
        for i, k in enumerate(keys):
            x = bx + i * bw
            d.rectangle((x, by - bh, x + bw - 5, by), fill=rgba(FAINT, 0.22 * a))
            hgt = bh * cur.get(k, 0) / mx
            if hgt > 0:
                on = (k[0], k[1]) == (now.year, now.month)
                d.rectangle((x, by - hgt, x + bw - 5, by), fill=rgba(FG if on else MUTED, (1.0 if on else 0.7) * a))
        text(d, (bx, by + 28), "2023", 18, 600, fill=MUTED, alpha=a)
        text(d, (bx + 12 * bw, by + 28), "2024", 18, 600, fill=MUTED, alpha=a)
        # legend
        for i, (lab, col) in enumerate((("DJI M30T", HOT), ("DJI M3T", SKY))):
            y = 820 + i * 36
            d.ellipse((rx, y - 6, rx + 12, y + 6), fill=rgba(col, a))
            text(d, (rx + 24, y), lab, 20, 500, fill=MUTED, alpha=a, anchor="lm")
        text(d, (96, 1010), "Jan 2023 – Nov 2024  ·  forests, wildlife crossings and near-natural enclosures",
             20, 500, fill=MUTED, alpha=a)
        return img


# ── Numbers ────────────────────────────────────────────────────────────────
class StatsScene(Scene):
    name = "stats"

    def __init__(self, backdrop: Optional[Image.Image] = None, duration: float = 7.0):
        self.duration = duration
        self.backdrop = darken(blur(cover(backdrop, (W, H)), 18), 0.8) if backdrop is not None else blank()
        self.items = [
            (FACTS["sequences"], "{:,}", "paired RGB + thermal videos", HOT),
            (FACTS["hours"] * 60 + FACTS["minutes"], "hm", "of flight footage · 372 GB", FG),
            (FACTS["tracks"], "{:,}", "annotated animal tracks", GREEN),
            (FACTS["keyframes"], "{:,}", "annotated key frames", GREEN),
            (FACTS["boxes"], "{:,}", "bounding boxes after interpolation", SKY),
            (FACTS["classes"], "{:,}", "classes · with age & sex", FG),
        ]

    def render(self, t):
        img = self.backdrop.copy()
        d = ImageDraw.Draw(img, "RGBA")
        section_title(d, "What", "One of the largest nadir wildlife video collections", alpha=ramp(t, 0, 0.6))
        for i, (val, fmt, label, col) in enumerate(self.items):
            r, c = divmod(i, 3)
            x = 120 + c * 580
            y = 330 + r * 330
            t0 = 0.4 + 0.18 * i
            a = ramp(t, t0, t0 + 0.6)
            v = val * ease_out(clamp((t - t0) / 1.8))
            if fmt == "hm":
                s = f"{int(v // 60)} h {int(v % 60):02d} min"
            else:
                s = fmt.format(int(round(v)))
            d.line([(x, y - 30), (x + 60 * a, y - 30)], fill=rgba(col, a), width=4)
            text(d, (x, y + 80), s, 84, 800, fill=FG, alpha=a, anchor="ls")
            text(d, (x, y + 130), label, 26, 500, fill=MUTED, alpha=a, anchor="ls")
        return img


# ── Species ────────────────────────────────────────────────────────────────
class SpeciesScene(Scene):
    name = "species"

    def __init__(self, duration: float = 7.0):
        self.duration = duration

    def render(self, t):
        img = blank()
        d = ImageDraw.Draw(img, "RGBA")
        section_title(d, "Who", "10 species, plus heat sources and unknowns", alpha=ramp(t, 0, 0.6))
        mx = TABLE1[0][1]
        x0, y0, bh, gap = 520, 220, 50, 16
        text(d, (x0, y0 - 22), "TRACKS", 16, 700, fill=MUTED, alpha=ramp(t, 0.3, 0.8), tracking=0.18)
        text(d, (1800 - text_width("KEY FRAMES", 16, 700, 0.18), y0 - 22), "KEY FRAMES", 16, 700, fill=MUTED,
             alpha=ramp(t, 0.3, 0.8), tracking=0.18)
        for i, (qid, tracks, keys) in enumerate(TABLE1):
            common, latin, col = SPECIES[qid]
            y = y0 + i * (bh + gap)
            t0 = 0.3 + 0.08 * i
            a = ramp(t, t0, t0 + 0.5)
            text(d, (x0 - 24, y + bh / 2 - 9), common, 26, 700, fill=FG, alpha=a, anchor="rm")
            text(d, (x0 - 24, y + bh / 2 + 16), latin, 16, 400, fill=col, alpha=a * 0.9, anchor="rm")
            # log scale keeps the rare classes visible, as Figure 5 of the paper does
            frac = math.log10(1 + tracks) / math.log10(1 + mx)
            wbar = 1000 * frac * ease_out(clamp((t - t0) / 1.2))
            d.rounded_rectangle((x0, y + 6, x0 + max(wbar, 8), y + bh - 6), radius=8, fill=rgba(col, 0.9 * a))
            text(d, (x0 + wbar + 14, y + bh / 2), fmt_int(tracks), 22, 700, fill=FG, alpha=a, anchor="lm")
            text(d, (1800, y + bh / 2), fmt_int(keys), 22, 500, fill=MUTED, alpha=a, anchor="rm")
        text(d, (x0, 1030), "log scale  ·  every track carries species, age and sex; every key frame a visibility flag",
             18, 500, fill=MUTED, alpha=ramp(t, 1.2, 1.8))
        return img


# ── Tasks ──────────────────────────────────────────────────────────────────
class TasksScene(Scene):
    """A grid of cards, each a task the dataset supports, with a still from earlier scenes."""
    name = "tasks"

    def __init__(self, cards: List[tuple], duration: float = 6.5):
        self.cards = cards                      # (title, subtitle, PIL image or None, accent)
        self.duration = duration

    def render(self, t):
        img = blank()
        d = ImageDraw.Draw(img, "RGBA")
        section_title(d, "Why", "Built for detection, tracking and beyond", alpha=ramp(t, 0, 0.6))
        cw, ch, gx, gy = 540, 360, 50, 48
        x0, y0 = 120, 210
        for i, (title, sub, pic, col) in enumerate(self.cards[:6]):
            r, c = divmod(i, 3)
            x, y = x0 + c * (cw + gx), y0 + r * (ch + gy)
            t0 = 0.3 + 0.12 * i
            a = ramp(t, t0, t0 + 0.6)
            if a <= 0:
                continue
            dy = (1 - a) * 30
            ph = 250
            if pic is not None:
                p = cover(pic, (cw, ph))
                m = rounded_mask((cw, ph), 16)
                m = m.point(lambda v: int(v * a))
                img.paste(p, (x, int(y + dy)), m)
                d = ImageDraw.Draw(img, "RGBA")
            d.rounded_rectangle((x, y + dy, x + cw, y + ph + dy), radius=16, outline=rgba(col, 0.6 * a), width=2)
            text(d, (x, y + ph + 44 + dy), title, 30, 700, fill=FG, alpha=a, anchor="ls")
            text(d, (x, y + ph + 78 + dy), sub, 20, 500, fill=MUTED, alpha=a, anchor="ls")
        return img


# ── Credits ────────────────────────────────────────────────────────────────
class OutroScene(Scene):
    name = "outro"

    def __init__(self, backdrop: Optional[Image.Image], links: Sequence[tuple], duration: float = 7.0,
                 footer: str = ""):
        self.duration = duration
        self.backdrop = darken(blur(cover(backdrop, (W, H)), 6), 0.72) if backdrop is not None else blank()
        self.links = links
        self.footer = footer

    def render(self, t):
        img = vignette(self.backdrop.copy(), 0.8)
        d = ImageDraw.Draw(img, "RGBA")
        a1, a2 = ramp(t, 0.2, 1.0), ramp(t, 0.7, 1.5)
        text(d, (W / 2, 380), "THE", 30, 600, fill=HOT, alpha=a1, anchor="ms", tracking=0.35)
        text(d, (W / 2, 490), "BAMBI Dataset", 110, 800, fill=FG, alpha=a1, anchor="ms")
        text(d, (W / 2, 560), "Multimodal Nadir UAV-Recordings of Forest Wildlife", 34, 500, fill=FG, alpha=a2,
             anchor="ms")
        for i, (label, url, col) in enumerate(self.links):
            a = ramp(t, 1.2 + 0.2 * i, 1.9 + 0.2 * i)
            y = 660 + i * 58
            text(d, (W / 2 - 16, y), label, 24, 700, fill=col, alpha=a, anchor="rm")
            text(d, (W / 2 + 16, y), url, 26, 500, fill=FG, alpha=a, anchor="lm")
        if self.footer:
            text(d, (W / 2, 1010), self.footer, 20, 500, fill=MUTED, alpha=ramp(t, 2.0, 2.8), anchor="ms")
        return img
