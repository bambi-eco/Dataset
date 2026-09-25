"""Real footage: the thermal and RGB halves side by side with their annotated tracks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import (BG, FAINT, FG, GREEN, HOT, MUTED, PANEL, Scene, blank, chip, clamp, corner_box,
                    ease_out, fade, font, ramp, rgba, rounded_mask, species_of, text, text_width)
from data import HALF, Flight

CROP = 118                       # px trimmed from every side of a 1024 half (undistortion rim)
SRC = HALF - 2 * CROP
PANEL_SIZE = 800
PANEL_Y = 176
PANEL_X = (120, 1000)
SCALE = PANEL_SIZE / SRC


@dataclass
class Clip:
    flight: str
    start: int                   # first source frame
    duration: float = 5.0        # seconds on screen
    speed: float = 1.0           # playback speed
    note: str = ""               # one line shown under the title
    rgb_boxes: bool = True       # draw the transferred RGB boxes too


def _to_panel(x, y, px):
    return px + (x - CROP) * SCALE, PANEL_Y + (y - CROP) * SCALE


class FootageScene(Scene):
    """One clip; a montage is several of these in a row."""

    def __init__(self, flight: Flight, clip: Clip, index: int = 0, count: int = 1):
        self.f = flight
        self.clip = clip
        self.duration = clip.duration
        self.index, self.count = index, count
        self.name = f"footage_{flight.key}"
        species = {}
        for f in range(clip.start, clip.start + int(clip.duration * 30 * clip.speed) + 1):
            for d in flight.thermal_tracks.at(f):
                species.setdefault(d["species"], set()).add(d["track_id"])
        # the clip's headline species: the one with the most tracks on screen
        self.species = sorted(species.items(), key=lambda kv: -len(kv[1]))

    def src_frame(self, t: float) -> int:
        return self.clip.start + int(round(t * 29.97 * self.clip.speed))

    # -- drawing -----------------------------------------------------------
    def _panel(self, bgr: np.ndarray) -> Image.Image:
        crop = bgr[CROP:HALF - CROP, CROP:HALF - CROP]
        crop = cv2.resize(crop, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
        return Image.fromarray(crop[:, :, ::-1])

    def _boxes(self, draw: ImageDraw.ImageDraw, tracks, frame: int, px: int, alpha: float, labels: bool):
        if tracks is None:
            return
        for d in tracks.at(frame):
            common, _, col = species_of(d["species"])
            x0, y0 = _to_panel(d["bb_left"], d["bb_top"], px)
            x1, y1 = _to_panel(d["bb_left"] + d["bb_width"], d["bb_top"] + d["bb_height"], px)
            lo, hi = px, px + PANEL_SIZE
            if x1 < lo or x0 > hi or y1 < PANEL_Y or y0 > PANEL_Y + PANEL_SIZE:
                continue
            x0, x1 = max(x0, lo), min(x1, hi)
            y0, y1 = max(y0, PANEL_Y), min(y1, PANEL_Y + PANEL_SIZE)
            occluded = d["visibility"] < 0.5
            a = alpha * (0.55 if occluded else 1.0)
            corner_box(draw, (x0, y0, x1, y1), col, alpha=a, width=3)
            if labels and x0 > lo + 4 and y0 > PANEL_Y + 24 and x1 < hi - 30:
                lab = f"{d['track_id']}"
                chip(draw, (x0, y0 - 4), lab, col, size=13, alpha=a, pad=(6, 2), filled=True,
                     anchor="lb")

    def render(self, t: float) -> Image.Image:
        img = blank()
        draw = ImageDraw.Draw(img, "RGBA")
        frame = self.src_frame(t)
        thermal, rgb = self.f.frame(frame)
        a_in = ramp(t, 0.0, 0.5)
        a_box = ramp(t, 0.35, 0.8)
        mask = rounded_mask((PANEL_SIZE, PANEL_SIZE), 18)
        for k, (src, px) in enumerate(((thermal, PANEL_X[0]), (rgb, PANEL_X[1]))):
            panel = self._panel(src)
            dy = (1 - ease_out(clamp((t - 0.08 * k) / 0.6))) * 30
            img.paste(panel, (px, int(PANEL_Y + dy)), mask)
        draw = ImageDraw.Draw(img, "RGBA")
        crowded = len(self.f.thermal_tracks.at(frame)) > 10
        self._boxes(draw, self.f.thermal_tracks, frame, PANEL_X[0], a_box, labels=not crowded)
        if self.clip.rgb_boxes:
            self._boxes(draw, self.f.rgb_tracks, frame, PANEL_X[1], a_box, labels=not crowded)

        # panel captions
        rgb_note = "boxes transferred from thermal" if self.clip.rgb_boxes and self.f.rgb_tracks else "wide-angle"
        for px, lab, col, note in ((PANEL_X[0], "THERMAL", HOT, "non-radiometric"),
                                   (PANEL_X[1], "RGB", GREEN, rgb_note)):
            w_ = text_width(lab, 16, 600) + 24 + 14 + text_width(note, 16, 500) + 16
            draw.rounded_rectangle((px + 14, PANEL_Y + 14, px + 14 + w_ + 8, PANEL_Y + 50), radius=18,
                                   fill=rgba(BG, 0.7 * a_in))
            x_ = chip(draw, (px + 18, PANEL_Y + 18), lab, col, size=16, alpha=a_in, filled=True)
            text(draw, (x_ + 12, PANEL_Y + 32), note, 16, 500, fill=FG, alpha=0.9 * a_in, anchor="lm")

        # headline: species present in the clip
        x = 120
        for i, (label, tids) in enumerate(self.species[:3]):
            common, latin, col = species_of(label)
            a = ramp(t, 0.1 + 0.12 * i, 0.6 + 0.12 * i)
            size = 46 if i == 0 else 30
            y = 118 if i == 0 else 126
            text(draw, (x, y), common, size, 700, fill=FG if i == 0 else MUTED, alpha=a, anchor="ls")
            x += text_width(common, size, 700) + 12
            text(draw, (x, y), latin, 20 if i else 24, 400, fill=col, alpha=a, anchor="ls")
            x += text_width(latin, 20 if i else 24, 400) + 36
        f = self.f
        kicker = f"FLIGHT {f.key}  ·  {f.date:%B %Y}  ·  {f.drone_name}".upper()
        text(draw, (120, 52), kicker, 18, 700, fill=HOT, alpha=a_in, tracking=0.14)
        if self.clip.note:
            text(draw, (1800, 118), self.clip.note, 22, 500, fill=MUTED, alpha=ramp(t, 0.3, 0.9), anchor="rs")

        # telemetry from the per-frame pose
        p = f.pose(frame)
        n = len(f.thermal_tracks.at(frame))
        tele = (f"{p['lat']:.6f}° N   {p['lng']:.6f}° E   {p['alt']:.1f} m   ·   " if f.poses_aligned else "")
        tele += f"frame {frame:,}   ·   {n} annotated {'animal' if n == 1 else 'animals'}"
        text(draw, (120, PANEL_Y + PANEL_SIZE + 44), tele, 20, 500, fill=MUTED, alpha=a_in, anchor="lm")
        # clip counter
        if self.count > 1:
            for k in range(self.count):
                cx = 1800 - (self.count - 1 - k) * 22
                on = k == self.index
                draw.ellipse((cx - 5, PANEL_Y + PANEL_SIZE + 39, cx + 5, PANEL_Y + PANEL_SIZE + 49),
                             fill=rgba(FG if on else FAINT, 1.0))
        return img
