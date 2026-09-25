"""Scenes built on the geo-referenced poses: the growing orthomosaic and the light-field refocus."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import (BG, FAINT, FG, GREEN, HOT, MUTED, PANEL, SKY, Scene, blank, chip, clamp, ease_in_out,
                    ease_out, fmt_int, lerp, ramp, rgba, rounded_mask, section_title, species_of, text,
                    text_width)


def _pil(bgr):
    return Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))


# ── Orthomosaic with world-coordinate tracks ───────────────────────────────
class MosaicScene(Scene):
    """The live frame on the left; on the right every frame lands on the ground where the pose says,
    and the animals' boxes become trajectories in metres."""
    name = "mosaic"

    def __init__(self, flight, cache: Path, duration: float = 10.0, layer: str = "rgb", end: int = None):
        self.f = flight
        self.duration = duration
        d = np.load(cache)
        self.layer = d[layer]
        self.valid = d["valid"]
        self.frames = d["frames"]
        self.end = int(end) if end is not None else int(self.frames[-1])
        self.path = d["path"]
        self.fp = d["footprints"]
        self.size = int(d["size"])
        self.metres = float(d["metres"])
        self.tracks = d["tracks"]
        self.species = d["species"]
        self.acc = None
        self.acc_k = -1
        self.disp = 900
        self.s = self.disp / self.size
        self.mx, self.my = 900, 150

    def src_frame(self, t):
        u = ease_in_out(clamp((t - 0.4) / (self.duration - 1.4)))
        return int(lerp(self.frames[0], self.end, u))

    def _mosaic(self, upto_k):
        if self.acc is None or upto_k < self.acc_k:
            self.acc = np.zeros((self.size, self.size, 3), np.uint8)
            self.acc_k = -1
        for k in range(self.acc_k + 1, upto_k + 1):
            # the RGB half has its own black rim inside the thermal footprint
            m = (self.valid[k] & (self.layer[k].max(axis=2) > 8)).astype(np.uint8)
            m = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool)
            self.acc[m] = self.layer[k][m]
        self.acc_k = upto_k
        return self.acc

    def render(self, t):
        img = blank()
        a = ramp(t, 0, 0.6)
        f = self.src_frame(t)
        k = int(np.searchsorted(self.frames, f, side="right")) - 1
        # left: the frame the drone sees now
        th, _ = self.f.frame(f)
        live = cv2.resize(th[118:906, 118:906], (640, 640), interpolation=cv2.INTER_AREA)
        img.paste(_pil(live), (120, 290), rounded_mask((640, 640), 16))
        d = ImageDraw.Draw(img, "RGBA")
        for det in self.f.thermal_tracks.at(f):
            _, _, col = species_of(det["species"])
            sc = 640 / 788
            x0 = 120 + (det["bb_left"] - 118) * sc
            y0 = 290 + (det["bb_top"] - 118) * sc
            x1, y1 = x0 + det["bb_width"] * sc, y0 + det["bb_height"] * sc
            if 120 <= x0 and x1 <= 760 and 290 <= y0 and y1 <= 930:
                d.rectangle((x0, y0, x1, y1), outline=rgba(col, a), width=2)
        chip(d, (136, 306), "THERMAL · IMAGE SPACE", HOT, size=15, alpha=a, filled=True)
        # right: the ground
        mos = self._mosaic(max(k, 0))
        view = cv2.resize(mos, (self.disp, self.disp), interpolation=cv2.INTER_AREA)
        base = Image.new("RGB", (self.disp, self.disp), PANEL)
        base.paste(_pil(view), (0, 0), Image.fromarray((view.max(axis=2) > 0).astype(np.uint8) * 255))
        img.paste(base, (self.mx, self.my), rounded_mask((self.disp, self.disp), 16))
        d = ImageDraw.Draw(img, "RGBA")
        P = lambda p: (self.mx + p[0] * self.s, self.my + p[1] * self.s)
        # drone path so far and the current footprint
        n = f - self.frames[0] + 1
        pts = [P(p) for p in self.path[:max(2, n):3]]
        d.line(pts, fill=rgba(FG, 0.8 * a), width=3, joint="curve")
        fp = [P(p) for p in self.fp[max(k, 0)] if np.all(np.isfinite(p))]
        if len(fp) > 2:
            d.polygon(fp, outline=rgba(HOT, a), width=3)
        dx, dy = P(self.path[min(n - 1, len(self.path) - 1)])
        d.ellipse((dx - 8, dy - 8, dx + 8, dy + 8), fill=rgba(HOT, a), outline=rgba(BG, a), width=2)
        # world tracks up to now
        sel = self.tracks[:, 0] <= f
        for tid in np.unique(self.tracks[sel, 1]):
            m = sel & (self.tracks[:, 1] == tid)
            ptr = self.tracks[m]
            _, _, col = species_of(str(self.species[m][0]))
            tp = [P(p) for p in ptr[:, 2:4]]
            if len(tp) > 1:
                d.line(tp, fill=rgba(col, 0.9 * a), width=3, joint="curve")
            live_now = ptr[-1, 0] >= f - 3
            x, y = tp[-1]
            r = 5 if live_now else 3
            d.ellipse((x - r, y - r, x + r, y + r), fill=rgba(col, a if live_now else 0.6 * a))
        chip(d, (self.mx + 16, self.my + 16), "RGB ORTHOMOSAIC · WORLD SPACE", GREEN, size=15, alpha=a,
             filled=True)
        # scale bar: 10 m
        px10 = 10 / self.metres * self.disp
        bx, by = self.mx + self.disp - 40 - px10, self.my + self.disp - 40
        d.rectangle((bx - 12, by - 34, bx + px10 + 12, by + 14), fill=rgba(BG, 0.7 * a))
        d.line([(bx, by), (bx + px10, by)], fill=rgba(FG, a), width=4)
        text(d, (bx + px10 / 2, by - 12), "10 m", 16, 700, fill=FG, alpha=a, anchor="ms")
        # north arrow
        nx, ny = self.mx + self.disp - 44, self.my + 44
        d.polygon([(nx, ny - 20), (nx - 10, ny + 10), (nx, ny + 4), (nx + 10, ny + 10)], fill=rgba(FG, a))
        text(d, (nx, ny + 32), "N", 16, 700, fill=FG, alpha=a, anchor="ms")

        section_title(d, "Where exactly", "Every frame geo-referenced", alpha=a)
        p = self.f.pose(f)
        lines = [("RTK GNSS pose per frame", FG), (f"{p['lat']:.6f}° N", MUTED), (f"{p['lng']:.6f}° E", MUTED),
                 (f"{p['alt']:.1f} m", MUTED)]
        text(d, (120, 214), "Frames and boxes projected to the ground:", 22, 500, fill=MUTED, alpha=a)
        text(d, (120, 244), "tracks become trajectories in metres.", 22, 500, fill=MUTED, alpha=a)
        text(d, (120, 980), f"flight {self.f.key}  ·  frame {f:,}  ·  {p['lat']:.6f}° N  {p['lng']:.6f}° E",
             20, 500, fill=MUTED, alpha=a)
        return img


# ── Light field ────────────────────────────────────────────────────────────
class AlfsScene(Scene):
    """A single frame next to the light-field integral while its focal plane sinks through the canopy."""
    name = "alfs"

    def __init__(self, flight, cache: Path, duration: float = 11.0, ground_agl: float = 55.0,
                 start_agl: float = None):
        self.f = flight
        self.duration = duration
        d = np.load(cache)
        self.integrals = d["integrals"]
        self.heights = d["heights"]
        self.single = d["single"]
        self.boxes = d["boxes"]
        self.n_frames = int(d["n_frames"])
        self.ground = ground_agl
        self.start = start_agl if start_agl is not None else float(self.heights[0])
        self.disp = 720
        cnts, _ = cv2.findContours((self.boxes > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        s = self.disp / self.boxes.shape[0]
        self.rects = [cv2.boundingRect(c) for c in cnts]
        self.rects = [(x * s, y * s, (x + w) * s, (y + h) * s) for x, y, w, h in self.rects]

    def agl(self, t):
        # hold on the canopy, sink to the ground, hold
        u = ease_in_out(clamp((t - 1.6) / 5.0))
        return lerp(self.start, self.ground, u)

    def integral_at(self, agl):
        h = self.heights
        i = int(np.clip(np.searchsorted(h, agl) - 1, 0, len(h) - 2))
        w = clamp((agl - h[i]) / (h[i + 1] - h[i]))
        a = self.integrals[i].astype(np.float32)
        b = self.integrals[i + 1].astype(np.float32)
        return (a * (1 - w) + b * w).astype(np.uint8)

    def render(self, t):
        img = blank()
        a = ramp(t, 0, 0.6)
        agl = self.agl(t)
        L, R, Y = 150, 1050, 200
        single = cv2.resize(self.single, (self.disp, self.disp), interpolation=cv2.INTER_AREA)
        integ = cv2.resize(self.integral_at(agl), (self.disp, self.disp), interpolation=cv2.INTER_AREA)
        m = rounded_mask((self.disp, self.disp), 16)
        img.paste(_pil(single), (L, Y), m)
        img.paste(_pil(integ), (R, Y), m)
        d = ImageDraw.Draw(img, "RGBA")
        section_title(d, "See through the canopy", "Airborne light-field sampling", alpha=a, kicker_col=GREEN)
        chip(d, (L + 16, Y + 16), "ONE FRAME", HOT, size=15, alpha=a, filled=True)
        chip(d, (R + 16, Y + 16), f"INTEGRAL OF {self.n_frames} FRAMES", GREEN, size=15, alpha=a, filled=True)
        # annotated animals: on the single frame from the start, on the integral once in focus
        a_box = ramp(t, 7.0, 7.8)
        for x0, y0, x1, y1 in self.rects:
            pad = 10
            d.rectangle((L + x0 - pad, Y + y0 - pad, L + x1 + pad, Y + y1 + pad), outline=rgba(HOT, 0.9 * a), width=2)
            d.rectangle((R + x0 - pad, Y + y0 - pad, R + x1 + pad, Y + y1 + pad), outline=rgba(HOT, a_box), width=3)
        # focal-plane gauge between the panels
        gx, gy0, gy1 = 945, Y + 60, Y + self.disp - 60
        span = (gy1 - gy0)
        to_y = lambda h: gy0 + span * h / (self.ground + 5)
        d.line([(gx, gy0), (gx, gy1)], fill=rgba(FAINT, a), width=2)
        # drone at the top, canopy band, ground
        d.polygon([(gx - 14, gy0 - 8), (gx + 14, gy0 - 8), (gx, gy0 + 8)], fill=rgba(FG, a))
        cy0, cy1 = to_y(self.ground - 35), to_y(self.ground - 12)
        d.rectangle((gx - 26, cy0, gx + 26, cy1), fill=rgba(GREEN, 0.18 * a))
        text(d, (gx, cy0 - 8), "canopy", 14, 600, fill=GREEN, alpha=a, anchor="ms")
        d.line([(gx - 30, to_y(self.ground)), (gx + 30, to_y(self.ground))], fill=rgba(MUTED, a), width=3)
        text(d, (gx, to_y(self.ground) + 22), "ground", 14, 600, fill=MUTED, alpha=a, anchor="ms")
        yy = to_y(agl)
        d.line([(gx - 40, yy), (gx + 40, yy)], fill=rgba(HOT, a), width=4)
        text(d, (gx + 48, yy), f"{agl:.0f} m", 18, 700, fill=HOT, alpha=a, anchor="lm")
        # captions
        text(d, (L, Y + self.disp + 52), "Canopy and branches hide the forest floor.", 22, 500, fill=MUTED,
             alpha=a)
        cap = ("Focusing on the ground blurs the canopy away" if agl > self.ground - 6 else
               "Frames re-projected onto a focal plane below the drone")
        text(d, (R, Y + self.disp + 52), cap, 22, 500, fill=FG if agl > self.ground - 6 else MUTED, alpha=a)
        text(d, (R, Y + self.disp + 84), f"flight {self.f.key} · per-frame poses and corrections from the dataset",
             18, 500, fill=MUTED, alpha=0.8 * a)
        return img
