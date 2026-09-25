"""Drawing primitives, typography, easing and the video writer shared by every scene.

Everything is drawn onto 1920x1080 RGB ``PIL.Image`` frames. Text and shapes
with opacity go through ``ImageDraw.Draw(img, "RGBA")``, which blends onto an
RGB image instead of overwriting it.
"""
from __future__ import annotations

import math
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS_DEFAULT = 30
HERE = Path(__file__).resolve().parent

# ── Palette ────────────────────────────────────────────────────────────────
BG = (9, 13, 12)
PANEL = (20, 27, 25)
FG = (238, 242, 238)
MUTED = (148, 160, 154)
FAINT = (70, 82, 77)
HOT = (255, 168, 38)        # thermal accent
GREEN = (118, 214, 152)     # RGB / forest accent
SKY = (110, 182, 255)

# One colour per class, keyed by Wikidata id (see docs/annotation-format.md).
SPECIES = {
    "Q58697":    ("Wild boar",    "Sus scrofa",             (255, 132, 64)),
    "Q1219579":  ("Red deer",     "Cervus elaphus",         (240, 84, 84)),
    "Q122069":   ("Roe deer",     "Capreolus capreolus",    (255, 205, 70)),
    "Q20908334": ("Fallow deer",  "Dama dama",              (198, 132, 255)),
    "Q168327":   ("Alpine ibex",  "Capra ibex",             (90, 200, 250)),
    "Q131340":   ("Chamois",      "Rupicapra rupicapra",    (80, 220, 190)),
    "Q5113":     ("Bird",         "Aves",                   (150, 230, 90)),
    "Q15978631": ("Human",        "Homo sapiens",           (235, 235, 235)),
    "Q26972265": ("Dog",          "Canis lupus familiaris", (255, 120, 190)),
    "Q602666":   ("Hybrid pig",   "Sus scrofa × S. domesticus", (255, 170, 140)),
    "Q10738":    ("No-animal",    "heat source",            (140, 150, 160)),
    "Q24238356": ("Unknown",      "unidentified",           (170, 170, 190)),
}
SPECIES_BY_NAME = {}
for _qid, (_common, _latin, _col) in SPECIES.items():
    SPECIES_BY_NAME[_common.lower()] = _qid


def species_of(label: str) -> Tuple[str, str, Tuple[int, int, int]]:
    """``(common, latin, colour)`` for a MOT species string like ``"Sus scrofa (Wild boar)"``."""
    common = label.split("(")[-1].rstrip(")").strip() if "(" in label else label
    qid = SPECIES_BY_NAME.get(common.lower())
    if qid is None:
        return common, "", MUTED
    return SPECIES[qid]


# ── Easing ─────────────────────────────────────────────────────────────────
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ramp(t, t0, t1, ease=ease_out):
    """0 before ``t0``, 1 after ``t1``, eased in between."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return ease((t - t0) / (t1 - t0))


def fade(t, t_in, t_out, dur, fade_len=0.5):
    """Opacity that fades in at ``t_in`` and out towards ``t_out`` (scene-local seconds)."""
    return min(ramp(t, t_in, t_in + fade_len), 1 - ramp(t, t_out - fade_len, t_out, ease_in_out))


# ── Typography ─────────────────────────────────────────────────────────────
FONT_FILE = HERE / "assets" / "Inter.ttf"
FALLBACK = {
    False: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    True: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}


@lru_cache(maxsize=256)
def font(size: int, weight: int = 400, italic: bool = False) -> ImageFont.FreeTypeFont:
    """Inter at ``weight`` (100-900); DejaVu when the asset is missing."""
    size = max(4, int(round(size)))
    if FONT_FILE.exists():
        f = ImageFont.truetype(str(FONT_FILE), size)
        try:
            axes = f.get_variation_axes()
            vals = []
            for ax in axes:
                name = ax["name"].decode() if isinstance(ax["name"], bytes) else ax["name"]
                if "eight" in name:
                    vals.append(weight)
                elif "ptical" in name:
                    vals.append(clamp(size, ax["minimum"], ax["maximum"]))
                else:
                    vals.append(ax["default"])
            f.set_variation_by_axes(vals)
        except (OSError, AttributeError):
            pass
        return f
    return ImageFont.truetype(FALLBACK[weight >= 600], size)


def rgba(col, alpha: float = 1.0):
    return (int(col[0]), int(col[1]), int(col[2]), int(round(255 * clamp(alpha))))


def text(draw: ImageDraw.ImageDraw, xy, s: str, size: int, weight: int = 400, fill=FG,
         alpha: float = 1.0, anchor: str = "la", tracking: float = 0.0):
    """Draw ``s``; ``tracking`` adds letter spacing in em (for small caps labels)."""
    if alpha <= 0.003 or not s:
        return
    f = font(size, weight)
    if tracking == 0.0:
        draw.text(xy, s, font=f, fill=rgba(fill, alpha), anchor=anchor)
        return
    # letter-spaced text: lay out glyph by glyph
    gap = tracking * size
    widths = [f.getlength(c) for c in s]
    total = sum(widths) + gap * (len(s) - 1)
    x, y = xy
    if anchor[0] == "m":
        x -= total / 2
    elif anchor[0] == "r":
        x -= total
    for c, w_ in zip(s, widths):
        draw.text((x, y), c, font=f, fill=rgba(fill, alpha), anchor="l" + anchor[1])
        x += w_ + gap


def text_width(s: str, size: int, weight: int = 400, tracking: float = 0.0) -> float:
    f = font(size, weight)
    return f.getlength(s) + tracking * size * max(0, len(s) - 1)


def fmt_int(n) -> str:
    return f"{int(round(n)):,}"


# ── Images ─────────────────────────────────────────────────────────────────
def blank(col=BG) -> Image.Image:
    return Image.new("RGB", (W, H), col)


def to_pil(arr_bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR array -> PIL RGB image."""
    return Image.fromarray(np.ascontiguousarray(arr_bgr[:, :, ::-1]))


def paste_alpha(dst: Image.Image, src: Image.Image, xy, alpha: float = 1.0, mask: Optional[Image.Image] = None):
    """Paste ``src`` onto ``dst`` at ``xy`` with global opacity ``alpha``."""
    if alpha <= 0.003:
        return
    xy = (int(round(xy[0])), int(round(xy[1])))
    if alpha >= 0.997 and mask is None:
        dst.paste(src, xy)
        return
    m = mask if mask is not None else Image.new("L", src.size, 255)
    if alpha < 0.997:
        m = m.point(lambda v: int(v * alpha))
    dst.paste(src, xy, m)


@lru_cache(maxsize=32)
def rounded_mask(size: Tuple[int, int], radius: int) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return m


def vignette(img: Image.Image, strength: float = 0.55) -> Image.Image:
    return Image.composite(img, Image.new("RGB", img.size, BG), _vignette_mask(img.size, strength))


@lru_cache(maxsize=4)
def _vignette_mask(size, strength):
    w, h = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    m = 1 - strength * np.clip(d, 0, 1) ** 2.2
    return Image.fromarray((m * 255).astype(np.uint8))


def darken(img: Image.Image, amount: float) -> Image.Image:
    return Image.blend(img, Image.new("RGB", img.size, BG), clamp(amount))


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius)) if radius > 0.2 else img


def cover(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Scale-and-crop ``img`` so it fills ``size``."""
    w, h = img.size
    s = max(size[0] / w, size[1] / h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    x = (img.width - size[0]) // 2
    y = (img.height - size[1]) // 2
    return img.crop((x, y, x + size[0], y + size[1]))


# ── Recurring overlay elements ─────────────────────────────────────────────
def chip(draw, xy, label: str, col, size: int = 20, alpha: float = 1.0, weight: int = 600,
         pad=(12, 6), filled: bool = False, anchor: str = "la"):
    """A small rounded label; returns its right edge."""
    w_ = text_width(label, size, weight) + 2 * pad[0]
    h_ = size + 2 * pad[1]
    x, y = xy
    if anchor[0] == "m":
        x -= w_ / 2
    elif anchor[0] == "r":
        x -= w_
    if anchor[1] == "m":
        y -= h_ / 2
    elif anchor[1] == "b":
        y -= h_
    if filled:
        draw.rounded_rectangle((x, y, x + w_, y + h_), radius=h_ / 2, fill=rgba(col, 0.92 * alpha))
        text(draw, (x + pad[0], y + h_ / 2 + 1), label, size, weight, fill=BG, alpha=alpha, anchor="lm")
    else:
        draw.rounded_rectangle((x, y, x + w_, y + h_), radius=h_ / 2, fill=rgba(BG, 0.72 * alpha),
                               outline=rgba(col, 0.9 * alpha), width=2)
        text(draw, (x + pad[0], y + h_ / 2 + 1), label, size, weight, fill=col, alpha=alpha, anchor="lm")
    return x + w_


def section_title(draw, kicker: str, title: str, alpha: float = 1.0, x: int = 96, y: int = 70,
                  kicker_col=HOT):
    text(draw, (x, y), kicker.upper(), 20, 700, fill=kicker_col, alpha=alpha, tracking=0.18)
    text(draw, (x, y + 34), title, 46, 700, fill=FG, alpha=alpha)


def corner_box(draw, box, col, alpha=1.0, width=3, frac=0.28):
    """Bounding box drawn as four corner brackets plus a faint full outline."""
    x0, y0, x1, y1 = box
    w_, h_ = x1 - x0, y1 - y0
    lx, ly = max(6, w_ * frac), max(6, h_ * frac)
    c = rgba(col, alpha)
    draw.rectangle(box, outline=rgba(col, 0.35 * alpha), width=1)
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line([(cx, cy + sy * ly), (cx, cy), (cx + sx * lx, cy)], fill=c, width=width, joint="curve")


# ── Timeline and output ────────────────────────────────────────────────────
class Scene:
    """A piece of the film: ``duration`` seconds, ``render(t)`` -> 1920x1080 RGB image."""
    name = "scene"
    duration = 5.0

    def prepare(self):
        """Load or compute whatever the scene needs; called once before rendering."""

    def render(self, t: float) -> Image.Image:  # pragma: no cover - abstract
        raise NotImplementedError


def crossfade(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    return Image.blend(a, b, ease_in_out(t))


def timeline(scenes: Sequence[Scene], overlap: float):
    """Start time of every scene and the total length; neighbours overlap by ``overlap`` s."""
    starts, t = [], 0.0
    for i, s in enumerate(scenes):
        starts.append(t)
        t += s.duration - (overlap if i < len(scenes) - 1 else 0.0)
    return starts, t


def frame_plan(scenes: Sequence[Scene], fps: float, overlap: float):
    """Per output frame: a list of ``(scene_index, local_time)`` and the crossfade weight.

    Outside an overlap the list has one entry; inside it has two and the
    weight says how far the fade to the second has progressed.
    """
    starts, total = timeline(scenes, overlap)
    for k in range(int(round(total * fps))):
        tg = k / fps
        active = [(i, tg - st) for i, st in enumerate(starts)
                  if st - 1e-9 <= tg < st + scenes[i].duration - 1e-9]
        if len(active) >= 2:
            (i0, l0), (i1, l1) = active[-2], active[-1]
            yield [(i0, l0), (i1, l1)], l1 / overlap
        else:
            yield active[-1:] or [(len(scenes) - 1, tg - starts[-1])], 0.0


class VideoWriter:
    """Pipe RGB frames into ffmpeg (the imageio-ffmpeg binary or ``ffmpeg`` on PATH)."""

    def __init__(self, path: Path, fps: float, size=(W, H), crf: int = 18, preset: str = "slow"):
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            exe = "ffmpeg"
        self.size = size
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-",
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img: Image.Image):
        if img.size != self.size:
            img = img.resize(self.size, Image.LANCZOS)
        self.proc.stdin.write(np.asarray(img.convert("RGB")).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
