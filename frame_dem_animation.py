#!/usr/bin/env python3
"""
Animate how a BAMBI frame relates to the terrain it was recorded over.

The animation has four phases:

1. **Hold** -- a single thermal or RGB frame fills the screen, seen exactly
   as the drone camera saw it (looking down along the optical axis).
2. **Roll** -- the viewpoint turns around the image's x-axis (or y-axis with
   ``--roll-axis y``).  At 90 degrees the image is seen edge-on and collapses
   to a one-pixel-thick line.  Above it the camera frustum becomes visible,
   below it the digital elevation model (DEM), which turns into the relief
   line of the cross-section under the image.  With a smaller roll (e.g.
   ``--roll 45``) the image stays a textured plane and everything is animated
   in 3D over the shaded DEM surface.
3. **Pause** -- a short delay.
4. **Fall** -- the image drops towards the terrain.  Each pixel comes to rest
   the moment it touches the DEM, the others keep falling until every pixel
   sits on the relief.  Optionally (``--neighbors K``) the neighbouring frames
   before and after the central one then fade in together with their own
   frustums and fall onto the terrain in sync, which is what an ALFS (airborne
   light-field sampling) integral rendering does.  ``--roll-axis y`` shows the
   flight direction across the screen so the neighbour cameras spread out.

Inputs are the artefacts the rest of this repository produces:

* ``<id>_matched_processed.mp4`` (thermal left, RGB right) or a folder of
  frames written by ``frame_extraction.py``;
* ``<id>_matched_poses.json`` (``lat``/``lng``/``alt`` and ``pitch``/``roll``/
  ``yaw``, or the ``location``/``rotation`` fields added by
  ``add_relative_dem_position_to_poses.py``);
* ``<id>_matched_dem.tif`` and optionally ``<id>_matched_dem.json`` from
  ``dem_from_poses.py``.

Examples::

    # 90 degree roll, thermal, single frame
    python frame_dem_animation.py --video bambi_downloads/146_matched_processed.mp4 \\
        --poses bambi_downloads/146_matched_poses.json \\
        --dem bambi_downloads/146_matched_dem.tif --frame 2125 -o 146_thermal.mp4

    # RGB, ALFS-style: 10 frames before and after (every 3rd frame) fall in sync
    python frame_dem_animation.py --frames-dir 146_frames --poses ... --dem ... \\
        --frame 2125 --modality rgb --neighbors 10 --neighbor-step 3 --roll-axis y -o 146_alfs.mp4

    # 45 degree roll, animated in 3D
    python frame_dem_animation.py ... --roll 45 -o 146_3d.mp4

    # no data at hand: a synthetic terrain, flight and frames
    python frame_dem_animation.py --demo -o demo.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

FRAME_SUFFIX_DIGITS = 8          # frame_extraction.py writes <stem>_00001234.png
DEFAULT_FOVY = 50.0              # vertical field of view in degrees (see introduction.ipynb)
EDGE_ON_TOLERANCE_DEG = 0.5      # |roll - 90| below this: edge-on (line) mode


# ============================================================================
# Small helpers
# ============================================================================

def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def parse_color(text: str) -> np.ndarray:
    """'#rrggbb', 'r,g,b' (0-255) or a few names -> float RGB in [0, 1]."""
    names = {
        "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
        "grey": (128, 128, 128), "darkgray": (32, 32, 36), "darkgrey": (32, 32, 36),
    }
    text = text.strip().lower()
    if text in names:
        rgb = names[text]
    elif text.startswith("#") and len(text) == 7:
        rgb = tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))
    else:
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"cannot parse colour '{text}'")
        rgb = tuple(parts)
    return np.array(rgb, dtype=np.float32) / 255.0


def to_cv_color(rgb01: np.ndarray, alpha: float = 1.0) -> tuple:
    """float RGB in [0,1] -> BGR tuple for OpenCV (alpha pre-multiplied later)."""
    c = np.clip(rgb01, 0, 1) * 255.0
    return (float(c[2]), float(c[1]), float(c[0]))


# ============================================================================
# Camera geometry (AlfsPY convention)
# ============================================================================

def camera_basis(tilt_deg: float, roll_deg: float, heading_deg: float):
    """
    Camera axes in an east/north/up world frame.

    ``tilt`` is measured from nadir (0 looks straight down, 90 at the horizon),
    ``heading`` is clockwise from north, ``roll`` turns the image about the
    optical axis.  This mirrors ``alfspy.core.util.pyrrs.quaternion_from_drone_pose``:
    a nadir camera's image top points along the heading.

    Returns (forward, right, up) -- ``up`` is the image's up direction.
    """
    t, ro, h = (math.radians(a) for a in (tilt_deg, roll_deg, heading_deg))
    f_level = np.array([math.sin(h), math.cos(h), 0.0])
    r_level = np.array([math.cos(h), -math.sin(h), 0.0])
    world_up = np.array([0.0, 0.0, 1.0])

    forward = math.sin(t) * f_level - math.cos(t) * world_up
    up = math.cos(t) * f_level + math.sin(t) * world_up
    right = r_level

    right_r = math.cos(ro) * right + math.sin(ro) * up
    up_r = -math.sin(ro) * right + math.cos(ro) * up
    return unit(forward), unit(right_r), unit(up_r)


@dataclass
class CameraPose:
    index: int
    position: np.ndarray          # (3,) relative to the DEM origin
    tilt: float                   # degrees from nadir
    roll: float
    heading: float
    fovy: float

    @property
    def basis(self):
        return camera_basis(self.tilt, self.roll, self.heading)


# ============================================================================
# DEM
# ============================================================================

class DEM:
    """A north-up elevation grid in DEM-origin-relative coordinates with bilinear lookup."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, z: np.ndarray):
        # xs ascending (columns), ys ascending (rows), z[row, col]
        assert z.shape == (len(ys), len(xs))
        self.xs = xs.astype(np.float64)
        self.ys = ys.astype(np.float64)
        self.z = z.astype(np.float32)
        self.dx = float(xs[1] - xs[0])
        self.dy = float(ys[1] - ys[0])

    # --- constructors -------------------------------------------------------

    @classmethod
    def from_geotiff(cls, path: Path, origin: Optional[list] = None):
        try:
            import rasterio
        except ImportError:  # pragma: no cover
            sys.exit("rasterio is required to read a GeoTIFF DEM (pip install rasterio)")

        with rasterio.open(path) as src:
            elev = src.read(1).astype(np.float64)
            if src.nodata is not None:
                elev[elev == src.nodata] = np.nan
            elev[elev == 0] = np.nan          # dem_from_poses.py treats 0 as void, too
            tr = src.transform
            if abs(tr.b) > 1e-9 or abs(tr.d) > 1e-9:
                sys.exit("rotated GeoTIFFs are not supported")
            bounds = src.bounds
            crs = str(src.crs)
            w, h = src.width, src.height

        if origin is None:
            valid = elev[~np.isnan(elev)]
            origin = [bounds.left, bounds.bottom, float(valid.min())]
        ox, oy, oz = (float(v) for v in origin[:3])

        # pixel centres
        xs = tr.c + tr.a * (np.arange(w) + 0.5) - ox
        ys = tr.f + tr.e * (np.arange(h) + 0.5) - oy
        if ys[0] > ys[-1]:
            ys = ys[::-1]
            elev = elev[::-1]
        elev = fill_nan_nearest(elev) - oz
        dem = cls(xs, ys, elev)
        dem.crs = crs
        dem.origin = [ox, oy, oz]
        return dem

    # --- queries --------------------------------------------------------------

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear elevation at (x, y); coordinates are clamped to the grid."""
        fx = (np.asarray(x, dtype=np.float64) - self.xs[0]) / self.dx
        fy = (np.asarray(y, dtype=np.float64) - self.ys[0]) / self.dy
        fx = np.clip(fx, 0, len(self.xs) - 1.000001)
        fy = np.clip(fy, 0, len(self.ys) - 1.000001)
        c0 = np.floor(fx).astype(np.int64)
        r0 = np.floor(fy).astype(np.int64)
        tx = (fx - c0).astype(np.float32)
        ty = (fy - r0).astype(np.float32)
        z = self.z
        top = z[r0, c0] * (1 - tx) + z[r0, c0 + 1] * tx
        bot = z[r0 + 1, c0] * (1 - tx) + z[r0 + 1, c0 + 1] * tx
        return top * (1 - ty) + bot * ty

    def crop_points(self, xmin, xmax, ymin, ymax, spacing: float):
        """Regular point cloud (N,3) of the DEM inside a box, plus hillshade (N,)."""
        xmin = max(xmin, self.xs[0]); xmax = min(xmax, self.xs[-1])
        ymin = max(ymin, self.ys[0]); ymax = min(ymax, self.ys[-1])
        nx = max(2, int((xmax - xmin) / spacing) + 1)
        ny = max(2, int((ymax - ymin) / spacing) + 1)
        gx = np.linspace(xmin, xmax, nx)
        gy = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(gx, gy)
        Z = self.sample(X, Y)
        # hillshade, light from the north-west, 45 degrees up
        dzdx = np.gradient(Z, gx, axis=1)
        dzdy = np.gradient(Z, gy, axis=0)
        az, alt = math.radians(315.0), math.radians(45.0)
        nx_, ny_, nz_ = -dzdx, -dzdy, np.ones_like(Z)
        nrm = np.sqrt(nx_ ** 2 + ny_ ** 2 + nz_ ** 2)
        lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
        shade = np.clip((nx_ * lx + ny_ * ly + nz_ * lz) / nrm, 0, 1)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
        return pts, shade.ravel().astype(np.float32)


def fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    """Replace NaNs with the nearest valid value (void areas at the DEM border)."""
    mask = np.isnan(a)
    if not mask.any():
        return a
    if mask.all():
        sys.exit("the DEM contains no valid elevation values")
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]


# ============================================================================
# Poses
# ============================================================================

def load_poses(path: Path, dem: DEM, pitch_convention: str, fovy_default: float) -> list[Optional[dict]]:
    with open(path) as f:
        data = json.load(f)
    images = data["images"]
    return [_pose_entry(i, img, dem, pitch_convention, fovy_default) for i, img in enumerate(images)]


def _pose_entry(i, img, dem, pitch_convention, fovy_default):
    return {"raw": img, "index": i, "dem": dem, "pitch_convention": pitch_convention, "fovy_default": fovy_default}


_transformer_cache: dict = {}


def pose_from_entry(entry: dict) -> CameraPose:
    img, dem = entry["raw"], entry["dem"]
    if "location" in img:
        pos = np.array(img["location"], dtype=np.float64)
    else:
        crs = getattr(dem, "crs", None)
        if crs is None:
            sys.exit("poses have no 'location' field and the DEM has no CRS; run "
                     "add_relative_dem_position_to_poses.py first")
        try:
            from pyproj import Transformer
        except ImportError:  # pragma: no cover
            sys.exit("pyproj is required to convert lat/lng poses (pip install pyproj)")
        if crs not in _transformer_cache:
            _transformer_cache[crs] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        e, n = _transformer_cache[crs].transform(img["lng"], img["lat"])
        ox, oy, oz = dem.origin
        pos = np.array([e - ox, n - oy, float(img["alt"]) - oz])

    if "rotation" in img:
        tilt, roll, heading = (float(v) for v in img["rotation"][:3])
    else:
        tilt, roll, heading = float(img.get("pitch", 0)), float(img.get("roll", 0)), float(img.get("yaw", 0))
    if entry["pitch_convention"] == "dji":
        tilt = tilt + 90.0            # DJI gimbal pitch: -90 = straight down

    fov = img.get("fovy", entry["fovy_default"])
    if isinstance(fov, (list, tuple)):
        fov = fov[0]
    return CameraPose(entry["index"], pos, tilt % 360.0, roll % 360.0, heading % 360.0, float(fov))


# ============================================================================
# Frame sources
# ============================================================================

class FrameSource:
    """Yields the thermal or RGB frame for a frame index as float RGB (H, W, 3) in [0, 1]."""

    def __init__(self, modality: str, video: Optional[Path] = None,
                 frames_dir: Optional[Path] = None, image: Optional[Path] = None):
        self.modality = modality
        self.video = video
        self.frames_dir = frames_dir
        self.image = image
        self._cache: dict[int, np.ndarray] = {}

    def preload(self, indices: list[int]):
        if self.video is not None:
            self._read_video(sorted(set(indices)))

    def get(self, index: int) -> np.ndarray:
        if index in self._cache:
            return self._cache[index]
        if self.image is not None:
            img = cv2.imread(str(self.image), cv2.IMREAD_COLOR)
            if img is None:
                sys.exit(f"cannot read image {self.image}")
        elif self.frames_dir is not None:
            img = self._read_frame_file(index)
        elif self.video is not None:
            self._read_video([index])
            return self._cache[index]
        else:
            raise RuntimeError("no frame source")
        self._cache[index] = self._to_rgb01(img)
        return self._cache[index]

    def _read_frame_file(self, index: int) -> np.ndarray:
        suffix = f"_{index:0{FRAME_SUFFIX_DIGITS}d}"
        candidates = []
        for folder in (self.frames_dir / self.modality, self.frames_dir):
            if folder.is_dir():
                candidates += sorted(p for p in folder.iterdir()
                                     if p.stem.endswith(suffix) and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not candidates:
            sys.exit(f"no frame file '*{suffix}.png' for index {index} under {self.frames_dir}")
        img = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"cannot read {candidates[0]}")
        return img

    def _read_video(self, indices: list[int]):
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            sys.exit(f"cannot open video {self.video}")
        wanted = [i for i in indices if i not in self._cache]
        if not wanted:
            return
        first, last = wanted[0], wanted[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        wanted_set = set(wanted)
        for idx in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                sys.exit(f"video ended before frame {idx}")
            if idx in wanted_set:
                w = frame.shape[1]
                half = frame[:, : w // 2] if self.modality == "thermal" else frame[:, w // 2:]
                self._cache[idx] = self._to_rgb01(half)
        cap.release()

    @staticmethod
    def _to_rgb01(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ============================================================================
# Scene: one entry per animated frame (central + neighbours)
# ============================================================================

@dataclass
class ShotGeometry:
    pose: CameraPose
    image: np.ndarray                       # (h, w, 3) float RGB, possibly downsampled
    plane_center: np.ndarray                # (3,)
    corners: np.ndarray                     # (4, 3) plane corners TL, TR, BR, BL
    p0: np.ndarray                          # (N, 3) initial pixel positions
    colors: np.ndarray                      # (N, 3)
    landing: np.ndarray                     # (N, 3) resting positions on the DEM
    line_mask: np.ndarray                   # (N,) bool: pixels of the edge-on line
    is_central: bool
    fall_mode: str
    # ray mode: parameter s at which each pixel lands, and the maximum
    s_land: Optional[np.ndarray] = None
    s_max: float = 1.0
    d_max: float = 0.0                      # vertical mode: largest drop
    profile: Optional[np.ndarray] = None    # (M, 3) relief cross-section under the line
    line_center: Optional[np.ndarray] = None
    line_dir: Optional[np.ndarray] = None
    line_extent: float = 0.0
    fall_progress: float = 0.0
    alpha: float = 1.0

    def positions(self) -> np.ndarray:
        p = self.fall_progress
        if self.fall_mode == "ray":
            s = 1.0 + (self.s_max - 1.0) * p
            s_eff = np.minimum(s, self.s_land)[:, None]
            return self.pose.position[None, :] + (self.p0 - self.pose.position[None, :]) * s_eff
        z = np.maximum(self.p0[:, 2] - self.d_max * p, self.landing[:, 2])
        out = self.p0.copy()
        out[:, 2] = z
        return out


def build_shot(pose: CameraPose, image: np.ndarray, dem: DEM, *, plane_height_frac: float,
               downsample: int, roll_axis: str, line_index: Optional[int], fall_mode: str,
               is_central: bool) -> ShotGeometry:
    fwd, right, up = pose.basis
    if fwd[2] >= -1e-3:
        sys.exit(f"frame {pose.index}: camera does not look downwards (tilt {pose.tilt:.1f} deg); "
                 "check --pitch-convention")

    img = image[::downsample, ::downsample] if downsample > 1 else image
    h, w = img.shape[:2]

    # --- where does the plane start? --------------------------------------
    # mean terrain height under the camera's nadir footprint, then place the
    # plane a fraction of the height above ground along the optical axis
    cam = pose.position
    ground_under = float(dem.sample(np.array([cam[0]]), np.array([cam[1]]))[0])
    agl = cam[2] - ground_under
    if agl <= 1.0:
        sys.exit(f"frame {pose.index}: camera is only {agl:.1f} m above the DEM")
    plane_z = ground_under + plane_height_frac * agl
    dist = (cam[2] - plane_z) / (-fwd[2])
    half = dist * math.tan(math.radians(pose.fovy) / 2.0)
    aspect = w / h
    half_x, half_y = half * aspect, half
    center = cam + dist * fwd

    corners = np.stack([
        center - half_x * right + half_y * up,   # TL
        center + half_x * right + half_y * up,   # TR
        center + half_x * right - half_y * up,   # BR
        center - half_x * right - half_y * up,   # BL
    ])

    # --- pixel positions on the plane --------------------------------------
    u = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
    v = (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0 - 1.0
    U, V = np.meshgrid(u, v)
    p0 = (center[None, None, :]
          + (U[..., None] * half_x) * right[None, None, :]
          - (V[..., None] * half_y) * up[None, None, :]).reshape(-1, 3).astype(np.float64)
    colors = img.reshape(-1, 3).astype(np.float32)

    # --- which pixels form the edge-on line? ---------------------------------
    if roll_axis == "x":
        li = h // 2 if line_index is None else min(max(line_index // downsample, 0), h - 1)
        line_mask = (np.repeat(np.arange(h), w) == li)
        line_dir, line_extent = right, half_x
        line_center = center - ((li + 0.5) / h * 2.0 - 1.0) * half_y * up
    else:
        li = w // 2 if line_index is None else min(max(line_index // downsample, 0), w - 1)
        line_mask = (np.tile(np.arange(w), h) == li)
        line_dir, line_extent = up, half_y
        line_center = center + ((li + 0.5) / w * 2.0 - 1.0) * half_x * right

    # --- landing positions ---------------------------------------------------
    shot = ShotGeometry(pose, img, center, corners, p0, colors, np.empty_like(p0),
                        line_mask, is_central, fall_mode)
    if fall_mode == "vertical":
        landing = p0.copy()
        landing[:, 2] = dem.sample(p0[:, 0], p0[:, 1])
        shot.landing = landing
        shot.d_max = float(np.max(p0[:, 2] - landing[:, 2]))
    else:
        s_land = march_rays(cam, p0, dem)
        shot.s_land = s_land
        shot.s_max = float(s_land.max())
        shot.landing = cam[None, :] + (p0 - cam[None, :]) * s_land[:, None]

    # the line's world position; the relief cross-section is sampled by the Animation
    shot.line_center = line_center
    shot.line_dir = line_dir
    shot.line_extent = line_extent
    return shot


def march_rays(cam: np.ndarray, p0: np.ndarray, dem: DEM) -> np.ndarray:
    """
    For rays cam + (p0 - cam) * s (s >= 1) find the first s at which the ray is
    below the terrain.  Coarse march followed by bisection, fully vectorised.
    """
    d = p0 - cam[None, :]
    dz = d[:, 2]
    if np.any(dz >= 0):
        sys.exit("ray fall mode needs every pixel ray to point downwards")
    z_floor = float(dem.z.min()) - 1.0
    s_end = (z_floor - cam[2]) / dz                     # per-ray s at which z reaches the floor
    s_end = np.maximum(s_end, 1.0 + 1e-6)
    n_coarse = 64
    lo = np.ones(len(p0))
    hi = s_end.copy()
    found = np.zeros(len(p0), dtype=bool)
    prev = lo.copy()
    for k in range(1, n_coarse + 1):
        s = 1.0 + (s_end - 1.0) * (k / n_coarse)
        pts = cam[None, :] + d * s[:, None]
        below = pts[:, 2] <= dem.sample(pts[:, 0], pts[:, 1])
        newly = below & ~found
        lo[newly] = prev[newly]
        hi[newly] = s[newly]
        found |= newly
        prev = s
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        pts = cam[None, :] + d * mid[:, None]
        below = pts[:, 2] <= dem.sample(pts[:, 0], pts[:, 1])
        hi = np.where(below, mid, hi)
        lo = np.where(below, lo, mid)
    return np.where(found, hi, s_end)


# ============================================================================
# View / framing
# ============================================================================

@dataclass
class View:
    e_x: np.ndarray
    e_y: np.ndarray
    dir: np.ndarray
    center: np.ndarray      # world point mapped to the screen centre
    scale: float            # pixels per metre

    def project(self, pts: np.ndarray, width: int, height: int):
        rel = pts - self.center[None, :]
        sx = rel @ self.e_x
        sy = rel @ self.e_y
        depth = rel @ self.dir
        px = width / 2.0 + sx * self.scale
        py = height / 2.0 - sy * self.scale
        return px, py, depth


def screen_basis(pose: CameraPose, roll_axis: str):
    """
    (forward, screen-right, screen-up) of the central frame at roll 0.

    For ``--roll-axis y`` the frame is shown turned by 90 degrees so that its
    y-axis runs across the screen; the viewpoint then turns about that axis and
    the edge-on line is an image column.  World up ends up pointing up on
    screen in both cases.
    """
    fwd, right, up = pose.basis
    if roll_axis == "x":
        return fwd, right, up
    return fwd, up, -right


def view_axes(fwd: np.ndarray, right: np.ndarray, up: np.ndarray, rho_deg: float):
    """Screen axes after turning the viewpoint by rho about the screen's x-axis."""
    c, s = math.cos(math.radians(rho_deg)), math.sin(math.radians(rho_deg))
    e_x = right
    e_y = c * up - s * fwd
    d = c * fwd + s * up
    return unit(e_x), unit(e_y), unit(d)


def fit_framing(pts: np.ndarray, e_x, e_y, width, height, margin: float):
    """Scale and world centre so that the points' projection fills the viewport."""
    sx = pts @ e_x
    sy = pts @ e_y
    w_world = max(sx.max() - sx.min(), 1e-6)
    h_world = max(sy.max() - sy.min(), 1e-6)
    scale = min(width * (1 - 2 * margin) / w_world, height * (1 - 2 * margin) / h_world)
    cx = 0.5 * (sx.max() + sx.min())
    cy = 0.5 * (sy.max() + sy.min())
    return scale, cx, cy


# ============================================================================
# Software renderer: z-buffered point splats + overlays
# ============================================================================

class Canvas:
    def __init__(self, width: int, height: int, bg: np.ndarray):
        self.w, self.h = width, height
        self.fb = np.empty((height, width, 3), dtype=np.float32)
        self.fb[:] = bg
        self.zbuf = np.full((height, width), np.inf, dtype=np.float32)

    def splat(self, px, py, depth, colors, size: int, alpha: float = 1.0):
        """Draw points with a size x size footprint, nearest depth wins."""
        if alpha <= 0.0 or len(px) == 0:
            return
        W, H = self.w, self.h
        ix = np.floor(px).astype(np.int64)
        iy = np.floor(py).astype(np.int64)
        pad = size + 1
        m = (ix >= -pad) & (ix < W + pad) & (iy >= -pad) & (iy < H + pad)
        if not m.any():
            return
        ix, iy, depth, colors = ix[m], iy[m], depth[m].astype(np.float32), colors[m]
        key = (iy + pad) * (W + 2 * pad) + (ix + pad)
        order = np.lexsort((depth, key))
        key_s = key[order]
        first = np.ones(len(key_s), dtype=bool)
        first[1:] = key_s[1:] != key_s[:-1]
        sel = order[first]
        ix, iy, depth, colors = ix[sel], iy[sel], depth[sel], colors[sel]

        off = -(size // 2)
        fb = self.fb.reshape(-1, 3)
        zb = self.zbuf.reshape(-1)
        for dy in range(size):
            for dx in range(size):
                x = ix + off + dx
                y = iy + off + dy
                inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                lin = y[inside] * W + x[inside]
                d = depth[inside]
                c = colors[inside]
                pass_z = d < zb[lin]
                lin, d, c = lin[pass_z], d[pass_z], c[pass_z]
                if alpha >= 1.0:
                    fb[lin] = c
                else:
                    fb[lin] = fb[lin] * (1.0 - alpha) + c * alpha
                zb[lin] = d

    def to_bgr8(self) -> np.ndarray:
        return cv2.cvtColor((np.clip(self.fb, 0, 1) * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def blend_overlay(base_bgr: np.ndarray, draw: Callable[[np.ndarray], None], alpha: float):
    """Run ``draw`` on a copy and blend the result over ``base_bgr`` in place."""
    if alpha <= 0.0:
        return
    if alpha >= 1.0:
        draw(base_bgr)
        return
    over = base_bgr.copy()
    draw(over)
    cv2.addWeighted(over, alpha, base_bgr, 1.0 - alpha, 0.0, dst=base_bgr)


# ============================================================================
# Timeline
# ============================================================================

@dataclass
class Timeline:
    hold: float
    roll: float
    pause: float
    fall: float
    neighbor_fade: float
    neighbor_fall: float
    neighbor_stagger: float
    neighbor_delay: float
    end_hold: float
    n_neighbors: int
    fall_easing: str

    def __post_init__(self):
        self.t_roll0 = self.hold
        self.t_roll1 = self.hold + self.roll
        self.t_fall0 = self.t_roll1 + self.pause
        self.t_fall1 = self.t_fall0 + self.fall
        self.neighbor_starts = [self.t_fall1 + self.neighbor_delay + i * self.neighbor_stagger
                                for i in range(self.n_neighbors)]
        last = self.t_fall1 if not self.neighbor_starts else \
            self.neighbor_starts[-1] + self.neighbor_fade + self.neighbor_fall
        self.total = last + self.end_hold

    def roll_progress(self, t: float) -> float:
        if self.roll <= 0:
            return 1.0 if t >= self.t_roll0 else 0.0
        return float(smoothstep((t - self.t_roll0) / self.roll))

    def _ease_fall(self, q: float) -> float:
        q = clamp01(q)
        return q * q if self.fall_easing == "gravity" else q

    def central_fall(self, t: float) -> float:
        if self.fall <= 0:
            return 1.0 if t >= self.t_fall0 else 0.0
        return self._ease_fall((t - self.t_fall0) / self.fall)

    def neighbor_state(self, i: int, t: float) -> tuple[float, float]:
        """(alpha, fall progress) of the i-th neighbour."""
        t0 = self.neighbor_starts[i]
        alpha = clamp01((t - t0) / self.neighbor_fade) if self.neighbor_fade > 0 else float(t >= t0)
        t1 = t0 + self.neighbor_fade
        fall = self._ease_fall((t - t1) / self.neighbor_fall) if self.neighbor_fall > 0 else float(t >= t1)
        return alpha, fall


# ============================================================================
# The animation
# ============================================================================

@dataclass
class Style:
    bg: np.ndarray
    fg: np.ndarray
    dem_cmap: str
    dem_alpha_behind: float
    frustum_color_central: np.ndarray
    frustum_color_neighbor: np.ndarray
    profile_color: np.ndarray
    fill_color: np.ndarray
    fill_alpha: float
    pixel_size: int
    dem_point_size: int
    caption: bool


class Animation:
    def __init__(self, shots: list[ShotGeometry], dem: DEM, timeline: Timeline, style: Style,
                 *, width: int, height: int, roll_deg: float, roll_axis: str,
                 dem_margin: float, caption_text: str, fit: str = "scene"):
        self.shots = shots
        self.central = next(s for s in shots if s.is_central)
        self.neighbors = [s for s in shots if not s.is_central]
        self.dem = dem
        self.tl = timeline
        self.style = style
        self.w, self.h = width, height
        self.roll_deg = roll_deg
        self.roll_axis = roll_axis
        self.edge_on = abs(roll_deg - 90.0) < EDGE_ON_TOLERANCE_DEG
        self.caption_text = caption_text

        self.basis = screen_basis(self.central.pose, roll_axis)
        self.axes0 = view_axes(*self.basis, 0.0)
        self.axes1 = view_axes(*self.basis, roll_deg)

        # framing at the start: the central image fills the screen
        self.scale0, cx, cy = fit_framing(self.central.corners, self.axes0[0], self.axes0[1],
                                          width, height, margin=0.04)
        self.center0 = self._center_from_screen(self.axes0, cx, cy, self.central.plane_center)

        # framing at the end: cameras, planes and landed pixels of every shot
        key_pts = []
        for s in shots:
            key_pts += [s.corners, s.landing[s.line_mask] if self.edge_on else s.landing[::7]]
            if fit == "scene":
                key_pts.append(s.pose.position[None, :])
        key_pts = np.concatenate(key_pts)
        self.scale1, cx, cy = fit_framing(key_pts, self.axes1[0], self.axes1[1], width, height, margin=0.06)
        self.center1 = self._center_from_screen(self.axes1, cx, cy, self.central.plane_center)

        # DEM point cloud around the action
        all_xy = np.concatenate([s.landing[:, :2] for s in shots] + [s.corners[:, :2] for s in shots]
                                + [s.pose.position[None, :2] for s in shots])
        foot = float(np.linalg.norm(self.central.corners[1] - self.central.corners[0]))
        pad = dem_margin * foot
        self.crop = (max(all_xy[:, 0].min() - pad, dem.xs[0]), min(all_xy[:, 0].max() + pad, dem.xs[-1]),
                     max(all_xy[:, 1].min() - pad, dem.ys[0]), min(all_xy[:, 1].max() + pad, dem.ys[-1]))
        spacing = max(1.35 * style.dem_point_size / self.scale1, min(dem.dx, dem.dy) * 0.5)
        self.dem_pts, shade = dem.crop_points(*self.crop, spacing)
        self.dem_colors = self._dem_colors(self.dem_pts[:, 2], shade)
        self.dem_spacing = spacing
        # depth offset that keeps lines and landed pixels in front of the terrain they rest on
        self.bias = 0.005 * foot + 1.5 * spacing

        # relief cross-sections under every shot's line, across the whole terrain crop
        for s in shots:
            s.profile = self._profile(s)

        # plane through the central line, used to hide terrain in front of the cross-section
        self.cut_normal = self.basis[2]
        self.cut_offset = float(self.central.line_center @ self.cut_normal)
        rel = (self.dem_pts @ self.cut_normal) - self.cut_offset
        self.dem_cut_min = float(rel.min())

    def _profile(self, s: ShotGeometry) -> np.ndarray:
        x0, x1, y0, y1 = self.crop
        L = math.hypot(x1 - x0, y1 - y0)
        lam = np.linspace(-L, L, 4096)
        pts = s.line_center[None, :] + lam[:, None] * s.line_dir[None, :]
        inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        pts = pts[inside]
        pts[:, 2] = self.dem.sample(pts[:, 0], pts[:, 1])
        return pts

    def _center_from_screen(self, axes, cx, cy, anchor):
        e_x, e_y, d = axes
        # the world point whose projection is (cx, cy) in the e_x/e_y frame, with the
        # depth of the anchor (depth is irrelevant for an orthographic view)
        return cx * e_x + cy * e_y + (anchor @ d) * d

    def _dem_colors(self, z: np.ndarray, shade: np.ndarray) -> np.ndarray:
        import matplotlib
        cmap = matplotlib.colormaps[self.style.dem_cmap]
        zmin, zmax = float(z.min()), float(z.max())
        t = (z - zmin) / max(zmax - zmin, 1e-6)
        base = cmap(0.15 + 0.7 * t)[:, :3].astype(np.float32)
        return base * (0.45 + 0.55 * shade)[:, None]

    # ---------------------------------------------------------------------

    def view_at(self, t: float) -> tuple[View, float]:
        k = self.tl.roll_progress(t)
        rho = self.roll_deg * k
        e_x, e_y, d = view_axes(*self.basis, rho)
        scale = self.scale0 * (1 - k) + self.scale1 * k
        center = self.center0 * (1 - k) + self.center1 * k
        return View(e_x, e_y, d, center, scale), rho

    def render(self, t: float) -> np.ndarray:
        view, rho = self.view_at(t)
        st = self.style
        W, H = self.w, self.h
        canvas = Canvas(W, H, st.bg)

        # --- state of every shot ----------------------------------------------
        self.central.fall_progress = self.tl.central_fall(t)
        self.central.alpha = 1.0
        for i, s in enumerate(self.neighbors):
            s.alpha, s.fall_progress = self.tl.neighbor_state(i, t)

        # --- edge-on cross-fade: DEM surface -> relief line + ground fill ------
        if self.edge_on:
            w_cut = clamp01((rho - 0.7 * self.roll_deg) / (0.3 * self.roll_deg))
        else:
            w_cut = 0.0
        surface_alpha = 1.0 - w_cut * (1.0 - st.dem_alpha_behind)
        line_alpha = w_cut if self.edge_on else clamp01(rho / max(self.roll_deg, 1e-6))

        # --- DEM surface ----------------------------------------------------
        if surface_alpha > 0.005:
            pts = self.dem_pts
            cols = self.dem_colors
            if self.edge_on and w_cut > 0:
                # sweep a cutting plane from the far edge to the cross-section
                thr = self.dem_cut_min * (1.0 - w_cut)
                keep = ((pts @ self.cut_normal) - self.cut_offset) >= thr
                pts, cols = pts[keep], cols[keep]
            px, py, depth = view.project(pts, W, H)
            size = int(np.clip(math.ceil(self.dem_spacing * view.scale * 1.6), st.dem_point_size, 12))
            canvas.splat(px, py, depth, cols, size, surface_alpha)

        # ground fill (2D look) below the central relief line
        if self.edge_on and w_cut > 0:
            px, py, _ = view.project(self.central.profile, W, H)
            poly = np.stack([px, py], axis=1)
            if px[0] > px[-1]:
                poly = poly[::-1]
            # continue the relief flat to both screen edges, then close along the bottom
            poly = np.concatenate([[[-10, poly[0, 1]]], poly, [[W + 10, poly[-1, 1]]],
                                   [[W + 10, H + 10], [-10, H + 10]]]).astype(np.int32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            a = st.fill_alpha * w_cut
            sel = mask > 0
            canvas.fb[sel] = canvas.fb[sel] * (1 - a) + st.fill_color * a

        # --- relief lines under every shot's line ------------------------------
        lines = []
        for s in self.shots:
            a = line_alpha * s.alpha * (1.0 if s.is_central else 0.45)
            if a <= 0.005:
                continue
            px, py, _ = view.project(s.profile, W, H)
            lines.append((np.stack([px, py], axis=1), a))
        if lines:
            bgr = canvas.to_bgr8()
            col = to_cv_color(st.profile_color)
            for poly, a in lines:
                pts = np.round(poly * 16).astype(np.int32).reshape(-1, 1, 2)
                blend_overlay(bgr, lambda im, pts=pts: cv2.polylines(im, [pts], False, col, 2, cv2.LINE_AA, shift=4), a)
            canvas.fb[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # --- the images -----------------------------------------------------
        bias = self.bias
        for s in self.shots:
            if s.alpha <= 0.005:
                continue
            pts = s.positions()
            cols = s.colors
            falling = (s.fall_progress > 0) or (t >= self.tl.t_fall0 and s.is_central)
            if self.edge_on and (falling or rho >= self.roll_deg - 1e-6):
                pts, cols = pts[s.line_mask], cols[s.line_mask]
                px, py, depth = view.project(pts, W, H)
                canvas.splat(px, py, depth - 2 * bias, cols, st.pixel_size, s.alpha)
            else:
                px, py, depth = view.project(pts, W, H)
                depth = depth - bias
                if self.edge_on:
                    # keep the future line row on top so the collapse is seamless
                    depth = depth - np.where(s.line_mask, bias, 0.0)
                canvas.splat(px, py, depth, cols, st.pixel_size, s.alpha)

        bgr = canvas.to_bgr8()

        # --- frustums, cameras and captions (drawn on top) ----------------------
        for s in self.shots:
            if s.alpha <= 0.005:
                continue
            color = st.frustum_color_central if s.is_central else st.frustum_color_neighbor
            self._draw_frustum(bgr, view, s, color, s.alpha * clamp01(0.15 + rho / max(self.roll_deg, 1e-6)))

        if st.caption:
            self._draw_caption(bgr, t, rho)
        return bgr

    def _draw_frustum(self, bgr, view: View, s: ShotGeometry, color, alpha):
        W, H = self.w, self.h
        apex = s.pose.position
        pts = np.concatenate([apex[None, :], s.corners])
        px, py, _ = view.project(pts, W, H)
        P = [(int(round(x * 16)), int(round(y * 16))) for x, y in zip(px, py)]   # 4-bit subpixels
        col = to_cv_color(color)

        def draw(im):
            for i in range(4):
                cv2.line(im, P[0], P[1 + i], col, 1, cv2.LINE_AA, shift=4)
            for i in range(4):
                cv2.line(im, P[1 + i], P[1 + (i + 1) % 4], col, 1, cv2.LINE_AA, shift=4)
            cv2.circle(im, P[0], 5 * 16, col, -1, cv2.LINE_AA, shift=4)
            cv2.circle(im, P[0], 8 * 16, col, 1, cv2.LINE_AA, shift=4)

        blend_overlay(bgr, draw, alpha)

    def _draw_caption(self, bgr, t, rho):
        st = self.style
        col = to_cv_color(st.fg)
        phase = ("frame" if t < self.tl.t_roll0 else
                 "roll" if t < self.tl.t_roll1 else
                 "pause" if t < self.tl.t_fall0 else
                 "projection" if t < self.tl.t_fall1 or not self.neighbors else "ALFS integral")
        text = f"{self.caption_text}   {phase}   roll {rho:5.1f} deg"
        cv2.putText(bgr, text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)


# ============================================================================
# Output
# ============================================================================

class Writer:
    def __init__(self, output: Path, fps: float, width: int, height: int):
        self.output = output
        self.fps = fps
        self.kind = None
        self.frames = []
        suffix = output.suffix.lower()
        if suffix in (".mp4", ".avi", ".mov", ".mkv"):
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix != ".avi" else "MJPG"))
            self.vw = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
            if not self.vw.isOpened():
                sys.exit(f"cannot open video writer for {output}")
            self.kind = "video"
        elif suffix == ".gif":
            self.kind = "gif"
        else:
            output.mkdir(parents=True, exist_ok=True)
            self.kind = "frames"
        self.count = 0

    def write(self, bgr: np.ndarray):
        if self.kind == "video":
            self.vw.write(bgr)
        elif self.kind == "gif":
            self.frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(str(self.output / f"frame_{self.count:05d}.png"), bgr)
        self.count += 1

    def close(self):
        if self.kind == "video":
            self.vw.release()
        elif self.kind == "gif":
            from PIL import Image
            ims = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in self.frames]
            ims[0].save(self.output, save_all=True, append_images=ims[1:],
                        duration=int(round(1000.0 / self.fps)), loop=0, optimize=False)


# ============================================================================
# Demo data
# ============================================================================

def make_demo(rng: np.random.Generator, modality: str, n_frames: int = 121):
    """A synthetic hilly terrain, a straight flight over it and frames rendered from a procedural texture."""
    size, step = 320.0, 1.0
    xs = np.arange(0, size + step, step)
    ys = np.arange(0, size + step, step)
    X, Y = np.meshgrid(xs, ys)
    Z = (18.0 * np.exp(-(((X - 120) ** 2 + (Y - 190) ** 2) / (2 * 55 ** 2)))
         + 12.0 * np.exp(-(((X - 230) ** 2 + (Y - 110) ** 2) / (2 * 40 ** 2)))
         - 6.0 * np.exp(-(((X - 170) ** 2 + (Y - 160) ** 2) / (2 * 22 ** 2)))
         + 0.04 * (X - 160) + 4.0 * np.sin(X / 23.0) * np.cos(Y / 31.0))
    Z += cv2.GaussianBlur(rng.normal(0, 0.6, Z.shape).astype(np.float32), (0, 0), 2.0)
    Z -= Z.min()
    dem = DEM(xs, ys, Z)
    dem.crs = None
    dem.origin = [0.0, 0.0, 0.0]

    heading = 35.0
    hr = math.radians(heading)
    start = np.array([120.0, 100.0])
    step_m = 1.2
    poses = []
    # procedural ground texture, sampled at the pixel footprints
    for i in range(n_frames):
        xy = start + step_m * i * np.array([math.sin(hr), math.cos(hr)])
        cam = np.array([xy[0], xy[1], 0.0])
        cam[2] = float(dem.sample(np.array([xy[0]]), np.array([xy[1]]))[0]) + 95.0
        img = {"lat": 0.0, "lng": 0.0, "alt": cam[2], "location": cam.tolist(),
               "rotation": [0.0, 0.0, heading], "fovy": [DEFAULT_FOVY]}
        poses.append({"raw": img, "index": i, "dem": dem, "pitch_convention": "nadir",
                      "fovy_default": DEFAULT_FOVY})
    return dem, poses, _demo_texture(rng, dem, modality, heading)


def _demo_texture(rng, dem: DEM, modality: str, heading: float):
    """Returns a callable frame_index, pose -> image, sampling a world texture under the camera."""
    X, Y = np.meshgrid(dem.xs, dem.ys)
    noise = rng.random((len(dem.ys), len(dem.xs))).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.2)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    road = np.exp(-((Y - (0.55 * X + 30)) ** 2) / (2 * 1.8 ** 2))
    trees = cv2.GaussianBlur(rng.random(noise.shape).astype(np.float32), (0, 0), 3.5)
    trees = (trees > 0.52).astype(np.float32)
    animals = np.zeros_like(noise)
    for _ in range(9):
        ax, ay = rng.uniform(100, 220), rng.uniform(100, 220)
        animals += np.exp(-(((X - ax) ** 2 + (Y - ay) ** 2) / (2 * 0.8 ** 2)))
    animals = np.clip(animals, 0, 1)

    if modality == "thermal":
        base = 0.25 + 0.25 * noise + 0.15 * trees + 0.1 * road
        tex = np.stack([base, base, base], axis=-1)
        tex = tex * (1 - animals[..., None]) + np.array([1.0, 1.0, 0.95]) * animals[..., None]
    else:
        g = np.stack([0.32 + 0.2 * noise, 0.45 + 0.25 * noise, 0.22 + 0.12 * noise], axis=-1)
        t = np.stack([0.12 + 0.05 * noise, 0.30 + 0.1 * noise, 0.12 + 0.0 * noise], axis=-1)
        r = np.array([0.6, 0.58, 0.55])
        tex = g * (1 - trees[..., None]) + t * trees[..., None]
        tex = tex * (1 - road[..., None]) + r * road[..., None]
        tex = tex * (1 - animals[..., None] * 0.7) + np.array([0.45, 0.3, 0.2]) * animals[..., None] * 0.7
    tex = np.clip(tex, 0, 1).astype(np.float32)
    tex_dem = [DEM(dem.xs, dem.ys, tex[..., c]) for c in range(3)]

    def render(pose: CameraPose, w: int = 1024, h: int = 1024) -> np.ndarray:
        fwd, right, up = pose.basis
        cam = pose.position
        ground = float(dem.sample(np.array([cam[0]]), np.array([cam[1]]))[0])
        dist = (cam[2] - ground) / (-fwd[2])
        half = dist * math.tan(math.radians(pose.fovy) / 2)
        u = (np.arange(w) + 0.5) / w * 2 - 1
        v = (np.arange(h) + 0.5) / h * 2 - 1
        U, V = np.meshgrid(u, v)
        P = cam + dist * fwd + (U[..., None] * half) * right - (V[..., None] * half) * up
        img = np.stack([td.sample(P[..., 0], P[..., 1]) for td in tex_dem], axis=-1)
        return np.clip(img, 0, 1).astype(np.float32)

    return render


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Animate a BAMBI frame rolling edge-on and falling onto the digital elevation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::", 1)[1] if "Examples::" in __doc__ else None,
    )
    src = p.add_argument_group("input data")
    src.add_argument("--video", type=Path, help="<id>_matched_processed.mp4 (thermal left, RGB right)")
    src.add_argument("--frames-dir", type=Path,
                     help="folder written by frame_extraction.py (with thermal/ and rgb/ subfolders)")
    src.add_argument("--image", type=Path, help="a single already extracted frame (no neighbours possible)")
    src.add_argument("--poses", type=Path, help="<id>_matched_poses.json")
    src.add_argument("--dem", type=Path, help="<id>_matched_dem.tif from dem_from_poses.py")
    src.add_argument("--dem-json", type=Path,
                     help="<id>_matched_dem.json with the DEM origin (default: derived from the GeoTIFF)")
    src.add_argument("--frame", type=int, help="index of the central frame")
    src.add_argument("--demo", action="store_true", help="use a synthetic terrain and flight instead of data")
    src.add_argument("--seed", type=int, default=7, help="random seed of the demo data")

    what = p.add_argument_group("what to show")
    what.add_argument("--modality", choices=["thermal", "rgb"], default="thermal")
    what.add_argument("--neighbors", type=int, default=0,
                      help="number of frames before AND after the central one that fall too (ALFS); 0 = single view")
    what.add_argument("--neighbor-step", type=int, default=1, help="frame stride between neighbours")
    what.add_argument("--roll", type=float, default=90.0,
                      help="how far the viewpoint turns (degrees); 90 = edge-on line, e.g. 45 = 3D")
    what.add_argument("--roll-axis", choices=["x", "y"], default="x",
                      help="turn about the image's x-axis (line = a row) or y-axis (line = a column)")
    what.add_argument("--line-index", type=int, default=None,
                      help="row (or column for --roll-axis y) shown in the edge-on view; default: centre")
    what.add_argument("--fall-mode", choices=["vertical", "ray"], default="vertical",
                      help="pixels fall straight down or slide along their camera rays")
    what.add_argument("--fall-easing", choices=["gravity", "linear"], default="gravity")
    what.add_argument("--plane-height", type=float, default=0.35,
                      help="start height of the image plane as a fraction of the height above ground")
    what.add_argument("--fovy", type=float, default=DEFAULT_FOVY, help="vertical field of view if the poses have none")
    what.add_argument("--pitch-convention", choices=["nadir", "dji"], default="nadir",
                      help="'nadir': pitch 0 looks straight down (BAMBI poses); 'dji': gimbal pitch -90 is down")
    what.add_argument("--image-downsample", type=int, default=None,
                      help="use every k-th pixel (default: 1 for edge-on, 2 for 3D rolls)")

    tim = p.add_argument_group("timing (seconds)")
    tim.add_argument("--fps", type=float, default=30.0)
    tim.add_argument("--hold", type=float, default=1.0, help="show the plain frame")
    tim.add_argument("--roll-duration", type=float, default=2.5)
    tim.add_argument("--pause", type=float, default=0.6, help="delay between roll and fall")
    tim.add_argument("--fall-duration", type=float, default=3.0)
    tim.add_argument("--neighbor-delay", type=float, default=0.4, help="after the central frame has landed")
    tim.add_argument("--neighbor-fade", type=float, default=0.5)
    tim.add_argument("--neighbor-fall", type=float, default=1.8)
    tim.add_argument("--neighbor-stagger", type=float, default=0.0,
                     help="delay between consecutive neighbours; 0 = all appear and fall in sync")
    tim.add_argument("--end-hold", type=float, default=1.5)

    out = p.add_argument_group("output and style")
    out.add_argument("-o", "--output", type=Path, default=Path("frame_dem_animation.mp4"),
                     help=".mp4/.avi (OpenCV), .gif (Pillow) or a folder for PNG frames")
    out.add_argument("--width", type=int, default=1280)
    out.add_argument("--height", type=int, default=720)
    out.add_argument("--theme", choices=["light", "dark"], default="dark")
    out.add_argument("--bg", type=parse_color, default=None, help="background colour (overrides the theme)")
    out.add_argument("--dem-cmap", default="gist_earth", help="matplotlib colormap of the DEM surface")
    out.add_argument("--dem-alpha-behind", type=float, default=0.3,
                     help="edge-on view: opacity of the terrain behind the cross-section (0 = hide)")
    out.add_argument("--dem-margin", type=float, default=0.6,
                     help="terrain shown around the frames, as a fraction of the footprint width")
    out.add_argument("--pixel-size", type=int, default=None, help="screen size of an image pixel (default: auto)")
    out.add_argument("--fit", choices=["scene", "image"], default="scene",
                     help="final framing: whole scene with the camera, or only image and relief (frustum runs off-screen)")
    out.add_argument("--dem-point-size", type=int, default=2)
    out.add_argument("--no-caption", action="store_true")
    out.add_argument("--caption", default=None, help="caption text (default: file name and frame index)")
    out.add_argument("--preview", type=float, default=None,
                     help="render only the frame at this time (seconds) as a PNG next to --output")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.demo:
        rng = np.random.default_rng(args.seed)
        dem, pose_entries, demo_render = make_demo(rng, args.modality)
        central_index = len(pose_entries) // 2 if args.frame is None else args.frame
        frame_getter = lambda idx: demo_render(pose_from_entry(pose_entries[idx]))  # noqa: E731
        caption_default = f"demo {args.modality}"
    else:
        missing = [n for n in ("poses", "dem", "frame") if getattr(args, n) is None]
        if missing:
            sys.exit("missing: " + ", ".join("--" + m for m in missing) + " (or use --demo)")
        if not (args.video or args.frames_dir or args.image):
            sys.exit("give one of --video, --frames-dir or --image")
        origin = None
        if args.dem_json is not None:
            with open(args.dem_json) as f:
                origin = json.load(f).get("origin")
        dem = DEM.from_geotiff(args.dem, origin)
        pose_entries = load_poses(args.poses, dem, args.pitch_convention, args.fovy)
        central_index = args.frame
        source = FrameSource(args.modality, args.video, args.frames_dir, args.image)
        frame_getter = source.get
        caption_default = f"{(args.video or args.frames_dir or args.image).stem}  {args.modality}  #{central_index}"

    if args.image is not None and args.neighbors > 0:
        sys.exit("--image holds a single frame; use --video or --frames-dir for --neighbors")

    # which frames take part, ordered by distance from the centre (1 before, 1 after, 2 before, ...)
    indices = [central_index]
    for k in range(1, args.neighbors + 1):
        for sign in (-1, 1):
            idx = central_index + sign * k * args.neighbor_step
            if 0 <= idx < len(pose_entries):
                indices.append(idx)
    if not args.demo:
        source.preload(indices)

    edge_on = abs(args.roll - 90.0) < EDGE_ON_TOLERANCE_DEG
    downsample = args.image_downsample or (1 if edge_on else 2)

    shots = []
    for n, idx in enumerate(indices):
        pose = pose_from_entry(pose_entries[idx])
        print(f"frame {idx:6d}: position {np.round(pose.position, 1).tolist()}  "
              f"tilt {pose.tilt:.1f}  roll {pose.roll:.1f}  heading {pose.heading:.1f}")
        image = frame_getter(idx)
        shots.append(build_shot(pose, image, dem, plane_height_frac=args.plane_height,
                                downsample=downsample, roll_axis=args.roll_axis,
                                line_index=args.line_index, fall_mode=args.fall_mode, is_central=(n == 0)))

    timeline = Timeline(args.hold, args.roll_duration, args.pause, args.fall_duration,
                        args.neighbor_fade, args.neighbor_fall, args.neighbor_stagger, args.neighbor_delay,
                        args.end_hold, len(indices) - 1, args.fall_easing)

    dark = args.theme == "dark"
    bg = args.bg if args.bg is not None else (parse_color("#101418") if dark else parse_color("white"))
    fg = parse_color("#e8e8e8") if dark else parse_color("#202020")
    style = Style(
        bg=bg, fg=fg, dem_cmap=args.dem_cmap, dem_alpha_behind=args.dem_alpha_behind,
        frustum_color_central=parse_color("#4fc3f7") if dark else parse_color("#0277bd"),
        frustum_color_neighbor=parse_color("#ffb74d") if dark else parse_color("#ef6c00"),
        profile_color=parse_color("#d0d0d0") if dark else parse_color("#3e2723"),
        fill_color=parse_color("#3a3f45") if dark else parse_color("#d7ccc8"),
        fill_alpha=0.85,
        pixel_size=args.pixel_size or 0, dem_point_size=max(1, args.dem_point_size),
        caption=not args.no_caption,
    )

    anim = Animation(shots, dem, timeline, style, width=args.width, height=args.height,
                     roll_deg=args.roll, roll_axis=args.roll_axis, dem_margin=args.dem_margin,
                     caption_text=args.caption if args.caption is not None else caption_default, fit=args.fit)
    if style.pixel_size <= 0:
        # an image pixel's size on screen at the end of the roll, but at least 2 px so the line stays closed
        h, w = shots[0].image.shape[:2]
        px_world = np.linalg.norm(shots[0].corners[1] - shots[0].corners[0]) / w
        style.pixel_size = int(np.clip(math.ceil(px_world * anim.scale1 * 1.2), 3 if edge_on else 2, 8))

    n_frames = int(math.ceil(timeline.total * args.fps))
    print(f"phases: hold {timeline.hold}s, roll {timeline.roll}s, pause {timeline.pause}s, "
          f"fall {timeline.fall}s, neighbours {len(indices) - 1}, total {timeline.total:.1f}s "
          f"({n_frames} frames at {args.fps:g} fps), edge-on: {edge_on}, pixel size {style.pixel_size}")

    if args.preview is not None:
        bgr = anim.render(args.preview)
        out = args.output.with_suffix("") .with_name(args.output.stem + f"_t{args.preview:.2f}.png")
        cv2.imwrite(str(out), bgr)
        print(f"wrote {out}")
        return

    writer = Writer(args.output, args.fps, args.width, args.height)
    try:
        for i in range(n_frames):
            t = i / args.fps
            writer.write(anim.render(t))
            if i % 30 == 0 or i == n_frames - 1:
                print(f"\r  frame {i + 1}/{n_frames}  t={t:5.2f}s", end="", flush=True)
    finally:
        writer.close()
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
