#!/usr/bin/env python3
"""
Show what lens undistortion does for the ray casting: geo-reference the boxed
animals of a flight twice, through the undistorted frame the dataset ships and
through the raw frame as it came off the drone with the distortion term left
out, and show both results as a video.

Same camera pose, same terrain model, same box -- only the pixel-to-ray mapping
differs.  The Mavic 3T thermal lens has barrel distortion (k1 ~ -0.37): a raw
pixel near the frame edge sits closer to the image centre than its true viewing
direction, so a ray cast straight through the raw pixel lands short of the
animal.

Panels: the raw thermal frame (640 x 512, as recorded) with the boxes mapped
back into it, the undistorted frame with the dataset's boxes, and a top-down
map with both ground positions per animal, their offsets and the accumulated
trails.  Below: the ground offset over the whole raw frame for the current
pose, and the running numbers.

Needs the raw thermal recording next to the SRT files of the flight (the `raw`
dataset version), the base-version files and the terrain model produced by
dem_from_poses.py.  The undistortion the dataset was built with is recovered
from `<id>_mask_t.png`: its border is the image of the raw frame under the
undistortion map, which pins the new camera matrix down to the pixel.
"""
import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import frame_dem_animation as fa   # noqa: E402
import srt_airdata_sync_eval as se             # noqa: E402

W = H = 1024
RAW_W, RAW_H = 640, 512
COL = {"undist": (180, 119, 31), "raw": (0, 128, 255)}     # BGR: blue / orange
NAME = {"undist": "ray through the undistorted pixel (BAMBI pipeline)",
        "raw": "ray through the raw pixel, distortion ignored"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
INK, GREY, LIGHT = (30, 30, 30), (110, 110, 110), (245, 245, 245)


# ----------------------------------------------------------------------------- calibration

def fit_new_camera_matrix(mask: np.ndarray, mtx: np.ndarray, dist: np.ndarray, raw_size, verbose=True):
    """
    Recover the new camera matrix the dataset frames were undistorted with.

    The valid-pixel mask is the raw frame's footprint under the undistortion
    map, so the focal length and principal point of the new matrix are the
    ones whose synthetic footprint matches the mask best.  Coordinate descent
    from a centred 50 degree start, coarse to fine.
    """
    h, w = mask.shape
    ones = np.full((raw_size[1], raw_size[0]), 255, np.uint8)

    def score(f, cx, cy):
        k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        mx, my = cv2.initUndistortRectifyMap(mtx, dist, None, k, (w, h), cv2.CV_32FC1)
        return float(((cv2.remap(ones, mx, my, cv2.INTER_LINEAR) > 127) == mask).mean())

    p = [h / 2 / math.tan(math.radians(25)), (w - 1) / 2, (h - 1) / 2]
    best = score(*p)
    for step in (40, 20, 10, 5, 2, 1, 0.5):
        improved = True
        while improved:
            improved = False
            for i in range(3):
                for d in (-step, step):
                    q = list(p)
                    q[i] += d
                    s = score(*q)
                    if s > best + 1e-7:
                        best, p, improved = s, q, True
    fovy = 2 * math.degrees(math.atan(h / 2 / p[0]))
    if verbose:
        print(f"undistortion recovered from the mask: f = {p[0]:.1f} px, principal point "
              f"({p[1]:.1f}, {p[2]:.1f}), fovy {fovy:.2f} deg, mask agreement {best:.4%}")
    return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]]), fovy


class Lens:
    """Raw <-> undistorted pixel mapping and the two ray models."""

    def __init__(self, calib: dict, new_k: np.ndarray):
        self.mtx = np.asarray(calib["mtx"], dtype=np.float64)
        self.dist = np.asarray(calib["dist"], dtype=np.float64).reshape(-1)
        self.new_k = new_k
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, new_k, (W, H), cv2.CV_32FC1)

    def undistort(self, raw_img):
        return cv2.remap(raw_img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def to_raw(self, uv: np.ndarray) -> np.ndarray:
        """Undistorted pixel coordinates (N, 2) -> raw pixel coordinates (N, 2)."""
        n = self.normalized_undist(uv)
        pts = np.concatenate([n, np.ones((len(n), 1))], axis=1).reshape(-1, 1, 3)
        raw, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.mtx, self.dist)
        return raw.reshape(-1, 2)

    def normalized_undist(self, uv: np.ndarray) -> np.ndarray:
        """Undistorted pixel -> normalized image coordinates (the correct ray)."""
        k = self.new_k
        return np.stack([(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1]], axis=1)

    def normalized_raw_naive(self, uv_raw: np.ndarray) -> np.ndarray:
        """Raw pixel treated as if the lens were a pinhole: distortion ignored."""
        k = self.mtx
        return np.stack([(uv_raw[:, 0] - k[0, 2]) / k[0, 0], (uv_raw[:, 1] - k[1, 2]) / k[1, 1]], axis=1)


def cast(cam, fwd, right, up, normalized, dem):
    """Normalized image coordinates (N, 2) -> ground points (N, 3)."""
    p0 = cam[None, :] + fwd[None, :] + normalized[:, :1] * right[None, :] - normalized[:, 1:] * up[None, :]
    s = fa.march_rays(cam, p0, dem)
    return cam[None, :] + (p0 - cam[None, :]) * s[:, None]


# ----------------------------------------------------------------------------- drawing helpers

def text(img, s, xy, scale=0.5, color=INK, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def fit_frame(img, box_w, box_h):
    """Scale an image to fit a box, keeping the aspect ratio; returns image and scale."""
    s = min(box_w / img.shape[1], box_h / img.shape[0])
    return cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA), s


def error_field_panel(lens, cam, fwd, right, up, dem, size, agl):
    """Ground offset (m) of the naive ray over the whole raw frame, as an image with contours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    step = 8
    us = np.arange(step / 2, RAW_W, step)
    vs = np.arange(step / 2, RAW_H, step)
    U, V = np.meshgrid(us, vs)
    raw_px = np.stack([U.ravel(), V.ravel()], axis=1)
    naive = lens.normalized_raw_naive(raw_px)
    true = cv2.undistortPoints(raw_px.reshape(-1, 1, 2), lens.mtx, lens.dist).reshape(-1, 2)
    g_naive = cast(cam, fwd, right, up, naive, dem)
    g_true = cast(cam, fwd, right, up, true, dem)
    off = np.linalg.norm((g_naive - g_true)[:, :2], axis=1).reshape(V.shape)
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_axes([0.06, 0.13, 0.72, 0.75])
    im = ax.imshow(off, extent=(0, RAW_W, RAW_H, 0), cmap="YlOrRd", vmin=0, vmax=max(2.5, float(np.ceil(off.max() * 2) / 2)))
    cs = ax.contour(U, V, off, levels=[0.5, 1, 1.5, 2, 2.5, 3], colors="k", linewidths=0.6)
    ax.clabel(cs, fmt="%g m", fontsize=7)
    ax.set_xticks([0, 320, 640])
    ax.set_yticks([0, 256, 512])
    ax.tick_params(labelsize=7)
    ax.set_title(f"how far a ray through the raw pixel lands from the true point ({agl:.0f} m above ground)", fontsize=8.5)
    cax = fig.add_axes([0.81, 0.13, 0.03, 0.75])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ground offset [m]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return cv2.cvtColor(np.ascontiguousarray(rgba), cv2.COLOR_RGB2BGR), off


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", default="146")
    ap.add_argument("--data", type=Path, default=Path("/home/user/bambi_downloads"), help="base-version files of the flight")
    ap.add_argument("--raw", type=Path, default=Path("/home/user/bambi_raw/146"), help="raw-version folder: SRT, air_data.csv, T_calib.json")
    ap.add_argument("--raw-video", type=Path, default=None, help="raw thermal recording (default: first *_T_*.MP4 under --raw)")
    ap.add_argument("--gt", type=Path, default=None, help="MOT file to take the boxes from (default <data>/interp/<flight>_gt.txt)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--window", type=float, default=32.0, help="map window size in metres")
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold the last frame with the summary")
    ap.add_argument("-o", "--output", type=Path, default=Path("undistortion_effect.mp4"))
    ap.add_argument("--preview", type=int, default=None, help="write only this video frame as PNG")
    args = ap.parse_args()
    fl = args.flight
    gt_path = args.gt or args.data / "interp" / f"{fl}_gt.txt"
    raw_video = args.raw_video or sorted(p for p in args.raw.rglob("*.MP4") if re.search(r"_\d{4}_T(\.|_)", p.name))[0]

    # --- terrain, poses (with correction), calibration ----------------------------
    dem = fa.DEM.from_geotiff(args.data / f"{fl}_matched_dem.tif")
    entries = fa.load_poses(args.data / f"{fl}_matched_poses.json", dem, "nadir", 50.0)
    t_corr, r_corr = fa.load_corrections(args.data / f"{fl}_correction.json", len(entries))
    for e, ct, cr in zip(entries, t_corr, r_corr):
        e["correction"] = (ct, cr)
    poses = [fa.pose_from_entry(e) for e in entries]
    calib = json.load(open(args.raw / "T_calib.json"))
    mask = cv2.imread(str(args.data / f"{fl}_mask_t.png"), 0) > 127
    new_k, fovy = fit_new_camera_matrix(mask, np.asarray(calib["mtx"], float), np.asarray(calib["dist"], float).reshape(-1), (RAW_W, RAW_H))
    lens = Lens(calib, new_k)

    # --- processed frame i -> raw recording frame k (via the SRT timestamps) -------------
    srt_files = sorted(p for p in args.raw.iterdir() if p.suffix.upper() == ".SRT" and re.search(r"_\d{4}_T(\.|_)", p.name))
    parts = [se.parse_srt(p) for p in srt_files]
    srt = {k: np.concatenate([s[k] for s in parts]) for k in parts[0]}
    ad = se.parse_airdata(args.raw / "air_data.csv")
    a, b = se.video_segment(ad, srt["t"][0]); seg = slice(a, b + 1); t0 = ad["t"][a]
    ad_s = se.seconds(ad["t"][seg], t0); srt_s = se.seconds(srt["t"], t0)
    offset = se.fit_offset(srt_s, srt["lat"], srt["lon"], ad_s, ad["lat"][seg], ad["lon"][seg])
    frame_s = srt_s + offset
    pub_s = se.seconds([dt.datetime.fromisoformat(e["raw"]["timestamp"]).replace(tzinfo=None) for e in entries], t0)
    k_of = np.searchsorted(frame_s, pub_s).clip(1, len(frame_s) - 1)
    k_of = np.where(np.abs(frame_s[k_of - 1] - pub_s) < np.abs(frame_s[k_of] - pub_s), k_of - 1, k_of)
    n_first = len(parts[0]["t"])        # frames of the first recording = the raw video used here

    # --- boxes ------------------------------------------------------------------------
    boxes = {}
    for line in open(gt_path):
        f = line.strip().split(",")
        if len(f) < 6:
            continue
        boxes.setdefault(int(f[0]), []).append((int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])))
    frames = sorted(fr for fr in boxes if k_of[fr] < n_first)
    if args.start is not None:
        frames = [f for f in frames if f >= args.start]
    if args.end is not None:
        frames = [f for f in frames if f <= args.end]
    frames = frames[::args.step]
    print(f"{len(frames)} frames with boxes ({frames[0]}..{frames[-1]}), raw frames {k_of[frames[0]]}..{k_of[frames[-1]]}")

    # --- geo-reference every box both ways -----------------------------------------------
    ground = {m: {} for m in COL}       # method -> frame -> list of (tid, x, y)
    raw_boxes = {}                      # frame -> list of (tid, quad in raw pixels)
    nadir = {}
    radius = []                         # per box: distance from the frame centre (fraction of half-size), offset
    for fr in frames:
        p = poses[fr]
        fwd, right, up = fa.camera_basis(p.tilt, 0.0, p.heading)
        cam = p.position
        bx = boxes[fr]
        centres = np.array([(l + w / 2, t + h / 2) for _, l, t, w, h in bx])
        g_und = cast(cam, fwd, right, up, lens.normalized_undist(centres), dem)
        g_raw = cast(cam, fwd, right, up, lens.normalized_raw_naive(lens.to_raw(centres)), dem)
        ground["undist"][fr] = [(tid, g[0], g[1]) for (tid, *_), g in zip(bx, g_und)]
        ground["raw"][fr] = [(tid, g[0], g[1]) for (tid, *_), g in zip(bx, g_raw)]
        quads = []
        for tid, l, t, w, h in bx:
            corners = np.array([(l, t), (l + w, t), (l + w, t + h), (l, t + h)])
            quads.append((tid, lens.to_raw(corners)))
        raw_boxes[fr] = quads
        nadir[fr] = cam[:2]
        for (u, v), gu, gr in zip(centres, g_und, g_raw):
            radius.append((math.hypot(u - W / 2, v - H / 2) / (W / 2), math.hypot(*(gu - gr)[:2])))
    radius = np.array(radius)
    off = []                             # per frame: mean and max offset
    for fr in frames:
        d = [math.hypot(x - xr, y - yr) for (_, x, y), (_, xr, yr) in zip(ground["undist"][fr], ground["raw"][fr])]
        off.append((np.mean(d), np.max(d)))
    off = np.array(off)
    print(f"offset per box: mean {radius[:, 1].mean():.2f} m, p95 {np.percentile(radius[:, 1], 95):.2f} m, max {radius[:, 1].max():.2f} m")
    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 2.0)):
        sel = (radius[:, 0] >= lo) & (radius[:, 0] < hi)
        if sel.any():
            print(f"  boxes {lo:.0%}-{hi:.0%} of the half-frame from the centre: mean offset {radius[sel, 1].mean():.2f} m (n={sel.sum()})")
    bins = [(0, 0.5, "inner half"), (0.5, 0.8, "50-80 %"), (0.8, 1.0, "80-100 %"), (1.0, 2.0, "corners")]
    bin_means = []
    for lo, hi, label in bins:
        sel = (radius[:, 0] >= lo) & (radius[:, 0] < hi)
        bin_means.append((label, radius[sel, 1].mean() if sel.any() else np.nan))

    # --- map background -------------------------------------------------------------------
    allp = np.array([(x, y) for fr in frames for _, x, y in ground["undist"][fr]])
    cx, cy = allp.mean(axis=0)
    half = args.window / 2
    map_px = 600
    scale = map_px / args.window
    gx = np.linspace(cx - half, cx + half, map_px); gy = np.linspace(cy + half, cy - half, map_px)
    GX, GY = np.meshgrid(gx, gy)
    shade = dem.sample(GX, GY, dem.hillshade)
    z = dem.sample(GX, GY)
    import matplotlib
    cmap = matplotlib.colormaps["gist_earth"]
    zn = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    base = cmap(0.25 + 0.6 * zn)[..., :3] * (0.55 + 0.45 * shade)[..., None]
    map_bg = (np.clip(base, 0, 1)[..., ::-1] * 255).astype(np.uint8)

    def to_px(x, y):
        return int(round((x - (cx - half)) * scale)), int(round(((cy + half) - y) * scale))

    route = np.array([p.position[:2] for p in poses])
    trails = np.zeros_like(map_bg)

    # --- error field for a representative pose (middle frame) ----------------------------
    mid = poses[frames[len(frames) // 2]]
    fwd, right, up = fa.camera_basis(mid.tilt, 0.0, mid.heading)
    agl = mid.position[2] - float(dem.sample(np.array([mid.position[0]]), np.array([mid.position[1]]))[0])
    field_img, field = error_field_panel(lens, mid.position, fwd, right, up, dem, (640, 330), agl)
    print(f"ground offset over the raw frame at {agl:.0f} m AGL: centre {field[field.shape[0] // 2, field.shape[1] // 2]:.2f} m, "
          f"edge (mid-right) {field[field.shape[0] // 2, -1]:.2f} m, corner {field[0, 0]:.2f} m")

    # --- video -------------------------------------------------------------------------------
    raw_cap = cv2.VideoCapture(str(raw_video))
    proc_cap = cv2.VideoCapture(str(args.data / f"{fl}_matched_processed.mp4"))
    out_w, out_h = 1920, 1080
    writer = None if args.preview is not None else cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    hold = int(args.hold * args.fps)
    seq = list(enumerate(frames)) + [(len(frames) - 1, frames[-1])] * hold
    x_raw, x_und, x_map, y_top = 16, 672, 1288, 112
    for n, (j, fr) in enumerate(seq):
        if args.preview is not None and n != args.preview:
            if n < len(frames):     # trails still have to accumulate
                for m in COL:
                    for _, x, y in ground[m][fr]:
                        cv2.circle(trails, to_px(x, y), 2, COL[m], -1, cv2.LINE_AA)
            continue
        raw_cap.set(cv2.CAP_PROP_POS_FRAMES, int(k_of[fr]))
        ok_r, raw_img = raw_cap.read()
        proc_cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok_p, proc_img = proc_cap.read()
        if not (ok_r and ok_p):
            print(f"could not read frame {fr}")
            break
        frame = np.full((out_h, out_w, 3), 255, np.uint8)

        # raw panel: the recording as it is, boxes mapped back through the lens
        raw_panel = raw_img.copy()
        for tid, quad in raw_boxes[fr]:
            cv2.polylines(raw_panel, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)], True, COL["raw"], 1, cv2.LINE_AA)
        raw_panel = cv2.copyMakeBorder(raw_panel, 44, 44, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        frame[y_top:y_top + 600, x_raw:x_raw + 640] = raw_panel
        cv2.rectangle(frame, (x_raw, y_top), (x_raw + 640, y_top + 600), (200, 200, 200), 1)
        text(frame, "raw thermal frame, 640 x 512 as recorded (barrel distortion)", (x_raw, y_top - 12), 0.55, INK)
        text(frame, f"raw frame {int(k_of[fr])}", (x_raw + 8, y_top + 22), 0.45, (220, 220, 220))

        # undistorted panel: the dataset frame with its boxes
        und = proc_img[:, :W].copy()
        for tid, l, t, w, h in boxes[fr]:
            cv2.rectangle(und, (int(l), int(t)), (int(l + w), int(t + h)), COL["undist"], 2)
        und = cv2.resize(und, (600, 600), interpolation=cv2.INTER_AREA)
        frame[y_top:y_top + 600, x_und:x_und + 600] = und
        cv2.rectangle(frame, (x_und, y_top), (x_und + 600, y_top + 600), (200, 200, 200), 1)
        text(frame, f"undistorted frame, 1024 x 1024 (dataset), fovy {fovy:.1f} deg", (x_und, y_top - 12), 0.55, INK)
        text(frame, f"frame {fr}", (x_und + 8, y_top + 22), 0.45, (220, 220, 220))

        # map
        if n < len(frames):
            for m in COL:
                for _, x, y in ground[m][fr]:
                    cv2.circle(trails, to_px(x, y), 2, COL[m], -1, cv2.LINE_AA)
        mp = map_bg.copy()
        pts = np.array([to_px(x, y) for x, y in route], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(mp, [pts], False, (90, 90, 90), 1, cv2.LINE_AA)
        tm = trails.any(axis=2)
        mp[tm] = trails[tm]
        nx, ny = to_px(*nadir[fr])
        cv2.drawMarker(mp, (nx, ny), INK, cv2.MARKER_TRIANGLE_UP, 16, 2, cv2.LINE_AA)
        for (_, x, y), (_, xr, yr) in zip(ground["undist"][fr], ground["raw"][fr]):
            cv2.line(mp, to_px(xr, yr), to_px(x, y), COL["raw"], 1, cv2.LINE_AA)
        for m in ("raw", "undist"):
            for _, x, y in ground[m][fr]:
                cv2.circle(mp, to_px(x, y), 7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(mp, to_px(x, y), 7, COL[m], 2, cv2.LINE_AA)
        x0, y0 = 20, map_px - 20
        cv2.line(mp, (x0, y0), (x0 + int(10 * scale), y0), INK, 3)
        text(mp, "10 m", (x0, y0 - 8), 0.5, INK)
        text(mp, "triangle: camera nadir    grey: flight route", (12, 22), 0.45, (60, 60, 60))
        frame[y_top:y_top + map_px, x_map:x_map + map_px] = mp
        cv2.rectangle(frame, (x_map, y_top), (x_map + map_px, y_top + map_px), (200, 200, 200), 1)
        text(frame, "ground position of each animal, top-down (1 m terrain model)", (x_map, y_top - 12), 0.55, INK)

        # title
        text(frame, "Does lens undistortion matter for ray casting?", (16, 42), 1.0, INK, 2)
        text(frame, f"BAMBI flight {fl}, {len(boxes[fr])} wild boar, same pose and terrain for both rays -- only the pixel-to-ray mapping differs",
             (16, 74), 0.55, GREY)

        # bottom band: error field, legend and numbers
        yb = y_top + 600 + 20
        frame[yb:yb + field_img.shape[0], x_raw:x_raw + field_img.shape[1]] = field_img
        xs = x_und
        yl = yb + 26
        for m in ("undist", "raw"):
            cv2.circle(frame, (xs + 8, yl - 5), 7, COL[m], -1, cv2.LINE_AA)
            text(frame, NAME[m], (xs + 24, yl), 0.55, INK)
            yl += 30
        yl += 6
        text(frame, f"this frame: offset {off[j, 0]:.2f} m mean, {off[j, 1]:.2f} m max over the {len(boxes[fr])} animals", (xs, yl), 0.55, INK); yl += 28
        text(frame, f"so far:     offset {off[: j + 1, 0].mean():.2f} m mean, {off[: j + 1, 1].max():.2f} m max", (xs, yl), 0.55, INK); yl += 34
        for line in (f"The M3T thermal lens has barrel distortion (k1 = {lens.dist[0]:.2f}).",
                     "A raw pixel near the edge sits closer to the image centre",
                     "than its true viewing direction, so a ray cast straight",
                     "through it lands short of the animal: by nothing at the",
                     "centre, by metres at the corners.",
                     "Undistortion: cv2.initUndistortRectifyMap with the",
                     "calibration shipped in the raw dataset version."):
            text(frame, line, (xs, yl), 0.48, GREY); yl += 22
        # offset by image position (running over all frames, shown always)
        yl = yb + 26
        text(frame, "offset by distance of the box from the frame centre, all frames:", (x_map, yl), 0.5, INK)
        bar_w, bar_top, bar_bot = 110, yl + 36, yl + 166
        bmax = max(v for _, v in bin_means if not np.isnan(v))
        for i, (label, v) in enumerate(bin_means):
            bx0 = x_map + 10 + i * (bar_w + 40)
            hgt = 0 if np.isnan(v) else int((bar_bot - bar_top) * v / max(bmax, 1e-6))
            cv2.rectangle(frame, (bx0, bar_bot - hgt), (bx0 + bar_w, bar_bot), COL["raw"], -1)
            text(frame, "n/a" if np.isnan(v) else f"{v:.2f} m", (bx0 + 22, bar_bot - hgt - 8), 0.5, INK)
            text(frame, label, (bx0 + 8, bar_bot + 22), 0.45, GREY)
        if n >= len(frames):
            yl = yb + 232
            cv2.rectangle(frame, (x_map, yl - 6), (x_map + map_px, yl + 96), LIGHT, -1)
            text(frame, f"all {len(radius)} boxes in {len(frames)} frames:", (x_map + 10, yl + 18), 0.55, INK)
            text(frame, f"mean offset {radius[:, 1].mean():.2f} m, p95 {np.percentile(radius[:, 1], 95):.2f} m, max {radius[:, 1].max():.2f} m",
                 (x_map + 10, yl + 46), 0.55, INK)
            text(frame, f"the boar are {(radius[:, 1] > 1).mean():.0%} of the time more than 1 m off without undistortion", (x_map + 10, yl + 74), 0.5, GREY)
        if args.preview is not None:
            cv2.imwrite(str(args.output.with_suffix(".png")), frame)
            print("wrote preview")
            return
        writer.write(frame)
        if n % 60 == 0:
            print(f"\r  {n + 1}/{len(seq)}", end="", flush=True)
    writer.release()
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
