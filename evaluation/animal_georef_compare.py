#!/usr/bin/env python3
"""
Geo-reference the annotated animals of a flight with three camera-pose
sources and show the offsets between them as a video:

  * AirData only   - flight-log positions, frames timed from the log's isVideo
                     flag at a constant 30 fps (no SRT used);
  * SRT only       - the GPS written next to every frame in the DJI SRT file;
  * SRT + AirData  - the BAMBI extractor: AirData positions at SRT frame times,
                     one clock offset fitted on the two GPS traces (the poses
                     shipped with the dataset).

Every box centre is cast from the camera onto the terrain model, so a wrong
pose shows up as a wrong ground position of the animal.  Left: the thermal
frame with its boxes.  Right: a top-down map with the accumulating ground
positions per method, the camera nadir points and the live offsets.
"""
import argparse
import datetime as dt
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
COL = {"combined": (180, 119, 31), "airdata": (14, 127, 255), "srt": (44, 160, 44)}   # BGR
NAME = {"combined": "SRT + AirData (BAMBI)", "airdata": "AirData only", "srt": "SRT only"}


def ground_point(cam, fwd, right, up, fovy, u, v, dem):
    half = math.tan(math.radians(fovy) / 2)
    p0 = cam + fwd + ((u + 0.5) / W * 2 - 1) * half * right - ((v + 0.5) / H * 2 - 1) * half * up
    s = fa.march_rays(cam, p0[None, :], dem)[0]
    return cam + (p0 - cam) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="146")
    ap.add_argument("--data", type=Path, default=Path("/home/user/bambi_downloads"))
    ap.add_argument("--raw", type=Path, default=Path("/home/user/bambi_raw/146"))
    ap.add_argument("--gt", type=Path, default=None, help="interpolated MOT file (default <data>/interp/<flight>_gt.txt)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--window", type=float, default=40.0, help="map window size in metres")
    ap.add_argument("-o", "--output", type=Path, default=Path("animal_georef_compare.mp4"))
    ap.add_argument("--preview", type=int, default=None, help="write only this video frame as PNG")
    args = ap.parse_args()
    fl = args.flight
    gt_path = args.gt or args.data / "interp" / f"{fl}_gt.txt"

    # --- terrain, published poses, correction ------------------------------------
    dem = fa.DEM.from_geotiff(args.data / f"{fl}_matched_dem.tif")
    entries = fa.load_poses(args.data / f"{fl}_matched_poses.json", dem, "nadir", 50.0)
    t_corr, r_corr = fa.load_corrections(args.data / f"{fl}_correction.json", len(entries))
    for e, ct, cr in zip(entries, t_corr, r_corr):
        e["correction"] = (ct, cr)
    poses = [fa.pose_from_entry(e) for e in entries]
    pub_t = [dt.datetime.fromisoformat(e["raw"]["timestamp"]).replace(tzinfo=None) for e in entries]

    # --- raw logs: SRT frames and the AirData video run -----------------------------
    srt_files = sorted(p for p in args.raw.iterdir() if p.suffix.upper() == ".SRT" and re.search(r"_\d{4}_T(\.|_)", p.name))
    parts = [se.parse_srt(p) for p in srt_files]
    srt = {k: np.concatenate([s[k] for s in parts]) for k in parts[0]}
    ad = se.parse_airdata(args.raw / "air_data.csv")
    a, b = se.video_segment(ad, srt["t"][0]); seg = slice(a, b + 1); t0 = ad["t"][a]
    ad_s = se.seconds(ad["t"][seg], t0); srt_s = se.seconds(srt["t"], t0)
    offset = se.fit_offset(srt_s, srt["lat"], srt["lon"], ad_s, ad["lat"][seg], ad["lon"][seg])
    frame_s = srt_s + offset
    ox, oy, oz = dem.origin
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)
    ad_x, ad_y = tr.transform(ad["lon"][seg], ad["lat"][seg])
    ad_x, ad_y, ad_z = ad_x - ox, ad_y - oy, ad["alt"][seg] - oz
    ad_head = se.unwrap_deg(ad["heading"][seg])
    srt_x, srt_y = tr.transform(srt["lon"], srt["lat"]); srt_x, srt_y = srt_x - ox, srt_y - oy
    rows = [{(k or "").strip(): v for k, v in r.items()} for r in __import__("csv").DictReader(open(args.raw / "air_data.csv", encoding="utf-8"))]
    takeoff_alt = float(rows[a]["altitude_above_seaLevel(feet)"]) * 0.3048 - float(rows[a]["height_above_takeoff(feet)"]) * 0.3048
    srt_rel = np.array([float(x) for x in re.findall(r"rel_alt: ([\-\d.]+)", "".join(open(p, encoding="utf-8", errors="replace").read() for p in srt_files))])
    if len(srt_rel) != len(srt_s):
        srt_rel = np.interp(np.arange(len(srt_s)), np.linspace(0, len(srt_s) - 1, len(srt_rel)), srt_rel)

    # processed frame i -> SRT frame k (duplicates were dropped by the extractor)
    pub_s = se.seconds(pub_t, t0)
    k_of = np.searchsorted(frame_s, pub_s).clip(1, len(frame_s) - 1)
    k_of = np.where(np.abs(frame_s[k_of - 1] - pub_s) < np.abs(frame_s[k_of] - pub_s), k_of - 1, k_of)

    def pose_variant(i, method):
        p = poses[i]
        ct, cr = entries[i]["correction"]
        if method == "combined":
            return p.position, p.tilt, p.heading
        k = k_of[i]
        if method == "airdata":
            s = k / 30.0                                  # isVideo onset + constant frame rate
            pos = np.array([np.interp(s, ad_s, ad_x), np.interp(s, ad_s, ad_y), np.interp(s, ad_s, ad_z)]) + ct
            head = (np.interp(s, ad_s, ad_head) - math.degrees(cr[2])) % 360
            return pos, p.tilt, head
        pos = np.array([srt_x[k], srt_y[k], takeoff_alt + srt_rel[k] - oz]) + ct
        head = (srt["gb_yaw"][k] - math.degrees(cr[2])) % 360
        return pos, p.tilt, head

    # --- boxes ----------------------------------------------------------------------
    boxes = {}
    for line in open(gt_path):
        f = line.strip().split(",")
        if len(f) < 6:
            continue
        fr, tid = int(f[0]), int(f[1])
        boxes.setdefault(fr, []).append((tid, float(f[2]), float(f[3]), float(f[4]), float(f[5])))
    frames = sorted(boxes)
    if args.start is not None:
        frames = [f for f in frames if f >= args.start]
    if args.end is not None:
        frames = [f for f in frames if f <= args.end]
    frames = frames[::args.step]
    print(f"{len(frames)} frames with boxes ({frames[0]}..{frames[-1]}), clock offset {offset:+.3f} s")

    # --- geo-reference everything up front -------------------------------------------
    ground = {m: {} for m in COL}           # method -> frame -> list of (tid, x, y)
    nadir = {m: {} for m in COL}
    for fr in frames:
        for m in COL:
            cam, tilt, head = pose_variant(fr, m)
            fwd, right, up = fa.camera_basis(tilt, 0.0, head)
            pts = []
            for tid, l, t, w, h in boxes[fr]:
                g = ground_point(cam, fwd, right, up, 50.0, l + w / 2, t + h / 2, dem)
                pts.append((tid, g[0], g[1]))
            ground[m][fr] = pts
            nadir[m][fr] = cam[:2]

    # offsets to the combined result, per frame (mean over the animals)
    off = {m: [] for m in ("airdata", "srt")}
    for fr in frames:
        ref = {tid: (x, y) for tid, x, y in ground["combined"][fr]}
        for m in off:
            d = [math.hypot(x - ref[tid][0], y - ref[tid][1]) for tid, x, y in ground[m][fr] if tid in ref]
            off[m].append(np.mean(d) if d else np.nan)
    for m in off:
        print(f"{NAME[m]:<24s} offset to combined: mean {np.nanmean(off[m]):.2f} m, p95 {np.nanpercentile(off[m], 95):.2f} m")
    # spread of each track's cluster per method
    spread = {}
    for m in COL:
        per_track = {}
        for fr in frames:
            for tid, x, y in ground[m][fr]:
                per_track.setdefault(tid, []).append((x, y))
        r = [np.sqrt(np.mean(np.sum((np.array(v) - np.mean(v, axis=0)) ** 2, axis=1))) for v in per_track.values() if len(v) > 10]
        spread[m] = float(np.mean(r))
        print(f"{NAME[m]:<24s} mean cluster radius per animal: {spread[m]:.2f} m")

    # --- map background --------------------------------------------------------------
    allp = np.array([(x, y) for fr in frames for _, x, y in ground["combined"][fr]])
    cx, cy = allp.mean(axis=0)
    half = args.window / 2
    map_px = 660
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

    # drone route
    route = np.array([p.position[:2] for p in poses])
    trails = np.zeros_like(map_bg)

    # --- video ----------------------------------------------------------------------
    cap = cv2.VideoCapture(str(args.data / f"{fl}_matched_processed.mp4"))
    out_w, out_h = 1280, 720
    writer = None if args.preview is not None else cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    hold = int(3 * args.fps)
    seq = list(enumerate(frames)) + [(len(frames) - 1, frames[-1])] * hold
    for n, (j, fr) in enumerate(seq):
        if args.preview is not None and n != args.preview:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            break
        thermal = img[:, :W].copy()
        for tid, l, t, w, h in boxes[fr]:
            cv2.rectangle(thermal, (int(l), int(t)), (int(l + w), int(t + h)), (255, 255, 255), 2)
        thermal = cv2.resize(thermal, (560, 560), interpolation=cv2.INTER_AREA)

        # trails accumulate (only while the sequence advances)
        if n < len(frames):
            for m in COL:
                for _, x, y in ground[m][fr]:
                    cv2.circle(trails, to_px(x, y), 2, COL[m], -1, cv2.LINE_AA)
        mp = map_bg.copy()
        # route and nadir points
        pts = np.array([to_px(x, y) for x, y in route], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(mp, [pts], False, (90, 90, 90), 1, cv2.LINE_AA)
        mp = cv2.addWeighted(mp, 1.0, trails, 1.0, 0)
        mask = trails.any(axis=2)
        mp[mask] = trails[mask]
        for m in ("airdata", "srt", "combined"):
            nx, ny = to_px(*nadir[m][fr])
            cv2.drawMarker(mp, (nx, ny), COL[m], cv2.MARKER_TRIANGLE_UP, 16, 2, cv2.LINE_AA)
        # current animals: offset lines then markers
        ref = {tid: (x, y) for tid, x, y in ground["combined"][fr]}
        for m in ("airdata", "srt"):
            for tid, x, y in ground[m][fr]:
                if tid in ref:
                    cv2.line(mp, to_px(x, y), to_px(*ref[tid]), COL[m], 1, cv2.LINE_AA)
        for m in ("airdata", "srt", "combined"):
            for _, x, y in ground[m][fr]:
                cv2.circle(mp, to_px(x, y), 7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(mp, to_px(x, y), 7, COL[m], 2, cv2.LINE_AA)
        cv2.putText(mp, "ground positions of the boxed animals, three camera-pose sources", (12, 22), font, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(mp, "triangles: camera nadir point    grey: flight route", (12, 42), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        # scale bar 10 m
        x0, y0 = 20, map_px - 20
        cv2.line(mp, (x0, y0), (x0 + int(10 * scale), y0), (20, 20, 20), 3)
        cv2.putText(mp, "10 m", (x0, y0 - 8), font, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

        frame = np.full((out_h, out_w, 3), 250, np.uint8)
        frame[80:640, 20:580] = thermal
        frame[30:30 + map_px, 600:600 + map_px] = mp
        cv2.putText(frame, f"BAMBI flight {fl}  thermal frame {fr}   {len(boxes[fr])} wild boar", (20, 62), font, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(frame, "box centres cast onto the 1 m terrain model", (20, 664), font, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        # legend + live offsets
        y = 700
        xs = 20
        for m in ("combined", "airdata", "srt"):
            cv2.circle(frame, (xs + 8, y - 5), 6, COL[m], -1, cv2.LINE_AA)
            txt = NAME[m]
            if m != "combined":
                txt += f": offset {off[m][j]:.1f} m (mean {np.nanmean(off[m][: j + 1]):.1f} m)"
            cv2.putText(frame, txt, (xs + 22, y), font, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
            xs += 24 + int(len(txt) * 9.2)
        if n >= len(frames):   # final hold: cluster spread summary
            cv2.rectangle(frame, (600, 30), (600 + map_px, 30 + 92), (255, 255, 255), -1)
            cv2.putText(frame, "same animals, all frames: mean radius of each animal's cluster", (612, 56), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(frame, f"SRT + AirData {spread['combined']:.2f} m     AirData only {spread['airdata']:.2f} m     SRT only {spread['srt']:.2f} m",
                        (612, 84), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(frame, "AirData only: frames timed from the isVideo flag, here 0.49 s late",
                        (612, 110), font, 0.48, (90, 90, 90), 1, cv2.LINE_AA)
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
