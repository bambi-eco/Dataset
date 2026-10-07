#!/usr/bin/env python3
"""
Lens distortion across the BAMBI dataset: what skipping the undistortion would
do to the ground position of every annotated animal.

For every flight with a calibration (raw dataset version) the script
  1. groups the calibrations (thermal and RGB) and the undistortion masks,
  2. recovers the new camera matrix each mask was made with (focal length and
     principal point of the released 1024 x 1024 frame),
  3. maps every annotated thermal box centre back into the raw frame and
     compares the ray through the raw pixel (distortion ignored) with the ray
     through the undistorted pixel, on flat ground at the frame's height above
     ground: offset = h * |n_naive - n_true| for a nadir camera,
  4. measures how much that offset varies along a track (the apparent motion a
     tracker or a light-field integral would see).

Outputs CSV tables and a JSON summary into <out>/tables.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C   # noqa: E402

F_NOMINAL = (C.H / 2) / math.tan(math.radians(25))     # the 50 deg fovy the projection assumes
RAW_SIZES = [(640, 512), (1280, 1024), (3840, 2160), (4000, 3000), (1920, 1080)]


def raw_size_of(calib: dict) -> tuple[int, int]:
    cx, cy = calib["mtx"][0, 2], calib["mtx"][1, 2]
    return min(RAW_SIZES, key=lambda s: abs(s[0] / 2 - cx) + abs(s[1] / 2 - cy))


def calib_key(calib: dict) -> str:
    v = np.concatenate([calib["mtx"].ravel(), calib["dist"]])
    return hashlib.md5(np.round(v, 4).tobytes()).hexdigest()[:8]


def mask_key(mask: np.ndarray) -> str:
    return hashlib.md5(np.packbits(mask).tobytes()).hexdigest()[:8]


# ----------------------------------------------------------------------------- lens model

def fit_new_camera_matrix(mask: np.ndarray, calib: dict, raw_size) -> tuple[np.ndarray, float, float]:
    """Focal length / principal point whose undistortion footprint matches the mask (coordinate descent)."""
    h, w = mask.shape
    ones = np.full((raw_size[1], raw_size[0]), 255, np.uint8)
    mtx, dist = calib["mtx"], calib["dist"]

    def score(f, cx, cy):
        k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        mx, my = cv2.initUndistortRectifyMap(mtx, dist, None, k, (w, h), cv2.CV_32FC1)
        return float(((cv2.remap(ones, mx, my, cv2.INTER_LINEAR) > 127) == mask).mean())

    # start: the raw focal length scaled to the output size, centred
    p = [mtx[1, 1] * h / raw_size[1], (w - 1) / 2, (h - 1) / 2]
    best = score(*p)
    for step in (160, 80, 40, 20, 10, 5, 2, 1, 0.5):
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
    return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]]), best, 2 * math.degrees(math.atan(h / 2 / p[0]))


def normalized_true(uv: np.ndarray, new_k: np.ndarray) -> np.ndarray:
    return np.stack([(uv[:, 0] - new_k[0, 2]) / new_k[0, 0], (uv[:, 1] - new_k[1, 2]) / new_k[1, 1]], axis=1)


def to_raw(n: np.ndarray, calib: dict) -> np.ndarray:
    pts = np.concatenate([n, np.ones((len(n), 1))], axis=1).reshape(-1, 1, 3)
    raw, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), calib["mtx"], calib["dist"])
    return raw.reshape(-1, 2)


def normalized_naive(uv_raw: np.ndarray, calib: dict) -> np.ndarray:
    k = calib["mtx"]
    return np.stack([(uv_raw[:, 0] - k[0, 2]) / k[0, 0], (uv_raw[:, 1] - k[1, 2]) / k[1, 1]], axis=1)


def offset_per_metre(n_true: np.ndarray, calib: dict) -> np.ndarray:
    """Ground offset vector per metre of height (image axes) when distortion is ignored, for undistorted normalized coords."""
    return normalized_naive(to_raw(n_true, calib), calib) - n_true


def radial_profile(calib: dict, raw_size, n=60):
    """Displacement between the raw pixel and its ideal pinhole position along the diagonal, up to the corner."""
    w, h = raw_size
    k = calib["mtx"]
    corner = np.array([(w - k[0, 2]) / k[0, 0], (h - k[1, 2]) / k[1, 1]])     # bottom-right corner, normalized (ideal)
    r = np.linspace(0, 1, n)
    n_true = corner[None, :] * r[:, None]
    raw = to_raw(n_true, calib)
    ideal = np.stack([n_true[:, 0] * k[0, 0] + k[0, 2], n_true[:, 1] * k[1, 1] + k[1, 2]], axis=1)
    centre = k[:2, 2][None, :]
    r_ideal = np.linalg.norm(ideal - centre, axis=1)
    r_raw = np.linalg.norm(raw - centre, axis=1)
    disp_px = r_raw - r_ideal                          # signed: negative = barrel (raw pixel pulled inward)
    frac = np.where(r > 0, disp_px / np.maximum(r_ideal, 1e-9), 0)
    return r, disp_px, frac


def field_per_metre(calib: dict, new_k: np.ndarray, step=16):
    """|offset| per metre AGL over the undistorted 1024 frame."""
    us = np.arange(step / 2, C.W, step)
    vs = np.arange(step / 2, C.H, step)
    U, V = np.meshgrid(us, vs)
    n = normalized_true(np.stack([U.ravel(), V.ravel()], axis=1), new_k)
    d = np.linalg.norm(offset_per_metre(n, calib), axis=1)
    return d.reshape(V.shape)


# ----------------------------------------------------------------------------- per flight

def frame_agl_airdata(fl: C.Flight, poses: dict):
    """Height above ground per published frame from the AirData log (terrain column, fallback: above take-off)."""
    ad = C.load_airdata(fl)
    if ad is None:
        return np.full(poses["n"], np.nan), "none"
    t0 = ad["t"][0]
    ad_s = C.seconds(ad["t"], t0)
    pose_s = C.seconds(poses["t"], t0)
    src = "terrain"
    col = ad["agl"]
    if not np.isfinite(col).any() or np.nanmax(col) <= 0:
        col, src = ad["above_takeoff"], "takeoff"
    ok = np.isfinite(col) & (col > 0)
    if ok.sum() < 2:
        return np.full(poses["n"], np.nan), "none"
    agl = np.interp(pose_s, ad_s[ok], col[ok])
    # outside the log: unknown
    agl[(pose_s < ad_s[ok][0] - 5) | (pose_s > ad_s[ok][-1] + 5)] = np.nan
    return agl, src


def frame_agl(fl: C.Flight, poses: dict, data: Path):
    """Height above ground per frame: corrected pose over the BEV terrain model (terrain.py), else the flight log."""
    p = data / "agl" / f"{fl.fid}_agl.csv"
    if p.exists():
        agl = np.full(poses["n"], np.nan)
        for row in __import__("csv").DictReader(open(p)):
            if row["agl"] != "":
                agl[int(row["frame"])] = float(row["agl"])
        if np.isfinite(agl).any():
            return agl, "dem"
    return frame_agl_airdata(fl, poses)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    ap.add_argument("--flights", nargs="*", default=None)
    args = ap.parse_args()
    tdir = args.out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    calibs, masks, newks = {}, {}, {}          # key -> record
    per_flight, boxes, tracks = [], [], []
    fls = C.flights(args.data, args.flights)
    print(f"{len(fls)} flights with base files")
    # calibrations are per drone; a flight without public raw logs borrows the calibration that the flights
    # of the same drone instance with the same undistortion mask carry (unique, otherwise none)
    by_pair = {}
    for fl in fls:
        inst = C.load_metadata(fl).get("flight_info", {}).get("drone_instance_id", "")
        m = C.load_mask(fl, "T")
        for cam in ("T", "W"):
            cal = C.load_calib(fl, cam)
            if cal is not None and m is not None:
                by_pair.setdefault((inst, mask_key(m), cam), {})[calib_key(cal)] = (cal, fl.fid)
    inferred = {k: next(iter(v.values()))[0] for k, v in by_pair.items() if len(v) == 1}

    def calib_of(fl, cam, inst, m):
        cal = C.load_calib(fl, cam)
        if cal is not None:
            return cal, "own"
        if m is not None and (inst, mask_key(m), cam) in inferred:
            return inferred[(inst, mask_key(m), cam)], "inferred"
        return None, ""

    for fl in fls:
        meta = C.load_metadata(fl).get("flight_info", {})
        rec = dict(flight=fl.fid, drone=meta.get("drone_name", ""), drone_instance=meta.get("drone_instance_id", ""),
                   calib_t="", calib_w="", mask_t="", mask_w="", newk_t="", fovy_t=np.nan, fovy_w=np.nan,
                   agl_source="", n_boxes=0, n_boxes_used=0, median_agl=np.nan, median_agl_airdata=np.nan, median_tilt=np.nan, fovy_stored=np.nan, pose_format="", calib_source="")
        poses = C.load_poses(fl)
        rec["median_tilt"] = float(np.nanmedian(poses["tilt"]))
        rec["fovy_stored"] = float(np.nanmedian(poses["fovy"])) if np.isfinite(poses["fovy"]).any() else np.nan
        rec["pose_format"] = "location" if np.isfinite(poses["fovy"]).any() else "latlng"
        gt = C.load_gt(fl)
        rec["n_boxes"] = int(len(gt))
        m_t = C.load_mask(fl, "T")
        for cam in ("T", "W"):
            m = C.load_mask(fl, cam)
            cal, src = calib_of(fl, cam, rec["drone_instance"], m_t)
            if cam == "T":
                rec["calib_source"] = src
            if cal is not None:
                k = calib_key(cal)
                rec[f"calib_{cam.lower()}"] = k
                if k not in calibs:
                    calibs[k] = dict(key=k, cam=cam, calib=cal, raw_size=raw_size_of(cal), flights=[], drones=set())
                calibs[k]["flights"].append(fl.fid)
                calibs[k]["drones"].add(f"{rec['drone']}#{rec['drone_instance']}")
            if m is not None:
                mk = mask_key(m)
                rec[f"mask_{cam.lower()}"] = mk
                if mk not in masks:
                    masks[mk] = dict(key=mk, cam=cam, mask=m, valid=float(m.mean()), flights=[])
                masks[mk]["flights"].append(fl.fid)
                if cal is not None:
                    nk = (mk, calib_key(cal))
                    if nk not in newks:
                        new_k, agree, fovy = fit_new_camera_matrix(m, cal, raw_size_of(cal))
                        newks[nk] = dict(mask=mk, calib=calib_key(cal), cam=cam, f=new_k[0, 0], cx=new_k[0, 2], cy=new_k[1, 2],
                                         fovy=fovy, agreement=agree, new_k=new_k)
                        print(f"  undistortion {cam} mask {mk} calib {calib_key(cal)}: f {new_k[0, 0]:.1f} px, fovy {fovy:.2f} deg, agreement {agree:.4%}")
                    rec[f"fovy_{cam.lower()}"] = newks[nk]["fovy"]
                    if cam == "T":
                        rec["newk_t"] = f"{mk}/{calib_key(cal)}"

        # --- per box offsets (thermal boxes, thermal calibration) --------------------------------
        m = m_t
        cal, _ = calib_of(fl, "T", rec["drone_instance"], m_t)
        if cal is not None and m is not None and len(gt):
            nk = newks[(mask_key(m), calib_key(cal))]["new_k"]
            agl, src = frame_agl(fl, poses, args.data)
            rec["agl_source"] = src
            agl_ad, _ = frame_agl_airdata(fl, poses)
            rec["median_agl_airdata"] = float(np.nanmedian(agl_ad)) if np.isfinite(agl_ad).any() else np.nan
            fr = gt["frame"].astype(int)
            valid = (fr >= 0) & (fr < poses["n"])
            g = gt[valid]
            fr = fr[valid]
            uv = np.stack([g["l"] + g["w"] / 2, g["t"] + g["h"] / 2], axis=1).astype(np.float64)
            n_true = normalized_true(uv, nk)
            dn = offset_per_metre(n_true, cal)                     # image axes, per metre
            h = agl[fr]
            head = np.radians(poses["heading"][fr])
            # world vector: image x along 'right' = (cos h, -sin h), image y (down) opposite to 'up' = (sin h, cos h)
            vx = h * (dn[:, 0] * np.cos(head) - dn[:, 1] * np.sin(head))
            vy = h * (-dn[:, 0] * np.sin(head) - dn[:, 1] * np.cos(head))
            off = np.hypot(vx, vy)
            off_fov = h * np.linalg.norm(n_true, axis=1) * abs(F_NOMINAL / nk[0, 0] - 1)
            ui = np.clip(uv[:, 0].astype(int), 0, C.W - 1)
            vi = np.clip(uv[:, 1].astype(int), 0, C.H - 1)
            in_mask = m[vi, ui]
            r_frac = np.hypot(uv[:, 0] - C.W / 2, uv[:, 1] - C.H / 2) / (C.W / 2)
            rec["n_boxes_used"] = int(np.isfinite(off).sum())
            rec["median_agl"] = float(np.nanmedian(h))
            for i in range(len(g)):
                boxes.append((fl.fid, int(fr[i]), int(g["tid"][i]), g["species"][i], float(uv[i, 0]), float(uv[i, 1]), float(r_frac[i]),
                              bool(in_mask[i]), float(h[i]), float(np.hypot(*dn[i])), float(off[i]), float(off_fov[i]), float(vx[i]), float(vy[i])))
            # --- per track: spread of the offset vector along the track ------------------------
            for tid in np.unique(g["tid"]):
                sel = (g["tid"] == tid) & np.isfinite(off)
                if sel.sum() < 2:
                    continue
                v = np.stack([vx[sel], vy[sel]], axis=1)
                c = v.mean(axis=0)
                rad = np.sqrt(np.mean(np.sum((v - c) ** 2, axis=1)))
                extent = float(np.max(np.linalg.norm(v - c, axis=1)) * 2)
                tracks.append((fl.fid, int(tid), g["species"][sel][0], int(sel.sum()), float(np.nanmedian(h[sel])), float(np.mean(off[sel])),
                               float(rad), extent, float(r_frac[sel].min()), float(r_frac[sel].max())))
        per_flight.append(rec)
        print(f"flight {fl.fid}: {rec['n_boxes']} boxes, {rec['n_boxes_used']} with height ({rec['agl_source']}), calib T {rec['calib_t'] or '-'}")

    # --- tables -------------------------------------------------------------------------------------
    import csv
    with open(tdir / "calibrations.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["key", "camera", "raw_w", "raw_h", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
                     "edge_disp_px", "edge_disp_pct", "corner_disp_px", "corner_disp_pct", "n_flights", "drones"])
        for k, c in sorted(calibs.items(), key=lambda kv: (kv[1]["cam"], -len(kv[1]["flights"]))):
            cal, (w, h) = c["calib"], c["raw_size"]
            r, dpx, frac = radial_profile(cal, (w, h))
            # edge: middle of the long side
            e_n = np.array([[(w - cal["mtx"][0, 2]) / cal["mtx"][0, 0], 0.0]])
            e_raw = to_raw(e_n, cal)
            e_ideal = np.array([[w, cal["mtx"][1, 2]]])
            e_px = float(np.linalg.norm(e_raw - cal["mtx"][:2, 2]) - np.linalg.norm(e_ideal - cal["mtx"][:2, 2]))
            e_pct = e_px / (w - cal["mtx"][0, 2]) * 100
            d = cal["dist"]
            wr.writerow([k, "thermal" if c["cam"] == "T" else "rgb", w, h, *np.round(cal["mtx"][[0, 1, 0, 1], [0, 1, 2, 2]], 2),
                         *np.round(d[:5], 5), round(e_px, 1), round(e_pct, 2), round(float(dpx[-1]), 1), round(float(frac[-1]) * 100, 2),
                         len(c["flights"]), "; ".join(sorted(c["drones"]))])
    with open(tdir / "undistortion.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mask", "calib", "camera", "f_px", "cx", "cy", "fovy_deg", "mask_agreement", "valid_share", "n_flights",
                     "f_nominal_px", "scale_error_pct"])
        for (mk, ck), r in newks.items():
            wr.writerow([mk, ck, "thermal" if r["cam"] == "T" else "rgb", round(r["f"], 1), round(r["cx"], 1), round(r["cy"], 1),
                         round(r["fovy"], 2), round(r["agreement"], 5), round(masks[mk]["valid"], 4), len(masks[mk]["flights"]),
                         round(F_NOMINAL, 1), round((F_NOMINAL / r["f"] - 1) * 100, 2)])
    with open(tdir / "distortion_flights.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_flight[0].keys()))
        wr.writeheader()
        wr.writerows(per_flight)
    with open(tdir / "distortion_boxes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["flight", "frame", "tid", "species", "u", "v", "r_frac", "in_mask", "agl", "offset_per_m", "offset", "offset_fov", "vx", "vy"])
        wr.writerows(boxes)
    with open(tdir / "distortion_tracks.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["flight", "tid", "species", "n_boxes", "median_agl", "mean_offset", "rms_radius", "extent", "r_min", "r_max"])
        wr.writerows(tracks)

    # --- field per metre and radial profiles for the figures -------------------------------------
    fields = {}
    for (mk, ck), r in newks.items():
        fields[f"{r['cam']}_{mk}_{ck}"] = field_per_metre(calibs[ck]["calib"], r["new_k"]).tolist()
    profiles = {}
    for k, c in calibs.items():
        r, dpx, frac = radial_profile(c["calib"], c["raw_size"])
        profiles[k] = dict(cam=c["cam"], r=r.tolist(), disp_px=dpx.tolist(), frac=frac.tolist(), n_flights=len(c["flights"]))
    json.dump(dict(fields=fields, profiles=profiles, f_nominal=F_NOMINAL), open(tdir / "distortion_model.json", "w"))

    # --- summary --------------------------------------------------------------------------------------
    off = np.array([b[10] for b in boxes]); agl = np.array([b[8] for b in boxes]); rf = np.array([b[6] for b in boxes])
    ok = np.isfinite(off)
    summ = dict(n_flights=len(fls), n_flights_calib=sum(1 for r in per_flight if r["calib_t"]),
                n_flights_calib_own=sum(1 for r in per_flight if r["calib_source"] == "own"),
                n_flights_calib_inferred=sum(1 for r in per_flight if r["calib_source"] == "inferred"), n_boxes=len(boxes), n_boxes_with_height=int(ok.sum()),
                n_calibrations_thermal=sum(1 for c in calibs.values() if c["cam"] == "T"),
                n_calibrations_rgb=sum(1 for c in calibs.values() if c["cam"] == "W"),
                offset_mean=float(np.nanmean(off)), offset_median=C.pct(off, 50), offset_p95=C.pct(off, 95), offset_max=float(np.nanmax(off)),
                share_over_match=float(np.mean(off[ok] > C.MATCH_RADIUS)), share_over_half_match=float(np.mean(off[ok] > C.MATCH_RADIUS / 2)),
                share_over_1m=float(np.mean(off[ok] > 1.0)),
                agl_median=C.pct(agl, 50), r_frac_median=C.pct(rf, 50), by_alt_bin=[], by_radius=[])
    bins = C.alt_bin(agl)
    for i, lab in enumerate(C.ALT_LABELS):
        s = ok & (bins == i)
        summ["by_alt_bin"].append(dict(bin=lab, n_boxes=int(s.sum()), n_flights=len({b[0] for b, m in zip(boxes, s) if m}),
                                       median=C.pct(off[s], 50), p95=C.pct(off[s], 95), share_over_match=float(np.mean(off[s] > C.MATCH_RADIUS)) if s.any() else np.nan))
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 2.0)):
        s = ok & (rf >= lo) & (rf < hi)
        summ["by_radius"].append(dict(lo=lo, hi=hi, n_boxes=int(s.sum()), median=C.pct(off[s], 50), p95=C.pct(off[s], 95),
                                      share_over_match=float(np.mean(off[s] > C.MATCH_RADIUS)) if s.any() else np.nan))
    if tracks:
        rad = np.array([t[6] for t in tracks]); ext = np.array([t[7] for t in tracks])
        summ["tracks"] = dict(n=len(tracks), rms_radius_median=C.pct(rad, 50), rms_radius_p95=C.pct(rad, 95), extent_median=C.pct(ext, 50), extent_p95=C.pct(ext, 95),
                              share_extent_over_match=float(np.mean(ext > C.MATCH_RADIUS)))
    json.dump(summ, open(tdir / "distortion_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    main()
