#!/usr/bin/env python3
"""
Pose correction across the BAMBI dataset: what the per-flight camera
correction of Praschl et al. (IET Computer Vision 2026) does to the ground
position of every annotated animal, and how it relates to quantities that can
be measured independently in the raw logs.

The correction files (`<id>_correction.json`) carry two parameters, each as a
flight default and, optionally, per frame range (`fine_corrections`, first
matching range wins): an altitude offset (translation z, metres, added to the
camera height) and a heading offset (rotation z, radians, subtracted from the
heading).  The other four components are zero in every file of the release.

Per annotated thermal box the ground position is computed with the corrected
and with the published (uncorrected) pose, for a camera over locally planar
terrain at the frame's corrected height above ground (terrain.py), through the
undistorted pixel of the box centre with the camera matrix recovered from the
flight's mask (distortion.py).  The corrected camera height is the take-off-
referenced height of terrain.py (terrain at take-off + barometric height above
take-off - terrain at the nadir); the uncorrected one is that minus the
altitude offset.  The displacement is split into the altitude
part (height only) and the heading part (heading only).

Independent references per flight, over the annotated frames:

  * altitude: DJI's take-off altitude (altitude above sea level minus height
    above take-off) against the terrain model at the take-off point; the
    resulting offset is what the published altitude is off by if the
    barometric height above take-off is right;
  * heading: the published heading (the aircraft's compass heading) against
    the gimbal yaw written in the SRT, i.e. the direction the camera faced;
  * scale: the renderer's default field of view (50 deg) against the frames'
    own (recovered from the mask), which a correction fitted in that renderer
    would absorb as an altitude change of h * (f_nominal / f_true - 1).

Outputs into <out>/tables: correction_boxes.csv, correction_flights.csv,
correction_summary.json.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import common as C   # noqa: E402
import frame_dem_animation as fa   # noqa: E402
import srt_airdata_sync_eval as se   # noqa: E402

F_NOMINAL = (C.H / 2) / math.tan(math.radians(25))


def ground_xy(height: float | np.ndarray, heading_deg: np.ndarray, tilt_deg: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Ground point (relative to the nadir column) of the ray through normalised image coords n, camera at `height` over a plane."""
    out = np.empty((len(n), 2))
    h = np.broadcast_to(np.asarray(height, float), (len(n),))
    for i in range(len(n)):
        fwd, right, up = fa.camera_basis(float(tilt_deg[i]), 0.0, float(heading_deg[i]))
        d = fwd + n[i, 0] * right - n[i, 1] * up
        t = h[i] / -d[2]
        out[i] = t * d[:2]
    return out


def load_agl(data: Path, fid: str, n: int) -> np.ndarray:
    agl = np.full(n, np.nan)
    p = data / "agl" / f"{fid}_agl.csv"
    if p.exists():
        for row in csv.DictReader(open(p)):
            if row["agl"] != "":
                agl[int(row["frame"])] = float(row["agl"])
    return agl


def camera_matrices(tables: Path) -> dict:
    """flight -> (f, cx, cy) of the undistorted thermal frame, from distortion.py's tables."""
    und = {}
    for r in csv.DictReader(open(tables / "undistortion.csv")):
        if r["camera"] == "thermal":
            und[r["mask"]] = (float(r["f_px"]), float(r["cx"]), float(r["cy"]))
    out = {}
    for r in csv.DictReader(open(tables / "distortion_flights.csv")):
        if r.get("mask_t") in und:
            out[r["flight"]] = und[r["mask_t"]]
    return out


def takeoff_offset(fl: C.Flight, poses: dict, frames: np.ndarray):
    """(terrain at take-off + height above take-off) - published altitude, median over the frames; and the take-off terrain error."""
    import terrain as T
    from pyproj import Transformer
    if not fl.has_raw:
        return np.nan, np.nan
    rows = [{(k or "").strip(): v for k, v in r.items()} for r in csv.DictReader(open(fl.raw / "air_data.csv", encoding="utf-8"))]
    try:
        hat = np.array([float(r["height_above_takeoff(feet)"]) * 0.3048 for r in rows])
        asl = np.array([float(r["altitude_above_seaLevel(feet)"]) * 0.3048 for r in rows])
        lat = np.array([float(r["latitude"]) for r in rows]); lon = np.array([float(r["longitude"]) for r in rows])
    except (KeyError, ValueError):
        return np.nan, np.nan
    ok = np.where((np.abs(lat) > 1) & (np.abs(hat) < 0.5))[0]
    if not len(ok):
        return np.nan, np.nan
    k0 = ok[0]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x0, y0 = tr.transform([lon[k0]], [lat[k0]])
    try:
        z0 = float(T.ground_heights(np.array(x0, float), np.array(y0, float), None)[0])
    except Exception:   # noqa: BLE001
        return np.nan, np.nan
    ad = C.load_airdata(fl)
    t0 = ad["t"][0]
    ads = C.seconds(ad["t"], t0)
    ps = C.seconds(poses["t"][frames], t0)
    inside = (ps >= ads[0]) & (ps <= ads[-1])
    if not inside.any():
        return np.nan, z0 - (asl[k0] - hat[k0])
    hat_f = np.interp(ps[inside], ads, hat)
    implied = z0 + hat_f - poses["alt"][frames][inside]
    return float(np.median(implied)), float(z0 - (asl[k0] - hat[k0]))


def heading_offset(fl: C.Flight, poses: dict, frames: np.ndarray) -> float:
    """Published heading - SRT gimbal yaw, median over the frames (degrees)."""
    parts = C.load_srt(fl, "T")
    if not parts:
        return np.nan
    t = np.concatenate([p["t"] for p in parts]); yaw = np.concatenate([p["gb_yaw"] for p in parts])
    order = np.argsort(t)
    t, yaw = t[order], yaw[order]
    t0 = t[0]
    ss = C.seconds(t, t0); ps = C.seconds(poses["t"][frames], t0)
    inside = (ps >= ss[0] - 1) & (ps <= ss[-1] + 1)
    if not inside.any():
        return np.nan
    j = C.nearest_index(ss, ps[inside])
    return float(np.median(se.ang_diff(poses["heading"][frames][inside], yaw[j])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    ap.add_argument("--flights", nargs="*", default=None)
    args = ap.parse_args()
    tdir = args.out / "tables"
    kmat = camera_matrices(tdir)
    box_rows, flight_rows = [], []
    fls = C.flights(args.data, args.flights)
    print(f"{len(fls)} flights")
    for fl in fls:
        poses = C.load_poses(fl)
        gt = C.load_gt(fl)
        corr = fl.path("correction.json")
        tc, rc = fa.load_corrections(corr, poses["n"]) if corr.exists() else (np.zeros((poses["n"], 3)), np.zeros((poses["n"], 3)))
        raw_corr = json.load(open(corr)) if corr.exists() else {}
        fine = raw_corr.get("fine_corrections") or []
        n_inverted = sum(1 for f in fine if int(f.get("end_frame", 0)) < int(f.get("start_frame", 0)))
        hts = C.load_heights(args.data, fl.fid, poses["n"])
        agl = np.where(C.valid_height(hts["takeoff"]), hts["takeoff"], np.nan)     # corrected camera height, take-off referenced
        f_true, cx, cy = kmat.get(fl.fid, (np.nan, (C.W - 1) / 2, (C.H - 1) / 2))
        fr = gt["frame"].astype(int)
        valid = (fr >= 0) & (fr < poses["n"]) & np.isfinite(agl[np.clip(fr, 0, poses["n"] - 1)]) & np.isfinite(f_true)
        g, fr = gt[valid], fr[valid]
        rec = dict(flight=fl.fid, n_boxes=int(len(gt)), n_used=int(len(g)), n_fine=len(fine), n_fine_inverted=n_inverted,
                   default_dz=float(tc[0, 2]) if not fine else float((raw_corr.get("translation") or {}).get("z", 0.0)),
                   default_rz_deg=math.degrees(float((raw_corr.get("rotation") or {}).get("z", 0.0))),
                   dz=np.nan, rz_deg=np.nan, agl=np.nan, f_true=f_true, takeoff_offset=np.nan, takeoff_terrain_error=np.nan,
                   heading_offset=np.nan, fov_term=np.nan, frames_in_fine=np.nan)
        if len(g):
            uv = np.stack([g["l"] + g["w"] / 2, g["t"] + g["h"] / 2], axis=1).astype(float)
            n = np.stack([(uv[:, 0] - cx) / f_true, (uv[:, 1] - cy) / f_true], axis=1)
            h_c = agl[fr]
            dz = tc[fr, 2]
            rz = np.degrees(rc[fr, 2])
            head = poses["heading"][fr]
            tilt = poses["tilt"][fr]
            h_u = np.maximum(h_c - dz, 1.0)
            p_c = ground_xy(h_c, head - rz, tilt, n)
            p_u = ground_xy(h_u, head, tilt, n)
            p_alt = ground_xy(h_u, head - rz, tilt, n)       # only the altitude left uncorrected
            p_head = ground_xy(h_c, head, tilt, n)           # only the heading left uncorrected
            d_tot = np.linalg.norm(p_u - p_c, axis=1)
            d_alt = np.linalg.norm(p_alt - p_c, axis=1)
            d_head = np.linalg.norm(p_head - p_c, axis=1)
            for i in range(len(g)):
                box_rows.append((fl.fid, int(fr[i]), int(g["tid"][i]), g["species"][i], round(float(h_c[i]), 2), round(float(dz[i]), 3),
                                 round(float(rz[i]), 3), round(float(np.hypot(*n[i])), 4), round(float(d_tot[i]), 3),
                                 round(float(d_alt[i]), 3), round(float(d_head[i]), 3)))
            frames = np.unique(fr)
            rec.update(dz=float(np.median(tc[frames, 2])), rz_deg=float(np.median(np.degrees(rc[frames, 2]))), agl=float(np.median(agl[frames])),
                       fov_term=float(np.median(agl[frames]) * (F_NOMINAL / f_true - 1)),
                       frames_in_fine=float(np.mean([any(int(f["start_frame"]) <= x <= int(f["end_frame"]) for f in fine) for x in frames])) if fine else 0.0)
            try:
                rec["takeoff_offset"], rec["takeoff_terrain_error"] = takeoff_offset(fl, poses, frames)
            except Exception as exc:   # noqa: BLE001
                print(f"  flight {fl.fid}: take-off lookup failed: {exc}")
            try:
                rec["heading_offset"] = heading_offset(fl, poses, frames)
            except Exception as exc:   # noqa: BLE001
                print(f"  flight {fl.fid}: heading lookup failed: {exc}")
        flight_rows.append(rec)
        print(f"flight {fl.fid}: {rec['n_used']} boxes, dz {rec['dz']:+.2f} m, rz {rec['rz_deg']:+.2f} deg | take-off offset {rec['takeoff_offset']:+.2f} m, "
              f"fov term {rec['fov_term']:+.2f} m | compass - gimbal {rec['heading_offset']:+.2f} deg", flush=True)

    with open(tdir / "correction_boxes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["flight", "frame", "tid", "species", "agl", "dz", "rz_deg", "r_norm", "disp_total", "disp_altitude", "disp_heading"])
        wr.writerows(box_rows)
    with open(tdir / "correction_flights.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(flight_rows[0].keys()))
        wr.writeheader()
        wr.writerows(flight_rows)

    # ---- summary ------------------------------------------------------------------------------------
    b = np.array([(r[8], r[9], r[10], r[4]) for r in box_rows], float)
    fr = [r for r in flight_rows if np.isfinite(r["dz"])]
    dzs = np.array([r["dz"] for r in fr]); rzs = np.array([r["rz_deg"] for r in fr])

    def dist(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return dict(n=int(len(x)), median=C.pct(x, 50), mean=float(x.mean()) if len(x) else np.nan, p95=C.pct(x, 95), max=float(x.max()) if len(x) else np.nan,
                    over_half=float(np.mean(x > C.MATCH_RADIUS / 2)), over_match=float(np.mean(x > C.MATCH_RADIUS)))
    alt_ref = np.array([r["takeoff_offset"] + r["fov_term"] for r in fr]); alt_res = dzs - alt_ref
    alt_ref0 = np.array([r["takeoff_offset"] for r in fr]); alt_res0 = dzs - alt_ref0
    head_ref = np.array([r["heading_offset"] for r in fr]); head_res = se.ang_diff(rzs, head_ref)
    corrected = (np.abs(dzs) > 0) | (np.abs(rzs) > 0)

    def agree(res, mask, tol):
        res = res[mask & np.isfinite(res)]
        return dict(n=int(len(res)), median_abs=C.pct(np.abs(res), 50), within=float(np.mean(np.abs(res) <= tol)), tol=tol)

    def corr(a, c, mask):
        m = mask & np.isfinite(a) & np.isfinite(c)
        return float(np.corrcoef(a[m], c[m])[0, 1]) if m.sum() > 2 else np.nan

    alt_mask = np.abs(dzs) > 0
    head_mask = np.abs(rzs) > 0
    summ = dict(
        n_flights=len(flight_rows), n_flights_used=len(fr), n_boxes=int(len(b)),
        flights_with_altitude_correction=int(alt_mask.sum()), flights_with_heading_correction=int(head_mask.sum()),
        flights_corrected=int(corrected.sum()), flights_with_fine=int(sum(1 for r in flight_rows if r["n_fine"] > 0)),
        fine_segments=int(sum(r["n_fine"] for r in flight_rows)), fine_segments_inverted=int(sum(r["n_fine_inverted"] for r in flight_rows)),
        dz=dist(np.abs(dzs[alt_mask])), dz_signed=dict(median=C.pct(dzs[alt_mask], 50), min=float(dzs.min()), max=float(dzs.max()), share_negative=float(np.mean(dzs[alt_mask] < 0))),
        rz_deg=dist(np.abs(rzs[head_mask])), rz_signed=dict(median=C.pct(rzs[head_mask], 50), min=float(rzs.min()), max=float(rzs.max())),
        disp_total=dist(b[:, 0]), disp_altitude=dist(b[:, 1]), disp_heading=dist(b[:, 2]),
        boxes_uncorrected_frames=float(np.mean((b[:, 0] == 0))),
        altitude_vs_takeoff=dict(with_fov_term=agree(alt_res, alt_mask, 1.0), without_fov_term=agree(alt_res0, alt_mask, 1.0),
                                 r_with_fov_term=corr(dzs, alt_ref, alt_mask), r_without=corr(dzs, alt_ref0, alt_mask),
                                 within_2m=agree(alt_res, alt_mask, 2.0)["within"]),
        heading_vs_gimbal=dict(**agree(head_res, head_mask, 2.0), r=corr(rzs, head_ref, head_mask), within_5deg=agree(head_res, head_mask, 5.0)["within"]),
        by_alt_bin=[])
    bins = C.alt_bin(b[:, 3])
    for i, lab in enumerate(C.ALT_LABELS):
        s = bins == i
        summ["by_alt_bin"].append(dict(bin=lab, n=int(s.sum()), median=C.pct(b[s, 0], 50), p95=C.pct(b[s, 0], 95),
                                       over_match=float(np.mean(b[s, 0] > C.MATCH_RADIUS)) if s.any() else np.nan))
    json.dump(summ, open(tdir / "correction_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "by_alt_bin"}, indent=1))


if __name__ == "__main__":
    main()
