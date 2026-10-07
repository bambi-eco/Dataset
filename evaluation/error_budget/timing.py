#!/usr/bin/env python3
"""
Camera-pose timing across the BAMBI dataset: what the SRT frame times add to
the AirData flight log, and what the published poses would look like without
them.

For every flight with raw logs and for every thermal recording of it the
script reproduces the pose extractor (AirData positions at the SRT frame times
with one fitted clock offset per recording) and measures three alternatives
per video frame:

  * AirData only: frames timed from the log's isVideo onset at 30 fps,
  * AirData only with the true mean frame rate of the recording,
  * SRT GPS only: the coordinates written next to every frame.

The per-frame errors are then looked up at every annotated box, so the
distributions are weighted the way the detection benchmark is.  A few side
quantities come along: AirData log rate and gaps, the interpolation error of
the log in turns at 1 Hz, the SRT coordinate quantisation, the RGB-thermal
frame-pairing time difference and the gimbal heading disagreement.

Outputs CSV tables and a JSON summary into <out>/tables.
"""
import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
import common as C   # noqa: E402
import srt_airdata_sync_eval as se   # noqa: E402

TR = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)


def analyse_recording(ad: dict, srt: dict, ad_xy):
    """The extractor's reproduction and the alternatives for one recording.  Returns None when the log has no video run."""
    if not ad["is_video"].any():
        return None
    a, b = se.video_segment(ad, srt["t"][0])
    seg = slice(a, b + 1)
    t0 = ad["t"][a]
    ad_s = C.seconds(ad["t"][seg], t0)
    srt_s = C.seconds(srt["t"], t0)
    if len(ad_s) < 3 or abs(srt_s[0]) > 600:          # no run anywhere near this recording
        return None
    ad_x, ad_y = ad_xy[0][seg], ad_xy[1][seg]
    offset = se.fit_offset(srt_s, srt["lat"], srt["lon"], ad_s, ad["lat"][seg], ad["lon"][seg])
    frame_s = srt_s + offset
    ref_x = np.interp(frame_s, ad_s, ad_x); ref_y = np.interp(frame_s, ad_s, ad_y)
    dts = np.diff(ad_s)
    log_hz = 1.0 / np.median(dts)
    vx = np.gradient(ad_x, ad_s); vy = np.gradient(ad_y, ad_s)
    speed = np.hypot(vx, vy)
    k = max(3, int(round(log_hz)))
    ker = np.ones(k) / k
    speed_s = np.convolve(speed, ker, mode="same")
    vxs = np.convolve(vx, ker, mode="same"); vys = np.convolve(vy, ker, mode="same")
    course_rate = np.abs(np.gradient(se.unwrap_deg(np.degrees(np.arctan2(vxs, vys))), ad_s))
    moving = speed_s > 1.0
    straight_ad = moving & (course_rate < 5.0)
    turning_ad = moving & (course_rate > 20.0)

    # alternatives
    naive_s = np.arange(len(srt_s)) / C.FPS                       # isVideo onset (ad_s[0] = 0) + 30 fps
    fps_true = (len(srt_s) - 1) / max(frame_s[-1] - frame_s[0], 1e-6)
    fixed_s = np.arange(len(srt_s)) / fps_true
    err_naive = np.hypot(np.interp(naive_s, ad_s, ad_x) - ref_x, np.interp(naive_s, ad_s, ad_y) - ref_y)
    err_fixed = np.hypot(np.interp(fixed_s, ad_s, ad_x) - ref_x, np.interp(fixed_s, ad_s, ad_y) - ref_y)
    srt_x, srt_y = TR.transform(srt["lon"], srt["lat"])
    err_srt = np.hypot(srt_x - ref_x, srt_y - ref_y)
    outside = (frame_s < ad_s[0] - 0.5) | (frame_s > ad_s[-1] + 0.5)      # no log there: the reference is unknown
    for e in (err_naive, err_fixed, err_srt):
        e[outside] = np.nan
    # 1 Hz interpolation of the log, error on the dropped rows
    step = max(int(round(log_hz)), 2)
    keep = np.zeros(len(ad_s), bool); keep[::step] = True
    e1 = np.hypot(np.interp(ad_s[~keep], ad_s[keep], ad_x[keep]) - ad_x[~keep], np.interp(ad_s[~keep], ad_s[keep], ad_y[keep]) - ad_y[~keep])
    turn1 = turning_ad[~keep]; straight1 = straight_ad[~keep]
    # gimbal heading disagreement (SRT gimbal yaw vs AirData gimbal heading)
    gy = np.interp(frame_s, ad_s, se.unwrap_deg(ad["gimbal"][seg]))
    yaw_diff = se.ang_diff(gy, srt["gb_yaw"])
    dec = max(len(str(v).split(".")[1]) if "." in str(v) else 0 for v in srt["lat"][:200])
    # SRT coordinate quantisation seen as the step between consecutive distinct positions
    return dict(t0=t0, ad_s=ad_s, frame_s=frame_s, offset=offset, onset_lag=float(-frame_s[0]), fps_true=fps_true,
                log_hz=log_hz, n_gaps=int((dts > 2).sum()), gap_max=float(dts.max()),
                coverage=float((ad_s[-1] - ad_s[0]) / max(frame_s[-1] - frame_s[0], 1e-6)),
                speed_frames=np.interp(frame_s, ad_s, speed_s), course_rate_frames=np.interp(frame_s, ad_s, course_rate),
                median_speed=float(np.median(speed_s[moving])) if moving.any() else 0.0,
                err_naive=err_naive, err_fixed=err_fixed, err_srt=err_srt,
                e1_turn_p95=C.pct(e1[turn1], 95) if turn1.any() else np.nan, e1_straight_p95=C.pct(e1[straight1], 95) if straight1.any() else np.nan,
                e1_all_p95=C.pct(e1, 95), yaw_diff_median=float(np.median(yaw_diff)), yaw_diff_mad=float(np.median(np.abs(yaw_diff - np.median(yaw_diff)))),
                srt_dec=dec, n_frames=len(srt_s), ref_x=ref_x, ref_y=ref_y, srt_x=srt_x, srt_y=srt_y, outside=outside,
                share_outside=float(outside.mean()), ad_hz_rows=len(ad_s), gimbal_frames=gy)


def concat_parts(parts: list[dict]) -> dict:
    keys = [k for k in parts[0] if k != "name"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["name"] = "+".join(p["name"] for p in parts)
    return out


def group_recordings(ad: dict, parts: list[dict]) -> list[dict]:
    """Concatenate SRT files that fall into the same AirData video run (DJI splits long recordings into several files)."""
    if not ad["is_video"].any():
        return parts
    groups = {}
    for p in parts:
        key = se.video_segment(ad, p["t"][0])
        groups.setdefault(key, []).append(p)
    return [concat_parts(sorted(g, key=lambda q: q["t"][0])) for _, g in sorted(groups.items())]


def pair_rgb_thermal(srt_t: dict, srt_v: dict):
    """|dt| between every thermal frame and the nearest RGB frame of the same recording (same camera clock)."""
    t0 = srt_t["t"][0]
    ts = C.seconds(srt_t["t"], t0); vs = C.seconds(srt_v["t"], t0)
    j = C.nearest_index(vs, ts)
    return np.abs(vs[j] - ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/home/user/data"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    ap.add_argument("--flights", nargs="*", default=None)
    args = ap.parse_args()
    tdir = args.out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    rec_rows, box_rows, skipped = [], [], []
    fls = [f for f in C.flights(args.data, args.flights) if f.has_raw]
    print(f"{len(fls)} flights with raw logs")
    for fl in fls:
        try:
            ad = C.load_airdata(fl)
            parts = C.load_srt(fl, "T")
            vparts = C.load_srt(fl, "V")
            if not parts:
                skipped.append((fl.fid, "no thermal SRT")); continue
            ad_xy = TR.transform(ad["lon"], ad["lat"])
            parts = group_recordings(ad, parts)          # files split by the camera inside one video run -> one recording
            vcat = concat_parts(vparts) if vparts else None
            poses = C.load_poses(fl)
            gt = C.load_gt(fl)
            meta = C.load_metadata(fl).get("flight_info", {})
            recs = []
            for i, srt in enumerate(parts):
                r = analyse_recording(ad, srt, ad_xy)
                if r is None:
                    rec_rows.append(dict(flight=fl.fid, recording=i, file=srt["name"], status="no AirData video run")); continue
                # RGB pairing: the V recording closest in start time
                pair = None
                if vcat is not None and abs((vcat["t"][0] - srt["t"][0]).total_seconds()) < 3600:
                    pair = pair_rgb_thermal(srt, vcat)
                r["pair_dt_median"] = float(np.median(pair)) if pair is not None else np.nan
                r["pair_dt_p95"] = C.pct(pair, 95) if pair is not None else np.nan
                r["abs_t"] = np.array([r["t0"] + dt.timedelta(seconds=s) for s in r["frame_s"]])
                recs.append(r)
                # published poses inside this recording: reproduction check
                pose_s = C.seconds(poses["t"], r["t0"])
                inside = (pose_s >= r["frame_s"][0] - 0.5) & (pose_s <= r["frame_s"][-1] + 0.5)
                repro = repro_srt = np.nan
                if inside.any():
                    j = C.nearest_index(r["frame_s"], pose_s[inside])
                    px, py = TR.transform(poses["lng"][inside], poses["lat"][inside])
                    d_ad = np.hypot(r["ref_x"][j] - px, r["ref_y"][j] - py)
                    d_ad[r["outside"][j]] = np.nan
                    repro = float(np.nanmedian(d_ad)) if np.isfinite(d_ad).any() else np.nan
                    repro_srt = float(np.median(np.hypot(r["srt_x"][j] - px, r["srt_y"][j] - py)))
                # what the release did with this recording: followed the log (reproduced), the SRT, or something else
                if r["n_frames"] < 300:
                    r["verdict"] = "degenerate"
                elif not inside.any():
                    r["verdict"] = "not released"
                elif repro < 1.0:
                    r["verdict"] = "reproduced"
                elif np.isfinite(repro_srt) and repro_srt < 2.0:
                    r["verdict"] = "release follows SRT"
                else:
                    r["verdict"] = "unmatched"
                rec_rows.append(dict(flight=fl.fid, recording=i, file=srt["name"], status="ok", drone=meta.get("drone_name", ""),
                                     n_frames=r["n_frames"], airdata_hz=round(r["log_hz"], 1), gaps_over_2s=r["n_gaps"], max_gap_s=round(r["gap_max"], 2),
                                     coverage=round(r["coverage"], 3), clock_offset_s=round(r["offset"], 3), onset_lag_s=round(r["onset_lag"], 3),
                                     fps_true=round(r["fps_true"], 4), srt_decimals=r["srt_dec"], median_speed=round(r["median_speed"], 2),
                                     nosrt_mean=round(float(np.nanmean(r["err_naive"])), 3), nosrt_p95=round(C.pct(r["err_naive"], 95), 3),
                                     nosrt_fixedfps_mean=round(float(np.nanmean(r["err_fixed"])), 3),
                                     srtonly_mean=round(float(np.nanmean(r["err_srt"])), 3), srtonly_p95=round(C.pct(r["err_srt"], 95), 3),
                                     interp1hz_turn_p95=round(r["e1_turn_p95"], 3), interp1hz_straight_p95=round(r["e1_straight_p95"], 3),
                                     yaw_diff_median_deg=round(r["yaw_diff_median"], 2), yaw_diff_mad_deg=round(r["yaw_diff_mad"], 2),
                                     rgb_pair_dt_median_ms=round(r["pair_dt_median"] * 1000, 1), rgb_pair_dt_p95_ms=round(r["pair_dt_p95"] * 1000, 1),
                                     n_poses_inside=int(inside.sum()), repro_median_m=round(repro, 3) if np.isfinite(repro) else "",
                                     repro_srt_m=round(repro_srt, 3) if np.isfinite(repro_srt) else "", share_outside_log=round(r["share_outside"], 3),
                                     verdict=r["verdict"]))
            if not recs:
                skipped.append((fl.fid, "no recording matched the log")); continue
            # per box: pose -> recording/frame -> errors
            agl_frames = None
            fr = gt["frame"].astype(int)
            valid = (fr >= 0) & (fr < poses["n"])
            pose_t = poses["t"]
            for g, f in zip(gt[valid], fr[valid]):
                tf = pose_t[f]
                best = None
                for ri, r in enumerate(recs):
                    s = (tf - r["t0"]).total_seconds()
                    if r["frame_s"][0] - 0.5 <= s <= r["frame_s"][-1] + 0.5:
                        j = int(C.nearest_index(r["frame_s"], np.array([s]))[0])
                        best = (ri, j, s); break
                if best is None:
                    continue
                ri, j, s = best
                r = recs[ri]
                if r["verdict"] != "reproduced" or r["outside"][j]:
                    continue
                agl = np.interp(s, r["ad_s"], np.nan_to_num(ad["agl"][se.video_segment(ad, parts[ri]["t"][0])[0]:][:len(r["ad_s"])], nan=np.nan)) if False else np.nan
                box_rows.append((fl.fid, int(f), int(g["tid"]), g["species"], ri, j, round(float(r["speed_frames"][j]), 3), round(float(r["course_rate_frames"][j]), 2),
                                 round(float(r["err_naive"][j]), 3), round(float(r["err_fixed"][j]), 3), round(float(r["err_srt"][j]), 3),
                                 round(float(r["onset_lag"]), 3)))
            ok = [x for x in rec_rows if x["flight"] == fl.fid and x["status"] == "ok"]
            print(f"flight {fl.fid}: {len(parts)} recording(s), {len(ok)} matched, onset lag " +
                  ", ".join(f"{x['onset_lag_s']:+.2f} s" for x in ok) + f", boxes {sum(1 for b in box_rows if b[0] == fl.fid)}")
        except Exception as exc:   # noqa: BLE001
            skipped.append((fl.fid, f"error: {exc}"))
            print(f"flight {fl.fid}: failed: {exc}")

    fields = ["flight", "recording", "file", "status", "drone", "n_frames", "airdata_hz", "gaps_over_2s", "max_gap_s", "coverage", "clock_offset_s",
              "onset_lag_s", "fps_true", "srt_decimals", "median_speed", "nosrt_mean", "nosrt_p95", "nosrt_fixedfps_mean", "srtonly_mean", "srtonly_p95",
              "interp1hz_turn_p95", "interp1hz_straight_p95", "yaw_diff_median_deg", "yaw_diff_mad_deg", "rgb_pair_dt_median_ms", "rgb_pair_dt_p95_ms",
              "n_poses_inside", "repro_median_m", "repro_srt_m", "share_outside_log", "verdict"]
    with open(tdir / "timing_recordings.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rec_rows:
            wr.writerow({k: r.get(k, "") for k in fields})
    with open(tdir / "timing_boxes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["flight", "frame", "tid", "species", "recording", "srt_frame", "speed", "course_rate", "err_nosrt", "err_nosrt_fixedfps", "err_srtonly", "onset_lag"])
        wr.writerows(box_rows)
    with open(tdir / "timing_skipped.csv", "w", newline="") as f:
        csv.writer(f).writerows([("flight", "reason")] + skipped)

    verdicts = {}
    for r in rec_rows:
        verdicts[r.get("verdict", r["status"])] = verdicts.get(r.get("verdict", r["status"]), 0) + 1
    ok = [r for r in rec_rows if r["status"] == "ok" and r.get("verdict") in ("reproduced", "release follows SRT", "not released")]
    lag = np.array([r["onset_lag_s"] for r in ok]); spd = np.array([r["median_speed"] for r in ok])
    bn = np.array([b[8] for b in box_rows]); bf = np.array([b[9] for b in box_rows]); bs = np.array([b[10] for b in box_rows]); bsp = np.array([b[6] for b in box_rows])
    summ = dict(n_flights_raw=len(fls), n_flights_ok=len({r["flight"] for r in ok}), n_recordings=len(ok), n_skipped=len(skipped), verdicts=verdicts,
                n_boxes_total=int(sum(len(C.load_gt(f)) for f in fls)),
                n_boxes=len(box_rows),
                onset_lag=dict(min=float(lag.min()), median=float(np.median(lag)), p95=C.pct(lag, 95), max=float(lag.max()), share_negative=float(np.mean(lag < 0))),
                clock_offset=dict(min=float(min(r["clock_offset_s"] for r in ok)), median=float(np.median([r["clock_offset_s"] for r in ok])), max=float(max(r["clock_offset_s"] for r in ok))),
                airdata_hz=dict(values=sorted({r["airdata_hz"] for r in ok}), share_10hz=float(np.mean([r["airdata_hz"] >= 9 for r in ok]))),
                gaps=dict(recordings_with_gap_over_2s=int(sum(r["gaps_over_2s"] > 0 for r in ok)), max_gap_s=float(max(r["max_gap_s"] for r in ok))),
                fps_true=dict(median=float(np.median([r["fps_true"] for r in ok])), min=float(min(r["fps_true"] for r in ok)), max=float(max(r["fps_true"] for r in ok))),
                srt_decimals=sorted({r["srt_decimals"] for r in ok}),
                speed=dict(median=float(np.median(spd)), p95=C.pct(spd, 95)),
                repro_median_m=float(np.nanmedian([r["repro_median_m"] for r in ok if r["repro_median_m"] != ""])),
                nosrt_boxes=dict(mean=float(bn.mean()), median=C.pct(bn, 50), p95=C.pct(bn, 95), share_over_match=float(np.mean(bn > C.MATCH_RADIUS)), share_over_1m=float(np.mean(bn > 1))),
                nosrt_fixedfps_boxes=dict(mean=float(bf.mean()), median=C.pct(bf, 50), p95=C.pct(bf, 95), share_over_match=float(np.mean(bf > C.MATCH_RADIUS))),
                srtonly_boxes=dict(mean=float(bs.mean()), median=C.pct(bs, 50), p95=C.pct(bs, 95), share_over_match=float(np.mean(bs > C.MATCH_RADIUS))),
                box_speed=dict(median=C.pct(bsp, 50), p95=C.pct(bsp, 95), share_stationary=float(np.mean(bsp < 0.5))),
                interp1hz=dict(turn_p95_median=float(np.nanmedian([r["interp1hz_turn_p95"] for r in ok])), straight_p95_median=float(np.nanmedian([r["interp1hz_straight_p95"] for r in ok]))),
                yaw_diff=dict(median_abs=float(np.median(np.abs([r["yaw_diff_median_deg"] for r in ok]))), p95_abs=C.pct(np.abs([r["yaw_diff_median_deg"] for r in ok]), 95)),
                rgb_pair_dt_ms=dict(median=float(np.nanmedian([r["rgb_pair_dt_median_ms"] for r in ok])), p95=float(np.nanmedian([r["rgb_pair_dt_p95_ms"] for r in ok]))))
    json.dump(summ, open(tdir / "timing_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
