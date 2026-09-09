#!/usr/bin/env python3
"""
Is the SRT + AirData combination of the BAMBI pose extractor necessary, or is
interpolating the AirData log alone good enough?

For one flight (raw SRT files + air_data.csv) this script
  1. reproduces the extractor: AirData positions, SRT frame times, one clock
     offset fitted by matching the two GPS traces -> the reference pose per frame;
  2. measures what the alternatives cost, split into straight and turning flight:
     - AirData only, synced by the isVideo flag and a constant frame rate
       (no SRT at all);
     - AirData only at reduced log rates (linear interpolation between rows);
     - SRT GPS only (no AirData);
     - heading interpolation between AirData rows.
"""
import csv
import datetime as dt
import math
import re
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.optimize import minimize_scalar

from zoneinfo import ZoneInfo

FPS = 30.0
VIENNA = ZoneInfo("Europe/Vienna")

# ----------------------------------------------------------------------------- parsing

def parse_srt(path: Path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    blocks = txt.strip().split("\n\n")
    ts, lat, lon, alt, yaw = [], [], [], [], []
    for b in blocks:
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[.,](\d+)", b)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") + dt.timedelta(milliseconds=int(m.group(2)[:3].ljust(3, "0")))
        g = re.search(r"latitude: ([\-\d.]+)\] \[longitude: ([\-\d.]+)\] \[rel_alt: ([\-\d.]+) abs_alt: ([\-\d.]+)\] \[gb_yaw: ([\-\d.]+)", b)
        if not g:
            continue
        ts.append(t); lat.append(float(g.group(1))); lon.append(float(g.group(2))); alt.append(float(g.group(4))); yaw.append(float(g.group(5)))
    return dict(t=np.array(ts), lat=np.array(lat), lon=np.array(lon), alt=np.array(alt), gb_yaw=np.array(yaw))


def parse_airdata(path: Path):
    rows = [{(k or "").strip(): v for k, v in r.items()} for r in csv.DictReader(open(path, encoding="utf-8"))]
    t_ms = np.array([float(r["time(millisecond)"]) for r in rows])
    d0 = dt.datetime.strptime(rows[0]["datetime(utc)"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    d0 = d0.astimezone(VIENNA).replace(tzinfo=None)          # SRT timestamps are local time
    t = np.array([d0 + dt.timedelta(milliseconds=m - t_ms[0]) for m in t_ms])
    f = lambda k: np.array([float(r[k]) for r in rows])
    return dict(t=t, t_ms=t_ms, lat=f("latitude"), lon=f("longitude"),
                alt=f("altitude_above_seaLevel(feet)") * 0.3048,
                heading=f("compass_heading(degrees)"), gimbal=f("gimbal_heading(degrees)"),
                is_video=np.array([r["isVideo"] == "1" for r in rows]),
                speed_mph=f("speed(mph)"))


def seconds(times, t0):
    return np.array([(x - t0).total_seconds() for x in times])


# ----------------------------------------------------------------------------- helpers

def video_segment(ad, video_start):
    """AirData rows of the isVideo run that starts closest to the video start (extractor logic)."""
    idx = np.where(ad["is_video"])[0]
    diffs = np.abs(seconds(ad["t"][idx], video_start))
    best = idx[np.argmin(diffs)]
    # walk to the start of this run, then to its end
    a = best
    while a > 0 and ad["is_video"][a - 1]:
        a -= 1
    b = best
    while b + 1 < len(ad["t"]) and ad["is_video"][b + 1]:
        b += 1
    return a, b


def fit_offset(srt_s, srt_lat, srt_lon, ad_s, ad_lat, ad_lon):
    """The extractor's clock offset: shift the SRT times so the SRT GPS trace matches the AirData trace."""
    def mse(s):
        la = np.interp(srt_s + s, ad_s, ad_lat)
        lo = np.interp(srt_s + s, ad_s, ad_lon)
        return np.mean((la - srt_lat) ** 2 + (lo - srt_lon) ** 2)
    res = minimize_scalar(mse, bounds=(-30, 30), method="bounded")
    return float(res.x)


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def stats(err, mask=None, label=""):
    e = err if mask is None else err[mask]
    if len(e) == 0:
        return f"{label:<42s}   (no samples)"
    return (f"{label:<42s} n={len(e):6d}  mean {np.mean(e):6.2f}  median {np.median(e):6.2f}  "
            f"p95 {np.percentile(e, 95):6.2f}  max {np.max(e):6.2f}")


# ----------------------------------------------------------------------------- main

def evaluate(flight: str, raw_dir: Path, poses_json=None, verbose: bool = True):
    srt_files = sorted(p for p in raw_dir.iterdir() if p.suffix.upper() == ".SRT" and re.search(r"_\d{4}_T(\.|_)", p.name))
    srt_parts = [parse_srt(p) for p in srt_files]
    srt = {k: np.concatenate([s[k] for s in srt_parts]) for k in srt_parts[0]}
    ad = parse_airdata(raw_dir / "air_data.csv")

    a, b = video_segment(ad, srt["t"][0])
    seg = slice(a, b + 1)
    t0 = ad["t"][a]
    ad_s = seconds(ad["t"][seg], t0)
    srt_s = seconds(srt["t"], t0)

    # metric coordinates
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    ad_x, ad_y = tr.transform(ad["lon"][seg], ad["lat"][seg])
    srt_x, srt_y = tr.transform(srt["lon"], srt["lat"])

    offset = fit_offset(srt_s, srt["lat"], srt["lon"], ad_s, ad["lat"][seg], ad["lon"][seg])
    frame_s = srt_s + offset                      # the extractor's frame times on the AirData clock
    ref_x = np.interp(frame_s, ad_s, ad_x)
    ref_y = np.interp(frame_s, ad_s, ad_y)

    # AirData log rate, gaps
    dts = np.diff(ad_s)
    log_hz = 1.0 / np.median(dts)

    # motion classes from the AirData track: course-over-ground rate and speed
    vx = np.gradient(ad_x, ad_s); vy = np.gradient(ad_y, ad_s)
    speed = np.hypot(vx, vy)
    k = max(3, int(round(log_hz)))                     # ~1 s smoothing
    ker = np.ones(k) / k
    vxs = np.convolve(vx, ker, mode="same"); vys = np.convolve(vy, ker, mode="same")
    course = np.degrees(np.arctan2(vxs, vys))
    course_rate = np.abs(np.gradient(unwrap_deg(course), ad_s))
    moving = np.convolve(speed, ker, mode="same") > 1.0
    straight_ad = moving & (course_rate < 5.0)
    turning_ad = moving & (course_rate > 20.0)
    # the same classes at frame times
    cr_f = np.interp(frame_s, ad_s, course_rate)
    sp_f = np.interp(frame_s, ad_s, np.convolve(speed, ker, mode="same"))
    straight_f = (sp_f > 1.0) & (cr_f < 5.0)
    turning_f = (sp_f > 1.0) & (cr_f > 20.0)

    # published poses as a sanity check of the reproduction
    import json
    repro_err = time_err = np.zeros(1)
    if poses_json is not None:
        pub = json.load(open(poses_json))["images"]
        pub_t = np.array([dt.datetime.fromisoformat(p["timestamp"]).replace(tzinfo=None) for p in pub])
        pub_s = seconds(pub_t, t0)
        pub_x, pub_y = tr.transform(np.array([p["lng"] for p in pub]), np.array([p["lat"] for p in pub]))
        j = np.searchsorted(frame_s, pub_s).clip(1, len(frame_s) - 1)
        j = np.where(np.abs(frame_s[j - 1] - pub_s) < np.abs(frame_s[j] - pub_s), j - 1, j)
        repro_err = np.hypot(ref_x[j] - pub_x, ref_y[j] - pub_y)
        time_err = np.abs(frame_s[j] - pub_s)
    if not verbose:
        import builtins
        _print = builtins.print
        print_ = lambda *a, **k: None
    else:
        print_ = print

    print_("=" * 100)
    print_(f"flight {flight}: {len(srt_files)} SRT file(s), {len(srt_s)} video frames, AirData {log_hz:.0f} Hz, "
          f"video segment {ad_s[-1] - ad_s[0]:.0f} s ({b - a + 1} rows), gaps > 2 s: {(dts > 2).sum()}, "
          f"largest gap {dts.max():.2f} s")
    print_(f"  median ground speed while moving {np.median(speed[moving]):.2f} m/s, "
          f"straight {straight_ad.mean() * 100:.0f} % / turning {turning_ad.mean() * 100:.0f} % of the video time")
    print_(f"  fitted SRT->AirData clock offset {offset:+.3f} s   "
          f"(first frame on the AirData clock: {(t0 + dt.timedelta(seconds=frame_s[0])).time()})")
    print_(f"  reproduction vs published poses: time |dt| median {np.median(time_err) * 1000:.0f} ms, "
          f"position {stats(repro_err, label='').strip()}")

    print_("\n[A] AirData only, no SRT: frame times from the isVideo flag onset + constant 30 fps")
    onset = 0.0                                     # ad_s[0] is the first isVideo row
    naive_s = onset + np.arange(len(srt_s)) / FPS
    dt_sync = naive_s[0] - frame_s[0]
    print_(f"    isVideo onset is {dt_sync * 1000:+.0f} ms from the fitted first-frame time; "
          f"over the video the SRT frame clock drifts {((frame_s[-1] - frame_s[0]) - (naive_s[-1] - naive_s[0])) * 1000:+.0f} ms against 30 fps")
    nx = np.interp(naive_s, ad_s, ad_x); ny = np.interp(naive_s, ad_s, ad_y)
    e = np.hypot(nx - ref_x, ny - ref_y)
    print_("   ", stats(e, None, "position error, all frames (m)"))
    print_("   ", stats(e, straight_f, "  straight flight"))
    print_("   ", stats(e, turning_f, "  turning"))
    fps_true = (len(srt_s) - 1) / (frame_s[-1] - frame_s[0])
    fixed_s = onset + np.arange(len(srt_s)) / fps_true
    fx = np.interp(fixed_s, ad_s, ad_x); fy = np.interp(fixed_s, ad_s, ad_y)
    e2 = np.hypot(fx - ref_x, fy - ref_y)
    print_("   ", stats(e2, None, f"same with the true frame rate ({fps_true:.3f} fps)"))
    print_("    a pure timing error of dt costs speed x dt: at the median speed "
          f"{np.median(speed[moving]):.1f} m/s -> 0.1 s = {0.1 * np.median(speed[moving]):.2f} m, "
          f"0.5 s = {0.5 * np.median(speed[moving]):.2f} m")

    print_("\n[B] AirData only, correctly synced, but at a lower log rate (linear interpolation between rows)")
    print_("    error of the interpolated rows against the rows that were dropped, metres")
    for step in sorted({int(round(log_hz / hz)) for hz in (5, 2, 1, 0.5, 0.2) if log_hz / hz >= 2}):
        keep = np.zeros(len(ad_s), bool); keep[::step] = True
        ix = np.interp(ad_s[~keep], ad_s[keep], ad_x[keep]); iy = np.interp(ad_s[~keep], ad_s[keep], ad_y[keep])
        e = np.hypot(ix - ad_x[~keep], iy - ad_y[~keep])
        m_s = straight_ad[~keep]; m_t = turning_ad[~keep]
        rate = log_hz / step
        print_("   ", stats(e, None, f"{rate:4.1f} Hz ({1 / rate:.1f} s between rows): all"))
        print_("   ", stats(e, m_s, "      straight"))
        print_("   ", stats(e, m_t, "      turning"))

    print_("\n[C] SRT GPS only (no AirData), after the same clock offset")
    e = np.hypot(srt_x - ref_x, srt_y - ref_y)
    print_("   ", stats(e, None, "position error (m), all frames"))
    print_("   ", stats(e, straight_f, "  straight"))
    print_("   ", stats(e, turning_f, "  turning"))
    dec = max(len(str(v).split(".")[1]) if "." in str(v) else 0 for v in srt["lat"][:200])
    print_(f"    SRT latitude/longitude have {dec} decimals -> {111320 / 10 ** dec:.2f} m quantisation")

    print_("\n[D] heading between AirData rows: compass heading interpolated at a lower log rate")
    for step in sorted({int(round(log_hz / hz)) for hz in (2, 1, 0.5) if log_hz / hz >= 2}):
        keep = np.zeros(len(ad_s), bool); keep[::step] = True
        h = unwrap_deg(ad["heading"][seg])
        ih = np.interp(ad_s[~keep], ad_s[keep], h[keep])
        e = np.abs(ang_diff(ih, h[~keep]))
        rate = log_hz / step
        print_("   ", stats(e, None, f"{rate:4.1f} Hz: heading error (deg), all"))
        print_("   ", stats(e, straight_ad[~keep], "      straight"))
        print_("   ", stats(e, turning_ad[~keep], "      turning"))
    gy = np.interp(frame_s, ad_s, unwrap_deg(ad["gimbal"][seg]))
    e = np.abs(ang_diff(gy, srt["gb_yaw"]))
    print_("   ", stats(e, None, "SRT gimbal yaw vs AirData gimbal heading (deg)"))

    return dict(ad_s=ad_s, ad_x=ad_x, ad_y=ad_y, frame_s=frame_s, ref_x=ref_x, ref_y=ref_y, srt_x=srt_x, srt_y=srt_y,
                course_rate=course_rate, turning_ad=turning_ad, straight_ad=straight_ad, log_hz=log_hz,
                naive_s=naive_s, speed=speed, offset=offset, onset_err=dt_sync, fps_true=fps_true,
                n_frames=len(srt_s), gap_max=float(dts.max()), n_gaps=int((dts > 2).sum()),
                srt_dec=dec, err_naive=np.hypot(nx - ref_x, ny - ref_y), err_fixed=e2, err_srt=np.hypot(srt_x - ref_x, srt_y - ref_y))


def scan(root: Path):
    """One line per flight: what the AirData log looks like and what dropping the SRT would cost."""
    print(f"{'flight':>6} {'frames':>6} {'AirData':>8} {'gap>2s':>6} {'maxgap':>6} {'SRT dec':>7} {'clock off':>9} "
          f"{'onset err':>9} {'true fps':>8} {'noSRT mean':>10} {'noSRT p95':>9} {'1Hz turn p95':>12} {'SRT-only mean':>13} {'speed':>5}")
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "air_data.csv").exists()):
        try:
            r = evaluate(d.name, d, None, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"{d.name:>6}  failed: {exc}")
            continue
        step = int(round(r["log_hz"]))
        keep = np.zeros(len(r["ad_s"]), bool); keep[::step] = True
        ix = np.interp(r["ad_s"][~keep], r["ad_s"][keep], r["ad_x"][keep]); iy = np.interp(r["ad_s"][~keep], r["ad_s"][keep], r["ad_y"][keep])
        e1 = np.hypot(ix - r["ad_x"][~keep], iy - r["ad_y"][~keep])
        turn = r["turning_ad"][~keep]
        p95_turn = np.percentile(e1[turn], 95) if turn.any() else float("nan")
        mv = r["speed"] > 1.0
        print(f"{d.name:>6} {r['n_frames']:6d} {r['log_hz']:6.0f} Hz {r['n_gaps']:6d} {r['gap_max']:5.1f}s {r['srt_dec']:7d} "
              f"{r['offset']:+8.2f}s {r['onset_err']:+8.2f}s {r['fps_true']:8.3f} {np.mean(r['err_naive']):9.2f}m "
              f"{np.percentile(r['err_naive'], 95):8.2f}m {p95_turn:11.2f}m {np.mean(r['err_srt']):12.2f}m {np.median(r['speed'][mv]) if mv.any() else 0:4.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        scan(Path(sys.argv[2]))
        sys.exit(0)
    out = {}
    for flight, raw, poses in (("146", "/home/user/bambi_raw/146", "/home/user/bambi_downloads/146_matched_poses.json"),
                               ("14", "/home/user/bambi_raw", "/home/user/bambi_downloads/14_matched_poses.json")):
        out[flight] = evaluate(flight, Path(raw), Path(poses))
    np.save(Path(__file__).with_name("sync_eval_out.npy"), out, allow_pickle=True)
