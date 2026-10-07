#!/usr/bin/env python3
"""
Figures and tables for the geometric error budget, from the CSV/JSON output of
distortion.py and timing.py.  Writes <out>/figures/*.pdf|png, <out>/tables/*.tex
and <out>/results.md (every number the text quotes, generated).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import common as C   # noqa: E402

# palette (validated categorical order, sequential blue ramp)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORD5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]     # ordinal, 5 steps, starts no lighter than step 250
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SHOW_DENSITY = False
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150})


def tex_table(df: pd.DataFrame, path: Path, caption: str, label: str, fmt=None, col_fmt=None):
    cols = list(df.columns)
    col_fmt = col_fmt or ("l" + "r" * (len(cols) - 1))
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule", " & ".join(str(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append((fmt or {}).get(c, "{:.2f}").format(v) if np.isfinite(v) else "--")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells).replace("%", "\\%").replace("_", "\\_") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def md_table(df: pd.DataFrame, fmt=None) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(((fmt or {}).get(c, "{:.2f}").format(v) if np.isfinite(v) else "--") if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def cdf(ax, x, color, label, lw=1.8, ls="-"):
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    if len(x) == 0:
        return
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, lw=lw, ls=ls, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    tdir, fdir = args.out / "tables", args.out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    boxes = pd.read_csv(tdir / "distortion_boxes.csv")
    tracks = pd.read_csv(tdir / "distortion_tracks.csv")
    flights = pd.read_csv(tdir / "distortion_flights.csv")
    calibs = pd.read_csv(tdir / "calibrations.csv")
    undist = pd.read_csv(tdir / "undistortion.csv")
    model = json.load(open(tdir / "distortion_model.json"))
    dsum = json.load(open(tdir / "distortion_summary.json"))
    tb = pd.read_csv(tdir / "timing_boxes.csv")
    trec = pd.read_csv(tdir / "timing_recordings.csv")
    tsum = json.load(open(tdir / "timing_summary.json"))
    trec_ok = trec[(trec.status == "ok") & trec.verdict.isin(["reproduced", "release follows SRT", "not released"])].copy()
    md = []

    # ------------------------------------------------------------------ figures: one panel per file (paper subfigures)
    def save(fig, name):
        fig.savefig(fdir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(fdir / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    PANEL = (3.4, 2.6)
    # (1) radial distortion profiles
    fig, ax = plt.subplots(figsize=PANEL)
    n_t = sum(1 for p in model["profiles"].values() if p["cam"] == "T")
    n_w = sum(1 for p in model["profiles"].values() if p["cam"] == "W")
    for k, p in model["profiles"].items():
        ax.plot(np.array(p["r"]) * 100, np.array(p["frac"]) * 100, color=BLUE if p["cam"] == "T" else ORANGE, lw=1.4, alpha=0.9)
    ax.plot([], [], color=BLUE, lw=1.4, label=f"Thermal ({n_t} calibrations)")
    ax.plot([], [], color=ORANGE, lw=1.4, label=f"RGB ({n_w} calibrations)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Distance from the optical centre [% of half-diagonal]")
    ax.set_ylabel("Radial displacement of the raw pixel [%]")
    ax.legend(loc="lower left")
    save(fig, "lens_profile")

    # (2) ground-offset field over the frame
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    key = next(k for k in model["fields"] if k.startswith("T_"))
    field = np.array(model["fields"][key])
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ex = np.linspace(0, C.W, field.shape[1] + 1); ey = np.linspace(0, C.H, field.shape[0] + 1)
    im = ax.pcolormesh(ex, ey, field * 100, cmap=cmap, vmin=0, vmax=max(5, np.ceil(field.max() * 100)), shading="flat", rasterized=False, linewidth=0, antialiased=False)
    ax.set_xlim(0, C.W); ax.set_ylim(C.H, 0); ax.set_aspect("equal")
    xs = np.linspace(0, C.W, field.shape[1]); ys = np.linspace(0, C.H, field.shape[0])
    cs = ax.contour(xs, ys, field * 100, levels=[1, 2, 3, 4, 5, 6], colors=INK, linewidths=0.5)
    ax.clabel(cs, fmt="%g", fontsize=6.5)
    if SHOW_DENSITY:   # kernel density of the annotated box centres; off by default, the shares by image position are in the tables
        from scipy.ndimage import gaussian_filter
        H2, xe, ye = np.histogram2d(boxes.u, boxes.v, bins=64, range=[[0, C.W], [0, C.H]])
        H2 = gaussian_filter(H2.T, 2.0); H2 = H2 / H2.max()
        ax.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, H2, levels=[0.25, 0.5, 0.75], colors=ORANGE, linewidths=0.9)
        ax.plot([], [], color=ORANGE, lw=0.9, label="Density of annotated animals")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_xticks([0, 512, 1024]); ax.set_yticks([0, 512, 1024]); ax.grid(False)
    ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.solids.set_rasterized(False)
    cb.set_label("Ground offset [cm per m of height]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, "lens_field")

    # (3) distortion offset per box, by height above ground
    ok = boxes[np.isfinite(boxes.offset)]
    bins = C.alt_bin(ok.agl.values)
    fig, ax = plt.subplots(figsize=PANEL)
    for i, lab in enumerate(C.ALT_LABELS):
        sel = bins == i
        if sel.sum():
            label = lab.replace(">= ", "$\\geq$ ")
            cdf(ax, ok.offset[sel], ORD5[i], f"{label} m (n = {sel.sum():,})", lw=1.5)
    ax.axvline(C.MATCH_RADIUS, color=MUTED, lw=0.9, ls="--")
    ax.text(C.MATCH_RADIUS + 0.06, 0.03, "Matching radius", color=INK2, fontsize=7, rotation=90, va="bottom")
    ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    ax.set_xlabel("Ground offset without undistortion [m]")
    ax.set_ylabel("Share of annotated boxes")
    ax.legend(title="Height above ground", fontsize=6.5, title_fontsize=7, loc="lower right")
    save(fig, "boxes_by_height")

    # (4) the three effects
    fig, ax = plt.subplots(figsize=PANEL)
    cdf(ax, ok.offset, BLUE, "Lens distortion ignored")
    cdf(ax, tb.err_nosrt, ORANGE, "Frame times from the flight log alone")
    cdf(ax, tb.err_srtonly, AQUA, "Positions from the SRT alone")
    cdf(ax, tracks.rms_radius, BLUE, "Distortion: apparent motion along a track", lw=1.2, ls=":")
    ax.axvline(C.MATCH_RADIUS, color=MUTED, lw=0.9, ls="--")
    ax.text(C.MATCH_RADIUS + 0.06, 0.03, "Matching radius", color=INK2, fontsize=7, rotation=90, va="bottom")
    ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    ax.set_xlabel("Ground error at the annotated animal [m]")
    ax.set_ylabel("Share of annotated boxes")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    save(fig, "boxes_effects")

    # (5) onset lag per recording
    fig, ax = plt.subplots(figsize=PANEL)
    lag = trec_ok.onset_lag_s.values
    shown = lag[(lag > -2) & (lag < 5)]
    ax.hist(shown, bins=np.arange(-2, 5.01, 0.2), color=ORANGE, edgecolor="white", linewidth=0.6)
    n_out = len(lag) - len(shown)
    if n_out:
        ax.text(0.98, 0.95, f"{n_out} recording{'s' if n_out != 1 else ''} outside the axis ({lag.min():+.0f} s)", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
    ax.set_xlabel("Onset of the isVideo flag after the first frame [s]")
    ax.set_ylabel("Recordings")
    save(fig, "onset_lag")

    # (6) cost of the onset lag per recording
    fig, ax = plt.subplots(figsize=PANEL)
    y = trec_ok.nosrt_mean.values
    ax.scatter(trec_ok.median_speed, np.minimum(y, 9.8), s=12, color=ORANGE, alpha=0.75, linewidths=0)
    ax.axhline(C.MATCH_RADIUS, color=MUTED, lw=0.9, ls="--")
    ax.text(0.05, C.MATCH_RADIUS + 0.2, "Matching radius", fontsize=7, color=INK2)
    ax.set_ylim(0, 10)
    n_out = int((y > 9.8).sum())
    if n_out:
        ax.text(0.98, 0.95, f"{n_out} recording{'s' if n_out != 1 else ''} above the axis (max {y.max():.0f} m)", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
    ax.set_xlabel("Median ground speed of the recording [m/s]")
    ax.set_ylabel("Mean pose error without the SRT [m]")
    save(fig, "onset_cost")

    # ------------------------------------------------------------------ tables
    # T1 calibrations
    t1 = calibs[["camera", "raw_w", "raw_h", "fx", "fy", "k1", "k2", "k3", "edge_disp_pct", "corner_disp_pct", "n_flights"]].copy()
    t1.columns = ["camera", "width", "height", "$f_x$", "$f_y$", "$k_1$", "$k_2$", "$k_3$", "edge [\\%]", "corner [\\%]", "flights"]
    tex_table(t1, tdir / "calibrations.tex", "Lens calibrations shipped with the raw dataset (one row per distinct calibration). "
              "Edge/corner: displacement of the raw pixel from its ideal pinhole position at the middle of the long edge and at the corner, "
              "in percent of its distance from the optical centre.", "tab:calib",
              fmt={"$f_x$": "{:.0f}", "$f_y$": "{:.0f}", "$k_1$": "{:.3f}", "$k_2$": "{:.3f}", "$k_3$": "{:.3f}"})
    md.append("## Calibrations\n\n" + md_table(t1.rename(columns=lambda c: c.replace("$", "").replace("\\%", "%")),
                                                fmt={"f_x": "{:.0f}", "f_y": "{:.0f}", "k_1": "{:.3f}", "k_2": "{:.3f}", "k_3": "{:.3f}"}))
    t1b = undist[["camera", "f_px", "fovy_deg", "valid_share", "mask_agreement", "scale_error_pct", "n_flights"]].copy()
    t1b.columns = ["camera", "focal length [px]", "fovy [deg]", "valid pixels", "mask agreement", "50 deg assumption off by [%]", "flights"]
    md.append("## Undistortion recovered from the masks\n\n" + md_table(t1b, fmt={"focal length [px]": "{:.0f}", "valid pixels": "{:.3f}", "mask agreement": "{:.4f}"}))

    # T2 distortion offsets by altitude bin, radius, species
    rows = []
    for i, lab in enumerate(C.ALT_LABELS):
        sel = bins == i
        o = ok.offset[sel]
        rows.append(dict(bin=f"{lab} m", flights=ok.flight[sel].nunique(), boxes=int(sel.sum()), median=C.pct(o, 50), p95=C.pct(o, 95),
                         over_half=float(np.mean(o > C.MATCH_RADIUS / 2)) * 100 if sel.any() else np.nan,
                         over_match=float(np.mean(o > C.MATCH_RADIUS)) * 100 if sel.any() else np.nan))
    o = ok.offset
    rows.append(dict(bin="all", flights=ok.flight.nunique(), boxes=len(ok), median=C.pct(o, 50), p95=C.pct(o, 95),
                     over_half=float(np.mean(o > C.MATCH_RADIUS / 2)) * 100, over_match=float(np.mean(o > C.MATCH_RADIUS)) * 100))
    t2 = pd.DataFrame(rows)
    t2.columns = ["height above ground", "flights", "boxes", "median [m]", "p95 [m]", "> 0.69 m [\\%]", "> 1.37 m [\\%]"]
    tex_table(t2, tdir / "distortion_by_altitude.tex", "Ground offset of the annotated thermal boxes if the lens distortion were ignored, "
              "by height above ground (the altitude bins of Table~3). Flat-ground nadir model with the flight's calibration and the "
              "undistortion recovered from the mask; height from the flight log.", "tab:distortion",
              fmt={"median [m]": "{:.2f}", "p95 [m]": "{:.2f}", "> 0.69 m [\\%]": "{:.0f}", "> 1.37 m [\\%]": "{:.0f}"})
    md.append("## Distortion offset by height above ground\n\n" + md_table(t2.rename(columns=lambda c: c.replace("\\%", "%")),
                                                                            fmt={"median [m]": "{:.2f}", "p95 [m]": "{:.2f}", "> 0.69 m [%]": "{:.0f}", "> 1.37 m [%]": "{:.0f}"}))
    rows = []
    for lo, hi, lab in ((0, 0.25, "0-25 %"), (0.25, 0.5, "25-50 %"), (0.5, 0.75, "50-75 %"), (0.75, 1.0, "75-100 %"), (1.0, 2.0, "corners (> 100 %)")):
        sel = (ok.r_frac >= lo) & (ok.r_frac < hi)
        o = ok.offset[sel]
        rows.append(dict(radius=lab, boxes=int(sel.sum()), share=float(sel.mean()) * 100, median=C.pct(o, 50), p95=C.pct(o, 95),
                         over_match=float(np.mean(o > C.MATCH_RADIUS)) * 100 if sel.any() else np.nan))
    t2b = pd.DataFrame(rows); t2b.columns = ["distance from the frame centre", "boxes", "share [%]", "median [m]", "p95 [m]", "> 1.37 m [%]"]
    md.append("## Distortion offset by position in the frame\n\n" + md_table(t2b, fmt={"share [%]": "{:.0f}", "median [m]": "{:.2f}", "p95 [m]": "{:.2f}", "> 1.37 m [%]": "{:.0f}"}))
    sp = ok.groupby("species").offset.agg(["size", "median", lambda x: np.percentile(x, 95), lambda x: np.mean(x > C.MATCH_RADIUS) * 100]).reset_index()
    sp.columns = ["species", "boxes", "median [m]", "p95 [m]", "> 1.37 m [%]"]
    sp = sp.sort_values("boxes", ascending=False).head(10)
    md.append("## Distortion offset by species (ten most annotated)\n\n" + md_table(sp, fmt={"> 1.37 m [%]": "{:.0f}"}))

    # T3 timing: per recording and per box
    def row(name, x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return dict(quantity=name, n=len(x), min=float(x.min()), median=float(np.median(x)), p95=C.pct(x, 95), max=float(x.max()))
    t3 = pd.DataFrame([row("isVideo onset lag [s]", trec_ok.onset_lag_s), row("SRT to AirData clock offset [s]", trec_ok.clock_offset_s),
                       row("true frame rate [fps]", trec_ok.fps_true), row("AirData log rate [Hz]", trec_ok.airdata_hz),
                       row("largest AirData gap [s]", trec_ok.max_gap_s), row("median ground speed [m/s]", trec_ok.median_speed),
                       row("pose error, AirData only, mean per recording [m]", trec_ok.nosrt_mean),
                       row("pose error, SRT only, mean per recording [m]", trec_ok.srtonly_mean),
                       row("AirData interpolation at 1 Hz, p95 in turns [m]", trec_ok.interp1hz_turn_p95),
                       row("AirData interpolation at 1 Hz, p95 straight [m]", trec_ok.interp1hz_straight_p95),
                       row("|gimbal yaw SRT - AirData| [deg]", np.abs(trec_ok.yaw_diff_median_deg)),
                       row("RGB-thermal frame pairing |dt| [ms]", trec_ok.rgb_pair_dt_median_ms),
                       row("reproduction of the published poses [m]", trec_ok.repro_median_m)])
    tex_table(t3, tdir / "timing_recordings.tex", f"Camera-pose timing over the {len(trec_ok)} recordings of the {trec_ok.flight.nunique()} flights "
              "with public raw logs: distribution over recordings.", "tab:timing", fmt={"n": "{:d}"})
    md.append("## Timing, distribution over recordings\n\n" + md_table(t3, fmt={"min": "{:.3f}", "median": "{:.3f}", "p95": "{:.3f}", "max": "{:.3f}"}))
    rows = []
    for name, col in (("lens distortion ignored", ok.offset), ("frame times from the flight log alone (30 fps from the isVideo onset)", tb.err_nosrt),
                      ("same with the recording's true frame rate", tb.err_nosrt_fixedfps), ("pose position from the SRT alone", tb.err_srtonly),
                      ("50 deg field of view assumed instead of the frames' own", ok.offset_fov),
                      ("distortion: apparent motion along a track (RMS radius)", tracks.rms_radius),
                      ("distortion: apparent motion along a track (extent)", tracks.extent)):
        x = np.asarray(col, float); x = x[np.isfinite(x)]
        rows.append(dict(effect=name, n=len(x), median=float(np.median(x)), mean=float(x.mean()), p95=C.pct(x, 95),
                         over_half=float(np.mean(x > C.MATCH_RADIUS / 2)) * 100, over_match=float(np.mean(x > C.MATCH_RADIUS)) * 100))
    t4 = pd.DataFrame(rows); t4.columns = ["effect", "n", "median [m]", "mean [m]", "p95 [m]", "> 0.69 m [\\%]", "> 1.37 m [\\%]"]
    tex_table(t4, tdir / "error_budget.tex", "Ground error at the annotated animals (thermal boxes) that each step of the pipeline removes, "
              "and the share of boxes it would move by more than half and more than the full matching radius of the detection protocol.",
              "tab:budget", fmt={"> 0.69 m [\\%]": "{:.0f}", "> 1.37 m [\\%]": "{:.0f}"}, col_fmt="lrrrrrr")
    md.append("## Error budget at the annotated animals\n\n" + md_table(t4.rename(columns=lambda c: c.replace("\\%", "%")),
                                                                          fmt={"> 0.69 m [%]": "{:.0f}", "> 1.37 m [%]": "{:.0f}"}))

    # headline numbers
    head = dict(flights_total=int(flights.shape[0]), flights_with_calibration=int((flights.calib_t != "").sum() if flights.calib_t.dtype == object else flights.calib_t.notna().sum()),
                boxes_total=int(flights.n_boxes.sum()), boxes_with_height=int(len(ok)), boxes_in_border=float(1 - boxes.in_mask.mean()) * 100,
                median_tilt_deg=float(flights.median_tilt.median()), agl_median=float(ok.agl.median()), agl_p5=C.pct(ok.agl, 5), agl_p95=C.pct(ok.agl, 95),
                r_frac_median=float(ok.r_frac.median()), timing_flights=int(trec_ok.flight.nunique()), timing_recordings=int(len(trec_ok)),
                timing_boxes=int(len(tb)), share_lag_negative=float(np.mean(trec_ok.onset_lag_s < 0)) * 100,
                recordings_with_gap=int((trec_ok.gaps_over_2s > 0).sum()), recordings_10hz=int((trec_ok.airdata_hz >= 9).sum()),
                tracks=int(len(tracks)), track_extent_median=float(tracks.extent.median()), track_extent_p95=C.pct(tracks.extent, 95),
                track_rms_median=float(tracks.rms_radius.median()), track_rms_p95=C.pct(tracks.rms_radius, 95),
                flights_calib_own=int((flights.calib_source == "own").sum()), flights_calib_inferred=int((flights.calib_source == "inferred").sum()),
                recordings_verdicts=trec.verdict.value_counts().to_dict())
    json.dump(head, open(tdir / "headline.json", "w"), indent=1)
    md.insert(0, "# Results (generated by report.py)\n\n```\n" + json.dumps(head, indent=1) + "\n```")
    (args.out / "results.md").write_text("\n\n".join(md) + "\n")
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
