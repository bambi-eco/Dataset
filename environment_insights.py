"""
What the environment layers say about the animals, and vice versa.

Reads the tables written by ``environment_features.py`` and answers a set of
questions that need both the animal boxes and the environment masks:

1. Does occlusion (``visibility`` 0.5 against 1.0) track canopy? Rate of
   half-visible boxes against the canopy fraction inside the box, against
   the distance to the nearest canopy, pooled and within flights.
2. What is each species standing on, and in? Enrichment of every class under
   the boxes and in the frames of each species.
3. How close do the animals come to roads, roofs, vehicles and water?
4. Group size against canopy openness.
5. Juveniles and adults, males and females, against canopy.
6. Where tracks end: is the last key frame of a track more often under
   canopy than the rest of it?
7. Season and snow.

Usage::

    python environment_insights.py features/ --figures figures/ --report report.md

Statistics are deliberately plain. Rates come with a 95% cluster bootstrap
over flights, because boxes of one flight are not independent samples;
regressions use cluster-robust standard errors by flight. All environment
labels are machine-generated and unreviewed (docs/environment.md), and the
boxes are annotated on the thermal view while the masks are computed on the
RGB view, so a per-box fraction is blurred by roughly 16 px of registration
error. Read every number with both in mind.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

CLASSES = ["snow", "water", "road", "grass", "rock", "bare_ground", "roof",
           "vehicle", "tree_cover", "deadwood"]
PRETTY = {c: c.replace("_", " ") for c in CLASSES}
SHORT_SPECIES = {
    "Sus scrofa (Wild boar)": "Wild boar",
    "Cervus elaphus (Red deer)": "Red deer",
    "Capreolus capreolus (Roe deer)": "Roe deer",
    "Dama dama (Fallow deer)": "Fallow deer",
    "Capra ibex (Alpine ibex)": "Alpine ibex",
    "Rupicapra rupicapra (Chamois)": "Chamois",
    "Aves (Bird)": "Bird",
    "Homo sapiens (Human)": "Human",
    "Canis lupus familiaris (Dog)": "Dog",
    "Sus scrofa x Sus domesticus (Hybrid pig)": "Hybrid pig",
    "No-animal": "No-animal",
    "Unknown": "Unknown",
}
MAIN_SPECIES = ["Wild boar", "Red deer", "Roe deer", "Fallow deer",
                "Alpine ibex", "Chamois"]
CANOPY_BINS = [-0.001, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0]
CANOPY_LABELS = ["0", "0–25%", "25–50%", "50–75%", "75–<100%", "100%"]
DIST_BINS = [-0.001, 0.0, 16, 48, 128, 1e9]
DIST_LABELS = ["under canopy", "<16 px", "16–48 px", "48–128 px", ">128 px"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_SHORT_LOWER = {k.lower(): v for k, v in SHORT_SPECIES.items()}


def short(name: str) -> str:
    """'Dama dama (Fallow Deer)' and 'Dama dama (Fallow deer)' both occur."""
    return _SHORT_LOWER.get(str(name).strip().lower(), name)


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, n: int = 1000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Mean of `values` with a 95% CI from resampling whole clusters."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    ok = ~np.isnan(values)
    values, clusters = values[ok], clusters[ok]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    ids, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=ids.size)
    cnts = np.bincount(inv, minlength=ids.size).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, ids.size, size=(n, ids.size))
    means = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


INT_COLUMNS = {"boxes", "flights", "frames", "tracks", "month", "n", "flights with ρ<0"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if math.isnan(v):
                    cells.append("")
                elif v.is_integer() and c in INT_COLUMNS:
                    cells.append(str(int(v)))
                else:
                    cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rate_table(df: pd.DataFrame, by: str, order=None) -> pd.DataFrame:
    rows = []
    groups = df.groupby(by, observed=True)
    keys = order if order is not None else list(groups.groups)
    for k in keys:
        if k not in groups.groups:
            continue
        g = groups.get_group(k)
        m, lo, hi = cluster_bootstrap(g["occluded"].values, g["flight"].values)
        rows.append({by: k, "boxes": len(g), "flights": g["flight"].nunique(),
                     "occluded": m, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(feat: Path):
    boxes = pd.read_csv(feat / "boxes.csv")
    frames = pd.read_csv(feat / "frames.csv")
    flights = pd.read_csv(feat / "flights.csv")
    boxes["species_short"] = boxes["species"].map(short)
    boxes["occluded"] = (boxes["visibility"] < 1.0).astype(float)
    flights["start_time"] = pd.to_datetime(flights["start_time"], errors="coerce")
    flights["month"] = flights["start_time"].dt.month
    flights["hour"] = flights["start_time"].dt.hour
    boxes = boxes.merge(flights[["flight", "month", "hour", "gsd_cm", "has_nc_layer", "split"]],
                        on="flight", how="left")
    frames = frames.merge(flights[["flight", "month", "hour", "gsd_cm", "has_nc_layer"]],
                          on="flight", how="left")
    return boxes, frames, flights


def usable_boxes(boxes: pd.DataFrame) -> pd.DataFrame:
    """Boxes the environment layers say something about: inside the imaged
    area, on a frame that is not undetermined, on a flight with the canopy
    layer."""
    b = boxes[(boxes["in_imaged_area"] == 1) & (boxes["undetermined"] == 0)
              & (boxes["has_nc_layer"] == 1)].copy()
    b["canopy_bin"] = pd.cut(b["tree_cover_box"], CANOPY_BINS, labels=CANOPY_LABELS)
    d = b["tree_cover_dist"].replace(np.inf, 1e8)
    b["dist_bin"] = pd.cut(d, DIST_BINS, labels=DIST_LABELS)
    # a box with canopy inside it sits "under canopy" whatever its distance
    b.loc[b["tree_cover_box"] > 0, "dist_bin"] = "under canopy"
    return b


# --------------------------------------------------------------------------
# 1. occlusion and canopy
# --------------------------------------------------------------------------
def occlusion_analysis(b: pd.DataFrame, report: list[str], figdir: Path,
                       frames_for_structure: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report.append("## 1. Occlusion and canopy\n")
    report.append(f"Usable boxes: {len(b)} on {b['flight'].nunique()} flights "
                  f"({int(b['occluded'].sum())} half-visible, "
                  f"{100 * b['occluded'].mean():.1f}%).\n")

    # 1a: rate by canopy fraction inside the box
    t = rate_table(b, "canopy_bin", CANOPY_LABELS)
    t = t.rename(columns={"canopy_bin": "canopy in box"})
    report.append("### Half-visible rate against canopy fraction inside the box\n")
    report.append(md_table(t) + "\n")

    # 1b: by distance to canopy
    t2 = rate_table(b, "dist_bin", DIST_LABELS).rename(columns={"dist_bin": "nearest canopy"})
    report.append("### Half-visible rate against distance from the box centre to the nearest canopy\n")
    report.append(md_table(t2) + "\n")

    # 1c: by species x canopy
    rows = []
    for sp in MAIN_SPECIES:
        g = b[b["species_short"] == sp]
        if len(g) < 200:
            continue
        for lab in ["0", "0–25%", "25–50%", "50–75%", "75–<100%", "100%"]:
            gg = g[g["canopy_bin"] == lab]
            if len(gg) < 20:
                continue
            m, lo, hi = cluster_bootstrap(gg["occluded"].values, gg["flight"].values)
            rows.append({"species": sp, "canopy in box": lab, "boxes": len(gg),
                         "occluded": m, "ci_low": lo, "ci_high": hi})
    sp_tab = pd.DataFrame(rows)
    report.append("### By species\n")
    report.append(md_table(sp_tab) + "\n")

    # 1d: within-flight comparison. For each flight with both kinds of box,
    # the mean canopy fraction under occluded minus under visible boxes.
    diffs = []
    for fl, g in b.groupby("flight"):
        occ = g[g["occluded"] == 1]["tree_cover_box"]
        vis = g[g["occluded"] == 0]["tree_cover_box"]
        if len(occ) >= 5 and len(vis) >= 5:
            diffs.append({"flight": fl, "n_occ": len(occ), "n_vis": len(vis),
                          "diff": occ.mean() - vis.mean(),
                          "occ_mean": occ.mean(), "vis_mean": vis.mean()})
    diffs = pd.DataFrame(diffs)
    if len(diffs):
        w = stats.wilcoxon(diffs["diff"])
        pos = (diffs["diff"] > 0).sum()
        report.append("### Within flights\n")
        report.append(
            f"On {len(diffs)} flights with at least five boxes of each kind, the mean canopy "
            f"fraction under half-visible boxes exceeds that under fully visible boxes on "
            f"{pos} flights ({100 * pos / len(diffs):.0f}%). Median difference "
            f"{diffs['diff'].median():+.3f} (IQR {diffs['diff'].quantile(.25):+.3f} to "
            f"{diffs['diff'].quantile(.75):+.3f}); Wilcoxon signed-rank p = {w.pvalue:.2g}.\n")

    # 1e: logistic regression with flight-clustered SEs
    try:
        import statsmodels.formula.api as smf
        d = b.copy()
        d["canopy"] = d["tree_cover_box"]
        d["log_dist"] = np.log1p(d["tree_cover_dist"].replace(np.inf, 1024).clip(upper=1024))
        d["sp"] = d["species_short"].where(d["species_short"].isin(MAIN_SPECIES), "other")
        d["window_canopy"] = d["tree_cover_win"]
        d["box_area"] = np.log(d["w"].clip(lower=1) * d["h"].clip(lower=1))
        m1 = smf.logit("occluded ~ canopy + C(sp)", d).fit(disp=0,
                                                          cov_type="cluster",
                                                          cov_kwds={"groups": d["flight"]})
        m2 = smf.logit("occluded ~ canopy + window_canopy + deadwood_box + snow_box + box_area + C(sp)", d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": d["flight"]})
        report.append("### Logistic regression, cluster-robust by flight\n")
        report.append("Model 1: `occluded ~ canopy_in_box + species`, where canopy_in_box is the "
                      "fraction of the box under the tree cover mask\n")
        report.append(f"- canopy in box: odds ratio {math.exp(m1.params['canopy']):.2f} "
                      f"(95% CI {math.exp(m1.conf_int().loc['canopy', 0]):.2f}–"
                      f"{math.exp(m1.conf_int().loc['canopy', 1]):.2f}), p = {m1.pvalues['canopy']:.2g}; "
                      f"McFadden pseudo-R² {m1.prsquared:.3f}\n")
        report.append("Model 2: adds `window_canopy`, the canopy fraction of the 256 px window around "
                      "the box centre, plus deadwood and snow in the box and log box area\n")
        for k in ["canopy", "window_canopy", "deadwood_box", "snow_box", "box_area"]:
            ci = m2.conf_int().loc[k]
            report.append(f"- {k}: OR {math.exp(m2.params[k]):.2f} "
                          f"({math.exp(ci[0]):.2f}–{math.exp(ci[1]):.2f}), p = {m2.pvalues[k]:.2g}")
        report.append(f"- McFadden pseudo-R² {m2.prsquared:.3f}\n")
    except Exception as e:  # statsmodels missing or a singular fit
        report.append(f"(logistic regression skipped: {e})\n")

    # 1f: canopy fragmentation given cover
    g = b[(b["tree_cover_box"] > 0)]
    if len(g) > 100:
        rho = stats.spearmanr(g["occluded"], g["tree_cover_ring"])
        report.append(f"Among boxes that touch canopy, Spearman ρ between half-visibility and canopy "
                      f"fraction in the 32 px ring around the box: {rho.statistic:.3f} (p = {rho.pvalue:.2g}).\n")

    # 1h: canopy structure at similar cover. Among frames whose canopy cover
    # is intermediate, does a more broken-up canopy (more edge per area) change
    # the rate for boxes that touch canopy?
    mid = b[(b["tree_cover_box"] > 0)].merge(
        frames_for_structure[["flight", "frame", "tree_cover_cov", "tree_edge_density", "tree_blobs"]],
        on=["flight", "frame"], how="left")
    mid = mid[(mid["tree_cover_cov"] >= 0.2) & (mid["tree_cover_cov"] <= 0.8)]
    if len(mid) > 300:
        q = mid["tree_edge_density"].quantile([1 / 3, 2 / 3]).values
        mid["struct"] = pd.cut(mid["tree_edge_density"], [-1, q[0], q[1], 1e9],
                               labels=["solid (low edge density)", "middle", "broken up (high edge density)"])
        t5 = rate_table(mid, "struct", ["solid (low edge density)", "middle", "broken up (high edge density)"]).rename(
            columns={"struct": "canopy structure of the frame"})
        t5["mean canopy cover"] = [mid[mid["struct"] == k]["tree_cover_cov"].mean() for k in t5["canopy structure of the frame"]]
        report.append("### Canopy structure at similar cover\n")
        report.append("Boxes touching canopy on frames with 20–80% canopy cover, split into tertiles of "
                      "canopy edge density (mask perimeter per imaged area): a solid block of canopy "
                      "against a canopy broken into many small crowns and gaps.\n")
        report.append(md_table(t5) + "\n")

    # 1g: what explains the half-visible boxes that have no canopy at all?
    # The thermal frame edge, and canopy just outside the box.
    z = b[b["tree_cover_box"] == 0].copy()
    if len(z) > 100:
        edge = np.minimum.reduce([z["x"], z["y"], 1024 - (z["x"] + z["w"]), 1024 - (z["y"] + z["h"])])
        z["edge_bin"] = pd.cut(edge, [-1e9, 0.5, 8, 32, 1e9], labels=["touches edge", "<8 px", "8–32 px", ">32 px"])
        t3 = rate_table(z, "edge_bin", ["touches edge", "<8 px", "8–32 px", ">32 px"]).rename(
            columns={"edge_bin": "distance to thermal frame edge"})
        report.append("### Half-visible boxes without any canopy\n")
        report.append("Boxes with no canopy inside them, by the distance of the box to the edge of the "
                      "thermal frame. An animal cut off by the frame border is annotated as half visible too.\n")
        report.append(md_table(t3) + "\n")
        z["win_bin"] = pd.cut(z["tree_cover_win"], [-0.001, 0.0, 0.1, 0.3, 0.6, 1.0],
                              labels=["0", "0–10%", "10–30%", "30–60%", ">60%"])
        zz = z[edge > 32]
        t4 = rate_table(zz, "win_bin", ["0", "0–10%", "10–30%", "30–60%", ">60%"]).rename(
            columns={"win_bin": "canopy in 256 px window"})
        report.append("The same boxes, away from the frame edge, by canopy in the 256 px window around "
                      "them. Canopy near but not inside the box still raises the rate, which is what "
                      "about 16 px of thermal-to-RGB registration error and a 40 cm-resolution canopy "
                      "mask would produce.\n")
        report.append(md_table(t4) + "\n")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(t))
    ax.bar(x, 100 * t["occluded"], yerr=[100 * (t["occluded"] - t["ci_low"]), 100 * (t["ci_high"] - t["occluded"])],
           color="#4c72b0", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(t["canopy in box"], rotation=20)
    ax.set_xlabel("canopy fraction inside the box"); ax.set_ylabel("half-visible boxes (%)")
    ax.set_title("All species")
    for xi, n in zip(x, t["boxes"]):
        ax.text(xi, 1, f"n={n}", ha="center", va="bottom", fontsize=7, color="white")
    ax = axes[1]
    for sp in sp_tab["species"].unique():
        s = sp_tab[sp_tab["species"] == sp]
        xs = [CANOPY_LABELS.index(l) for l in s["canopy in box"]]
        ax.errorbar(xs, 100 * s["occluded"], yerr=[100 * (s["occluded"] - s["ci_low"]), 100 * (s["ci_high"] - s["occluded"])],
                    marker="o", capsize=2, label=sp)
    ax.set_xticks(range(len(CANOPY_LABELS))); ax.set_xticklabels(CANOPY_LABELS, rotation=20)
    ax.set_xlabel("canopy fraction inside the box"); ax.set_title("By species")
    ax.legend(fontsize=8)
    fig.savefig(figdir / "environment_occlusion_canopy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.arange(len(t2))
    ax.bar(x, 100 * t2["occluded"], yerr=[100 * (t2["occluded"] - t2["ci_low"]), 100 * (t2["ci_high"] - t2["occluded"])],
           color="#55a868", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(t2["nearest canopy"], rotation=20)
    ax.set_ylabel("half-visible boxes (%)"); ax.set_xlabel("distance from box centre to nearest canopy")
    for xi, n in zip(x, t2["boxes"]):
        ax.text(xi, 1, f"n={n}", ha="center", va="bottom", fontsize=7, color="white")
    fig.savefig(figdir / "environment_occlusion_distance.png", dpi=150)
    plt.close(fig)

    if len(diffs):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.hist(diffs["diff"], bins=25, color="#8172b2")
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("mean canopy fraction: half-visible minus fully visible boxes, per flight")
        ax.set_ylabel("flights")
        fig.savefig(figdir / "environment_occlusion_within_flight.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------
# 2. species habitat profiles
# --------------------------------------------------------------------------
def habitat_profiles(b: pd.DataFrame, frames: pd.DataFrame, report: list[str], figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report.append("## 2. What each species is standing on\n")
    # per flight-species means first so that a long flight does not dominate,
    # and the baseline is the frame coverage of the same flights, so that a
    # species recorded only in the mountains is compared with mountain frames
    box_cols = [f"{c}_box" for c in CLASSES]
    cov_cols = [f"{c}_cov" for c in CLASSES]
    fr = frames[(frames["undetermined"] == 0) & (frames["has_nc_layer"] == 1)]
    flight_cov = fr.groupby("flight")[cov_cols].mean()
    flight_cov.columns = CLASSES
    per_flight = b.groupby(["species_short", "flight"])[box_cols].mean()
    per_flight.columns = CLASSES
    per_flight = per_flight.join(flight_cov, rsuffix="_frame")
    counts = b.groupby("species_short").agg(boxes=("flight", "size"), flights=("flight", "nunique"))
    prof = per_flight.groupby("species_short")[CLASSES].mean().join(counts)
    base = per_flight.groupby("species_short")[[f"{c}_frame" for c in CLASSES]].mean()
    base.columns = CLASSES
    prof = prof[prof["flights"] >= 3].sort_values("boxes", ascending=False)
    base = base.loc[prof.index]
    diff = prof[CLASSES] - base[CLASSES]

    report.append("Mean fraction of the box covered by each class, averaged per flight first and "
                  "then over flights, so that every flight counts once. Species on fewer than three "
                  "flights are left out. The second table is the mean coverage of the key frames of "
                  "the same flights, and the third is the difference in percentage points: positive "
                  "means the animal is on the class more than the frames it was recorded in are "
                  "covered by it.\n")
    show = prof.reset_index().rename(columns={"species_short": "species"})
    report.append(md_table(show[["species", "boxes", "flights"] + CLASSES]) + "\n")
    report.append("Frame coverage of the same flights:\n")
    report.append(md_table(base.reset_index().rename(columns={"species_short": "species"})) + "\n")
    report.append("Difference, box minus frame, in percentage points:\n")
    report.append(md_table((100 * diff).reset_index().rename(columns={"species_short": "species"}), "{:+.1f}") + "\n")

    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(prof) + 1.8), constrained_layout=True)
    im = ax.imshow(100 * diff.values, cmap="RdBu_r", vmin=-25, vmax=25, aspect="auto")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels([PRETTY[c] for c in CLASSES], rotation=30, ha="right")
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([f"{s} ({int(n)} boxes, {int(f)} flights)" for s, n, f in zip(prof.index, prof["boxes"], prof["flights"])])
    for i in range(len(prof)):
        for j, c in enumerate(CLASSES):
            ax.text(j, i, f"{100 * prof.iloc[i][c]:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if abs(diff.iloc[i, j]) > 0.15 else "black")
    fig.colorbar(im, ax=ax, label="box minus frame coverage (percentage points)")
    ax.set_title("Under the box: mean fraction per class, coloured against the frames of the same flights")
    fig.savefig(figdir / "environment_species_habitat.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. distance to anthropogenic features and water
# --------------------------------------------------------------------------
def proximity(b: pd.DataFrame, report: list[str]) -> None:
    report.append("## 3. Roads, roofs, vehicles, water\n")
    report.append("Share of boxes with the class in the same frame, and within 128 px of the box centre "
                  "(about 4.5 m at the median GSD). A frame is roughly 36 m across, so 'in frame' means "
                  "within about 20 m. Boxes are only counted on flights where the class fires at least once, "
                  "which limits the sample to the terrain where the class exists at all.\n")
    rows = []
    for cls in ["road", "roof", "vehicle", "water", "snow", "deadwood"]:
        d = b[f"{cls}_dist"]
        for sp in MAIN_SPECIES + ["Human", "Dog"]:
            g = b[b["species_short"] == sp]
            if len(g) < 50:
                continue
            # flights where the class fires at least once
            fl_ok = g.groupby("flight")[f"{cls}_dist"].apply(lambda s: np.isfinite(s).any())
            g = g[g["flight"].isin(fl_ok[fl_ok].index)]
            if len(g) < 50:
                continue
            in_frame = np.isfinite(g[f"{cls}_dist"]).astype(float)
            near = (g[f"{cls}_dist"] <= 128).astype(float)
            m1, lo1, hi1 = cluster_bootstrap(in_frame.values, g["flight"].values)
            m2, lo2, hi2 = cluster_bootstrap(near.values, g["flight"].values)
            rows.append({"class": cls, "species": sp, "boxes": len(g), "flights": g["flight"].nunique(),
                         "in frame": m1, "in frame CI": f"{lo1:.2f}–{hi1:.2f}",
                         "within 128 px": m2, "within 128 px CI": f"{lo2:.2f}–{hi2:.2f}"})
    report.append(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")


# --------------------------------------------------------------------------
# 4. group size vs canopy
# --------------------------------------------------------------------------
def group_size(b: pd.DataFrame, frames: pd.DataFrame, report: list[str], figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report.append("## 4. Group size and canopy\n")
    fr = frames[(frames["undetermined"] == 0) & (frames["has_nc_layer"] == 1) & (frames["n_boxes"] > 0)].copy()
    fr["group"] = pd.cut(fr["n_boxes"], [0, 1, 2, 4, 8, 1000], labels=["1", "2", "3–4", "5–8", "9+"])
    rows = []
    for lab, g in fr.groupby("group", observed=True):
        m, lo, hi = cluster_bootstrap(g["tree_cover_cov"].values, g["flight"].values)
        rows.append({"animals in frame": lab, "frames": len(g), "flights": g["flight"].nunique(),
                     "mean canopy cover": m, "ci_low": lo, "ci_high": hi})
    t = pd.DataFrame(rows)
    report.append("Frame canopy cover against the number of annotated animals in the frame.\n")
    report.append(md_table(t) + "\n")
    # per species, spearman within flights
    rows = []
    for sp in MAIN_SPECIES:
        g = b[b["species_short"] == sp]
        if g["flight"].nunique() < 5:
            continue
        rhos = []
        for fl, gg in g.groupby("flight"):
            if gg["n_boxes_in_frame"].nunique() > 1 and len(gg) >= 20:
                r = stats.spearmanr(gg["n_boxes_in_frame"], gg["tree_cover_win"]).statistic
                if not math.isnan(r):
                    rhos.append(r)
        if len(rhos) >= 3:
            rows.append({"species": sp, "flights": len(rhos), "median within-flight ρ": float(np.median(rhos)),
                         "flights with ρ<0": int(sum(r < 0 for r in rhos))})
    report.append("Within each flight, Spearman correlation between the number of animals in the frame "
                  "and canopy in the 256 px window around each box; negative means larger groups in the open.\n")
    report.append(md_table(pd.DataFrame(rows)) + "\n")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.arange(len(t))
    ax.bar(x, 100 * t["mean canopy cover"], yerr=[100 * (t["mean canopy cover"] - t["ci_low"]),
                                                  100 * (t["ci_high"] - t["mean canopy cover"])],
           color="#dd8452", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(t["animals in frame"])
    ax.set_xlabel("annotated animals in the frame"); ax.set_ylabel("canopy cover of the frame (%)")
    fig.savefig(figdir / "environment_group_size.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. age and sex
# --------------------------------------------------------------------------
def age_sex(b: pd.DataFrame, report: list[str]) -> None:
    report.append("## 5. Juveniles, adults, males, females\n")
    # whether sex and age were annotated at all, against canopy
    d = b.copy()
    d["cb"] = pd.cut(d["tree_cover_box"], [-0.001, 0.0, 0.5, 0.999, 1.0],
                     labels=["0", "0–50%", "50–<100%", "100%"])
    rows = []
    for lab, g in d.groupby("cb", observed=True):
        gk, glo, ghi = cluster_bootstrap((g["gender"] > 0).values.astype(float), g["flight"].values)
        ak, alo, ahi = cluster_bootstrap((g["age"] > 0).values.astype(float), g["flight"].values)
        rows.append({"canopy in box": lab, "boxes": len(g), "sex annotated": gk, "sex CI": f"{glo:.2f}–{ghi:.2f}",
                     "age annotated": ak, "age CI": f"{alo:.2f}–{ahi:.2f}"})
    report.append("Whether sex and age could be annotated at all, against canopy in the box. "
                  "`gender` and `age` are 0 when unknown.\n")
    report.append(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")

    report.append("Canopy under each group, pooled over flights (species with at least 100 boxes "
                  "on three flights in the group).\n")
    rows = []
    for sp in MAIN_SPECIES:
        g = b[b["species_short"] == sp]
        for name, mask in [("juvenile", g["age"] == 1), ("adult", g["age"] == 2),
                           ("male", g["gender"] == 1), ("female", g["gender"] == 2)]:
            gg = g[mask]
            if len(gg) < 100 or gg["flight"].nunique() < 3:
                continue
            m, lo, hi = cluster_bootstrap(gg["tree_cover_box"].values, gg["flight"].values)
            o, olo, ohi = cluster_bootstrap(gg["occluded"].values, gg["flight"].values)
            rows.append({"species": sp, "group": name, "boxes": len(gg), "flights": gg["flight"].nunique(),
                         "canopy in box": m, "canopy CI": f"{lo:.2f}–{hi:.2f}",
                         "half-visible": o, "half-visible CI": f"{olo:.2f}–{ohi:.2f}"})
    report.append(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")

    # the same within flights, where both groups were recorded together
    rows = []
    for sp in MAIN_SPECIES:
        g = b[b["species_short"] == sp]
        for name, a_mask, b_mask in [("juvenile − adult", g["age"] == 1, g["age"] == 2),
                                     ("male − female", g["gender"] == 1, g["gender"] == 2)]:
            diffs = []
            for fl, gg in g.groupby("flight"):
                a = gg[a_mask.loc[gg.index]]["tree_cover_box"]
                c = gg[b_mask.loc[gg.index]]["tree_cover_box"]
                if len(a) >= 10 and len(c) >= 10:
                    diffs.append(a.mean() - c.mean())
            if len(diffs) >= 5:
                w = stats.wilcoxon(diffs)
                rows.append({"species": sp, "comparison": name, "flights": len(diffs),
                             "median difference in canopy": float(np.median(diffs)),
                             "flights with difference > 0": int(sum(x > 0 for x in diffs)),
                             "Wilcoxon p": w.pvalue})
    if rows:
        report.append("Within flights that recorded both groups (at least ten boxes each): "
                      "difference in mean canopy fraction under the box.\n")
        report.append(md_table(pd.DataFrame(rows), "{:.3f}") + "\n")


# --------------------------------------------------------------------------
# 6. where tracks end
# --------------------------------------------------------------------------
def track_ends(b: pd.DataFrame, report: list[str], figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report.append("## 6. Where tracks end\n")
    rows = []
    for (fl, tid), g in b.groupby(["flight", "track_id"]):
        if len(g) < 4:
            continue
        g = g.sort_values("frame")
        first, last, mid = g.iloc[0], g.iloc[-1], g.iloc[1:-1]
        rows.append({"flight": fl, "track": tid, "n": len(g), "species": g["species_short"].iloc[0],
                     "first_canopy": first["tree_cover_box"], "last_canopy": last["tree_cover_box"],
                     "mid_canopy": mid["tree_cover_box"].mean(),
                     "first_occ": first["occluded"], "last_occ": last["occluded"], "mid_occ": mid["occluded"].mean(),
                     "last_edge": last["dist_edge"], "first_edge": first["dist_edge"]})
    t = pd.DataFrame(rows)
    if len(t) == 0:
        report.append("(no tracks with four or more key frames)\n")
        return
    for what, a, c in [("canopy fraction", "last_canopy", "mid_canopy"), ("half-visible", "last_occ", "mid_occ")]:
        m1, lo1, hi1 = cluster_bootstrap(t[a].values, t["flight"].values)
        m2, lo2, hi2 = cluster_bootstrap(t[c].values, t["flight"].values)
        w = stats.wilcoxon(t[a] - t[c], zero_method="zsplit")
        report.append(f"- **{what}**, last key frame vs the middle of the track: "
                      f"{m1:.3f} ({lo1:.3f}–{hi1:.3f}) vs {m2:.3f} ({lo2:.3f}–{hi2:.3f}), "
                      f"Wilcoxon p = {w.pvalue:.2g}, {len(t)} tracks on {t['flight'].nunique()} flights")
    # how many tracks end at the frame edge vs under canopy
    edge = (t["last_edge"] < 40).mean()
    canopy_end = ((t["last_canopy"] >= 0.5) & (t["last_edge"] >= 40)).mean()
    report.append(f"- Track ends within 40 px of the imaged-area edge: {100 * edge:.1f}%; "
                  f"ends away from the edge with at least half the box under canopy: {100 * canopy_end:.1f}%\n")
    per = t.groupby("species").agg(tracks=("n", "size"), last_minus_mid=("last_canopy", "mean")).reset_index()
    per["last_minus_mid"] = t.groupby("species").apply(lambda g: (g["last_canopy"] - g["mid_canopy"]).mean()).values
    per = per[per["tracks"] >= 20]
    report.append(md_table(per) + "\n")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(t["last_canopy"] - t["mid_canopy"], bins=30, color="#64b5cd")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("canopy fraction in the last key frame minus track middle")
    ax.set_ylabel("tracks")
    fig.savefig(figdir / "environment_track_ends.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. season and snow
# --------------------------------------------------------------------------
def season(b: pd.DataFrame, frames: pd.DataFrame, report: list[str]) -> None:
    report.append("## 7. Season and snow\n")
    fr = frames[frames["undetermined"] == 0].copy()
    m = fr.groupby("flight").agg(month=("month", "first"), snow=("snow_cov", "mean"),
                                 canopy=("tree_cover_cov", "mean"), grass=("grass_cov", "mean")).reset_index()
    t = m.groupby("month").agg(flights=("flight", "size"), snow=("snow", "mean"),
                               canopy=("canopy", "mean"), grass=("grass", "mean")).reset_index()
    report.append("Mean frame coverage per flight, by month of recording.\n")
    report.append(md_table(t) + "\n")
    rows = []
    for sp in MAIN_SPECIES:
        g = b[b["species_short"] == sp]
        if g["flight"].nunique() < 3:
            continue
        on_snow = (g["snow_box"] > 0.5).astype(float)
        mm, lo, hi = cluster_bootstrap(on_snow.values, g["flight"].values)
        rows.append({"species": sp, "boxes": len(g), "flights": g["flight"].nunique(),
                     "months": ",".join(str(int(x)) for x in sorted(g["month"].dropna().unique())),
                     "box on snow": mm, "CI": f"{lo:.2f}–{hi:.2f}"})
    report.append("Share of boxes with more than half their area on snow.\n")
    report.append(md_table(pd.DataFrame(rows), "{:.2f}") + "\n")


# --------------------------------------------------------------------------
# 8. data quality
# --------------------------------------------------------------------------
def quality(boxes: pd.DataFrame, frames: pd.DataFrame, flights: pd.DataFrame, report: list[str]) -> None:
    report.append("## 8. Coverage of the join itself\n")
    n = len(boxes)
    lb = (boxes["in_imaged_area"] == 0).sum()
    und = ((boxes["in_imaged_area"] == 1) & (boxes["undetermined"] == 1)).sum()
    nc = (boxes["has_nc_layer"] == 0).sum()
    report.append(f"- {n} boxes on {boxes['flight'].nunique()} flights and {len(frames)} key frames.")
    report.append(f"- {lb} boxes ({100 * lb / n:.1f}%) have their centre in the RGB letterbox band, "
                  f"where no environment mask exists.")
    report.append(f"- {und} boxes ({100 * und / n:.1f}%) sit on frames flagged `undetermined`.")
    report.append(f"- {nc} boxes ({100 * nc / n:.1f}%) are on the seven flights without a canopy layer.")
    g = flights["gsd_cm"].dropna()
    report.append(f"- Estimated GSD: median {g.median():.2f} cm/px (p10 {g.quantile(.1):.2f}, "
                  f"p90 {g.quantile(.9):.2f}) over {len(g)} flights.")
    # occlusion on undetermined frames
    ub = boxes[boxes["undetermined"] == 1]
    if len(ub):
        report.append(f"- Half-visible rate on `undetermined` frames: {100 * ub['occluded'].mean():.1f}% "
                      f"against {100 * boxes[boxes['undetermined'] == 0]['occluded'].mean():.1f}% elsewhere.")
    report.append("")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("features", type=Path)
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--report", type=Path, default=Path("environment_insights.md"))
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    boxes, frames, flights = load(args.features)
    b = usable_boxes(boxes)
    report = ["# Environment layers against the animal annotations\n",
              f"Generated by `environment_insights.py` from {len(flights)} flights.\n"]
    occlusion_analysis(b, report, args.figures, frames)
    habitat_profiles(b, frames, report, args.figures)
    proximity(b, report)
    group_size(b, frames, report, args.figures)
    age_sex(b, report)
    track_ends(b, report, args.figures)
    season(b, frames, report)
    quality(boxes, frames, flights, report)
    args.report.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
