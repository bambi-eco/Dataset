#!/usr/bin/env python3
"""Render the BAMBI dataset teaser video.

Two steps, both idempotent:

    # 1. fetch the sample flights, the annotations of every flight (for the map)
    #    and the fonts, then pre-render the light-field sweep and the mosaic
    python teaser_video/make_video.py prepare --data bambi_video

    # 2. render the film (about 90 s, 1920x1080, H.264)
    python teaser_video/make_video.py render --data bambi_video -o bambi_teaser.mp4

    # quick look: half resolution, every scene's middle frame as a PNG
    python teaser_video/make_video.py stills --data bambi_video -o stills/

``--anonymous`` drops the repository links from the closing card, for a
double-blind submission.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from common import (FPS_DEFAULT, GREEN, HOT, MUTED, SKY, FG, H, W, VideoWriter, crossfade,  # noqa: E402
                    frame_plan, timeline)
from data import Flight  # noqa: E402
from scene_footage import Clip, FootageScene  # noqa: E402

# ── Storyboard ─────────────────────────────────────────────────────────────
# Flights shown as footage: (flight, first frame, note). Chosen for variety of
# species, season, habitat and drone.
CLIPS = [
    Clip("17", 19380, 5.0, 1.0, "Winter · snow · a herd of fallow deer"),
    Clip("6", 3100, 5.0, 1.0, "Summer · open enclosure"),
    Clip("142", 3380, 5.0, 1.5, "Spring · mixed wild boar group"),
    Clip("11", 4960, 5.0, 1.0, "Rock field · Alpine ibex"),
    Clip("143", 3400, 5.0, 1.0, "Autumn · feeding station"),
    Clip("31", 2470, 5.0, 1.0, "Dense forest · animals under the canopy"),
]
TITLE = ("17", 19240)                         # footage behind the title
ALFS = dict(flight="11", centre=12885, heights=(18, 62, 1.0), ground=55.0, half_window=75, step=3)
MOSAIC = dict(flight="142", frames=(3000, 4100, 5), ground=30.0, centre=(-10.0, 10.5), radius=36.0,
              size=900, anchor=3500, end=3880)
FLIGHTS = sorted({c.flight for c in CLIPS} | {TITLE[0], ALFS["flight"], MOSAIC["flight"]}, key=int)

FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf"


def _dirs(data: Path):
    return dict(flights=data / "flights", owl=data / "owl", ann=data / "annotations", cache=data / "cache")


def prepare(args):
    data = Path(args.data)
    d = _dirs(data)
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    font = HERE / "assets" / "Inter.ttf"
    if not font.exists():
        print("fetching Inter (SIL OFL)")
        urllib.request.urlretrieve(FONT_URL, font)
    dl = [sys.executable, str(REPO / "download_from_zenodo.py")]
    missing = [f for f in FLIGHTS if not (d["flights"] / f"{f}_matched_processed.mp4").exists()]
    if missing:
        print(f"downloading flights {missing} (about 1 GB each)")
        subprocess.run(dl + ["-f", *missing, "-o", str(d["flights"]), "--unzip"], check=True)
    if not all((d["owl"] / f"{f}_rgb_gt.txt").exists() for f in FLIGHTS):
        subprocess.run(dl + ["--version", "owl-transferred", "--annotations-only", "-f", *FLIGHTS,
                             "-o", str(d["owl"])], check=False)
    if not args.skip_map:
        print("annotations and poses of every flight, for the map (a few MB each)")
        subprocess.run(dl + ["--annotations-only", "-o", str(d["ann"])], check=False)

    import numpy as np
    from precompute import focal_sweep, ortho_strip

    alfs = d["cache"] / f"alfs_{ALFS['flight']}.npz"
    if not alfs.exists():
        print("light-field focal sweep")
        a, b, s = ALFS["heights"]
        focal_sweep(Flight(ALFS["flight"], d["flights"]), ALFS["centre"], list(np.arange(a, b + 1e-6, s)),
                    half_window=ALFS["half_window"], step=ALFS["step"], out=alfs, ground_agl=ALFS["ground"])
    mos = d["cache"] / f"ortho_{MOSAIC['flight']}.npz"
    if not mos.exists():
        print("orthomosaic")
        a, b, s = MOSAIC["frames"]
        ortho_strip(Flight(MOSAIC["flight"], d["flights"], d["owl"]), list(range(a, b + 1, s)), MOSAIC["ground"],
                    MOSAIC["centre"], MOSAIC["radius"], MOSAIC["size"], mos, anchor=MOSAIC["anchor"])
    print("prepared", data)


def build_scenes(args):
    import numpy as np
    from PIL import Image

    from scene_geo import AlfsScene, MosaicScene
    from scenes_info import (MapScene, OutroScene, SpeciesScene, StatsScene, TasksScene, TitleScene,
                             flight_dates, flight_locations)

    d = _dirs(Path(args.data))
    flights = {}

    def flight(key):
        if key not in flights:
            flights[key] = Flight(key, d["flights"], d["owl"])
        return flights[key]

    title = TitleScene(flight(TITLE[0]), TITLE[1])
    locs = flight_locations(d["ann"], REPO / "flight_metadata")
    mapscene = MapScene(HERE / "assets", locs, flight_dates(REPO / "flight_metadata"))
    th, rgb = flight("143").frame(3500)
    stats = StatsScene(Image.fromarray(rgb[:, :, ::-1].copy()))
    footage = [FootageScene(flight(c.flight), c, i, len(CLIPS)) for i, c in enumerate(CLIPS)]
    mosaic = MosaicScene(flight(MOSAIC["flight"]), d["cache"] / f"ortho_{MOSAIC['flight']}.npz",
                         end=MOSAIC["end"])
    alfs = AlfsScene(flight(ALFS["flight"]), d["cache"] / f"alfs_{ALFS['flight']}.npz",
                     ground_agl=ALFS["ground"], start_agl=24.0)
    species = SpeciesScene()

    # stills for the task cards, cut from the scenes above
    def cut(scene, t, box):
        return scene.render(t).crop(box)

    def pair(scene, t):
        # thermal and RGB of one moment, side by side
        img = scene.render(t)
        return Image.fromarray(np.hstack([np.asarray(img.crop((400, 420, 800, 760))),
                                          np.asarray(img.crop((1280, 420, 1680, 760)))]))

    cards = [
        ("Detection", "small objects: median box 39 × 39 px", cut(footage[0], 2.5, (160, 240, 900, 800)), HOT),
        ("Multi-object tracking", "5,100 tracks with key-frame interpolation",
         cut(footage[2], 3.0, (120, 176, 920, 976)), GREEN),
        ("Species · age · sex", "fine-grained labels per track", cut(footage[4], 2.5, (1000, 176, 1800, 976)),
         SKY),
        ("RGB ↔ thermal", "5,305 co-registered image pairs", pair(footage[1], 2.5), HOT),
        ("Geo-referenced analysis", "orthographic projection, world tracks",
         cut(mosaic, mosaic.duration - 0.5, (900, 150, 1800, 1050)), GREEN),
        ("Light-field sampling", "occlusion removal under canopy",
         cut(alfs, alfs.duration - 0.5, (1040, 230, 1800, 990)), SKY),
    ]
    tasks = TasksScene(cards)
    if args.anonymous:
        links = [("Data", "Zenodo · CC BY 4.0", HOT), ("Code", "toolkits for download, tracks, geo-processing", GREEN)]
        footer = ""
    else:
        links = [("Data", "zenodo.org · CC BY 4.0", HOT), ("Code", "github.com/bambi-eco/Dataset", GREEN),
                 ("Models", "huggingface.co/cpraschl/bambi-models", SKY)]
        footer = "bambi.eco"
    th, rgb = flight("6").frame(3300)
    outro = OutroScene(Image.fromarray(rgb[:, :, ::-1].copy()), links, footer=footer)
    return [title, mapscene, stats, *footage, species, mosaic, alfs, tasks, outro]


def render(args):
    scenes = build_scenes(args)
    only = set(args.only.split(",")) if args.only else None
    if only:
        scenes = [s for s in scenes if s.name in only or any(s.name.startswith(o) for o in only)]
    fps = args.fps
    starts, total = timeline(scenes, args.crossfade)
    print(f"{len(scenes)} scenes, {total:.1f} s at {fps} fps")
    size = (W // 2, H // 2) if args.preview else (W, H)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(out, fps, size=size, crf=args.crf)
    t0 = time.time()
    for k, (parts, blend) in enumerate(frame_plan(scenes, fps, args.crossfade)):
        (i, tl), *rest = parts
        img = scenes[i].render(tl)
        if rest:
            j, tj = rest[0]
            img = crossfade(img, scenes[j].render(tj), blend)
        writer.write(img)
        if k % (fps * 5) == 0:
            print(f"  {k / fps:5.1f} s  ({time.time() - t0:.0f} s elapsed)")
    writer.close()
    print("wrote", out)


def stills(args):
    scenes = build_scenes(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(scenes):
        for frac in (0.5, 0.9):
            s.render(s.duration * frac).save(out / f"{i:02d}_{s.name}_{int(frac * 100)}.jpg", quality=88)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "render", "stills"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="bambi_video", help="working folder for flights and caches")
        if name == "prepare":
            p.add_argument("--skip-map", action="store_true", help="do not fetch every flight's poses")
        else:
            p.add_argument("-o", "--output", default="bambi_teaser.mp4" if name == "render" else "stills")
            p.add_argument("--anonymous", action="store_true", help="no repository links on the closing card")
        if name == "render":
            p.add_argument("--fps", type=int, default=FPS_DEFAULT)
            p.add_argument("--crf", type=int, default=18)
            p.add_argument("--crossfade", type=float, default=0.6, help="seconds between scenes")
            p.add_argument("--preview", action="store_true", help="half resolution")
            p.add_argument("--only", help="comma-separated scene names, e.g. title,map")
    args = ap.parse_args(argv)
    {"prepare": prepare, "render": render, "stills": stills}[args.cmd](args)


if __name__ == "__main__":
    main()
