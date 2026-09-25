"""Heavy renders done once and cached: light-field focal sweeps and orthographic mosaics.

Both go through the BAMBI geo stack (``bambi_detection`` + ``alfspy``), so the
pictures in the film are the ones the public tooling produces. The DEM is
replaced by a flat plane: a nadir window of a few seconds covers a few dozen
metres, over which the plane is a fair focal surface, and sweeping its height
*is* the synthetic refocus the light field allows.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

os.environ.setdefault("ALFS_ENGINE", "torch")      # the one engine that needs no GL driver

FOVY = 50.0
EPSG = 32633                                        # UTM 33N covers all of Austria


def _stack():
    from bambi.geo.poses import Poses, make_origin
    from bambi.io.corrections import corrections_for_frames, read_corrections
    from bambi.io.dem import render_data_from_arrays
    from bambi.io.poses import read_poses, to_local_poses
    from bambi.render.ortho import make_shot, make_shots, render_orthographic
    from bambi.util.render_context import make_render_context
    return locals()


def local_poses(flight, anchor_frame: int):
    s = _stack()
    pf = s["read_poses"](flight.folder / f"{flight.key}_matched_poses.json")
    origin = s["make_origin"](*pf.lla[anchor_frame, :2], 0.0, EPSG)
    poses = s["to_local_poses"](pf, epsg=EPSG, origin=origin)
    tc, rc = s["corrections_for_frames"](
        s["read_corrections"](flight.folder / f"{flight.key}_correction.json"), len(poses))
    return poses, tc, rc


def _plane(s, cx, cy, h, r):
    v = np.array([[-r, -r, h], [r, -r, h], [r, r, h], [-r, r, h]], float) + [cx, cy, 0]
    md, td = s["render_data_from_arrays"](v, np.array([[0, 1, 2], [0, 2, 3]]))
    return md, td, (v[0, 0], v[0, 1], v[2, 0], v[2, 1])


def box_mask(boxes, size=1024) -> np.ndarray:
    """A black frame with the annotated boxes filled white (projected like any image)."""
    m = np.zeros((size, size, 3), np.uint8)
    for d in boxes:
        x, y, w, h = d["bb_left"], d["bb_top"], d["bb_width"], d["bb_height"]
        cv2.rectangle(m, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return m


def focal_sweep(flight, centre: int, heights_agl: Sequence[float], half_window: int = 120, step: int = 4,
                radius: float = 22.0, size: int = 900, out: Path = None, ground_agl: float = None):
    """Integrals of the thermal frames around ``centre`` on planes ``heights_agl`` metres below the drone.

    Also stores the single centre frame projected on the ground plane and the
    annotated boxes of that frame projected the same way. Returns the npz path.
    """
    s = _stack()
    poses, tc, rc = local_poses(flight, centre)
    ctx = s["make_render_context"]()
    win = np.arange(centre - half_window, centre + half_window + 1, step)
    win = win[(win >= 0) & (win < len(poses))]
    images = [flight.frame(i)[0].copy() for i in win]
    cx, cy, z = poses.positions[centre]
    ground_agl = ground_agl if ground_agl is not None else max(heights_agl)
    integrals = []
    for agl in heights_agl:
        # the same view cone at every depth, so refocusing does not also zoom
        md, td, b = _plane(s, cx, cy, z - agl, radius * agl / ground_agl)
        shots = s["make_shots"](ctx, images, poses, FOVY, 1.0, tc, rc, indices=win, lazy=True)
        img = s["render_orthographic"](ctx, md, td, shots, b, (size, size), integral=True, release_shots=True)
        integrals.append(img[:, :, :3])
        print(f"  focal plane {agl:5.1f} m below the drone")
    md, td, b = _plane(s, cx, cy, z - ground_agl, radius)
    one = lambda im: s["render_orthographic"](
        ctx, md, td, s["make_shot"](ctx, im, poses.positions[centre], poses.rotations[centre], FOVY, 1.0,
                                    tc[centre], rc[centre], lazy=False), b, (size, size))
    single = one(flight.frame(centre)[0].copy())
    single_rgb = one(flight.frame(centre)[1].copy())
    boxes = one(box_mask(flight.thermal_tracks.at(centre)))
    np.savez_compressed(out, integrals=np.stack(integrals), heights=np.asarray(heights_agl, float),
                        single=single[:, :, :3], single_alpha=single[:, :, 3], single_rgb=single_rgb[:, :, :3],
                        boxes=boxes[:, :, 0], n_frames=len(win), radius=radius)
    return out


def ortho_strip(flight, frames: Sequence[int], ground_agl: float, centre_xy, radius: float, size: int,
                out: Path, anchor: int = None, track_step: int = 3):
    """Per-frame orthophotos of ``frames`` on one ground grid, for the growing-mosaic scene.

    Saves each frame's thermal and RGB projection with its validity mask, the
    drone path, every frame's ground footprint, and the annotated animals'
    box centres cast to the ground - their tracks in world coordinates -
    all in the mosaic's pixel grid.
    """
    import trimesh
    from bambi.geo.camera import cameras_from_poses
    from bambi.geo.georef import footprint, pixels_to_world

    s = _stack()
    anchor = frames[len(frames) // 2] if anchor is None else anchor
    poses, tc, rc = local_poses(flight, anchor)
    ctx = s["make_render_context"]()
    h = poses.positions[anchor, 2] - ground_agl
    md, td, b = _plane(s, centre_xy[0], centre_xy[1], h, radius)
    v = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * radius * 3 + centre_xy
    mesh = trimesh.Trimesh(np.column_stack([v, np.full(4, h)]), [[0, 1, 2], [0, 2, 3]], process=False)
    to_px = lambda xy: np.column_stack([(xy[:, 0] - b[0]) / (b[2] - b[0]) * size,
                                        (b[3] - xy[:, 1]) / (b[3] - b[1]) * size])
    th, rg, va, fp = [], [], [], []
    for f in frames:
        t_img, r_img = (a.copy() for a in flight.frame(f))
        pr = []
        for im in (t_img, r_img):
            shot = s["make_shot"](ctx, im, poses.positions[f], poses.rotations[f], FOVY, 1.0, tc[f], rc[f],
                                  lazy=False)
            pr.append(s["render_orthographic"](ctx, md, td, shot, b, (size, size)))
        th.append(pr[0][:, :, :3]); rg.append(pr[1][:, :, :3]); va.append(pr[0][:, :, 3] > 0)
        cam = cameras_from_poses(poses, FOVY, 1.0, tc, rc, indices=[f])[0]
        fp.append(to_px(footprint(cam, 1024, 1024, mesh, samples_per_edge=4)[:, :2]))
    # animals on the ground: box centres of every track, cast frame by frame
    tracks = []                                     # rows: frame, track_id, x_px, y_px, species
    for f in range(frames[0], frames[-1] + 1, track_step):
        dets = flight.thermal_tracks.at(f)
        if not dets:
            continue
        cam = cameras_from_poses(poses, FOVY, 1.0, tc, rc, indices=[f])[0]
        px = np.array([[d["bb_left"] + d["bb_width"] / 2, d["bb_top"] + d["bb_height"] / 2] for d in dets])
        w = pixels_to_world(px, cam, 1024, 1024, mesh)
        g = to_px(w[:, :2])
        for d, (x, y) in zip(dets, g):
            if np.isfinite(x):
                tracks.append((f, d["track_id"], x, y, d["species"]))
    path = to_px(poses.positions[frames[0]:frames[-1] + 1, :2])
    np.savez_compressed(out, thermal=np.stack(th), rgb=np.stack(rg), valid=np.stack(va),
                        frames=np.asarray(frames), path=path, footprints=np.stack(fp), bounds=np.asarray(b),
                        size=size, metres=2 * radius,
                        tracks=np.array([t[:4] for t in tracks], float),
                        species=np.array([t[4] for t in tracks]))
    return out
