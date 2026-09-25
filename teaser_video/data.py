"""Reading a downloaded flight: the split video, its tracks, poses and metadata."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from mot_video_viewer import build_tracks, interpolate_tracks, parse_mot_file  # noqa: E402

HALF = 1024      # each half of the processed video is 1024 x 1024


class FrameReader:
    """Random access into a video that is cheap when the access is sequential."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(self.path)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 29.97
        self.pos = -1
        self.last = None

    def get(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.count - 1))
        if idx == self.pos and self.last is not None:
            return self.last
        if not (self.pos < idx <= self.pos + 8):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx - 1
        while self.pos < idx:
            ok, img = self.cap.read()
            if not ok:
                return self.last if self.last is not None else np.zeros((HALF, 2 * HALF, 3), np.uint8)
            self.pos += 1
            self.last = img
        return self.last


@dataclass
class Tracks:
    """Interpolated boxes, by frame and by track."""
    by_frame: Dict[int, List[dict]]
    by_track: Dict[int, Dict[int, dict]]

    @classmethod
    def load(cls, path: Path) -> "Tracks":
        dets = parse_mot_file(str(path)) if Path(path).exists() else []
        tracks = interpolate_tracks(build_tracks(dets))
        by_frame, by_track = defaultdict(list), {}
        for tid, lst in tracks.items():
            by_track[tid] = {d["frame"]: d for d in lst}
            for d in lst:
                by_frame[d["frame"]].append(d)
        return cls(dict(by_frame), by_track)

    def at(self, frame: int) -> List[dict]:
        return self.by_frame.get(int(frame), [])

    def trail(self, tid: int, frame: int, length: int, step: int = 2) -> List[tuple]:
        """Box centres of track ``tid`` over the ``length`` frames up to ``frame``."""
        t = self.by_track.get(tid, {})
        pts = []
        for f in range(frame - length, frame + 1, step):
            d = t.get(f)
            if d is not None:
                pts.append((d["bb_left"] + d["bb_width"] / 2, d["bb_top"] + d["bb_height"] / 2))
        return pts


@dataclass
class Flight:
    key: str
    folder: Path
    owl_folder: Optional[Path] = None
    _reader: Optional[FrameReader] = field(default=None, repr=False)

    def __post_init__(self):
        self.folder = Path(self.folder)
        self.meta = json.loads(self._file("_metadata.json").read_text())
        self.thermal_tracks = Tracks.load(self._file("_gt.txt"))
        rgb = self._owl_file("_rgb_gt.txt")
        self.rgb_tracks = Tracks.load(rgb) if rgb is not None else None
        poses = json.loads(self._file("_matched_poses.json").read_text())
        self.poses = poses["images"]
        self.drone = poses.get("drone", "")

    def _file(self, suffix: str) -> Path:
        p = self.folder / f"{self.key}{suffix}"
        if not p.exists() and self.owl_folder is not None:
            p = Path(self.owl_folder) / f"{self.key}{suffix}"
        return p

    def _owl_file(self, suffix: str) -> Optional[Path]:
        for d in (self.owl_folder, self.folder):
            if d is not None and (Path(d) / f"{self.key}{suffix}").exists():
                return Path(d) / f"{self.key}{suffix}"
        return None

    @property
    def video(self) -> Path:
        return self.folder / f"{self.key}_matched_processed.mp4"

    @property
    def reader(self) -> FrameReader:
        if self._reader is None:
            self._reader = FrameReader(self.video)
        return self._reader

    def frame(self, idx: int):
        """``(thermal, rgb)`` BGR halves of frame ``idx``."""
        img = self.reader.get(idx)
        return img[:, :HALF], img[:, HALF:2 * HALF]

    @property
    def poses_aligned(self) -> bool:
        """One pose per video frame (not the case for every flight)."""
        return len(self.poses) == self.reader.count

    def pose(self, idx: int) -> dict:
        return self.poses[int(np.clip(idx, 0, len(self.poses) - 1))]

    @property
    def date(self) -> datetime:
        return datetime.fromisoformat(self.meta["flight_info"]["start_time"])

    @property
    def drone_name(self) -> str:
        name = self.meta["flight_info"].get("drone_name", "")
        return {"Matric 30 Thermal": "DJI M30T", "Mavic 3 Thermal": "DJI M3T"}.get(name, name)
