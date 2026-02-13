#!/usr/bin/env python3
"""
Extract frames from a split video where the left half is thermal and the right half is RGB.
Frames are saved to separate subfolders with configurable sampling rate.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_rate: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    export_thermal: bool = True,
    export_rgb: bool = True,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video_name = video_path.stem

    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if not export_thermal and not export_rgb:
        print("Error: At least one of --thermal or --rgb must be enabled.")
        sys.exit(1)

    # Create output subdirectories
    thermal_dir = output_dir / "thermal"
    rgb_dir = output_dir / "rgb"

    if export_thermal:
        thermal_dir.mkdir(parents=True, exist_ok=True)
    if export_rgb:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_width = width // 2

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height} (each half: {half_width}x{height})")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"Frame range: {start_frame} to {end_frame if end_frame is not None else 'end'} (exclusive)")
    print(f"Sample rate: every {sample_rate} frame(s)")
    print(f"Export thermal: {export_thermal}, Export RGB: {export_rgb}")
    print()

    # Seek to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    exported_count = 0

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % sample_rate == 0:
            filename = f"{video_name}_{frame_idx:08d}.png"

            if export_thermal:
                thermal_frame = frame[:, :half_width]
                cv2.imwrite(str(thermal_dir / filename), thermal_frame)

            if export_rgb:
                rgb_frame = frame[:, half_width:]
                cv2.imwrite(str(rgb_dir / filename), rgb_frame)

            exported_count += 1
            if exported_count % 100 == 0:
                print(f"  Exported {exported_count} frames (frame index {frame_idx})...")

        frame_idx += 1

    cap.release()
    print(f"\nDone. Exported {exported_count} frames out of {frame_idx} total.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a split video (left=thermal, right=RGB)."
    )
    parser.add_argument("video",
                        help="Path to the input video file.")
    parser.add_argument("output", help="Path to the output directory.")
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=1,
        help="Extract every N-th frame (default: 1 = every frame).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame index to consider (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last frame index (exclusive). Default: process until end of video.",
    )
    parser.add_argument(
        "--thermal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export thermal frames (left half). Use --no-thermal to disable.",
    )
    parser.add_argument(
        "--rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export RGB frames (right half). Use --no-rgb to disable.",
    )

    args = parser.parse_args()

    if not args.thermal and not args.rgb:
        parser.error("At least one of --thermal or --rgb must be enabled.")

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        start_frame=args.start,
        end_frame=args.end,
        export_thermal=args.thermal,
        export_rgb=args.rgb,
    )


if __name__ == "__main__":
    main()