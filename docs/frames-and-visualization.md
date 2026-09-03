# Frames and visualization

Getting images out of the split videos and drawing annotations on them.

## Frame Extraction

Extract thermal and RGB frames from the side-by-side video files. The left half of each frame contains the thermal channel, the right half the RGB channel.

```bash
# Extract every frame
python frame_extraction.py video.mp4 ./frames

# Extract every 10th frame, RGB only
python frame_extraction.py video.mp4 ./frames --sample-rate 10 --thermal false

# Extract a specific frame range
python frame_extraction.py video.mp4 ./frames --start 1000 --end 2000
```

Output structure:
```
frames/
├── thermal/
│   ├── video_00001000.png
│   ├── video_00001001.png
│   └── ...
└── rgb/
    ├── video_00001000.png
    ├── video_00001001.png
    └── ...
```

## Visualization

### On Extracted Frames

Overlay bounding boxes on individual extracted frames:

```bash
# Visualize key frame annotations
python mot_frame_viewer.py frame_image.png annotations.txt --show

# With interpolation of in-between frames
python mot_frame_viewer.py frame_image.png annotations.txt --interpolate --show

# Save the visualization to a file
python mot_frame_viewer.py frame_image.png annotations.txt -o output.png
```

### On Video

Overlay bounding box tracks directly on video:

```bash
# Live preview
python mot_video_viewer.py video.mp4 annotations.txt --show

# Save annotated video
python mot_video_viewer.py video.mp4 annotations.txt -o annotated_output.mp4

# With interpolated tracks
python mot_video_viewer.py video.mp4 annotations.txt -o output.mp4 --interpolate
```
