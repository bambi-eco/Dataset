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

## Frame-to-terrain animation

`frame_dem_animation.py` renders a short video that shows how one frame relates
to the terrain it was recorded over. You first see the plain thermal or RGB
frame. The viewpoint then turns around the image's x-axis until the image is
seen edge-on as a one-pixel-thick line, with the camera frustum above it and
the relief line of the digital elevation model (DEM) below it; the DEM surface
is visible in the background while the view turns. After a short pause the
image falls: every pixel stops the moment it touches the relief, the rest keep
falling until all of them rest on the terrain. With `--neighbors` the frames
before and after the central one then fade in together with their own frustums
and fall in sync, which is what an ALFS integral rendering does.

It needs the DEM from [`dem_from_poses.py`](geospatial.md) and the poses file;
frames come from the split video or from a folder written by
`frame_extraction.py`.

```bash
# thermal, single central frame, 90 degree roll (the default)
python frame_dem_animation.py --video bambi_downloads/146_matched_processed.mp4 \
    --poses bambi_downloads/146_matched_poses.json \
    --dem bambi_downloads/146_matched_dem.tif --dem-json bambi_downloads/146_matched_dem.json \
    --frame 2125 -o 146_thermal.mp4

# RGB, ALFS-style: 8 frames before and 8 after (every 20th frame) fall in together;
# the view turns so that the flight direction runs across the screen (neighbours left and right)
python frame_dem_animation.py --frames-dir 146_frames --poses ... --dem ... \
    --frame 2125 --modality rgb --neighbors 8 --neighbor-step 20 -o 146_alfs.mp4

# 45 degree roll instead of 90: the image stays a textured plane, animated in 3D
python frame_dem_animation.py ... --roll 45 -o 146_3d.mp4

# no data at hand? a synthetic terrain and flight
python frame_dem_animation.py --demo -o demo.mp4
```

| Option | Default | Description |
|---|---|---|
| `--video` / `--frames-dir` / `--image` | *(one required)* | Split video, `frame_extraction.py` output folder, or a single extracted frame |
| `--poses`, `--dem`, `--frame` | *(required)* | Poses JSON, DEM GeoTIFF, index of the central frame |
| `--dem-json` | derived from the GeoTIFF | DEM origin metadata written next to the `.glb` |
| `--modality` | `thermal` | `thermal` or `rgb` |
| `--neighbors` | `0` | Frames before *and* after the central one that fall in afterwards (`0` = single view) |
| `--neighbor-step` | `1` | Frame stride between neighbours. Flight 146 moves about 0.18 m per frame, so a stride of 20 puts the cameras 3.5 m apart |
| `--roll` | `90` | How far the viewpoint turns; `90` gives the edge-on line, e.g. `45` animates in 3D |
| `--roll-axis` | `auto` | Turn about the image's x-axis (line = a row) or y-axis (line = a column). `auto` picks the axis that puts the flight direction across the screen, so neighbour frames sit left and right of the central one |
| `--fall-mode` | `vertical` | Pixels fall straight down, or `ray`: slide along their camera rays |
| `--fall-easing` | `gravity` | Accelerating or `linear` fall |
| `--plane-height` | `0.35` | Start height of the image plane as a fraction of the height above ground |
| `--pitch-convention` | `nadir` | BAMBI poses store the tilt from nadir; use `dji` for raw gimbal pitch (-90 = down) |
| `--hold`, `--roll-duration`, `--pause`, `--fall-duration`, `--end-hold` | `1`, `2.5`, `0.6`, `3`, `1.5` | Phase durations in seconds |
| `--neighbor-delay`, `--neighbor-fade`, `--neighbor-fall`, `--neighbor-stagger` | `0.4`, `0.5`, `1.8`, `0` | Timing of the neighbour frames; by default they all appear and fall in sync, a stagger > 0 drops them one after another |
| `--fit` | `scene` | Final framing: whole scene with the camera, or `image` to zoom onto image and relief |
| `--dem-alpha-behind`, `--dem-alpha-fall` | `0.3`, `0` | Edge-on view: opacity of the terrain behind the cross-section after the roll, and of the terrain while the frames fall (fully transparent by default so the landed pixels stay visible) |
| `--theme`, `--bg`, `--dem-cmap` | `dark`, theme, `gist_earth` | Colours |
| `--width`, `--height`, `--fps` | `1280`, `720`, `30` | Output format; `-o` takes `.mp4`, `.gif` or a folder for PNG frames |
| `--preview T` | | Render only the frame at `T` seconds as a PNG, handy for tuning |

Poses are read like AlfsPY reads them: `rotation` is `[tilt, roll, heading]`
with the tilt measured from nadir and the heading clockwise from north; a
nadir frame's top points along the heading. If the poses have no `location`
yet, the script converts `lat`/`lng`/`alt` into the DEM's coordinate system
itself, so `add_relative_dem_position_to_poses.py` is optional here.
