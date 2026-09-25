# Dataset teaser video

An ~86 s, 1920×1080 film of the BAMBI dataset for the paper's project page and
talk, rendered entirely from the public data and toolkits:

| Scene | What it shows | Built from |
|---|---|---|
| Title | thermal / RGB split over a fallow deer herd in snow | flight 17 |
| Where | flight sites on a map of Austria, flights per month (Fig. 5d) | poses of all flights, `flight_metadata/` |
| What | headline numbers (386 flights, 49 h 44 min, 5,100 tracks, …) | paper, Sec. 4 |
| Footage ×6 | thermal and RGB side by side with the annotated tracks; RGB boxes from the `owl-transferred` version | flights 17, 6, 142, 11, 143, 31 |
| Who | tracks and key frames per class (Table 1) | paper, Table 1 |
| Where exactly | frames projected to the ground as an orthomosaic, boxes cast to world-coordinate trajectories | flight 142, `bambi_detection` + `alfspy` |
| See through the canopy | one frame vs. an airborne light-field integral whose focal plane sinks from the canopy to the forest floor | flight 11, `bambi_detection` + `alfspy` |
| Why | the tasks the dataset supports | stills of the scenes above |
| Credits | links to data, code and models (`--anonymous` drops them) | |

## Rendering

```bash
pip install -r requirements.txt imageio-ffmpeg pillow
pip install "git+https://github.com/bambi-eco/bambi_detection.git"
pip install "AlfsPy[torch] @ git+https://github.com/bambi-eco/alfs_py.git"

# downloads ~7 GB of flights, the poses of every flight (for the map) and the
# font, then pre-renders the light-field sweep and the mosaic (CPU is fine)
python teaser_video/make_video.py prepare --data bambi_video

python teaser_video/make_video.py stills --data bambi_video -o stills      # check the look
python teaser_video/make_video.py render --data bambi_video -o bambi_teaser.mp4
python teaser_video/make_video.py render --data bambi_video --anonymous -o bambi_teaser_anon.mp4
```

`render --preview` writes half resolution, `--only title,map` renders a subset
of scenes. The storyboard (clips, frames, the ALFS and mosaic settings) sits at
the top of `make_video.py`.

## Notes

- The light field and the mosaic use a flat focal plane instead of a DEM: over
  a few seconds of flight the terrain is close to flat, and sweeping the
  plane's height is exactly the synthetic refocus. The heights (55 m below the
  drone for flight 11, 30 m for flight 142) were picked from focal sweeps.
- Some flights carry fewer poses than video frames (e.g. flight 6); the footage
  scene then shows no coordinates rather than wrong ones.
- `assets/austria.json` is a simplified extract of Natural Earth (public
  domain). Inter (SIL OFL) is downloaded by `prepare`.
