# Precomputed OWL detections

Point detections produced by **OWL-D** (ViT-H/16, zero-shot: the checkpoint was
trained on public overhead datasets, not on BAMBI) over the RGB frames of two
flights of the `matched` release. They exist so that
[`owl_label_transfer.ipynb`](../../owl_label_transfer.ipynb) can be run without
a GPU; section 4 of the notebook shows how to regenerate them.

| file | flight | frames scored | detections | why this flight |
|---|---|---|---|---|
| `owl_detections_flight119.csv` | 119 | 734 | 838 | one animal at a time, large (32 px) shift that drifts over the flight |
| `owl_detections_flight163.csv` | 163 | 532 | 3,169 | 16 tracks, several animals per frame, exercises the identity check |

## Format

```
frame,x,y,score
0,430.0,854.0,0.1094
```

* `frame`: source-video frame index, matching the `<frame>.jpg` file names and
  the `frame` column of the MOT labels.
* `x`, `y`: the detected point in **image pixels** (1024x1024 frames).
* `score`: OWL's detection score.

Note that OWL's own `detections.csv` stores points in *heatmap* coordinates: the
released configs stitch with `down_ratio: 2` and `up: False`, so a point in a
1024x1024 frame comes back in `0..511`. These files already have that factor of
2 applied; a raw `detections.csv` passed to `transfer_labels.py` needs
`--detection-scale 2`.

Frames that OWL scored but found nothing in are not listed. `transfer_labels.py`
treats any frame absent from this file as "not processed" and falls back to the
consensus curve for it, which is the same outcome.
