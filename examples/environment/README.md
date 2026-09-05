# A dense run of environment masks

All ten environment classes over **34 consecutive-ish frames of flight 332** —
every fifth frame from 5450 to 5600, plus the five key frames that do not fall
on that grid. Produced with
[`environment_segmentation.py`](../../environment_segmentation.py), on the same
GPU and with the same settings as the release.

It exists so that
[`environment_segmentation.ipynb`](../../environment_segmentation.ipynb) can be
run without a GPU, without the gated SAM 3 checkpoint, and without the two
SegFormer checkpoints; section 5 of the notebook shows how to regenerate it.

**29 of these masks are also published**, on the five key frames the release
covers, and they are **pixel-identical** to the released ones — the notebook
checks this in section 6. The frames were extracted from the same video with
the same encoder settings, so they are byte-identical to the ones the release
was computed from and the comparison is exact rather than approximate.

| | |
|---|---|
| file | `flight332_dense.json.gz` (444 KB, 1.8 MB uncompressed) |
| flight | 332, a forest road beside an alpine river |
| frames | 34, spanning 5450 to 5600 (five seconds at 29.97 fps) |
| of those, published | 5 (frames 5458, 5533, 5567, 5570, 5585) |
| masks | 189 |
| classes present | road, grass, bare ground, rock, water, tree cover |

> ⚠️ **Mixed licence.** The eight SAM 3 classes (snow, water, road, grass, rock,
> bare ground, roof, vehicle) are CC-BY-4.0. `tree cover` and `deadwood` come
> from models built on NVIDIA's SegFormer and are **CC-BY-NC-4.0**: research and
> evaluation use only. See
> [docs/dataset-versions.md](../../docs/dataset-versions.md#licensing-and-why-the-environment-layers-are-split-in-two).

## Format

Gzipped JSON. `environment_segmentation.load_masks` reads it directly and
returns `masks[frame][class]` as boolean arrays.

```json
{
  "flight": 332,
  "frame_range": [5450, 5600],
  "step": 5,
  "extra_frames": [5458, 5533, 5567, 5570, 5585],
  "classes": ["snow", "water", ..., "tree cover", "deadwood"],
  "mask_format": "COCO RLE, column-major, 1024x1024 full frame",
  "dynamic_range": {"005450": 148.0, ...},
  "coverage":      {"005450": {"road": 0.0484, ...}, ...},
  "masks":         {"005450": {"road": {"size": [1024, 1024], "counts": [...]}, ...}, ...}
}
```

* Frame keys are zero-padded strings; `load_masks` converts them to integers.
* `coverage` is the fraction of the **imaged area**, excluding the letterbox
  bands — not of the padded 1024x1024 square.
* `dynamic_range` is the luminance p98 − p2 over the imaged area, the statistic
  the release's `undetermined` flag is built on.
* A class absent from a frame was not detected there. Empty masks are not
  stored.
