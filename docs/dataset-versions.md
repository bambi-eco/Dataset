# Dataset versions

The recordings are published in several versions. They share the annotation
format described in [annotation-format.md](annotation-format.md) and are all
fetched with the same script, via `--version`; see
[downloading.md](downloading.md).

| `--version` | What it is | Per flight |
|---|---|---|
| `base` *(default)* | The processed recordings: undistorted, FCC calibration phases removed, thermal and RGB synced from the flight logs. | `<id>_matched_processed.mp4` (2048x1024, thermal left / RGB right), `<id>_gt.txt`, `<id>_matched_poses.json`, `<id>_mask_t.png`, `<id>_mask_w.png`, `<id>_metadata.json`, `<id>_correction.json`, `<id>_track_mapping.json` |
| `raw` | The original, unprocessed videos as they came off the drone, with undistortion parameters, SRT subtitles and Airdata logs. | one folder per flight |
| `matched` | A red deer subset carrying human-accepted **RGB** boxes produced by the template-matching toolkit, plus extracted frames. | `<id>_accepted_thermal_mot.txt`, `<id>_accepted_rgb_mot.txt` |
| `orthographic` | The matched subset reprojected to an orthographic view. | |
| `owl-transferred` ⚠️ | **Experimental.** RGB boxes for the `base` annotations, transferred from thermal with OWL. An **annotation layer**, not a standalone release. | `<id>_rgb_gt.txt`, `<id>_provenance.csv`, `<id>_owl_detections.csv` |
| `environment` ⚠️ | **Experimental.** Where the trees are and where the snow is, per frame. An annotation layer. | `<id>_tree_positions.json`, `<id>_snow.json` |
| `environment-nc` ⚠️🔒 | **Experimental, non-commercial.** Canopy extent and standing deadwood, per frame. An annotation layer. | `<id>_tree_cover.json`, `<id>_deadwood.json` |
| `environment-all` ⚠️🔒 | Both environment layers at once. **Partly non-commercial**; see below. | the union of the two above |

`owl-transferred` ships no imagery, so selecting it downloads the `base`
recordings as well and both land in the same directory. It covers 238 of the
386 flights — see [label-transfer.md](label-transfer.md) for what it contains,
how accurate it is, and why the other flights are missing.

The `environment` layers are likewise annotation-only and pull `base` with them.
See [environment.md](environment.md) for what each class means and how it was
produced.

> ⚠️ **`owl-transferred` is experimental.** Unlike the other versions, its
> annotations are machine-generated and have not been reviewed by hand. They
> agree closely with the human-accepted annotations on average, but an
> individual flight can be confidently wrong in a way none of the pipeline's
> internal checks catch. Every box carries a provenance record saying where its
> position came from — use it. Coverage and content may change in a future
> revision, so cite the DOI you actually used.

## Licensing, and why the environment layers are split in two

Every version of this dataset is **CC-BY-4.0** except `environment-nc`, which is
**CC-BY-NC-4.0**. That one exception is the reason the environment annotations
are published as two versions rather than one, so it is worth explaining rather
than leaving as a surprise.

The environment layers are produced by four different methods, and two of them
carry a restriction the other two do not:

| layer | produced by | licence of the method | released as |
|---|---|---|---|
| tree positions | [DeepForest](https://github.com/weecology/DeepForest) | MIT | `environment`, CC-BY-4.0 |
| snow | brightness/saturation threshold, in this repository | none | `environment`, CC-BY-4.0 |
| tree cover | [Restor TCD](https://huggingface.co/restor/tcd-segformer-mit-b5) | CC-BY-NC + NVIDIA Source Code License | `environment-nc`, CC-BY-NC-4.0 |
| deadwood | [deadtrees.earth](https://github.com/cmosig/deadtreesmodels) | MIT, but see below | `environment-nc`, CC-BY-NC-4.0 |

Both restricted layers trace back to the same root: their models are built on
**SegFormer**, which NVIDIA released under the NVIDIA Source Code License,
permitting use "non-commercially, meaning for research or evaluation purposes
only". Restor state this directly on their model card. The deadwood model is
less obvious — deadtrees.earth release their work under MIT, and so is
`segmentation_models_pytorch`, but the MiT encoder file inside that MIT package
carries an explicit carve-out:

```
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License
```

so the restriction is inherited whether or not the wrapping project mentions it.

Splitting on that boundary keeps the layers that are free of it genuinely free:
tree positions and snow can be used commercially, redistributed, and combined
with the rest of the dataset under one licence. Merging everything into a single
non-commercial release would have restricted them for no reason.

`--version environment-all` downloads both and is therefore **partly
non-commercial**. It prints a warning before and after the download, because
the files from all layers land in the same directory and nothing about a mask
file itself records which licence it arrived under. `--licences` prints the
table for every version:

```bash
python download_from_zenodo.py --licences
```

Two notes on how this may change. Restor's model cards say "CC-BY-NC; CC-BY to
follow", so the restriction on tree cover may be lifted in future. And the
underlying training data, [OAM-TCD](https://huggingface.co/datasets/restor/tcd),
is itself CC-BY-4.0 with smaller CC-BY-NC and CC-BY-SA subsets — the
non-commercial constraint here comes from the *models*, not from the imagery
they were trained on.

## Zenodo records

The processed dataset is publicly available on Zenodo:

- Part 1: [10.5281/zenodo.18692354](https://doi.org/10.5281/zenodo.18692354)
- Part 2: [10.5281/zenodo.18698508](https://doi.org/10.5281/zenodo.18698508)
- Part 3: [10.5281/zenodo.18703312](https://doi.org/10.5281/zenodo.18703312)
- Part 4: [10.5281/zenodo.18705705](https://doi.org/10.5281/zenodo.18705705)
- Part 5: [10.5281/zenodo.18707610](https://doi.org/10.5281/zenodo.18707610)
- Part 6: [10.5281/zenodo.18711217](https://doi.org/10.5281/zenodo.18711217)
- Part 7: [10.5281/zenodo.18715162](https://doi.org/10.5281/zenodo.18715162)
- Part 8: [10.5281/zenodo.18717601](https://doi.org/10.5281/zenodo.18717601)

The unprocessed dataset is also publicly available on Zenodo:

- Part 1:  [10.5281/zenodo.18885436](https://doi.org/10.5281/zenodo.18885436 )
- Part 2:  [10.5281/zenodo.18895587](https://doi.org/10.5281/zenodo.18895587 )
- Part 3:  [10.5281/zenodo.18898626](https://doi.org/10.5281/zenodo.18898626 )
- Part 4:  [10.5281/zenodo.18902130](https://doi.org/10.5281/zenodo.18902130 )
- Part 5:  [10.5281/zenodo.18905738](https://doi.org/10.5281/zenodo.18905738 )
- Part 6:  [10.5281/zenodo.18908258](https://doi.org/10.5281/zenodo.18908258 )
- Part 7:  [10.5281/zenodo.18911860](https://doi.org/10.5281/zenodo.18911860 )
- Part 8:  [10.5281/zenodo.18916791](https://doi.org/10.5281/zenodo.18916791 )
- Part 9:  [10.5281/zenodo.18920616](https://doi.org/10.5281/zenodo.18920616 )
- Part 10: [10.5281/zenodo.18928160](https://doi.org/10.5281/zenodo.18928160)
- Part 11: [10.5281/zenodo.18931665](https://doi.org/10.5281/zenodo.18931665)
- Part 12: [10.5281/zenodo.18937467](https://doi.org/10.5281/zenodo.18937467)
- Part 13: [10.5281/zenodo.18943988](https://doi.org/10.5281/zenodo.18943988)
- Part 14: [10.5281/zenodo.18949040](https://doi.org/10.5281/zenodo.18949040)
- Part 15: [10.5281/zenodo.18958982](https://doi.org/10.5281/zenodo.18958982)
- Part 16: [10.5281/zenodo.18968497](https://doi.org/10.5281/zenodo.18968497)
- Part 17: [10.5281/zenodo.18975298](https://doi.org/10.5281/zenodo.18975298)
- Part 18: [10.5281/zenodo.18983480](https://doi.org/10.5281/zenodo.18983480)
- Part 19: [10.5281/zenodo.18989873](https://doi.org/10.5281/zenodo.18989873)
- Part 20: [10.5281/zenodo.18995072](https://doi.org/10.5281/zenodo.18995072)
- Part 21: [10.5281/zenodo.19004434](https://doi.org/10.5281/zenodo.19004434)
- Part 22: [10.5281/zenodo.19011010](https://doi.org/10.5281/zenodo.19011010)
- Part 23: [10.5281/zenodo.19016292](https://doi.org/10.5281/zenodo.19016292)
- Part 24: [10.5281/zenodo.19022059](https://doi.org/10.5281/zenodo.19022059)
- Part 25: [10.5281/zenodo.19040833](https://doi.org/10.5281/zenodo.19040833)
- Part 26: [10.5281/zenodo.19043198](https://doi.org/10.5281/zenodo.19043198)
- Part 27: [10.5281/zenodo.19047570](https://doi.org/10.5281/zenodo.19047570)
- Part 28: [10.5281/zenodo.19052469](https://doi.org/10.5281/zenodo.19052469)
- Part 29: [10.5281/zenodo.19056538](https://doi.org/10.5281/zenodo.19056538)
- Part 30: [10.5281/zenodo.19058735](https://doi.org/10.5281/zenodo.19058735)
- Part 31: [10.5281/zenodo.19060823](https://doi.org/10.5281/zenodo.19060823)
- Part 32: [10.5281/zenodo.19066975](https://doi.org/10.5281/zenodo.19066975)
- Part 33: [10.5281/zenodo.19073897](https://doi.org/10.5281/zenodo.19073897)
- Part 34: [10.5281/zenodo.19077836](https://doi.org/10.5281/zenodo.19077836)
- Part 35: [10.5281/zenodo.19080733](https://doi.org/10.5281/zenodo.19080733)
- Part 36: [10.5281/zenodo.19135787](https://doi.org/10.5281/zenodo.19135787)
- Part 37: [10.5281/zenodo.19140480](https://doi.org/10.5281/zenodo.19140480)
- Part 38: [10.5281/zenodo.19142769](https://doi.org/10.5281/zenodo.19142769)
- Part 39: [10.5281/zenodo.19145596](https://doi.org/10.5281/zenodo.19145596)
- Part 40: [10.5281/zenodo.19150949](https://doi.org/10.5281/zenodo.19150949)
- Part 41: [10.5281/zenodo.19153807](https://doi.org/10.5281/zenodo.19153807)
- Part 42: [10.5281/zenodo.19155449](https://doi.org/10.5281/zenodo.19155449)
- Part 43: [10.5281/zenodo.19158628](https://doi.org/10.5281/zenodo.19158628)
- Part 44: [10.5281/zenodo.19161728](https://doi.org/10.5281/zenodo.19161728)
- Part 45: [10.5281/zenodo.19166111](https://doi.org/10.5281/zenodo.19166111)
- Part 46: [10.5281/zenodo.19172900](https://doi.org/10.5281/zenodo.19172900)
- Part 47: [10.5281/zenodo.19177615](https://doi.org/10.5281/zenodo.19177615)

The matched Red Deer subset:
- Part 1: [10.5281/zenodo.21291168](https://doi.org/10.5281/zenodo.21291168)
- Part 2: [10.5281/zenodo.21298044](https://doi.org/10.5281/zenodo.21298044)

The matched, orthographic Red Deer subset:
- Part 1:   [10.5281/zenodo.21301636](https://doi.org/10.5281/zenodo.21301636) 
- Part 2:   [10.5281/zenodo.21304284](https://doi.org/10.5281/zenodo.21304284) 
- Part 3:   [10.5281/zenodo.21308829](https://doi.org/10.5281/zenodo.21308829) 
- Part 4:   [10.5281/zenodo.21313119](https://doi.org/10.5281/zenodo.21313119) 
- Part 5:   [10.5281/zenodo.21315950](https://doi.org/10.5281/zenodo.21315950) 
- Part 6:   [10.5281/zenodo.21318796](https://doi.org/10.5281/zenodo.21318796) 
- Part 7:   [10.5281/zenodo.21322508](https://doi.org/10.5281/zenodo.21322508) 
- Part 8:   [10.5281/zenodo.21326737](https://doi.org/10.5281/zenodo.21326737) 
- Part 9:   [10.5281/zenodo.21330351](https://doi.org/10.5281/zenodo.21330351) 
- Part 10:  [10.5281/zenodo.21338084](https://doi.org/10.5281/zenodo.21338084) 
- Part 11:  [10.5281/zenodo.21342007](https://doi.org/10.5281/zenodo.21342007) 

The OWL-transferred RGB annotations (an annotation layer over the processed dataset — see [label-transfer.md](label-transfer.md)):
- Part 1: [10.5281/zenodo.22271027](https://doi.org/10.5281/zenodo.22271027)
- Part 2: [10.5281/zenodo.22271042](https://doi.org/10.5281/zenodo.22271042)
- Part 3: [10.5281/zenodo.22271073](https://doi.org/10.5281/zenodo.22271073)

---
