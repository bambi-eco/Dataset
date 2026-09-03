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
| `owl-transferred` | RGB boxes for the `base` annotations, transferred from thermal with OWL. An **annotation layer**, not a standalone release. | `<id>_rgb_gt.txt`, `<id>_provenance.csv`, `<id>_owl_detections.csv` |

`owl-transferred` ships no imagery, so selecting it downloads the `base`
recordings as well and both land in the same directory. It covers 238 of the
386 flights — see [label-transfer.md](label-transfer.md) for what it contains,
how accurate it is, and why the other flights are missing.

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
