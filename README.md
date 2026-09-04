![header image bambi dataset](./figures/header.png)

# 🦌 BAMBI Dataset

**Multimodal Nadir UAV-Recordings of Forest Wildlife**

The BAMBI dataset is a large-scale airborne multispectral wildlife dataset comprising 389 paired RGB and thermal aerial video sequences recorded across diverse forest and forest-adjacent habitats in Austria. Each frame is geo-referenced with precise global coordinates (longitude, latitude, and altitude), enabling learning and evaluation in both image space and geographic space.

This repository provides sample scripts for downloading, processing, and visualizing the dataset.

> **Citation:** If you use this dataset in your research, please cite our paper (see [Citation](#citation)).

---

## Overview

| Property | Value |
|---|---|
| Video sequences | 389 paired RGB + thermal |
| Total flight time | ~45 hours |
| Annotated tracks | 5,100 |
| Key frames | 92,701 |
| Interpolated bounding boxes | 1,218,903 |
| Species classes | 12 |
| Recording period | January 2023 – November 2024 |
| Location | Austria (Tyrol, Upper Austria, Lower Austria, Salzburg, Styria, Carinthia) |

Each flight is one video in which the **left half is thermal and the right half
is RGB**, both 1024x1024, accompanied by an annotation file in MOT format and
the drone poses. Annotations are made on the thermal view.

## Getting started

```bash
git clone https://github.com/bambi-eco/Dataset.git && cd Dataset
pip install -r requirements.txt

# find the flights you care about
python filter_flights.py --folder flight_metadata/ --species "Red deer"

# fetch one, unpacked
python download_from_zenodo.py -f 146 --unzip

# pull a frame out of the video and draw the boxes on it
python frame_extraction.py bambi_downloads/146_matched_processed.mp4 frames --start 3448 --end 3449
python mot_frame_viewer.py frames/thermal/146_00003448.png bambi_downloads/146_gt.txt --show
```

[`introduction.ipynb`](introduction.ipynb) walks through all of this end to
end, including the geo-referenced tooling.

## Documentation

| | |
|---|---|
| [Dataset versions](docs/dataset-versions.md) | What each `--version` contains, how the licences differ, and the Zenodo DOIs |
| [Annotation format](docs/annotation-format.md) | The MOT columns, the species table, and how `class_id` is composed |
| [Downloading](docs/downloading.md) | Installing, filtering flights by metadata, and fetching them |
| [Working with annotations](docs/annotation-tools.md) | Interpolating, filtering, and converting to YOLO |
| [Frames and visualization](docs/frames-and-visualization.md) | Extracting frames, drawing boxes on images and video |
| [Thermal to RGB label transfer](docs/label-transfer.md) | Moving thermal boxes onto the RGB view, and the ⚠️ experimental `owl-transferred` annotations |
| [Environment annotations](docs/environment.md) | ⚠️ Experimental. Snow, water, roads, vegetation, canopy and deadwood per frame |
| [Geospatial tools](docs/geospatial.md) | Terrain models from flight poses |

Two notebooks: [`introduction.ipynb`](introduction.ipynb) for a first tour, and
[`owl_label_transfer.ipynb`](owl_label_transfer.ipynb) for the label transfer in
detail.

## RGB annotations

Annotations are made on the thermal view, and the two sensors are not perfectly
aligned, so a thermal box does not sit on the animal in the RGB frame. There
are RGB boxes for two different scopes:

- **`matched`** — human-accepted RGB boxes for a red deer subset, produced by
  the template-matching toolkit.
- **`owl-transferred`** — ⚠️ **experimental.** RGB boxes for 238 of the 386
  flights, produced by detecting the animals in RGB with OWL and re-centring
  each thermal box on its match. Machine-generated and **not reviewed by
  hand**: 5.22 px mean centre error against the accepted annotations, against
  15.98 px for leaving the thermal boxes alone, but individual flights can be
  confidently wrong. Treat it as a starting point to be checked, not as ground
  truth, and read the [known failure mode](docs/label-transfer.md#the-released-owl-transferred-annotations)
  before training on it. The content and coverage of this version may change.

```bash
# the recordings and the transferred RGB annotations, in one folder
python download_from_zenodo.py --version owl-transferred -f 146 --unzip
```

See [docs/label-transfer.md](docs/label-transfer.md).

## Environment annotations

Alongside the animals, ten per-frame classes describe what they are moving
through, over **301 flights and 29,832 frames**:

- ⚠️ **`environment`** — **experimental.** **snow**, **water**, **road**,
  **grass**, **rock**, **bare ground**, **roof** and **vehicle**, from SAM 3
  prompted with each class name. Machine-generated and **not reviewed by
  hand**; BAMBI has no ground truth for these classes, so no accuracy figures
  are given and none should be inferred. Treat it as a starting point to be
  checked, not as ground truth.
- ⚠️🔒 **`environment-nc`** — **experimental and non-commercial.** **tree
  cover** and standing **deadwood**. Same caveats, plus a CC-BY-NC licence.

**The per-frame masks are the raw model output and are not smoothed.** On
frames that are one material edge to edge — a flat snowfield, a fog whiteout —
detection is unstable and the masks **flicker** between adjacent frames. Each
flight ships a smoothed coverage series and three flags (`undetermined`,
`unstable`, `unreliable_classes`), but nothing is filled in and no mask pixel is
invented, so a reader who ignores the flags will see the flicker. Read
[the note on the masks](docs/environment.md#read-this-before-using-the-masks)
before training on them.

Across the release, 5.2% of frames are `undetermined` — too little visible
content to classify at all — and 3.1% carry at least one unstable class.

They are published as **two versions, split by licence**:

```bash
# snow, water, road, grass, rock, bare ground, roof, vehicle -- CC-BY-4.0
python download_from_zenodo.py --version environment -f 146 --unzip

# tree cover + deadwood -- CC-BY-NC-4.0, non-commercial use only
python download_from_zenodo.py --version environment-nc -f 146 --unzip

# both at once; prints a warning because the result is then partly NC
python download_from_zenodo.py --version environment-all -f 146 --unzip

# what every version is licensed under
python download_from_zenodo.py --licences
```

The split exists because the tree-cover and deadwood models are both built on
NVIDIA's **SegFormer**, licensed for research and evaluation only, while SAM 3
carries no non-commercial restriction. Putting them in one release would have
made the unrestricted classes non-commercial for no reason. The full reasoning,
including the non-obvious case where an MIT-licensed package carries an NVIDIA
carve-out internally, is in
[docs/dataset-versions.md](docs/dataset-versions.md#licensing-and-why-the-environment-layers-are-split-in-two).

See [docs/environment.md](docs/environment.md) for what each class means, how it
was produced, and where each one fails.

## Additional related repositories:

- [AlfsPY](https://github.com/bambi-eco/alfs_py): Framework for orthographic projections and light field renderings based on the drone recordings.
- [Detection](https://github.com/bambi-eco/bambi_detection): Examples on using AlfsPY for different tasks like geo-tiff generation.
- [Geo-Referenced Tracking](https://github.com/bambi-eco/Geo-Referenced-Tracking): Implementation of tracking algorithms based on local image as well as global world coordinates.
- [Bambi-QGIS](https://github.com/bambi-eco/Bambi-QGIS): Plugin for integrating drone video processing to the geo-information system QIGS.
- [BAMBI BBox Corrections](https://github.com/HugoMarkoff/BAMBI_BBox_Corrections): Template-matching toolkit for aligning thermal and RGB annotations.

---

## Citation

```bibtex
@misc{praschl2026bambi,
  title        = {The BAMBI Dataset: Multimodal Nadir UAV-Recordings of Forest Wildlife},
  author       = {Praschl, Christoph and Markoff, Hugo and Maschek, Anna and Jantsch, Wolfram and Wohlfahrt, Stephanie and Leitner, Horst and Beery, Sara and {\O}rsted, Michael and Schedl, David C.},
  year         = {2026},
  howpublished = {Presented at the CV4Animals Workshop, held in conjunction with the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  note         = {Non-archival workshop paper},
}
```

---

## License

This repository is licensed under the [MIT License](LICENSE).
