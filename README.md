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
| [Dataset versions](docs/dataset-versions.md) | What `base`, `raw`, `matched`, `orthographic` and `owl-transferred` contain, and the Zenodo DOIs |
| [Annotation format](docs/annotation-format.md) | The MOT columns, the species table, and how `class_id` is composed |
| [Downloading](docs/downloading.md) | Installing, filtering flights by metadata, and fetching them |
| [Working with annotations](docs/annotation-tools.md) | Interpolating, filtering, and converting to YOLO |
| [Frames and visualization](docs/frames-and-visualization.md) | Extracting frames, drawing boxes on images and video |
| [Thermal to RGB label transfer](docs/label-transfer.md) | Moving thermal boxes onto the RGB view, and the released `owl-transferred` annotations |
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
- **`owl-transferred`** — RGB boxes for 238 of the 386 flights, produced by
  detecting the animals in RGB with OWL and re-centring each thermal box on its
  match. Machine-generated and not reviewed by hand; 5.22 px mean centre error
  against the accepted annotations, against 15.98 px for leaving the thermal
  boxes alone.

```bash
# the recordings and the transferred RGB annotations, in one folder
python download_from_zenodo.py --version owl-transferred -f 146 --unzip
```

See [docs/label-transfer.md](docs/label-transfer.md).

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
