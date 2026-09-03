# Working with annotations

Interpolating, filtering and converting MOT files. For what the columns mean,
see [annotation-format.md](annotation-format.md).

## MOT Interpolation

Interpolate missing frames between annotated key frames using linear interpolation of bounding box coordinates. Interpolated entries are marked with `is_propagated=1`.

```bash
# Interpolate a single annotation file
python mot_interpolation.py annotations.txt ./output

# Interpolate all files in a folder with a custom frame step
python mot_interpolation.py ./annotations_folder ./output --step 2
```

## MOT Filter

Filter annotation files by species, class ID, visibility, bounding box size, gender, or age.

```bash
# Keep only wild boar annotations
python mot_filter.py annotations.txt -o ./filtered --species "Sus scrofa (Wild boar)"

# Filter by class ID and minimum bounding box width
python mot_filter.py ./annotations/ -o ./filtered --class-id 50 51 --min-width 25

# Combine multiple filters
python mot_filter.py annotations.txt -o ./filtered --species "Cervus elaphus (Red deer)" --visibility 1.0 --min-width 30
```

## MOT to YOLO Conversion

Convert MOT annotations to YOLO label format for training object detection models.

```bash
# Convert with default label (class_id only)
python mot_to_yolo.py annotations.txt -o ./yolo_labels

# Include species, gender, and age in the label
python mot_to_yolo.py ./annotations/ -o ./yolo_labels --img-width 640 --img-height 512 --labels class_id gender age
```
