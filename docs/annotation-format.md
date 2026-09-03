# Annotation format

How a BAMBI annotation file is laid out, what each column means, and how
`class_id` is derived. The same format is used by every dataset version; see
[dataset-versions.md](dataset-versions.md) for which file names each version
ships.

## Species

The dataset covers the following 12 classes:

| # | Species | Common Name | Wikidata                                                                                                  | Tracks | Key Frames |
|---|---|---|-----------------------------------------------------------------------------------------------------------|---:|---:|
| 1 | *Sus scrofa* | Wild boar | [Q58697](https://www.wikidata.org/wiki/Q58697)                                                            | 1,770 | 26,132 |
| 2 | *Cervus elaphus* | Red deer | [Q1219579](https://www.wikidata.org/wiki/Q1219579)                                                        | 1,613 | 26,674 |
| 3 | *Capreolus capreolus* | Roe deer | [Q122069](https://www.wikidata.org/wiki/Q122069)                                                          | 682 | 9,404 |
| 4 | *Dama dama* | Fallow deer | [Q20908334](https://www.wikidata.org/wiki/Q20908334)                                                      | 297 | 20,003 |
| 5 | *Capra ibex* | Alpine ibex | [Q168327](https://www.wikidata.org/wiki/Q168327)                                                          | 100 | 3,005 |
| 6 | *Rupicapra rupicapra* | Chamois | [Q131340](https://www.wikidata.org/wiki/Q131340)                                                          | 15 | 747 |
| 7 | *Aves* | Bird | [Q5113](https://www.wikidata.org/wiki/Q5113)                                                              | 75 | 942 |
| 8 | *Homo sapiens* | Human | [Q15978631](https://www.wikidata.org/wiki/Q15978631)                                                      | 93 | 1,158 |
| 9 | *Canis lupus familiaris* | Dog | [Q26972265](https://www.wikidata.org/wiki/Q26972265)                                                      | 7 | 95 |
| 10 | *Sus scrofa × Sus domesticus* | Hybrid pig | [Q602666](https://www.wikidata.org/wiki/Q602666) (no matching wikidata id, so workaround with forest hog) | 44 | 1,484 |
| 11 | — | No-animal | [Q10738](https://www.wikidata.org/wiki/Q10738) (it is not an animal, it is a rock, THE ROCK)              | 60 | 521 |
| 12 | — | Unknown | [Q24238356](https://www.wikidata.org/wiki/Q24238356)                                                      | 344 | 2,536 |

## File layout

Annotations are stored as CSV files (no header) following a custom MOT format:

```
frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility, species, gender, age, is_propagated
```

| Column | Type | Description |
|---|---|---|
| `frame` | int | Frame index |
| `track_id` | int | Unique track identifier |
| `bb_left` | int | Bounding box left coordinate (px) |
| `bb_top` | int | Bounding box top coordinate (px) |
| `bb_width` | int | Bounding box width (px) |
| `bb_height` | int | Bounding box height (px) |
| `conf` | float | Confidence score |
| `class_id` | int | Species class identifier |
| `visibility` | float | Visibility (1.0 = fully visible, 0.0 = fully occluded) |
| `species` | str | Species name |
| `gender` | int | Gender (0 = unknown, 1 = male, 2 = female) |
| `age` | int | Age (0 = unknown, 1 = juvenile, 2 = adult) |
| `is_propagated` | int | 0 = annotated key frame, 1 = interpolated |

Class_ids are defined as `<wikidata_id>-<gender>-<age>-<visibility>` with the following mapping:
```json
{
  "class_mapping": {
    "Q10738-0-0-0": 0,
    "Q10738-0-0-1": 1,
    "Q1219579-0-0-0": 2,
    "Q1219579-0-0-1": 3,
    "Q1219579-0-1-0": 4,
    "Q1219579-0-1-1": 5,
    "Q1219579-0-2-0": 6,
    "Q1219579-0-2-1": 7,
    "Q1219579-1-0-0": 8,
    "Q1219579-1-0-1": 9,
    "Q1219579-1-2-0": 10,
    "Q1219579-1-2-1": 11,
    "Q1219579-2-0-0": 12,
    "Q1219579-2-0-1": 13,
    "Q1219579-2-1-0": 14,
    "Q1219579-2-2-0": 15,
    "Q1219579-2-2-1": 16,
    "Q122069-0-0-0": 17,
    "Q122069-0-0-1": 18,
    "Q122069-0-1-0": 19,
    "Q122069-0-1-1": 20,
    "Q122069-0-2-0": 21,
    "Q122069-0-2-1": 22,
    "Q131340-0-0-0": 23,
    "Q131340-0-0-1": 24,
    "Q131340-0-2-0": 25,
    "Q131340-0-2-1": 26,
    "Q15978631-0-0-0": 27,
    "Q15978631-0-0-1": 28,
    "Q15978631-0-2-0": 29,
    "Q15978631-0-2-1": 30,
    "Q15978631-2-0-0": 31,
    "Q15978631-2-2-0": 32,
    "Q15978631-2-2-1": 33,
    "Q168327-0-0-0": 34,
    "Q168327-0-0-1": 35,
    "Q168327-0-1-0": 36,
    "Q168327-0-1-1": 37,
    "Q168327-0-2-0": 38,
    "Q168327-0-2-1": 39,
    "Q168327-2-2-0": 40,
    "Q168327-2-2-1": 41,
    "Q20908334-0-0-0": 42,
    "Q20908334-0-1-0": 43,
    "Q20908334-1-0-0": 44,
    "Q20908334-1-1-0": 45,
    "Q20908334-1-2-0": 46,
    "Q20908334-2-2-0": 47,
    "Q20908334-2-2-1": 48,
    "Q24238356-0-0-0": 49,
    "Q24238356-0-0-1": 50,
    "Q24238356-0-2-0": 51,
    "Q24238356-0-2-1": 52,
    "Q26972265-0-0-0": 53,
    "Q26972265-0-2-0": 54,
    "Q26972265-0-2-1": 55,
    "Q5113-0-0-0": 56,
    "Q5113-0-0-1": 57,
    "Q5113-0-2-0": 58,
    "Q5113-0-2-1": 59,
    "Q58697-0-0-0": 60,
    "Q58697-0-0-1": 61,
    "Q58697-0-1-0": 62,
    "Q58697-0-1-1": 63,
    "Q58697-0-2-0": 64,
    "Q58697-0-2-1": 65,
    "Q58697-1-2-0": 66,
    "Q58697-1-2-1": 67,
    "Q58697-2-2-0": 68,
    "Q58697-2-2-1": 69,
    "Q602666-0-0-0": 70,
    "Q602666-0-0-1": 71,
    "Q602666-0-1-0": 72,
    "Q602666-0-2-0": 73,
    "Q602666-0-2-1": 74
  }
}
```

---
