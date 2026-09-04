# Environment annotations

> ⚠️ **Experimental.** These annotations are machine-generated and have not
> been reviewed by hand. BAMBI carries no ground truth for these classes, so no
> accuracy figures are given and none should be inferred. Coverage and content
> may change in a future revision, so cite the DOI you actually used.

Published over **301 flights and 29,832 frames**. The non-commercial layer
covers 294 of those flights and 27,613 frames. The seven missing flights
(56, 57, 221, 222, 223, 247, 254) did not fail: both models ran over every frame
and returned zero coverage throughout, so there was no mask to publish and no
file is written.

Where the animals are is only half of a scene. These layers describe what they
are moving through: whether the ground is under snow, where the water and the
tracks are, how much canopy is overhead, and where dead wood is standing.

They are annotation layers, not standalone releases — selecting one downloads
the `base` recordings alongside it, and everything lands in one directory.

## The layers

| classes | method | version | licence |
|---|---|---|---|
| snow, water, road, grass, rock, bare ground, roof, vehicle | [SAM 3](https://github.com/facebookresearch/sam3), prompted with each class name | `environment` | CC-BY-4.0 |
| tree cover | [Restor TCD](https://huggingface.co/restor/tcd-segformer-mit-b5) SegFormer-B5 | `environment-nc` | CC-BY-NC-4.0 |
| deadwood | [deadtrees.earth](https://github.com/cmosig/deadtreesmodels) SegFormer-B5 | `environment-nc` | CC-BY-NC-4.0 |

The split is by **licence**, not by subject. See
[dataset-versions.md](dataset-versions.md#licensing-and-why-the-environment-layers-are-split-in-two)
for why, and `--version environment-all` to fetch both.

## Read this before using the masks

**The per-frame masks flicker, and the published masks are not smoothed.**

SAM 3 is a promptable concept *detector*: it finds instances of a concept and
delimits them. A frame that is one material edge to edge offers no boundary to
find, and detection there becomes unstable. On one pilot flight over an almost
featureless snowfield, snow was found in 37 of 80 frames — at 98–99% coverage
when found and nothing when not, flipping between *adjacent* frames of a
slowly-moving drone. Lowering the confidence threshold does not help; it is not
a confidence problem.

Each flight therefore ships a smoothed coverage series and three flags
alongside the raw output. **The masks themselves are published exactly as the
model produced them.** Nothing is filled in, and no mask pixel is invented — so
a user who reads raw per-frame masks and ignores the flags will still see the
flicker. If you need a stable per-frame signal, use `smoothed`, or drop frames
where the class is flagged, or aggregate over a window yourself.

### The flags

| flag | scope | meaning |
|---|---|---|
| `undetermined` | frame | Luminance dynamic range (p98 − p2) below 90. The frame has no visible content; nothing is claimed about it. |
| `unstable` | frame × class | This frame's coverage differs from its temporal neighbourhood by more than 0.25, so it is more likely a detector dropout than a real change. |
| `unreliable_classes` | flight | That class flips presence between consecutive frames more than 10% of the time across the flight. |

Across the release (301 flights, 29,832 frames): **1,558 frames (5.2%)** are
`undetermined`, **921 frames (3.1%)** carry at least one `unstable` class, and
**171 class/flight pairs** are marked unreliable. On the 15-flight pilot,
rolling-median smoothing halved the mean flicker rate, from 1.7% to 0.7%.

The `undetermined` threshold is measured, not guessed. Two pilot flights are
featureless — one a fog whiteout, one a flat snowfield — and their luminance
range is 66 and 18 against 125 to 200 for every normal flight. Both are excluded
in full. That does discard a correct answer on the snowfield, which really is
snow: a person would say so from context. But no rule separates it from the fog
flight, where the same emptiness produced a confident and wrong `water` label
over the whole frame, so both are declined.

The unreliable flag falls overwhelmingly on two classes. Of the 171
class/flight pairs it marks, `bare ground` accounts for 77 and `grass` for 74 —
88% between them — against 9 for `road`, 5 for `snow`, 4 for `roof` and 2 for
`water`. `rock` and `vehicle` never trip it at all. `bare ground` and `grass`
are the classes whose boundaries are genuinely gradual — there is no crisp
answer to where grass ends — so on those two the flag describes a property of
the class as much as a failure of the model. On the other six it is rarer, and
more likely to mean the detector actually dropped out.

For reference, the number of flights on which each class fires at least once:
`grass` 224, `bare ground` 206, `rock` 137, `road` 104, `roof` 103, `water` 93,
`snow` 56, `vehicle` 46.

## Resolution, and why it shapes everything

BAMBI frames are far higher resolution than the imagery these models were built
for. Estimated from 96,746 annotated animal boxes across seven species of known
body length, the ground sampling distance is a **median of 3.5 cm/px** (p10 2.3,
p90 4.8; per flight 1.4 to 9.8). A 1024 px frame therefore covers only about
**36 × 36 m**.

The two SegFormer layers are resampled to the resolution their models expect —
10 cm for tree cover, 5 cm for deadwood — rather than run at native scale, which
neither has seen. SAM 3 runs at native resolution.

That 36 m footprint is also why there is no land-cover-style class scheme: most
frames contain no road and no water at all, and a model built for 20–50 cm
country-scale mapping sees something quite different from a 3.5 cm nadir frame.

## The letterbox

A BAMBI RGB frame is 16:9 content letterboxed into a 1024 × 1024 square, so
roughly the top and bottom 100 rows are black, and the exact extent varies per
flight. Every layer here excludes those bands: they are not imagery, and a
segmentation model fed black bars predicts confidently inside them. An early
deadwood run reported 8.2% coverage on a flight with no dead wood at all, and
all of it was in the letterbox.

Coverage fractions are reported over the **imaged area**, not the padded square,
so flights with different letterboxing stay comparable.

## Choosing the class names

The class names are SAM 3 prompts, so they were chosen by measurement rather
than taste. Sweeping candidates over 30 flights:

* `meadow` and `mud` never fired once, on any flight. Dropped.
* `grass` fires on 24 of 30 flights; `"meadow, grass"` gives a mask with **IoU
  0.977** against plain `grass`, so the compound phrase adds nothing.
* `roof` fires on 6 flights against 1 for `building` — from nadir you see roofs,
  not buildings. Note it also finds *car* roofs, which is literally correct;
  `vehicle` exists to separate them.
* `bare ground` and `rock` overlap at **IoU 0.002** — they are cleanly disjoint
  rather than two names for one region.

## Where each layer fails

**Tree cover** is robust across seasons — closed summer canopy, bare winter
trees and snow alike. Its weakness is granularity: at 10 cm inference with
SegFormer's quarter-resolution logits, effective mask resolution is about 40 cm
on the ground, so it gives smooth blobs rather than crown outlines. Good for
"how much canopy", not "which tree".

**Deadwood** flags standing dead wood in the canopy, and marks bare grey branch
structure against green canopy where one would expect it. It has had the least
validation of the layers here.

**`road`** is the least reliable of the SAM 3 classes. It is right where there
is a real surface — a gravel yard came out at 19.5% — but on snow-covered ground
it has been seen to segment broad shadowed bands while missing the actual visible
tracks.

**`snow` and `water` can be confused on featureless frames**, which is what the
`undetermined` flag exists to catch.

## Validation

There is no ground truth for these classes in BAMBI, so unlike the transferred
RGB annotations there is nothing to score against. These layers are **reviewed
qualitatively only** and carry no accuracy figures. The flags above report
self-consistency — agreement between a frame and its neighbours, or between two
layers — and self-consistency is not accuracy.
