# Environment annotations

> **Status: in preparation.** The methods below are implemented and validated on
> sample flights, but the Zenodo records are not published yet, so
> `--version environment` will report a missing summary file until they are.
> This page documents what the layers will contain.

Where the animals are is only half of a scene. These layers describe what they
are moving through: how much canopy is overhead, where the individual trees
stand, whether the ground is under snow, and where dead wood is standing.

They are annotation layers, not standalone releases — selecting one downloads
the `base` recordings alongside it, and everything lands in one directory.

## The four layers

| class | method | what it gives you | version |
|---|---|---|---|
| tree positions | DeepForest, release model | one box and centre point per tree | `environment` |
| snow | brightness/saturation threshold | binary mask | `environment` |
| tree cover | Restor TCD SegFormer-B5 | binary canopy mask | `environment-nc` |
| deadwood | deadtrees.earth SegFormer-B5 | binary mask of standing dead wood | `environment-nc` |

The split is by **licence**, not by subject: the two SegFormer-based models are
restricted to non-commercial use. See
[dataset-versions.md](dataset-versions.md#licensing-and-why-the-environment-layers-are-split-in-two)
for why, and what that means in practice.

## Resolution, and why it matters here

BAMBI frames are far higher resolution than the imagery these models were built
for. Estimated from 96,746 annotated animal boxes across seven species of known
body length, the ground sampling distance is a **median of 3.5 cm/px** (p10 2.3,
p90 4.8; per flight 1.4 to 9.8). A 1024 px frame therefore covers only about
**36 x 36 m**.

Both the Restor and DeepForest models are trained at **10 cm/px**, and the
deadwood model downsamples anything finer to 5 cm. Each layer is therefore
resampled to the resolution its model expects rather than being run at native
scale, which a network trained at 10 cm has never seen.

That 36 m footprint is also why there is no "road" or "river" class. Land-cover
models assume a far wider view; at this scale most frames contain no road and no
water at all, so the classes would be empty almost everywhere.

## The letterbox

A BAMBI RGB frame is 16:9 content letterboxed into a 1024 x 1024 square, so
roughly the top and bottom 100 rows are black, and the exact extent varies per
flight. Every layer here excludes those bands: they are not imagery, and a
segmentation model fed black bars predicts confidently inside them. The first
deadwood run reported 8.2% coverage on a flight that has no dead wood at all,
and all of it was in the letterbox.

Coverage fractions are therefore reported over the **imaged area**, not the
padded square, so flights with different letterboxing stay comparable.

## What each layer is good for, and where it fails

**Tree cover** is robust across seasons — it handles closed summer canopy, bare
winter trees and snow alike. Its weakness is granularity: at 10 cm inference
with SegFormer's quarter-resolution logits, the effective mask resolution is
about 40 cm on the ground, so it produces smooth blobs rather than crown
outlines. Good for "how much canopy", not for "which tree".

**Tree positions** are convincing on green summer canopy and clearly
under-detect on leaf-off and snow-covered flights, which are outside DeepForest's
NEON training distribution. On one January flight it found 7 trees where the
tree-cover layer correctly marked 34% of the frame as canopy. Treat position
counts from winter flights as a lower bound.

Because the two layers disagree in a predictable direction, they cross-check
each other: **a flight with high tree cover but few detected positions is
unreliable for positions**, and that comparison costs nothing to compute.

**Snow** is the most reliable layer, and the only one that uses no learned model
at all. Snow in RGB is bright and almost completely desaturated, which nothing
else in a nadir forest scene reliably is. The rule is `V > 140 and S < 40` in
HSV, followed by morphological cleanup and a minimum-area filter. Thresholds
were measured, not guessed: over sample frames, the snow flights peak at
saturation p90 of 30 and 38, while the snow-free flights start at p10 of 42 to
58. Measured coverage is 99.6% on an open snow field, 66-70% on a snow flight
with scattered trees, and 0.000% on closed summer canopy.

Its failure mode is honest and predictable: it means "bright achromatic
surface", so overexposed limestone, scree and white gravel tracks read as snow.
An alpine September flight sits at about 2% for this reason.

**Deadwood** flags standing dead wood in the canopy. It has been checked on
sample flights, where it marks bare grey branch structure against green canopy,
but it has had the least validation of the four.

## Validation

There is no ground truth for these classes in BAMBI, so unlike the transferred
RGB annotations there is nothing to score against. These layers are **reviewed
qualitatively only** and carry no accuracy figures. Where two layers can be
compared — snow against tree cover, positions against cover — their agreement is
reported, but agreement is not accuracy.
