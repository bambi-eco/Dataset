"""Reproduce the BAMBI `environment` annotation layers on any frame.

The published layers cover the **key frames** of each flight -- the frames that
carry animal annotations in `<id>_gt.txt` -- so that an environment mask can be
read alongside an animal box without interpolating anything. That is a few
dozen to a few hundred frames out of the ten thousand or so in a flight. If you
want the classes on the frames in between, you have to run the models yourself,
and this module is what the release was produced with.

Three models, in two licence tiers:

| classes | model | tier |
|---|---|---|
| snow, water, road, grass, rock, bare ground, roof, vehicle | SAM 3, prompted with each class name | `environment`, CC-BY-4.0 |
| tree cover | `restor/tcd-segformer-mit-b5` | `environment-nc`, CC-BY-NC-4.0 |
| deadwood | deadtrees.earth SegFormer-B5 | `environment-nc`, CC-BY-NC-4.0 |

Both SegFormer models inherit NVIDIA's Source Code License through their
encoder, which permits research and evaluation use only. That is why the
release is split in two, and it applies to anything you produce with them here.

Two details matter more than the model choice, and getting either wrong
produces confident nonsense rather than an obvious failure:

* **The letterbox.** A BAMBI RGB frame is 16:9 content in a 1024x1024 square,
  so roughly the top and bottom 100 rows are black and the exact extent varies
  per flight. Fed the whole square, a segmenter predicts inside the black
  bands: an early deadwood run reported 8.2% coverage on a flight with no dead
  wood in it, all of it letterbox. `valid_rows` finds the imaged strip, and
  every model here runs on that strip alone.
* **The scale.** BAMBI is far finer than the imagery these models were
  trained on -- a median of about 3.5 cm/px against 10 cm for tree cover and
  5 cm for deadwood -- so the two SegFormer models are resampled to the
  resolution they expect. SAM 3 runs at native resolution.

The heavy dependencies are imported inside the classes that need them, so
importing this module costs nothing and the analysis helpers below work with
numpy alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "SAM3_CLASSES", "NC_CLASSES",
    "valid_rows", "coverage",
    "Sam3Segmenter", "TreeCoverSegmenter", "DeadwoodSegmenter",
    "rle_encode", "rle_decode", "iou",
    "rolling_median", "flicker_rate", "dynamic_range", "flag_series",
    "load_masks",
]

SAM3_CLASSES = ["snow", "water", "road", "grass", "rock", "bare ground",
                "roof", "vehicle"]
NC_CLASSES = ["tree cover", "deadwood"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------
# the letterbox
# --------------------------------------------------------------------------
def valid_rows(img: np.ndarray, thresh: int = 12) -> tuple[int, int]:
    """The half-open row range holding imagery, excluding the letterbox bands.

    A row is imagery if any pixel in it exceeds `thresh`. The bands are not
    exactly zero -- video compression leaves a little noise in them -- so a
    small threshold rather than a test against 0.
    """
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    rowmax = g.max(axis=1)
    if not (rowmax > thresh).any():
        return 0, img.shape[0]
    return (int(np.argmax(rowmax > thresh)),
            img.shape[0] - int(np.argmax(rowmax[::-1] > thresh)))


def coverage(mask: np.ndarray, rows: tuple[int, int] | None = None) -> float:
    """Fraction of the *imaged* area covered, not of the padded square.

    Reporting over the full 1024x1024 would make flights with different
    letterboxing incomparable, and would understate every coverage by the
    fraction of the frame that is black bar.
    """
    if rows is None:
        return float(mask.mean())
    top, bot = rows
    denom = max((bot - top) * mask.shape[1], 1)
    return float(mask[top:bot].sum()) / denom


# --------------------------------------------------------------------------
# the models
# --------------------------------------------------------------------------
class Sam3Segmenter:
    """SAM 3 prompted with a class name, one binary mask per class.

    SAM 3 is a promptable concept *detector*: it finds instances of a concept
    and delimits them, so the union of its instances is taken as the class
    mask. The backbone is encoded once per frame and each prompt is then a
    cheap pass over the cached features, so eight classes cost far less than
    eight times one class.

    The checkpoint is gated on HuggingFace; `checkpoint` is a local path to it.
    """

    def __init__(self, checkpoint, device=None, confidence=0.5):
        import os

        import sam3
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        pkg = os.path.dirname(sam3.__file__)
        bpe = next((c for c in
                    [os.path.join(pkg, "assets", "bpe_simple_vocab_16e6.txt.gz"),
                     os.path.join(pkg, "..", "assets",
                                  "bpe_simple_vocab_16e6.txt.gz")]
                    if os.path.exists(c)), None)
        if bpe is None:
            raise RuntimeError("no BPE vocabulary found in the sam3 package")

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam3_image_model(bpe_path=bpe, device=self.device,
                                            checkpoint_path=str(checkpoint),
                                            load_from_HF=False)
        self.proc = Sam3Processor(self.model, confidence_threshold=confidence)

    def __call__(self, img_bgr: np.ndarray,
                 classes=SAM3_CLASSES) -> dict[str, np.ndarray]:
        import cv2
        from PIL import Image

        torch = self.torch
        top, bot = valid_rows(img_bgr)
        strip = cv2.cvtColor(img_bgr[top:bot], cv2.COLOR_BGR2RGB)

        # SAM 3 wraps its backbone in bfloat16 autocast. On a card without
        # native bfloat16 this is emulated -- correct, only slower.
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device == "cuda"
                    else torch.autocast("cpu", enabled=False))

        out = {}
        with autocast:
            state = self.proc.set_image(Image.fromarray(strip))
            for name in classes:
                self.proc.reset_all_prompts(state)
                st = self.proc.set_text_prompt(state=state, prompt=name)
                masks = st.get("masks")
                if masks is None or len(masks) == 0:
                    continue
                m = masks.squeeze(1).any(0).to(torch.uint8).cpu().numpy()
                full = np.zeros(img_bgr.shape[:2], np.uint8)
                full[top:bot] = m[: bot - top]
                if full.any():
                    out[name] = full
        return out


class _SegformerBase:
    """Shared plumbing: crop the letterbox, resample to the model's GSD,
    predict, then put the result back on the full frame."""

    target_gsd_cm = 0.0
    class_name = ""

    def _prepare(self, img_bgr, gsd_cm):
        import cv2

        top, bot = valid_rows(img_bgr)
        strip = img_bgr[top:bot]
        scale = (gsd_cm / self.target_gsd_cm) if self.target_gsd_cm else 1.0
        if scale != 1.0:
            strip = cv2.resize(
                strip, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        return strip, (top, bot), scale

    @staticmethod
    def _restore(prob_small, shape, rows):
        """Probabilities on the resampled strip -> a full-frame mask.

        Bilinear back to the strip's own size, then written into a zero frame,
        so the letterbox comes out as not-predicted rather than as background.
        """
        import cv2

        top, bot = rows
        prob = cv2.resize(prob_small, (shape[1], bot - top),
                          interpolation=cv2.INTER_LINEAR)
        full = np.zeros(shape[:2], np.float32)
        full[top:bot] = prob
        return full


class TreeCoverSegmenter(_SegformerBase):
    """Semantic tree cover from `restor/tcd-segformer-mit-b5`.

    Tree / no-tree per pixel, not individual crowns. Inference at 10 cm/px and
    thresholding the confidence map rather than taking the argmax both follow
    the deadtrees project's own tree-cover config.

    NVIDIA Source Code License via the MiT encoder: research and evaluation
    use only.
    """

    target_gsd_cm = 10.0
    class_name = "tree cover"

    def __init__(self, model="restor/tcd-segformer-mit-b5", device=None):
        import torch
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model).to(self.device).eval()

    def __call__(self, img_bgr, gsd_cm=3.5, threshold=0.5):
        import cv2

        torch = self.torch
        strip, rows, _ = self._prepare(img_bgr, gsd_cm)
        rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits          # (1, C, h/4, w/4)
        prob = torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()
        full = self._restore(prob, img_bgr.shape, rows)
        return (full > threshold).astype(np.uint8)


class DeadwoodSegmenter(_SegformerBase):
    """Standing deadwood from the deadtrees.earth model.

    Upstream runs this over georeferenced orthophotos; none of that geo
    plumbing applies to a video frame, but the model underneath is an ordinary
    binary `smp.Unet` with a MiT-B5 encoder, so it is rebuilt and fed frames
    directly. Upstream never lets the network see finer than 5 cm/px, and that
    is reproduced here.

    The checkpoint was saved from a `torch.compile`d model, so its keys carry
    an `_orig_mod.` prefix that has to be stripped.

    NVIDIA Source Code License via the MiT encoder: research and evaluation
    use only.
    """

    target_gsd_cm = 5.0
    class_name = "deadwood"

    def __init__(self, checkpoint, device=None):
        import segmentation_models_pytorch as smp
        import torch
        from safetensors.torch import load_file

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet(encoder_name="mit_b5", encoder_weights=None,
                         in_channels=3, classes=1)
        state = {k.replace("_orig_mod.", "", 1): v
                 for k, v in load_file(str(checkpoint)).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  warning: {len(missing)} missing / "
                  f"{len(unexpected)} unexpected key(s)")
        self.model = model.to(self.device).eval()

    def __call__(self, img_bgr, gsd_cm=3.5, threshold=0.5):
        import cv2

        torch = self.torch
        strip, rows, _ = self._prepare(img_bgr, gsd_cm)

        # The U-Net downsamples five times, so both sides must be divisible by
        # 32; resampling rarely lands on such a size. Replicate the edge rather
        # than padding with zeros, which would look like a hard black border to
        # a network that has never seen one.
        h, w = strip.shape[:2]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            strip = cv2.copyMakeBorder(strip, 0, ph, 0, pw, cv2.BORDER_REPLICATE)

        rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(t))[0, 0].float().cpu().numpy()

        full = self._restore(prob[:h, :w], img_bgr.shape, rows)
        return (full > threshold).astype(np.uint8)


# --------------------------------------------------------------------------
# the published mask format
# --------------------------------------------------------------------------
def rle_encode(mask: np.ndarray) -> dict:
    """Binary mask -> COCO run-length encoding.

    Column-major, and the first run is always of zeros -- a mask whose very
    first pixel is set therefore starts with a run of length 0. This is the
    format the published `<id>_environment.json` files use.
    """
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    if flat.size == 0:
        return {"size": list(mask.shape), "counts": []}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0] == 1:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    """COCO run-length encoding -> boolean mask."""
    h, w = rle["size"]
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for n in rle["counts"]:
        if val:
            flat[pos:pos + n] = 1
        pos += n
        val ^= 1
    return flat.reshape((h, w), order="F").astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


# --------------------------------------------------------------------------
# stability over a series of frames
# --------------------------------------------------------------------------
def dynamic_range(img_bgr: np.ndarray) -> float:
    """Luminance p98 - p2 over the imaged strip.

    The statistic the `undetermined` flag is built on. A frame that is one
    material edge to edge gives a detector no boundary to find, and this
    separates such frames from normal ones cleanly: the two featureless pilot
    flights measure 18 and 66, every normal flight 125 and up.
    """
    import cv2

    top, bot = valid_rows(img_bgr)
    g = cv2.cvtColor(img_bgr[top:bot], cv2.COLOR_BGR2GRAY)
    return float(np.percentile(g, 98) - np.percentile(g, 2))


def rolling_median(frames: np.ndarray, values: np.ndarray,
                   half_window: float = 120.0) -> np.ndarray:
    """Median of the values whose frame index is within +/- half_window.

    Indexed by frame rather than by sample, because key frames are irregularly
    spaced and what matters is temporal locality, not how many samples happen
    to lie nearby. Run this over a dense series and the window covers far more
    samples than it does over key frames alone -- which is the main practical
    reason to compute in-between frames at all.
    """
    frames = np.asarray(frames, float)
    values = np.asarray(values, float)
    out = np.empty(len(frames))
    lo = np.searchsorted(frames, frames - half_window, side="left")
    hi = np.searchsorted(frames, frames + half_window, side="right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        out[i] = np.median(values[a:b]) if b > a else values[i]
    return out


def flicker_rate(values: np.ndarray, thresh: float = 0.05) -> float:
    """How often presence flips between consecutive samples."""
    on = np.asarray(values, float) > thresh
    return float((on[1:] != on[:-1]).mean()) if len(on) > 1 else 0.0


def flag_series(frames, values, ranges=None, half_window=120.0,
                min_range=90.0, unstable_delta=0.25,
                unreliable_flicker=0.10) -> dict:
    """The three published flags for one class over one series of frames.

    Returns `smoothed`, the per-frame booleans `undetermined` and `unstable`,
    and the flight-level `unreliable`. Thresholds default to the ones the
    release used.
    """
    frames = np.asarray(frames, float)
    values = np.asarray(values, float)
    smoothed = rolling_median(frames, values, half_window)
    undetermined = (np.asarray(ranges, float) < min_range if ranges is not None
                    else np.zeros(len(frames), bool))
    return {
        "smoothed": smoothed,
        "undetermined": undetermined,
        "unstable": np.abs(values - smoothed) > unstable_delta,
        "flicker": flicker_rate(values),
        "unreliable": flicker_rate(values) > unreliable_flicker,
    }


# --------------------------------------------------------------------------
# reading the shipped example
# --------------------------------------------------------------------------
def load_masks(path, decode: bool = True) -> dict:
    """Read a mask document, plain or gzipped.

    Reads both the shipped `examples/environment/*.json.gz` and a published
    `<id>_environment.json`, whose frames are a list rather than a mapping.
    With `decode`, every RLE becomes a boolean array; either way the result is
    keyed `masks[frame_number][class]`.
    """
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            doc = json.load(fh)
    else:
        doc = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(doc.get("frames"), list):        # a published release file
        doc["masks"] = {int(f["frame"]): f["masks"] for f in doc["frames"]}
    doc["masks"] = {int(f): {c: (rle_decode(r) if decode else r)
                             for c, r in per.items()}
                    for f, per in doc["masks"].items()}
    return doc
