#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Save a 1x5 grid: original / global / fg_only / bg_only / fgbg.

A4 ``fgbg`` uses FG×N slot pick + random complex BG (not same-sample bg).
Other columns use ``augment()``. Change variables in ``__main__``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fgbg_aug import (
    FgSpec,
    GlobalSpec,
    augment,
    augment_fgbg_pick,
    foreground_l2,
    list_by_class_samples,
    load_by_class_sample,
)
from platform_config import BY_CLASS_ROOT

MODES = ("none", "global", "fg_only", "bg_only", "fgbg")
PHOTO_FG = FgSpec(rotate_deg=0.0)
PHOTO_GLOBAL = GlobalSpec(rotate_deg=0.0, scale=0.0, shear_deg=0.0)


def _label(img: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 28), (255, 255, 255))
    canvas.paste(img, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), text, fill=(0, 0, 0))
    return canvas


def render_row(
    root: Path,
    samples: list[tuple[str, str]],
    idx: int,
    seed: int,
    img_size: int,
    *,
    fg_aug_count: int = 4,
    fg_aug_pick: int = 0,
    other_image_prob: float = 0.65,
) -> tuple[Image.Image, dict]:
    breed, stem = samples[idx]
    exclude = (breed, stem)
    img, mask, _ = load_by_class_sample(root, breed, stem)
    bg_breed, bg_stem = samples[(idx + 31) % len(samples)]
    bg, _, _ = load_by_class_sample(root, bg_breed, bg_stem)
    orig = np.array(img.resize((img_size, img_size), Image.BILINEAR))
    mask_r = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((img_size, img_size), Image.NEAREST)) > 0
    tiles = []
    stats = {}
    for mode in MODES:
        rng = np.random.default_rng(seed + idx)
        if mode == "none":
            vis = img.resize((img_size, img_size), Image.BILINEAR)
            photo = vis
        elif mode == "fgbg":
            vis = augment_fgbg_pick(
                img,
                mask,
                rng,
                root=root,
                samples=samples,
                exclude=exclude,
                n_slots=fg_aug_count,
                pick=fg_aug_pick,
                other_image_prob=other_image_prob,
            ).resize((img_size, img_size), Image.BILINEAR)
            rng_photo = np.random.default_rng(seed + idx)
            photo = augment_fgbg_pick(
                img,
                mask,
                rng_photo,
                root=root,
                samples=samples,
                exclude=exclude,
                n_slots=fg_aug_count,
                pick=fg_aug_pick,
                fg_spec=PHOTO_FG,
                other_image_prob=other_image_prob,
            ).resize((img_size, img_size), Image.BILINEAR)
        else:
            vis = augment(img, mask, rng, mode, bg_image=bg).resize((img_size, img_size), Image.BILINEAR)
            rng_photo = np.random.default_rng(seed + idx)
            photo = augment(
                img,
                mask,
                rng_photo,
                mode,
                bg_image=bg,
                fg_spec=PHOTO_FG,
                global_spec=PHOTO_GLOBAL,
            ).resize((img_size, img_size), Image.BILINEAR)
        stats[mode] = round(foreground_l2(orig, np.array(photo), mask_r), 2)
        tiles.append(_label(vis, f"{mode}  photo-L2={stats[mode]}"))
    w, h = tiles[0].size
    row = Image.new("RGB", (w * len(tiles), h), (255, 255, 255))
    for i, t in enumerate(tiles):
        row.paste(t, (i * w, 0))
    return row, stats


if __name__ == "__main__":
    # Source by_class：platform_config.py
    OUT_DIR = Path(f"{BY_CLASS_ROOT}-viz")  # mechanism grid → OUT_DIR/pet_aug_grid.png
    N_IMAGES = 6  # how many sample rows in the grid
    IMG_SIZE = 224  # tile side length (px) after resize
    SEED = 0  # RNG seed; same seed → reproducible aug per row
    # A4 fgbg only: generate FG_AUG_COUNT slot augs, show FG_AUG_PICK (0-based).
    FG_AUG_COUNT = 4
    FG_AUG_PICK = 0
    # Prob of sampling BG from another by_class image; else solid+noise+stripes.
    OTHER_IMAGE_PROB = 0.65

    samples = list_by_class_samples(BY_CLASS_ROOT)
    if not samples:
        raise SystemExit(f"no samples under {BY_CLASS_ROOT / 'image'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stride = max(1, len(samples) // (N_IMAGES + 2))
    rows = []
    all_stats = []
    for k in range(N_IMAGES):
        idx = min((k + 1) * stride, len(samples) - 1)
        row, stats = render_row(
            BY_CLASS_ROOT,
            samples,
            idx,
            SEED,
            IMG_SIZE,
            fg_aug_count=FG_AUG_COUNT,
            fg_aug_pick=FG_AUG_PICK,
            other_image_prob=OTHER_IMAGE_PROB,
        )
        rows.append(row)
        all_stats.append({"idx": idx, "sample": f"{samples[idx][0]}/{samples[idx][1]}", **stats})
        print(f"idx={idx} {samples[idx][0]}/{samples[idx][1]} {stats}")
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), (255, 255, 255))
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * row.height))
    out_path = OUT_DIR / "pet_aug_grid.png"
    canvas.save(out_path)
    print(f"wrote {out_path}")
    print(
        "photo-L2: RGB L2 on by_class FG alpha, geometry off. "
        "fg_only/fgbg stay relatively low on FG; A1 is mild global (not harmful). "
        "A4 fgbg = FG×N pick + random complex BG (not same-sample bg). "
        "FG/global rotate uses expand so subject is not clipped."
    )
