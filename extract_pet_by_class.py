#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract Oxford-IIIT Pet into by_class (pipeline step ①).

Layout::

    <out>/
      image/<breed>/<id>.jpg   # RGB crop (trimap body bbox, pixels 1|3)
      fg/<breed>/<id>.png      # RGBA crop: keep trimap==1, else transparent
      bg/<breed>/<id>.png      # RGBA crop: keep trimap==2, else transparent
      bbox/<breed>/<id>.jpg    # full RGB with body bbox drawn

Next: ``split_by_class.py`` (②) → ``dump_aug.py`` (③) / ``compare.py`` (④).
Trimap pixel 3 (uncertain) is transparent in both fg and bg. Change variables
under ``if __name__ == "__main__"`` then run.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from platform_config import BY_CLASS_ROOT, MAX_CLASSES, MAX_PER_CLASS, PET_ROOT

SUBSETS = ("image", "fg", "bg", "bbox")


def breed_from_stem(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def cap_image_paths(
    paths: list[Path],
    *,
    max_classes: int,
    max_per_class: int,
) -> list[Path]:
    """Mac smoke / optional subset: first N breeds, then ≤K images per breed."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        buckets[breed_from_stem(p.stem)].append(p)
    breeds = sorted(buckets)
    if max_classes > 0:
        breeds = breeds[: int(max_classes)]
    out: list[Path] = []
    for b in breeds:
        items = sorted(buckets[b], key=lambda x: x.stem)
        if max_per_class > 0:
            items = items[: int(max_per_class)]
        out.extend(items)
    return out


def body_bbox(trimap: np.ndarray, pad_ratio: float) -> tuple[int, int, int, int] | None:
    """BBox from trimap body (1|3). Returns (left, top, right, bottom) xyxy, or None."""
    body = (trimap == 1) | (trimap == 3)
    ys, xs = np.where(body)
    if len(xs) == 0:
        return None
    h, w = trimap.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in exts and not p.name.startswith("._")
    )


def draw_bbox(img: Image.Image, box: tuple[int, int, int, int], width: int = 3) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = box
    # PIL rectangle is inclusive; shrink right/bottom by 1 for visual edge
    draw.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=(0, 255, 0), width=width)
    return out


def rgba_keep(crop_rgb: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Build RGBA from RGB crop; alpha=255 where keep, else 0 (transparent)."""
    h, w = keep.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = crop_rgb
    out[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
    return out


def extract(
    pet_root: Path,
    out_dir: Path,
    *,
    pad_ratio: float = 0.05,
    skip_existing: bool = True,
    max_classes: int = 0,
    max_per_class: int = 0,
) -> dict:
    images_dir = pet_root / "images"
    trimap_dir = pet_root / "annotations" / "trimaps"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing images: {images_dir}")
    if not trimap_dir.is_dir():
        raise FileNotFoundError(f"missing trimaps: {trimap_dir}")

    paths_all = list_images(images_dir)
    paths = cap_image_paths(
        paths_all, max_classes=max_classes, max_per_class=max_per_class
    )
    for name in SUBSETS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    skipped = 0
    no_fg = 0
    missing_trimap = 0

    for img_path in tqdm(paths, desc="extract pet"):
        breed = breed_from_stem(img_path.stem)
        stem = img_path.stem
        paths_out = {
            "image": out_dir / "image" / breed / f"{stem}.jpg",
            "fg": out_dir / "fg" / breed / f"{stem}.png",
            "bg": out_dir / "bg" / breed / f"{stem}.png",
            "bbox": out_dir / "bbox" / breed / f"{stem}.jpg",
        }
        if skip_existing and all(p.is_file() for p in paths_out.values()):
            skipped += 1
            counts[breed] += 1
            continue

        trimap_path = trimap_dir / f"{stem}.png"
        if not trimap_path.is_file():
            missing_trimap += 1
            continue

        img = Image.open(img_path).convert("RGB")
        trimap = np.array(Image.open(trimap_path))
        if trimap.ndim == 3:
            trimap = trimap[..., 0]

        box = body_bbox(trimap, pad_ratio)
        if box is None:
            no_fg += 1
            box = (0, 0, img.width, img.height)

        x0, y0, x1, y1 = box
        crop = np.array(img.crop(box))
        trimap_crop = trimap[y0:y1, x0:x1]
        fg = rgba_keep(crop, trimap_crop == 1)
        bg = rgba_keep(crop, trimap_crop == 2)
        bbox_vis = draw_bbox(img, box)

        for p in paths_out.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(paths_out["image"], quality=95)
        Image.fromarray(fg, mode="RGBA").save(paths_out["fg"])
        Image.fromarray(bg, mode="RGBA").save(paths_out["bg"])
        bbox_vis.save(paths_out["bbox"], quality=95)
        counts[breed] += 1

    summary = {
        "n_images_all": len(paths_all),
        "n_images": len(paths),
        "n_saved": int(sum(counts.values())),
        "n_classes": len(counts),
        "max_classes": int(max_classes),
        "max_per_class": int(max_per_class),
        "skipped_existing": skipped,
        "missing_trimap": missing_trimap,
        "no_fg_fallback_full": no_fg,
        "out_dir": str(out_dir),
        "layout": {s: str(out_dir / s / "<breed>") for s in SUBSETS},
        "per_class": dict(sorted(counts.items())),
    }
    return summary


if __name__ == "__main__":
    # 路径 / Mac冒烟·Linux全量：platform_config.py
    OUT_DIR = BY_CLASS_ROOT
    result = extract(
        PET_ROOT,
        OUT_DIR,
        max_classes=MAX_CLASSES,
        max_per_class=MAX_PER_CLASS,
    )

    print(
        f"saved {result['n_saved']}/{result['n_images']} images "
        f"(of {result['n_images_all']} listed) "
        f"→ {result['n_classes']} classes under {result['out_dir']}"
    )
    print("layout:", result["layout"])
    print(
        f"max_classes={result['max_classes']} max_per_class={result['max_per_class']} "
        f"skipped_existing={result['skipped_existing']} "
        f"missing_trimap={result['missing_trimap']} "
        f"no_fg={result['no_fg_fallback_full']}"
    )
