#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract CUB-200-2011 into by_class (pipeline step ①, CUB).

Layout (same as Pet)::

    <out>/
      image/<class>/<id>.jpg   # RGB crop (seg body bbox)
      fg/<class>/<id>.png      # RGBA crop: keep seg >= fg_thresh
      bg/<class>/<id>.png      # RGBA crop: keep seg <  fg_thresh
      bbox/<class>/<id>.jpg    # full RGB with body bbox drawn

CUB soft segmentations are quantized ``{0,51,102,153,204,255}``. Default
``fg_thresh=128`` → FG ``{153,204,255}``, BG ``{0,51,102}``.

Next: ``split_by_class.py`` (②) → ``dump_aug.py`` (③) / ``compare.py`` (④).
Change variables under ``if __name__ == "__main__"`` then run.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from platform_config import CUB_BY_CLASS_ROOT, CUB_ROOT, MAX_CLASSES, MAX_PER_CLASS

SUBSETS = ("image", "fg", "bg", "bbox")


def body_bbox(fg_mask: np.ndarray, pad_ratio: float) -> tuple[int, int, int, int] | None:
    """BBox from boolean FG mask. Returns (left, top, right, bottom) xyxy, or None."""
    ys, xs = np.where(fg_mask)
    if len(xs) == 0:
        return None
    h, w = fg_mask.shape[:2]
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


def draw_bbox(img: Image.Image, box: tuple[int, int, int, int], width: int = 3) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=(0, 255, 0), width=width)
    return out


def rgba_keep(crop_rgb: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Build RGBA from RGB crop; alpha=255 where keep, else 0 (transparent)."""
    h, w = keep.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = crop_rgb
    out[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
    return out


def iter_cub_images(cub_root: Path) -> list[tuple[str, str, Path, Path]]:
    """Return list of (class_name, stem, image_path, seg_path) from images.txt."""
    images_txt = cub_root / "images.txt"
    images_dir = cub_root / "images"
    seg_dir = cub_root / "segmentations"
    if not images_txt.is_file():
        raise FileNotFoundError(f"missing images.txt: {images_txt}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing images: {images_dir}")
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"missing segmentations: {seg_dir}")

    rows: list[tuple[str, str, Path, Path]] = []
    for line in images_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _iid, rel = line.split(maxsplit=1)
        rel_path = Path(rel)
        if rel_path.name.startswith("._"):
            continue
        class_name = rel_path.parent.name
        stem = rel_path.stem
        img_path = images_dir / rel_path
        seg_path = seg_dir / rel_path.with_suffix(".png")
        rows.append((class_name, stem, img_path, seg_path))
    return rows


def cap_cub_rows(
    rows: list[tuple[str, str, Path, Path]],
    *,
    max_classes: int,
    max_per_class: int,
) -> list[tuple[str, str, Path, Path]]:
    """Mac smoke / optional subset: first N classes, then ≤K images per class."""
    buckets: dict[str, list[tuple[str, str, Path, Path]]] = defaultdict(list)
    for row in rows:
        buckets[row[0]].append(row)
    classes = sorted(buckets)
    if max_classes > 0:
        classes = classes[: int(max_classes)]
    out: list[tuple[str, str, Path, Path]] = []
    for c in classes:
        items = sorted(buckets[c], key=lambda r: r[1])
        if max_per_class > 0:
            items = items[: int(max_per_class)]
        out.extend(items)
    return out


def extract(
    cub_root: Path,
    out_dir: Path,
    *,
    fg_thresh: int = 128,
    pad_ratio: float = 0.05,
    skip_existing: bool = True,
    max_classes: int = 0,
    max_per_class: int = 0,
) -> dict:
    rows_all = iter_cub_images(cub_root)
    rows = cap_cub_rows(
        rows_all, max_classes=max_classes, max_per_class=max_per_class
    )
    for name in SUBSETS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    skipped = 0
    no_fg = 0
    missing_seg = 0
    missing_img = 0
    size_mismatch = 0

    for class_name, stem, img_path, seg_path in tqdm(rows, desc="extract cub"):
        paths_out = {
            "image": out_dir / "image" / class_name / f"{stem}.jpg",
            "fg": out_dir / "fg" / class_name / f"{stem}.png",
            "bg": out_dir / "bg" / class_name / f"{stem}.png",
            "bbox": out_dir / "bbox" / class_name / f"{stem}.jpg",
        }
        if skip_existing and all(p.is_file() for p in paths_out.values()):
            skipped += 1
            counts[class_name] += 1
            continue

        if not img_path.is_file():
            missing_img += 1
            continue
        if not seg_path.is_file():
            missing_seg += 1
            continue

        img = Image.open(img_path).convert("RGB")
        seg = np.array(Image.open(seg_path))
        if seg.ndim == 3:
            seg = seg[..., 0]
        if seg.shape[:2] != (img.height, img.width):
            size_mismatch += 1
            seg = np.array(
                Image.fromarray(seg).resize((img.width, img.height), Image.NEAREST)
            )

        fg_mask = seg >= fg_thresh
        box = body_bbox(fg_mask, pad_ratio)
        if box is None:
            no_fg += 1
            box = (0, 0, img.width, img.height)

        x0, y0, x1, y1 = box
        crop = np.array(img.crop(box))
        seg_crop = seg[y0:y1, x0:x1]
        fg = rgba_keep(crop, seg_crop >= fg_thresh)
        bg = rgba_keep(crop, seg_crop < fg_thresh)
        bbox_vis = draw_bbox(img, box)

        for p in paths_out.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(paths_out["image"], quality=95)
        Image.fromarray(fg, mode="RGBA").save(paths_out["fg"])
        Image.fromarray(bg, mode="RGBA").save(paths_out["bg"])
        bbox_vis.save(paths_out["bbox"], quality=95)
        counts[class_name] += 1

    summary = {
        "n_images_all": len(rows_all),
        "n_images": len(rows),
        "n_saved": int(sum(counts.values())),
        "n_classes": len(counts),
        "max_classes": int(max_classes),
        "max_per_class": int(max_per_class),
        "skipped_existing": skipped,
        "missing_img": missing_img,
        "missing_seg": missing_seg,
        "size_mismatch_resized": size_mismatch,
        "no_fg_fallback_full": no_fg,
        "fg_thresh": fg_thresh,
        "out_dir": str(out_dir),
        "layout": {s: str(out_dir / s / "<class>") for s in SUBSETS},
        "per_class": dict(sorted(counts.items())),
    }
    return summary


if __name__ == "__main__":
    # 路径 / Mac冒烟·Linux全量：platform_config.py
    OUT_DIR = CUB_BY_CLASS_ROOT
    FG_THRESH = 128
    result = extract(
        CUB_ROOT,
        OUT_DIR,
        fg_thresh=FG_THRESH,
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
        f"fg_thresh={result['fg_thresh']} "
        f"max_classes={result['max_classes']} max_per_class={result['max_per_class']} "
        f"skipped_existing={result['skipped_existing']} "
        f"missing_img={result['missing_img']} "
        f"missing_seg={result['missing_seg']} "
        f"size_mismatch={result['size_mismatch_resized']} "
        f"no_fg={result['no_fg_fallback_full']}"
    )
