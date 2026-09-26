#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dump offline A1–A4 augmented ImageFolders (pipeline step ③).

Runs **after** ``split_by_class.py``. Source is the copied tree
``SPLIT_DIR/trainval/`` (all images there — no further train/val split).
``test/`` is never augmented.

Output root: ``AUG_OUT_ROOT`` from ``platform_config``.

Each source image writes ``AUG_COUNT`` slots (default 4) for **A0–A4**
(same multiplicity for fair compare)::

    AUG_OUT_ROOT/
      A0_none/<breed>/<stem>__aug01.jpg …   # originals if INCLUDE_A0
      A1_global/<breed>/<stem>__aug01.jpg …
      A2_fg_only/…
      A3_bg_only/…
      A4_fgbg/…

A0 is unaugmented copies (optional, ``INCLUDE_A0`` in ``platform_config``).
A4 uses FG×N slot augs + random complex BG (same as viz). A1–A3 use ``augment()``.
``SKIP_EXISTING=True`` (default): skip files already on disk — do **not**
re-augment and do **not** delete prior outputs.

``compare.py`` / ``train.py`` can instead call ``fgbg_aug`` on the fly (skip this dump).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from fgbg_aug import (
    AugMode,
    augment,
    augment_fgbg_pick,
    list_by_class_samples,
    load_by_class_sample,
)
from platform_config import (
    AUG_COUNT,
    AUG_OUT_ROOT,
    INCLUDE_A0,
    SPLIT_DIR,
)
from split_by_class import ensure_split

# Ablation id → (folder name, mode). A0 optional via INCLUDE_A0.
MODE_DIRS: list[tuple[str, AugMode]] = [
    ("A1_global", "global"),
    ("A2_fg_only", "fg_only"),
    ("A3_bg_only", "bg_only"),
    ("A4_fgbg", "fgbg"),
]


def _pick_other_sample(
    samples: list[tuple[str, str]],
    idx: int,
) -> tuple[str, str]:
    j = (idx + 17 * (idx % 9 + 1)) % len(samples)
    if j == idx:
        j = (idx + 1) % len(samples)
    return samples[j]


def _save_jpg(path: Path, img: Image.Image, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=quality)


def dump_mode(
    *,
    root: Path,
    samples: list[tuple[str, str]],
    out_dir: Path,
    mode: AugMode,
    seed: int,
    img_size: int,
    skip_existing: bool,
    aug_count: int,
    other_image_prob: float,
) -> dict[str, int]:
    """Write ``aug_count`` slots per source as ``<stem>__augNN.jpg``."""
    n_slots = max(1, int(aug_count))
    stats = {"ok": 0, "skipped": 0, "fail": 0}
    for idx, (breed, stem) in enumerate(tqdm(samples, desc=out_dir.name, leave=False)):
        try:
            img, mask, _ = load_by_class_sample(root, breed, stem)
        except OSError:
            stats["fail"] += 1
            continue

        for slot in range(n_slots):
            dst = out_dir / breed / f"{stem}__aug{slot + 1:02d}.jpg"
            if skip_existing and dst.is_file() and dst.stat().st_size > 64:
                stats["skipped"] += 1
                continue
            rng = np.random.default_rng(seed + idx * 10007 + slot * 17)

            if mode == "fgbg":
                out = augment_fgbg_pick(
                    img,
                    mask,
                    rng,
                    root=root,
                    samples=samples,
                    exclude=(breed, stem),
                    n_slots=n_slots,
                    pick=slot,
                    other_image_prob=other_image_prob,
                )
            elif mode == "none":
                out = img
            else:
                bg_b, bg_s = _pick_other_sample(samples, idx + slot * 31)
                try:
                    bg_img, _, _ = load_by_class_sample(root, bg_b, bg_s)
                except OSError:
                    bg_img = None
                out = augment(img, mask, rng, mode, bg_image=bg_img)

            if img_size > 0:
                out = out.resize((img_size, img_size), Image.BILINEAR)
            _save_jpg(dst, out)
            stats["ok"] += 1
    return stats


def run_dump(
    *,
    trainval_root: Path,
    out_root: Path,
    modes: list[tuple[str, AugMode]],
    seed: int,
    img_size: int,
    skip_existing: bool,
    aug_count: int,
    other_image_prob: float,
) -> None:
    """Augment **all** samples under ``trainval_root`` (by_class layout)."""
    samples = list_by_class_samples(trainval_root)
    if not samples:
        raise SystemExit(f"no samples under {trainval_root / 'image'}")

    print(f"source={trainval_root}")
    print(f"out={out_root}")
    print(f"n_samples={len(samples)} modes={[m[0] for m in modes]} aug_count={aug_count}")
    out_root.mkdir(parents=True, exist_ok=True)

    for folder, mode in modes:
        stats = dump_mode(
            root=trainval_root,
            samples=samples,
            out_dir=out_root / folder,
            mode=mode,
            seed=seed,
            img_size=img_size,
            skip_existing=skip_existing,
            aug_count=aug_count,
            other_image_prob=other_image_prob,
        )
        print(f"{folder}: ok={stats['ok']} skipped={stats['skipped']} fail={stats['fail']}")


if __name__ == "__main__":
    # 路径 / 输出：platform_config；源 = SPLIT_DIR/trainval（整目录增强，不区分 train/val）
    from platform_config import BY_CLASS_ROOT, MAX_CLASSES, MAX_PER_CLASS

    TRAINVAL_ROOT = SPLIT_DIR / "trainval"

    # ② 若尚无划分，先 ensure（复用已有 trainval/test 复制树）
    TEST_RATIO = 0.1
    VAL_RATIO = 0
    SPLIT_SEED = 0
    FORCE_RESPLIT = False
    split_info = ensure_split(
        BY_CLASS_ROOT,
        SPLIT_DIR,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SPLIT_SEED,
        force=FORCE_RESPLIT,
        max_classes=MAX_CLASSES,
        max_per_class=MAX_PER_CLASS,
    )
    action = "reused" if split_info.get("reused") else "created"
    print(
        f"split {action}: trainval={split_info.get('n_trainval')} "
        f"test={split_info.get('n_test')} → {SPLIT_DIR}"
    )
    if not (TRAINVAL_ROOT / "image").is_dir():
        raise SystemExit(f"missing trainval tree: {TRAINVAL_ROOT / 'image'} (run split_by_class first)")

    SEED = 0
    IMG_SIZE = 224  # 0 = keep crop size; >0 resize to square
    # True: skip files that already exist; never delete prior aug samples.
    SKIP_EXISTING = False
    OTHER_IMAGE_PROB = 0.4  # A4 BG from other images; else solid+noise+stripes
    # INCLUDE_A0 / AUG_COUNT：platform_config（原图 A0_none ×N，与 A1–A4 同倍数）

    modes: list[tuple[str, AugMode]] = list(MODE_DIRS)
    if INCLUDE_A0:
        modes = [("A0_none", "none"), *modes]

    run_dump(
        trainval_root=TRAINVAL_ROOT,
        out_root=AUG_OUT_ROOT,
        modes=modes,
        seed=SEED,
        img_size=IMG_SIZE,
        skip_existing=SKIP_EXISTING,
        aug_count=AUG_COUNT,
        other_image_prob=OTHER_IMAGE_PROB,
    )
