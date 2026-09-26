#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split dump_aug ImageFolders into train / val (pipeline step ③b).

Source::

    AUG_OUT_ROOT/
      A0_none/<breed>/<stem>__augNN.jpg   # originals (optional, INCLUDE_A0)
      A1_global/<breed>/<stem>__augNN.jpg
      …

Output (ImageFolder-ready)::

    AUG_SPLIT_ROOT/
      A0_none/train/<breed>/…   # baseline for comparison
      A0_none/val/<breed>/…
      A1_global/train|val/…
      …
      split_summary.json

If ``INCLUDE_A0`` and ``A0_none`` is missing/incomplete, copies originals from
``SPLIT_DIR/trainval/image`` (same ``AUG_COUNT`` slots + optional resize)
before splitting — so you can add A0 without re-dumping A1–A4.

``AUG_VAL_USE_AUG`` (``platform_config``):

- **True (default)**: current behavior — val also uses dump/aug files.
- **False**: val = **originals only** (from ``trainval/image``); ratio is over
  **source stems** (pick val stems first, then remaining stems' aug → train).
  Forces stem-based split (``AUG_RANDOM_SPLIT`` ignored).

When ``AUG_VAL_USE_AUG=True``, split modes (``AUG_RANDOM_SPLIT``):

- **False (default)**: same **source-stem** → same train/val for all ``__augNN``
  (and the same assignment across every A* folder).
- **True**: each **file** randomly train/val; same ``breed/filename`` shared
  across A* so modes stay comparable.

Per breed: hold out ``AUG_VAL_RATIO`` to val; rest train.
Default 8:2 → ``AUG_VAL_RATIO=0.2`` in ``platform_config``.

Change variables under ``if __name__ == "__main__"`` then run.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from fgbg_aug import list_by_class_samples
from platform_config import (
    AUG_COUNT,
    AUG_OUT_ROOT,
    AUG_RANDOM_SPLIT,
    AUG_SPLIT_ROOT,
    AUG_VAL_RATIO,
    AUG_VAL_USE_AUG,
    INCLUDE_A0,
    SPLIT_DIR,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_STEM_AUG_RE = re.compile(r"^(?P<source>.+)__aug\d+$", re.IGNORECASE)
A0_DIR_NAME = "A0_none"


def source_stem_from_name(name: str) -> str:
    """``Abyssinian_82__aug01.jpg`` → ``Abyssinian_82``; plain stem otherwise."""
    stem = Path(name).stem
    m = _STEM_AUG_RE.match(stem)
    return m.group("source") if m else stem


def _is_image_file(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("._"):
        return False
    return path.suffix.lower() in IMAGE_EXTS


def _count_images(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if _is_image_file(p))


def list_mode_dirs(aug_root: Path, *, include_a0: bool) -> list[Path]:
    """Top-level dirs that look like ablation dumps (A0_none / A1_global / …)."""
    if not aug_root.is_dir():
        raise FileNotFoundError(f"missing aug root: {aug_root}")
    out: list[Path] = []
    for p in sorted(aug_root.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name.startswith("._"):
            continue
        if p.name in {"train", "val", "test"}:
            continue
        if p.name == A0_DIR_NAME and not include_a0:
            continue
        out.append(p)
    return out


def _find_image(breed_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = breed_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def ensure_a0_from_trainval(
    trainval_root: Path,
    aug_root: Path,
    *,
    aug_count: int = 4,
    img_size: int = 224,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Materialize ``A0_none`` as resized original copies × ``aug_count`` (fair vs A1–A4)."""
    image_root = Path(trainval_root) / "image"
    if not image_root.is_dir():
        raise FileNotFoundError(f"missing trainval images: {image_root}")
    out_a0 = Path(aug_root) / A0_DIR_NAME
    out_a0.mkdir(parents=True, exist_ok=True)
    n_slots = max(1, int(aug_count))
    stats = {"ok": 0, "skipped": 0, "fail": 0}
    samples = list_by_class_samples(trainval_root)
    for breed, stem in samples:
        src = _find_image(image_root / breed, stem)
        if src is None:
            stats["fail"] += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
        except OSError:
            stats["fail"] += 1
            continue
        if img_size > 0:
            img = img.resize((img_size, img_size), Image.BILINEAR)
        for slot in range(n_slots):
            dst = out_a0 / breed / f"{stem}__aug{slot + 1:02d}.jpg"
            if skip_existing and dst.is_file() and dst.stat().st_size > 64:
                stats["skipped"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, quality=95)
            stats["ok"] += 1
    return stats


def collect_breed_sources(mode_dir: Path) -> dict[str, list[str]]:
    """breed → sorted unique source stems that have at least one image."""
    by_breed: dict[str, set[str]] = defaultdict(set)
    for breed_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for f in breed_dir.iterdir():
            if not _is_image_file(f):
                continue
            by_breed[breed_dir.name].add(source_stem_from_name(f.name))
    return {b: sorted(stems) for b, stems in sorted(by_breed.items())}


def collect_breed_files(mode_dir: Path) -> dict[str, list[str]]:
    """breed → sorted image filenames."""
    by_breed: dict[str, list[str]] = {}
    for breed_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        names = sorted(f.name for f in breed_dir.iterdir() if _is_image_file(f))
        by_breed[breed_dir.name] = names
    return by_breed


def split_items_per_breed(
    by_breed: dict[str, list[str]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Per breed: shuffle items, hold out ``val_ratio`` to val, rest train."""
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    rng = random.Random(int(seed))
    train: dict[str, list[str]] = {}
    val: dict[str, list[str]] = {}
    for breed, items_in in by_breed.items():
        items = list(items_in)
        rng.shuffle(items)
        n = len(items)
        if n == 0:
            train[breed], val[breed] = [], []
            continue
        if n == 1:
            train[breed], val[breed] = items, []
            continue
        n_val = max(1, int(round(n * float(val_ratio))))
        n_val = min(n_val, n - 1)
        val[breed] = sorted(items[:n_val])
        train[breed] = sorted(items[n_val:])
    return train, val


def split_stems_per_breed(
    by_breed: dict[str, list[str]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Per breed: shuffle stems, hold out ``val_ratio`` to val, rest train."""
    return split_items_per_breed(by_breed, val_ratio=val_ratio, seed=seed)


def _copy_mode_split(
    mode_dir: Path,
    out_mode: Path,
    train_map: dict[str, set[str]],
    val_map: dict[str, set[str]],
    *,
    skip_existing: bool,
    key_fn,
) -> dict[str, int]:
    """Copy files; ``key_fn(filename) → membership key`` (stem or filename)."""
    stats = {"train": 0, "val": 0, "skipped": 0, "orphan": 0}
    for breed_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        breed = breed_dir.name
        train_keys = train_map.get(breed, set())
        val_keys = val_map.get(breed, set())
        for f in breed_dir.iterdir():
            if not _is_image_file(f):
                continue
            key = key_fn(f.name)
            if key in train_keys:
                split = "train"
            elif key in val_keys:
                split = "val"
            else:
                stats["orphan"] += 1
                continue
            dst = out_mode / split / breed / f.name
            if skip_existing and dst.is_file() and dst.stat().st_size > 64:
                stats["skipped"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            stats[split] += 1
    return stats


def _copy_train_by_stem(
    mode_dir: Path,
    out_mode: Path,
    train_stems: dict[str, set[str]],
    *,
    skip_existing: bool,
) -> dict[str, int]:
    """Copy dump files whose source stem is in ``train_stems`` → ``out_mode/train``."""
    stats = {"train": 0, "skipped": 0, "orphan": 0}
    for breed_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        breed = breed_dir.name
        allow = train_stems.get(breed, set())
        for f in breed_dir.iterdir():
            if not _is_image_file(f):
                continue
            stem = source_stem_from_name(f.name)
            if stem not in allow:
                # Val-stem dump files are unused when clean-val (originals go to val).
                continue
            dst = out_mode / "train" / breed / f.name
            if skip_existing and dst.is_file() and dst.stat().st_size > 64:
                stats["skipped"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            stats["train"] += 1
    return stats


def _copy_clean_val_originals(
    trainval_root: Path,
    out_mode: Path,
    val_stems: dict[str, set[str]],
    *,
    img_size: int,
    skip_existing: bool,
) -> dict[str, int]:
    """Write one resized original per val stem under ``out_mode/val/<breed>/``."""
    image_root = Path(trainval_root) / "image"
    stats = {"val": 0, "skipped": 0, "fail": 0}
    for breed, stems in sorted(val_stems.items()):
        for stem in sorted(stems):
            src = _find_image(image_root / breed, stem)
            if src is None:
                stats["fail"] += 1
                continue
            dst = out_mode / "val" / breed / f"{stem}.jpg"
            if skip_existing and dst.is_file() and dst.stat().st_size > 64:
                stats["skipped"] += 1
                continue
            try:
                img = Image.open(src).convert("RGB")
            except OSError:
                stats["fail"] += 1
                continue
            if img_size > 0:
                img = img.resize((img_size, img_size), Image.BILINEAR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, quality=95)
            stats["val"] += 1
    return stats


def split_aug_train_val(
    aug_root: Path,
    out_root: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 0,
    force: bool = False,
    skip_existing: bool = True,
    reference_mode: str | None = None,
    include_a0: bool = True,
    random_split: bool = False,
    val_use_aug: bool = True,
    trainval_root: Path | None = None,
    aug_count: int = 4,
    img_size: int = 224,
) -> dict:
    """Write train/val trees under ``out_root`` for every mode folder in ``aug_root``.

    ``val_use_aug=True``: legacy path (aug files in val; optional file-random split).
    ``val_use_aug=False``: pick val **stems** first by ``val_ratio``, copy **originals**
    into every mode's val; train gets remaining stems' dump/aug files.
    """
    aug_root = Path(aug_root)
    out_root = Path(out_root)
    trainval_root = Path(trainval_root) if trainval_root is not None else SPLIT_DIR / "trainval"
    val_use_aug = bool(val_use_aug)
    # Clean-val path always allocates by source stem (ratio over originals).
    effective_random = bool(random_split) and val_use_aug
    if not val_use_aug and random_split:
        print("AUG_VAL_USE_AUG=False → ignore AUG_RANDOM_SPLIT; stem-first clean val")

    a0_stats = None
    if include_a0:
        a0_dir = aug_root / A0_DIR_NAME
        a0_n = _count_images(a0_dir)
        # Prefer matching A1 count when present; else any other mode; else force if empty.
        peer_n = 0
        for peer_name in ("A1_global", "A2_fg_only", "A3_bg_only", "A4_fgbg"):
            peer_n = _count_images(aug_root / peer_name)
            if peer_n > 0:
                break
        need_a0 = force or a0_n == 0 or (peer_n > 0 and a0_n < peer_n)
        if need_a0:
            print(
                f"materialize {A0_DIR_NAME} from {trainval_root / 'image'} "
                f"(have={a0_n} peer={peer_n}) …"
            )
            a0_stats = ensure_a0_from_trainval(
                trainval_root,
                aug_root,
                aug_count=aug_count,
                img_size=img_size,
                skip_existing=skip_existing and not force,
            )
            print(
                f"{A0_DIR_NAME}: ok={a0_stats['ok']} skipped={a0_stats['skipped']} "
                f"fail={a0_stats['fail']}"
            )
        a0_n_after = _count_images(aug_root / A0_DIR_NAME)
        if a0_n_after <= 0:
            raise RuntimeError(
                f"INCLUDE_A0=True but no images under {aug_root / A0_DIR_NAME} "
                f"(check trainval: {trainval_root / 'image'})"
            )

    mode_dirs = list_mode_dirs(aug_root, include_a0=include_a0)
    if not mode_dirs:
        raise RuntimeError(f"no mode folders under {aug_root}")
    if include_a0 and not any(p.name == A0_DIR_NAME for p in mode_dirs):
        raise RuntimeError(f"INCLUDE_A0=True but {A0_DIR_NAME} missing under {aug_root}")

    if reference_mode:
        ref = aug_root / reference_mode
        if not ref.is_dir():
            raise FileNotFoundError(f"reference_mode missing: {ref}")
    else:
        named = {p.name: p for p in mode_dirs}
        ref = named.get(A0_DIR_NAME) or named.get("A1_global") or mode_dirs[0]

    if force and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    skip = skip_existing and not force

    if not val_use_aug:
        # 1) val stems by original ratio → 2) remaining stems → train (with augs)
        by_breed = collect_breed_sources(ref)
        if not by_breed:
            raise RuntimeError(f"no images under {ref}")
        train_lists, val_lists = split_items_per_breed(by_breed, val_ratio=val_ratio, seed=seed)
        train_map = {b: set(items) for b, items in train_lists.items()}
        val_map = {b: set(items) for b, items in val_lists.items()}
        list_label = "stems"
        per_mode: dict[str, dict] = {}
        for mode_dir in mode_dirs:
            out_mode = out_root / mode_dir.name
            train_stats = _copy_train_by_stem(mode_dir, out_mode, train_map, skip_existing=skip)
            val_stats = _copy_clean_val_originals(
                trainval_root,
                out_mode,
                val_map,
                img_size=img_size,
                skip_existing=skip,
            )
            stats = {
                "train": train_stats["train"],
                "val": val_stats["val"],
                "skipped": train_stats["skipped"] + val_stats["skipped"],
                "orphan": train_stats["orphan"],
                "val_fail": val_stats["fail"],
            }
            per_mode[mode_dir.name] = stats
            print(
                f"{mode_dir.name}: train_files={stats['train']} val_orig={stats['val']} "
                f"skipped={stats['skipped']} orphan={stats['orphan']} "
                f"val_fail={stats['val_fail']}"
            )
        n_train = sum(len(v) for v in train_lists.values())
        n_val = sum(len(v) for v in val_lists.values())
        split_unit = "stem"
        effective_random = False
    else:
        if effective_random:
            by_breed = collect_breed_files(ref)

            def key_fn(name: str) -> str:
                return name

            list_label = "files"
        else:
            by_breed = collect_breed_sources(ref)
            key_fn = source_stem_from_name
            list_label = "stems"

        if not by_breed:
            raise RuntimeError(f"no images under {ref}")

        train_lists, val_lists = split_items_per_breed(by_breed, val_ratio=val_ratio, seed=seed)
        train_map = {b: set(items) for b, items in train_lists.items()}
        val_map = {b: set(items) for b, items in val_lists.items()}

        per_mode = {}
        for mode_dir in mode_dirs:
            stats = _copy_mode_split(
                mode_dir,
                out_root / mode_dir.name,
                train_map,
                val_map,
                skip_existing=skip,
                key_fn=key_fn,
            )
            per_mode[mode_dir.name] = stats
            print(
                f"{mode_dir.name}: train_files={stats['train']} val_files={stats['val']} "
                f"skipped={stats['skipped']} orphan={stats['orphan']}"
            )

        n_train = sum(len(v) for v in train_lists.values())
        n_val = sum(len(v) for v in val_lists.values())
        split_unit = "file" if effective_random else "stem"

    summary = {
        "aug_root": str(aug_root),
        "out_root": str(out_root),
        "reference_mode": ref.name,
        "include_a0": bool(include_a0),
        "a0_materialize": a0_stats,
        "val_use_aug": val_use_aug,
        "random_split": bool(effective_random),
        "random_split_requested": bool(random_split),
        "split_unit": split_unit,
        "val_ratio": float(val_ratio),
        "train_ratio": 1.0 - float(val_ratio),
        "seed": int(seed),
        "n_breeds": len(by_breed),
        "n_train_stems": n_train if split_unit == "stem" else None,
        "n_val_stems": n_val if split_unit == "stem" else None,
        "n_train_items": n_train,
        "n_val_items": n_val,
        "modes": [p.name for p in mode_dirs],
        "per_mode": per_mode,
        "per_class": {
            b: {
                f"train_{list_label}": len(train_lists[b]),
                f"val_{list_label}": len(val_lists[b]),
            }
            for b in sorted(by_breed)
        },
    }
    (out_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    train_lines = [f"{b}/{s}" for b in sorted(train_lists) for s in train_lists[b]]
    val_lines = [f"{b}/{s}" for b in sorted(val_lists) for s in val_lists[b]]
    list_name = "train_files.txt" if effective_random else "train_stems.txt"
    val_list_name = "val_files.txt" if effective_random else "val_stems.txt"
    (out_root / list_name).write_text(
        "\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8"
    )
    (out_root / val_list_name).write_text(
        "\n".join(val_lines) + ("\n" if val_lines else ""), encoding="utf-8"
    )
    # Keep stem filenames for tooling that always looks for them.
    if effective_random:
        (out_root / "train_stems.txt").write_text(
            "# random_split=True: see train_files.txt / val_files.txt\n", encoding="utf-8"
        )
        (out_root / "val_stems.txt").write_text(
            "# random_split=True: see train_files.txt / val_files.txt\n", encoding="utf-8"
        )
    return summary


if __name__ == "__main__":
    # 路径 / 比例 / A0 / 随机划分 / val 是否用增强：platform_config
    SEED = 0
    FORCE = False  # True：清空 AUG_SPLIT_ROOT 后重写；并重做 A0（若 INCLUDE_A0）
    SKIP_EXISTING = False
    IMG_SIZE = 224  # A0 / clean-val 原图 resize；与 dump_aug 对齐
    # None = 优先 A0_none，否则 A1_global，定划分后套用到所有模式
    REFERENCE_MODE: str | None = None

    print(f"aug_root={AUG_OUT_ROOT}")
    print(f"out_root={AUG_SPLIT_ROOT}")
    print(f"val_ratio={AUG_VAL_RATIO} (train={1.0 - AUG_VAL_RATIO:.2f})")
    print(
        f"include_a0={INCLUDE_A0} aug_count={AUG_COUNT} "
        f"random_split={AUG_RANDOM_SPLIT} val_use_aug={AUG_VAL_USE_AUG}"
    )

    summary = split_aug_train_val(
        AUG_OUT_ROOT,
        AUG_SPLIT_ROOT,
        val_ratio=AUG_VAL_RATIO,
        seed=SEED,
        force=FORCE,
        skip_existing=SKIP_EXISTING,
        reference_mode=REFERENCE_MODE,
        include_a0=INCLUDE_A0,
        random_split=AUG_RANDOM_SPLIT,
        val_use_aug=AUG_VAL_USE_AUG,
        trainval_root=SPLIT_DIR / "trainval",
        aug_count=AUG_COUNT,
        img_size=IMG_SIZE,
    )
    print(
        f"done {summary['split_unit']}s train={summary['n_train_items']} "
        f"val={summary['n_val_items']} breeds={summary['n_breeds']} "
        f"val_use_aug={summary['val_use_aug']} modes={summary['modes']} → {AUG_SPLIT_ROOT}"
    )
