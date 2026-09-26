#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split by_class into trainval / test (pipeline step ②).

**Ratios** (``TEST_RATIO`` / ``VAL_RATIO``; train = rest after test+val).
Procedure: hold out **test** first, then split remainder into **train / val**.
``trainval`` = train ∪ val.

Also **copies** class images (and fg/bg/bbox when present) into::

    <out>/
      trainval/{image,fg,bg,bbox}/<breed>/<stem>.*
      test/{image,fg,bg,bbox}/<breed>/<stem>.*
      train.txt / val.txt / test.txt / trainval.txt
      split_summary.json

Mac smoke vs Linux full: use ``MAX_CLASSES`` / ``MAX_PER_CLASS`` from
``platform_config`` (same idea as ``compare.py``).

Change variables under ``if __name__ == "__main__"`` then run.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from fgbg_aug import list_by_class_samples
from platform_config import BY_CLASS_ROOT, MAX_CLASSES, MAX_PER_CLASS, SPLIT_DIR

REQUIRED_LISTS = ("train.txt", "val.txt", "test.txt")
REQUIRED_COPY_ROOTS = ("trainval", "test")
COPY_SUBSETS = ("image", "fg", "bg", "bbox")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def split_stratified(
    samples: list[tuple[str, str]],
    *,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Per-breed: hold out test first, then val from remainder, rest train.

    ``test_ratio`` / ``val_ratio`` are fractions of that breed's **total** count.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if float(test_ratio) + float(val_ratio) >= 1.0:
        raise ValueError("test_ratio + val_ratio must be < 1")

    by_breed: dict[str, list[str]] = defaultdict(list)
    for breed, stem in samples:
        by_breed[breed].append(stem)

    rng = random.Random(int(seed))
    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []

    for breed in sorted(by_breed):
        stems = list(by_breed[breed])
        rng.shuffle(stems)
        n = len(stems)
        if n == 1:
            train.append((breed, stems[0]))
            continue

        # 1) test first
        n_test = max(1, int(round(n * float(test_ratio))))
        n_test = min(n_test, n - 1)
        test_stems = stems[:n_test]
        remain = stems[n_test:]

        # 2) val from remainder (ratio vs total n)
        n_val = 0
        if val_ratio > 0 and len(remain) >= 2:
            n_val = max(1, int(round(n * float(val_ratio))))
            n_val = min(n_val, len(remain) - 1)
        val_stems = remain[:n_val]
        train_stems = remain[n_val:]

        train.extend((breed, s) for s in train_stems)
        val.extend((breed, s) for s in val_stems)
        test.extend((breed, s) for s in test_stems)

    key = lambda x: (x[0], x[1])
    train.sort(key=key)
    val.sort(key=key)
    test.sort(key=key)
    return train, val, test


def cap_samples(
    samples: list[tuple[str, str]],
    *,
    max_classes: int,
    max_per_class: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Mac smoke / optional subset: first N breeds, then ≤K stems per breed."""
    breeds = sorted({b for b, _ in samples})
    if max_classes > 0:
        breeds = breeds[: int(max_classes)]
    keep = set(breeds)
    samples = [(b, s) for b, s in samples if b in keep]
    if max_per_class <= 0:
        return samples

    rng = random.Random(int(seed))
    buckets: dict[str, list[str]] = defaultdict(list)
    for b, s in samples:
        buckets[b].append(s)
    out: list[tuple[str, str]] = []
    for b in sorted(buckets):
        stems = list(buckets[b])
        rng.shuffle(stems)
        out.extend((b, s) for s in stems[: int(max_per_class)])
    return out


def write_stem_list(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [stem for _, stem in pairs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _count_lines(path: Path) -> int:
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            n += 1
    return n


def _find_source_file(subset_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = subset_dir / f"{stem}{ext}"
        if p.is_file():
            return p
        p = subset_dir / f"{stem}{ext.upper()}"
        if p.is_file():
            return p
    return None


def _copy_pairs(
    by_class_root: Path,
    dest_root: Path,
    pairs: list[tuple[str, str]],
) -> dict[str, int]:
    """Copy image/fg/bg/bbox for each (breed, stem) into ``dest_root``."""
    counts = {k: 0 for k in COPY_SUBSETS}
    counts["missing_image"] = 0
    for breed, stem in pairs:
        img_src = _find_source_file(by_class_root / "image" / breed, stem)
        if img_src is None:
            counts["missing_image"] += 1
            continue
        for subset in COPY_SUBSETS:
            src_dir = by_class_root / subset / breed
            if not src_dir.is_dir():
                continue
            if subset == "image":
                src = img_src
            else:
                src = _find_source_file(src_dir, stem)
                if src is None:
                    continue
            dst = dest_root / subset / breed / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            counts[subset] += 1
    return counts


def copy_split_dirs(
    by_class_root: Path,
    out_dir: Path,
    trainval: list[tuple[str, str]],
    test: list[tuple[str, str]],
) -> dict:
    """Rewrite ``out_dir/trainval`` and ``out_dir/test`` class trees."""
    out_dir = Path(out_dir)
    copied: dict[str, dict] = {}
    for name, pairs in (("trainval", trainval), ("test", test)):
        dest = out_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        copied[name] = _copy_pairs(by_class_root, dest, pairs)
    return copied


def _copy_dirs_ready(out_dir: Path) -> bool:
    for name in REQUIRED_COPY_ROOTS:
        image_root = out_dir / name / "image"
        if not image_root.is_dir():
            return False
        if not any(p.is_dir() for p in image_root.iterdir()):
            return False
    return True


def _per_class_counts(
    train: list[tuple[str, str]],
    val: list[tuple[str, str]],
    test: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    per_class: dict[str, dict[str, int]] = {}
    for breed, _ in train:
        per_class.setdefault(breed, {"train": 0, "val": 0, "test": 0, "trainval": 0})
        per_class[breed]["train"] += 1
        per_class[breed]["trainval"] += 1
    for breed, _ in val:
        per_class.setdefault(breed, {"train": 0, "val": 0, "test": 0, "trainval": 0})
        per_class[breed]["val"] += 1
        per_class[breed]["trainval"] += 1
    for breed, _ in test:
        per_class.setdefault(breed, {"train": 0, "val": 0, "test": 0, "trainval": 0})
        per_class[breed]["test"] += 1
    return dict(sorted(per_class.items()))


def split_by_class(
    by_class_root: Path,
    out_dir: Path,
    *,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 0,
    max_classes: int = 0,
    max_per_class: int = 0,
) -> dict:
    """Always rewrite lists + copy trainval/test image trees."""
    samples = list_by_class_samples(by_class_root)
    if not samples:
        raise RuntimeError(f"no samples under {by_class_root / 'image'}")

    samples = cap_samples(
        samples,
        max_classes=max_classes,
        max_per_class=max_per_class,
        seed=seed,
    )
    if not samples:
        raise RuntimeError(
            f"no samples left after cap "
            f"(max_classes={max_classes}, max_per_class={max_per_class})"
        )

    train, val, test = split_stratified(
        samples, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
    )
    trainval = sorted(train + val, key=lambda x: (x[0], x[1]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.txt",
        "val": out_dir / "val.txt",
        "test": out_dir / "test.txt",
        "trainval": out_dir / "trainval.txt",
    }
    write_stem_list(paths["train"], train)
    write_stem_list(paths["val"], val)
    write_stem_list(paths["test"], test)
    write_stem_list(paths["trainval"], trainval)

    copied = copy_split_dirs(by_class_root, out_dir, trainval, test)

    summary = {
        "by_class_root": str(by_class_root),
        "out_dir": str(out_dir),
        "test_ratio": float(test_ratio),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "max_classes": int(max_classes),
        "max_per_class": int(max_per_class),
        "n_total": len(samples),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_trainval": len(trainval),
        "n_classes": len({b for b, _ in samples}),
        "train_file": str(paths["train"]),
        "val_file": str(paths["val"]),
        "test_file": str(paths["test"]),
        "trainval_file": str(paths["trainval"]),
        "trainval_dir": str(out_dir / "trainval"),
        "test_dir": str(out_dir / "test"),
        "copied": copied,
        "per_class": _per_class_counts(train, val, test),
        "reused": False,
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def ensure_split(
    by_class_root: Path,
    out_dir: Path,
    *,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 0,
    force: bool = False,
    max_classes: int = 0,
    max_per_class: int = 0,
) -> dict:
    """Create split+copies if incomplete; otherwise reuse (do not re-roll).

    Requires txt lists **and** ``trainval/image`` + ``test/image`` class trees.
    """
    out_dir = Path(out_dir)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    test_path = out_dir / "test.txt"
    trainval_path = out_dir / "trainval.txt"
    summary_path = out_dir / "split_summary.json"

    lists_ok = all((out_dir / name).is_file() for name in REQUIRED_LISTS)
    dirs_ok = _copy_dirs_ready(out_dir)
    complete = lists_ok and dirs_ok
    if not force and complete:
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {
                "by_class_root": str(by_class_root),
                "out_dir": str(out_dir),
                "test_ratio": float(test_ratio),
                "val_ratio": float(val_ratio),
                "seed": int(seed),
                "max_classes": int(max_classes),
                "max_per_class": int(max_per_class),
                "n_train": _count_lines(train_path),
                "n_val": _count_lines(val_path),
                "n_test": _count_lines(test_path),
                "n_trainval": _count_lines(trainval_path) if trainval_path.is_file() else None,
            }
        summary["reused"] = True
        summary["train_file"] = str(train_path)
        summary["val_file"] = str(val_path)
        summary["test_file"] = str(test_path)
        summary["trainval_file"] = str(trainval_path)
        summary["trainval_dir"] = str(out_dir / "trainval")
        summary["test_dir"] = str(out_dir / "test")
        return summary

    return split_by_class(
        by_class_root,
        out_dir,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
        max_classes=max_classes,
        max_per_class=max_per_class,
    )


if __name__ == "__main__":
    # 路径 / 冒烟：platform_config.py（Mac 子集测试；Linux 全量真实分割）
    OUT_DIR = SPLIT_DIR
    # 一开始就配好三比例；流程仍是先抽 test，再在剩余里划 train/val
    TEST_RATIO = 0.1
    VAL_RATIO = 0
    SEED = 0
    FORCE = False

    result = ensure_split(
        BY_CLASS_ROOT,
        OUT_DIR,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        force=FORCE,
        max_classes=MAX_CLASSES,
        max_per_class=MAX_PER_CLASS,
    )
    action = "reused" if result.get("reused") else "wrote"
    print(
        f"{action} split train={result.get('n_train')} "
        f"val={result.get('n_val')} test={result.get('n_test')} "
        f"classes={result.get('n_classes')} "
        f"max_classes={result.get('max_classes')} max_per_class={result.get('max_per_class')} "
        f"test_ratio={result.get('test_ratio')} val_ratio={result.get('val_ratio')} "
        f"seed={result.get('seed')}"
    )
    print(f"trainval dir → {result.get('trainval_dir')}")
    print(f"test dir → {result.get('test_dir')}")
    print(f"lists → {result.get('train_file')} / {result.get('val_file')} / {result.get('test_file')}")
    if result.get("copied"):
        print(f"copied → {result['copied']}")
