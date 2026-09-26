#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Platform-dependent defaults for FgBgAug demos / pipeline ``__main__``.

Edit **this file only** when Mac / Linux paths, active ``DATASET``, smoke limits,
aug dump/split roots, or ``AUG_*`` knobs change.
Scripts import resolved roots (``DATA_ROOT`` / ``BY_CLASS_ROOT`` / …) instead of
branching on ``sys.platform`` themselves.

Mac (``IS_DARWIN``): smoke subset — small ``MAX_CLASSES`` / ``MAX_PER_CLASS``.
Linux / servers: full run — both caps ``0`` (= no limit).
"""
from __future__ import annotations

import sys
from pathlib import Path

IS_DARWIN = sys.platform == "darwin"

# ---------------------------------------------------------------------------
# Active dataset for pipeline ①–④ (``compare.py`` / extract / split / dump)
# ---------------------------------------------------------------------------
# "pet" | "cub"
DATASET = "cub"

# ---------------------------------------------------------------------------
# Raw dataset roots (platform paths)
# ---------------------------------------------------------------------------
if IS_DARWIN:
    PET_ROOT = Path("/Volumes/shunyao-h1/基线数据/Oxford-IIIT Pet")
    CUB_ROOT = Path("/Volumes/shunyao-h1/基线数据/CUB_200_2011/CUB_200_2011")
else:
    PET_ROOT = Path("/data/samples/base/Oxford-IIIT Pet")
    CUB_ROOT = Path("/data/samples/base/CUB_200_2011")

if DATASET == "cub":
    DATA_ROOT = CUB_ROOT
elif DATASET == "pet":
    DATA_ROOT = PET_ROOT
else:
    raise ValueError(f"unknown DATASET={DATASET!r}; expected 'pet' or 'cub'")

# by_class + splits (pipeline ① / ②)
BY_CLASS_ROOT = DATA_ROOT / "by_class"
CUB_BY_CLASS_ROOT = CUB_ROOT / "by_class"  # alias when calling extract_cub directly
SPLIT_DIR = BY_CLASS_ROOT / "splits"

# Offline A1–A4 dump root (pipeline step ③; source = SPLIT_DIR/trainval)
AUG_OUT_ROOT = DATA_ROOT / "by_class-aug-A1A4"
# After dump: split each A* ImageFolder into train/val (pipeline step ③b)
AUG_SPLIT_ROOT = DATA_ROOT / "by_class-aug-A1A4-split"
# Fraction of items per breed assigned to val (train = 1 - ratio)
AUG_VAL_RATIO = 0.2
# True: current behavior — val also takes dump/aug images (same unit as train).
# False: val = originals only; ratio is over source stems (pick val stems first,
# then remaining stems' aug files go to train). Implies stem-based split.
AUG_VAL_USE_AUG = False
# Copy originals as A0_none (× AUG_COUNT) for fair comparison with A1–A4
INCLUDE_A0 = True
AUG_COUNT = 4
# False: keep source-stem grouping (all __augNN of one image → same train or val)
# True: randomly assign each file to train/val (same relative path shared across A*)
# Ignored when AUG_VAL_USE_AUG=False (always stem-first for clean val).
AUG_RANDOM_SPLIT = False


# ---------------------------------------------------------------------------
# Smoke subset (Mac: small; Linux / servers: full run)
# Applied by extract ① / split ② / dump ③ / compare ④
# ---------------------------------------------------------------------------
if IS_DARWIN:
    MAX_CLASSES = 8
    MAX_PER_CLASS = 40
else:
    MAX_CLASSES = 0  # 0 = all classes
    MAX_PER_CLASS = 0  # 0 = no per-class cap
