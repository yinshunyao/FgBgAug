#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Platform-dependent defaults for FgBgAug demos / pipeline ``__main__``.

Edit **this file only** when Mac / Linux paths, smoke limits, aug dump/split roots,
``AUG_VAL_RATIO``, ``AUG_VAL_USE_AUG``, ``INCLUDE_A0``, ``AUG_COUNT``, or
``AUG_RANDOM_SPLIT`` change.
Scripts import ``BY_CLASS_ROOT`` / ``AUG_OUT_ROOT`` / ``AUG_SPLIT_ROOT`` / … instead of
branching on ``sys.platform`` themselves.
"""
from __future__ import annotations

import sys
from pathlib import Path

IS_DARWIN = sys.platform == "darwin"

# ---------------------------------------------------------------------------
# Data roots
# ---------------------------------------------------------------------------
if IS_DARWIN:
    PET_ROOT = Path("/Volumes/shunyao-h1/基线数据/Oxford-IIIT Pet")
else:
    PET_ROOT = Path("/data/samples/base/Oxford-IIIT Pet")

BY_CLASS_ROOT = PET_ROOT / "by_class"
SPLIT_DIR = BY_CLASS_ROOT / "splits"
# Offline A1–A4 dump root (pipeline step ③; source = SPLIT_DIR/trainval)
AUG_OUT_ROOT = PET_ROOT / "by_class-aug-A1A4"
# After dump: split each A* ImageFolder into train/val (pipeline step ③b)
AUG_SPLIT_ROOT = PET_ROOT / "by_class-aug-A1A4-split"
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
# ---------------------------------------------------------------------------
if IS_DARWIN:
    MAX_CLASSES = 8
    MAX_PER_CLASS = 40
else:
    MAX_CLASSES = 0  # 0 = all classes
    MAX_PER_CLASS = 0  # 0 = no per-class cap
