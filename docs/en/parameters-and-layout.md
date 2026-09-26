# Augmentation parameters and directory layout

> Language: English · [中文](../zh/参数与目录说明.md) · [README](../../README.md)

Self-contained notes for this repo. Numeric defaults live in each script’s `__main__` and the dataclasses in `fgbg_aug.py`.

**A1–A4 processing details** (per-mode pipelines, specs, offline ×N slots, online vs offline A4): see [a1-a4-augmentation.md](a1-a4-augmentation.md).

---

## 0. Recommended order

```text
① extract_pet_by_class.py  → by_class/{image,fg,bg,bbox}/
② split_by_class.py        → splits/{trainval,test}/ + txt (Mac smoke / Linux full)
③ dump_aug.py              → AUG_OUT_ROOT/A{1..4}_*/ (augment all of trainval)
③b split_aug.py            → AUG_SPLIT_ROOT/{A*}/{train,val}/ (default 8:2)
④ compare.py               → A0–A4 + yolo_auto; val+test (eval: no aug)
```

| Step | Script | Role |
|:---|:---|:---|
| ① | `extract_pet_by_class.py` | by_class crop + fg/bg |
| ② | `split_by_class.py` / `ensure_split` | hold-out test first; write txt **and copy** `trainval/`/`test/`; Mac smoke / Linux full; reuse when lists + dirs are complete |
| ③ | `dump_aug.py` | augment **all** of **SPLIT_DIR/trainval** → `AUG_OUT_ROOT` |
| ③b | `split_aug.py` | train/val split (`AUG_VAL_RATIO`); `AUG_VAL_USE_AUG` controls whether val uses aug; `INCLUDE_A0` requires A0 → `AUG_SPLIT_ROOT` |
| ④ | `compare.py` | prefer `AUG_SPLIT_ROOT` train/val + `splits/test/image`; skip aug when ready |

---

## 1. Ablation / production modes

| Value | Train-time aug | Role |
|:---|:---|:---|
| `none` | write unaugmented train images → `YOLO.train` (YOLO aug off) | A0 |
| `global` / `fg_only` / `bg_only` / `fgbg` | write corresponding fgbg images → `YOLO.train` (YOLO aug off) | A1–A4 |
| `yolo_auto` | clean images → `YOLO.train` (YOLO aug on) | production baseline |

Ablation and production both use the Ultralytics classify trainer — **same log format**. Difference is only “images already fgbg or not” and “YOLO built-in aug on/off”.

**Eval (val / test) always disables aug.** Model selection uses Ultralytics internal val; after training, re-measure `best_val_top1` / `test_top1_at_best_val` with a clean loader.

---

## 2. Augmentation specs (`fgbg_aug.py`)

Photometric fields are relative amplitudes: factor `1 + Uniform(-x, x)`, then clamp `max(0.2, ·)`.

### 2.1 `FgSpec` (foreground, weak by default)

| Param | Default | Meaning |
|:---|:---|:---|
| `brightness` | `0.08` | FG brightness jitter amplitude |
| `contrast` | `0.08` | FG contrast jitter amplitude |
| `rotate_deg` | `90.0` | rigid-rotation half-range (deg); angle ∈ `[-rotate_deg, +rotate_deg]` |

Foreground branch **never** applies Color jitter (hue/saturation), to avoid twisting coat patterns.

### 2.2 `BgSpec` (background, strong by default)

| Param | Default | Meaning |
|:---|:---|:---|
| `brightness` | `0.28` | BG region brightness jitter |
| `contrast` | `0.28` | BG region contrast jitter |
| `color` | `0.35` | BG color (saturation-like) jitter |
| `replace_prob` | `0.55` | with `bg_image`, probability of full replace (resized) |
| `noise_sigma` | `10.0` | BG Gaussian noise σ (pixel units) |
| `blur_prob` | `0.25` | BG Gaussian blur probability (radius ≈ 1.2) |

When replace does not fire: dig out FG, ~35% chance solid canvas, then photometric / noise / blur on the BG mask.

### 2.3 `GlobalSpec` (A1: normal global aug)

| Param | Default | Meaning |
|:---|:---|:---|
| `brightness` / `contrast` / `color` | `0.15` / `0.15` / `0.10` | mild whole-image photometric |
| `rotate_deg` | `20.0` | whole-image rotation half-range; `expand=True` avoids crop |
| `scale` | `0.10` | mild scale half-range |
| `shear_deg` | `0.0` | no shear by default |

After FG / whole-image rotation the canvas may exceed the original size; dump / `IMG_SIZE` resize back uniformly.

---

## 3. Script entry variables

### 3.1 Data and splits

| Variable | Script | Meaning |
|:---|:---|:---|
| `DATA_ROOT` / `BY_CLASS_ROOT` | `fgbg_aug` / `train` / `visualize` | extracted `by_class` root |
| `SPLIT_DIR` | `train.py` / `compare.py` | dir of `trainval.txt` / `test.txt`; prefer `BY_CLASS_ROOT/splits` (from `split_by_class.py`); official `annotations` also OK |
| `SAMPLE_INDEX` | `fgbg_aug.py` | which sample for smoke |

### 3.2 Training (`train.py`)

| Variable | Meaning |
|:---|:---|
| `AUG_MODE` | see §1 |
| `WEIGHTS` | backbone: `resnet18` / `yolo11s.pt` / `yolo11l.pt` (detect names map to `*-cls.pt`) |
| `IMG_SIZE` | input side length (default 224) |
| `BATCH` / `EPOCHS` / `LR` / `SEED` | batch, epochs, AdamW lr, seed |
| `MAX_CLASSES` | `0`=all breeds; `>0` first N classes |
| `MAX_PER_CLASS` | `0`=full; `>0` at most N train images per class |
| `NUM_WORKERS` | DataLoader workers |
| `RUN_ROOT` | output root (default `runs/` in this repo) |

### 3.3 Visualization (`visualize.py`)

Five columns remain A0–A4. **Only the A4 `fgbg` column** uses “FG×N pick one + random complex BG” (not same-sample `bg`); other columns still call `augment()`.

| Variable | Meaning |
|:---|:---|
| `N_IMAGES` | how many sample rows |
| `IMG_SIZE` / `SEED` | cell size, seed |
| `OUT_DIR` | mechanism-figure dir (`pet_aug_grid.png`) |
| `FG_AUG_COUNT` | A4 FG slot count (default 4) |
| `FG_AUG_PICK` | which slot to show (0-based; pick 1 is fine) |
| `OTHER_IMAGE_PROB` | A4 BG from other `by_class/image` wrap-crop; else solid+noise+stripes |

Note: **online A4 in train / compare** still uses `augment(..., mode="fgbg")`; **mechanism grid / `dump_aug.py` A4** uses `augment_fgbg_pick` (FG×N + random complex BG).

### 3.4 Extract (`extract_pet_by_class.py`) — ① first by_class step

| Variable | Meaning |
|:---|:---|
| `PET_ROOT` | raw Oxford Pet (`images/` + `annotations/trimaps/`) |
| `OUT_DIR` | target root for `by_class` |
| `PAD_RATIO` | body-bbox relative padding |
| `SKIP_EXISTING` | skip when all four file types exist |

### 3.5 Split (`split_by_class.py` / `ensure_split`) — ②

Configure three ratios up front; **flow unchanged**: hold-out test, then in remainder carve val by `VAL_RATIO` (of total), rest = train.  
Subset: ``MAX_CLASSES`` / ``MAX_PER_CLASS`` from `platform_config` (Mac smoke; Linux full).

| Variable | Meaning |
|:---|:---|
| `TEST_RATIO` | test fraction of total (default `0.1`) |
| `VAL_RATIO` | val fraction of total; train = 1 − test − val (`0` → no val) |
| `SEED` / `SPLIT_SEED` | split seed |
| `MAX_CLASSES` / `MAX_PER_CLASS` | smoke caps; `0`=full |
| `FORCE` / `FORCE_RESPLIT` | default reuse complete lists+copy dirs; rewrite if any missing |

Outputs:

- Lists: `train.txt`, `val.txt`, `test.txt`, `trainval.txt` (=train∪val), `split_summary.json`
- Copy trees (same layout as by_class): `trainval/{image,fg,bg,bbox}/<breed>/`, `test/{image,fg,bg,bbox}/<breed>/`

### 3.6 Offline dump (`dump_aug.py`) — ③

After ②. Source is always ``SPLIT_DIR/trainval/`` (whole tree; **no** train/val split here); write to ``AUG_OUT_ROOT`` (`platform_config`).

| Variable | Meaning |
|:---|:---|
| `SPLIT_DIR` / `trainval` | trainval by_class tree copied in ② |
| `AUG_OUT_ROOT` | output root; contains `A1_global` … `A4_fgbg` |
| `SKIP_EXISTING` | default `True`: skip existing files; **do not re-aug or delete** |
| `AUG_COUNT` | slots per source image for A1–A4 (default `4`; same multiplier for fairness) |
| `OTHER_IMAGE_PROB` | A4 random BG from other images |
| `IMG_SIZE` | `>0` uniform resize; `0` keep crop size |
| `INCLUDE_A0` | if `True`, also write `A0_none` (same ×`AUG_COUNT`) |

Example: `AUG_OUT_ROOT/A1_global/<breed>/<stem>__aug01.jpg` … `__aug04.jpg` (A2–A4 same).

### 3.6b Post-aug train/val (`split_aug.py`) — ③b

After ③. Default train:val = 8:2; **same split applied to every A\*** for fair compare.

| Variable (`platform_config`) | Meaning |
|:---|:---|
| `AUG_OUT_ROOT` | ③ dump root |
| `AUG_SPLIT_ROOT` | output root; `A*_*/{train,val}/<breed>/` (incl. `A0_none`) |
| `AUG_VAL_RATIO` | val fraction (default `0.2` → 8:2) |
| `AUG_VAL_USE_AUG` | `True` (default): val also uses dump/aug files; `False`: carve val by **original stem** first (one original per stem in val), remaining stems’ aug → train; ignores `AUG_RANDOM_SPLIT` |
| `INCLUDE_A0` | `True`: must include `A0_none`; fill from trainval if missing/incomplete |
| `AUG_COUNT` | same multiplier as dump; A0 also writes N original copies for matched sample size |
| `AUG_RANDOM_SPLIT` | only when `AUG_VAL_USE_AUG=True`. `False` (default): all `__augNN` of one source stay on the same side; `True`: file-level random (same `breed/filename` assignment across A*) |

If `AUG_OUT_ROOT` has no / incomplete `A0_none`, `split_aug` copies originals from `SPLIT_DIR/trainval/image` (same resize as dump) then splits — **no need to re-run A1–A4**. With `INCLUDE_A0=False`, A0 on disk is still not written into the split.

Also writes `train_stems.txt` / `val_stems.txt` (or `train_files.txt` / `val_files.txt` under random mode) and `split_summary.json`.

### 3.7 Compare train (`compare.py` / `train.py`) — ④

**Prefer offline production data** (if dirs ready, **no** online aug / re-dump):

| Use | Path |
|:---|:---|
| train / val | `AUG_SPLIT_ROOT/<A0_none\|A1_…\|A4_…>/{train,val}/` |
| test | `SPLIT_DIR/test/image/` |

`yolo_auto` uses clean `A0_none` train + YOLO built-in aug on; ablation arms keep YOLO aug off.  
If incomplete, fall back to older path (materialize from `by_class` via `splits/*.txt`).

Each arm reports `best_val_top1` and `test_top1_at_best_val` (eval: no aug). Default `MODES` includes A0–A4 and `yolo_auto`.

| Variable (compare) | Meaning |
|:---|:---|
| `TEST_RATIO` / `VAL_RATIO` / `SPLIT_SEED` | passed to `ensure_split` |
| `MODES` | default `ALL_MODES` = five ablations + `yolo_auto` |
| `WEIGHTS` | YOLO production path needs `yolo11*.pt` |
| `EPOCHS` | max epochs (Pet default **100**; with early stop) |
| `PATIENCE` | early stop after this many val epochs with no gain (default **25**; `0`=off) |
| `OPTIMIZER` | shared across arms, default **`AdamW`** (fair compare) |
| `LR` / `LRF` / `COS_LR` / `WEIGHT_DECAY` | → Ultralytics `lr0`/`lrf`/`cos_lr`/`weight_decay` |
| other | `SEEDS` / smoke subset, etc. |

---

## 4. Directory roles

### 4.1 Input: `by_class/` (usually on external disk)

Examples (see `platform_config.py`, branched by `sys.platform`):

- Mac: `/Volumes/shunyao-h1/基线数据/Oxford-IIIT Pet/by_class`
- Linux: `/data/samples/base/Oxford-IIIT Pet/by_class`

| Subdir | Content | Use |
|:---|:---|:---|
| `image/<breed>/*.jpg` | body-box RGB crop | `augment` `image`; train main image |
| `fg/<breed>/*.png` | RGBA, trimap==1 opaque | `mask = alpha>0` |
| `bg/<breed>/*.png` | RGBA, trimap==2 opaque | optional; demo may synthesize BG RGB |
| `bbox/<breed>/*.jpg` | original + green body box | **human review only**; unused in train/aug |
| `splits/train.txt` | train stems | may be augmented |
| `splits/val.txt` | val stems | **never augmented**; model selection |
| `splits/test.txt` | test stems | **never augmented**; final compare |
| `splits/trainval.txt` | train∪val | legacy |
| `splits/trainval/` | copied train∪val by_class tree | `{image,fg,bg,bbox}/<breed>/` |
| `splits/test/` | copied test by_class tree | same; no aug |

### 4.2 Output: repo `runs/`

| Path | Use |
|:---|:---|
| `runs/compare/<mode>_<tag>_s<seed>/metrics.json` | includes `best_val_top1`, `test_top1_at_best_val` |
| `runs/compare/compare_summary.json` | mean±std val/test per mode |
| `runs/compare/yolo_auto_*/ultralytics/` | production Ultralytics train dir |

### 4.3 Files in this repo

| Path | Use |
|:---|:---|
| `fgbg_aug.py` | core `augment()` + by_class load (③④ / viz) |
| `extract_pet_by_class.py` | ① raw Pet → `by_class` |
| `split_by_class.py` | ② → `splits/{trainval,test}/` + txt (Mac smoke / Linux full) |
| `dump_aug.py` | ③ trainval → `AUG_OUT_ROOT` A1–A4 (whole-dir aug) |
| `split_aug.py` | ③b → `AUG_SPLIT_ROOT/{A*}/{train,val}/` (default 8:2) |
| `compare.py` | ④ calls `train_one` for A0–A4 (online aug when needed) |
| `train.py` | single-mode train (same data paths as ④, online aug) |
| `visualize.py` | mechanism figure (optional) |
| `docs/en/` / `docs/zh/` | docs (incl. [a1-a4-augmentation.md](a1-a4-augmentation.md)) |
| external `PET_ROOT` | raw download; read by extract script |

---

## 5. Tuning tips

1. Main path: §0 — ① → ② → (optional ③) → ④.
2. Fair ablation: only change `AUG_MODE` / compare list; keep the three Spec defaults.
3. Strength: edit the matching dataclass; **do not** add Color or scale to FG.
4. Mechanism grid: `python visualize.py` → `runs/viz/`.
5. Single mode: `python train.py`. Compare table: `python compare.py` → `runs/compare/compare_summary.json`.
