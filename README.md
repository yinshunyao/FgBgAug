# FgBgAug

Foreground–background **dual-intensity** data augmentation for morphology-sensitive recognition.

[中文文档](README.zh-CN.md) · Docs: [en](docs/en/) / [zh](docs/zh/)

Same image, two strengths — not a new MixUp/CutMix operator:

| Branch | Behavior |
|:---|:---|
| Foreground | Rigid rotation + weak photometric (**no** hue shift, **no** scale) |
| Background | Strong photometric and/or resampled from another image / solid canvas |

Inference: unchanged model, zero extra cost (masks used only when building training samples).

## Modes (ablation)

| `AUG_MODE` | Meaning |
|:---|:---|
| `none` | Resize / flip only |
| `global` | Mild whole-image photometric + affine (normal global baseline) |
| `fg_only` | Weak FG only |
| `bg_only` | Strong BG only; FG pixels unchanged |
| `fgbg` | Full dual-intensity (this method) |

## Pipeline (recommended order)

```text
① extract_pet_by_class.py   → by_class/{image,fg,bg,bbox}/<breed>/
② split_by_class.py         → splits/{trainval,test}/ + txt lists
                              (Mac smoke subset; Linux full; copies image/fg/bg/bbox)
                              (configure TEST_RATIO / VAL_RATIO first; hold-out test, then train/val)
③ dump_aug.py               → augment all SPLIT_DIR/trainval → AUG_OUT_ROOT (A1–A4)
③b split_aug.py             → AUG_OUT_ROOT → AUG_SPLIT_ROOT/{A*}/{train,val}/ (default 8:2)
④ compare.py                → A0–A4 + yolo_auto; each arm reports val + test (eval: no aug)
```

| Step | Script | Role |
|:---|:---|:---|
| ① | `extract_pet_by_class.py` | by_class crop + fg/bg |
| ② | `split_by_class.py` | ratios preconfigured; hold-out test; write txt **and copy** `trainval/` / `test/` trees (Mac smoke / Linux full) |
| ③ | `dump_aug.py` | augment **all** of **trainval/** ×`AUG_COUNT` (default 4) → `AUG_OUT_ROOT`; leave test untouched |
| ③b | `split_aug.py` | with `INCLUDE_A0`, ensure/fill `A0_none`; if `AUG_VAL_USE_AUG=True`, val may include aug images (`AUG_RANDOM_SPLIT` optional file-level random); if `False`, split val by original stems (clean val) then put remaining stems’ aug into train; same split shared across modes |
| ④ | `compare.py` | prefer `AUG_SPLIT_ROOT` train/val + `SPLIT_DIR/test`; if ready, **no re-augment**. `__main__` flags `RERUN_SPLIT_BY_CLASS` / `RERUN_DUMP_AUG` / `RERUN_SPLIT_AUG` delete product dirs then re-run ②/③/③b (earlier steps cascade later) |

Compare arms:

| mode | Training | Ultralytics logs | Eval |
|:---|:---|:---|:---|
| `none` / `global` / `fg_only` / `bg_only` / `fgbg` | write fgbg into train ImageFolder, then `YOLO.train` with **YOLO built-in aug off** | same as production | val + test, no aug |
| `yolo_auto` | clean train images + `YOLO.train` with **YOLO built-in aug on** | standard classify logs | same val + test, no aug |

Note: older ablations used a custom PyTorch loop (one-line `mode=… loss=…`). Production used Ultralytics banners. Both arms now use `YOLO.train`, so log style matches.

Idempotency:

- **Split**: if full `train/val/test` lists **and** `trainval/image`+`test/image` trees exist, do not re-draw; only `FORCE_RESPLIT=True` rewrites. Legacy txt-only (no copy dirs) is treated incomplete and rewritten.
- **Offline aug**: `SKIP_EXISTING=True` — skip already-augmented samples; do not delete.

Helpers (optional, off the critical path):

```bash
python fgbg_aug.py     # smoke: load one sample, print photo-L2
python visualize.py    # mechanism grid 1×5 (A4 = FG×N pick + random complex BG)
python train.py        # single AUG_MODE (same data paths as compare; online aug)
```

## Quick start

```bash
pip install -r requirements.txt

# ① by_class
python extract_pet_by_class.py

# ② hold out test (default TEST_RATIO=0.1)
python split_by_class.py

# ③ dump aug (trainval only)
python dump_aug.py

# ③b split aug products into train/val (default 8:2)
python split_aug.py

# ④ compare (A0–A4 + yolo_auto; each arm val+test)
#    to re-run upstream: set RERUN_* in compare.py __main__
#    (delete products then run; ②⇒③⇒③b)
python compare.py
```

Default data roots / smoke (see ``platform_config.py``; branched by `sys.platform`):

```text
Mac:   /Volumes/shunyao-h1/基线数据/Oxford-IIIT Pet/by_class
Linux: /data/samples/base/Oxford-IIIT Pet/by_class
SPLIT_DIR        = <BY_CLASS_ROOT>/splits
AUG_OUT_ROOT     = <PET_ROOT>/by_class-aug-A1A4
AUG_SPLIT_ROOT   = <PET_ROOT>/by_class-aug-A1A4-split
AUG_VAL_RATIO    = 0.2   # train:val = 8:2 (by split unit)
AUG_VAL_USE_AUG  = True  # True: val also uses aug images; False: clean val by stem first
INCLUDE_A0       = True  # A0_none baseline (dump / split both include)
AUG_COUNT        = 4
AUG_RANDOM_SPLIT = False # False: same-source stems stay together; True: file-level random
                         # (only when AUG_VAL_USE_AUG=True)
TEST_RATIO  = 0.1
VAL_RATIO   = 0.1
Smoke: Mac 8×40; Linux MAX_CLASSES=MAX_PER_CLASS=0 (full)
```

`dump_aug.py` reads `SPLIT_DIR/trainval/`, writes `AUG_OUT_ROOT` (optional `A0_none`).  
`split_aug.py` builds `AUG_SPLIT_ROOT/{A0,A1…}/{train,val}/<breed>/` (shared split; if `INCLUDE_A0` and A0 missing/incomplete, fill from trainval).

Change paths, smoke, aug output, A0, `AUG_VAL_USE_AUG`, random split, or train/val ratio only in `platform_config.py`.

Training schedule (shared by all arms for fair compare):

```text
EPOCHS=100  PATIENCE=25  OPTIMIZER=AdamW  LR/lr0=1e-3  cos_lr=True
```

**Val / test are never augmented.** Production arm `yolo_auto` enables Ultralytics auto-aug only during **training**; ablation arms keep YOLO built-in aug off.

## Layout

```text
platform_config.py       # Mac/Linux paths + smoke defaults (shared by __main__)
fgbg_aug.py              # core augment() + by_class load (train / dump)
extract_pet_by_class.py  # ① Pet → by_class/{image,fg,bg,bbox}
split_by_class.py        # ② by_class → splits/{trainval,test}/ + txt
dump_aug.py              # ③ SPLIT_DIR/trainval → AUG_OUT_ROOT A1–A4 ImageFolder
split_aug.py             # ③b AUG_OUT_ROOT → AUG_SPLIT_ROOT/{A*}/{train,val}/
compare.py               # ④ A0–A4 compare (online fgbg_aug when needed)
train.py                 # single-mode train (same data paths as ④)
visualize.py             # mechanism figure (optional)
requirements.txt
docs/en/                 # English docs
docs/zh/                 # Chinese docs
```

`by_class` layout (① output; ②–④ / aug / viz consume):

```text
by_class/image/<breed>/*.jpg   # RGB crop → augment `image`
by_class/fg/<breed>/*.png      # RGBA → mask = alpha>0
by_class/bg/<breed>/*.png      # RGBA (optional; demo may synthesize bg_image)
by_class/bbox/<breed>/*.jpg    # human review only; not used in train/aug
by_class/splits/train.txt      # ② train stems (may be augmented)
by_class/splits/val.txt        # ② val stems (never augmented)
by_class/splits/test.txt       # ② test stems (never augmented)
by_class/splits/trainval.txt   # train∪val (legacy compatibility)
by_class/splits/trainval/      # ② copy: {image,fg,bg,bbox}/<breed>/
by_class/splits/test/          # ② copy: same (test set, no aug)
```

Metrics: `runs/compare/compare_summary.json` (val/test per mode).

Details: [docs/en/parameters-and-layout.md](docs/en/parameters-and-layout.md) · [docs/zh/参数与目录说明.md](docs/zh/参数与目录说明.md)  
A1–A4: [docs/en/a1-a4-augmentation.md](docs/en/a1-a4-augmentation.md) · [docs/zh/A1-A4增强处理说明.md](docs/zh/A1-A4增强处理说明.md)
