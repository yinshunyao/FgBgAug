# A1–A4 augmentation processing

> Language: English · [中文](../zh/A1-A4增强处理说明.md) · [README](../../README.md)

What each FgBgAug ablation arm **A1–A4** does to an image, default parameters, offline dump conventions, and how they relate to A0 / `yolo_auto`. Implementation source of truth: `fgbg_aug.py`, `dump_aug.py`.

Overview: [parameters-and-layout.md](parameters-and-layout.md)

---

## 1. Claim and ablation roles

This is **not** a MixUp / CutMix-style “sample mixing” operator. It is: **same image, weak FG + strong BG (dual intensity)**. At inference the model is unchanged with no extra cost; masks are used only when building training samples.

| ID | Mode (`AugMode`) | Offline dir | Role |
|:---|:---|:---|:---|
| A0 | `none` | `A0_none` (optional) | no fgbg; resize etc. only |
| **A1** | `global` | `A1_global` | **normal global aug baseline**: mild photometric + affine (not a harmful warp baseline) |
| **A2** | `fg_only` | `A2_fg_only` | FG only (weak photometric + rigid rotate); BG dug out / no strong BG aug |
| **A3** | `bg_only` | `A3_bg_only` | BG only (strong photometric / replace / noise-blur); **FG pixels pasted back unchanged** |
| **A4** | `fgbg` | `A4_fgbg` | **main method**: weak FG + strong BG (dual intensity) |
| — | `yolo_auto` | (production; not via dump) | clean images + Ultralytics built-in classify aug |

Mechanism narrative:

- A1: normal whole-image aug; rotation with **expand** so subject is not cropped.
- A2 / A4: weak FG, no hue, no scale; canvas expands after rotation.
- A3: FG almost unchanged (pixel paste-back); large BG change.
- A4 vs A1: spatially dual intensity; vs A2/A3: uses both strong BG and weak FG geometry.

---

## 2. Inputs and mask

Augmentation entry always needs:

| Input | Source (`by_class` / `splits/trainval`) | Use |
|:---|:---|:---|
| `image` RGB | `image/<breed>/<stem>.jpg` | main image |
| FG mask | `fg/<breed>/<stem>.png` **alpha > 0** | region split |
| `bg_image` (optional) | `bg/<breed>/<stem>.png` → RGB, or another sample at dump | A1/A3/A4 BG branch |

`bbox/` is for human review only — **not** used in augmentation.

Photometric jitter: amplitude `x` means factor `1 + Uniform(-x, x)`, then clamp `max(0.2, ·)`.

---

## 3. Default specs (`FgSpec` / `BgSpec` / `GlobalSpec`)

### 3.1 Foreground `FgSpec` (weak)

| Param | Default | Meaning |
|:---|:---|:---|
| `brightness` | `0.08` | FG brightness half-range |
| `contrast` | `0.08` | FG contrast half-range |
| `rotate_deg` | `90.0` | rigid rotation half-range (deg), angle ∈ `[-rotate_deg, +rotate_deg]` |

**Hard rule**: FG has **no Color (hue/saturation) jitter** and no scale, to avoid twisting patterns / coat color.

### 3.2 Background `BgSpec` (strong)

| Param | Default | Meaning |
|:---|:---|:---|
| `brightness` / `contrast` / `color` | `0.28` / `0.28` / `0.35` | strong BG photometric |
| `replace_prob` | `0.55` | with `bg_image`, full replace (resized) probability |
| `noise_sigma` | `10.0` | BG Gaussian noise σ |
| `blur_prob` | `0.25` | BG Gaussian blur probability (radius ≈ 1.2) |

When replace does not fire: dig out FG; ~35% solid canvas; then photometric / noise / blur on BG mask.

### 3.3 Whole-image `GlobalSpec` (A1: normal global)

| Param | Default | Meaning |
|:---|:---|:---|
| `brightness` / `contrast` / `color` | `0.15` / `0.15` / `0.10` | mild whole-image photometric |
| `rotate_deg` | `20.0` | whole-image rotation half-range; **expand=True** avoids crop |
| `scale` | `0.10` | mild scale half-range |
| `shear_deg` | `0.0` | no shear by default (keep “normal” global aug) |

---

## 4. Per-mode pipelines

Entry points:

- **A1 / A2 / A3**, and online A4: `augment(..., mode=...)`
- **Offline dump / mechanism-grid A4**: `augment_fgbg_pick(...)` (per-slot FG rotation + random complex BG; not same-sample `bg`)

### 4.1 A1 · `global` (normal global aug)

```text
RGB
  → affine: rotate (expand canvas) + mild scale; no shear by default
  → whole-image mild Brightness / Contrast / Color
  → output (may be larger than original; dump resizes by IMG_SIZE)
```

- Does **not** use mask regions (empty mask makes other modes fall back to original; global still processes the whole image).
- Baseline “ordinary global aug” vs dual-intensity A4 — not an intentionally harmful warp.

**No crop on rotate**: FG rigid rotate and A1 whole-image rotate both use `expand=True`; A2/A4 center-pad the BG canvas to the expanded FG size before blend.

### 4.2 A2 · `fg_only` (FG only)

```text
RGB + mask
  → canvas = dig out FG (fill FG region with BG mean color)
  → FG: weak Brightness/Contrast (no Color)
  → FG: rigid rotate (angle Uniform[-rotate_deg, +rotate_deg]), paste back
  → output = blend(canvas, rotated FG, new mask)
```

- BG region does **not** get `BgSpec` strong aug (quieter BG than A3/A4).
- Answers: “is FG-only enough / does weak FG preserve shape?”

### 4.3 A3 · `bg_only` (BG only)

```text
RGB + mask + optional bg_image
  → canvas = _make_background(...)   # replace / solid / strong photo / noise / blur
  → output = blend(canvas, original RGB, original mask)   # FG pixels unchanged
```

- FG geometry and photometric unchanged.
- Answers: “does strong BG help without hurting FG appearance?”

### 4.4 A4 · `fgbg` (dual intensity, main method)

#### (1) Online train path: `augment(..., mode="fgbg")`

```text
RGB + mask + optional bg_image
  → canvas = _make_background(...)     # same BG branch as A3
  → FG: weak photometric (no Color) + rigid rotate
  → output = blend(canvas, rotated FG, new mask)
```

#### (2) Offline dump / mechanism-grid path: `augment_fgbg_pick`

```text
For slot pick (0 … N-1):
  → FG: weak photometric + **per-slot angle** rotate (prefer larger ± angles; see _sample_angle_slot)
  → BG: make_random_complex_bg
        · with other_image_prob (default 0.65) wrap-crop another sample’s image
        · else: solid + noise blobs + stripes (not same-sample bg)
  → output = blend(complex BG, rotated FG, new mask)
```

Offline A4 and online A4 intentionally differ in **how BG is sampled**: dump/viz emphasize **cross-image random complex BG** for mechanism figures and offline ImageFolders; online may still use in-distribution `bg` + `BgSpec`. Document which path you used in the paper protocol.

---

## 5. Offline dump (`dump_aug.py`)

After `split_by_class`:

| Item | Convention |
|:---|:---|
| Source | `SPLIT_DIR/trainval/` (whole tree; **no** train/val split here) |
| Out | `AUG_OUT_ROOT` (`platform_config`, default `<PET_ROOT>/by_class-aug-A1A4`) |
| Multiplier | **`AUG_COUNT` (default 4)**: **same** for A1–A4 for fair offline compare |
| Naming | `<breed>/<stem>__aug01.jpg` … `__augNN.jpg` |
| test | **never augmented** |
| Idempotent | `SKIP_EXISTING=True`: skip existing large-enough files; do not delete old files |

Directory example:

```text
AUG_OUT_ROOT/
  A1_global/<breed>/<stem>__aug01.jpg … __aug04.jpg
  A2_fg_only/<breed>/…
  A3_bg_only/<breed>/…
  A4_fgbg/<breed>/…
```

Slot randomness:

- Per source image, per slot: `seed + idx * 10007 + slot * 17`
- A1–A3: independent `augment()` per slot; BG reference staggered by `idx + slot * 31`
- A4: per slot `augment_fgbg_pick(..., pick=slot, n_slots=AUG_COUNT)`

Optional `INCLUDE_A0=True` writes `A0_none`, also ×`AUG_COUNT` (original copies).

---

## 6. Relation to train / eval

| Path | How A1–A4 enter the net | Eval |
|:---|:---|:---|
| Offline dump then train | ImageFolder reads `AUG_OUT_ROOT/A*_*/` | val/test use clean splits, aug off |
| `compare.py` / `train.py` online | train calls `fgbg_aug.augment` (or online A4 branch); YOLO built-in aug **off** | same |
| `yolo_auto` | clean train + YOLO built-in aug **on** | same |

**Fair ablation**: only swap mode (A0–A4); keep the three Spec defaults and the same `AUG_COUNT` — do not boost only one arm’s multiplier.

---

## 7. Mechanism checks

1. `python visualize.py`: 1×5 grid — subject intact after rotate (no edge crop), A3 FG paste-back, A4 dual intensity.
2. Tuning: edit matching dataclass; **do not** add Color or scale to FG.
3. Main-table ablation: Pet/CUB protocols live in the paper experiment notes; this repo owns reproducible code and dirs.

---

## 8. Quick parameter map

| Want to change | Where |
|:---|:---|
| Paths / smoke / aug output root | `platform_config.py` |
| Aug copies per source | `dump_aug.py` → `AUG_COUNT` (default 4) |
| A4 cross-image BG probability | `dump_aug.py` → `OTHER_IMAGE_PROB`; same name in viz |
| FG / BG / whole-image strength | `FgSpec` / `BgSpec` / `GlobalSpec` in `fgbg_aug.py` |
| Whether to dump A0 | `dump_aug.py` → `INCLUDE_A0` |

Core code: `fgbg_aug.augment`, `fgbg_aug.augment_fgbg_pick`, `dump_aug.dump_mode`.
