#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Foreground / background dual-intensity augmentation (paper demo).

Same image, two strengths. Not a new MixUp/CutMix operator.
Foreground: rigid rotation + weak photometric (no hue, no scale).
Background: strong photometric and/or resampled from another image / solid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

AugMode = Literal["none", "global", "fg_only", "bg_only", "fgbg"]


@dataclass(frozen=True)
class FgSpec:
    brightness: float = 0.08
    contrast: float = 0.08
    rotate_deg: float = 90.0


@dataclass(frozen=True)
class BgSpec:
    brightness: float = 0.28
    contrast: float = 0.28
    color: float = 0.35
    replace_prob: float = 0.55
    noise_sigma: float = 10.0
    blur_prob: float = 0.25


@dataclass(frozen=True)
class GlobalSpec:
    """Whole-image mild/standard aug (A1 comparison arm — not a harmful baseline)."""

    brightness: float = 0.15
    contrast: float = 0.15
    color: float = 0.10
    rotate_deg: float = 20.0
    scale: float = 0.10
    shear_deg: float = 0.0


def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _blend(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    out = src.copy()
    out[m] = dst[m]
    return out


def _jitter_whole(
    img: Image.Image,
    rng: np.random.Generator,
    *,
    brightness: float,
    contrast: float,
    color: float,
) -> Image.Image:
    b = 1.0 + float(rng.uniform(-brightness, brightness))
    c = 1.0 + float(rng.uniform(-contrast, contrast))
    k = 1.0 + float(rng.uniform(-color, color))
    out = ImageEnhance.Brightness(img).enhance(max(0.2, b))
    out = ImageEnhance.Contrast(out).enhance(max(0.2, c))
    if color > 1e-6:
        out = ImageEnhance.Color(out).enhance(max(0.2, k))
    return out


def _photometric_masked(
    rgb: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    brightness: float,
    contrast: float,
    color: float,
) -> np.ndarray:
    jittered = _to_np(_jitter_whole(_to_pil(rgb), rng, brightness=brightness, contrast=contrast, color=color))
    return _blend(rgb, jittered, mask)


def _solid_bg(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    roll = float(rng.random())
    if roll < 0.45:
        v = int(rng.integers(210, 251))
    elif roll < 0.7:
        v = int(rng.integers(8, 46))
    else:
        v = int(rng.integers(90, 171))
    color = np.clip(v + rng.integers(-12, 13, size=3), 0, 255).astype(np.uint8)
    return np.broadcast_to(color, (h, w, 3)).copy()


def _resize_np(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return _to_np(_to_pil(arr).resize((size[1], size[0]), Image.BILINEAR))


def _punch_fg(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    inv = ~mask.astype(bool)
    if np.any(inv):
        fill = rgb[inv].mean(axis=0).astype(np.uint8)
    else:
        fill = np.array([128, 128, 128], dtype=np.uint8)
    out[mask.astype(bool)] = fill
    return out


def _make_background(
    rgb: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    bg_rgb: Optional[np.ndarray],
    spec: BgSpec,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    if bg_rgb is not None and float(rng.random()) < spec.replace_prob:
        return _resize_np(bg_rgb, (h, w))

    bg = _punch_fg(rgb, mask)
    if float(rng.random()) < 0.35:
        bg = _solid_bg(h, w, rng)
    inv = ~mask.astype(bool)
    bg = _photometric_masked(
        bg,
        inv,
        rng,
        brightness=spec.brightness,
        contrast=spec.contrast,
        color=spec.color,
    )
    if spec.noise_sigma > 0:
        noise = rng.normal(0.0, spec.noise_sigma, size=bg.shape)
        noisy = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        bg = _blend(bg, noisy, inv)
    if spec.blur_prob > 0 and float(rng.random()) < spec.blur_prob:
        blurred = _to_np(_to_pil(bg).filter(ImageFilter.GaussianBlur(radius=1.2)))
        bg = _blend(bg, blurred, inv)
    return bg


def _expand_rgb_to(
    rgb: np.ndarray,
    nh: int,
    nw: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Center-pad (or center-crop) RGB to ``(nh, nw)``."""
    h, w = rgb.shape[:2]
    if h == nh and w == nw:
        return rgb
    out = np.empty((nh, nw, 3), dtype=np.uint8)
    out[:, :] = np.asarray(fill, dtype=np.uint8)
    if h > nh or w > nw:
        sy = max(0, (h - nh) // 2)
        sx = max(0, (w - nw) // 2)
        patch = rgb[sy : sy + nh, sx : sx + nw]
        ph, pw = patch.shape[:2]
        y0 = (nh - ph) // 2
        x0 = (nw - pw) // 2
        out[y0 : y0 + ph, x0 : x0 + pw] = patch
        return out
    y0 = (nh - h) // 2
    x0 = (nw - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = rgb
    return out


def _rotate_foreground(
    rgb: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-rotate FG with ``expand=True`` so the full silhouette fits (no clip)."""
    alpha = (mask.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate([rgb, alpha], axis=2)
    rot = Image.fromarray(rgba, mode="RGBA").rotate(
        angle_deg, resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0, 0)
    )
    arr = np.array(rot, dtype=np.uint8)
    return arr[:, :, :3], arr[:, :, 3] > 0


def _sample_angle_slot(
    rng: np.random.Generator,
    *,
    degrees: float,
    rotate_min_abs: float,
    slot: int,
    n_slots: int,
) -> float:
    """Layered angles across slots (insect 03 style): prefer large ±angles first."""
    d = float(degrees)
    if d <= 0.0:
        return 0.0
    ma = min(float(rotate_min_abs), d) if rotate_min_abs > 0 else 0.0
    n = max(1, int(n_slots))
    i = int(slot) % n
    if n <= 1:
        ang = float(rng.uniform(-d, d))
        if ma > 0.0 and abs(ang) < ma:
            sign = 1.0 if ang >= 0.0 else -1.0
            ang = sign * float(rng.uniform(ma, d))
        return ang
    if ma <= 0.0:
        lin = np.linspace(-d, d, num=n, dtype=np.float64)
        base = float(lin[i])
        span = float(2.0 * d / float(max(n - 1, 1)))
        jitter = min(span * 0.28, d * 0.12)
        return float(np.clip(base + float(rng.uniform(-jitter, jitter)), -d, d))
    n_neg = (n + 1) // 2
    n_pos = n - n_neg
    neg = np.linspace(-d, -ma, num=max(1, n_neg), dtype=np.float64)
    pos = np.linspace(d, ma, num=max(1, n_pos), dtype=np.float64)
    if i % 2 == 0:
        k = min(i // 2, n_neg - 1)
        base = float(neg[k])
    else:
        k = min(i // 2, max(0, n_pos - 1))
        base = float(pos[k])
    step_neg = float(abs(neg[1] - neg[0])) if n_neg > 1 else float(d - ma)
    step_pos = float(abs(pos[1] - pos[0])) if n_pos > 1 else float(d - ma)
    slot_w = max(step_neg, step_pos, 1e-6)
    jitter = min(slot_w * 0.18, float(d - ma) * 0.08, 4.0)
    ang = float(np.clip(base + float(rng.uniform(-jitter, jitter)), -d, d))
    if -ma < ang < ma:
        ang = float(ma if ang >= 0.0 else -ma)
    return ang


def augment_foreground_slot(
    image: Image.Image,
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    slot: int,
    n_slots: int = 4,
    fg_spec: FgSpec = FgSpec(),
    rotate_min_abs: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Weak FG photometric + rigid rotate for one of N slots.

    Returns ``(rgb, mask)`` on an **expanded** canvas when rotation needs more room.
    """
    rgb = _to_np(image)
    mask = mask.astype(bool)
    if mask.shape[:2] != rgb.shape[:2]:
        mask = (
            np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.NEAREST
                )
            )
            > 0
        )
    if not np.any(mask):
        return rgb, mask
    fg_rgb = _photometric_masked(
        rgb,
        mask,
        rng,
        brightness=fg_spec.brightness,
        contrast=fg_spec.contrast,
        color=0.0,
    )
    angle = _sample_angle_slot(
        rng,
        degrees=fg_spec.rotate_deg,
        rotate_min_abs=rotate_min_abs,
        slot=slot,
        n_slots=n_slots,
    )
    return _rotate_foreground(fg_rgb, mask, angle)


def _wrap_crop_rgb(src: np.ndarray, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    hb, wb = src.shape[:2]
    y0 = int(rng.integers(0, max(1, hb)))
    x0 = int(rng.integers(0, max(1, wb)))
    ys = (np.arange(h, dtype=np.int32) + y0) % hb
    xs = (np.arange(w, dtype=np.int32) + x0) % wb
    return src[np.ix_(ys, xs)].copy()


def _luma_u8(rgb: np.ndarray) -> float:
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    return 0.299 * r + 0.587 * g + 0.114 * b


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """h∈[0,1), s,v∈[0,1] → RGB uint8."""
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _contrast_rgb(
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    intensity: int,
    multicolor: bool = True,
) -> tuple[int, int, int]:
    """Pick a mark color that stays visible on ``base`` (opposite luma + optional hue)."""
    intensity = max(1, int(intensity))
    want_dark = _luma_u8(base) >= 128.0
    if float(rng.random()) < 0.12:
        want_dark = not want_dark
    if multicolor and float(rng.random()) < 0.75:
        hh = float(rng.random())
        s = float(rng.uniform(0.02, 0.12)) if float(rng.random()) < 0.22 else float(rng.uniform(0.22, 0.72))
        if want_dark:
            vv = float(rng.uniform(0.10, min(0.55, 0.18 + intensity / 120.0)))
        else:
            vv = float(rng.uniform(max(0.58, 0.72 - intensity / 200.0), 0.97))
        return _hsv_to_rgb(hh, s, vv)
    lo = max(1, intensity // 2)
    if want_dark:
        return tuple(
            int(np.clip(int(c) - int(rng.integers(lo, intensity + 1)), 0, 255)) for c in base[:3]
        )
    return tuple(
        int(np.clip(int(c) + int(rng.integers(lo, intensity + 1)), 0, 255)) for c in base[:3]
    )


def _complex_solid_bg(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Solid canvas + dense spots / blocks / stripes (stronger than mild pad defaults)."""
    roll = float(rng.random())
    if roll < 0.5:
        v = int(rng.integers(210, 251))
    elif roll < 0.6:
        v = int(rng.integers(100, 171))
    elif roll < 0.85:
        v = int(rng.integers(8, 46))
    else:
        v = int(rng.integers(0, 256))
    base = np.clip(v + rng.integers(-10, 11, size=3), 0, 255).astype(np.uint8)
    patch = np.empty((h, w, 3), dtype=np.uint8)
    patch[:, :] = base
    pil = _to_pil(patch)
    draw = ImageDraw.Draw(pil)
    min_side = min(h, w)

    # Spots (dots): higher density + contrast colors (was density 0.04 / ± ±22)
    spot_density = 0.10
    n_spots = max(8, int(h * w * spot_density / 40.0))
    for _ in range(n_spots):
        rad = int(rng.choice([1, 2, 3, 4, 5, 7, 9], p=[0.22, 0.22, 0.2, 0.14, 0.1, 0.07, 0.05]))
        cy = int(rng.integers(0, h))
        cx = int(rng.integers(0, w))
        color = _contrast_rgb(base, rng, intensity=42, multicolor=True)
        draw.rectangle([cx - rad, cy - rad, cx + rad, cy + rad], fill=color)

    # Blocks: larger rectangles / squares (was missing as a separate layer)
    n_blocks = max(2, int(rng.integers(3, 8)))
    for _ in range(n_blocks):
        bw = int(rng.integers(max(6, min_side // 18), max(8, min_side // 5)))
        bh = int(rng.integers(max(6, min_side // 18), max(8, min_side // 5)))
        if float(rng.random()) < 0.35:
            bh = bw
        x0 = int(rng.integers(0, max(1, w - bw)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        color = _contrast_rgb(base, rng, intensity=48, multicolor=True)
        if float(rng.random()) < 0.55:
            draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=color)
        else:
            draw.ellipse([x0, y0, x0 + bw, y0 + bh], fill=color)

    # Stripes / wavy lines: denser, thicker, high-contrast multicolor
    stripe_density = 0.055
    n_stripes = max(4, int((w + h) * stripe_density))
    max_width = max(2, min(10, int(min_side * 0.022) + 1))
    for _ in range(n_stripes):
        color = _contrast_rgb(base, rng, intensity=48, multicolor=True)
        width = int(rng.integers(1, max_width + 1))
        amp = float(rng.uniform(0.0, 0.18)) * min_side * float(rng.uniform(0.35, 1.0))
        period = float(rng.uniform(max(8.0, min_side * 0.10), max(8.0, min_side * 0.60)))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        orient = float(rng.random())
        pts: list[tuple[int, int]] = []
        if orient < 0.4:
            y_base = float(rng.uniform(0, max(1, h - 1)))
            for x in range(0, w, 3):
                yi = int(np.clip(y_base + amp * np.sin(2.0 * np.pi * x / period + phase), 0, h - 1))
                pts.append((x, yi))
        elif orient < 0.8:
            x_base = float(rng.uniform(0, max(1, w - 1)))
            for y in range(0, h, 3):
                xi = int(np.clip(x_base + amp * np.sin(2.0 * np.pi * y / period + phase), 0, w - 1))
                pts.append((xi, y))
        else:
            angle = float(rng.uniform(0.0, np.pi))
            x0 = float(rng.uniform(0, max(1, w - 1)))
            y0 = float(rng.uniform(0, max(1, h - 1)))
            dx, dy = float(np.cos(angle)), float(np.sin(angle))
            nx, ny = -dy, dx
            length = int(np.hypot(w, h))
            for t in range(0, length, 3):
                px = x0 + dx * t + nx * amp * np.sin(2.0 * np.pi * t / period + phase)
                py = y0 + dy * t + ny * amp * np.sin(2.0 * np.pi * t / period + phase)
                pts.append((int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)
    return _to_np(pil)


def _load_by_class_image_rgb(root: Path, breed: str, stem: str) -> Optional[np.ndarray]:
    for ext in (".jpg", ".jpeg", ".png"):
        path = root / "image" / breed / f"{stem}{ext}"
        if path.is_file():
            return _to_np(Image.open(path).convert("RGB"))
    return None


def make_random_complex_bg(
    h: int,
    w: int,
    rng: np.random.Generator,
    *,
    root: Optional[Path] = None,
    samples: Optional[list[tuple[str, str]]] = None,
    exclude: Optional[tuple[str, str]] = None,
    other_image_prob: float = 0.65,
) -> np.ndarray:
    """Random complex BG: other by_class image crop/tile, or solid+spots/blocks/stripes. Never uses ``exclude``."""
    pool = [s for s in (samples or []) if exclude is None or s != exclude]
    if (
        root is not None
        and pool
        and float(rng.random()) < float(other_image_prob)
    ):
        for _ in range(8):
            breed, stem = pool[int(rng.integers(0, len(pool)))]
            src = _load_by_class_image_rgb(root, breed, stem)
            if src is None:
                continue
            return _wrap_crop_rgb(src, h, w, rng)
    return _complex_solid_bg(h, w, rng)


def augment_fgbg_pick(
    image: Image.Image,
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    root: Path,
    samples: list[tuple[str, str]],
    exclude: tuple[str, str],
    n_slots: int = 4,
    pick: int = 0,
    fg_spec: FgSpec = FgSpec(),
    rotate_min_abs: float = 30.0,
    other_image_prob: float = 0.65,
) -> Image.Image:
    """Viz / pipeline A4: one of N FG augs + random complex BG (not same-sample bg)."""
    rot_rgb, rot_mask = augment_foreground_slot(
        image,
        mask,
        rng,
        slot=int(pick),
        n_slots=int(n_slots),
        fg_spec=fg_spec,
        rotate_min_abs=rotate_min_abs,
    )
    h, w = rot_rgb.shape[:2]
    canvas = make_random_complex_bg(
        h,
        w,
        rng,
        root=root,
        samples=samples,
        exclude=exclude,
        other_image_prob=other_image_prob,
    )
    return _to_pil(_blend(canvas, rot_rgb, rot_mask))


def _affine_global(rgb: np.ndarray, rng: np.random.Generator, spec: GlobalSpec) -> np.ndarray:
    """Mild whole-image affine; rotation uses ``expand=True`` to avoid clipping."""
    angle = float(rng.uniform(-spec.rotate_deg, spec.rotate_deg))
    scale = float(rng.uniform(1.0 - spec.scale, 1.0 + spec.scale))
    shear = float(rng.uniform(-spec.shear_deg, spec.shear_deg))
    im = _to_pil(rgb)
    im = im.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0))
    nw, nh = im.size
    nw2, nh2 = max(1, int(round(nw * scale))), max(1, int(round(nh * scale)))
    im = im.resize((nw2, nh2), Image.BILINEAR)
    if abs(shear) > 1e-6:
        # Shear on expanded canvas; pad width so horizontal shear does not clip.
        pad_x = int(abs(np.tan(np.deg2rad(shear))) * nh2) + 2
        canvas = Image.new("RGB", (nw2 + 2 * pad_x, nh2), (0, 0, 0))
        canvas.paste(im, (pad_x, 0))
        w_out, h_out = canvas.size
        canvas = canvas.transform(
            (w_out, h_out),
            Image.AFFINE,
            (1.0, np.tan(np.deg2rad(shear)), 0.0, 0.0, 1.0, 0.0),
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        im = canvas
    return _to_np(im)


def augment(
    image: Image.Image,
    mask: np.ndarray,
    rng: np.random.Generator,
    mode: AugMode,
    bg_image: Optional[Image.Image] = None,
    fg_spec: FgSpec = FgSpec(),
    bg_spec: BgSpec = BgSpec(),
    global_spec: GlobalSpec = GlobalSpec(),
) -> Image.Image:
    rgb = _to_np(image)
    mask = mask.astype(bool)
    if mask.shape[:2] != rgb.shape[:2]:
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 0
    if mode == "none" or not np.any(mask):
        return _to_pil(rgb)

    bg_rgb = _to_np(bg_image) if bg_image is not None else None

    if mode == "global":
        out = _affine_global(rgb, rng, global_spec)
        out = _to_np(
            _jitter_whole(
                _to_pil(out),
                rng,
                brightness=global_spec.brightness,
                contrast=global_spec.contrast,
                color=global_spec.color,
            )
        )
        return _to_pil(out)

    if mode == "bg_only":
        canvas = _make_background(rgb, mask, rng, bg_rgb, bg_spec)
        return _to_pil(_blend(canvas, rgb, mask))

    canvas = _make_background(rgb, mask, rng, bg_rgb, bg_spec) if mode == "fgbg" else _punch_fg(rgb, mask)
    fg_rgb = _photometric_masked(
        rgb,
        mask,
        rng,
        brightness=fg_spec.brightness,
        contrast=fg_spec.contrast,
        color=0.0,
    )
    angle = float(rng.uniform(-fg_spec.rotate_deg, fg_spec.rotate_deg))
    rot_rgb, rot_mask = _rotate_foreground(fg_rgb, mask, angle)
    nh, nw = rot_rgb.shape[:2]
    if canvas.shape[0] != nh or canvas.shape[1] != nw:
        # Expand BG to fit rotated FG (avoid truncation after expand=True rotate).
        fill = tuple(int(x) for x in np.median(canvas.reshape(-1, 3), axis=0))
        canvas = _expand_rgb_to(canvas, nh, nw, fill=fill)
    return _to_pil(_blend(canvas, rot_rgb, rot_mask))


def foreground_l2(orig: np.ndarray, aug: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.sum() < 32:
        return 0.0
    d = orig[m].astype(np.float32) - aug[m].astype(np.float32)
    return float(np.sqrt((d * d).mean()))


def trimap_to_fg(seg: Image.Image | np.ndarray) -> np.ndarray:
    """Oxford-IIIT Pet trimap: 1=fg, 2=bg, 3=border. Keep 1 and 3 as foreground."""
    arr = np.array(seg)
    return arr != 2


def mask_from_fg_rgba(fg: Image.Image | np.ndarray) -> np.ndarray:
    """Boolean FG mask from by_class ``fg/*.png`` (alpha > 0)."""
    arr = np.array(fg)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        return arr[:, :, 3] > 0
    if arr.ndim == 2:
        return arr > 0
    raise ValueError(f"expected RGBA fg, got shape={arr.shape}")


def bg_rgb_from_rgba(bg: Image.Image | np.ndarray, fill: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """Composite by_class ``bg/*.png`` onto a solid canvas → RGB for ``bg_image``."""
    arr = np.array(bg.convert("RGBA") if isinstance(bg, Image.Image) else bg)
    rgb = np.empty(arr.shape[:2] + (3,), dtype=np.uint8)
    rgb[:, :] = np.asarray(fill, dtype=np.uint8)
    a = arr[:, :, 3] > 0
    rgb[a] = arr[:, :, :3][a]
    return _to_pil(rgb)


def load_by_class_sample(
    root: Path,
    breed: str,
    stem: str,
) -> tuple[Image.Image, np.ndarray, Image.Image]:
    """Load one by_class sample for ``augment``.

    Expects::

        <root>/image/<breed>/<stem>.jpg
        <root>/fg/<breed>/<stem>.png
        <root>/bg/<breed>/<stem>.png   # optional; falls back to punched RGB

    Returns ``(image_rgb, fg_mask, bg_rgb)``. ``bbox/`` is not required.
    """
    img_path = root / "image" / breed / f"{stem}.jpg"
    fg_path = root / "fg" / breed / f"{stem}.png"
    bg_path = root / "bg" / breed / f"{stem}.png"
    if not img_path.is_file():
        raise FileNotFoundError(img_path)
    if not fg_path.is_file():
        raise FileNotFoundError(fg_path)
    img = Image.open(img_path).convert("RGB")
    mask = mask_from_fg_rgba(Image.open(fg_path))
    if bg_path.is_file():
        bg = bg_rgb_from_rgba(Image.open(bg_path))
    else:
        bg = _to_pil(_punch_fg(_to_np(img), mask))
    return img, mask, bg


def list_by_class_samples(root: Path) -> list[tuple[str, str]]:
    """Return ``[(breed, stem), ...]`` from ``<root>/image/<breed>/*.jpg``."""
    image_root = root / "image"
    if not image_root.is_dir():
        raise FileNotFoundError(f"missing image root: {image_root}")
    out: list[tuple[str, str]] = []
    for breed_dir in sorted(p for p in image_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for p in sorted(breed_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith("._"):
                out.append((breed_dir.name, p.stem))
    return out


def _synthetic_sample(h: int = 192, w: int = 192) -> tuple[Image.Image, np.ndarray, Image.Image]:
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - h * 0.52) ** 2) / (38 ** 2) + ((xx - w * 0.48) ** 2) / (52 ** 2) <= 1.0
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = (70, 130, 80)
    rgb[mask] = (190, 60, 40)
    rgb[mask & (yy % 8 < 3)] = (230, 200, 40)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = (40, 40, 120)
    bg[::12, :] = (200, 200, 200)
    return _to_pil(rgb), mask, _to_pil(bg)


if __name__ == "__main__":
    # Source by_class：platform_config.py；Missing → synthetic demo.
    from platform_config import BY_CLASS_ROOT

    DATA_ROOT = BY_CLASS_ROOT
    SAMPLE_INDEX = 0  # index into list_by_class_samples(DATA_ROOT)

    if DATA_ROOT.is_dir() and (DATA_ROOT / "image").is_dir():
        samples = list_by_class_samples(DATA_ROOT)
        if not samples:
            raise SystemExit(f"no samples under {DATA_ROOT / 'image'}")
        breed, stem = samples[min(SAMPLE_INDEX, len(samples) - 1)]
        img, mask, bg = load_by_class_sample(DATA_ROOT, breed, stem)
        print(f"source={DATA_ROOT} sample={breed}/{stem} size={img.size} fg_px={int(mask.sum())}")
    else:
        img, mask, bg = _synthetic_sample()
        print(f"DATA_ROOT missing ({DATA_ROOT}); using synthetic sample")

    orig = _to_np(img)
    photo_fg = FgSpec(rotate_deg=0.0)
    photo_g = GlobalSpec(rotate_deg=0.0, scale=0.0, shear_deg=0.0)
    for mode in ("none", "global", "fg_only", "bg_only", "fgbg"):
        rng = np.random.default_rng(0)
        vis = augment(img, mask, rng, mode, bg_image=bg)
        rng = np.random.default_rng(0)
        photo = augment(img, mask, rng, mode, bg_image=bg, fg_spec=photo_fg, global_spec=photo_g)
        print(
            mode,
            "photo-L2",
            round(foreground_l2(orig, _to_np(photo), mask), 2),
            "vis_size",
            vis.size,
        )
