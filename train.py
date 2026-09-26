#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Oxford-IIIT Pet classification with dual-intensity / YOLO-auto aug.

Reads ``by_class`` + ``splits/{train,val,test}.txt``.

All YOLO-backed arms train via Ultralytics ``YOLO.train`` (same log style):

- Ablation (``none`` / ``global`` / ``fg_only`` / ``bg_only`` / ``fgbg``):
  write train ImageFolder with ``fgbg_aug`` applied; YOLO **train-aug OFF**
  (custom ClassificationTrainer so train also uses ``augment=False``).
- Production (``yolo_auto``): clean train ImageFolder; YOLO **train-aug ON**
  (default Ultralytics classify aug).

**Val / test** always scored with our clean loaders (no fgbg, no YOLO aug).
Best checkpoint follows Ultralytics val inside ``YOLO.train``; we re-measure
val/test top-1 after training.

``resnet18`` still uses a small custom loop (no Ultralytics trainer).

For A0–A4 + yolo_auto sweep, run ``compare.py``.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF
from tqdm import tqdm

from fgbg_aug import AugMode, augment, list_by_class_samples, load_by_class_sample
from platform_config import BY_CLASS_ROOT, MAX_CLASSES, MAX_PER_CLASS, SPLIT_DIR

HERE = Path(__file__).resolve().parent
YOLO_AUTO_MODE = "yolo_auto"
ABLATION_MODES: tuple[str, ...] = ("none", "global", "fg_only", "bg_only", "fgbg")

# Offline dump/split_aug folder names under AUG_SPLIT_ROOT
MODE_AUG_FOLDER: dict[str, str] = {
    "none": "A0_none",
    "global": "A1_global",
    "fg_only": "A2_fg_only",
    "bg_only": "A3_bg_only",
    "fgbg": "A4_fgbg",
    YOLO_AUTO_MODE: "A0_none",  # clean originals + YOLO train-aug ON
}


def mode_aug_folder(mode: str) -> str:
    if mode not in MODE_AUG_FOLDER:
        raise KeyError(f"unknown mode for aug split folder: {mode}")
    return MODE_AUG_FOLDER[mode]


def read_split_stems(split_file: Path) -> set[str]:
    stems: set[str] = set()
    if not split_file.is_file():
        raise FileNotFoundError(split_file)
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stems.add(line.split()[0])
    return stems


def resolve_weight_name(weights: str) -> str:
    w = (weights or "resnet18").strip()
    if not w or w.lower() in {"resnet18", "resnet"}:
        return "resnet18"
    low = w.lower()
    if low.endswith(".pt") and "yolo" in low and "-cls" not in low:
        return w[:-3] + "-cls.pt"
    return w


class _YoloClsLogits(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if isinstance(out, tuple):
            return out[1]
        return out


def build_model(weights: str, n_class: int) -> tuple[nn.Module, str]:
    resolved = resolve_weight_name(weights)
    if resolved == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_class)
        return model, resolved

    from ultralytics import YOLO

    yolo = YOLO(resolved)
    net = yolo.model
    head = net.model[-1]
    if not hasattr(head, "linear"):
        raise RuntimeError(f"unexpected YOLO head (no linear): {type(head)} for {resolved}")
    in_f = int(head.linear.in_features)
    head.linear = nn.Linear(in_f, n_class)
    return _YoloClsLogits(net), resolved


def weight_tag(weights: str) -> str:
    resolved = resolve_weight_name(weights)
    if resolved == "resnet18":
        return "resnet18"
    name = Path(resolved).stem
    if name.endswith("-cls"):
        name = name[: -len("-cls")]
    return name


class PetByClassDualAug(Dataset):
    """by_class samples; train-time fgbg only when ``train=True``."""

    def __init__(
        self,
        by_class_root: Path,
        stems: set[str] | None,
        mode: AugMode,
        img_size: int,
        seed: int,
        train: bool,
        breed_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(by_class_root)
        self.mode = mode
        self.img_size = int(img_size)
        self.train = bool(train)
        self.seed = int(seed)

        samples = list_by_class_samples(self.root)
        if stems is not None:
            samples = [(b, s) for b, s in samples if s in stems]
        if not samples:
            raise RuntimeError(f"no samples under {self.root} for given split")

        if breed_to_idx is None:
            breeds = sorted({b for b, _ in samples})
            breed_to_idx = {b: i for i, b in enumerate(breeds)}
        self.class_to_idx = dict(breed_to_idx)
        self.samples = [(b, s, self.class_to_idx[b]) for b, s in samples if b in self.class_to_idx]

    def __len__(self) -> int:
        return len(self.samples)

    def _bg_image(self, idx: int) -> Image.Image:
        j = (idx + 17 * (idx % 9 + 1)) % len(self.samples)
        if j == idx:
            j = (idx + 1) % len(self.samples)
        breed, stem, _ = self.samples[j]
        img, _, _ = load_by_class_sample(self.root, breed, stem)
        return img

    def __getitem__(self, idx: int):
        breed, stem, label = self.samples[idx]
        img, mask, _ = load_by_class_sample(self.root, breed, stem)
        if self.train and self.mode != "none":
            rng = np.random.default_rng(self.seed + idx * 10007)
            img = augment(img, mask, rng, self.mode, bg_image=self._bg_image(idx))
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x, int(label)


def subset_by_class(ds: PetByClassDualAug, max_classes: int, max_per_class: int, seed: int) -> Subset:
    if max_classes <= 0 and max_per_class <= 0:
        return Subset(ds, list(range(len(ds))))
    rng = random.Random(seed)
    buckets: dict[int, list[int]] = {}
    for i, (_, _, label) in enumerate(ds.samples):
        buckets.setdefault(int(label), []).append(i)
    classes = sorted(buckets)
    if max_classes > 0:
        classes = classes[: int(max_classes)]
    keep: list[int] = []
    for c in classes:
        ids = buckets[c]
        rng.shuffle(ids)
        n = len(ids) if max_per_class <= 0 else min(len(ids), int(max_per_class))
        keep.extend(ids[:n])
    keep.sort()
    return Subset(ds, keep)


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _breed_to_idx(by_class_root: Path, max_classes: int) -> dict[str, int]:
    all_breeds = sorted({b for b, _ in list_by_class_samples(by_class_root)})
    if max_classes > 0:
        all_breeds = all_breeds[: int(max_classes)]
    return {b: i for i, b in enumerate(all_breeds)}


def _cap_stems(
    by_class_root: Path,
    stems: set[str],
    breed_to_idx: dict[str, int],
    max_per_class: int,
    seed: int,
) -> set[str]:
    if max_per_class <= 0:
        return stems
    rng = random.Random(seed)
    by_breed: dict[str, list[str]] = {}
    for breed, stem in list_by_class_samples(by_class_root):
        if stem in stems and breed in breed_to_idx:
            by_breed.setdefault(breed, []).append(stem)
    capped: set[str] = set()
    for stems_b in by_breed.values():
        rng.shuffle(stems_b)
        capped.update(stems_b[: int(max_per_class)])
    return capped


def _eval_loaders(
    *,
    by_class_root: Path,
    split_dir: Path,
    breed_to_idx: dict[str, int],
    img_size: int,
    seed: int,
    batch: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    val_stems = read_split_stems(split_dir / "val.txt")
    test_stems = read_split_stems(split_dir / "test.txt")
    val_full = PetByClassDualAug(
        by_class_root, val_stems, "none", img_size, seed, train=False, breed_to_idx=breed_to_idx
    )
    test_full = PetByClassDualAug(
        by_class_root, test_stems, "none", img_size, seed, train=False, breed_to_idx=breed_to_idx
    )
    val_loader = DataLoader(val_full, batch_size=batch, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_full, batch_size=batch, shuffle=False, num_workers=num_workers)
    return val_loader, test_loader, len(val_full), len(test_full)


def _materialize_imagefolder(
    by_class_root: Path,
    stems: set[str],
    breed_to_idx: dict[str, int],
    out_dir: Path,
) -> int:
    """Symlink (fallback copy) clean by_class/image → ImageFolder."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for breed, stem in list_by_class_samples(by_class_root):
        if stem not in stems or breed not in breed_to_idx:
            continue
        src = by_class_root / "image" / breed / f"{stem}.jpg"
        if not src.is_file():
            continue
        dst_dir = out_dir / breed
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{stem}.jpg"
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
        n += 1
    return n


def _materialize_fgbg_train(
    by_class_root: Path,
    stems: set[str],
    breed_to_idx: dict[str, int],
    out_dir: Path,
    *,
    mode: AugMode,
    img_size: int,
    seed: int,
) -> int:
    """Write train ImageFolder with fgbg_aug already applied (YOLO train-aug will be off)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(b, s) for b, s in list_by_class_samples(by_class_root) if s in stems and b in breed_to_idx]
    if not pairs:
        return 0

    n = 0
    for i, (breed, stem) in enumerate(pairs):
        img, mask, _ = load_by_class_sample(by_class_root, breed, stem)
        if mode != "none":
            j = (i + 17 * (i % 9 + 1)) % len(pairs)
            if j == i:
                j = (i + 1) % len(pairs)
            bg_breed, bg_stem = pairs[j]
            bg_img, _, _ = load_by_class_sample(by_class_root, bg_breed, bg_stem)
            rng = np.random.default_rng(seed + i * 10007)
            img = augment(img, mask, rng, mode, bg_image=bg_img)
        img = img.resize((img_size, img_size), Image.BILINEAR)
        dst_dir = out_dir / breed
        dst_dir.mkdir(parents=True, exist_ok=True)
        img.save(dst_dir / f"{stem}.jpg", quality=95)
        n += 1
    return n


def _cls_trainer_cls(*, yolo_train_aug: bool):
    """Ultralytics ClassificationTrainer; optionally force train ``augment=False``."""
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    class _Trainer(ClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            # Default Ultralytics: augment = (mode == "train").
            # Ablation: images already fgbg-augmented → never apply YOLO aug.
            aug = (mode == "train") and bool(yolo_train_aug)
            return ClassificationDataset(root=img_path, args=self.args, augment=aug, prefix=mode)

    return _Trainer


def _run_yolo_classify_train(
    *,
    resolved_weights: str,
    data_root: Path,
    run_dir: Path,
    img_size: int,
    batch: int,
    epochs: int,
    seed: int,
    num_workers: int,
    yolo_train_aug: bool,
    patience: int = 25,
    optimizer: str = "AdamW",
    lr0: float = 1e-3,
    lrf: float = 0.01,
    cos_lr: bool = True,
    weight_decay: float = 5e-4,
) -> tuple[Path, float | None]:
    """Train; return ``(best_pt, val_top1)``.

    ``val_top1`` comes from Ultralytics' own final ``Validating best.pt`` (do not
    re-run val afterwards — that duplicates the banner the user saw twice).
    """
    from ultralytics import YOLO

    yolo = YOLO(resolved_weights)
    metrics = yolo.train(
        data=str(data_root),
        epochs=int(epochs),
        patience=int(patience),
        imgsz=int(img_size),
        batch=int(batch),
        seed=int(seed),
        workers=int(num_workers),
        project=str(run_dir),
        name="ultralytics",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        optimizer=str(optimizer),
        lr0=float(lr0),
        lrf=float(lrf),
        cos_lr=bool(cos_lr),
        weight_decay=float(weight_decay),
        trainer=_cls_trainer_cls(yolo_train_aug=yolo_train_aug),
    )
    best_path = run_dir / "ultralytics" / "weights" / "best.pt"
    if not best_path.is_file():
        best_path = run_dir / "ultralytics" / "weights" / "last.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"YOLO train finished but no weights under {run_dir}/ultralytics")

    val_top1: float | None = None
    if metrics is not None:
        raw = getattr(metrics, "top1", None)
        if raw is None and hasattr(metrics, "results_dict"):
            raw = metrics.results_dict.get("metrics/accuracy_top1")
        if raw is not None:
            val_top1 = float(raw)
    return best_path, val_top1


def _count_imagefolder(root: Path) -> int:
    if not root.is_dir():
        return 0
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and not p.name.startswith("._"):
            n += 1
    return n


def prebuilt_aug_split_ready(
    aug_split_root: Path,
    modes: tuple[str, ...] | list[str],
    *,
    test_image_root: Path,
) -> bool:
    """True when every mode has train/val under ``aug_split_root`` and test ImageFolder exists."""
    if not Path(test_image_root).is_dir():
        return False
    if not any(p.is_dir() for p in Path(test_image_root).iterdir() if not p.name.startswith(".")):
        return False
    root = Path(aug_split_root)
    for mode in modes:
        folder = mode_aug_folder(mode)
        train_d = root / folder / "train"
        val_d = root / folder / "val"
        if not train_d.is_dir() or not val_d.is_dir():
            return False
        if _count_imagefolder(train_d) <= 0 or _count_imagefolder(val_d) <= 0:
            return False
    return True


def _symlink_dir(src: Path, dst: Path) -> None:
    """Point ``dst`` at ``src`` (symlink; fallback copytree)."""
    src = Path(src).resolve()
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def _prepare_prebuilt_cls_data(
    data_root: Path,
    *,
    train_src: Path,
    val_src: Path,
    test_src: Path,
) -> tuple[int, int, int]:
    """Wire YOLO classify layout from existing ImageFolders (no re-aug)."""
    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    _symlink_dir(train_src, data_root / "train")
    _symlink_dir(val_src, data_root / "val")
    _symlink_dir(test_src, data_root / "test")
    return (
        _count_imagefolder(data_root / "train"),
        _count_imagefolder(data_root / "val"),
        _count_imagefolder(data_root / "test"),
    )


def _breeds_from_imagefolder(root: Path) -> dict[str, int]:
    breeds = sorted(
        p.name for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not breeds:
        raise RuntimeError(f"no class folders under {root}")
    return {b: i for i, b in enumerate(breeds)}


def _top1_from_metrics(metrics) -> float:
    if metrics is None:
        return 0.0
    v = getattr(metrics, "top1", None)
    if v is not None:
        return float(v)
    rd = getattr(metrics, "results_dict", None) or {}
    return float(rd.get("metrics/accuracy_top1", 0.0) or 0.0)


def _eval_yolo_best(
    *,
    best_path: Path,
    data_root: Path,
    img_size: int,
    batch: int,
    num_workers: int,
    val_top1: float | None = None,
) -> tuple[float, float, int, int]:
    """Score ``best.pt`` on test (and val only if ``val_top1`` not already known).

    After ``YOLO.train``, Ultralytics already prints one ``Validating best.pt`` on
    **val**. Passing that ``val_top1`` here skips a duplicate val pass; we only
    run ``split=test``.
    """
    from ultralytics import YOLO

    val_dir = data_root / "val"
    test_dir = data_root / "test"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"missing val ImageFolder: {val_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"missing test ImageFolder: {test_dir}")

    yolo = YOLO(str(best_path))
    if val_top1 is None:
        m_val = yolo.val(
            data=str(data_root),
            split="val",
            imgsz=int(img_size),
            batch=int(batch),
            workers=int(num_workers),
            plots=False,
            verbose=False,
        )
        val_top1 = _top1_from_metrics(m_val)

    m_test = yolo.val(
        data=str(data_root),
        split="test",
        imgsz=int(img_size),
        batch=int(batch),
        workers=int(num_workers),
        plots=False,
        verbose=False,
    )
    test_top1 = _top1_from_metrics(m_test)
    return float(val_top1), test_top1, _count_imagefolder(val_dir), _count_imagefolder(test_dir)


def train_one(
    *,
    by_class_root: Path,
    split_dir: Path,
    run_dir: Path,
    mode: str,
    img_size: int,
    batch: int,
    epochs: int,
    lr: float,
    seed: int,
    max_classes: int,
    max_per_class: int,
    num_workers: int,
    weights: str = "resnet18",
    patience: int = 25,
    optimizer: str = "AdamW",
    lrf: float = 0.01,
    cos_lr: bool = True,
    weight_decay: float = 5e-4,
    aug_split_root: Path | None = None,
    test_image_root: Path | None = None,
) -> dict:
    resolved = resolve_weight_name(weights)
    opt_kw = dict(
        patience=patience,
        optimizer=optimizer,
        lr0=lr,
        lrf=lrf,
        cos_lr=cos_lr,
        weight_decay=weight_decay,
    )
    # Prefer offline split_aug ImageFolders when present (no online / re-dump aug).
    use_prebuilt = False
    if aug_split_root is not None:
        test_root = Path(test_image_root) if test_image_root is not None else Path(split_dir) / "test" / "image"
        folder = mode_aug_folder(mode)
        train_src = Path(aug_split_root) / folder / "train"
        val_src = Path(aug_split_root) / folder / "val"
        use_prebuilt = (
            train_src.is_dir()
            and val_src.is_dir()
            and test_root.is_dir()
            and _count_imagefolder(train_src) > 0
            and _count_imagefolder(val_src) > 0
            and _count_imagefolder(test_root) > 0
        )
        if use_prebuilt and resolved == "resnet18":
            raise ValueError("prebuilt aug_split currently requires YOLO-cls weights (not resnet18)")
        if use_prebuilt:
            return train_one_yolo_ultralytics(
                by_class_root=by_class_root,
                split_dir=split_dir,
                run_dir=run_dir,
                mode=mode,
                img_size=img_size,
                batch=batch,
                epochs=epochs,
                seed=seed,
                max_classes=max_classes,
                max_per_class=max_per_class,
                num_workers=num_workers,
                weights=weights,
                yolo_train_aug=(mode == YOLO_AUTO_MODE),
                fgbg_mode=None if mode == YOLO_AUTO_MODE else mode,  # type: ignore[arg-type]
                aug_split_root=Path(aug_split_root),
                test_image_root=test_root,
                **opt_kw,
            )

    if mode == YOLO_AUTO_MODE:
        if resolved == "resnet18":
            raise ValueError("yolo_auto requires a YOLO-cls weight (e.g. yolo11s.pt)")
        return train_one_yolo_ultralytics(
            by_class_root=by_class_root,
            split_dir=split_dir,
            run_dir=run_dir,
            mode=YOLO_AUTO_MODE,
            img_size=img_size,
            batch=batch,
            epochs=epochs,
            seed=seed,
            max_classes=max_classes,
            max_per_class=max_per_class,
            num_workers=num_workers,
            weights=weights,
            yolo_train_aug=True,
            fgbg_mode=None,
            **opt_kw,
        )
    if resolved == "resnet18":
        return train_one_resnet_loop(
            by_class_root=by_class_root,
            split_dir=split_dir,
            run_dir=run_dir,
            mode=mode,  # type: ignore[arg-type]
            img_size=img_size,
            batch=batch,
            epochs=epochs,
            lr=lr,
            seed=seed,
            max_classes=max_classes,
            max_per_class=max_per_class,
            num_workers=num_workers,
            weights=weights,
            patience=patience,
            weight_decay=weight_decay,
        )
    return train_one_yolo_ultralytics(
        by_class_root=by_class_root,
        split_dir=split_dir,
        run_dir=run_dir,
        mode=mode,
        img_size=img_size,
        batch=batch,
        epochs=epochs,
        seed=seed,
        max_classes=max_classes,
        max_per_class=max_per_class,
        num_workers=num_workers,
        weights=weights,
        yolo_train_aug=False,
        fgbg_mode=mode,  # type: ignore[arg-type]
        **opt_kw,
    )


def train_one_yolo_ultralytics(
    *,
    by_class_root: Path,
    split_dir: Path,
    run_dir: Path,
    mode: str,
    img_size: int,
    batch: int,
    epochs: int,
    seed: int,
    max_classes: int,
    max_per_class: int,
    num_workers: int,
    weights: str,
    yolo_train_aug: bool,
    fgbg_mode: AugMode | None,
    patience: int = 25,
    optimizer: str = "AdamW",
    lr0: float = 1e-3,
    lrf: float = 0.01,
    cos_lr: bool = True,
    weight_decay: float = 5e-4,
    aug_split_root: Path | None = None,
    test_image_root: Path | None = None,
) -> dict:
    """Shared Ultralytics classify train path (ablation + yolo_auto).

    When ``aug_split_root`` is set, train/val come from
    ``aug_split_root/<A*_>/{{train,val}}`` and test from ``test_image_root``
    (default ``split_dir/test/image``) — **no** online fgbg materialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    resolved = resolve_weight_name(weights)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = run_dir / "yolo_cls_data"

    if aug_split_root is not None:
        folder = mode_aug_folder(mode)
        train_src = Path(aug_split_root) / folder / "train"
        val_src = Path(aug_split_root) / folder / "val"
        test_src = Path(test_image_root) if test_image_root is not None else Path(split_dir) / "test" / "image"
        if not train_src.is_dir() or not val_src.is_dir():
            raise FileNotFoundError(f"missing prebuilt train/val: {train_src} / {val_src}")
        if not test_src.is_dir():
            raise FileNotFoundError(f"missing test ImageFolder: {test_src}")
        n_train, n_val_link, n_test_link = _prepare_prebuilt_cls_data(
            data_root,
            train_src=train_src,
            val_src=val_src,
            test_src=test_src,
        )
        breed_to_idx = _breeds_from_imagefolder(train_src)
        if yolo_train_aug:
            train_aug_desc = f"prebuilt:{folder}; ultralytics_classify_auto"
        else:
            train_aug_desc = f"prebuilt:{folder}; yolo_train_aug=off"
        print(
            f"[{mode}] prebuilt data | train={train_src} | val={val_src} | test={test_src}"
        )
    else:
        train_stems = read_split_stems(split_dir / "train.txt")
        val_stems = read_split_stems(split_dir / "val.txt")
        breed_to_idx = _breed_to_idx(by_class_root, max_classes)
        train_stems = _cap_stems(by_class_root, train_stems, breed_to_idx, max_per_class, seed)
        test_stems = read_split_stems(split_dir / "test.txt")
        if fgbg_mode is None:
            n_train = _materialize_imagefolder(by_class_root, train_stems, breed_to_idx, data_root / "train")
            train_aug_desc = "ultralytics_classify_auto"
        else:
            n_train = _materialize_fgbg_train(
                by_class_root,
                train_stems,
                breed_to_idx,
                data_root / "train",
                mode=fgbg_mode,
                img_size=img_size,
                seed=seed,
            )
            train_aug_desc = f"fgbg:{fgbg_mode}; yolo_train_aug=off"
        # Val/test: always clean RGB (never fgbg / never YOLO train-aug).
        n_val_link = _materialize_imagefolder(by_class_root, val_stems, breed_to_idx, data_root / "val")
        n_test_link = _materialize_imagefolder(by_class_root, test_stems, breed_to_idx, data_root / "test")

    if n_train == 0:
        raise RuntimeError(f"{mode}: empty train ImageFolder")
    if n_val_link == 0:
        raise RuntimeError(f"{mode}: empty val ImageFolder")
    if n_test_link == 0:
        raise RuntimeError(f"{mode}: empty test ImageFolder")

    print(
        f"[{mode}] Ultralytics classify | yolo_train_aug={yolo_train_aug} | "
        f"opt={optimizer} lr0={lr0} patience={patience} epochs={epochs} | "
        f"train_n={n_train} val_n={n_val_link} test_n={n_test_link} | {train_aug_desc}"
    )
    best_path, train_val_top1 = _run_yolo_classify_train(
        resolved_weights=resolved,
        data_root=data_root,
        run_dir=run_dir,
        img_size=img_size,
        batch=batch,
        epochs=epochs,
        seed=seed,
        num_workers=num_workers,
        yolo_train_aug=yolo_train_aug,
        patience=patience,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        weight_decay=weight_decay,
    )

    # Reuse train's final val top1; only run test once (avoid duplicate "Validating best.pt" on val).
    val_acc, test_acc, n_val, n_test = _eval_yolo_best(
        best_path=best_path,
        data_root=data_root,
        img_size=img_size,
        batch=batch,
        num_workers=num_workers,
        val_top1=train_val_top1,
    )
    print(
        f"mode={mode} best_val_top1={val_acc:.4f} test_top1={test_acc:.4f} "
        f"(val from train final_eval; test via ultralytics.val)"
    )

    from ultralytics import YOLO

    trained = YOLO(str(best_path))
    torch.save(_YoloClsLogits(trained.model).state_dict(), run_dir / "best.pt")

    result = {
        "mode": mode,
        "weights": weights,
        "resolved_weights": resolved,
        "best_val_top1": val_acc,
        "test_top1_at_best_val": test_acc,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_class": len(breed_to_idx),
        "seed": seed,
        "by_class_root": str(by_class_root),
        "aug_at_eval": False,
        "eval_backend": "ultralytics.val",
        "yolo_train_aug": bool(yolo_train_aug),
        "train_aug": train_aug_desc,
        "optimizer": optimizer,
        "lr0": lr0,
        "patience": patience,
        "prebuilt_aug_split": aug_split_root is not None,
        "aug_split_root": str(aug_split_root) if aug_split_root is not None else None,
        "epochs": epochs,
        "ultralytics_best": str(best_path),
        "history": [],
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_one_resnet_loop(
    *,
    by_class_root: Path,
    split_dir: Path,
    run_dir: Path,
    mode: AugMode,
    img_size: int,
    batch: int,
    epochs: int,
    lr: float,
    seed: int,
    max_classes: int,
    max_per_class: int,
    num_workers: int,
    weights: str = "resnet18",
    patience: int = 25,
    weight_decay: float = 5e-4,
) -> dict:
    """Fallback custom loop when WEIGHTS=resnet18 (no Ultralytics logs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _pick_device()

    train_stems = read_split_stems(split_dir / "train.txt")
    breed_to_idx = _breed_to_idx(by_class_root, max_classes)
    train_full = PetByClassDualAug(
        by_class_root, train_stems, mode, img_size, seed, train=True, breed_to_idx=breed_to_idx
    )
    train_ds = subset_by_class(train_full, 0, max_per_class, seed)
    val_loader, test_loader, n_val, n_test = _eval_loaders(
        by_class_root=by_class_root,
        split_dir=split_dir,
        breed_to_idx=breed_to_idx,
        img_size=img_size,
        seed=seed,
        batch=batch,
        num_workers=num_workers,
    )

    n_class = len(breed_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers)
    model, resolved_weights = build_model(weights, n_class)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val = -1.0
    best_test_at_val = 0.0
    bad_epochs = 0
    run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"{mode} ep{epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(y.numel())
            n += int(y.numel())
        sched.step()
        val_acc = accuracy(model, val_loader, device)
        test_acc = accuracy(model, test_loader, device)
        row = {"epoch": epoch, "loss": running / max(n, 1), "val_top1": val_acc, "test_top1": test_acc}
        history.append(row)
        print(
            f"mode={mode} epoch={epoch} loss={row['loss']:.4f} "
            f"val_top1={val_acc:.4f} test_top1={test_acc:.4f}"
        )
        if val_acc >= best_val:
            best_val = val_acc
            best_test_at_val = test_acc
            bad_epochs = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= int(patience):
                print(f"early stop at epoch={epoch} (patience={patience}, best_val={best_val:.4f})")
                break

    result = {
        "mode": mode,
        "weights": weights,
        "resolved_weights": resolved_weights,
        "best_val_top1": best_val,
        "test_top1_at_best_val": best_test_at_val,
        "n_train": len(train_ds),
        "n_val": n_val,
        "n_test": n_test,
        "n_class": n_class,
        "seed": seed,
        "by_class_root": str(by_class_root),
        "aug_at_eval": False,
        "train_aug": f"fgbg:{mode}; trainer=resnet_loop",
        "optimizer": "AdamW",
        "lr0": lr,
        "patience": patience,
        "epochs": epochs,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    # 数据路径 / 冒烟：platform_config.py
    RUN_ROOT = HERE / "runs"

    WEIGHTS = "yolo11s.pt"
    AUG_MODE: str = "fgbg"
    IMG_SIZE = 224
    BATCH = 32
    # Pet 全量：100 + patience 早停；冒烟可临时改小
    EPOCHS = 100
    PATIENCE = 25
    OPTIMIZER = "AdamW"
    LR = 1e-3
    LRF = 0.01
    COS_LR = True
    WEIGHT_DECAY = 5e-4
    SEED = 0
    NUM_WORKERS = 2

    tag = weight_tag(WEIGHTS)
    out = train_one(
        by_class_root=BY_CLASS_ROOT,
        split_dir=SPLIT_DIR,
        run_dir=RUN_ROOT / f"{AUG_MODE}_{tag}_s{SEED}",
        mode=AUG_MODE,
        img_size=IMG_SIZE,
        batch=BATCH,
        epochs=EPOCHS,
        lr=LR,
        seed=SEED,
        max_classes=MAX_CLASSES,
        max_per_class=MAX_PER_CLASS,
        num_workers=NUM_WORKERS,
        weights=WEIGHTS,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        lrf=LRF,
        cos_lr=COS_LR,
        weight_decay=WEIGHT_DECAY,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=2))
