#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-shot ablation + production YOLO-auto comparison (pipeline step ④).

**Preferred data path** (no online / re-dump aug when ready)::

    train/val ← AUG_SPLIT_ROOT/<A0_none|A1_…|A4_…>/{train,val}/
    test      ← SPLIT_DIR/test/image/

If those dirs exist, ``compare`` only trains + evaluates. Otherwise falls back
to online materialize from ``by_class`` + ``splits/*.txt`` (legacy).

Upstream rebuild switches (``__main__``)::

    RERUN_EXTRACT_BY_CLASS  # ① delete BY_CLASS_ROOT → extract_{pet|cub}_by_class
    RERUN_SPLIT_BY_CLASS    # ② delete SPLIT_DIR → split_by_class
    RERUN_DUMP_AUG          # ③ delete AUG_OUT_ROOT → dump_aug
    RERUN_SPLIT_AUG         # ③b delete AUG_SPLIT_ROOT → split_aug

Enabling an earlier step auto-cascades later ones (①⇒②⇒③⇒③b, ②⇒③⇒③b, ③⇒③b).

Dataset / Mac smoke vs Linux full: ``platform_config.DATASET`` + ``MAX_*``.

Runs:
  - A0–A4 (``none`` / ``global`` / ``fg_only`` / ``bg_only`` / ``fgbg``)
  - production ``yolo_auto`` (Ultralytics classify train aug ON; clean A0 train)

Every arm reports **val** and **test** top-1 with **augmentation off** at eval.

Edit variables under ``if __name__ == "__main__"`` then run this file.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from dump_aug import MODE_DIRS, run_dump
from extract_cub_by_class import extract as extract_cub
from extract_pet_by_class import extract as extract_pet
from fgbg_aug import AugMode
from platform_config import (
    AUG_COUNT,
    AUG_OUT_ROOT,
    AUG_RANDOM_SPLIT,
    AUG_SPLIT_ROOT,
    AUG_VAL_RATIO,
    AUG_VAL_USE_AUG,
    BY_CLASS_ROOT,
    CUB_ROOT,
    DATASET,
    INCLUDE_A0,
    MAX_CLASSES,
    MAX_PER_CLASS,
    PET_ROOT,
    SPLIT_DIR,
)
from split_aug import split_aug_train_val
from split_by_class import ensure_split
from train import (
    YOLO_AUTO_MODE,
    mode_aug_folder,
    prebuilt_aug_split_ready,
    train_one,
    weight_tag,
)

HERE = Path(__file__).resolve().parent
ABLATION_MODES: tuple[AugMode, ...] = ("none", "global", "fg_only", "bg_only", "fgbg")
ALL_MODES: tuple[str, ...] = (*ABLATION_MODES, YOLO_AUTO_MODE)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, var**0.5


def _rmtree(path: Path) -> None:
    """Delete a tree; tolerate ENOENT races (AppleDouble ``._*`` on external volumes)."""
    path = Path(path)
    if not (path.exists() or path.is_symlink()):
        return
    print(f"  remove → {path}")
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    def _onerror(func, p, exc_info):
        err = exc_info[1]
        if isinstance(err, FileNotFoundError):
            return
        raise err

    shutil.rmtree(path, onerror=_onerror)
    # External FS may leave a phantom root after ENOENT races; retry once if needed.
    if path.exists() or path.is_symlink():
        shutil.rmtree(path, onerror=_onerror)


def prepare_upstream(
    *,
    rerun_extract_by_class: bool,
    rerun_split_by_class: bool,
    rerun_dump_aug: bool,
    rerun_split_aug: bool,
    by_class_root: Path = BY_CLASS_ROOT,
    split_dir: Path = SPLIT_DIR,
    aug_out_root: Path = AUG_OUT_ROOT,
    aug_split_root: Path = AUG_SPLIT_ROOT,
    dataset: str = DATASET,
    test_ratio: float = 0.1,
    val_ratio: float = 0.0,
    split_seed: int = 0,
    img_size: int = 224,
    other_image_prob: float = 0.4,
    dump_seed: int = 0,
    max_classes: int = MAX_CLASSES,
    max_per_class: int = MAX_PER_CLASS,
) -> dict:
    """Optionally rebuild ① / ② / ③ / ③b. Earlier steps cascade to later ones."""
    do_1 = bool(rerun_extract_by_class)
    do_2 = bool(rerun_split_by_class) or do_1
    do_3 = bool(rerun_dump_aug) or do_2
    do_3b = bool(rerun_split_aug) or do_3

    log = {
        "dataset": dataset,
        "max_classes": int(max_classes),
        "max_per_class": int(max_per_class),
        "requested": {
            "extract_by_class": bool(rerun_extract_by_class),
            "split_by_class": bool(rerun_split_by_class),
            "dump_aug": bool(rerun_dump_aug),
            "split_aug": bool(rerun_split_aug),
        },
        "effective": {
            "extract_by_class": do_1,
            "split_by_class": do_2,
            "dump_aug": do_3,
            "split_aug": do_3b,
        },
        "steps": {},
    }
    if not (do_1 or do_2 or do_3 or do_3b):
        print("upstream: no rebuild (all RERUN_* = False)")
        return log

    print(
        "upstream rebuild plan: "
        f"dataset={dataset} max_classes={max_classes} max_per_class={max_per_class} | "
        f"①extract={do_1} ②split_by_class={do_2} ③dump_aug={do_3} ③b_split_aug={do_3b}"
    )

    if do_1:
        print(f"\n=== ① extract_{dataset}_by_class: delete BY_CLASS_ROOT then rewrite ===")
        _rmtree(by_class_root)
        if dataset == "pet":
            extract_info = extract_pet(
                PET_ROOT,
                by_class_root,
                skip_existing=False,
                max_classes=max_classes,
                max_per_class=max_per_class,
            )
        elif dataset == "cub":
            extract_info = extract_cub(
                CUB_ROOT,
                by_class_root,
                skip_existing=False,
                max_classes=max_classes,
                max_per_class=max_per_class,
            )
        else:
            raise SystemExit(f"unknown dataset={dataset!r}; expected 'pet' or 'cub'")
        log["steps"]["extract_by_class"] = {
            "out": str(by_class_root),
            "n_saved": extract_info.get("n_saved"),
            "n_classes": extract_info.get("n_classes"),
            "n_images": extract_info.get("n_images"),
        }
        print(
            f"  saved {extract_info.get('n_saved')}/{extract_info.get('n_images')} "
            f"→ {extract_info.get('n_classes')} classes under {by_class_root}"
        )

    if do_2:
        print("\n=== ② split_by_class: delete SPLIT_DIR then rewrite ===")
        _rmtree(split_dir)
        split_info = ensure_split(
            by_class_root,
            split_dir,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=split_seed,
            force=True,
            max_classes=max_classes,
            max_per_class=max_per_class,
        )
        log["steps"]["split_by_class"] = {
            "out": str(split_dir),
            "n_train": split_info.get("n_train"),
            "n_val": split_info.get("n_val"),
            "n_test": split_info.get("n_test"),
            "n_trainval": split_info.get("n_trainval"),
        }
        print(
            f"  wrote trainval={split_info.get('n_trainval')} "
            f"test={split_info.get('n_test')} → {split_dir}"
        )

    if do_3:
        print("\n=== ③ dump_aug: delete AUG_OUT_ROOT then rewrite ===")
        trainval_root = Path(split_dir) / "trainval"
        if not (trainval_root / "image").is_dir():
            raise SystemExit(
                f"missing {trainval_root / 'image'}; run ② split_by_class first "
                f"(or set RERUN_SPLIT_BY_CLASS=True / RERUN_EXTRACT_BY_CLASS=True)"
            )
        _rmtree(aug_out_root)
        modes = list(MODE_DIRS)
        if INCLUDE_A0:
            modes = [("A0_none", "none"), *modes]
        run_dump(
            trainval_root=trainval_root,
            out_root=Path(aug_out_root),
            modes=modes,
            seed=dump_seed,
            img_size=img_size,
            skip_existing=False,
            aug_count=AUG_COUNT,
            other_image_prob=other_image_prob,
        )
        log["steps"]["dump_aug"] = {
            "out": str(aug_out_root),
            "modes": [m[0] for m in modes],
            "aug_count": AUG_COUNT,
            "include_a0": INCLUDE_A0,
        }

    if do_3b:
        print("\n=== ③b split_aug: delete AUG_SPLIT_ROOT then rewrite ===")
        if not Path(aug_out_root).is_dir():
            raise SystemExit(
                f"missing {aug_out_root}; run ③ dump_aug first "
                f"(or set RERUN_DUMP_AUG=True)"
            )
        _rmtree(aug_split_root)
        summary = split_aug_train_val(
            Path(aug_out_root),
            Path(aug_split_root),
            val_ratio=AUG_VAL_RATIO,
            seed=dump_seed,
            force=True,
            skip_existing=False,
            reference_mode=None,
            include_a0=INCLUDE_A0,
            random_split=AUG_RANDOM_SPLIT,
            val_use_aug=AUG_VAL_USE_AUG,
            trainval_root=Path(split_dir) / "trainval",
            aug_count=AUG_COUNT,
            img_size=img_size,
        )
        log["steps"]["split_aug"] = {
            "out": str(aug_split_root),
            "n_train_items": summary.get("n_train_items"),
            "n_val_items": summary.get("n_val_items"),
            "val_use_aug": summary.get("val_use_aug"),
            "modes": summary.get("modes"),
        }
        print(
            f"  wrote train={summary.get('n_train_items')} "
            f"val={summary.get('n_val_items')} → {aug_split_root}"
        )

    return log


def run_compare(
    *,
    by_class_root: Path,
    split_dir: Path,
    run_root: Path,
    modes: tuple[str, ...],
    seeds: tuple[int, ...],
    img_size: int,
    batch: int,
    epochs: int,
    lr: float,
    max_classes: int,
    max_per_class: int,
    num_workers: int,
    weights: str = "yolo11s.pt",
    aug_split_root: Path | None = None,
    test_image_root: Path | None = None,
    prefer_prebuilt: bool = True,
    patience: int = 25,
    optimizer: str = "AdamW",
    lrf: float = 0.01,
    cos_lr: bool = True,
    weight_decay: float = 5e-4,
    upstream: dict | None = None,
) -> dict:
    test_root = Path(test_image_root) if test_image_root is not None else Path(split_dir) / "test" / "image"
    aug_root = Path(aug_split_root) if aug_split_root is not None else None

    use_prebuilt = bool(
        prefer_prebuilt
        and aug_root is not None
        and prebuilt_aug_split_ready(aug_root, modes, test_image_root=test_root)
    )
    if use_prebuilt:
        print(
            f"prebuilt ready → train/val from {aug_root} | test from {test_root} "
            f"(skip online / re-dump aug)"
        )
        for m in modes:
            print(f"  mode={m} → {aug_root / mode_aug_folder(m)}")
    else:
        print(
            "prebuilt missing or incomplete → fallback online materialize "
            f"(need dump_aug + split_aug under {aug_root}; test={test_root})"
        )

    tag = weight_tag(weights)
    print(
        f"weights={weights} → tag={tag} | "
        f"opt={optimizer} lr0={lr} patience={patience} epochs={epochs}"
    )

    rows: list[dict] = []
    for seed in seeds:
        for mode in modes:
            run_dir = run_root / f"{mode}_{tag}_s{seed}"
            print(f"\n=== compare mode={mode} weights={weights} seed={seed} → {run_dir} ===")
            out = train_one(
                by_class_root=by_class_root,
                split_dir=split_dir,
                run_dir=run_dir,
                mode=mode,
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
                optimizer=optimizer,
                lrf=lrf,
                cos_lr=cos_lr,
                weight_decay=weight_decay,
                aug_split_root=aug_root if use_prebuilt else None,
                test_image_root=test_root if use_prebuilt else None,
            )
            rows.append(
                {
                    "mode": mode,
                    "weights": weights,
                    "resolved_weights": out.get("resolved_weights"),
                    "seed": seed,
                    "best_val_top1": out["best_val_top1"],
                    "test_top1_at_best_val": out["test_top1_at_best_val"],
                    "n_train": out["n_train"],
                    "n_val": out["n_val"],
                    "n_test": out["n_test"],
                    "n_class": out["n_class"],
                    "aug_at_eval": out.get("aug_at_eval", False),
                    "optimizer": out.get("optimizer", optimizer),
                    "lr0": out.get("lr0", lr),
                    "patience": out.get("patience", patience),
                    "prebuilt_aug_split": out.get("prebuilt_aug_split", False),
                    "train_aug": out.get("train_aug"),
                    "run_dir": str(run_dir),
                }
            )

    by_mode: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        m = str(r["mode"])
        by_mode.setdefault(m, {"val": [], "test": []})
        by_mode[m]["val"].append(float(r["best_val_top1"]))
        by_mode[m]["test"].append(float(r["test_top1_at_best_val"]))

    summary_modes = {}
    for mode, packs in by_mode.items():
        v_mean, v_std = _mean_std(packs["val"])
        t_mean, t_std = _mean_std(packs["test"])
        summary_modes[mode] = {
            "mean_best_val_top1": round(v_mean, 4),
            "std_best_val_top1": round(v_std, 4),
            "val_values": [round(v, 4) for v in packs["val"]],
            "mean_test_top1_at_best_val": round(t_mean, 4),
            "std_test_top1_at_best_val": round(t_std, 4),
            "test_values": [round(v, 4) for v in packs["test"]],
            "n_seeds": len(packs["val"]),
        }

    report = {
        "by_class_root": str(by_class_root),
        "split_dir": str(split_dir),
        "aug_split_root": str(aug_root) if aug_root is not None else None,
        "test_image_root": str(test_root),
        "data_source": "prebuilt_aug_split" if use_prebuilt else "online_materialize",
        "upstream": upstream,
        "weights": weights,
        "weight_tag": tag,
        "modes": list(modes),
        "seeds": list(seeds),
        "epochs": epochs,
        "patience": patience,
        "optimizer": optimizer,
        "lr0": lr,
        "lrf": lrf,
        "cos_lr": cos_lr,
        "weight_decay": weight_decay,
        "max_classes": max_classes,
        "max_per_class": max_per_class,
        "eval_aug": False,
        "runs": rows,
        "by_mode": summary_modes,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    out_path = run_root / "compare_summary.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(
        f"data_source={report['data_source']} | eval_aug=False | "
        f"optimizer={optimizer} lr0={lr} patience={patience} epochs={epochs}"
    )
    for mode in modes:
        s = summary_modes[mode]
        print(
            f"  {mode:10s}  val={s['mean_best_val_top1']:.4f}±{s['std_best_val_top1']:.4f}  "
            f"test={s['mean_test_top1_at_best_val']:.4f}±{s['std_test_top1_at_best_val']:.4f}"
        )
    return report


if __name__ == "__main__":
    # nohup /home/beyond/.conda/envs/yolo11/bin/python3 compare.py > cub.log 2>&1 &
    # 路径 / 数据集 / Mac冒烟·Linux全量：platform_config.py（DATASET, MAX_*）

    # ========================= 上游重跑开关（先删产物目录，再执行）=========================
    # ① extract_by_class → BY_CLASS_ROOT（pet / cub 由 DATASET 决定）
    # ② split_by_class   → SPLIT_DIR
    # ③ dump_aug         → AUG_OUT_ROOT
    # ③b split_aug       → AUG_SPLIT_ROOT
    # 打开更早步骤会自动级联更晚步骤（①⇒②⇒③⇒③b；②⇒③⇒③b；③⇒③b）
    RERUN_EXTRACT_BY_CLASS = True
    RERUN_SPLIT_BY_CLASS = True
    RERUN_DUMP_AUG = True
    RERUN_SPLIT_AUG = True

    # 与脚本 __main__ 对齐的上游参数
    UPSTREAM_TEST_RATIO = 0.1
    UPSTREAM_VAL_RATIO = 0.0  # ② 仍只 hold-out test；train/val 留给 ③b
    UPSTREAM_SPLIT_SEED = 0
    UPSTREAM_IMG_SIZE = 224
    UPSTREAM_OTHER_IMAGE_PROB = 0.4

    RUN_ROOT = HERE / "runs" / "compare"
    WEIGHTS = "yolo11n.pt"

    MODES: tuple[str, ...] = ALL_MODES
    SEEDS: tuple[int, ...] = (0,)
    IMG_SIZE = 224
    BATCH = 32
    NUM_WORKERS = 8

    EPOCHS = 100
    PATIENCE = 25
    OPTIMIZER = "AdamW"
    LR = 1e-3
    LRF = 0.01
    COS_LR = True
    WEIGHT_DECAY = 5e-4

    # True：目录齐全则直接训；不齐才回退在线物化
    PREFER_PREBUILT = True
    TEST_IMAGE_ROOT = SPLIT_DIR / "test" / "image"

    print(
        f"pipeline config: DATASET={DATASET} BY_CLASS_ROOT={BY_CLASS_ROOT} "
        f"MAX_CLASSES={MAX_CLASSES} MAX_PER_CLASS={MAX_PER_CLASS}"
    )

    upstream_log = prepare_upstream(
        rerun_extract_by_class=RERUN_EXTRACT_BY_CLASS,
        rerun_split_by_class=RERUN_SPLIT_BY_CLASS,
        rerun_dump_aug=RERUN_DUMP_AUG,
        rerun_split_aug=RERUN_SPLIT_AUG,
        by_class_root=BY_CLASS_ROOT,
        split_dir=SPLIT_DIR,
        aug_out_root=AUG_OUT_ROOT,
        aug_split_root=AUG_SPLIT_ROOT,
        dataset=DATASET,
        test_ratio=UPSTREAM_TEST_RATIO,
        val_ratio=UPSTREAM_VAL_RATIO,
        split_seed=UPSTREAM_SPLIT_SEED,
        img_size=UPSTREAM_IMG_SIZE,
        other_image_prob=UPSTREAM_OTHER_IMAGE_PROB,
        dump_seed=UPSTREAM_SPLIT_SEED,
        max_classes=MAX_CLASSES,
        max_per_class=MAX_PER_CLASS,
    )

    run_compare(
        by_class_root=BY_CLASS_ROOT,
        split_dir=SPLIT_DIR,
        run_root=RUN_ROOT,
        modes=MODES,
        seeds=SEEDS,
        img_size=IMG_SIZE,
        batch=BATCH,
        epochs=EPOCHS,
        lr=LR,
        max_classes=MAX_CLASSES,
        max_per_class=MAX_PER_CLASS,
        num_workers=NUM_WORKERS,
        weights=WEIGHTS,
        aug_split_root=AUG_SPLIT_ROOT,
        test_image_root=TEST_IMAGE_ROOT,
        prefer_prebuilt=PREFER_PREBUILT,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        lrf=LRF,
        cos_lr=COS_LR,
        weight_decay=WEIGHT_DECAY,
        upstream=upstream_log,
    )
