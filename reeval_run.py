#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-score all existing compare runs with Ultralytics ``val`` (no retrain).

Mirrors ``compare.py`` layout: scan ``runs/compare/<mode>_<tag>_s<seed>/``,
fix val+test with official YOLO classify eval, then write a comparison
summary (same shape as ``compare_summary.json``).

Expects each run to have ``ultralytics/weights/best.pt`` and
``yolo_cls_data/{train,val}/``. Creates ``yolo_cls_data/test/`` if missing.

Edit variables under ``__main__`` then run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from platform_config import BY_CLASS_ROOT, MAX_CLASSES, MAX_PER_CLASS, SPLIT_DIR
from split_by_class import ensure_split
from train import YOLO_AUTO_MODE, _breed_to_idx, _eval_yolo_best, _materialize_imagefolder, read_split_stems, weight_tag

HERE = Path(__file__).resolve().parent
ABLATION_MODES = ("none", "global", "fg_only", "bg_only", "fgbg")
ALL_MODES: tuple[str, ...] = (*ABLATION_MODES, YOLO_AUTO_MODE)
_RUN_RE = re.compile(r"^(?P<mode>.+)_(?P<tag>[^_]+)_s(?P<seed>\d+)$")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, var**0.5


def _parse_run_dir(name: str) -> tuple[str, str, int] | None:
    """Parse ``fgbg_yolo11s_s0`` → (mode, tag, seed)."""
    m = _RUN_RE.match(name)
    if not m:
        return None
    return m.group("mode"), m.group("tag"), int(m.group("seed"))


def _test_dir_empty(test_dir: Path) -> bool:
    if not test_dir.is_dir():
        return True
    return not any(test_dir.iterdir())


def reeval_run(
    *,
    run_dir: Path,
    by_class_root: Path,
    split_dir: Path,
    img_size: int = 224,
    batch: int = 32,
    num_workers: int = 2,
    max_classes: int = 0,
) -> dict:
    run_dir = Path(run_dir)
    best = run_dir / "ultralytics" / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    data_root = run_dir / "yolo_cls_data"
    if not (data_root / "val").is_dir():
        raise FileNotFoundError(data_root / "val")

    breed_to_idx = _breed_to_idx(by_class_root, max_classes)
    test_dir = data_root / "test"
    if _test_dir_empty(test_dir):
        test_stems = read_split_stems(split_dir / "test.txt")
        n = _materialize_imagefolder(by_class_root, test_stems, breed_to_idx, test_dir)
        print(f"  materialized test/ n={n} → {test_dir}")

    val_acc, test_acc, n_val, n_test = _eval_yolo_best(
        best_path=best,
        data_root=data_root,
        img_size=img_size,
        batch=batch,
        num_workers=num_workers,
    )
    parsed = _parse_run_dir(run_dir.name)
    mode = parsed[0] if parsed else run_dir.name
    seed = parsed[2] if parsed else -1
    tag = parsed[1] if parsed else ""

    out = {
        "mode": mode,
        "weight_tag": tag,
        "seed": seed,
        "run_dir": str(run_dir),
        "best": str(best),
        "best_val_top1": val_acc,
        "test_top1_at_best_val": test_acc,
        "n_val": n_val,
        "n_test": n_test,
        "n_class": len(breed_to_idx),
        "eval_backend": "ultralytics.val",
        "aug_at_eval": False,
    }
    path = run_dir / "metrics_reeval.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Also refresh metrics.json headline numbers so compare readers stay consistent.
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            old = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
        old.update(
            {
                "best_val_top1": val_acc,
                "test_top1_at_best_val": test_acc,
                "n_val": n_val,
                "n_test": n_test,
                "eval_backend": "ultralytics.val",
                "aug_at_eval": False,
                "reeval_from": str(path),
            }
        )
        metrics_path.write_text(json.dumps(old, indent=2), encoding="utf-8")

    print(
        f"  mode={mode} seed={seed}  val={val_acc:.4f}  test={test_acc:.4f}  "
        f"(n_val={n_val} n_test={n_test})"
    )
    return out


def run_reeval_compare(
    *,
    by_class_root: Path,
    split_dir: Path,
    compare_root: Path,
    modes: tuple[str, ...] | None,
    seeds: tuple[int, ...] | None,
    weights: str,
    img_size: int,
    batch: int,
    num_workers: int,
    max_classes: int,
    test_ratio: float,
    val_ratio: float,
    split_seed: int,
    max_per_class: int = 0,
) -> dict:
    """Scan compare_root, reeval matching runs, write summary like compare.py."""
    ensure_split(
        by_class_root,
        split_dir,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
        force=False,
        max_classes=max_classes,
        max_per_class=max_per_class,
    )
    tag = weight_tag(weights)
    compare_root = Path(compare_root)
    compare_root.mkdir(parents=True, exist_ok=True)

    # Discover runs
    candidates: list[tuple[str, int, Path]] = []
    for d in sorted(compare_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "ultralytics" / "weights" / "best.pt").is_file():
            continue
        parsed = _parse_run_dir(d.name)
        if parsed is None:
            print(f"skip (name): {d.name}")
            continue
        mode, run_tag, seed = parsed
        if run_tag != tag:
            print(f"skip (tag {run_tag}!={tag}): {d.name}")
            continue
        if modes is not None and mode not in modes:
            continue
        if seeds is not None and seed not in seeds:
            continue
        candidates.append((mode, seed, d))

    # Stable order: ALL_MODES then seed
    mode_order = {m: i for i, m in enumerate(ALL_MODES)}
    candidates.sort(key=lambda x: (mode_order.get(x[0], 999), x[1], x[2].name))

    print(f"reeval {len(candidates)} runs under {compare_root} (tag={tag})")
    rows: list[dict] = []
    for mode, seed, run_dir in candidates:
        print(f"\n=== reeval mode={mode} seed={seed} → {run_dir} ===")
        try:
            out = reeval_run(
                run_dir=run_dir,
                by_class_root=by_class_root,
                split_dir=split_dir,
                img_size=img_size,
                batch=batch,
                num_workers=num_workers,
                max_classes=max_classes,
            )
            rows.append(out)
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "error": str(e),
                }
            )

    by_mode: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if "error" in r or "best_val_top1" not in r:
            continue
        m = str(r["mode"])
        by_mode.setdefault(m, {"val": [], "test": []})
        by_mode[m]["val"].append(float(r["best_val_top1"]))
        by_mode[m]["test"].append(float(r["test_top1_at_best_val"]))

    summary_modes = {}
    ordered_modes = [m for m in ALL_MODES if m in by_mode] + [m for m in by_mode if m not in ALL_MODES]
    for mode in ordered_modes:
        packs = by_mode[mode]
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
        "compare_root": str(compare_root),
        "weights": weights,
        "weight_tag": tag,
        "max_classes": max_classes,
        "eval_aug": False,
        "eval_backend": "ultralytics.val",
        "note": "Re-scored existing best.pt; no retrain. Replaces broken custom-loader numbers.",
        "runs": rows,
        "by_mode": summary_modes,
    }
    out_path = compare_root / "compare_summary_reeval.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Also overwrite compare_summary.json so the headline file is trustworthy.
    main_summary = compare_root / "compare_summary.json"
    main_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nwrote {out_path}")
    print(f"wrote {main_summary}")
    print("eval_aug=False | backend=ultralytics.val")
    for mode in ordered_modes:
        s = summary_modes[mode]
        print(
            f"  {mode:10s}  val={s['mean_best_val_top1']:.4f}±{s['std_best_val_top1']:.4f}  "
            f"test={s['mean_test_top1_at_best_val']:.4f}±{s['std_test_top1_at_best_val']:.4f}"
        )
    return report


if __name__ == "__main__":
    # nohup .../python reeval_run.py > reeval.log 2>&1 &
    # 数据路径 / 冒烟：platform_config.py
    COMPARE_ROOT = HERE / "runs" / "compare"

    # 划分（只 ensure / 复用，不重训）
    TEST_RATIO = 0.1
    VAL_RATIO = 0.2
    SPLIT_SEED = 0

    # 与当时训练一致（图中六组均为 yolo11s / s0 / 冒烟 8 类）
    WEIGHTS = "yolo11s.pt"
    MODES: tuple[str, ...] | None = ALL_MODES  # None=目录里扫到的都测
    SEEDS: tuple[int, ...] | None = None  # None=所有 seed；或 (0,)
    IMG_SIZE = 224
    BATCH = 32
    NUM_WORKERS = 2

    run_reeval_compare(
        by_class_root=BY_CLASS_ROOT,
        split_dir=SPLIT_DIR,
        compare_root=COMPARE_ROOT,
        modes=MODES,
        seeds=SEEDS,
        weights=WEIGHTS,
        img_size=IMG_SIZE,
        batch=BATCH,
        num_workers=NUM_WORKERS,
        max_classes=MAX_CLASSES,
        max_per_class=MAX_PER_CLASS,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
    )
