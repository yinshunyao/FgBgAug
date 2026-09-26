# FgBgAug

面向形态敏感识别的前景–背景**双强度**数据增强。

[English](README.md) · 文档：[中文](docs/zh/) / [English](docs/en/)

同一张图、两种强度——不是新的 MixUp/CutMix 算子：

| 分支 | 行为 |
|:---|:---|
| 前景 | 刚体旋转 + 弱光度（**无**色相抖动、**无**缩放） |
| 背景 | 强光度，和/或从其它图 / 纯色画布重采样 |

推理：模型不变，零额外开销（mask 仅在构造训练样本时使用）。

## 消融模式

| `AUG_MODE` | 含义 |
|:---|:---|
| `none` | 仅 resize / flip |
| `global` | 轻度整图光度 + 仿射（常规全局基线） |
| `fg_only` | 仅弱前景 |
| `bg_only` | 仅强背景；前景像素不变 |
| `fgbg` | 完整双强度（本方法） |

## 流水线（推荐执行顺序）

```text
① extract_pet_by_class.py   → by_class/{image,fg,bg,bbox}/<breed>/
② split_by_class.py         → splits/{trainval,test}/ + txt 列表
                              （Mac 冒烟子集；Linux 全量；复制 image/fg/bg/bbox）
                              （一开始配 TEST_RATIO / VAL_RATIO；流程：先抽 test，再划 train/val）
③ dump_aug.py               → 增强 SPLIT_DIR/trainval 全部样本 → AUG_OUT_ROOT（A1–A4）
③b split_aug.py             → AUG_OUT_ROOT → AUG_SPLIT_ROOT/{A*}/{train,val}/（默认 8:2）
④ compare.py                → A0–A4 + yolo_auto；每组报 val + test（评估关增强）
```

| 步 | 脚本 | 做什么 |
|:---|:---|:---|
| ① | `extract_pet_by_class.py` | by_class 裁切 + fg/bg |
| ② | `split_by_class.py` | 比例预先配置；先 hold-out test；写出 txt **并复制** `trainval/` / `test/` 类别树（Mac 冒烟 / Linux 全量） |
| ③ | `dump_aug.py` | 统一增强 **trainval/** 全部图 ×`AUG_COUNT`(默认4) → `AUG_OUT_ROOT`；test 不动 |
| ③b | `split_aug.py` | `INCLUDE_A0` 时必含/补齐 `A0_none`；`AUG_VAL_USE_AUG=True` 时 val 也可含增强图（`AUG_RANDOM_SPLIT` 可选按文件随机）；`False` 时先按原图 stem 划 val（仅原图）再把剩余 stem 的增强进 train；各模式共用同一划分 |
| ④ | `compare.py` | 优先用 `AUG_SPLIT_ROOT` train/val + `SPLIT_DIR/test`；目录齐全则**不再增强**。`__main__` 里 `RERUN_SPLIT_BY_CLASS` / `RERUN_DUMP_AUG` / `RERUN_SPLIT_AUG` 可先删对应产物目录再重跑 ②/③/③b（早开级联晚开） |

对比臂：

| mode | 训练 | Ultralytics 日志 | 评估 |
|:---|:---|:---|:---|
| `none` / `global` / `fg_only` / `bg_only` / `fgbg` | 先把 fgbg 写进 train ImageFolder，再 `YOLO.train`，**YOLO 自带增强关闭** | 与产线相同 | val + test，无增强 |
| `yolo_auto` | 干净 train 图 + `YOLO.train`，**YOLO 自带增强打开** | 标准 classify 日志 | 同样 val + test，无增强 |

说明：以前消融走自定义 PyTorch 循环，所以只有一行 `mode=… loss=…`；产线才有 Ultralytics 横幅。现在两边都走 `YOLO.train`，日志风格一致。

幂等约定：

- **划分**：完整 `train/val/test` 列表 **且** `trainval/image`+`test/image` 目录就绪后默认不重抽；仅 `FORCE_RESPLIT=True` 才重写。旧版只有 txt、无复制目录视为不完整并重写。
- **落盘增强**：`SKIP_EXISTING=True`，已增强样本不重算、不删除。

辅助（可选，不挡主链路）：

```bash
python fgbg_aug.py     # smoke：加载一张样本，打印 photo-L2
python visualize.py    # 机制图 1×5（A4 = FG×N pick + 随机复杂 BG）
python train.py        # 单 AUG_MODE（与 compare 同数据路径，在线增强）
```

## 快速开始

```bash
pip install -r requirements.txt

# ① by_class
python extract_pet_by_class.py

# ② 分出 test（默认 TEST_RATIO=0.1）
python split_by_class.py

# ③ 输出增强（仅 trainval）
python dump_aug.py

# ③b 增强产物划 train/val（默认 8:2）
python split_aug.py

# ④ 对比训练（A0–A4 + yolo_auto；每组 val+test）
#    需重跑上游时：在 compare.py __main__ 打开 RERUN_*（先删产物再执行；②⇒③⇒③b）
python compare.py
```

默认数据根 / 冒烟（统一见 ``platform_config.py``；按 `sys.platform` 分流）:

```text
Mac:   /Volumes/shunyao-h1/基线数据/Oxford-IIIT Pet/by_class
Linux: /data/samples/base/Oxford-IIIT Pet/by_class
SPLIT_DIR        = <BY_CLASS_ROOT>/splits
AUG_OUT_ROOT     = <PET_ROOT>/by_class-aug-A1A4
AUG_SPLIT_ROOT   = <PET_ROOT>/by_class-aug-A1A4-split
AUG_VAL_RATIO    = 0.2   # train:val = 8:2（按划分单元）
AUG_VAL_USE_AUG  = True  # True=现有：val 也用增强图；False=val 仅原图（先划 val stem）
INCLUDE_A0       = True  # 原图 A0_none 对照（dump / split 都会纳入）
AUG_COUNT        = 4
AUG_RANDOM_SPLIT = False # False=同源 stem 同侧；True=按文件全随机（仅 AUG_VAL_USE_AUG=True 时生效）
TEST_RATIO  = 0.1
VAL_RATIO   = 0.1
冒烟: Mac 8×40；Linux MAX_CLASSES=MAX_PER_CLASS=0（全量）
```

`dump_aug.py` 读 `SPLIT_DIR/trainval/`，写出到 `AUG_OUT_ROOT`（含可选 `A0_none`）。  
`split_aug.py` 再划成 `AUG_SPLIT_ROOT/{A0,A1…}/{train,val}/<breed>/`（各模式共用划分；`INCLUDE_A0` 且缺/不全 A0 时从 trainval 补齐）。

改路径、冒烟、增强输出、A0、`AUG_VAL_USE_AUG`、随机划分或 train/val 比例时只改 `platform_config.py`。

训练日程（各臂统一，公平对比）:

```text
EPOCHS=100  PATIENCE=25  OPTIMIZER=AdamW  LR/lr0=1e-3  cos_lr=True
```

**Val / test 永不增强。** 产线臂 `yolo_auto` 仅在 Ultralytics **训练**时开自动增强；消融臂 YOLO 自带增强关闭。

## 目录布局

```text
platform_config.py       # Mac/Linux 路径 + 冒烟默认（各脚本 __main__ 共用）
fgbg_aug.py              # 核心 augment() + by_class 加载（训练/落盘共用）
extract_pet_by_class.py  # ① Pet → by_class/{image,fg,bg,bbox}
split_by_class.py        # ② by_class → splits/{trainval,test}/ + txt（Mac 冒烟 / Linux 全量）
dump_aug.py              # ③ SPLIT_DIR/trainval → AUG_OUT_ROOT A1–A4 ImageFolder
split_aug.py             # ③b AUG_OUT_ROOT → AUG_SPLIT_ROOT/{A*}/{train,val}/
compare.py               # ④ A0–A4 对比训练（在线调用 fgbg_aug）
train.py                 # 单模式训练（同 ④ 数据路径）
visualize.py             # 机制图（可选）
requirements.txt
docs/zh/                 # 中文说明
docs/en/                 # 英文说明
```

`by_class` 布局（① 产出；②～④ / aug / viz 消费）:

```text
by_class/image/<breed>/*.jpg   # RGB crop → augment `image`
by_class/fg/<breed>/*.png      # RGBA → mask = alpha>0
by_class/bg/<breed>/*.png      # RGBA（可选；demo 可合成 bg_image）
by_class/bbox/<breed>/*.jpg    # 仅人工检视；训练/增强不用
by_class/splits/train.txt      # ② 训练 stem（可增强）
by_class/splits/val.txt        # ② 验证 stem（永不增强）
by_class/splits/test.txt       # ② 测试 stem（永不增强）
by_class/splits/trainval.txt   # train∪val（兼容旧脚本）
by_class/splits/trainval/      # ② 复制：{image,fg,bg,bbox}/<breed>/
by_class/splits/test/          # ② 复制：同上（测试集，不增强）
```

指标：`runs/compare/compare_summary.json`（含各 mode 的 val/test）。

详情：[docs/zh/参数与目录说明.md](docs/zh/参数与目录说明.md) · [docs/en/parameters-and-layout.md](docs/en/parameters-and-layout.md)  
A1–A4 增强细则：[docs/zh/A1-A4增强处理说明.md](docs/zh/A1-A4增强处理说明.md) · [docs/en/a1-a4-augmentation.md](docs/en/a1-a4-augmentation.md)
