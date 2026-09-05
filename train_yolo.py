"""用 Ultralytics 在本地 YOLO 数据集上微调检测模型。"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(r"D:/Atrain")
PROJECT_DIR = ROOT / "runs"
# 角色专模：Aset/<name> 图片 + Xout/<name> 标签，run 名与角色目录名相同。
# 过图五类：runs/solarwarden_b。自动标注小模型另开 runs/<角色>（现 varien_t）。
CHARACTER = "varien_t"
RUN_NAME = "solarwarden_b"
# None = 从 yolo26n 预训练重训；暖启动可填上次 best.pt
RESUME_WEIGHTS = None
CLASS_NAMES = ["boss", "gate", "loot", "mon", "player"]
CHAR_CLASS_NAMES = ["player"]
# 小数据默认冻骨架（YOLO26 前 10 层）；--freeze 0 关闭
FREEZE = 10
# 上限轮数 + early stop：val mAP 连续 PATIENCE 轮不升就停，best.pt 仍是峰值
EPOCHS = 100
PATIENCE = 20
SPLIT_SEED = 0


def resolve_dataset_dirs(root: Path, character: str | None) -> tuple[Path, Path, str]:
    """返回 (图片目录, 标签目录, run 名)。"""
    if character:
        name = character.strip()
        image_dir = root / "Aset" / name
        label_dir = root / "Xout" / name
        pngs = list(image_dir.glob("*.png")) if image_dir.is_dir() else []
        txts = (
            [t for t in label_dir.glob("*.txt") if t.name.lower() != "classes.txt"]
            if label_dir.is_dir()
            else []
        )
        if len(pngs) < 10 or len(txts) < 10:
            raise FileNotFoundError(
                f"角色数据集不足: {image_dir} png={len(pngs)}, {label_dir} txt={len(txts)}"
            )
        return image_dir, label_dir, name
    return (*discover_image_and_label_dirs(root), RUN_NAME)


def discover_image_and_label_dirs(root: Path) -> tuple[Path, Path]:
    """扫描 D:/Atrain：优先 Aset/<角色> + Xout/<角色>；不再假设 PNG 堆在 Aset 根目录。"""
    aset = root / "Aset"
    xout = root / "Xout"
    if aset.is_dir() and xout.is_dir():
        for sub in sorted(p for p in aset.iterdir() if p.is_dir()):
            labels = xout / sub.name
            pngs = list(sub.glob("*.png"))
            txts = (
                [t for t in labels.glob("*.txt") if t.name.lower() != "classes.txt"]
                if labels.is_dir()
                else []
            )
            if len(pngs) >= 10 and len(txts) >= 10:
                return sub, labels
    image_dir = None
    label_dir = None
    skip = {"runs", "aset", "xout"}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.lower() in skip:
            continue
        pngs = list(p.glob("*.png"))
        txts = [t for t in p.glob("*.txt") if t.name.lower() != "classes.txt"]
        if image_dir is None and len(pngs) >= 10:
            image_dir = p
        if label_dir is None and len(txts) >= 10:
            label_dir = p
    if image_dir is None or label_dir is None:
        raise FileNotFoundError(
            f"在 {root} 下未找到图片目录(*.png) 或标签目录(*.txt)。"
            f" image={image_dir} label={label_dir}"
        )
    return image_dir, label_dir


def load_class_names(image_dir: Path, label_dir: Path, character: str | None) -> list[str]:
    for cand in (image_dir / "classes.txt", label_dir / "classes.txt"):
        if cand.is_file():
            names = [ln.strip() for ln in cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if names:
                return names
    return list(CHAR_CLASS_NAMES if character else CLASS_NAMES)


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def paired_stems(image_dir: Path, label_dir: Path) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    pngs = {p.stem: p for p in image_dir.glob("*.png")} if image_dir.is_dir() else {}
    txts = {}
    if label_dir.is_dir():
        for p in label_dir.glob("*.txt"):
            if p.name.lower() != "classes.txt" and p.parent == label_dir:
                txts[p.stem] = p
    stems = sorted(set(pngs) & set(txts))
    return pngs, txts, stems


def prepare_dataset(
    image_dir: Path,
    label_dir: Path,
    class_names: list[str],
    stage_dir: Path,
    extra_train: list[tuple[Path, Path]] | None = None,
    seed: int = SPLIT_SEED,
) -> Path:
    """主集按 7/2/1 切分；extra 只进 train（blank 负样本不要进 val，便于和 SOLAR_F 对比）。"""
    pngs, txts, stems = paired_stems(image_dir, label_dir)
    if not stems:
        raise FileNotFoundError(f"标签与 {image_dir} 图片没有同名配对。")
    missing_img = sorted(set(txts) - set(pngs))
    missing_lbl = sorted(set(pngs) - set(txts))
    if missing_img:
        print(f"警告: {len(missing_img)} 个标签没有对应 PNG，已跳过")
    if missing_lbl:
        print(f"警告: {len(missing_lbl)} 张图没有对应 txt，已跳过")

    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train : n_train + n_val],
        "test": stems[n_train + n_val :],
    }
    if not splits["test"]:
        splits["test"] = splits["val"][-1:]
        splits["val"] = splits["val"][:-1]

    extra_pngs: dict[str, Path] = {}
    extra_txts: dict[str, Path] = {}
    extra_added = 0
    for eimg, elbl in extra_train or []:
        ep, et, es = paired_stems(eimg, elbl)
        for stem in es:
            key = stem if stem not in pngs else f"{eimg.name}_{stem}"
            extra_pngs[key] = ep[stem]
            extra_txts[key] = et[stem]
            splits["train"].append(key)
            extra_added += 1
        print(f"extra train: {eimg.name} +{len(es)}（只进 train）")
    pngs = {**pngs, **extra_pngs}
    txts = {**txts, **extra_txts}

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    for split, names in splits.items():
        img_out = stage_dir / "images" / split
        lbl_out = stage_dir / "labels" / split
        for stem in names:
            _copy_or_link(pngs[stem], img_out / f"{stem}.png")
            _copy_or_link(txts[stem], lbl_out / f"{stem}.txt")

    data_yaml = stage_dir / "data.yaml"
    yaml_text = (
        f"path: {stage_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(class_names))
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")
    print(
        f"数据集已准备: train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} extra_train={extra_added} -> {data_yaml}"
    )
    return data_yaml


def parse_args():
    p = argparse.ArgumentParser(description="训练 YOLO 检测模型")
    p.add_argument(
        "--char",
        default=CHARACTER,
        help="主集目录名（Aset/<name> + Xout/<name>，val/test 只来自这里）",
    )
    p.add_argument(
        "--extra",
        default="",
        help="只进 train 的附加集，逗号分隔，如 blank",
    )
    p.add_argument("--name", default="", help="runs 目录名，默认与 --char 相同")
    p.add_argument(
        "--freeze",
        type=int,
        default=FREEZE,
        help="冻前 N 层（YOLO26 骨架约 10）。0=不冻",
    )
    p.add_argument(
        "--weights",
        default="",
        help="初始权重。空则 yolo26n.pt；续训填 runs/.../best.pt",
    )
    return p.parse_args()


def train_my_yolo(
    character: str | None = CHARACTER,
    extra: str = "",
    run_name: str = "",
    freeze: int = FREEZE,
    weights: str = "",
):
    character = (character or "").strip() or None
    image_dir, label_dir, default_name = resolve_dataset_dirs(ROOT, character)
    run_name = (run_name or "").strip() or default_name
    class_names = load_class_names(image_dir, label_dir, character)
    extra_dirs: list[tuple[Path, Path]] = []
    for raw in extra.split(","):
        name = raw.strip()
        if not name:
            continue
        extra_dirs.append((ROOT / "Aset" / name, ROOT / "Xout" / name))

    print(f"图片目录: {image_dir}")
    print(f"标签目录: {label_dir}")
    print(f"run: {run_name}")
    print(f"类别: {class_names}")
    print(f"freeze: {freeze}  extra: {[p[0].name for p in extra_dirs] or '(无)'}")
    n_img = len(list(image_dir.glob("*.png")))
    print(
        f"轮数策略: 主集 {n_img} 张 / YOLO26n / 关 mosaic / freeze={freeze} → "
        f"epochs={EPOCHS} 上限, patience={PATIENCE}"
    )

    stage_dir = ROOT / "yolo_data" / run_name
    data_yaml = prepare_dataset(
        image_dir, label_dir, class_names, stage_dir, extra_train=extra_dirs
    )
    if not torch.cuda.is_available():
        print("未检测到 CUDA，请先安装 GPU 版 PyTorch。")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"data: {data_yaml}")

    weights_arg = (weights or "").strip()
    if weights_arg and Path(weights_arg).is_file():
        weights_path = weights_arg
    elif RESUME_WEIGHTS and Path(RESUME_WEIGHTS).is_file():
        weights_path = str(RESUME_WEIGHTS)
    else:
        weights_path = "yolo26n.pt"
    print(f"weights: {weights_path}")
    model = YOLO(weights_path)

    n_train_png = len(list((stage_dir / "images" / "train").glob("*.png")))
    batch = 16 if n_train_png < 120 else 32

    train_kw = dict(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=640,
        device=0,
        batch=batch,
        workers=4,
        amp=True,
        mosaic=0.0,
        mixup=0.0,
        conf=0.1,
        iou=0.7,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
    if freeze and freeze > 0:
        train_kw["freeze"] = freeze

    results = model.train(**train_kw)

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n训练完成，最佳权重: {best}")


if __name__ == "__main__":
    args = parse_args()
    train_my_yolo(
        args.char,
        extra=args.extra,
        run_name=args.name,
        freeze=args.freeze,
        weights=args.weights,
    )
