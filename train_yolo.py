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
# 过图集：Aset/solarwarden/<name>/*.png + labels/*.txt。Xout 只给 X-AnyLabeling 导出。
# GUI 过图默认权重仍是 solarwarden_b，未切到 d。
CHARACTER = "varien_t"
RUN_NAME = "solarwarden_d"
RESUME_WEIGHTS = None
CLASS_NAMES = ["boss", "gate", "loot", "mon", "player"]
CHAR_CLASS_NAMES = ["player"]
FREEZE = 0
MOSAIC = 0.0
# 训练/val 用；实机推理下次再问，不要当长期默认写进 GUI
TRAIN_CONF = 0.15
TRAIN_IOU = 0.5
# 上限轮数 + early stop：val mAP 连续 PATIENCE 轮不升就停，best.pt 仍是峰值
EPOCHS = 100
PATIENCE = 20
SPLIT_SEED = 0


def _pngs(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return list(image_dir.glob("*.png"))


def _txts(label_dir: Path) -> list[Path]:
    if not label_dir.is_dir():
        return []
    return [t for t in label_dir.glob("*.txt") if t.name.lower() != "classes.txt"]


def resolve_dataset_dirs(root: Path, character: str | None) -> tuple[Path, Path, str]:
    """返回 (图片目录, 标签目录, run 名)。过图优先 Aset/.../labels，不再默认 Xout。"""
    if character:
        name = character.strip()
        for image_dir, label_dir in (
            (root / "Aset" / "solarwarden" / name, root / "Aset" / "solarwarden" / name / "labels"),
            (root / "Aset" / name, root / "Aset" / name / "labels"),
        ):
            if len(_pngs(image_dir)) >= 10 and len(_txts(label_dir)) >= 10:
                return image_dir, label_dir, name
        raise FileNotFoundError(
            f"未找到 {name}：需要 png + labels/*.txt（例如 Aset/solarwarden/{name}）"
        )
    return (*discover_image_and_label_dirs(root), RUN_NAME)


def discover_image_and_label_dirs(root: Path) -> tuple[Path, Path]:
    """优先 Aset/solarwarden/<集>/labels，其次 Aset/<集>/labels。"""
    aset = root / "Aset"
    search: list[Path] = []
    sw = aset / "solarwarden"
    if sw.is_dir():
        search.extend(sorted(p for p in sw.iterdir() if p.is_dir()))
    if aset.is_dir():
        search.extend(sorted(p for p in aset.iterdir() if p.is_dir() and p.name.lower() != "solarwarden"))
    for image_dir in search:
        label_dir = image_dir / "labels"
        if len(_pngs(image_dir)) >= 10 and len(_txts(label_dir)) >= 10:
            return image_dir, label_dir
    raise FileNotFoundError(f"在 {root}/Aset 下未找到 png + labels/*.txt")


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
        help="主集目录名（Aset/solarwarden/<name> 或 Aset/<name>，标签在 labels/）",
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
        "--mosaic",
        type=float,
        default=MOSAIC,
        help="mosaic 概率。默认 0=关",
    )
    p.add_argument(
        "--images",
        default="",
        help="图片目录。空则 Aset/<char>",
    )
    p.add_argument(
        "--labels",
        default="",
        help="YOLO txt 目录。空则 <图片目录>/labels",
    )
    p.add_argument(
        "--weights",
        default="",
        help="初始权重。空则 yolo26n.pt；续训填 runs/.../best.pt",
    )
    p.add_argument(
        "--suite",
        default="",
        help="solarwarden_c = 同一拆分连训 A/B/C",
    )
    return p.parse_args()


def resolve_start_weights(weights: str) -> str:
    weights_arg = (weights or "").strip()
    if weights_arg and Path(weights_arg).is_file():
        return weights_arg
    if RESUME_WEIGHTS and Path(RESUME_WEIGHTS).is_file():
        return str(RESUME_WEIGHTS)
    return "yolo26n.pt"


def run_ultralytics_train(
    data_yaml: Path,
    run_name: str,
    freeze: int,
    mosaic: float,
    weights: str = "",
    n_train_png: int | None = None,
    conf: float = TRAIN_CONF,
    iou: float = TRAIN_IOU,
):
    if not torch.cuda.is_available():
        print("未检测到 CUDA，请先安装 GPU 版 PyTorch。")
        return None

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"data: {data_yaml}")
    print(f"run: {run_name}  freeze={freeze}  mosaic={mosaic}")
    weights_path = resolve_start_weights(weights)
    print(f"weights: {weights_path}")
    model = YOLO(weights_path)

    if n_train_png is None:
        n_train_png = len(list((data_yaml.parent / "images" / "train").glob("*.png")))
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
        mosaic=float(mosaic),
        mixup=0.0,
        conf=float(conf),
        iou=float(iou),
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
    return best


def train_solarwarden_c_suite():
    """同一 7/2/1 拆分、同一 yolo26n 起点，连训 A/B/C。"""
    image_dir = ROOT / "Aset" / "solarwarden" / "solarwarden_c"
    label_dir = image_dir / "labels"
    class_names = load_class_names(image_dir, label_dir, None)
    print(f"图片目录: {image_dir}")
    print(f"标签目录: {label_dir}")
    print(f"类别: {class_names}")
    n_img = len(list(image_dir.glob("*.png")))
    print(
        f"solarwarden_c suite: 主集 {n_img} 张 / YOLO26n / "
        f"epochs={EPOCHS} 上限, patience={PATIENCE} / conf=0.1 iou=0.7"
    )
    stage_dir = ROOT / "yolo_data" / "solarwarden_c"
    data_yaml = prepare_dataset(image_dir, label_dir, class_names, stage_dir)
    jobs = [
        ("solarwarden_c_a", 0, 0.0),
        ("solarwarden_c_b", 0, 1.0),
        ("solarwarden_c_c", 10, 0.0),
    ]
    for run_name, freeze, mosaic in jobs:
        print(f"\n======== {run_name} freeze={freeze} mosaic={mosaic} ========")
        run_ultralytics_train(data_yaml, run_name, freeze, mosaic)


def train_my_yolo(
    character: str | None = CHARACTER,
    extra: str = "",
    run_name: str = "",
    freeze: int = FREEZE,
    mosaic: float = MOSAIC,
    weights: str = "",
    images: str = "",
    labels: str = "",
):
    character = (character or "").strip() or None
    if images.strip() and labels.strip():
        image_dir = Path(images)
        label_dir = Path(labels)
        default_name = image_dir.name
    else:
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
    print(f"freeze: {freeze}  mosaic: {mosaic}  extra: {[p[0].name for p in extra_dirs] or '(无)'}")
    n_img = len(list(image_dir.glob("*.png")))
    print(
        f"轮数策略: 主集 {n_img} 张 / YOLO26n / mosaic={mosaic} / freeze={freeze} → "
        f"epochs={EPOCHS} 上限, patience={PATIENCE} / conf={TRAIN_CONF} iou={TRAIN_IOU}"
    )

    stage_dir = ROOT / "yolo_data" / run_name
    data_yaml = prepare_dataset(
        image_dir, label_dir, class_names, stage_dir, extra_train=extra_dirs
    )
    run_ultralytics_train(data_yaml, run_name, freeze, mosaic, weights=weights)


if __name__ == "__main__":
    args = parse_args()
    if (args.suite or "").strip().lower() == "solarwarden_c":
        train_solarwarden_c_suite()
    else:
        train_my_yolo(
            args.char,
            extra=args.extra,
            run_name=args.name,
            freeze=args.freeze,
            mosaic=args.mosaic,
            weights=args.weights,
            images=args.images,
            labels=args.labels,
        )
