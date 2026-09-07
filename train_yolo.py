"""用 Ultralytics 在本地 YOLO 数据集上微调检测模型。"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(r"D:/Atrain")
PROJECT_DIR = ROOT / "runs"
# 过图集：Aset/solarwarden/<name>/*.png + labels/*.txt。Xout 只给 X-AnyLabeling 导出。
# GUI 过图默认权重已切 mix_a。
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
MIX_A_CHARS = ("solarwarden_d", "varien_a", "Sam1ra_a")
MIX_A_BLANK_TRAIN = 12
MIX_A_BLANK_EVAL = 5


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


def split_721(stems: list[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    ordered = list(stems)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    splits = {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }
    if not splits["test"]:
        splits["test"] = splits["val"][-1:]
        splits["val"] = splits["val"][:-1]
    return splits


def _write_names_yaml(stage_dir: Path, class_names: list[str], val_rel: str, dest: Path) -> None:
    dest.write_text(
        f"path: {stage_dir.as_posix()}\n"
        "train: images/train\n"
        f"val: {val_rel}\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(class_names)),
        encoding="utf-8",
    )


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

    splits = split_721(stems, seed)

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
    _write_names_yaml(stage_dir, class_names, "images/val", data_yaml)
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
    p.add_argument(
        "--suite",
        default="",
        help="solarwarden_c = 同一拆分连训 A/B/C；mix_a = 混合角色分层重训",
    )
    p.add_argument("--epochs", type=int, default=EPOCHS, help="上限轮数")
    p.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="早停：val 连续 N 轮不升则停。0=关闭，跑满 epochs",
    )
    p.add_argument("--name", dest="name", default="", help="runs 目录名，默认与 --char 相同")
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
        help="初始权重。空则 weights/yolo26n.pt；续训填 runs/.../best.pt",
    )
    return p.parse_args()


def resolve_start_weights(weights: str) -> str:
    weights_arg = (weights or "").strip()
    if weights_arg and Path(weights_arg).is_file():
        return weights_arg
    if RESUME_WEIGHTS and Path(RESUME_WEIGHTS).is_file():
        return str(RESUME_WEIGHTS)
    return str(Path("weights") / "yolo26n.pt")


def run_ultralytics_train(
    data_yaml: Path,
    run_name: str,
    freeze: int,
    mosaic: float,
    weights: str = "",
    n_train_png: int | None = None,
    conf: float = TRAIN_CONF,
    iou: float = TRAIN_IOU,
    epochs: int | None = None,
    patience: int | None = None,
):
    if not torch.cuda.is_available():
        print("未检测到 CUDA，请先安装 GPU 版 PyTorch。")
        return None

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"data: {data_yaml}")
    n_epochs = int(epochs) if epochs is not None else EPOCHS
    n_patience = PATIENCE if patience is None else int(patience)
    print(f"run: {run_name}  freeze={freeze}  mosaic={mosaic}  epochs={n_epochs}  patience={n_patience}")
    weights_path = resolve_start_weights(weights)
    print(f"weights: {weights_path}")
    model = YOLO(weights_path)

    if n_train_png is None:
        n_train_png = len(list((data_yaml.parent / "images" / "train").glob("*.png")))
    batch = 16 if n_train_png < 120 else 32

    train_kw = dict(
        data=str(data_yaml),
        epochs=n_epochs,
        patience=n_patience,
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
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\n训练完成，最佳权重: {best}")
    return save_dir


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


def _empty_txt_count(label_dir: Path, stems: list[str], txts: dict[str, Path]) -> int:
    n = 0
    for stem in stems:
        text = txts[stem].read_text(encoding="utf-8")
        if not any(ln.strip() for ln in text.splitlines()):
            n += 1
    return n


def mix_a_dataset_text(
    counts: dict[str, int],
    empty_in_char: dict[str, int],
    split_sizes: dict,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> str:
    total = sum(counts.values())
    return (
        f"数据集: solarwarden_d {counts['solarwarden_d']} + varien_a {counts['varien_a']} "
        f"+ Sam1ra_a {counts['Sam1ra_a']} + blank {counts['blank']}"
        f"（{MIX_A_BLANK_TRAIN} train / {MIX_A_BLANK_EVAL} 评估，不进 val/test） = {total}\n"
        f"角色集内空框 txt: solarwarden_d {empty_in_char['solarwarden_d']} "
        f"+ Sam1ra_a {empty_in_char['Sam1ra_a']} + varien_a {empty_in_char['varien_a']}\n"
        "预标注:\n"
        "  solarwarden_d 560: 本批未逐张记录预标模型（集内已有人工校对 txt）\n"
        "  varien_a 90: solarwarden_d 权重, conf=0.15, iou=0.5, 人工修改\n"
        "  Sam1ra_a 124: solarwarden_d 权重, conf=0.1, iou=0.7, save_conf/score, 人工修改\n"
        "  blank 17: 纯负样本，无框\n"
        "blank 结案（mix_a_e100，5 张独立评估，best=last）:\n"
        "  conf=0.15: 总误检 1（mon×1）；conf=0.05: 总误检 2（mon×2）\n"
        "  均在 20260902_185757_472357；其余 4 张全空。\n"
        "  结论: 空图抑制可用，本轮不作为主矛盾，不扩 blank。\n"
        "本 run: Sam1ra 已按 json 重导修正 txt；patience=0 跑满 300 为最终版。\n"
        "起点: yolo26n.pt 全量重训, freeze=0, mosaic=0, mixup=0, imgsz=640\n"
        "切分: 三角色各自 7/2/1 seed=0 再合并；blank 12 train + 5 评估 seed=0\n"
        f"训练: epochs={epochs}, patience={patience}（0=关早停）, batch=32（train>=120）, "
        f"conf={TRAIN_CONF}, iou={TRAIN_IOU}, device=0\n"
        "评估: 训完 best.pt 与 last.pt 各跑三角色 val（含每类 P/R/mAP50/mAP50-95）"
        " + blank 评估集 conf=0.15/0.05 误检。不每 epoch 打分角色 mAP。\n"
        f"切分张数: {json.dumps(split_sizes, ensure_ascii=False)}\n"
        "GUI 过图默认权重已切 mix_a。\n"
    )


def prepare_mix_a_dataset(stage_dir: Path, class_names: list[str]) -> tuple[Path, dict]:
    """三角色各自 7/2/1；blank 12 进 train、5 张独立评估。"""
    root = ROOT / "Aset" / "mix_a"
    if not root.is_dir():
        raise FileNotFoundError(root)

    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    counts: dict[str, int] = {}
    empty_in_char: dict[str, int] = {}
    split_sizes: dict[str, dict[str, int]] = {}
    split_stems: dict[str, dict[str, list[str]]] = {}

    for name in MIX_A_CHARS:
        image_dir = root / name
        label_dir = image_dir / "labels"
        pngs, txts, stems = paired_stems(image_dir, label_dir)
        if not stems:
            raise FileNotFoundError(f"{name} 没有 png+txt 配对")
        missing_lbl = sorted(set(pngs) - set(txts))
        if missing_lbl:
            print(f"警告: {name} {len(missing_lbl)} 张图没有 txt，已跳过")
        counts[name] = len(stems)
        empty_in_char[name] = _empty_txt_count(label_dir, stems, txts)
        parts = split_721(stems, SPLIT_SEED)
        split_stems[name] = parts
        split_sizes[name] = {k: len(v) for k, v in parts.items()}
        for split, stem_list in parts.items():
            img_out = stage_dir / "images" / split
            lbl_out = stage_dir / "labels" / split
            char_img = stage_dir / "images" / f"{split}_{name}"
            char_lbl = stage_dir / "labels" / f"{split}_{name}"
            for stem in stem_list:
                key = f"{name}_{stem}"
                _copy_or_link(pngs[stem], img_out / f"{key}.png")
                _copy_or_link(txts[stem], lbl_out / f"{key}.txt")
                if split == "val":
                    _copy_or_link(pngs[stem], char_img / f"{key}.png")
                    _copy_or_link(txts[stem], char_lbl / f"{key}.txt")

    blank_dir = root / "blank"
    blank_pngs = sorted(blank_dir.glob("*.png"), key=lambda p: p.name.lower()) if blank_dir.is_dir() else []
    if len(blank_pngs) < MIX_A_BLANK_TRAIN + MIX_A_BLANK_EVAL:
        raise FileNotFoundError(
            f"blank 需要至少 {MIX_A_BLANK_TRAIN + MIX_A_BLANK_EVAL} 张，实际 {len(blank_pngs)}"
        )
    counts["blank"] = len(blank_pngs)
    rng = random.Random(SPLIT_SEED)
    blank_stems = [p.stem for p in blank_pngs]
    rng.shuffle(blank_stems)
    blank_train = blank_stems[:MIX_A_BLANK_TRAIN]
    blank_eval = blank_stems[MIX_A_BLANK_TRAIN : MIX_A_BLANK_TRAIN + MIX_A_BLANK_EVAL]
    leftover = blank_stems[MIX_A_BLANK_TRAIN + MIX_A_BLANK_EVAL :]
    if leftover:
        print(f"警告: blank 多出 {len(leftover)} 张未使用")
    split_sizes["blank"] = {"train": len(blank_train), "eval": len(blank_eval), "unused": len(leftover)}
    split_stems["blank"] = {"train": blank_train, "eval": blank_eval}
    png_by_stem = {p.stem: p for p in blank_pngs}

    empty_path = stage_dir / "_empty.txt"
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    empty_path.write_text("", encoding="utf-8")
    for stem in blank_train:
        key = f"blank_{stem}"
        _copy_or_link(png_by_stem[stem], stage_dir / "images" / "train" / f"{key}.png")
        _copy_or_link(empty_path, stage_dir / "labels" / "train" / f"{key}.txt")
    for stem in blank_eval:
        key = f"blank_{stem}"
        _copy_or_link(png_by_stem[stem], stage_dir / "images" / "blank_eval" / f"{key}.png")
        _copy_or_link(empty_path, stage_dir / "labels" / "blank_eval" / f"{key}.txt")
    empty_path.unlink(missing_ok=True)

    data_yaml = stage_dir / "data.yaml"
    _write_names_yaml(stage_dir, class_names, "images/val", data_yaml)
    for name in MIX_A_CHARS:
        _write_names_yaml(
            stage_dir,
            class_names,
            f"images/val_{name}",
            stage_dir / f"val_{name}.yaml",
        )

    manifest = {
        "counts": counts,
        "empty_in_char": empty_in_char,
        "split_sizes": split_sizes,
        "split_stems": split_stems,
    }
    (stage_dir / "splits.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    n_train = len(list((stage_dir / "images" / "train").glob("*.png")))
    n_val = len(list((stage_dir / "images" / "val").glob("*.png")))
    n_test = len(list((stage_dir / "images" / "test").glob("*.png")))
    print(
        f"mix_a 已准备: train={n_train} val={n_val} test={n_test} "
        f"blank_eval={len(blank_eval)} -> {data_yaml}"
    )
    for name in MIX_A_CHARS:
        print(f"  {name}: {split_sizes[name]}")
    print(f"  blank: {split_sizes['blank']}")
    return data_yaml, manifest


def _as_float_list(x) -> list[float] | None:
    if x is None:
        return None
    try:
        return [float(v) for v in list(x)]
    except TypeError:
        return [float(x)]


def _box_metrics(metrics, class_names: list[str]) -> dict:
    box = metrics.box
    p = _as_float_list(getattr(box, "p", None))
    r = _as_float_list(getattr(box, "r", None))
    maps = _as_float_list(getattr(box, "maps", None))
    ap50 = _as_float_list(getattr(box, "ap50", None))
    if ap50 is None:
        all_ap = getattr(box, "all_ap", None)
        if all_ap is not None and len(all_ap) and len(all_ap[0]) > 0:
            ap50 = [float(row[0]) for row in all_ap]
    per: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        per[name] = {
            "precision": p[i] if p and i < len(p) else None,
            "recall": r[i] if r and i < len(r) else None,
            "map50": ap50[i] if ap50 and i < len(ap50) else None,
            "map50_95": maps[i] if maps and i < len(maps) else None,
        }
    return {
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per,
    }


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def _yolo_txt_xyxy(txt: Path, iw: int, ih: int, class_id: int) -> list[list[float]]:
    out = []
    if not txt.is_file():
        return out
    for raw in txt.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        if int(float(parts[0])) != class_id:
            continue
        cx, cy, bw, bh = (float(parts[i]) for i in range(1, 5))
        x1 = (cx - bw * 0.5) * iw
        y1 = (cy - bh * 0.5) * ih
        x2 = (cx + bw * 0.5) * iw
        y2 = (cy + bh * 0.5) * ih
        out.append([x1, y1, x2, y2])
    return out


def dump_sam1ra_loot_fn(
    weights: Path,
    stage_dir: Path,
    out_dir: Path,
    class_names: list[str],
    max_images: int = 8,
) -> Path:
    """Sam1ra val 上 loot 漏检（GT 无 IoU>=0.5 匹配），绿=GT 红=预测 黄=漏检。"""
    import cv2
    from auto_label import class_aware_nms, png_size

    cid = class_names.index("loot")
    img_dir = stage_dir / "images" / "val_Sam1ra_a"
    lbl_dir = stage_dir / "labels" / "val_Sam1ra_a"
    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    pngs = sorted(img_dir.glob("*.png"), key=lambda p: p.name.lower())
    for path in pngs:
        iw, ih = png_size(path)
        gt = _yolo_txt_xyxy(lbl_dir / f"{path.stem}.txt", iw, ih, cid)
        if not gt:
            continue
        r = model.predict(
            source=str(path),
            conf=TRAIN_CONF,
            iou=TRAIN_IOU,
            imgsz=640,
            device=0,
            verbose=False,
            save=False,
        )[0]
        pred = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.tolist()
            clses = [int(c) for c in r.boxes.cls.tolist()]
            confs = r.boxes.conf.tolist()
            keep = class_aware_nms(xyxy, confs, clses, TRAIN_IOU)
            pred = [xyxy[i] for i in keep if clses[i] == cid]
        matched = [False] * len(pred)
        fn = []
        for g in gt:
            best_j, best_iou = -1, 0.0
            for j, pbox in enumerate(pred):
                if matched[j]:
                    continue
                iou = _iou_xyxy(g, pbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= 0.5:
                matched[best_j] = True
            else:
                fn.append(g)
        if not fn:
            continue
        records.append({"png": path.name, "gt": len(gt), "pred": len(pred), "fn": len(fn), "path": str(path)})
        im = cv2.imread(str(path))
        if im is None:
            continue
        for box in gt:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for box in pred:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for box in fn:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            im,
            f"loot GT={len(gt)} pred={len(pred)} FN={len(fn)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imwrite(str(out_dir / path.name), im)
    records.sort(key=lambda x: (-x["fn"], x["png"]))
    keep_names = {x["png"] for x in records[:max_images]}
    for p in out_dir.glob("*.png"):
        if p.name not in keep_names:
            p.unlink()
    summary = out_dir / "fn.json"
    summary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Sam1ra loot 漏检图 {len(records)} 张，抽出 {min(max_images, len(records))} -> {out_dir}")
    return summary


def _blank_fp_counts(model, image_dir: Path, class_names: list[str], conf: float, iou: float) -> dict:
    from auto_label import class_aware_nms

    pngs = sorted(image_dir.glob("*.png"), key=lambda p: p.name.lower())
    total = 0
    by_class = {name: 0 for name in class_names}
    per_image = []
    for path in pngs:
        r = model.predict(
            source=str(path),
            conf=conf,
            iou=iou,
            imgsz=640,
            device=0,
            verbose=False,
            save=False,
        )[0]
        dist = {name: 0 for name in class_names}
        n = 0
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.tolist()
            clses = [int(c) for c in r.boxes.cls.tolist()]
            confs = r.boxes.conf.tolist()
            keep = class_aware_nms(xyxy, confs, clses, iou)
            for i in keep:
                cid = clses[i]
                name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                dist[name] = dist.get(name, 0) + 1
                by_class[name] = by_class.get(name, 0) + 1
                n += 1
        total += n
        per_image.append({"png": path.name, "fp": n, "by_class": dist})
    return {
        "n_images": len(pngs),
        "total_fp": total,
        "by_class": by_class,
        "per_image": per_image,
    }


def eval_mix_a_weights(save_dir: Path, stage_dir: Path, class_names: list[str]) -> Path:
    out: dict = {}
    blank_dir = stage_dir / "images" / "blank_eval"
    for weight_name in ("best.pt", "last.pt"):
        wpath = save_dir / "weights" / weight_name
        if not wpath.is_file():
            print(f"跳过评估，没有 {wpath}")
            continue
        print(f"\n======== eval {weight_name} ========")
        model = YOLO(str(wpath))
        block: dict = {}
        for name in MIX_A_CHARS:
            yaml_path = stage_dir / f"val_{name}.yaml"
            print(f"  val {name}")
            metrics = model.val(
                data=str(yaml_path),
                split="val",
                imgsz=640,
                conf=TRAIN_CONF,
                iou=TRAIN_IOU,
                device=0,
                plots=False,
                verbose=True,
                project=str(save_dir / "eval"),
                name=f"{weight_name.replace('.pt', '')}_{name}",
                exist_ok=True,
            )
            block[name] = _box_metrics(metrics, class_names)
        print("  blank_eval FP")
        block["blank_eval"] = {
            "conf_0.15": _blank_fp_counts(model, blank_dir, class_names, 0.15, TRAIN_IOU),
            "conf_0.05": _blank_fp_counts(model, blank_dir, class_names, 0.05, TRAIN_IOU),
        }
        out[weight_name] = block
        if weight_name == "best.pt":
            dump_sam1ra_loot_fn(
                wpath,
                stage_dir,
                save_dir / "sam1ra_loot_fn",
                class_names,
            )
    dest = save_dir / "metrics_by_char.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写 {dest}")
    return dest


def train_mix_a(
    freeze: int = 0,
    mosaic: float = 0.0,
    weights: str = "",
    epochs: int | None = None,
    patience: int | None = None,
):
    n_epochs = int(epochs) if epochs is not None else EPOCHS
    n_patience = PATIENCE if patience is None else int(patience)
    mix_root = ROOT / "Aset" / "mix_a"
    class_names = load_class_names(mix_root / "solarwarden_d", mix_root / "solarwarden_d" / "labels", None)
    if not class_names:
        class_names = list(CLASS_NAMES)
    print(f"图片目录: {mix_root}")
    print(f"run: mix_a")
    print(f"类别: {class_names}")
    print(
        f"mix_a: YOLO26n 全量重训 / mosaic={mosaic} / freeze={freeze} → "
        f"epochs={n_epochs} 上限, patience={n_patience} / conf={TRAIN_CONF} iou={TRAIN_IOU}"
    )

    stage_dir = ROOT / "yolo_data" / "mix_a"
    data_yaml, manifest = prepare_mix_a_dataset(stage_dir, class_names)
    meta_text = mix_a_dataset_text(
        manifest["counts"],
        manifest["empty_in_char"],
        manifest["split_sizes"],
        epochs=n_epochs,
        patience=n_patience,
    )
    run_dir = PROJECT_DIR / "mix_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "DATASET.txt").write_text(meta_text, encoding="utf-8")
    (stage_dir / "DATASET.txt").write_text(meta_text, encoding="utf-8")
    print(f"已写元数据 {run_dir / 'DATASET.txt'}")
    print(meta_text)

    save_dir = run_ultralytics_train(
        data_yaml, "mix_a", freeze, mosaic, weights=weights, epochs=n_epochs, patience=n_patience
    )
    if save_dir is None:
        return
    (save_dir / "DATASET.txt").write_text(meta_text, encoding="utf-8")
    shutil.copy2(stage_dir / "splits.json", save_dir / "splits.json")
    eval_mix_a_weights(save_dir, stage_dir, class_names)


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
    suite = (args.suite or "").strip().lower()
    char = (args.char or "").strip().lower()
    if suite == "solarwarden_c":
        train_solarwarden_c_suite()
    elif suite == "mix_a" or char == "mix_a":
        train_mix_a(
            freeze=args.freeze,
            mosaic=args.mosaic,
            weights=args.weights,
            epochs=args.epochs,
            patience=args.patience,
        )
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
