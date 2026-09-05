"""自动标注：写出 X-AnyLabeling 的 json，和 png 放在同一文件夹。

参照 D:/Atrain/Aset/solarwarden/solarwarden_b（同名 .json，rectangle 像素框）。
Xout 只给 X-AnyLabeling「导出」YOLO txt 用，自动标不要往那里写。

用法:
  python auto_label.py --images D:\\Atrain\\Aset\\solarwarden\\深渊 --weights D:\\Atrain\\runs\\solarwarden_b\\weights\\best.pt --conf 0.1
  python auto_label.py --convert-txt
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

ROOT = Path(r"D:/Atrain")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
XAL_VERSION = "0.4.43"
DEFAULT_NAMES = ["boss", "gate", "loot", "mon", "player"]


def png_size(path: Path) -> tuple[int, int]:
    """只读 PNG IHDR，不解码像素。"""
    with path.open("rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"不是 PNG: {path}")
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def iter_images(image_dir: Path, recursive: bool = True) -> list[Path]:
    if not image_dir.is_dir():
        return []
    it = image_dir.rglob("*") if recursive else image_dir.iterdir()
    out = [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXT]
    out.sort(key=lambda p: str(p).lower())
    return out


def load_class_names(*dirs: Path, model_names=None) -> list[str]:
    for d in dirs:
        if d is None:
            continue
        cand = d / "classes.txt"
        if cand.is_file():
            names = [ln.strip() for ln in cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if names:
                return names
    ref = ROOT / "Aset" / "solarwarden" / "solarwarden_b" / "classes.txt"
    if ref.is_file():
        names = [ln.strip() for ln in ref.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names:
            return names
    if isinstance(model_names, dict):
        return [model_names[i] for i in sorted(model_names)]
    return list(DEFAULT_NAMES)


def ensure_classes_txt(folder: Path, names: list[str]) -> None:
    path = folder / "classes.txt"
    if not path.exists():
        path.write_text("\n".join(names) + "\n", encoding="utf-8")


def yolo_xywhn_to_xyxy(cx: float, cy: float, w: float, h: float, iw: int, ih: int) -> list[list[float]]:
    x1 = (cx - w * 0.5) * iw
    y1 = (cy - h * 0.5) * ih
    x2 = (cx + w * 0.5) * iw
    y2 = (cy + h * 0.5) * ih
    return [[float(x1), float(y1)], [float(x2), float(y2)]]


def parse_yolo_txt(text: str, names: list[str], iw: int, ih: int) -> list[dict]:
    shapes = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(parts[i]) for i in range(1, 5))
        label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
        shapes.append(
            {
                "label": label,
                "text": "",
                "points": yolo_xywhn_to_xyxy(cx, cy, bw, bh, iw, ih),
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )
    return shapes


def xal_payload(image_name: str, iw: int, ih: int, shapes: list[dict]) -> dict:
    return {
        "version": XAL_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": ih,
        "imageWidth": iw,
        "text": "",
    }


def write_xal_json(dest: Path, payload: dict) -> None:
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_txt_tree(txt_root: Path, image_root: Path, names: list[str], overwrite: bool = False) -> tuple[int, int, int]:
    """Xout 镜像 txt → 与 png 同目录的 json。返回 (写, 跳过, 缺图)。"""
    wrote = skipped = missing = 0
    txts = sorted(p for p in txt_root.rglob("*.txt") if p.name.lower() != "classes.txt")
    for txt in txts:
        rel = txt.relative_to(txt_root)
        png = image_root / rel.with_suffix(".png")
        if not png.is_file():
            missing += 1
            continue
        dest = png.with_suffix(".json")
        if dest.exists() and not overwrite:
            skipped += 1
            continue
        iw, ih = png_size(png)
        shapes = parse_yolo_txt(txt.read_text(encoding="utf-8"), names, iw, ih)
        write_xal_json(dest, xal_payload(png.name, iw, ih, shapes))
        ensure_classes_txt(png.parent, names)
        wrote += 1
        if wrote % 400 == 0:
            print(f"  … 已转 {wrote}")
    return wrote, skipped, missing


def label_folder(
    model,
    image_dir: Path,
    conf: float,
    iou: float,
    overwrite: bool,
    recursive: bool = True,
) -> tuple[int, int, int, int]:
    images = iter_images(image_dir, recursive=recursive)
    if not images:
        print(f"  跳过（无图）: {image_dir}")
        return 0, 0, 0, 0

    names = load_class_names(image_dir, model_names=getattr(model, "names", None))
    todo: list[Path] = []
    skipped = 0
    for path in images:
        dest = path.with_suffix(".json")
        if dest.exists() and not overwrite:
            skipped += 1
            continue
        todo.append(path)

    print(f"图片: {image_dir}  ({len(images)})")
    print(f"标签: 与 png 同目录 .json")
    print(f"已有 json 跳过 {skipped}，待推理 {len(todo)}")
    if not todo:
        return len(images), skipped, 0, 0

    wrote = 0
    boxes_total = 0
    empty: list[str] = []
    for path in todo:
        # 只传单张路径；列表会被 Ultralytics 整批读进内存
        r = model.predict(
            source=str(path),
            conf=conf,
            iou=iou,
            imgsz=640,
            device=0,
            verbose=False,
            save=False,
        )[0]
        iw, ih = png_size(path)
        shapes: list[dict] = []
        if r.boxes is not None and len(r.boxes):
            for cls_id, xywhn in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                cid = int(cls_id)
                label = names[cid] if 0 <= cid < len(names) else str(cid)
                cx, cy, bw, bh = (float(v) for v in xywhn)
                shapes.append(
                    {
                        "label": label,
                        "text": "",
                        "points": yolo_xywhn_to_xyxy(cx, cy, bw, bh, iw, ih),
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                )
        write_xal_json(path.with_suffix(".json"), xal_payload(path.name, iw, ih, shapes))
        ensure_classes_txt(path.parent, names)
        wrote += 1
        boxes_total += len(shapes)
        if not shapes:
            empty.append(path.relative_to(image_dir).as_posix())
        if wrote % 200 == 0:
            print(f"  … {wrote}/{len(todo)}  框累计 {boxes_total}")

    print(f"已写 {wrote} 个 json，共 {boxes_total} 框")
    print(f"无框 {len(empty)}")
    if empty[:20]:
        print("  " + ", ".join(empty[:20]) + (" …" if len(empty) > 20 else ""))
    print("X-AnyLabeling 直接打开", image_dir)
    print("类别:", ", ".join(names))
    return len(images), skipped, wrote, boxes_total


def parse_args():
    p = argparse.ArgumentParser(description="自动标注 → 与 png 同目录的 X-AnyLabeling json")
    p.add_argument("--char", default="", help="Aset/<char>；不填则用 --images")
    p.add_argument("--images", default="", help="图片目录（递归）")
    p.add_argument("--weights", default="", help="默认 solarwarden_b/best.pt")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--overwrite", action="store_true", help="覆盖已有 json")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument(
        "--convert-txt",
        action="store_true",
        help="把 Xout/solarwarden 的 yolo txt 转成 Aset 同目录 json，然后删掉误放的 txt",
    )
    return p.parse_args()


def main():
    args = parse_args()
    names = load_class_names()
    if args.convert_txt:
        txt_root = ROOT / "Xout" / "solarwarden"
        img_root = ROOT / "Aset" / "solarwarden"
        print(f"txt → json: {txt_root} → {img_root}")
        wrote, skipped, missing = convert_txt_tree(txt_root, img_root, names, overwrite=args.overwrite)
        print(f"已写 json {wrote}，已有跳过 {skipped}，缺 png {missing}")
        return

    image_dir = Path(args.images) if args.images.strip() else ROOT / "Aset" / args.char.strip()
    weights = (
        Path(args.weights)
        if args.weights.strip()
        else ROOT / "runs" / "solarwarden_b" / "weights" / "best.pt"
    )
    if not image_dir.is_dir():
        raise FileNotFoundError(f"没有图片目录: {image_dir}")
    if not weights.is_file():
        raise FileNotFoundError(f"没有权重: {weights}")

    from ultralytics import YOLO

    print(f"权重: {weights}")
    print(f"conf={args.conf} iou={args.iou}  overwrite={args.overwrite}")
    model = YOLO(str(weights))
    label_folder(
        model,
        image_dir,
        conf=args.conf,
        iou=args.iou,
        overwrite=args.overwrite,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
