"""在 Aset/blank 上数误检，对比过图 YOLO（conf=0.1）。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

BLANK = Path(__file__).resolve().parent / "images" / "Atrain" / "Aset" / "blank"
MODELS = {
    "solarwarden_b": Path(__file__).resolve().parent / "images" / "Atrain" / "runs" / "solarwarden_b" / "weights" / "best.pt",
}


def eval_one(name: str, weights: Path, conf: float, iou: float) -> dict:
    if not weights.is_file():
        return {"name": name, "error": f"missing {weights}"}
    pngs = sorted(BLANK.glob("*.png"))
    model = YOLO(str(weights))
    per_cls: Counter = Counter()
    imgs_hit = 0
    boxes_total = 0
    for png in pngs:
        r = model.predict(str(png), conf=conf, iou=iou, imgsz=640, device=0, verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        boxes_total += n
        if n:
            imgs_hit += 1
        if r.boxes is not None and r.boxes.cls is not None:
            names = r.names
            for c in r.boxes.cls.tolist():
                per_cls[names.get(int(c), str(int(c)))] += 1
    n = max(len(pngs), 1)
    return {
        "name": name,
        "weights": str(weights),
        "n_blank": len(pngs),
        "imgs_with_any_box": imgs_hit,
        "fp_img_rate": round(imgs_hit / n, 4),
        "boxes_total": boxes_total,
        "boxes_per_img": round(boxes_total / n, 3),
        "by_class": dict(per_cls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()
    rows = [eval_one(n, p, args.conf, args.iou) for n, p in MODELS.items()]
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
