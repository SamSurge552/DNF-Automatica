"""给只有 png、没有检测框的 frames.jsonl 补 YOLO 特征。保留原 t_ns。

  python backfill_session_yolo.py
  python backfill_session_yolo.py --session recordings/深渊：最终调律者/20260903_190326_SolarWarden
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from yolo_engine import DEFAULT_YOLO_WEIGHTS, YoloEngine

ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "recordings"
IMAGES = ROOT / "images"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def needs_yolo(rows: list[dict]) -> bool:
    if not rows:
        return False
    sample = rows[: min(8, len(rows))]
    return not any("mon" in r or "player_xy" in r or "mon_xy" in r for r in sample)


def png_dir_of(session: Path, meta: dict) -> Path | None:
    raw = str((meta or {}).get("png_dir") or "").strip()
    if raw:
        p = Path(raw)
        if p.is_dir():
            return p
    stamp = str((meta or {}).get("started_at") or "").strip()
    if not stamp:
        name = session.name
        stamp = "_".join(name.split("_")[:2]) if "_" in name else name
    p = IMAGES / stamp
    return p if p.is_dir() else None


def load_bgr(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def backfill_session(session: Path, engine: YoloEngine, force: bool = False) -> tuple[int, str]:
    frames_path = session / "frames.jsonl"
    rows = load_jsonl(frames_path)
    if not rows:
        return 0, "无 frames.jsonl"
    if not force and not needs_yolo(rows):
        return 0, "已有检测字段，跳过"
    meta = {}
    mp = session / "meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    png_dir = png_dir_of(session, meta)
    if png_dir is None:
        return 0, "找不到 PNG 目录"
    out = []
    n_ok = 0
    for fr in rows:
        png_name = str(fr.get("png") or "")
        img = load_bgr(png_dir / png_name) if png_name else None
        feats = engine.infer_features(img) if img is not None else None
        if not feats:
            feats = YoloEngine.empty_features()
        else:
            n_ok += 1
        row = dict(fr)
        row.update({k: v for k, v in feats.items() if k != "_result"})
        out.append(row)
    tmp = frames_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(frames_path)
    meta["format"] = "keys.jsonl + frames.jsonl (t_ns absolute)"
    meta["yolo_weights"] = str(engine.weights)
    meta["yolo_classes"] = [str(v) for v in (engine.names or {}).values()]
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return n_ok, f"补了 {n_ok}/{len(rows)} 帧 -> {png_dir}"


def iter_sessions(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("frames.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="", help="某一段目录；空则扫描 recordings 里缺检测的段")
    ap.add_argument("--weights", default=str(DEFAULT_YOLO_WEIGHTS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    engine = YoloEngine(args.weights)
    if not engine.apply(log=print):
        raise SystemExit("YOLO 加载失败")
    if args.session.strip():
        sessions = [Path(args.session)]
    else:
        sessions = [p for p in iter_sessions(RECORDINGS) if args.force or needs_yolo(load_jsonl(p / "frames.jsonl"))]
    if not sessions:
        print("没有需要补的段")
        return
    for ses in sessions:
        n, msg = backfill_session(ses, engine, force=args.force)
        print(f"{ses}: {msg}")


if __name__ == "__main__":
    main()
