"""
按 TRAIN_ALIGN.md 把 recordings jsonl **组集**为过图训练集（dataset_export/）。

这不是写盘。不改原始 jsonl。组集时才做：
  - 键盘去 auto-repeat，重建 hold；meta.keys_held_at_start 初始化已按住
  - 相邻帧特征完全一致则丢弃（仅导出；录制 jsonl 原样保留给 FSM）
  - 每帧真实 dt 上的 held_frac（已去掉 count）
  - 丢掉无左端点的第一帧
  - 过短段默认不进训练集
坐标原样拷贝（player 可 null，*_xy 为检测器坐标）。相对 / 填 player / 原点翻转不在导出里做。

用法:
  python export_dataset.py
  python export_dataset.py --min-duration 8 --min-frames 20
  python export_dataset.py --include-short
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDINGS = ROOT / "recordings"
DEFAULT_OUT = ROOT / "dataset_export"

# 过短段：约十几帧 / 数秒的误切不要进训练（可用 --include-short 关掉）
DEFAULT_MIN_DURATION_S = 8.0
DEFAULT_MIN_FRAMES = 20


def normalize_key(raw: Any) -> str:
    s = str(raw).strip()
    if s.lower().startswith("key."):
        s = s[4:]
    return s.lower() if s.isascii() else s


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rebuild_holds(
    events: list[dict],
    last_frame_t: int,
    held_at_start: Iterable[str] | None = None,
    started_t_ns: int | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """
    扫 keys.jsonl：第一次 press 且当前未按住 → 开 hold；已按时再 press 忽略；release 关。
    开录已按住的键用 held_at_start 在 started_t_ns 初始化，避免要等到第一条 auto-repeat 才开始算 held_frac。
    段末仍按住则延续到最后一帧 t_ns。
    返回 {key: [(start_ns, end_ns), ...]}，区间为半开半闭 [start, end)。
    """
    events = sorted(events, key=lambda e: int(e["t_ns"]))
    seed_t = int(started_t_ns) if started_t_ns is not None else None
    held: dict[str, int] = {}
    if held_at_start and seed_t is not None:
        for raw in held_at_start:
            key = normalize_key(raw)
            if key:
                held[key] = seed_t
    holds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ev in events:
        typ = ev.get("type")
        key = normalize_key(ev.get("key", ""))
        t = int(ev["t_ns"])
        if not key:
            continue
        if typ == "press":
            if key not in held:
                held[key] = t
        elif typ == "release":
            start = held.pop(key, None)
            if start is not None and t > start:
                holds[key].append((start, t))
    end = int(last_frame_t)
    for key, start in list(held.items()):
        if end > start:
            holds[key].append((start, end))
    return dict(holds)


def _xy_tuples(fr: dict, field: str) -> tuple:
    arr = fr.get(field) or []
    if not isinstance(arr, list):
        return ()
    out = []
    for p in arr:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((round(float(p[0]), 1), round(float(p[1]), 1)))
    return tuple(out)


def raw_feature_tuple(fr: dict) -> tuple:
    xy = fr.get("player_xy")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        xy_t: tuple | None = (float(xy[0]), float(xy[1]))
    else:
        xy_t = None
    return (
        xy_t,
        int(fr.get("mon") or 0),
        int(fr.get("loot") or 0),
        int(fr.get("gate") or 0),
        int(fr.get("boss") or 0),
        _xy_tuples(fr, "mon_xy"),
        _xy_tuples(fr, "loot_xy"),
        _xy_tuples(fr, "gate_xy"),
        _xy_tuples(fr, "boss_xy"),
    )


def drop_duplicate_feature_frames(frames: list[dict]) -> tuple[list[dict], int]:
    """仅导出用。相邻帧特征完全一致则丢掉后者。录制端不得调用。"""
    if not frames:
        return [], 0
    kept = [frames[0]]
    skipped = 0
    prev = raw_feature_tuple(frames[0])
    for fr in frames[1:]:
        cur = raw_feature_tuple(fr)
        if cur == prev:
            skipped += 1
            continue
        kept.append(fr)
        prev = cur
    return kept, skipped


def interval_held_frac(
    holds: dict[str, list[tuple[int, int]]],
    t0: int,
    t1: int,
    keys: Iterable[str],
) -> dict[str, float]:
    """I = (t0, t1]，dt = t1-t0。held_frac = 交叠时长 / dt。"""
    dt = t1 - t0
    frac: dict[str, float] = {}
    if dt <= 0:
        return frac
    for key in keys:
        overlap = 0
        for start, end in holds.get(key) or []:
            lo = max(start, t0)
            hi = min(end, t1)
            if hi > lo:
                overlap += hi - lo
        if overlap:
            frac[key] = min(1.0, overlap / dt)
    return frac


def forward_fill_player(frames: list[dict]) -> list[dict | None]:
    """
    返回与 frames 等长的列表。
    每项为 {"xy": [x,y], "hold": 0|1}，开场连续 null（还没见过人）为 None。
    """
    last = None
    out: list[dict | None] = []
    for fr in frames:
        xy = fr.get("player_xy")
        if xy is not None:
            last = [float(xy[0]), float(xy[1])]
            out.append({"xy": last, "hold": 0})
        elif last is not None:
            out.append({"xy": list(last), "hold": 1})
        else:
            out.append(None)
    return out


def session_duration_s(frames: list[dict]) -> float:
    if len(frames) < 2:
        return 0.0
    return (int(frames[-1]["t_ns"]) - int(frames[0]["t_ns"])) / 1e9


def export_session(
    session_dir: Path,
    min_duration_s: float,
    min_frames: int,
    include_short: bool,
) -> dict[str, Any]:
    meta_path = session_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    frames_raw = sorted(load_jsonl(session_dir / "frames.jsonl"), key=lambda r: int(r["t_ns"]))
    keys_raw = load_jsonl(session_dir / "keys.jsonl")
    dungeon = meta.get("dungeon") or session_dir.parent.name
    character = meta.get("character") or ""
    started_at = meta.get("started_at") or session_dir.name
    held_at_start = [normalize_key(k) for k in (meta.get("keys_held_at_start") or [])]
    started_t_ns = meta.get("started_t_ns")
    if started_t_ns is not None:
        started_t_ns = int(started_t_ns)
    elif frames_raw:
        started_t_ns = int(frames_raw[0]["t_ns"])

    frames, dup_skipped = drop_duplicate_feature_frames(frames_raw)

    info: dict[str, Any] = {
        "session_dir": str(session_dir),
        "dungeon": dungeon,
        "character": character,
        "started_at": started_at,
        "raw_frames": len(frames_raw),
        "raw_keys": len(keys_raw),
        "dup_skipped": dup_skipped,
        "skipped": None,
        "samples": 0,
        "player_null_raw": sum(1 for f in frames_raw if f.get("player_xy") is None),
        "player_hold_samples": 0,
        "key_press_raw": sum(1 for e in keys_raw if e.get("type") == "press"),
        "key_repeat_ignored": 0,
        "keys_held_at_start": held_at_start,
        "keys_seen": [],
        "duration_s": round(session_duration_s(frames_raw), 3),
    }

    if not frames:
        info["skipped"] = "no_frames"
        return info

    duration = session_duration_s(frames_raw)
    if not include_short and (duration < min_duration_s or len(frames) < min_frames):
        info["skipped"] = f"too_short duration={duration:.2f}s frames={len(frames)} dup_skipped={dup_skipped}"
        return info

    last_t = int(frames[-1]["t_ns"])
    held: set[str] = set(held_at_start)
    ignored = 0
    for ev in sorted(keys_raw, key=lambda e: int(e["t_ns"])):
        if ev.get("seed"):
            continue
        k = normalize_key(ev.get("key", ""))
        if ev.get("type") == "press":
            if k in held:
                ignored += 1
            else:
                held.add(k)
        elif ev.get("type") == "release":
            held.discard(k)
    info["key_repeat_ignored"] = ignored

    holds = rebuild_holds(
        keys_raw,
        last_t,
        held_at_start=held_at_start,
        started_t_ns=started_t_ns,
    )
    vocab = sorted(holds.keys())
    info["keys_seen"] = vocab

    samples: list[dict] = []
    for i in range(1, len(frames)):
        t0 = int(frames[i - 1]["t_ns"])
        t1 = int(frames[i]["t_ns"])
        dt = t1 - t0
        if dt <= 0:
            continue
        frac = interval_held_frac(holds, t0, t1, vocab)
        fr = frames[i]
        samples.append(
            {
                "dungeon": dungeon,
                "character": character,
                "session": started_at,
                "t_ns": t1,
                "dt_ns": dt,
                "dt_s": round(dt / 1e9, 6),
                "player_xy": fr.get("player_xy"),
                "mon_xy": list(fr.get("mon_xy") or []),
                "loot_xy": list(fr.get("loot_xy") or []),
                "gate_xy": list(fr.get("gate_xy") or []),
                "boss_xy": list(fr.get("boss_xy") or []),
                "mon": int(fr.get("mon") or 0),
                "loot": int(fr.get("loot") or 0),
                "gate": int(fr.get("gate") or 0),
                "boss": int(fr.get("boss") or 0),
                "infer_ms": fr.get("infer_ms"),
                "held_frac": frac,
            }
        )

    info["samples"] = len(samples)
    info["player_hold_samples"] = 0
    info["rows"] = samples
    return info


def iter_sessions(recordings: Path) -> list[Path]:
    out: list[Path] = []
    if not recordings.is_dir():
        return out
    for dungeon_dir in sorted(p for p in recordings.iterdir() if p.is_dir()):
        for sess in sorted(p for p in dungeon_dir.iterdir() if p.is_dir()):
            if (sess / "frames.jsonl").is_file():
                out.append(sess)
    return out


def summarize_keys(rows: list[dict], vocab: list[str]) -> dict[str, Any]:
    per: dict[str, dict[str, float]] = {}
    for k in vocab:
        per[k] = {
            "frames_held_gt0": 0,
            "sum_held_frac": 0.0,
        }
    for row in rows:
        f = row.get("held_frac") or {}
        for k in vocab:
            if f.get(k, 0):
                per[k]["frames_held_gt0"] += 1
                per[k]["sum_held_frac"] += float(f[k])
    n = max(len(rows), 1)
    for k in vocab:
        per[k]["mean_held_frac"] = round(per[k]["sum_held_frac"] / n, 4)
        per[k]["sum_held_frac"] = round(per[k]["sum_held_frac"], 3)
    return per


def write_export(infos: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    vocab: set[str] = set()
    kept: list[dict] = []
    skipped: list[dict] = []
    for info in infos:
        vocab.update(info.get("keys_seen") or [])
        if info.get("skipped"):
            skipped.append({k: v for k, v in info.items() if k != "rows"})
            continue
        kept.append({k: v for k, v in info.items() if k != "rows"})

    key_order = sorted(vocab)
    n_rows = 0
    all_rows: list[dict] = []
    with samples_path.open("w", encoding="utf-8") as f:
        for info in infos:
            for row in info.get("rows") or []:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_rows.append(row)
                n_rows += 1

    manifest = {
        "align": "TRAIN_ALIGN.md",
        "y": "per-key held_frac; missing key = 0",
        "player_fill": "not applied; player_xy copied from recordings (null kept)",
        "xy": "copied from recordings; relative/origin is replay-only for now",
        "first_frame": "dropped (no left endpoint / no fabricated dt)",
        "dup_frames": "export-only: drop consecutive identical features; recordings keep them for FSM",
        "stamp_bias": "use disk t_ns as-is (dxcam capture wall median ~8ms; old Grab ~41ms)",
        "keys": key_order,
        "n_samples": n_rows,
        "n_sessions_kept": len(kept),
        "n_sessions_skipped": len(skipped),
        "dup_skipped": sum(int(i.get("dup_skipped") or 0) for i in infos),
        "sessions_kept": kept,
        "sessions_skipped": skipped,
        "key_stats": summarize_keys(all_rows, key_order),
        "player_null_rate": round(
            (sum(1 for r in all_rows if r.get("player_xy") is None) / n_rows) if n_rows else 0.0,
            4,
        ),
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return man_path


def self_test() -> None:
    t0 = 1_000_000_000
    frames = [
        {"t_ns": t0, "player_xy": None, "mon": 0, "loot": 0, "gate": 0, "boss": 0},
        {"t_ns": t0 + 200_000_000, "player_xy": [10.0, 20.0], "mon": 1, "loot": 0, "gate": 0, "boss": 0},
        {"t_ns": t0 + 400_000_000, "player_xy": None, "mon": 0, "loot": 2, "gate": 0, "boss": 0},
        {"t_ns": t0 + 600_000_000, "player_xy": [12.0, 21.0], "mon": 0, "loot": 0, "gate": 1, "boss": 0},
    ]
    events = [
        {"t_ns": t0 + 50_000_000, "type": "press", "key": "right"},
        {"t_ns": t0 + 80_000_000, "type": "press", "key": "right"},
        {"t_ns": t0 + 110_000_000, "type": "press", "key": "right"},
        {"t_ns": t0 + 450_000_000, "type": "release", "key": "right"},
        {"t_ns": t0 + 410_000_000, "type": "press", "key": "x"},
        {"t_ns": t0 + 430_000_000, "type": "release", "key": "x"},
        {"t_ns": t0 + 460_000_000, "type": "press", "key": "x"},
        {"t_ns": t0 + 480_000_000, "type": "release", "key": "x"},
    ]
    filled = forward_fill_player(frames)
    assert filled[0] is None
    assert filled[1]["hold"] == 0 and filled[1]["xy"] == [10.0, 20.0]
    assert filled[2]["hold"] == 1 and filled[2]["xy"] == [10.0, 20.0]
    assert filled[3]["hold"] == 0 and filled[3]["xy"] == [12.0, 21.0]

    holds = rebuild_holds(events, int(frames[-1]["t_ns"]))
    assert len(holds["right"]) == 1
    f = interval_held_frac(holds, t0 + 200_000_000, t0 + 400_000_000, ["right", "x"])
    assert abs(f["right"] - 1.0) < 1e-9
    assert "x" not in f
    f = interval_held_frac(holds, t0 + 400_000_000, t0 + 600_000_000, ["right", "x"])
    assert 0 < f["right"] < 1
    assert 0 < f["x"] < 1

    t_s = 10_000_000_000
    ev_repeat = [
        {"t_ns": t_s + 80_000_000, "type": "press", "key": "right"},
        {"t_ns": t_s + 110_000_000, "type": "press", "key": "right"},
        {"t_ns": t_s + 400_000_000, "type": "release", "key": "right"},
    ]
    holds2 = rebuild_holds(
        ev_repeat,
        t_s + 600_000_000,
        held_at_start=["right"],
        started_t_ns=t_s,
    )
    assert holds2["right"] == [(t_s, t_s + 400_000_000)]
    f = interval_held_frac(holds2, t_s, t_s + 200_000_000, ["right"])
    assert abs(f["right"] - 1.0) < 1e-9

    d0 = {"t_ns": 1, "player_xy": [1.0, 2.0], "mon": 1, "loot": 0, "gate": 0, "boss": 0}
    d1 = {"t_ns": 2, "player_xy": [1.0, 2.0], "mon": 1, "loot": 0, "gate": 0, "boss": 0}
    d2 = {"t_ns": 3, "player_xy": [3.0, 2.0], "mon": 1, "loot": 0, "gate": 0, "boss": 0}
    kept, n_dup = drop_duplicate_feature_frames([d0, d1, d2])
    assert n_dup == 1 and [k["t_ns"] for k in kept] == [1, 3]
    print("self_test ok")


def main():
    ap = argparse.ArgumentParser(description="Export recordings → held_frac dataset")
    ap.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    ap.add_argument("--include-short", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    sessions = iter_sessions(args.recordings)
    if not sessions:
        print(f"未找到录制段: {args.recordings}")
        return
    infos = [
        export_session(s, args.min_duration, args.min_frames, args.include_short)
        for s in sessions
    ]
    man = write_export(infos, args.out)
    kept = [i for i in infos if not i.get("skipped")]
    skipped = [i for i in infos if i.get("skipped")]
    n = sum(i.get("samples", 0) for i in kept)
    dups = sum(int(i.get("dup_skipped") or 0) for i in infos)
    print(f"段: 保留 {len(kept)} / 跳过 {len(skipped)} / 样本 {n} / dup_skipped {dups}")
    for i in skipped:
        print(f"  skip {i['started_at']}: {i['skipped']}")
    for i in kept:
        print(
            f"  keep {i['started_at']}: samples={i['samples']} "
            f"dur={i['duration_s']}s null_player={i['player_null_raw']} "
            f"hold={i['player_hold_samples']} dup_skipped={i.get('dup_skipped', 0)} "
            f"repeat_ignored={i['key_repeat_ignored']} "
            f"keys={i['keys_seen']}"
        )
    print(f"写出 {args.out / 'samples.jsonl'} 与 {man}")


if __name__ == "__main__":
    main()
