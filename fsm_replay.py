"""回放 recordings/ 与 FSM_TEST/ 的 frames.jsonl：站在模型视角看结构化标签。

图标对应 YOLO 写入 jsonl 的字段。可选叠 PNG 对比（与检测同一套截屏像素）。
旧段只有 player_xy、其它类是计数；新段还有 mon_xy/loot_xy/gate_xy/boss_xy。

用法:
  python fsm_replay.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

from relative_xy import as_xy_list, player_abs, views_for_frames
from fsm_core import (
    FsmFlag,
    FsmParams,
    FsmState,
    run_track,
    MON_CORR_MAX,
    mon_corr_from_dict,
    mon_off_from_corr,
    DEFAULT_TOWN_S,
    send_keys_label,
    json_ms,
    json_s_from_frames,
    parse_hold_ms,
    HOLD_MS_KEY,
)
from fsm_execute import mash_n_for_slot
from window_geom import apply as apply_window_geom
from window_geom import remember as remember_window_geom
from skill_feature_extract import (
    casts_from_tracks,
    count_casts,
    detect_cd_resets,
    feature_path,
    fight_plan_from_features,
    format_report,
    dist_table_from_features,
    hotbar_from_binds,
    bind_hotkey,
    legacy_feature_path,
    load_features,
    maybe_adopt_legacy,
    merge_casts,
    pack_dist_from_player,
    prev_path,
    rollback_features,
    save_features,
    skill_range_of,
    clear_features,
    combo_slots_of,
    set_combo_slots,
    set_multi_slots,
    multi_n_of,
    apply_f,
    has_map_reset,
    set_map_reset,
    MAP_RESET,
    DEFAULT_F,
    DEFAULT_MULTI,
    parse_multi_n,
    cast_is_fake,
)

ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "recordings"
FSM_TEST_DIR = ROOT / "FSM_TEST"
IMAGES = ROOT / "images"
ICON_DIR = ROOT / "fsm_icons"
SETTINGS_PATH = ROOT / "_fsm_replay_ui.json"
BINDS_DIR = ROOT / "skill_binds"
CLASSES = ("player", "mon", "loot", "gate", "boss")
CLASS_CN = {
    "player": "玩家",
    "mon": "怪物",
    "loot": "掉落",
    "gate": "前进/门",
    "boss": "BOSS",
}
DIRS = ("left", "right", "up", "down")
RUN_PRESS_GT = 3
LOOT_HOLD_DEFAULT = 250
DEFAULT_PM = 10
DEFAULT_PC = 5
DEFAULT_PT = 3
HOLD_FRAMES_KEY = HOLD_MS_KEY
COMBO_KEY = "combo"
MULTI_KEY = "multi"
MULTI_N_KEY = "multi_n"
MASH_KEY = "mash"
SKILL_HOLD_DEFAULT = 250
PLAYFIELD_W = 800
PLAYFIELD_H = 450
FLOW_H = 78
COUNT_PANEL_W = 400
STATUS_VALUE_H = 52
HELP_PARAMS = (
    "M/L/G：连续同值才改判定。BOSS 暂与 MON 共用 M。\n"
    "回城秒：连续无地下城关键词达该秒数即回城（核心直接用秒，不换帧）。回放 jsonl 无 OCR 时默认已进图。\n"
    "S%：开打出现时 MON 群相似。\n"
    "F%：杀 MON 效率低于该百分比记假释放（不进序列 / 范围 / CD）。\n"
    "PM：掉落位移超过该像素算在动。连续 PT 帧不动才判定停下。数量 > PC 一键拾取，否则挨个捡。\n"
    "卡住：同一流程状态连续 X 秒则改为卡住，上下左右各 HOLD Y 毫秒（不是前进）。\n"
    "GX/GY：相对当前门，距离大于该像素才继续接近。接近与捡物依次走：点按方向 → 等移动间隔 TH 毫秒 → 按住（TH 默认等于连按间隔）。AX/AY：两轴停后、或前进时门消失，按过门方向走的毫秒。XXX：快捷栏全 CD 时按住普攻 X 的毫秒。\n"
    "点按 ms：发键层技能/左Alt 按下保持的毫秒，组合键步骤间隔相同。与主面板共用 json。\n"
    "连按 COUNT / 间隔 ms：执行层参数（不进核心）。红字由回放按槽是否勾连按 + COUNT 拼出来。\n"
    "MON/BOSS 补正：上下左右滑块，角色要往哪边站就把 YOLO 的 MON/BOSS 坐标往哪边挪（上=Y 减，左=X 减）。只影响 FSM，不改 jsonl。\n"
    "叠图：把该帧 PNG 铺到坐标网上（半透明、最上层）。对齐=截屏像素与 jsonl 同一套；相对坐标时图平移使 player 落在盘面中心。\n"
    "改完立刻写入共用 json，回放当场重算；FSM测试点开始（或测试中再改）会读这份，不必另同步。"
)
HELP_SKILLS = (
    "只列出已勾选快捷栏的技能（space 即使未勾快捷栏也保留）。普攻 X 无 CD，不在这张表里。\n"
    "连续释放：该槽 CD 按 270 秒。\n"
    "多次释放：勾选后填 MULTI（默认 2），FSM 拆成多份独立 CD。\n"
    "连按：执行层把该技能展开成 COUNT 次点按（COUNT/间隔在参数区）。与连续释放/多次释放不是一回事。\n"
    "连续释放 / 多次释放都不触发提取时的 CD 重置检测。\n"
    "勾选或改次数后点「保存技能表」写入键位表。键位工具改完后点「重新读取」。"
)
HELP_EXTRACT = (
    "按下快捷栏技能（含 SPACE）即提取，不要求开打。\n"
    "持续结束后杀 MON 数 > E → 群，否则为单。\n"
    "杀 MON 效率 < F% → 假释放（不进序列 / 范围 / CD）。\n"
    "怪物分布按按下那一帧现算，不沿用 FSM 当时的状态。\n"
    "地图【CD重置】：提取发现短于 CD 的间隔后写入；运行时击败 BOSS +1 才清 CD。"
)


def _safe_folder_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in (name or "").strip())
    return out or "unknown"


def character_from_session(session: Path, meta: dict | None = None) -> str:
    ch = str((meta or {}).get("character") or "").strip()
    if ch:
        return ch
    parts = Path(session).name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return ""


def png_dir_candidates(session: Path, meta: dict | None = None) -> list[Path]:
    """采集图按角色夹；jsonl 在地下城段目录。靠文件名对齐，不靠地下城路径。"""
    meta = meta or {}
    session = Path(session)
    seen: set[str] = set()
    out: list[Path] = []

    def add(p: Path | None) -> None:
        if p is None:
            return
        p = Path(p)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        if p.is_dir():
            out.append(p)

    add(session / "png")
    raw = str(meta.get("png_dir") or "").strip()
    if raw:
        add(Path(raw))
    stamp = str(meta.get("started_at") or "").strip()
    if not stamp:
        name = session.name
        stamp = "_".join(name.split("_")[:2]) if "_" in name else name
    if stamp:
        add(IMAGES / stamp)
    char = character_from_session(session, meta)
    if char:
        add(IMAGES / _safe_folder_name(char))
    return out


def png_dir_of(session: Path, meta: dict) -> Path | None:
    found = png_dir_candidates(session, meta)
    return found[0] if found else None


def png_file_of(session: Path, meta: dict | None, png_name: str) -> Path | None:
    name = str(png_name or "").strip()
    if not name:
        return None
    for folder in png_dir_candidates(session, meta):
        p = folder / name
        if p.is_file():
            return p
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def list_sessions(*roots: Path) -> list[Path]:
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("frames.jsonl"):
            found.append(p.parent)
    return sorted(found, key=lambda x: x.name, reverse=True)


def session_kind(session: Path) -> str:
    """采集 recordings → rec；FSM测试 FSM_TEST → fsm。"""
    try:
        fsm = FSM_TEST_DIR.resolve()
        cur = session.resolve()
        if cur == fsm or fsm in cur.parents:
            return "fsm"
    except OSError:
        pass
    return "rec"


def class_xys(fr: dict, name: str) -> list[list[float]]:
    if name == "player":
        return as_xy_list(fr.get("player_xy"))
    return as_xy_list(fr.get(f"{name}_xy"))


def feature_sig(fr: dict) -> tuple:
    def _round_pts(pts: list[list[float]]) -> tuple:
        return tuple((round(p[0], 1), round(p[1], 1)) for p in pts)

    return (
        _round_pts(class_xys(fr, "player")),
        int(fr.get("mon") or 0),
        int(fr.get("loot") or 0),
        int(fr.get("gate") or 0),
        int(fr.get("boss") or 0),
        _round_pts(class_xys(fr, "mon")),
        _round_pts(class_xys(fr, "loot")),
        _round_pts(class_xys(fr, "gate")),
        _round_pts(class_xys(fr, "boss")),
    )


def compute_draft_track(
    frames: list[dict],
    m: int,
    l: int,
    g: int,
    x_s: float = 1.5,
    gx: int = 50,
    gy: int = 10,
    ax_ms: int = 250,
    ay_ms: int = 250,
    y_ms: int = 250,
    s: int = 20,
    pm: int = 10,
    pc: int = 5,
    pt: int = 3,
    xxx_ms: int = 1000,
    th_ms: int = 50,
    fight_plan=(),
    hotbar=(),
    dist_table=(),
    map_reset: bool = False,
    mon_off_x: int = 0,
    mon_off_y: int = 0,
    tn_s: float = DEFAULT_TOWN_S,
) -> list[dict]:
    """回放宿主入口；t_ns 用 jsonl 原值。判定在 fsm_core.step。"""
    return run_track(
        frames,
        FsmParams(
            m=m,
            l=l,
            g=g,
            x_s=float(x_s),
            gx=gx,
            gy=gy,
            ax_ms=ax_ms,
            ay_ms=ay_ms,
            y_ms=y_ms,
            s=s,
            pm=pm,
            pc=pc,
            pt=pt,
            xxx_ms=xxx_ms,
            th_ms=th_ms,
            fight_plan=tuple(fight_plan or ()),
            hotbar=tuple(hotbar or ()),
            dist_table=tuple(dist_table or ()),
            map_reset=bool(map_reset),
            mon_off_x=int(mon_off_x),
            mon_off_y=int(mon_off_y),
            tn_s=float(tn_s),
        ),
    )


def rebuild_held_at(events: list[dict], t_ns: int, held_at_start: list[str] | None = None) -> list[str]:
    held: set[str] = {str(k).lower() for k in (held_at_start or []) if k}
    for ev in events:
        t = int(ev.get("t_ns") or 0)
        if t > t_ns:
            break
        key = str(ev.get("key") or "").lower()
        typ = ev.get("type")
        if not key:
            continue
        if typ in ("press", "seed") or ev.get("seed"):
            held.add(key)
        elif typ == "release":
            held.discard(key)
    return sorted(held)


def events_in_range(events: list[dict], t0: int, t1: int) -> list[dict]:
    out = []
    for ev in events:
        t = int(ev.get("t_ns") or 0)
        if t <= t0:
            continue
        if t > t1:
            break
        if ev.get("seed"):
            continue
        out.append(ev)
    return out


def run_burst_active(edges: list[dict], gt: int = RUN_PRESS_GT) -> bool:
    """本帧沿任一方向 press 次数 > gt → 跑。方向先不管。"""
    counts = {d: 0 for d in DIRS}
    for ev in edges:
        if ev.get("type") != "press":
            continue
        key = str(ev.get("key") or "").lower()
        if key in counts:
            counts[key] += 1
            if counts[key] > gt:
                return True
    return False


def move_active(edges: list[dict], held: list[str] | None) -> bool:
    """移动：任一方向正按住，或本帧沿有方向键 press。"""
    if any(str(k).lower() in DIRS for k in (held or [])):
        return True
    return any(
        ev.get("type") == "press" and str(ev.get("key") or "").lower() in DIRS
        for ev in edges
    )


def _is_alt_key(key: str) -> bool:
    k = str(key or "").lower()
    return k == "alt" or k.startswith("alt_")


def basic_attack_active(edges: list[dict], held: list[str] | None) -> bool:
    """普攻：x 正按住，或本帧沿有 x 的 press（同帧按下又松开也能认）。"""
    if any(str(k).lower() == "x" for k in (held or [])):
        return True
    return any(
        ev.get("type") == "press" and str(ev.get("key") or "").lower() == "x"
        for ev in edges
    )


def loot_hold_track(frames: list[dict], keys: list[dict], dur_ms: int) -> list[bool]:
    """本帧沿出现 alt press 后，持续 dur_ms 毫秒判定为捡物（含按下那一帧）。再按 alt 会重计。"""
    n = len(frames)
    dur_ms = max(1, int(dur_ms))
    if n == 0:
        return []
    last_t = None
    out = [False] * n
    for i, fr in enumerate(frames):
        t = int(fr["t_ns"])
        t_prev = int(frames[i - 1]["t_ns"]) if i > 0 else t - 1
        edges = events_in_range(keys, t_prev, t)
        if any(ev.get("type") == "press" and _is_alt_key(str(ev.get("key") or "")) for ev in edges):
            last_t = t
        out[i] = last_t is not None and (t - last_t) / 1e6 < dur_ms
    return out


def bind_file(character: str) -> Path:
    safe = "".join(ch for ch in character.strip() if ch not in r'\/:*?"<>|') or "character"
    return BINDS_DIR / f"{safe}.json"


def character_of_session(session: Path, meta: dict) -> str:
    ch = str((meta or {}).get("character") or "").strip()
    if ch:
        return ch
    parts = session.name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return ""


def dungeon_of_session(session: Path, meta: dict) -> str:
    """meta.dungeon 优先；占位名则退回 recordings 或 FSM_TEST 下的地下城目录。"""
    d = str((meta or {}).get("dungeon") or "").strip()
    if d and d not in ("采集", "未知", "FSM测试"):
        return d
    for root in (RECORDINGS, FSM_TEST_DIR):
        try:
            rel = session.resolve().relative_to(root.resolve())
            if rel.parts:
                return rel.parts[0]
        except ValueError:
            continue
    return d or "未知"


def load_skill_binds(character: str) -> dict | None:
    if not character:
        return None
    path = bind_file(character)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def parse_hold_frames(item, default: int | None = None) -> int | None:
    return parse_hold_ms(item, default)


def write_hold_frames(character: str, slot_durs: dict[int, int]) -> bool:
    """把回放里调的持续 ms 写进键位表，其它字段原样保留。"""
    return write_skill_table(character, slot_durs=slot_durs, combo_slots=None, multi_n=None)


def write_skill_table(
    character: str,
    slot_durs: dict[int, int] | None = None,
    combo_slots: set[int] | None = None,
    multi_n: dict[int, int] | None = None,
    mash_slots: set[int] | None = None,
    edit_slots: set[int] | None = None,
) -> bool:
    """把持续帧 / 连续释放 / 多次释放 / 连按写进键位表，其它字段原样保留。"""
    path = bind_file(character)
    if not path.is_file():
        return False
    if not slot_durs and combo_slots is None and multi_n is None and mash_slots is None:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    skills = data.get("skills")
    if not isinstance(skills, list):
        return False
    want_combo = None if combo_slots is None else {int(s) for s in combo_slots if int(s) > 0}
    want_mash = None if mash_slots is None else {int(s) for s in mash_slots if int(s) > 0}
    want_multi = None
    if multi_n is not None:
        want_multi = {}
        for s, n in multi_n.items():
            try:
                slot, times = int(s), int(n)
            except (TypeError, ValueError):
                continue
            if slot > 0:
                want_multi[slot] = max(DEFAULT_MULTI, min(9, times))
    touch = None
    if edit_slots is not None:
        touch = set()
        for s in edit_slots:
            try:
                n = int(s)
            except (TypeError, ValueError):
                continue
            if n > 0:
                touch.add(n)
    changed = False
    for item in skills:
        if not isinstance(item, dict):
            continue
        try:
            slot = int(item.get("slot") or 0)
        except (TypeError, ValueError):
            continue
        if slot <= 0:
            continue
        if touch is not None and slot not in touch:
            continue
        if slot_durs and slot in slot_durs:
            n = max(1, int(slot_durs[slot]))
            if item.get(HOLD_MS_KEY) != n:
                item[HOLD_MS_KEY] = n
                item.pop("hold_frames", None)
                changed = True
        if want_combo is not None:
            on = slot in want_combo
            if bool(item.get(COMBO_KEY)) is not on or COMBO_KEY not in item:
                item[COMBO_KEY] = on
                changed = True
        if want_multi is not None:
            on = slot in want_multi
            if bool(item.get(MULTI_KEY)) is not on or MULTI_KEY not in item:
                item[MULTI_KEY] = on
                changed = True
            n_val = want_multi.get(slot)
            if n_val is None:
                try:
                    n_val = max(DEFAULT_MULTI, int(item.get(MULTI_N_KEY) or DEFAULT_MULTI))
                except (TypeError, ValueError):
                    n_val = DEFAULT_MULTI
            if item.get(MULTI_N_KEY) != n_val:
                item[MULTI_N_KEY] = n_val
                changed = True
        if want_mash is not None:
            on = slot in want_mash
            if bool(item.get(MASH_KEY)) is not on or MASH_KEY not in item:
                item[MASH_KEY] = on
                changed = True
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def skill_rows(data: dict | None) -> list[dict]:
    """键位表里带快捷键的技能（含 space）；没有快捷键的排除。"""
    out = []
    for item in (data or {}).get("skills") or []:
        try:
            slot = int(item.get("slot") or 0)
        except (TypeError, ValueError):
            continue
        if slot <= 0:
            continue
        cmd = str(item.get("command") or "").strip().lower()
        row = {
            "slot": slot,
            "command": cmd,
            "hotbar": bool(item.get("hotbar")),
        }
        hf = parse_hold_frames(item)
        if hf is not None:
            row[HOLD_FRAMES_KEY] = hf
        row["_combo_set"] = COMBO_KEY in item
        row[COMBO_KEY] = bool(item.get(COMBO_KEY))
        row["_multi_set"] = MULTI_KEY in item
        row[MULTI_KEY] = bool(item.get(MULTI_KEY))
        row[MULTI_N_KEY] = parse_multi_n(item) if item.get(MULTI_KEY) else DEFAULT_MULTI
        row[MASH_KEY] = bool(item.get(MASH_KEY))
        try:
            stored = int(item.get(MULTI_N_KEY) or DEFAULT_MULTI)
            if stored >= DEFAULT_MULTI:
                row[MULTI_N_KEY] = stored
        except (TypeError, ValueError):
            pass
        try:
            if item.get("cooldown_s") is not None:
                row["cooldown_s"] = float(item.get("cooldown_s"))
        except (TypeError, ValueError):
            pass
        if bind_hotkey(row):
            out.append(row)
    out.sort(key=lambda s: s["slot"])
    return out


def hotbar_single_key(skill: dict) -> str | None:
    """快捷栏单键，或不带 + 的 space，都当成技能快捷键。"""
    return bind_hotkey(skill)


def skill_hold_track(
    frames: list[dict],
    keys: list[dict],
    hotbar: list[dict],
    slot_dur: dict[int, int],
) -> list[tuple[int, str] | None]:
    """本帧沿 press 快捷栏单键后，按该技能持续毫秒显示。再按别的技能会打断。"""
    n = len(frames)
    if n == 0:
        return []
    by_key: dict[str, list[dict]] = {}
    for sk in hotbar:
        k = hotbar_single_key(sk)
        if not k:
            continue
        by_key.setdefault(k, []).append(sk)
    last_t = None
    last: tuple[int, str] | None = None
    last_dur = SKILL_HOLD_DEFAULT
    out: list[tuple[int, str] | None] = [None] * n
    for i, fr in enumerate(frames):
        t = int(fr["t_ns"])
        t_prev = int(frames[i - 1]["t_ns"]) if i > 0 else t - 1
        edges = events_in_range(keys, t_prev, t)
        hit = None
        for ev in edges:
            if ev.get("type") != "press":
                continue
            k = str(ev.get("key") or "").lower()
            cands = by_key.get(k)
            if cands:
                hit = cands[0]
        if hit:
            last_t = t
            key = hotbar_single_key(hit) or ""
            last = (int(hit["slot"]), key)
            last_dur = max(1, int(slot_dur.get(int(hit["slot"]), SKILL_HOLD_DEFAULT)))
        if last is not None and last_t is not None and (t - last_t) / 1e6 < last_dur:
            out[i] = last
        else:
            last = None
    return out


def keys_vocab(events: list[dict], held_at_start: list[str] | None = None) -> list[str]:
    seen: set[str] = {str(k).lower() for k in (held_at_start or []) if k}
    for ev in events:
        key = str(ev.get("key") or "").lower()
        if key:
            seen.add(key)
    return sorted(seen)


def load_session(session: Path) -> dict:
    frames = sorted(load_jsonl(session / "frames.jsonl"), key=lambda r: int(r["t_ns"]))
    keys = sorted(load_jsonl(session / "keys.jsonl"), key=lambda r: int(r.get("t_ns") or 0))
    meta = {}
    mp = session / "meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    stuck = [0.0] * len(frames)
    run = 0
    for i, fr in enumerate(frames):
        if i == 0:
            stuck[i] = 0.0
            run = 1
            continue
        if feature_sig(fr) == feature_sig(frames[i - 1]):
            run += 1
            dt = (int(fr["t_ns"]) - int(frames[i - run + 1]["t_ns"])) / 1e9
            stuck[i] = dt
        else:
            run = 1
            stuck[i] = 0.0
    region = meta.get("region") or {}
    keys_path = session / "keys.jsonl"
    held_at_start = [str(k).lower() for k in (meta.get("keys_held_at_start") or []) if k]
    return {
        "session": session,
        "meta": meta,
        "frames": frames,
        "keys": keys,
        "keys_path_exists": keys_path.is_file(),
        "held_at_start": held_at_start,
        "keys_vocab": keys_vocab(keys, held_at_start),
        "stuck": stuck,
        "game_w": int(region.get("width") or 1600),
        "game_h": int(region.get("height") or 900),
        "has_class_xy": any(
            fr.get("mon_xy") or fr.get("loot_xy") or fr.get("gate_xy") or fr.get("boss_xy")
            for fr in frames[: min(20, len(frames))]
        ),
    }


class FsmReplayApp(tk.Tk):
    def __init__(self, start_session: Path | None = None):
        super().__init__()
        self.title("YOLO 录像回放 · FSM 对照")
        self.minsize(1100, 640)
        self.data = None
        self.idx = 0
        self.playing = False
        self.speed = 1.0
        self._photos: dict[str, ImageTk.PhotoImage] = {}
        self._stack_photos: dict[str, ImageTk.PhotoImage] = {}
        self._player_photo = None
        self._count_overlay_photo = None
        self._count_overlay_font = None
        self._frame_overlay_photo = None
        self._frame_overlay_key = None
        self._persist_ok = False
        self._skill_ui_guard = False
        self._skill_hold_saved: dict[str, dict[str, int]] = {}
        self._skill_hold_vars: dict[int, tk.StringVar] = {}
        self._bind_character = ""
        self._bind_skills: list[dict] = []
        self._hotbar_skills: list[dict] = []
        self._load_icons()

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="来源").pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value="全部")
        self.source_combo = ttk.Combobox(
            top,
            textvariable=self.source_var,
            values=("全部", "采集", "FSM测试"),
            state="readonly",
            width=8,
        )
        self.source_combo.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(top, text="录像").pack(side=tk.LEFT, padx=(8, 0))
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(top, textvariable=self.session_var, state="readonly", width=70)
        self.session_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.session_combo.bind("<<ComboboxSelected>>", lambda e: self._open_selected())
        ttk.Button(top, text="刷新", width=6, command=self._refresh_sessions).pack(side=tk.LEFT)
        ttk.Button(top, text="打开技能绑定工具", command=self._open_skill_bind_tool).pack(side=tk.LEFT, padx=(8, 0))

        body = ttk.Frame(self, padding=(8, 0))
        body.pack(fill=tk.BOTH, expand=True)

        count_wrap = tk.Frame(body, width=COUNT_PANEL_W, bg=self.cget("bg"))
        count_wrap.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        count_wrap.pack_propagate(False)

        self.fill_var = tk.BooleanVar(value=True)
        self.rel_var = tk.BooleanVar(value=False)
        self.overlay_var = tk.BooleanVar(value=False)
        self.overlay_alpha_var = tk.IntVar(value=45)
        self.m_var = tk.StringVar(value="5")
        self.l_var = tk.StringVar(value="5")
        self.g_var = tk.StringVar(value="5")
        self.x_var = tk.StringVar(value="1.5")
        self.gx_var = tk.StringVar(value="50")
        self.gy_var = tk.StringVar(value="10")
        self.ax_var = tk.StringVar(value="250")
        self.ay_var = tk.StringVar(value="250")
        self.y_var = tk.StringVar(value="250")
        self.s_var = tk.StringVar(value="20")
        self.xxx_var = tk.StringVar(value="1000")
        self.tap_ms_var = tk.StringVar(value="50")
        self.mash_count_var = tk.StringVar(value="3")
        self.mash_gap_var = tk.StringVar(value="50")
        self.th_var = tk.StringVar(value="50")
        self.pm_var = tk.StringVar(value=str(DEFAULT_PM))
        self.pc_var = tk.StringVar(value=str(DEFAULT_PC))
        self.pt_var = tk.StringVar(value=str(DEFAULT_PT))
        self.town_s_var = tk.StringVar(value=str(int(DEFAULT_TOWN_S)))
        self.run_press_var = tk.StringVar(value=str(RUN_PRESS_GT))
        self.loot_hold_var = tk.StringVar(value=str(LOOT_HOLD_DEFAULT))
        self.corr_u = tk.IntVar(value=0)
        self.corr_d = tk.IntVar(value=0)
        self.corr_l = tk.IntVar(value=0)
        self.corr_r = tk.IntVar(value=0)
        self._corr_lock = False
        self._corr_prev = (0, 0, 0, 0)
        self.e_var = tk.StringVar(value="3")
        self.f_var = tk.StringVar(value=str(DEFAULT_F))
        self._extract_busy = False
        self._feature_cache = None
        self._last_session_path: Path | None = None
        self._session_paths: dict[str, Path] = {}
        self._combo_vars: dict[int, tk.BooleanVar] = {}
        self._combo_ui_guard = False
        self._mash_vars: dict[int, tk.BooleanVar] = {}
        self._mash_ui_guard = False
        self._multi_vars: dict[int, tk.BooleanVar] = {}
        self._multi_n_vars: dict[int, tk.StringVar] = {}
        self._multi_spins: dict[int, ttk.Spinbox] = {}
        self._multi_ui_guard = False
        self._load_ui_settings()
        self.source_combo.bind("<<ComboboxSelected>>", lambda e: self._on_source())

        params = ttk.LabelFrame(count_wrap, text="参数", padding=8)
        params.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        r_mlg = ttk.Frame(params)
        r_mlg.pack(fill=tk.X)
        ttk.Label(r_mlg, text="判定 M").pack(side=tk.LEFT)
        self._spin(r_mlg, self.m_var).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_mlg, text="L").pack(side=tk.LEFT)
        self._spin(r_mlg, self.l_var).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_mlg, text="G").pack(side=tk.LEFT)
        self._spin(r_mlg, self.g_var).pack(side=tk.LEFT, padx=(2, 6))
        self._help_btn(r_mlg, "参数说明", HELP_PARAMS).pack(side=tk.RIGHT)
        r_x = ttk.Frame(params)
        r_x.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r_x, text="卡住 X秒").pack(side=tk.LEFT)
        self._spin(r_x, self.x_var, to=120, frm=0.05, width=5).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_x, text="恢复 Y ms").pack(side=tk.LEFT)
        self._spin(r_x, self.y_var, to=20000, frm=1, width=6).pack(side=tk.LEFT, padx=(2, 0))
        r_gate = ttk.Frame(params)
        r_gate.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r_gate, text="GX").pack(side=tk.LEFT)
        self._spin(r_gate, self.gx_var, frm=0, to=400).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_gate, text="GY").pack(side=tk.LEFT)
        self._spin(r_gate, self.gy_var, frm=0, to=400).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_gate, text="AX ms").pack(side=tk.LEFT)
        self._spin(r_gate, self.ax_var, frm=0, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_gate, text="AY ms").pack(side=tk.LEFT)
        self._spin(r_gate, self.ay_var, frm=0, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 0))
        r_sf = ttk.Frame(params)
        r_sf.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r_sf, text="相似 S%").pack(side=tk.LEFT)
        self._spin(r_sf, self.s_var, frm=0, to=100).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_sf, text="假释放 F%").pack(side=tk.LEFT)
        self._spin(r_sf, self.f_var, frm=0, to=100).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_sf, text="普攻 XXX ms").pack(side=tk.LEFT)
        self._spin(r_sf, self.xxx_var, frm=1, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_sf, text="点按 ms").pack(side=tk.LEFT)
        self._spin(r_sf, self.tap_ms_var, frm=1, to=200).pack(side=tk.LEFT, padx=(2, 0))
        r_mash = ttk.Frame(params)
        r_mash.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r_mash, text="连按 COUNT").pack(side=tk.LEFT)
        self._spin(r_mash, self.mash_count_var, frm=1, to=15).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_mash, text="间隔 ms").pack(side=tk.LEFT)
        self._spin(r_mash, self.mash_gap_var, frm=10, to=300).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_mash, text="移动间隔 ms").pack(side=tk.LEFT)
        self._spin(r_mash, self.th_var, frm=0, to=300).pack(side=tk.LEFT, padx=(2, 0))
        r_loot_fsm = ttk.Frame(params)
        r_loot_fsm.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r_loot_fsm, text="掉落动 PM").pack(side=tk.LEFT)
        self._spin(r_loot_fsm, self.pm_var, frm=0, to=200).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_loot_fsm, text="一键拾取 PC").pack(side=tk.LEFT)
        self._spin(r_loot_fsm, self.pc_var, frm=0, to=40).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_loot_fsm, text="停下 PT").pack(side=tk.LEFT)
        self._spin(r_loot_fsm, self.pt_var).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_loot_fsm, text="回城秒").pack(side=tk.LEFT)
        self._spin(r_loot_fsm, self.town_s_var, frm=1, to=120).pack(side=tk.LEFT, padx=(2, 0))
        r_run = ttk.Frame(params)
        r_run.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r_run, text="跑 press>").pack(side=tk.LEFT)
        self._spin(r_run, self.run_press_var, frm=0, to=30).pack(side=tk.LEFT, padx=(2, 0))
        r_loot = ttk.Frame(params)
        r_loot.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r_loot, text="捡物持续 ms").pack(side=tk.LEFT)
        self._spin(r_loot, self.loot_hold_var, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 0))
        r_corr = ttk.Frame(params)
        r_corr.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(r_corr, text="MON/BOSS补正").pack(side=tk.LEFT)
        self.corr_summary = ttk.Label(r_corr, text="X+0 Y+0", foreground="#06c")
        self.corr_summary.pack(side=tk.LEFT, padx=(8, 0))
        pad = ttk.Frame(params)
        pad.pack(fill=tk.X, pady=(2, 0))
        mx = MON_CORR_MAX
        tk.Scale(pad, from_=mx, to=0, orient=tk.VERTICAL, length=72, width=10, showvalue=True, variable=self.corr_u, label="上", command=self._on_corr_edit).pack(side=tk.LEFT)
        mid = ttk.Frame(pad)
        mid.pack(side=tk.LEFT, fill=tk.Y, padx=4)
        tk.Scale(mid, from_=mx, to=0, orient=tk.HORIZONTAL, length=90, width=10, showvalue=True, variable=self.corr_l, label="左", command=self._on_corr_edit).pack()
        tk.Scale(mid, from_=0, to=mx, orient=tk.HORIZONTAL, length=90, width=10, showvalue=True, variable=self.corr_r, label="右", command=self._on_corr_edit).pack()
        tk.Scale(pad, from_=0, to=mx, orient=tk.VERTICAL, length=72, width=10, showvalue=True, variable=self.corr_d, label="下", command=self._on_corr_edit).pack(side=tk.LEFT)
        self._refresh_corr_summary()
        ttk.Checkbutton(
            params,
            text="player 缺失沿用上次",
            variable=self.fill_var,
            command=self._on_view_opts,
        ).pack(anchor=tk.W, pady=(6, 0))
        ttk.Checkbutton(
            params,
            text="相对坐标（player 原点）",
            variable=self.rel_var,
            command=self._on_view_opts,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            params,
            text="叠图对比 PNG",
            variable=self.overlay_var,
            command=self._on_overlay_opts,
        ).pack(anchor=tk.W)
        r_ov = ttk.Frame(params)
        r_ov.pack(fill=tk.X)
        ttk.Label(r_ov, text="透明度").pack(side=tk.LEFT)
        tk.Scale(
            r_ov,
            from_=8,
            to=100,
            orient=tk.HORIZONTAL,
            length=160,
            width=10,
            showvalue=True,
            variable=self.overlay_alpha_var,
            command=lambda _=None: self._on_overlay_opts(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        skill_box = ttk.LabelFrame(count_wrap, text="技能表", padding=6)
        skill_box.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        skill_head = ttk.Frame(skill_box)
        skill_head.pack(fill=tk.X)
        self.skill_bind_label = ttk.Label(skill_head, text="技能表: —", foreground="#668")
        self.skill_bind_label.pack(side=tk.LEFT)
        ttk.Button(skill_head, text="保存技能表", command=self._save_skill_table).pack(side=tk.RIGHT)
        ttk.Button(skill_head, text="重新读取", command=self._reload_skill_table).pack(side=tk.RIGHT, padx=(0, 4))
        self._help_btn(skill_head, "技能表说明", HELP_SKILLS).pack(side=tk.RIGHT, padx=(0, 4))
        skill_cols = ttk.Frame(skill_box)
        skill_cols.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(skill_cols, text="槽 / 键", width=12).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="持续", width=5).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="连续", width=4).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="多次", width=4).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="MULTI", width=6).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="连按", width=4).pack(side=tk.LEFT)
        self.skill_hold_host = ttk.Frame(skill_box)
        self.skill_hold_host.pack(fill=tk.X)

        feat = ttk.LabelFrame(count_wrap, text="过图技能特征（快捷栏技能段）", padding=6)
        feat.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        feat_r = ttk.Frame(feat)
        feat_r.pack(fill=tk.X)
        ttk.Label(feat_r, text="E").pack(side=tk.LEFT)
        self._e_spin = ttk.Spinbox(feat_r, from_=0, to=40, width=4, textvariable=self.e_var)
        self._e_spin.pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(feat_r, text="F%").pack(side=tk.LEFT)
        ttk.Spinbox(feat_r, from_=0, to=100, width=4, textvariable=self.f_var).pack(side=tk.LEFT, padx=(2, 8))
        self._help_btn(feat_r, "过图技能特征说明", HELP_EXTRACT).pack(side=tk.RIGHT)
        feat_btns = ttk.Frame(feat)
        feat_btns.pack(fill=tk.X, pady=(4, 0))
        self.extract_one_btn = ttk.Button(feat_btns, text="提取本段", command=lambda: self._start_extract(False))
        self.extract_one_btn.pack(side=tk.LEFT)
        self.extract_all_btn = ttk.Button(feat_btns, text="提取同地下城全部", command=lambda: self._start_extract(True))
        self.extract_all_btn.pack(side=tk.LEFT, padx=(4, 0))
        feat_btns2 = ttk.Frame(feat)
        feat_btns2.pack(fill=tk.X, pady=(4, 0))
        self.extract_reset_btn = ttk.Button(
            feat_btns2, text="重新提取本图", command=lambda: self._start_extract(True, wipe=True)
        )
        self.extract_reset_btn.pack(side=tk.LEFT)
        self.rollback_btn = ttk.Button(feat_btns2, text="回退上次叠加", command=self._rollback_extract)
        self.rollback_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.map_reset_host = ttk.Frame(feat)
        self.map_reset_host.pack(fill=tk.X, pady=(4, 0))
        feat_p = ttk.Frame(feat)
        feat_p.pack(fill=tk.X, pady=(4, 0))
        self.extract_prog_var = tk.StringVar(value="")
        ttk.Label(feat_p, textvariable=self.extract_prog_var, width=18).pack(side=tk.LEFT)
        self.extract_prog = ttk.Progressbar(feat_p, mode="determinate")
        self.extract_prog.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.extract_result = tk.Text(feat, height=6, state=tk.DISABLED, font=("Consolas", 9), wrap=tk.WORD)
        self.extract_result.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.e_var.trace_add("write", lambda *_: self._save_ui_settings())
        self.f_var.trace_add("write", lambda *_: self._save_ui_settings())

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1, minsize=PLAYFIELD_H)
        left.rowconfigure(1, weight=0, minsize=FLOW_H)
        left.rowconfigure(2, weight=0)
        self.canvas = tk.Canvas(
            left, width=PLAYFIELD_W, height=PLAYFIELD_H, bg="#1a1d24", highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._field_size = (PLAYFIELD_W, PLAYFIELD_H)
        self._redrawing_field = False
        self.canvas.bind("<Configure>", self._on_field_configure)
        flow_host = tk.Frame(left, height=FLOW_H, bg="#14171d")
        flow_host.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        flow_host.grid_propagate(False)
        flow_host.pack_propagate(False)
        self.flow_canvas = tk.Canvas(flow_host, height=FLOW_H, bg="#14171d", highlightthickness=0)
        self.flow_canvas.pack(fill=tk.BOTH, expand=True)
        status = ttk.Frame(left)
        status.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._status_wrap_labels: list[tk.Label] = []
        state_row = ttk.Frame(status)
        state_row.pack(anchor=tk.W, fill=tk.X)
        self.state_var = tk.StringVar(value="FSM状态: —")
        self.intent_var = tk.StringVar(value="")
        self.send_var = tk.StringVar(value="")
        self.action_var = tk.StringVar(value="")
        # 四列固定宽+固定高，空字也占位。
        for col, w, title, color, var in (
            (0, 200, "FSM 状态", "#0a7", self.state_var),
            (1, 190, "意图（蓝）该干什么", "#4aa3ff", self.intent_var),
            (2, 190, "发键（红）按了什么", "#e85d5d", self.send_var),
            (3, 220, "录像（黄）按了什么", "#e8c547", self.action_var),
        ):
            state_row.columnconfigure(col, minsize=w, weight=0)
            cell = tk.Frame(state_row, width=w, height=STATUS_VALUE_H)
            cell.grid(row=0, column=col, sticky="nw")
            cell.grid_propagate(False)
            cell.pack_propagate(False)
            ttk.Label(cell, text=title, font=("Microsoft YaHei", 9), foreground=color).pack(anchor=tk.W)
            tk.Label(
                cell,
                textvariable=var,
                font=("Microsoft YaHei", 16, "bold"),
                fg=color,
                wraplength=w - 8,
                justify=tk.LEFT,
                anchor=tk.NW,
                height=2,
            ).pack(anchor=tk.W, fill=tk.X)
        self.trigger_var = tk.StringVar(value="")
        self.trigger2_var = tk.StringVar(value="")
        self.cd_var = tk.StringVar(value="")
        self.info_var = tk.StringVar(value="")
        self.keys_all_var = tk.StringVar(value="本段键: —")
        self.keys_var = tk.StringVar(value="按住: —")
        self.keys_edge_var = tk.StringVar(value="本帧沿(上一帧→本帧]: —")
        for var, kw in (
            (self.trigger_var, {"fg": "#0a7", "height": 1}),
            (self.trigger2_var, {"fg": "#0a7", "height": 2}),
            (self.cd_var, {"fg": "#e87a2a", "font": ("Consolas", 10, "bold"), "height": 2}),
            (self.info_var, {"height": 2}),
            (self.keys_all_var, {"height": 1}),
            (self.keys_var, {"fg": "#e8c547", "font": ("Microsoft YaHei", 12, "bold"), "height": 1}),
            (self.keys_edge_var, {"height": 1}),
        ):
            opts = {
                "textvariable": var,
                "font": ("Consolas", 10),
                "anchor": tk.NW,
                "justify": tk.LEFT,
                "wraplength": PLAYFIELD_W,
            }
            opts.update(kw)
            lab = tk.Label(status, **opts)
            lab.pack(anchor=tk.W, fill=tk.X)
            self._status_wrap_labels.append(lab)

        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(fill=tk.X)
        self.play_btn = ttk.Button(ctrl, text="播放", width=8, command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT)
        ttk.Button(ctrl, text="|<", width=4, command=lambda: self._seek(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="<", width=4, command=lambda: self._step(-1)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text=">", width=4, command=lambda: self._step(1)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text=">|", width=4, command=lambda: self._seek(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="速度").pack(side=tk.LEFT, padx=(12, 4))
        self.speed_var = tk.StringVar(value="1x")
        sp = ttk.Combobox(ctrl, textvariable=self.speed_var, values=("0.25x", "0.5x", "1x", "2x", "4x"), width=6, state="readonly")
        sp.pack(side=tk.LEFT)
        sp.bind("<<ComboboxSelected>>", self._on_speed)
        self.scale = ttk.Scale(ctrl, from_=0, to=1, orient=tk.HORIZONTAL, command=self._on_scale)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.frame_var = tk.StringVar(value="0 / 0")
        ttk.Label(ctrl, textvariable=self.frame_var, width=14).pack(side=tk.LEFT)

        self._refresh_sessions()
        self._persist_ok = True
        self._refresh_rollback_btn()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after_idle(self._fit_window)
        opened = False
        if start_session:
            opened = self._select_path(Path(start_session))
        if not opened and self._last_session_path:
            opened = self._select_path(self._last_session_path)
        if not opened and self.session_combo["values"]:
            self.session_combo.current(0)
            self._open_selected()

    def _load_icons(self):
        from make_fsm_icons import write_icons

        write_icons(ICON_DIR)
        for name in CLASSES:
            p = ICON_DIR / f"{name}.png"
            if not p.is_file():
                continue
            img = Image.open(p).convert("RGB")
            self._stack_photos[name] = ImageTk.PhotoImage(img.resize((28, 28)))
            if name == "player":
                self._player_photo = ImageTk.PhotoImage(img.resize((36, 36)))

    def _help_btn(self, parent, title: str, body: str) -> ttk.Button:
        return ttk.Button(
            parent,
            text="说明",
            width=4,
            command=lambda: messagebox.showinfo(title, body, parent=self),
        )

    def _spin(self, parent, var: tk.StringVar, to: int = 60, frm: int = 1, width: int = 4) -> ttk.Spinbox:
        sp = ttk.Spinbox(parent, from_=frm, to=to, width=width, textvariable=var, command=self._on_fsm_opts, increment=0.05 if isinstance(frm, float) or (isinstance(to, float)) else 1)
        var.trace_add("write", lambda *_: self._on_fsm_opts())
        return sp

    def _fsm_track_kwargs(self) -> dict:
        ox, oy = mon_off_from_corr(*self._corr_tuple())
        return {
            "m": self._spin_n(self.m_var, 5),
            "l": self._spin_n(self.l_var, 5),
            "g": self._spin_n(self.g_var, 5),
            "x_s": self._spin_f(self.x_var, 1.5, lo=0.05),
            "gx": self._spin_n(self.gx_var, 50, lo=0),
            "gy": self._spin_n(self.gy_var, 10, lo=0),
            "ax_ms": self._spin_n(self.ax_var, 250, lo=0),
            "ay_ms": self._spin_n(self.ay_var, 250, lo=0),
            "y_ms": self._spin_n(self.y_var, 250),
            "s": self._spin_n(self.s_var, 20, lo=0),
            "pm": self._spin_n(self.pm_var, DEFAULT_PM, lo=0),
            "pc": self._spin_n(self.pc_var, DEFAULT_PC, lo=0),
            "pt": self._spin_n(self.pt_var, DEFAULT_PT),
            "xxx_ms": self._spin_n(self.xxx_var, 1000),
            "th_ms": self._spin_n(self.th_var, 50, lo=0),
            "fight_plan": self._fight_plan(),
            "hotbar": hotbar_from_binds(self._bind_skills),
            "dist_table": dist_table_from_features(self._feature_cache),
            "map_reset": has_map_reset(self._feature_cache),
            "mon_off_x": ox,
            "mon_off_y": oy,
            "tn_s": self._spin_f(self.town_s_var, float(DEFAULT_TOWN_S), lo=0.05),
        }

    def _send_text(self, dr: dict) -> str:
        mash_n = mash_n_for_slot(
            dr.get("skill_slot"),
            self._mash_slots_from_ui(),
            self._spin_n(self.mash_count_var, 3),
        )
        return send_keys_label(
            dr.get("action"),
            tuple(dr.get("move_dirs") or ()),
            dr.get("move_dir"),
            dr.get("skill_key"),
            mash_n=mash_n,
        )

    def _corr_tuple(self) -> tuple[int, int, int, int]:
        def n(var: tk.IntVar) -> int:
            try:
                return max(0, min(MON_CORR_MAX, int(var.get())))
            except (TypeError, ValueError, tk.TclError):
                return 0

        return n(self.corr_u), n(self.corr_d), n(self.corr_l), n(self.corr_r)

    def _refresh_corr_summary(self):
        lab = getattr(self, "corr_summary", None)
        if lab is None:
            return
        ox, oy = mon_off_from_corr(*self._corr_tuple())
        try:
            lab.config(text=f"X{ox:+d} Y{oy:+d}")
        except tk.TclError:
            pass

    def _on_corr_edit(self, _=None):
        if getattr(self, "_corr_lock", False):
            return
        self._corr_lock = True
        try:
            cur = self._corr_tuple()
            prev = getattr(self, "_corr_prev", (0, 0, 0, 0))
            u, dwn, left, right = cur
            if u > 0 and dwn > 0:
                if u != prev[0]:
                    self.corr_d.set(0)
                elif dwn != prev[1]:
                    self.corr_u.set(0)
                elif u >= dwn:
                    self.corr_d.set(0)
                else:
                    self.corr_u.set(0)
            if left > 0 and right > 0:
                if left != prev[2]:
                    self.corr_r.set(0)
                elif right != prev[3]:
                    self.corr_l.set(0)
                elif left >= right:
                    self.corr_r.set(0)
                else:
                    self.corr_l.set(0)
            self._corr_prev = self._corr_tuple()
        finally:
            self._corr_lock = False
        self._refresh_corr_summary()
        self._on_fsm_opts()

    @staticmethod
    def _spin_n(var: tk.StringVar, default: int, lo: int = 1) -> int:
        try:
            return max(lo, int(str(var.get()).strip() or default))
        except ValueError:
            return default

    @staticmethod
    def _spin_f(var: tk.StringVar, default: float, lo: float = 0.05) -> float:
        try:
            return max(lo, float(str(var.get()).strip() or default))
        except ValueError:
            return float(default)

    def _fit_window(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.minsize(min(980, max(sw - 20, 640)), 520)
        if apply_window_geom(self, "fsm_replay", min_w=min(980, max(sw - 20, 640)), min_h=520):
            return
        w = max(min(sw - 20, 1680), min(sw - 20, 980))
        h = max(sh - 72, 560)
        self.geometry(f"{w}x{h}+6+6")

    def _draw_flow(self, state, method_steps, method_hit: int):
        c = getattr(self, "flow_canvas", None)
        if c is None:
            return
        c.delete("all")
        w = max(int(c.winfo_width() or PLAYFIELD_W), 400)
        states = (
            FsmState.WAIT,
            FsmState.FIGHT,
            FsmState.LOOT,
            FsmState.ADVANCE,
            FsmState.RETURN,
            FsmState.IDLE,
            FsmState.STUCK,
        )
        n = len(states)
        box_w = max(56, min(100, (w - 16) // n - 8))
        y0 = 8
        h0 = 26
        cur = state if isinstance(state, FsmState) else None
        if cur is None and state is not None:
            for s in states:
                if getattr(state, "value", state) == s.value:
                    cur = s
                    break
        for i, st in enumerate(states):
            x = 8 + i * (box_w + 8)
            on = cur is st
            c.create_rectangle(
                x, y0, x + box_w, y0 + h0,
                fill="#1e8a55" if on else "#2a3140",
                outline="#3d9a6a" if on else "#445",
            )
            c.create_text(
                x + box_w / 2, y0 + h0 / 2,
                text=st.value,
                fill="#dff" if on else "#9ab",
                font=("Microsoft YaHei", 9, "bold" if on else "normal"),
            )
            if i < n - 1:
                c.create_line(x + box_w + 1, y0 + h0 / 2, x + box_w + 7, y0 + h0 / 2, fill="#667", arrow=tk.LAST)
        steps = list(method_steps or ())
        if not steps:
            c.create_text(10, 58, anchor=tk.W, fill="#667", font=("Microsoft YaHei", 9), text="方法：当前状态无执行步骤")
            return
        m = len(steps)
        mw = max(48, min(120, (w - 16) // m - 6))
        y1 = 42
        h1 = 28
        for i, name in enumerate(steps):
            x = 8 + i * (mw + 6)
            on = i == method_hit
            c.create_rectangle(
                x, y1, x + mw, y1 + h1,
                fill="#2b6cb0" if on else "#2a3140",
                outline="#5aa0e0" if on else "#445",
            )
            c.create_text(
                x + mw / 2, y1 + h1 / 2,
                text=name,
                fill="#eef" if on else "#9ab",
                font=("Microsoft YaHei", 8, "bold" if on else "normal"),
            )
            if i < m - 1:
                c.create_line(x + mw + 1, y1 + h1 / 2, x + mw + 5, y1 + h1 / 2, fill="#667", arrow=tk.LAST)

    def _on_field_configure(self, event):
        if event.widget is not self.canvas:
            return
        nw, nh = max(event.width, 2), max(event.height, 2)
        wrap_w = max(nw - 8, 200)
        for lab in getattr(self, "_status_wrap_labels", []):
            try:
                lab.configure(wraplength=wrap_w)
            except tk.TclError:
                pass
        if abs(nw - self._field_size[0]) < 2 and abs(nh - self._field_size[1]) < 2:
            return
        self._field_size = (nw, nh)
        if self.data and not self._redrawing_field:
            self._redrawing_field = True
            try:
                self._show()
            finally:
                self._redrawing_field = False

    def _field_wh(self) -> tuple[int, int]:
        w, h = self._field_size
        if w < 80:
            w = PLAYFIELD_W
        if h < 80:
            h = PLAYFIELD_H
        return w, h

    def _session_label(self, p: Path) -> str:
        tag = "[FSM测试]" if session_kind(p) == "fsm" else "[采集]"
        try:
            root = FSM_TEST_DIR if session_kind(p) == "fsm" else RECORDINGS
            rel = str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            rel = str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).replace("\\", "/")
        keys_p = p / "keys.jsonl"
        n = 0
        if keys_p.is_file():
            with keys_p.open("r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
        return f"{tag} {rel}  [keys={n}]"

    def _session_roots(self) -> tuple[Path, ...]:
        src = (self.source_var.get() or "全部").strip()
        if src == "采集":
            return (RECORDINGS,)
        if src == "FSM测试":
            return (FSM_TEST_DIR,)
        return (RECORDINGS, FSM_TEST_DIR)

    def _on_source(self):
        keep = self.data["session"] if self.data else None
        self._refresh_sessions()
        if keep and any(p.resolve() == keep.resolve() for p in self._session_paths.values()):
            self._select_path(keep)
            return
        if self.session_combo["values"]:
            self.session_combo.current(0)
            self._open_selected()
            return
        self._clear_session()
        self._save_ui_settings()

    def _refresh_sessions(self):
        keep = None
        if self.data:
            keep = self.data["session"]
        elif self.session_var.get().strip():
            keep = self._session_paths.get(self.session_var.get().strip())
        sessions = list_sessions(*self._session_roots())
        labels = [self._session_label(p) for p in sessions]
        self._session_paths = {lab: p for lab, p in zip(labels, sessions)}
        self.session_combo["values"] = labels
        if keep:
            for lab, p in self._session_paths.items():
                if p.resolve() == keep.resolve():
                    self.session_var.set(lab)
                    return
        if labels:
            if not self.session_var.get().strip():
                self.session_combo.current(0)
        else:
            self.session_var.set("")

    def _select_path(self, path: Path) -> bool:
        path = path.resolve()
        if not path.exists():
            return False
        src = (self.source_var.get() or "全部").strip()
        kind = session_kind(path)
        if src == "采集" and kind == "fsm":
            self.source_var.set("全部")
            self._refresh_sessions()
        elif src == "FSM测试" and kind == "rec":
            self.source_var.set("全部")
            self._refresh_sessions()
        for lab, p in self._session_paths.items():
            if p.resolve() == path:
                self.session_var.set(lab)
                self._open_selected()
                return True
        self._session_paths[str(path)] = path
        lab = self._session_label(path)
        vals = list(self.session_combo["values"]) + [lab]
        self._session_paths[lab] = path
        self.session_combo["values"] = vals
        self.session_var.set(lab)
        self._open_selected()
        return True

    def _clear_session(self):
        self.playing = False
        self.play_btn.config(text="播放")
        self.data = None
        self.idx = 0
        self.scale.config(to=1)
        self.frame_var.set("0 / 0")
        src = (self.source_var.get() or "全部").strip()
        if src == "FSM测试":
            self.info_var.set("没有 FSM_TEST 段。FSM测试写盘后点刷新；目录还不存在时这里是空的。")
        elif src == "采集":
            self.info_var.set("没有采集段（recordings/*/frames.jsonl）。")
        else:
            self.info_var.set("没有可回放的段。")

    def _open_selected(self):
        lab = self.session_var.get().strip()
        path = self._session_paths.get(lab)
        if not path:
            return
        self.playing = False
        self.play_btn.config(text="播放")
        self.data = load_session(path)
        self._last_session_path = path
        n = max(len(self.data["frames"]) - 1, 1)
        self.scale.config(to=n)
        self.idx = 0
        self._sync_skill_binds()
        self._reload_feature_cache()
        self._rebuild_views()
        self._show()
        self._refresh_rollback_btn()

    def _on_view_opts(self):
        self._save_ui_settings()
        if not self.data:
            return
        self._rebuild_views()
        self._show()

    def _on_overlay_opts(self):
        self._save_ui_settings()
        if self.data:
            self._show()

    def _on_fsm_opts(self):
        if self._skill_ui_guard:
            return
        self._save_ui_settings()
        self._sync_f_labels()
        if not self.data:
            return
        self._rebuild_draft()
        self._show()

    def _sync_f_labels(self):
        data = self._feature_cache
        if not data:
            return
        f = min(100, self._spin_n(self.f_var, DEFAULT_F, lo=0))
        try:
            prev = int(data.get("f"))
        except (TypeError, ValueError):
            prev = -1
        if prev == f:
            return
        apply_f(data, f)
        char = self._bind_character
        dun = self._current_dungeon() if self.data else ""
        if char and dun and dun != "未知":
            save_features(char, dun, data, keep_prev=False)
        if hasattr(self, "extract_result"):
            self._set_extract_result(format_report(data))

    def _load_ui_settings(self):
        if not SETTINGS_PATH.is_file():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        for key, var, default, lo in (
            ("m", self.m_var, 5, 1),
            ("l", self.l_var, 5, 1),
            ("g", self.g_var, 5, 1),
            ("gx", self.gx_var, 50, 0),
            ("gy", self.gy_var, 10, 0),
            ("tap_ms", self.tap_ms_var, 50, 1),
            ("mash_count", self.mash_count_var, 3, 1),
            ("mash_gap_ms", self.mash_gap_var, 50, 10),
            ("th_ms", self.th_var, 50, 0),
            ("s", self.s_var, 20, 0),
            ("pm", self.pm_var, DEFAULT_PM, 0),
            ("pc", self.pc_var, DEFAULT_PC, 0),
            ("pt", self.pt_var, DEFAULT_PT, 1),
            ("town_s", self.town_s_var, int(DEFAULT_TOWN_S), 1),
            ("run_press", self.run_press_var, RUN_PRESS_GT, 0),
            ("e", self.e_var, 3, 0),
            ("f", self.f_var, DEFAULT_F, 0),
        ):
            if key in data:
                try:
                    var.set(str(max(lo, int(data[key]))))
                except (TypeError, ValueError):
                    var.set(str(default))
        if "th_ms" not in data:
            try:
                self.th_var.set(str(max(0, int(str(self.mash_gap_var.get() or 50)))))
            except (TypeError, ValueError):
                self.th_var.set("50")
        self.x_var.set(str(json_s_from_frames(data, "x_s", "x", 30)))
        self.ax_var.set(str(json_ms(data, "ax_ms", "ax", 5)))
        self.ay_var.set(str(json_ms(data, "ay_ms", "ay", 5)))
        if "ax_ms" not in data and "ax" not in data and "a" in data:
            try:
                n = max(0, int(data["a"])) * 50
                self.ax_var.set(str(n))
                self.ay_var.set(str(n))
            except (TypeError, ValueError):
                pass
        self.xxx_var.set(str(json_ms(data, "xxx_ms", "xxx", 20, lo=1)))
        self.y_var.set(str(json_ms(data, "y_ms", "y", 5, lo=1)))
        self.loot_hold_var.set(str(json_ms(data, "loot_hold_ms", "loot_hold", 5, lo=1)))
        if "pm" not in data and "lm" in data:
            try:
                self.pm_var.set(str(max(0, int(data["lm"]))))
            except (TypeError, ValueError):
                pass
        if "pc" not in data and "lc" in data:
            try:
                self.pc_var.set(str(max(0, int(data["lc"]))))
            except (TypeError, ValueError):
                pass
        if "fill" in data:
            self.fill_var.set(bool(data["fill"]))
        if "relative" in data:
            self.rel_var.set(bool(data["relative"]))
        if "overlay" in data:
            self.overlay_var.set(bool(data["overlay"]))
        if "overlay_alpha" in data:
            try:
                self.overlay_alpha_var.set(max(8, min(100, int(data["overlay_alpha"]))))
            except (TypeError, ValueError):
                pass
        src = str(data.get("session_source") or "").strip()
        if src in ("全部", "采集", "FSM测试"):
            self.source_var.set(src)
        last = str(data.get("last_session") or "").strip()
        if last:
            p = Path(last)
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                self._last_session_path = p
        cu, cd, cl, cr = mon_corr_from_dict(data)
        self._corr_lock = True
        self.corr_u.set(cu)
        self.corr_d.set(cd)
        self.corr_l.set(cl)
        self.corr_r.set(cr)
        self._corr_lock = False
        self._corr_prev = (cu, cd, cl, cr)
        sh = data.get("skill_hold_ms")
        legacy = False
        if not isinstance(sh, dict):
            sh = data.get("skill_hold")
            legacy = True
        if isinstance(sh, dict):
            saved: dict[str, dict[str, int]] = {}
            for ch, slots in sh.items():
                if not isinstance(slots, dict):
                    continue
                one = {}
                for sk, sv in slots.items():
                    try:
                        n = max(1, int(sv))
                    except (TypeError, ValueError):
                        continue
                    if legacy:
                        n *= 50
                    one[str(sk)] = n
                saved[str(ch)] = one
            self._skill_hold_saved = saved

    def _save_ui_settings(self):
        if not self._persist_ok:
            return
        self._flush_skill_hold_saved()
        data = {
            "m": self._spin_n(self.m_var, 5),
            "l": self._spin_n(self.l_var, 5),
            "g": self._spin_n(self.g_var, 5),
            "x_s": self._spin_f(self.x_var, 1.5, lo=0.05),
            "gx": self._spin_n(self.gx_var, 50, lo=0),
            "gy": self._spin_n(self.gy_var, 10, lo=0),
            "ax_ms": self._spin_n(self.ax_var, 250, lo=0),
            "ay_ms": self._spin_n(self.ay_var, 250, lo=0),
            "xxx_ms": self._spin_n(self.xxx_var, 1000),
            "tap_ms": self._spin_n(self.tap_ms_var, 50),
            "mash_count": self._spin_n(self.mash_count_var, 3),
            "mash_gap_ms": self._spin_n(self.mash_gap_var, 50, lo=10),
            "th_ms": self._spin_n(self.th_var, 50, lo=0),
            "y_ms": self._spin_n(self.y_var, 250),
            "s": self._spin_n(self.s_var, 20, lo=0),
            "pm": self._spin_n(self.pm_var, DEFAULT_PM, lo=0),
            "pc": self._spin_n(self.pc_var, DEFAULT_PC, lo=0),
            "pt": self._spin_n(self.pt_var, DEFAULT_PT),
            "town_s": self._spin_f(self.town_s_var, float(DEFAULT_TOWN_S), lo=0.05),
            "run_press": self._spin_n(self.run_press_var, RUN_PRESS_GT, lo=0),
            "loot_hold_ms": self._spin_n(self.loot_hold_var, LOOT_HOLD_DEFAULT),
            "e": self._spin_n(self.e_var, 3, lo=0),
            "f": self._spin_n(self.f_var, DEFAULT_F, lo=0),
            "fill": bool(self.fill_var.get()),
            "relative": bool(self.rel_var.get()),
            "overlay": bool(self.overlay_var.get()),
            "overlay_alpha": max(8, min(100, int(self.overlay_alpha_var.get() or 45))),
            "skill_hold_ms": self._skill_hold_saved,
            "session_source": (self.source_var.get() or "全部").strip() or "全部",
            "last_session": (
                self._session_rel(self.data["session"])
                if self.data
                else (self._session_rel(self._last_session_path) if self._last_session_path else "")
            ),
        }
        u, dwn, left, right = self._corr_tuple()
        data["mon_corr_u"] = u
        data["mon_corr_d"] = dwn
        data["mon_corr_l"] = left
        data["mon_corr_r"] = right
        try:
            SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        self._write_hold_frames_to_binds()

    def _on_close(self):
        self._persist_ok = True
        remember_window_geom(self, "fsm_replay")
        self._save_ui_settings()
        self.destroy()

    def _flush_skill_hold_saved(self):
        char = self._bind_character
        if not char:
            return
        cur = dict(self._skill_hold_saved.get(char) or {})
        for slot, var in self._skill_hold_vars.items():
            cur[str(slot)] = self._spin_n(var, SKILL_HOLD_DEFAULT)
        self._skill_hold_saved[char] = cur

    def _write_hold_frames_to_binds(self):
        char = self._bind_character
        if not char:
            return
        durs: dict[int, int] = {}
        for k, v in (self._skill_hold_saved.get(char) or {}).items():
            try:
                durs[int(k)] = max(1, int(v))
            except (TypeError, ValueError):
                continue
        durs.update(self._slot_durs())
        write_skill_table(
            char,
            slot_durs=durs or None,
            combo_slots=self._combo_slots_from_ui(),
            multi_n=self._multi_n_from_ui(),
            mash_slots=self._mash_slots_from_ui(),
            edit_slots=self._edit_slots_from_ui(),
        )

    def _open_skill_bind_tool(self):
        path = ROOT / "skill_bind_tool.py"
        if not path.is_file():
            messagebox.showerror("技能绑定工具", f"找不到 {path}", parent=self)
            return
        kwargs = {"cwd": str(ROOT)}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        try:
            subprocess.Popen([sys.executable, str(path)], **kwargs)
        except Exception as e:
            messagebox.showerror("技能绑定工具", str(e), parent=self)

    def _save_skill_table(self):
        char = self._bind_character
        if not char:
            messagebox.showinfo("技能表", "当前录像没有角色名，没法写入键位表。", parent=self)
            return
        if not bind_file(char).is_file():
            messagebox.showinfo("技能表", f"没有 {bind_file(char).name}，请先打开技能绑定工具生成。", parent=self)
            return
        self._flush_skill_hold_saved()
        write_skill_table(
            char,
            slot_durs=self._slot_durs(),
            combo_slots=self._combo_slots_from_ui(),
            multi_n=self._multi_n_from_ui(),
            mash_slots=self._mash_slots_from_ui(),
            edit_slots=self._edit_slots_from_ui(),
        )
        self._sync_combo_to_features(save=True)
        self._sync_multi_to_features(save=True)
        if self.data:
            self._rebuild_draft()
            self._show()
        messagebox.showinfo("技能表", f"已写入 {bind_file(char).name}（持续 ms + 连续释放 + 多次释放 + 连按）。", parent=self)

    def _sync_combo_to_features(self, save: bool = False):
        slots = self._combo_slots_from_ui()
        data = self._feature_cache
        if not data:
            return
        set_combo_slots(data, slots)
        if save:
            char = self._bind_character
            dun = self._current_dungeon() if self.data else ""
            if char and dun and dun != "未知":
                save_features(char, dun, data, keep_prev=False)
        if hasattr(self, "extract_result"):
            self._set_extract_result(format_report(data))

    def _sync_multi_to_features(self, save: bool = False):
        mapping = self._multi_n_from_ui()
        data = self._feature_cache
        if not data:
            return
        set_multi_slots(data, mapping)
        if save:
            char = self._bind_character
            dun = self._current_dungeon() if self.data else ""
            if char and dun and dun != "未知":
                save_features(char, dun, data, keep_prev=False)
        if hasattr(self, "extract_result"):
            self._set_extract_result(format_report(data))

    def _slot_durs(self) -> dict[int, int]:
        out = {}
        for slot, var in self._skill_hold_vars.items():
            out[int(slot)] = self._spin_n(var, SKILL_HOLD_DEFAULT)
        return out

    def _reload_skill_table(self):
        """从磁盘重新读键位表，丢掉面板上未保存的改动。"""
        if not self.data:
            messagebox.showinfo("技能表", "先打开一段录像。", parent=self)
            return
        char = character_of_session(self.data["session"], self.data.get("meta") or {})
        if char:
            self._skill_hold_saved.pop(char, None)
        self._sync_skill_binds(reload_disk=True)
        self._rebuild_draft()
        self._show()

    def _sync_skill_binds(self, reload_disk: bool = False):
        if not reload_disk:
            self._flush_skill_hold_saved()
        d = self.data
        if not d:
            return
        char = character_of_session(d["session"], d.get("meta") or {})
        if reload_disk and char:
            self._skill_hold_saved.pop(char, None)
        binds = load_skill_binds(char)
        self._bind_character = char
        self._bind_skills = skill_rows(binds)
        self._hotbar_skills = list(self._bind_skills)
        if not char:
            self.skill_bind_label.config(text="技能表: 录像无角色名")
        elif binds is None:
            self.skill_bind_label.config(text=f"技能表: 没有 {bind_file(char).name}")
        else:
            self.skill_bind_label.config(
                text=f"技能表: {char}  {len(self._bind_skills)}个快捷键（含space）"
            )
        self._rebuild_skill_hold_ui(from_binds=reload_disk)

    def _rebuild_skill_hold_ui(self, *, from_binds: bool = False):
        self._skill_ui_guard = True
        self._combo_ui_guard = True
        self._mash_ui_guard = True
        self._multi_ui_guard = True
        try:
            if not from_binds:
                self._flush_skill_hold_saved()
            for child in self.skill_hold_host.winfo_children():
                child.destroy()
            self._skill_hold_vars = {}
            self._combo_vars = {}
            self._mash_vars = {}
            self._multi_vars = {}
            self._multi_n_vars = {}
            self._multi_spins = {}
            saved = dict(self._skill_hold_saved.get(self._bind_character) or {})
            for sk in self._bind_skills:
                slot = int(sk["slot"])
                hf = sk.get(HOLD_FRAMES_KEY)
                if hf is None:
                    continue
                key = str(slot)
                if from_binds or key not in saved:
                    saved[key] = max(1, int(hf))
            bind_combo = {int(sk["slot"]) for sk in self._bind_skills if sk.get(COMBO_KEY)}
            if any(sk.get("_combo_set") for sk in self._bind_skills):
                marked = bind_combo
            else:
                marked = bind_combo or combo_slots_of(self._feature_cache)
            bind_multi = {int(sk["slot"]) for sk in self._bind_skills if sk.get(MULTI_KEY)}
            if any(sk.get("_multi_set") for sk in self._bind_skills):
                marked_m = bind_multi
            else:
                marked_m = bind_multi or set(multi_n_of(self._feature_cache))
            for sk in self._bind_skills:
                slot = int(sk["slot"])
                cmd = sk["command"] or "—"
                row = ttk.Frame(self.skill_hold_host)
                row.pack(fill=tk.X, pady=1)
                mark = "*" if hotbar_single_key(sk) else " "
                ttk.Label(row, text=f"{mark}{slot} {cmd}", width=12, anchor=tk.W).pack(side=tk.LEFT)
                prev = saved.get(str(slot), SKILL_HOLD_DEFAULT)
                var = tk.StringVar(value=str(prev))
                self._spin(row, var, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 4))
                self._skill_hold_vars[slot] = var
                cvar = tk.BooleanVar(value=slot in marked)
                self._combo_vars[slot] = cvar
                ttk.Checkbutton(row, variable=cvar, command=self._on_combo_toggle).pack(side=tk.LEFT)
                on_m = slot in marked_m
                mvar = tk.BooleanVar(value=on_m)
                self._multi_vars[slot] = mvar
                ttk.Checkbutton(row, variable=mvar, command=self._on_multi_toggle).pack(side=tk.LEFT, padx=(6, 0))
                n_prev = int(sk.get(MULTI_N_KEY) or DEFAULT_MULTI)
                nvar = tk.StringVar(value=str(max(DEFAULT_MULTI, n_prev)))
                self._multi_n_vars[slot] = nvar
                sp = ttk.Spinbox(
                    row,
                    from_=DEFAULT_MULTI,
                    to=9,
                    width=3,
                    textvariable=nvar,
                    command=self._on_multi_n_change,
                    state=tk.NORMAL if on_m else tk.DISABLED,
                )
                sp.pack(side=tk.LEFT, padx=(2, 0))
                nvar.trace_add("write", lambda *_: self._on_multi_n_change())
                self._multi_spins[slot] = sp
                mash_var = tk.BooleanVar(value=bool(sk.get(MASH_KEY)))
                self._mash_vars[slot] = mash_var
                ttk.Checkbutton(row, variable=mash_var, command=self._on_mash_toggle).pack(side=tk.LEFT, padx=(6, 0))
            if self._bind_character:
                self._skill_hold_saved[self._bind_character] = {
                    **saved,
                    **{str(s): self._spin_n(v, SKILL_HOLD_DEFAULT) for s, v in self._skill_hold_vars.items()},
                }
                for sk in self._bind_skills:
                    sk[HOLD_FRAMES_KEY] = self._spin_n(
                        self._skill_hold_vars[int(sk["slot"])], SKILL_HOLD_DEFAULT
                    )
        finally:
            self._skill_ui_guard = False
            self._combo_ui_guard = False
            self._mash_ui_guard = False
            self._multi_ui_guard = False
        if self._persist_ok:
            self._write_hold_frames_to_binds()
            self._sync_combo_to_features(save=False)
            self._sync_multi_to_features(save=False)

    def _rebuild_views(self):
        d = self.data
        if not d:
            return
        d["views"] = views_for_frames(
            d["frames"],
            fill=bool(self.fill_var.get()),
            relative=bool(self.rel_var.get()),
            height=float(d["game_h"]) if d.get("game_h") else None,
        )
        self._rebuild_draft()

    def _fight_plan(self):
        data = self._feature_cache
        if data:
            apply_f(data, self._spin_n(self.f_var, DEFAULT_F, lo=0))
        return fight_plan_from_features(data, self._bind_skills)

    def _rebuild_draft(self):
        d = self.data
        if not d:
            return
        d["draft"] = compute_draft_track(
            d["frames"],
            **self._fsm_track_kwargs(),
        )
        d["loot_hold"] = loot_hold_track(
            d["frames"],
            d.get("keys") or [],
            self._spin_n(self.loot_hold_var, LOOT_HOLD_DEFAULT),
        )
        d["skill_hold"] = skill_hold_track(
            d["frames"],
            d.get("keys") or [],
            self._hotbar_skills,
            self._slot_durs(),
        )
        f = self._spin_n(self.f_var, DEFAULT_F, lo=0)
        live_casts, _ = casts_from_tracks(
            self._session_rel(d["session"]),
            d["draft"],
            d["skill_hold"],
            d.get("views") or [],
            self._spin_n(self.e_var, 3, lo=0),
            self._bind_skills,
            dist_table=dist_table_from_features(self._feature_cache),
            s=self._spin_n(self.s_var, 20, lo=0),
            slot_durs=self._slot_durs(),
        )
        skip_ids = {str(c["id"]) for c in live_casts if c.get("id") and cast_is_fake(c, f)}
        d["cd_resets"] = detect_cd_resets(
            self._session_rel(d["session"]),
            d["frames"],
            d["draft"],
            d["skill_hold"],
            self._bind_skills,
            combo_slots=self._combo_slots_from_ui(),
            multi_slots=set(self._multi_n_from_ui()),
            skip_ids=skip_ids,
        )

    def _on_speed(self, _=None):
        raw = self.speed_var.get().replace("x", "")
        try:
            self.speed = float(raw)
        except ValueError:
            self.speed = 1.0

    def _toggle_play(self):
        if not self.data or not self.data["frames"]:
            return
        self.playing = not self.playing
        self.play_btn.config(text="暂停" if self.playing else "播放")
        if self.playing:
            self._tick()

    def _tick(self):
        if not self.playing or not self.data:
            return
        frames = self.data["frames"]
        if self.idx >= len(frames) - 1:
            self.playing = False
            self.play_btn.config(text="播放")
            return
        t0 = int(frames[self.idx]["t_ns"])
        t1 = int(frames[self.idx + 1]["t_ns"])
        delay_ms = max(10, int((t1 - t0) / 1e6 / max(self.speed, 0.01)))
        self._step(1)
        self.after(delay_ms, self._tick)

    def _step(self, d: int):
        if not self.data:
            return
        n = len(self.data["frames"])
        if not n:
            return
        self.idx = max(0, min(n - 1, self.idx + d))
        self.scale.set(self.idx)
        self._show()

    def _seek(self, i: int):
        if not self.data:
            return
        n = len(self.data["frames"])
        self.idx = n - 1 if i < 0 else 0
        self.scale.set(self.idx)
        self._show()

    def _on_scale(self, val):
        if not self.data:
            return
        self.idx = int(float(val))
        self._show()

    def _show(self):
        d = self.data
        if not d or not d["frames"]:
            return
        i = self.idx
        fr = d["frames"][i]
        n = len(d["frames"])
        t0 = int(d["frames"][0]["t_ns"])
        t = (int(fr["t_ns"]) - t0) / 1e9
        if i > 0:
            dt_s = f"  dt={(int(fr['t_ns']) - int(d['frames'][i - 1]['t_ns'])) / 1e6:.0f}ms"
        else:
            dt_s = "  dt=—"
        self.frame_var.set(f"{i + 1} / {n}")
        counts = {
            "player": 0 if fr.get("player_xy") is None else 1,
            "mon": int(fr.get("mon") or 0),
            "loot": int(fr.get("loot") or 0),
            "gate": int(fr.get("gate") or 0),
            "boss": int(fr.get("boss") or 0),
        }

        draft_rows = d.get("draft") or []
        dr = draft_rows[i] if i < len(draft_rows) else None
        x_lim = self._spin_f(self.x_var, 1.5, lo=0.05)
        run_on = False
        move_on = False
        edges: list[dict] = []
        held: list[str] = []
        if d.get("keys_path_exists") and (d["keys"] or d.get("held_at_start")):
            t_prev = int(d["frames"][i - 1]["t_ns"]) if i > 0 else int(fr["t_ns"]) - 1
            edges = events_in_range(d["keys"], t_prev, int(fr["t_ns"]))
            held = rebuild_held_at(d["keys"], int(fr["t_ns"]), d.get("held_at_start"))
            gt = self._spin_n(self.run_press_var, RUN_PRESS_GT, lo=0)
            run_on = run_burst_active(edges, gt=gt)
            move_on = (not run_on) and move_active(edges, held)
        loot_on = False
        loot_hold = d.get("loot_hold") or []
        if i < len(loot_hold):
            loot_on = bool(loot_hold[i])
        skill_hit = None
        skill_hold = d.get("skill_hold") or []
        if i < len(skill_hold):
            skill_hit = skill_hold[i]
        x_atk = basic_attack_active(edges, held)
        if skill_hit:
            action_s = f"技能{skill_hit[0]}：{skill_hit[1]}"
        else:
            bits = []
            if loot_on:
                bits.append("捡物")
            if x_atk:
                bits.append("普攻")
            if run_on:
                bits.append("跑")
            elif move_on:
                bits.append("移动")
            action_s = "  ".join(bits)
        self.action_var.set(action_s)
        if dr is None:
            self.state_var.set("FSM状态: —")
            self.intent_var.set("")
            self.send_var.set("")
            self.trigger_var.set("")
            self.trigger2_var.set("")
            fsm_note = ""
            self._draw_flow(None, None, -1)
        else:
            st = dr["state"]
            label = st.value if hasattr(st, "value") else str(st)
            flags = dr.get("flags") or frozenset()
            flag_s = " ".join(sorted((f.value for f in flags), key=lambda s: s))
            self.state_var.set(f"FSM状态: {label}" + (f"  | {flag_s}" if flag_s else ""))
            self.intent_var.set(str(dr.get("intent_label") or ""))
            self.send_var.set(self._send_text(dr))
            warm = "  【预热】" if FsmFlag.WARMUP in flags else ""
            self.trigger_var.set(f"触发: {dr['why']}{warm}")
            self.trigger2_var.set(
                f"怪物分布={dr.get('dist_key') or '—'}【{('BOSS' if dr.get('dist_kind')=='boss' else '小怪' if dr.get('dist_kind')=='mob' else dr.get('dist_kind') or '—')}】  "
                f"已过房间={dr.get('gate_crosses', 0)}  击败BOSS={dr.get('boss_kills', 0)}  "
                f"状态连续 {dr['fsm_run']} 帧 / {dr['fsm_run_s']:.2f}s  "
                f"(X={x_lim:g}s GX={self._spin_n(self.gx_var, 50, lo=0)} GY={self._spin_n(self.gy_var, 10, lo=0)} "
                f"AX={self._spin_n(self.ax_var, 250, lo=0)}ms AY={self._spin_n(self.ay_var, 250, lo=0)}ms "
                f"S={self._spin_n(self.s_var, 20, lo=0)}%)"
            )
            self._draw_flow(st, dr.get("flow_steps") or (), int(dr.get("flow_hit") if dr.get("flow_hit") is not None else -1))
            fsm_note = f"  状态连续={dr['fsm_run']}帧"
        resets = d.get("cd_resets") or []
        reset_here = bool((dr or {}).get("cd_reset"))
        cd_bits = []
        for slot, key, rem in (dr or {}).get("skill_cds") or ():
            if rem and rem > 0:
                cd_bits.append(f"{slot}{key or ''} {rem:.1f}s")
            else:
                cd_bits.append(f"{slot}{key or ''} 就绪")
        live = "CD " + " | ".join(cd_bits) if cd_bits else ""
        if has_map_reset(self._feature_cache):
            tag = "【本帧重置】 " if reset_here else "【CD重置】击败BOSS+1清CD "
            ev = ""
            if resets:
                ev = "  短间隔证据 " + "；".join(
                    f"t={r.get('t_s')}s/帧{int(r.get('i') or 0)+1}" for r in resets
                )
            self.cd_var.set((live + "  " if live else "") + tag + ev)
        else:
            self.cd_var.set(live)
        infer = fr.get("infer_ms")
        infer_s = f"  推理={infer:.0f}ms" if infer is not None else ""
        stuck = d["stuck"][i]
        has_xy = d.get("has_class_xy")
        views = d.get("views") or []
        view = views[i] if i < len(views) else None
        xy_note = "" if has_xy else "  旧段无 mon/loot/gate/boss 坐标"
        mode = "相对" if self.rel_var.get() else "左下角绝对"
        fill_on = "沿用" if self.fill_var.get() else "不填"
        if view is None:
            origin_s = "view=—"
        elif view.get("player_xy") is None:
            origin_s = f"{mode}/{fill_on} player=null"
        else:
            tag = "fill" if view["player_hold"] else "本帧"
            origin_s = f"{mode}/{fill_on} xy={view['player_xy']}({tag}) origin_bl={view.get('origin_bl')}"
        self.info_var.set(
            f"t={t:7.2f}s{dt_s}  特征未变={stuck:5.2f}s{fsm_note}  {origin_s}  "
            f"盘面左上={fr.get('player_xy')}{infer_s}  {d['session'].name}{xy_note}"
        )
        skill_range = None
        if skill_hit:
            room_n = (dr or {}).get("dist_key") or (dr or {}).get("rooms") or 0
            skill_range = skill_range_of(self._feature_cache, room_n, int(skill_hit[0]))
            now_dist = pack_dist_from_player(view)
            bits = [f"技能{skill_hit[0]}：{skill_hit[1]}"]
            if skill_range is not None:
                bits.append(f"范围{skill_range:.0f}")
            if now_dist is not None:
                bits.append(f"现距{now_dist:.0f}")
            self.action_var.set("  ".join(bits))
        if not d.get("keys_path_exists"):
            self.keys_all_var.set("本段键: （没有 keys.jsonl，这段不能训过图）")
            self.keys_var.set("按住: （无文件）")
            self.keys_edge_var.set("本帧沿(上一帧→本帧]: —")
        elif not d["keys"] and not d.get("held_at_start"):
            self.keys_all_var.set("本段键: （keys.jsonl 为空）")
            self.keys_var.set("按住: （无）")
            self.keys_edge_var.set("本帧沿(上一帧→本帧]: —")
        else:
            vocab = d.get("keys_vocab") or []
            if not held:
                held = rebuild_held_at(d["keys"], int(fr["t_ns"]), d.get("held_at_start"))
            edge_s = "  ".join(f"{e.get('type')}:{e.get('key')}" for e in edges[:24])
            if len(edges) > 24:
                edge_s += f"  +{len(edges) - 24}"
            self.keys_all_var.set("本段键: " + (" ".join(vocab) if vocab else "（无）"))
            self.keys_var.set("按住: " + (" ".join(held) if held else "（无）"))
            self.keys_edge_var.set(
                "本帧沿(上一帧→本帧]: " + (edge_s if edge_s else "（无 press/release）")
            )
        self._draw_field(
            view,
            d["game_w"],
            d["game_h"],
            has_xy=bool(has_xy),
            skill_range=skill_range,
            cd_reset=reset_here,
            counts=counts,
            overlay_path=self._png_path_for_frame(fr) if self.overlay_var.get() else None,
            player_tl=self._overlay_player_tl(fr, view, d.get("game_h") or 0),
        )

    def _session_rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    def _reload_feature_cache(self):
        char = self._bind_character
        dun = self._current_dungeon() if self.data else ""
        if char and dun and dun != "未知":
            self._feature_cache = load_features(char, dun)
        else:
            self._feature_cache = None
        self._rebuild_map_reset_ui()

    def _edit_slots_from_ui(self) -> set[int]:
        return (
            {int(s) for s in self._skill_hold_vars}
            | {int(s) for s in self._combo_vars}
            | {int(s) for s in self._mash_vars}
        )

    def _combo_slots_from_ui(self) -> set[int]:
        if self._combo_vars:
            return {slot for slot, var in self._combo_vars.items() if var.get()}
        return combo_slots_of(self._feature_cache)

    def _mash_slots_from_ui(self) -> set[int]:
        return {slot for slot, var in self._mash_vars.items() if var.get()}

    def _multi_n_from_ui(self) -> dict[int, int]:
        if self._multi_vars:
            out: dict[int, int] = {}
            for slot, var in self._multi_vars.items():
                if not var.get():
                    continue
                nvar = self._multi_n_vars.get(slot)
                try:
                    n = int(str((nvar.get() if nvar else "") or DEFAULT_MULTI))
                except ValueError:
                    n = DEFAULT_MULTI
                out[int(slot)] = max(DEFAULT_MULTI, min(9, n))
            return out
        return dict(multi_n_of(self._feature_cache))

    def _rebuild_map_reset_ui(self):
        host = getattr(self, "map_reset_host", None)
        if host is None:
            return
        for child in host.winfo_children():
            child.destroy()
        map_row = ttk.Frame(host)
        map_row.pack(fill=tk.X)
        if has_map_reset(self._feature_cache):
            ttk.Label(map_row, text=f"地图特征:【{MAP_RESET}】", foreground="#e87a2a").pack(side=tk.LEFT)
            ttk.Button(map_row, text=f"删除【{MAP_RESET}】", command=self._delete_map_reset).pack(
                side=tk.LEFT, padx=(8, 0)
            )
        else:
            ttk.Label(map_row, text="地图特征: 无【CD重置】", foreground="#99a").pack(side=tk.LEFT)

    def _delete_map_reset(self):
        char = self._bind_character
        dun = self._current_dungeon() if self.data else ""
        data = self._feature_cache
        if not data or not char or not dun or dun == "未知":
            return
        if not messagebox.askyesno(
            "删除【CD重置】",
            "从该图特征表去掉【CD重置】。击败 BOSS 时将不再清 CD。确定？",
            parent=self,
        ):
            return
        set_map_reset(data, False)
        save_features(char, dun, data, keep_prev=False)
        self._feature_cache = data
        self._rebuild_map_reset_ui()
        self._set_extract_result(format_report(data))
        if self.data:
            self._rebuild_draft()
            self._show()

    def _on_combo_toggle(self):
        if self._combo_ui_guard or self._skill_ui_guard:
            return
        self._sync_combo_to_features(save=True)
        if self.data:
            self._rebuild_draft()
            self._show()

    def _on_mash_toggle(self):
        if getattr(self, "_mash_ui_guard", False) or self._skill_ui_guard:
            return
        self._write_hold_frames_to_binds()
        if self.data:
            self._rebuild_draft()
            self._show()

    def _on_multi_toggle(self):
        if self._multi_ui_guard or self._skill_ui_guard:
            return
        for slot, var in self._multi_vars.items():
            sp = self._multi_spins.get(slot)
            if sp is None:
                continue
            on = bool(var.get())
            sp.config(state=tk.NORMAL if on else tk.DISABLED)
            if on:
                nvar = self._multi_n_vars.get(slot)
                try:
                    n = int(str((nvar.get() if nvar else "") or DEFAULT_MULTI))
                except ValueError:
                    n = 0
                if nvar is not None and n < DEFAULT_MULTI:
                    nvar.set(str(DEFAULT_MULTI))
        self._write_hold_frames_to_binds()
        self._sync_multi_to_features(save=True)
        if self.data:
            self._rebuild_draft()
            self._show()

    def _on_multi_n_change(self):
        if self._multi_ui_guard or self._skill_ui_guard or not getattr(self, "_persist_ok", False):
            return
        self._write_hold_frames_to_binds()
        self._sync_multi_to_features(save=True)
        if self.data:
            self._rebuild_draft()
            self._show()

    def _current_dungeon(self) -> str:
        d = self.data
        if not d:
            return "未知"
        return dungeon_of_session(d["session"], d.get("meta") or {})

    def _refresh_rollback_btn(self):
        char = self._bind_character
        dun = self._current_dungeon() if char else ""
        on = bool(char and dun and prev_path(char, dun).is_file())
        try:
            self.rollback_btn.config(state=tk.NORMAL if on else tk.DISABLED)
        except tk.TclError:
            pass

    def _set_extract_result(self, text: str):
        self.extract_result.config(state=tk.NORMAL)
        self.extract_result.delete("1.0", tk.END)
        self.extract_result.insert(tk.END, text)
        self.extract_result.config(state=tk.DISABLED)

    def _tracks_for_extract(self, data: dict, slot_durs: dict[int, int] | None = None):
        frames = data.get("frames") or []
        draft = compute_draft_track(
            frames,
            **self._fsm_track_kwargs(),
        )
        skill_hold = skill_hold_track(
            frames,
            data.get("keys") or [],
            self._hotbar_skills,
            slot_durs if slot_durs is not None else self._slot_durs(),
        )
        gh = data.get("game_h")
        views = views_for_frames(
            frames,
            fill=True,
            relative=True,
            height=float(gh) if gh else None,
        )
        return draft, skill_hold, views

    def _sessions_for_extract(self, all_same: bool) -> list[Path]:
        d = self.data
        if not d:
            return []
        if not all_same:
            return [d["session"]]
        char = self._bind_character
        dun = self._current_dungeon()
        out = []
        for p in list_sessions(RECORDINGS, FSM_TEST_DIR):
            meta = {}
            mp = p / "meta.json"
            if mp.is_file():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            if character_of_session(p, meta) == char and dungeon_of_session(p, meta) == dun:
                out.append(p)
        return out

    def _start_extract(self, all_same: bool, wipe: bool = False):
        if self._extract_busy:
            return
        char = self._bind_character
        if not char:
            messagebox.showinfo("过图技能特征", "当前录像没有角色名，没法写入特征表。")
            return
        dun = self._current_dungeon()
        if not dun or dun == "未知":
            messagebox.showinfo("过图技能特征", "当前录像没有地下城名，没法按图分类写入。")
            return
        if not self._hotbar_skills:
            messagebox.showinfo("过图技能特征", "键位表没有快捷栏单键（或 space），黄字技能段为空。")
            return
        sessions = self._sessions_for_extract(True if wipe else all_same)
        if not sessions:
            messagebox.showinfo("过图技能特征", "没有可提取的录像。")
            return
        dest = feature_path(char, dun)
        existing = load_features(char, dun)
        if existing is None and not wipe:
            old = legacy_feature_path(char)
            if old.is_file():
                try:
                    raw = json.loads(old.read_text(encoding="utf-8"))
                    existing = raw if isinstance(raw, dict) else None
                except Exception:
                    existing = None
        have = count_casts(existing)
        e = self._spin_n(self.e_var, 3, lo=0)
        if wipe:
            ok = messagebox.askyesno(
                "确认重新提取本图",
                f"将清空\n{dest}\n里现有 {have} 条，再从同地下城同角色全部 {len(sessions)} 段重新提取。\n\n"
                f"杀MON数>E={e} 记为群。清空前另存一份副本，可用「回退上次叠加」还原（用过即删）。\n\n"
                f"确定重新提取？",
                parent=self,
            )
        else:
            scope = f"同地下城同角色全部 {len(sessions)} 段" if all_same else "当前这一段"
            ok = messagebox.askyesno(
                "确认提取并叠加",
                f"从{scope}提取快捷栏技能段（含 SPACE，不要求开打），叠加进\n{dest}\n\n"
                f"现有 {have} 条。杀MON数>E={e} 记为群。\n"
                f"同一段、同一起始帧、同一技能槽已经在表里的会跳过"
                f"（避免对同一录像点两次提取写出重复）。\n"
                f"若表已存在，叠加前另存一份未叠加副本，\n"
                f"可用「回退上次叠加」一次性还原（用过即删）。\n\n确定提取？",
                parent=self,
            )
        if not ok:
            return
        self._flush_skill_hold_saved()
        self._write_hold_frames_to_binds()
        slot_durs = dict(self._slot_durs())
        for sk in self._bind_skills:
            slot = int(sk["slot"])
            if slot in slot_durs:
                sk[HOLD_FRAMES_KEY] = slot_durs[slot]
        f = self._spin_n(self.f_var, DEFAULT_F, lo=0)
        self._extract_busy = True
        self.extract_one_btn.config(state=tk.DISABLED)
        self.extract_all_btn.config(state=tk.DISABLED)
        self.extract_reset_btn.config(state=tk.DISABLED)
        self.rollback_btn.config(state=tk.DISABLED)
        self.extract_prog["value"] = 0
        self.extract_prog["maximum"] = max(len(sessions), 1)
        self.extract_prog_var.set("提取 0/" + str(len(sessions)))
        threading.Thread(
            target=self._extract_worker,
            args=(char, dun, sessions, e, wipe, set(self._combo_slots_from_ui()), dict(self._multi_n_from_ui()), slot_durs, f),
            daemon=True,
        ).start()

    def _extract_worker(
        self,
        char: str,
        dun: str,
        sessions: list[Path],
        e: int,
        wipe: bool = False,
        combo_slots=None,
        multi_n=None,
        slot_durs=None,
        f: int = DEFAULT_F,
    ):
        all_casts = []
        all_resets = []
        total = len(sessions)
        err = ""
        combo_slots = set(combo_slots or [])
        multi_n = dict(multi_n or {})
        slot_durs = dict(slot_durs or {})
        multi_slots = set(multi_n)
        table = () if wipe else dist_table_from_features(self._feature_cache)
        s_pct = self._spin_n(self.s_var, 20, lo=0)
        extra_sigs = ()
        try:
            for i, path in enumerate(sessions):
                self.after(
                    0,
                    lambda i=i, t=total, n=path.name: self._extract_progress(i, t, n),
                )
                if path == (self.data or {}).get("session"):
                    data = self.data
                else:
                    data = load_session(path)
                draft, skill_hold, views = self._tracks_for_extract(data, slot_durs)
                rel = self._session_rel(path)
                casts, extra_sigs = casts_from_tracks(
                    rel,
                    draft,
                    skill_hold,
                    views,
                    e,
                    self._bind_skills,
                    dist_table=table,
                    s=s_pct,
                    extra_sigs=extra_sigs,
                    slot_durs=slot_durs,
                )
                all_casts.extend(casts)
                skip_ids = {str(c["id"]) for c in casts if c.get("id") and cast_is_fake(c, f)}
                all_resets.extend(
                    detect_cd_resets(
                        rel,
                        data.get("frames") or [],
                        draft,
                        skill_hold,
                        self._bind_skills,
                        combo_slots=combo_slots,
                        multi_slots=multi_slots,
                        skip_ids=skip_ids,
                    )
                )
            if wipe:
                clear_features(char, dun)
                existing = None
                dest = feature_path(char, dun)
                keep_prev = False
            else:
                maybe_adopt_legacy(char, dun)
                dest = feature_path(char, dun)
                existing = load_features(char, dun)
                keep_prev = dest.is_file()
            merged, added, skipped = merge_casts(
                existing,
                all_casts,
                char,
                dun,
                e,
                resets=all_resets,
                f=self._spin_n(self.f_var, DEFAULT_F, lo=0),
            )
            set_combo_slots(merged, combo_slots)
            set_multi_slots(merged, multi_n)
            save_features(char, dun, merged, keep_prev=keep_prev)
            report = format_report(merged, added=added, skipped=skipped)
            self.after(0, lambda r=report, a=added, s=skipped: self._extract_done(True, r, a, s))
        except Exception as ex:
            err = str(ex)
            self.after(0, lambda msg=err: self._extract_done(False, msg, 0, 0))

    def _extract_progress(self, i: int, total: int, name: str):
        self.extract_prog["maximum"] = max(total, 1)
        self.extract_prog["value"] = i
        self.extract_prog_var.set(f"提取 {i}/{total}  {name}")

    def _extract_done(self, ok: bool, report: str, added: int, skipped: int):
        self._extract_busy = False
        self.extract_one_btn.config(state=tk.NORMAL)
        self.extract_all_btn.config(state=tk.NORMAL)
        self.extract_reset_btn.config(state=tk.NORMAL)
        self._refresh_rollback_btn()
        self._reload_feature_cache()
        if ok:
            self.extract_prog["value"] = self.extract_prog["maximum"]
            self.extract_prog_var.set(f"完成  +{added}  跳过{skipped}")
            self._set_extract_result(report)
            self._show_extract_window(report)
            if self.data:
                self._rebuild_draft()
            self._show()
        else:
            self.extract_prog_var.set("失败")
            self._set_extract_result(report)
            messagebox.showerror("过图技能特征", report, parent=self)

    def _show_extract_window(self, report: str):
        win = tk.Toplevel(self)
        win.title("过图技能特征 · 结果")
        win.geometry("720x480")
        apply_window_geom(win, "fsm_extract", min_w=480, min_h=280, fallback="720x480")
        txt = tk.Text(win, font=("Consolas", 10))
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", report)
        txt.config(state=tk.DISABLED)

        def _close():
            remember_window_geom(win, "fsm_extract")
            win.destroy()

        ttk.Button(win, text="关闭", command=_close).pack(pady=6)
        win.protocol("WM_DELETE_WINDOW", _close)

    def _rollback_extract(self):
        char = self._bind_character
        dun = self._current_dungeon()
        if not char or not dun:
            return
        if not messagebox.askyesno(
            "回退上次叠加",
            "用叠加前的副本覆盖当前特征表，然后删掉该副本（一次性）。确定？",
            parent=self,
        ):
            return
        ok, msg = rollback_features(char, dun)
        self._refresh_rollback_btn()
        self._reload_feature_cache()
        if ok:
            data = load_features(char, dun)
            self._set_extract_result(format_report(data))
            self._show()
            messagebox.showinfo("回退上次叠加", msg, parent=self)
        else:
            messagebox.showinfo("回退上次叠加", msg, parent=self)

    def _png_path_for_frame(self, fr: dict) -> Path | None:
        d = self.data
        if not d:
            return None
        name = str(fr.get("png") or "").strip()
        if not name:
            return None
        hit = png_file_of(d["session"], d.get("meta") or {}, name)
        if hit is not None:
            d["png_dir"] = hit.parent
        return hit

    def _overlay_player_tl(self, fr: dict, view: dict | None, gh: float) -> list[float] | None:
        raw = player_abs(fr)
        if raw is not None:
            return raw
        if not self.fill_var.get() or not view:
            return None
        origin = view.get("origin_bl")
        if not origin or len(origin) < 2 or not gh:
            return None
        return [float(origin[0]), float(gh) - float(origin[1])]

    def _overlay_photo(
        self,
        path: Path,
        gw: int,
        gh: int,
        cw: int,
        ch: int,
        relative: bool,
        player_tl: list[float] | None,
        alpha_pct: int,
    ) -> tuple[ImageTk.PhotoImage | None, float, float]:
        try:
            alpha_pct = max(8, min(100, int(alpha_pct)))
        except (TypeError, ValueError):
            alpha_pct = 45
        px = round(player_tl[0], 2) if player_tl and len(player_tl) >= 2 else None
        py = round(player_tl[1], 2) if player_tl and len(player_tl) >= 2 else None
        key = (str(path), cw, ch, gw, gh, relative, px, py, alpha_pct)
        if key == self._frame_overlay_key and self._frame_overlay_photo is not None:
            ox = cw / 2.0 - px / gw * cw if relative and px is not None else 0.0
            oy = ch / 2.0 - py / gh * ch if relative and py is not None else 0.0
            return self._frame_overlay_photo, ox, oy
        if relative and (px is None or py is None):
            self._frame_overlay_photo = None
            self._frame_overlay_key = None
            return None, 0.0, 0.0
        try:
            img = Image.open(path).convert("RGBA")
        except OSError:
            self._frame_overlay_photo = None
            self._frame_overlay_key = None
            return None, 0.0, 0.0
        iw, ih = img.size
        dw = max(1, int(round(iw / max(gw, 1) * cw)))
        dh = max(1, int(round(ih / max(gh, 1) * ch)))
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img = img.resize((dw, dh), resample)
        a = int(round(255 * alpha_pct / 100.0))
        r, g, b, aa = img.split()
        aa = aa.point(lambda v, m=a: int(v * m / 255))
        img = Image.merge("RGBA", (r, g, b, aa))
        photo = ImageTk.PhotoImage(img)
        self._frame_overlay_photo = photo
        self._frame_overlay_key = key
        ox = cw / 2.0 - px / gw * cw if relative else 0.0
        oy = ch / 2.0 - py / gh * ch if relative else 0.0
        return photo, ox, oy

    def _draw_field(
        self,
        view: dict | None,
        gw: int,
        gh: int,
        has_xy: bool = False,
        skill_range: float | None = None,
        cd_reset: bool = False,
        counts: dict[str, int] | None = None,
        overlay_path: Path | None = None,
        player_tl: list[float] | None = None,
    ):
        c = self.canvas
        cw, ch = self._field_wh()
        c.delete("all")
        c.create_rectangle(1, 1, cw - 2, ch - 2, outline="#445")
        relative = bool(self.rel_var.get())
        cx0, cy0 = cw / 2, ch / 2
        if relative:
            c.create_line(cx0, 0, cx0, ch, fill="#334")
            c.create_line(0, cy0, cw, cy0, fill="#334")

        if gw > 0 and gh > 0:
            step = self._ruler_step(gw, gh, cw, ch)
            sx = step * cw / gw
            sy = step * ch / gh
            x = sx
            while x < cw:
                c.create_line(x, 0, x, ch, fill="#252830")
                x += sx
            y = sy
            while y < ch:
                c.create_line(0, y, cw, y, fill="#252830")
                y += sy
        else:
            step = 0
            for x in range(0, cw, 80):
                c.create_line(x, 0, x, ch, fill="#252830")
            for y in range(0, ch, 45):
                c.create_line(0, y, cw, y, fill="#252830")

        if relative:
            hint = "相对 player（y 向上）"
        else:
            hint = "区域左下角绝对（y 向上）"
        if not has_xy:
            hint += " · 旧段仅 player"

        if view is not None and gw > 0 and gh > 0:
            def to_canvas(pt: list[float]) -> tuple[float, float]:
                if relative:
                    return cx0 + pt[0] / gw * cw, cy0 - pt[1] / gh * ch
                return pt[0] / gw * cw, ch - pt[1] / gh * ch

            order = ("mon", "loot", "gate", "boss", "player")
            for name in order:
                photo = self._stack_photos.get(name) if name != "player" else self._player_photo
                if name == "player":
                    pxy = view.get("player_xy")
                    pts = [pxy] if pxy is not None else []
                else:
                    pts = view.get(f"{name}_xy") or []
                for pt in pts:
                    px, py = to_canvas(pt)
                    if photo:
                        c.create_image(px, py, image=photo)
                    else:
                        c.create_oval(px - 6, py - 6, px + 6, py + 6, fill="#6cf")
            if relative and view.get("player_xy") is None:
                c.create_text(cx0, cy0, fill="#a64", text="无 player 原点（可勾选沿用上次）")
            elif view.get("player_hold"):
                c.create_text(cw / 2, 40, fill="#a84", text="player 漏检，位置沿用上次")
            if skill_range is not None and skill_range > 0:
                pxy = view.get("player_xy") if view else None
                if isinstance(pxy, (list, tuple)) and len(pxy) >= 2:
                    px, py = to_canvas(pxy)
                    rx = skill_range / gw * cw
                    ry = skill_range / gh * ch
                    c.create_oval(
                        px - rx,
                        py - ry,
                        px + rx,
                        py + ry,
                        outline="#e8c547",
                        width=2,
                        dash=(8, 4),
                    )
                    c.create_text(
                        px,
                        py - ry - 10,
                        fill="#e8c547",
                        font=("Consolas", 12, "bold"),
                        text=f"范围 {skill_range:.0f}px",
                    )

        if overlay_path is not None and gw > 0 and gh > 0 and self.overlay_var.get():
            photo, ox, oy = self._overlay_photo(
                overlay_path,
                gw,
                gh,
                cw,
                ch,
                relative,
                player_tl,
                self.overlay_alpha_var.get(),
            )
            if photo is not None:
                c.create_image(ox, oy, image=photo, anchor=tk.NW)
        elif not self.overlay_var.get():
            self._frame_overlay_photo = None
            self._frame_overlay_key = None

        if step:
            self._draw_edge_rulers(c, cw, ch, gw, gh, relative, step)
        self._draw_count_overlay(c, counts or {}, x=(44 + 8) if step else 10, y=8)
        c.create_text(cw - 8, 8, anchor=tk.NE, fill="#668", font=("Microsoft YaHei", 9), text=hint)
        if cd_reset:
            c.create_text(
                cw / 2,
                22,
                fill="#ff9a3c",
                font=("Microsoft YaHei", 18, "bold"),
                text="CD重置",
            )

    @staticmethod
    def _ruler_step(gw: int, gh: int, cw: int, ch: int) -> int:
        span = max(gw / max(cw, 1), gh / max(ch, 1))
        for n in (50, 100, 200, 250, 400, 500):
            if n / span >= 64:
                return n
        return max(25, int(round(80 * span)))

    def _load_count_overlay_font(self):
        if self._count_overlay_font is not None:
            return self._count_overlay_font
        fonts = Path(r"C:\Windows\Fonts")
        for name in ("msyh.ttc", "msyhbd.ttc", "msyh.ttf", "simhei.ttf", "consola.ttf"):
            p = fonts / name
            if not p.is_file():
                continue
            try:
                self._count_overlay_font = ImageFont.truetype(str(p), 16)
                return self._count_overlay_font
            except OSError:
                continue
        self._count_overlay_font = ImageFont.load_default()
        return self._count_overlay_font

    def _draw_count_overlay(self, c: tk.Canvas, counts: dict[str, int], x: float, y: float):
        """半透明数量字叠在坐标系左上，不占右侧栏。"""
        lines = [f"{CLASS_CN[name]}  {int(counts.get(name) or 0)}" for name in CLASSES]
        font = self._load_count_overlay_font()
        probe = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        probe_d = ImageDraw.Draw(probe)
        max_w = 0
        line_h = 0
        for line in lines:
            box = probe_d.textbbox((0, 0), line, font=font)
            max_w = max(max_w, box[2] - box[0])
            line_h = max(line_h, box[3] - box[1])
        gap, pad = 3, 4
        img = Image.new(
            "RGBA",
            (max(1, max_w + pad * 2), max(1, len(lines) * (line_h + gap) + pad * 2 - gap)),
            (0, 0, 0, 0),
        )
        draw = ImageDraw.Draw(img)
        cy = pad
        for line in lines:
            draw.text((pad + 1, cy + 1), line, font=font, fill=(12, 14, 18, 90))
            draw.text((pad, cy), line, font=font, fill=(236, 242, 250, 150))
            cy += line_h + gap
        self._count_overlay_photo = ImageTk.PhotoImage(img)
        c.create_image(x, y, image=self._count_overlay_photo, anchor=tk.NW)

    @staticmethod
    def _tick_values(lo: float, hi: float, step: int) -> list[int]:
        if step <= 0:
            return []
        a = int(lo // step) * step
        if a < lo - 1e-6:
            a += step
        out = []
        v = a
        while v <= hi + 1e-6:
            out.append(int(v))
            v += step
        return out

    def _draw_edge_rulers(
        self,
        c: tk.Canvas,
        cw: int,
        ch: int,
        gw: int,
        gh: int,
        relative: bool,
        step: int,
    ):
        """左、下边缘尺子：刻度 = 画面像素。"""
        band_l, band_b = 44, 22
        c.create_rectangle(0, ch - band_b, cw, ch, fill="#12151b", outline="")
        c.create_rectangle(0, 0, band_l, ch, fill="#12151b", outline="")
        c.create_line(band_l, 0, band_l, ch - band_b, fill="#3a4555")
        c.create_line(band_l, ch - band_b, cw, ch - band_b, fill="#3a4555")

        if relative:
            x0, x1 = -gw / 2.0, gw / 2.0
            y0, y1 = -gh / 2.0, gh / 2.0

            def x_to_c(gx: float) -> float:
                return cw / 2.0 + gx / gw * cw

            def y_to_c(gy: float) -> float:
                return ch / 2.0 - gy / gh * ch
        else:
            x0, x1 = 0.0, float(gw)
            y0, y1 = 0.0, float(gh)

            def x_to_c(gx: float) -> float:
                return gx / gw * cw

            def y_to_c(gy: float) -> float:
                return ch - gy / gh * ch

        minor = step // 2 if step >= 100 else 0
        tick_col = "#8a96a8"
        label_col = "#e8c547"
        font = ("Consolas", 11, "bold")
        y_edge = ch - band_b
        if minor:
            for gx in self._tick_values(x0, x1, minor):
                if gx % step == 0:
                    continue
                px = x_to_c(gx)
                if px < band_l or px > cw - 2:
                    continue
                c.create_line(px, y_edge, px, y_edge + 7, fill=tick_col)
        for gx in self._tick_values(x0, x1, step):
            px = x_to_c(gx)
            if px < band_l - 1 or px > cw - 2:
                continue
            c.create_line(px, y_edge, px, ch - 2, fill=label_col, width=2)
            c.create_text(px, ch - 3, anchor=tk.S, text=str(gx), fill=label_col, font=font)

        x_edge = band_l
        if minor:
            for gy in self._tick_values(y0, y1, minor):
                if gy % step == 0:
                    continue
                py = y_to_c(gy)
                if py < 2 or py > y_edge:
                    continue
                c.create_line(x_edge - 7, py, x_edge, py, fill=tick_col)
        for gy in self._tick_values(y0, y1, step):
            py = y_to_c(gy)
            if py < 10 or py > y_edge:
                continue
            c.create_line(2, py, x_edge, py, fill=label_col, width=2)
            c.create_text(4, py, anchor=tk.W, text=str(gy), fill=label_col, font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="", help="某一段目录（recordings/ 或 FSM_TEST/）")
    args = ap.parse_args()
    start = Path(args.session) if args.session.strip() else None
    app = FsmReplayApp(start_session=start)
    app.mainloop()


if __name__ == "__main__":
    main()
