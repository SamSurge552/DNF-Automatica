"""回放 recordings/ 与 FSM_TEST/ 的 frames.jsonl：站在模型视角看结构化标签。

图标对应 YOLO 写入 jsonl 的字段。可选叠 PNG 对比（与检测同一套截屏像素）。
旧段只有 player_xy、其它类是计数；新段还有 mon_xy/loot_xy/gate_xy/boss_xy。

用法:
  python fsm_replay.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
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
    apply_mon_boss_corr,
    apply_loot_corr,
    apply_gate_corr,
    apply_detect_xy_corr,
    mon_corr_from_dict,
    loot_corr_from_dict,
    gate_corr_from_dict,
    mon_off_from_corr,
    DEFAULT_PW_MS,
    DEFAULT_TOWN_S,
    DEFAULT_ADVANCE_TIMEOUT_MS,
    DEFAULT_APPROACH_TIMEOUT_MS,
    DEFAULT_ADVANCE_TIMEOUT_ESC_N,
    DEFAULT_STUCK_RECOVER_S,
    DEFAULT_RANGE_X,
    DEFAULT_RANGE_Y,
    parse_stuck_recover,
    stuck_recover_to_json,
    send_keys_label,
    json_ms,
    json_s,
    parse_hold_ms,
    warn_legacy_key,
    HOLD_MS_KEY,
)
from fsm_execute import (
    mash_n_for_slot,
    parse_mash_n,
    parse_tap_ms_range,
    DEFAULT_TAP_MS_MIN,
    DEFAULT_TAP_MS_MAX,
    DEFAULT_MASH_COUNT,
    MASH_COUNT_MIN,
    MASH_COUNT_MAX,
    MASH_N_KEY,
    clamp_mash_count,
)
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
    skill_range_xy_of,
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
    EXTRACT_PRESS_LAG_FRAMES,
    parse_multi_n,
    cast_is_fake,
    fake_member_ids,
)

ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "recordings"
FSM_TEST_DIR = ROOT / "FSM_TEST"
IMAGES = ROOT / "images"
MISS_PNG_DIR = IMAGES / "miss_png"  # 漏检图收集：放 images 下，避免根目录自动上传
ICON_DIR = ROOT / "fsm_icons"
SETTINGS_PATH = ROOT / "_fsm_replay_ui.json"
BINDS_DIR = ROOT / "skill_binds"
FEATURES_DIR = ROOT / "skill_features"
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
    "S_pos%：分布中心相似。S_size%：多 MON 范围框尺寸相似。全表取最相似再过阈值。\n"
    "F%：杀 MON 效率低于该百分比记假释放（不进序列 / 范围 / CD）。\n"
    "PM：掉落位移超过该像素算在动。连续 PT 帧不动才判定停下。PW：等停下上限毫秒，超时结束等待并按当前掉落继续捡（蓝字「捡物等待超时 PW」）。数量 > PC 一键拾取，否则挨个捡。\n"
    "卡住：同一流程状态标签连续 X 秒则改为卡住（同标签方法重入含前进/走近超时不重置计时）。然后跑配置序列 stuck_recover（默认：ESC 点按，再右/左 HOLD 2×恢复s、上/下 HOLD 1×恢复s）。步骤在 json 的 stuck_recover 里改，不必改代码。基准秒字段 stuck_recover_s。一轮跑完退出卡住并重置监测（可再入）。X 秒应大于 max(最长技能持续, AX, AY, PW, 恢复序列 HOLD)（秒）。\n"
    "GX/GY：相对当前门，距离大于该像素才继续接近。接近与捡物依次走：点按方向 → 等移动间隔 TH 毫秒 → 按住（TH 默认等于连按间隔）。开打范围外同样走近（朝分布中心）。AX/AY：两轴停后、或前进时门消失，按过门方向走的毫秒。前进超时：方法开始计时，超过 advance_timeout_ms 结束本次前进并重入状态判定（重识门/重记过门方向）。连续超时达 advance_timeout_esc_n 次点按 ESC 并清零计数；正常过门也清零。走近超时：开打范围外走近超过 approach_timeout_ms 结束本次走近并重选技能（不进卡住、不强行对该技能 CAST/CD）。技能无提取 XY 且无 range_px 时用 default_range_x/y 矩形（默认 250/100）。XXX：快捷栏全 CD 时按住普攻 X 的毫秒。\n"
    "点按 min/max：发键层每次点按（移动 TAP、技能 CAST/连按每次、左Alt 拾取）按下保持 uniform[min,max] 毫秒。默认 80–120。HOLD/普攻不是点按。PICK 节流 = max×5。与主面板共用 json。\n"
    "连按 COUNT / 间隔 ms：全局默认（不进核心）。该槽有 mash_n 用槽值，否则用这里的 COUNT。红字按槽 COUNT 拼。\n"
    "MON/BOSS、LOOT、GATE 补正：各类上下左右滑块。FSM / 回放逻辑点用同一偏移；jsonl 与叠图 PNG 仍是原始检测。改 MON/BOSS 补正后旧 skill_features 作废，请重新提取本图；改 LOOT/GATE 不必重提。\n"
    "叠图：把该帧 PNG 铺到坐标网上（半透明、最上层）。对齐=截屏像素与 jsonl 同一套；相对坐标时图平移使 player 落在盘面中心。\n"
    "改完立刻写入共用 json，回放当场重算；FSM测试点开始（或测试中再改）会读这份，不必另同步。"
)
HELP_SKILLS = (
    "只列出已勾选快捷栏的技能（space 即使未勾快捷栏也保留）。普攻 X 无 CD，不在这张表里。\n"
    "多次释放：勾选后填 MULTI（默认 2），FSM 拆成多份独立 CD。\n"
    "连按：执行层按该槽 COUNT 点按（无槽值用参数区全局 COUNT；间隔仍全局）。勾选连按=连续释放（CD 270 秒、提取 CD 重置跳过）；表上不再单独一列连续。\n"
    "多次释放不并进连按；连按/多次都会让提取跳过该槽的 CD 重置启发。\n"
    "持续/多次/连按会在切录像、勾选变更、提取前、关闭回放时自动写入该角色键位表。键位工具改完后点「重新读取」。"
)

HELP_EXTRACT = (
    "按下快捷栏技能（含 SPACE）即提取，不要求开打。hold_ms 未结束又按下下一个 → 技能组（group_key 如 a>b）。\n"
    "组结束 = 组内 (按下+hold_ms) 最大。组间隔 gaps_ms 取中位数，FSM 按间隔复现。起始数量 = 按下时 MON+BOSS。效率 100% 才记范围框；不足不计范围。该分布对已记范围取最大值。\n"
    "持续结束后杀 MON 数 > E → 群，否则为单。\n"
    "杀 MON 效率 < F% → 假释放（不进序列 / 范围 / CD）。\n"
    "怪物分布按 onset 后 press_lag 帧快照现算（与黄字延后同一值），不沿用 FSM 当时的状态。\n"
    "lag 只延后特征快照和黄字显示；hold/CD 仍按真实按下。改 lag 后须重新提取。\n"
    "MON/BOSS 用当前补正（与 FSM 同一套）；jsonl 不改。改补正后请重新提取本图。\n"
    "地图【CD重置】：提取发现短于 CD 的间隔后写入；运行时击败 BOSS +1 才清 CD。\n"
    "「提取同地下城全部 / 重新提取本图」默认只扫 recordings/ 采集段。勾「含FSM测试」才并入 FSM_TEST。提取本段始终用当前这一段。"
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


def session_storage_root(session: Path) -> Path | None:
    """仅允许删 recordings/ 或 FSM_TEST/ 下的段目录，不能是根目录本身。"""
    try:
        cur = Path(session).resolve()
    except OSError:
        return None
    for root in (RECORDINGS, FSM_TEST_DIR):
        try:
            rr = root.resolve()
        except OSError:
            continue
        if cur == rr:
            return None
        if rr in cur.parents:
            return rr
    return None


def _png_name_list(raw) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(val) -> None:
        s = str(val or "").strip()
        if not s:
            return
        name = Path(s).name
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    if isinstance(raw, str):
        add(raw)
    elif isinstance(raw, list):
        for item in raw:
            add(item)
    return out


def png_names_of_session(session: Path, meta: dict | None = None, frames: list[dict] | None = None) -> list[str]:
    session = Path(session)
    if frames is None:
        frames = load_jsonl(session / "frames.jsonl")
    if meta is None:
        mp = session / "meta.json"
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    names: list[str] = []
    seen: set[str] = set()
    for fr in frames or []:
        for n in _png_name_list(fr.get("png")):
            if n not in seen:
                seen.add(n)
                names.append(n)
    for key in ("png", "pngs", "png_files"):
        for n in _png_name_list((meta or {}).get(key)):
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


def _path_under(child: Path, parent: Path) -> bool:
    try:
        cr = child.resolve()
        pr = parent.resolve()
    except OSError:
        return False
    return cr == pr or pr in cr.parents


def resolve_session_pngs(
    session: Path,
    meta: dict | None = None,
    frames: list[dict] | None = None,
    log=None,
) -> list[Path]:
    """本段点名的 PNG 文件；缺文件跳过。不返回目录，不进 skill_features / skill_binds。"""
    def _log(msg: str) -> None:
        if callable(log):
            log(msg)
        else:
            print(msg)

    session = Path(session)
    found: list[Path] = []
    seen: set[str] = set()
    for name in png_names_of_session(session, meta, frames):
        hit = png_file_of(session, meta, name)
        if hit is None:
            _log(f"[replay] 缺 PNG，跳过: {name}")
            continue
        if not hit.is_file():
            _log(f"[replay] 不是文件，跳过: {hit}")
            continue
        if _path_under(hit, FEATURES_DIR) or _path_under(hit, BINDS_DIR):
            _log(f"[replay] 禁止路径，跳过: {hit}")
            continue
        key = str(hit.resolve())
        if key in seen:
            continue
        seen.add(key)
        found.append(hit)
    return found


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
    stuck_recover_s: float = DEFAULT_STUCK_RECOVER_S,
    stuck_recover=(),
    s: int = 20,
    s_pos: int | None = None,
    s_size: int | None = None,
    pm: int = 10,
    pc: int = 5,
    pt: int = 3,
    xxx_ms: int = 1000,
    th_ms: int = 50,
    pw_ms: int = 3000,
    advance_timeout_ms: int = DEFAULT_ADVANCE_TIMEOUT_MS,
    approach_timeout_ms: int = DEFAULT_APPROACH_TIMEOUT_MS,
    advance_timeout_esc_n: int = DEFAULT_ADVANCE_TIMEOUT_ESC_N,
    default_range_x: int = DEFAULT_RANGE_X,
    default_range_y: int = DEFAULT_RANGE_Y,
    fight_plan=(),
    hotbar=(),
    dist_table=(),
    map_reset: bool = False,
    mon_off_x: int = 0,
    mon_off_y: int = 0,
    loot_off_x: int = 0,
    loot_off_y: int = 0,
    gate_off_x: int = 0,
    gate_off_y: int = 0,
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
            stuck_recover_s=float(stuck_recover_s),
            stuck_recover=parse_stuck_recover(stuck_recover),
            s=s,
            s_pos=s_pos,
            s_size=s_size,
            pm=pm,
            pc=pc,
            pt=pt,
            xxx_ms=xxx_ms,
            th_ms=th_ms,
            pw_ms=int(pw_ms),
            advance_timeout_ms=max(1, int(advance_timeout_ms)),
            approach_timeout_ms=max(1, int(approach_timeout_ms)),
            advance_timeout_esc_n=max(1, int(advance_timeout_esc_n)),
            default_range_x=max(1, int(default_range_x)),
            default_range_y=max(1, int(default_range_y)),
            fight_plan=tuple(fight_plan or ()),
            hotbar=tuple(hotbar or ()),
            dist_table=tuple(dist_table or ()),
            map_reset=bool(map_reset),
            mon_off_x=int(mon_off_x),
            mon_off_y=int(mon_off_y),
            loot_off_x=int(loot_off_x),
            loot_off_y=int(loot_off_y),
            gate_off_x=int(gate_off_x),
            gate_off_y=int(gate_off_y),
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


def write_hold_frames(character: str, slot_durs: dict[int, int]) -> bool:
    """把回放里调的持续 ms 写进键位表，其它字段原样保留。"""
    return write_skill_table(character, slot_durs=slot_durs, combo_slots=None, multi_n=None)


def write_skill_table(
    character: str,
    slot_durs: dict[int, int] | None = None,
    combo_slots: set[int] | None = None,
    multi_n: dict[int, int] | None = None,
    mash_slots: set[int] | None = None,
    mash_n: dict[int, int] | None = None,
    edit_slots: set[int] | None = None,
) -> bool:
    """把持续毫秒 / 连续释放 / 多次释放 / 连按写进键位表，其它字段原样保留。"""
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
        if want_combo is not None and want_mash is None:
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
            if bool(item.get(COMBO_KEY)) is not on or COMBO_KEY not in item:
                item[COMBO_KEY] = on
                changed = True
            if mash_n is not None and on:
                n_val = mash_n.get(slot)
                if n_val is None:
                    n_val = parse_mash_n(item, DEFAULT_MASH_COUNT)
                n_val = clamp_mash_count(n_val)
                if item.get(MASH_N_KEY) != n_val:
                    item[MASH_N_KEY] = n_val
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
        hf = parse_hold_ms(item)
        if hf is not None:
            row[HOLD_FRAMES_KEY] = hf
        row["_combo_set"] = COMBO_KEY in item
        row[COMBO_KEY] = bool(item.get(COMBO_KEY) or item.get(MASH_KEY))
        row["_multi_set"] = MULTI_KEY in item
        row[MULTI_KEY] = bool(item.get(MULTI_KEY))
        row[MULTI_N_KEY] = parse_multi_n(item) if item.get(MULTI_KEY) else DEFAULT_MULTI
        row[MASH_KEY] = bool(item.get(MASH_KEY) or item.get(COMBO_KEY))
        row[MASH_N_KEY] = parse_mash_n(item, DEFAULT_MASH_COUNT)
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




DET_KEYS = ("mon", "boss", "loot", "gate")


def _frame_count(fr: dict, key: str) -> int:
    try:
        return max(0, int(fr.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _interior_run_spans(seq: list[bool], want_on: bool) -> list[tuple[int, int]]:
    """两侧被相反状态夹住的连续段 [start, end]（含，0-based）。
    want_on=False → 掉检：有→无→有 中间「无」；
    want_on=True  → 误检：无→有→无 中间「有」。
    """
    n = len(seq)
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and seq[j + 1] == seq[i]:
            j += 1
        if (
            seq[i] == want_on
            and i > 0
            and j + 1 < n
            and seq[i - 1] != want_on
            and seq[j + 1] != want_on
        ):
            out.append((i, j))
        i = j + 1
    return out


def _run_hist_from_spans(spans: list[tuple[int, int]]) -> dict[str, int]:
    h = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for a, b in spans:
        r = b - a + 1
        if r <= 0:
            continue
        if r >= 5:
            h["5+"] += 1
        else:
            h[str(r)] += 1
    return h


def _fmt_hist(h: dict[str, int]) -> str:
    return (
        f"  1帧: {h.get('1', 0)}   2帧: {h.get('2', 0)}   "
        f"3帧: {h.get('3', 0)}   4帧: {h.get('4', 0)}   ≥5帧: {h.get('5+', 0)}"
    )


def _fmt_span_1based(a: int, b: int) -> str:
    """回放 UI 帧号为 1-based。"""
    if a == b:
        return str(a + 1)
    return f"{a + 1}-{b + 1}"


def _fmt_short_span_lines(spans: list[tuple[int, int]]) -> list[str]:
    """仅列出长度 1–4 的游程帧号。"""
    by_len: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    for a, b in spans:
        r = b - a + 1
        if 1 <= r <= 4:
            by_len[r].append(_fmt_span_1based(a, b))
    lines: list[str] = []
    for r in (1, 2, 3, 4):
        if by_len[r]:
            lines.append(f"  {r}帧 → " + ", ".join(by_len[r]))
    return lines


def _count_jitters(frames: list[dict], key: str) -> list[dict]:
    """连续「有」内 count 上升：记录帧(0-based)、旧值、新值。"""
    n = len(frames)
    counts = [_frame_count(fr, key) for fr in frames]
    on = [c > 0 for c in counts]
    out: list[dict] = []
    i = 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        for k in range(i + 1, j + 1):
            if counts[k] > counts[k - 1]:
                out.append({"i": k, "prev": counts[k - 1], "cur": counts[k]})
        i = j + 1
    return out


def build_det_stats(frames: list[dict]) -> dict:
    """类存在布尔游程 + mon COUNT 抖动（连续「有」内上升）。"""
    drop_spans: dict[str, list[tuple[int, int]]] = {}
    false_spans: dict[str, list[tuple[int, int]]] = {}
    drop_hist: dict[str, dict[str, int]] = {}
    false_hist: dict[str, dict[str, int]] = {}
    for key in DET_KEYS:
        seq = [_frame_count(fr, key) > 0 for fr in frames]
        ds = _interior_run_spans(seq, False)
        fs = _interior_run_spans(seq, True)
        drop_spans[key] = ds
        false_spans[key] = fs
        drop_hist[key] = _run_hist_from_spans(ds)
        false_hist[key] = _run_hist_from_spans(fs)
    mon_jit = _count_jitters(frames, "mon")
    mon_present = sum(1 for fr in frames if _frame_count(fr, "mon") > 0)
    return {
        "drop_spans": drop_spans,
        "false_spans": false_spans,
        "drop_hist": drop_hist,
        "false_hist": false_hist,
        "mon_jitters": mon_jit,
        "mon_present_frames": mon_present,
    }


def format_det_run_report(det: dict | None) -> str:
    if not det:
        return "本段无检测帧统计。"
    lines: list[str] = []
    for key in DET_KEYS:
        dh = (det.get("drop_hist") or {}).get(key) or _run_hist_from_spans([])
        fh = (det.get("false_hist") or {}).get(key) or _run_hist_from_spans([])
        ds = (det.get("drop_spans") or {}).get(key) or []
        fs = (det.get("false_spans") or {}).get(key) or []
        lines.append(f"{key}  掉检游程（连续帧内有→无→有，中间那段「无」的长度）")
        lines.append(_fmt_hist(dh))
        lines.extend(_fmt_short_span_lines(ds))
        lines.append(f"{key}  误检游程（无→有→无，中间那段「有」的长度）")
        lines.append(_fmt_hist(fh))
        lines.extend(_fmt_short_span_lines(fs))
        lines.append("")
    jits = det.get("mon_jitters") or []
    pf = int(det.get("mon_present_frames") or 0)
    lines.append("mon  COUNT抖动（连续「有」内 count 上升；帧号与回放一致，从1起）")
    if not jits:
        lines.append(f"  （无）  有怪帧 {pf}")
    else:
        for j in jits:
            lines.append(f"  帧 {int(j['i']) + 1}: {int(j['prev'])}→{int(j['cur'])}")
        lines.append(f"  共 {len(jits)} 次 / 有怪帧 {pf}")
    return "\n".join(lines).rstrip() + "\n"


def state_hint_for_frame(states: list[dict], frames: list[dict], i: int) -> str:
    """右上角：states.jsonl 当前状态 + frame_count（有则带进度）。"""
    if not states:
        return "无 states.jsonl"
    if i < 0 or i >= len(frames):
        return "无 states.jsonl"
    try:
        t = int(frames[i]["t_ns"])
    except (KeyError, TypeError, ValueError):
        return "无 states.jsonl"
    hit = None
    for row in states:
        try:
            a = int(row.get("enter_t_ns"))
            b = int(row.get("exit_t_ns"))
        except (TypeError, ValueError):
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        if lo <= t <= hi:
            hit = row
            break
    if hit is None:
        # 落在空隙时取最近已结束段
        best = None
        best_d = None
        for row in states:
            try:
                b = int(row.get("exit_t_ns"))
            except (TypeError, ValueError):
                continue
            if b <= t:
                d = t - b
                if best_d is None or d < best_d:
                    best, best_d = row, d
        hit = best
    if hit is None:
        return "states 未覆盖本帧"
    st = str(hit.get("state") or "—")
    try:
        fc = max(1, int(hit.get("frame_count") or 1))
    except (TypeError, ValueError):
        fc = 1
    try:
        enter = int(hit.get("enter_t_ns"))
        exit_ = int(hit.get("exit_t_ns"))
    except (TypeError, ValueError):
        enter = exit_ = None
    cur = None
    if enter is not None and exit_ is not None:
        lo, hi = (enter, exit_) if enter <= exit_ else (exit_, enter)
        n = 0
        for fr in frames:
            try:
                tj = int(fr["t_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            if lo <= tj <= hi:
                n += 1
                if tj == t:
                    cur = n
        if cur is None and lo <= t <= hi:
            cur = n
    try:
        dur = int(hit.get("duration_ms") or 0)
    except (TypeError, ValueError):
        dur = 0
    if cur is not None and fc > 0:
        bit = f"{st} · {cur}/{fc}帧"
    else:
        bit = f"{st} · {fc}帧"
    if dur > 0:
        bit += f" · {dur}ms"
    nxt = hit.get("next_state")
    if nxt:
        bit += f" → {nxt}"
    return bit



ROUND_BAR_SKIP_STATES = frozenset()
ROUND_BAR_COLORS = {
    "前进": "#4aa3ff",
    "开打": "#e85d5d",
    "捡物": "#e8c547",
    "空闲": "#6b7280",
    "未知": "#888888",
}
_ROUND_BAR_FALLBACK = ("#7c6af2", "#2bb673", "#e67e22", "#1abc9c", "#9b59b6", "#16a085")


def _state_dur_ms(row: dict) -> int:
    try:
        if row.get("duration_ms") is not None:
            return max(0, int(row["duration_ms"]))
    except (TypeError, ValueError):
        pass
    try:
        return max(0, int((int(row["exit_t_ns"]) - int(row["enter_t_ns"])) / 1e6))
    except (KeyError, TypeError, ValueError):
        return 0


def _round_bar_color(state: str, seen: dict[str, str]) -> str:
    if state in ROUND_BAR_COLORS:
        return ROUND_BAR_COLORS[state]
    if state not in seen:
        seen[state] = _ROUND_BAR_FALLBACK[len(seen) % len(_ROUND_BAR_FALLBACK)]
    return seen[state]


def _fmt_dual_ms(ms: float) -> str:
    if ms is None:
        return "—"
    ms = float(ms)
    return f"{ms / 1000.0:.2f}s ({ms:.0f}ms)"


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return float(ys[f])
    return float(ys[f] + (ys[c] - ys[f]) * (k - f))


def load_round_segments(states: list[dict]) -> list[tuple[str, int]]:
    """一轮内按状态分段：(state, duration_ms)，跳过 0 时长。"""
    out: list[tuple[str, int]] = []
    for row in states:
        st = str(row.get("state") or "未知")
        if st in ROUND_BAR_SKIP_STATES:
            continue
        ms = _state_dur_ms(row)
        if ms <= 0:
            continue
        out.append((st, ms))
    return out


def collect_dungeon_rounds(
    dun: str,
    character: str = "",
    session: Path | None = None,
) -> list[tuple[str, Path, list[tuple[str, int]]]]:
    """同角色+同地下城含 states.jsonl 的段，按目录名新→旧，不设数量上限。标签=目录名。"""
    found: list[tuple[str, Path, list[tuple[str, int]]]] = []
    seen: set[Path] = set()
    roots = (FSM_TEST_DIR, RECORDINGS)
    candidates: list[Path] = []
    char = (character or "").strip()
    if dun and dun not in ("未知", "采集", "FSM测试"):
        for root in roots:
            ddir = root / dun
            if ddir.is_dir():
                for p in ddir.iterdir():
                    if p.is_dir() and (p / "states.jsonl").is_file():
                        candidates.append(p)
    if session is not None:
        sp = Path(session)
        if sp.is_dir() and (sp / "states.jsonl").is_file():
            candidates.append(sp)
    for p in sorted(candidates, key=lambda x: x.name, reverse=True):
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key in seen:
            continue
        seen.add(key)
        meta = {}
        mp = p / "meta.json"
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        # 角色：meta / 目录名后缀；states 里 char_name 兜底
        ch = character_of_session(p, meta)
        if not ch:
            rows0 = load_jsonl(p / "states.jsonl")
            for row in rows0[:5]:
                c2 = str(row.get("char_name") or "").strip()
                if c2:
                    ch = c2
                    break
        if char and ch and ch != char:
            continue
        if char and not ch:
            # 当前指定了角色但这段解析不出角色 → 跳过，避免串角色
            continue
        rows = load_jsonl(p / "states.jsonl")
        segs = load_round_segments(rows)
        if not segs:
            continue
        found.append((p.name, p, segs))
    return found


def summarize_round_segments(
    rounds: list[tuple[str, Path, list[tuple[str, int]]]],
    *,
    include_idle: bool = False,
) -> list[dict]:
    """每状态：占比 / 次数 / 中位 / p90 / max（ms）。按占比降序。"""
    by: dict[str, list[int]] = {}
    for _name, _path, segs in rounds:
        for st, ms in segs:
            if st == "空闲" and not include_idle:
                continue
            by.setdefault(st, []).append(ms)
    total = sum(sum(v) for v in by.values()) or 1
    rows: list[dict] = []
    for st, xs in by.items():
        s = sum(xs)
        rows.append(
            {
                "state": st,
                "share": 100.0 * s / total,
                "n": len(xs),
                "med": statistics.median(xs) if xs else None,
                "p90": _percentile([float(x) for x in xs], 90.0),
                "max": float(max(xs)) if xs else None,
                "total_ms": s,
            }
        )
    rows.sort(key=lambda r: (-r["share"], r["state"]))
    return rows


def format_round_summary_table(summary: list[dict]) -> str:
    lines = [
        "状态汇总（占比 | 次数 | 中位 | p90 | max；时长双单位 s/ms）",
        f"{'状态':<6} {'占比':>7} {'次数':>5} {'中位':>18} {'p90':>18} {'max':>18}",
    ]
    for r in summary:
        lines.append(
            f"{r['state']:<6} {r['share']:6.1f}% {r['n']:5d} "
            f"{_fmt_dual_ms(r['med']):>18} {_fmt_dual_ms(r['p90']):>18} {_fmt_dual_ms(r['max']):>18}"
        )
    return "\n".join(lines) + "\n"




def _frame_index_1based(frames: list[dict], t_ns: int) -> int | None:
    """enter_t_ns → 回放帧号（1-based）；找不到则 None。"""
    if not frames:
        return None
    try:
        t = int(t_ns)
    except (TypeError, ValueError):
        return None
    best_i = None
    for i, fr in enumerate(frames):
        try:
            ft = int(fr["t_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if ft >= t:
            return i + 1
        best_i = i + 1
    return best_i


def classify_fight_casts(casts: list | None) -> str:
    """只描述 casts 形状，不做因果判断。"""
    xs = list(casts or [])
    if not xs:
        return "无释放"
    slots: list[int] = []
    ts: list[int] = []
    for c in xs:
        if not isinstance(c, dict):
            continue
        try:
            slots.append(int(c.get("slot")))
        except (TypeError, ValueError):
            continue
        try:
            ts.append(int(c.get("t") or 0))
        except (TypeError, ValueError):
            ts.append(0)
    if not slots:
        return "无释放"
    uniq = set(slots)
    if len(uniq) == 1 and len(slots) >= 2:
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        gaps = [g for g in gaps if g >= 0]
        if gaps:
            avg = sum(gaps) / len(gaps)
            if avg > 0 and all(0.5 * avg <= g <= 1.5 * avg for g in gaps):
                return "同槽反复"
        return "同槽反复"
    return "有释放"


def _fmt_casts_cell(casts: list | None) -> str:
    xs = list(casts or [])
    if not xs:
        return "[]"
    parts: list[str] = []
    for c in xs:
        if not isinstance(c, dict):
            continue
        t = c.get("t", 0)
        slot = c.get("slot")
        parts.append(f"{t}:s{slot}" if slot is not None else f"{t}:s?")
    return "[" + ", ".join(parts) + "]"


def collect_fight_segments(
    dun: str,
    character: str = "",
    session: Path | None = None,
) -> list[dict]:
    """同角色+同图各段 states.jsonl 里的开打段，附起始帧与邻接卡住标记。"""
    rounds = collect_dungeon_rounds(dun, character=character, session=session)
    # also need sessions that only have states — collect_dungeon_rounds skips empty combat segs
    # Re-scan sessions for states including those with only idle:
    found_paths: list[Path] = []
    seen: set[Path] = set()
    char = (character or "").strip()
    roots = (FSM_TEST_DIR, RECORDINGS)
    cands: list[Path] = []
    if dun and dun not in ("未知", "采集", "FSM测试"):
        for root in roots:
            ddir = root / dun
            if ddir.is_dir():
                for p in ddir.iterdir():
                    if p.is_dir() and (p / "states.jsonl").is_file():
                        cands.append(p)
    if session is not None:
        sp = Path(session)
        if sp.is_dir() and (sp / "states.jsonl").is_file():
            cands.append(sp)
    for p in sorted(cands, key=lambda x: x.name, reverse=True):
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key in seen:
            continue
        seen.add(key)
        meta = {}
        mp = p / "meta.json"
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        ch = character_of_session(p, meta)
        if not ch:
            rows0 = load_jsonl(p / "states.jsonl")
            for row in rows0[:5]:
                c2 = str(row.get("char_name") or "").strip()
                if c2:
                    ch = c2
                    break
        if char and ch and ch != char:
            continue
        if char and not ch:
            continue
        found_paths.append(p)

    out: list[dict] = []
    for p in found_paths:
        states = load_jsonl(p / "states.jsonl")
        frames = load_jsonl(p / "frames.jsonl")
        for i, row in enumerate(states):
            if str(row.get("state") or "") != "开打":
                continue
            if "mon_boss_enter" not in row:
                # 旧段无新字段，跳过取数（先验字段）
                continue
            try:
                dur = int(row.get("duration_ms") or 0)
            except (TypeError, ValueError):
                dur = 0
            try:
                enter_n = int(row.get("mon_boss_enter") or 0)
            except (TypeError, ValueError):
                enter_n = 0
            per = (dur / enter_n) if enter_n > 0 else float(dur)
            casts = row.get("casts") if isinstance(row.get("casts"), list) else []
            stuck_near = False
            if i > 0 and str(states[i - 1].get("state") or "") == "卡住":
                stuck_near = True
            if i + 1 < len(states) and str(states[i + 1].get("state") or "") == "卡住":
                stuck_near = True
            try:
                et = int(row.get("enter_t_ns") or 0)
            except (TypeError, ValueError):
                et = 0
            out.append(
                {
                    "session": p.name,
                    "path": p,
                    "map_name": str(row.get("map_name") or dun),
                    "frame": _frame_index_1based(frames, et),
                    "duration_ms": dur,
                    "mon_boss_enter": enter_n,
                    "per_ms": per,
                    "casts": casts,
                    "class": classify_fight_casts(casts),
                    "stuck_near": stuck_near,
                    "enter_t_ns": et,
                }
            )
    out.sort(key=lambda r: (-r["per_ms"], -r["duration_ms"]))
    return out


def format_fight_sort_report(rows: list[dict]) -> str:
    if not rows:
        return (
            "无带 mon_boss_enter/casts 的开打段。\n"
            "请用新写入端跑一轮 FSM 测试后再打开本表。\n"
        )
    lines: list[str] = []
    lines.append(
        f"{'#':>2}  {'轮次':<28} {'图':<12} {'起始帧':>8}  {'时长':>8}  {'进入怪数':>6}  {'每只怪':>8}  {'casts':<32}  分类  卡住邻接"
    )
    for i, r in enumerate(rows, 1):
        fr = f"f{r['frame']:05d}" if r["frame"] is not None else "f?????"
        map_s = (r["map_name"] or "")[:12]
        casts_s = _fmt_casts_cell(r["casts"])
        stuck = "是" if r["stuck_near"] else ""
        lines.append(
            f"{i:>2}  {r['session']:<28} {map_s:<12} {fr:>8}  {r['duration_ms']:>6}ms  "
            f"{r['mon_boss_enter']:>6}  {r['per_ms']:>6.0f}ms  {casts_s:<32}  {r['class']:<6}  {stuck}"
        )
    # summary
    n = len(rows)
    c_cast = sum(1 for r in rows if r["class"] == "有释放")
    c_none = sum(1 for r in rows if r["class"] == "无释放")
    c_same = sum(1 for r in rows if r["class"] == "同槽反复")
    pers = [r["per_ms"] for r in rows]
    med = statistics.median(pers)
    p90 = _percentile(pers, 90.0) if pers else None
    mx = max(pers) if pers else None
    lines.append("")
    lines.append(
        f"开打段 {n} 个：有释放 {c_cast} / 无释放 {c_none} / 同槽反复 {c_same}"
    )
    lines.append(
        f"每只怪耗时：中位 {_fmt_dual_ms(med)}  p90 {_fmt_dual_ms(p90)}  max {_fmt_dual_ms(mx)}"
    )
    lines.append("排序键 = duration_ms / mon_boss_enter（降序）。分类只描述 casts 形状。")
    return "\n".join(lines) + "\n"


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
    states_path = session / "states.jsonl"
    states = load_jsonl(states_path) if states_path.is_file() else []
    return {
        "session": session,
        "meta": meta,
        "frames": frames,
        "keys": keys,
        "states": states,
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
        "det_stats": build_det_stats(frames),
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
        ttk.Button(top, text="删除本段", width=10, command=self._delete_current_session).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(top, text="检测游程", width=10, command=self._show_det_runs).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(top, text="每轮时间条", width=10, command=self._show_round_time_bars).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(top, text="开打排序", width=10, command=self._show_fight_sort).pack(
            side=tk.LEFT, padx=(6, 0)
        )
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
        self.stuck_recover_s_var = tk.StringVar(value=str(DEFAULT_STUCK_RECOVER_S))
        self.s_var = tk.StringVar(value="20")
        self.s_size_var = tk.StringVar(value="20")
        self.xxx_var = tk.StringVar(value="1000")
        self.tap_ms_min_var = tk.StringVar(value=str(DEFAULT_TAP_MS_MIN))
        self.tap_ms_max_var = tk.StringVar(value=str(DEFAULT_TAP_MS_MAX))
        self.mash_count_var = tk.StringVar(value="3")
        self.mash_gap_var = tk.StringVar(value="50")
        self.th_var = tk.StringVar(value="50")
        self.pm_var = tk.StringVar(value=str(DEFAULT_PM))
        self.pc_var = tk.StringVar(value=str(DEFAULT_PC))
        self.pt_var = tk.StringVar(value=str(DEFAULT_PT))
        self.pw_var = tk.StringVar(value=str(DEFAULT_PW_MS))
        self.advance_timeout_var = tk.StringVar(value=str(DEFAULT_ADVANCE_TIMEOUT_MS))
        self.approach_timeout_var = tk.StringVar(value=str(DEFAULT_APPROACH_TIMEOUT_MS))
        self.default_range_x_var = tk.StringVar(value=str(DEFAULT_RANGE_X))
        self.default_range_y_var = tk.StringVar(value=str(DEFAULT_RANGE_Y))
        self.advance_timeout_esc_n_var = tk.StringVar(value=str(DEFAULT_ADVANCE_TIMEOUT_ESC_N))
        self.town_s_var = tk.StringVar(value=str(int(DEFAULT_TOWN_S)))
        self.run_press_var = tk.StringVar(value=str(RUN_PRESS_GT))
        self.loot_hold_var = tk.StringVar(value=str(LOOT_HOLD_DEFAULT))
        self.corr_u = tk.IntVar(value=0)
        self.corr_d = tk.IntVar(value=0)
        self.corr_l = tk.IntVar(value=0)
        self.corr_r = tk.IntVar(value=0)
        self.loot_corr_u = tk.IntVar(value=0)
        self.loot_corr_d = tk.IntVar(value=0)
        self.loot_corr_l = tk.IntVar(value=0)
        self.loot_corr_r = tk.IntVar(value=0)
        self.gate_corr_u = tk.IntVar(value=0)
        self.gate_corr_d = tk.IntVar(value=0)
        self.gate_corr_l = tk.IntVar(value=0)
        self.gate_corr_r = tk.IntVar(value=0)
        self._corr_lock = False
        self._corr_prev = (0, 0, 0, 0)
        self.e_var = tk.StringVar(value="3")
        self.f_var = tk.StringVar(value=str(DEFAULT_F))
        self.press_lag_var = tk.StringVar(value=str(EXTRACT_PRESS_LAG_FRAMES))
        self.extract_fsm_var = tk.BooleanVar(value=False)
        self._extract_busy = False
        self._feature_cache = None
        self._last_session_path: Path | None = None
        self._session_paths: dict[str, Path] = {}
        self._combo_vars: dict[int, tk.BooleanVar] = {}
        self._combo_ui_guard = False
        self._mash_vars: dict[int, tk.BooleanVar] = {}
        self._mash_n_vars: dict[int, tk.StringVar] = {}
        self._mash_n_spins: dict[int, ttk.Spinbox] = {}
        self._mash_ui_guard = False
        self._multi_vars: dict[int, tk.BooleanVar] = {}
        self._multi_n_vars: dict[int, tk.StringVar] = {}
        self._multi_spins: dict[int, ttk.Spinbox] = {}
        self._multi_ui_guard = False
        self._stuck_recover_spec = stuck_recover_to_json()
        self._load_ui_settings()
        self.source_combo.bind("<<ComboboxSelected>>", lambda e: self._on_source())

        feat = ttk.LabelFrame(count_wrap, text="过图技能特征（快捷栏技能段）", padding=6)
        feat.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        feat_r = ttk.Frame(feat)
        feat_r.pack(fill=tk.X)
        ttk.Label(feat_r, text="E").pack(side=tk.LEFT)
        self._e_spin = ttk.Spinbox(feat_r, from_=0, to=40, width=4, textvariable=self.e_var)
        self._shield_right_wheel(self._e_spin)
        self._e_spin.pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(feat_r, text="F%").pack(side=tk.LEFT)
        _f_spin = ttk.Spinbox(feat_r, from_=0, to=100, width=4, textvariable=self.f_var)
        self._shield_right_wheel(_f_spin)
        _f_spin.pack(side=tk.LEFT, padx=(2, 8))
        ttk.Checkbutton(
            feat_r,
            text="含FSM测试",
            variable=self.extract_fsm_var,
            command=self._save_ui_settings,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self._help_btn(feat_r, "过图技能特征说明", HELP_EXTRACT).pack(side=tk.RIGHT)
        feat_lag = ttk.Frame(feat)
        feat_lag.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(feat_lag, text="lag 帧").pack(side=tk.LEFT)
        self._spin(feat_lag, self.press_lag_var, frm=0, to=30).pack(side=tk.LEFT, padx=(2, 0))
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
        ttk.Button(feat_btns2, text="查看当前过图技能特征", command=self._show_current_features).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        self.map_reset_host = ttk.Frame(feat)
        self.map_reset_host.pack(fill=tk.X, pady=(4, 0))
        feat_p = ttk.Frame(feat)
        feat_p.pack(fill=tk.X, pady=(4, 0))
        self.extract_prog_var = tk.StringVar(value="")
        ttk.Label(feat_p, textvariable=self.extract_prog_var, width=18).pack(side=tk.LEFT)
        self.extract_prog = ttk.Progressbar(feat_p, mode="determinate")
        self.extract_prog.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.extract_result = tk.Text(feat, height=4, state=tk.DISABLED, font=("Consolas", 9), wrap=tk.WORD)
        self.extract_result.pack(fill=tk.X, pady=(4, 0))
        self.e_var.trace_add("write", lambda *_: self._save_ui_settings())
        self.f_var.trace_add("write", lambda *_: self._save_ui_settings())

        scroll_host = tk.Frame(count_wrap)
        scroll_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._right_canvas = tk.Canvas(scroll_host, highlightthickness=0, borderwidth=0)
        right_vsb = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=self._right_canvas.yview)
        self._right_inner = tk.Frame(self._right_canvas)
        self._right_inner.bind(
            "<Configure>",
            lambda e: self._right_canvas.configure(scrollregion=self._right_canvas.bbox("all")),
        )
        self._right_win = self._right_canvas.create_window((0, 0), window=self._right_inner, anchor="nw")
        self._right_canvas.configure(yscrollcommand=right_vsb.set)
        self._right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._right_canvas.bind("<Configure>", self._on_right_canvas_cfg)
        self._bind_right_wheel(scroll_host)

        def _pf(title: str) -> ttk.LabelFrame:
            return ttk.LabelFrame(self._right_inner, text=title, padding=6)

        def _prow(parent) -> ttk.Frame:
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=(2, 0))
            return row

        box_judge = _pf("判定")
        r_mlg = _prow(box_judge)
        ttk.Label(r_mlg, text="M").pack(side=tk.LEFT)
        self._spin(r_mlg, self.m_var).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_mlg, text="L").pack(side=tk.LEFT)
        self._spin(r_mlg, self.l_var).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_mlg, text="G").pack(side=tk.LEFT)
        self._spin(r_mlg, self.g_var).pack(side=tk.LEFT, padx=(2, 0))
        self._help_btn(r_mlg, "参数说明", HELP_PARAMS).pack(side=tk.RIGHT)
        box_stuck = _pf("卡住")
        r_x = _prow(box_stuck)
        ttk.Label(r_x, text="X秒").pack(side=tk.LEFT)
        self._spin(r_x, self.x_var, to=120, frm=0.05, width=5).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_x, text="恢复s").pack(side=tk.LEFT)
        self._spin(r_x, self.stuck_recover_s_var, to=30, frm=0.05, width=5).pack(side=tk.LEFT, padx=(2, 0))
        box_gate = _pf("过门")
        r_gate = _prow(box_gate)
        ttk.Label(r_gate, text="GX").pack(side=tk.LEFT)
        self._spin(r_gate, self.gx_var, frm=0, to=400).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_gate, text="GY").pack(side=tk.LEFT)
        self._spin(r_gate, self.gy_var, frm=0, to=400).pack(side=tk.LEFT, padx=(2, 0))
        r_ax = _prow(box_gate)
        ttk.Label(r_ax, text="AX ms").pack(side=tk.LEFT)
        self._spin(r_ax, self.ax_var, frm=0, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(r_ax, text="AY ms").pack(side=tk.LEFT)
        self._spin(r_ax, self.ay_var, frm=0, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 0))
        r_adv_to = _prow(box_gate)
        ttk.Label(r_adv_to, text="前进超时 ms").pack(side=tk.LEFT)
        self._spin(r_adv_to, self.advance_timeout_var, frm=1, to=60000, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_adv_to, text="超时ESC N").pack(side=tk.LEFT)
        self._spin(r_adv_to, self.advance_timeout_esc_n_var, frm=1, to=20, width=4).pack(side=tk.LEFT, padx=(2, 0))
        box_fight = _pf("开打")
        r_sf = _prow(box_fight)
        ttk.Label(r_sf, text="S_pos%").pack(side=tk.LEFT)
        self._spin(r_sf, self.s_var, frm=0, to=100).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_sf, text="S_size%").pack(side=tk.LEFT)
        self._spin(r_sf, self.s_size_var, frm=0, to=100).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_sf, text="F%").pack(side=tk.LEFT)
        self._spin(r_sf, self.f_var, frm=0, to=100).pack(side=tk.LEFT, padx=(2, 0))
        r_xxx = _prow(box_fight)
        ttk.Label(r_xxx, text="XXX ms").pack(side=tk.LEFT)
        self._spin(r_xxx, self.xxx_var, frm=1, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 0))
        r_appr_to = _prow(box_fight)
        ttk.Label(r_appr_to, text="走近超时 ms").pack(side=tk.LEFT)
        self._spin(r_appr_to, self.approach_timeout_var, frm=1, to=60000, width=6).pack(side=tk.LEFT, padx=(2, 0))
        r_rng = _prow(box_fight)
        ttk.Label(r_rng, text="默认范围X").pack(side=tk.LEFT)
        self._spin(r_rng, self.default_range_x_var, frm=1, to=4000, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_rng, text="Y").pack(side=tk.LEFT)
        self._spin(r_rng, self.default_range_y_var, frm=1, to=4000, width=6).pack(side=tk.LEFT, padx=(2, 0))
        r_tap = _prow(box_fight)
        ttk.Label(r_tap, text="点按 min").pack(side=tk.LEFT)
        self._spin(r_tap, self.tap_ms_min_var, frm=1, to=200).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_tap, text="max").pack(side=tk.LEFT)
        self._spin(r_tap, self.tap_ms_max_var, frm=1, to=200).pack(side=tk.LEFT, padx=(2, 0))
        box_mash = _pf("连按·移动")
        r_mash = _prow(box_mash)
        ttk.Label(r_mash, text="COUNT").pack(side=tk.LEFT)
        self._spin(r_mash, self.mash_count_var, frm=1, to=15).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_mash, text="间隔 ms").pack(side=tk.LEFT)
        self._spin(r_mash, self.mash_gap_var, frm=10, to=300).pack(side=tk.LEFT, padx=(2, 0))
        r_th = _prow(box_mash)
        ttk.Label(r_th, text="th ms").pack(side=tk.LEFT)
        self._spin(r_th, self.th_var, frm=0, to=300).pack(side=tk.LEFT, padx=(2, 0))
        box_loot = _pf("捡物·回城")
        r_loot_fsm = _prow(box_loot)
        ttk.Label(r_loot_fsm, text="PM").pack(side=tk.LEFT)
        self._spin(r_loot_fsm, self.pm_var, frm=0, to=200).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_loot_fsm, text="PC").pack(side=tk.LEFT)
        self._spin(r_loot_fsm, self.pc_var, frm=0, to=40).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_loot_fsm, text="PT").pack(side=tk.LEFT)
        self._spin(r_loot_fsm, self.pt_var).pack(side=tk.LEFT, padx=(2, 0))
        r_pw = _prow(box_loot)
        ttk.Label(r_pw, text="PW ms").pack(side=tk.LEFT)
        self._spin(r_pw, self.pw_var, frm=0, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_pw, text="回城秒").pack(side=tk.LEFT)
        self._spin(r_pw, self.town_s_var, frm=1, to=120).pack(side=tk.LEFT, padx=(2, 0))
        box_misc = _pf("其它")
        r_run = _prow(box_misc)
        ttk.Label(r_run, text="跑>").pack(side=tk.LEFT)
        self._spin(r_run, self.run_press_var, frm=0, to=30).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r_run, text="捡物 ms").pack(side=tk.LEFT)
        self._spin(r_run, self.loot_hold_var, to=20000, width=6).pack(side=tk.LEFT, padx=(2, 0))
        box_corr = _pf("补正")
        mx = MON_CORR_MAX

        def _corr_pad(parent, title: str, vu, vd, vl, vr, summary_attr: str, kind: str):
            r = _prow(parent)
            ttk.Label(r, text=title).pack(side=tk.LEFT)
            lab = ttk.Label(r, text="X+0 Y+0", foreground="#06c")
            lab.pack(side=tk.LEFT, padx=(8, 0))
            setattr(self, summary_attr, lab)
            pad = ttk.Frame(parent)
            pad.pack(fill=tk.X, pady=(2, 6))
            cmd = lambda _=None, k=kind: self._on_corr_edit(kind=k)
            sc_u = tk.Scale(
                pad, from_=mx, to=0, orient=tk.VERTICAL, length=64, width=10,
                showvalue=True, variable=vu, label="上", command=cmd,
            )
            self._shield_right_wheel(sc_u)
            sc_u.pack(side=tk.LEFT)
            mid = ttk.Frame(pad)
            mid.pack(side=tk.LEFT, fill=tk.Y, padx=4)
            sc_l = tk.Scale(
                mid, from_=mx, to=0, orient=tk.HORIZONTAL, length=84, width=10,
                showvalue=True, variable=vl, label="左", command=cmd,
            )
            self._shield_right_wheel(sc_l)
            sc_l.pack()
            sc_r = tk.Scale(
                mid, from_=0, to=mx, orient=tk.HORIZONTAL, length=84, width=10,
                showvalue=True, variable=vr, label="右", command=cmd,
            )
            self._shield_right_wheel(sc_r)
            sc_r.pack()
            sc_d = tk.Scale(
                pad, from_=0, to=mx, orient=tk.VERTICAL, length=64, width=10,
                showvalue=True, variable=vd, label="下", command=cmd,
            )
            self._shield_right_wheel(sc_d)
            sc_d.pack(side=tk.LEFT)

        _corr_pad(box_corr, "MON/BOSS", self.corr_u, self.corr_d, self.corr_l, self.corr_r, "corr_summary", "mon")
        _corr_pad(
            box_corr, "LOOT", self.loot_corr_u, self.loot_corr_d, self.loot_corr_l, self.loot_corr_r,
            "loot_corr_summary", "loot",
        )
        _corr_pad(
            box_corr, "GATE", self.gate_corr_u, self.gate_corr_d, self.gate_corr_l, self.gate_corr_r,
            "gate_corr_summary", "gate",
        )
        self._refresh_corr_summary()
        ttk.Checkbutton(
            box_corr,
            text="player 缺失沿用上次",
            variable=self.fill_var,
            command=self._on_view_opts,
        ).pack(anchor=tk.W, pady=(6, 0), fill=tk.X)
        ttk.Checkbutton(
            box_corr,
            text="相对坐标（player 原点）",
            variable=self.rel_var,
            command=self._on_view_opts,
        ).pack(anchor=tk.W, fill=tk.X)
        ttk.Checkbutton(
            box_corr,
            text="叠图对比 PNG",
            variable=self.overlay_var,
            command=self._on_overlay_opts,
        ).pack(anchor=tk.W, fill=tk.X)
        r_ov = _prow(box_corr)
        ttk.Label(r_ov, text="透明度").pack(side=tk.LEFT)
        ov_sc = tk.Scale(
            r_ov,
            from_=8,
            to=100,
            orient=tk.HORIZONTAL,
            length=120,
            width=10,
            showvalue=True,
            variable=self.overlay_alpha_var,
            command=lambda _=None: self._on_overlay_opts(),
        )
        self._shield_right_wheel(ov_sc)
        ov_sc.pack(side=tk.LEFT, fill=tk.X, expand=True)

        skill_box = ttk.LabelFrame(self._right_inner, text="技能表", padding=6)
        skill_box.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        skill_head = ttk.Frame(skill_box)
        skill_head.pack(fill=tk.X)
        self.skill_bind_label = ttk.Label(skill_head, text="技能表: —", foreground="#668")
        self.skill_bind_label.pack(side=tk.LEFT)
        ttk.Button(skill_head, text="重新读取", command=self._reload_skill_table).pack(side=tk.RIGHT)
        self._help_btn(skill_head, "技能表说明", HELP_SKILLS).pack(side=tk.RIGHT, padx=(0, 4))
        skill_cols = ttk.Frame(skill_box)
        skill_cols.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(skill_cols, text="槽 / 键", width=12).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="持续", width=5).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="多次", width=4).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="MULTI", width=6).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="连按", width=4).pack(side=tk.LEFT)
        ttk.Label(skill_cols, text="COUNT", width=6).pack(side=tk.LEFT)
        self.skill_hold_host = ttk.Frame(skill_box)
        self.skill_hold_host.pack(fill=tk.X)
        for box in (box_judge, box_stuck, box_gate, box_fight, box_mash, box_loot, box_misc, box_corr):
            box.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

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
        ttk.Label(ctrl, text="跳到").pack(side=tk.LEFT, padx=(8, 2))
        self.jump_var = tk.StringVar()
        jump_ent = ttk.Entry(ctrl, textvariable=self.jump_var, width=6)
        jump_ent.pack(side=tk.LEFT)
        jump_ent.bind("<Return>", self._jump_to_frame)
        ttk.Button(ctrl, text="跳转", width=4, command=self._jump_to_frame).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(ctrl, text="保存PNG", width=8, command=self._save_current_png).pack(
            side=tk.RIGHT, padx=(8, 0)
        )

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
            path = ICON_DIR / f"{name}.png"
            if not path.is_file():
                continue
            img = Image.open(path).convert("RGB")
            self._stack_photos[name] = ImageTk.PhotoImage(img.resize((28, 28)))
            if name == "player":
                self._player_photo = ImageTk.PhotoImage(img.resize((36, 36)))

    def _ensure_icons(self):
        if self._player_photo is not None and all(n in self._stack_photos for n in CLASSES):
            return
        self._load_icons()

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
        self._shield_right_wheel(sp)
        return sp

    def _press_lag_frames(self) -> int:
        return self._spin_n(self.press_lag_var, EXTRACT_PRESS_LAG_FRAMES, lo=0)

    def _fsm_track_kwargs(self) -> dict:
        ox, oy = mon_off_from_corr(*self._corr_tuple("mon"))
        lox, loy = mon_off_from_corr(*self._corr_tuple("loot"))
        gox, goy = mon_off_from_corr(*self._corr_tuple("gate"))
        return {
            "m": self._spin_n(self.m_var, 5),
            "l": self._spin_n(self.l_var, 5),
            "g": self._spin_n(self.g_var, 5),
            "x_s": self._spin_f(self.x_var, 1.5, lo=0.05),
            "gx": self._spin_n(self.gx_var, 50, lo=0),
            "gy": self._spin_n(self.gy_var, 10, lo=0),
            "ax_ms": self._spin_n(self.ax_var, 250, lo=0),
            "ay_ms": self._spin_n(self.ay_var, 250, lo=0),
            "stuck_recover_s": self._spin_f(self.stuck_recover_s_var, DEFAULT_STUCK_RECOVER_S, lo=0.05),
            "stuck_recover": list(getattr(self, "_stuck_recover_spec", None) or stuck_recover_to_json()),
            "s": self._spin_n(self.s_var, 20, lo=0),
            "s_pos": self._spin_n(self.s_var, 20, lo=0),
            "s_size": self._spin_n(self.s_size_var, 20, lo=0),
            "pm": self._spin_n(self.pm_var, DEFAULT_PM, lo=0),
            "pc": self._spin_n(self.pc_var, DEFAULT_PC, lo=0),
            "pt": self._spin_n(self.pt_var, DEFAULT_PT),
            "pw_ms": self._spin_n(self.pw_var, DEFAULT_PW_MS, lo=0),
            "advance_timeout_ms": self._spin_n(
                self.advance_timeout_var, DEFAULT_ADVANCE_TIMEOUT_MS, lo=1
            ),
            "approach_timeout_ms": self._spin_n(
                self.approach_timeout_var, DEFAULT_APPROACH_TIMEOUT_MS, lo=1
            ),
            "advance_timeout_esc_n": self._spin_n(
                self.advance_timeout_esc_n_var, DEFAULT_ADVANCE_TIMEOUT_ESC_N, lo=1
            ),
            "default_range_x": self._spin_n(self.default_range_x_var, DEFAULT_RANGE_X, lo=1),
            "default_range_y": self._spin_n(self.default_range_y_var, DEFAULT_RANGE_Y, lo=1),
            "xxx_ms": self._spin_n(self.xxx_var, 1000),
            "th_ms": self._spin_n(self.th_var, 50, lo=0),
            "fight_plan": self._fight_plan(),
            "hotbar": hotbar_from_binds(self._bind_skills),
            "dist_table": dist_table_from_features(self._feature_cache),
            "map_reset": has_map_reset(self._feature_cache),
            "mon_off_x": ox,
            "mon_off_y": oy,
            "loot_off_x": lox,
            "loot_off_y": loy,
            "gate_off_x": gox,
            "gate_off_y": goy,
            "tn_s": self._spin_f(self.town_s_var, float(DEFAULT_TOWN_S), lo=0.05),
        }

    def _send_text(self, dr: dict) -> str:
        mash_n = mash_n_for_slot(
            dr.get("skill_slot"),
            self._mash_slots_from_ui(),
            self._spin_n(self.mash_count_var, 3),
            self._mash_n_from_ui(),
        )
        return send_keys_label(
            dr.get("action"),
            tuple(dr.get("move_dirs") or ()),
            dr.get("move_dir"),
            dr.get("skill_key"),
            mash_n=mash_n,
            fight_dir=dr.get("fight_dir"),
        )

    def _corr_vars(self, kind: str = "mon") -> tuple[tk.IntVar, tk.IntVar, tk.IntVar, tk.IntVar]:
        if kind == "loot":
            return self.loot_corr_u, self.loot_corr_d, self.loot_corr_l, self.loot_corr_r
        if kind == "gate":
            return self.gate_corr_u, self.gate_corr_d, self.gate_corr_l, self.gate_corr_r
        return self.corr_u, self.corr_d, self.corr_l, self.corr_r

    def _corr_tuple(self, kind: str = "mon") -> tuple[int, int, int, int]:
        def n(var: tk.IntVar) -> int:
            try:
                return max(0, min(MON_CORR_MAX, int(var.get())))
            except (TypeError, ValueError, tk.TclError):
                return 0

        u, dwn, left, right = self._corr_vars(kind)
        return n(u), n(dwn), n(left), n(right)

    def _corr_off(self, kind: str = "mon") -> tuple[int, int]:
        return mon_off_from_corr(*self._corr_tuple(kind))

    def _apply_xy_corr(self, fr: dict) -> dict:
        mox, moy = self._corr_off("mon")
        lox, loy = self._corr_off("loot")
        gox, goy = self._corr_off("gate")
        return apply_detect_xy_corr(
            fr, mon_ox=mox, mon_oy=moy, loot_ox=lox, loot_oy=loy, gate_ox=gox, gate_oy=goy
        )

    def _refresh_corr_summary(self):
        for kind, attr in (
            ("mon", "corr_summary"),
            ("loot", "loot_corr_summary"),
            ("gate", "gate_corr_summary"),
        ):
            lab = getattr(self, attr, None)
            if lab is None:
                continue
            ox, oy = self._corr_off(kind)
            try:
                lab.config(text=f"X{ox:+d} Y{oy:+d}")
            except tk.TclError:
                pass

    def _on_corr_edit(self, _=None, kind: str = "mon"):
        if getattr(self, "_corr_lock", False):
            return
        self._corr_lock = True
        try:
            vu, vd, vl, vr = self._corr_vars(kind)
            cur = self._corr_tuple(kind)
            prev_map = getattr(self, "_corr_prev_map", None)
            if not isinstance(prev_map, dict):
                prev_map = {}
            prev = prev_map.get(kind, (0, 0, 0, 0))
            u, dwn, left, right = cur
            if u > 0 and dwn > 0:
                if u != prev[0]:
                    vd.set(0)
                elif dwn != prev[1]:
                    vu.set(0)
                elif u >= dwn:
                    vd.set(0)
                else:
                    vu.set(0)
            if left > 0 and right > 0:
                if left != prev[2]:
                    vr.set(0)
                elif right != prev[3]:
                    vl.set(0)
                elif left >= right:
                    vr.set(0)
                else:
                    vl.set(0)
            prev_map[kind] = self._corr_tuple(kind)
            self._corr_prev_map = prev_map
            if kind == "mon":
                self._corr_prev = prev_map[kind]
        finally:
            self._corr_lock = False
        self._refresh_corr_summary()
        if kind == "mon":
            ox, oy = self._corr_off("mon")
            prev_off = getattr(self, "_corr_off_warned", None)
            if prev_off is not None and prev_off != (ox, oy) and getattr(self, "_persist_ok", False):
                self._warn_corr_reextract()
            self._corr_off_warned = (ox, oy)
        self._save_ui_settings()
        if self.data:
            self._rebuild_views()
            self._show()
        else:
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

    def _on_right_canvas_cfg(self, event):
        self._right_canvas.itemconfigure(self._right_win, width=event.width)

    def _right_wheel_event(self, event):
        """右侧参数列表滚轮：只滚动画布，不改 Spinbox/Scale 数值。"""
        if getattr(event, "delta", 0):
            self._right_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _shield_right_wheel(self, widget) -> None:
        """Spinbox/Scale 默认会先吃掉滚轮；绑在控件上 return break，再转给列表。"""
        widget.bind("<MouseWheel>", self._right_wheel_event)

    def _bind_right_wheel(self, host):
        def _enter(_):
            self.bind_all("<MouseWheel>", self._right_wheel_event)

        def _leave(_):
            self.unbind_all("<MouseWheel>")

        host.bind("<Enter>", _enter)
        host.bind("<Leave>", _leave)

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

    def _refresh_sessions(self, *, select_first: bool = True):
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
            try:
                keep_r = Path(keep).resolve()
            except OSError:
                keep_r = None
            if keep_r is not None:
                for lab, p in self._session_paths.items():
                    try:
                        if p.resolve() == keep_r:
                            self.session_var.set(lab)
                            return
                    except OSError:
                        continue
        if labels:
            if select_first and not self.session_var.get().strip():
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
        self._photos.clear()
        # 分类 icon 是全局 UI 资源，清会话时不要丢；否则再打开段会退化成蓝点
        self._count_overlay_photo = None
        self._frame_overlay_photo = None
        self._frame_overlay_key = None
        self._feature_cache = None
        try:
            self.canvas.delete("all")
        except Exception:
            pass
        self._draw_flow(None, (), -1)
        self.keys_all_var.set("本段键: —")
        self.keys_var.set("按住: —")
        self.keys_edge_var.set("本帧沿(上一帧→本帧]: —")
        src = (self.source_var.get() or "全部").strip()
        if src == "FSM测试":
            self.info_var.set("没有 FSM_TEST 段。FSM测试写盘后点刷新；目录还不存在时这里是空的。")
        elif src == "采集":
            self.info_var.set("没有采集段（recordings/*/frames.jsonl）。")
        else:
            self.info_var.set("没有可回放的段。")

    def _selected_session_path(self) -> Path | None:
        if self.data and self.data.get("session"):
            return Path(self.data["session"])
        lab = self.session_var.get().strip()
        p = self._session_paths.get(lab)
        return Path(p) if p else None

    def _delete_current_session(self):
        session = self._selected_session_path()
        if session is None:
            messagebox.showinfo("删除本段", "先选中一段录像。", parent=self)
            return
        session = session.resolve() if session.exists() else session
        root = session_storage_root(session)
        if root is None:
            messagebox.showerror(
                "删除本段",
                f"只能删除 recordings/ 或 FSM_TEST/ 下的段目录，当前不是：\n{session}",
                parent=self,
            )
            return
        meta = (self.data or {}).get("meta") if self.data and Path(self.data["session"]).resolve() == session else None
        frames = (self.data or {}).get("frames") if self.data and Path(self.data["session"]).resolve() == session else None
        if meta is None or frames is None:
            meta = {}
            mp = session / "meta.json"
            if mp.is_file():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            frames = load_jsonl(session / "frames.jsonl")
        skipped: list[str] = []

        def _log(msg: str) -> None:
            print(msg)
            if msg.startswith("[replay] 缺 PNG"):
                skipped.append(msg)

        pngs = resolve_session_pngs(session, meta, frames, log=_log)
        char_dir = None
        ch = character_from_session(session, meta)
        if ch:
            char_dir = (IMAGES / _safe_folder_name(ch)).resolve()
        outside = []
        for p in pngs:
            try:
                pr = p.resolve()
            except OSError:
                continue
            if char_dir is not None and pr == char_dir:
                _log(f"[replay] 拒绝删除角色图目录: {pr}")
                continue
            if _path_under(pr, session):
                continue
            outside.append(p)
        msg = (
            "确定删除当前录像段？此操作不能撤销。\n\n"
            f"{session}\n\n"
            "将删除该目录（keys/frames/meta 等），以及本段 frames.jsonl / meta 点名的 PNG。\n"
            "不会删除整个 images/<角色>/，也不会改 skill_features / skill_binds。"
        )
        if not messagebox.askyesno("删除本段", msg, parent=self, default=messagebox.NO):
            return
        self.playing = False
        n_png = 0
        for p in outside:
            try:
                if p.is_file():
                    p.unlink()
                    n_png += 1
            except OSError as e:
                print(f"[replay] 删 PNG 失败，跳过: {p} ({e})")
        try:
            shutil.rmtree(session)
        except OSError as e:
            messagebox.showerror("删除本段", f"删除目录失败：\n{session}\n{e}", parent=self)
            return
        if self._last_session_path is not None:
            try:
                if self._last_session_path.resolve() == session:
                    self._last_session_path = None
            except OSError:
                self._last_session_path = None
        self.data = None
        self.session_var.set("")
        self._refresh_sessions(select_first=False)
        self.session_var.set("")
        self._clear_session()
        self._save_ui_settings()
        note = f"已删除 {session}"
        if n_png:
            note += f"；另删段外 PNG {n_png} 个"
        if skipped:
            note += f"；缺图跳过 {len(skipped)} 个"
        self.info_var.set(note)

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
            ("mash_count", self.mash_count_var, 3, 1),
            ("mash_gap_ms", self.mash_gap_var, 50, 10),
            ("th_ms", self.th_var, 50, 0),
            ("s", self.s_var, 20, 0),
            ("s_size", self.s_size_var, 20, 0),
            ("pm", self.pm_var, DEFAULT_PM, 0),
            ("pc", self.pc_var, DEFAULT_PC, 0),
            ("pt", self.pt_var, DEFAULT_PT, 1),
            ("pw_ms", self.pw_var, DEFAULT_PW_MS, 0),
            ("advance_timeout_ms", self.advance_timeout_var, DEFAULT_ADVANCE_TIMEOUT_MS, 1),
            ("approach_timeout_ms", self.approach_timeout_var, DEFAULT_APPROACH_TIMEOUT_MS, 1),
            ("advance_timeout_esc_n", self.advance_timeout_esc_n_var, DEFAULT_ADVANCE_TIMEOUT_ESC_N, 1),
            ("default_range_x", self.default_range_x_var, DEFAULT_RANGE_X, 1),
            ("default_range_y", self.default_range_y_var, DEFAULT_RANGE_Y, 1),
            ("town_s", self.town_s_var, int(DEFAULT_TOWN_S), 1),
            ("run_press", self.run_press_var, RUN_PRESS_GT, 0),
            ("e", self.e_var, 3, 0),
            ("f", self.f_var, DEFAULT_F, 0),
            ("press_lag_frames", self.press_lag_var, EXTRACT_PRESS_LAG_FRAMES, 0),
        ):
            if key in data:
                try:
                    var.set(str(max(lo, int(data[key]))))
                except (TypeError, ValueError):
                    var.set(str(default))
        if "s_pos" in data:
            try:
                self.s_var.set(str(max(0, int(data["s_pos"]))))
            except (TypeError, ValueError):
                pass
        if "s_size" not in data:
            self.s_size_var.set(self.s_var.get())
        if "th_ms" not in data:
            try:
                self.th_var.set(str(max(0, int(str(self.mash_gap_var.get() or 50)))))
            except (TypeError, ValueError):
                self.th_var.set("50")
        self.x_var.set(str(json_s(data, "x_s", 30.0)))
        self.stuck_recover_s_var.set(str(json_s(data, "stuck_recover_s", DEFAULT_STUCK_RECOVER_S, lo=0.05)))
        self._stuck_recover_spec = stuck_recover_to_json(parse_stuck_recover(data.get("stuck_recover")))
        self.ax_var.set(str(json_ms(data, "ax_ms", 1000)))
        self.ay_var.set(str(json_ms(data, "ay_ms", 1000)))
        self.advance_timeout_var.set(
            str(json_ms(data, "advance_timeout_ms", DEFAULT_ADVANCE_TIMEOUT_MS, lo=1))
        )
        self.approach_timeout_var.set(
            str(json_ms(data, "approach_timeout_ms", DEFAULT_APPROACH_TIMEOUT_MS, lo=1))
        )
        self.advance_timeout_esc_n_var.set(
            str(json_ms(data, "advance_timeout_esc_n", DEFAULT_ADVANCE_TIMEOUT_ESC_N, lo=1))
        )
        if "a" in data:
            warn_legacy_key("a")
        self.xxx_var.set(str(json_ms(data, "xxx_ms", 2000, lo=1)))
        tap_lo, tap_hi = parse_tap_ms_range(data)
        self.tap_ms_min_var.set(str(tap_lo))
        self.tap_ms_max_var.set(str(tap_hi))
        self.y_var.set(str(json_ms(data, "y_ms", 500, lo=1)))
        self.loot_hold_var.set(str(json_ms(data, "loot_hold_ms", 250, lo=1)))
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
        if "extract_include_fsm" in data:
            self.extract_fsm_var.set(bool(data["extract_include_fsm"]))
        last = str(data.get("last_session") or "").strip()
        if last:
            p = Path(last)
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                self._last_session_path = p
        cu, cd, cl, cr = mon_corr_from_dict(data)
        lu, ld, ll, lr = loot_corr_from_dict(data)
        gu, gd, gl, gr = gate_corr_from_dict(data)
        self._corr_lock = True
        self.corr_u.set(cu)
        self.corr_d.set(cd)
        self.corr_l.set(cl)
        self.corr_r.set(cr)
        self.loot_corr_u.set(lu)
        self.loot_corr_d.set(ld)
        self.loot_corr_l.set(ll)
        self.loot_corr_r.set(lr)
        self.gate_corr_u.set(gu)
        self.gate_corr_d.set(gd)
        self.gate_corr_l.set(gl)
        self.gate_corr_r.set(gr)
        self._corr_lock = False
        self._corr_prev = (cu, cd, cl, cr)
        self._corr_prev_map = {
            "mon": (cu, cd, cl, cr),
            "loot": (lu, ld, ll, lr),
            "gate": (gu, gd, gl, gr),
        }
        sh = data.get("skill_hold_ms")
        if not isinstance(sh, dict):
            if isinstance(data.get("skill_hold"), dict):
                warn_legacy_key("skill_hold")
            sh = None
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
            "stuck_recover_s": self._spin_f(self.stuck_recover_s_var, DEFAULT_STUCK_RECOVER_S, lo=0.05),
            "stuck_recover": list(getattr(self, "_stuck_recover_spec", None) or stuck_recover_to_json()),
            "xxx_ms": self._spin_n(self.xxx_var, 1000),
            "tap_ms_min": self._spin_n(self.tap_ms_min_var, DEFAULT_TAP_MS_MIN),
            "tap_ms_max": self._spin_n(self.tap_ms_max_var, DEFAULT_TAP_MS_MAX),
            "mash_count": self._spin_n(self.mash_count_var, 3),
            "mash_gap_ms": self._spin_n(self.mash_gap_var, 50, lo=10),
            "th_ms": self._spin_n(self.th_var, 50, lo=0),
            "s": self._spin_n(self.s_var, 20, lo=0),
            "s_pos": self._spin_n(self.s_var, 20, lo=0),
            "s_size": self._spin_n(self.s_size_var, 20, lo=0),
            "pm": self._spin_n(self.pm_var, DEFAULT_PM, lo=0),
            "pc": self._spin_n(self.pc_var, DEFAULT_PC, lo=0),
            "pt": self._spin_n(self.pt_var, DEFAULT_PT),
            "pw_ms": self._spin_n(self.pw_var, DEFAULT_PW_MS, lo=0),
            "advance_timeout_ms": self._spin_n(
                self.advance_timeout_var, DEFAULT_ADVANCE_TIMEOUT_MS, lo=1
            ),
            "approach_timeout_ms": self._spin_n(
                self.approach_timeout_var, DEFAULT_APPROACH_TIMEOUT_MS, lo=1
            ),
            "advance_timeout_esc_n": self._spin_n(
                self.advance_timeout_esc_n_var, DEFAULT_ADVANCE_TIMEOUT_ESC_N, lo=1
            ),
            "default_range_x": self._spin_n(self.default_range_x_var, DEFAULT_RANGE_X, lo=1),
            "default_range_y": self._spin_n(self.default_range_y_var, DEFAULT_RANGE_Y, lo=1),
            "town_s": self._spin_f(self.town_s_var, float(DEFAULT_TOWN_S), lo=0.05),
            "run_press": self._spin_n(self.run_press_var, RUN_PRESS_GT, lo=0),
            "loot_hold_ms": self._spin_n(self.loot_hold_var, LOOT_HOLD_DEFAULT),
            "e": self._spin_n(self.e_var, 3, lo=0),
            "f": self._spin_n(self.f_var, DEFAULT_F, lo=0),
            "press_lag_frames": self._press_lag_frames(),
            "fill": bool(self.fill_var.get()),
            "relative": bool(self.rel_var.get()),
            "overlay": bool(self.overlay_var.get()),
            "overlay_alpha": max(8, min(100, int(self.overlay_alpha_var.get() or 45))),
            "skill_hold_ms": self._skill_hold_saved,
            "session_source": (self.source_var.get() or "全部").strip() or "全部",
            "extract_include_fsm": bool(self.extract_fsm_var.get()),
            "last_session": (
                self._session_rel(self.data["session"])
                if self.data
                else (self._session_rel(self._last_session_path) if self._last_session_path else "")
            ),
        }
        u, dwn, left, right = self._corr_tuple("mon")
        data["mon_corr_u"] = u
        data["mon_corr_d"] = dwn
        data["mon_corr_l"] = left
        data["mon_corr_r"] = right
        lu, ld, ll, lr = self._corr_tuple("loot")
        data["loot_corr_u"] = lu
        data["loot_corr_d"] = ld
        data["loot_corr_l"] = ll
        data["loot_corr_r"] = lr
        gu, gd, gl, gr = self._corr_tuple("gate")
        data["gate_corr_u"] = gu
        data["gate_corr_d"] = gd
        data["gate_corr_l"] = gl
        data["gate_corr_r"] = gr
        lo = int(data["tap_ms_min"])
        hi = int(data["tap_ms_max"])
        if lo > hi:
            data["tap_ms_min"], data["tap_ms_max"] = hi, lo
        data.pop("tap_ms", None)
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
            mash_n=self._mash_n_from_ui(),
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
        prev = self._bind_character
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
        # 换角色时必须从该角色键位表重灌；否则会把上一角色旋钮值 flush 进新角色
        self._rebuild_skill_hold_ui(from_binds=reload_disk or (char != prev))

    def _rebuild_skill_hold_ui(self, *, from_binds: bool = False):
        self._skill_ui_guard = True
        self._combo_ui_guard = True
        self._mash_ui_guard = True
        self._multi_ui_guard = True
        try:
            # 不在这里 flush：旋钮仍属上一角色时，_bind_character 已切走会污染新角色缓存
            for child in self.skill_hold_host.winfo_children():
                child.destroy()
            self._skill_hold_vars = {}
            self._combo_vars = {}
            self._mash_vars = {}
            self._mash_n_vars = {}
            self._mash_n_spins = {}
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
                # 连按与连续释放合并：无单独「连续」勾选；勾选连按即写 combo 标签
                mash_on = bool(sk.get(MASH_KEY)) or slot in marked
                cvar = tk.BooleanVar(value=mash_on)
                self._combo_vars[slot] = cvar
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
                self._shield_right_wheel(sp)
                sp.pack(side=tk.LEFT, padx=(2, 0))
                nvar.trace_add("write", lambda *_: self._on_multi_n_change())
                self._multi_spins[slot] = sp
                mash_var = tk.BooleanVar(value=mash_on)
                self._mash_vars[slot] = mash_var
                ttk.Checkbutton(row, variable=mash_var, command=self._on_mash_toggle).pack(side=tk.LEFT, padx=(6, 0))
                mn = parse_mash_n(sk, self._spin_n(self.mash_count_var, DEFAULT_MASH_COUNT))
                mash_n_var = tk.StringVar(value=str(mn))
                self._mash_n_vars[slot] = mash_n_var
                mash_sp = ttk.Spinbox(
                    row,
                    from_=MASH_COUNT_MIN,
                    to=MASH_COUNT_MAX,
                    width=3,
                    textvariable=mash_n_var,
                    command=self._on_mash_n_change,
                    state=tk.NORMAL if mash_on else tk.DISABLED,
                )
                self._shield_right_wheel(mash_sp)
                mash_sp.pack(side=tk.LEFT, padx=(2, 0))
                mash_n_var.trace_add("write", lambda *_: self._on_mash_n_change())
                self._mash_n_spins[slot] = mash_sp
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
        ox, oy = mon_off_from_corr(*self._corr_tuple())
        frames = d["frames"]
        if ox or oy:
            frames = [self._apply_xy_corr(fr) for fr in frames]
        d["views"] = views_for_frames(
            frames,
            fill=bool(self.fill_var.get()),
            relative=bool(self.rel_var.get()),
            height=float(d["game_h"]) if d.get("game_h") else None,
        )
        self._rebuild_draft()

    def _warn_corr_reextract(self):
        msg = "补正已改：旧 skill_features 作废，请「重新提取本图」。jsonl 未改。"
        try:
            self.extract_prog_var.set("补正已改·请重提")
            self._set_extract_result(msg)
        except (tk.TclError, AttributeError):
            pass

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
            s_pos=self._spin_n(self.s_var, 20, lo=0),
            s_size=self._spin_n(self.s_size_var, 20, lo=0),
            slot_durs=self._slot_durs(),
            press_lag_frames=self._press_lag_frames(),
        )
        skip_ids = fake_member_ids(live_casts, f)
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

    def _jump_to_frame(self, _event=None):
        """跳到指定帧（与回放显示一致，从 1 起）。"""
        if not self.data:
            return
        n = len(self.data["frames"])
        if not n:
            return
        raw = (self.jump_var.get() or "").strip()
        try:
            f = int(raw)
        except ValueError:
            return
        i = max(1, min(n, f)) - 1
        self.idx = i
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
        lag = self._press_lag_frames()
        yi = i - lag
        run_on = False
        move_on = False
        edges: list[dict] = []
        held: list[str] = []
        if d.get("keys_path_exists") and (d["keys"] or d.get("held_at_start")):
            t_prev = int(d["frames"][i - 1]["t_ns"]) if i > 0 else int(fr["t_ns"]) - 1
            edges = events_in_range(d["keys"], t_prev, int(fr["t_ns"]))
            held = rebuild_held_at(d["keys"], int(fr["t_ns"]), d.get("held_at_start"))
        loot_on = False
        skill_hit = None
        y_edges: list[dict] = []
        y_held: list[str] = []
        if yi >= 0:
            fr_y = d["frames"][yi]
            loot_hold = d.get("loot_hold") or []
            if yi < len(loot_hold):
                loot_on = bool(loot_hold[yi])
            skill_hold = d.get("skill_hold") or []
            if yi < len(skill_hold):
                skill_hit = skill_hold[yi]
            if d.get("keys_path_exists") and (d["keys"] or d.get("held_at_start")):
                t_prev_y = int(d["frames"][yi - 1]["t_ns"]) if yi > 0 else int(fr_y["t_ns"]) - 1
                y_edges = events_in_range(d["keys"], t_prev_y, int(fr_y["t_ns"]))
                y_held = rebuild_held_at(d["keys"], int(fr_y["t_ns"]), d.get("held_at_start"))
            gt = self._spin_n(self.run_press_var, RUN_PRESS_GT, lo=0)
            run_on = run_burst_active(y_edges, gt=gt)
            move_on = (not run_on) and move_active(y_edges, y_held)
            x_atk = basic_attack_active(y_edges, y_held)
        else:
            x_atk = False
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
            origin_s = f"{mode}/{fill_on} xy={view['player_xy']}({tag})"
        self.info_var.set(
            f"t={t:7.2f}s{dt_s}  特征未变={stuck:5.2f}s{fsm_note}  {origin_s}"
            f"{infer_s}  {d['session'].name}{xy_note}"
        )
        skill_range = None
        skill_range_xy = None
        if skill_hit:
            room_n = (dr or {}).get("dist_key") or (dr or {}).get("rooms") or 0
            skill_range_xy = skill_range_xy_of(self._feature_cache, room_n, int(skill_hit[0]))
            skill_range = skill_range_of(self._feature_cache, room_n, int(skill_hit[0]))
            now_dist = pack_dist_from_player(view)
            bits = [f"技能{skill_hit[0]}：{skill_hit[1]}"]
            if skill_range_xy is not None:
                bits.append(f"范围X{skill_range_xy[0]:.0f}/Y{skill_range_xy[1]:.0f}")
            elif skill_range is not None:
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
            skill_range_xy=skill_range_xy,
            cd_reset=reset_here,
            counts=counts,
            overlay_path=self._png_path_for_frame(fr) if self.overlay_var.get() else None,
            player_tl=self._overlay_player_tl(fr, view, d.get("game_h") or 0),
            state_hint=state_hint_for_frame(d.get("states") or [], d["frames"], i),
        )

    def _show_det_runs(self):
        if not self.data:
            messagebox.showinfo("检测游程", "未打开录像段。")
            return
        report = format_det_run_report(self.data.get("det_stats"))
        win = tk.Toplevel(self)
        win.title("检测游程")
        win.minsize(520, 360)
        txt = tk.Text(win, wrap=tk.NONE, font=("Consolas", 11), padx=10, pady=10)
        ys = ttk.Scrollbar(win, orient=tk.VERTICAL, command=txt.yview)
        xs = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=txt.xview)
        txt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        txt.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)
        txt.insert("1.0", report)
        txt.configure(state=tk.DISABLED)

    def _show_fight_sort(self):
        """同角色+同图开打段：按每只怪耗时降序 + 汇总。"""
        if not self.data:
            messagebox.showinfo("开打排序", "未打开录像段。", parent=self)
            return
        dun = self._current_dungeon()
        char = character_of_session(self.data["session"], self.data.get("meta") or {}) or (
            self._bind_character or ""
        )
        rows = collect_fight_segments(dun, character=char, session=self.data.get("session"))
        report = format_fight_sort_report(rows)
        win = tk.Toplevel(self)
        win.title(f"开打排序 · {char or '?'} · {dun} · {len(rows)} 段")
        win.minsize(960, 420)
        win.geometry("1100x520")
        txt = tk.Text(win, wrap=tk.NONE, font=("Consolas", 10), padx=10, pady=10)
        ys = ttk.Scrollbar(win, orient=tk.VERTICAL, command=txt.yview)
        xs = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=txt.xview)
        txt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        txt.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)
        txt.insert("1.0", report)
        txt.configure(state=tk.DISABLED)

    def _show_round_time_bars(self):

        """同角色+同图 states.jsonl：按文件名纵向叠分段横条 + 汇总表（不设轮数上限）。"""
        if not self.data:
            messagebox.showinfo("每轮时间条", "未打开录像段。", parent=self)
            return
        dun = self._current_dungeon()
        char = character_of_session(self.data["session"], self.data.get("meta") or {}) or (
            self._bind_character or ""
        )
        rounds = collect_dungeon_rounds(dun, character=char, session=self.data.get("session"))
        if not rounds:
            messagebox.showinfo(
                "每轮时间条",
                f"角色「{char or '?'}」图「{dun}」没有可用的 states.jsonl。",
                parent=self,
            )
            return
        cur_name = Path(self.data["session"]).name

        win = tk.Toplevel(self)
        win.title(f"每轮时间条 · {char or '?'} · {dun} · {len(rounds)} 段")
        win.minsize(900, 520)
        win.geometry("1000x640")

        top = ttk.Frame(win, padding=8)
        top.pack(fill=tk.BOTH, expand=True)
        tip = ttk.Label(
            top,
            text=f"范围：角色 {char or '?'} + 图 {dun}（标签=目录名，不限段数）。点击条可跳到该段。",
        )
        tip.pack(anchor=tk.W)

        opts = ttk.Frame(top)
        opts.pack(fill=tk.X, pady=(4, 0))
        include_idle_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="含空闲（图+统计）", variable=include_idle_var).pack(side=tk.LEFT)

        canvas_host = ttk.Frame(top)
        canvas_host.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        ys = ttk.Scrollbar(canvas_host, orient=tk.VERTICAL)
        xs = ttk.Scrollbar(canvas_host, orient=tk.HORIZONTAL)
        cv = tk.Canvas(canvas_host, bg="#1a1d24", highlightthickness=0)
        ys.config(command=cv.yview)
        xs.config(command=cv.xview)
        cv.config(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        xs.pack(side=tk.BOTTOM, fill=tk.X)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tbl = ttk.LabelFrame(top, text="汇总表", padding=6)
        tbl.pack(fill=tk.X, pady=(8, 0))
        txt = tk.Text(tbl, height=8, wrap=tk.NONE, font=("Consolas", 10))
        txt.pack(fill=tk.X)

        label_w = 260
        bar_h = 22
        gap = 8
        left = 8
        top_y = 12
        bar_x0 = left + label_w + 8
        bar_max_w = 620
        path_by_tag: dict[str, Path] = {}

        def _segs_filtered(segs: list[tuple[str, int]]) -> list[tuple[str, int]]:
            if include_idle_var.get():
                return list(segs)
            return [(st, ms) for st, ms in segs if st != "空闲"]

        def _redraw(*_a):
            cv.delete("all")
            path_by_tag.clear()
            color_seen: dict[str, str] = {}
            drawn = [(name, sp, _segs_filtered(segs)) for name, sp, segs in rounds]
            drawn = [(n, p, s) for n, p, s in drawn if s]
            max_total = max((sum(ms for _s, ms in segs) for _n, _p, segs in drawn), default=1) or 1
            y = top_y
            for name, sess_path, segs in drawn:
                total = sum(ms for _s, ms in segs) or 1
                mark = " ◆" if name == cur_name else ""
                fill = "#e8c547" if name == cur_name else "#c8cdd8"
                cv.create_text(
                    left + label_w,
                    y + bar_h / 2,
                    anchor=tk.E,
                    fill=fill,
                    font=("Consolas", 9),
                    text=name + mark,
                )
                x = bar_x0
                scale = bar_max_w / max_total
                for st, ms in segs:
                    w = max(2.0, ms * scale)
                    col = _round_bar_color(st, color_seen)
                    tag = f"bar:{name}"
                    path_by_tag[tag] = sess_path
                    cv.create_rectangle(
                        x, y, x + w, y + bar_h, fill=col, outline="#111", width=1, tags=(tag,)
                    )
                    if w >= 36:
                        cv.create_text(
                            x + w / 2,
                            y + bar_h / 2,
                            fill="#111",
                            font=("Microsoft YaHei", 8),
                            text=st,
                            tags=(tag,),
                        )
                    x += w
                cv.create_text(
                    x + 8,
                    y + bar_h / 2,
                    anchor=tk.W,
                    fill="#9ab",
                    font=("Consolas", 10),
                    text=f"{total / 1000.0:.1f}s",
                )
                y += bar_h + gap

            lx = bar_x0
            ly = y + 6
            legend_states: list[str] = []
            want = ("前进", "开打", "捡物", "空闲") if include_idle_var.get() else ("前进", "开打", "捡物")
            for st in want:
                legend_states.append(st)
            for st in color_seen:
                if st not in legend_states:
                    legend_states.append(st)
            for st in legend_states:
                col = _round_bar_color(st, color_seen)
                cv.create_rectangle(lx, ly, lx + 14, ly + 14, fill=col, outline="#111")
                cv.create_text(
                    lx + 18,
                    ly + 7,
                    anchor=tk.W,
                    fill="#c8cdd8",
                    font=("Microsoft YaHei", 9),
                    text=st,
                )
                lx += 18 + 12 * max(2, len(st)) + 16
            y = ly + 28
            cv.configure(scrollregion=(0, 0, bar_x0 + bar_max_w + 120, y + 8))
            for tag in path_by_tag:
                cv.tag_bind(tag, "<Button-1>", _on_bar_click)

            summary = summarize_round_segments(rounds, include_idle=bool(include_idle_var.get()))
            txt.configure(state=tk.NORMAL)
            txt.delete("1.0", tk.END)
            txt.insert("1.0", format_round_summary_table(summary))
            txt.configure(state=tk.DISABLED)

        def _on_bar_click(ev):
            items = cv.find_withtag("current")
            if not items:
                return
            tags = cv.gettags(items[0])
            for t in tags:
                if t.startswith("bar:") and t in path_by_tag:
                    self._select_path(path_by_tag[t])
                    break

        include_idle_var.trace_add("write", _redraw)
        _redraw()


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
            | {int(s) for s in self._mash_n_vars}
        )

    def _combo_slots_from_ui(self) -> set[int]:
        if self._combo_vars:
            return {slot for slot, var in self._combo_vars.items() if var.get()}
        return combo_slots_of(self._feature_cache)

    def _mash_slots_from_ui(self) -> set[int]:
        return {slot for slot, var in self._mash_vars.items() if var.get()}

    def _mash_n_from_ui(self) -> dict[int, int]:
        default = self._spin_n(self.mash_count_var, DEFAULT_MASH_COUNT)
        out: dict[int, int] = {}
        for slot, var in self._mash_vars.items():
            if not var.get():
                continue
            nvar = self._mash_n_vars.get(slot)
            try:
                n = int(str((nvar.get() if nvar else "") or default))
            except ValueError:
                n = default
            out[int(slot)] = clamp_mash_count(n, default)
        return out

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
        self._combo_ui_guard = True
        try:
            for slot, mvar in self._mash_vars.items():
                on = bool(mvar.get())
                cvar = self._combo_vars.get(slot)
                if cvar is not None:
                    cvar.set(on)
                sp = self._mash_n_spins.get(slot)
                if sp is not None:
                    sp.config(state=tk.NORMAL if on else tk.DISABLED)
                    if on:
                        nvar = self._mash_n_vars.get(slot)
                        try:
                            n = int(str((nvar.get() if nvar else "") or DEFAULT_MASH_COUNT))
                        except ValueError:
                            n = 0
                        if nvar is not None and (n < MASH_COUNT_MIN or n > MASH_COUNT_MAX):
                            nvar.set(str(self._spin_n(self.mash_count_var, DEFAULT_MASH_COUNT)))
        finally:
            self._combo_ui_guard = False
        self._sync_combo_to_features(save=True)
        self._write_hold_frames_to_binds()
        if self.data:
            self._rebuild_draft()
            self._show()

    def _on_mash_n_change(self):
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
        ox, oy = mon_off_from_corr(*self._corr_tuple())
        view_frames = frames
        if ox or oy:
            view_frames = [self._apply_xy_corr(fr) for fr in frames]
        views = views_for_frames(
            view_frames,
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
        roots = (RECORDINGS, FSM_TEST_DIR) if bool(self.extract_fsm_var.get()) else (RECORDINGS,)
        out = []
        for p in list_sessions(*roots):
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

    def _extract_scope_text(self, sessions: list[Path], all_same: bool) -> str:
        n_fsm = sum(1 for p in sessions if session_kind(p) == "fsm")
        n_rec = len(sessions) - n_fsm
        if not all_same:
            kind = "FSM测试" if n_fsm else "采集"
            return f"当前这一段（{kind}）"
        extra = f"采集 {n_rec} 段 + FSM测试 {n_fsm} 段" if bool(self.extract_fsm_var.get()) else f"采集 {n_rec} 段（不含 FSM测试）"
        return f"同地下城同角色全部 {len(sessions)} 段，{extra}"

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
        scope = self._extract_scope_text(sessions, True if wipe else all_same)
        if wipe:
            ok = messagebox.askyesno(
                "确认重新提取本图",
                f"将清空\n{dest}\n里现有 {have} 条，再从{scope}重新提取。\n\n"
                f"杀MON数>E={e} 记为群。清空前另存一份副本，可用「回退上次叠加」还原（用过即删）。\n\n"
                f"确定重新提取？",
                parent=self,
            )
        else:
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
        lag = self._press_lag_frames()
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
            args=(char, dun, sessions, e, wipe, set(self._combo_slots_from_ui()) | set(self._mash_slots_from_ui()), dict(self._multi_n_from_ui()), slot_durs, f, lag),
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
        press_lag_frames: int = EXTRACT_PRESS_LAG_FRAMES,
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
        s_size = self._spin_n(self.s_size_var, 20, lo=0)
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
                    s_pos=s_pct,
                    s_size=s_size,
                    extra_sigs=extra_sigs,
                    slot_durs=slot_durs,
                    press_lag_frames=press_lag_frames,
                )
                all_casts.extend(casts)
                skip_ids = fake_member_ids(casts, f)
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

    def _show_current_features(self):
        """打开当前录像对应地下城/角色的特征表（磁盘最新）。"""
        char = self._bind_character
        dun = self._current_dungeon() if self.data else ""
        if not char:
            messagebox.showinfo("过图技能特征", "当前录像没有角色名。", parent=self)
            return
        if not dun or dun == "未知":
            messagebox.showinfo("过图技能特征", "当前录像没有地下城名。", parent=self)
            return
        self._reload_feature_cache()
        dest = feature_path(char, dun)
        data = self._feature_cache
        if not data:
            messagebox.showinfo("过图技能特征", f"没有特征表：\n{dest}", parent=self)
            return
        report = format_report(data)
        self._set_extract_result(report)
        self._show_extract_window(report, title=f"过图技能特征 · {dun} / {char}")

    def _show_extract_window(self, report: str, title: str = "过图技能特征 · 结果"):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("720x480")
        apply_window_geom(win, "fsm_extract", min_w=480, min_h=280, fallback="720x480")
        txt = tk.Text(win, font=("Consolas", 10), wrap=tk.WORD)
        ys = ttk.Scrollbar(win, orient=tk.VERTICAL, command=txt.yview)
        txt.configure(yscrollcommand=ys.set)
        txt.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        txt.insert("1.0", report)
        txt.config(state=tk.DISABLED)

        def _close():
            remember_window_geom(win, "fsm_extract")
            win.destroy()

        ttk.Button(win, text="关闭", command=_close).grid(row=1, column=0, columnspan=2, pady=6)
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)
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


    def _save_current_png(self):
        """把当前帧叠图用的背景 PNG 拷到 images/miss_png，便于漏检收集。"""
        d = self.data
        if not d or not d.get("frames"):
            messagebox.showinfo("保存PNG", "没有加载录像。", parent=self)
            return
        i = int(self.idx)
        frames = d["frames"]
        if i < 0 or i >= len(frames):
            messagebox.showinfo("保存PNG", "帧号无效。", parent=self)
            return
        fr = frames[i]
        src = self._png_path_for_frame(fr)
        if src is None or not src.is_file():
            messagebox.showinfo(
                "保存PNG",
                "当前帧没有对应 PNG（帧字段 png 为空或文件不存在）。",
                parent=self,
            )
            return
        MISS_PNG_DIR.mkdir(parents=True, exist_ok=True)
        sess = getattr(d.get("session"), "name", None) or "session"
        dest = MISS_PNG_DIR / f"{sess}_f{i + 1:05d}_{src.name}"
        n = 1
        while dest.exists():
            dest = MISS_PNG_DIR / f"{sess}_f{i + 1:05d}_{n}_{src.name}"
            n += 1
        shutil.copy2(src, dest)
        messagebox.showinfo("保存PNG", f"已保存：\n{dest}", parent=self)

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
        skill_range_xy: tuple[float, float] | None = None,
        cd_reset: bool = False,
        counts: dict[str, int] | None = None,
        overlay_path: Path | None = None,
        player_tl: list[float] | None = None,
        state_hint: str = "",
    ):
        self._ensure_icons()
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

        hint = (state_hint or "").strip() or "无 states.jsonl"

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
            range_xy = skill_range_xy
            if range_xy is None and skill_range is not None and skill_range > 0:
                range_xy = (skill_range, skill_range)
            if range_xy is not None:
                rx_g, ry_g = float(range_xy[0]), float(range_xy[1])
                if rx_g > 0 and ry_g > 0:
                    pxy = view.get("player_xy") if view else None
                    if isinstance(pxy, (list, tuple)) and len(pxy) >= 2:
                        px, py = to_canvas(pxy)
                        hx = rx_g / gw * cw
                        hy = ry_g / gh * ch
                        c.create_rectangle(
                            px - hx,
                            py - hy,
                            px + hx,
                            py + hy,
                            outline="#e8c547",
                            width=2,
                            dash=(8, 4),
                        )
                        c.create_text(
                            px,
                            py - hy - 10,
                            fill="#e8c547",
                            font=("Consolas", 12, "bold"),
                            text=f"范围 X{rx_g:.0f}/Y{ry_g:.0f}",
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
