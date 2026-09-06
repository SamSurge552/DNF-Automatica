"""过图技能特征：快捷栏技能（含 SPACE）按下 → 按按下时现算的怪物分布整理。

不进 fsm_core。相对坐标只在提取时算，不改 jsonl。
调用方须对 frames/views 做 apply_mon_boss_corr（与 FSM 同一 mon_off），再算最远敌对与范围。
落盘：skill_features/<地下城>/<角色>.json
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

from fsm_core import DistSig, DistSkillPlan, FightSkill, classify_dist, expand_fight_skills, parse_hold_ms

FEATURES_DIR = Path(__file__).resolve().parent / "skill_features"
BINDS_DIR = Path(__file__).resolve().parent / "skill_binds"
TZ8 = timezone(timedelta(hours=8))
DEFAULT_E = 3
_UNSAFE = r'\/:*?"<>|'
TAG_COMBO = "连续释放"
TAG_MULTI = "多次释放"
MAP_RESET = "CD重置"
MAP_RESET_LEGACY = "重置"
TAG_RESET_POINT = "重置点"
COMBO_CD_S = 270.0
DEFAULT_F = 20
DEFAULT_MULTI = 2
MULTI_N_MAX = 9
EXTRACT_PRESS_LAG_FRAMES = 1  # 仅默认；提取 i_feat 与回放黄字共用 UI 传入的 press_lag_frames


def _now_iso() -> str:
    return datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def _safe_name(name: str, fallback: str) -> str:
    s = "".join(ch for ch in (name or "").strip() if ch not in _UNSAFE)
    return s or fallback


def feature_path(character: str, dungeon: str) -> Path:
    return FEATURES_DIR / _safe_name(dungeon, "未知") / f"{_safe_name(character, 'character')}.json"


def prev_path(character: str, dungeon: str) -> Path:
    p = feature_path(character, dungeon)
    return p.with_name(p.stem + ".prev.json")


def legacy_feature_path(character: str) -> Path:
    """旧路径 skill_features/<角色>.json（不再写入）。"""
    return FEATURES_DIR / f"{_safe_name(character, 'character')}.json"


def bind_path(character: str) -> Path:
    return BINDS_DIR / f"{_safe_name(character, 'character')}.json"


def skill_table_missing(character: str) -> bool:
    """技能表文件不存在或 skills 为空。"""
    if not (character or "").strip():
        return True
    path = bind_path(character)
    if not path.is_file():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if not isinstance(data, dict):
        return True
    skills = data.get("skills")
    return not isinstance(skills, list) or len(skills) == 0


def maybe_adopt_legacy(character: str, dungeon: str) -> Path | None:
    """若新路径还没有表，把旧的角色级 json 挪进该地下城目录。"""
    new = feature_path(character, dungeon)
    if new.is_file():
        return None
    old = legacy_feature_path(character)
    if not old.is_file():
        return None
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    old_prev = old.with_name(old.stem + ".prev.json")
    if old_prev.is_file():
        shutil.move(str(old_prev), str(prev_path(character, dungeon)))
    return new


def load_features(character: str, dungeon: str) -> dict | None:
    path = feature_path(character, dungeon)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def guess_dungeon(character: str) -> str:
    """该角色只有一份特征表时返回地下城名，否则空。"""
    name = _safe_name(character, "")
    if not name:
        return ""
    hits = [p.parent.name for p in FEATURES_DIR.glob(f"*/{name}.json") if p.is_file()]
    return hits[0] if len(hits) == 1 else ""


def load_fight_plan(
    character: str, dungeon: str, f: int | None = None
) -> tuple[DistSkillPlan, ...]:
    data = load_features(character, dungeon)
    if data:
        apply_f(data, data.get("f", DEFAULT_F) if f is None else f)
    binds_path = bind_path(character)
    skills: list[dict] = []
    if binds_path.is_file():
        try:
            raw = json.loads(binds_path.read_text(encoding="utf-8"))
            skills = list((raw or {}).get("skills") or [])
        except Exception:
            skills = []
    return fight_plan_from_features(data, skills)


def load_hotbar(character: str) -> tuple[FightSkill, ...]:
    binds_path = bind_path(character)
    skills: list[dict] = []
    if binds_path.is_file():
        try:
            raw = json.loads(binds_path.read_text(encoding="utf-8"))
            skills = list((raw or {}).get("skills") or [])
        except Exception:
            skills = []
    return hotbar_from_binds(skills)


def load_dist_table(character: str, dungeon: str) -> tuple[DistSig, ...]:
    return dist_table_from_features(load_features(character, dungeon))


def count_casts(data: dict | None) -> int:
    n = 0
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            n += len(sk.get("casts") or [])
    return n


def _rooms_or_dists(data: dict | None) -> dict:
    data = data or {}
    if data.get("dists"):
        return data.get("dists") or {}
    return data.get("rooms") or {}


def skill_spans(skill_hold: list) -> list[tuple[int, int, int, str]]:
    """连续黄字同一技能 → (i0, i1, slot, key)，闭区间。"""
    out: list[tuple[int, int, int, str]] = []
    i0 = None
    cur = None
    for i, hit in enumerate(skill_hold):
        if hit != cur:
            if cur is not None and i0 is not None:
                out.append((i0, i - 1, int(cur[0]), str(cur[1])))
            cur = hit
            i0 = i if hit is not None else None
    if cur is not None and i0 is not None:
        out.append((i0, len(skill_hold) - 1, int(cur[0]), str(cur[1])))
    return out


def _pts(view: dict, name: str) -> list[list[float]]:
    raw = (view or {}).get(f"{name}_xy") or []
    out = []
    for p in raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
    return out


def farthest_enemy(view: dict | None) -> dict:
    """按下时：MON+BOSS 总数；数量>1 取相对玩家最远点。"""
    view = view or {}
    mon_n = int(view.get("mon") or 0)
    boss_n = int(view.get("boss") or 0)
    pts = _pts(view, "mon") + _pts(view, "boss")
    pxy = view.get("player_xy")
    origin = (0.0, 0.0)
    if isinstance(pxy, (list, tuple)) and len(pxy) >= 2:
        origin = (float(pxy[0]), float(pxy[1]))
    far = None
    dist = None
    direction = None
    if pts:
        if len(pts) == 1:
            far = pts[0]
        else:
            far = max(pts, key=lambda p: math.hypot(p[0] - origin[0], p[1] - origin[1]))
        rel = [round(far[0] - origin[0], 2), round(far[1] - origin[1], 2)]
        dist = round(math.hypot(rel[0], rel[1]), 2)
        cx, cy = rel
        if abs(cx) < 1e-6 and abs(cy) < 1e-6:
            direction = "重叠"
        elif abs(cx) >= abs(cy):
            direction = "right" if cx > 0 else "left"
        else:
            direction = "up" if cy > 0 else "down"
        far = rel
    return {
        "mon": mon_n,
        "boss": boss_n,
        "enemy": mon_n + boss_n,
        "center": far,
        "dist": dist,
        "dir": direction,
    }


def _ratio(killed: float, start: float) -> float | None:
    if start <= 0:
        return None
    return round(killed / start, 3)


def _xy_tuples(view: dict, name: str) -> tuple[tuple[float, float], ...]:
    return tuple((p[0], p[1]) for p in _pts(view, name))


def _view_player(view: dict | None) -> tuple[float, float] | None:
    pxy = (view or {}).get("player_xy")
    if isinstance(pxy, (list, tuple)) and len(pxy) >= 2:
        return (float(pxy[0]), float(pxy[1]))
    return None


def casts_from_tracks(
    session: str,
    draft: list[dict],
    skill_hold: list,
    views: list[dict],
    e: int,
    skills: list[dict] | None = None,
    dist_table: tuple[DistSig, ...] = (),
    s: int = 20,
    extra_sigs: tuple[DistSig, ...] = (),
    slot_durs: dict[int, int] | None = None,
    press_lag_frames: int = EXTRACT_PRESS_LAG_FRAMES,
) -> tuple[list[dict], tuple[DistSig, ...]]:
    """快捷栏技能（含 SPACE）。skill_hold 必须是真实 press onset，不要预先平移。
    分布/最远敌对用 views[i_feat]；t0/hold/i_end/CD 仍从黄字 i0 起算。"""
    e = max(0, int(e))
    s_pct = max(0.0, float(s))
    lag = max(0, int(press_lag_frames))
    hold_of = {slot: hf for slot, (_k, _cd, hf) in _bind_by_slot(skills).items()}
    for slot, hf in (slot_durs or {}).items():
        try:
            hold_of[int(slot)] = max(0, int(hf))
        except (TypeError, ValueError):
            continue
    n = len(skill_hold)
    extra = extra_sigs
    out = []
    n_views = len(views)
    for i0, i1, slot, key in skill_spans(skill_hold):
        if i0 >= len(draft):
            continue
        hold = max(0, int(hold_of.get(slot, 0)))
        t0 = int((views[i0] if i0 < n_views else {}).get("t_ns") or (draft[i0] or {}).get("t_ns") or 0)
        end_i = i0
        lim = n if n else n_views
        for j in range(i0, lim):
            row = views[j] if j < n_views else (draft[j] if j < len(draft) else {})
            tj = int(row.get("t_ns") or t0)
            end_i = j
            if (tj - t0) / 1e6 >= hold:
                break
        last_v = max(n_views - 1, 0)
        i_feat = min(i0 + lag, end_i, last_v)
        v_feat = views[i_feat] if i_feat < n_views else {}
        v1 = views[end_i] if end_i < n_views else {}
        a = farthest_enemy(v_feat)
        b = farthest_enemy(v1)
        killed_mon = a["mon"] - b["mon"]
        killed_enemy = a["enemy"] - b["enemy"]
        dist_key, dist_kind, extra, _rk = classify_dist(
            _view_player(v_feat),
            _xy_tuples(v_feat, "mon"),
            _xy_tuples(v_feat, "boss"),
            int(v_feat.get("mon") or a["mon"] or 0),
            int(v_feat.get("boss") or a["boss"] or 0),
            dist_table,
            extra,
            s_pct,
        )
        is_boss = dist_kind == "boss"
        kind = None if is_boss else ("群" if killed_mon > e else "单")
        out.append(
            {
                "id": f"{session}|{i0}|{slot}",
                "session": session,
                "dist_key": dist_key,
                "dist_kind": dist_kind,
                "room": dist_key or "X",
                "slot": slot,
                "key": key,
                "i0": i0,
                "i_feat": i_feat,
                "i1": i1,
                "i_end": end_i,
                "mon": a["mon"],
                "boss": a["boss"],
                "enemy": a["enemy"],
                "center": a["center"],
                "dist": a["dist"],
                "dir": a["dir"],
                "kind": kind,
                "boss_room": is_boss,
                "mon_end": b["mon"],
                "boss_end": b["boss"],
                "enemy_end": b["enemy"],
                "killed_mon": killed_mon,
                "killed_enemy": None if is_boss else killed_enemy,
                "kill_ratio": None if is_boss else _ratio(killed_mon, a["mon"]),
                "kill_ratio_enemy": None if is_boss else _ratio(killed_enemy, a["enemy"]),
            }
        )
    return out, extra


def _cooldown_by_slot(skills: list[dict] | None) -> dict[int, float]:
    out: dict[int, float] = {}
    for sk in skills or []:
        try:
            slot = int(sk.get("slot") or 0)
            cd = float(sk.get("cooldown_s"))
        except (TypeError, ValueError):
            continue
        if slot > 0 and cd > 0:
            out[slot] = cd
    return out


def bind_hotkey(skill: dict | None) -> str | None:
    """有快捷栏勾选的单键，或指令为 space。没有快捷键的技能排除。"""
    if not isinstance(skill, dict):
        return None
    cmd = str(skill.get("command") or skill.get("key") or "").strip().lower()
    if not cmd or "+" in cmd or "＋" in cmd:
        return None
    parts = cmd.split()
    if len(parts) != 1:
        return None
    key = parts[0]
    if skill.get("hotbar") or key == "space":
        return key
    return None


def _bind_by_slot(skills: list[dict] | None) -> dict[int, tuple[str, float, int]]:
    out: dict[int, tuple[str, float, int]] = {}
    for sk in skills or []:
        try:
            slot = int(sk.get("slot") or 0)
        except (TypeError, ValueError):
            continue
        if slot <= 0:
            continue
        key = str(sk.get("command") or sk.get("key") or "").strip().lower()
        cd = 0.0
        try:
            if sk.get("cooldown_s") is not None:
                cd = max(0.0, float(sk.get("cooldown_s")))
        except (TypeError, ValueError):
            pass
        hf = parse_hold_ms(sk, 0) or 0
        out[slot] = (key, cd, hf)
    return out


def parse_multi_n(item: dict | None) -> int:
    """键位表：勾选多次释放 → MULTI（默认 2，至少 2）；未勾选 → 1。"""
    if not isinstance(item, dict) or not item.get("multi"):
        return 1
    try:
        n = int(item.get("multi_n", DEFAULT_MULTI))
    except (TypeError, ValueError):
        n = DEFAULT_MULTI
    return max(DEFAULT_MULTI, min(MULTI_N_MAX, n))


def multi_n_of(data: dict | None) -> dict[int, int]:
    """特征表里标了多次释放的槽 → 次数。"""
    out: dict[int, int] = {}
    raw = (data or {}).get("skill_multi_n") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                slot, n = int(k), int(v)
            except (TypeError, ValueError):
                continue
            if slot > 0:
                out[slot] = max(DEFAULT_MULTI, min(MULTI_N_MAX, n))
    tags = (data or {}).get("skill_tags") or {}
    if isinstance(tags, dict):
        for k, v in tags.items():
            try:
                slot = int(k)
            except (TypeError, ValueError):
                continue
            names = v if isinstance(v, list) else [v]
            if TAG_MULTI in names and slot > 0:
                out.setdefault(slot, DEFAULT_MULTI)
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            names = sk.get("tags") or []
            if sk.get("multi") or TAG_MULTI in names:
                try:
                    slot = int(sk.get("slot") or 0)
                except (TypeError, ValueError):
                    continue
                if slot > 0:
                    try:
                        n = int(sk.get("multi_n", out.get(slot, DEFAULT_MULTI)))
                    except (TypeError, ValueError):
                        n = out.get(slot, DEFAULT_MULTI)
                    out[slot] = max(DEFAULT_MULTI, min(MULTI_N_MAX, n))
    return out


def set_multi_slots(data: dict, mapping) -> dict:
    """写入 skill_multi_n / skill_tags【多次释放】，并同步到各房技能。"""
    want: dict[int, int] = {}
    if isinstance(mapping, dict):
        items = mapping.items()
    else:
        items = ((s, DEFAULT_MULTI) for s in (mapping or []))
    for s, n in items:
        try:
            slot, times = int(s), int(n)
        except (TypeError, ValueError):
            continue
        if slot > 0:
            want[slot] = max(DEFAULT_MULTI, min(MULTI_N_MAX, times))
    data["skill_multi_n"] = {str(k): v for k, v in sorted(want.items())}
    tags = dict(data.get("skill_tags") or {})
    known = set(want)
    for k in list(tags):
        try:
            known.add(int(k))
        except (TypeError, ValueError):
            continue
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            try:
                known.add(int(sk.get("slot") or 0))
            except (TypeError, ValueError):
                continue
    new_tags: dict[str, list[str]] = {}
    for slot in sorted(known):
        if slot <= 0:
            continue
        prev = tags.get(str(slot)) or []
        if isinstance(prev, str):
            prev = [prev]
        names = [str(x) for x in prev if x and x != TAG_MULTI]
        if slot in want:
            names.append(TAG_MULTI)
        if names:
            new_tags[str(slot)] = names
    data["skill_tags"] = new_tags
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            try:
                slot = int(sk.get("slot") or 0)
            except (TypeError, ValueError):
                continue
            t = [x for x in (sk.get("tags") or []) if x != TAG_MULTI]
            if slot in want:
                t.append(TAG_MULTI)
                sk["multi"] = True
                sk["multi_n"] = want[slot]
            else:
                sk.pop("multi", None)
                sk.pop("multi_n", None)
            if t:
                sk["tags"] = t
            else:
                sk.pop("tags", None)
    for room in _rooms_or_dists(data).values():
        if isinstance(room.get("skills"), dict):
            room["skills"] = _sort_skills(room["skills"], str(room.get("kind") or ""))
    data["sequences"] = sequences_of(data)
    return data


def _real_casts(sk: dict) -> list[dict]:
    return [c for c in (sk.get("casts") or []) if not c.get("fake")]


def _cast_n(sk: dict) -> int:
    return len(_real_casts(sk))


def sort_skills_by_freq(skills: dict | None) -> dict:
    """释放次数从高到低，次数相同按槽位。"""

    def key(kv):
        k, sk = kv
        try:
            slot = int(k)
        except (TypeError, ValueError):
            slot = 0
        return (-_cast_n(sk or {}), slot)

    return dict(sorted((skills or {}).items(), key=key))


def sort_skills_by_ratio(skills: dict | None) -> dict:
    """杀MON效率从高到低；效率相同按释放次数比率从高到低。"""
    items = list((skills or {}).items())
    total = sum(_cast_n(sk or {}) for _, sk in items) or 1

    def key(kv):
        k, sk = kv
        try:
            slot = int(k)
        except (TypeError, ValueError):
            slot = 0
        ratio = (sk or {}).get("kill_ratio_mean")
        try:
            r = float(ratio) if ratio is not None else -1.0
        except (TypeError, ValueError):
            r = -1.0
        n = _cast_n(sk or {})
        return (-r, -(n / total), slot)

    return dict(sorted(items, key=key))


def _sort_skills(skills: dict | None, kind: str) -> dict:
    if kind == "boss":
        return sort_skills_by_freq(skills)
    return sort_skills_by_ratio(skills)


def sequences_of(data: dict | None) -> dict[str, list[int]]:
    """BOSS 分布按次数；小怪分布按效率，效率相同按释放次数比率。"""
    out: dict[str, list[int]] = {}
    for rk, room in _rooms_or_dists(data).items():
        kind = str(room.get("kind") or "")
        ordered = _sort_skills(room.get("skills") or {}, kind)
        slots = []
        for sk in ordered.values():
            try:
                slot = int(sk.get("slot") or 0)
            except (TypeError, ValueError):
                continue
            if slot > 0 and _cast_n(sk) > 0:
                slots.append(slot)
        out[str(rk)] = slots
    return out


def combo_slots_of(data: dict | None) -> set[int]:
    """特征表里标了「连续释放」的槽。"""
    out: set[int] = set()
    tags = (data or {}).get("skill_tags") or {}
    if isinstance(tags, dict):
        combo_list = tags.get("combo")
        if isinstance(combo_list, list):
            for x in combo_list:
                try:
                    slot = int(x)
                except (TypeError, ValueError):
                    continue
                if slot > 0:
                    out.add(slot)
        for k, v in tags.items():
            if k == "combo":
                continue
            try:
                slot = int(k)
            except (TypeError, ValueError):
                continue
            names = v if isinstance(v, list) else [v]
            if TAG_COMBO in names and slot > 0:
                out.add(slot)
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            names = sk.get("tags") or []
            if sk.get("combo") or TAG_COMBO in names:
                try:
                    slot = int(sk.get("slot") or 0)
                except (TypeError, ValueError):
                    continue
                if slot > 0:
                    out.add(slot)
    return out


def set_combo_slots(data: dict, slots) -> dict:
    """写入 skill_tags，并同步到各房技能上的 tags/combo。"""
    want = set()
    for s in slots or []:
        try:
            n = int(s)
        except (TypeError, ValueError):
            continue
        if n > 0:
            want.add(n)
    tags = dict(data.get("skill_tags") or {})
    tags.pop("combo", None)
    known = set(want)
    for k in list(tags):
        try:
            known.add(int(k))
        except (TypeError, ValueError):
            continue
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            try:
                known.add(int(sk.get("slot") or 0))
            except (TypeError, ValueError):
                continue
    new_tags: dict[str, list[str]] = {}
    for slot in sorted(known):
        if slot <= 0:
            continue
        prev = tags.get(str(slot)) or []
        if isinstance(prev, str):
            prev = [prev]
        names = [str(x) for x in prev if x and x != TAG_COMBO]
        if slot in want:
            names.append(TAG_COMBO)
        if names:
            new_tags[str(slot)] = names
    data["skill_tags"] = new_tags
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            try:
                slot = int(sk.get("slot") or 0)
            except (TypeError, ValueError):
                continue
            t = [x for x in (sk.get("tags") or []) if x != TAG_COMBO]
            if slot in want:
                t.append(TAG_COMBO)
                sk["combo"] = True
            else:
                sk.pop("combo", None)
            if t:
                sk["tags"] = t
            else:
                sk.pop("tags", None)
    for room in _rooms_or_dists(data).values():
        if isinstance(room.get("skills"), dict):
            room["skills"] = _sort_skills(room["skills"], str(room.get("kind") or ""))
    data["sequences"] = sequences_of(data)
    return data


def has_map_reset(data: dict | None) -> bool:
    tags = (data or {}).get("map_tags") or []
    return MAP_RESET in tags or MAP_RESET_LEGACY in tags


def set_map_reset(data: dict, on: bool) -> dict:
    tags = [t for t in (data.get("map_tags") or []) if t not in (MAP_RESET, MAP_RESET_LEGACY)]
    if on:
        tags.append(MAP_RESET)
    data["map_tags"] = tags
    return data


def cast_is_fake(c: dict, f: int | float, *, boss_room: bool | None = None) -> bool:
    """非 BOSS 房间且杀 MON 效率 < F%。假释放不进序列 / 范围 / CD。"""
    try:
        f_n = int(f)
    except (TypeError, ValueError):
        f_n = DEFAULT_F
    f_n = max(0, min(100, f_n))
    if boss_room is None:
        try:
            boss_n = int(c.get("boss") or 0)
        except (TypeError, ValueError):
            boss_n = 0
        boss_room = bool(c.get("boss_room") or str(c.get("dist_kind") or "") == "boss" or boss_n > 0)
    if boss_room:
        return False
    try:
        mon0 = int(c.get("mon") or 0)
    except (TypeError, ValueError):
        mon0 = 0
    kr = c.get("kill_ratio")
    if mon0 <= 0 or kr is None:
        return False
    try:
        return float(kr) < (f_n / 100.0)
    except (TypeError, ValueError):
        return False


def apply_f(data: dict, f: int | float) -> dict:
    """杀MON效率 < F% 标假释放；统计/序列/范围只用真释放。"""
    try:
        f_n = int(f)
    except (TypeError, ValueError):
        f_n = DEFAULT_F
    f_n = max(0, min(100, f_n))
    data["f"] = f_n
    for room in _rooms_or_dists(data).values():
        room_boss = str(room.get("kind") or "") == "boss"
        skills = dict(room.get("skills") or {})
        for sk_key, sk in skills.items():
            lst = list(sk.get("casts") or [])
            for c in lst:
                try:
                    boss_n = int(c.get("boss") or 0)
                except (TypeError, ValueError):
                    boss_n = 0
                boss = bool(
                    c.get("boss_room") or room_boss or boss_n > 0 or str(c.get("dist_kind") or "") == "boss"
                )
                c["fake"] = cast_is_fake(c, f_n, boss_room=boss)
            sk = dict(sk)
            sk["casts"] = lst
            sk.update(_skill_summary(lst))
            skills[sk_key] = sk
        room["skills"] = _sort_skills(skills, str(room.get("kind") or ""))
    data["sequences"] = sequences_of(data)
    return data


def fight_plan_from_features(
    data: dict | None, skills: list[dict] | None
) -> tuple[DistSkillPlan, ...]:
    """特征表 → 按怪物分布的开打序列。"""
    if not data:
        return ()
    binds = _bind_by_slot(skills)
    have_bind_combo = any(isinstance(s, dict) and "combo" in s for s in (skills or []))
    if have_bind_combo:
        combo = set()
        for s in skills or []:
            if not isinstance(s, dict) or not s.get("combo"):
                continue
            try:
                slot = int(s.get("slot") or 0)
            except (TypeError, ValueError):
                continue
            if slot > 0:
                combo.add(slot)
    else:
        combo = combo_slots_of(data)
    have_bind_multi = any(isinstance(s, dict) and "multi" in s for s in (skills or []))
    if have_bind_multi:
        multi_map: dict[int, int] = {}
        for s in skills or []:
            if not isinstance(s, dict):
                continue
            try:
                slot = int(s.get("slot") or 0)
            except (TypeError, ValueError):
                continue
            n = parse_multi_n(s)
            if slot > 0 and n > 1:
                multi_map[slot] = n
    else:
        multi_map = multi_n_of(data)
    mash_slots: set[int] = set()
    for s in skills or []:
        if not isinstance(s, dict) or not s.get("mash"):
            continue
        try:
            slot = int(s.get("slot") or 0)
        except (TypeError, ValueError):
            continue
        if slot > 0:
            mash_slots.add(slot)
    hotbar_key: dict[int, str] = {}
    for s in skills or []:
        hk = bind_hotkey(s)
        if not hk:
            continue
        try:
            slot = int(s.get("slot") or 0)
        except (TypeError, ValueError):
            continue
        if slot > 0:
            hotbar_key[slot] = hk
    rooms = _rooms_or_dists(data)
    slot_range: dict[int, float] = {}
    for rk in rooms:
        for sk_key in ((rooms.get(rk) or {}).get("skills") or {}):
            try:
                slot = int(sk_key)
            except (TypeError, ValueError):
                continue
            rng = skill_range_of(data, str(rk), slot)
            if rng is not None:
                slot_range.setdefault(slot, rng)

    def _skill(slot: int, key: str, cd: float, rng_f, hf: int) -> FightSkill:
        is_combo = slot in combo
        return FightSkill(
            slot=slot,
            key=key,
            cooldown_s=COMBO_CD_S if is_combo else cd,
            range_px=rng_f,
            combo=is_combo,
            hold_ms=max(0, int(hf)),
            multi=multi_map.get(slot, 1),
            mash=slot in mash_slots,
        )

    def _seq(room: dict) -> tuple[FightSkill, ...]:
        ordered = _sort_skills(room.get("skills") or {}, str(room.get("kind") or ""))
        items: list[FightSkill] = []
        for sk in ordered.values():
            try:
                slot = int(sk.get("slot") or 0)
            except (TypeError, ValueError):
                continue
            if slot <= 0 or _cast_n(sk) <= 0:
                continue
            if hotbar_key:
                key = hotbar_key.get(slot)
                if not key:
                    continue
            else:
                key = str(sk.get("key") or "").strip().lower()
                if not key or "+" in key or "＋" in key or len(key.split()) != 1:
                    continue
            _bk, cd, hf = binds.get(slot, ("", 0.0, 0))
            rng = sk.get("range_med")
            try:
                rng_f = float(rng) if rng is not None else slot_range.get(slot)
            except (TypeError, ValueError):
                rng_f = slot_range.get(slot)
            items.append(_skill(slot, key, cd, rng_f, hf))
        return expand_fight_skills(tuple(items))

    plans: list[DistSkillPlan] = []
    for rk, room in rooms.items():
        seq = _seq(room or {})
        tags = room.get("tags") or []
        reset = bool(room.get("reset_point") or TAG_RESET_POINT in tags)
        if seq or reset:
            plans.append(DistSkillPlan(key=str(rk), skills=seq, reset_point=reset))
    plans.sort(key=lambda p: p.key)
    return tuple(plans)


def hotbar_from_binds(skills: list[dict] | None) -> tuple[FightSkill, ...]:
    rows = [s for s in (skills or []) if isinstance(s, dict) and bind_hotkey(s)]
    out: list[FightSkill] = []
    for slot, (key, cd, hf) in sorted(_bind_by_slot(rows).items()):
        hk = None
        combo = False
        multi_n = 1
        mash = False
        for s in rows:
            try:
                if int(s.get("slot") or 0) == slot:
                    hk = bind_hotkey(s)
                    combo = bool(s.get("combo"))
                    multi_n = parse_multi_n(s)
                    mash = bool(s.get("mash"))
                    break
            except (TypeError, ValueError):
                continue
        if not hk:
            continue
        out.append(
            FightSkill(
                slot=slot,
                key=hk,
                cooldown_s=COMBO_CD_S if combo else cd,
                combo=combo,
                hold_ms=hf,
                multi=multi_n,
                mash=mash,
            )
        )
    return expand_fight_skills(tuple(out))


def dist_table_from_features(data: dict | None) -> tuple[DistSig, ...]:
    out: list[DistSig] = []
    for key, room in _rooms_or_dists(data).items():
        kind = str(room.get("kind") or "")
        if kind not in ("boss", "mob"):
            continue
        rel = room.get("rel") or room.get("center") or [0, 0]
        try:
            rx, ry = float(rel[0]), float(rel[1])
        except (TypeError, ValueError, IndexError):
            continue
        tags = room.get("tags") or []
        n = int(room.get("n") or 0)
        out.append(
            DistSig(
                key=str(key),
                kind=kind,
                n=n,
                rx=rx,
                ry=ry,
                reset_point=bool(room.get("reset_point") or TAG_RESET_POINT in tags),
                multi_boss=bool(room.get("multi_boss") or "多BOSS" in tags),
            )
        )
    return tuple(out)


def detect_cd_resets(
    session: str,
    frames: list[dict],
    draft: list[dict],
    skill_hold: list,
    skills: list[dict] | None,
    combo_slots=None,
    multi_slots=None,
    skip_ids: set[str] | None = None,
) -> list[dict]:
    """同段间隔短于 CD（连续释放 / 多次释放 / 假释放不算）→ 给该图加地图特征【CD重置】。"""
    if not frames or not draft:
        return []
    skip = {int(s) for s in (combo_slots or []) if str(s).lstrip("-").isdigit()}
    skip |= {int(s) for s in (multi_slots or []) if str(s).lstrip("-").isdigit()}
    for s in skills or []:
        if not isinstance(s, dict):
            continue
        try:
            slot = int(s.get("slot") or 0)
        except (TypeError, ValueError):
            continue
        if slot <= 0:
            continue
        if s.get("combo") or parse_multi_n(s) > 1:
            skip.add(slot)
    cd_map = _cooldown_by_slot(skills)
    if not cd_map:
        return []
    last_t: dict[int, int] = {}
    t0 = int(frames[0]["t_ns"])
    out: list[dict] = []
    skip_ids = {str(x) for x in (skip_ids or []) if x}
    for i0, _i1, slot, key in skill_spans(skill_hold):
        cid = f"{session}|{i0}|{slot}"
        if i0 >= len(frames) or slot not in cd_map or slot in skip or cid in skip_ids:
            continue
        t = int(frames[i0]["t_ns"])
        if slot in last_t:
            dt = (t - last_t[slot]) / 1e9
            if dt + 1e-9 < cd_map[slot]:
                dr = draft[i0] if i0 < len(draft) else {}
                out.append(
                    {
                        "id": f"{session}|{i0}|{slot}",
                        "session": session,
                        "i": i0,
                        "t_ns": t,
                        "t_s": round((t - t0) / 1e9, 3),
                        "dist_key": str(dr.get("dist_key") or ""),
                        "slot": slot,
                        "key": key,
                        "dt_s": round(dt, 3),
                        "cooldown_s": cd_map[slot],
                    }
                )
        last_t[slot] = t
    return out


def _skill_summary(casts: list[dict]) -> dict:
    real = [c for c in casts if not c.get("fake")]
    dists = [c["dist"] for c in real if c.get("dist") is not None]
    kinds = [c.get("kind") for c in real if c.get("kind")]
    ratios = [c["kill_ratio"] for c in real if c.get("kill_ratio") is not None]
    n_group = sum(1 for k in kinds if k == "群")
    kind_maj = "群" if n_group * 2 >= len(kinds) and kinds else ("单" if kinds else None)
    return {
        "n": len(real),
        "n_fake": sum(1 for c in casts if c.get("fake")),
        "range_med": None if not dists else round(float(median(dists)), 2),
        "kind_maj": kind_maj,
        "kind_group": n_group,
        "kill_mean": None
        if not real
        else round(sum(float(c.get("killed_mon") or 0) for c in real) / len(real), 2),
        "kill_ratio_mean": None if not ratios else round(sum(ratios) / len(ratios), 3),
    }


def empty_features(character: str, dungeon: str, e: int) -> dict:
    return {
        "dungeon": dungeon,
        "character": character,
        "updated_at": _now_iso(),
        "e": max(0, int(e)),
        "dists": {},
        "rooms": {},
        "cd_resets": [],
        "skill_tags": {},
        "map_tags": [],
        "sequences": {},
        "f": DEFAULT_F,
    }


def merge_casts(
    existing: dict | None,
    casts: list[dict],
    character: str,
    dungeon: str,
    e: int,
    resets: list[dict] | None = None,
    f: int = DEFAULT_F,
) -> tuple[dict, int, int]:
    """按怪物分布+技能叠加。同 id 已在表里则跳过。"""
    out = dict(existing) if isinstance(existing, dict) else empty_features(character, dungeon, e)
    out["character"] = character
    out["dungeon"] = dungeon
    out["e"] = max(0, int(e))
    rooms = dict(out.get("dists") or out.get("rooms") or {})
    seen: set[str] = set()
    for room in rooms.values():
        for sk in (room.get("skills") or {}).values():
            for c in sk.get("casts") or []:
                if c.get("id"):
                    seen.add(str(c["id"]))
    added = 0
    skipped = 0
    for c in casts:
        cid = str(c.get("id") or "")
        if not cid or cid in seen:
            skipped += 1
            continue
        seen.add(cid)
        added += 1
        rk = str(c.get("dist_key") or c.get("room") or "X")
        kind = str(c.get("dist_kind") or "")
        room = dict(rooms.get(rk) or {"skills": {}, "kind": kind, "tags": []})
        if kind:
            room["kind"] = kind
        tags = list(room.get("tags") or [])
        if kind == "boss" and "BOSS" not in tags:
            tags.append("BOSS")
        if kind == "mob" and "小怪" not in tags:
            tags.append("小怪")
        if kind == "abnormal" and "异常" not in tags:
            tags.append("异常")
        room["tags"] = tags
        if c.get("center") and room.get("rel") is None:
            room["rel"] = c.get("center")
        if kind == "mob":
            room["n"] = int(c.get("mon") or room.get("n") or 0)
        skills = dict(room.get("skills") or {})
        sk_key = str(int(c.get("slot") or 0))
        sk = dict(skills.get(sk_key) or {"slot": int(c.get("slot") or 0), "key": c.get("key") or "", "casts": []})
        sk["slot"] = int(c.get("slot") or sk.get("slot") or 0)
        if c.get("key"):
            sk["key"] = c["key"]
        lst = list(sk.get("casts") or [])
        lst.append(c)
        sk["casts"] = lst
        skills[sk_key] = sk
        room["skills"] = skills
        rooms[rk] = room
    for rk, room in rooms.items():
        skills = dict(room.get("skills") or {})
        for sk_key, sk in skills.items():
            summ = _skill_summary(sk.get("casts") or [])
            sk = dict(sk)
            sk.update(summ)
            skills[sk_key] = sk
        room = dict(room)
        room["skills"] = _sort_skills(skills, str(room.get("kind") or ""))
        rooms[rk] = room
    out["dists"] = dict(sorted(rooms.items(), key=lambda kv: kv[0]))
    out["rooms"] = out["dists"]
    out["cd_resets"] = _merge_reset_list(out.get("cd_resets"), resets or [])
    out.setdefault("skill_tags", {})
    set_combo_slots(out, combo_slots_of(out))
    apply_f(out, f)
    _drop_fake_cd_resets(out)
    _stamp_after_cd_reset(out)
    if out.get("cd_resets"):
        set_map_reset(out, True)
    out["updated_at"] = _now_iso()
    return out, added, skipped


def _drop_fake_cd_resets(data: dict) -> None:
    fake_ids = set()
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            for c in sk.get("casts") or []:
                if c.get("fake") and c.get("id"):
                    fake_ids.add(str(c["id"]))
    if not fake_ids:
        return
    data["cd_resets"] = [
        r for r in (data.get("cd_resets") or []) if str(r.get("id") or "") not in fake_ids
    ]


def _merge_reset_list(existing, incoming: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in list(existing or []) + list(incoming or []):
        rid = str(r.get("id") or "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    out.sort(key=lambda x: (str(x.get("session") or ""), int(x.get("i") or 0)))
    return out


def _stamp_after_cd_reset(data: dict) -> None:
    by_sess: dict[str, list[int]] = {}
    for r in data.get("cd_resets") or []:
        by_sess.setdefault(str(r.get("session") or ""), []).append(int(r.get("i") or 0))
    for room in _rooms_or_dists(data).values():
        for sk in (room.get("skills") or {}).values():
            for c in sk.get("casts") or []:
                hits = by_sess.get(str(c.get("session") or ""), [])
                c["after_cd_reset"] = any(ri <= int(c.get("i0") or 0) for ri in hits)


def clear_features(character: str, dungeon: str) -> Path:
    """清空该图该角色特征表。若已有表，先另存 .prev 供一次性回退。"""
    path = feature_path(character, dungeon)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        shutil.copy2(path, prev_path(character, dungeon))
        path.unlink()
    return path


def save_features(character: str, dungeon: str, data: dict, *, keep_prev: bool) -> Path:
    path = feature_path(character, dungeon)
    path.parent.mkdir(parents=True, exist_ok=True)
    if keep_prev and path.is_file():
        shutil.copy2(path, prev_path(character, dungeon))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def rollback_features(character: str, dungeon: str) -> tuple[bool, str]:
    """一次性：有 .prev 则覆盖当前表并删掉 prev。"""
    cur = feature_path(character, dungeon)
    prev = prev_path(character, dungeon)
    if not prev.is_file():
        return False, "没有可回退的未叠加副本。"
    shutil.copy2(prev, cur)
    try:
        prev.unlink()
    except OSError:
        pass
    return True, f"已回退到叠加前：{cur.parent.name}/{cur.name}"


def format_report(data: dict | None, added: int = 0, skipped: int = 0) -> str:
    if not data:
        return "没有特征表。"
    lines = [
        f"{data.get('dungeon') or '—'}  /  {data.get('character') or '—'}  "
        f"更新 {data.get('updated_at') or '—'}  E={data.get('e')}  F={data.get('f', DEFAULT_F)}%",
        f"本次新增 {added} 条，跳过已在表里的 {skipped} 条，表内共 {count_casts(data)} 条。",
        "",
    ]
    resets = data.get("cd_resets") or []
    if resets:
        lines.append(f"CD重置证据 {len(resets)} 处（短于CD的间隔 → 该图【CD重置】；击败BOSS+1时清CD）")
        for r in resets:
            lines.append(
                f"  {r.get('session')}  帧{int(r.get('i') or 0)+1}  t={r.get('t_s')}s  "
                f"分布{r.get('dist_key') or '—'}  技能{r.get('slot')} {r.get('key') or ''} "
                f"间隔{r.get('dt_s')}s < CD{r.get('cooldown_s')}s"
            )
        lines.append("")
    combo = combo_slots_of(data)
    if combo:
        lines.append("连续释放（CD按270秒）: " + "、".join(f"技能{s}" for s in sorted(combo)))
        lines.append("")
    multi = multi_n_of(data)
    if multi:
        bits = [f"技能{s}×{n}" for s, n in sorted(multi.items())]
        lines.append("多次释放（各份独立CD，不触发CD重置检测）: " + "、".join(bits))
        lines.append("")
    if has_map_reset(data):
        lines.append("地图特征:【CD重置】（击败BOSS数+1时清CD；回放可删）")
        lines.append("")
    rooms = _rooms_or_dists(data)
    if not rooms:
        lines.append("还没有释放记录。")
        return "\n".join(lines)
    for rk in rooms:
        room = rooms[rk]
        seq = (data.get("sequences") or sequences_of(data)).get(str(rk)) or []
        seq_s = " > ".join(f"技能{s}" for s in seq) if seq else "—"
        tags = " ".join(f"【{t}】" for t in (room.get("tags") or []))
        lines.append(f"怪物分布 {rk} {tags}  序列 {seq_s}")
        skills = _sort_skills(room.get("skills") or {}, str(room.get("kind") or ""))
        for sk_key, sk in skills.items():
            rng = sk.get("range_med")
            rng_s = "—" if rng is None else str(rng)
            ratio = sk.get("kill_ratio_mean")
            ratio_s = "—" if ratio is None else f"{ratio:.0%}" if ratio <= 1 else str(ratio)
            km = sk.get("kill_mean")
            if km is None:
                cst = _real_casts(sk)
                km = (
                    None
                    if not cst
                    else sum(float(c.get("killed_mon") or 0) for c in cst) / len(cst)
                )
            km_s = "—" if km is None else f"{km:.2f}"
            combo_s = f"  {TAG_COMBO}" if (sk.get("combo") or TAG_COMBO in (sk.get("tags") or [])) else ""
            multi_s = ""
            if sk.get("multi") or TAG_MULTI in (sk.get("tags") or []):
                try:
                    mn = int(sk.get("multi_n") or DEFAULT_MULTI)
                except (TypeError, ValueError):
                    mn = DEFAULT_MULTI
                multi_s = f"  {TAG_MULTI}×{max(DEFAULT_MULTI, mn)}"
            fake_n = int(sk.get("n_fake") or 0)
            fake_s = f"  假释放{fake_n}" if fake_n else ""
            lines.append(
                f"  技能{sk.get('slot')} {sk.get('key') or '—'}  "
                f"n={sk.get('n', 0)}  范围中位={rng_s}  "
                f"类型={sk.get('kind_maj') or '—'}（群{sk.get('kind_group', 0)}）  "
                f"杀MON效率={ratio_s}  杀MON均={km_s}{combo_s}{multi_s}{fake_s}"
            )
            for i, c in enumerate(sk.get("casts") or [], 1):
                dist = c.get("dist")
                dist_s = "—" if dist is None else str(dist)
                kr = c.get("kill_ratio")
                kr_s = "—" if kr is None else f"{kr:.0%}"
                fake_s = "  假释放" if c.get("fake") else ""
                sess = str(c.get("session") or "—")
                try:
                    fr = int(c.get("i0") or 0) + 1
                except (TypeError, ValueError):
                    fr = 1
                lines.append(
                    f"    #{i}  {sess}  帧{fr}  "
                    f"mon{c.get('mon')}→{c.get('mon_end')}  "
                    f"boss{c.get('boss')}→{c.get('boss_end')}  "
                    f"距离{dist_s}  "
                    f"{c.get('kind')}  杀MON{c.get('killed_mon')} {kr_s}{fake_s}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def skill_range_of(data: dict | None, room, slot: int) -> float | None:
    """该分布该槽的范围中位；没有则用各分布同槽中位。"""
    if not data:
        return None
    rooms = _rooms_or_dists(data)
    sk = ((rooms.get(str(room)) or {}).get("skills") or {}).get(str(int(slot))) or {}
    rng = sk.get("range_med")
    if rng is not None:
        try:
            return float(rng)
        except (TypeError, ValueError):
            pass
    dists = []
    for room_d in rooms.values():
        other = ((room_d or {}).get("skills") or {}).get(str(int(slot))) or {}
        v = other.get("range_med")
        if v is None:
            continue
        try:
            dists.append(float(v))
        except (TypeError, ValueError):
            continue
    if not dists:
        return None
    return round(float(median(dists)), 2)


def pack_dist_from_player(view: dict | None) -> float | None:
    """当前最远敌对单位相对 player 的距离。"""
    return farthest_enemy(view).get("dist")
