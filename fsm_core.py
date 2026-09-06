"""FSM 核心：纯函数。step(snapshot, ctx, params) → (decision, new_ctx)

禁止：time / sleep / 读文件 / 发按键 / 截屏 / random / 模块级可变状态。
t_ns 只信快照：回放必须用 jsonl 原值；实机必须用采样时刻（截图完成、推理之前）。
防抖是因果的：只看当前帧和 ctx，不向后看。「与旧批量等价」是断言不是实测，见 DECISIONS §8。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class FsmState(Enum):
    WAIT = "等待"
    FIGHT = "开打"
    LOOT = "捡物"
    ADVANCE = "前进"
    RETURN = "返回"
    IDLE = "空闲"
    STUCK = "卡住"


class FsmFlag(Enum):
    STUCK = "卡住"
    WARMUP = "预热"


class FsmDir(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class FsmAction(Enum):
    NONE = "无"
    TAP = "点按"
    HOLD = "按住"
    CAST = "释放"
    PICK = "一键拾取"
    ATTACK = "普攻"


_RECOVER_DIRS = (FsmDir.UP, FsmDir.DOWN, FsmDir.LEFT, FsmDir.RIGHT)
_MOVE_DIR_ORDER = (FsmDir.UP, FsmDir.DOWN, FsmDir.LEFT, FsmDir.RIGHT)
DEFAULT_TH_MS = 50
DEFAULT_PW_MS = 3000


def _norm_dirs(dirs) -> tuple[FsmDir, ...]:
    got = {d for d in (dirs or ()) if d is not None}
    return tuple(d for d in _MOVE_DIR_ORDER if d in got)


def _tap_hold(
    want: tuple[FsmDir, ...],
    prev_want: tuple[FsmDir, ...],
    phase: int,
    left: int,
    th_ms: int,
    dt: int,
) -> tuple[FsmAction, tuple[FsmDir, ...], tuple[FsmDir, ...], int, int]:
    """接近/捡物：点按 → 等 th_ms → 按住。方向变了重来；少掉的轴若仍是子集则继续按住。"""
    want = _norm_dirs(want)
    prev_want = _norm_dirs(prev_want)
    th_ms = max(0, int(th_ms))
    dt = max(0, int(dt))
    phase = int(phase)
    left = max(0, int(left))
    if not want:
        return FsmAction.NONE, (), (), 0, 0
    if phase == 2 and want != prev_want and set(want) <= set(prev_want):
        return FsmAction.HOLD, want, want, 2, 0
    if want == prev_want and phase == 2:
        return FsmAction.HOLD, want, want, 2, 0
    if want == prev_want and phase == 1:
        left = max(0, left - dt)
        if left <= 0:
            return FsmAction.HOLD, want, want, 2, 0
        return FsmAction.NONE, (), want, 1, left
    return FsmAction.TAP, want, want, 1, th_ms


DEFAULT_TOWN_S = 30.0
HOLD_MS_KEY = "hold_ms"
HOLD_FRAMES_KEY = "hold_frames"  # 仅清理/警告，不再换算
_REF_W = 1600.0
_REF_H = 900.0

_LEGACY_MS_KEYS = {
    "ax_ms": ("ax", "a"),
    "ay_ms": ("ay", "a"),
    "y_ms": ("y",),
    "xxx_ms": ("xxx",),
    "loot_hold_ms": ("loot_hold",),
}
_LEGACY_S_KEYS = {
    "x_s": ("x",),
}
_warned_legacy: set[str] = set()


def warn_legacy_key(*names: str) -> None:
    """旧帧字段只警告、不换算。"""
    for n in names:
        s = str(n).strip()
        if not s or s in _warned_legacy:
            continue
        _warned_legacy.add(s)
        print(f"忽略旧帧字段 {s!r}（不再换算毫秒/秒）", flush=True)


def dt_ms(prev_t_ns: int | None, t_ns: int) -> int:
    if prev_t_ns is None:
        return 0
    return max(0, int(round((int(t_ns) - int(prev_t_ns)) / 1e6)))


def json_ms(data: dict | None, key_ms: str, default_ms: int, lo: int = 0) -> int:
    """只读 *_ms；没有则用毫秒默认。旧帧键忽略。"""
    src = data or {}
    for old in _LEGACY_MS_KEYS.get(key_ms, ()):
        if old in src:
            warn_legacy_key(old)
    if key_ms in src and src[key_ms] is not None:
        try:
            return max(lo, int(src[key_ms]))
        except (TypeError, ValueError):
            pass
    return max(lo, int(default_ms))


def json_s(data: dict | None, key_s: str, default_s: float, lo: float = 0.05) -> float:
    """只读秒字段；没有则用秒默认。旧帧键忽略。"""
    src = data or {}
    for old in _LEGACY_S_KEYS.get(key_s, ()):
        if old in src:
            warn_legacy_key(old)
    if key_s in src and src[key_s] is not None:
        try:
            return max(lo, float(src[key_s]))
        except (TypeError, ValueError):
            pass
    return max(lo, float(default_s))


def parse_hold_ms(item, default: int | None = None) -> int | None:
    """只读 hold_ms。有 hold_frames/frames 则警告并忽略。"""
    if not isinstance(item, dict):
        return default
    if item.get(HOLD_FRAMES_KEY) is not None:
        warn_legacy_key(HOLD_FRAMES_KEY)
    if item.get("frames") is not None and HOLD_MS_KEY not in item:
        warn_legacy_key("frames")
    if item.get(HOLD_MS_KEY) is not None:
        try:
            return max(0, int(item[HOLD_MS_KEY]))
        except (TypeError, ValueError):
            return default
    return default


@dataclass(frozen=True)
class FightSkill:
    """过图技能序列里的一项。range_px 来自特征表该房（或各房同槽）范围中位。"""

    slot: int
    key: str
    cooldown_s: float
    range_px: float | None = None
    combo: bool = False
    hold_ms: int = 0
    multi: int = 1
    charge: int = 0
    mash: bool = False  # 键位表【连按】：宿主/执行层用，核心不读


@dataclass(frozen=True)
class DistSkillPlan:
    """该怪物分布在过图技能文件里的序列。"""

    key: str
    skills: tuple[FightSkill, ...] = ()
    reset_point: bool = False  # 旧特征字段；运行时清 CD 用 map_reset + 击败 BOSS


@dataclass(frozen=True)
class DistSig:
    """BOSS位置表 / 小怪分布表一项。rel 是相对玩家。"""

    key: str
    kind: str
    n: int
    rx: float
    ry: float
    reset_point: bool = False
    multi_boss: bool = False


@dataclass(frozen=True)
class FsmParams:
    m: int = 5
    l: int = 5
    g: int = 5
    x_s: float = 1.5  # 卡住：同一流程状态连续秒数
    gx: int = 50
    gy: int = 10
    ax_ms: int = 250
    ay_ms: int = 250
    y_ms: int = 250  # 卡住恢复：每向按住毫秒
    s: int = 20
    pm: int = 10
    pc: int = 5
    pt: int = 3
    xxx_ms: int = 1000
    th_ms: int = DEFAULT_TH_MS  # 捡物/前进接近：点按后等到按住的间隔
    pw_ms: int = DEFAULT_PW_MS  # 捡物等停下上限；超时后按当前掉落继续步骤 2
    tn_s: float = DEFAULT_TOWN_S  # 回城：连续无地下城关键词的秒数
    fight_plan: tuple[DistSkillPlan, ...] = ()
    hotbar: tuple[FightSkill, ...] = ()
    dist_table: tuple[DistSig, ...] = ()
    map_reset: bool = False
    mon_off_x: int = 0  # >0 右：YOLO MON/BOSS x 加；<0 左：减
    mon_off_y: int = 0  # >0 下：YOLO MON/BOSS y 加；<0 上：减


@dataclass(frozen=True)
class FsmSnapshot:
    t_ns: int
    mon: int = 0
    loot: int = 0
    gate: int = 0
    boss: int = 0
    player_xy: tuple[float, float] | None = None
    gate_xy: tuple[float, float] | None = None
    mon_xy: tuple[tuple[float, float], ...] = ()
    boss_xy: tuple[tuple[float, float], ...] = ()
    loot_xy: tuple[tuple[float, float], ...] = ()
    in_dungeon: bool = True
    town_return: bool = False
    dungeon_kw: bool | None = None  # None=回放默认已进图；bool=本帧 OCR 是否看到地下城关键词


@dataclass(frozen=True)
class _Deb:
    judged: bool = False
    run_v: bool | None = None
    run_n: int = 0
    run_t: int | None = None


@dataclass(frozen=True)
class FsmContext:
    mon: _Deb = field(default_factory=_Deb)
    boss: _Deb = field(default_factory=_Deb)
    loot: _Deb = field(default_factory=_Deb)
    gate: _Deb = field(default_factory=_Deb)
    state: FsmState | None = None
    fsm_run: int = 0
    run_start_t: int | None = None
    rooms: int = 0
    boss_seen: bool = False
    dist_key: str = ""
    dist_kind: str = ""
    extra_sigs: tuple[DistSig, ...] = ()
    last_player_xy: tuple[float, float] | None = None
    last_t_ns: int | None = None
    n_seen: int = 0
    adv_dir: FsmDir | None = None
    adv_dirs: tuple[FsmDir, ...] = ()
    adv_phase: int = 0
    extra_h: int = 0
    extra_v: int = 0
    adv_lock: tuple[int, int, int, int] = (0, 0, 0, 0)
    adv_dir_set: bool = False
    recover_on: bool = False
    recover_dir_i: int = 0
    recover_ms: int = 0
    through_done: bool = False
    gate_crosses: int = 0
    boss_enters: int = 0
    boss_kills: int = 0
    skill_cd: tuple[tuple[int, int, int], ...] = ()
    fight_dir: FsmDir | None = None
    cast_wait_left: int = 0
    attack_left: int = 0
    cast_slot: int | None = None
    cast_key: str | None = None
    loot_xy_prev: tuple[tuple[float, float], ...] = ()
    loot_targets: tuple[tuple[float, float], ...] = ()
    loot_dir: FsmDir | None = None
    loot_stop: _Deb = field(default_factory=_Deb)
    loot_wait_t: int | None = None  # 进入「等停下」的 t_ns
    loot_wait_done: bool = False  # 本轮捡物已 PW 超时，不再无限等
    dash_want: tuple[FsmDir, ...] = ()
    dash_phase: int = 0
    dash_left: int = 0
    dungeon: _Deb = field(default_factory=_Deb)
    saw_dungeon: bool = False
    watch_state: FsmState | None = None
    watch_run: int = 0
    watch_start_t: int | None = None


@dataclass(frozen=True)
class FsmDecision:
    state: FsmState
    flags: frozenset[FsmFlag]
    action: FsmAction
    move_dir: FsmDir | None
    why: str
    chain: str
    rooms: int
    boss_seen: bool
    room_kind: str | None
    fsm_run: int
    fsm_run_s: float
    player_hold: bool
    eff_mon: bool
    eff_loot: bool
    eff_gate: bool
    eff_boss: bool
    move_dirs: tuple[FsmDir, ...] = ()
    skill_slot: int | None = None
    skill_key: str | None = None
    skill_range: float | None = None
    pack_dist: float | None = None
    intent_label: str = ""
    cd_reset: bool = False
    skill_cds: tuple[tuple[int, str, float], ...] = ()
    flow_steps: tuple[str, ...] = ()
    flow_hit: int = -1
    send_label: str = ""
    dist_key: str = ""
    dist_kind: str = ""
    gate_crosses: int = 0
    boss_enters: int = 0
    boss_kills: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "state_label": self.state.value,
            "flags": self.flags,
            "action": self.action,
            "action_label": self.action.value,
            "move_dir": self.move_dir,
            "move_dir_label": None if self.move_dir is None else self.move_dir.value,
            "move_dirs": self.move_dirs,
            "move_dirs_label": "+".join(d.value for d in self.move_dirs) if self.move_dirs else "",
            "why": self.why,
            "chain": self.chain,
            "rooms": self.rooms,
            "boss_seen": self.boss_seen,
            "room_kind": self.room_kind,
            "dist_key": self.dist_key,
            "dist_kind": self.dist_kind,
            "gate_crosses": self.gate_crosses,
            "boss_enters": self.boss_enters,
            "boss_kills": self.boss_kills,
            "fsm_run": self.fsm_run,
            "fsm_run_s": self.fsm_run_s,
            "player_hold": self.player_hold,
            "eff_mon": self.eff_mon,
            "eff_loot": self.eff_loot,
            "eff_gate": self.eff_gate,
            "eff_boss": self.eff_boss,
            "skill_slot": self.skill_slot,
            "skill_key": self.skill_key,
            "skill_range": self.skill_range,
            "pack_dist": self.pack_dist,
            "intent_label": self.intent_label,
            "cd_reset": self.cd_reset,
            "skill_cds": self.skill_cds,
            "flow_steps": self.flow_steps,
            "flow_hit": self.flow_hit,
            "send_label": self.send_label,
        }


def _dirs_label(dirs: tuple[FsmDir, ...]) -> str:
    return "+".join(d.value for d in dirs)


def intent_text(decision: FsmDecision) -> str:
    if decision.intent_label:
        return decision.intent_label
    dirs = decision.move_dirs or (() if decision.move_dir is None else (decision.move_dir,))
    if decision.action is FsmAction.NONE or not dirs:
        return ""
    recover = "恢复 " if FsmFlag.STUCK in decision.flags else ""
    return f"{recover}{decision.action.value} {_dirs_label(dirs)}"


def send_keys_label(
    action: FsmAction,
    move_dirs: tuple[FsmDir, ...] = (),
    move_dir: FsmDir | None = None,
    skill_key: str | None = None,
    mash_n: int = 0,
) -> str:
    """这一帧会按的键（展示用）。mash_n 由宿主按执行层同一套槽+COUNT 传入，核心不填。"""
    dirs = move_dirs or (() if move_dir is None else (move_dir,))
    names = [d.value for d in dirs]
    if action is FsmAction.HOLD and names:
        return "按住 " + "+".join(names)
    if action is FsmAction.TAP and names:
        return "点按 " + "+".join(names)
    if action is FsmAction.CAST:
        cmd = str(skill_key or "").strip()
        if not cmd:
            return ""
        n = max(0, int(mash_n or 0))
        return f"连按{n}× {cmd}" if n > 1 else f"点按 {cmd}"
    if action is FsmAction.ATTACK:
        return "按住 x"
    if action is FsmAction.PICK:
        return "点按 左Alt"
    return ""


def _parse_xy(xy) -> tuple[float, float] | None:
    if xy is None or len(xy) < 2:
        return None
    return (float(xy[0]), float(xy[1]))


def _parse_xy_list(raw) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for p in raw or []:
        xy = _parse_xy(p)
        if xy is not None:
            out.append(xy)
    return tuple(out)


def _pack_center(pts: tuple[tuple[float, float], ...]) -> tuple[float, float] | None:
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _count_pct(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0) * 100.0


def rooms_similar(
    n1: int,
    c1: tuple[float, float] | None,
    n2: int,
    c2: tuple[float, float] | None,
    s: float,
) -> bool:
    """数量相对差、相对坐标两项都不超过 S%。"""
    s = max(0.0, float(s))
    if _count_pct(n1, n2) > s:
        return False
    return _rel_similar(c1, c2, s)


def _rel_similar(
    a: tuple[float, float] | None, b: tuple[float, float] | None, s: float
) -> bool:
    s = max(0.0, float(s))
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a[0] - b[0]) / _REF_W * 100.0 <= s and abs(a[1] - b[1]) / _REF_H * 100.0 <= s


def _rel_of(
    player: tuple[float, float] | None, pts: tuple[tuple[float, float], ...]
) -> tuple[float, float] | None:
    c = _pack_center(pts)
    if player is None or c is None:
        return None
    return (c[0] - player[0], c[1] - player[1])


def _farthest_dist(
    player: tuple[float, float] | None, pts: tuple[tuple[float, float], ...]
) -> float | None:
    if player is None or not pts:
        return None
    return max(math.hypot(p[0] - player[0], p[1] - player[1]) for p in pts)


def _next_dist_key(kind: str, table: tuple[DistSig, ...]) -> str:
    prefix = "B" if kind == "boss" else "M"
    used = {e.key for e in table if e.key.startswith(prefix)}
    i = 0
    while f"{prefix}{i}" in used:
        i += 1
    return f"{prefix}{i}"


def _match_dist(
    table: tuple[DistSig, ...],
    extra: tuple[DistSig, ...],
    kind: str,
    n: int,
    rel: tuple[float, float] | None,
    s: float,
    *,
    multi_boss: bool = False,
) -> tuple[str, tuple[DistSig, ...], str, bool]:
    combined = table + extra
    for e in combined:
        if e.kind != kind:
            continue
        if kind == "boss":
            ok = _rel_similar((e.rx, e.ry), rel, s)
        else:
            ok = rooms_similar(n, rel, e.n, (e.rx, e.ry), s)
        if ok:
            return e.key, extra, "similar", e.reset_point
    if rel is None:
        return "X", extra, "new", False
    key = _next_dist_key(kind, combined)
    sig = DistSig(
        key=key,
        kind=kind,
        n=n,
        rx=rel[0],
        ry=rel[1],
        multi_boss=multi_boss,
    )
    return key, extra + (sig,), "new", False


def classify_dist(
    pos: tuple[float, float] | None,
    mon_xy: tuple[tuple[float, float], ...],
    boss_xy: tuple[tuple[float, float], ...],
    mon_n: int,
    boss_n: int,
    table: tuple[DistSig, ...],
    extra: tuple[DistSig, ...],
    s: float,
) -> tuple[str, str, tuple[DistSig, ...], str]:
    """按「关于怪物分布判定」现算：有 BOSS → BOSS 表，否则有 MON → 小怪表，否则异常。"""
    if int(boss_n) > 0 or boss_xy:
        rel = _rel_of(pos, tuple(boss_xy))
        multi = len(boss_xy) > 1
        key, extra, room_kind, _rp = _match_dist(
            table,
            extra,
            "boss",
            len(boss_xy) or int(boss_n),
            rel,
            s,
            multi_boss=multi,
        )
        return key, "boss", extra, room_kind
    if int(mon_n) > 0 or mon_xy:
        rel = _rel_of(pos, tuple(mon_xy))
        key, extra, room_kind, _rp = _match_dist(
            table,
            extra,
            "mob",
            int(mon_n) or len(mon_xy),
            rel,
            s,
        )
        return key, "mob", extra, room_kind
    return "X", "abnormal", extra, "new"


def _nearest_gate(
    player: tuple[float, float] | None, gates
) -> tuple[float, float] | None:
    pts: list[tuple[float, float]] = []
    for g in gates or []:
        p = _parse_xy(g)
        if p is not None:
            pts.append(p)
    if not pts:
        return None
    if player is None:
        return pts[0]
    return min(pts, key=lambda q: (q[0] - player[0]) ** 2 + (q[1] - player[1]) ** 2)


MON_CORR_MAX = 80


def mon_corr_from_dict(data: dict | None) -> tuple[int, int, int, int]:
    """json → (上, 下, 左, 右) 像素，均 ≥0。"""
    d = data if isinstance(data, dict) else {}

    def _n(key: str) -> int:
        try:
            return max(0, min(MON_CORR_MAX, int(d.get(key, 0) or 0)))
        except (TypeError, ValueError):
            return 0

    return _n("mon_corr_u"), _n("mon_corr_d"), _n("mon_corr_l"), _n("mon_corr_r")


def mon_off_from_corr(u: int, dwn: int, left: int, right: int) -> tuple[int, int]:
    """滑块 → 加到 YOLO MON/BOSS 上的 (dx, dy)。上/左为负。"""
    return int(right) - int(left), int(dwn) - int(u)


def mon_off_from_dict(data: dict | None) -> tuple[int, int]:
    return mon_off_from_corr(*mon_corr_from_dict(data))


def apply_mon_boss_corr(data: dict | None, ox: int = 0, oy: int = 0) -> dict:
    """读时平移 mon_xy / boss_xy。浅拷贝，不改入参、不写 jsonl。"""
    src = dict(data or {})
    dx, dy = int(ox), int(oy)
    if not dx and not dy:
        return src
    src["mon_xy"] = [list(p) for p in _shift_xy(_parse_xy_list(src.get("mon_xy")), dx, dy)]
    src["boss_xy"] = [list(p) for p in _shift_xy(_parse_xy_list(src.get("boss_xy")), dx, dy)]
    return src


def stuck_x_s_floor(params: FsmParams) -> float:
    """卡住 x_s 建议下限（秒）：max(XXX, AX, AY, PW, 四向 Y) / 1000。技能表最长持续由宿主另算。"""
    ms = max(
        int(params.xxx_ms),
        int(params.ax_ms),
        int(params.ay_ms),
        int(params.pw_ms),
        int(params.y_ms) * 4,
        1,
    )
    return ms / 1000.0


def _shift_xy(
    pts: tuple[tuple[float, float], ...], dx: float, dy: float
) -> tuple[tuple[float, float], ...]:
    if (not dx and not dy) or not pts:
        return pts
    return tuple((p[0] + dx, p[1] + dy) for p in pts)


def snapshot_from_detect(
    t_ns: int,
    features: dict,
    *,
    in_dungeon: bool = True,
    town_return: bool = False,
    dungeon_kw: bool | None = None,
) -> FsmSnapshot:
    player = _parse_xy(features.get("player_xy"))
    gate = _nearest_gate(player, features.get("gate_xy"))
    return FsmSnapshot(
        t_ns=int(t_ns),
        mon=int(features.get("mon") or 0),
        loot=int(features.get("loot") or 0),
        gate=int(features.get("gate") or 0),
        boss=int(features.get("boss") or 0),
        player_xy=player,
        gate_xy=gate,
        mon_xy=_parse_xy_list(features.get("mon_xy")),
        boss_xy=_parse_xy_list(features.get("boss_xy")),
        loot_xy=_parse_xy_list(features.get("loot_xy")),
        in_dungeon=bool(in_dungeon),
        town_return=bool(town_return),
        dungeon_kw=None if dungeon_kw is None else bool(dungeon_kw),
    )


def _axis_dir(player: tuple[float, float], gate: tuple[float, float]) -> FsmDir:
    """单轴接近（开打/捡物）。前进穿门用 `_gate_dirs`。"""
    dx = gate[0] - player[0]
    dy = gate[1] - player[1]
    if abs(dx) >= abs(dy):
        return FsmDir.RIGHT if dx >= 0 else FsmDir.LEFT
    return FsmDir.DOWN if dy >= 0 else FsmDir.UP


def _gate_lock(pos: tuple[float, float], gate: tuple[float, float]) -> tuple[int, int, int, int]:
    """一次性过门方向：X+/X-/Y+/Y-，各 0/1。Y+ 为屏幕向下。"""
    return (
        1 if pos[0] < gate[0] else 0,
        1 if pos[0] > gate[0] else 0,
        1 if pos[1] < gate[1] else 0,
        1 if pos[1] > gate[1] else 0,
    )


def _lock_hold(lock: tuple[int, int, int, int]) -> tuple[FsmDir | None, FsmDir | None]:
    xp, xm, yp, ym = lock
    hx = FsmDir.RIGHT if xp else (FsmDir.LEFT if xm else None)
    vy = FsmDir.DOWN if yp else (FsmDir.UP if ym else None)
    return hx, vy


def _intent(
    *,
    state: FsmState,
    stuck: bool,
    pos: tuple[float, float] | None,
    gate_xy: tuple[float, float] | None,
    prev_state: FsmState | None,
    ctx: FsmContext,
    gx: int,
    gy: int,
    ax_ms: int,
    ay_ms: int,
    y_ms: int,
    dt_ms: int,
) -> tuple[
    FsmAction,
    FsmDir | None,
    tuple[FsmDir, ...],
    tuple[FsmDir, ...],
    int,
    int,
    int,
    tuple[int, int, int, int],
    bool,
    int,
    int,
    bool,
    bool,
]:
    """action, move_dir, move_dirs, adv_dirs, extra_h, extra_v, adv_phase, adv_lock, recover_on, recover_dir_i, recover_ms, through_done, adv_dir_set。"""
    gx, gy = max(0, int(gx)), max(0, int(gy))
    ax_ms, ay_ms = max(0, int(ax_ms)), max(0, int(ay_ms))
    y_ms = max(1, int(y_ms))
    step_ms = max(0, int(dt_ms))

    if stuck:
        i = ctx.recover_dir_i if ctx.recover_on else 0
        k = ctx.recover_ms if ctx.recover_on else 0
        i = i % 4
        d = _RECOVER_DIRS[i]
        k += step_ms
        if k >= y_ms:
            k = 0
            i = (i + 1) % 4
        return (
            FsmAction.HOLD,
            d,
            (d,),
            (d,),
            ctx.extra_h,
            ctx.extra_v,
            ctx.adv_phase,
            ctx.adv_lock,
            True,
            i,
            k,
            ctx.through_done,
            ctx.adv_dir_set,
        )

    empty = (
        FsmAction.NONE,
        None,
        (),
        (),
        0,
        0,
        0,
        (0, 0, 0, 0),
        False,
        0,
        0,
        False,
        False,
    )
    if state is not FsmState.ADVANCE or pos is None:
        return empty

    just_entered = prev_state is not FsmState.ADVANCE
    phase = 0 if just_entered else ctx.adv_phase
    extra_h = 0 if just_entered else ctx.extra_h
    extra_v = 0 if just_entered else ctx.extra_v
    lock = (0, 0, 0, 0) if just_entered else ctx.adv_lock
    dir_set = False if just_entered else ctx.adv_dir_set
    if (not dir_set) and gate_xy is not None:
        lock = _gate_lock(pos, gate_xy)
        dir_set = True

    def _pack(
        action: FsmAction,
        dirs: tuple[FsmDir, ...],
        extra_h: int,
        extra_v: int,
        phase: int,
        done: bool,
    ):
        return (
            action,
            dirs[0] if dirs else None,
            dirs,
            dirs,
            extra_h,
            extra_v,
            phase,
            lock,
            False,
            0,
            0,
            done,
            dir_set,
        )

    def _enter_through():
        eh = ax_ms if (lock[0] or lock[1]) else 0
        ev = ay_ms if (lock[2] or lock[3]) else 0
        return _pack(FsmAction.NONE, (), eh, ev, 1, eh <= 0 and ev <= 0)

    if phase == 0:
        if gate_xy is None:
            if dir_set:
                return _enter_through()
            return _pack(FsmAction.NONE, (), 0, 0, 0, False)
        dx = gate_xy[0] - pos[0]
        dy = gate_xy[1] - pos[1]
        hold_h = None
        if abs(dx) > gx:
            hold_h = FsmDir.RIGHT if dx > 0 else FsmDir.LEFT
        hold_v = None
        if abs(dy) > gy:
            hold_v = FsmDir.DOWN if dy > 0 else FsmDir.UP
        dirs = tuple(d for d in (hold_h, hold_v) if d is not None)
        if dirs:
            return _pack(FsmAction.HOLD, dirs, 0, 0, 0, False)
        return _enter_through()

    if extra_h <= 0 and extra_v <= 0:
        return _pack(FsmAction.NONE, (), 0, 0, 1, True)

    hx, vy = _lock_hold(lock)
    hold_h = hx if extra_h > 0 else None
    hold_v = vy if extra_v > 0 else None
    dirs = tuple(d for d in (hold_h, hold_v) if d is not None)
    extra_h = max(0, extra_h - (step_ms if hold_h is not None else 0))
    extra_v = max(0, extra_v - (step_ms if hold_v is not None else 0))
    done = extra_h <= 0 and extra_v <= 0
    if not dirs:
        return _pack(FsmAction.NONE, (), extra_h, extra_v, 1, done)
    return _pack(FsmAction.HOLD, dirs, extra_h, extra_v, 1, done)


def _skills_for_dist(plan: tuple[DistSkillPlan, ...], key: str) -> tuple[FightSkill, ...]:
    for p in plan:
        if p.key == key:
            return p.skills
    return ()


def expand_fight_skills(skills: tuple[FightSkill, ...]) -> tuple[FightSkill, ...]:
    """多次释放：拆成 MULTI 份相同技能，charge 分开计 CD。"""
    out: list[FightSkill] = []
    for sk in skills:
        n = max(1, int(sk.multi or 1))
        if n <= 1:
            out.append(
                FightSkill(
                    slot=sk.slot,
                    key=sk.key,
                    cooldown_s=sk.cooldown_s,
                    range_px=sk.range_px,
                    combo=sk.combo,
                    hold_ms=sk.hold_ms,
                    multi=1,
                    charge=0,
                    mash=bool(sk.mash),
                )
            )
            continue
        for i in range(n):
            out.append(
                FightSkill(
                    slot=sk.slot,
                    key=sk.key,
                    cooldown_s=sk.cooldown_s,
                    range_px=sk.range_px,
                    combo=sk.combo,
                    hold_ms=sk.hold_ms,
                    multi=n,
                    charge=i,
                    mash=bool(sk.mash),
                )
            )
    return tuple(out)


def _cd_map(skill_cd: tuple[tuple[int, int, int], ...]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for item in skill_cd:
        if len(item) == 2:
            slot, ready = item
            out[(int(slot), 0)] = int(ready)
        else:
            slot, charge, ready = item
            out[(int(slot), int(charge))] = int(ready)
    return out


def _cd_tuple(ready: dict[tuple[int, int], int]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((int(s), int(c), int(v)) for (s, c), v in ready.items()))


def _ready_skills(skills: tuple[FightSkill, ...], ready: dict[tuple[int, int], int], t: int) -> list[FightSkill]:
    out = []
    for sk in skills:
        until = ready.get((sk.slot, sk.charge))
        if until is None or t >= until:
            out.append(sk)
    return out


def _first_file_ready(
    seq: tuple[FightSkill, ...], ready: dict[tuple[int, int], int], t: int
) -> FightSkill | None:
    """排除 CD 后，过图文件序列里第 1 个就绪技能（多次释放各份按序列顺序）。"""
    for sk in seq:
        until = ready.get((sk.slot, sk.charge))
        if until is None or t >= until:
            return sk
    return None


def _start_attack(xxx_ms: int, step_ms: int) -> tuple:
    n = max(1, int(xxx_ms))
    return FsmAction.ATTACK, None, None, None, "x", None, None, (), "普攻X", 0, max(0, n - max(0, int(step_ms)))


def _fight_intent(
    *,
    t: int,
    pos: tuple[float, float] | None,
    enemy_xy: tuple[tuple[float, float], ...],
    dist_key: str,
    plan: tuple[DistSkillPlan, ...],
    hotbar: tuple[FightSkill, ...],
    skill_cd: tuple[tuple[int, int, int], ...],
    cast_wait_left: int,
    attack_left: int,
    xxx_ms: int,
    dt_ms: int,
    last_slot: int | None = None,
    last_key: str | None = None,
) -> tuple[
    FsmAction,
    FsmDir | None,
    FsmDir | None,
    int | None,
    str | None,
    float | None,
    float | None,
    tuple[tuple[int, int, int], ...],
    str,
    int,
    int,
]:
    """action, move_dir, fight_dir, slot, key, range, dist, new_cd, note, cast_wait_left, attack_left。"""
    wait = max(0, int(cast_wait_left))
    atk = max(0, int(attack_left))
    step_ms = max(0, int(dt_ms))
    if wait > 0:
        return (
            FsmAction.NONE,
            None,
            None,
            last_slot,
            last_key,
            None,
            None,
            skill_cd,
            "等待释放",
            max(0, wait - step_ms),
            0,
        )
    if atk > 0:
        return (
            FsmAction.ATTACK,
            None,
            None,
            None,
            "x",
            None,
            None,
            skill_cd,
            "普攻X",
            0,
            max(0, atk - step_ms),
        )
    file_seq = _skills_for_dist(plan, dist_key)
    ready = _cd_map(skill_cd)
    file_first = _first_file_ready(file_seq, ready, t)
    bar_ready = _ready_skills(hotbar, ready, t)
    if file_first is not None:
        picked = file_first
        dist = _farthest_dist(pos, enemy_xy)
        rng = picked.range_px
        in_range = rng is None or dist is None or dist <= rng
        cd_s = max(0.0, float(picked.cooldown_s))
        ready[(picked.slot, picked.charge)] = t + int(round(cd_s * 1e9))
        new_cd = _cd_tuple(ready)
        hold = max(0, int(picked.hold_ms))
        return (
            FsmAction.CAST,
            None,
            None,
            picked.slot,
            picked.key,
            rng,
            dist,
            new_cd,
            "" if in_range else "范围异常",
            hold,
            0,
        )
    if bar_ready:
        picked = min(bar_ready, key=lambda sk: (max(0, int(sk.hold_ms)), sk.slot, sk.charge))
        cd_s = max(0.0, float(picked.cooldown_s))
        ready[(picked.slot, picked.charge)] = t + int(round(cd_s * 1e9))
        new_cd = _cd_tuple(ready)
        hold = max(0, int(picked.hold_ms))
        return (
            FsmAction.CAST,
            None,
            None,
            picked.slot,
            picked.key,
            picked.range_px,
            _farthest_dist(pos, enemy_xy),
            new_cd,
            "无技能可放",  # 不再看过图文件；快捷栏最短持续
            hold,
            0,
        )
    act, md, fd, slot, key, rng, dist, cd, note, cw, aw = _start_attack(xxx_ms, step_ms)
    return act, md, fd, slot, key, rng, dist, skill_cd, note, cw, aw


def _xy_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _loot_moving(
    prev: tuple[tuple[float, float], ...],
    now: tuple[tuple[float, float], ...],
    pm: float,
) -> bool:
    """当前掉落相对上一帧是否还在位移（超过 PM 像素）。新出现也算在动。"""
    pm = max(0.0, float(pm))
    if not now:
        return False
    if not prev:
        return True
    if len(now) > len(prev):
        return True
    leftover = list(prev)
    for p in now:
        if not leftover:
            return True
        j = min(range(len(leftover)), key=lambda i: _xy_dist(p, leftover[i]))
        if _xy_dist(p, leftover[j]) > pm:
            return True
        leftover.pop(j)
    return False


def _loot_intent(
    *,
    pos: tuple[float, float] | None,
    loot_xy: tuple[tuple[float, float], ...],
    loot_n: int,
    targets: tuple[tuple[float, float], ...],
    loot_dir: FsmDir | None,
    just_entered: bool,
    still_ok: bool,
    pm: int,
    pc: int,
) -> tuple[FsmAction, FsmDir | None, tuple[tuple[float, float], ...], FsmDir | None, str]:
    """action, move_dir, loot_targets, loot_dir, note。"""
    pm_px = max(0, int(pm))
    pc_n = max(0, int(pc))
    pts = loot_xy
    n = int(loot_n)
    if just_entered:
        targets = ()
        loot_dir = None
    if not still_ok:
        return FsmAction.NONE, None, (), None, "等待停下"
    if n > pc_n:
        return FsmAction.PICK, None, (), None, "一键拾取"
    match_r = max(float(pm_px) * 3.0, 24.0)
    kept: list[tuple[float, float]] = []
    for tgt in targets:
        if any(_xy_dist(p, tgt) <= match_r for p in pts):
            kept.append(tgt)
    if not kept:
        if pos is None:
            kept = list(pts)
        else:
            kept = sorted(pts, key=lambda p: _xy_dist(p, pos))
    if not kept:
        return FsmAction.NONE, None, (), None, ""
    tgt = kept[0]
    if pos is None:
        return FsmAction.NONE, None, tuple(kept), None, "无坐标"
    if _xy_dist(pos, tgt) <= max(float(pm_px), 8.0):
        return FsmAction.NONE, None, tuple(kept), loot_dir, "等消失"
    want = _axis_dir(pos, tgt)
    return FsmAction.HOLD, want, tuple(kept), want, "依次捡"


def _intent_label(
    *,
    state: FsmState,
    flags: frozenset[FsmFlag],
    action: FsmAction,
    move_dir: FsmDir | None,
    move_dirs: tuple[FsmDir, ...] = (),
    skill_slot: int | None,
    skill_key: str | None,
    fight_note: str,
    loot_note: str = "",
    through_done: bool = False,
    adv_phase: int = 0,
    fsm_run: int = 0,
    cast_wait_left: int = 0,
) -> str:
    recover = "恢复 " if FsmFlag.STUCK in flags else ""
    dirs = move_dirs or (() if move_dir is None else (move_dir,))
    dir_s = _dirs_label(dirs)
    act = f"{action.value} {dir_s}".strip() if dirs else action.value
    if FsmFlag.STUCK in flags or state is FsmState.STUCK:
        if action is FsmAction.NONE or not dirs:
            return "卡住"
        return f"卡住 {recover}{act}".replace("  ", " ").strip()
    if state is FsmState.FIGHT:
        sk = f"{skill_slot} {skill_key or ''}".strip()
        head = "提取分布+选技能" if fsm_run <= 1 else "选技能"
        if fight_note == "等待释放":
            left = max(1, int(cast_wait_left))
            return f"{head} 等待释放 {sk} 剩{left}ms".replace("  ", " ").strip()
        if fight_note == "范围异常":
            if action is FsmAction.CAST:
                return f"{head} 范围异常 仍然释放 {sk}".strip()
            return f"{head} 范围异常"
        if action is FsmAction.ATTACK:
            return f"{head} 普攻X"
        if action is FsmAction.CAST:
            if fight_note == "无技能可放":
                return f"{head} 无技能可放 最短释放 {sk}".strip()
            return f"{head} 释放 {sk}".strip()
        return f"{head} {fight_note}".strip() if fight_note else head
    if state is FsmState.LOOT:
        if loot_note == "等待停下":
            return "掉落在动 等待停下"
        if loot_note.startswith("捡物等待超时 PW"):
            return loot_note
        if action is FsmAction.PICK or loot_note == "一键拾取":
            return "数量>PC 一键拾取"
        if loot_note == "等消失":
            return "数量≤PC 等掉落消失"
        if loot_note == "无坐标":
            return "数量≤PC 无坐标"
        if action in (FsmAction.TAP, FsmAction.HOLD) and dirs:
            return f"数量≤PC 依次捡 {act}"
        if loot_note == "依次捡":
            return "数量≤PC 依次捡"
        return loot_note
    if state is FsmState.ADVANCE:
        through = through_done or adv_phase == 1
        if through:
            if action is FsmAction.NONE or not dirs:
                return "过门 完成" if through_done else "过门"
            return f"过门 {act}"
        head = "确认方向+前进" if fsm_run <= 1 else "前进"
        if action is FsmAction.NONE or not dirs:
            return head
        return f"{head} {act}"
    if action is FsmAction.NONE or not dirs:
        return ""
    return act


def _cd_remain(
    t: int, skill_cd: tuple[tuple[int, int, int], ...], skills: tuple[FightSkill, ...]
) -> tuple[tuple[int, str, float], ...]:
    ready = _cd_map(skill_cd)
    out: list[tuple[int, str, float]] = []
    for sk in skills:
        until = ready.get((sk.slot, sk.charge))
        rem = 0.0 if until is None or t >= until else (until - t) / 1e9
        label = sk.key if sk.multi <= 1 else f"{sk.key}·{sk.charge + 1}"
        out.append((sk.slot, label, round(max(0.0, rem), 2)))
    return tuple(out)


def _method_flow(
    *,
    state: FsmState,
    flags: frozenset[FsmFlag],
    action: FsmAction,
    fight_note: str,
    extra_h: int,
    extra_v: int,
    through_done: bool,
    recover_dir_i: int,
    adv_phase: int = 0,
    loot_note: str = "",
    fsm_run: int = 0,
) -> tuple[tuple[str, ...], int]:
    if FsmFlag.STUCK in flags or state is FsmState.STUCK:
        return ("卡住", "上", "下", "左", "右"), (recover_dir_i % 4) + 1
    if state is FsmState.FIGHT:
        steps = ("1提取分布", "2选技能")
        if fsm_run <= 1:
            return steps, 0
        return steps, 1
    if state is FsmState.LOOT:
        steps = ("1掉落在动?", "2数量>PC?")
        if loot_note == "等待停下":
            return steps, 0
        if loot_note.startswith("捡物等待超时 PW"):
            return steps, 1
        return steps, 1
    if state is FsmState.ADVANCE:
        steps = ("1确认方向", "2前进", "3过门")
        through = through_done or adv_phase == 1
        moving_through = through and action in (FsmAction.HOLD, FsmAction.TAP)
        if moving_through or (through_done and fsm_run > 1):
            return steps, 2
        if fsm_run <= 1:
            return steps, 0
        if through:
            return steps, 2
        return steps, 1
    return (), -1


def _deb_step(d: _Deb, raw: bool, n: int) -> _Deb:
    """因果防抖：连续 n 帧同值才改 judged。不看未来帧。"""
    n = max(1, int(n))
    v = bool(raw)
    run_v, run_n, judged = d.run_v, d.run_n, d.judged
    if v is run_v:
        run_n += 1
    else:
        run_v, run_n = v, 1
    if run_n >= n:
        judged = v
    return _Deb(judged=judged, run_v=run_v, run_n=run_n)


def dungeon_deb_step(d: _Deb, kw: bool, t_ns: int, tn_s: float) -> _Deb:
    """进图：看到关键词立刻进。回城：连续 tn_s 秒无关键词才出。"""
    t = int(t_ns)
    if kw:
        return _Deb(judged=True, run_v=True, run_n=1, run_t=t)
    sec = max(0.0, float(tn_s))
    if d.run_v is False and d.run_t is not None:
        start = d.run_t
    else:
        start = t
    elapsed = (t - start) / 1e9
    if d.judged and elapsed < sec:
        return _Deb(judged=True, run_v=False, run_n=d.run_n + 1, run_t=start)
    return _Deb(judged=False, run_v=False, run_n=d.run_n + 1 if d.run_v is False else 1, run_t=start)


def _draft(
    *,
    in_dungeon: bool,
    has_enemy: bool,
    has_loot: bool,
    has_gate: bool,
    town_return: bool,
    raw_mon: int,
    raw_loot: int,
    raw_gate: int,
    raw_boss: int,
    m: int,
    l: int,
    g: int,
) -> tuple[FsmState, str, str]:
    chain = [
        (
            "1 进图?",
            in_dungeon,
            "已进图 → 往下" if in_dungeon else "未进图 → 等待",
        ),
        (
            "2 MON/BOSS?",
            has_enemy,
            f"raw mon={raw_mon} boss={raw_boss}  判定={int(has_enemy)} (M={m}) → "
            + ("开打" if has_enemy else "否，往下"),
        ),
        (
            "3 LOOT?",
            has_loot,
            f"raw loot={raw_loot}  判定={int(has_loot)} (L={l}) → " + ("捡物" if has_loot else "否，往下"),
        ),
        (
            "4 GATE?",
            has_gate,
            f"raw gate={raw_gate}  判定={int(has_gate)} (G={g}) → " + ("前进" if has_gate else "否，往下"),
        ),
        (
            "5 返回城镇?",
            town_return,
            "连续无地下城关键词 → 返回" if town_return else "未进过图或仍在图内",
        ),
    ]
    if not in_dungeon:
        if town_return:
            state, hit_i = FsmState.RETURN, 4
        else:
            state, hit_i = FsmState.WAIT, 0
    elif has_enemy:
        state, hit_i = FsmState.FIGHT, 1
    elif has_loot:
        state, hit_i = FsmState.LOOT, 2
    elif has_gate:
        state, hit_i = FsmState.ADVANCE, 3
    elif town_return:
        state, hit_i = FsmState.RETURN, 4
    else:
        state, hit_i = FsmState.IDLE, 4

    why = f"命中步骤{hit_i + 1}：{chain[hit_i][2]}"
    if hit_i < 4:
        skipped = "、".join(chain[j][0] for j in range(hit_i + 1, 5))
        why += f"  （短路，未看 {skipped}）"
    lines = []
    for i, (title, _ok, detail) in enumerate(chain):
        mark = "▶" if i == hit_i else "·"
        lines.append(f"{mark} {title}  {detail}")
    return state, why, "\n".join(lines)


def _method_hold(
    ctx: FsmContext, drafted: FsmState, loot_on: bool
) -> tuple[FsmState | None, str]:
    """方法未跑完时不让总流程短路切走。"""
    prev = ctx.state
    if prev is FsmState.ADVANCE and not ctx.through_done:
        if drafted is not FsmState.ADVANCE:
            return FsmState.ADVANCE, "方法未完·前进"
        return None, ""
    if prev is FsmState.FIGHT and (ctx.cast_wait_left > 0 or ctx.attack_left > 0):
        if drafted is not FsmState.FIGHT:
            return FsmState.FIGHT, "方法未完·开打"
        return None, ""
    if prev is FsmState.LOOT and loot_on:
        if drafted is not FsmState.LOOT:
            return FsmState.LOOT, "方法未完·捡物"
        return None, ""
    return None, ""


def step(
    snap: FsmSnapshot,
    ctx: FsmContext | None,
    params: FsmParams,
) -> tuple[FsmDecision, FsmContext]:
    if ctx is None:
        ctx = FsmContext()
    t = int(snap.t_ns)
    if ctx.last_t_ns is not None and t < ctx.last_t_ns:
        raise ValueError(f"t_ns 必须单调不减: prev={ctx.last_t_ns} got={t}")

    m, l, g = max(1, int(params.m)), max(1, int(params.l)), max(1, int(params.g))
    x_s = max(0.05, float(params.x_s))
    gx, gy = max(0, int(params.gx)), max(0, int(params.gy))
    ax_ms, ay_ms = max(0, int(params.ax_ms)), max(0, int(params.ay_ms))
    y_ms = max(1, int(params.y_ms))
    s = max(0, int(params.s))
    pm = max(0, int(params.pm))
    pc = max(0, int(params.pc))
    pt = max(1, int(params.pt))
    xxx_ms = max(1, int(params.xxx_ms))
    th_ms = max(0, int(params.th_ms))
    pw_ms = max(0, int(params.pw_ms))
    tn_s = max(0.0, float(params.tn_s))
    step_ms = dt_ms(ctx.last_t_ns, t)
    ox, oy = int(params.mon_off_x), int(params.mon_off_y)
    if ox or oy:
        corr = apply_mon_boss_corr({"mon_xy": snap.mon_xy, "boss_xy": snap.boss_xy}, ox, oy)
        snap = replace(
            snap,
            mon_xy=_parse_xy_list(corr.get("mon_xy")),
            boss_xy=_parse_xy_list(corr.get("boss_xy")),
        )
    n_seen = ctx.n_seen + 1

    mon = _deb_step(ctx.mon, snap.mon > 0, m)
    boss = _deb_step(ctx.boss, snap.boss > 0, m)  # BOSS 暂与 MON 共用 M
    loot = _deb_step(ctx.loot, snap.loot > 0, l)
    gate = _deb_step(ctx.gate, snap.gate > 0, g)
    has_enemy = mon.judged or boss.judged

    if snap.dungeon_kw is None:
        dun = _Deb(judged=True, run_v=True, run_n=1)
        in_dungeon = True
        saw_dungeon = True
        town_return = False
    else:
        dun = dungeon_deb_step(ctx.dungeon, bool(snap.dungeon_kw), t, tn_s)
        in_dungeon = dun.judged
        saw_dungeon = ctx.saw_dungeon or in_dungeon
        town_return = saw_dungeon and not in_dungeon

    drafted, why, chain = _draft(
        in_dungeon=in_dungeon,
        has_enemy=has_enemy,
        has_loot=loot.judged,
        has_gate=gate.judged,
        town_return=town_return,
        raw_mon=snap.mon,
        raw_loot=snap.loot,
        raw_gate=snap.gate,
        raw_boss=snap.boss,
        m=m,
        l=l,
        g=g,
    )
    hold, hold_why = (None, "")
    if ctx.state is not FsmState.STUCK:
        hold, hold_why = _method_hold(ctx, drafted, loot.judged)
    flow = hold if hold is not None else drafted
    if hold is not None:
        why += f"  {hold_why}"
    if ctx.watch_state is flow and ctx.watch_start_t is not None:
        watch_start_t = ctx.watch_start_t
        watch_run = ctx.watch_run + 1
    else:
        watch_start_t = t
        watch_run = 1
    watch_s = (t - watch_start_t) / 1e9
    if watch_s >= x_s:
        flow = drafted
        if ctx.watch_state is flow and ctx.watch_start_t is not None:
            watch_start_t = ctx.watch_start_t
            watch_run = ctx.watch_run + 1
        else:
            watch_start_t = t
            watch_run = 1
        watch_s = (t - watch_start_t) / 1e9
        state = FsmState.STUCK if watch_s >= x_s else flow
        if state is FsmState.STUCK:
            why += "  卡住打断"
    else:
        state = flow

    if snap.player_xy is not None:
        last_xy = snap.player_xy
        player_hold = False
        pos = snap.player_xy
    else:
        last_xy = ctx.last_player_xy
        player_hold = last_xy is not None
        pos = last_xy

    rooms = ctx.rooms
    extra_sigs = ctx.extra_sigs
    boss_seen = ctx.boss_seen
    dist_key = ctx.dist_key
    dist_kind = ctx.dist_kind
    room_kind = None
    gate_crosses = ctx.gate_crosses
    boss_enters = ctx.boss_enters
    boss_kills = ctx.boss_kills
    if ctx.gate.judged and not gate.judged:
        gate_crosses += 1
    if (not ctx.boss.judged) and boss.judged:
        boss_enters += 1
    cd_reset = False
    skill_cd = ctx.skill_cd
    if ctx.boss.judged and not boss.judged:
        boss_kills += 1
        if params.map_reset:
            skill_cd = ()
            cd_reset = True
            why += "  CD重置"
    if state in (FsmState.WAIT, FsmState.RETURN):
        rooms = 0
        extra_sigs = ()
        boss_seen = False
        dist_key = ""
        dist_kind = ""
        gate_crosses = 0
        boss_enters = 0
        boss_kills = 0
    elif state is FsmState.FIGHT and ctx.state is not FsmState.FIGHT:
        if boss.judged or snap.boss_xy:
            rel = _rel_of(pos, tuple(snap.boss_xy))
            multi = len(snap.boss_xy) > 1
            dist_kind = "boss"
            dist_key, extra_sigs, room_kind, _reset_pt = _match_dist(
                params.dist_table,
                extra_sigs,
                "boss",
                len(snap.boss_xy),
                rel,
                s,
                multi_boss=multi,
            )
        elif mon.judged or snap.mon_xy:
            rel = _rel_of(pos, tuple(snap.mon_xy))
            dist_kind = "mob"
            dist_key, extra_sigs, room_kind, _reset_pt = _match_dist(
                params.dist_table,
                extra_sigs,
                "mob",
                int(snap.mon),
                rel,
                s,
            )
        else:
            dist_kind = "abnormal"
            dist_key = "X"
            room_kind = "new"
        rooms = int("".join(ch for ch in dist_key if ch.isdigit()) or 0)
        tag = "BOSS" if dist_kind == "boss" else ("小怪" if dist_kind == "mob" else "异常")
        why += f"  怪物分布 {dist_key}【{tag}】{' 相似' if room_kind == 'similar' else ' 新'}"
        boss_seen = boss.judged
    if boss.judged:
        boss_seen = True

    if state is ctx.state:
        fsm_run = ctx.fsm_run + 1
        run_start_t = ctx.run_start_t if ctx.run_start_t is not None else t
    else:
        fsm_run = 1
        run_start_t = t

    flags_list: list[FsmFlag] = []
    if n_seen <= max(m, l, g):
        flags_list.append(FsmFlag.WARMUP)
    stuck = state is FsmState.STUCK
    if stuck:
        flags_list.append(FsmFlag.STUCK)
    flags = frozenset(flags_list)

    skill_slot = None
    skill_key = None
    skill_range = None
    pack_dist = None
    fight_note = ""
    loot_note = ""
    fight_dir = None
    cast_wait_left = 0
    attack_left = 0
    loot_targets: tuple[tuple[float, float], ...] = ()
    loot_dir = None
    loot_stop = _Deb()
    loot_wait_t = None
    loot_wait_done = False
    move_dirs: tuple[FsmDir, ...] = ()
    adv_dirs: tuple[FsmDir, ...] = ()
    adv_phase = 0
    extra_h = 0
    extra_v = 0
    adv_lock = (0, 0, 0, 0)
    adv_dir_set = False
    recover_on = False
    recover_dir_i = 0
    recover_ms = 0
    through_done = False
    if stuck or state is FsmState.ADVANCE:
        (
            action,
            move_dir,
            move_dirs,
            adv_dirs,
            extra_h,
            extra_v,
            adv_phase,
            adv_lock,
            recover_on,
            recover_dir_i,
            recover_ms,
            through_done,
            adv_dir_set,
        ) = _intent(
            state=state,
            stuck=stuck,
            pos=pos,
            gate_xy=snap.gate_xy,
            prev_state=ctx.state,
            ctx=ctx,
            gx=gx,
            gy=gy,
            ax_ms=ax_ms,
            ay_ms=ay_ms,
            y_ms=y_ms,
            dt_ms=step_ms,
        )
    elif state is FsmState.FIGHT:
        enemy_xy = tuple(snap.boss_xy) + tuple(snap.mon_xy)
        (
            action,
            move_dir,
            fight_dir,
            skill_slot,
            skill_key,
            skill_range,
            pack_dist,
            skill_cd,
            fight_note,
            cast_wait_left,
            attack_left,
        ) = _fight_intent(
            t=t,
            pos=pos,
            enemy_xy=enemy_xy,
            dist_key=dist_key,
            plan=params.fight_plan,
            hotbar=params.hotbar,
            skill_cd=skill_cd,
            cast_wait_left=ctx.cast_wait_left if ctx.state is FsmState.FIGHT else 0,
            attack_left=ctx.attack_left if ctx.state is FsmState.FIGHT else 0,
            xxx_ms=xxx_ms,
            dt_ms=step_ms,
            last_slot=ctx.cast_slot if ctx.state is FsmState.FIGHT else None,
            last_key=ctx.cast_key if ctx.state is FsmState.FIGHT else None,
        )
        if fight_note:
            why += f"  {fight_note}"
        elif action is FsmAction.CAST:
            why += f"  释放技能{skill_slot} {skill_key or ''}".rstrip()
    elif state is FsmState.LOOT:
        just_entered = ctx.state is not FsmState.LOOT
        prev_xy = ctx.loot_xy_prev if not just_entered else ()
        raw_moving = _loot_moving(prev_xy, snap.loot_xy, pm)
        loot_stop = _deb_step(
            ctx.loot_stop if not just_entered else _Deb(),
            (not raw_moving),
            pt,
        )
        loot_wait_done = False if just_entered else bool(ctx.loot_wait_done)
        loot_wait_t = None if just_entered else ctx.loot_wait_t
        timed_out = False
        if loot_wait_done:
            still_ok = True
        elif loot_stop.judged:
            still_ok = True
            loot_wait_t = None
        else:
            if loot_wait_t is None:
                loot_wait_t = t
            waited_ms = (t - loot_wait_t) / 1e6
            if pw_ms > 0 and waited_ms >= pw_ms:
                still_ok = True
                loot_wait_done = True
                timed_out = True
                loot_wait_t = None
            else:
                still_ok = False
        action, move_dir, loot_targets, loot_dir, loot_note = _loot_intent(
            pos=pos,
            loot_xy=snap.loot_xy,
            loot_n=int(snap.loot),
            targets=ctx.loot_targets if not just_entered else (),
            loot_dir=ctx.loot_dir if not just_entered else None,
            just_entered=just_entered,
            still_ok=still_ok,
            pm=pm,
            pc=pc,
        )
        if timed_out or loot_wait_done:
            if loot_note == "等待停下":
                loot_note = "捡物等待超时 PW"
            else:
                loot_note = f"捡物等待超时 PW {loot_note}".strip()
        if loot_note:
            why += f"  {loot_note}"
    else:
        action, move_dir = FsmAction.NONE, None

    dash_want: tuple[FsmDir, ...] = ()
    dash_phase = 0
    dash_left = 0
    if state is FsmState.LOOT and not move_dirs and move_dir is not None:
        move_dirs = (move_dir,)
    approaching = (
        not stuck
        and action in (FsmAction.HOLD, FsmAction.TAP)
        and (
            (state is FsmState.ADVANCE and adv_phase == 0)
            or state is FsmState.LOOT
        )
    )
    if approaching:
        want = _norm_dirs(move_dirs or ((move_dir,) if move_dir else ()))
        same_move = ctx.state in (FsmState.ADVANCE, FsmState.LOOT)
        action, move_dirs, dash_want, dash_phase, dash_left = _tap_hold(
            want,
            ctx.dash_want if same_move else (),
            ctx.dash_phase if same_move else 0,
            ctx.dash_left if same_move else 0,
            th_ms,
            step_ms,
        )
        move_dir = move_dirs[0] if move_dirs else None

    intent_label = _intent_label(
        state=state,
        flags=flags,
        action=action,
        move_dir=move_dir,
        move_dirs=move_dirs,
        skill_slot=skill_slot,
        skill_key=skill_key,
        fight_note=fight_note,
        loot_note=loot_note,
        through_done=through_done,
        adv_phase=adv_phase,
        fsm_run=fsm_run,
        cast_wait_left=cast_wait_left,
    )
    skill_cds = _cd_remain(
        t,
        skill_cd,
        _skills_for_dist(params.fight_plan, dist_key) or params.hotbar,
    )
    flow_steps, flow_hit = _method_flow(
        state=state,
        flags=flags,
        action=action,
        fight_note=fight_note,
        extra_h=extra_h,
        extra_v=extra_v,
        through_done=through_done,
        recover_dir_i=recover_dir_i,
        adv_phase=adv_phase,
        loot_note=loot_note,
        fsm_run=fsm_run,
    )

    new_ctx = FsmContext(
        mon=mon,
        boss=boss,
        loot=loot,
        gate=gate,
        state=state,
        fsm_run=fsm_run,
        run_start_t=run_start_t,
        rooms=rooms,
        boss_seen=boss_seen,
        dist_key=dist_key if state not in (FsmState.WAIT, FsmState.RETURN) else "",
        dist_kind=dist_kind if state not in (FsmState.WAIT, FsmState.RETURN) else "",
        extra_sigs=extra_sigs,
        last_player_xy=last_xy,
        last_t_ns=t,
        n_seen=n_seen,
        adv_dir=adv_dirs[0] if adv_dirs else None,
        adv_dirs=adv_dirs,
        adv_phase=adv_phase if state is FsmState.ADVANCE else 0,
        extra_h=extra_h if state is FsmState.ADVANCE else 0,
        extra_v=extra_v if state is FsmState.ADVANCE else 0,
        adv_lock=adv_lock if state is FsmState.ADVANCE else (0, 0, 0, 0),
        adv_dir_set=adv_dir_set if state is FsmState.ADVANCE else False,
        recover_on=recover_on,
        recover_dir_i=recover_dir_i,
        recover_ms=recover_ms,
        through_done=through_done if state is FsmState.ADVANCE else False,
        gate_crosses=gate_crosses,
        boss_enters=boss_enters,
        boss_kills=boss_kills,
        skill_cd=skill_cd,
        fight_dir=fight_dir,
        cast_wait_left=cast_wait_left if state is FsmState.FIGHT else 0,
        attack_left=attack_left if state is FsmState.FIGHT else 0,
        cast_slot=skill_slot if state is FsmState.FIGHT else None,
        cast_key=skill_key if state is FsmState.FIGHT else None,
        loot_xy_prev=snap.loot_xy if state is FsmState.LOOT else (),
        loot_targets=loot_targets if state is FsmState.LOOT else (),
        loot_dir=loot_dir if state is FsmState.LOOT else None,
        loot_stop=loot_stop if state is FsmState.LOOT else _Deb(),
        loot_wait_t=loot_wait_t if state is FsmState.LOOT else None,
        loot_wait_done=loot_wait_done if state is FsmState.LOOT else False,
        dash_want=dash_want,
        dash_phase=dash_phase,
        dash_left=dash_left,
        dungeon=dun,
        saw_dungeon=saw_dungeon if state not in (FsmState.WAIT,) else False,
        watch_state=flow,
        watch_run=watch_run,
        watch_start_t=watch_start_t,
    )
    decision = FsmDecision(
        state=state,
        flags=flags,
        action=action,
        move_dir=move_dir,
        move_dirs=move_dirs,
        why=why,
        chain=chain,
        rooms=rooms,
        boss_seen=boss_seen,
        room_kind=room_kind,
        fsm_run=fsm_run,
        fsm_run_s=(t - run_start_t) / 1e9,
        player_hold=player_hold,
        eff_mon=mon.judged,
        eff_loot=loot.judged,
        eff_gate=gate.judged,
        eff_boss=boss.judged,
        skill_slot=skill_slot,
        skill_key=skill_key,
        skill_range=skill_range,
        pack_dist=pack_dist,
        intent_label=intent_label,
        cd_reset=cd_reset,
        skill_cds=skill_cds,
        flow_steps=flow_steps,
        flow_hit=flow_hit,
        send_label=send_keys_label(
            action,
            move_dirs,
            move_dir,
            skill_key,
        ),
        dist_key=dist_key,
        dist_kind=dist_kind,
        gate_crosses=gate_crosses,
        boss_enters=boss_enters,
        boss_kills=boss_kills,
    )
    return decision, new_ctx


def run_track(frames: list[dict], params: FsmParams) -> list[dict]:
    """回放宿主：按 jsonl 的 t_ns 原样逐步 step。缺戳或墙钟填充都不允许。"""
    ctx = FsmContext()
    out: list[dict] = []
    for i, fr in enumerate(frames):
        if "t_ns" not in fr or fr["t_ns"] is None:
            raise ValueError(f"回放第 {i} 帧缺少 t_ns，禁止用当前时间填充")
        snap = snapshot_from_detect(
            int(fr["t_ns"]),
            fr,
            dungeon_kw=None if fr.get("dungeon_kw") is None else bool(fr.get("dungeon_kw")),
        )
        decision, ctx = step(snap, ctx, params)
        out.append(decision.as_row())
    return out
