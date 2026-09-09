"""FSM 核心：纯函数。step(snapshot, ctx, params) → (decision, new_ctx)

禁止：time / sleep / 读文件 / 发按键 / 截屏 / random / 模块级可变状态。
t_ns 只信快照：回放必须用 jsonl 原值；实机必须用 grab 后、save/YOLO 前的采样戳。
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


_MOVE_DIR_ORDER = (FsmDir.UP, FsmDir.DOWN, FsmDir.LEFT, FsmDir.RIGHT)
_DIR_BY_KEY = {
    "up": FsmDir.UP,
    "down": FsmDir.DOWN,
    "left": FsmDir.LEFT,
    "right": FsmDir.RIGHT,
}
DEFAULT_TH_MS = 50
DEFAULT_PW_MS = 3000
DEFAULT_ADVANCE_TIMEOUT_MS = 3000
DEFAULT_APPROACH_TIMEOUT_MS = 3000
DEFAULT_STUCK_RECOVER_S = 1.0


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
    """过图技能序列里的一项。range_x/y 为范围框半宽高；range_px 旧圆形兼容。"""

    slot: int
    key: str
    cooldown_s: float
    range_px: float | None = None
    range_x: float | None = None
    range_y: float | None = None
    combo: bool = False
    hold_ms: int = 0
    multi: int = 1
    charge: int = 0
    mash: bool = False  # 键位表【连按】：宿主/执行层用，核心不读
    group_key: str = ""
    group_slots: tuple[int, ...] = ()
    group_keys: tuple[str, ...] = ()
    group_holds: tuple[int, ...] = ()
    gaps_ms: tuple[int, ...] = ()


@dataclass(frozen=True)
class DistSkillPlan:
    """该怪物分布在过图技能文件里的序列。"""

    key: str
    skills: tuple[FightSkill, ...] = ()
    reset_point: bool = False  # 旧特征字段；运行时清 CD 用 map_reset + 击败 BOSS


@dataclass(frozen=True)
class DistSig:
    """BOSS位置表 / 小怪分布表一项。rel 是相对玩家；小怪多目标另存框宽高（单怪为 0）。"""

    key: str
    kind: str
    n: int
    rx: float
    ry: float
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    reset_point: bool = False
    multi_boss: bool = False


@dataclass(frozen=True)
class StuckRecoverStep:
    """卡住恢复一步。HOLD 时长：有 ms 用绝对毫秒，否则 s × stuck_recover_s 秒。TAP 忽略时长。"""

    action: str  # tap | hold
    key: str
    s: float | None = None
    ms: int | None = None


DEFAULT_STUCK_RECOVER: tuple[StuckRecoverStep, ...] = (
    StuckRecoverStep("tap", "esc"),
    StuckRecoverStep("hold", "right", s=2),
    StuckRecoverStep("hold", "left", s=2),
    StuckRecoverStep("hold", "up", s=1),
    StuckRecoverStep("hold", "down", s=1),
)


def stuck_recover_step_ms(step: StuckRecoverStep, base_s: float) -> int:
    if str(step.action) == "tap":
        return 0
    if step.ms is not None:
        return max(0, int(step.ms))
    mult = 1.0 if step.s is None else float(step.s)
    return max(0, int(round(mult * max(0.0, float(base_s)) * 1000.0)))


def parse_stuck_recover(raw) -> tuple[StuckRecoverStep, ...]:
    """读配置列表；空/缺/无效 → 默认序列。有 ms 用绝对时长，否则 s 为 stuck_recover_s 的倍数。"""
    if isinstance(raw, dict):
        raw = raw.get("stuck_recover")
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], StuckRecoverStep):
        return tuple(raw)
    if not isinstance(raw, (list, tuple)) or not raw:
        return DEFAULT_STUCK_RECOVER
    out: list[StuckRecoverStep] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "hold").strip().lower()
        action = "tap" if action in ("tap", "点按") else "hold"
        key = str(item.get("key") or item.get("dir") or "").strip().lower()
        if not key:
            continue
        ms_v = None
        if item.get("ms") is not None:
            try:
                ms_v = max(0, int(item["ms"]))
            except (TypeError, ValueError):
                ms_v = None
        s_v = None
        if item.get("s") is not None:
            try:
                s_v = float(item["s"])
            except (TypeError, ValueError):
                s_v = None
        out.append(StuckRecoverStep(action=action, key=key, s=s_v, ms=ms_v))
    return tuple(out) if out else DEFAULT_STUCK_RECOVER


def stuck_recover_to_json(steps: tuple[StuckRecoverStep, ...] | None = None) -> list[dict]:
    rows = []
    for st in steps or DEFAULT_STUCK_RECOVER:
        row: dict = {"action": st.action, "key": st.key}
        if st.ms is not None:
            row["ms"] = int(st.ms)
        elif st.s is not None:
            row["s"] = st.s
        rows.append(row)
    return rows


def stuck_recover_seq(params: FsmParams) -> tuple[StuckRecoverStep, ...]:
    return params.stuck_recover or DEFAULT_STUCK_RECOVER


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
    y_ms: int = 250  # 旧字段；卡住恢复不再用四向 Y
    stuck_recover_s: float = DEFAULT_STUCK_RECOVER_S  # 卡住恢复 HOLD 基准秒；步骤 s 为倍数
    stuck_recover: tuple[StuckRecoverStep, ...] = ()  # 空则用 DEFAULT_STUCK_RECOVER
    s: int = 20  # 兼容：s_pos/s_size 为空时两者都用它
    s_pos: int | None = None
    s_size: int | None = None
    pm: int = 10
    pc: int = 5
    pt: int = 3
    xxx_ms: int = 1000
    th_ms: int = DEFAULT_TH_MS  # 捡物/前进接近：点按后等到按住的间隔
    pw_ms: int = DEFAULT_PW_MS  # 捡物等停下上限；超时后按当前掉落继续步骤 2
    advance_timeout_ms: int = DEFAULT_ADVANCE_TIMEOUT_MS  # 前进方法超时；须>0；结束方法并重入判定
    approach_timeout_ms: int = DEFAULT_APPROACH_TIMEOUT_MS  # 开打范围外走近超时；须>0；结束走近并重选技能
    tn_s: float = DEFAULT_TOWN_S  # 回城：连续无地下城关键词的秒数
    fight_plan: tuple[DistSkillPlan, ...] = ()
    hotbar: tuple[FightSkill, ...] = ()
    dist_table: tuple[DistSig, ...] = ()
    map_reset: bool = False
    mon_off_x: int = 0  # >0 右：YOLO MON/BOSS x 加；<0 左：减
    mon_off_y: int = 0  # >0 下：YOLO MON/BOSS y 加；<0 上：减
    loot_off_x: int = 0  # >0 右：YOLO loot x 加；<0 左：减
    loot_off_y: int = 0  # >0 下：YOLO loot y 加；<0 上：减
    gate_off_x: int = 0  # >0 右：YOLO gate x 加；<0 左：减
    gate_off_y: int = 0  # >0 下：YOLO gate y 加；<0 上：减


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
    recover_key: str = ""
    through_done: bool = False
    adv_start_t: int | None = None  # 本次前进方法起点 t_ns
    approach_start_t: int | None = None  # 本次范围外走近起点 t_ns
    approach_charge: int = 0
    approach_skip: tuple[tuple[int, int], ...] = ()  # 走近超时跳过的 (slot, charge)，离开关打清空；不进 CD
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
    fight_rest: tuple[tuple[int, str, int], ...] = ()
    fight_gaps: tuple[int, ...] = ()
    fight_tail: int = 0
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
    fight_dir: FsmDir | None = None
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
            "fight_dir": self.fight_dir,
            "fight_dir_label": None if self.fight_dir is None else self.fight_dir.value,
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
    fight_dir: FsmDir | None = None,
) -> str:
    """这一帧会按的键（展示用）。mash_n 由宿主按执行层同一套槽+COUNT 传入，核心不填。"""
    dirs = move_dirs or (() if move_dir is None else (move_dir,))
    names = [d.value for d in dirs]
    if action is FsmAction.HOLD and names:
        return "按住 " + "+".join(names)
    if action is FsmAction.HOLD and skill_key:
        return f"按住 {skill_key}"
    if action is FsmAction.TAP and names:
        return "点按 " + "+".join(names)
    if action is FsmAction.TAP:
        cmd = str(skill_key or "").strip()
        return f"点按 {cmd}" if cmd else ""
    if action is FsmAction.CAST:
        cmd = str(skill_key or "").strip()
        if not cmd:
            return ""
        n = max(0, int(mash_n or 0))
        skill = f"连按{n}× {cmd}" if n > 1 else f"点按 {cmd}"
        if fight_dir in (FsmDir.LEFT, FsmDir.RIGHT):
            return f"点按 {fight_dir.value} 再 {skill}"
        return skill
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


def _s_pos_size(params: FsmParams | None = None, s: float | None = None, s_pos: float | None = None, s_size: float | None = None) -> tuple[float, float]:
    if params is not None:
        base = float(params.s)
        sp = base if params.s_pos is None else float(params.s_pos)
        ss = base if params.s_size is None else float(params.s_size)
        return max(0.0, sp), max(0.0, ss)
    base = 0.0 if s is None else float(s)
    sp = base if s_pos is None else float(s_pos)
    ss = base if s_size is None else float(s_size)
    return max(0.0, sp), max(0.0, ss)


def _pack_center(pts: tuple[tuple[float, float], ...]) -> tuple[float, float] | None:
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def bbox_wh_rel(
    player: tuple[float, float] | None, pts: tuple[tuple[float, float], ...]
) -> tuple[float, float]:
    """多目标相对玩家的范围框宽高；单点或无人则为 0。"""
    if player is None or len(pts) <= 1:
        return 0.0, 0.0
    xs = [p[0] - player[0] for p in pts]
    ys = [p[1] - player[1] for p in pts]
    return float(max(xs) - min(xs)), float(max(ys) - min(ys))


def bbox_rel(
    player: tuple[float, float] | None, pts: tuple[tuple[float, float], ...]
) -> tuple[float, float, float, float] | None:
    """相对玩家的 minx,miny,maxx,maxy；单点为 None（框尺寸 0）。"""
    if player is None or len(pts) <= 1:
        return None
    xs = [p[0] - player[0] for p in pts]
    ys = [p[1] - player[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_range_xy(box: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    """相对玩家的范围框 → (|X|上限, |Y|上限)。单点敌对时为到该点的 |dx|,|dy|。"""
    if box is None or len(box) < 4:
        return None
    x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    rx = max(abs(x0), abs(x1))
    ry = max(abs(y0), abs(y1))
    return (round(rx, 2), round(ry, 2))


def bbox_range_px(box: tuple[float, float, float, float] | None) -> float | None:
    """旧圆形半径（框角最大hypot）；新逻辑优先用 bbox_range_xy。"""
    xy = bbox_range_xy(box)
    if xy is None:
        return None
    rx, ry = xy
    return round(math.hypot(rx, ry), 2)


def _count_pct(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0) * 100.0


def _pos_err_pct(
    a: tuple[float, float] | None, b: tuple[float, float] | None
) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1e9
    return max(abs(a[0] - b[0]) / _REF_W, abs(a[1] - b[1]) / _REF_H) * 100.0


def _size_err_pct(w1: float, h1: float, w2: float, h2: float) -> float:
    return max(abs(w1 - w2) / _REF_W, abs(h1 - h2) / _REF_H) * 100.0


def rooms_similar(
    n1: int,
    c1: tuple[float, float] | None,
    n2: int,
    c2: tuple[float, float] | None,
    s: float,
) -> bool:
    """旧接口：数量+中心都不超过 S%。新匹配走 _match_dist（最相似 + S_pos/S_size）。"""
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
    s_pos: float,
    s_size: float,
    *,
    bbox_w: float = 0.0,
    bbox_h: float = 0.0,
    multi_boss: bool = False,
) -> tuple[str, tuple[DistSig, ...], str, bool]:
    """全表取最相似一条，再过 S_pos（中心）/ S_size（多怪框）。过则视为同一分布。"""
    combined = table + extra
    s_pos = max(0.0, float(s_pos))
    s_size = max(0.0, float(s_size))
    bw, bh = float(bbox_w), float(bbox_h)
    best: tuple[float, DistSig] | None = None
    for e in combined:
        if e.kind != kind:
            continue
        pe = _pos_err_pct((e.rx, e.ry), rel)
        if kind == "boss":
            score = pe
        elif int(n) <= 1 and int(e.n) <= 1:
            score = pe
        else:
            score = pe + _size_err_pct(bw, bh, e.bbox_w, e.bbox_h)
        if best is None or score < best[0]:
            best = (score, e)
    if best is not None:
        e = best[1]
        if kind == "boss" or int(n) <= 1:
            ok = _pos_err_pct((e.rx, e.ry), rel) <= s_pos + 1e-9
        else:
            ok = (
                _pos_err_pct((e.rx, e.ry), rel) <= s_pos + 1e-9
                and _size_err_pct(bw, bh, e.bbox_w, e.bbox_h) <= s_size + 1e-9
            )
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
        bbox_w=0.0 if int(n) <= 1 else bw,
        bbox_h=0.0 if int(n) <= 1 else bh,
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
    s_pos: float | None = None,
    s_size: float | None = None,
) -> tuple[str, str, tuple[DistSig, ...], str]:
    """按「关于怪物分布判定」现算：有 BOSS → BOSS 表，否则有 MON → 小怪表，否则异常。"""
    sp, ss = _s_pos_size(s=s, s_pos=s_pos, s_size=s_size)
    if int(boss_n) > 0 or boss_xy:
        rel = _rel_of(pos, tuple(boss_xy))
        multi = len(boss_xy) > 1
        key, extra, room_kind, _rp = _match_dist(
            table,
            extra,
            "boss",
            len(boss_xy) or int(boss_n),
            rel,
            sp,
            ss,
            multi_boss=multi,
        )
        return key, "boss", extra, room_kind
    if int(mon_n) > 0 or mon_xy:
        pts = tuple(mon_xy)
        rel = _rel_of(pos, pts)
        n = int(mon_n) or len(pts)
        bw, bh = bbox_wh_rel(pos, pts)
        key, extra, room_kind, _rp = _match_dist(
            table,
            extra,
            "mob",
            n,
            rel,
            sp,
            ss,
            bbox_w=bw,
            bbox_h=bh,
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


MON_CORR_MAX = 80  # loot/gate 共用上限
XY_CORR_MAX = MON_CORR_MAX


def corr_udlr_from_dict(data: dict | None, prefix: str) -> tuple[int, int, int, int]:
    """json → (上, 下, 左, 右)。prefix 如 mon_corr / loot_corr / gate_corr。"""
    d = data if isinstance(data, dict) else {}

    def _n(suffix: str) -> int:
        try:
            return max(0, min(XY_CORR_MAX, int(d.get(f"{prefix}_{suffix}", 0) or 0)))
        except (TypeError, ValueError):
            return 0

    return _n("u"), _n("d"), _n("l"), _n("r")


def mon_corr_from_dict(data: dict | None) -> tuple[int, int, int, int]:
    return corr_udlr_from_dict(data, "mon_corr")


def loot_corr_from_dict(data: dict | None) -> tuple[int, int, int, int]:
    return corr_udlr_from_dict(data, "loot_corr")


def gate_corr_from_dict(data: dict | None) -> tuple[int, int, int, int]:
    return corr_udlr_from_dict(data, "gate_corr")


def mon_off_from_corr(u: int, dwn: int, left: int, right: int) -> tuple[int, int]:
    """滑块 → (dx, dy)。上/左为负。loot/gate 同用。"""
    return int(right) - int(left), int(dwn) - int(u)


def mon_off_from_dict(data: dict | None) -> tuple[int, int]:
    return mon_off_from_corr(*mon_corr_from_dict(data))


def loot_off_from_dict(data: dict | None) -> tuple[int, int]:
    return mon_off_from_corr(*loot_corr_from_dict(data))


def gate_off_from_dict(data: dict | None) -> tuple[int, int]:
    return mon_off_from_corr(*gate_corr_from_dict(data))


def apply_mon_boss_corr(data: dict | None, ox: int = 0, oy: int = 0) -> dict:
    """读时平移 mon_xy / boss_xy。浅拷贝，不改入参、不写 jsonl。"""
    src = dict(data or {})
    dx, dy = int(ox), int(oy)
    if not dx and not dy:
        return src
    src["mon_xy"] = [list(p) for p in _shift_xy(_parse_xy_list(src.get("mon_xy")), dx, dy)]
    src["boss_xy"] = [list(p) for p in _shift_xy(_parse_xy_list(src.get("boss_xy")), dx, dy)]
    return src


def apply_loot_corr(data: dict | None, ox: int = 0, oy: int = 0) -> dict:
    """读时平移 loot_xy。浅拷贝，不写 jsonl。"""
    src = dict(data or {})
    dx, dy = int(ox), int(oy)
    if not dx and not dy:
        return src
    src["loot_xy"] = [list(p) for p in _shift_xy(_parse_xy_list(src.get("loot_xy")), dx, dy)]
    return src


def apply_gate_corr(data: dict | None, ox: int = 0, oy: int = 0) -> dict:
    """读时平移 gate_xy（点列表，或单点 list/tuple）。浅拷贝，不写 jsonl。"""
    src = dict(data or {})
    dx, dy = int(ox), int(oy)
    if not dx and not dy:
        return src
    raw = src.get("gate_xy")
    # [[x,y], ...] 或 ((x,y), ...)
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        pts = _parse_xy_list(raw)
        src["gate_xy"] = [list(p) for p in _shift_xy(pts, dx, dy)]
        return src
    one = _parse_xy(raw)
    if one is not None:
        shifted = (one[0] + dx, one[1] + dy)
        src["gate_xy"] = list(shifted) if isinstance(raw, list) else shifted
    return src


def apply_detect_xy_corr(
    data: dict | None,
    *,
    mon_ox: int = 0,
    mon_oy: int = 0,
    loot_ox: int = 0,
    loot_oy: int = 0,
    gate_ox: int = 0,
    gate_oy: int = 0,
) -> dict:
    """一次套上 mon/boss、loot、gate 补正。"""
    out = apply_mon_boss_corr(data, mon_ox, mon_oy)
    out = apply_loot_corr(out, loot_ox, loot_oy)
    out = apply_gate_corr(out, gate_ox, gate_oy)
    return out


def stuck_x_s_floor(params: FsmParams) -> float:
    """卡住 x_s 建议下限（秒）：max(XXX, AX, AY, PW, 恢复序列 HOLD) / 1000。技能表最长持续由宿主另算。"""
    rec_ms = 0
    base = max(0.0, float(params.stuck_recover_s))
    for st in stuck_recover_seq(params):
        rec_ms += stuck_recover_step_ms(st, base)
    ms = max(
        int(params.xxx_ms),
        int(params.ax_ms),
        int(params.ay_ms),
        int(params.pw_ms),
        rec_ms,
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
    dt_ms: int,
    recover_seq: tuple[StuckRecoverStep, ...] = (),
    recover_s: float = DEFAULT_STUCK_RECOVER_S,
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
    step_ms = max(0, int(dt_ms))
    seq = tuple(recover_seq) or DEFAULT_STUCK_RECOVER
    base_s = max(0.0, float(recover_s))

    if stuck:
        extra = (
            ctx.extra_h,
            ctx.extra_v,
            ctx.adv_phase,
            ctx.adv_lock,
        )
        tail = (ctx.through_done, ctx.adv_dir_set)

        def _pack_rec(action: FsmAction, dirs: tuple[FsmDir, ...], on: bool, i: int, k: int):
            return (
                action,
                dirs[0] if dirs else None,
                dirs,
                dirs,
                *extra,
                on,
                i,
                k,
                *tail,
            )

        i = int(ctx.recover_dir_i) if ctx.recover_on else 0
        k = int(ctx.recover_ms) if ctx.recover_on else 0
        i = max(0, i)
        n = len(seq)
        while True:
            if n <= 0 or i >= n:
                return _pack_rec(FsmAction.NONE, (), False, 0, 0)
            st = seq[i]
            d = _DIR_BY_KEY.get(str(st.key))
            dirs = (d,) if d is not None else ()
            if st.action == "tap":
                if k > 0:
                    i += 1
                    k = 0
                    continue
                return _pack_rec(FsmAction.TAP, dirs, True, i, 1)
            dur = stuck_recover_step_ms(st, base_s)
            if dur <= 0:
                i += 1
                k = 0
                continue
            k = k + step_ms
            if k >= dur:
                i += 1
                k = 0
                continue
            return _pack_rec(FsmAction.HOLD, dirs, True, i, k)

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


def _copy_fight_skill(sk: FightSkill, *, charge: int | None = None, multi: int | None = None) -> FightSkill:
    return FightSkill(
        slot=sk.slot,
        key=sk.key,
        cooldown_s=sk.cooldown_s,
        range_px=sk.range_px,
        range_x=sk.range_x,
        range_y=sk.range_y,
        combo=sk.combo,
        hold_ms=sk.hold_ms,
        multi=sk.multi if multi is None else multi,
        charge=sk.charge if charge is None else charge,
        mash=bool(sk.mash),
        group_key=sk.group_key,
        group_slots=sk.group_slots,
        group_keys=sk.group_keys,
        group_holds=sk.group_holds,
        gaps_ms=sk.gaps_ms,
    )


def expand_fight_skills(skills: tuple[FightSkill, ...]) -> tuple[FightSkill, ...]:
    """多次释放：拆成 MULTI 份相同技能，charge 分开计 CD。"""
    out: list[FightSkill] = []
    for sk in skills:
        n = max(1, int(sk.multi or 1))
        if n <= 1:
            out.append(_copy_fight_skill(sk, charge=0, multi=1))
            continue
        for i in range(n):
            out.append(_copy_fight_skill(sk, charge=i, multi=n))
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


def _slot_ready(slot: int, ready: dict[tuple[int, int], int], t: int) -> bool:
    hits = [until for (s, _c), until in ready.items() if s == slot]
    if not hits:
        return True
    return any(t >= until for until in hits)


def _first_file_ready(
    seq: tuple[FightSkill, ...],
    ready: dict[tuple[int, int], int],
    t: int,
    skip: tuple[tuple[int, int], ...] = (),
) -> FightSkill | None:
    """排除 CD 后，过图文件序列里第 1 个就绪项。技能组 = 组内所有技能均就绪。skip 不改优先级，只去掉走近超时项。"""
    skip_set = set(skip or ())
    for sk in seq:
        if (int(sk.slot), int(sk.charge)) in skip_set:
            continue
        slots = sk.group_slots
        if slots:
            if all(_slot_ready(int(sl), ready, t) for sl in slots):
                return sk
            continue
        until = ready.get((sk.slot, sk.charge))
        if until is None or t >= until:
            return sk
    return None


def _group_members(sk: FightSkill) -> tuple[tuple[int, str, int], ...]:
    slots = sk.group_slots or (sk.slot,)
    n = len(slots)
    keys = sk.group_keys or ((sk.key,) * n)
    holds = sk.group_holds or ((sk.hold_ms,) * n)
    out: list[tuple[int, str, int]] = []
    for i, slot in enumerate(slots):
        key = keys[i] if i < len(keys) else sk.key
        hold = holds[i] if i < len(holds) else sk.hold_ms
        out.append((int(slot), str(key or sk.key), max(0, int(hold))))
    return tuple(out)


def _group_tail_ms(holds: tuple[int, ...], gaps: tuple[int, ...]) -> int:
    hs = [max(0, int(x)) for x in (holds or (0,))]
    gs = [max(0, int(x)) for x in (gaps or ())]
    times = [0]
    t = 0
    for g in gs:
        t += g
        times.append(t)
    while len(hs) < len(times):
        hs.append(hs[-1] if hs else 0)
    end = max(times[i] + hs[i] for i in range(len(times)))
    return max(0, end - times[-1])


def _in_skill_range(
    pos: tuple[float, float] | None,
    enemy_xy: tuple[tuple[float, float], ...],
    rng: float | None,
    range_x: float | None = None,
    range_y: float | None = None,
) -> tuple[bool, float | None]:
    """优先 XY 范围框：最远敌对须 |dx|<=range_x 且 |dy|<=range_y；否则退回圆形 range_px。"""
    dist = _farthest_dist(pos, enemy_xy)
    if pos is None or not enemy_xy:
        return True, dist
    rx = range_x
    ry = range_y
    if rx is not None and ry is not None:
        try:
            rx_f, ry_f = float(rx), float(ry)
        except (TypeError, ValueError):
            rx_f = ry_f = None  # type: ignore
        else:
            far = max(enemy_xy, key=lambda p: (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2)
            ok = abs(far[0] - pos[0]) <= rx_f + 1e-9 and abs(far[1] - pos[1]) <= ry_f + 1e-9
            return ok, dist
    if rng is None or dist is None:
        return True, dist
    return dist <= float(rng) + 1e-9, dist


def _dist_center(
    pos: tuple[float, float] | None,
    dist_key: str,
    table: tuple[DistSig, ...],
    extra: tuple[DistSig, ...],
    enemy_xy: tuple[tuple[float, float], ...],
) -> tuple[float, float] | None:
    for e in extra + table:
        if e.key == dist_key:
            if pos is None:
                return None
            return (pos[0] + float(e.rx), pos[1] + float(e.ry))
    return _pack_center(enemy_xy)


def _cast_tgt(
    pos: tuple[float, float] | None,
    dist_key: str,
    table: tuple[DistSig, ...],
    extra: tuple[DistSig, ...],
    enemy_xy: tuple[tuple[float, float], ...],
) -> tuple[float, float] | None:
    """CAST 朝向目标：与范围异常走近同一套 _dist_center；缺则最远怪 / 敌对中心。"""
    tgt = _dist_center(pos, dist_key, table, extra, enemy_xy)
    if tgt is not None:
        return tgt
    if pos is not None and enemy_xy:
        return max(enemy_xy, key=lambda p: (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2)
    return _pack_center(enemy_xy)


def _cast_face_dir(
    pos: tuple[float, float] | None, tgt: tuple[float, float] | None
) -> FsmDir | None:
    """仅左右：dx>=0→RIGHT 否则 LEFT。无迟滞。pos 或目标缺则 None。"""
    if pos is None or tgt is None:
        return None
    return FsmDir.RIGHT if (tgt[0] - pos[0]) >= 0 else FsmDir.LEFT


def _start_attack(xxx_ms: int, step_ms: int) -> tuple:
    n = max(1, int(xxx_ms))
    return (
        FsmAction.ATTACK,
        None,
        None,
        None,
        "x",
        None,
        None,
        (),
        "普攻X",
        0,
        max(0, n - max(0, int(step_ms))),
        (),
        (),
        0,
    )


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
    dist_table: tuple[DistSig, ...] = (),
    extra_sigs: tuple[DistSig, ...] = (),
    fight_rest: tuple[tuple[int, str, int], ...] = (),
    fight_gaps: tuple[int, ...] = (),
    fight_tail: int = 0,
    approach_skip: tuple[tuple[int, int], ...] = (),
) -> tuple:
    """action, move_dir, fight_dir, slot, key, range, dist, new_cd, note, wait, atk, rest, gaps, tail。"""
    wait = max(0, int(cast_wait_left))
    atk = max(0, int(attack_left))
    step_ms = max(0, int(dt_ms))
    rest = tuple(fight_rest or ())
    gaps = tuple(int(x) for x in (fight_gaps or ()))
    tail = max(0, int(fight_tail))
    empty_q = ((), (), 0)

    def _idle_wait(note: str, left: int):
        return (
            FsmAction.NONE,
            None,
            None,
            last_slot,
            last_key,
            None,
            None,
            skill_cd,
            note,
            max(0, left),
            0,
            rest,
            gaps,
            tail,
        )

    if wait > 0:
        return _idle_wait("等待释放", wait - step_ms)
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
            *empty_q,
        )
    if rest:
        slot, key, _hold = rest[0]
        ready = _cd_map(skill_cd)
        cd_s = 0.0
        charge = 0
        for sk in _skills_for_dist(plan, dist_key) + hotbar:
            if sk.slot == int(slot):
                cd_s = max(0.0, float(sk.cooldown_s))
                charge = int(sk.charge)
                break
        ready[(int(slot), charge)] = t + int(round(cd_s * 1e9))
        new_rest = rest[1:]
        new_gaps = gaps[1:] if len(gaps) > 1 else ()
        new_wait = int(new_gaps[0]) if new_rest else tail
        face = _cast_face_dir(pos, _cast_tgt(pos, dist_key, dist_table, extra_sigs, enemy_xy))
        return (
            FsmAction.CAST,
            None,
            face,
            int(slot),
            key,
            None,
            _farthest_dist(pos, enemy_xy),
            _cd_tuple(ready),
            "",
            max(0, new_wait),
            0,
            new_rest,
            new_gaps,
            tail if new_rest else 0,
        )

    file_seq = _skills_for_dist(plan, dist_key)
    ready = _cd_map(skill_cd)
    skip = tuple(approach_skip or ())
    file_first = _first_file_ready(file_seq, ready, t, skip)
    bar_ready = [
        sk for sk in _ready_skills(hotbar, ready, t) if (int(sk.slot), int(sk.charge)) not in set(skip)
    ]
    picked = file_first
    note_prefix = ""
    if picked is None and bar_ready:
        picked = min(bar_ready, key=lambda sk: (max(0, int(sk.hold_ms)), sk.slot, sk.charge))
        note_prefix = "无技能可放"
    if picked is not None:
        in_range, dist = _in_skill_range(
            pos, enemy_xy, picked.range_px, picked.range_x, picked.range_y
        )
        if not in_range:
            tgt = _dist_center(pos, dist_key, dist_table, extra_sigs, enemy_xy)
            md = _axis_dir(pos, tgt) if pos is not None and tgt is not None else None
            return (
                FsmAction.HOLD if md is not None else FsmAction.NONE,
                md,
                None,
                picked.slot,
                picked.key,
                picked.range_px,
                dist,
                skill_cd,
                "范围异常" if not note_prefix else f"{note_prefix} 范围异常",
                0,
                0,
                *empty_q,
            )
        members = _group_members(picked)
        first_slot, first_key, _first_hold = members[0]
        rest_m = members[1:]
        gaps_m = tuple(max(0, int(x)) for x in (picked.gaps_ms or ()))
        if len(gaps_m) < len(rest_m):
            gaps_m = gaps_m + (0,) * (len(rest_m) - len(gaps_m))
        elif len(gaps_m) > len(rest_m):
            gaps_m = gaps_m[: len(rest_m)]
        holds = tuple(h for _s, _k, h in members)
        tail_m = _group_tail_ms(holds, gaps_m)
        cd_s = max(0.0, float(picked.cooldown_s))
        ready[(picked.slot, picked.charge)] = t + int(round(cd_s * 1e9))
        wait_m = int(gaps_m[0]) if rest_m else tail_m
        face = _cast_face_dir(pos, _cast_tgt(pos, dist_key, dist_table, extra_sigs, enemy_xy))
        return (
            FsmAction.CAST,
            None,
            face,
            first_slot,
            first_key,
            picked.range_px,
            dist,
            _cd_tuple(ready),
            note_prefix,
            max(0, wait_m),
            0,
            rest_m,
            gaps_m,
            tail_m if rest_m else 0,
        )
    act, md, fd, slot, key, rng, dist, cd, note, cw, aw, r, g, tl = _start_attack(xxx_ms, step_ms)
    return act, md, fd, slot, key, rng, dist, skill_cd, note, cw, aw, r, g, tl


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
    if action is FsmAction.TAP and skill_key and not dirs:
        act = f"{action.value} {skill_key}"
    if action is FsmAction.HOLD and skill_key and not dirs:
        act = f"{action.value} {skill_key}"
    if FsmFlag.STUCK in flags or state is FsmState.STUCK:
        if action is FsmAction.NONE:
            return "卡住"
        return f"卡住 {recover}{act}".replace("  ", " ").strip()
    if state is FsmState.FIGHT:
        sk = f"{skill_slot} {skill_key or ''}".strip()
        head = "提取分布+选技能" if fsm_run <= 1 else "选技能"
        if fight_note == "等待释放":
            left = max(1, int(cast_wait_left))
            return f"{head} 等待释放 {sk} 剩{left}ms".replace("  ", " ").strip()
        if fight_note == "范围异常" or fight_note.endswith("范围异常"):
            if action is FsmAction.CAST:
                return f"{head} 范围异常 仍然释放 {sk}".strip()
            if action in (FsmAction.HOLD, FsmAction.TAP):
                return f"{head} 范围异常 走近 {sk}".strip()
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
    recover_keys: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], int]:
    if FsmFlag.STUCK in flags or state is FsmState.STUCK:
        keys = recover_keys or ("esc", "right", "left", "up", "down")
        hit = min(max(0, int(recover_dir_i)), max(0, len(keys) - 1)) + 1
        return ("卡住",) + tuple(keys), hit
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
    if prev is FsmState.FIGHT and (
        ctx.cast_wait_left > 0 or ctx.attack_left > 0 or ctx.fight_rest
    ):
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
    stuck_recover_s = max(0.0, float(params.stuck_recover_s))
    recover_seq = stuck_recover_seq(params)
    s_pos, s_size = _s_pos_size(params)
    pm = max(0, int(params.pm))
    pc = max(0, int(params.pc))
    pt = max(1, int(params.pt))
    xxx_ms = max(1, int(params.xxx_ms))
    th_ms = max(0, int(params.th_ms))
    pw_ms = max(0, int(params.pw_ms))
    advance_timeout_ms = max(1, int(params.advance_timeout_ms))
    approach_timeout_ms = max(1, int(params.approach_timeout_ms))
    tn_s = max(0.0, float(params.tn_s))
    step_ms = dt_ms(ctx.last_t_ns, t)
    ox, oy = int(params.mon_off_x), int(params.mon_off_y)
    lox, loy = int(params.loot_off_x), int(params.loot_off_y)
    gox, goy = int(params.gate_off_x), int(params.gate_off_y)
    if ox or oy or lox or loy or gox or goy:
        corr = apply_detect_xy_corr(
            {
                "mon_xy": snap.mon_xy,
                "boss_xy": snap.boss_xy,
                "loot_xy": snap.loot_xy,
                "gate_xy": snap.gate_xy,
            },
            mon_ox=ox,
            mon_oy=oy,
            loot_ox=lox,
            loot_oy=loy,
            gate_ox=gox,
            gate_oy=goy,
        )
        g_raw = corr.get("gate_xy")
        if isinstance(g_raw, (list, tuple)) and g_raw and isinstance(g_raw[0], (list, tuple)):
            g_pts = _parse_xy_list(g_raw)
            g_one = g_pts[0] if g_pts else None
        else:
            g_one = _parse_xy(g_raw)
        snap = replace(
            snap,
            mon_xy=_parse_xy_list(corr.get("mon_xy")),
            boss_xy=_parse_xy_list(corr.get("boss_xy")),
            loot_xy=_parse_xy_list(corr.get("loot_xy")),
            gate_xy=g_one,
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
    adv_timeout = False
    if (
        ctx.state is FsmState.ADVANCE
        and ctx.adv_start_t is not None
        and (t - ctx.adv_start_t) / 1e6 >= advance_timeout_ms
    ):
        adv_timeout = True
    appr_timeout = False
    if (
        ctx.state is FsmState.FIGHT
        and ctx.approach_start_t is not None
        and (t - ctx.approach_start_t) / 1e6 >= approach_timeout_ms
    ):
        appr_timeout = True
    method_timeout = adv_timeout or appr_timeout
    hold, hold_why = (None, "")
    if ctx.state is not FsmState.STUCK and not method_timeout:
        hold, hold_why = _method_hold(ctx, drafted, loot.judged)
    flow = hold if hold is not None else drafted
    if hold is not None:
        why += f"  {hold_why}"
    if adv_timeout:
        why += "  前进超时"
    if appr_timeout:
        why += "  走近超时"
    if ctx.state is FsmState.STUCK and ctx.recover_on:
        # 恢复进行中：卡住本身不计入连续计时；保持 STUCK 直到序列一轮结束
        watch_start_t = ctx.watch_start_t if ctx.watch_start_t is not None else t
        watch_run = ctx.watch_run
        state = FsmState.STUCK
    else:
        watch_same = ctx.watch_state is flow and ctx.watch_start_t is not None and not method_timeout
        if watch_same:
            watch_start_t = ctx.watch_start_t
            watch_run = ctx.watch_run + 1
        else:
            watch_start_t = t
            watch_run = 1
        watch_s = (t - watch_start_t) / 1e9
        if watch_s >= x_s:
            flow = drafted
            if ctx.watch_state is flow and ctx.watch_start_t is not None and not method_timeout:
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
                s_pos,
                s_size,
                multi_boss=multi,
            )
        elif mon.judged or snap.mon_xy:
            rel = _rel_of(pos, tuple(snap.mon_xy))
            dist_kind = "mob"
            bw, bh = bbox_wh_rel(pos, tuple(snap.mon_xy))
            dist_key, extra_sigs, room_kind, _reset_pt = _match_dist(
                params.dist_table,
                extra_sigs,
                "mob",
                int(snap.mon),
                rel,
                s_pos,
                s_size,
                bbox_w=bw,
                bbox_h=bh,
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
    if adv_timeout and state is FsmState.ADVANCE:
        fsm_run = 1
        run_start_t = t
    if appr_timeout and state is FsmState.FIGHT:
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
    fight_rest: tuple[tuple[int, str, int], ...] = ()
    fight_gaps: tuple[int, ...] = ()
    fight_tail = 0
    fight_skip: tuple[tuple[int, int], ...] = ()
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
            prev_state=None if adv_timeout else ctx.state,
            ctx=ctx,
            gx=gx,
            gy=gy,
            ax_ms=ax_ms,
            ay_ms=ay_ms,
            dt_ms=step_ms,
            recover_seq=recover_seq,
            recover_s=stuck_recover_s,
        )
        if stuck and recover_on and 0 <= recover_dir_i < len(recover_seq):
            st = recover_seq[recover_dir_i]
            if action is FsmAction.TAP or (action is FsmAction.HOLD and not move_dirs):
                skill_key = st.key
        if ctx.state is FsmState.STUCK and ctx.recover_on and not recover_on:
            # 配置序列一轮完成 → 退出卡住，重置监测，避免立刻再进
            state = flow
            why += "  卡住恢复结束"
            watch_start_t = t
            watch_run = 1
            fsm_run = 1
            run_start_t = t
            flags = frozenset(f for f in flags if f is not FsmFlag.STUCK)
            stuck = False
            action = FsmAction.NONE
            move_dir = None
            move_dirs = ()
    elif state is FsmState.FIGHT:
        enemy_xy = tuple(snap.boss_xy) + tuple(snap.mon_xy)
        fight_skip = ctx.approach_skip if ctx.state is FsmState.FIGHT else ()
        if appr_timeout and ctx.cast_slot is not None:
            item = (int(ctx.cast_slot), int(ctx.approach_charge))
            if item not in fight_skip:
                fight_skip = fight_skip + (item,)
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
            fight_rest,
            fight_gaps,
            fight_tail,
        ) = _fight_intent(
            t=t,
            pos=pos,
            enemy_xy=enemy_xy,
            dist_key=dist_key,
            plan=params.fight_plan,
            hotbar=params.hotbar,
            skill_cd=skill_cd,
            cast_wait_left=0 if appr_timeout else (ctx.cast_wait_left if ctx.state is FsmState.FIGHT else 0),
            attack_left=0 if appr_timeout else (ctx.attack_left if ctx.state is FsmState.FIGHT else 0),
            xxx_ms=xxx_ms,
            dt_ms=step_ms,
            last_slot=ctx.cast_slot if ctx.state is FsmState.FIGHT else None,
            last_key=ctx.cast_key if ctx.state is FsmState.FIGHT else None,
            dist_table=params.dist_table,
            extra_sigs=extra_sigs,
            fight_rest=() if appr_timeout else (ctx.fight_rest if ctx.state is FsmState.FIGHT else ()),
            fight_gaps=() if appr_timeout else (ctx.fight_gaps if ctx.state is FsmState.FIGHT else ()),
            fight_tail=0 if appr_timeout else (ctx.fight_tail if ctx.state is FsmState.FIGHT else 0),
            approach_skip=fight_skip,
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
            or state is FsmState.FIGHT
        )
    )
    if approaching:
        want = _norm_dirs(move_dirs or ((move_dir,) if move_dir else ()))
        same_move = (not method_timeout) and ctx.state in (
            FsmState.ADVANCE,
            FsmState.LOOT,
            FsmState.FIGHT,
        )
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
        recover_keys=tuple(st.key for st in recover_seq),
    )

    is_appr = state is FsmState.FIGHT and str(fight_note or "").endswith("范围异常")
    approach_charge = 0
    approach_start_t = None
    if is_appr:
        skip_set = set(fight_skip)
        if skill_slot is not None:
            for sk in _skills_for_dist(params.fight_plan, dist_key) + params.hotbar:
                if int(sk.slot) == int(skill_slot) and (int(sk.slot), int(sk.charge)) not in skip_set:
                    approach_charge = int(sk.charge)
                    break
        same_appr = (
            not appr_timeout
            and ctx.approach_start_t is not None
            and ctx.cast_slot == skill_slot
            and int(ctx.approach_charge) == int(approach_charge)
        )
        approach_start_t = ctx.approach_start_t if same_appr else t

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
        recover_key=(
            recover_seq[recover_dir_i].key
            if recover_on and 0 <= recover_dir_i < len(recover_seq)
            else ""
        ),
        through_done=through_done if state is FsmState.ADVANCE else False,
        adv_start_t=(
            t
            if state is FsmState.ADVANCE and (ctx.state is not FsmState.ADVANCE or adv_timeout or ctx.adv_start_t is None)
            else (ctx.adv_start_t if state is FsmState.ADVANCE else None)
        ),
        approach_start_t=approach_start_t if state is FsmState.FIGHT else None,
        approach_charge=approach_charge if state is FsmState.FIGHT and is_appr else 0,
        approach_skip=fight_skip if state is FsmState.FIGHT else (),
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
        fight_rest=fight_rest if state is FsmState.FIGHT else (),
        fight_gaps=fight_gaps if state is FsmState.FIGHT else (),
        fight_tail=fight_tail if state is FsmState.FIGHT else 0,
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
        fight_dir=fight_dir,
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
            fight_dir=fight_dir,
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
