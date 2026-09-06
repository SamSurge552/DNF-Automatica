"""FSM 意图 → 按键。核心只出 TAP/HOLD/CAST/PICK；这里查名注入。"""
from __future__ import annotations

import random
import time

from fsm_core import FsmAction, FsmDecision, FsmDir
from key_inject import key_down, key_tap, key_up, vk_of

PICK_KEY = "alt_l"
DEFAULT_TAP_MS_MIN = 80
DEFAULT_TAP_MS_MAX = 120
DEFAULT_TAP_MS = DEFAULT_TAP_MS_MIN  # 旧单值入口；点按实际抽 [min,max]
TAP_MS_MIN = 1
TAP_MS_MAX = 200
PICK_THROTTLE_MULT = 5  # 节流 = tap_ms_max × 5，避免连发比最长点按还密
DEFAULT_MASH_COUNT = 3
MASH_COUNT_MIN = 1
MASH_COUNT_MAX = 15
DEFAULT_MASH_GAP_MS = 50
MASH_GAP_MIN = 10
MASH_GAP_MAX = 300


def clamp_tap_ms(raw, default: int = DEFAULT_TAP_MS) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    return max(TAP_MS_MIN, min(TAP_MS_MAX, n))


def parse_tap_ms_range(data: dict | None) -> tuple[int, int]:
    """优先 tap_ms_min/max。旧 tap_ms 忽略（不再当 50ms 点按）。都没有则 80–120。固定时长请写 min=max。"""
    data = data if isinstance(data, dict) else {}
    if "tap_ms_min" not in data and "tap_ms_max" not in data:
        return DEFAULT_TAP_MS_MIN, DEFAULT_TAP_MS_MAX
    lo = clamp_tap_ms(data.get("tap_ms_min"), DEFAULT_TAP_MS_MIN)
    hi = clamp_tap_ms(data.get("tap_ms_max"), DEFAULT_TAP_MS_MAX)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def clamp_mash_count(raw, default: int = DEFAULT_MASH_COUNT) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    return max(MASH_COUNT_MIN, min(MASH_COUNT_MAX, n))


def clamp_mash_gap_ms(raw, default: int = DEFAULT_MASH_GAP_MS) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    return max(MASH_GAP_MIN, min(MASH_GAP_MAX, n))


_DIRS = {FsmDir.LEFT.value, FsmDir.RIGHT.value, FsmDir.UP.value, FsmDir.DOWN.value}


def parse_command(cmd: str) -> list[list[str]]:
    """`down right+z` → [['down','right'],['z']]；`+` 是先后，空格是同时。"""
    raw = str(cmd or "").strip().lower()
    if not raw:
        return []
    steps = []
    for part in raw.split("+"):
        keys = [t for t in part.strip().split() if t]
        if keys:
            steps.append(keys)
    return steps


def mash_n_for_slot(slot, mash_slots, mash_count: int) -> int:
    """和 apply() 同一套：槽在连按表里则返回 COUNT，否则 0。"""
    if slot is None:
        return 0
    try:
        sid = int(slot)
    except (TypeError, ValueError):
        return 0
    slots = {int(s) for s in (mash_slots or ()) if int(s) > 0}
    if sid not in slots:
        return 0
    return clamp_mash_count(mash_count)


class FsmExecutor:
    def __init__(self, log=None, tap_ms: int | None = None, tap_ms_min: int | None = None, tap_ms_max: int | None = None):
        self.log = log
        self._on = True
        self._held: set[str] = set()
        self._bad: set[str] = set()
        self._last_pick_t = 0.0
        self._mash_slots: set[int] = set()
        if tap_ms_min is None and tap_ms_max is None and tap_ms is not None:
            self.set_tap_ms(tap_ms)
        else:
            self.set_tap_range(
                DEFAULT_TAP_MS_MIN if tap_ms_min is None else tap_ms_min,
                DEFAULT_TAP_MS_MAX if tap_ms_max is None else tap_ms_max,
            )
        self.set_mash(DEFAULT_MASH_COUNT, DEFAULT_MASH_GAP_MS)

    def set_tap_ms(self, tap_ms: int) -> None:
        n = clamp_tap_ms(tap_ms)
        self.set_tap_range(n, n)

    def set_tap_range(self, lo, hi) -> None:
        a = clamp_tap_ms(lo, DEFAULT_TAP_MS_MIN)
        b = clamp_tap_ms(hi, DEFAULT_TAP_MS_MAX)
        if a > b:
            a, b = b, a
        self.tap_ms_min = a
        self.tap_ms_max = b
        self.tap_ms = (a + b) // 2

    def _tap_hold_s(self) -> float:
        return random.randint(self.tap_ms_min, self.tap_ms_max) / 1000.0

    def _tap_down_up(self, name: str) -> None:
        if vk_of(name) is None:
            self._warn(name)
            return
        if name in self._held:
            key_up(name)
            self._held.discard(name)
        key_tap(name, random.randint(self.tap_ms_min, self.tap_ms_max))

    def set_mash(self, count: int, gap_ms: int, slots=None) -> None:
        self.mash_count = clamp_mash_count(count)
        self.mash_gap_ms = clamp_mash_gap_ms(gap_ms)
        self._mash_gap_s = self.mash_gap_ms / 1000.0
        if slots is not None:
            self._mash_slots = {int(s) for s in slots if int(s) > 0}

    def set_mash_slots(self, slots) -> None:
        self._mash_slots = {int(s) for s in (slots or ()) if int(s) > 0}

    def apply(self, decision: FsmDecision) -> None:
        if not self._on or decision is None:
            return
        action = decision.action
        dirs = tuple(decision.move_dirs or ())
        if not dirs and decision.move_dir is not None:
            dirs = (decision.move_dir,)
        names = {d.value for d in dirs if d.value in _DIRS}
        if action is FsmAction.HOLD and names:
            self._hold_only(names)
            return
        if action is FsmAction.TAP and names:
            self._hold_only(set())
            for name in sorted(names):
                self._tap(name)
            return
        if action is FsmAction.CAST:
            self._hold_only(set())
            slot = decision.skill_slot
            mash = slot is not None and int(slot) in self._mash_slots
            self._cast(str(decision.skill_key or ""), mash=mash)
            return
        if action is FsmAction.ATTACK:
            self._hold_only({"x"})
            return
        if action is FsmAction.PICK:
            self._hold_only(set())
            self._pick()
            return
        self._hold_only(set())

    def stop(self) -> None:
        self._on = False
        self._hold_only(set())

    def _hold_only(self, want: set[str]) -> None:
        for name in list(self._held - want):
            if key_up(name):
                self._held.discard(name)
        for name in want - self._held:
            if self._press(name):
                self._held.add(name)

    def _press(self, name: str) -> bool:
        if vk_of(name) is None:
            self._warn(name)
            return False
        return key_down(name)

    def _cast_tap(self, name: str) -> None:
        self._tap_down_up(name)

    def _pick(self) -> None:
        now = time.monotonic()
        gap = (self.tap_ms_max / 1000.0) * PICK_THROTTLE_MULT
        if now - self._last_pick_t < gap:
            return
        self._last_pick_t = now
        self._tap_down_up(PICK_KEY)

    def _tap(self, name: str) -> None:
        self._tap_down_up(name)

    def _cast(self, cmd: str, mash: bool = False) -> None:
        n = self.mash_count if mash else 1
        n = max(1, int(n))
        for i in range(n):
            self._cast_once(cmd)
            if i + 1 < n:
                time.sleep(self._mash_gap_s)

    def _cast_once(self, cmd: str) -> None:
        steps = parse_command(cmd)
        if not steps:
            self._warn(cmd or "(空技能键)")
            return
        held: list[str] = []
        try:
            for i, keys in enumerate(steps):
                last = i == len(steps) - 1
                if last:
                    if held:
                        time.sleep(self._tap_hold_s())
                    for name in keys:
                        self._cast_tap(name)
                else:
                    for name in keys:
                        if name not in self._held and name not in held:
                            if self._press(name):
                                held.append(name)
        finally:
            for name in reversed(held):
                key_up(name)

    def _warn(self, name: str) -> None:
        key = str(name or "").strip().lower() or "?"
        if key in self._bad:
            return
        self._bad.add(key)
        if self.log:
            self.log(f"发键: 不认识的键 `{name}`，已跳过")
