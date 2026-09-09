"""FSM测试 states.jsonl：只写不读。不进 step，不改决策。

一条记录 = 一段刚结束的 FSM state。t_ns 只用帧戳（与 frames.jsonl 对齐）。
"""
from __future__ import annotations

from typing import Any

from fsm_core import FsmAction, FsmContext, FsmDecision, FsmState


def _val(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "value") and not isinstance(x, (str, bytes, int, float, bool)):
        return x.value
    if isinstance(x, tuple):
        return [_val(i) for i in x]
    if isinstance(x, frozenset):
        return sorted(_val(i) for i in x)
    return x


def _mon_boss_n(features: dict | None) -> int:
    feats = features or {}
    try:
        mon = max(0, int(feats.get("mon") or 0))
    except (TypeError, ValueError):
        mon = 0
    try:
        boss = max(0, int(feats.get("boss") or 0))
    except (TypeError, ValueError):
        boss = 0
    return mon + boss


def context_fields(state: FsmState, decision: FsmDecision, ctx: FsmContext, features: dict | None = None) -> dict:
    """按状态带方法层上下文（开打/前进/捡物/卡住）；公共计数各状态都有。"""
    feats = features or {}
    out: dict[str, Any] = {
        "rooms": int(decision.rooms),
        "gate_crosses": int(decision.gate_crosses),
        "boss_kills": int(decision.boss_kills),
        "why": str(decision.why or ""),
        "flags": _val(decision.flags),
        "action": _val(decision.action),
        "flow_steps": list(decision.flow_steps or ()),
        "flow_hit": int(decision.flow_hit),
        "intent_label": str(decision.intent_label or ""),
    }
    if state is FsmState.FIGHT:
        out.update(
            {
                "dist_key": decision.dist_key,
                "dist_kind": decision.dist_kind,
                "boss_seen": bool(decision.boss_seen),
                "skill_slot": decision.skill_slot,
                "skill_key": decision.skill_key,
                "skill_range": decision.skill_range,
                "pack_dist": decision.pack_dist,
                "cd_reset": bool(decision.cd_reset),
                "skill_cds": [_val(row) for row in (decision.skill_cds or ())],
                "mon": int(feats.get("mon") or 0),
                "boss": int(feats.get("boss") or 0),
            }
        )
    elif state is FsmState.LOOT:
        out.update(
            {
                "loot": int(feats.get("loot") or 0),
                "loot_wait_done": bool(ctx.loot_wait_done),
                "loot_wait_t": ctx.loot_wait_t,
                "loot_targets_n": len(ctx.loot_targets or ()),
            }
        )
    elif state is FsmState.ADVANCE:
        out.update(
            {
                "adv_dirs": _val(ctx.adv_dirs),
                "adv_phase": int(ctx.adv_phase),
                "adv_dir_set": bool(ctx.adv_dir_set),
                "through_done": bool(ctx.through_done),
                "move_dirs": _val(decision.move_dirs),
            }
        )
    elif state is FsmState.STUCK:
        rec_dir = None
        dirs = ("up", "down", "left", "right")
        i = int(ctx.recover_dir_i)
        if 0 <= i < len(dirs):
            rec_dir = dirs[i]
        out.update(
            {
                "recover_on": bool(ctx.recover_on),
                "recover_dir": str(ctx.recover_key or "") or rec_dir,
                "recover_key": str(ctx.recover_key or ""),
                "recover_ms": int(ctx.recover_ms),
                "watch_state": _val(ctx.watch_state),
            }
        )
    elif state is FsmState.WAIT:
        out["saw_dungeon"] = bool(ctx.saw_dungeon)
    elif state is FsmState.IDLE:
        out["dist_key"] = decision.dist_key
    elif state is FsmState.RETURN:
        out["saw_dungeon"] = bool(ctx.saw_dungeon)
    return out


class FsmStateLogger:
    """主循环末尾：state 变了就交出刚结束那一段。回城/停进程 flush 未闭合段。

    额外只写字段（不进决策）：
      mon_boss_enter — 开段帧 mon+boss
      casts — 段内每次 CAST 追加 {t: 相对段起点ms, slot}
    """

    def __init__(self) -> None:
        self.reset()
        self.map_name = ""
        self.char_name = ""

    def reset(self) -> None:
        self._state: str | None = None
        self._enter_t: int | None = None
        self._last_t: int | None = None
        self._n = 0
        self._ctx: dict = {}
        self._mon_boss_enter = 0
        self._casts: list[dict] = []
        self._cast_armed = True  # 边沿：非 CAST 后下一次 CAST 才记

    def set_names(self, map_name: str, char_name: str) -> None:
        self.map_name = str(map_name or "")
        self.char_name = str(char_name or "")

    def observe(
        self,
        t_ns: int,
        decision: FsmDecision,
        ctx: FsmContext,
        features: dict | None = None,
    ) -> dict | None:
        t_ns = int(t_ns)
        name = decision.state.value
        snap = context_fields(decision.state, decision, ctx, features)
        if self._state is None:
            self._open(name, t_ns, snap, features)
            self._note_cast(t_ns, decision)
            return None
        if name == self._state:
            self._last_t = t_ns
            self._n += 1
            self._ctx = snap
            self._note_cast(t_ns, decision)
            return None
        rec = self._close(next_state=name, exit_t=self._last_t)
        self._open(name, t_ns, snap, features)
        self._note_cast(t_ns, decision)
        return rec

    def flush(self, next_state: str | None = None) -> dict | None:
        if self._state is None:
            return None
        rec = self._close(next_state=next_state, exit_t=self._last_t)
        self.reset()
        return rec

    def _open(self, state: str, t_ns: int, snap: dict, features: dict | None) -> None:
        self._state = state
        self._enter_t = t_ns
        self._last_t = t_ns
        self._n = 1
        self._ctx = snap
        self._mon_boss_enter = _mon_boss_n(features)
        self._casts = []
        self._cast_armed = True

    def _note_cast(self, t_ns: int, decision: FsmDecision) -> None:
        """每次进入 CAST 动作记一条（连续多帧同一 CAST 只记首帧）。"""
        is_cast = decision.action is FsmAction.CAST
        if not is_cast:
            self._cast_armed = True
            return
        if not self._cast_armed:
            return
        self._cast_armed = False
        enter = int(self._enter_t or t_ns)
        t_ms = max(0, int(round((int(t_ns) - enter) / 1e6)))
        slot = decision.skill_slot
        try:
            slot_i = int(slot) if slot is not None else None
        except (TypeError, ValueError):
            slot_i = None
        self._casts.append({"t": t_ms, "slot": slot_i})

    def _close(self, *, next_state: str | None, exit_t: int | None) -> dict:
        enter = int(self._enter_t or 0)
        exit_t = int(exit_t if exit_t is not None else enter)
        dur = max(0, int(round((exit_t - enter) / 1e6)))
        rec = {
            "state": self._state,
            "next_state": next_state,
            "enter_t_ns": enter,
            "exit_t_ns": exit_t,
            "duration_ms": dur,
            "frame_count": int(self._n),
            "map_name": self.map_name,
            "char_name": self.char_name,
            "mon_boss_enter": int(self._mon_boss_enter),
            "casts": list(self._casts),
        }
        rec.update(self._ctx)
        return rec
