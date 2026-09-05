"""后期坐标转换。录制 jsonl 不调用本模块。

磁盘：检测器输出，原点在截屏图左上角（y 向下）。
转换：先翻成录制区域左下角绝对坐标（x 向右、y 向上），再按需填 player、再按需改成相对 player。
"""
from __future__ import annotations

CLASS_XY = ("mon", "loot", "gate", "boss")


def as_xy_list(val) -> list[list[float]]:
    if not isinstance(val, (list, tuple)) or not val:
        return []
    if isinstance(val[0], (int, float)):
        if len(val) >= 2:
            return [[float(val[0]), float(val[1])]]
        return []
    out: list[list[float]] = []
    for p in val:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
    return out


def player_abs(fr: dict) -> list[float] | None:
    pts = as_xy_list(fr.get("player_xy"))
    if not pts:
        return None
    return pts[0]


def to_bl(pt: list[float], height: float | None) -> list[float]:
    x, y = float(pt[0]), float(pt[1])
    if height is None:
        return [round(x, 2), round(y, 2)]
    return [round(x, 2), round(float(height) - y, 2)]


def minus(pts: list[list[float]], origin: list[float]) -> list[list[float]]:
    ox, oy = float(origin[0]), float(origin[1])
    return [[round(p[0] - ox, 2), round(p[1] - oy, 2)] for p in pts]


def transform_frame(
    fr: dict,
    last_tl: list[float] | None,
    *,
    fill: bool,
    relative: bool,
    height: float | None,
) -> tuple[dict, list[float] | None]:
    raw_tl = player_abs(fr)
    new_last = raw_tl if raw_tl is not None else last_tl
    used_tl = raw_tl
    hold = 0
    if used_tl is None and fill and last_tl is not None:
        used_tl = last_tl
        hold = 1

    def pts_out(disk_pts: list[list[float]]) -> list[list[float]]:
        bl = [to_bl(p, height) for p in disk_pts]
        if not relative:
            return bl
        if used_tl is None:
            return []
        return minus(bl, to_bl(used_tl, height))

    view = {
        "t_ns": int(fr["t_ns"]) if fr.get("t_ns") is not None else None,
        "player_hold": hold,
        "player_missing": raw_tl is None,
        "relative": relative,
        "origin_bl": to_bl(used_tl, height) if used_tl is not None else None,
        "player_xy": None if used_tl is None else ([0.0, 0.0] if relative else to_bl(used_tl, height)),
        "mon": int(fr.get("mon") or 0),
        "loot": int(fr.get("loot") or 0),
        "gate": int(fr.get("gate") or 0),
        "boss": int(fr.get("boss") or 0),
    }
    for name in CLASS_XY:
        view[f"{name}_xy"] = pts_out(as_xy_list(fr.get(f"{name}_xy")))
    return view, new_last


def views_for_frames(
    frames: list[dict],
    *,
    fill: bool,
    relative: bool,
    height: float | None,
) -> list[dict]:
    last = None
    out: list[dict] = []
    for fr in frames:
        view, last = transform_frame(fr, last, fill=fill, relative=relative, height=height)
        out.append(view)
    return out
