"""记住各 GUI 上次关闭时的窗口大小（和位置）。

框选遮罩、测距这类一次性全屏窗不走这里。
"""
from __future__ import annotations

import json
from pathlib import Path

STORE = Path(__file__).resolve().parent / "_window_geom.json"


def _load() -> dict:
    if not STORE.is_file():
        return {}
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _dump(data: dict) -> None:
    try:
        STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _screen() -> tuple[int, int, int, int]:
    try:
        from window_align import get_virtual_screen

        vs = get_virtual_screen()
        return int(vs["x"]), int(vs["y"]), int(vs["width"]), int(vs["height"])
    except Exception:
        return 0, 0, 1920, 1080


def _int(val, default: int | None = None) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def capture(win) -> dict | None:
    try:
        win.update_idletasks()
        w = int(win.winfo_width())
        h = int(win.winfo_height())
        x = int(win.winfo_x())
        y = int(win.winfo_y())
    except Exception:
        return None
    if w < 80 or h < 80:
        return None
    zoomed = False
    try:
        zoomed = str(win.state()) == "zoomed"
    except Exception:
        pass
    return {"w": w, "h": h, "x": x, "y": y, "zoomed": zoomed}


def remember(win, key: str) -> None:
    geom = capture(win)
    if not geom or not key:
        return
    data = _load()
    data[str(key)] = geom
    _dump(data)


def apply(win, key: str, *, min_w: int, min_h: int, fallback: str | None = None) -> bool:
    """有记录则还原。返回是否用了上次关闭的尺寸。"""
    data = _load()
    geom = data.get(str(key))
    if not isinstance(geom, dict):
        if fallback:
            try:
                win.geometry(fallback)
            except Exception:
                pass
        return False
    w = max(min_w, _int(geom.get("w"), min_w) or min_w)
    h = max(min_h, _int(geom.get("h"), min_h) or min_h)
    x = _int(geom.get("x"))
    y = _int(geom.get("y"))
    sx, sy, sw, sh = _screen()
    w = min(w, max(min_w, sw))
    h = min(h, max(min_h, sh))
    if x is None or y is None:
        try:
            win.geometry(f"{w}x{h}")
        except Exception:
            return False
    else:
        x = min(max(x, sx), sx + sw - 80)
        y = min(max(y, sy), sy + sh - 80)
        try:
            win.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            return False
    if geom.get("zoomed"):
        try:
            win.state("zoomed")
        except Exception:
            pass
    return True
