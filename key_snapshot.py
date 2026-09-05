"""开录时轮询一次键盘按下状态（GetAsyncKeyState），不是每帧标签。"""
from __future__ import annotations

import ctypes

user32 = ctypes.windll.user32

# 与 pynput Listener 写入 keys.jsonl 的名字对齐（key.name / key.char）
_VK_NAMED = (
    (0x08, "backspace"),
    (0x09, "tab"),
    (0x0D, "enter"),
    (0x1B, "esc"),
    (0x20, "space"),
    (0x21, "page_up"),
    (0x22, "page_down"),
    (0x23, "end"),
    (0x24, "home"),
    (0x25, "left"),
    (0x26, "up"),
    (0x27, "right"),
    (0x28, "down"),
    (0x2D, "insert"),
    (0x2E, "delete"),
    (0xA0, "shift_l"),
    (0xA1, "shift_r"),
    (0xA2, "ctrl_l"),
    (0xA3, "ctrl_r"),
    (0xA4, "alt_l"),
    (0xA5, "alt_r"),
)


def _down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def snapshot_held_keys() -> list[str]:
    """返回当前按下的键名；只应在开录时调用一次。"""
    held: list[str] = []
    seen: set[str] = set()

    def add(name: str):
        if name and name not in seen:
            seen.add(name)
            held.append(name)

    for vk, name in _VK_NAMED:
        if _down(vk):
            add(name)
    for i in range(10):
        if _down(0x30 + i):
            add(str(i))
    for i in range(26):
        if _down(0x41 + i):
            add(chr(ord("a") + i))
    for i in range(12):
        if _down(0x70 + i):
            add(f"f{i + 1}")
    return held
