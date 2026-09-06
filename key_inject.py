"""SendInput 键盘注入。给执行层用，不进 fsm_core / 回放。"""
from __future__ import annotations

import ctypes
import time

user32 = ctypes.windll.user32

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MAPVK_VK_TO_VSC = 0

_EXTENDED_VK = {
    0x21,
    0x22,
    0x23,
    0x24,
    0x25,
    0x26,
    0x27,
    0x28,
    0x2D,
    0x2E,
    0xA3,
    0xA5,
}

# 左右修饰键用 VK 发：MapVirtualKey+SCANCODE 对 Alt 经常进不了游戏的系统键通道。
_VK_DIRECT = {
    0xA0,
    0xA1,
    0xA2,
    0xA3,
    0xA4,
    0xA5,
}

_NAME_VK = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "page_up": 0x21,
    "page_down": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "delete": 0x2E,
    "shift": 0xA0,
    "shift_l": 0xA0,
    "shift_r": 0xA1,
    "ctrl": 0xA2,
    "ctrl_l": 0xA2,
    "ctrl_r": 0xA3,
    "alt": 0xA4,
    "alt_l": 0xA4,
    "lalt": 0xA4,
    "alt_r": 0xA5,
}
for i in range(10):
    _NAME_VK[str(i)] = 0x30 + i
for i in range(26):
    _NAME_VK[chr(ord("a") + i)] = 0x41 + i
for i in range(12):
    _NAME_VK[f"f{i + 1}"] = 0x70 + i


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", INPUTUNION)]


def vk_of(name: str) -> int | None:
    key = str(name or "").strip().lower()
    if not key:
        return None
    return _NAME_VK.get(key)


def _send(vk: int, up: bool) -> None:
    scan = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) & 0xFF
    if vk in _VK_DIRECT:
        flags = 0
        if vk in _EXTENDED_VK:
            flags |= KEYEVENTF_EXTENDEDKEY
        if up:
            flags |= KEYEVENTF_KEYUP
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki = KEYBDINPUT(vk, scan, flags, 0, None)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        return
    flags = KEYEVENTF_SCANCODE
    if vk in _EXTENDED_VK:
        flags |= KEYEVENTF_EXTENDEDKEY
    if up:
        flags |= KEYEVENTF_KEYUP
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(0, scan, flags, 0, None)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def key_down(name: str) -> bool:
    vk = vk_of(name)
    if vk is None:
        return False
    _send(vk, False)
    return True


def key_up(name: str) -> bool:
    vk = vk_of(name)
    if vk is None:
        return False
    _send(vk, True)
    return True


def key_tap(name: str, hold_ms: int) -> bool:
    """按下 → 保持 hold_ms 毫秒 → 抬起。禁止瞬时 down+up。"""
    try:
        ms = int(hold_ms)
    except (TypeError, ValueError):
        return False
    if ms < 1:
        return False
    if not key_down(name):
        return False
    time.sleep(ms / 1000.0)
    key_up(name)
    return True
