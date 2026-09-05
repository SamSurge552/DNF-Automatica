# 从 DNF 战斗统计悬停读技能信息，生成角色键位表草稿（不写 recordings）。
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

if not getattr(sys, "frozen", False):
    os.chdir(Path(__file__).resolve().parent)

from admin_runtime import apply_saved_env, is_admin, restart_as_admin, suppress_windows_error_dialogs

apply_saved_env()
suppress_windows_error_dialogs()

from window_align import enable_dpi_awareness, focus_point, get_virtual_screen

enable_dpi_awareness()

import ctypes
import numpy as np
import tkinter as tk
from tkinter import ttk, font, messagebox, simpledialog
from PIL import Image, ImageDraw, ImageTk

from screen_capture import ScreenCaptureModule
from window_geom import apply as apply_window_geom
from window_geom import remember as remember_window_geom
from fsm_core import parse_hold_ms, HOLD_MS_KEY, HOLD_FRAMES_KEY as HOLD_FRAMES_LEGACY

MAX_SKILLS = 9
BINDS_DIR = Path("skill_binds")
SETTINGS_PATH = BINDS_DIR / "_tool_ui.json"
CHAR_FILE = Path("character_names_custom.txt")
EXAMPLE_PNG = BINDS_DIR / "example_combat_stats.png"
COOLDOWN_KEY = "cooldown_s"
MASH_KEY = "mash"
COMBO_KEY = "combo"
MULTI_KEY = "multi"
MULTI_N_KEY = "multi_n"
DEFAULT_MULTI = 2
TZ8 = timezone(timedelta(hours=8))

user32 = ctypes.windll.user32
VK_ESCAPE = 0x1B
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT),
    ]


def _now_iso() -> str:
    return datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def get_cursor() -> tuple[int, int]:
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


def _sendinput_abs(x: int, y: int) -> None:
    vs = get_virtual_screen()
    w = max(int(vs["width"]) - 1, 1)
    h = max(int(vs["height"]) - 1, 1)
    nx = int(round((int(x) - int(vs["x"])) * 65535 / w))
    ny = int(round((int(y) - int(vs["y"])) * 65535 / h))
    nx = min(max(nx, 0), 65535)
    ny = min(max(ny, 0), 65535)
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(
        nx,
        ny,
        0,
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        0,
        None,
    )
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def set_cursor(x: int, y: int) -> tuple[int, int]:
    x, y = int(x), int(y)
    try:
        _sendinput_abs(x, y)
    except Exception:
        pass
    user32.SetCursorPos(x, y)
    return get_cursor()


def esc_down() -> bool:
    return bool(user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)


def load_characters() -> list[str]:
    if not CHAR_FILE.is_file():
        return ["SolarWarden"]
    names = [ln.strip() for ln in CHAR_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names or ["SolarWarden"]


def bind_path(character: str) -> Path:
    safe = "".join(ch for ch in character.strip() if ch not in r'\/:*?"<>|') or "character"
    return BINDS_DIR / f"{safe}.json"


def _skill_slot(item: dict) -> int:
    try:
        return int(item.get("slot") or 0)
    except (TypeError, ValueError):
        return 0


def parse_hold_frames(item) -> int | None:
    v = parse_hold_ms(item)
    if v is None:
        return None
    return max(1, int(v))


def merge_bind_payload(existing: dict | None, payload: dict) -> dict:
    """保存时合并旧表：不丢 hold_frames / cooldown_s / combo / multi / mash 和未出现在新列表里的技能行。"""
    old = existing if isinstance(existing, dict) else {}
    out = dict(old)
    out.update(payload)
    if not payload.get("region") and old.get("region"):
        out["region"] = old["region"]
    if not payload.get("hover") and old.get("hover"):
        out["hover"] = old["hover"]
    by_slot: dict[int, dict] = {}
    for item in old.get("skills") or []:
        if not isinstance(item, dict):
            continue
        slot = _skill_slot(item)
        if slot > 0:
            by_slot[slot] = dict(item)
    merged = []
    seen: set[int] = set()
    for item in payload.get("skills") or []:
        if not isinstance(item, dict):
            continue
        slot = _skill_slot(item)
        prev = by_slot.get(slot, {})
        row = dict(prev)
        row.update(item)
        for keep in (HOLD_MS_KEY, HOLD_FRAMES_LEGACY, COOLDOWN_KEY):
            if keep not in item and keep in prev:
                row[keep] = prev[keep]
        if COMBO_KEY in item:
            row[COMBO_KEY] = bool(item.get(COMBO_KEY))
        if MASH_KEY in item:
            row[MASH_KEY] = bool(item.get(MASH_KEY))
        if MULTI_KEY in item:
            row[MULTI_KEY] = bool(item.get(MULTI_KEY))
        if MULTI_N_KEY in item:
            try:
                row[MULTI_N_KEY] = max(DEFAULT_MULTI, min(9, int(item.get(MULTI_N_KEY))))
            except (TypeError, ValueError):
                row[MULTI_N_KEY] = prev.get(MULTI_N_KEY, DEFAULT_MULTI)
        elif MULTI_N_KEY in prev:
            row[MULTI_N_KEY] = prev[MULTI_N_KEY]
        merged.append(row)
        if slot > 0:
            seen.add(slot)
    for slot, prev in sorted(by_slot.items()):
        if slot not in seen:
            merged.append(prev)
    merged.sort(key=lambda s: _skill_slot(s) or 999)
    out["skills"] = merged
    return out


def _seq(val) -> list:
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return [val[i] for i in range(val.shape[0])] if val.size else []
    try:
        return list(val)
    except TypeError:
        return []


def ocr_texts(ocr, frame) -> list[str]:
    result = ocr(frame)
    out = []
    for t in _seq(getattr(result, "txts", None)):
        text = str(t).strip()
        if text:
            out.append(text)
    return out


_HAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_KONGGE_RE = re.compile(r"空\s*格")
_ARROW_MAP = {
    "↑": "up",
    "↓": "down",
    "←": "left",
    "→": "right",
}


def normalize_command(cmd: str) -> str:
    """「空格」→ space；↑↓←→ → up/down/left/right；字母小写。与 keys.jsonl 词表对齐。"""
    t = _KONGGE_RE.sub("space", str(cmd or ""))
    t = t.replace("＋", "+")
    parts: list[str] = []
    for ch in t:
        word = _ARROW_MAP.get(ch)
        if word:
            if parts and parts[-1][-1:].isalnum():
                parts.append(" ")
            parts.append(word)
        else:
            parts.append(ch)
    t = "".join(parts).lower()
    return re.sub(r"[ \t]+", " ", t).strip()


def command_has_han(cmd: str) -> bool:
    """normalize 之后还剩汉字则不支持（箭头、字母、+、space 可以）。"""
    return bool(_HAN_RE.search(str(cmd or "")))


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def command_from_ocr_texts(texts: list[str], marker: str = "操作指令") -> str:
    """同一块 OCR 文本里找 GUI 填的匹配串；要后面的原文。匹配串前后空白忽略，OCR 空格也忽略。后面若有冒号会剥掉。认不到就空。"""
    marker = str(marker or "").strip()
    compact_m = _compact(marker)
    if not compact_m:
        return ""
    for t in texts:
        right = _after_marker(t, marker, compact_m)
        if right is not None:
            return normalize_command(right)
    return ""


_FULLWIDTH_NUM = str.maketrans("０１２３４５６７８９．，", "0123456789..")


def format_cooldown_s(value: float) -> str:
    n = float(value)
    if n == int(n):
        return str(int(n))
    return f"{n:.3f}".rstrip("0").rstrip(".")


def cooldown_json_value(value: float):
    n = float(value)
    if n < 0:
        n = 0.0
    if n == int(n):
        return int(n)
    return round(n, 3)


def parse_cooldown_text(raw: str, *, allow_bare: bool = True) -> float | None:
    """从「3秒 / 3.5秒 / 无 / 1分30秒」取出秒。认不到就 None。"""
    t = str(raw or "").strip().translate(_FULLWIDTH_NUM)
    t = re.sub(r"\s+", "", t)
    if not t:
        return None
    if "%" in t:
        return None
    if t.startswith("减少"):
        return None
    if t in ("无", "无冷却", "无冷却时间", "-", "—", "－"):
        return 0.0
    if t == "无冷却" or t.startswith("无"):
        return 0.0
    m = re.search(
        r"(?:(\d+(?:\.\d+)?)分(?:钟)?)(?:(\d+(?:\.\d+)?)秒)?",
        t,
    )
    if m and "分" in t:
        mins = float(m.group(1))
        secs = float(m.group(2) or 0)
        return mins * 60.0 + secs
    m = re.search(r"(\d+(?:\.\d+)?)秒", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)s\b", t, flags=re.I)
    if m:
        return float(m.group(1))
    if allow_bare:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)", t)
        if m:
            return float(m.group(1))
    return None


def parse_cooldown_item(item) -> float | None:
    if not isinstance(item, dict):
        return None
    raw = item.get(COOLDOWN_KEY, item.get("cooldown"))
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return max(0.0, float(raw))
    return parse_cooldown_text(str(raw))


def cooldown_from_ocr_texts(texts: list[str], marker: str = "冷却时间") -> float | None:
    """和操作指令同一套：同一块里找匹配串，要冒号后面。标签和数字被拆开时看随后几块。跳过「冷却时间减少」。"""
    marker = str(marker or "").strip()
    compact_m = _compact(marker)
    if not compact_m:
        return None
    for i, t in enumerate(texts):
        right = _after_marker(t, marker, compact_m)
        if right is None:
            continue
        if _compact(right).startswith("减少"):
            continue
        parsed = parse_cooldown_text(right)
        if parsed is not None:
            return parsed
        for nxt in texts[i + 1 : i + 4]:
            ns = str(nxt or "").strip()
            if not ns:
                continue
            if _after_marker(ns, marker, compact_m) is not None:
                break
            if _compact(ns).startswith("减少"):
                continue
            parsed = parse_cooldown_text(ns, allow_bare=False)
            if parsed is not None:
                return parsed
            break
    return None


def _after_marker(text: str, marker: str, compact_m: str) -> str | None:
    idx = text.find(marker)
    if idx >= 0:
        right = text[idx + len(marker) :].strip()
        return re.sub(r"^[:：]\s*", "", right)
    compact_t = _compact(text)
    j = compact_t.find(compact_m)
    if j < 0:
        return None
    right = compact_t[j + len(compact_m) :]
    return re.sub(r"^[:：]+", "", right)


class ScreenRectSelector(tk.Toplevel):
    """全屏拖拽：得到虚拟屏像素矩形（战斗统计面板 / 技能信息区）。"""

    def __init__(self, parent, on_done):
        super().__init__(parent)
        self.on_done = on_done
        self.screen = get_virtual_screen()
        self.drag_start = None
        self.rect_id = None

        self.geometry(
            f"{self.screen['width']}x{self.screen['height']}+{self.screen['x']}+{self.screen['y']}"
        )
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.28)

        self.canvas = tk.Canvas(self, cursor="crosshair", bg="gray", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_text(
            self.screen["width"] // 2,
            40,
            text="拖拽框选战斗统计区域（悬停后要 OCR 的那块）  |  ESC 取消",
            fill="white",
            font=("Microsoft YaHei", 16, "bold"),
        )
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.bind("<Escape>", self.cancel)
        self.focus_force()

    def _press(self, event):
        self.drag_start = (event.x, event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def _drag(self, event):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#00ff66", width=2
        )

    def _release(self, event):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        cx = min(x0, x1)
        cy = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w < 24 or h < 24:
            self.drag_start = None
            return
        rect = {
            "x": int(self.screen["x"] + cx),
            "y": int(self.screen["y"] + cy),
            "width": int(w),
            "height": int(h),
        }
        self.on_done(rect)
        self.destroy()

    def cancel(self, event=None):
        self.destroy()


class ScreenTwoPointPicker(tk.Toplevel):
    """全屏点两下：用竖直间距定行距。第1点也可当作1号悬停点。"""

    def __init__(self, parent, on_done):
        super().__init__(parent)
        self.on_done = on_done
        self.screen = get_virtual_screen()
        self.p1 = None

        self.geometry(
            f"{self.screen['width']}x{self.screen['height']}+{self.screen['x']}+{self.screen['y']}"
        )
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.22)

        self.canvas = tk.Canvas(self, cursor="crosshair", bg="gray", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.hint_id = self.canvas.create_text(
            self.screen["width"] // 2,
            40,
            text="点第1行（1号技能）  |  ESC 取消",
            fill="white",
            font=("Microsoft YaHei", 16, "bold"),
        )
        self.canvas.bind("<ButtonPress-1>", self._click)
        self.bind("<Escape>", self.cancel)
        self.focus_force()

    def _screen_xy(self, event):
        return int(self.screen["x"] + event.x), int(self.screen["y"] + event.y)

    def _mark(self, event, label: str, color: str):
        r = 8
        self.canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r, outline=color, width=2
        )
        self.canvas.create_text(event.x + 12, event.y, text=label, fill=color, anchor="w")

    def _click(self, event):
        sx, sy = self._screen_xy(event)
        if self.p1 is None:
            self.p1 = (sx, sy)
            self._mark(event, "1", "#00ff66")
            self.canvas.itemconfig(self.hint_id, text="再点第2行（下一技能）  |  ESC 取消")
            return
        self._mark(event, "2", "#ffdc00")
        self.update_idletasks()
        self.on_done(self.p1, (sx, sy))
        self.destroy()

    def cancel(self, event=None):
        self.destroy()


class SkillBindTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("键位表 · DNF 战斗统计")
        self.geometry("1100x820")
        self.minsize(960, 700)
        if is_admin():
            self.title(self.title() + " [管理员]")

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Microsoft YaHei", size=10)
        self.option_add("*Font", default_font)

        self.capture = ScreenCaptureModule(base_output_dir="skill_binds/_preview")
        self.ocr = None
        self.ocr_error = None
        self.region = None
        self.preview_bgr = None
        self._photo = None
        self._preview_scale = 1.0
        self.busy = False
        self._stop = threading.Event()
        self._stop_win = None
        self._preview_is_example = True
        self._rows = []
        self._hotbar_guard = False
        self._persist_ok = False
        self._hold_frames: dict[int, int] = {}

        BINDS_DIR.mkdir(parents=True, exist_ok=True)

        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(main, text="角色与区域", padding=8)
        top.pack(fill=tk.X)

        r1 = ttk.Frame(top)
        r1.pack(fill=tk.X)
        ttk.Label(r1, text="角色", width=8).pack(side=tk.LEFT)
        self.character_var = tk.StringVar()
        self.character_combo = ttk.Combobox(r1, textvariable=self.character_var, state="readonly", width=22)
        self.character_combo.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(r1, text="刷新角色", command=self._refresh_characters).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(r1, text="框选区域", command=self.select_region).pack(side=tk.LEFT)
        self.region_label = ttk.Label(r1, text="[未框选]", foreground="#06c")
        self.region_label.pack(side=tk.LEFT, padx=8)

        r2 = ttk.Frame(top)
        r2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(r2, text="1号坐标", width=8).pack(side=tk.LEFT)
        ttk.Label(r2, text="x").pack(side=tk.LEFT)
        self.origin_x = tk.StringVar(value="24")
        ttk.Entry(r2, textvariable=self.origin_x, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r2, text="y").pack(side=tk.LEFT)
        self.origin_y = tk.StringVar(value="24")
        ttk.Entry(r2, textvariable=self.origin_y, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r2, text="行距").pack(side=tk.LEFT)
        self.row_dy = tk.StringVar(value="28")
        ttk.Entry(r2, textvariable=self.row_dy, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(r2, text="测行距", command=self.measure_row_pitch).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(r2, text="个数").pack(side=tk.LEFT)
        self.count_var = tk.StringVar(value="9")
        ttk.Spinbox(r2, from_=1, to=MAX_SKILLS, textvariable=self.count_var, width=4).pack(
            side=tk.LEFT, padx=(2, 8)
        )
        ttk.Label(r2, text="停留秒").pack(side=tk.LEFT)
        self.dwell_var = tk.StringVar(value="0.45")
        ttk.Entry(r2, textvariable=self.dwell_var, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r2, text="匹配").pack(side=tk.LEFT)
        self.marker_var = tk.StringVar(value="操作指令")
        ttk.Entry(r2, textvariable=self.marker_var, width=10).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(r2, text="冷却").pack(side=tk.LEFT)
        self.cd_marker_var = tk.StringVar(value="冷却时间")
        ttk.Entry(r2, textvariable=self.cd_marker_var, width=10).pack(side=tk.LEFT, padx=(2, 0))

        hint = ttk.Label(
            top,
            text="读取前把战斗统计按「平均伤害」从高到低排。战斗统计窗口放在游戏左上角。选完区域然后点击预览图里技能1即可。",
            foreground="#a40",
            wraplength=920,
            justify=tk.LEFT,
        )
        hint.pack(fill=tk.X, pady=(6, 0))

        r3 = ttk.Frame(top)
        r3.pack(fill=tk.X, pady=(6, 0))
        self.read_btn = ttk.Button(r3, text="读取技能", command=self.start_read)
        self.read_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(r3, text="停止", command=self.stop_read, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(r3, text="保存键位表", command=self.save_binds).pack(side=tk.LEFT, padx=8)
        ttk.Button(r3, text="载入已有表", command=self.load_binds).pack(side=tk.LEFT)
        ttk.Button(r3, text="以管理员重启", command=self.restart_elevated).pack(side=tk.LEFT, padx=8)
        ttk.Label(
            r3,
            text="截图后再 OCR。日志会逐条刷结果。停止或 ESC 中止。",
            foreground="#666",
        ).pack(side=tk.LEFT, padx=12)

        mid = ttk.Frame(main)
        mid.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        prev_box = ttk.LabelFrame(mid, text="区域预览", padding=6)
        prev_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview = tk.Canvas(prev_box, background="#222", highlightthickness=0)
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.preview.bind("<Button-1>", self._on_preview_click)
        self.preview.create_text(
            160, 40, text="先框选战斗统计区域", fill="#ddd", font=("Microsoft YaHei", 11)
        )

        right = ttk.LabelFrame(mid, text="结果：序号=优先级；指令/冷却=匹配串冒号右原文", padding=6)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        header = ttk.Frame(right)
        header.pack(fill=tk.X)
        ttk.Label(header, text="#", width=3).pack(side=tk.LEFT)
        ttk.Label(header, text="行", width=8).pack(side=tk.LEFT)
        ttk.Label(header, text="操作指令", width=16).pack(side=tk.LEFT)
        ttk.Label(header, text="冷却秒", width=8).pack(side=tk.LEFT)
        ttk.Label(header, text="快捷栏").pack(side=tk.LEFT)
        ttk.Label(header, text="按键", width=6).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(header, text="连续释放").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(header, text="多次释放").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(header, text="MULTI", width=6).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(header, text="连按").pack(side=tk.LEFT, padx=(8, 0))

        self.rows_host = ttk.Frame(right)
        self.rows_host.pack(fill=tk.BOTH, expand=True)
        self._build_rows()
        ttk.Label(
            right,
            text="连续释放：CD 按 270 秒。多次释放：勾选后填 MULTI（默认 2），FSM 拆成多份独立 CD。连按：执行层打 COUNT 次点按（COUNT/间隔在主面板 FSM设置）。连续/多次不触发提取 CD 重置检测。勾选后点「保存键位表」。",
            foreground="#668",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

        log_box = ttk.LabelFrame(main, text="日志", padding=6)
        log_box.pack(fill=tk.X, pady=(8, 0))
        prog_row = ttk.Frame(log_box)
        prog_row.pack(fill=tk.X, pady=(0, 4))
        self.progress_var = tk.StringVar(value="")
        ttk.Label(prog_row, textvariable=self.progress_var, width=18).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(prog_row, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.log_text = tk.Text(log_box, height=6, state=tk.DISABLED)
        self.log_text.pack(fill=tk.X)

        self._refresh_characters()
        self._load_ui_settings()
        self._persist_ok = True
        for var in (
            self.origin_x,
            self.origin_y,
            self.row_dy,
            self.count_var,
            self.dwell_var,
            self.character_var,
            self.marker_var,
            self.cd_marker_var,
        ):
            var.trace_add("write", lambda *_: self._save_ui_settings())
        if EXAMPLE_PNG.is_file():
            self.log(f"字段对照图: {EXAMPLE_PNG}")
        if is_admin():
            self.log("当前是管理员，可以向游戏注入鼠标。")
        else:
            self.log("当前不是管理员：点「读取技能」会被拦截。请先「以管理员重启」。")
        self.log("对照图已放进预览。请按平均伤害从高到低排列后再读（序号=优先级）。只认「操作指令：」冒号右。")
        if self.region:
            self.log(
                f"已记住选区 {self.region['width']}×{self.region['height']}，预览先显示对照图。"
            )
        self.after(200, self._show_example_preview)
        self.after(200, self._init_ocr_async)
        self.after_idle(lambda: apply_window_geom(self, "skill_bind", min_w=960, min_h=700))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')}  {msg}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _refresh_characters(self):
        names = load_characters()
        self.character_combo["values"] = names
        cur = self.character_var.get()
        if cur not in names:
            self.character_var.set(names[0])

    def _build_rows(self):
        for child in self.rows_host.winfo_children():
            child.destroy()
        self._rows = []
        for i in range(MAX_SKILLS):
            host = ttk.Frame(self.rows_host)
            host.pack(fill=tk.X, pady=2)
            ttk.Label(host, text=str(i + 1), width=3).pack(side=tk.LEFT)
            ttk.Label(host, text=f"技能{i + 1}", width=8).pack(side=tk.LEFT, padx=(0, 4))
            cmd_var = tk.StringVar(value="")
            ttk.Entry(host, textvariable=cmd_var, width=16).pack(side=tk.LEFT, padx=(0, 4))
            cd_var = tk.StringVar(value="")
            ttk.Entry(host, textvariable=cd_var, width=7).pack(side=tk.LEFT, padx=(0, 4))
            hotbar_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                host,
                variable=hotbar_var,
                command=lambda idx=i: self._on_hotbar_toggle(idx),
            ).pack(side=tk.LEFT)
            key_var = tk.StringVar(value="")
            key_entry = ttk.Entry(host, textvariable=key_var, width=6, state=tk.DISABLED)
            key_entry.pack(side=tk.LEFT, padx=(2, 0))
            key_entry.bind("<Return>", lambda _e, idx=i: self._apply_hotbar_key(idx))
            key_entry.bind("<FocusOut>", lambda _e, idx=i: self._apply_hotbar_key(idx))
            combo_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(host, variable=combo_var).pack(side=tk.LEFT, padx=(8, 0))
            multi_var = tk.BooleanVar(value=False)
            multi_n_var = tk.StringVar(value=str(DEFAULT_MULTI))
            multi_spin = ttk.Spinbox(
                host,
                from_=DEFAULT_MULTI,
                to=9,
                width=3,
                textvariable=multi_n_var,
                state=tk.DISABLED,
            )
            ttk.Checkbutton(
                host,
                variable=multi_var,
                command=lambda idx=i: self._on_multi_toggle(idx),
            ).pack(side=tk.LEFT, padx=(8, 0))
            multi_spin.pack(side=tk.LEFT, padx=(4, 0))
            mash_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(host, variable=mash_var).pack(side=tk.LEFT, padx=(8, 0))
            self._rows.append(
                {
                    "command": cmd_var,
                    "cooldown": cd_var,
                    "hotbar": hotbar_var,
                    "hotbar_key": key_var,
                    "key_entry": key_entry,
                    "combo": combo_var,
                    "multi": multi_var,
                    "multi_n": multi_n_var,
                    "multi_spin": multi_spin,
                    "mash": mash_var,
                    "_ocr_cmd": "",
                }
            )

    def _set_hotbar_enabled(self, idx: int, on: bool):
        row = self._rows[idx]
        row["key_entry"].config(state=tk.NORMAL if on else tk.DISABLED)

    def _on_multi_toggle(self, idx: int):
        row = self._rows[idx]
        on = bool(row["multi"].get())
        row["multi_spin"].config(state=tk.NORMAL if on else tk.DISABLED)
        if on:
            try:
                n = int(str(row["multi_n"].get() or DEFAULT_MULTI))
            except ValueError:
                n = 0
            if n < DEFAULT_MULTI:
                row["multi_n"].set(str(DEFAULT_MULTI))

    def _on_hotbar_toggle(self, idx: int):
        if self._hotbar_guard:
            return
        row = self._rows[idx]
        if not row["hotbar"].get():
            self._set_hotbar_enabled(idx, False)
            if row.get("_ocr_cmd"):
                row["command"].set(row["_ocr_cmd"])
            return
        self._set_hotbar_enabled(idx, True)
        suggested = row["hotbar_key"].get().strip()
        key = simpledialog.askstring(
            "快捷栏",
            f"技能{idx + 1} 对应快捷栏按键：",
            initialvalue=suggested,
            parent=self,
        )
        if key is None or not str(key).strip():
            self._hotbar_guard = True
            row["hotbar"].set(False)
            self._hotbar_guard = False
            self._set_hotbar_enabled(idx, False)
            return
        key = normalize_command(key)
        if command_has_han(key):
            messagebox.showwarning("键位表", f"技能{idx + 1} 快捷栏按键含中文，不支持。")
            self._hotbar_guard = True
            row["hotbar"].set(False)
            self._hotbar_guard = False
            self._set_hotbar_enabled(idx, False)
            return
        if not row.get("_ocr_cmd"):
            row["_ocr_cmd"] = row["command"].get()
        self._hotbar_guard = True
        row["hotbar_key"].set(key)
        self._hotbar_guard = False
        row["command"].set(key)
        self.log(f"技能{idx + 1} 快捷栏 → {key}")

    def _apply_hotbar_key(self, idx: int):
        if self._hotbar_guard:
            return
        row = self._rows[idx]
        if not row["hotbar"].get():
            return
        raw = row["hotbar_key"].get()
        key = normalize_command(raw)
        if command_has_han(key):
            return
        self._hotbar_guard = True
        try:
            if key != raw:
                row["hotbar_key"].set(key)
            if key:
                row["command"].set(key)
        finally:
            self._hotbar_guard = False

    def _clear_row_command(self, row: dict, keep_hotbar: bool = False):
        if keep_hotbar and row["hotbar"].get() and normalize_command(row["hotbar_key"].get()):
            return
        row["command"].set("")
        row["_ocr_cmd"] = ""

    def _init_ocr_async(self):
        self.log("正在初始化 RapidOCR…")
        threading.Thread(target=self._init_ocr_worker, daemon=True).start()

    def _init_ocr_worker(self):
        try:
            from cuda_runtime import ensure_cuda_runtime
            from rapidocr import RapidOCR

            ensure_cuda_runtime()
            ocr = RapidOCR(
                params={
                    "EngineConfig.onnxruntime.use_cuda": True,
                    "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
                    "Global.use_cls": False,
                    "Global.text_score": 0.5,
                    "Global.max_side_len": 8192,
                    "Global.log_level": "error",
                }
            )
            self.ocr = ocr
            self.after(0, lambda: self.log("RapidOCR 就绪。打开游戏战斗统计后框选区域。"))
        except Exception as e:
            self.ocr_error = str(e)
            self.after(0, lambda: self.log(f"RapidOCR 初始化失败: {e}"))

    def select_region(self):
        if self.busy:
            return
        ScreenRectSelector(self, self._on_region)

    def measure_row_pitch(self):
        if self.busy:
            return
        self.log("测行距：先点第1行，再点第2行。")
        ScreenTwoPointPicker(self, self._on_pitch_points)

    def _on_pitch_points(self, p1: tuple[int, int], p2: tuple[int, int]):
        dy = abs(int(p2[1]) - int(p1[1]))
        if dy < 1:
            self.log("两点几乎同一高度，行距未改。")
            return
        self.row_dy.set(str(dy))
        if self.region:
            rx = int(p1[0]) - int(self.region["x"])
            ry = int(p1[1]) - int(self.region["y"])
            rw = int(self.region["width"])
            rh = int(self.region["height"])
            if 0 <= rx < rw and 0 <= ry < rh:
                self.origin_x.set(str(rx))
                self.origin_y.set(str(ry))
                self.log(f"行距 {dy}px；1号点已用第1点 ({rx}, {ry})")
            else:
                self.log(f"行距 {dy}px；第1点不在选区内，只改行距。")
        else:
            self.log(f"行距 {dy}px。还没框选区域，1号点请稍后在预览上点。")
        self._save_ui_settings()
        if self.preview_bgr is not None:
            self._show_preview()

    def _on_region(self, rect: dict):
        self.region = rect
        self.region_label.config(
            text=f"X:{rect['x']} Y:{rect['y']}  {rect['width']}×{rect['height']}"
        )
        self.log(f"已框选 {rect['width']}×{rect['height']} @ ({rect['x']},{rect['y']})")
        self._save_ui_settings()
        self.refresh_preview()

    def refresh_preview(self):
        if not self.region:
            return
        grabbed = self.capture.capture(self.region, gui=None, save=False)
        if not grabbed or grabbed.get("frame") is None:
            self.log("截取预览失败。")
            return
        self.preview_bgr = grabbed["frame"]
        self._preview_is_example = False
        self._show_preview()

    def _show_example_preview(self):
        if not EXAMPLE_PNG.is_file():
            self.preview.delete("all")
            self.preview.create_text(
                180,
                40,
                text="缺少对照图 skill_binds/example_combat_stats.png",
                fill="#ddd",
                font=("Microsoft YaHei", 11),
            )
            return
        img = Image.open(EXAMPLE_PNG).convert("RGB")
        box_w = max(int(self.preview.winfo_width()), 320)
        box_h = max(int(self.preview.winfo_height()), 280)
        scale = min(box_w / img.width, box_h / img.height, 1.0)
        self._preview_scale = scale
        if scale < 1.0:
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.BILINEAR,
            )
        self._photo = ImageTk.PhotoImage(img)
        self._preview_is_example = True
        self.preview.delete("all")
        self.preview.create_image(0, 0, anchor=tk.NW, image=self._photo)
        self.preview.create_text(
            8,
            8,
            anchor=tk.NW,
            text="对照图（非当前截屏）",
            fill="#ffcc66",
            font=("Microsoft YaHei", 11, "bold"),
        )

    def _hover_params(self) -> tuple[int, int, int, int, float]:
        x = int(float(self.origin_x.get().strip() or 0))
        y = int(float(self.origin_y.get().strip() or 0))
        dy = int(float(self.row_dy.get().strip() or 0))
        n = int(float(self.count_var.get().strip() or 1))
        n = max(1, min(MAX_SKILLS, n))
        dwell = float(self.dwell_var.get().strip() or 0.45)
        return x, y, dy, n, max(0.05, dwell)

    def _show_preview(self, live_caption: str | None = None):
        if self.preview_bgr is None:
            return
        rgb = self.preview_bgr[:, :, ::-1].copy()
        img = Image.fromarray(rgb)
        try:
            ox, oy, dy, n, _ = self._hover_params()
        except ValueError:
            ox, oy, dy, n = 24, 24, 28, 9
        draw = ImageDraw.Draw(img)
        for i in range(n):
            px, py = ox, oy + i * dy
            if 0 <= px < img.width and 0 <= py < img.height:
                r = 6
                color = (0, 255, 100) if i == 0 else (255, 220, 0)
                draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=2)
                draw.text((px + 8, py - 8), str(i + 1), fill=color)

        box_w = max(int(self.preview.winfo_width()), 320)
        box_h = max(int(self.preview.winfo_height()), 280)
        scale = min(box_w / img.width, box_h / img.height, 1.0)
        self._preview_scale = scale
        if scale < 1.0:
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.NEAREST,
            )
        self._photo = ImageTk.PhotoImage(img)
        self.preview.delete("all")
        self.preview.create_image(0, 0, anchor=tk.NW, image=self._photo)
        if live_caption:
            self.preview.create_text(
                8,
                8,
                anchor=tk.NW,
                text=live_caption,
                fill="#ffcc66",
                font=("Microsoft YaHei", 11, "bold"),
            )

    def _show_live_frame(self, frame, caption: str):
        if frame is None:
            return
        self.preview_bgr = np.ascontiguousarray(frame)
        self._preview_is_example = False
        self._show_preview(live_caption=caption)

    def _on_preview_click(self, event):
        if self.preview_bgr is None or self.busy or self._preview_is_example:
            return
        scale = self._preview_scale or 1.0
        x = int(event.x / scale)
        y = int(event.y / scale)
        h, w = self.preview_bgr.shape[:2]
        x = min(max(0, x), w - 1)
        y = min(max(0, y), h - 1)
        self.origin_x.set(str(x))
        self.origin_y.set(str(y))
        self.log(f"1号悬停点（相对选区）: ({x}, {y})")
        self._save_ui_settings()
        self._show_preview()

    def restart_elevated(self):
        if is_admin():
            self.log("已经是管理员。")
            return
        if self.busy:
            self.log("读取中不能提权，先等结束或 ESC。")
            return
        self._save_ui_settings()
        ok, msg = restart_as_admin(cwd=os.getcwd())
        self.log(msg)
        if ok:
            remember_window_geom(self, "skill_bind")
            try:
                self.destroy()
            except Exception:
                pass
            os._exit(0)

    def start_read(self):
        if self.busy:
            return
        if not is_admin():
            messagebox.showerror(
                "读取需要管理员",
                "读取技能必须先以管理员身份启动。\n\n"
                "请点「以管理员重启」，再点读取。\n"
                "没有「继续读取」选项。",
                parent=self,
            )
            self.log("读取已拦截：当前不是管理员。")
            return
        if not self.region:
            messagebox.showinfo("键位表", "请先框选战斗统计区域。")
            return
        try:
            self._hover_params()
        except ValueError:
            messagebox.showinfo("键位表", "坐标 / 行距 / 个数 / 停留秒 填数字。")
            return
        self._save_ui_settings()
        self._stop.clear()
        self.busy = True
        self.read_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        for row in self._rows:
            self._clear_row_command(row, keep_hotbar=True)
        params = self._hover_params()
        region = dict(self.region)
        self.log("开始截图（不跑 OCR）。点「停止」或 ESC 中止。")
        self._set_progress("截图", 0, params[3])
        self._show_stop_overlay()
        threading.Thread(target=self._read_worker, args=(region, params), daemon=True).start()

    def stop_read(self):
        if not self.busy:
            return
        self._stop.set()
        self.log("已点停止，当前这一步结束后中止。")

    def _show_stop_overlay(self):
        self._hide_stop_overlay()
        w = tk.Toplevel(self)
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        w.geometry("180x52+24+24")
        btn = ttk.Button(w, text="停止读取", command=self.stop_read)
        btn.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._stop_win = w

    def _hide_stop_overlay(self):
        win = getattr(self, "_stop_win", None)
        self._stop_win = None
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def _set_progress(self, label: str, cur: int, total: int):
        total = max(int(total), 1)
        cur = max(0, min(int(cur), total))
        self.progress["maximum"] = total
        self.progress["value"] = cur
        self.progress_var.set(f"{label} {cur}/{total}")

    def _grab(self, region: dict):
        grabbed = self.capture.capture(region, gui=None, save=False)
        if not grabbed:
            return None
        frame = grabbed.get("frame")
        return None if frame is None else np.ascontiguousarray(frame.copy())

    def _read_worker(self, region: dict, params: tuple):
        parked = get_cursor()
        aborted = False
        hover_frames: list = []
        try:
            ox, oy, dy, n, dwell = params
            rx = int(region["x"])
            ry = int(region["y"])
            time.sleep(0.05)
            esc_down()
            first_xy = (rx + ox, ry + oy)
            try:
                focus_point(*first_xy)
            except Exception:
                pass
            time.sleep(0.05)
            for i in range(n):
                if self._stop.is_set() or esc_down():
                    aborted = True
                    break
                mx = rx + ox
                my = ry + oy + i * dy
                got = set_cursor(mx, my)
                if i == 0 and (abs(got[0] - mx) > 8 or abs(got[1] - my) > 8):
                    self.after(
                        0,
                        lambda g=got, t=(mx, my): self.log(
                            f"鼠标没到目标 {t}，实际 {g}。多半不是管理员或游戏吃掉了注入。"
                        ),
                    )
                t_end = time.time() + dwell
                while time.time() < t_end:
                    if self._stop.is_set() or esc_down():
                        aborted = True
                        break
                    set_cursor(mx, my)
                    time.sleep(0.02)
                if aborted:
                    break
                frame = self._grab(region)
                hover_frames.append(frame)
                self.after(
                    0,
                    lambda f=None if frame is None else frame.copy(), k=i + 1, tot=n: (
                        self._set_progress("截图", k, tot),
                        self._show_live_frame(f, f"截图 #{k}/{tot}"),
                    ),
                )
        finally:
            set_cursor(*parked)
            self.after(0, lambda: self._after_capture(hover_frames, aborted, parked, params))

    def _after_capture(self, hover_frames, aborted, parked, params):
        self.deiconify()
        self.lift()
        n_got = len(hover_frames)
        if aborted:
            self.log(f"截图中止。鼠标已放回。拿到 {n_got} 张。")
        else:
            self.log(f"截图完毕，已退出鼠标控制。悬停 {n_got} 张，开始 OCR。")
        if self._stop.is_set() and n_got == 0:
            self._finish_read([])
            return
        threading.Thread(
            target=self._ocr_worker,
            args=(hover_frames, params),
            daemon=True,
        ).start()

    def _wait_ocr(self) -> bool:
        for _ in range(200):
            if self._stop.is_set():
                return self.ocr is not None
            if self.ocr is not None:
                return True
            if self.ocr_error:
                return False
            time.sleep(0.1)
        return self.ocr is not None

    def _ocr_worker(self, hover_frames, params):
        slots = []
        stopped = False
        try:
            if not self._wait_ocr():
                self.after(0, lambda: self.log(f"OCR 不可用: {self.ocr_error or '超时'}"))
                self.after(0, lambda: self._finish_read([]))
                return
            total = max(len(hover_frames), 1)
            self.after(0, lambda: self._set_progress("OCR", 0, total))
            for i, frame in enumerate(hover_frames):
                if self._stop.is_set():
                    stopped = True
                    break
                command = ""
                cooldown = None
                if frame is not None:
                    try:
                        texts = ocr_texts(self.ocr, frame)
                        command = command_from_ocr_texts(
                            texts, marker=self.marker_var.get()
                        )
                        cooldown = cooldown_from_ocr_texts(
                            texts, marker=self.cd_marker_var.get()
                        )
                        raw = " | ".join(texts[:24])
                    except Exception as e:
                        command = ""
                        cooldown = None
                        raw = f"[OCR失败] {e}"
                else:
                    raw = ""
                item = {
                    "slot": i + 1,
                    "command": command,
                    "ocr_raw": raw,
                    "_frame": None if frame is None else frame.copy(),
                }
                if cooldown is not None:
                    item[COOLDOWN_KEY] = cooldown_json_value(cooldown)
                slots.append(item)
                self.after(
                    0,
                    lambda it=item, c=i + 1, tot=total: self._on_slot_ready(it, c, tot),
                )
        except Exception as e:
            self.after(0, lambda msg=str(e): self.log(f"OCR 失败: {msg}"))
        self.after(0, lambda: self._finish_read(slots, stopped=stopped))

    def _fill_command_from_ocr(self, idx: int, command: str) -> str:
        row = self._rows[idx]
        row["_ocr_cmd"] = command
        if row["hotbar"].get():
            key = normalize_command(row["hotbar_key"].get())
            if key:
                row["command"].set(key)
                return key
        row["command"].set(command)
        return command

    def _fill_cooldown_from_ocr(self, idx: int, cooldown) -> None:
        if cooldown is None:
            return
        try:
            n = float(cooldown)
        except (TypeError, ValueError):
            return
        self._rows[idx]["cooldown"].set(format_cooldown_s(n))

    def _on_slot_ready(self, item: dict, cur: int, total: int):
        idx = int(item.get("slot", 0)) - 1
        command = item.get("command") or ""
        shown = command
        if 0 <= idx < len(self._rows):
            shown = self._fill_command_from_ocr(idx, command)
            self._fill_cooldown_from_ocr(idx, parse_cooldown_item(item))
        self._set_progress("OCR", cur, total)
        fr = item.get("_frame")
        if fr is not None:
            self._show_live_frame(fr, f"OCR #{item.get('slot')}")
        cd = parse_cooldown_item(item)
        cd_txt = f"{format_cooldown_s(cd)}秒" if cd is not None else "（未识别冷却）"
        self.log(f"#{item.get('slot')} 技能{item.get('slot')}  |  {shown or '（无指令）'}  |  {cd_txt}")
        raw = item.get("ocr_raw") or ""
        if raw:
            self.log(f"    原文 {raw[:220]}")

    def _finish_read(self, slots: list, stopped: bool = False):
        self.busy = False
        self._hide_stop_overlay()
        self.read_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.deiconify()
        self.lift()
        for item in slots:
            idx = int(item.get("slot", 0)) - 1
            if 0 <= idx < len(self._rows):
                self._fill_command_from_ocr(idx, item.get("command") or "")
                self._fill_cooldown_from_ocr(idx, parse_cooldown_item(item))
        self._set_progress("完成", 1, 1)
        if stopped or self._stop.is_set():
            self.log(f"已停止。已识别 {len(slots)} 条。")
        else:
            self.log(f"OCR 完毕。共 {len(slots)} 条。指令或冷却不对就手改，然后保存。")
        # 留下最后一张 OCR 图，不再重新截当前桌面。

    def _collect_skills(self) -> tuple[list[dict], list[str]]:
        out = []
        rejected = []
        for i, row in enumerate(self._rows):
            cmd = normalize_command(row["command"].get())
            if cmd != row["command"].get().strip():
                row["command"].set(cmd)
            if not cmd:
                continue
            if command_has_han(cmd):
                rejected.append(f"技能{i + 1}：{cmd}")
                continue
            item = {"slot": i + 1, "command": cmd}
            if row["hotbar"].get():
                item["hotbar"] = True
            hf = self._hold_frames.get(i + 1)
            if hf is not None:
                item[HOLD_MS_KEY] = max(1, int(hf))
            cd = parse_cooldown_text(row["cooldown"].get())
            if cd is not None:
                item[COOLDOWN_KEY] = cooldown_json_value(cd)
            item[COMBO_KEY] = bool(row["combo"].get())
            on_m = bool(row["multi"].get())
            item[MULTI_KEY] = on_m
            try:
                n = int(str(row["multi_n"].get() or DEFAULT_MULTI))
            except ValueError:
                n = DEFAULT_MULTI
            item[MULTI_N_KEY] = max(DEFAULT_MULTI, min(9, n))
            item[MASH_KEY] = bool(row["mash"].get())
            out.append(item)
        return out, rejected

    def save_binds(self):
        character = self.character_var.get().strip()
        if not character:
            messagebox.showinfo("键位表", "先选角色。")
            return
        try:
            ox, oy, dy, n, dwell = self._hover_params()
        except ValueError:
            messagebox.showinfo("键位表", "悬停参数不是数字。")
            return
        skills, rejected = self._collect_skills()
        if rejected:
            for line in rejected:
                self.log(f"不支持，已跳过 {line}")
            messagebox.showwarning(
                "键位表",
                "下列技能指令含中文，不支持，已抛弃。「空格」会改成 space，不在此列：\n\n"
                + "\n".join(rejected),
            )
        payload = {
            "character": character,
            "updated_at": _now_iso(),
            "source": "dnf_combat_stats",
            "region": dict(self.region) if self.region else None,
            "hover": {"x": ox, "y": oy, "dy": dy, "count": n, "dwell_s": dwell},
            "skills": skills,
        }
        path = bind_path(character)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        payload = merge_bind_payload(existing, payload)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._hold_frames = {}
        kept = 0
        for item in payload.get("skills") or []:
            hf = parse_hold_frames(item)
            slot = _skill_slot(item)
            if hf is not None and slot > 0:
                self._hold_frames[slot] = hf
                kept += 1
        self._save_ui_settings()
        extra = f"，保留 {kept} 条 hold_frames" if kept else ""
        n_cd = sum(1 for it in payload.get("skills") or [] if parse_cooldown_item(it) is not None)
        if n_cd:
            extra += f"，{n_cd} 条 cooldown_s"
        n_combo = sum(1 for it in payload.get("skills") or [] if it.get(COMBO_KEY))
        if n_combo:
            extra += f"，{n_combo} 条连续释放"
        n_multi = sum(1 for it in payload.get("skills") or [] if it.get(MULTI_KEY))
        if n_multi:
            extra += f"，{n_multi} 条多次释放"
        n_mash = sum(1 for it in payload.get("skills") or [] if it.get(MASH_KEY))
        if n_mash:
            extra += f"，{n_mash} 条连按"
        self.log(f"已写 {path}{extra}")

    def load_binds(self):
        character = self.character_var.get().strip()
        path = bind_path(character)
        if not path.is_file():
            messagebox.showinfo("键位表", f"没有 {path}")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        hover = data.get("hover") or {}
        if "x" in hover:
            self.origin_x.set(str(hover["x"]))
        if "y" in hover:
            self.origin_y.set(str(hover["y"]))
        if "dy" in hover:
            self.row_dy.set(str(hover["dy"]))
        if "count" in hover:
            self.count_var.set(str(hover["count"]))
        if "dwell_s" in hover:
            self.dwell_var.set(str(hover["dwell_s"]))
        region = data.get("region")
        if isinstance(region, dict) and region.get("width"):
            self.region = {
                "x": int(region["x"]),
                "y": int(region["y"]),
                "width": int(region["width"]),
                "height": int(region["height"]),
            }
            self.region_label.config(
                text=f"X:{self.region['x']} Y:{self.region['y']}  {self.region['width']}×{self.region['height']}"
            )
        for row in self._rows:
            self._hotbar_guard = True
            row["command"].set("")
            row["hotbar"].set(False)
            row["hotbar_key"].set("")
            row["_ocr_cmd"] = ""
            row["cooldown"].set("")
            row["combo"].set(False)
            row["mash"].set(False)
            row["multi"].set(False)
            row["multi_n"].set(str(DEFAULT_MULTI))
            row["multi_spin"].config(state=tk.DISABLED)
            row["key_entry"].config(state=tk.DISABLED)
            self._hotbar_guard = False
        self._hold_frames = {}
        for item in data.get("skills") or []:
            if not isinstance(item, dict):
                continue
            slot = _skill_slot(item)
            idx = slot - 1
            if 0 <= idx < len(self._rows):
                cmd = normalize_command(str(item.get("command") or item.get("ocr_text") or ""))
                row = self._rows[idx]
                row["command"].set(cmd)
                cd = parse_cooldown_item(item)
                if cd is not None:
                    row["cooldown"].set(format_cooldown_s(cd))
                if item.get("hotbar"):
                    self._hotbar_guard = True
                    row["hotbar"].set(True)
                    row["hotbar_key"].set(cmd)
                    row["key_entry"].config(state=tk.NORMAL)
                    self._hotbar_guard = False
                row["combo"].set(bool(item.get(COMBO_KEY)))
                row["mash"].set(bool(item.get(MASH_KEY)))
                on_m = bool(item.get(MULTI_KEY))
                row["multi"].set(on_m)
                try:
                    n = int(item.get(MULTI_N_KEY) or DEFAULT_MULTI)
                except (TypeError, ValueError):
                    n = DEFAULT_MULTI
                row["multi_n"].set(str(max(DEFAULT_MULTI, min(9, n))))
                row["multi_spin"].config(state=tk.NORMAL if on_m else tk.DISABLED)
            hf = parse_hold_frames(item)
            if hf is not None and slot > 0:
                self._hold_frames[slot] = hf
        n_hf = len(self._hold_frames)
        extra = f"，{n_hf} 条 hold_frames" if n_hf else ""
        n_cd = sum(1 for it in data.get("skills") or [] if parse_cooldown_item(it) is not None)
        if n_cd:
            extra += f"，{n_cd} 条 cooldown_s"
        n_combo = sum(1 for it in data.get("skills") or [] if isinstance(it, dict) and it.get(COMBO_KEY))
        if n_combo:
            extra += f"，{n_combo} 条连续释放"
        n_multi = sum(1 for it in data.get("skills") or [] if isinstance(it, dict) and it.get(MULTI_KEY))
        if n_multi:
            extra += f"，{n_multi} 条多次释放"
        n_mash = sum(1 for it in data.get("skills") or [] if isinstance(it, dict) and it.get(MASH_KEY))
        if n_mash:
            extra += f"，{n_mash} 条连按"
        self.log(f"已载入 {path}{extra}")
        self._save_ui_settings()
        if self.region:
            self.refresh_preview()

    def _save_ui_settings(self):
        if not self._persist_ok:
            return
        try:
            ox, oy, dy, n, dwell = self._hover_params()
        except ValueError:
            return
        data = {
            "character": self.character_var.get().strip(),
            "region": self.region,
            "hover": {"x": ox, "y": oy, "dy": dy, "count": n, "dwell_s": dwell},
            "ocr": {
                "marker": self.marker_var.get().strip() or "操作指令",
                "cd_marker": self.cd_marker_var.get().strip() or "冷却时间",
            },
        }
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_ui_settings(self):
        if not SETTINGS_PATH.is_file():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        ch = data.get("character")
        names = list(self.character_combo["values"] or [])
        if ch and ch in names:
            self.character_var.set(ch)
        hover = data.get("hover") or {}
        for src, var in (
            ("x", self.origin_x),
            ("y", self.origin_y),
            ("dy", self.row_dy),
            ("count", self.count_var),
            ("dwell_s", self.dwell_var),
        ):
            if src in hover:
                var.set(str(hover[src]))
        ocr_ui = data.get("ocr") or {}
        marker = ocr_ui.get("marker")
        if isinstance(marker, str) and marker.strip():
            self.marker_var.set(marker.strip())
        cd_marker = ocr_ui.get("cd_marker")
        if isinstance(cd_marker, str) and cd_marker.strip():
            self.cd_marker_var.set(cd_marker.strip())
        region = data.get("region")
        if isinstance(region, dict) and region.get("width"):
            self.region = {
                "x": int(region["x"]),
                "y": int(region["y"]),
                "width": int(region["width"]),
                "height": int(region["height"]),
            }
            self.region_label.config(
                text=f"X:{self.region['x']} Y:{self.region['y']}  {self.region['width']}×{self.region['height']}"
            )

    def _on_close(self):
        self._stop.set()
        self._hide_stop_overlay()
        self._persist_ok = True
        remember_window_geom(self, "skill_bind")
        self._save_ui_settings()
        self.destroy()


if __name__ == "__main__":
    app = SkillBindTool()
    app.mainloop()
