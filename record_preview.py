"""录制时另开两个轻量预览窗：按键时间线 + YOLO 画面/计数。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from PIL import Image, ImageTk

from window_geom import apply as apply_window_geom
from window_geom import remember as remember_window_geom


class RecordPreviewWindows:
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.keys_win: tk.Toplevel | None = None
        self.yolo_win: tk.Toplevel | None = None
        self.keys_list: tk.Listbox | None = None
        self.yolo_label: ttk.Label | None = None
        self.yolo_canvas: tk.Label | None = None
        self._photo = None
        self._key_count = 0
        self._closed = True

    @property
    def is_open(self) -> bool:
        return (not self._closed) and (self.keys_win is not None or self.yolo_win is not None)

    def open(self, keys: bool = True, yolo: bool = True):
        self.close()
        self._closed = False
        self._key_count = 0

        if keys:
            self.keys_win = tk.Toplevel(self.parent)
            self.keys_win.title("采集 · 按键")
            self.keys_win.geometry("320x420+40+80")
            self.keys_win.attributes("-topmost", True)
            ttk.Label(self.keys_win, text="实时按键（press / release）").pack(anchor=tk.W, padx=8, pady=(8, 4))
            self.keys_list = tk.Listbox(self.keys_win, font=("Consolas", 10), height=18)
            self.keys_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            self.keys_win.protocol("WM_DELETE_WINDOW", lambda: self._close_one("keys"))
            apply_window_geom(self.keys_win, "record_keys", min_w=240, min_h=180, fallback="320x420+40+80")

        if yolo:
            self.yolo_win = tk.Toplevel(self.parent)
            self.yolo_win.title("YOLO 预览")
            self.yolo_win.geometry("520x420+380+80")
            self.yolo_win.attributes("-topmost", True)
            self.yolo_label = ttk.Label(self.yolo_win, text="等待第一帧推理…", font=("Microsoft YaHei", 10))
            self.yolo_label.pack(anchor=tk.W, padx=8, pady=(8, 4))
            self.yolo_canvas = tk.Label(self.yolo_win, bg="#111")
            self.yolo_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            self.yolo_win.protocol("WM_DELETE_WINDOW", lambda: self._close_one("yolo"))
            apply_window_geom(self.yolo_win, "record_yolo", min_w=360, min_h=240, fallback="520x420+380+80")

    def _close_one(self, which: str):
        if which == "keys":
            if self.keys_win is not None:
                remember_window_geom(self.keys_win, "record_keys")
            win, self.keys_win, self.keys_list = self.keys_win, None, None
        else:
            if self.yolo_win is not None:
                remember_window_geom(self.yolo_win, "record_yolo")
            win, self.yolo_win, self.yolo_label, self.yolo_canvas = (
                self.yolo_win,
                None,
                None,
                None,
            )
            self._photo = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass
        if self.keys_win is None and self.yolo_win is None:
            self._closed = True

    def close(self):
        self._closed = True
        self._photo = None
        if self.keys_win is not None:
            remember_window_geom(self.keys_win, "record_keys")
        if self.yolo_win is not None:
            remember_window_geom(self.yolo_win, "record_yolo")
        wins = (self.keys_win, self.yolo_win)
        self.keys_win = None
        self.yolo_win = None
        self.keys_list = None
        self.yolo_label = None
        self.yolo_canvas = None
        for win in wins:
            if win is not None:
                try:
                    win.destroy()
                except tk.TclError:
                    pass

    def push_key(self, action: str, key: str):
        if self.keys_list is None:
            return
        self._key_count += 1
        self.keys_list.insert(tk.END, f"{self._key_count:04d}  {action:<7}  {key}")
        if int(self.keys_list.size()) > 200:
            self.keys_list.delete(0, 50)
        self.keys_list.see(tk.END)

    def push_yolo(self, features: dict[str, Any], bgr_small):
        if self.yolo_label is not None:
            xy = features.get("player_xy")
            xy_s = f"({xy[0]:.0f},{xy[1]:.0f})" if xy else "—"
            text = (
                f"player={xy_s}  mon={features.get('mon', 0)}  loot={features.get('loot', 0)}  "
                f"gate={features.get('gate', 0)}  boss={features.get('boss', 0)}  "
                f"{features.get('infer_ms', 0):.0f}ms"
            )
            gxy = features.get("gate_xy") or []
            if gxy:
                text += f"  gate_xy={gxy[0]}"
            self.yolo_label.config(text=text)
        if bgr_small is None or self.yolo_canvas is None:
            return
        try:
            rgb = bgr_small[:, :, ::-1]
            if getattr(rgb, "copy", None):
                rgb = rgb.copy()
            img = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(img)
            self.yolo_canvas.config(image=self._photo)
        except Exception:
            pass
