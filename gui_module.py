# 必须先恢复提权 PATH、关掉系统 DLL 错误框，再碰 CUDA
import os
import sys
import json
import subprocess
from pathlib import Path

FSM_UI_PATH = Path(__file__).resolve().parent / "_fsm_replay_ui.json"

if not getattr(sys, "frozen", False):
    os.chdir(Path(__file__).resolve().parent)

from admin_runtime import apply_saved_env, is_admin, restart_as_admin, suppress_windows_error_dialogs

apply_saved_env()
suppress_windows_error_dialogs()

from cuda_runtime import ensure_cuda_runtime
ensure_cuda_runtime()

import tkinter as tk
from tkinter import ttk, font, messagebox
import threading
from central_controller import CentralController
from window_align import enable_dpi_awareness, get_virtual_screen, get_window_at_point, find_window_by_process
from window_geom import apply as apply_window_geom
from window_geom import remember as remember_window_geom
from fsm_core import MON_CORR_MAX, mon_corr_from_dict, mon_off_from_corr, DEFAULT_PW_MS, DEFAULT_TOWN_S, json_ms, json_s_from_frames
from skill_feature_extract import DEFAULT_E, DEFAULT_F, skill_table_missing


class WindowSelector(tk.Toplevel):
    """全屏遮罩：点击目标窗口后自动对齐到该窗口客户区。"""
    def __init__(self, parent, on_selection_complete):
        super().__init__(parent)
        self.on_selection_complete = on_selection_complete
        self.exclude_hwnds = {int(parent.winfo_id()), int(self.winfo_id())}
        self.screen = get_virtual_screen()
        self.aligned = None
        self.rect_id = None
        self.label_id = None

        self.geometry(
            f"{self.screen['width']}x{self.screen['height']}+{self.screen['x']}+{self.screen['y']}"
        )
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        self.attributes('-alpha', 0.28)

        self.canvas = tk.Canvas(self, cursor="hand2", bg="gray", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        hint = "点击游戏窗口自动对齐  |  ESC 取消"
        self.canvas.create_text(
            self.screen['width'] // 2,
            40,
            text=hint,
            fill="white",
            font=("Microsoft YaHei", 16, "bold"),
        )

        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.bind("<Escape>", self.cancel_selection)
        self.focus_force()

    def _screen_to_canvas(self, x, y):
        return x - self.screen['x'], y - self.screen['y']

    def on_click(self, event):
        screen_x = event.x_root
        screen_y = event.y_root

        self.withdraw()
        self.update_idletasks()
        aligned = get_window_at_point(screen_x, screen_y, exclude_hwnds=self.exclude_hwnds)
        self.deiconify()
        self.attributes('-topmost', True)
        self.lift()
        self.focus_force()

        if not aligned:
            return

        self.aligned = aligned
        self._draw_aligned_rect(aligned)
        self.on_selection_complete(aligned)
        self.destroy()

    def _draw_aligned_rect(self, aligned):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.label_id:
            self.canvas.delete(self.label_id)
        x1, y1 = self._screen_to_canvas(aligned['x'], aligned['y'])
        x2, y2 = self._screen_to_canvas(aligned['x'] + aligned['width'], aligned['y'] + aligned['height'])
        self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline='#00ff66', width=3)
        self.label_id = self.canvas.create_text(
            x1 + 8,
            y1 + 18,
            text=aligned.get('title', ''),
            anchor='w',
            fill='#00ff66',
            font=("Microsoft YaHei", 12, "bold"),
        )
        self.update_idletasks()

    def cancel_selection(self, event=None):
        self.destroy()


def default_ocr_region(game_w: int, game_h: int) -> dict:
    """未框选时：游戏客户区右上角（地下城名常见位置）。"""
    w = max(int(game_w), 1)
    h = max(int(game_h), 1)
    rw = max(int(w * 0.40), 80)
    rh = max(int(h * 0.28), 60)
    return {"x": max(w - rw, 0), "y": 0, "width": rw, "height": rh}


class OcrRegionSelector(tk.Toplevel):
    """在已对齐的游戏窗口上拖拽矩形，得到相对客户区的 OCR 裁剪区。"""

    def __init__(self, parent, game_rect, on_done, frame_bgr=None):
        super().__init__(parent)
        self.on_done = on_done
        self.game_rect = game_rect
        self.drag_start = None
        self.rect_id = None
        self._photo = None

        self.geometry(
            f"{game_rect['width']}x{game_rect['height']}+{game_rect['x']}+{game_rect['y']}"
        )
        self.overrideredirect(True)
        self.attributes("-topmost", True)

        self.canvas = tk.Canvas(self, cursor="crosshair", bg="#123", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        if frame_bgr is not None:
            from PIL import Image, ImageTk

            rgb = frame_bgr[:, :, ::-1]
            if getattr(rgb, "copy", None):
                rgb = rgb.copy()
            img = Image.fromarray(rgb)
            gw = int(game_rect["width"])
            gh = int(game_rect["height"])
            if img.size != (gw, gh):
                img = img.resize((gw, gh), Image.Resampling.NEAREST)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
            self.attributes("-alpha", 1.0)
        else:
            self.attributes("-alpha", 0.32)
        self.canvas.create_text(
            game_rect["width"] // 2,
            28,
            text="拖拽框选 OCR 区域（含右上角地名+小地图，不要只框一行字）  |  ESC 取消",
            fill="white",
            font=("Microsoft YaHei", 14, "bold"),
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
        x = max(0, min(x0, x1))
        y = max(0, min(y0, y1))
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        gw = int(self.game_rect["width"])
        gh = int(self.game_rect["height"])
        w = min(w, gw - x)
        h = min(h, gh - y)
        if w < 80 or h < 60:
            self.drag_start = None
            return
        self.on_done({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
        self.destroy()

    def cancel(self, event=None):
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("自动化控制面板")
        self.geometry("540x780")
        self.region_coords = None
        self.window_title = ""
        self.ocr_region = None

        self.controller = CentralController(self)
        self.record_preview = None

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Microsoft YaHei", size=10)
        self.option_add("*Font", default_font)

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.minsize(520, 720)

        live_group = ttk.LabelFrame(main_frame, text="实时状态", padding="8")
        live_group.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        char_row = ttk.Frame(live_group)
        char_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(char_row, text="当前角色", width=10).pack(side=tk.LEFT)
        self.character_var = tk.StringVar(value="")
        self.character_combo = ttk.Combobox(
            char_row,
            textvariable=self.character_var,
            state="readonly",
        )
        self.character_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.character_refresh_btn = ttk.Button(char_row, text="刷新", width=6, command=self._refresh_characters)
        self.character_refresh_btn.pack(side=tk.LEFT)
        self.character_combo.bind("<<ComboboxSelected>>", self._on_character_selected)

        state_row = ttk.Frame(live_group)
        state_row.pack(fill=tk.X)
        ttk.Label(state_row, text="当前状态", width=10).pack(side=tk.LEFT)
        self.live_state_var = tk.StringVar(value="城镇")
        self.live_state_label = ttk.Label(
            state_row, textvariable=self.live_state_var, foreground="#06c", font=("Microsoft YaHei", 12, "bold")
        )
        self.live_state_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._wrap_label(self.live_state_label)

        perm_row = ttk.Frame(live_group)
        perm_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(perm_row, text="运行权限", width=10).pack(side=tk.LEFT)
        self.live_admin_var = tk.StringVar(value="")
        self.live_admin_label = ttk.Label(
            perm_row, textvariable=self.live_admin_var, font=("Microsoft YaHei", 10, "bold")
        )
        self.live_admin_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._wrap_label(self.live_admin_label)

        fsm_row = ttk.Frame(live_group)
        fsm_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(fsm_row, text="FSM状态", width=10).pack(side=tk.LEFT)
        self.live_fsm_var = tk.StringVar(value="—")
        self.live_fsm_label = ttk.Label(
            fsm_row, textvariable=self.live_fsm_var, foreground="#0a7", font=("Microsoft YaHei", 12, "bold")
        )
        self.live_fsm_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._wrap_label(self.live_fsm_label)

        log_group = ttk.LabelFrame(main_frame, text="状态日志", padding="10")

        self.log_text = tk.Text(log_group, height=12, state=tk.DISABLED)
        self.log_max_lines = 200
        self.log_text.pack(fill=tk.BOTH, expand=True)

        settings_group = ttk.LabelFrame(main_frame, text="游戏窗口与间隔", padding="8")

        top_frame = ttk.Frame(settings_group)
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="游戏窗口", width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(top_frame, text="对齐窗口", command=self.select_region).pack(side=tk.LEFT)
        self.region_status_label = ttk.Label(top_frame, text="[未设置]", foreground="blue")
        self.region_status_label.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        self._wrap_label(self.region_status_label)

        bottom_frame = ttk.Frame(settings_group)
        bottom_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(bottom_frame, text="截图间隔(秒):", width=15).pack(side=tk.LEFT, padx=(0, 5))
        self.interval_entry = ttk.Entry(bottom_frame, width=8)
        self.interval_entry.insert(0, "0.05")
        self.interval_entry.pack(side=tk.LEFT)
        self.apply_interval_btn = ttk.Button(bottom_frame, text="应用", width=6, command=self.apply_interval)
        self.apply_interval_btn.pack(side=tk.LEFT, padx=5)

        mode_group = ttk.LabelFrame(main_frame, text="模式设置", padding="8")

        mode_row = ttk.Frame(mode_group)
        mode_row.pack(fill=tk.X)
        self.mode_var = tk.StringVar(value="collect")
        self.record_rb = ttk.Radiobutton(
            mode_row, text="采集模式", variable=self.mode_var, value="collect", command=self._sync_exclusive_modes
        )
        self.record_rb.pack(anchor=tk.W, side=tk.LEFT, padx=5)
        self.yolo_mode_rb = ttk.Radiobutton(
            mode_row, text="YOLO测试", variable=self.mode_var, value="yolo_test", command=self._sync_exclusive_modes
        )
        self.yolo_mode_rb.pack(anchor=tk.W, side=tk.LEFT, padx=5)
        self.automation_rb = ttk.Radiobutton(
            mode_row, text="自动化模式", variable=self.mode_var, value="automation", command=self._sync_exclusive_modes
        )
        self.automation_rb.pack(anchor=tk.W, side=tk.LEFT, padx=5)

        self.yolo_test_var = tk.BooleanVar(value=False)
        self.yolo_test_cb = ttk.Checkbutton(
            mode_row,
            text="FSM测试",
            variable=self.yolo_test_var,
            command=self._on_yolo_test_toggle,
        )
        self.yolo_test_cb.pack(anchor=tk.W, side=tk.LEFT, padx=(15, 5))

        self.fsm_settings_btn = ttk.Button(mode_row, text="FSM设置", command=self._open_fsm_settings)
        self.fsm_settings_btn.pack(anchor=tk.W, side=tk.LEFT, padx=(12, 0))

        self.fsm_m_var = tk.StringVar(value="5")
        self.fsm_l_var = tk.StringVar(value="5")
        self.fsm_g_var = tk.StringVar(value="5")
        self.fsm_x_var = tk.StringVar(value="1.5")
        self.fsm_gx_var = tk.StringVar(value="50")
        self.fsm_gy_var = tk.StringVar(value="10")
        self.fsm_ax_var = tk.StringVar(value="250")
        self.fsm_ay_var = tk.StringVar(value="250")
        self.fsm_y_var = tk.StringVar(value="250")
        self.fsm_s_var = tk.StringVar(value="20")
        self.fsm_xxx_var = tk.StringVar(value="1000")
        self.fsm_pm_var = tk.StringVar(value="10")
        self.fsm_pc_var = tk.StringVar(value="5")
        self.fsm_pt_var = tk.StringVar(value="3")
        self.fsm_pw_var = tk.StringVar(value=str(DEFAULT_PW_MS))
        self.fsm_e_var = tk.StringVar(value=str(DEFAULT_E))
        self.fsm_f_var = tk.StringVar(value=str(DEFAULT_F))
        self.fsm_town_s_var = tk.StringVar(value=str(int(DEFAULT_TOWN_S)))
        self.fsm_tap_ms_var = tk.StringVar(value="50")
        self.fsm_mash_count_var = tk.StringVar(value="3")
        self.fsm_mash_gap_var = tk.StringVar(value="50")
        self.fsm_th_var = tk.StringVar(value="50")
        self.fsm_corr_u = tk.IntVar(value=0)
        self.fsm_corr_d = tk.IntVar(value=0)
        self.fsm_corr_l = tk.IntVar(value=0)
        self.fsm_corr_r = tk.IntVar(value=0)
        self._fsm_corr_lock = False
        self._fsm_corr_prev = (0, 0, 0, 0)
        self._fsm_settings_win = None
        self._fsm_corr_summary = None
        self._fsm_mlg_persist = False
        self._load_fsm_mlg_vars()

        self._fsm_mlg_persist = True
        for var in (
            self.fsm_m_var,
            self.fsm_l_var,
            self.fsm_g_var,
            self.fsm_x_var,
            self.fsm_gx_var,
            self.fsm_gy_var,
            self.fsm_ax_var,
            self.fsm_ay_var,
            self.fsm_y_var,
            self.fsm_s_var,
            self.fsm_xxx_var,
            self.fsm_pm_var,
            self.fsm_pc_var,
            self.fsm_pt_var,
            self.fsm_pw_var,
            self.fsm_e_var,
            self.fsm_f_var,
            self.fsm_town_s_var,
            self.fsm_tap_ms_var,
            self.fsm_mash_count_var,
            self.fsm_mash_gap_var,
            self.fsm_th_var,
        ):
            var.trace_add("write", lambda *_: self._on_fsm_mlg_edit())
        for cvar in (self.fsm_corr_u, self.fsm_corr_d, self.fsm_corr_l, self.fsm_corr_r):
            cvar.trace_add("write", lambda *_: self._on_fsm_corr_edit())

        yolo_group = ttk.LabelFrame(main_frame, text="YOLO 设置", padding="8")

        ver_row = ttk.Frame(yolo_group)
        ver_row.pack(fill=tk.X)
        ttk.Label(ver_row, text="模型版本", width=10).pack(side=tk.LEFT)
        self.yolo_version_var = tk.StringVar(value="")
        self.yolo_version_combo = ttk.Combobox(
            ver_row,
            textvariable=self.yolo_version_var,
            state="readonly",
        )
        self.yolo_version_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(ver_row, text="刷新", width=6, command=self._refresh_yolo_versions).pack(side=tk.LEFT)

        param_row = ttk.Frame(yolo_group)
        param_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(param_row, text="conf", width=10).pack(side=tk.LEFT)
        self.yolo_conf_entry = ttk.Entry(param_row, width=8)
        self.yolo_conf_entry.insert(0, "0.1")
        self.yolo_conf_entry.pack(side=tk.LEFT)
        ttk.Label(param_row, text="iou", width=6).pack(side=tk.LEFT, padx=(12, 0))
        self.yolo_iou_entry = ttk.Entry(param_row, width=8)
        self.yolo_iou_entry.insert(0, "0.7")
        self.yolo_iou_entry.pack(side=tk.LEFT)

        ocr_group = ttk.LabelFrame(main_frame, text="OCR 设置", padding="8")

        ocr_int_row = ttk.Frame(ocr_group)
        ocr_int_row.pack(fill=tk.X)
        ttk.Label(ocr_int_row, text="识别间隔(秒)", width=12).pack(side=tk.LEFT)
        self.ocr_interval_entry = ttk.Entry(ocr_int_row, width=8)
        self.ocr_interval_entry.insert(0, "0.3")
        self.ocr_interval_entry.pack(side=tk.LEFT)
        ttk.Label(ocr_int_row, text="conf", width=5).pack(side=tk.LEFT, padx=(12, 0))
        self.ocr_conf_entry = ttk.Entry(ocr_int_row, width=8)
        self.ocr_conf_entry.insert(0, "0.95")
        self.ocr_conf_entry.pack(side=tk.LEFT)
        ttk.Label(ocr_int_row, text="关键词最低分", foreground="#666").pack(side=tk.LEFT, padx=8)

        ocr_reg_row = ttk.Frame(ocr_group)
        ocr_reg_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(ocr_reg_row, text="识别区域", width=12).pack(side=tk.LEFT)
        self.ocr_select_btn = ttk.Button(ocr_reg_row, text="框选", width=6, command=self.select_ocr_region)
        self.ocr_select_btn.pack(side=tk.LEFT)
        self.apply_ocr_btn = ttk.Button(ocr_reg_row, text="应用", width=6, command=self.apply_ocr_settings)
        self.apply_ocr_btn.pack(side=tk.LEFT, padx=5)
        self.ocr_region_label = ttk.Label(ocr_reg_row, text="[未框选·将用右上角]", foreground="blue")
        self.ocr_region_label.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        self._wrap_label(self.ocr_region_label)

        control_frame = ttk.Frame(main_frame, padding=(0, 5, 0, 0))
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.admin_button = ttk.Button(control_frame, text="以管理员身份重启", command=self.restart_elevated)
        self.admin_button.pack(side=tk.LEFT)
        ttk.Button(control_frame, text="FSM回放", command=lambda: self._launch_tool("fsm_replay.py")).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(control_frame, text="键位表", command=lambda: self._launch_tool("skill_bind_tool.py")).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        self.stop_button = ttk.Button(control_frame, text="停止", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=(0, 5))
        self.start_button = ttk.Button(control_frame, text="开始", command=self.start)
        self.start_button.pack(side=tk.RIGHT, padx=5)

        ocr_group.pack(side=tk.BOTTOM, fill=tk.X, pady=3)
        yolo_group.pack(side=tk.BOTTOM, fill=tk.X, pady=3)
        mode_group.pack(side=tk.BOTTOM, fill=tk.X, pady=3)
        settings_group.pack(side=tk.BOTTOM, fill=tk.X, pady=3, ipady=2)
        log_group.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self._yolo_weight_map = {}
        self._refresh_yolo_versions(select_default="solarwarden_b")
        self._refresh_characters()

        self.controller.load_config()
        self.log("GUI初始化完成。")
        self.log(f"找到项目路径: {os.getcwd()}")
        self.log("中央控制器已加载。")
        self._refresh_admin_status()
        self._auto_align_dnf()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after_idle(lambda: apply_window_geom(self, "main", min_w=520, min_h=720))

    @staticmethod
    def _wrap_label(label: ttk.Label):
        def _on_cfg(event, lab=label):
            w = int(event.width) - 4
            if w >= 40:
                lab.configure(wraplength=w)

        label.bind("<Configure>", _on_cfg)

    def _refresh_admin_status(self):
        if is_admin():
            self.live_admin_var.set("管理员（可采集按键）")
            self.live_admin_label.config(foreground="#0a5")
            self.admin_button.config(state=tk.DISABLED)
            if not self.title().endswith(" [管理员]"):
                self.title(self.title() + " [管理员]")
            self.log("当前进程已是管理员。")
        else:
            self.live_admin_var.set("普通用户 — 采集会被拦截，请点左下角提权重启")
            self.live_admin_label.config(foreground="#c60")
            self.admin_button.config(state=tk.NORMAL)
            self.log("当前不是管理员：采集模式会直接拒绝开始。请点「以管理员身份重启」。")

    def _launch_tool(self, script: str):
        root = Path(__file__).resolve().parent
        path = root / script
        if not path.is_file():
            self.log(f"找不到 {path}")
            return
        kwargs = {"cwd": str(root)}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        try:
            subprocess.Popen([sys.executable, str(path)], **kwargs)
            self.log(f"已启动 {script}")
        except Exception as e:
            self.log(f"启动 {script} 失败: {e}")

    def restart_elevated(self):
        if is_admin():
            self.log("已经是管理员，无需重启。")
            return
        if str(self.start_button.cget("state")) == "disabled":
            self.log("请先点「停止」再提权重启，避免录制中途断开。")
            return
        ok, msg = restart_as_admin(cwd=os.getcwd())
        self.log(msg)
        if ok:
            remember_window_geom(self, "main")
            try:
                self.destroy()
            except Exception:
                pass
            os._exit(0)

    def _on_close(self):
        remember_window_geom(self, "main")
        try:
            self.destroy()
        except Exception:
            pass

    def _auto_align_dnf(self):
        aligned = find_window_by_process("dnf.exe")
        if not aligned:
            self.log("未找到 DNF.exe 窗口，将使用已保存坐标；也可手动点「对齐窗口」。")
            return
        self.log("启动时已自动对齐 DNF.exe 窗口。")
        self._on_region_selected(aligned)

    def _refresh_yolo_versions(self, select_default=None):
        """扫描 D:/Atrain/runs/*/weights/best.pt，填充版本下拉框。"""
        runs_root = Path(r"D:/Atrain/runs")
        weight_map = {}
        if runs_root.is_dir():
            for best in sorted(runs_root.glob("*/weights/best.pt")):
                weight_map[best.parent.parent.name] = str(best)

        self._yolo_weight_map = weight_map
        names = list(weight_map.keys())
        self.yolo_version_combo["values"] = names

        current = self.yolo_version_var.get()
        if select_default and select_default in weight_map:
            self.yolo_version_var.set(select_default)
        elif current in weight_map:
            self.yolo_version_var.set(current)
        elif names:
            skip_default = {n.lower() for n in names if n.lower().startswith("varien")}
            preferred = (
                [n for n in names if n.lower() == "solarwarden_b"]
                or [n for n in names if n.lower() == "solarwarden"]
                or [n for n in names if n.lower() not in skip_default]
                or names
            )
            self.yolo_version_var.set(preferred[-1] if preferred else names[-1])
        else:
            self.yolo_version_var.set("")

        if hasattr(self, "log_text"):
            if names:
                self.log(f"YOLO 可用版本: {', '.join(names)}")
            else:
                self.log("警告: 未在 D:/Atrain/runs 下找到任何 best.pt。")

    def _selected_yolo_weights(self):
        name = self.yolo_version_var.get().strip()
        return self._yolo_weight_map.get(name)

    def _refresh_characters(self, select_name=None):
        """从 character_names_custom.txt 填充角色下拉框。"""
        path = Path("character_names_custom.txt")
        names = []
        if path.is_file():
            names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.character_combo["values"] = names
        current = (select_name or self.character_var.get() or "").strip()
        if current in names:
            self.character_var.set(current)
        elif names:
            self.character_var.set(names[0])
        else:
            self.character_var.set("")
        self._sync_character_to_controller()
        if hasattr(self, "log_text"):
            if names:
                self.log(f"角色列表: {', '.join(names)}")
            else:
                self.log("警告: character_names_custom.txt 为空，请先填写角色名。")

    def _on_character_selected(self, event=None):
        self._sync_character_to_controller()
        name = self.character_var.get().strip()
        if name:
            self.log(f"已选择角色: {name}")
            self.controller.update_and_save_config()

    def _sync_character_to_controller(self):
        if hasattr(self, "controller"):
            self.controller.current_character_name = self.character_var.get().strip() or None

    def is_collect_mode(self) -> bool:
        if self.yolo_test_var.get():
            return False
        return self.mode_var.get() in ("collect", "record")

    def is_yolo_test_mode(self) -> bool:
        if self.yolo_test_var.get():
            return False
        return self.mode_var.get() == "yolo_test"

    def is_fsm_test_mode(self) -> bool:
        return bool(self.yolo_test_var.get())

    def _on_yolo_test_toggle(self):
        if self.yolo_test_var.get():
            self._refresh_yolo_versions()
            self._load_fsm_mlg_vars()
        self._sync_exclusive_modes()

    def _sync_exclusive_modes(self):
        exclusive = self.yolo_test_var.get()
        mode_state = tk.DISABLED if exclusive else tk.NORMAL
        self.record_rb.config(state=mode_state)
        self.yolo_mode_rb.config(state=mode_state)
        self.automation_rb.config(state=mode_state)

        if self.is_fsm_test_mode():
            ver = self.yolo_version_var.get() or "(未选择)"
            self.log("已开启「FSM测试」：实时 YOLO + FSM + 发键；进图后写盘到 FSM_TEST（YOLO jsonl + 键盘 + 截图 + 当时参数）。城镇不录。")
            self.log(f"提示: 采集间隔默认 0.05s；调 debounce 请与采集同档，不要放到 0.2～0.5。当前版本 {ver}。")
            if ver.lower() != "solarwarden_b":
                self.log("过图检测请点「刷新」后选 solarwarden_b；varien_t 仅自动标注，不要当过图 YOLO。")
        else:
            if self.is_yolo_test_mode():
                ver = self.yolo_version_var.get() or "(未选择)"
                self.log("已选「YOLO测试」：只看检测框和新图识别，不跑 FSM、不存图、不采集键盘。")
                self.log(f"当前模型 {ver}。过图请用 solarwarden_b。")
            elif self.is_collect_mode():
                self.log("采集模式：截图 + 键盘 + YOLO。开始时必须已是管理员，并选过图权重。")
            else:
                self.log("自动化模式：等 OCR 进图后占位，不采集。")

    def select_region(self):
        self.log("请点击游戏窗口以自动对齐...(按ESC取消)")
        selector = WindowSelector(self, self._on_region_selected)
        self.withdraw()
        self.wait_window(selector)
        self.deiconify()
        self.lift()

    def _on_region_selected(self, rect):
        self.region_coords = {
            'x': rect['x'],
            'y': rect['y'],
            'width': rect['width'],
            'height': rect['height'],
        }
        self.window_title = rect.get('title', '')
        self._refresh_region_label()
        title_hint = f" ({self.window_title})" if self.window_title else ""
        self.log(
            f"已对齐窗口{title_hint}: "
            f"X:{rect['x']}, Y:{rect['y']}, W:{rect['width']}, H:{rect['height']}"
        )
        self.controller.update_and_save_config()

    def _refresh_region_label(self):
        if not isinstance(self.region_coords, dict):
            self.region_status_label.config(text="[未设置]")
            return
        coords = self.region_coords
        title = f" {self.window_title}" if self.window_title else ""
        self.region_status_label.config(
            text=f"{title} X:{coords['x']}, Y:{coords['y']}, W:{coords['width']}, H:{coords['height']}".strip()
        )

    def _refresh_ocr_region_label(self):
        r = self.ocr_region
        if not isinstance(r, dict) or r.get("width", 0) <= 0:
            self.ocr_region_label.config(text="[未框选·将用右上角]")
            return
        self.ocr_region_label.config(
            text=f"相对窗口  X:{r['x']} Y:{r['y']} W:{r['width']} H:{r['height']}"
        )

    def select_ocr_region(self):
        if not isinstance(self.region_coords, dict) or self.region_coords.get("width", 0) <= 0:
            self.log("请先对齐游戏窗口，再框选 OCR 区域。")
            return
        self.log("请在游戏画面上拖拽 OCR 区域（点「应用」后写入配置）...")
        frame_bgr = None
        try:
            cap = getattr(self.controller, "capture_module", None)
            if cap is not None:
                result = cap.capture(rect=self.region_coords, gui=None, save=False)
                if result and result.get("frame") is not None:
                    frame_bgr = result["frame"]
        except Exception as e:
            self.log(f"框选预览截图失败，改用半透明遮罩: {e}")
        selector = OcrRegionSelector(
            self, self.region_coords, self._on_ocr_region_selected, frame_bgr=frame_bgr
        )
        self.wait_window(selector)
        self.lift()

    def _on_ocr_region_selected(self, rect):
        self.ocr_region = {
            "x": int(rect["x"]),
            "y": int(rect["y"]),
            "width": int(rect["width"]),
            "height": int(rect["height"]),
        }
        self._refresh_ocr_region_label()
        self.log(
            f"已框选 OCR 区域（相对游戏窗口）: "
            f"X:{self.ocr_region['x']} Y:{self.ocr_region['y']} "
            f"W:{self.ocr_region['width']} H:{self.ocr_region['height']} — 请点「应用」保存"
        )
        self._push_ocr_live()

    def _effective_ocr_region(self):
        if isinstance(self.ocr_region, dict) and self.ocr_region.get("width", 0) > 0:
            return dict(self.ocr_region)
        game = self.region_coords if isinstance(self.region_coords, dict) else None
        if not game:
            return None
        return default_ocr_region(int(game.get("width", 0)), int(game.get("height", 0)))

    def apply_interval(self):
        try:
            val = float(self.interval_entry.get())
            self.log(f"已应用: 截图间隔 {val:g}s（录制/YOLO 用）")
            self.controller.update_and_save_config()
        except ValueError:
            self.log(f"错误: 间隔 '{self.interval_entry.get()}' 不是有效数字。")

    def apply_ocr_settings(self):
        try:
            ocr_interval = float(self.ocr_interval_entry.get())
            if ocr_interval < 0.05:
                ocr_interval = 0.05
                self.ocr_interval_entry.delete(0, tk.END)
                self.ocr_interval_entry.insert(0, "0.05")
        except ValueError:
            self.log(f"错误: OCR 间隔 '{self.ocr_interval_entry.get()}' 不是有效数字。")
            return
        try:
            ocr_conf = float(self.ocr_conf_entry.get())
            ocr_conf = min(1.0, max(0.0, ocr_conf))
            self.ocr_conf_entry.delete(0, tk.END)
            self.ocr_conf_entry.insert(0, str(ocr_conf))
        except ValueError:
            self.log(f"错误: OCR conf '{self.ocr_conf_entry.get()}' 不是有效数字。")
            return
        ocr = self._effective_ocr_region()
        if ocr and not (isinstance(self.ocr_region, dict) and self.ocr_region.get("width", 0) > 0):
            self.ocr_region = ocr
            self._refresh_ocr_region_label()
        self.log(f"已应用: OCR 间隔 {ocr_interval:g}s  conf>={ocr_conf:g}")
        if ocr:
            self.log(
                f"已应用: OCR 区域 相对窗口 X:{ocr['x']} Y:{ocr['y']} "
                f"W:{ocr['width']} H:{ocr['height']}（建议含右上角地名和小地图）"
            )
        self.controller.update_and_save_config()
        self._push_ocr_live(ocr_interval, ocr_conf)

    def _read_ocr_conf(self) -> float:
        try:
            return min(1.0, max(0.0, float(self.ocr_conf_entry.get())))
        except ValueError:
            return 0.95

    def _push_ocr_live(self, ocr_interval=None, ocr_conf=None):
        analyzer = getattr(self.controller, "status_analyzer", None)
        if analyzer is None or not getattr(analyzer, "is_running", False):
            return
        ocr = self._effective_ocr_region()
        analyzer.ocr_region = dict(ocr) if ocr else None
        if ocr_interval is None:
            try:
                ocr_interval = float(self.ocr_interval_entry.get())
            except ValueError:
                ocr_interval = 0.3
        analyzer.ocr_interval = max(0.05, float(ocr_interval))
        if ocr_conf is None:
            ocr_conf = self._read_ocr_conf()
        analyzer.ocr_conf = float(ocr_conf)
        self.log(
            f"已同步运行中 OCR: 间隔 {analyzer.ocr_interval:g}s  conf>={analyzer.ocr_conf:g}"
            + (
                f"  区域 X:{ocr['x']} Y:{ocr['y']} W:{ocr['width']} H:{ocr['height']}"
                if ocr
                else ""
            )
        )

    def apply_initial_config(self, config):
        """应用配置；兼容旧版 A/B 分区字段。"""
        self.log("GUI: 正在应用初始配置...")
        try:
            region = None
            title = ""
            interval = None

            if isinstance(config.get('region'), dict):
                region = config['region'].get('rect')
                title = config['region'].get('window_title', "") or ""
            if interval is None and 'interval' in config:
                interval = config['interval']

            if not isinstance(region, dict):
                candidates = []
                for key in ('A', 'B'):
                    if isinstance(config.get(key), dict) and isinstance(config[key].get('rect'), dict):
                        candidates.append(config[key]['rect'])
                if candidates:
                    region = max(candidates, key=lambda r: r.get('width', 0) * r.get('height', 0))
                    self.log("GUI: 已将旧版 A/B 区域合并为单一游戏窗口。")

            if interval is None:
                interval = config.get('interval_A', config.get('interval_B', 0.05))

            if isinstance(region, dict):
                self.region_coords = region
                self.window_title = title
                self._refresh_region_label()

            if interval is not None:
                self.interval_entry.delete(0, tk.END)
                self.interval_entry.insert(0, str(interval))

            ocr_interval = config.get("ocr_interval")
            if ocr_interval is not None:
                self.ocr_interval_entry.delete(0, tk.END)
                self.ocr_interval_entry.insert(0, str(ocr_interval))
            ocr_conf = config.get("ocr_conf")
            if ocr_conf is not None:
                self.ocr_conf_entry.delete(0, tk.END)
                self.ocr_conf_entry.insert(0, str(ocr_conf))

            ocr = config.get("ocr_region")
            if isinstance(ocr, dict) and ocr.get("width", 0) > 0:
                self.ocr_region = {
                    "x": int(ocr.get("x", 0)),
                    "y": int(ocr.get("y", 0)),
                    "width": int(ocr["width"]),
                    "height": int(ocr["height"]),
                }
            self._refresh_ocr_region_label()

            char = (config.get("character") or "").strip()
            if char:
                self._refresh_characters(select_name=char)

            self.log("GUI: 初始配置应用完成。")
        except Exception as e:
            self.log(f"错误: 应用初始配置时发生错误: {e}")

    def log(self, message):
        self.after(0, self._log_threadsafe, message)

    def _log_threadsafe(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

        num_lines = int(self.log_text.index('end-1c').split('.')[0])
        if num_lines > self.log_max_lines:
            lines_to_delete = num_lines - self.log_max_lines
            self.log_text.delete('1.0', f'{lines_to_delete + 1}.0')

        self.log_text.config(state=tk.DISABLED)

    def open_record_preview(self):
        self.after(0, self._open_record_preview_ui)

    def _open_record_preview_ui(self):
        from record_preview import RecordPreviewWindows

        if self.record_preview is None:
            self.record_preview = RecordPreviewWindows(self)
        self.record_preview.open(keys=True, yolo=False)

    def close_record_preview(self):
        if self.record_preview is not None:
            self.after(0, self.record_preview.close)

    def preview_key(self, action: str, key: str):
        self.after(0, self._preview_key_ui, action, key)

    def _preview_key_ui(self, action, key):
        if self.record_preview:
            self.record_preview.push_key(action, key)

    def preview_yolo(self, features: dict, bgr_small):
        self.after(0, self._preview_yolo_ui, features, bgr_small)

    def _preview_yolo_ui(self, features, bgr_small):
        if self.record_preview:
            self.record_preview.push_yolo(features, bgr_small)

    def update_runtime_status(self, character_name=None, state_text=None, fsm_text=None):
        """供分析线程更新城镇/地下城 / FSM；角色名只来自 GUI 下拉。"""
        self.after(0, self._update_runtime_status_threadsafe, character_name, state_text, fsm_text)

    def _update_runtime_status_threadsafe(self, character_name, state_text, fsm_text=None):
        if state_text is not None:
            self.live_state_var.set(state_text)
        if fsm_text is not None:
            self.live_fsm_var.set(fsm_text)

    def start(self):
        self.log("GUI: '开始'按钮被点击。")
        if self.is_collect_mode():
            if not is_admin():
                messagebox.showerror(
                    "采集需要管理员",
                    "采集必须先以管理员身份启动。\n\n"
                    "请点左下角「以管理员身份重启」，再点开始。\n"
                    "没有「继续采集」选项。",
                    parent=self,
                )
                self.log("采集已拦截：当前不是管理员。")
                return
            if not self._selected_yolo_weights():
                messagebox.showerror(
                    "采集需要 YOLO",
                    "采集必须先在「YOLO 设置」里选过图权重（solarwarden_b）。\n"
                    "frames.jsonl 要带检测框，否则回放/技能特征提不出来。",
                    parent=self,
                )
                self.log("采集已拦截：未选择 YOLO 权重。")
                return
        if self.is_fsm_test_mode() or self.is_yolo_test_mode():
            weights = self._selected_yolo_weights()
            if not weights:
                self.log("错误: 请先在「YOLO 设置」中选择有效权重。")
                return
            if self.is_fsm_test_mode():
                self._load_fsm_mlg_vars()
                if not is_admin():
                    messagebox.showerror(
                        "FSM测试需要管理员",
                        "FSM测试会发键，必须以管理员启动，否则游戏收不到键。\n"
                        "请点左下角「以管理员身份重启」，再开始。",
                        parent=self,
                    )
                    self.log("FSM测试已拦截：当前不是管理员。")
                    return
        if self.is_fsm_test_mode() or self.mode_var.get() == "automation":
            char = (self.character_var.get() or "").strip()
            if skill_table_missing(char):
                messagebox.showerror(
                    "没有技能表",
                    "FSM测试 / 自动化需要玩家技能表（skill_binds 下该角色 json，且 skills 非空）。\n"
                    "请先用技能绑定工具保存技能表。",
                    parent=self,
                )
                self.log("已拦截：没有技能表。")
                return
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self._toggle_settings_state("disabled")
        self.controller.start_process()
        if self.is_fsm_test_mode():
            self.after(80, self._focus_game_for_keys)
            self.after(500, self._focus_game_for_keys)

    def _focus_game_for_keys(self):
        if not self.is_fsm_test_mode():
            return
        from window_align import focus_region

        focus_region(self.region_coords)

    def abort_collect(self, message: str):
        """采集硬拦截：弹窗后停止。必须在 Tk 线程调用。"""
        messagebox.showerror("采集已中止", message, parent=self)
        self.log(f"采集已中止: {message.splitlines()[0]}")
        self.stop()

    def stop(self):
        self.log("GUI: '停止'按钮被点击 或 启动失败。")
        # 停止会 join 后台线程；放到工作线程里，避免 Tk 主线程卡死
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self):
        try:
            self.controller.stop_process()
        finally:
            self.after(0, self._after_stop_ui)

    def _after_stop_ui(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._toggle_settings_state("normal")
        self.live_fsm_var.set("—")
        self._load_fsm_mlg_vars()
        self.log("GUI: 停止完成，界面已恢复。")

    def _open_fsm_settings(self):
        win = self._fsm_settings_win
        if win is not None:
            try:
                if win.winfo_exists():
                    win.deiconify()
                    win.lift()
                    win.focus_force()
                    return
            except tk.TclError:
                self._fsm_settings_win = None
        win = tk.Toplevel(self)
        win.title("FSM 设置")
        win.transient(self)
        win.resizable(False, False)
        self._fsm_settings_win = win
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_fsm_settings(win))

        body = ttk.Frame(win, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        def spin(parent, label, var, frm, to, width=4):
            ttk.Label(parent, text=label).pack(side=tk.LEFT)
            ttk.Spinbox(parent, from_=frm, to=to, width=width, textvariable=var).pack(
                side=tk.LEFT, padx=(2, 8)
            )

        r1 = ttk.Frame(body)
        r1.pack(fill=tk.X)
        spin(r1, "判定 M", self.fsm_m_var, 1, 60)
        spin(r1, "L", self.fsm_l_var, 1, 60)
        spin(r1, "G", self.fsm_g_var, 1, 60)
        spin(r1, "卡住 X秒", self.fsm_x_var, 0.05, 120, 5)

        r2 = ttk.Frame(body)
        r2.pack(fill=tk.X, pady=(6, 0))
        spin(r2, "GX", self.fsm_gx_var, 0, 400)
        spin(r2, "GY", self.fsm_gy_var, 0, 400)
        spin(r2, "AX ms", self.fsm_ax_var, 0, 20000, 6)
        spin(r2, "AY ms", self.fsm_ay_var, 0, 20000, 6)

        r3 = ttk.Frame(body)
        r3.pack(fill=tk.X, pady=(6, 0))
        spin(r3, "恢复 Y ms", self.fsm_y_var, 1, 20000, 6)
        spin(r3, "相似 S%", self.fsm_s_var, 0, 100)
        spin(r3, "普攻 XXX ms", self.fsm_xxx_var, 1, 20000, 6)
        spin(r3, "点按 ms", self.fsm_tap_ms_var, 1, 200)

        r_mash = ttk.Frame(body)
        r_mash.pack(fill=tk.X, pady=(6, 0))
        spin(r_mash, "连按 COUNT", self.fsm_mash_count_var, 1, 15)
        spin(r_mash, "连按间隔 ms", self.fsm_mash_gap_var, 10, 300)
        spin(r_mash, "移动间隔 ms", self.fsm_th_var, 0, 300)

        r4 = ttk.Frame(body)
        r4.pack(fill=tk.X, pady=(6, 0))
        spin(r4, "掉落动 PM", self.fsm_pm_var, 0, 200)
        spin(r4, "一键拾取 PC", self.fsm_pc_var, 0, 40)
        spin(r4, "停下 PT", self.fsm_pt_var, 1, 60)
        spin(r4, "PW ms", self.fsm_pw_var, 0, 20000, 6)

        r_ef = ttk.Frame(body)
        r_ef.pack(fill=tk.X, pady=(6, 0))
        spin(r_ef, "群单 E", self.fsm_e_var, 0, 40)
        spin(r_ef, "假释放 F%", self.fsm_f_var, 0, 100)
        spin(r_ef, "回城秒", self.fsm_town_s_var, 1, 120, 5)

        pad = ttk.LabelFrame(body, text="MON/BOSS 位置补正", padding=8)
        pad.pack(fill=tk.X, pady=(10, 0))
        self._fsm_corr_summary = ttk.Label(pad, text="", foreground="#06c")
        self._fsm_corr_summary.pack(anchor=tk.W)
        grid = ttk.Frame(pad)
        grid.pack(pady=(6, 0))
        mx = MON_CORR_MAX
        tk.Scale(
            grid,
            from_=mx,
            to=0,
            orient=tk.VERTICAL,
            length=100,
            showvalue=True,
            variable=self.fsm_corr_u,
            label="上",
        ).grid(row=0, column=1)
        tk.Scale(
            grid,
            from_=mx,
            to=0,
            orient=tk.HORIZONTAL,
            length=120,
            showvalue=True,
            variable=self.fsm_corr_l,
            label="左",
        ).grid(row=1, column=0)
        tk.Scale(
            grid,
            from_=0,
            to=mx,
            orient=tk.HORIZONTAL,
            length=120,
            showvalue=True,
            variable=self.fsm_corr_r,
            label="右",
        ).grid(row=1, column=2)
        tk.Scale(
            grid,
            from_=0,
            to=mx,
            orient=tk.VERTICAL,
            length=100,
            showvalue=True,
            variable=self.fsm_corr_d,
            label="下",
        ).grid(row=2, column=1)
        ttk.Label(
            pad,
            text="滑块方向=你希望角色往哪边站。上：YOLO 的 MON/BOSS 的 Y 全部减去该值；左：X 全部减去。同轴互斥。检测框/jsonl 仍原始；FSM、提取、回放逻辑点用补正。改补正后请重提 skill_features。",
            foreground="#666",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(6, 0))
        self._refresh_fsm_corr_summary()

        ttk.Label(
            body,
            text="M/L/G 连续同值才改判定。BOSS 暂与 MON 共用 M。回城秒 tn_s：连续无地下城关键词达该秒数即回城（不换帧）。OCR 无关键词则沿用上一帧再进进图/回城。前进接近与捡物：点按 → TH 毫秒 → 按住。距门 >GX/>GY 才接近；两轴停或门没了再 AX/AY 毫秒。卡住后 HOLD Y 毫秒（不是前进）。X 秒应大于 max(最长技能持续, AX, AY, PW, 四向 Y)。S=分布相似百分比。PM/PT=掉落在动；PW=等停下上限毫秒，超时按当前掉落继续捡。数量>PC 一键拾取。E/F=提取。XXX=全 CD 普攻毫秒。点按 ms=技能/左Alt。连按 COUNT 在执行层。MON/BOSS 补正用于 FSM/提取/回放逻辑点，jsonl 仍原始，改后请重提。与回放共用 json：改完立刻写入；点开始会先读文件。",
            foreground="#666",
            wraplength=460,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 0))
        ttk.Button(body, text="关闭", command=lambda: self._close_fsm_settings(win)).pack(pady=(10, 0))

    def _close_fsm_settings(self, win):
        try:
            win.destroy()
        except tk.TclError:
            pass
        if self._fsm_settings_win is win:
            self._fsm_settings_win = None
        self._fsm_corr_summary = None

    def _corr_tuple(self) -> tuple[int, int, int, int]:
        def n(var: tk.IntVar) -> int:
            try:
                return max(0, min(MON_CORR_MAX, int(var.get())))
            except (TypeError, ValueError, tk.TclError):
                return 0

        return n(self.fsm_corr_u), n(self.fsm_corr_d), n(self.fsm_corr_l), n(self.fsm_corr_r)

    def _refresh_fsm_corr_summary(self):
        lab = getattr(self, "_fsm_corr_summary", None)
        if lab is None:
            return
        try:
            if not lab.winfo_exists():
                return
        except tk.TclError:
            return
        ox, oy = mon_off_from_corr(*self._corr_tuple())
        xs = "左" if ox < 0 else ("右" if ox > 0 else "无")
        ys = "上" if oy < 0 else ("下" if oy > 0 else "无")
        lab.config(text=f"当前：X {ox:+d}（{xs}）  Y {oy:+d}（{ys}）  → YOLO MON/BOSS 坐标加上该偏移")

    def _on_fsm_corr_edit(self):
        if self._fsm_corr_lock:
            return
        self._fsm_corr_lock = True
        try:
            cur = self._corr_tuple()
            prev = getattr(self, "_fsm_corr_prev", (0, 0, 0, 0))
            u, dwn, left, right = cur
            if u > 0 and dwn > 0:
                if u != prev[0]:
                    self.fsm_corr_d.set(0)
                elif dwn != prev[1]:
                    self.fsm_corr_u.set(0)
                elif u >= dwn:
                    self.fsm_corr_d.set(0)
                else:
                    self.fsm_corr_u.set(0)
            if left > 0 and right > 0:
                if left != prev[2]:
                    self.fsm_corr_r.set(0)
                elif right != prev[3]:
                    self.fsm_corr_l.set(0)
                elif left >= right:
                    self.fsm_corr_r.set(0)
                else:
                    self.fsm_corr_l.set(0)
            self._fsm_corr_prev = self._corr_tuple()
        finally:
            self._fsm_corr_lock = False
        self._refresh_fsm_corr_summary()
        ox, oy = mon_off_from_corr(*self._corr_tuple())
        prev_off = getattr(self, "_corr_off_warned", None)
        if self._fsm_mlg_persist and prev_off is not None and prev_off != (ox, oy):
            self.log("补正已改：旧 skill_features 作废，请在回放「重新提取本图」。jsonl 未改。")
        self._corr_off_warned = (ox, oy)
        self._on_fsm_mlg_edit()

    @staticmethod
    def _parse_mlg(var: tk.StringVar, default: int = 5) -> int:
        try:
            return max(1, int(str(var.get()).strip() or default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_float(var: tk.StringVar, default: float = 1.5, lo: float = 0.05) -> float:
        try:
            return max(lo, float(str(var.get()).strip() or default))
        except (TypeError, ValueError):
            return float(default)

    def _load_fsm_mlg_vars(self):
        m, l, g, gx, gy, s, pm, pc, pt = 5, 5, 5, 50, 10, 20, 10, 5, 3
        mash_count, mash_gap = 3, 50
        th_ms = 50
        e, f, town_s = DEFAULT_E, DEFAULT_F, float(DEFAULT_TOWN_S)
        tap_ms = 50
        pw_ms = DEFAULT_PW_MS
        data = {}
        if FSM_UI_PATH.is_file():
            try:
                data = json.loads(FSM_UI_PATH.read_text(encoding="utf-8"))
                m = max(1, int(data.get("m", m)))
                l = max(1, int(data.get("l", l)))
                g = max(1, int(data.get("g", g)))
                gx = max(0, int(data.get("gx", gx)))
                gy = max(0, int(data.get("gy", gy)))
                s = max(0, min(100, int(data.get("s", s))))
                pm = max(0, int(data.get("pm", data.get("lm", pm))))
                pc = max(0, int(data.get("pc", data.get("lc", pc))))
                pt = max(1, int(data.get("pt", pt)))
                try:
                    pw_ms = max(0, int(data.get("pw_ms", pw_ms)))
                except (TypeError, ValueError):
                    pw_ms = DEFAULT_PW_MS
                tap_ms = max(1, min(200, int(data.get("tap_ms", tap_ms))))
                mash_count = max(1, min(15, int(data.get("mash_count", mash_count))))
                mash_gap = max(10, min(300, int(data.get("mash_gap_ms", mash_gap))))
                th_ms = max(0, min(300, int(data.get("th_ms", mash_gap))))
                e = max(0, int(data.get("e", e)))
                f = max(0, min(100, int(data.get("f", f))))
                town_s = float(data.get("town_s", town_s))
            except Exception:
                data = {}
        x_s = json_s_from_frames(data, "x_s", "x", 30)
        ax_ms = json_ms(data, "ax_ms", "ax", 5)
        ay_ms = json_ms(data, "ay_ms", "ay", 5)
        y_ms = json_ms(data, "y_ms", "y", 5, lo=1)
        xxx_ms = json_ms(data, "xxx_ms", "xxx", 20, lo=1)
        self.fsm_m_var.set(str(m))
        self.fsm_l_var.set(str(l))
        self.fsm_g_var.set(str(g))
        self.fsm_x_var.set(str(x_s))
        self.fsm_gx_var.set(str(gx))
        self.fsm_gy_var.set(str(gy))
        self.fsm_ax_var.set(str(ax_ms))
        self.fsm_ay_var.set(str(ay_ms))
        self.fsm_y_var.set(str(y_ms))
        self.fsm_s_var.set(str(s))
        self.fsm_xxx_var.set(str(xxx_ms))
        cu, cd, cl, cr = mon_corr_from_dict(data)
        prev = self._fsm_mlg_persist
        self._fsm_mlg_persist = False
        self._fsm_corr_lock = True
        self.fsm_pm_var.set(str(pm))
        self.fsm_pc_var.set(str(pc))
        self.fsm_pt_var.set(str(pt))
        self.fsm_pw_var.set(str(pw_ms))
        self.fsm_e_var.set(str(e))
        self.fsm_f_var.set(str(f))
        self.fsm_town_s_var.set(str(town_s))
        self.fsm_tap_ms_var.set(str(tap_ms))
        self.fsm_mash_count_var.set(str(mash_count))
        self.fsm_mash_gap_var.set(str(mash_gap))
        self.fsm_th_var.set(str(th_ms))
        self.fsm_corr_u.set(cu)
        self.fsm_corr_d.set(cd)
        self.fsm_corr_l.set(cl)
        self.fsm_corr_r.set(cr)
        self._fsm_corr_lock = False
        self._fsm_mlg_persist = prev
        self._fsm_corr_prev = self._corr_tuple()
        self._refresh_fsm_corr_summary()

    def _save_fsm_mlg_vars(self):
        data = {}
        if FSM_UI_PATH.is_file():
            try:
                loaded = json.loads(FSM_UI_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}
        data["m"] = self._parse_mlg(self.fsm_m_var)
        data["l"] = self._parse_mlg(self.fsm_l_var)
        data["g"] = self._parse_mlg(self.fsm_g_var)
        data["x_s"] = self._parse_float(self.fsm_x_var, 1.5, lo=0.05)
        try:
            data["gx"] = max(0, int(str(self.fsm_gx_var.get()).strip() or 50))
        except (TypeError, ValueError):
            data["gx"] = 50
        try:
            data["gy"] = max(0, int(str(self.fsm_gy_var.get()).strip() or 10))
        except (TypeError, ValueError):
            data["gy"] = 10
        try:
            data["ax_ms"] = max(0, int(str(self.fsm_ax_var.get()).strip() or 250))
        except (TypeError, ValueError):
            data["ax_ms"] = 250
        try:
            data["ay_ms"] = max(0, int(str(self.fsm_ay_var.get()).strip() or 250))
        except (TypeError, ValueError):
            data["ay_ms"] = 250
        try:
            data["xxx_ms"] = max(1, int(str(self.fsm_xxx_var.get()).strip() or 1000))
        except (TypeError, ValueError):
            data["xxx_ms"] = 1000
        try:
            data["tap_ms"] = max(1, min(200, int(str(self.fsm_tap_ms_var.get()).strip() or 50)))
        except (TypeError, ValueError):
            data["tap_ms"] = 50
        try:
            data["mash_count"] = max(1, min(15, int(str(self.fsm_mash_count_var.get()).strip() or 3)))
        except (TypeError, ValueError):
            data["mash_count"] = 3
        try:
            data["mash_gap_ms"] = max(10, min(300, int(str(self.fsm_mash_gap_var.get()).strip() or 50)))
        except (TypeError, ValueError):
            data["mash_gap_ms"] = 50
        try:
            data["th_ms"] = max(0, min(300, int(str(self.fsm_th_var.get()).strip() or 50)))
        except (TypeError, ValueError):
            data["th_ms"] = 50
        data["y_ms"] = self._parse_mlg(self.fsm_y_var, 250)
        try:
            data["s"] = max(0, min(100, int(str(self.fsm_s_var.get()).strip() or 20)))
        except (TypeError, ValueError):
            data["s"] = 20
        try:
            data["pm"] = max(0, int(str(self.fsm_pm_var.get()).strip() or 10))
        except (TypeError, ValueError):
            data["pm"] = 10
        try:
            data["pc"] = max(0, int(str(self.fsm_pc_var.get()).strip() or 5))
        except (TypeError, ValueError):
            data["pc"] = 5
        try:
            data["pt"] = max(1, int(str(self.fsm_pt_var.get()).strip() or 3))
        except (TypeError, ValueError):
            data["pt"] = 3
        try:
            data["pw_ms"] = max(0, int(str(self.fsm_pw_var.get()).strip() or DEFAULT_PW_MS))
        except (TypeError, ValueError):
            data["pw_ms"] = DEFAULT_PW_MS
        try:
            data["e"] = max(0, int(str(self.fsm_e_var.get()).strip() or DEFAULT_E))
        except (TypeError, ValueError):
            data["e"] = DEFAULT_E
        try:
            data["f"] = max(0, min(100, int(str(self.fsm_f_var.get()).strip() or DEFAULT_F)))
        except (TypeError, ValueError):
            data["f"] = DEFAULT_F
        try:
            data["town_s"] = max(1, int(float(str(self.fsm_town_s_var.get()).strip() or DEFAULT_TOWN_S)))
        except (TypeError, ValueError):
            data["town_s"] = int(DEFAULT_TOWN_S)
        data.pop("lm", None)
        data.pop("lc", None)
        u, dwn, left, right = self._corr_tuple()
        data["mon_corr_u"] = u
        data["mon_corr_d"] = dwn
        data["mon_corr_l"] = left
        data["mon_corr_r"] = right
        try:
            FSM_UI_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log(f"警告: 未能写入 {FSM_UI_PATH.name}: {e}")

    def _on_fsm_mlg_edit(self):
        if not getattr(self, "_fsm_mlg_persist", False):
            return
        self._save_fsm_mlg_vars()

    def get_current_config(self):
        weights = self._selected_yolo_weights()
        try:
            yolo_conf = float(self.yolo_conf_entry.get())
        except ValueError:
            yolo_conf = 0.1
        try:
            yolo_iou = float(self.yolo_iou_entry.get())
        except ValueError:
            yolo_iou = 0.7
        return {
            'mode': self.mode_var.get() if self.mode_var.get() != "record" else "collect",
            'region_coords': self.region_coords,
            'window_title': self.window_title,
            'interval': self.interval_entry.get(),
            'ocr_interval': self.ocr_interval_entry.get(),
            'ocr_conf': self.ocr_conf_entry.get(),
            'capture_only': False,
            'fsm_test': bool(self.is_fsm_test_mode()),
            'yolo_test': bool(self.is_fsm_test_mode() or self.is_yolo_test_mode()),
            'send_keys': bool(self.is_fsm_test_mode()),
            'yolo_version': self.yolo_version_var.get(),
            'yolo_weights': weights,
            'yolo_conf': yolo_conf,
            'yolo_iou': yolo_iou,
            'ocr_region': self._effective_ocr_region(),
            'character': self.character_var.get().strip(),
        }

    def _toggle_settings_state(self, state):
        for group in self.winfo_children():
            if isinstance(group, ttk.Frame):
                for sub_group in group.winfo_children():
                    if isinstance(sub_group, ttk.LabelFrame):
                        title = sub_group.cget("text")
                        if "窗口" in title or "模式" in title or "YOLO" in title or title.startswith("OCR"):
                            for widget in sub_group.winfo_children():
                                self._set_widget_state(widget, state)
        if state == "normal":
            try:
                self.yolo_version_combo.config(state="readonly")
            except tk.TclError:
                pass
            try:
                self.character_combo.config(state="readonly")
                self.character_refresh_btn.config(state="normal")
            except tk.TclError:
                pass
        else:
            try:
                self.character_combo.config(state="disabled")
                self.character_refresh_btn.config(state="disabled")
            except tk.TclError:
                pass
            try:
                self.ocr_select_btn.config(state="normal")
                self.apply_ocr_btn.config(state="normal")
                self.ocr_interval_entry.config(state="normal")
                self.ocr_conf_entry.config(state="normal")
            except tk.TclError:
                pass
        try:
            self.fsm_settings_btn.config(state="normal")
        except tk.TclError:
            pass

    def _set_widget_state(self, widget, state):
        try:
            widget.config(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_state(child, state)


if __name__ == '__main__':
    if not getattr(sys, "frozen", False):
        os.chdir(Path(__file__).resolve().parent)
    enable_dpi_awareness()
    app = App()
    app.mainloop()
