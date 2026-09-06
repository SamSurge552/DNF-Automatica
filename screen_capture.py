"""截取游戏窗口：SDR 用 dxcam；HDR 用 FP16 DXGI + OBS 色调映射。返回 BGR ndarray。"""
from __future__ import annotations

import datetime
import os
import threading
import time

import numpy as np
from PIL import Image, ImageGrab

from hdr_sdr import is_hdr_on, scrgb_to_sdr_bgr_u8, sdr_white_nits

_dxcam = None
_dxcam_import_error = None
try:
    import dxcam as _dxcam
except Exception as e:
    _dxcam_import_error = e


class ScreenCaptureModule:
    def __init__(self, base_output_dir="images"):
        self.base_output_dir = base_output_dir
        self.session_dir = None
        self._lock = threading.Lock()
        self._camera = None
        self._fp16 = None
        self._sdr_white = 203.0
        self._started_region = None
        self._use_dxcam = _dxcam is not None
        self._logged_backend = False
        self._session_files = []
        self._save_threads: list[threading.Thread] = []
        os.makedirs(self.base_output_dir, exist_ok=True)

    @staticmethod
    def _safe_folder(name: str) -> str:
        bad = '<>:"/\\|?*'
        out = "".join("_" if c in bad else c for c in (name or "").strip())
        return out or "unknown"

    def begin_session(self, gui=None, character=None):
        folder = self._safe_folder(character)
        self.session_dir = os.path.join(self.base_output_dir, folder)
        os.makedirs(self.session_dir, exist_ok=True)
        self._session_files = []
        self._log(gui, f"本轮截图目录: {self.session_dir}")
        return self.session_dir

    def discard_session_pngs(self):
        """只删本段写下的 PNG，不删角色目录里其它段。"""
        self.flush_saves()
        for fp in list(self._session_files or []):
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except OSError:
                pass
        self._session_files = []

    def _stop_camera(self):
        cam = self._camera
        fp16 = self._fp16
        self._camera = None
        self._fp16 = None
        self._started_region = None
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass
        if fp16 is not None:
            try:
                fp16.release()
            except Exception:
                pass

    def _ensure_camera(self, region: tuple[int, int, int, int], gui=None):
        if self._fp16 is not None and self._started_region == region:
            return "hdr"
        if self._camera is not None and self._started_region == region:
            return "dxcam"

        self._stop_camera()
        hdr = False
        try:
            hdr = is_hdr_on(0)
        except Exception:
            hdr = False

        if hdr:
            try:
                from fp16_capture import FP16Capture

                cap = FP16Capture(output_idx=0)
                self._fp16 = cap
                self._sdr_white = sdr_white_nits(0)
                self._started_region = region
                self._log(
                    gui,
                    f"检测到 HDR，改用 FP16 采集 + SDR 映射（纸白 {self._sdr_white:.0f} nits）",
                )
                return "hdr"
            except Exception as e:
                self._fp16 = None
                self._log(gui, f"HDR FP16 采集失败，回退 dxcam（画面可能过曝）: {e}")

        if not self._use_dxcam:
            return None
        try:
            cam = _dxcam.create(output_idx=0, output_color="BGR")
            if cam is None:
                self._use_dxcam = False
                return None
            cam.start(region=region, target_fps=120)
            self._camera = cam
            self._started_region = region
            return "dxcam"
        except Exception:
            self._stop_camera()
            self._use_dxcam = False
            return None

    def flush_saves(self, timeout: float = 8.0):
        """等本段未写完的 PNG 落盘（异步 save 用）。"""
        threads = list(self._save_threads)
        self._save_threads = []
        deadline = time.time() + max(0.1, float(timeout))
        for th in threads:
            remain = deadline - time.time()
            if remain <= 0:
                break
            th.join(remain)

    def _grab_hdr(self, region: tuple[int, int, int, int]):
        left, top, right, bottom = region
        for _ in range(8):
            try:
                bgra = self._fp16.crop_virtual(left, top, right, bottom, timeout_ms=40)
            except Exception:
                self._stop_camera()
                return None
            if bgra is not None and getattr(bgra, "size", 0) > 0:
                return scrgb_to_sdr_bgr_u8(bgra, self._sdr_white)
            time.sleep(0.004)
        return None

    def _grab_dxcam(self, region: tuple[int, int, int, int]):
        cam = self._camera
        if cam is None:
            return None
        for _ in range(8):
            try:
                frame = cam.get_latest_frame()
            except Exception:
                self._stop_camera()
                return None
            if frame is not None and getattr(frame, "size", 0) > 0:
                return np.ascontiguousarray(frame)
            time.sleep(0.004)
        return None

    def _grab_pillow(self, left, top, right, bottom):
        img = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
        return np.ascontiguousarray(np.array(img)[:, :, ::-1])

    def capture(self, rect, gui=None, save=True):
        if not all(k in rect for k in ["x", "y", "width", "height"]):
            self._log(gui, f"错误: 区域坐标格式不正确: {rect}")
            return None
        if save and not self.session_dir:
            self._log(gui, "错误: 尚未创建本轮截图目录。")
            return None

        left = int(rect["x"])
        top = int(rect["y"])
        right = left + int(rect["width"])
        bottom = top + int(rect["height"])
        if right <= left or bottom <= top:
            self._log(gui, "错误: 截图区域宽高无效。")
            return None

        if (right - left) % 2:
            right += 1
        if (bottom - top) % 2:
            bottom += 1
        region = (left, top, right, bottom)
        try:
            with self._lock:
                frame = None
                backend = "pillow"
                mode = self._ensure_camera(region, gui=gui)
                if mode == "hdr":
                    frame = self._grab_hdr(region)
                    backend = "dxcam-hdr" if frame is not None else "pillow-fallback"
                    if frame is None:
                        frame = self._grab_pillow(left, top, right, bottom)
                elif mode == "dxcam":
                    frame = self._grab_dxcam(region)
                    if frame is not None:
                        backend = "dxcam"
                    else:
                        frame = self._grab_pillow(left, top, right, bottom)
                        backend = "pillow-fallback"
                else:
                    frame = self._grab_pillow(left, top, right, bottom)

            if frame is None:
                self._log(gui, "错误: 截图失败（空帧）。")
                return None

            t_ns = time.time_ns()

            if not self._logged_backend:
                self._logged_backend = True
                extra = f"（dxcam 不可用: {_dxcam_import_error}）" if _dxcam_import_error else ""
                self._log(gui, f"截图后端: {backend}{extra}")

            filepath = None
            if save:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{timestamp}.png"
                filepath = os.path.join(self.session_dir, filename)
                rgb = np.ascontiguousarray(frame[:, :, ::-1])
                th = threading.Thread(
                    target=self._save_png,
                    args=(rgb, filepath),
                    name="png-save",
                    daemon=True,
                )
                self._save_threads = [t for t in self._save_threads if t.is_alive()]
                self._save_threads.append(th)
                self._session_files.append(filepath)
                th.start()

            return {"frame": frame, "path": filepath, "backend": backend, "t_ns": t_ns}
        except Exception as e:
            self._log(gui, f"错误: 截图失败 - {e}")
            return None

    @staticmethod
    def _save_png(rgb, filepath: str) -> None:
        Image.fromarray(rgb).save(filepath)

    def _log(self, gui, message):
        if gui:
            gui.log(f"截图模块: {message}")
        else:
            print(f"截图模块: {message}")
