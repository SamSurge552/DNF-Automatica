"""操作分析：采集（截图+键盘，无模型）；自动化占位。"""
from __future__ import annotations

import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from pynput import keyboard

from key_snapshot import snapshot_held_keys
from record_writer import RecordWriter
from yolo_engine import YoloEngine

COLLECT_KEYS_GRACE_S = 3.0
ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "recordings"


class OperationAnalysisModule:
    def __init__(self, gui_app, central_controller):
        self.gui = gui_app
        self.controller = central_controller

        self.is_recording = False
        self.keyboard_listener = None
        self.writer: RecordWriter | None = None
        self.session_dir: Path | None = None
        self.images_dir: Path | None = None
        self.vision_thread = None
        self.infer_thread = None
        self.frame_queue = None
        self._rec_stop = threading.Event()
        self._seed_keys = 0
        self._collect_aborted = False
        self._cleanup_lock = threading.Lock()
        self._yolo_conf = 0.1
        self._yolo_iou = 0.7

        self.is_automating = False

    def handle_dungeon_entry(self, dungeon_name, mode):
        self.gui.log(f"操作分析模块: 已接收到进入地下城 [{dungeon_name}] 的指令，模式为 [{mode}]。")
        if mode in ("collect", "record"):
            self.gui.log("  - 采集不经 OCR 开录（点开始即写 PNG+键盘）。")
            return
        if mode == "automation":
            self.start_automation(dungeon_name)

    def update_session_dungeon(self, dungeon_name: str):
        """录制已开始后，用 OCR 识别到的地下城名更新 meta / 目录名偏好。"""
        if not self.writer or not self.session_dir:
            return
        try:
            import json

            meta_path = self.session_dir / "meta.json"
            meta = {}
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["dungeon"] = dungeon_name
            self.writer.write_meta(meta)
            self.gui.log(f"  - 录制会话地下城已更新为: {dungeon_name}")
        except Exception as e:
            self.gui.log(f"  - 警告: 更新录制 meta 失败: {e}")

    def start_recording(self, dungeon_name="采集"):
        return self.start_collect(dungeon_name)

    def start_collect(self, dungeon_name="采集") -> bool:
        if self.is_recording:
            self.gui.log("  - 警告: 已经在采集中，无法重复启动。")
            return False

        region = None
        interval = 0.05
        yolo_conf, yolo_iou = 0.1, 0.7
        weights = None
        try:
            cfg = self.gui.get_current_config()
            region = cfg.get("region_coords")
            interval = float(cfg.get("interval", 0.05))
            yolo_conf = float(cfg.get("yolo_conf", 0.1))
            yolo_iou = float(cfg.get("yolo_iou", 0.7))
            weights = cfg.get("yolo_weights")
        except Exception as e:
            self.gui.log(f"  - 错误: 读取采集配置失败 - {e}")
            return False

        if not isinstance(region, dict) or region.get("width", 0) <= 0:
            self.gui.log("  - 错误: 游戏窗口未对齐，无法开始采集。")
            return False

        try:
            from admin_runtime import is_admin

            if not is_admin():
                self.gui.log("  - 错误: 采集必须先以管理员身份启动。")
                return False
        except Exception as e:
            self.gui.log(f"  - 错误: 检查管理员权限失败 - {e}")
            return False

        if not weights:
            self.gui.log("  - 错误: 采集必须先选 YOLO 权重（过图用 solarwarden_b）。")
            return False
        engine = getattr(self.controller, "yolo_engine", None)
        if engine is None or not engine.apply(
            weights=weights, conf=yolo_conf, iou=yolo_iou, log=self.gui.log
        ):
            self.gui.log("  - 错误: YOLO 未能加载，采集中止。")
            return False
        self._yolo_conf = yolo_conf
        self._yolo_iou = yolo_iou

        try:
            capture = self.controller.capture_module
            char = self.controller.current_character_name
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            images = Path(capture.begin_session(self.gui, character=char))
            self.images_dir = images
            dun = dungeon_name or "采集"
            if dun in ("采集", "未知"):
                try:
                    from skill_feature_extract import guess_dungeon

                    guessed = guess_dungeon(char or "")
                    if guessed:
                        dun = guessed
                except Exception:
                    pass
            rec_dir = self._recordings_session_dir(dun, stamp, char)
            rec_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir = rec_dir

            self.writer = RecordWriter(self.session_dir, log=self.gui.log)
            self.writer.start()

            started_t_ns = time.time_ns()
            held_at_start = snapshot_held_keys()
            self._seed_keys = len(held_at_start)
            self._collect_aborted = False
            self.writer.write_meta(
                {
                    "dungeon": dun,
                    "started_at": stamp,
                    "started_t_ns": started_t_ns,
                    "interval": interval,
                    "region": region,
                    "format": "keys.jsonl + frames.jsonl (t_ns absolute)",
                    "yolo_weights": str(engine.weights),
                    "yolo_classes": [str(v) for v in (engine.names or {}).values()],
                    "yolo": {
                        "weights": str(engine.weights),
                        "conf": yolo_conf,
                        "iou": yolo_iou,
                    },
                    "session_dir": str(self.session_dir.resolve()),
                    "png_dir": str(self.images_dir.resolve()),
                    "character": char,
                    "keys_held_at_start": held_at_start,
                }
            )

            self._rec_stop.clear()
            self.is_recording = True
            self._ensure_keyboard_listener()

            self.vision_thread = threading.Thread(
                target=self._vision_capture_loop,
                args=(region, interval),
                name="collect-vision",
                daemon=True,
            )
            self.vision_thread.start()
            threading.Thread(
                target=self._collect_keys_watchdog,
                name="collect-keys-watchdog",
                daemon=True,
            ).start()

            for key_name in held_at_start:
                self._enqueue_key_event("press", key_name, t_ns=started_t_ns, seed=True)
            rec_path = str(self.session_dir.resolve())
            img_path = str(self.images_dir.resolve()) if self.images_dir else ""
            try:
                self.gui.open_record_preview()
            except Exception as e:
                self.gui.log(f"  - 警告: 预览窗打开失败（不影响采集）: {e}")
            self.gui.log("  -> 采集已开始（截图 + 键盘 + YOLO）")
            self.gui.log(f"  -> 按键/jsonl: {rec_path}")
            if img_path:
                self.gui.log(f"  -> PNG: {img_path}")
            if held_at_start:
                self.gui.log(f"  - 开录时已按住（GetAsyncKeyState 一次）: {', '.join(held_at_start)}")
            self.gui.log(f"  - {COLLECT_KEYS_GRACE_S:g}s 内无新按键将中止并删除本段。")
            return True
        except Exception as e:
            self.gui.log(f"  - 错误: 启动采集失败 - {e}")
            self._cleanup_recording(rename=False)
            return False

    @staticmethod
    def _safe_name(name: str) -> str:
        bad = '<>:"/\\|?*'
        out = "".join("_" if c in bad else c for c in name).strip()
        return out or "unknown"

    def _recordings_session_dir(self, dungeon: str, stamp: str, character: str | None) -> Path:
        dun = self._safe_name(dungeon or "采集")
        char = self._safe_name(character) if character else ""
        name = f"{stamp}_{char}" if char else stamp
        dest = RECORDINGS / dun / name
        if not dest.exists():
            return dest
        n = 2
        while True:
            alt = RECORDINGS / dun / f"{name}_{n}"
            if not alt.exists():
                return alt
            n += 1

    def _ensure_keyboard_listener(self):
        """Windows 上 pynput Listener.stop 后再 start，钩子经常不再进键。换段只换 writer，钩子进程内只挂一次。"""
        lis = self.keyboard_listener
        if lis is not None:
            running = getattr(lis, "running", False)
            if running:
                self.gui.log("  - 按键钩子沿用（不 stop/start，避免换段丢键）")
                return
            try:
                lis.stop()
            except Exception:
                pass
            self.keyboard_listener = None
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self.keyboard_listener.start()
        self.gui.log("  - 按键钩子已安装（本进程只应看到这一次，除非钩子自己死了）")

    def _get_key_str(self, key):
        if hasattr(key, "char") and key.char is not None:
            return key.char
        if hasattr(key, "name"):
            return key.name
        return str(key)

    def _enqueue_key_event(self, action: str, key_str: str, t_ns: int | None = None, seed: bool = False):
        if not self.is_recording or not self.writer:
            return
        try:
            item = {
                "stream": "keys",
                "t_ns": int(t_ns if t_ns is not None else time.time_ns()),
                "type": action,
                "key": key_str,
            }
            if seed:
                item["seed"] = True
            if not self.writer.enqueue(item) and self.writer.dropped % 100 == 1:
                self.gui.log("  - 警告: 采集写盘队列已满，部分按键可能丢弃。")
            preview = getattr(self.gui, "preview_key", None)
            if preview:
                preview(action, key_str)
        except Exception as e:
            self.gui.log(f"[REC] 按键入队失败: {e}")

    def _enqueue_key(self, action: str, key):
        self._enqueue_key_event(action, self._get_key_str(key))

    def _on_key_press(self, key):
        self._enqueue_key("press", key)

    def _on_key_release(self, key):
        self._enqueue_key("release", key)

    def _vision_capture_loop(self, region, interval: float):
        capture = self.controller.capture_module
        last = 0.0
        n = 0
        while not self._rec_stop.is_set():
            now = time.time()
            if now - last < interval:
                time.sleep(min(0.005, max(0.001, interval * 0.1)))
                continue
            last = now
            try:
                result = capture.capture(rect=region, gui=None, save=True)
            except Exception as e:
                self.gui.log(f"  - 采集截图失败: {e}")
                continue
            if not result or result.get("frame") is None:
                continue
            t_ns = time.time_ns()
            png_path = result.get("path") or ""
            png_name = Path(png_path).name if png_path else f"{t_ns}.png"
            engine = getattr(self.controller, "yolo_engine", None)
            feats = None
            if engine is not None:
                try:
                    feats = engine.infer_features(
                        result["frame"], conf=self._yolo_conf, iou=self._yolo_iou
                    )
                except Exception as e:
                    self.gui.log(f"  - YOLO 推理失败: {e}")
            if not feats:
                feats = YoloEngine.empty_features()
            if self.writer:
                item = {"stream": "frames", "t_ns": t_ns, "png": png_name}
                item.update({k: v for k, v in feats.items() if k != "_result"})
                self.writer.enqueue(item)
            n += 1
            if n == 1 or n % 20 == 0:
                self.gui.log(
                    f"  - 已保存 #{n}: {png_name}  "
                    f"mon={feats.get('mon') or 0} loot={feats.get('loot') or 0} "
                    f"gate={feats.get('gate') or 0} boss={feats.get('boss') or 0} "
                    f"infer={feats.get('infer_ms') or 0:.0f}ms"
                )
            offer = getattr(self.controller.status_analyzer, "offer_ocr_frame", None)
            if callable(offer):
                try:
                    offer(result["frame"])
                except Exception:
                    pass

    def _collect_keys_watchdog(self):
        time.sleep(COLLECT_KEYS_GRACE_S)
        if self._collect_aborted or not self.is_recording or not self.writer:
            return
        if self.writer.keys_count <= self._seed_keys:
            self._abort_empty_keys()

    def _abort_empty_keys(self):
        if self._collect_aborted or not self.is_recording:
            return
        self._collect_aborted = True
        rec_dir = self.session_dir
        capture = getattr(self.controller, "capture_module", None)
        discard = getattr(capture, "discard_session_pngs", None) if capture is not None else None
        self._cleanup_recording(rename=False)
        if callable(discard):
            discard()
        if rec_dir and rec_dir.exists():
            try:
                shutil.rmtree(rec_dir)
                self.gui.log(f"  - 已删除空采集目录: {rec_dir}")
            except Exception as e:
                self.gui.log(f"  - 删除空采集目录失败: {e}")
        msg = (
            f"{COLLECT_KEYS_GRACE_S:g} 秒内 keys.jsonl 没有新按键事件。\n"
            "钩子没挂上或没在打。本段已删除，没有继续采集选项。"
        )
        abort = getattr(self.gui, "abort_collect", None)
        if abort:
            self.gui.after(0, lambda: abort(msg))
        else:
            self.gui.after(0, self.gui.stop)

    def _cleanup_recording(self, rename: bool = True):
        with self._cleanup_lock:
            self._cleanup_recording_locked(rename)

    def _cleanup_recording_locked(self, rename: bool):
        if not self.is_recording and self.writer is None and self.session_dir is None:
            return
        self.is_recording = False
        self._rec_stop.set()

        for th in (self.vision_thread, self.infer_thread):
            if th and th.is_alive():
                th.join(timeout=2.0)
        self.vision_thread = None
        self.infer_thread = None
        self.frame_queue = None

        if self.writer:
            self.writer.stop(timeout=5.0)
            keys_n = self.writer.keys_count
            frames_n = self.writer.frames_count
        else:
            keys_n = frames_n = 0

        final_dir = self.session_dir
        if self.session_dir and self.session_dir.exists():
            try:
                import json

                meta_path = self.session_dir / "meta.json"
                meta = {}
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["character"] = self.controller.current_character_name
                meta["keys_count"] = keys_n
                meta["frames_count"] = frames_n
                meta["session_dir"] = str(Path(self.session_dir).resolve())
                if self.images_dir:
                    meta["png_dir"] = str(Path(self.images_dir).resolve())
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as e:
                self.gui.log(f"  - 警告: 采集 meta 更新失败: {e}")

        img_dir = self.images_dir
        self.writer = None
        self.is_recording = False
        self.session_dir = None
        self.images_dir = None
        closer = getattr(self.gui, "close_record_preview", None)
        if closer:
            closer()
        if final_dir:
            extra = f" PNG={Path(img_dir).resolve()}" if img_dir else ""
            self.gui.log(
                f"  - 采集已停止。jsonl: {Path(final_dir).resolve()} "
                f"(keys={keys_n}, frames={frames_n}){extra}"
            )
        else:
            self.gui.log("  - 采集已停止。")

    def start_automation(self, dungeon_name):
        """过图小模型自动化占位：旧 .script 回放已移除。"""
        char = self.controller.current_character_name or "未识别角色"
        self.gui.log(f"  -> 自动化模式: [{dungeon_name}] / [{char}]")
        self.gui.log("  - 过图小模型尚未接入；本段仅占位，不执行按键。")
        self.gui.log("  - 请先用采集模式写盘（PNG + YOLO 特征 + 键盘）。")
        self.is_automating = False

    def stop_all_operations(self):
        self.gui.log("操作分析模块: 收到停止所有操作的指令。")
        if self.is_recording:
            self._cleanup_recording(rename=not self._collect_aborted)

        if self.is_automating:
            self.gui.log("  - 正在停止自动化...")
            self.is_automating = False
            self.gui.log("  - 自动化已停止。")
