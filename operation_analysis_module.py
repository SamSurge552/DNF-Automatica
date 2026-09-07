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
        self._segment_gen = 0
        self._cleanup_lock = threading.Lock()
        self._yolo_conf = 0.1
        self._yolo_iou = 0.7

        self.is_automating = False

    def handle_dungeon_entry(self, dungeon_name, mode):
        self.gui.log(f"操作分析模块: 已接收到进入地下城 [{dungeon_name}] 的指令，模式为 [{mode}]。")
        if mode in ("collect", "record"):
            self.open_segment(dungeon_name)
            return
        if mode == "automation":
            self.start_automation(dungeon_name)

    def update_session_dungeon(self, dungeon_name: str):
        """禁止只改 meta.dungeon：地下城名变了就停段再新开。"""
        if not dungeon_name:
            return
        self.open_segment(dungeon_name)

    def start_recording(self, dungeon_name="采集"):
        return self.arm_collect()

    def start_collect(self, dungeon_name="采集") -> bool:
        """已废：点开始不要带图名建段。请走 arm_collect。"""
        return self.arm_collect()

    def arm_collect(self) -> bool:
        """点开始：YOLO + 键钩 + 截图循环供 OCR。不建 recordings、不写 keys/frames/PNG。"""
        if self.is_recording:
            self.gui.log("  - 警告: 采集已待命或正在写段，无法重复启动。")
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
            self.gui.log("  - 错误: 采集必须先选 YOLO 权重（过图默认 mix_a）。")
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
            images = Path(capture.begin_session(self.gui, character=char))
            self.images_dir = images
            self.session_dir = None
            self.writer = None
            self._collect_aborted = False
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
            try:
                self.gui.open_record_preview()
            except Exception as e:
                self.gui.log(f"  - 警告: 预览窗打开失败（不影响采集）: {e}")
            self.gui.log("  -> 采集待命（YOLO + 键钩 + OCR）。城镇不写盘。")
            self.gui.log(f"  -> 进图后 PNG 仍: {images.resolve()}")
            return True
        except Exception as e:
            self.gui.log(f"  - 错误: 启动采集失败 - {e}")
            self._stop_armed()
            return False

    def open_segment(self, dungeon_name: str) -> bool:
        """OCR 进图：建 recordings/<OCR图名>/<时间戳>_<角色>/ 并开始写盘。"""
        if not self.is_recording:
            self.gui.log("  - 警告: 采集未待命，忽略进图建段。")
            return False
        dun = (dungeon_name or "").strip()
        if not dun or dun in ("采集", "未知"):
            self.gui.log(f"  - 错误: 进图建段需要 OCR 图名，收到 [{dungeon_name}]。")
            return False
        if self.writer and self.session_dir:
            cur = ""
            try:
                import json

                meta_path = self.session_dir / "meta.json"
                if meta_path.is_file():
                    cur = str(json.loads(meta_path.read_text(encoding="utf-8")).get("dungeon") or "")
            except Exception:
                cur = ""
            if cur == dun:
                return True
            self.close_segment()
        try:
            capture = self.controller.capture_module
            mark = getattr(capture, "mark_segment", None)
            if callable(mark):
                mark()
            char = self.controller.current_character_name
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_dir = self._recordings_session_dir(dun, stamp, char)
            rec_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir = rec_dir
            self.writer = RecordWriter(self.session_dir, log=self.gui.log)
            self.writer.start()
            started_t_ns = time.time_ns()
            held_at_start = snapshot_held_keys()
            self._seed_keys = len(held_at_start)
            self._collect_aborted = False
            engine = getattr(self.controller, "yolo_engine", None)
            self.writer.write_meta(
                {
                    "dungeon": dun,
                    "started_at": stamp,
                    "started_t_ns": started_t_ns,
                    "interval": float((self.gui.get_current_config() or {}).get("interval", 0.05)),
                    "region": (self.gui.get_current_config() or {}).get("region_coords"),
                    "format": "keys.jsonl + frames.jsonl (t_ns absolute)",
                    "yolo_weights": str(getattr(engine, "weights", "") or ""),
                    "yolo_classes": [str(v) for v in (getattr(engine, "names", None) or {}).values()],
                    "yolo": {
                        "weights": str(getattr(engine, "weights", "") or ""),
                        "conf": self._yolo_conf,
                        "iou": self._yolo_iou,
                    },
                    "session_dir": str(self.session_dir.resolve()),
                    "png_dir": str(self.images_dir.resolve()) if self.images_dir else "",
                    "character": char,
                    "keys_held_at_start": held_at_start,
                }
            )
            self._segment_gen += 1
            gen = self._segment_gen
            threading.Thread(
                target=self._collect_keys_watchdog,
                args=(gen,),
                name="collect-keys-watchdog",
                daemon=True,
            ).start()
            for key_name in held_at_start:
                self._enqueue_key_event("press", key_name, t_ns=started_t_ns, seed=True)
            self.gui.log(f"  -> 采集建段: {self.session_dir.resolve()}")
            if held_at_start:
                self.gui.log(f"  - 开段时已按住: {', '.join(held_at_start)}")
            self.gui.log(f"  - {COLLECT_KEYS_GRACE_S:g}s 内无新按键将中止并删除本段。")
            return True
        except Exception as e:
            self.gui.log(f"  - 错误: 采集建段失败 - {e}")
            self.close_segment()
            return False

    def close_segment(self) -> None:
        """回城：只停本段，保留文件。键钩/YOLO/截图循环继续。"""
        self._flush_segment(delete=False, keep_armed=True)

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
                writing = self.writer is not None
                result = capture.capture(rect=region, gui=None, save=writing)
            except Exception as e:
                self.gui.log(f"  - 采集截图失败: {e}")
                continue
            if not result or result.get("frame") is None:
                continue
            t_ns = int(result["t_ns"])
            if writing and self.writer:
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

    def _collect_keys_watchdog(self, gen: int):
        time.sleep(COLLECT_KEYS_GRACE_S)
        if gen != self._segment_gen:
            return
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
        self._flush_segment(delete=False, keep_armed=False)
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
        self._flush_segment(delete=False, keep_armed=False)

    def _flush_segment(self, *, delete: bool, keep_armed: bool) -> None:
        with self._cleanup_lock:
            self._flush_segment_locked(keep_armed=keep_armed)

    def _flush_segment_locked(self, *, keep_armed: bool) -> None:
        self._segment_gen += 1
        writer = self.writer
        session = self.session_dir
        self.writer = None
        keys_n = frames_n = 0
        if writer:
            writer.stop(timeout=5.0)
            keys_n = writer.keys_count
            frames_n = writer.frames_count
        if session and session.exists():
            try:
                import json

                meta_path = session / "meta.json"
                meta = {}
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["character"] = self.controller.current_character_name
                meta["keys_count"] = keys_n
                meta["frames_count"] = frames_n
                meta["session_dir"] = str(Path(session).resolve())
                if self.images_dir:
                    meta["png_dir"] = str(Path(self.images_dir).resolve())
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as e:
                self.gui.log(f"  - 警告: 采集 meta 更新失败: {e}")
        self.session_dir = None
        if session:
            self.gui.log(
                f"  - 采集本段已停（保留文件）: {Path(session).resolve()} "
                f"(keys={keys_n}, frames={frames_n})"
            )
        if keep_armed:
            flush = getattr(self.controller.capture_module, "flush_saves", None)
            if callable(flush):
                flush()
            return
        if not self.is_recording and writer is None and session is None:
            return
        self.is_recording = False
        self._rec_stop.set()
        for th in (self.vision_thread, self.infer_thread):
            if th and th.is_alive():
                th.join(timeout=2.0)
        self.vision_thread = None
        self.infer_thread = None
        self.frame_queue = None
        flush = getattr(self.controller.capture_module, "flush_saves", None)
        if callable(flush):
            flush()
        self.images_dir = None
        closer = getattr(self.gui, "close_record_preview", None)
        if closer:
            closer()
        self.gui.log("  - 采集待命已停止。")

    def _stop_armed(self) -> None:
        self._flush_segment(delete=False, keep_armed=False)

    def start_automation(self, dungeon_name):
        """过图小模型自动化占位：旧 .script 回放已移除。"""
        char = self.controller.current_character_name or "未识别角色"
        self.gui.log(f"  -> 自动化模式: [{dungeon_name}] / [{char}]")
        self.gui.log("  - 过图小模型尚未接入；本段仅占位，不执行按键。")
        self.gui.log("  - 请先用采集模式写盘（PNG + YOLO 特征 + 键盘）。")
        self.is_automating = False

    def stop_all_operations(self):
        self.gui.log("操作分析模块: 收到停止所有操作的指令。")
        self._flush_segment(delete=False, keep_armed=False)
        if self.is_automating:
            self.gui.log("  - 正在停止自动化...")
            self.is_automating = False
            self.gui.log("  - 自动化已停止。")
