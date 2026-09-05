import os
import json
import time
import threading
import queue
from pathlib import Path

from cuda_runtime import ensure_cuda_runtime
from rapidocr import RapidOCR
from yolo_engine import DEFAULT_YOLO_WEIGHTS
from fsm_core import (
    FsmContext,
    FsmParams,
    intent_text,
    mon_off_from_dict,
    send_keys_label,
    snapshot_from_detect,
    step,
)
from fsm_execute import (
    FsmExecutor,
    DEFAULT_TAP_MS,
    DEFAULT_MASH_COUNT,
    DEFAULT_MASH_GAP_MS,
    clamp_tap_ms,
    clamp_mash_count,
    clamp_mash_gap_ms,
)
from window_align import (
    find_window_by_title,
    focus_region,
    foreground_hwnd,
    prevent_activate,
)
from skill_feature_extract import (
    DEFAULT_F,
    feature_path,
    guess_dungeon,
    has_map_reset,
    load_dist_table,
    load_features,
    load_fight_plan,
    load_hotbar,
)

YOLO_PREVIEW_WINDOW_FSM = "FSM Test"
YOLO_PREVIEW_WINDOW_YOLO = "YOLO Test"
DETECT_LOG_S = 8.0
ROOT = Path(__file__).resolve().parent
FSM_TEST_DIR = ROOT / "FSM_TEST"


class StatusAnalysisModule:
    def __init__(
        self,
        gui_app,
        capture_module,
        yolo_engine=None,
        on_state_change=None,
    ):
        self.gui = gui_app
        self.capture_module = capture_module
        self.yolo_engine = yolo_engine
        self.on_state_change_callback = on_state_change

        self.is_running = False
        self.capture_only = False
        self.yolo_test = False
        self.fsm_test = False
        self.save_images = True
        self.yolo_conf = 0.1
        self.yolo_iou = 0.7
        self.ocr_region = None
        self.ocr_interval = 0.3
        self.ocr_conf = 0.95
        self.capture_thread = None
        self.ocr_thread = None
        self.ocr_queue = None
        self.game_state = "TOWN"
        self.current_dungeon_name = None
        self.last_capture_time = 0
        self.frame_index = 0
        self._fsm_ctx = FsmContext()
        self._fsm_params = FsmParams()
        self._fsm_ui_mtime = None
        self._executor = None
        self._region_coords = None
        self._preview_named: set[str] = set()
        self._preview_hwnds: dict[str, int] = {}
        self._writer = None
        self._fsm_session_dir = None
        self._fsm_png_dir = None
        self._key_listener = None
        self._last_png_name = ""
        self._want_game_fg = False
        self._want_game_fg_until = 0.0
        self._last_detect_log_key = None
        self._last_detect_log_t = 0.0
        self._last_key_fail_t = 0.0

        self.dungeon_keywords = self._load_keywords("dun_keywords_custom.txt")
        self.gui.log("状态分析模块: 正在初始化 RapidOCR (CUDA 13)...")
        try:
            ensure_cuda_runtime()
            self.ocr = RapidOCR(
                params={
                    "EngineConfig.onnxruntime.use_cuda": True,
                    "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
                    "Global.use_cls": False,
                    "Global.text_score": 0.5,
                    "Global.max_side_len": 8192,
                    "Global.log_level": "error",
                }
            )
            self.gui.log("  - RapidOCR 初始化成功（优先 CUDA 13，失败会回退 CPU）。")
        except Exception as e:
            self.gui.log(f"  - 错误: RapidOCR 初始化失败: {e}")
            self.ocr = None

    def _load_keywords(self, filename):
        if not os.path.exists(filename):
            self.gui.log(f"提示: 关键词文件 '{filename}' 不存在，将创建空文件。")
            open(filename, "w").close()
            return []
        try:
            with open(filename, "r", encoding="utf-8") as f:
                keywords = [line.strip() for line in f if line.strip()]
            self.gui.log(f"状态分析模块: 从 {filename} 加载 {len(keywords)} 个关键词。")
            return keywords
        except Exception as e:
            self.gui.log(f"警告: 读取关键词文件 '{filename}' 失败: {e}")
            return []

    def start(self, config):
        if self.is_running:
            self.gui.log("错误: 状态分析模块已在运行。")
            return

        self.capture_only = bool(config.get("capture_only", False))
        self.fsm_test = bool(config.get("fsm_test"))
        self.yolo_test = bool(config.get("yolo_test") or self.fsm_test)
        mode = config.get("mode", "collect")
        if mode == "record":
            mode = "collect"
        if mode == "yolo_test":
            self.yolo_test = True
        if mode == "collect" and not self.yolo_test:
            self.gui.log("状态分析模块: 采集由操作模块负责，这里不截图。")
            return
        # YOLO测试：不落盘。FSM测试：写 FSM_TEST。
        self.save_images = bool(self.fsm_test)
        self.yolo_conf = float(config.get("yolo_conf", 0.1))
        self.yolo_iou = float(config.get("yolo_iou", 0.7))
        self.ocr_region = config.get("ocr_region") if isinstance(config.get("ocr_region"), dict) else None
        try:
            self.ocr_interval = max(0.05, float(config.get("ocr_interval", 0.3)))
        except (TypeError, ValueError):
            self.ocr_interval = 0.3
        try:
            self.ocr_conf = min(1.0, max(0.0, float(config.get("ocr_conf", 0.95))))
        except (TypeError, ValueError):
            self.ocr_conf = 0.95

        if self.capture_only and self.yolo_test:
            self.gui.log("错误: 采集与检测测试不能同时开启。")
            self.gui.stop()
            return

        if self.yolo_test:
            if not self.yolo_engine:
                self.gui.log("错误: 未注入 YOLO 引擎。")
                self.gui.stop()
                return
            weights = config.get("yolo_weights")
            if not self.yolo_engine.apply(
                weights=weights,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                log=self.gui.log,
            ):
                self.gui.stop()
                return
            self.gui.log(f"  - YOLO 推理参数 conf={self.yolo_conf}, iou={self.yolo_iou}")
        elif not self.capture_only and not self.ocr:
            self.gui.log("错误: OCR引擎未初始化，无法启动分析模块。")
            self.gui.stop()
            return

        if not self.capture_only and not self.yolo_test:
            self.dungeon_keywords = self._load_keywords("dun_keywords_custom.txt")

        self.game_state = "TOWN"
        self.current_dungeon_name = None
        self.frame_index = 0
        self.last_capture_time = 0
        self._fsm_ctx = FsmContext()
        self._fsm_params = self._load_fsm_params()
        self._fsm_ui_mtime = self._fsm_ui_file_mtime()
        self._executor = None
        self._region_coords = config.get("region_coords") if isinstance(config.get("region_coords"), dict) else None
        self._preview_named = set()
        self._preview_hwnds = {}
        self._want_game_fg = False
        self._want_game_fg_until = 0.0
        self._last_detect_log_key = None
        self._last_detect_log_t = 0.0
        self._last_key_fail_t = 0.0
        self._last_png_name = ""
        self._stop_fsm_record()
        if self.fsm_test:
            self._executor = FsmExecutor(log=self.gui.log, tap_ms=getattr(self, "_tap_ms", DEFAULT_TAP_MS))
            self._sync_executor_mash()
            self._want_game_fg = True
            self._want_game_fg_until = time.time() + 2.0
            if focus_region(self._region_coords):
                self.gui.log("FSM测试发键：已把游戏窗口拉到前台。点停止会松开我们按下的键。")
            else:
                self.gui.log("FSM测试发键：未能切到游戏窗口，请点一下游戏画面。点停止会松开我们按下的键。")
            self._start_fsm_record(config)

        if self.fsm_test:
            self.gui.log("状态分析模块: FSM测试模式 — 发键 + 写盘 FSM_TEST。")
        elif self.yolo_test:
            self.gui.log("状态分析模块: YOLO测试模式 — 只检测，不跑 FSM，不落盘。")
        else:
            self.gui.log("状态分析模块: 自动化 — OCR 用内存帧，截图不落盘。")

        self.is_running = True
        self.ocr_queue = None
        self.ocr_thread = None
        self.capture_thread = threading.Thread(target=self._capture_loop, args=(config,), daemon=True)
        self.capture_thread.start()

        if self.fsm_test:
            p = self._fsm_params
            self.gui.log(
                f"状态分析模块: 已启动【FSM测试】（YOLO+FSM+发键，写盘 FSM_TEST）。"
                f"M={p.m} L={p.l} G={p.g} X={p.x} GX={p.gx} GY={p.gy} AX={p.ax} AY={p.ay} Y={p.y} S={p.s}% "
                f"点按{getattr(self, '_tap_ms', DEFAULT_TAP_MS)}ms 连按COUNT={getattr(self, '_mash_count', DEFAULT_MASH_COUNT)} "
                f"间隔{getattr(self, '_mash_gap_ms', DEFAULT_MASH_GAP_MS)}ms  开打序列{len(p.fight_plan)}分布 发键开"
            )
        elif self.yolo_test:
            self.gui.log("状态分析模块: 已启动【YOLO测试】（只画检测框，不跑 FSM，无OCR/无存图）。")
        else:
            self.ocr_queue = queue.Queue(maxsize=1)
            self.ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True)
            self.ocr_thread.start()
            if isinstance(self.ocr_region, dict) and self.ocr_region.get("width", 0) > 0:
                r = self.ocr_region
                self.gui.log(
                    f"状态分析模块: 已启动 RapidOCR — 间隔 {self.ocr_interval:g}s  conf>={self.ocr_conf:g}，仅识别裁剪区 "
                    f"X:{r.get('x')} Y:{r.get('y')} W:{r.get('width')} H:{r.get('height')}"
                )
            else:
                self.gui.log(
                    f"状态分析模块: 已启动 RapidOCR（间隔 {self.ocr_interval:g}s  conf>={self.ocr_conf:g}，未设置裁剪则识别整帧）。"
                )

    def get_current_dungeon_name(self):
        return self.current_dungeon_name

    def stop(self):
        """只发停止信号并短等；避免在 GUI 主线程长时间阻塞。"""
        self.is_running = False
        exe = self._executor
        self._executor = None
        if exe is not None:
            try:
                exe.stop()
            except Exception:
                pass
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.ocr_thread and self.ocr_thread.is_alive():
            self.ocr_thread.join(timeout=1.5)
        self.capture_thread = None
        self.ocr_thread = None
        self._stop_fsm_record()
        try:
            import cv2
            cv2.destroyWindow(YOLO_PREVIEW_WINDOW_FSM)
            cv2.destroyWindow(YOLO_PREVIEW_WINDOW_YOLO)
        except Exception:
            pass
        self._preview_named = set()
        self._preview_hwnds = {}
        self._want_game_fg = False
        self._want_game_fg_until = 0.0
        if self.capture_only:
            saved = self.frame_index
            session = getattr(self.capture_module, "session_dir", None)
            if session:
                self.gui.log(f"状态分析模块: 采集截图结束，共 {saved} 张 -> {os.path.abspath(session)}")
            else:
                self.gui.log(f"状态分析模块: 采集截图结束，共 {saved} 张。")
        else:
            self.gui.log("状态分析模块: 已停止。")

    def _capture_loop(self, config):
        self.gui.log("状态分析模块: 进入截图循环。")
        while self.is_running:
            current_time = time.time()
            if self.yolo_test or self.capture_only:
                interval = float(config.get("interval", 0.05))
            else:
                interval = float(getattr(self, "ocr_interval", 0.3) or 0.3)
            if current_time - self.last_capture_time >= interval:
                self.last_capture_time = current_time
                frame = self._capture_frame(config.get("region_coords"))
                if frame is None:
                    pass
                elif self.yolo_test:
                    t_ns = time.time_ns()
                    self._run_yolo_frame(frame, t_ns)
                elif not self.capture_only:
                    self._submit_ocr(frame)
            time.sleep(0.02)
        self.gui.log("状态分析模块: 退出截图循环。")

    def _fsm_ui_path(self) -> Path:
        return Path(__file__).resolve().parent / "_fsm_replay_ui.json"

    @staticmethod
    def _safe_name(name: str) -> str:
        bad = '<>:"/\\|?*'
        out = "".join("_" if c in bad else c for c in name).strip()
        return out or "unknown"

    def _sync_executor_mash(self) -> None:
        exe = self._executor
        if exe is None:
            return
        slots = {int(sk.slot) for sk in (self._fsm_params.hotbar or ()) if sk.mash}
        exe.set_mash(
            getattr(self, "_mash_count", DEFAULT_MASH_COUNT),
            getattr(self, "_mash_gap_ms", DEFAULT_MASH_GAP_MS),
            slots,
        )

    def _fsm_session_path(self, dungeon: str, stamp: str, character: str | None) -> Path:
        dun = self._safe_name(dungeon or "FSM测试")
        char = self._safe_name(character) if character else ""
        name = f"{stamp}_{char}" if char else stamp
        dest = FSM_TEST_DIR / dun / name
        if not dest.exists():
            return dest
        n = 2
        while True:
            alt = FSM_TEST_DIR / dun / f"{name}_{n}"
            if not alt.exists():
                return alt
            n += 1

    def _start_fsm_record(self, config: dict) -> None:
        from datetime import datetime

        from key_snapshot import snapshot_held_keys
        from record_writer import RecordWriter

        try:
            char = (self.gui.character_var.get() or "").strip()
        except Exception:
            char = ""
        dun = self.current_dungeon_name or guess_dungeon(char) or "FSM测试"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = self._fsm_session_path(dun, stamp, char or None)
        png_dir = session / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        self._fsm_session_dir = session
        self._fsm_png_dir = png_dir
        self.capture_module.session_dir = str(png_dir)
        writer = RecordWriter(session, log=self.gui.log)
        writer.start()
        self._writer = writer
        started_t_ns = time.time_ns()
        held = snapshot_held_keys()
        engine = self.yolo_engine
        meta = {
            "kind": "fsm_test",
            "dungeon": dun,
            "started_at": stamp,
            "started_t_ns": started_t_ns,
            "interval": config.get("interval"),
            "region": config.get("region_coords"),
            "format": "keys.jsonl + frames.jsonl (t_ns absolute)",
            "yolo_weights": str(getattr(engine, "weights", "") or ""),
            "yolo": {
                "weights": str(getattr(engine, "weights", "") or ""),
                "conf": self.yolo_conf,
                "iou": self.yolo_iou,
            },
            "session_dir": str(session.resolve()),
            "png_dir": str(png_dir.resolve()),
            "character": char,
            "keys_held_at_start": held,
        }
        writer.write_meta(meta)
        self._write_fsm_params_file()
        self._start_key_listener()
        for key_name in held:
            self._enqueue_fsm_key("press", key_name, t_ns=started_t_ns, seed=True)
        self.gui.log(f"  -> FSM_TEST: {session.resolve()}")
        self.gui.log(f"  -> PNG: {png_dir.resolve()}")

    def _write_fsm_params_file(self) -> None:
        session = self._fsm_session_dir
        if session is None:
            return
        p = self._fsm_params
        ui = {}
        path = self._fsm_ui_path()
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    ui = raw
            except Exception:
                ui = {}
        try:
            char = (self.gui.character_var.get() or "").strip()
        except Exception:
            char = ""
        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "character": char,
            "dungeon": self.current_dungeon_name or guess_dungeon(char) or "",
            "yolo_conf": self.yolo_conf,
            "yolo_iou": self.yolo_iou,
            "tap_ms": getattr(self, "_tap_ms", DEFAULT_TAP_MS),
            "mash_count": getattr(self, "_mash_count", DEFAULT_MASH_COUNT),
            "mash_gap_ms": getattr(self, "_mash_gap_ms", DEFAULT_MASH_GAP_MS),
            "mash_slots": sorted({int(sk.slot) for sk in (p.hotbar or ()) if sk.mash}),
            "fsm": {
                "m": p.m,
                "l": p.l,
                "g": p.g,
                "x": p.x,
                "gx": p.gx,
                "gy": p.gy,
                "ax": p.ax,
                "ay": p.ay,
                "y": p.y,
                "s": p.s,
                "pm": p.pm,
                "pc": p.pc,
                "pt": p.pt,
                "xxx": p.xxx,
                "mon_off_x": p.mon_off_x,
                "mon_off_y": p.mon_off_y,
                "mash_count": p.mash_count,
                "mash_gap_ms": p.mash_gap_ms,
                "map_reset": p.map_reset,
                "fight_plan_n": len(p.fight_plan),
                "hotbar_n": len(p.hotbar),
            },
            "ui": ui,
        }
        try:
            (session / "fsm_params.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            self.gui.log(f"  - 警告: 写入 fsm_params.json 失败: {e}")
        try:
            from skill_feature_extract import _safe_name as skill_safe

            bind = ROOT / "skill_binds" / f"{skill_safe(char, 'character')}.json"
            if char and bind.is_file():
                import shutil

                shutil.copy2(bind, session / "skill_binds.json")
        except Exception:
            pass

    def _start_key_listener(self) -> None:
        if self._key_listener is not None:
            return
        try:
            from pynput import keyboard
        except Exception as e:
            self.gui.log(f"  - 警告: 键盘钩子不可用，本段没有 keys.jsonl: {e}")
            return
        self._key_listener = keyboard.Listener(
            on_press=lambda k: self._on_fsm_key("press", k),
            on_release=lambda k: self._on_fsm_key("release", k),
        )
        self._key_listener.start()
        self.gui.log("  - FSM_TEST 按键钩子已安装")

    def _key_str(self, key) -> str:
        if hasattr(key, "char") and key.char is not None:
            return key.char
        if hasattr(key, "name"):
            return key.name
        return str(key)

    def _on_fsm_key(self, action: str, key) -> None:
        self._enqueue_fsm_key(action, self._key_str(key))

    def _enqueue_fsm_key(self, action: str, key_str: str, t_ns: int | None = None, seed: bool = False) -> None:
        writer = self._writer
        if writer is None:
            return
        item = {
            "stream": "keys",
            "t_ns": int(t_ns if t_ns is not None else time.time_ns()),
            "type": action,
            "key": key_str,
        }
        if seed:
            item["seed"] = True
        writer.enqueue(item)

    def _enqueue_fsm_frame(self, t_ns: int, features: dict) -> None:
        writer = self._writer
        if writer is None:
            return
        item = {"stream": "frames", "t_ns": int(t_ns), "png": self._last_png_name}
        item.update({k: v for k, v in (features or {}).items() if k != "_result"})
        writer.enqueue(item)

    def _stop_fsm_record(self) -> None:
        lis = self._key_listener
        self._key_listener = None
        if lis is not None:
            try:
                lis.stop()
            except Exception:
                pass
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                self._write_fsm_params_file()
            except Exception:
                pass
            writer.stop(timeout=5.0)
            session = self._fsm_session_dir
            if session:
                self.gui.log(
                    f"  - FSM_TEST 已停止: {Path(session).resolve()} "
                    f"(keys={writer.keys_count}, frames={writer.frames_count})"
                )
        self._fsm_session_dir = None
        self._fsm_png_dir = None
        if getattr(self.capture_module, "session_dir", None) and str(self.capture_module.session_dir).replace("\\", "/").find("/FSM_TEST/") >= 0:
            self.capture_module.session_dir = None
        elif getattr(self.capture_module, "session_dir", None) and "FSM_TEST" in str(self.capture_module.session_dir):
            self.capture_module.session_dir = None

    def _fsm_ui_file_mtime(self) -> float | None:
        times: list[float] = []
        path = self._fsm_ui_path()
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            pass
        try:
            char = (self.gui.character_var.get() or "").strip()
        except Exception:
            char = ""
        dun = self.current_dungeon_name or (guess_dungeon(char) if char else "")
        if char and dun:
            try:
                times.append(feature_path(char, dun).stat().st_mtime)
            except OSError:
                pass
        return max(times) if times else None

    def _load_fsm_params(self) -> FsmParams:
        path = self._fsm_ui_path()
        m, l, g, x, gx, gy, ax, ay, y, s, f, pm, pc, pt, xxx, tap_ms = 5, 5, 5, 30, 50, 10, 5, 5, 5, 20, DEFAULT_F, 10, 5, 3, 20, DEFAULT_TAP_MS
        mash_count, mash_gap = DEFAULT_MASH_COUNT, DEFAULT_MASH_GAP_MS
        ox, oy = 0, 0
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                m = max(1, int(data.get("m", m)))
                l = max(1, int(data.get("l", l)))
                g = max(1, int(data.get("g", g)))
                x = max(1, int(data.get("x", x)))
                old_a = max(0, int(data.get("a", 5)))
                gx = max(0, int(data.get("gx", gx)))
                gy = max(0, int(data.get("gy", gy)))
                ax = max(0, int(data.get("ax", old_a)))
                ay = max(0, int(data.get("ay", old_a)))
                y = max(1, int(data.get("y", y)))
                s = max(0, min(100, int(data.get("s", s))))
                f = max(0, min(100, int(data.get("f", f))))
                pm = max(0, int(data.get("pm", data.get("lm", pm))))
                pc = max(0, int(data.get("pc", data.get("lc", pc))))
                pt = max(1, int(data.get("pt", pt)))
                xxx = max(1, int(data.get("xxx", xxx)))
                tap_ms = clamp_tap_ms(data.get("tap_ms", tap_ms))
                mash_count = clamp_mash_count(data.get("mash_count", mash_count))
                mash_gap = clamp_mash_gap_ms(data.get("mash_gap_ms", mash_gap))
                ox, oy = mon_off_from_dict(data)
            except Exception:
                pass
        self._f = f
        self._tap_ms = tap_ms
        self._mash_count = mash_count
        self._mash_gap_ms = mash_gap
        if self._executor is not None:
            self._executor.set_tap_ms(tap_ms)
            self._sync_executor_mash()
        char = ""
        try:
            char = (self.gui.character_var.get() or "").strip()
        except Exception:
            pass
        dun = self.current_dungeon_name or guess_dungeon(char)
        return FsmParams(
            m=m,
            l=l,
            g=g,
            x=x,
            gx=gx,
            gy=gy,
            ax=ax,
            ay=ay,
            y=y,
            s=s,
            pm=pm,
            pc=pc,
            pt=pt,
            xxx=xxx,
            fight_plan=load_fight_plan(char, dun, f=f) if char and dun else (),
            hotbar=load_hotbar(char) if char else (),
            dist_table=load_dist_table(char, dun) if char and dun else (),
            map_reset=has_map_reset(load_features(char, dun)) if char and dun else False,
            mon_off_x=ox,
            mon_off_y=oy,
            mash_count=mash_count,
            mash_gap_ms=mash_gap,
        )

    def _fsm_state_from_features(self, features: dict, t_ns: int):
        mtime = self._fsm_ui_file_mtime()
        if mtime != getattr(self, "_fsm_ui_mtime", None):
            self._fsm_ui_mtime = mtime
            self._fsm_params = self._load_fsm_params()
        snap = snapshot_from_detect(int(t_ns), features)
        decision, self._fsm_ctx = step(snap, self._fsm_ctx, self._fsm_params)
        return decision

    def _overlay_fsm(self, plotted, decision):
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("msyh.ttc", 28)
        except Exception:
            font = ImageFont.load_default()
        try:
            font2 = ImageFont.truetype("msyh.ttc", 18)
        except Exception:
            font2 = font
        flag_s = " ".join(sorted((f.value for f in decision.flags), key=lambda s: s))
        intent = intent_text(decision)
        send = send_keys_label(
            decision.action, decision.move_dirs, decision.move_dir, decision.skill_key
        )
        text = f"FSM {decision.state.value}"
        if flag_s:
            text += f" | {flag_s}"
        if intent:
            text += f" | {intent}"
        if send:
            text += f" | {send}"
        steps = decision.flow_steps or ()
        hit = decision.flow_hit
        if steps:
            bits = []
            for i, name in enumerate(steps):
                bits.append(f"[{name}]" if i == hit else name)
            text2 = "方法 " + "→".join(bits)
        else:
            text2 = ""
        cds = decision.skill_cds or ()
        extra = (
            f"已过房间={decision.gate_crosses}  击败BOSS={decision.boss_kills}"
            + ("  CD重置" if decision.cd_reset else "")
        )
        if cds:
            cd_s = "CD " + " | ".join(
                f"{slot}{key or ''} {rem:.0f}s" if rem > 0 else f"{slot}{key or ''} 就绪"
                for slot, key, rem in cds
            )
            text2 = (text2 + "  " + cd_s).strip()
        text2 = (text2 + "  " + extra).strip()
        try:
            box = draw.textbbox((14, 10), text, font=font)
        except Exception:
            box = (8, 8, 200, 44)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(20, 24, 32))
        draw.text((14, 10), text, fill=(0, 220, 120), font=font)
        if text2:
            try:
                box2 = draw.textbbox((14, 48), text2, font=font2)
            except Exception:
                box2 = (8, 44, 400, 72)
            draw.rectangle((box2[0] - 6, box2[1] - 4, box2[2] + 6, box2[3] + 4), fill=(20, 24, 32))
            draw.text((14, 48), text2, fill=(80, 180, 255), font=font2)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _run_yolo_frame(self, frame, t_ns: int):
        if not self.yolo_engine:
            return
        try:
            t0 = time.time()
            result = self.yolo_engine.predict_result(
                frame, conf=self.yolo_conf, iou=self.yolo_iou
            )
            elapsed = time.time() - t0
            if result is None:
                return
            features = self.yolo_engine._extract_features(result, infer_ms=elapsed * 1000.0)
            self._enqueue_fsm_frame(t_ns, features)
            summary = ", ".join(
                f"{k}×{v}"
                for k, v in (
                    ("mon", features.get("mon") or 0),
                    ("loot", features.get("loot") or 0),
                    ("gate", features.get("gate") or 0),
                    ("boss", features.get("boss") or 0),
                    ("player", 0 if features.get("player_xy") is None else 1),
                )
                if v
            ) or "无目标"
            import cv2
            plotted = result.plot()
            if self.fsm_test:
                decision = self._fsm_state_from_features(features, t_ns)
                if self._executor is not None:
                    try:
                        self._executor.apply(decision)
                    except Exception as e:
                        self._log_key_fail(e)
                flag_s = " ".join(sorted((f.value for f in decision.flags), key=lambda s: s))
                intent = intent_text(decision)
                send = send_keys_label(
                    decision.action, decision.move_dirs, decision.move_dir, decision.skill_key
                )
                fsm_s = decision.state.value
                if flag_s:
                    fsm_s += f" | {flag_s}"
                if intent:
                    fsm_s += f" | {intent}"
                if send:
                    fsm_s += f" | {send}"
                self._log_detect(
                    (decision.state.value, flag_s, decision.action, decision.skill_slot, decision.skill_key, summary),
                    f"  - FSM#{self.frame_index} {elapsed*1000:.0f}ms | {fsm_s} | {summary}",
                )
                self.gui.update_runtime_status(
                    character_name=None,
                    state_text=f"检测 · {summary}",
                    fsm_text=fsm_s,
                )
                plotted = self._overlay_fsm(plotted, decision)
                win_name = YOLO_PREVIEW_WINDOW_FSM
            else:
                self._log_detect(
                    ("yolo", summary),
                    f"  - YOLO#{self.frame_index} {elapsed*1000:.0f}ms | {summary}",
                )
                self.gui.update_runtime_status(
                    character_name=None,
                    state_text=f"检测 · {summary}",
                    fsm_text="YOLO测试",
                )
                win_name = YOLO_PREVIEW_WINDOW_YOLO
            h, w = plotted.shape[:2]
            max_w = 960
            if w > max_w:
                scale = max_w / w
                plotted = cv2.resize(plotted, (max_w, int(h * scale)))
            self._show_preview(cv2, win_name, plotted)
        except Exception as e:
            tag = "FSM测试" if self.fsm_test else "YOLO测试"
            self.gui.log(f"  - 错误: {tag}推理失败 - {e}")

    def _log_detect(self, key, line: str) -> None:
        """状态/意图变了打一条；否则最多 DETECT_LOG_S 秒一条。顶栏仍每帧更新。"""
        now = time.time()
        if key != self._last_detect_log_key or now - self._last_detect_log_t >= DETECT_LOG_S:
            self._last_detect_log_key = key
            self._last_detect_log_t = now
            self.gui.log(line)

    def _log_key_fail(self, err) -> None:
        now = time.time()
        if now - self._last_key_fail_t < DETECT_LOG_S:
            return
        self._last_key_fail_t = now
        self.gui.log(f"  - 发键失败: {err}")

    def _show_preview(self, cv2, win_name: str, plotted) -> None:
        if win_name not in self._preview_named:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            self._preview_named.add(win_name)
        hwnd = self._preview_hwnds.get(win_name) or find_window_by_title(win_name)
        if hwnd:
            self._preview_hwnds[win_name] = hwnd
            prevent_activate(hwnd)
        cv2.imshow(win_name, plotted)
        cv2.waitKey(1)
        hwnd = self._preview_hwnds.get(win_name) or find_window_by_title(win_name)
        if hwnd:
            self._preview_hwnds[win_name] = hwnd
            prevent_activate(hwnd)
        if not self._executor:
            return
        fg = foreground_hwnd()
        stolen = hwnd and fg == hwnd
        grace = self._want_game_fg and time.time() < self._want_game_fg_until
        if stolen or grace:
            if focus_region(self._region_coords):
                if not stolen:
                    self._want_game_fg = False
        elif self._want_game_fg:
            self._want_game_fg = False

    def _submit_ocr(self, frame):
        try:
            self.ocr_queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self.ocr_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.ocr_queue.put_nowait(frame)
        except queue.Full:
            return

    def _ocr_loop(self):
        self.gui.log("状态分析模块: 进入OCR循环。")
        while self.is_running:
            if self.ocr_queue is None:
                time.sleep(0.05)
                continue
            try:
                frame = self.ocr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.is_running:
                break
            t0 = time.time()
            detected_texts = self._analyze_image(frame)
            elapsed = time.time() - t0
            if not self.is_running:
                break
            if elapsed >= 2.0:
                self.gui.log(f"  - 警告: OCR耗时 {elapsed:.2f}s（偏慢，已静默丢旧帧保最新）")
            if detected_texts is not None:
                self._update_dungeon_state(detected_texts)
                self._refresh_runtime_display()
        self.gui.log("状态分析模块: 退出OCR循环。")

    def _capture_frame(self, rect):
        if not (rect and rect.get("width", 0) > 0 and rect.get("height", 0) > 0):
            self.gui.log("  - 警告: 游戏窗口坐标无效，跳过。")
            return None
        result = self.capture_module.capture(
            rect=rect, gui=None if self.fsm_test else self.gui, save=self.save_images
        )
        if not result:
            self.gui.log("  - 游戏窗口截图失败。")
            return None
        self.frame_index += 1
        frame = result["frame"]
        path = result.get("path")
        self._last_png_name = Path(path).name if path else ""
        h, w = frame.shape[:2]
        if path:
            if self.fsm_test:
                if self.frame_index == 1 or self.frame_index % 20 == 0:
                    self.gui.log(f"  - FSM_TEST 已保存 #{self.frame_index}: {self._last_png_name}")
            elif self.capture_only:
                if self.frame_index == 1 or self.frame_index % 20 == 0:
                    self.gui.log(f"  - 已保存 #{self.frame_index}: {os.path.basename(path)}")
            else:
                self.gui.log(f"  - 截图#{self.frame_index}: {w}x{h} -> {os.path.basename(path)}")
        elif self.frame_index == 1:
            self.gui.log(f"  - 截图#{self.frame_index}: {w}x{h}（不落盘；后续截图不再刷日志）")
        return frame

    def _match_keywords(self, pairs, keywords):
        hits = []
        seen = set()
        for text, score in pairs:
            for keyword in keywords:
                if keyword in text and keyword not in seen:
                    seen.add(keyword)
                    hits.append((keyword, score, text))
        return hits

    @staticmethod
    def _fmt_score(score):
        if score is None:
            return "—"
        return f"{float(score):.3f}"

    @staticmethod
    def _clip_text(text, n=24):
        s = (text or "").replace("\n", " ").strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    def _update_dungeon_state(self, pairs):
        thr = float(getattr(self, "ocr_conf", 0.95))
        hits = self._match_keywords(pairs, self.dungeon_keywords)
        above = [h for h in hits if h[1] is None or float(h[1]) >= thr]
        matched_dungeon = above[0][0] if above else None
        if len(above) > 1:
            self.gui.log(f"  - 警告: 匹配到多个地下城关键词 {[h[0] for h in above]}，默认使用第一个: '{matched_dungeon}'")

        if matched_dungeon:
            entered_new = self.game_state != "DUNGEON" or self.current_dungeon_name != matched_dungeon
            if entered_new:
                kw, score, raw = above[0]
                self.current_dungeon_name = matched_dungeon
                self.game_state = "DUNGEON"
                self.gui.log(
                    f">>> 状态变更: [模式] 进入地下城 [{self.current_dungeon_name}]  "
                    f"conf={self._fmt_score(score)}  阈值>={thr:g}  原文「{self._clip_text(raw)}」"
                )
                if self.on_state_change_callback:
                    self.on_state_change_callback(self.current_dungeon_name)
        elif self.game_state != "TOWN":
            old_dungeon = self.current_dungeon_name
            self.current_dungeon_name = None
            self.game_state = "TOWN"
            if hits:
                kw, score, raw = max(hits, key=lambda h: float(h[1] or 0.0))
                why = (
                    f"关键词「{kw}」conf={self._fmt_score(score)} 低于阈值 {thr:g}  "
                    f"原文「{self._clip_text(raw)}」"
                )
            elif pairs:
                raw, score = max(pairs, key=lambda p: float(p[1] or 0.0))
                why = (
                    f"本帧无关键词  最高OCR conf={self._fmt_score(score)}  "
                    f"「{self._clip_text(raw)}」"
                )
            else:
                why = "本帧无OCR结果"
            self.gui.log(f">>> 状态变更: [模式] 从 [{old_dungeon}] 返回 城镇  {why}")
            if self.on_state_change_callback:
                self.on_state_change_callback(self.current_dungeon_name)

    def _refresh_runtime_display(self):
        if self.game_state == "DUNGEON" and self.current_dungeon_name:
            state_text = f"地下城 · {self.current_dungeon_name}"
        else:
            state_text = "城镇"
        self.gui.update_runtime_status(state_text=state_text)

    def _crop_for_ocr(self, frame):
        """全屏帧上裁出 OCR 区；无效则回退整帧。"""
        if frame is None:
            return frame
        r = self.ocr_region
        if not isinstance(r, dict) or r.get("width", 0) <= 0 or r.get("height", 0) <= 0:
            return frame
        fh, fw = frame.shape[:2]
        x = max(0, int(r.get("x", 0)))
        y = max(0, int(r.get("y", 0)))
        w = int(r["width"])
        h = int(r["height"])
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        if x2 - x < 16 or y2 - y < 16:
            return frame
        return frame[y:y2, x:x2]

    def _analyze_image(self, frame):
        if not self.ocr:
            return None
        try:
            roi = self._crop_for_ocr(frame)
            result = self.ocr(roi)
            txts = list(getattr(result, "txts", None) or [])
            scores = list(getattr(result, "scores", None) or [])

            if not txts:
                return []

            if len(txts) == len(scores):
                return list(zip(txts, scores))
            return [(t, None) for t in txts]
        except Exception as e:
            self.gui.log(f"  - 错误: OCR分析失败 - {e}")
            return None
