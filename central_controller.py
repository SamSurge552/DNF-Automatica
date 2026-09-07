import time
import threading
import json
import os
from admin_runtime import is_admin
from screen_capture import ScreenCaptureModule
from status_analysis_module import StatusAnalysisModule
from operation_analysis_module import OperationAnalysisModule
from yolo_engine import YoloEngine, DEFAULT_YOLO_WEIGHTS


class CentralController:
    def __init__(self, gui_app):
        self.gui = gui_app
        self.capture_module = ScreenCaptureModule()
        self.yolo_engine = YoloEngine(DEFAULT_YOLO_WEIGHTS)
        self.operation_analyzer = OperationAnalysisModule(self.gui, self)
        self.status_analyzer = StatusAnalysisModule(
            self.gui,
            self.capture_module,
            yolo_engine=self.yolo_engine,
            on_state_change=self.on_dungeon_state_change,
        )

        self.current_dungeon = None
        self.current_character_name = None

    def on_dungeon_state_change(self, new_dungeon_name):
        """由SAM调用的回调函数，用于更新游戏上下文"""
        self.current_dungeon = new_dungeon_name
        mode = self.gui.mode_var.get()

        if self.current_dungeon:
            self.operation_analyzer.handle_dungeon_entry(self.current_dungeon, mode)
            self.gui.log(f"中央控制器: [回调] 检测到地下城变更为 -> {self.current_dungeon}，模式: {mode}")
        else:
            if mode in ("collect", "record"):
                self.operation_analyzer.close_segment()
            else:
                self.operation_analyzer.stop_all_operations()
            self.gui.log("中央控制器: [回调] 已退出地下城，返回城镇。")

    def load_config(self):
        """加载配置文件并更新GUI"""
        config_path = 'config.json'
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.gui.log(f"成功加载配置文件: {config_path}")
                # 将加载的配置应用到GUI
                self.gui.apply_initial_config(config)
            else:
                self.gui.log(f"警告: 配置文件 {config_path} 未找到，使用默认值。")
        except Exception as e:
            self.gui.log(f"错误: 加载配置文件失败 - {e}")

    def update_and_save_config(self):
        """从GUI获取最新配置并保存到文件"""
        self.gui.log("中央控制器: 正在更新并保存配置...")
        try:
            current_gui_config = self.gui.get_current_config()
            region = current_gui_config.get('region_coords')
            if not isinstance(region, dict):
                self.gui.log("警告: 游戏窗口尚未对齐，部分配置无法保存。")

            config_to_save = {
                "interval": float(current_gui_config['interval']),
                "region": {
                    "rect": region if isinstance(region, dict) else None,
                    "window_title": current_gui_config.get('window_title', ""),
                },
                "ocr_region": current_gui_config.get("ocr_region"),
                "ocr_interval": float(current_gui_config.get("ocr_interval", 0.3)),
                "ocr_conf": float(current_gui_config.get("ocr_conf", 0.95)),
                "character": current_gui_config.get("character") or "",
            }

            config_path = 'config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
            
            self.gui.log(f"配置已成功保存到 {config_path}")

        except Exception as e:
            self.gui.log(f"错误: 保存配置文件失败 - {e}")

    def start_process(self):
        """由GUI的'开始'按钮调用，负责启动核心模块"""
        self.gui.log("中央控制器: 接收到启动指令。")
        self.current_dungeon = None
        self.current_character_name = (self.gui.character_var.get() or "").strip() or None
        self.gui.update_runtime_status(state_text="城镇")
        if self.current_character_name:
            self.gui.log(f"本轮角色: {self.current_character_name}")
        else:
            self.gui.log("警告: 未选择角色，采集目录将不带角色后缀。")
        
        try:
            config = self.gui.get_current_config()
            region = config.get('region_coords')
            if not isinstance(region, dict) or region.get('width', 0) <= 0:
                 self.gui.log("错误: 游戏窗口未对齐或坐标无效，无法启动。")
                 self.gui.stop()
                 return
            self.gui.log("中央控制器: 成功获取GUI配置，准备启动分析模块...")
        except Exception as e:
            self.gui.log(f"错误: 获取GUI配置失败 - {e}")
            self.gui.stop()
            return

        fsm_test = bool(config.get("fsm_test"))
        yolo_test = bool(config.get("yolo_test"))
        mode = config.get("mode", "collect")
        if mode == "record":
            mode = "collect"
        if mode == "yolo_test":
            yolo_test = True
        collect = (not fsm_test) and (not yolo_test) and mode == "collect"

        if collect:
            if not is_admin():
                self.gui.log("错误: 采集必须先以管理员身份启动。")
                self.gui.stop()
                return
            if not self.operation_analyzer.arm_collect():
                self.gui.stop()
                return
            ocr_cfg = dict(config)
            ocr_cfg["ocr_only"] = True
            ocr_cfg["fsm_test"] = False
            ocr_cfg["yolo_test"] = False
            self.status_analyzer.start(ocr_cfg)
            self.gui.update_runtime_status(state_text="城镇")
            self.gui.log("中央控制器: 采集已待命 — YOLO + 键钩 + OCR；进图后才写 recordings。")
            return

        self.status_analyzer.start(config)

        if fsm_test:
            self.gui.log("中央控制器: FSM测试已交给状态模块（发键；进图后写盘 FSM_TEST）。")
        elif yolo_test:
            self.gui.log("中央控制器: YOLO测试已交给状态模块（只检测，不跑 FSM、不写盘）。")
        elif mode == "automation":
            self.gui.log("中央控制器: 自动化模式 — 等待进入地下城。")

    def stop_process(self):
        """由GUI的'停止'按钮调用，负责停止核心模块"""
        self.gui.log("中央控制器: 接收到停止指令。")
        # 将停止任务委托给状态分析模块
        self.status_analyzer.stop()
        
        # 3. 停止操作分析模块
        self.operation_analyzer.stop_all_operations()

        self.gui.log("中央控制器: 所有模块的停止指令已发送。") 