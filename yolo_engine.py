"""共享 YOLO 检测引擎（默认 solarwarden_b：旧五类 + blank，未冻）。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from cuda_runtime import ensure_cuda_runtime

DEFAULT_YOLO_WEIGHTS = Path(r"D:/Atrain/runs/solarwarden_b/weights/best.pt")


class YoloEngine:
    """线程安全的懒加载推理；busy 时 try_infer 返回 None（跳过该帧）。"""

    def __init__(self, weights: str | Path | None = None, conf: float = 0.1, iou: float = 0.7):
        self.weights = Path(weights) if weights else DEFAULT_YOLO_WEIGHTS
        self.conf = float(conf)
        self.iou = float(iou)
        self.model = None
        self.names = {}
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._warmed = False
        self._imgsz = 640

    def _same_weights(self, weights: str | Path) -> bool:
        other = Path(weights)
        try:
            return self.weights.resolve() == other.resolve()
        except OSError:
            return str(self.weights) == str(other)

    def apply(self, weights=None, conf: float | None = None, iou: float | None = None, log=None) -> bool:
        """更新 conf/iou；权重已加载且路径未变则复用，不重新读盘。"""
        if conf is not None:
            self.conf = float(conf)
        if iou is not None:
            self.iou = float(iou)
        if weights:
            new_w = Path(weights)
            if self.model is not None and self._same_weights(new_w):
                if log:
                    log(f"YOLO: 复用已加载权重 {self.weights.name}")
                return True
            if self.model is not None and not self._same_weights(new_w):
                self.model = None
                self.names = {}
                self._warmed = False
            self.weights = new_w
        return self.ensure_loaded(log=log)

    def ensure_loaded(self, log=None) -> bool:
        if self.model is not None:
            return True
        with self._load_lock:
            if self.model is not None:
                return True
            if not self.weights.is_file():
                if log:
                    log(f"YOLO: 找不到权重 {self.weights}")
                return False
            try:
                ensure_cuda_runtime()
                from ultralytics import YOLO
                import numpy as np
                import torch

                if log:
                    log(f"YOLO: 正在加载 {self.weights} ...")
                self.model = YOLO(str(self.weights))
                self.names = dict(self.model.names or {})
                if torch.cuda.is_available():
                    self.model.to("cuda")
                dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
                self._predict(dummy)
                self._warmed = True
                if log:
                    log(f"YOLO: 加载成功并已 GPU warmup，类别={list(self.names.values())}")
                return True
            except Exception as e:
                if log:
                    log(f"YOLO: 加载失败 - {e}")
                self.model = None
                self._warmed = False
                return False

    def _predict(self, frame, conf: float | None = None, iou: float | None = None):
        """复用同一 YOLO/predictor；参数保持稳定以免重建。"""
        return self.model.predict(
            source=frame,
            conf=self.conf if conf is None else float(conf),
            iou=self.iou if iou is None else float(iou),
            imgsz=self._imgsz,
            device=0,
            verbose=False,
            save=False,
            stream=False,
        )

    def try_infer_features(
        self,
        frame,
        conf: float | None = None,
        iou: float | None = None,
        return_result: bool = False,
    ):
        """
        若正在推理则立即返回 None（调用方可跳过该帧）。
        成功时返回特征 dict（不含 t_ns，由调用方在截图完成时打戳）。
        return_result=True 时额外带 ultralytics Result，供预览窗画框（不要写入 jsonl）。
        """
        if self.model is None:
            return None
        if not self._infer_lock.acquire(blocking=False):
            return None
        try:
            t0 = time.perf_counter()
            results = self._predict(frame, conf=conf, iou=iou)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            features = self._extract_features(results[0], infer_ms=infer_ms)
            if return_result:
                features["_result"] = results[0]
            return features
        except Exception:
            return None
        finally:
            self._infer_lock.release()

    def infer_features(self, frame, conf: float | None = None, iou: float | None = None):
        """阻塞推理。录制必须每帧都有特征，不能因 busy 跳过。"""
        if self.model is None:
            return None
        with self._infer_lock:
            t0 = time.perf_counter()
            results = self._predict(frame, conf=conf, iou=iou)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            return self._extract_features(results[0], infer_ms=infer_ms)

    @staticmethod
    def empty_features(infer_ms: float = 0.0) -> dict:
        return {
            "player_xy": None,
            "mon_xy": [],
            "loot_xy": [],
            "gate_xy": [],
            "boss_xy": [],
            "mon": 0,
            "loot": 0,
            "gate": 0,
            "boss": 0,
            "infer_ms": round(float(infer_ms), 2),
        }

    def predict_result(self, frame, conf: float | None = None, iou: float | None = None):
        """阻塞推理，返回 ultralytics Result（供 YOLO 测试预览）。"""
        if self.model is None:
            return None
        with self._infer_lock:
            results = self._predict(frame, conf=conf, iou=iou)
            return results[0]

    def _extract_features(self, result, infer_ms: float):
        """客户区绝对中心点。相对坐标 / player 填充 / 门方向都不是检测器的事。"""
        names = result.names or self.names or {}
        mon_xy: list[list[float]] = []
        loot_xy: list[list[float]] = []
        gate_xy: list[list[float]] = []
        boss_xy: list[list[float]] = []
        best_player = None  # (conf, cx, cy)

        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                label = str(names.get(cls_id, cls_id)).lower()
                xyxy = box.xyxy[0].tolist()
                cx = round((xyxy[0] + xyxy[2]) * 0.5, 2)
                cy = round((xyxy[1] + xyxy[3]) * 0.5, 2)
                pt = [cx, cy]

                if label in ("mon", "monster"):
                    mon_xy.append(pt)
                elif label == "loot":
                    loot_xy.append(pt)
                elif label == "gate":
                    gate_xy.append(pt)
                elif label == "boss":
                    boss_xy.append(pt)
                elif label == "player":
                    if best_player is None or conf > best_player[0]:
                        best_player = (conf, cx, cy)

        player_xy = None
        if best_player is not None:
            player_xy = [best_player[1], best_player[2]]

        return {
            "player_xy": player_xy,
            "mon_xy": mon_xy,
            "loot_xy": loot_xy,
            "gate_xy": gate_xy,
            "boss_xy": boss_xy,
            "mon": len(mon_xy),
            "loot": len(loot_xy),
            "gate": len(gate_xy),
            "boss": len(boss_xy),
            "infer_ms": round(infer_ms, 2),
        }
