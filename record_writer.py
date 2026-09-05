"""采集写盘队列：键事件与帧索引异步落盘为 jsonl。"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path


class RecordWriter:
    """
    单写盘线程消费队列。
    条目约定:
      {"stream": "keys"|"frames"|"meta", ...}
      {"stream": "_stop"} 结束
    """

    def __init__(self, session_dir: str | Path, log=None):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self._q: queue.Queue = queue.Queue(maxsize=20000)
        self._thread = None
        self._running = False
        self.keys_path = self.session_dir / "keys.jsonl"
        self.frames_path = self.session_dir / "frames.jsonl"
        self.meta_path = self.session_dir / "meta.json"
        self._keys_f = None
        self._frames_f = None
        self.keys_count = 0
        self.frames_count = 0
        self.dropped = 0

    def start(self):
        if self._running:
            return
        self._keys_f = open(self.keys_path, "a", encoding="utf-8")
        self._frames_f = open(self.frames_path, "a", encoding="utf-8")
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="record-writer", daemon=True)
        self._thread.start()

    def enqueue(self, item: dict) -> bool:
        if not self._running:
            return False
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def write_meta(self, meta: dict):
        """同步写 meta（启动/结束时调用即可）。"""
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def stop(self, timeout: float = 5.0):
        if not self._running:
            return
        try:
            self._q.put({"stream": "_stop"}, timeout=1.0)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._running = False
        self._thread = None
        for f in (self._keys_f, self._frames_f):
            if f:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
        self._keys_f = None
        self._frames_f = None
        if self.log:
            self.log(
                f"采集写盘: keys={self.keys_count} frames={self.frames_count} "
                f"dropped={self.dropped} -> {self.session_dir}"
            )

    def _loop(self):
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                if not self._running:
                    break
                continue
            if not item or item.get("stream") == "_stop":
                # 排空剩余
                while True:
                    try:
                        more = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if more and more.get("stream") != "_stop":
                        self._write_item(more)
                break
            self._write_item(item)

    def _write_item(self, item: dict):
        stream = item.get("stream")
        payload = {k: v for k, v in item.items() if k != "stream"}
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            if stream == "keys" and self._keys_f:
                self._keys_f.write(line)
                self.keys_count += 1
                if self.keys_count % 50 == 0:
                    self._keys_f.flush()
            elif stream == "frames" and self._frames_f:
                self._frames_f.write(line)
                self.frames_count += 1
                if self.frames_count % 10 == 0:
                    self._frames_f.flush()
        except Exception as e:
            if self.log:
                self.log(f"采集写盘错误: {e}")
