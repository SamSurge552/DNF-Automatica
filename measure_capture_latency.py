"""重测 capture() 打戳前耗时（TRAIN_ALIGN.md 第 2 节）。不改录制打戳逻辑。"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from screen_capture import ScreenCaptureModule


def _stats(ms: list[float]) -> dict:
    xs = sorted(ms)
    n = len(xs)
    return {
        "n": n,
        "median_ms": round(statistics.median(xs), 2),
        "mean_ms": round(statistics.mean(xs), 2),
        "min_ms": round(xs[0], 2),
        "max_ms": round(xs[-1], 2),
        "p90_ms": round(xs[min(n - 1, int(n * 0.9))], 2),
    }


def main(n: int = 40, warmup: int = 8):
    cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8"))
    rect = cfg["region"]["rect"]
    cap = ScreenCaptureModule()
    times: list[float] = []
    backend = None
    for i in range(warmup + n):
        t0 = time.perf_counter_ns()
        result = cap.capture(rect=rect, gui=None, save=False)
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        if not result or result.get("frame") is None:
            print("截图失败，中止")
            return
        backend = result.get("backend")
        if i >= warmup:
            times.append(dt_ms)
    s = _stats(times)
    print(f"backend={backend} region={rect['width']}x{rect['height']}")
    print(
        f"capture() wall (warmup={warmup}): median={s['median_ms']}ms "
        f"mean={s['mean_ms']}ms range={s['min_ms']}-{s['max_ms']} p90={s['p90_ms']} n={s['n']}"
    )
    print("说明: 录制仍在 capture() 返回后打 t_ns；本数是打戳前墙钟，不是帧内容的采集时刻。")
    print("dxcam get_latest_frame 可能返回已缓冲的上一帧，墙钟短 ≠ 画面更新延迟小。")


if __name__ == "__main__":
    main()
