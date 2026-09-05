"""
对齐验证：把 held_frac 标签平移 K 帧，各训一个小模型，看 val 指标 vs K。

约定（按 TRAIN_ALIGN.md §7）：
  样本 i 用特征 x[i]、标签 y[i+K]（同一段内）。
  K>0：标签来自更晚的帧 = 人看见画面后再按（反应延迟 Δ）。
  指标用 val R²（越高越好）：单峰且峰在 K=0 → 对齐正常；
  峰在 K>0 → Δ ≈ K × 名义间隔；平坦无峰 → 对齐坏或特征没信息。

不要用「训练 loss 降了」当对齐对了。

  python shift_align_check.py
  python shift_align_check.py --k-min -10 --k-max 10 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES = ROOT / "dataset_export" / "samples.jsonl"
DEFAULT_MANIFEST = ROOT / "dataset_export" / "manifest.json"
DEFAULT_OUT = ROOT / "dataset_export" / "shift_align.json"

FEAT_W, FEAT_H = 1600.0, 900.0


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def keys_from(rows: list[dict], manifest: Path | None) -> list[str]:
    vocab: set[str] = set()
    if manifest and manifest.is_file():
        man = json.loads(manifest.read_text(encoding="utf-8"))
        vocab.update(man.get("keys") or [])
    for r in rows:
        vocab.update((r.get("held_frac") or {}).keys())
    return sorted(vocab)


def feat_vec(row: dict) -> np.ndarray:
    xy = row.get("player_xy") or [0.0, 0.0]
    return np.array(
        [
            float(xy[0]) / FEAT_W,
            float(xy[1]) / FEAT_H,
            0.0 if row.get("player_xy") is None else 1.0,
            float(row.get("mon") or 0) / 20.0,
            float(row.get("loot") or 0) / 10.0,
            float(row.get("gate") or 0),
            float(row.get("boss") or 0),
            min(float(row.get("dt_s") or 0.05) / 0.2, 1.0),
        ],
        dtype=np.float32,
    )


def y_vec(row: dict, keys: list[str]) -> np.ndarray:
    frac = row.get("held_frac") or {}
    return np.array([float(frac.get(k, 0.0)) for k in keys], dtype=np.float32)


def by_session(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("session") or "")].append(r)
    for sid in groups:
        groups[sid].sort(key=lambda x: int(x["t_ns"]))
    return dict(groups)


def split_sessions(sids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    ids = list(sids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) > 1 else 0
    if n_val >= len(ids):
        n_val = max(1, len(ids) // 2) if len(ids) > 1 else 0
    val = ids[:n_val]
    train = ids[n_val:] or ids
    if not val and len(ids) > 1:
        val = [ids[-1]]
        train = ids[:-1]
    return train, val


def pair_shift(
    groups: dict[str, list[dict]],
    sids: list[str],
    keys: list[str],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for sid in sids:
        seq = groups[sid]
        n = len(seq)
        if n <= abs(k):
            continue
        for i in range(n):
            j = i + k
            if j < 0 or j >= n:
                continue
            xs.append(feat_vec(seq[i]))
            ys.append(y_vec(seq[j], keys))
    if not xs:
        return np.zeros((0, 8), np.float32), np.zeros((0, len(keys)), np.float32)
    return np.stack(xs), np.stack(ys)


class TinyNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0 if ss_res < 1e-12 else float("nan")
    return 1.0 - ss_res / ss_tot


def train_one(
    x_tr, y_tr, x_va, y_va, epochs: int, batch: int, lr: float, seed: int, device: torch.device
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyNet(x_tr.shape[1], y_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=min(batch, max(1, len(ds))), shuffle=True, drop_last=False)
    xt = torch.from_numpy(x_va).to(device)
    best_mse = math.inf
    best_pred = None
    patience, bad = 8, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xt).cpu().numpy()
            mse = float(np.mean((pred - y_va) ** 2))
        if mse + 1e-8 < best_mse:
            best_mse = mse
            best_pred = pred
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_pred is None:
        best_pred = np.zeros_like(y_va)
        best_mse = float(np.mean(y_va ** 2))
    return {
        "val_mse": round(best_mse, 6),
        "val_r2": round(r2_score(y_va, best_pred), 4),
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
    }


def ascii_curve(points: list[tuple[int, float]]) -> str:
    vals = [p[1] for p in points if p[1] == p[1]]
    if not vals:
        return "(no scores)"
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    width = 28
    lines = []
    for k, v in points:
        if v != v:
            bar = ""
            mark = " nan"
        else:
            n = int(round((v - lo) / span * width))
            bar = "#" * n
            mark = f" {v:+.3f}"
        peak = "  <-- peak" if v == hi and v == v else ""
        lines.append(f"K={k:+3d} |{bar:<{width}}|{mark}{peak}")
    return "\n".join(lines)


def interpret(best_k: int, r2: float, flat: bool, interval_s: float) -> str:
    if flat or r2 != r2 or r2 < 0.02:
        return "曲线平坦或 R² 接近 0：对齐可能错，或特征几乎没有信息量。"
    delay_s = best_k * interval_s
    if best_k == 0:
        return "峰值在 K=0：特征与 held_frac 同时对齐，时钟偏差不明显。"
    if best_k > 0:
        return (
            f"峰值在 K=+{best_k}：标签晚于画面约 {delay_s:.2f}s "
            f"（{best_k} 帧 x {interval_s:g}s）。可视为人类反应延迟 delta。"
        )
    return (
        f"峰值在 K={best_k}：标签早于画面约 {abs(delay_s):.2f}s。"
        f"更像帧戳偏晚或键被对到了上一帧特征。"
    )


def run(args) -> dict:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rows = load_rows(args.samples)
    if len(rows) < 20:
        raise SystemExit(f"样本太少（{len(rows)}），先重录再导出。")
    keys = keys_from(rows, args.manifest)
    groups = by_session(rows)
    train_s, val_s = split_sessions(list(groups.keys()), args.val_frac, args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    ks = list(range(args.k_min, args.k_max + 1))
    curve = []
    for k in ks:
        x_tr, y_tr = pair_shift(groups, train_s, keys, k)
        x_va, y_va = pair_shift(groups, val_s, keys, k)
        if len(x_tr) < 8 or len(x_va) < 4:
            curve.append({"k": k, "val_mse": None, "val_r2": None, "skipped": "too_few"})
            continue
        stats = train_one(
            x_tr, y_tr, x_va, y_va, args.epochs, args.batch, args.lr, args.seed + 1000 + k, device
        )
        stats["k"] = k
        curve.append(stats)

    scored = [c for c in curve if c.get("val_r2") is not None]
    best = max(scored, key=lambda c: c["val_r2"]) if scored else None
    r2s = [(c["k"], c["val_r2"]) for c in curve]
    finite = [c["val_r2"] for c in scored]
    flat = False
    if len(finite) >= 5:
        flat = (max(finite) - min(finite)) < 0.03 and max(finite) < 0.1

    interval = args.interval
    summary = {
        "keys": keys,
        "n_rows": len(rows),
        "train_sessions": train_s,
        "val_sessions": val_s,
        "device": str(device),
        "k_range": [args.k_min, args.k_max],
        "best_k": None if not best else best["k"],
        "best_val_r2": None if not best else best["val_r2"],
        "interval_s": interval,
        "delta_s_est": None if not best else round(best["k"] * interval, 4),
        "flat": flat,
        "read": interpret(
            0 if not best else best["k"],
            0.0 if not best else best["val_r2"],
            flat or best is None,
            interval,
        ),
        "curve": curve,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train sessions={train_s}  val sessions={val_s}  keys={keys}  device={device}")
    print(ascii_curve([(c["k"], c.get("val_r2") if c.get("val_r2") is not None else float("nan")) for c in curve]))
    print(summary["read"])
    print(f"写出 {args.out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Shift labels by K frames and plot val R²")
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--k-min", type=int, default=-10)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.05, help="名义间隔，只用于把 K 换成秒")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
