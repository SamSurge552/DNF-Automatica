"""
随时看 D:/Desktop/T/test/images/Atrain/Aset/<集> 的正负样本与各类占比（统计表）。

配对：Aset/<name>/*.png  ↔  Xout/<name>/<同名>.txt
  - 有框 → 正样本
  - 空 txt → 负样本（blank 读图/黑屏）
  - 有图无 txt → 未标注（训练前用 --ensure-empty-labels）

用法:
  python aset_balance.py
  python aset_balance.py solarw blank
  python aset_balance.py --csv
  python aset_balance.py --verbose
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "images" / "Atrain"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_NAMES = ["boss", "gate", "loot", "mon", "player"]
DEFAULT_CSV = ROOT / "aset_balance.csv"


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def list_asets(root: Path) -> list[str]:
    aset = root / "Aset"
    if not aset.is_dir():
        return []
    names = []
    for p in sorted(aset.iterdir()):
        if p.is_dir() and any(f.suffix.lower() in IMG_EXT for f in p.iterdir() if f.is_file()):
            names.append(p.name)
    return names


def load_class_names(root: Path, name: str) -> list[str]:
    for cand in (
        root / "Aset" / name / "classes.txt",
        root / "Xout" / name / "classes.txt",
    ):
        if cand.is_file():
            names = [ln.strip() for ln in cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if names:
                return names
    return list(DEFAULT_NAMES)


def class_name(names: list[str], cid: int) -> str:
    if 0 <= cid < len(names):
        return names[cid]
    return f"id{cid}"


def iter_images(image_dir: Path) -> dict[str, Path]:
    out = {}
    if not image_dir.is_dir():
        return out
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out[p.stem] = p
    return out


def parse_label(path: Path) -> list[int]:
    ids = []
    text = path.read_text(encoding="utf-8")
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ids.append(int(ln.split()[0]))
    return ids


def analyze_one(root: Path, name: str) -> dict:
    image_dir = root / "Aset" / name
    label_dir = root / "Xout" / name
    images = iter_images(image_dir)
    labels = {}
    if label_dir.is_dir():
        for p in label_dir.glob("*.txt"):
            if p.name.lower() == "classes.txt":
                continue
            labels[p.stem] = p
    names = load_class_names(root, name)
    box_n: Counter[int] = Counter()
    img_has: Counter[int] = Counter()
    pos = neg = 0
    unlabeled = []
    for stem, _img in sorted(images.items()):
        lab = labels.get(stem)
        if lab is None:
            unlabeled.append(stem)
            continue
        cids = parse_label(lab)
        if not cids:
            neg += 1
            continue
        pos += 1
        for cid in cids:
            box_n[cid] += 1
        for cid in set(cids):
            img_has[cid] += 1
    txt_only = sorted(set(labels) - set(images))
    n_img = len(images)
    return {
        "name": name,
        "n_img": n_img,
        "n_txt": len(labels),
        "paired": n_img - len(unlabeled),
        "unlabeled": unlabeled,
        "txt_only": txt_only,
        "pos": pos,
        "neg": neg,
        "box_n": box_n,
        "img_has": img_has,
        "class_names": names,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
    }


def merge_stats(parts: list[dict], title: str) -> dict:
    box_n: Counter[int] = Counter()
    img_has: Counter[int] = Counter()
    unlabeled = []
    names = parts[0]["class_names"] if parts else list(DEFAULT_NAMES)
    pos = neg = n_img = n_txt = 0
    for s in parts:
        n_img += s["n_img"]
        n_txt += s["n_txt"]
        pos += s["pos"]
        neg += s["neg"]
        unlabeled.extend(f"{s['name']}/{u}" for u in s["unlabeled"])
        box_n.update(s["box_n"])
        img_has.update(s["img_has"])
        if len(s["class_names"]) > len(names):
            names = s["class_names"]
    return {
        "name": title,
        "n_img": n_img,
        "n_txt": n_txt,
        "paired": n_img - len(unlabeled),
        "unlabeled": unlabeled,
        "txt_only": [],
        "pos": pos,
        "neg": neg,
        "box_n": box_n,
        "img_has": img_has,
        "class_names": names,
    }


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def _ratio(pos: int, neg: int) -> str:
    return f"{pos}:{neg}"


def overview_row(s: dict, unlabeled_as_neg: bool) -> dict:
    n = s["n_img"]
    u = len(s["unlabeled"])
    neg = s["neg"] + (u if unlabeled_as_neg else 0)
    pos = s["pos"]
    return {
        "set": s["name"],
        "png": n,
        "txt": s["n_txt"],
        "unlabeled": u,
        "pos": pos,
        "neg": neg,
        "pos%": _pct(pos, n),
        "neg%": _pct(neg, n),
        "正:负": _ratio(pos, neg),
        "classes": ",".join(s["class_names"]),
    }


def print_table(headers: list[str], rows: list[dict]) -> None:
    cols = []
    for h in headers:
        width = max(len(h), *(len(str(r.get(h, ""))) for r in rows)) if rows else len(h)
        cols.append((h, width))
    line = "  ".join(h.ljust(w) for h, w in cols)
    print(line)
    print("  ".join("-" * w for _, w in cols))
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(w) for h, w in cols))
    print()


def class_table_rows(stats: list[dict]) -> tuple[list[str], list[dict]]:
    """同类 schema 才能并排。每行一个 class，列为各集 img% / boxes。"""
    names = stats[0]["class_names"] if stats else []
    headers = ["class"]
    for s in stats:
        headers.extend([f"{s['name']}_img%", f"{s['name']}_boxes"])
    rows = []
    for cid, cname in enumerate(names):
        row = {"class": cname}
        for s in stats:
            n = s["n_img"]
            row[f"{s['name']}_img%"] = _pct(s["img_has"].get(cid, 0), n)
            row[f"{s['name']}_boxes"] = s["box_n"].get(cid, 0)
        rows.append(row)
    return headers, rows


def group_by_schema(stats: list[dict]) -> list[list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in stats:
        groups[tuple(s["class_names"])].append(s)
    return list(groups.values())


def write_csv(path: Path, overview: list[dict], class_blocks: list[tuple[list[str], list[dict]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overview[0].keys()) if overview else ["set"])
        f.write("# overview\n")
        w.writeheader()
        w.writerows(overview)
        for headers, rows in class_blocks:
            f.write("\n# class\n")
            cw = csv.DictWriter(f, fieldnames=headers)
            cw.writeheader()
            cw.writerows(rows)
    print(f"CSV: {path}")


def print_stats(s: dict, unlabeled_as_neg: bool) -> None:
    n = s["n_img"]
    u = len(s["unlabeled"])
    neg_eff = s["neg"] + (u if unlabeled_as_neg else 0)
    pos = s["pos"]
    print(f"== {s['name']} ==")
    print(f"  png={n}  txt={s['n_txt']}  paired={s['paired']}  unlabeled={u}  txt_only={len(s.get('txt_only') or [])}")
    print(f"  正样本(有框) {pos} {_pct(pos, n)}")
    print(f"  负样本(空txt) {s['neg']} {_pct(s['neg'], n)}")
    if u:
        tag = "计入负样本" if unlabeled_as_neg else "未计入；训练需空 txt"
        print(f"  未标注png     {u} {_pct(u, n)}  ({tag})")
    print(f"  正:负(仅空txt) {pos}:{s['neg'] or 0}" + (f"  正:负(未标注当负) {pos}:{neg_eff}" if u else ""))
    names = s["class_names"]
    all_ids = sorted(set(s["box_n"]) | set(s["img_has"]) | set(range(len(names))))
    print(f"  {'class':<10} {'imgs':>6} {'img%':>7} {'boxes':>7} {'box%':>7}")
    tot_box = sum(s["box_n"].values()) or 1
    for cid in all_ids:
        ni = s["img_has"].get(cid, 0)
        nb = s["box_n"].get(cid, 0)
        print(
            f"  {class_name(names, cid):<10} {ni:6d} {_pct(ni, n):>7} {nb:7d} {_pct(nb, tot_box):>7}"
        )
    print()


def ensure_empty_labels(root: Path, name: str) -> int:
    """给未标注 png 写空 txt，YOLO 才把它们当背景。"""
    image_dir = root / "Aset" / name
    label_dir = root / "Xout" / name
    label_dir.mkdir(parents=True, exist_ok=True)
    images = iter_images(image_dir)
    n = 0
    for stem in images:
        dest = label_dir / f"{stem}.txt"
        if dest.exists():
            continue
        dest.write_text("", encoding="utf-8")
        n += 1
    return n


def main():
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="Aset 正负样本 / 各类占比")
    ap.add_argument("sets", nargs="*", help="Aset 子目录名；省略则扫全部")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument(
        "--unlabeled-as-neg",
        action="store_true",
        help="把无 txt 的图也算进负样本（看 blank 混进去之后的比例）",
    )
    ap.add_argument(
        "--ensure-empty-labels",
        action="store_true",
        help="给指定集里还没有 txt 的 png 写空标签（blank 进训练集前用）",
    )
    ap.add_argument("--verbose", action="store_true", help="每个集再打一份明细")
    ap.add_argument("--csv", nargs="?", const=str(DEFAULT_CSV), default=None, help="写出 CSV（默认 D:/Desktop/T/test/images/Atrain/aset_balance.csv）")
    args = ap.parse_args()
    names = args.sets or list_asets(args.root)
    if not names:
        raise SystemExit(f"未找到 Aset 子目录: {args.root / 'Aset'}")

    if args.ensure_empty_labels:
        for name in names:
            n = ensure_empty_labels(args.root, name)
            print(f"空标签已补: {name} +{n} -> {args.root / 'Xout' / name}")
        print()

    unlabeled_as_neg = args.unlabeled_as_neg
    stats = [analyze_one(args.root, n) for n in names]
    if args.verbose:
        for s in stats:
            print_stats(s, unlabeled_as_neg)

    overview = [overview_row(s, unlabeled_as_neg) for s in stats]
    print("表1  正负样本")
    print_table(["set", "png", "pos", "neg", "pos%", "neg%", "正:负", "unlabeled"], overview)

    class_blocks: list[tuple[list[str], list[dict]]] = []
    print("表2  各类出现率（同类 schema 才并排；varien 单类 player 单独一张）")
    for group in group_by_schema(stats):
        if len(group) > 1:
            mixed = merge_stats(group, "MIX " + "+".join(s["name"] for s in group))
            group = group + [mixed]
        headers, rows = class_table_rows(group)
        class_blocks.append((headers, rows))
        print_table(headers, rows)

    if args.csv:
        write_csv(Path(args.csv), overview, class_blocks)


if __name__ == "__main__":
    main()
