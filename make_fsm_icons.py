"""生成 FSM 回放用的手绘抽象图标（不是游戏截图像素）。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ICON_DIR = ROOT / "fsm_icons"
SIZE = 96
BG = (32, 34, 42, 255)


def _new():
    return Image.new("RGBA", (SIZE, SIZE), BG)


def _font(size: int):
    for name in ("seguiemj.ttf", "segoui.ttf", "msyh.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, text: str):
    font = _font(14)
    draw.text((6, SIZE - 20), text, fill=(220, 220, 230, 255), font=font)


def icon_player() -> Image.Image:
    im = _new()
    d = ImageDraw.Draw(im)
    # 人：圆头 + 三角身，模型里这是「有一个可定位的点」
    d.ellipse((36, 14, 60, 38), outline=(90, 200, 255, 255), width=4, fill=(40, 90, 130, 255))
    d.polygon([(48, 40), (22, 82), (74, 82)], outline=(90, 200, 255, 255), fill=(50, 110, 150, 255))
    d.line((48, 40, 48, 58), fill=(200, 230, 255, 255), width=3)
    _label(d, "player")
    return im


def icon_mon() -> Image.Image:
    im = _new()
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((18, 16, 78, 72), radius=10, outline=(230, 70, 70, 255), width=4, fill=(90, 30, 30, 255))
    d.ellipse((32, 30, 44, 42), fill=(240, 200, 200, 255))
    d.ellipse((52, 30, 64, 42), fill=(240, 200, 200, 255))
    d.arc((34, 44, 62, 64), 20, 160, fill=(240, 160, 160, 255), width=3)
    _label(d, "mon")
    return im


def icon_loot() -> Image.Image:
    im = _new()
    d = ImageDraw.Draw(im)
    d.ellipse((28, 28, 68, 68), outline=(240, 200, 50, 255), width=4, fill=(180, 140, 20, 255))
    d.ellipse((38, 36, 58, 56), outline=(255, 230, 120, 255), width=2)
    d.polygon([(48, 18), (58, 32), (38, 32)], fill=(240, 210, 80, 255))
    _label(d, "loot")
    return im


def icon_gate() -> Image.Image:
    im = _new()
    d = ImageDraw.Draw(im)
    d.polygon(
        [(18, 36), (50, 36), (50, 22), (78, 48), (50, 74), (50, 60), (18, 60)],
        fill=(40, 160, 90, 255),
        outline=(120, 255, 170, 255),
    )
    _label(d, "gate")
    return im


def icon_boss() -> Image.Image:
    im = _new()
    d = ImageDraw.Draw(im)
    cx, cy, r = 48, 42, 26
    pts = []
    import math

    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d.polygon(pts, fill=(90, 40, 140, 255), outline=(210, 140, 255, 255))
    d.ellipse((40, 34, 56, 50), fill=(40, 20, 60, 255), outline=(230, 180, 255, 255), width=2)
    _label(d, "boss")
    return im


MAKERS = {
    "player": icon_player,
    "mon": icon_mon,
    "loot": icon_loot,
    "gate": icon_gate,
    "boss": icon_boss,
}


def write_icons(out_dir: Path | None = None) -> Path:
    dest = out_dir or ICON_DIR
    dest.mkdir(parents=True, exist_ok=True)
    for name, fn in MAKERS.items():
        fn().convert("RGB").save(dest / f"{name}.png")
    return dest


if __name__ == "__main__":
    p = write_icons()
    print(f"wrote {p}")
