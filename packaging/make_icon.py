#!/usr/bin/env python3
"""从一张满幅方图生成 macOS 应用图标（.icns）。

用法：
    .venv/bin/python packaging/make_icon.py                     # 用默认母版
    .venv/bin/python packaging/make_icon.py <源图.png>           # 换一张

母版：assets/icon/laya-server-icon-1024.png（1024×1024，满幅出血，
主形是「单输入 → 三条类型化出口」的芯片分叉图）。产物写到
packaging/LayaServer.icns，被 laya_server.spec 和 make_macos_app.sh 引用。

为什么要额外做一层几何处理，而不是直接把方图塞进 iconutil：

  macOS 不会像 iOS 那样自动给应用图标加圆角遮罩。系统按原样显示 .icns 里的
  位图，所以「圆角方形 + 四周透明留白」必须画进图里。Apple 的规范是 1024 画布、
  824 见方的图标体居中，四角是连续曲率的圆角矩形（≈ 超椭圆）。

  直接把满幅方图当图标，结果是 Dock 里一个方角黑块 —— 能用，但一看就不是
  macOS 原生应用。

不画投影：投影会撑大图标的实际外接尺寸，在浅色界面上不好控制。要更贴近 Apple
模板的话，给画布加一层柔和的接触投影即可，改动在 _tile() 里。

依赖 Pillow 和 numpy（打包链路里本来就有），以及系统自带的 iconutil。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO / "assets" / "icon" / "laya-server-icon-1024.png"
OUT_ICNS = REPO / "packaging" / "LayaServer.icns"

CANVAS = 1024   # .icns 的最大边长
BODY = 824      # Apple 规范：图标体占画布 824/1024
SUPERN = 5.0    # 超椭圆指数，近似 Apple 的连续曲率圆角
SS = 4          # 遮罩超采样倍率，用来把圆角边缘磨平

# iconutil 要求的文件名 → 边长
ICONSET = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}


def _squircle_mask(size: int, body: int | None = None, n: float = SUPERN) -> Image.Image:
    """超椭圆遮罩：形状在 size×size 画布上占 body 见方、居中。body 省略即贴满。"""
    body = size if body is None else body
    big = size * SS
    R = body * SS / 2.0
    ax = np.arange(big) - (big - 1) / 2.0
    d = (np.abs(ax[None, :]) / R) ** n + (np.abs(ax[:, None]) / R) ** n
    # |∇d| ≈ n/R，所以 (1-d)*R/n 是「离边界还有几个超采像素」，换算到最终像素
    # 再除以 1.5 得到一条约 1.5px 的抗锯齿过渡带。
    a = np.clip((1.0 - d) * R / (n * SS * 1.5) + 0.5, 0.0, 1.0)
    mask = Image.fromarray((a * 255).astype(np.uint8), "L")
    return mask.resize((size, size), Image.LANCZOS)


def _tile(src: Path) -> Image.Image:
    """满幅方图 → 1024 画布上的圆角图标体（RGBA，四角透明）。"""
    art = Image.open(src).convert("RGB")
    if art.size != (CANVAS, CANVAS):
        art = art.resize((CANVAS, CANVAS), Image.LANCZOS)

    # 满幅设计缩成 824 的图标体，再套 824 见方的超椭圆遮罩后居中贴上
    body = art.resize((BODY, BODY), Image.LANCZOS).convert("RGBA")
    body.putalpha(_squircle_mask(BODY))
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.alpha_composite(body, ((CANVAS - BODY) // 2, (CANVAS - BODY) // 2))
    return canvas


def main() -> int:
    src = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.is_file():
        print(f"找不到源图 {src}", file=sys.stderr)
        return 1

    tile = _tile(src)
    iconset = REPO / "build" / "LayaServer.iconset"
    iconset.mkdir(parents=True, exist_ok=True)

    # 从 1024 的成品逐级降采样，而不是各自独立地缩原图 ——
    # 圆角边缘和发光描边在小尺寸下才会保持一致。
    cache: dict[int, Image.Image] = {}
    for name, px in ICONSET.items():
        if px not in cache:
            cache[px] = tile.resize((px, px), Image.LANCZOS)
        cache[px].save(iconset / name)

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(OUT_ICNS)],
        check=True,
    )
    print(f"✓ {OUT_ICNS.relative_to(REPO)}  ({OUT_ICNS.stat().st_size // 1024} KB)")
    print(f"  源图 {src.relative_to(REPO) if src.is_relative_to(REPO) else src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
