"""PyInstaller 的入口脚本。

打包出来的 `.app` 是 GUI 应用（spec 里 `console=False`）：双击启动时**没有终端**。
而 `print()` 在这个项目里用得很密 —— 哪份权重从哪加载的、花了多久、为什么失败，
全在 stdout 上。不接住它们，「双击没反应」就又变成一件无从查起的事
（这个坑我们在薄壳 `.app` 上已经踩过一次了）。

所以要做的第一件事就是在**导入任何 `laya_server` 模块之前**把输出接走：
`config.py` 在导入期就读 `LAYA_SERVER_HOME`，晚了就连日志该写哪都不知道。

第二件事才是调 `laya_server.desktop:main`。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: 只留一份、超过 2MB 就轮转 —— 这是控制台应用的日志，不是审计日志。
_MAX_LOG_BYTES = 2 * 1024 * 1024


def _redirect_output_to_log() -> Path | None:
    """把 stdout/stderr 落到 `~/.laya-server/app.log`，返回日志路径。"""
    home = Path(os.environ.get("LAYA_SERVER_HOME") or "~/.laya-server").expanduser()
    try:
        home.mkdir(parents=True, exist_ok=True)
        log = home / "app.log"
        if log.is_file() and log.stat().st_size > _MAX_LOG_BYTES:
            log.replace(log.with_suffix(".log.1"))
        stream = open(log, "a", encoding="utf-8", buffering=1)
    except OSError:
        # 写不了日志不该让 app 起不来，只是下次出问题会难查一点。
        return None

    # 先接文件描述符，再换 sys.std*。
    #
    # 顺序不能反，也不能只换 Python 层的对象：Metal 和 MLX 的警告是 C 层直接
    # 写 fd 2 的（我们之前就被一条「这份 checkpoint 的温度参数会让 confidence
    # 失真」的警告提示过），只在 Python 层替换接不到它们。
    for fd in (1, 2):
        try:
            os.dup2(stream.fileno(), fd)
        except OSError:
            pass
    sys.stdout = sys.stderr = stream
    return log


def _runtime_packages_dir() -> Path | None:
    """打包后额外包的目录：`LayaServer.app/Contents/Resources/runtime`。

    里面放的是**不经过 PyInstaller** 的包（目前只有 `mlx`）—— 它必须保持着
    site-packages 里的原始相对结构，否则会找不到自己的 dylib 和 181MB 的
    Metal 着色器库。原因写在 `laya_server.spec` 顶部，动之前先读那段。
    """
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    # Contents/MacOS/laya-console → Contents/Resources/runtime
    for candidate in (exe.parents[1] / "Resources" / "runtime", exe.parent / "runtime"):
        if candidate.is_dir():
            return candidate
    return None


def main() -> int:
    log = _redirect_output_to_log()

    if log is not None:
        from datetime import datetime

        print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 启动 =====", flush=True)
        print(f"[laya-server] 日志：{log}", flush=True)
        print(
            "[laya-server] 首次使用会从 Hugging Face 下载约 2.2GB 权重，"
            "这一步可能要几分钟；进度也会写在这个文件里。",
            flush=True,
        )

    # 加在导入 laya_server 之前。mlx 是**懒导入**的（MlxBackend.laya 那个 property 里），
    # 严格说晚一点也能行，但放在这里最不容易被后来的改动破坏。
    runtime = _runtime_packages_dir()
    if runtime is not None:
        sys.path.insert(0, str(runtime))
        print(f"[laya-server] 额外包目录：{runtime}", flush=True)

    from laya_server.desktop import main as desktop_main

    return desktop_main()


if __name__ == "__main__":
    raise SystemExit(main())
