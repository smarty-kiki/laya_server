"""桌面启动器：双击起服务，不用碰命令行。

它做的事只有四件，其余全部交给控制台页面：

  1. 决定用哪个端口（被占了就往后找，而不是直接失败）
  2. 在后台线程里跑 uvicorn
  3. 等 `/healthz` 真的通了再打开界面
  4. 把「停止服务」这条线接上（控制台的 /admin/shutdown 落到 server.should_exit）

**不自己实现界面。** UI 就是服务自带的那一页（`GET /`），启动器只是把它送进
原生窗口或浏览器。这样只有一份前端代码，改一次两边都变；反过来，如果启动器
自己画一套，用户就会看到两个互相漂移的版本。

界面只有两种打开方式：**原生窗口**（系统 WebKit，`laya_server/native.py`，装了
pyobjc 就是默认）和 **`--no-native` 交给默认浏览器**。没有第三种 —— 曾经还有
借 Chrome `--app=` / Safari AppleScript / pywebview 的三级回退，但那条链在窗口
生命周期上误杀过服务（借来的进程不保证活到窗口关闭），且要求用户装特定浏览器，
已整体删除。
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import socket
import sys
import threading
import time
import webbrowser
from dataclasses import replace
from typing import Optional, Sequence

from .config import AUTO_LOAD_MODES, HOME_DIR, Settings, load_settings

DEFAULT_PORT_SCAN = 20
HEALTH_TIMEOUT_S = 180.0


def port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def running_laya_server(host: str, port: int) -> bool:
    """这个端口上是不是**已经有一个 laya-server** 在跑。

    桌面端必须回答这个问题，否则「关掉窗口再双击一次」会起出第二个服务
    （第一次占着 8077，第二次就自动换到 8078），用户看到的是两个互不相干的实例。

    判定不能只看端口通不通 —— 那里可能是任何东西。看 `/healthz` 的响应形状：
    我们的 healthz 一定同时带 `status: ok` 和 `models`。
    """
    import httpx

    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        # trust_env=False：这是回环地址上的探测，**绝不能走代理**。公司机器上
        # 普遍设着 HTTP_PROXY，走代理的话要么被拒要么挂住等到超时 ——
        # 表现就是「明明服务在跑，却一直等不到 /healthz 就绪」。
        with httpx.Client(timeout=1.5, trust_env=False) as client:
            response = client.get(f"http://{probe_host}:{port}/healthz")
    except Exception:  # noqa: BLE001 —— 连不上就是没在跑
        return False
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return payload.get("status") == "ok" and "models" in payload


def pick_port(host: str, preferred: int, scan: int = DEFAULT_PORT_SCAN) -> tuple[int, Optional[str]]:
    """返回 (端口, 说明)。

    换端口而不是报错，是因为「8077 被别的进程占了」对双击启动的用户来说
    不是他能处理的错误。但一定要告诉他换了 —— 否则客户端连的还是老地址。
    """
    if port_available(host, preferred):
        return preferred, None
    for candidate in range(preferred + 1, preferred + 1 + scan):
        if port_available(host, candidate):
            return candidate, f"端口 {preferred} 已被占用，改用 {candidate}"
    raise SystemExit(
        f"端口 {preferred} 到 {preferred + scan} 全都不可用。"
        "用 --port 指定一个别的端口，或关掉占着这些端口的程序。"
    )


def wait_until_ready(url: str, timeout: float = HEALTH_TIMEOUT_S) -> bool:
    """轮询 /healthz。

    不能用固定 sleep：冷启动加载权重要几十秒，而且第一次还要下模型，
    时间完全不可预期。轮询到通为止，超时上限只是防止无限卡住。
    """
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # 同 running_laya_server：回环探测不吃代理环境变量。
            with httpx.Client(timeout=2.0, trust_env=False) as client:
                if client.get(url).status_code == 200:
                    return True
        except Exception:  # noqa: BLE001 —— 还没起来，什么异常都正常
            pass
        time.sleep(0.25)
    return False


def open_in_browser(url: str) -> None:
    """把控制台交给默认浏览器打开（`--no-native` 或原生壳不可用时的路径）。

    `webbrowser.open` 的返回值不可信 —— 它几乎永远返回 True，哪怕什么都没弹出来。
    所以日志必须带上地址：「没看到页面」的时候至少有一个能手敲的 URL。
    """
    webbrowser.open(url)
    print(f"[laya-console] 已交给默认浏览器。没看到页面就手动访问：{url}/", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-console",
        description="起一个 laya-server 并打开控制台。等价于 laya-server + 界面，"
        "外加一个能改配置、看统计、停服务的页面。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="JSON/YAML 配置文件路径。")
    parser.add_argument("--port", type=int, help="监听端口。被占用时会自动往后找一个。")
    parser.add_argument("--host", help="监听地址，默认只对本机开放。")
    parser.add_argument("--backend", choices=["mlx", "echo"], help="推理后端。")
    parser.add_argument(
        "--auto-load",
        dest="auto_load",
        choices=list(AUTO_LOAD_MODES),
        help="启动时加载哪些权重：local = 只加载本地已有的（默认，不联网）；"
        "all = 本地没有的就去下载；off = 全部等首个请求。",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="等价于 --auto-load all（保留这个写法，免得弄坏已经在用的启动脚本）。",
    )
    parser.add_argument("--api-key", help="设置后所有接口都要带 Authorization: Bearer <key>。")
    parser.add_argument(
        "--native",
        action="store_true",
        help="用 macOS 原生壳（系统 WebKit，不需要任何浏览器）。装了 pyobjc 时这是默认行为。",
    )
    parser.add_argument(
        "--no-native", action="store_true", help="强制走浏览器路径，不要原生壳。"
    )
    parser.add_argument("--no-browser", action="store_true", help="只起服务，不打开任何界面。")
    parser.add_argument(
        "--strict-port", action="store_true", help="端口被占用时直接失败，不自动换端口。"
    )
    parser.add_argument(
        "--no-attach",
        action="store_true",
        help="即使这个端口上已经有一个 laya-server，也另起一个（默认是直接连上去）。",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = load_settings(args.config)
    updates = {}
    for field in ("host", "port", "backend", "auto_load"):
        value = getattr(args, field, None)
        if value is not None:
            updates[field] = value
    if args.api_key is not None:
        updates["api_key"] = args.api_key
    if args.preload and "auto_load" not in updates:
        updates["auto_load"] = "all"
    return replace(settings, **updates).resolved if updates else settings


def wants_native_window(args: argparse.Namespace) -> bool:
    """要不要用 macOS 原生窗口（系统 WebKit，不依赖任何浏览器）。

    **这个判断只能有一处。** 之前 attach 分支看的是 `--window`，主路径看的是另一套
    推导，两者默认值不同 —— 表现就是「第一次双击弹原生窗口，第二次双击（连上已有的
    服务）掉进浏览器标签页」。同一件事两种结果，用户既看不出原因，也想不到是 bug。

    `--no-browser` 一律优先，它说的是「不要任何界面」。
    """
    if args.no_browser:
        return False
    return bool(args.native or not args.no_native)


def run_native_shell(url: str, *, wait_ready, on_quit) -> Optional[int]:
    """起原生窗口并阻塞到用户退出；返回退出码。

    原生壳不可用（没装 pyobjc）或者起不来时返回 None，让调用方退回浏览器路径 ——
    装不上不该等于用不了。
    """
    from . import native

    if not native.available():
        print(
            "[laya-console] 没装 pyobjc，用不了原生壳（pip install "
            "'laya-server[native]'）。退回浏览器路径。",
            flush=True,
        )
        return None

    print("[laya-console] 用 macOS 原生壳（系统 WebKit，不需要浏览器）", flush=True)
    try:
        return native.run(
            url=url,
            wait_ready=wait_ready,
            on_quit=on_quit,
            title="laya-server",
            lock_path=HOME_DIR / "app.lock",
        )
    except Exception as exc:  # noqa: BLE001 —— 壳起不来就退回浏览器
        print(
            f"[laya-console] 原生壳起不来（{type(exc).__name__}: {exc}），退回浏览器路径。",
            flush=True,
        )
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)

    import uvicorn

    from .app import create_app

    # 已经有一个在跑就直接连上去，不要再起第二个。
    #
    # ★ 这一步必须在 pick_port **之前**。反过来的话，第二个实例会先把「8077 被占」
    #   解释成「换个端口」，于是 attach 探测打到了空着的 8078 上、当然探测不到，
    #   接着就在 8078 起了一个新服务 —— 用户凭空多出一个看不见的实例，
    #   而他想看到的那个「连上已有服务」的行为一次也不会发生。
    #
    # 桌面端最容易踩的就是这个：用户关掉窗口（服务还在），再双击一次图标。
    # 判定要认「是不是 laya-server」，不能只看端口通不通。
    if not args.no_attach and running_laya_server(settings.host, settings.port):
        display_host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
        base_url = f"http://{display_host}:{settings.port}"
        print(
            f"[laya-console] {base_url} 已经有一个 laya-server 在跑，直接连上去（不重复启动）。",
            flush=True,
        )
        if args.no_browser:
            return 0
        # 和「自己起服务」那条路走同一套界面决策，否则会一块儿出现两种表现。
        if wants_native_window(args):
            code = run_native_shell(
                f"{base_url}",
                wait_ready=lambda t: wait_until_ready(f"{base_url}/healthz", t),
                # 连上去的实例**不归我们管**：关窗口不该停掉别人起的那个服务。
                on_quit=lambda: None,
            )
            if code is not None:
                return code
        open_in_browser(base_url)
        return 0

    port = settings.port
    if not args.strict_port:
        port, note = pick_port(settings.host, port)
        if note:
            print(f"[laya-console] {note}", flush=True)
    elif not port_available(settings.host, port):
        raise SystemExit(f"端口 {port} 已被占用（--strict-port 要求直接失败）。")

    if port != settings.port:
        settings = replace(settings, port=port).resolved

    display_host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    base_url = f"http://{display_host}:{settings.port}"

    # 控制台的「停止服务」需要一个能碰到 Server 对象的回调，而 Server 又要拿到 app，
    # 所以中间放一个可变的 holder 打破这个环。
    holder: dict = {}

    def request_shutdown() -> None:
        server = holder.get("server")
        if server is not None:
            server.should_exit = True

    app = create_app(settings, on_shutdown=request_shutdown)
    config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info")
    server = uvicorn.Server(config)
    holder["server"] = server

    health_url = f"{base_url}/healthz"

    thread = threading.Thread(
        target=server.run, name="laya-server-uvicorn", daemon=True
    )
    thread.start()

    # uvicorn 只在主线程装信号处理器，所以信号得由我们接住再转成 should_exit。
    def on_signal(signum, frame):  # noqa: ANN001, ARG001
        print(f"\n[laya-console] 收到信号 {signum}，正在停止…", flush=True)
        request_shutdown()

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            with contextlib.suppress(ValueError):
                signal.signal(getattr(signal, name), on_signal)

    print(f"[laya-console] 服务地址 {base_url}", flush=True)
    print(f"[laya-console] 控制台   {base_url}/", flush=True)

    # ---------------------------------------------------------------- 原生壳
    #
    # 原生壳会占住主线程跑 Cocoa 事件循环、一直阻塞到用户退出，所以它必须在这里
    # 就地接管，而不是像浏览器路径那样「打开完就返回」。
    #
    # 它会自己等服务就绪再显示窗口（先显示再加载的话，用户会先看到一个
    # 「无法连接到服务器」的错误页），所以这里不需要预先 wait_until_ready。
    if wants_native_window(args):

        def shut_down_and_wait() -> None:
            request_shutdown()
            thread.join(timeout=8)

        code = run_native_shell(
            base_url,
            wait_ready=lambda t: wait_until_ready(health_url, t),
            on_quit=shut_down_and_wait,
        )
        if code is not None:
            return code

    if not args.no_browser and not wait_until_ready(health_url):
        print(
            "[laya-console] 健康检查超时，界面照样给你打开，但服务可能还没就绪。",
            flush=True,
        )

    if not args.no_browser:
        open_in_browser(base_url)

    try:
        # 只等一件事：服务自己被停（控制台按钮 / 信号 / 崩溃）。
        # 原生壳路径到不了这里（它在上面阻塞到用户退出才返回），
        # 浏览器路径下窗口归浏览器管，我们既不需要也不应该盯着它。
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        request_shutdown()
        thread.join(timeout=10)

    print("[laya-console] 已停止。", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
