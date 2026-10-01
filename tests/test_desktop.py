"""桌面启动器的测试。

这里只测**不需要真的起界面**的那部分：端口挑选、attach 判定、启动顺序。
窗口由原生壳负责（tests/test_native.py）；浏览器路径就是一句 webbrowser.open，
没什么可测的。

两个 bug 是真实踩过的，都补了回归测试：
  * attach 判定排在 pick_port 之后 —— 第二个实例把「端口被占」解释成「换端口」，
    于是探测打到空端口上、永远探测不到，然后在 8078 又起一个，用户凭空多一个实例；
  * 回环探测走代理 —— httpx 默认 trust_env=True，公司机器上普遍设着 HTTP_PROXY，
    结果是「服务明明在跑，/healthz 却一直等不到」。
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import replace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from laya_server.desktop import (
    port_available,
    pick_port,
    running_laya_server,
    wait_until_ready,
)
from laya_server.backend import EchoBackend
from laya_server.config import Settings


class RunningServer:
    """在后台线程跑一个真的 uvicorn。attach 判定必须打真端口，桩测不出来。

    **端口交给系统分配（port=0）然后从 server 对象上读回来**，不用「先找一个空闲端口
    再 bind 上去」那种写法 —— 那两步之间有窗口期，别的进程（包括上一个测试还没释放的
    监听套接字）可能正好抢走，于是测试偶发失败而代码本身没问题。
    """

    def __init__(self, app: FastAPI) -> None:
        config = uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="error", access_log=False
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.port: int = 0

    def _bound_port(self) -> int:
        for srv in getattr(self.server, "servers", []) or []:
            for sock in srv.sockets or []:
                try:
                    return int(sock.getsockname()[1])
                except OSError:
                    continue
        return 0

    def __enter__(self) -> "RunningServer":
        self.thread.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            port = self._bound_port()
            if port:
                try:
                    with httpx.Client(timeout=1.0, trust_env=False) as client:
                        if client.get(f"http://127.0.0.1:{port}/healthz").status_code == 200:
                            self.port = port
                            return self
                except Exception:  # noqa: BLE001 —— 还没 start 完
                    pass
            time.sleep(0.1)
        raise RuntimeError(f"测试用的 uvicorn 没起来（已绑定端口 {self._bound_port() or '无'}）")

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def free_port() -> int:
    """要一个「现在没人监听」的端口号。仅用于测 pick_port / wait_until_ready，
    不用于 RunningServer（那边走 port=0，避开竞态）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_laya_app(backend: str = "echo") -> FastAPI:
    from laya_server.app import create_app

    settings = replace(Settings(), backend=backend).resolved
    return create_app(settings, EchoBackend(settings))


def make_other_app() -> FastAPI:
    """长得像但不是 laya-server 的服务：只回了 200，没有我们的 healthz 形状。"""
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# 端口
# ---------------------------------------------------------------------------


def test_port_available_reports_true_for_a_free_port():
    assert port_available("127.0.0.1", free_port()) is True


def test_pick_port_falls_back_when_preferred_is_taken():
    with RunningServer(make_other_app()) as taken:
        picked, note = pick_port("127.0.0.1", taken.port, scan=5)
        assert picked != taken.port
        assert note and str(taken.port) in note
        assert port_available("127.0.0.1", picked)


def test_pick_port_keeps_preferred_when_free():
    port = free_port()
    picked, note = pick_port("127.0.0.1", port)
    assert (picked, note) == (port, None)


# ---------------------------------------------------------------------------
# attach 判定
# ---------------------------------------------------------------------------


def test_running_laya_server_detects_our_own_service():
    with RunningServer(make_laya_app()) as server:
        assert running_laya_server("127.0.0.1", server.port) is True


def test_running_laya_server_rejects_a_lookalike():
    """只看端口通不通是不够的 —— 那里可能是任何东西，被误认就会去 attach 一个
    根本不存在的控制台。所以判定要认 /healthz 的形状。"""
    with RunningServer(make_other_app()) as server:
        assert running_laya_server("127.0.0.1", server.port) is False


def test_running_laya_server_false_when_nothing_listening():
    assert running_laya_server("127.0.0.1", free_port()) is False


def test_probe_ignores_proxy_environment(monkeypatch):
    """设置 HTTP_PROXY 之后，回环探测必须照样通。

    这是真出过的 bug：httpx 默认 trust_env=True，会读 HTTP_PROXY。/healthz 的探测
    因此被送去代理，等超时——表现为「服务在跑，界面却一直不出来」。
    """
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    with RunningServer(make_laya_app()) as server:
        assert running_laya_server("127.0.0.1", server.port) is True
        assert wait_until_ready(f"http://127.0.0.1:{server.port}/healthz", timeout=5) is True


def test_wait_until_ready_times_out_on_a_dead_port():
    started = time.monotonic()
    assert wait_until_ready(f"http://127.0.0.1:{free_port()}/healthz", timeout=1.0) is False
    assert time.monotonic() - started < 5


# ---------------------------------------------------------------------------
# 启动顺序
# ---------------------------------------------------------------------------


def test_attach_check_runs_before_port_fallback(monkeypatch):
    """attach 判定必须排在 pick_port 之前。

    否则第二个实例会先把「端口被占」当成「换端口」，attach 探测打到空端口上、
    永远探测不到，然后在下一个端口又起一个服务 —— 用户凭空多一个看不见的实例，
    而「连上已有服务」这条路径一次也不会被执行到。

    测法是盯住 pick_port：一旦它被调用，就说明顺序错了。
    """
    from laya_server import desktop

    calls: list[str] = []

    def spy_pick_port(host, preferred, scan=20):
        calls.append("pick_port")
        return preferred, None

    with RunningServer(make_laya_app()) as server:
        monkeypatch.setattr(desktop, "pick_port", spy_pick_port)
        # --no-browser：不起真的窗口，跑完就该干净退出。
        exit_code = desktop.main(["--port", str(server.port), "--no-browser"])

    assert exit_code == 0
    assert calls == [], "attach 命中时不该去挑端口"


def test_no_attach_bypasses_the_check(monkeypatch):
    from laya_server import desktop

    calls: list[str] = []

    def spy_pick_port(host, preferred, scan=20):
        calls.append("pick_port")
        return preferred, None

    with RunningServer(make_laya_app()) as server:
        monkeypatch.setattr(desktop, "pick_port", spy_pick_port)
        # --no-attach + --strict-port：跳过 attach，直接发现端口被占并失败。
        with pytest.raises(SystemExit):
            desktop.main(["--port", str(server.port), "--no-browser", "--no-attach", "--strict-port"])

    assert calls == [], "--strict-port 不该调用 pick_port"


def test_both_launch_paths_agree_on_the_window():
    """「连上已有服务」和「自己起服务」必须给出同一个界面决策。

    真出过这个 bug：attach 分支看的是 `--window`、主路径看的是另一套推导，
    默认值不同 —— 表现是「第一次双击弹原生窗口，第二次双击掉进浏览器标签页」。
    同一件事两种结果，用户既看不出原因，也想不到是 bug。
    """
    from laya_server.desktop import build_parser, wants_native_window

    parser = build_parser()
    assert wants_native_window(parser.parse_args([])) is True, "默认就该是原生窗口"
    assert wants_native_window(parser.parse_args(["--native"])) is True
    assert wants_native_window(parser.parse_args(["--no-native"])) is False
    assert wants_native_window(parser.parse_args(["--no-browser"])) is False
    # --no-browser 说的是「不要任何界面」，它压过 --native。
    assert wants_native_window(parser.parse_args(["--native", "--no-browser"])) is False
