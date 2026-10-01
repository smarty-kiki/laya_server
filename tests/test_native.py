"""原生壳里**不需要图形界面**的那部分测试。

窗口、菜单、WKWebView 这些东西没法在测试里断言（要真起一个 Cocoa 事件循环，
而且得占住主线程）。所以这里只测那些一旦写错就会静默出问题的纯逻辑：
跳转判定和单实例锁。GUI 部分靠人工跑一遍 + 看服务端日志确认 ——
控制台自己的 JS 会每 3 秒轮询 `/admin/*`，日志里出现这些请求就说明
WKWebView 真的把页面跑起来了。
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from laya_server import native

BASE = "http://127.0.0.1:8077"


# ---------------------------------------------------------------------------
# 可用性
# ---------------------------------------------------------------------------


def test_available_matches_importability():
    """available() 必须如实反映「能不能 import」，不能用平台字符串猜 ——
    装的 wheel 和系统版本对不上（cp313 的包在 3.11 上）时它也是不可用的。"""
    try:
        import AppKit  # noqa: F401
        import WebKit  # noqa: F401
    except Exception:
        assert native.available() is False
    else:
        assert native.available() is True


# ---------------------------------------------------------------------------
# 站外跳转判定
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        f"{BASE}/",
        f"{BASE}/admin/stats",
        f"{BASE}/docs",
        "http://localhost:8077/",  # 同一个服务的另一种写法
        "http://127.0.0.1:8077",
        "about:blank",
        "data:text/html,<p>x</p>",
        "#anchor",
    ],
)
def test_internal_navigation_is_allowed(target):
    assert native.is_external_url(target, BASE) is False


@pytest.mark.parametrize(
    "target",
    [
        "https://docs.typesafe.ai/api",
        "https://example.com",
        "http://127.0.0.1:9999/",  # 同机不同端口 = 别的服务
        "https://127.0.0.1:8077/",  # 同主机但换了协议
    ],
)
def test_external_navigation_goes_to_the_browser(target):
    assert native.is_external_url(target, BASE) is True


def test_localhost_and_loopback_are_the_same_origin():
    """`localhost:8077` 和 `127.0.0.1:8077` 是同一个服务。

    按字符串前缀比会把它判成站外，于是点一下就在浏览器里多开一个标签 ——
    对用户来说是纯打扰，而且他会以为是应用出错了。
    """
    assert native.is_external_url("http://localhost:8077/x", "http://127.0.0.1:8077") is False
    assert native.is_external_url("http://127.0.0.1:8077/x", "http://localhost:8077") is False


def test_default_port_is_normalised():
    assert native.is_external_url("https://example.com/a", "https://example.com:443/b") is False
    assert native.is_external_url("http://example.com/a", "http://example.com:80/b") is False


# ---------------------------------------------------------------------------
# 单实例锁
# ---------------------------------------------------------------------------


def test_lock_is_exclusive_and_releasable(temp_dir):
    path = temp_dir / "app.lock"
    first = native.SingleInstanceLock(path)
    second = native.SingleInstanceLock(path)

    assert first.acquire("pid=1 url=http://127.0.0.1:8077") is True
    assert second.acquire("pid=2") is False
    assert first.holder() == "pid=1 url=http://127.0.0.1:8077"

    first.release()
    assert second.acquire("pid=2") is True
    second.release()


def test_lock_creates_parent_directory(temp_dir):
    """首次运行时 ~/.laya-server 可能还不存在。"""
    path = temp_dir / "nested" / "deeper" / "app.lock"
    lock = native.SingleInstanceLock(path)
    assert lock.acquire("pid=1") is True
    assert path.is_file()
    lock.release()


def test_lock_is_released_when_the_process_dies(temp_dir):
    """锁必须随进程消亡自动释放。

    否则用户强退一次（或崩溃一次）之后，那个文件永远锁着，应用再也起不来 ——
    而且报错信息会指向「已经在运行」，跟他实际做的事完全对不上。
    这里靠子进程真死一次来验证，而不是靠「我们记得调了 release」。

    用 subprocess 而不是 multiprocessing：后者默认的 spawn 上下文没法 pickle
    定义在测试函数里的目标函数，而把辅助函数提到模块级只是为了这一个断言，
    不划算。子进程里 import 得到 `laya_server`（本包是 editable 安装）。
    """
    path = temp_dir / "app.lock"
    code = (
        "import pathlib, sys, time\n"
        "from laya_server.native import SingleInstanceLock\n"
        f"lock = SingleInstanceLock(pathlib.Path({str(path)!r}))\n"
        "assert lock.acquire('child'), 'child could not lock'\n"
        "print('locked', flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "locked", "子进程没能拿到锁"

        contender = native.SingleInstanceLock(path)
        assert contender.acquire("parent") is False, "子进程持锁时不该拿得到"

        child.kill()
        child.wait(timeout=20)

        deadline = time.monotonic() + 10
        acquired = False
        while time.monotonic() < deadline:
            if contender.acquire("parent"):
                acquired = True
                break
            time.sleep(0.2)
        assert acquired, "子进程死了，锁却没释放"
        contender.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_holder_is_empty_when_unreadable(temp_dir):
    lock = native.SingleInstanceLock(temp_dir / "missing.lock")
    assert lock.holder() == ""
