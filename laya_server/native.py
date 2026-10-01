"""macOS 原生壳：用系统自带的 WebKit 显示控制台，不需要任何浏览器。

和 StaffDeck 走的是同一条路（`pyobjc` → `AppKit` + `WebKit` → `WKWebView`）。
选它的理由是**窗口的生死从此由我们掌握**：借浏览器 `--app=` 时拿到的那个进程
不保证活到窗口关闭，我们只能靠猜测，而这个猜测已经造成过一次「会话中途服务被
误杀」的事故。自己持有 `NSWindow` 就没有这个猜测了。

几个必须写对、否则会静默失效的地方（都是踩过才知道的）：

* **delegate 是弱引用。** `WKWebView` 的 `UIDelegate`、`NSApplication` 的 delegate，
  Python 侧不持有就会被 GC 掉。掉了之后表现是 `alert/confirm/prompt` 永远不返回 ——
  控制台里「停止服务」按钮点了没反应，就是这个原因。所以全部挂在模块级引用上。
* **没有菜单栏，⌘C/⌘V/⌘Q 全都是废的。** Cocoa 的快捷键靠菜单项的 keyEquivalent
  派发，不是自动的。
* **页面必须等服务真的通了再加载。** 先开窗口再加载的话，用户会看到一个
  「无法连接到服务器」的错误页，一秒后再跳一下。

窗口语义上刻意和「网页应用」区分开：

* **关窗口 ≠ 退服务。** 这是个服务端应用，窗口只是它的界面；关掉窗口之后
  API 还要继续给脚本用。状态栏图标和 Dock 图标都能把窗口叫回来。
* **⌘Q / 菜单里的「停止服务并退出」才是真的停。**
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Optional

WINDOW_SIZE = (1200, 820)
WINDOW_MIN_SIZE = (960, 640)

#: 模块级引用。见上面「delegate 是弱引用」那条 —— 这些对象必须活得和进程一样久。
_SHELL: Optional["Shell"] = None
_UI_DELEGATE_CLASS = None
_NAV_DELEGATE_CLASS = None
_APP_DELEGATE_CLASS = None


def available() -> bool:
    """这台机器上能不能用原生壳（装了 pyobjc 且是 macOS）。"""
    try:
        import AppKit  # noqa: F401
        import WebKit  # noqa: F401
    except Exception:  # noqa: BLE001 —— 缺 wheel / 不是 mac，都算不可用
        return False
    return True


# ---------------------------------------------------------------------------
# 单实例
# ---------------------------------------------------------------------------


class SingleInstanceLock:
    """用文件锁保证同一时间只有一个实例。

    没有它的话双击两次会起两个服务（第二个自动换到 8078）加两个窗口，
    用户完全不知道自己正在看哪一个。锁随进程退出自动释放 —— 靠的是 fd 被内核
    关掉，不需要任何清理代码，所以进程被 kill -9 也不会留下死锁。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def acquire(self, info: str = "") -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._handle.close()
            self._handle = None
            return False
        if info:
            self._handle.seek(0)
            self._handle.truncate()
            self._handle.write(info + "\n")
            self._handle.flush()
        return True

    def holder(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def notify_already_running(detail: str) -> None:
    """已经有一个在跑了。

    这时候**不能**默默退出 —— 「双击了图标但什么都没发生」是这类工具最糟的失败
    方式。用系统弹窗说清楚，用户至少知道该去看 Dock 图标。
    """
    script = (
        'display alert "laya-server 已经在运行" message '
        f'"{detail}" as informational buttons {{"好"}} default button "好"'
    )
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        print(f"[laya-server] 已经在运行：{detail}", flush=True)


#: 指向本机的几种写法。`localhost:8077` 和 `127.0.0.1:8077` 是同一个服务，
#: 按字符串前缀比会被判成站外、白白弹一个浏览器出去。
_LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "[::1]"}


def _origin(value: str) -> Optional[tuple]:
    from urllib.parse import urlsplit

    parts = urlsplit(value)
    if not parts.hostname:
        return None
    host = parts.hostname.lower()
    if host in _LOOPBACK_NAMES:
        host = "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return (parts.scheme, host, port)


def is_external_url(target: str, base: str) -> bool:
    """这个跳转该不该交给系统浏览器。

    只对 http/https 判断：`about:blank`、`data:` 这些是 WebKit 内部用的，
    拦下来会把页面搞坏。同源的（控制台自己的路由）当然也放行。

    比较的是 **origin**（协议 + 主机 + 端口），不是字符串前缀。前缀比法会把
    `http://localhost:8077/` 判成站外 —— 那是同一个服务，弹浏览器是纯打扰。
    """
    if not target.startswith(("http://", "https://")):
        return False
    target_origin = _origin(target)
    base_origin = _origin(base)
    if target_origin is None or base_origin is None:
        return False
    return target_origin != base_origin


# ---------------------------------------------------------------------------
# 壳
# ---------------------------------------------------------------------------


class Shell:
    def __init__(
        self,
        *,
        url: str,
        wait_ready: Callable[[float], bool],
        on_quit: Callable[[], None],
        title: str,
    ) -> None:
        self.url = url
        self.wait_ready = wait_ready
        self.on_quit = on_quit
        self.title = title
        self.window = None
        self.webview = None
        self._ui_delegate = None
        self._nav_delegate = None
        self._status_item = None
        self._quitting = False

    # ------------------------------------------------------------------ 构建
    def build(self) -> None:
        import AppKit
        import WebKit

        self._appkit = AppKit
        self._webkit = WebKit

        global _UI_DELEGATE_CLASS, _NAV_DELEGATE_CLASS, _APP_DELEGATE_CLASS

        if _UI_DELEGATE_CLASS is None:

            class LayaWebViewUIDelegate(AppKit.NSObject):
                """没有它，页面里的 alert/confirm/prompt 会被 WebKit 静默丢弃 ——
                不是报错，是**永远不返回**。控制台靠 confirm 做二次确认、
                靠 prompt 收 API Key，少了这个类两个功能一起变哑巴。"""

                def webView_runJavaScriptAlertPanelWithMessage_initiatedByFrame_completionHandler_(
                    self, _webview, message, _frame, handler
                ):  # noqa: N802
                    alert = AppKit.NSAlert.alloc().init()
                    alert.setMessageText_("laya-server")
                    alert.setInformativeText_(str(message))
                    alert.addButtonWithTitle_("好")
                    alert.runModal()
                    handler()

                def webView_runJavaScriptConfirmPanelWithMessage_initiatedByFrame_completionHandler_(
                    self, _webview, message, _frame, handler
                ):  # noqa: N802
                    alert = AppKit.NSAlert.alloc().init()
                    alert.setMessageText_("laya-server")
                    alert.setInformativeText_(str(message))
                    alert.addButtonWithTitle_("确定")
                    alert.addButtonWithTitle_("取消")
                    handler(alert.runModal() == AppKit.NSAlertFirstButtonReturn)

                def webView_runJavaScriptTextInputPanelWithPrompt_defaultText_initiatedByFrame_completionHandler_(
                    self, _webview, prompt, default_text, _frame, handler
                ):  # noqa: N802
                    alert = AppKit.NSAlert.alloc().init()
                    alert.setMessageText_("laya-server")
                    alert.setInformativeText_(str(prompt))
                    alert.addButtonWithTitle_("确定")
                    alert.addButtonWithTitle_("取消")
                    field = AppKit.NSTextField.alloc().initWithFrame_(
                        AppKit.NSMakeRect(0, 0, 300, 24)
                    )
                    field.setStringValue_(str(default_text or ""))
                    alert.setAccessoryView_(field)
                    accepted = alert.runModal() == AppKit.NSAlertFirstButtonReturn
                    handler(field.stringValue() if accepted else None)

            _UI_DELEGATE_CLASS = LayaWebViewUIDelegate

        if _NAV_DELEGATE_CLASS is None:
            base = self.url

            class LayaNavigationDelegate(AppKit.NSObject):
                """站外链接交给系统浏览器。不拦的话点一个外链会把控制台顶掉，
                而应用窗口没有后退按钮 —— 用户只能重开。"""

                def webView_decidePolicyForNavigationAction_decisionHandler_(
                    self, _webview, action, handler
                ):  # noqa: N802
                    request_url = action.request().URL()
                    target = str(request_url.absoluteString()) if request_url is not None else ""
                    if is_external_url(target, base):
                        webbrowser.open(target)
                        handler(WebKit.WKNavigationActionPolicyCancel)
                        return
                    handler(WebKit.WKNavigationActionPolicyAllow)

            _NAV_DELEGATE_CLASS = LayaNavigationDelegate

        if _APP_DELEGATE_CLASS is None:
            shell = self

            class LayaAppDelegate(AppKit.NSObject):
                def applicationShouldTerminateAfterLastWindowClosed_(self, _app):  # noqa: N802
                    # False：关窗口只是把界面收起来，服务照常跑。
                    # 这是个服务端应用，窗口只是它的一个视图。
                    return False

                def applicationShouldHandleReopen_hasVisibleWindows_(self, _app, _flag):  # noqa: N802
                    # 点 Dock 图标把窗口叫回来。
                    shell.show_window()
                    return True

                def applicationWillTerminate_(self, _note):  # noqa: N802
                    shell.quit_server()

                def showConsole_(self, _sender):  # noqa: N802
                    shell.show_window()

                def openInBrowser_(self, _sender):  # noqa: N802
                    webbrowser.open(shell.url)

                def reload_(self, _sender):  # noqa: N802
                    shell.load()

                def quitServer_(self, _sender):  # noqa: N802
                    AppKit.NSApplication.sharedApplication().terminate_(None)

            _APP_DELEGATE_CLASS = LayaAppDelegate

        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
        self.app = app
        self.delegate = _APP_DELEGATE_CLASS.alloc().init()
        app.setDelegate_(self.delegate)
        app.setMainMenu_(self._build_menu())

        self.window = self._build_window()
        self.webview = self._build_webview()
        self.window.setContentView_(self.webview)
        self._build_status_item()

    def _build_window(self):
        AppKit = self._appkit
        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskMiniaturizable
            | AppKit.NSWindowStyleMaskResizable
        )
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(0, 0, *WINDOW_SIZE),
            style,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setTitle_(self.title)
        window.setMinSize_(AppKit.NSMakeSize(*WINDOW_MIN_SIZE))
        # 关键：默认关窗时窗口会被释放，之后「点 Dock 图标再打开」就是野指针。
        window.setReleasedWhenClosed_(False)
        window.center()
        return window

    def _build_webview(self):
        AppKit, WebKit = self._appkit, self._webkit
        configuration = WebKit.WKWebViewConfiguration.alloc().init()
        webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(
            AppKit.NSMakeRect(0, 0, *WINDOW_SIZE), configuration
        )
        webview.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        self._ui_delegate = _UI_DELEGATE_CLASS.alloc().init()
        self._nav_delegate = _NAV_DELEGATE_CLASS.alloc().init()
        webview.setUIDelegate_(self._ui_delegate)
        webview.setNavigationDelegate_(self._nav_delegate)
        return webview

    def _build_menu(self):
        """菜单栏。

        不是为了好看：Cocoa 的 ⌘C/⌘V/⌘A/⌘Q 全靠菜单项的 keyEquivalent 派发。
        没有菜单栏，应用窗口里连复制粘贴都是废的。Edit 菜单各项不设 target，
        让它们沿响应链走到当前聚焦的 WKWebView 上。
        """
        AppKit = self._appkit
        main_menu = AppKit.NSMenu.alloc().init()

        def add_menu(title, items):
            root = AppKit.NSMenuItem.alloc().init()
            main_menu.addItem_(root)
            submenu = AppKit.NSMenu.alloc().initWithTitle_(title)
            root.setSubmenu_(submenu)
            for item_title, action, key, target in items:
                if item_title is None:  # 分隔线
                    submenu.addItem_(AppKit.NSMenuItem.separatorItem())
                    continue
                item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    item_title, action, key
                )
                if target is not None:
                    item.setTarget_(target)
                submenu.addItem_(item)
            return submenu

        add_menu(
            "laya-server",
            [
                ("显示主窗口", "showConsole:", "", self.delegate),
                ("在浏览器中打开", "openInBrowser:", "", self.delegate),
                (None, None, None, None),
                ("隐藏", "hide:", "h", None),
                ("隐藏其他", "hideOtherApplications:", "h", None),
                (None, None, None, None),
                ("停止服务并退出", "quitServer:", "q", self.delegate),
            ],
        )
        add_menu(
            "编辑",
            [
                ("撤销", "undo:", "z", None),
                ("重做", "redo:", "Z", None),
                (None, None, None, None),
                ("剪切", "cut:", "x", None),
                ("复制", "copy:", "c", None),
                ("粘贴", "paste:", "v", None),
                ("全选", "selectAll:", "a", None),
            ],
        )
        add_menu("视图", [("重新载入控制台", "reload:", "r", self.delegate)])
        return main_menu

    def _build_status_item(self):
        """菜单栏常驻图标。

        它的价值在于「关掉窗口之后，服务还在不在」这件事有个地方能看见 ——
        否则用户只能靠 ps 去猜。
        """
        AppKit = self._appkit
        item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(
            AppKit.NSVariableStatusItemLength
        )
        button = item.button()
        button.setTitle_("laya")
        button.setToolTip_(self.url)
        menu = AppKit.NSMenu.alloc().init()
        for title, action in (
            ("打开控制台", "showConsole:"),
            ("在浏览器中打开", "openInBrowser:"),
            (None, None),
            ("停止服务并退出", "quitServer:"),
        ):
            if title is None:
                menu.addItem_(AppKit.NSMenuItem.separatorItem())
                continue
            menu_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, action, ""
            )
            menu_item.setTarget_(self.delegate)
            menu.addItem_(menu_item)
        item.setMenu_(menu)
        self._status_item = item

    # ------------------------------------------------------------------ 运行
    def start_when_ready(self, timeout_s: float) -> None:
        """在后台线程等服务就绪，然后回主线程把窗口显示出来。

        顺序不能反：先显示再加载的话，用户会先看到一个「无法连接到服务器」的错误页。
        """

        def worker():
            try:
                ready = bool(self.wait_ready(timeout_s))
            except Exception:  # noqa: BLE001 —— 等就绪本身出错也算没就绪
                ready = False
            from PyObjCTools import AppHelper

            AppHelper.callAfter(self._present, ready)

        threading.Thread(target=worker, name="laya-shell-ready", daemon=True).start()

    def _present(self, ready: bool) -> None:
        if ready:
            self.load()
        else:
            self.load_message(
                "服务没有就绪",
                "控制台暂时连不上。窗口先开着，服务起来之后按 ⌘R 重新载入。",
            )
        self.show_window()

    def load(self) -> None:
        import Foundation

        page = Foundation.NSURL.URLWithString_(self.url)
        if page is None:
            return
        self.webview.loadRequest_(Foundation.NSURLRequest.requestWithURL_(page))

    def load_message(self, heading: str, body: str) -> None:
        import Foundation

        html = (
            "<!DOCTYPE html><meta charset='utf-8'><body style=\"font:14px -apple-system,"
            "'PingFang SC';padding:48px;color:#1c1e21\">"
            f"<h2 style='font-weight:500'>{heading}</h2>"
            f"<p style='color:#5a5f68'>{body}</p></body>"
        )
        self.webview.loadHTMLString_baseURL_(html, Foundation.NSURL.URLWithString_(self.url))

    def show_window(self) -> None:
        AppKit = self._appkit
        if self.window is None:
            return
        self.window.makeKeyAndOrderFront_(None)
        AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def quit_server(self) -> None:
        """退出前把服务停掉。

        会被调用两次（菜单的「停止服务并退出」和 `applicationWillTerminate_`），
        用标志位保证只真的停一次 —— uvicorn 收到第二次 should_exit 不会有事，
        但日志里会出现两遍「正在停止」，看起来像出了故障。
        """
        if self._quitting:
            return
        self._quitting = True
        try:
            self.on_quit()
        except Exception as exc:  # noqa: BLE001 —— 退出路径不能因为异常卡住
            print(f"[laya-server] 停止服务时出错：{exc}", flush=True)


def run(
    *,
    url: str,
    wait_ready: Callable[[float], bool],
    on_quit: Callable[[], None],
    title: str = "laya-server",
    timeout_s: float = 180.0,
    lock_path: Optional[Path] = None,
) -> int:
    """起原生窗口并阻塞到用户退出。返回进程退出码。

    调用方负责把服务跑起来（原生壳只拥有窗口，不拥有服务）。
    """
    import AppKit
    from PyObjCTools import AppHelper

    lock = None
    if lock_path is not None:
        lock = SingleInstanceLock(lock_path)
        if not lock.acquire(info=f"pid={os.getpid()} url={url}"):
            holder = lock.holder()
            notify_already_running(
                f"laya-server 已经开着了{('（' + holder + '）') if holder else ''}。"
                "请点 Dock 或菜单栏上的图标把窗口叫回来。"
            )
            return 1

    global _SHELL
    shell = Shell(url=url, wait_ready=wait_ready, on_quit=on_quit, title=title)
    _SHELL = shell  # 模块级强引用，见文件头「delegate 是弱引用」
    try:
        shell.build()
        shell.start_when_ready(timeout_s)
        AppHelper.runEventLoop()
    finally:
        if lock is not None:
            lock.release()
    return 0


__all__ = [
    "Shell",
    "SingleInstanceLock",
    "available",
    "is_external_url",
    "notify_already_running",
    "run",
]
