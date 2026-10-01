"""本机资源占用：CPU / 内存 / GPU / MLX 显存。

只做 macOS。这是刻意的：这个服务的推理后端是 MLX，而 MLX 只在 Apple Silicon
上有意义；与其写一堆跨平台分支再声称它们可用，不如把「读不到」明确说出来。

**不引入 psutil**，理由有两条：

1. GPU 利用率 psutil 根本拿不到（它没有 Metal 的概念），一定要自己读
   IOAccelerator。既然 GPU 必须自己解析，再为 CPU/内存多引一个 C 扩展依赖
   （安装、架构、wheel 三件麻烦事）就不划算了。
2. 系统自带接口已经够用，而且更快：CPU 用 mach 调用是微秒级，比任何
   「起个 `ps` 再解析输出」的方案都准 —— 后面那种还得等采样间隔。

四类数据源，各自的特点写在对应函数上：

    mach host_statistics    CPU 累计 tick → 求增量得使用率。微秒级
    mach task_info          本进程 RSS。比 `ps` 直接，也不用起子进程
    sysctl / vm_stat        整机内存与 swap
    ioreg                   GPU 利用率与显存。约 90ms，所以带 TTL 缓存
    mlx.core                模型真实占用。只有导入过 mlx 才有

采集一律**尽力而为**：任何一项失败就那一项标 `available: false`，其余照常返回。
一个监控面板不该因为读不到 GPU 就把页面搞崩。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

#: 两次真采样之间的最小间隔。控制台 3 秒轮询一次，这里挡掉重复的并发采集
#: （多个页面同时打开时尤其有用），又不至于把数据缓存到看不出变化。
DEFAULT_TTL = 2.0

#: 子进程超时。ioreg 正常 90ms；超过这个数说明系统状态不对，宁可不要这个数据。
_SUBPROCESS_TIMEOUT = 4.0

_MB = 1024 * 1024

#: 间隔超过这么久就重新起一个采样窗口。开机几小时后再看「CPU 使用率」，
#: 拿几小时的平均值当实时值是一种误导 —— 宁可重新等一小会儿。
_MAX_WINDOW_S = 120.0

#: 采样窗口下限。这个值不是随便定的：内核刷新 host_statistics 的 CPU 计数器
#: **有粒度**，实测 0.25 秒的窗口里大约五次就有一次拿到和上次完全一样的读数
#: （两个窗口的 tick 增量是 0），而 0.5 秒的窗口连续 20 次都没有出过 0。
#: 取 0.5 是这个粒度下的安全值。
_MIN_WINDOW_S = 0.5

#: 万一窗口够宽仍然撞上没刷新的那一拍：再读几次等它往前走。
#: 放弃并报错会把一次本可以成功的采样变成一格「—」，不值得。
_ZERO_DELTA_RETRIES = 3
_ZERO_DELTA_WAIT_S = 0.3


def _run(cmd: List[str], timeout: float = _SUBPROCESS_TIMEOUT) -> Optional[str]:
    """跑一个只读的系统命令。失败返回 None —— 调用方按「这项没有」处理。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _mb(value: float) -> float:
    return round(value / _MB, 1)


# ---------------------------------------------------------------------------
# sysctl：一次调用读多个 oid（省掉四五次子进程）
# ---------------------------------------------------------------------------

#: 固定按这个顺序读，输出按行对应。
_SYSCTL_KEYS = ("hw.memsize", "hw.pagesize", "hw.model", "machdep.cpu.brand_string", "vm.swapusage")


def _sysctl(keys: tuple = _SYSCTL_KEYS) -> Dict[str, str]:
    """批量读 sysctl。`sysctl -n a b c` 会按顺序每行一个值。

    合并成一次调用是有意义的：单次约 10ms，五项分开读就是 50ms，
    而这是个每几秒就要采一次的路径。
    """
    out = _run(["sysctl", "-n", *keys])
    if out is None:
        return {}
    lines = out.splitlines()
    if len(lines) < len(keys):
        return {}
    return dict(zip(keys, (line.strip() for line in lines)))


def parse_swapusage(text: str) -> Dict[str, float]:
    """解析 `sysctl vm.swapusage` 的输出。

    形如：`total = 6144.00M  used = 4878.56M  free = 1265.44M  (encrypted)`。
    单位只能是 M，但这里仍然按数值算，不假设单位。
    """
    match = re.search(
        r"total\s*=\s*([\d.]+)M\s+used\s*=\s*([\d.]+)M\s+free\s*=\s*([\d.]+)M", text
    )
    if not match:
        return {}
    return {
        "total_mb": round(float(match.group(1)), 1),
        "used_mb": round(float(match.group(2)), 1),
        "free_mb": round(float(match.group(3)), 1),
    }


# ---------------------------------------------------------------------------
# vm_stat
# ---------------------------------------------------------------------------

#: `Pages free:  5414.` / `"Translation faults":  123.` —— 键可能带引号，值后面有个点。
_VM_STAT_LINE = re.compile(r'^\s*"?([A-Za-z][^":]*?)"?:\s+(\d+)\.?\s*$')


def parse_vm_stat(text: str) -> Dict[str, int]:
    """`vm_stat` 输出 → {页类别: 页数}。

    键名统一成小写下划线（`Pages wired down` → `pages_wired_down`），
    这样调用方不用记住原文里的空格和大小写。
    """
    pages: Dict[str, int] = {}
    for line in text.splitlines():
        match = _VM_STAT_LINE.match(line)
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        pages[key] = int(match.group(2))
    return pages


# ---------------------------------------------------------------------------
# ioreg：GPU 利用率与显存
# ---------------------------------------------------------------------------

#: ioreg 里的键名 → 我们对外用的字段名。
_GPU_KEYS = {
    "percent": "Device Utilization %",
    "renderer_percent": "Renderer Utilization %",
    "tiler_percent": "Tiler Utilization %",
    "in_use_mb": "In use system memory",
    "alloc_mb": "Alloc system memory",
}


def parse_ioreg_accelerator(text: str) -> Optional[Dict[str, Any]]:
    """从 `ioreg -c IOAccelerator` 的输出里抠出 GPU 统计。

    数据长这样，整个字典挤在一行里：

        "PerformanceStatistics" = {"In use system memory (driver)"=0,
                                   "Alloc system memory"=5635031040,
                                   "Device Utilization %"=14,
                                   "In use system memory"=730251264, ...}

    匹配时**引号必须紧跟在键名后面**：否则 `"In use system memory"` 会命中
    `"In use system memory (driver)"` —— 那个是驱动自己的记账，值常年是 0，
    取错了会显示成「显存占用 0」而没人看得出哪里不对。
    """
    if not text or not text.strip():
        return None

    stats: Dict[str, Any] = {}
    for field, key in _GPU_KEYS.items():
        match = re.search(rf'"{re.escape(key)}"=(-?\d+)', text)
        if match is None:
            continue
        value = int(match.group(1))
        if field.endswith("_mb"):
            stats[field] = _mb(value)
        else:
            stats[field] = value

    if not stats:
        return None

    cores = re.search(r'"gpu-core-count"\s*=\s*(\d+)', text)
    if cores:
        stats["cores"] = int(cores.group(1))
    stats["available"] = True
    return stats


# ---------------------------------------------------------------------------
# mach 调用：CPU tick 与本进程内存
# ---------------------------------------------------------------------------


class _HostCpuLoadInfo(ctypes.Structure):
    _fields_ = [("cpu_ticks", ctypes.c_uint32 * 4)]


class _TimeValue(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_int32), ("microseconds", ctypes.c_int32)]


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", _TimeValue),
        ("system_time", _TimeValue),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    ]


class MachReader:
    """所有 mach 调用集中在这里，加载失败就整体标成不可用。

    为什么不用 `ps` / `top`：

      * 起子进程是几十毫秒，而这两项都是每次采样都要读的；
      * `ps` 给的是「进程平均 %cpu」，不是瞬时值；要瞬时就只能调两次再自己算差，
        那不如直接用内核给的 tick 计数器。
    """

    HOST_CPU_LOAD_INFO = 3
    MACH_TASK_BASIC_INFO = 20
    CPU_STATE_MAX = 4
    STATE_USER, STATE_SYSTEM, STATE_IDLE, STATE_NICE = range(4)

    def __init__(self) -> None:
        self._lib = None
        if platform.system() != "Darwin":
            return
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("System") or "libSystem.B.dylib")
            lib.mach_host_self.restype = ctypes.c_uint32
            lib.mach_host_self.argtypes = []
            lib.host_statistics.restype = ctypes.c_int
            lib.host_statistics.argtypes = [
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.mach_task_self.restype = ctypes.c_uint32
            lib.mach_task_self.argtypes = []
            lib.task_info.restype = ctypes.c_int
            lib.task_info.argtypes = [
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
        except (OSError, AttributeError):
            return
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def cpu_ticks(self) -> Optional[Dict[str, int]]:
        """全机 CPU 累计 tick（user / system / idle / nice）。

        这是**累计值**，使用率要靠两次读数求差 —— 所以调用方得自己留着上一次。
        tick 的含义在 macOS 上实测是每核 100Hz（8 核约 800 tick/秒），
        但这里不依赖这个常数：全程按增量的比例算，换机型也不会错。
        """
        if self._lib is None:
            return None
        info = _HostCpuLoadInfo()
        count = ctypes.c_uint32(self.CPU_STATE_MAX)
        rc = self._lib.host_statistics(
            self._lib.mach_host_self(),
            self.HOST_CPU_LOAD_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if rc != 0:
            return None
        ticks = list(info.cpu_ticks)
        return {
            "user": ticks[self.STATE_USER],
            "system": ticks[self.STATE_SYSTEM],
            "idle": ticks[self.STATE_IDLE],
            "nice": ticks[self.STATE_NICE],
            "total": sum(ticks),
        }

    def task_memory(self) -> Optional[Dict[str, float]]:
        """本进程的常驻内存（RSS）。

        比 `ps -o rss=` 直接：不起子进程，而且拿到的是内核记的当前值。
        """
        if self._lib is None:
            return None
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint32(ctypes.sizeof(_MachTaskBasicInfo) // 4)
        rc = self._lib.task_info(
            self._lib.mach_task_self(),
            self.MACH_TASK_BASIC_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if rc != 0:
            return None
        return {
            "resident_mb": _mb(info.resident_size),
            "peak_mb": _mb(info.resident_size_max),
            "virtual_mb": _mb(info.virtual_size),
            "cpu_user_s": round(
                info.user_time.seconds + info.user_time.microseconds / 1e6, 2
            ),
            "cpu_system_s": round(
                info.system_time.seconds + info.system_time.microseconds / 1e6, 2
            ),
        }


# ---------------------------------------------------------------------------
# MLX 显存
# ---------------------------------------------------------------------------


def _mlx_memory() -> Dict[str, Any]:
    """MLX 自己记的显存占用。

    这是整个面板里**唯一**能回答「模型到底吃了多少」的数字：系统层面看到的只是
    进程 RSS，里面混着 Python 堆、tokenizer、各种缓存；MLX 记的是它实际持有的
    Metal 缓冲。

    Apple Silicon 是统一内存，没有独立显存 —— 这个数字和「内存」是同一块物理内存，
    只是归属不同，**不要把它和内存读数相加**。

    取不到就返回 available: false。老版本 MLX 把这三个函数放在 `mx.metal` 下，
    新版本提到了顶层，两边都试。

    **顺序很重要**：必须先试顶层。`mx.metal.get_active_memory` 现在只是个转发，
    调用它会往 stderr 打一条 deprecation 警告 —— 而这个函数每几秒就被调一次，
    日志会被刷满。
    """
    try:
        import mlx.core as mx  # noqa: PLC0415 —— 只在真的要看这个数时才导入
    except Exception:  # noqa: BLE001 —— 不是 Apple Silicon / 没装，都算「没有」
        return {"available": False}

    legacy = getattr(mx, "metal", None)

    def read(name: str) -> Optional[float]:
        func = getattr(mx, name, None) or getattr(legacy, name, None)
        if func is None:
            return None
        try:
            return _mb(float(func()))
        except Exception:  # noqa: BLE001
            return None

    active = read("get_active_memory")
    if active is None:
        return {"available": False}
    return {
        "available": True,
        "active_mb": active,
        "peak_mb": read("get_peak_memory"),
        "cache_mb": read("get_cache_memory"),
    }


# ---------------------------------------------------------------------------
# 采样器
# ---------------------------------------------------------------------------


def _safe(func, *args) -> Dict[str, Any]:
    """跑一个采集块；出任何问题都转成「这项没有」，而不是把整个快照带崩。"""
    try:
        return func(*args)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


class SystemMonitor:
    """按需采集本机资源，带 TTL 缓存。

    CPU 是**增量**指标，必须跨调用保存上一次读数，所以这个对象要活到进程结束
    （挂在 app.state 上），不能每次请求新建一个 —— 那样每次都会得到「首次采样，
    没有参照」。

    线程安全靠一把锁：`snapshot()` 在 FastAPI 的线程池里跑，可能被并发调用。
    锁把并发的采集也一并挡掉了，顺带省掉重复的 ioreg。
    """

    def __init__(self, ttl: float = DEFAULT_TTL, *, prime: bool = True) -> None:
        self._ttl = ttl
        self._lock = threading.Lock()
        self._cached: Optional[Dict[str, Any]] = None
        self._cached_at = 0.0
        self._mach = MachReader()
        self._prev_ticks: Optional[Dict[str, int]] = None
        self._prev_ticks_at = 0.0
        if prime:
            # 启动时先读一次 tick：控制台通常几秒后才第一次轮询，那时两次读数
            # 一相减就有真实的使用率，用户不会看到一格「—」。
            self._prev_ticks = self._mach.cpu_ticks()
            self._prev_ticks_at = time.monotonic()

    # ------------------------------------------------------------------ 对外
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._cached is not None and now - self._cached_at < self._ttl:
                return self._cached
            data = self._collect()
            self._cached, self._cached_at = data, now
            return data

    # ------------------------------------------------------------------ 采集
    def _collect(self) -> Dict[str, Any]:
        sysctl = _sysctl()
        return {
            "sampled_at": time.time(),
            "host": _host_block(sysctl),
            "cpu": _safe(self._cpu_block),
            "memory": _safe(_memory_block, sysctl),
            "process": _safe(self._process_block),
            "gpu": _safe(_gpu_block),
            "mlx": _safe(_mlx_memory),
            "notes": [
                "Apple Silicon 是统一内存：这里没有独立的「显存」，"
                "MLX 显存指的是它持有的那部分内存，不要和内存读数相加。",
                "内存口径：已用 = 应用内存（匿名页 - 可回收页）+ 有线内存 + 压缩内存，"
                "与活动监视器的「已用内存」接近；不是简单的「总量 - 空闲」。",
            ],
        }

    def _cpu_block(self) -> Dict[str, Any]:
        block: Dict[str, Any] = {"available": True, "cores": os.cpu_count()}
        try:
            load = os.getloadavg()
            block["load"] = [round(value, 2) for value in load]
        except (OSError, AttributeError):  # pragma: no cover —— 非 POSIX
            block["load"] = None

        sample = self._mach.cpu_ticks()
        if sample is None:
            block["available"] = False
            block["reason"] = (
                "只有 macOS 支持（走 mach host_statistics）"
                if platform.system() != "Darwin"
                else "mach 调用失败"
            )
            return block

        now = time.monotonic()
        previous, previous_at = self._prev_ticks, self._prev_ticks_at
        span = (now - previous_at) if previous is not None else 0.0
        if previous is None or span < _MIN_WINDOW_S or span > _MAX_WINDOW_S:
            # 三种情况都要临时采一个短窗口：
            #   * 没有参照（进程刚起来）
            #   * 窗口太短 —— 每个 tick 就是一个百分点，几毫秒的窗口噪声太大，
            #     屏幕上会看到使用率在 40% 和 80% 之间乱跳
            #   * 窗口太长（控制台几小时没打开）—— 那是平均值，不是「当前」
            time.sleep(_MIN_WINDOW_S)
            previous, previous_at = sample, now
            sample = self._mach.cpu_ticks() or sample
            now = time.monotonic()

        # 窗口够宽也可能撞上内核还没刷新的那一拍。这时**不要**推进 previous，
        # 只是再读一次等它自己往前走 —— 否则这次采样会白丢。
        attempts = 0
        while sample["total"] <= previous["total"] and attempts < _ZERO_DELTA_RETRIES:
            time.sleep(_ZERO_DELTA_WAIT_S)
            sample = self._mach.cpu_ticks() or sample
            now = time.monotonic()
            attempts += 1

        delta_total = sample["total"] - previous["total"]
        if delta_total <= 0:
            # 到这里说明计数器在这段时间里确实一直没动。如实说，不编数字。
            block["available"] = False
            block["reason"] = "内核还没刷新 CPU 计数器，这一次取不到增量"
            return block

        self._prev_ticks, self._prev_ticks_at = sample, now
        delta_idle = sample["idle"] - previous["idle"]
        block.update(
            {
                "percent": round(100.0 * (delta_total - delta_idle) / delta_total, 1),
                "user_percent": round(
                    100.0 * (sample["user"] - previous["user"]) / delta_total, 1
                ),
                "system_percent": round(
                    100.0 * (sample["system"] - previous["system"]) / delta_total, 1
                ),
                "idle_percent": round(100.0 * delta_idle / delta_total, 1),
                "window_s": round(now - previous_at, 2),
            }
        )
        return block

    def _process_block(self) -> Dict[str, Any]:
        memory = self._mach.task_memory()
        if memory is None:
            return {
                "available": False,
                "reason": "只有 macOS 支持（走 mach task_info）",
                "pid": os.getpid(),
            }
        return {"available": True, "pid": os.getpid(), **memory}


def _host_block(sysctl: Dict[str, str]) -> Dict[str, Any]:
    def as_int(key: str) -> Optional[int]:
        try:
            return int(sysctl[key])
        except (KeyError, ValueError):
            return None

    total = as_int("hw.memsize")
    macos = platform.mac_ver()[0]
    block: Dict[str, Any] = {
        "chip": sysctl.get("machdep.cpu.brand_string") or None,
        "model": sysctl.get("hw.model") or None,
        "cpu_cores": os.cpu_count(),
        "os": f"macOS {macos}" if macos else f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    if total:
        block["total_mb"] = _mb(total)
    return block


def _memory_block(sysctl: Dict[str, str]) -> Dict[str, Any]:
    try:
        total = int(sysctl["hw.memsize"])
        page_size = int(sysctl["hw.pagesize"])
    except (KeyError, ValueError):
        return {"available": False, "reason": "读不到 hw.memsize / hw.pagesize"}

    text = _run(["vm_stat"])
    if text is None:
        return {"available": False, "reason": "vm_stat 执行失败"}
    pages = parse_vm_stat(text)
    if not pages:
        return {"available": False, "reason": "vm_stat 输出无法解析"}

    def bytes_of(name: str) -> int:
        return pages.get(name, 0) * page_size

    # 「应用内存」用**匿名页 - 可回收页**，而不是 vm_stat 里的 pages_active。
    # 后者把文件缓存（读过的文件映射）也算进去，数字明显偏大，而且随文件读写
    # 乱跳 —— 那样展示出来的是「内核用掉多少页框」，不是「应用占了多少内存」。
    # 这个口径与活动监视器的「已用内存」接近。
    anonymous = bytes_of("anonymous_pages")
    purgeable = bytes_of("pages_purgeable")
    app = max(0, anonymous - purgeable)
    wired = bytes_of("pages_wired_down")
    compressed = bytes_of("pages_occupied_by_compressor")
    used = app + wired + compressed

    return {
        "available": True,
        "total_mb": _mb(total),
        "used_mb": _mb(used),
        "available_mb": _mb(max(0, total - used)),
        "percent": round(100.0 * used / total, 1),
        "breakdown_mb": {
            "app": _mb(app),
            "wired": _mb(wired),
            "compressed": _mb(compressed),
            "cached": _mb(bytes_of("file_backed_pages")),
            "inactive": _mb(bytes_of("pages_inactive")),
            "free": _mb(bytes_of("pages_free")),
        },
        "swap": parse_swapusage(sysctl.get("vm.swapusage", "")),
    }


def _gpu_block() -> Dict[str, Any]:
    text = _run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"])
    if text is None:
        return {"available": False, "reason": "ioreg 执行失败（非 macOS？）"}
    stats = parse_ioreg_accelerator(text)
    if stats is None:
        return {"available": False, "reason": "没有找到 Metal 设备的统计信息"}
    return stats


__all__ = [
    "DEFAULT_TTL",
    "MachReader",
    "SystemMonitor",
    "parse_ioreg_accelerator",
    "parse_swapusage",
    "parse_vm_stat",
]
