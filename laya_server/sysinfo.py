"""本机与服务进程的资源占用：CPU / 内存 / GPU / MLX 显存。

只做 macOS。这是刻意的：这个服务的推理后端是 MLX，而 MLX 只在 Apple Silicon
上有意义；与其写一堆跨平台分支再声称它们可用，不如把「读不到」明确说出来。

**不引入 psutil**，理由有两条：

1. GPU 利用率 psutil 根本拿不到（它没有 Metal 的概念），一定要自己读
   IOAccelerator。既然 GPU 必须自己解析，再为 CPU/内存多引一个 C 扩展依赖
   （安装、架构、wheel 三件麻烦事）就不划算了。
2. 系统自带接口已经够用，而且更快：整机 CPU 走 mach 调用是微秒级，进程取数走
   libproc —— `ps` 和活动监视器自己用的就是它，我们只是直接调库，不起子进程。

五类数据源，各自的特点写在对应函数上：

    mach host_statistics    整机 CPU 累计 tick → 求增量得使用率。微秒级
    libproc                 本进程累计 CPU 时间 / RSS / phys_footprint，ps 同源
    sysctl / vm_stat        整机内存与 swap
    ioreg                   GPU 利用率与显存（整机）。约 90ms，所以带 TTL 缓存
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
# mach 调用：整机 CPU tick
# ---------------------------------------------------------------------------


class _HostCpuLoadInfo(ctypes.Structure):
    _fields_ = [("cpu_ticks", ctypes.c_uint32 * 4)]


class MachReader:
    """整机 CPU 的 mach 调用集中在这里，加载失败就整体标成不可用。"""

    HOST_CPU_LOAD_INFO = 3
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

# ---------------------------------------------------------------------------
# libproc：本进程 CPU 与内存
# ---------------------------------------------------------------------------
#
# 进程取数走 libproc（`ps` / 活动监视器同源），**不要**走 mach 的
# task_info(MACH_TASK_BASIC_INFO)：在 macOS 26 上实测，它的 CPU 时间字段在
# 进程正用着 CPU 的时候会返回 0 —— 烧一个核，连续 6 次读取全是 0.00s（同一
# 时刻 `ps -o time=` 一路正常增长到 0:07.60），停火后同一次调用才把累计的
# 7.3 秒补出来。拿它做实时采样是死的，换成 libproc 后逐秒读数完全正常。
#
# 两个调用各管一头：
#   proc_pidinfo(PROC_PIDTASKINFO)  累计 CPU 时间（mach 时钟 tick）与 RSS
#   proc_pid_rusage(RUSAGE_INFO_V6) phys_footprint —— 活动监视器口径。
#       MLX 的 Metal 缓冲只有这个口径记得到：实测分配 256MB，RSS 只涨 2.6MB，
#       phys_footprint 涨 258MB。拿 RSS 当「服务内存」会在加载模型后严重低报。


class _MachTimebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class _ProcTaskInfo(ctypes.Structure):
    """sys/proc_info.h 的 proc_taskinfo，96 字节。字段顺序照抄头文件。"""

    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


#: rusage_info_v6 里 uuid 之后的字段名，顺序照抄 sys/resource.h。声明到
#: ri_lifetime_max_phys_footprint 为止就够用，剩下的用占位数组顶住 —— 总大小
#: 对了，内核按固定布局填，关心的字段偏移就不会错。
_RUSAGE_V6_FIELDS = (
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
    "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility",
    "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated",
    "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time",
    "ri_serviced_system_time",
    "ri_logical_writes",
    "ri_lifetime_max_phys_footprint",
)


class _RusageInfoV6(ctypes.Structure):
    _fields_ = (
        [("ri_uuid", ctypes.c_uint8 * 16)]
        + [(name, ctypes.c_uint64) for name in _RUSAGE_V6_FIELDS]
        + [("_reserved", ctypes.c_uint64 * 27)]
    )


class LibprocReader:
    """本进程的 CPU 与内存。加载失败就整体标成不可用。

    为什么不用 `ps` / `top` 命令：起子进程是几十到几百毫秒，而这两项每次采样
    都要读；而且命令输出是给人看的，字段会随系统版本变。libproc 是同一份数据
    的内核接口，直接用。
    """

    PROC_PIDTASKINFO = 4
    RUSAGE_INFO_V6 = 6

    def __init__(self) -> None:
        self._lib = None
        self._timebase = None
        if platform.system() != "Darwin":
            return
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib")
            lib.proc_pidinfo.restype = ctypes.c_int
            lib.proc_pidinfo.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            lib.proc_pid_rusage.restype = ctypes.c_int
            lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            system = ctypes.CDLL(
                ctypes.util.find_library("System") or "libSystem.B.dylib"
            )
            system.mach_timebase_info.restype = ctypes.c_int
            system.mach_timebase_info.argtypes = [ctypes.POINTER(_MachTimebase)]
        except (OSError, AttributeError):
            return
        # pti 的时间是 mach 时钟 tick，要换算成秒。取不到 timebase 也照样
        # 返回内存，只是 CPU 那几项为 None —— 不猜一个系数出来用。
        tb = _MachTimebase()
        if system.mach_timebase_info(ctypes.byref(tb)) == 0 and tb.denom:
            self._timebase = (tb.numer, tb.denom)
        self._lib = lib

    @property
    def available(self) -> bool:
        return self._lib is not None

    def proc_stats(self) -> Optional[Dict[str, Any]]:
        """本进程的累计 CPU 时间与内存。

        `_cpu_total_s` 是给增量算 CPU% 用的完整精度值，调用方取走后再拼 payload。
        """
        if self._lib is None:
            return None
        pid = os.getpid()

        info = _ProcTaskInfo()
        if self._lib.proc_pidinfo(
            pid, self.PROC_PIDTASKINFO, 0, ctypes.byref(info), ctypes.sizeof(info)
        ) != ctypes.sizeof(info):
            return None
        usage = _RusageInfoV6()
        if self._lib.proc_pid_rusage(pid, self.RUSAGE_INFO_V6, ctypes.byref(usage)) != 0:
            return None

        stats: Dict[str, Any] = {
            "resident_mb": _mb(info.pti_resident_size),
            "footprint_mb": _mb(usage.ri_phys_footprint),
            "peak_footprint_mb": _mb(usage.ri_lifetime_max_phys_footprint),
        }
        if self._timebase is None:
            stats.update(cpu_user_s=None, cpu_system_s=None, _cpu_total_s=None)
            return stats

        numer, denom = self._timebase

        def to_seconds(ticks: int) -> float:
            return ticks * numer / denom / 1e9

        user = to_seconds(info.pti_total_user)
        system = to_seconds(info.pti_total_system)
        stats.update(
            cpu_user_s=round(user, 2),
            cpu_system_s=round(system, 2),
            _cpu_total_s=round(user + system, 3),
        )
        return stats


# ---------------------------------------------------------------------------
# MLX 显存
# ---------------------------------------------------------------------------


def _mlx_memory() -> Dict[str, Any]:
    """MLX 自己记的显存占用。

    这是整个面板里**唯一**能回答「模型到底吃了多少」的数字：系统层面只能看到
    进程总账（phys_footprint），里面混着 Python 堆、tokenizer、各种缓存；
    MLX 记的是它实际持有的 Metal 缓冲。

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
        self._libproc = LibprocReader()
        self._prev_ticks: Optional[Dict[str, int]] = None
        self._prev_ticks_at = 0.0
        self._prev_proc_cpu_s: Optional[float] = None
        self._prev_proc_cpu_at = 0.0
        if prime:
            # 启动时先把两个参照都读一遍：控制台通常几秒后才第一次轮询，那时
            # 两次读数一相减就有真实的使用率，用户不会看到一格「—」。
            self._prev_ticks = self._mach.cpu_ticks()
            self._prev_ticks_at = time.monotonic()
            started = self._libproc.proc_stats() or {}
            self._prev_proc_cpu_s = started.get("_cpu_total_s")
            self._prev_proc_cpu_at = time.monotonic()

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
                "服务 CPU / 内存取的是 laya 进程自己的占用：CPU 是进程累计时间的"
                "增量（多线程会超过 100%），内存是 phys_footprint（活动监视器同口径），"
                "MLX 持有的那部分内存也在里面 —— 不要和「MLX 显存」相加，"
                "它们是同一块统一内存的总账与模型账。",
                "GPU 利用率是整机的：系统没有提供按进程的 GPU 占用接口；"
                "推理跑起来时这个数基本就是这个服务打上去的。",
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
        stats = self._libproc.proc_stats()
        if stats is None:
            return {
                "available": False,
                "reason": "只有 macOS 支持（走 libproc）",
                "pid": os.getpid(),
            }

        read_at = time.monotonic()
        total = stats.pop("_cpu_total_s", None)
        previous, previous_at = self._prev_proc_cpu_s, self._prev_proc_cpu_at
        # 和整机 CPU 同样的窗口规则：参照太近（进程刚起来）或太远（面板几小时
        # 没打开）就临时起一个 0.5 秒的窗口再读一次 —— 宁可晚半秒，也不给
        # 用户一格看不出是「0%」还是「没数据」的「—」。
        if total is not None and (
            previous is None or not _MIN_WINDOW_S <= read_at - previous_at <= _MAX_WINDOW_S
        ):
            time.sleep(_MIN_WINDOW_S)
            again = self._libproc.proc_stats()
            if again is not None:
                previous, previous_at = total, read_at
                stats = again
                total = again.pop("_cpu_total_s", None)
                read_at = time.monotonic()

        self._prev_proc_cpu_s, self._prev_proc_cpu_at = total, read_at
        span = read_at - previous_at

        block: Dict[str, Any] = {"available": True, "pid": os.getpid(), **stats}
        # 进程可以多线程，超出 100% 是正常读数（top 同款口径），不做截断。
        block["cpu_percent"] = (
            round(100.0 * (total - previous) / span, 1)
            if total is not None
            and previous is not None
            and _MIN_WINDOW_S <= span <= _MAX_WINDOW_S
            else None
        )
        block["window_s"] = round(span, 2) if previous is not None else None
        return block


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
    "LibprocReader",
    "MachReader",
    "SystemMonitor",
    "parse_ioreg_accelerator",
    "parse_swapusage",
    "parse_vm_stat",
]
