"""资源采集的测试。

解析部分喂**固定文本**，不依赖任何系统状态：ioreg 和 vm_stat 的输出格式跟着
macOS 版本走，把真实片段钉在这里，格式一变测试就红 —— 这正是想要的。

采集部分只在 macOS 上跑，而且只断言「形状」和「不许抛异常」：CPU 用多少取决于
这台机器当时在干嘛，写成期望值只会在别人的机器上变红。
"""

from __future__ import annotations

import platform

import pytest

from laya_server.sysinfo import (
    MachReader,
    SystemMonitor,
    parse_ioreg_accelerator,
    parse_swapusage,
    parse_vm_stat,
)

DARWIN = platform.system() == "Darwin"
darwin_only = pytest.mark.skipif(not DARWIN, reason="这些采集路径只在 macOS 上有")

#: 真实输出的节选。注意 `"In use system memory (driver)"=0` 排在真正的
#: `"In use system memory"=730251264` **前面** —— 这是最容易取错值的地方，
#: 取错了会显示成「显存占用 0」而且没人看得出哪里不对。
IOREG_SAMPLE = """+-o AGXAcceleratorG13G_B0  <class AGXAcceleratorG13G_B0, id 0x100000abc>
    {
      "IOClass" = "AGXAcceleratorG13G_B0"
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=5635031040,"Tiler Utilization %"=14,"Renderer Utilization %"=14,"Device Utilization %"=14,"In use system memory"=730251264}
      "model" = "Apple M1"
      "gpu-core-count" = 8
      "IONameMatch" = ("gpu,t8103")
    }
"""

VM_STAT_SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     5414.
Pages active:                                 252779.
Pages inactive:                               251802.
Pages speculative:                               490.
Pages throttled:                                   0.
Pages wired down:                             148910.
Pages purgeable:                                2438.
"Translation faults":                     2272267274.
Pages copy-on-write:                        71697090.
File-backed pages:                            169233.
Anonymous pages:                              289342.
Pages stored in compressor:                  1216557.
Pages occupied by compressor:                 396546.
"""


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def test_parse_ioreg_takes_the_real_memory_key_not_the_driver_one():
    stats = parse_ioreg_accelerator(IOREG_SAMPLE)
    assert stats is not None
    assert stats["available"] is True
    assert stats["percent"] == 14
    assert stats["renderer_percent"] == 14
    assert stats["tiler_percent"] == 14
    assert stats["cores"] == 8
    # 关键断言：不能是 driver 那个恒为 0 的键。
    assert stats["in_use_mb"] == round(730251264 / 1024 / 1024, 1)
    assert stats["in_use_mb"] > 0
    assert stats["alloc_mb"] == round(5635031040 / 1024 / 1024, 1)


def test_parse_ioreg_returns_none_when_there_is_nothing_to_read():
    assert parse_ioreg_accelerator("") is None
    assert parse_ioreg_accelerator("   \n") is None
    assert parse_ioreg_accelerator("这是个完全无关的输出") is None


def test_parse_ioreg_survives_a_partial_record():
    """只有部分键时也要能给出部分数据，而不是整块丢掉。"""
    stats = parse_ioreg_accelerator('"PerformanceStatistics" = {"Device Utilization %"=7}')
    assert stats is not None
    assert stats["percent"] == 7
    assert "in_use_mb" not in stats
    assert "cores" not in stats


def test_parse_vm_stat_normalises_key_names():
    pages = parse_vm_stat(VM_STAT_SAMPLE)
    assert pages["pages_free"] == 5414
    assert pages["pages_wired_down"] == 148910
    assert pages["pages_occupied_by_compressor"] == 396546
    # 带引号的键也要认
    assert pages["translation_faults"] == 2272267274
    # 连字符要变成下划线
    assert pages["file_backed_pages"] == 169233


def test_parse_vm_stat_ignores_the_header_line():
    pages = parse_vm_stat(VM_STAT_SAMPLE)
    assert "mach_virtual_memory_statistics" not in pages
    assert all(isinstance(value, int) for value in pages.values())


def test_parse_swapusage():
    swap = parse_swapusage("total = 6144.00M  used = 4878.56M  free = 1265.44M  (encrypted)")
    # 面板上不需要 0.01MB 的精度，统一留一位小数。
    assert swap == {"total_mb": 6144.0, "used_mb": 4878.6, "free_mb": 1265.4}
    assert parse_swapusage("没有 swap 信息") == {}


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------


def test_mach_reader_is_unavailable_off_macos(monkeypatch):
    monkeypatch.setattr("laya_server.sysinfo.platform.system", lambda: "Linux")
    reader = MachReader()
    assert reader.available is False
    # 不可用时返回 None 而不是抛异常 —— 调用方按「这项没有」处理。
    assert reader.cpu_ticks() is None
    assert reader.task_memory() is None


def test_a_failing_block_does_not_take_down_the_snapshot(monkeypatch):
    """一个采集项炸了，其余项照常出数。

    监控面板最忌讳的就是「因为读不到 GPU，整页数据全没了」。
    """

    def boom():
        raise RuntimeError("ioreg 炸了")

    monkeypatch.setattr("laya_server.sysinfo._gpu_block", boom)
    snapshot = SystemMonitor(ttl=0).snapshot()

    assert snapshot["gpu"]["available"] is False
    assert "ioreg 炸了" in snapshot["gpu"]["reason"]
    assert snapshot["cpu"]["available"] is True
    assert set(snapshot) == {
        "sampled_at", "host", "cpu", "memory", "process", "gpu", "mlx", "notes",
    }


def test_snapshot_is_reused_within_the_ttl():
    """TTL 之内直接复用同一个对象，不再真采一次。

    没有这层缓存的话，每 3 秒一次的轮询就要起三四个子进程 ——
    一个「看资源占用」的面板自己把 CPU 吃掉，说不过去。
    """
    monitor = SystemMonitor(ttl=60)
    first = monitor.snapshot()
    assert monitor.snapshot() is first


def test_snapshot_recollects_after_the_ttl():
    monitor = SystemMonitor(ttl=0)
    assert monitor.snapshot() is not monitor.snapshot()


@darwin_only
def test_first_cpu_reading_already_has_a_number():
    """首次调用就该给出使用率，不能显示一格「—」。

    构造时预读了一次 tick，所以「没有参照」这个状态在用户看到之前就已经过去了。
    """
    monitor = SystemMonitor(ttl=0)
    cpu = monitor.snapshot()["cpu"]
    assert cpu["available"] is True
    assert cpu["percent"] is not None


@darwin_only
def test_cpu_window_is_never_too_short_to_be_meaningful():
    """采样窗口不能太短。

    内核刷新 CPU 计数器有粒度：实测 0.25 秒的窗口里五次就有一次增量是 0，
    而 0.5 秒的窗口连续 20 次都没出过。实现里按 0.5 秒兜底，这条钉住它。
    """
    monitor = SystemMonitor(ttl=0)
    monitor.snapshot()
    assert monitor.snapshot()["cpu"]["window_s"] >= 0.5


@darwin_only
def test_cpu_reading_survives_a_frozen_counter():
    """连采 12 次都不该出现「取不到增量」——窗口够宽就撞不上那个粒度。"""
    monitor = SystemMonitor(ttl=0)
    for _ in range(12):
        cpu = monitor.snapshot()["cpu"]
        assert cpu["available"] is True, cpu.get("reason")
        assert 0.0 <= cpu["percent"] <= 100.0


@darwin_only
def test_snapshot_shape_on_this_machine():
    snapshot = SystemMonitor().snapshot()

    host = snapshot["host"]
    assert host["cpu_cores"] and host["cpu_cores"] > 0
    assert host["total_mb"] > 0

    cpu = snapshot["cpu"]
    assert cpu["available"] is True
    assert 0.0 <= cpu["percent"] <= 100.0

    memory = snapshot["memory"]
    assert memory["available"] is True
    assert memory["total_mb"] > 0
    assert 0 < memory["percent"] <= 100
    assert memory["used_mb"] <= memory["total_mb"]
    assert memory["available_mb"] >= 0

    process = snapshot["process"]
    assert process["available"] is True
    assert process["resident_mb"] > 0, "跑着测试的进程不可能不占内存"

    gpu = snapshot["gpu"]
    if gpu["available"]:
        # 「可用」只保证至少抠到了一个字段，不保证是全套 —— 解析器有意支持部分
        # 记录（见 test_parse_ioreg_survives_a_partial_record）。虚拟机上的半虚拟化
        # GPU 就只给显存、不给利用率，所以逐字段判断，别整块假设全套都在。
        if "percent" in gpu:
            assert 0 <= gpu["percent"] <= 100
        if "in_use_mb" in gpu:
            assert gpu["in_use_mb"] >= 0
        if "cores" in gpu:
            assert gpu["cores"] >= 1

    assert snapshot["notes"], "口径说明要给出来，否则没人知道这些数字怎么算的"
