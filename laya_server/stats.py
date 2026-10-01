"""请求统计：环形缓冲 + 聚合视图，给控制台用。

设计取舍：
  * **内存态，不落盘。** 这是「刚才那几分钟发生了什么」的工具，不是审计日志。
    落盘会带来轮转、权限、清理三件麻烦事，而排查「为什么突然变慢」只需要最近几百条。
  * **有界。** 记录条数写死上限（环形缓冲），长时间挂机不会吃内存。
  * **聚合在读取时算。** 分位数、分桶都从记录里现算。写路径只做 append，
    这样中间件加进去的开销是个位数微秒，不会成为推理路径上的噪音。
  * **线程安全用一把锁。** 写来自线程池里的推理线程和事件循环两个地方，
    但每次只 append 一条，锁竞争可以忽略。

注意这里统计的是**推理请求**（`/v1/systemone`）——被 schema 挡下来的 422 也算，
因为「今天有多少请求是被参数写错挡住的」本身就是个要看的问题。
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

#: 保留的明细条数。够翻「刚才那几十条为什么慢」，又不至于吃内存。
DEFAULT_CAPACITY = 1000
#: 时间序列分桶粒度与桶数（最近 30 个 1 分钟桶）。
BUCKET_SECONDS = 60
BUCKET_COUNT = 30


@dataclass
class Trace:
    """一条请求的可变记录。中间件建它，端点往里填，最后统一记账。"""

    request_id: str
    started_at: float
    model_alias: Optional[str] = None
    model_reported: Optional[str] = None
    slot: Optional[str] = None
    question_count: int = 0
    answer_types: Dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    inference_ms: Optional[float] = None
    status: Optional[int] = None
    error_code: Optional[str] = None

    def as_dict(self, latency_ms: float) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "at": self.started_at,
            "model_alias": self.model_alias,
            "model": self.model_reported,
            "slot": self.slot,
            "question_count": self.question_count,
            "answer_types": self.answer_types,
            "input_tokens": self.input_tokens,
            "status": self.status,
            "error_code": self.error_code,
            "latency_ms": round(latency_ms, 2),
            "inference_ms": self.inference_ms,
        }


def _percentile(values: List[float], fraction: float) -> Optional[float]:
    """线性插值分位数。样本太少时返回 None，而不是编一个看着像真的数字。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 2)


def _latency_block(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p90": None, "p99": None, "mean": None, "max": None}
    return {
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "mean": round(statistics.fmean(values), 2),
        "max": round(max(values), 2),
    }


class StatsCollector:
    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._records: Deque[Dict[str, Any]] = deque(maxlen=max(1, capacity))
        self._started_at = time.time()
        self._started_monotonic = time.monotonic()
        self._dropped = 0

    # ------------------------------------------------------------------ 写入
    def record(self, trace: Trace, latency_ms: float) -> None:
        entry = trace.as_dict(latency_ms)
        with self._lock:
            self._records.append(entry)

    def new_trace(self, request_id: str, model_alias: Optional[str], question_count: int) -> Trace:
        return Trace(
            request_id=request_id,
            started_at=time.time(),
            model_alias=model_alias,
            question_count=question_count,
        )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._started_at = time.time()
            self._started_monotonic = time.monotonic()

    # ------------------------------------------------------------------ 读取
    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._records)
        # 新的在前。
        return list(reversed(items))[: max(0, limit)]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records)
            started_at = self._started_at
            uptime = time.monotonic() - self._started_monotonic

        total = len(records)
        by_status: Counter = Counter()
        by_error: Counter = Counter()
        by_model: Counter = Counter()
        by_type: Counter = Counter()
        latencies: List[float] = []
        inference: List[float] = []
        tokens = 0
        for entry in records:
            by_status[str(entry["status"])] += 1
            if entry.get("error_code"):
                by_error[entry["error_code"]] += 1
            if entry.get("model"):
                by_model[entry["model"]] += 1
            for qtype, count in (entry.get("answer_types") or {}).items():
                by_type[qtype] += count
            latencies.append(float(entry["latency_ms"]))
            if entry.get("inference_ms") is not None:
                inference.append(float(entry["inference_ms"]))
            tokens += int(entry.get("input_tokens") or 0)

        succeeded = by_status.get("200", 0) + by_status.get("201", 0)
        return {
            "uptime_s": round(uptime, 1),
            "started_at": started_at,
            "capacity": self._records.maxlen,
            "total": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "success_rate": round(succeeded / total, 4) if total else None,
            "by_status": dict(sorted(by_status.items())),
            "by_error_code": dict(by_error),
            "by_model": dict(by_model),
            "answer_types": dict(by_type),
            "input_tokens": tokens,
            "latency_ms": _latency_block(latencies),
            "inference_ms": _latency_block(inference),
            "throughput": self._throughput(records),
            "series": self._series(records),
        }

    # ------------------------------------------------------------------ 派生
    @staticmethod
    def _throughput(records: List[Dict[str, Any]], window: float = 60.0) -> Dict[str, Any]:
        now = time.time()
        recent = [entry for entry in records if now - entry["at"] <= window]
        return {
            "window_s": window,
            "count": len(recent),
            "per_minute": round(len(recent) * 60.0 / window, 2),
        }

    @staticmethod
    def _series(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最近 N 个分钟桶。空桶补 0，否则图上会「断格」，看不出到底是没流量还是没画。"""
        now = time.time()
        buckets: Dict[int, Dict[str, Any]] = {}
        for offset in range(BUCKET_COUNT - 1, -1, -1):
            start = int((now - offset * BUCKET_SECONDS) // BUCKET_SECONDS) * BUCKET_SECONDS
            buckets[start] = {"at": start, "count": 0, "errors": 0, "p50": None, "_lat": []}

        for entry in records:
            start = int(entry["at"] // BUCKET_SECONDS) * BUCKET_SECONDS
            bucket = buckets.get(start)
            if bucket is None:
                continue
            bucket["count"] += 1
            if entry["status"] != 200:
                bucket["errors"] += 1
            bucket["_lat"].append(float(entry["latency_ms"]))

        series = []
        for start in sorted(buckets):
            bucket = buckets[start]
            latencies = bucket.pop("_lat")
            bucket["p50"] = _percentile(latencies, 0.50)
            series.append(bucket)
        return series


__all__ = ["BUCKET_COUNT", "BUCKET_SECONDS", "DEFAULT_CAPACITY", "StatsCollector", "Trace"]
