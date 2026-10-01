"""管理接口：给控制台用的 `/admin/*`。

这些端点能干的事情很重（改 API key、卸载模型、停机），所以默认**只接受本机来源**。
`--host 0.0.0.0` 的部署方式下，管理面不会跟着一起暴露到局域网上 —— 这是一个刻意
的失败方向：远程改不了配置，比远程能改配置要好得多。

设了 `api_key` 的话还要额外过一遍 Bearer 校验。两层是叠加的，不是二选一。
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import JSONResponse

from .backend import Backend, BackendError, RequestError, UnknownModelError
from .config import (
    AUTO_LOAD_MODES,
    COLD_FIELDS,
    HOME_DIR,
    HOT_FIELDS,
    Settings,
    apply_updates,
    config_path,
    save_settings,
    settings_diff,
)
from .stats import StatsCollector
from .sysinfo import SystemMonitor

SERVICE_NAME = "laya-server"
SERVICE_VERSION = "0.1.0"

#: 字段的中文标签。没列到的按字段名显示——与其编一个不准确的译名，不如让原始名字露出来。
FIELD_LABELS: Dict[str, str] = {
    "host": "监听地址",
    "port": "监听端口",
    "api_key": "API Key",
    "backend": "推理后端",
    "default_model": "默认模型",
    "dtype": "权重精度",
    "device": "MLX 设备",
    "batch_size": "前向批量",
    "max_concurrency": "并发推理数",
    "queue_timeout_s": "排队超时（秒）",
    "max_loaded": "常驻模型数上限",
    "max_state_chars": "state 字符上限",
    "max_questions": "单请求问题上限",
    "auto_load": "启动时加载权重",
    "debug": "调试模式",
    "docs": "导出 OpenAPI 文档",
    "compile": "MLX 编译优化",
    "pad_to_multiple": "padding 对齐",
    "cache_prompts": "前缀缓存",
    "allow_raw_checkpoint": "允许直接给 checkpoint id",
    "console": "托管控制台",
    "admin_local_only": "管理接口仅限本机",
    "stats_capacity": "请求明细保留条数",
    "hf_endpoint": "权重下载端点",
    "hf_offline": "只用本地缓存",
}

#: 需要成对解读的字段，界面上给点提示比让人猜强。
FIELD_HINTS: Dict[str, str] = {
    "hf_endpoint": "国内直连 huggingface.co 通常不通，填 https://hf-mirror.com 即可。",
    "hf_offline": "权重已经下过 / 完全离线时打开，省掉每次启动的联网检查。",
    "max_concurrency": "MLX 是单设备，>1 只在 IO 等待上有意义；调高更容易 OOM。",
    "max_loaded": "常驻权重数上限（LRU）。三个槽位全驻留约 2.3GB。",
    "dtype": "float32 数值更准，更慢更占内存。",
    "api_key": "留空表示不校验 Authorization。",
    "auto_load": "local = 只加载本地已有的权重（不联网）；all = 本地没有就去下；off = 全部等首个请求。",
    "max_state_chars": "0 表示不限制；checkpoint 自身的 context 才是硬约束。",
    "debug": "响应里附带路由、延迟等 Jev 没有的字段。",
    "queue_timeout_s": "等推理槽位超过这个时间返回 429。",
}

#: 取值只有固定几档的字段。有这一条，控制台就会渲染成下拉框，
#: 而不是让人对着一个空文本框猜该填什么拼写。
FIELD_CHOICES: Dict[str, List[str]] = {
    "auto_load": list(AUTO_LOAD_MODES),
}

_MASK = "••••••••"


def _mask(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    tail = value[-4:] if len(value) > 8 else ""
    return _MASK + tail


def _kind(name: str, default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, dict):
        return "json"
    if default is None:
        # Optional[str] / Optional[int]：按字段名给个更准的默认输入类型。
        return "int" if name in {"pad_to_multiple", "max_state_chars"} else "text"
    return "text"


def _field_meta() -> Dict[str, Dict[str, Any]]:
    base = Settings()
    meta: Dict[str, Dict[str, Any]] = {}
    for name in sorted(Settings.__dataclass_fields__):
        default = getattr(base, name)
        choices = FIELD_CHOICES.get(name)
        meta[name] = {
            "label": FIELD_LABELS.get(name, name),
            "hint": FIELD_HINTS.get(name, ""),
            "kind": "choice" if choices else _kind(name, default),
            "choices": choices,
            "default": default,
            "mutable": name in HOT_FIELDS,
        }
    return meta


def _serialise(settings: Settings, reveal: bool) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for name in Settings.__dataclass_fields__:
        value = getattr(settings, name)
        if name == "api_key":
            value = value if reveal else _mask(value)
        values[name] = value
    return values


def build_admin_router(
    settings: Settings,
    backend: Backend,
    stats: StatsCollector,
    *,
    is_local: Callable[[Request], bool],
    baseline: Optional[Settings] = None,
    monitor: Optional[SystemMonitor] = None,
    on_settings_changed: Optional[Callable[[Settings, Settings], List[str]]] = None,
    on_shutdown: Optional[Callable[[], None]] = None,
) -> APIRouter:
    """组装管理路由。

    `settings` 是**可变引用**：热改字段直接就地更新，端点每次读到的都是新值，
    不需要为了让配置生效去重启进程。冷改字段写进配置文件，靠 `restart_required` 提示。

    `baseline` 是本次启动的生效值。存盘时以它为基准，这样命令行/环境变量带来的值
    不会被顺手写进配置文件（详见 `save_settings` 的注释）。
    """
    router = APIRouter(prefix="/admin", tags=["admin"])
    field_meta = _field_meta()

    def guard(request: Request) -> Optional[JSONResponse]:
        if settings.admin_local_only and not is_local(request):
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": "admin_local_only",
                        "message": "管理接口只接受本机请求。要在别处访问，"
                        "请改成把 admin_local_only 设成 false（不推荐）。",
                    }
                },
            )
        expected = settings.api_key
        if expected:
            header = request.headers.get("Authorization", "")
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or token.strip() != expected:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "unauthorized",
                            "message": "管理接口需要 Authorization: Bearer <API Key>。",
                        }
                    },
                )
        return None

    # ------------------------------------------------------------------ 概览
    @router.get("/overview")
    async def overview(request: Request) -> Any:
        denied = guard(request)
        if denied:
            return denied
        status = backend.status()
        snapshot = stats.snapshot()
        return {
            "service": {
                "name": SERVICE_NAME,
                "version": SERVICE_VERSION,
                "python": sys.version.split()[0],
                "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
                "pid": os.getpid(),
                "started_at": snapshot["started_at"],
                "uptime_s": snapshot["uptime_s"],
                "address": f"{settings.host}:{settings.port}",
                "openapi": "/docs" if settings.docs else None,
            },
            "backend": status,
            "models": _model_rows(settings, status),
            "stats": {
                key: snapshot[key]
                for key in (
                    "total",
                    "succeeded",
                    "failed",
                    "success_rate",
                    "latency_ms",
                    "inference_ms",
                    "throughput",
                )
            },
        }

    # ------------------------------------------------------------------ 配置
    @router.get("/config")
    async def read_config(request: Request, reveal: bool = Query(False)) -> Any:
        denied = guard(request)
        if denied:
            return denied
        active = config_path(None)
        return {
            "values": _serialise(settings, reveal),
            "defaults": _serialise(Settings(), reveal=False),
            "fields": field_meta,
            "hot_fields": sorted(HOT_FIELDS),
            "cold_fields": sorted(COLD_FIELDS),
            # 本次启动实际读的是哪个文件（可能与即将写入的目标不同：先读到了
            # 工作目录的 server.json，但控制台只会往用户目录写）。
            "active_config_file": str(active) if active else None,
            "config_file": str(HOME_DIR / "config.json"),
            "overridden": sorted(settings_diff(settings)),
        }

    @router.put("/config")
    async def update_config(
        request: Request, updates: Dict[str, Any] = Body(...)
    ) -> Any:
        denied = guard(request)
        if denied:
            return denied

        # 用 apply_updates 空改一遍拿到「当前生效值」的独立副本：直接 Settings(**当前值)
        # 拿到的是未 resolve 的版本，aliases 之类可能差一项，会让下面的变更检测误报。
        previous = apply_updates(settings, {})
        try:
            updated = apply_updates(previous, updates)
        except (ValueError, TypeError) as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "error": {"code": "invalid_config", "message": str(exc)},
                },
            )

        # 按**设置字段**逐个比对，而不是遍历请求体的键：请求体里的键不一定是字段名
        # （`preload` 会被迁移成 `auto_load`），拿它去 getattr 会炸；而「前后两个
        # Settings 差在哪」本来就是这个问题的准确答案，跟请求体怎么写的无关。
        changed = sorted(
            name
            for name in Settings.__dataclass_fields__
            if getattr(updated, name) != getattr(previous, name)
        )
        # 就地更新：settings 是共享引用，端点与中间件下一请求就能看到新值。
        for name in Settings.__dataclass_fields__:
            object.__setattr__(settings, name, getattr(updated, name))

        restart_required = sorted(set(changed) & COLD_FIELDS)
        applied = sorted(set(changed) & HOT_FIELDS)
        side_effects: List[str] = []
        if on_settings_changed is not None:
            side_effects = on_settings_changed(previous, updated)

        saved_to: Optional[str] = None
        try:
            saved_to = str(save_settings(settings, baseline))
        except OSError as exc:
            # 存不进磁盘不等于配置没生效，但必须说清楚——否则用户下次启动会
            # 发现改动「自己没了」，而且无从查起。
            side_effects.append(f"配置已生效但写入失败：{exc}")

        return {
            "changed": changed,
            "applied_now": applied,
            "restart_required": restart_required,
            "side_effects": side_effects,
            "saved_to": saved_to,
            "values": _serialise(settings, reveal=False),
        }

    @router.delete("/config")
    async def reset_config(request: Request) -> Any:
        denied = guard(request)
        if denied:
            return denied
        target = HOME_DIR / "config.json"
        if target.is_file():
            target.unlink()
        return {"removed": str(target)}

    # ------------------------------------------------------------------ 统计
    @router.get("/stats")
    async def read_stats(request: Request) -> Any:
        denied = guard(request)
        if denied:
            return denied
        return stats.snapshot()

    @router.post("/stats/reset")
    async def reset_stats(request: Request) -> Any:
        denied = guard(request)
        if denied:
            return denied
        stats.reset()
        return {"reset": True}

    @router.get("/requests")
    async def read_requests(request: Request, limit: int = Query(50, ge=1, le=1000)) -> Any:
        denied = guard(request)
        if denied:
            return denied
        return {"items": stats.recent(limit)}

    # ------------------------------------------------------------------ 资源
    @router.get("/system")
    def read_system(request: Request) -> Any:
        """CPU / 内存 / GPU / MLX 显存。

        这个端点是**同步**的（没有 `async def`）：采集要起 `ioreg`、`vm_stat`
        两三个子进程，外加等一个 0.25 秒的 CPU 采样窗口。声明成同步函数，
        FastAPI 会自动把它扔进线程池，事件循环不会被堵住 —— 比自己在里面
        再套一层 `to_thread` 干净。

        不可用的项会是 `{"available": false, "reason": "..."}`，而不是缺字段，
        前端不用为「这个系统没有 GPU 统计」写特例。
        """
        denied = guard(request)
        if denied:
            return denied
        if monitor is None:
            return {"available": False, "reason": "这个实例没有接资源采集器"}
        return monitor.snapshot()

    # ------------------------------------------------------------------ 模型
    @router.get("/models")
    async def read_models(request: Request) -> Any:
        denied = guard(request)
        if denied:
            return denied
        return {"models": _model_rows(settings, backend.status())}

    @router.post("/models/preload")
    async def preload_models(request: Request) -> Any:
        """加载**全部**槽位，本地没有的会去 Hugging Face 下。

        单个槽位失败不会让整个请求失败：失败原因记在 `load_errors` 里，
        由 `models` 那一栏标出来。「三个里有两个好了」是要让人看见的状态，
        不是要变成一个 503。
        """
        denied = guard(request)
        if denied:
            return denied
        import anyio

        started = time.perf_counter()
        # 首次下载是分钟级阻塞，放到线程里，别把事件循环顶住。
        loaded = await anyio.to_thread.run_sync(backend.preload)
        return {
            "loaded": loaded,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "models": _model_rows(settings, backend.status()),
        }

    @router.post("/models/load")
    async def load_model(
        request: Request, model: str = Body(..., embed=True)
    ) -> Any:
        """加载指定的一个模型。

        这是「不靠发推理请求也能触发下载」的正规入口 —— 走的是和首个请求**完全同一条**
        路径（`_agent()` → `laya.load()` → `snapshot_download`），只是不需要为了过
        schema 校验去编一个假的 state。

        注意它是同步阻塞的：权重没缓存时要下几百 MB，这个请求会一直挂着，直到下完
        （或者超时）。控制台的按钮会一直转，这是预期行为而不是卡死。
        """
        denied = guard(request)
        if denied:
            return denied
        import anyio

        started = time.perf_counter()
        try:
            repo = await anyio.to_thread.run_sync(backend.load, model)
        except BackendError as exc:
            status, code = (
                (422, "unknown_model")
                if isinstance(exc, (UnknownModelError, RequestError))
                else (503, "model_unavailable")
            )
            return JSONResponse(
                status_code=status,
                content={"error": {"code": code, "message": str(exc)}},
            )
        return {
            "loaded": repo,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "models": _model_rows(settings, backend.status()),
        }

    @router.post("/models/unload")
    async def unload_models(request: Request, model: Optional[str] = Body(None, embed=True)) -> Any:
        denied = guard(request)
        if denied:
            return denied
        try:
            backend.unload(model)
        except BackendError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": {"code": "unknown_model", "message": str(exc)}},
            )
        return {"models": _model_rows(settings, backend.status())}

    # ------------------------------------------------------------------ 生命周期
    @router.post("/shutdown")
    async def shutdown(request: Request) -> Any:
        denied = guard(request)
        if denied:
            return denied
        if on_shutdown is None:
            return JSONResponse(
                status_code=501,
                content={
                    "error": {
                        "code": "shutdown_unavailable",
                        "message": "当前进程没有注册停机回调（命令行直接用 uvicorn 起的时候"
                        "控制台停不了它）。用 laya-server 或桌面端启动即可。",
                    }
                },
            )
        on_shutdown()
        return {"stopping": True}

    return router


def _model_rows(settings: Settings, status: Dict[str, Any]) -> List[Dict[str, Any]]:
    loaded = set(status.get("loaded") or [])
    errors = status.get("load_errors") or {}
    rows: List[Dict[str, Any]] = []
    for slot, repo in settings.slots.items():
        rows.append(
            {
                "slot": slot,
                "checkpoint": repo,
                "loaded": repo in loaded,
                "error": errors.get(repo),
                "aliases": sorted(a for a, s in settings.aliases.items() if s == slot),
                "default": slot == settings.default_slot,
            }
        )
    # 用户直接用 checkpoint id 加载过的、不属于任何槽位的权重也要列出来，
    # 否则「内存里到底驻留着什么」这个问题就答不全。
    slots_repos = set(settings.slots.values())
    for repo in sorted(loaded - slots_repos):
        rows.append(
            {
                "slot": None,
                "checkpoint": repo,
                "loaded": True,
                "error": errors.get(repo),
                "aliases": [],
                "default": False,
            }
        )
    return rows


__all__ = ["SERVICE_NAME", "SERVICE_VERSION", "build_admin_router"]
