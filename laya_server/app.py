"""HTTP 层：把 Jev 的 `/v1/systemone` 接口原样搬过来，底下换成本地 laya-mlx。

端点清单
--------
POST /v1/systemone   Jev 兼容主端点。请求/响应形状与 TypeSafe 文档一致。
GET  /v1/models      本地可用别名与槽位。Jev 没有这个端点，但本地服务需要它
                     ——否则客户端只能靠试错发现「这个模型到底存不存在」。
GET  /healthz        存活与模型驻留状态。
GET  /docs           OpenAPI（可用 docs=false 关掉）。

错误语义对齐 Jev 文档：401 缺/错 API key，422 请求体不合法，429 排队超时。
Jev 文档里还有 529「服务端过载」，本地单机场景下对应的就是排队超时，统一归到 429。
"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import anyio
from fastapi import Depends, FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .admin import SERVICE_VERSION, build_admin_router
from .backend import (
    Backend,
    BackendError,
    InferenceError,
    ModelLoadError,
    RequestError,
    UnknownModelError,
    build_backend,
    describe_error,
    to_jev_answers,
)
from .config import Settings, apply_updates, load_settings
from .schemas import ErrorResponse, SystemOneRequest, SystemOneResponse
from .stats import StatsCollector, Trace
from .sysinfo import SystemMonitor

REQUEST_ID_HEADER = "X-Request-Id"
INFERENCE_PATH = "/v1/systemone"


def console_path() -> Path:
    """控制台 HTML 的位置。

    PyInstaller 打包后 `__file__` 指向解包目录里合成的 .pyc 路径，旁边的 static/
    是我们在 spec 里显式收进来的那一份（`_MEIPASS/laya_server/static`）。
    两个候选都试一遍，谁存在用谁 —— 比在导入期 assert 好，至少开发态和打包态
    能共用同一段代码。
    """
    candidates = [Path(__file__).parent / "static" / "console.html"]
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidates.insert(0, Path(base) / "laya_server" / "static" / "console.html")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# 错误 --> HTTP
# ---------------------------------------------------------------------------


class QueueTimeout(Exception):
    """等推理槽位超时。对应 Jev 的 429 Too Many Requests。"""


class Unauthorized(Exception):
    """API key 缺失或不匹配。对应 Jev 的 401。"""


#: 后端异常 → (状态码, 错误码)。顺序有意义：子类要排在父类前面。
_ERROR_MAP: tuple[tuple[type[BaseException], int, str], ...] = (
    (UnknownModelError, 422, "unknown_model"),
    (RequestError, 422, "invalid_request"),
    (ModelLoadError, 503, "model_unavailable"),
    (InferenceError, 500, "inference_failed"),
)


def _error_body(code: str, message: str, *, detail: Any = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail is not None:
        body["detail"] = detail
    return body


def _jsonable_errors(errors: list) -> list:
    """把 pydantic 的 errors() 洗成可 JSON 序列化的形状。

    两件事必须做：自定义校验器抛的 ValueError 会以**对象**形式挂在 ctx 里，
    直接 JSONResponse 会在序列化时炸掉（于是本该 422 的请求变成一个二次异常）；
    而 jsonable_encoder 遇到这类对象只会编码成空字典，把「到底哪里错了」弄丢。
    所以 ctx 里的非标量值统一 str()，保住那句人话。
    """
    cleaned = []
    for error in errors:
        item = dict(error)
        ctx = item.get("ctx")
        if isinstance(ctx, dict):
            item["ctx"] = {
                key: value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
                for key, value in ctx.items()
            }
        cleaned.append(jsonable_encoder(item))
    return cleaned


def _classify(exc: BaseException) -> tuple[int, str]:
    for exc_type, status, code in _ERROR_MAP:
        if isinstance(exc, exc_type):
            return status, code
    return 500, "internal_error"


# ---------------------------------------------------------------------------
# 服务对象
# ---------------------------------------------------------------------------


class InferenceService:
    """把「排队 → 线程里跑推理 → 归一化」这段串起来。

    MLX 是单设备，多个线程同时前向只会互相抢 GPU 且更容易 OOM，所以用一个信号量
    把推理串行化；但推理本身跑在线程里，事件循环不会被堵住，健康检查和排队等待
    仍能正常响应。
    """

    def __init__(self, settings: Settings, backend: Backend) -> None:
        self.settings = settings
        self.backend = backend
        self._gate: Optional[asyncio.Semaphore] = None

    @property
    def gate(self) -> asyncio.Semaphore:
        # 在事件循环里首次使用时创建：Semaphore 绑定 loop，模块导入期创建会在
        # 多个测试 event loop 之间复用出问题。
        if self._gate is None:
            self._gate = asyncio.Semaphore(max(1, int(self.settings.max_concurrency)))
        return self._gate

    def reset_gate(self) -> None:
        """并发数改了之后重建闸门。

        正在等旧闸门的请求会继续等旧的（它们本来就已经在队列里），新请求走新的。
        这不完美，但比「改了不生效、必须重启」好，也比「中途把等待者踢掉」温和。
        """
        self._gate = None

    async def run(self, payload: SystemOneRequest, trace: Optional[Trace] = None) -> Dict[str, Any]:
        state = payload.state
        # 不能用 exclude_none：choice 的选项描述允许是 null（「这个选项不用额外说明」），
        # 递归剔掉 None 会让那个选项整条消失。
        questions = {qid: definition.model_dump() for qid, definition in payload.questions.items()}

        limit = int(self.settings.max_state_chars)
        if limit > 0 and isinstance(state, str) and len(state) > limit:
            raise RequestError(
                f"state 长度 {len(state)} 超过上限 {limit}（max_state_chars）。"
                " 注意 checkpoint 自身的 context 更小，截断会静默丢信息。"
            )

        acquire_started = time.perf_counter()
        try:
            await asyncio.wait_for(
                self.gate.acquire(), timeout=float(self.settings.queue_timeout_s)
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise QueueTimeout(
                f"等待推理槽位超过 {self.settings.queue_timeout_s}s"
                f"（max_concurrency={self.settings.max_concurrency}）"
            ) from exc

        try:
            inference_started = time.perf_counter()
            raw, routing = await anyio.to_thread.run_sync(
                functools.partial(self.backend.infer, state, questions, payload.model),
                # 推理一旦开始就不该被取消：中途抛 CancelledError 会留下半个推理状态，
                # 而 MLX 缓冲本来就只在栈解开后回收。
                abandon_on_cancel=False,
            )
            inference_ms = round((time.perf_counter() - inference_started) * 1000, 2)
        finally:
            self.gate.release()

        queue_ms = round((inference_started - acquire_started) * 1000, 2)
        answers = to_jev_answers(raw.get("answers") or {}, questions, passthrough=self.settings.debug)
        usage = raw.get("usage") or {}

        result: Dict[str, Any] = {
            "model": raw.get("model") or payload.model,
            "answers": answers,
            "usage": {
                "input_tokens": int(usage.get("input_tokens") or 0),
                # System One 不逐 token 解码，恒为 0。字段保留是为了让按 Jev 写的
                # 计量代码不至于 KeyError。
                "output_tokens": int(usage.get("output_tokens") or 0),
            },
        }
        if self.settings.debug:
            kind, target = self.backend.resolve(payload.model)
            result["debug"] = {
                "requested_model": payload.model,
                "resolved": {"kind": kind, "target": target},
                "routing": routing,
                "latency_ms": inference_ms,
                "queue_ms": queue_ms,
                "question_count": len(questions),
            }

        if trace is not None:
            # 记账放在服务层而不是中间件：这里才知道真实模型、token 数和纯推理耗时。
            # 排队等待单独记，否则「变慢了」在两个原因（排队 vs 推理）之间分不出来。
            trace.model_reported = result["model"]
            trace.input_tokens = result["usage"]["input_tokens"]
            trace.inference_ms = inference_ms
            # 显式别名时后端不做路由，routing 是空的；从别名表补一个槽位，
            # 否则控制台上「走的哪份权重」这一列会一直空着。
            trace.slot = (routing or {}).get("slot") or self.settings.aliases.get(payload.model)
            counts: Dict[str, int] = {}
            for answer in answers.values():
                counts[answer["type"]] = counts.get(answer["type"], 0) + 1
            trace.answer_types = counts

        result["_inference_ms"] = inference_ms
        result["_queue_ms"] = queue_ms
        return result


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------


def is_local_client(request: Request) -> bool:
    """请求是不是从本机发起的。

    只信 TCP 层的 client 地址，不看 `X-Forwarded-For` 之类的头 —— 那些头是客户端
    自己写的，拿它做权限判断等于没做。
    """
    client = request.client
    if client is None:
        return False
    host = client.host or ""
    if host in {"127.0.0.1", "::1", "localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def start_autoload(settings: Settings, backend: Backend) -> Optional[threading.Thread]:
    """按 `auto_load` 的档位在**后台**把本地已有的权重读进内存。

    为什么不等它加载完再让服务就绪：读一份权重是秒级（首次下载是分钟级），
    而这段时间里服务完完全全是可用的。挡在启动路径上只会让「双击图标」
    白等好几秒、控制台连都连不上，还看不出是在干嘛。放到后台，页面立刻能打开，
    模型那一栏会自己长出来。

    失败一律不往外抛：真实原因（磁盘满、权重损坏）下次请求还会再报一次，
    而在这里抛只会变成一个没人接的后台异常。
    """
    if settings.backend == "echo" or settings.auto_load == "off":
        return None

    only_local = settings.auto_load == "local"
    print(
        "[laya-server] 后台自动加载权重（auto_load="
        + settings.auto_load
        + "）："
        + ("只加载本地已有的，不联网" if only_local else "本地没有的会去 Hugging Face 下载"),
        flush=True,
    )

    def worker() -> None:
        started = time.perf_counter()
        try:
            loaded = backend.preload(only_local=only_local)
        except Exception as exc:  # noqa: BLE001 —— 后台线程里抛出去就没人接了
            print(f"[laya-server] 自动加载失败：{type(exc).__name__}: {exc}", flush=True)
            return
        elapsed = time.perf_counter() - started
        if loaded:
            print(
                f"[laya-server] 自动加载完成：{len(loaded)} 份权重已就绪，耗时 {elapsed:.2f}s",
                flush=True,
            )
        else:
            print(
                f"[laya-server] 没有可自动加载的权重（本地一份都没有），"
                f"首次请求时会去下载。耗时 {elapsed:.2f}s",
                flush=True,
            )

    thread = threading.Thread(target=worker, name="laya-autoload", daemon=True)
    thread.start()
    return thread


def create_app(
    settings: Optional[Settings] = None,
    backend: Optional[Backend] = None,
    *,
    monitor: Optional[SystemMonitor] = None,
    on_shutdown: Optional[Callable[[], None]] = None,
) -> FastAPI:
    settings = settings or load_settings()
    backend = backend or build_backend(settings)
    service = InferenceService(settings, backend)
    stats = StatsCollector(capacity=settings.stats_capacity)
    # 资源采集器要活到进程结束：CPU 是增量指标，得跨请求记住上一次的 tick 读数。
    # 允许注入是为了让测试不用真的去读 ioreg / vm_stat。
    monitor = monitor or SystemMonitor()
    # 启动时的生效值。存盘以它为基准：命令行/环境变量给的值不该被写进配置文件。
    baseline = apply_updates(settings, {})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(
            f"[laya-server] backend={settings.backend} dtype={settings.dtype} "
            f"max_concurrency={settings.max_concurrency} max_loaded={settings.max_loaded}",
            flush=True,
        )
        print(
            "[laya-server] aliases: "
            + ", ".join(f"{alias}→{slot}" for alias, slot in sorted(settings.aliases.items())),
            flush=True,
        )
        if settings.backend == "echo":
            print(
                "[laya-server] ⚠️  echo 后端：不会加载任何权重，答案全是假的。"
                " 只用于联调与测试，别拿它做任何判断。",
                flush=True,
            )
        start_autoload(settings, backend)
        if settings.api_key is None:
            print("[laya-server] 未设置 api_key：不校验 Authorization。", flush=True)
        yield

    app = FastAPI(
        title="laya-server",
        version=SERVICE_VERSION,
        summary="Jev 兼容的 System One 接口，后端是本地 laya-mlx 推理",
        description=(
            "接口形状对齐 TypeSafe Jev（`POST /v1/systemone`），权重与推理全部在本地。\n\n"
            "**注意**：本地跑的是 Laya 的 MLX 移植，与 TypeSafe 云端的 Jev 不是同一份权重。"
            "按 Jev 标定过的 confidence 阈值需要重新标定。\n\n"
            "控制台在 `/`，管理接口在 `/admin/*`（默认只接受本机来源）。"
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs else None,
    )
    app.state.settings = settings
    app.state.backend = backend
    app.state.service = service
    app.state.stats = stats
    app.state.monitor = monitor

    # ---------------------------------------------------------------- 中间件
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Laya-Latency-Ms"] = f"{elapsed:.2f}"

        if request.url.path == INFERENCE_PATH:
            # 端点正常走完会留下 trace；被 schema 挡下来（422）或鉴权挡住（401）的
            # 请求没有 trace，照样记一条 —— 「今天有多少请求是被参数写错挡住的」
            # 本身就是个要看的问题。
            trace = getattr(request.state, "trace", None)
            if trace is None:
                trace = stats.new_trace(request_id, None, 0)
            trace.status = response.status_code
            if trace.error_code is None:
                trace.error_code = getattr(request.state, "error_code", None)
            stats.record(trace, elapsed)
        return response

    # ---------------------------------------------------------------- 异常处理
    def _respond(request: Request, status: int, code: str, message: str, detail: Any = None,
                 headers: Optional[Dict[str, str]] = None) -> JSONResponse:
        # 记在 request.state 上，中间件收尾时取走；从异常处理器里反读响应体做不到
        # （body 是流），而错误码是统计里最该看清的一列。
        request.state.error_code = code
        return JSONResponse(
            status_code=status,
            content=_error_body(code, message, detail=detail),
            headers={REQUEST_ID_HEADER: getattr(request.state, "request_id", ""), **(headers or {})},
        )

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Jev 的 422 说「body details the offending field」。FastAPI 的 detail 列表
        # 正好是逐字段的，原样带出去；外面再套一层 error 便于客户端统一解析。
        return _respond(
            request,
            422,
            "invalid_request",
            "请求体校验失败",
            detail=_jsonable_errors(exc.errors()),
        )

    @app.exception_handler(BackendError)
    async def on_backend_error(request: Request, exc: BackendError) -> JSONResponse:
        # 只登记具体类型（BackendError 及其子类），不登记裸 Exception：
        # Starlette 在裸 Exception 的处理器返回响应之后仍会把异常重新抛出去
        # （好让服务器日志和测试客户端能看到），那会污染正常路径。
        status, code = _classify(exc)
        print(f"[laya-server] {code}: {type(exc).__name__}: {exc}", flush=True)
        return _respond(request, status, code, str(exc))

    @app.exception_handler(QueueTimeout)
    async def on_queue_timeout(request: Request, exc: QueueTimeout) -> JSONResponse:
        return _respond(
            request,
            429,
            "too_many_requests",
            str(exc),
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _respond(request, exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        """兜底。Starlette 会在这个处理器返回响应之后继续把异常抛给服务器日志，
        这是想要的：线上有结构化 500，测试里照样能看见真实堆栈。"""
        print(f"[laya-server] internal_error: {type(exc).__name__}: {exc}", flush=True)
        return _respond(request, 500, "internal_error", f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- 鉴权
    async def require_api_key(request: Request) -> None:
        expected = settings.api_key
        if not expected:
            return
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or token.strip() != expected:
            raise Unauthorized()

    @app.exception_handler(Unauthorized)
    async def on_unauthorized(request: Request, exc: Unauthorized) -> JSONResponse:
        return _respond(
            request,
            401,
            "unauthorized",
            "API key 缺失或不正确，检查 Authorization: Bearer <key>。",
        )

    # ---------------------------------------------------------------- 端点
    @app.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        dependencies=[Depends(require_api_key)],
        responses={
            401: {"model": ErrorResponse, "description": "API key 缺失或不正确"},
            422: {"model": ErrorResponse, "description": "请求体不合法"},
            429: {"model": ErrorResponse, "description": "排队超时"},
            503: {"model": ErrorResponse, "description": "模型不可用"},
        },
    )
    async def system_one(
        payload: SystemOneRequest, request: Request, response: Response
    ) -> Any:
        """评估 state 与一组带类型的问题，每个问题返回一个结构化答案。"""
        trace = stats.new_trace(
            request.state.request_id, payload.model, len(payload.questions)
        )
        request.state.trace = trace

        result = await service.run(payload, trace)
        inference = result.pop("_inference_ms", None)
        result.pop("_queue_ms", None)
        if inference is not None:
            response.headers["X-Laya-Inference-Ms"] = f"{inference}"

        if settings.debug:
            # debug 模式下会带上 Jev 没有的字段（routing / action 等），
            # 不能再过一遍 response_model，否则会被逐字段校验剔掉。
            return JSONResponse(content=result)
        return SystemOneResponse.model_validate(result)

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        """存活检查。

        **这个端点本身不能失败。** 它是「服务怎么了」的第一个入口，摔在这里等于把
        整条排查路径也一起断掉。打包之后真发生过：MLX 扩展初始化失败 →
        `backend.status()` 去读包版本时抛异常 → `/healthz` 返回 500，
        看起来像「服务根本没起来」，其实服务好好的、只是推理不可用。

        所以后端探测失败时降级成 `status: "degraded"` + `error`，仍然 200 ——
        「进程活着」和「权重可用」是两件事，健康检查只该回答第一件。
        """
        snapshot = stats.snapshot()
        body: Dict[str, Any] = {
            "status": "ok",
            "models": _model_index(settings),
            "uptime_s": snapshot["uptime_s"],
            "requests": snapshot["total"],
        }
        try:
            body.update(backend.status())
        except Exception as exc:  # noqa: BLE001
            body["status"] = "degraded"
            body["backend"] = settings.backend
            body["loaded"] = []
            body["load_errors"] = {}
            body["error"] = describe_error(exc)
        return body

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """本地可用的别名与槽位。Jev 没有这个端点，但本地服务缺了它很难用。"""
        status = backend.status()
        loaded = set(status.get("loaded") or [])
        return {
            "default": settings.default_model,
            "objects": [
                {
                    "alias": alias,
                    "slot": slot,
                    "checkpoint": settings.slots.get(slot),
                    "loaded": settings.slots.get(slot) in loaded,
                }
                for alias, slot in sorted(settings.aliases.items())
            ],
            "backend": status,
        }

    @app.get("/", include_in_schema=False)
    async def root(request: Request) -> Any:
        """控制台页面。控制台自己会去调 `/admin/*`，所以这里不需要鉴权 ——
        只有一堆静态 HTML，认证发生在数据接口上。"""
        if settings.console and console_path().is_file():
            return FileResponse(
                console_path(),
                media_type="text/html; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )
        return {
            "service": "laya-server",
            "endpoint": "POST /v1/systemone",
            "console": "/" if settings.console else None,
            "docs": "/docs" if settings.docs else None,
            "models": _model_index(settings),
        }

    # ---------------------------------------------------------------- 管理接口
    def on_settings_changed(previous: Settings, updated: Settings) -> list:
        """热改字段的副作用。返回给控制台显示，让人知道「我刚才那一下到底动了什么」。"""
        notes = []
        if updated.max_concurrency != previous.max_concurrency:
            service.reset_gate()
            notes.append(
                f"并发闸门已重建：{previous.max_concurrency} → {updated.max_concurrency}"
            )
        if updated.max_loaded != previous.max_loaded:
            if updated.max_loaded < previous.max_loaded:
                backend.trim()
                notes.append(f"常驻上限降到 {updated.max_loaded}，已按 LRU 立即淘汰")
            else:
                notes.append(f"常驻上限升到 {updated.max_loaded}，下次加载时生效")
        if updated.api_key != previous.api_key:
            notes.append("API Key 已变更，所有调用方（包括本控制台）都要用新的")
        return notes

    app.include_router(
        build_admin_router(
            settings,
            backend,
            stats,
            is_local=is_local_client,
            baseline=baseline,
            monitor=monitor,
            on_settings_changed=on_settings_changed,
            on_shutdown=on_shutdown,
        )
    )

    return app


def _model_index(settings: Settings) -> Dict[str, Optional[str]]:
    """槽位 → checkpoint。给健康检查和根路径复用。"""
    return {slot: repo for slot, repo in settings.slots.items()}


__all__ = [
    "InferenceService",
    "QueueTimeout",
    "console_path",
    "create_app",
    "is_local_client",
    "start_autoload",
]
