"""命令行入口：`laya-server` 或 `python -m laya_server`。

命令行参数故意只覆盖「每次起服务都可能要调」的那几项；剩下的走配置文件或环境变量，
避免把 argparse 写成第二个配置文件。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Optional, Sequence

from .config import AUTO_LOAD_MODES, Settings, load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-server",
        description="Jev 兼容（POST /v1/systemone）的本地推理服务，后端是 laya-mlx。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="JSON/YAML 配置文件路径。命令行参数覆盖它。")
    parser.add_argument("--host", help="监听地址。默认 127.0.0.1（只对本机开放）。")
    parser.add_argument("--port", type=int, help="监听端口。")
    parser.add_argument(
        "--model",
        help="默认模型：别名（jev-latest / laya-multilingual / ...）或 checkpoint id。",
    )
    parser.add_argument(
        "--api-key",
        help="设置后所有请求都要带 Authorization: Bearer <key>。不设则不校验。",
    )
    parser.add_argument("--dtype", help="权重精度：float16 / float32 / bfloat16。")
    parser.add_argument("--device", help="MLX 设备：gpu / cpu。不填用 MLX 默认。")
    parser.add_argument("--batch-size", type=int, help="单次前向最多处理多少个问题。")
    parser.add_argument(
        "--max-concurrency", type=int, help="同时进行的推理数。MLX 单设备，默认 1。"
    )
    parser.add_argument("--max-loaded", type=int, help="常驻权重数量上限（LRU）。")
    parser.add_argument(
        "--backend",
        choices=["mlx", "echo"],
        help="推理后端。echo 不加载权重、答案全假，只用于联调与测试。",
    )
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
    parser.add_argument("--debug", action="store_true", help="响应里附带路由与延迟等调试字段。")
    parser.add_argument("--no-docs", action="store_true", help="关闭 /docs 与 /openapi.json。")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = load_settings(args.config)
    updates = {}
    for field in (
        "host",
        "port",
        "model",
        "dtype",
        "device",
        "batch_size",
        "max_concurrency",
        "max_loaded",
        "backend",
        "auto_load",
    ):
        value = getattr(args, field, None)
        if value is None:
            continue
        updates["default_model" if field == "model" else field] = value
    if args.api_key is not None:
        updates["api_key"] = args.api_key
    if args.preload and "auto_load" not in updates:
        updates["auto_load"] = "all"
    if args.debug:
        updates["debug"] = True
    if args.no_docs:
        updates["docs"] = False
    return replace(settings, **updates).resolved if updates else settings


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)

    import uvicorn

    from .app import create_app

    if settings.host not in {"127.0.0.1", "localhost", "::1"} and settings.api_key is None:
        print(
            f"[laya-server] ⚠️  正在监听 {settings.host}:{settings.port} 且没有设置 api_key——"
            "任何能连到这个地址的人都能调用你的模型。",
            flush=True,
        )

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
