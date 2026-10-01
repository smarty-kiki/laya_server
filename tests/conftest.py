"""测试公共配置。

两件事必须在**任何 `laya_server` 模块被导入之前**做掉，conftest.py 是 pytest
保证会最先加载的文件：

1. `LAYA_SERVER_HOME` 指到临时目录。
   `config.HOME_DIR` 是导入期求值的模块级常量。不这么做的话，跑一次
   `test_console.py` 就会往用户真实的 `~/.laya-server/config.json` 里写进
   测试用的端口和 debug 开关，用户下次启动会发现配置「自己变了」。

2. `HF_HUB_CACHE` 指到**空**的临时目录。
   否则 backend 的 `local_snapshot()` 会命中开发机真实的 Hugging Face 缓存 ——
   于是「本地没缓存时应该把 repo id 透传下去」这类断言会突然失败，
   而失败原因取决于这台机器上碰巧下过什么模型。测试不能依赖那个。
   真的有 2GB 权重的机器上跑测试还会白读一堆文件。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

HOME = Path(tempfile.mkdtemp(prefix="laya-tests-"))
os.environ.setdefault("LAYA_SERVER_HOME", str(HOME))

HF_CACHE = Path(tempfile.mkdtemp(prefix="laya-tests-hf-"))
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
os.environ.pop("HF_HOME", None)

# 仓库工作目录里如果放了 server.json，也会被 load_settings 捞进去，测试同样要隔离。
os.environ.pop("LAYA_SERVER_CONFIG", None)


@pytest.fixture()
def temp_dir() -> Iterator[Path]:
    """一个一次性的临时目录。

    **不用 pytest 的 `tmp_path`。** 那个 fixture 依赖会话级 basetemp
    （`.../T/pytest-of-unknown`），它只在第一次运行时 mkdir 成功；在受限沙箱里，
    第二次跑测试时「目录已存在」会被报成 `PermissionError`，于是整个文件全红，
    而失败原因和被测代码毫无关系。自己建就没有这个共享状态。
    """
    path = Path(tempfile.mkdtemp(prefix="laya-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
