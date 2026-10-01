"""控制台与管理接口的测试。

配置持久化会写磁盘，落到哪个目录由 `conftest.py` 在导入前决定（临时目录）。
这里只负责每个测试前把它清干净。
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from laya_server import config as config_module
from laya_server.app import create_app, start_autoload
from laya_server.backend import EchoBackend
from laya_server.config import Settings, apply_updates, save_settings, settings_diff
from laya_server.stats import StatsCollector, Trace

HOME = Path(os.environ["LAYA_SERVER_HOME"])
CONFIG_FILE = HOME / "config.json"

STATE = "I was charged twice, please refund."
QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "pay", "tech": "bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
    "refund": {"type": "noul", "instructions": "Refund requested?"},
}

#: Starlette 的 TestClient 默认把 client 设成 ("testclient", 50000) —— 那不是个 IP，
#: 会被「只认本机」的判定挡住。真给一个回环地址，测的才是真实路径。
LOOPBACK = ("127.0.0.1", 50000)


@pytest.fixture(autouse=True)
def clean_home():
    """每个测试都从一个干净的 home 开始。"""
    for item in HOME.glob("*.json*"):
        item.unlink()
    yield
    for item in HOME.glob("*.json*"):
        item.unlink()


def make_client(**overrides):
    settings = replace(Settings(backend="echo"), **overrides).resolved
    app = create_app(settings, EchoBackend(settings))
    return TestClient(app, client=LOOPBACK), settings


class StubMonitor:
    """桩资源采集器。真的那个要起 ioreg / vm_stat，测试不该依赖那些。"""

    def __init__(self, payload):
        self.payload = payload

    def snapshot(self):
        return self.payload


def make_client_with_monitor(payload, **overrides):
    settings = replace(Settings(backend="echo"), **overrides).resolved
    app = create_app(settings, EchoBackend(settings), monitor=StubMonitor(payload))
    return TestClient(app, client=LOOPBACK), settings


def infer(client, **payload_overrides):
    payload = {"state": STATE, "model": "jev-latest", "questions": QUESTIONS}
    payload.update(payload_overrides)
    return client.post("/v1/systemone", json=payload)


# ---------------------------------------------------------------------------
# 控制台页面
# ---------------------------------------------------------------------------


def test_root_serves_console_html():
    client, _ = make_client()
    with client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # 页面得真的引用管理接口，否则就是一张静态海报。
    assert "/admin/overview" in response.text
    assert "laya-server" in response.text


def test_root_falls_back_to_json_when_console_disabled():
    client, _ = make_client(console=False)
    with client:
        body = client.get("/").json()
    assert body["endpoint"] == "POST /v1/systemone"
    assert body["console"] is None


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def test_stats_counts_successes_and_failures():
    client, _ = make_client()
    with client:
        assert infer(client).status_code == 200
        assert infer(client).status_code == 200
        assert infer(client, model="gpt-4o").status_code == 422
        stats = client.get("/admin/stats").json()

    assert stats["total"] == 3
    assert stats["succeeded"] == 2
    assert stats["failed"] == 1
    assert stats["by_status"] == {"200": 2, "422": 1}
    # 被 schema 挡下来的请求也要能归因到具体错误码，否则「为什么失败」答不上来。
    assert stats["by_error_code"] == {"unknown_model": 1}
    assert stats["by_model"] == {"aac6fef/laya-mlx": 2}
    assert stats["answer_types"] == {"choice": 2, "score": 2, "noul": 2}


def test_stats_records_inference_time_separately_from_queue():
    client, _ = make_client()
    with client:
        infer(client)
        items = client.get("/admin/requests").json()["items"]
    entry = items[0]
    assert entry["status"] == 200
    assert entry["inference_ms"] is not None
    assert entry["latency_ms"] >= entry["inference_ms"]
    assert entry["slot"] == "english"
    assert entry["model"] == "aac6fef/laya-mlx"
    assert entry["question_count"] == 3


def test_stats_ignores_admin_traffic():
    """控制台自己每 3 秒轮询一次，要是被算进请求统计，数字就永远是假的。"""
    client, _ = make_client()
    with client:
        for _ in range(5):
            client.get("/admin/overview")
            client.get("/admin/stats")
        assert client.get("/admin/stats").json()["total"] == 0


def test_requests_are_newest_first():
    client, _ = make_client()
    with client:
        infer(client)
        infer(client, model="laya-multilingual")
        items = client.get("/admin/requests").json()["items"]
    assert items[0]["at"] >= items[1]["at"]


def test_stats_reset():
    client, _ = make_client()
    with client:
        infer(client)
        assert client.get("/admin/stats").json()["total"] == 1
        client.post("/admin/stats/reset")
        assert client.get("/admin/stats").json()["total"] == 0


def test_series_has_fixed_bucket_count():
    """空桶也要补 0：否则图上断格，分不清是没流量还是没画出来。"""
    client, _ = make_client()
    with client:
        stats = client.get("/admin/stats").json()
    assert len(stats["series"]) == 30
    assert all(bucket["count"] == 0 for bucket in stats["series"])


def test_stats_collector_bounds_memory():
    collector = StatsCollector(capacity=5)
    for i in range(20):
        collector.record(Trace(request_id=str(i), started_at=0.0), float(i))
    assert collector.snapshot()["total"] == 5
    assert collector.snapshot()["capacity"] == 5


# ---------------------------------------------------------------------------
# 概览
# ---------------------------------------------------------------------------


def test_overview_shape():
    client, _ = make_client()
    with client:
        body = client.get("/admin/overview").json()
    assert body["service"]["name"] == "laya-server"
    assert body["service"]["address"].endswith(":8077")
    assert {row["slot"] for row in body["models"]} == {"english", "multilingual", "typed-decisions"}
    assert body["models"][0]["aliases"]
    assert set(body["stats"]) == {
        "total", "succeeded", "failed", "success_rate",
        "latency_ms", "inference_ms", "throughput",
    }


# ---------------------------------------------------------------------------
# 配置读写
# ---------------------------------------------------------------------------


def test_config_reads_and_masks_api_key():
    client, _ = make_client(api_key="super-secret-key-1234")
    with client:
        body = client.get(
            "/admin/config", headers={"Authorization": "Bearer super-secret-key-1234"}
        ).json()
    assert body["values"]["api_key"] == "••••••••1234"
    assert body["hot_fields"] and body["cold_fields"]
    # 字段元数据要能让前端直接渲染表单，不用把类型硬编码在 JS 里。
    assert body["fields"]["port"]["kind"] == "int"
    assert body["fields"]["debug"]["kind"] == "bool"
    assert body["fields"]["debug"]["mutable"] is True
    assert body["fields"]["port"]["mutable"] is False


def test_config_hot_change_takes_effect_immediately():
    client, settings = make_client()
    with client:
        response = client.put("/admin/config", json={"debug": True, "max_concurrency": 4})
        assert response.status_code == 200
        body = response.json()

        assert body["applied_now"] == ["debug", "max_concurrency"]
        assert body["restart_required"] == []
        assert any("并发闸门" in note for note in body["side_effects"])
        # 热改必须就地生效：同一个进程、同一个 app，下一个请求就要看到新值。
        assert settings.debug is True
        assert settings.max_concurrency == 4
        assert "debug" in infer(client).json()


def test_config_cold_change_is_persisted_and_flagged():
    client, settings = make_client()
    with client:
        body = client.put("/admin/config", json={"port": 9999, "dtype": "float32"}).json()
    assert body["restart_required"] == ["dtype", "port"]
    assert body["applied_now"] == []
    assert body["saved_to"] == str(CONFIG_FILE)

    saved = (Path(HOME) / "config.json").read_text(encoding="utf-8")
    # 只存 diff：存全量会把今天的默认值冻进文件，以后改默认值对老用户就不生效了。
    assert '"port": 9999' in saved.replace("\n", " ")
    assert "max_questions" not in saved
    assert settings.port == 9999


def test_config_rejects_unknown_field():
    client, _ = make_client()
    with client:
        response = client.put("/admin/config", json={"home": "/tmp"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_config"


def test_config_rejects_emptying_slots():
    """空 map 会被 resolved 悄悄换成默认值，那不是用户想要的「清空」。"""
    client, _ = make_client()
    with client:
        response = client.put("/admin/config", json={"slots": {}})
    assert response.status_code == 422
    assert "不能清空" in response.json()["error"]["message"]


def test_config_rejects_broken_alias_mapping():
    client, _ = make_client()
    with client:
        response = client.put(
            "/admin/config",
            json={"aliases": {"jev-latest": "nonexistent-slot"}},
        )
    assert response.status_code == 422
    assert "不存在的槽位" in response.json()["error"]["message"]


def test_settings_diff_and_save_round_trip():
    settings = replace(Settings(), port=1234, debug=True).resolved
    diff = settings_diff(settings)
    assert diff == {"port": 1234, "debug": True}

    written = save_settings(settings)
    assert written == CONFIG_FILE
    assert not written.with_suffix(".json.tmp").exists()

    reloaded = config_module.load_settings(str(written))
    assert reloaded.port == 1234
    assert reloaded.debug is True


def test_save_does_not_persist_values_that_came_from_the_command_line():
    """命令行/环境变量给的值不能被写进配置文件。

    真出过这个 bug：`laya-console --backend echo` 跑一次，再在控制台保存任意一项，
    用户的配置文件里就永久多了一条 `backend: echo`。下次他不加参数启动，
    拿到的是一个答案全是假的假后端，而且完全不知道为什么。
    """
    baseline = replace(Settings(), backend="echo").resolved  # 启动时就长这样
    current = apply_updates(baseline, {"max_concurrency": 4})  # 用户在控制台只动了这一项
    save_settings(current, baseline)

    on_disk = CONFIG_FILE.read_text(encoding="utf-8")
    assert "backend" not in on_disk, on_disk
    assert '"max_concurrency": 4' in on_disk


def test_save_preserves_hand_written_file_entries():
    """存盘要合并已有文件，不能把用户手写的其他项抹掉。"""
    CONFIG_FILE.write_text('{"queue_timeout_s": 30}\n', encoding="utf-8")
    baseline = Settings().resolved
    current = apply_updates(baseline, {"debug": True})
    save_settings(current, baseline)

    saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    assert saved["queue_timeout_s"] == 30
    assert saved["debug"] is True


def test_save_removes_override_when_value_returns_to_default():
    """把改过的项改回默认，应当从文件里删掉，而不是留下一条等于默认值的记录。"""
    CONFIG_FILE.write_text('{"port": 9999, "debug": true}\n', encoding="utf-8")
    baseline = replace(Settings(), port=9999, debug=True).resolved
    current = apply_updates(baseline, {"port": Settings().port, "debug": False})
    save_settings(current, baseline)

    saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    assert "port" not in saved
    assert "debug" not in saved


def test_config_reset_removes_file():
    client, _ = make_client()
    with client:
        client.put("/admin/config", json={"port": 1234})
        assert (Path(HOME) / "config.json").is_file()
        body = client.delete("/admin/config").json()
    assert body["removed"].endswith("config.json")
    assert not (Path(HOME) / "config.json").is_file()


# ---------------------------------------------------------------------------
# 管理面隔离
# ---------------------------------------------------------------------------


def test_admin_requires_api_key_when_configured():
    client, _ = make_client(api_key="k")
    with client:
        assert client.get("/admin/overview").status_code == 401
        assert client.get("/admin/overview", headers={"Authorization": "Bearer k"}).status_code == 200


def test_admin_denied_for_non_local_client():
    """`--host 0.0.0.0` 的时候，管理面不能跟着一起暴露到局域网。"""
    from fastapi import Request

    from laya_server.app import is_local_client

    assert is_local_client(Request({"type": "http", "client": ("127.0.0.1", 1)}))
    assert is_local_client(Request({"type": "http", "client": ("::1", 1)}))
    assert not is_local_client(Request({"type": "http", "client": ("192.168.1.20", 1)}))
    assert not is_local_client(Request({"type": "http"}))


def test_admin_denied_for_remote_client_through_app():
    from laya_server import app as app_module

    original = app_module.is_local_client
    app_module.is_local_client = lambda request: False
    try:
        client, _ = make_client()
        with client:
            response = client.get("/admin/overview")
    finally:
        app_module.is_local_client = original
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "admin_local_only"


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------


def test_shutdown_calls_registered_callback():
    settings = Settings(backend="echo").resolved
    called = []
    app = create_app(settings, EchoBackend(settings), on_shutdown=lambda: called.append(True))
    with TestClient(app, client=LOOPBACK) as client:
        assert client.post("/admin/shutdown").json() == {"stopping": True}
    assert called == [True]


def test_shutdown_reports_when_no_callback():
    client, _ = make_client()
    with client:
        response = client.post("/admin/shutdown")
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "shutdown_unavailable"


def test_unload_endpoint():
    client, _ = make_client()
    with client:
        body = client.post("/admin/models/unload", json={"model": None}).json()
    assert len(body["models"]) == 3


def test_load_one_model_resolves_to_its_checkpoint():
    """单独加载走的是和首个请求同一条路径（_agent → laya.load），
    但不需要为了过 schema 去编一个假的 state。

    之前这个按钮是「发一个 state='warmup' 的假推理请求，再顺手把所有槽位都加载一遍」——
    两条都错：假的推理请求会混进请求统计里，而按钮写的是单个加载、实际做的是全量加载。
    """
    client, _ = make_client()
    with client:
        response = client.post("/admin/models/load", json={"model": "laya-multilingual"})
    assert response.status_code == 200
    assert response.json()["loaded"] == "aac6fef/laya-multilingual-mlx"


def test_load_one_model_does_not_touch_request_stats():
    client, _ = make_client()
    with client:
        client.post("/admin/models/load", json={"model": "jev-latest"})
        assert client.get("/admin/stats").json()["total"] == 0


def test_load_one_model_rejects_unknown():
    client, _ = make_client()
    with client:
        response = client.post("/admin/models/load", json={"model": "gpt-4o"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_model"


def test_load_one_model_rejects_auto():
    """auto 要靠 state 的语言才能决定加载哪个，这里没有 state。"""
    client, _ = make_client()
    with client:
        response = client.post("/admin/models/load", json={"model": "auto"})
    assert response.status_code == 422
    assert "auto" in response.json()["error"]["message"]


def test_healthz_includes_uptime_and_request_count():
    client, _ = make_client()
    with client:
        infer(client)
        body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["requests"] == 1
    assert body["uptime_s"] >= 0


# ---------------------------------------------------------------------------
# Hugging Face 端点
# ---------------------------------------------------------------------------


def test_hf_env_reads_the_official_variable_names(monkeypatch):
    """HF_ENDPOINT / HF_HUB_OFFLINE 用官方名字，不加我们自己的前缀。

    用户很可能已经在别处设过这两个变量了，要求他再学一套 LAYA_SERVER_* 只是
    制造「我明明设了为什么没用」的困惑。
    """
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    settings = config_module.load_settings()
    assert settings.hf_endpoint == "https://hf-mirror.com"
    assert settings.hf_offline is True


def test_prefixed_alias_overrides_the_official_name(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    monkeypatch.setenv("LAYA_SERVER_HF_ENDPOINT", "https://hf-mirror.com")
    assert config_module.load_settings().hf_endpoint == "https://hf-mirror.com"


def test_apply_hf_env_writes_environment(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    config_module.apply_hf_env(
        replace(Settings(), hf_endpoint="https://hf-mirror.com", hf_offline=True).resolved
    )
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert os.environ["HF_HUB_OFFLINE"] == "1"

    # 留空则不动环境变量 —— 别把用户自己设的值覆盖掉。
    monkeypatch.setenv("HF_ENDPOINT", "https://example.invalid")
    config_module.apply_hf_env(Settings().resolved)
    assert os.environ["HF_ENDPOINT"] == "https://example.invalid"


# ---------------------------------------------------------------------------
# 资源占用接口
# ---------------------------------------------------------------------------


def test_system_endpoint_returns_the_collector_payload():
    payload = {
        "sampled_at": 1.0,
        "host": {"chip": "Apple M1", "cpu_cores": 8, "total_mb": 16384.0},
        "cpu": {"available": True, "percent": 12.5},
        "memory": {"available": True, "percent": 60.0},
        "process": {"available": True, "pid": 1, "resident_mb": 900.0},
        "gpu": {"available": False, "reason": "没有 Metal 设备"},
        "mlx": {"available": True, "active_mb": 780.0},
        "notes": ["统一内存"],
    }
    client, _ = make_client_with_monitor(payload)
    with client:
        body = client.get("/admin/system").json()
    assert body == payload


def test_system_endpoint_requires_the_api_key():
    client, _ = make_client_with_monitor({"available": True}, api_key="k")
    with client:
        assert client.get("/admin/system").status_code == 401
        assert (
            client.get("/admin/system", headers={"Authorization": "Bearer k"}).status_code
            == 200
        )


def test_system_endpoint_respects_the_local_only_guard():
    """资源占用能看出这台机器在干嘛，同样不该暴露到局域网。"""
    from laya_server import app as app_module

    original = app_module.is_local_client
    app_module.is_local_client = lambda request: False
    try:
        client, _ = make_client_with_monitor({"available": True})
        with client:
            assert client.get("/admin/system").status_code == 403
    finally:
        app_module.is_local_client = original


# ---------------------------------------------------------------------------
# auto_load：启动时自动加载本地已有的权重
# ---------------------------------------------------------------------------


class RecordingBackend:
    """只记下被怎么调用的后端，不加载任何东西。"""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def preload(self, only_local=False):
        self.calls.append(only_local)
        if self.error is not None:
            raise self.error
        return ["aac6fef/laya-mlx"]


def test_autoload_local_only_never_touches_the_network():
    """默认档位：只读本地已有的那份，绝不联网。"""
    backend = RecordingBackend()
    settings = replace(Settings(), auto_load="local").resolved
    thread = start_autoload(settings, backend)
    assert thread is not None
    thread.join(timeout=10)
    assert backend.calls == [True]


def test_autoload_all_is_willing_to_download():
    backend = RecordingBackend()
    settings = replace(Settings(), auto_load="all").resolved
    thread = start_autoload(settings, backend)
    thread.join(timeout=10)
    assert backend.calls == [False]


def test_autoload_off_starts_nothing():
    backend = RecordingBackend()
    settings = replace(Settings(), auto_load="off").resolved
    assert start_autoload(settings, backend) is None
    assert backend.calls == []


def test_autoload_is_skipped_for_the_echo_backend():
    """echo 后端没有权重可加载，没必要为它去 import mlx。"""
    settings = replace(Settings(), backend="echo", auto_load="all").resolved
    assert start_autoload(settings, EchoBackend(settings)) is None


def test_autoload_failure_is_reported_not_raised(capsys):
    """后台线程里抛出去就没人接了 —— 失败要打成日志，而且不能挡住启动。"""
    settings = replace(Settings(), auto_load="local").resolved
    thread = start_autoload(settings, RecordingBackend(error=RuntimeError("磁盘满了")))
    thread.join(timeout=10)
    assert "磁盘满了" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# auto_load 的配置层
# ---------------------------------------------------------------------------


def test_auto_load_is_rendered_as_a_dropdown():
    """取值只有三档的字段要让前端渲染成下拉框，而不是让人手打字符串。"""
    client, _ = make_client()
    with client:
        meta = client.get("/admin/config").json()["fields"]["auto_load"]
    assert meta["kind"] == "choice"
    assert meta["choices"] == ["off", "local", "all"]
    assert meta["mutable"] is False, "启动时才读的东西不该标成即时生效"


def test_legacy_preload_true_becomes_auto_load_all():
    """升级之后「配置文件突然不被认了」是最难查的那类失败，所以老键要平移。"""
    CONFIG_FILE.write_text('{"preload": true}\n', encoding="utf-8")
    assert config_module.load_settings(str(CONFIG_FILE)).auto_load == "all"


def test_legacy_preload_false_is_treated_as_no_opinion():
    """`preload: false` 在老版本里**就是默认值**，没携带信息。

    当成「明确要求关掉」会改变那些只是照抄了示例配置的人的行为。
    """
    CONFIG_FILE.write_text('{"preload": false}\n', encoding="utf-8")
    assert config_module.load_settings(str(CONFIG_FILE)).auto_load == "local"


def test_legacy_preload_is_accepted_over_the_admin_api():
    client, settings = make_client()
    with client:
        body = client.put("/admin/config", json={"preload": True}).json()
    assert body["changed"] == ["auto_load"]
    assert body["restart_required"] == ["auto_load"]
    assert settings.auto_load == "all"


def test_auto_load_rejects_a_typo():
    client, _ = make_client()
    with client:
        response = client.put("/admin/config", json={"auto_load": "lcoal"})
    assert response.status_code == 422
    assert "auto_load" in response.json()["error"]["message"]


class BrokenBackend:
    """`status()` 会抛异常的后端 —— 模拟「MLX 扩展初始化失败」那种情况。"""

    def status(self):
        raise ImportError("Encountered an error while initializing the extension.")


def test_healthz_stays_200_even_when_the_backend_is_broken():
    """健康检查**不能**在服务有问题时自己挂掉 —— 它是排查的第一步。

    打包之后真发生过：MLX 扩展初始化失败 → `backend.status()` 去读包版本时抛异常
    → `/healthz` 返回 500，看起来像「服务根本没起来」，其实服务好好的、
    只是推理不可用。所以这里降级成 degraded + error，仍然 200。
    """
    settings = replace(Settings(backend="echo")).resolved
    app = create_app(settings, BrokenBackend())
    with TestClient(app, client=LOOPBACK) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "Encountered an error" in body["error"]
    assert body["loaded"] == []
    assert body["uptime_s"] >= 0


def test_load_failure_message_walks_the_cause_chain():
    """外层文案往往是笼统的，真正的原因挂在 `__cause__` 上。

    打包时就是靠这一条才定位到「扩展初始化失败」的根因 —— 只看 `str(exc)`
    只会得到一句 "Encountered an error while initializing the extension."，
    完全不知道从哪下手。
    """
    from laya_server.backend import describe_error

    try:
        try:
            raise OSError("Failed to load the default metallib")
        except OSError as inner:
            raise ImportError("Encountered an error while initializing the extension.") from inner
    except ImportError as exc:  # noqa: PERF203 —— 这里就是要拿到这个异常对象
        text = describe_error(exc)

    assert "Encountered an error" in text
    assert "Failed to load the default metallib" in text
