"""注册表 / 路由 / 生命周期的测试。用桩替代 laya_mlx，不碰权重。

这里锁的是**我写的那部分**：别名解析、auto 路由判定、同一份权重只构建一次、
LRU 淘汰、未知模型报错。laya-mlx 自己那部分（前向、分词、温度标定）由它自带的
benchmarks 负责，重复一遍没意义。
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from laya_server.backend import (
    REQUIRED_CHECKPOINT_FILES,
    MlxBackend,
    ModelLoadError,
    UnknownModelError,
    local_snapshot,
    local_source,
    to_jev_answers,
)
from laya_server.config import Settings

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "payments", "technical": "bugs"},
    },
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
    "refund": {"type": "noul", "instructions": "Refund requested?"},
}


class StubAgent:
    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.calls = 0

    def predict(self, state, questions):
        self.calls += 1
        answers = {}
        for qid, definition in questions.items():
            qtype = definition["type"]
            if qtype == "choice":
                labels = list(definition["criteria"])
                answers[qid] = {
                    "type": "choice",
                    "choice": labels[0],
                    "probabilities": {label: (1.0 if i == 0 else 0.0) for i, label in enumerate(labels)},
                    "confidence": 1.0,
                    "action": {"act_probability": 0.5},
                }
            elif qtype == "score":
                levels = list(definition["criteria"])
                answers[qid] = {
                    "type": "score",
                    "score": 0.0,
                    "legend": {str(i): value for i, value in enumerate(levels)},
                    "probabilities": {str(i): (1.0 if i == 0 else 0.0) for i in range(len(levels))},
                    "confidence": 1.0,
                    "action": {"act_probability": 0.5},
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": 0.25,
                    "confidence": 0.75,
                    "action": {"act_probability": 0.5},
                }
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }


class StubLaya:
    """最小可用的 laya_mlx 替身：只实现我们真正调用的两个入口。"""

    __version__ = "0.0.0-stub"

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.loads: list[tuple[str, dict]] = []
        self.agents: dict[str, StubAgent] = {}
        self.fail_on = fail_on or set()

    def load(self, repo, **kwargs):
        self.loads.append((repo, kwargs))
        if repo in self.fail_on:
            raise RuntimeError("权重下载失败")
        agent = StubAgent(repo)
        self.agents[repo] = agent
        return agent

    @staticmethod
    def detect_language(state):
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        return {
            "script": "han" if has_cjk else "latin",
            "script_profile": {},
            "language": None if has_cjk else "en",
            "is_english": not has_cjk,
            "language_undecided": has_cjk,
            "diacritic_rate": 0.0,
            "non_latin_fraction": 1.0 if has_cjk else 0.0,
        }


@pytest.fixture()
def backend_and_stub():
    settings = replace(Settings(), max_loaded=2).resolved
    backend = MlxBackend(settings)
    stub = StubLaya()
    backend._laya = stub
    return backend, stub


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested, expected_kind, expected_target",
    [
        ("jev-latest", "slot", "english"),
        ("jev-1.13.0", "slot", "english"),
        ("jev-preview", "slot", "english"),
        ("JEV-LATEST", "slot", "english"),
        ("laya-multilingual", "slot", "multilingual"),
        ("auto", "auto", ""),
        ("acme/custom-weights", "repo", "acme/custom-weights"),
    ],
)
def test_resolve(backend_and_stub, requested, expected_kind, expected_target):
    backend, _ = backend_and_stub
    assert backend.resolve(requested) == (expected_kind, expected_target)


def test_unknown_alias_is_rejected(backend_and_stub):
    backend, _ = backend_and_stub
    with pytest.raises(UnknownModelError):
        backend.resolve("gpt-4o")


def test_raw_checkpoint_refused_when_disabled():
    settings = replace(Settings(), allow_raw_checkpoint=False).resolved
    backend = MlxBackend(settings)
    backend._laya = StubLaya()
    with pytest.raises(UnknownModelError):
        backend.resolve("acme/custom-weights")


# ---------------------------------------------------------------------------
# auto 路由
# ---------------------------------------------------------------------------

EN_REPO = "aac6fef/laya-mlx"
MULTI_REPO = "aac6fef/laya-multilingual-mlx"


def test_auto_routes_english_state_to_english_slot(backend_and_stub):
    backend, _ = backend_and_stub
    repo, slot, routing = backend.route("I was charged twice, please refund.", "auto")
    assert (repo, slot) == (EN_REPO, "english")
    assert "英文" in routing["reason"]


def test_auto_routes_cjk_state_to_multilingual_slot(backend_and_stub):
    backend, _ = backend_and_stub
    repo, slot, routing = backend.route("发票被重复扣款了，请尽快退款。", "auto")
    # 中文走英文 checkpoint 不是「效果差一点」，是直接崩（laya 文档给的数据）。
    assert (repo, slot) == (MULTI_REPO, "multilingual")
    assert "非拉丁文字" in routing["reason"]
    assert routing["detection"]["script"] == "han"


def test_explicit_alias_skips_detection(backend_and_stub):
    backend, stub = backend_and_stub
    repo, slot, routing = backend.route("发票被重复扣款了", "jev-latest")
    assert (repo, slot) == (EN_REPO, "english")
    # 显式指定时不该白跑一次语言检测。
    assert routing["detection"] is None if "detection" in routing else True
    assert "alias" in routing["reason"]


# ---------------------------------------------------------------------------
# 本地快照（不联网加载）
# ---------------------------------------------------------------------------


def _make_snapshot(cache: Path, repo: str, revision: str, *, complete: bool = True) -> Path:
    snapshot = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    names = list(REQUIRED_CHECKPOINT_FILES) if complete else ["rl_agent_config.json"]
    for name in names:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    return snapshot


def test_local_snapshot_finds_a_complete_cache_entry(monkeypatch, temp_dir):
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    snapshot = _make_snapshot(temp_dir, "acme/laya", "abc123")
    assert local_snapshot("acme/laya") == snapshot


def test_local_snapshot_returns_none_when_missing(monkeypatch, temp_dir):
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    assert local_snapshot("acme/nope") is None


def test_local_snapshot_rejects_a_half_downloaded_checkpoint(monkeypatch, temp_dir):
    """半截快照不能当完整的用 —— 否则会得到一个语焉不详的 FileNotFoundError。"""
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    _make_snapshot(temp_dir, "acme/laya", "abc123", complete=False)
    assert local_snapshot("acme/laya") is None


def test_local_snapshot_respects_hf_home(monkeypatch, temp_dir):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(temp_dir))
    snapshot = _make_snapshot(temp_dir / "hub", "acme/laya", "abc123")
    assert local_snapshot("acme/laya") == snapshot


def test_local_snapshot_ignores_bare_names(monkeypatch, temp_dir):
    """没有 / 的不是 repo id，别去猜缓存路径。"""
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    assert local_snapshot("laya-mlx") is None


# ---------------------------------------------------------------------------
# 「本地有没有」这个判断，是 auto_load=local 的全部依据
# ---------------------------------------------------------------------------


def test_local_source_accepts_a_plain_directory(monkeypatch, temp_dir):
    """slots 里直接写本地目录也要认。

    这种配置下 huggingface_hub 完全不参与，是最「离线」的用法。
    """
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir / "empty"))
    weights = temp_dir / "laya-fp16"
    weights.mkdir()
    assert local_source(str(weights)) == (str(weights), "本地目录")


def test_local_source_falls_back_to_the_cache(monkeypatch, temp_dir):
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    snapshot = _make_snapshot(temp_dir, "acme/laya", "abc123")
    assert local_source("acme/laya") == (str(snapshot), "本地缓存")


def test_local_source_reports_nothing_local(monkeypatch, temp_dir):
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    assert local_source("acme/nope") is None
    assert local_source(str(temp_dir / "not-there")) is None


def test_agent_is_loaded_from_the_local_snapshot_when_available(monkeypatch, temp_dir):
    """本地有完整快照时，交给 laya 的必须是本地路径。

    这是「权重都在磁盘上，加载还是 503」那个 bug 的回归测试：只要把 repo id 原样
    递下去，huggingface_hub 就会先去 HF 问最新版本，网络不通整个加载就失败。
    """
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    snapshot = _make_snapshot(temp_dir, "aac6fef/laya-mlx", "cafe")

    settings = replace(Settings(), max_loaded=3).resolved
    backend = MlxBackend(settings)
    stub = StubLaya()
    backend._laya = stub

    backend.infer("hello", QUESTIONS, "jev-latest")

    loaded_path, _kwargs = stub.loads[0]
    assert loaded_path == str(snapshot), "应当把本地快照路径交给 laya，而不是 repo id"


def test_repo_id_is_passed_through_when_cache_is_cold(monkeypatch, temp_dir):
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    settings = replace(Settings()).resolved
    backend = MlxBackend(settings)
    stub = StubLaya()
    backend._laya = stub
    backend.infer("hello", QUESTIONS, "jev-latest")
    loaded_path, _kwargs = stub.loads[0]
    assert loaded_path == EN_REPO, "本地没有缓存时才该让 laya 自己去下"


def test_load_failure_mentions_the_missing_cache(monkeypatch, temp_dir):
    """加载失败时的报错要能指路：是没有缓存，还是下载本身挂了。"""
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    settings = replace(Settings()).resolved
    backend = MlxBackend(settings)
    backend._laya = StubLaya(fail_on={EN_REPO})

    with pytest.raises(ModelLoadError) as excinfo:
        backend.infer("hello", QUESTIONS, "jev-latest")
    assert "本地没有" in str(excinfo.value)
    assert "hf_endpoint" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------


def test_same_checkpoint_is_built_once(backend_and_stub):
    backend, stub = backend_and_stub
    for _ in range(3):
        backend.infer("hello", QUESTIONS, "jev-latest")
    assert len(stub.loads) == 1
    assert stub.agents[EN_REPO].calls == 3


def test_lru_evicts_when_over_limit(backend_and_stub):
    backend, stub = backend_and_stub
    assert backend.settings.max_loaded == 2
    backend.infer("hello", QUESTIONS, "jev-latest")
    backend.infer("hello", QUESTIONS, "laya-multilingual")
    backend.infer("hello", QUESTIONS, "laya-typed-decisions")

    assert len(stub.loads) == 3
    # 上限 2，最久没用过的 english 应该被淘汰。
    assert backend.status()["loaded"] == [MULTI_REPO, "aac6fef/laya-typed-decisions-mlx"]


def test_evicted_checkpoint_reloads_but_in_flight_result_is_unaffected():
    settings = replace(Settings(), max_loaded=1).resolved
    backend = MlxBackend(settings)
    stub = StubLaya()
    backend._laya = stub

    first_raw, _ = backend.infer("hello", QUESTIONS, "jev-latest")
    backend.infer("发票", QUESTIONS, "laya-multilingual")
    assert len(stub.loads) == 2  # 已经被淘汰，重新建过

    # 淘汰前拿到的结果仍然完整可用 —— 调用栈里的强引用让 Agent 不会被中途回收。
    answers = to_jev_answers(first_raw["answers"], QUESTIONS)
    assert answers["route"]["choice"] == "billing"
    assert answers["refund"]["noul"] == 0.25


def test_preload_builds_every_slot(backend_and_stub):
    backend, stub = backend_and_stub
    backend.preload()
    assert {repo for repo, _ in stub.loads} == {EN_REPO, MULTI_REPO, "aac6fef/laya-typed-decisions-mlx"}


def test_preload_local_only_touches_nothing_when_the_cache_is_empty(
    backend_and_stub, monkeypatch, temp_dir
):
    """`auto_load: local` 的底线：本地一份都没有时，一个加载都不该发起。

    这条比它看起来重要 —— `laya.load()` 一旦被调用，传下去的是 repo id 就会去
    Hugging Face 下载。启动时「悄悄下 2GB」正是这个开关要避免的事。
    """
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    backend, stub = backend_and_stub
    assert backend.preload(only_local=True) == []
    assert stub.loads == [], "本地没有权重时不该调用 laya.load"


def test_preload_local_only_loads_exactly_what_is_on_disk(
    backend_and_stub, monkeypatch, temp_dir
):
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    snapshot = _make_snapshot(temp_dir, EN_REPO, "cafe")
    backend, stub = backend_and_stub

    assert backend.preload(only_local=True) == [EN_REPO]
    # 本地没有的槽位一次都不该被碰 —— 碰了就会触发下载。
    # 注意 stub 记的是**传给 laya 的路径**：只有一份加载，而且是快照路径。
    assert [path for path, _ in stub.loads] == [str(snapshot)]


def test_preload_local_only_uses_the_snapshot_path(backend_and_stub, monkeypatch, temp_dir):
    """本地加载必须把**路径**交给 laya，而不是 repo id —— 否则又会去联网校验版本。"""
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    snapshot = _make_snapshot(temp_dir, EN_REPO, "cafe")
    backend, stub = backend_and_stub

    backend.preload(only_local=True)
    assert stub.loads[0][0] == str(snapshot)


def test_preload_keeps_going_when_one_slot_fails(backend_and_stub, monkeypatch, temp_dir):
    """一份权重炸了不该让另外两份白加载，也不该把异常抛给调用方。

    「三个里有两个已经好了」是要让人看见的状态；抛出去只会变成一个 503，
    然后用户以为一个都没成。
    """
    monkeypatch.setenv("HF_HUB_CACHE", str(temp_dir))
    backend, stub = backend_and_stub
    stub.fail_on = {MULTI_REPO}

    loaded = backend.preload()

    assert EN_REPO in loaded
    assert MULTI_REPO not in loaded
    assert MULTI_REPO in backend.status()["load_errors"]


def test_parameters_reach_the_agent():
    settings = replace(
        Settings(), dtype="float32", batch_size=32, pad_to_multiple=16, cache_prompts=True
    ).resolved
    backend = MlxBackend(settings)
    stub = StubLaya()
    backend._laya = stub
    backend.infer("hello", QUESTIONS, "jev-latest")
    _, kwargs = stub.loads[0]
    assert kwargs == {
        "dtype": "float32",
        "device": None,
        "batch_size": 32,
        "compile": False,
        "pad_to_multiple": 16,
        "cache_prompts": True,
    }


def test_load_failure_becomes_model_unavailable_and_is_remembered():
    settings = replace(Settings()).resolved
    backend = MlxBackend(settings)
    backend._laya = StubLaya(fail_on={EN_REPO})

    with pytest.raises(ModelLoadError):
        backend.infer("hello", QUESTIONS, "jev-latest")

    # 失败原因要留痕：5xx 的 message 是排查时唯一能拿到的一手信息。
    assert EN_REPO in backend.status()["load_errors"]

    # 失败不缓存 Agent，下次请求会重试。
    with pytest.raises(ModelLoadError):
        backend.infer("hello", QUESTIONS, "jev-latest")


def test_unload():
    settings = replace(Settings()).resolved
    backend = MlxBackend(settings)
    backend._laya = StubLaya()
    backend.infer("hello", QUESTIONS, "jev-latest")
    assert backend.status()["loaded"]
    backend.unload()
    assert backend.status()["loaded"] == []


# ---------------------------------------------------------------------------
# 端到端（桩后端）：确认服务层拿到的是 Jev 形状
# ---------------------------------------------------------------------------


def test_endpoint_with_stub_laya():
    from fastapi.testclient import TestClient

    from laya_server.app import create_app

    settings = Settings().resolved
    backend = MlxBackend(settings)
    backend._laya = StubLaya()

    with TestClient(create_app(settings, backend)) as client:
        response = client.post(
            "/v1/systemone",
            json={"state": "发票重复扣款", "model": "auto", "questions": QUESTIONS},
        )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "answers", "usage"}
    # auto 判定为中文 → 多语言权重。
    assert body["model"] == MULTI_REPO
    assert set(body["answers"]["route"]) == {"type", "choice", "probabilities", "confidence"}
    assert set(body["answers"]["refund"]) == {"type", "noul"}
