"""契约测试：脱离权重跑，锁住「接口形状」而不是「模型准不准」。

模型准不准要用真实权重和真实样本量去量（laya-mlx 自带 benchmarks），
把那种测试塞进 CI 只会让每次跑都要下载几个 GB。这里只保证：

  * 请求校验按 Jev 的取值域执行（选项上限、score 等级数、noul criteria 键）；
  * 响应**逐字段**是 Jev 的形状，不夹带多余的 key；
  * 错误码对齐 Jev 文档（401 / 422 / 429）；
  * 归一化逻辑对边界情况（null 选项描述、legend 原来是对象）不静默丢数据。
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from laya_server.app import create_app
from laya_server.backend import EchoBackend, InferenceError, to_jev_answers
from laya_server.config import Settings

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency"},
    },
}

STATE = "I've been trying to connect my Stripe account for 3 days and it keeps failing."


def make_settings(**overrides) -> Settings:
    """默认的 echo 后端配置，带上需要覆盖的字段。"""
    return replace(Settings(backend="echo"), **overrides).resolved


@pytest.fixture()
def client():
    settings = make_settings()
    with TestClient(create_app(settings, EchoBackend(settings))) as c:
        yield c


def post(client: TestClient, **payload_overrides):
    payload = {"state": STATE, "model": "jev-latest", "questions": QUESTIONS}
    payload.update(payload_overrides)
    return client.post("/v1/systemone", json=payload)


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------


def test_response_is_jevs_exact_shape(client: TestClient):
    response = post(client)
    assert response.status_code == 200
    body = response.json()

    # 顶层三个键，不多不少。
    assert set(body) == {"model", "answers", "usage"}, body
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert isinstance(body["usage"]["input_tokens"], int)
    assert body["usage"]["output_tokens"] == 0

    # 答案与请求同键、同序。
    assert list(body["answers"]) == list(QUESTIONS)

    choice = body["answers"]["department"]
    assert set(choice) == {"type", "choice", "probabilities", "confidence"}
    assert choice["type"] == "choice"
    assert choice["choice"] in QUESTIONS["department"]["criteria"]
    assert set(choice["probabilities"]) == set(QUESTIONS["department"]["criteria"])
    assert abs(sum(choice["probabilities"].values()) - 1.0) < 0.05

    score = body["answers"]["frustration"]
    assert set(score) == {"type", "score", "legend", "probabilities", "confidence"}
    assert score["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert set(score["probabilities"]) == {"0", "1", "2"}

    noul = body["answers"]["is_urgent"]
    # Jev 的 noul 答案只有 type + noul；laya 多给的 confidence 必须被剔掉。
    assert set(noul) == {"type", "noul"}
    assert 0.0 <= noul["noul"] <= 1.0


def test_headers_carry_request_id(client: TestClient):
    response = post(client)
    assert response.headers["X-Request-Id"]
    assert float(response.headers["X-Laya-Inference-Ms"]) >= 0


def test_debug_mode_adds_extras_without_changing_default(client: TestClient):
    settings = make_settings(debug=True)
    with TestClient(create_app(settings, EchoBackend(settings))) as debug_client:
        body = post(debug_client).json()
    assert "debug" in body
    assert body["debug"]["resolved"]["kind"] == "slot"
    # debug 模式下保留 laya 的原生字段，方便排查「置信度为什么这么高」。
    assert "action" in body["answers"]["is_urgent"]


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "questions",
    [
        {},  # 空
        {"q": {"type": "nope", "instructions": "x"}},  # 未知类型
        {"q": {"type": "choice", "instructions": "x", "criteria": {}}},  # 没有选项
        {"q": {"type": "choice", "instructions": "x"}},  # 缺 criteria
        {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}},  # 等级不足
        {
            "q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}
        },  # 等级超上限
        {"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}},  # 多余的键
        {"q": {"type": "noul"}},  # 缺 instructions
    ],
)
def test_invalid_questions_are_rejected(client: TestClient, questions):
    response = post(client, questions=questions)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["detail"]  # Jev 的 422 会指出问题出在哪个字段


def test_missing_state_is_rejected(client: TestClient):
    response = client.post("/v1/systemone", json={"model": "jev-latest", "questions": QUESTIONS})
    assert response.status_code == 422


def test_state_may_be_structured(client: TestClient):
    response = post(client, state={"messages": [{"role": "user", "content": STATE}]})
    assert response.status_code == 200


def test_unknown_model_is_422(client: TestClient):
    response = post(client, model="gpt-4o")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_model"


def test_raw_checkpoint_id_is_accepted(client: TestClient):
    response = post(client, model="some-org/some-checkpoint")
    assert response.status_code == 200
    assert response.json()["model"] == "some-org/some-checkpoint"


def test_state_length_guard():
    settings = make_settings(max_state_chars=32)
    with TestClient(create_app(settings, EchoBackend(settings))) as c:
        response = post(c)
        assert response.status_code == 422
        assert "max_state_chars" in response.json()["error"]["message"]


def test_api_key_enforced_when_configured():
    settings = make_settings(api_key="s3cret")
    with TestClient(create_app(settings, EchoBackend(settings))) as c:
        assert post(c).status_code == 401
        assert post(c).json()["error"]["code"] == "unauthorized"

        headers = {"Authorization": "Bearer s3cret"}
        payload = {"state": STATE, "model": "jev-latest", "questions": QUESTIONS}
        ok = c.post("/v1/systemone", json=payload, headers=headers)
        assert ok.status_code == 200

        wrong = c.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401


# ---------------------------------------------------------------------------
# 元数据端点
# ---------------------------------------------------------------------------


def test_models_endpoint_lists_aliases(client: TestClient):
    body = client.get("/v1/models").json()
    aliases = {item["alias"] for item in body["objects"]}
    assert {"jev-latest", "jev-1.13.0", "laya-multilingual"} <= aliases
    assert body["default"] == "jev-latest"


def test_healthz(client: TestClient):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert "english" in body["models"]


# ---------------------------------------------------------------------------
# 归一化单元测试
# ---------------------------------------------------------------------------


def test_choice_option_with_null_description_survives(client: TestClient):
    # Jev 允许 null 表示「这个选项不用额外说明」。归一化不能把它整条丢掉。
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Where?",
            "criteria": {"fast": None, "cheap": None},
        }
    }
    body = post(client, questions=questions).json()
    assert set(body["answers"]["route"]["probabilities"]) == {"fast", "cheap"}


def test_legend_stringifies_structured_criteria():
    raw = {
        "answers": {
            "sev": {
                "type": "score",
                "score": 1.0,
                "probabilities": {"0": 0.0, "1": 1.0},
                "confidence": 1.0,
            }
        }
    }
    questions = {
        "sev": {
            "type": "score",
            "instructions": "severity",
            "criteria": [{"label": "none"}, {"label": "bad"}],
        }
    }
    answers = to_jev_answers(raw["answers"], questions)
    assert answers["sev"]["legend"] == {"0": '{"label":"none"}', "1": '{"label":"bad"}'}


def test_missing_answer_raises_instead_of_returning_none():
    questions = {"a": {"type": "noul", "instructions": "x", "criteria": None}}
    with pytest.raises(InferenceError):
        to_jev_answers({"other": {"type": "noul", "noul": 0.5}}, questions)


def test_choice_falls_back_to_argmax_when_inconsistent():
    # 模型给的 choice 不在它自己返回的概率表里 —— 不自洽。Jev 的语义是
    # choice = argmax，所以按概率重算，而不是把不存在的选项名透给下游。
    raw = {
        "a": {
            "type": "choice",
            "choice": "ghost",
            "probabilities": {"billing": 0.2, "technical": 0.8},
            "confidence": 0.5,
        }
    }
    questions = {
        "a": {"type": "choice", "instructions": "x", "criteria": {"billing": "", "technical": ""}}
    }
    assert to_jev_answers(raw, questions)["a"]["choice"] == "technical"


def test_choice_is_kept_when_consistent():
    raw = {
        "a": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.75, "technical": 0.25},
            "confidence": 0.5,
        }
    }
    questions = {
        "a": {"type": "choice", "instructions": "x", "criteria": {"billing": "", "technical": ""}}
    }
    assert to_jev_answers(raw, questions)["a"]["choice"] == "billing"


# ---------------------------------------------------------------------------
# 排队 / 过载
# ---------------------------------------------------------------------------


class SlowEchoBackend(EchoBackend):
    """每个请求睡 200ms，用来把并发闸门撑到超时。"""

    def infer(self, state, questions, model):
        time.sleep(0.2)
        return super().infer(state, questions, model)


def test_queue_timeout_returns_429():
    settings = make_settings(max_concurrency=1, queue_timeout_s=0.05)
    with TestClient(create_app(settings, SlowEchoBackend(settings))) as c:
        import threading

        first = {}

        def call():
            first["response"] = post(c)

        thread = threading.Thread(target=call)
        thread.start()
        time.sleep(0.05)
        second = post(c)
        thread.join()

        assert first["response"].status_code == 200
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "too_many_requests"
        assert second.headers["Retry-After"] == "1"
