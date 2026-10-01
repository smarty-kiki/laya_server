"""与 laya-mlx 的接缝测试：不需要权重，但需要真的装上 laya-mlx。

这些断言的价值在于**跨包边界**：我的 pydantic 模型吐出来的字典，是不是 laya
的 `_to_internal` / `render_options` 真正接受的东西。这类接缝靠桩测不出来 ——
桩只证明我自洽，证明不了我和对方一致。

顺带钉死两个容易在重构里被悄悄改坏的地方：
  * choice 的 `null` 描述必须保留成选项（前端一行 exclude_none 就能把它删掉）；
  * noul 的选项顺序必须是 [false, true]，因为 laya 取 `p[1]` 当「是」的概率。
"""

from __future__ import annotations

import pytest

laya = pytest.importorskip("laya_mlx", reason="需要装 laya-mlx 才能跑接缝测试")

from laya_mlx.agent import Agent  # noqa: E402
from laya_mlx.common import render_options  # noqa: E402

from laya_server.schemas import SystemOneRequest  # noqa: E402

PAYLOAD = {
    "state": "I was billed twice. Please refund the duplicate.",
    "model": "jev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs and outages",
                "sales": "new purchases",
            },
        },
        "route": {
            "type": "choice",
            "instructions": "Which queue?",
            "criteria": {"fast": None, "cheap": None},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "critical"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
            "criteria": {"true": "asks for money back", "false": "does not"},
        },
        "bare_noul": {"type": "noul", "instructions": "Is this a bug report?"},
        "structured": {
            "type": "noul",
            "instructions": {"duplicate": {"amount": 20}, "question": "Is this a duplicate?"},
        },
    },
}


@pytest.fixture()
def questions() -> dict:
    """走完整条链：HTTP 请求体 → 校验 → 交给 laya 的字典。"""
    request = SystemOneRequest.model_validate(PAYLOAD)
    return {qid: definition.model_dump() for qid, definition in request.questions.items()}


def test_every_question_is_accepted_by_laya(questions):
    for qid, definition in questions.items():
        Agent._to_internal(definition)  # 不抛异常即通过


def test_choice_with_null_criterion_keeps_the_option(questions):
    internal = Agent._to_internal(questions["route"])
    options = render_options(internal)
    # 两个选项都要在，且顺序与 criteria 的插入顺序一致 —— laya 是按这个顺序
    # zip 出 probabilities 的，顺序错了 label 就跟着错。
    assert options == ["fast", "cheap"]
    assert list(internal["crit"]) == ["fast", "cheap"]


def test_choice_option_order_matches_criteria_order(questions):
    internal = Agent._to_internal(questions["department"])
    assert render_options(internal) == [
        "billing: invoices, payments, refunds",
        "technical: bugs and outages",
        "sales: new purchases",
    ]


def test_noul_options_are_false_then_true(questions):
    for qid in ("refund", "bare_noul"):
        options = render_options(Agent._to_internal(questions[qid]))
        # laya 的实现是 noul = p[1]，「是」必须落在下标 1。
        assert options[0].startswith("false:")
        assert options[1].startswith("true:")


def test_score_levels_are_indexed_in_order(questions):
    internal = Agent._to_internal(questions["urgency"])
    assert render_options(internal) == [
        "level 0: not urgent",
        "level 1: soon",
        "level 2: critical",
    ]
    # Jev 的 legend 是 "0"/"1"/"2"，与这里的下标必须一一对应。
    assert internal["crit"] == ["not urgent", "soon", "critical"]


def test_structured_instructions_are_serialised(questions):
    internal = Agent._to_internal(questions["structured"])
    # 对象型 instructions 由 laya 自己 json.dumps；我们不要抢着先拼成字符串，
    # 不然分隔符和转义就由我们决定，分词结果也跟着变。
    assert isinstance(internal["ins"], str)
    assert "Is this a duplicate?" in internal["ins"]


def test_laya_rejects_what_our_schema_already_rejected(questions):
    """反向确认：schema 挡掉的形状，laya 自己也确实不接受。

    如果 laya 其实能接受而我们挡了，那就是把可用的能力误伤了。
    """
    with pytest.raises(ValueError):
        Agent._to_internal({"type": "choice", "instructions": "x", "criteria": {}})
    with pytest.raises(ValueError):
        Agent._to_internal({"type": "sentiment", "instructions": "x"})
