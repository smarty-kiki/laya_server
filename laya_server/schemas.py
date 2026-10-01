"""TypeSafe / TypeSafe System One 兼容的请求 / 响应模型。

字段、取值域、错误语义都对着 https://docs.typesafe.ai/api 写，目标是
「按 Jev 写的客户端改一个 base_url 就能连上来」。三处刻意的取舍写在下面：

1. `instructions` / `criteria` 里的描述项按文档允许 string | object | array。
   这里原样透传给 laya-mlx，不做 flatten——laya 的 prompt 构造与上游一致，
   自己先拼成字符串会改变分词结果，进而改变概率。
2. `questions` 用可辨识联合（discriminator="type"）而不是 `Any`，
   这样 OpenAPI 里能看到三种问题的真实结构，客户端代码生成器才有东西可生成。
3. 响应模型照抄 Jev 的 `{model, answers, usage}`。任何额外信息只在
   `debug` 打开时以附加字段出现，默认逐字段兼容。
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# 基础别名
# ---------------------------------------------------------------------------

#: 被评估的内容。Jev 允许纯文本、结构化对象、数组三种形态。
State = Union[str, Dict[str, Any], List[Any]]

#: 问题正文。可以是一句话，也可以是「一个字段放问题、其他字段放数据」的对象。
Instructions = Union[str, Dict[str, Any], List[Any]]

#: Choice 的单个选项描述。文档明确允许 null（选项不需要额外说明时）。
CriteriaValue = Union[str, Dict[str, Any], List[Any], None]

#: Score 的单个等级描述。
ScoreLevel = Union[str, Dict[str, Any], List[Any]]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
MIN_QUESTIONS = 1
MAX_QUESTIONS = 255


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------


class ChoiceQuestion(_Base):
    """从你给定的选项里挑一个，返回选中项和完整概率分布。"""

    type: Literal["choice"]
    instructions: Instructions
    criteria: Dict[str, CriteriaValue]

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, value: Dict[str, CriteriaValue]) -> Dict[str, CriteriaValue]:
        if not value:
            raise ValueError("choice 的 criteria 至少要有一个选项")
        if len(value) > MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"choice 的选项上限是 {MAX_CHOICE_OPTIONS}，收到 {len(value)} 个。"
                " 选项数过多时应当先用 predict_shortlist 之类的方式收窄。"
            )
        return value


class ScoreQuestion(_Base):
    """按你给定的有序评分表打分，返回概率加权后的分值（可以落在两档之间）。"""

    type: Literal["score"]
    instructions: Instructions
    criteria: List[ScoreLevel]

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, value: List[ScoreLevel]) -> List[ScoreLevel]:
        if len(value) < MIN_SCORE_LEVELS:
            raise ValueError(f"score 至少要 {MIN_SCORE_LEVELS} 个等级")
        if len(value) > MAX_SCORE_LEVELS:
            raise ValueError(
                f"score 的等级上限是 {MAX_SCORE_LEVELS}，收到 {len(value)} 个"
            )
        return value


class NoulQuestion(_Base):
    """是非题，返回「是」的概率。criteria 可选，用来描述什么算「是」/「否」。"""

    type: Literal["noul"]
    instructions: Instructions
    criteria: Optional[Dict[str, Instructions]] = None

    @field_validator("criteria")
    @classmethod
    def _check_criteria(
        cls, value: Optional[Dict[str, Instructions]]
    ) -> Optional[Dict[str, Instructions]]:
        if value is None:
            return None
        unknown = set(value) - {"true", "false"}
        if unknown:
            raise ValueError(
                f"noul 的 criteria 只认 true / false 两个键，收到多余的：{sorted(unknown)}"
            )
        return value


Question = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------


class SystemOneRequest(_Base):
    """POST /v1/systemone 的请求体。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "state": "Hi, I've been trying to connect my Stripe account for 3 days "
                    "and the integration keeps failing. I'm losing sales. Please help ASAP.",
                    "model": "jev-latest",
                    "questions": {
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
                            "criteria": [
                                "Calm, just stating facts",
                                "Frustrated but civil",
                                "Very angry, strong language",
                            ],
                        },
                        "is_urgent": {
                            "type": "noul",
                            "instructions": "The message conveys urgency or time-sensitivity",
                        },
                    },
                }
            ]
        },
    )

    state: State = Field(
        description="被评估的内容。字符串、对象或数组，同一份 state 会被所有问题共用。"
    )
    model: str = Field(
        default="jev-latest",
        description="模型别名或本地 checkpoint id。默认 jev-latest，映射到本地 MLX 权重。",
    )
    questions: Dict[str, Question] = Field(
        description="问题映射，键由你起名，答案会以同样的键返回。",
    )

    @field_validator("questions")
    @classmethod
    def _check_questions(cls, value: Dict[str, Question]) -> Dict[str, Question]:
        if len(value) < MIN_QUESTIONS:
            raise ValueError("questions 不能为空")
        if len(value) > MAX_QUESTIONS:
            raise ValueError(
                f"单请求问题上限 {MAX_QUESTIONS}，收到 {len(value)} 个"
            )
        return value


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------


class NoulAnswer(_Base):
    type: Literal["noul"] = "noul"
    noul: float = Field(description="0 = 否，1 = 是。")


class ChoiceAnswer(_Base):
    type: Literal["choice"] = "choice"
    choice: str = Field(description="概率最高的选项。")
    probabilities: Dict[str, float] = Field(description="每个选项的概率，和为 1。")
    confidence: float = Field(description="由概率分布导出的确定度，0–1。")


class ScoreAnswer(_Base):
    type: Literal["score"] = "score"
    score: float = Field(description="跨等级的概率加权分值，可以落在两档之间。")
    legend: Dict[str, str] = Field(description="等级序号 → 等级描述。")
    probabilities: Dict[str, float] = Field(description="每个等级的概率，和为 1。")
    confidence: float = Field(description="由概率分布导出的确定度，0–1。")


Answer = Annotated[
    Union[ChoiceAnswer, ScoreAnswer, NoulAnswer],
    Field(discriminator="type"),
]


class Usage(_Base):
    input_tokens: int
    #: System One 不逐 token 解码，这里是 0。保留字段是为了让按 Jev 写的
    #: 计量代码不需要特判（少了这个键会 KeyError）。
    output_tokens: int = 0


class SystemOneResponse(_Base):
    """POST /v1/systemone 的响应体。"""

    model_config = ConfigDict(extra="allow")

    model: str = Field(description="实际服务的模型标识（本地为解析后的 checkpoint）。")
    answers: Dict[str, Answer] = Field(description="与请求 questions 同键的答案。")
    usage: Usage


class ErrorDetail(_Base):
    code: str
    message: str


class ErrorResponse(_Base):
    error: ErrorDetail


__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "ErrorResponse",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "Usage",
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
]
