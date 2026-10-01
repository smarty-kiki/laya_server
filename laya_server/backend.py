"""推理后端：模型注册表（别名解析 / 懒加载 / LRU）、并发闸门、答案归一化。

为什么自己管模型生命周期而不用 `laya_mlx.Router`：

1. Router 内部构造 Agent 时写死了参数（只透传 device/token/subfolder/dtype），
   `batch_size` / `compile` / `pad_to_multiple` / `cache_prompts` 这几个调优项
   根本传不进去。服务端恰恰需要 batch_size（问题多时决定分几次前向）。
2. Router 的 LRU 默认 max_loaded=1。服务端在「中英文交替」的流量下会**每个请求
   重载一次模型**（冷加载是秒级，语言检测是微秒级）——这是文档自己点名的坑。
   所以服务的正确姿势是常驻，而不是靠 LRU 换进换出。
3. 服务还需要「哪个 slot 现在驻留了哪些权重」这类状态，Router 只暴露 loaded 名字。

路由判定仍然复用 laya 自己的 `detect_language`（= lang.analyse），
因为「英文 checkpoint 读不了非拉丁文字」这条规则是模型特性，不是我们该自造的。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .config import AUTO_ALIAS, Settings, apply_hf_env


# ---------------------------------------------------------------------------
# 异常。每个对应一个明确的 HTTP 状态码，映射在 app.py 里，这里不带 HTTP 概念。
# ---------------------------------------------------------------------------


class BackendError(RuntimeError):
    """推理后端基类异常。"""


class UnknownModelError(BackendError):
    """请求里的 model 既不是已知别名，也不是可用的 checkpoint id。"""


class ModelLoadError(BackendError):
    """权重下载或构建失败。"""


class RequestError(BackendError):
    """请求本身让模型没法处理（例如选项多到塞不进 token 预算）→ 422。"""


class InferenceError(BackendError):
    """推理本身失败，或模型漏答了某个问题 → 500。"""


def _hf_cache_dir() -> Path:
    """Hugging Face 的 hub 缓存目录。

    自己按官方优先级读环境变量，而不是 `import huggingface_hub.constants` ——
    那个包在导入期就把 HF_ENDPOINT 冻成常量了，早导一秒就会让 `apply_hf_env`
    变成一句废话（见 config.apply_hf_env 的注释）。
    """
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path("~/.cache/huggingface").expanduser()
    return base / "hub"


#: 一个 Laya checkpoint 要能被加载，这几样缺一不可。
#: 和 laya_mlx.resolve_model 末尾那段完整性检查保持一致 —— 这里先查一遍是为了
#: 别把半截快照当成完整的，那样会得到一个语焉不详的 FileNotFoundError。
REQUIRED_CHECKPOINT_FILES = ("model.safetensors", "rl_agent_config.json", "encoder/config.json")


def local_snapshot(repo: str) -> Optional[Path]:
    """如果这个 HF repo 已经**完整**落在本地缓存里，返回它的 snapshot 路径。

    为什么要自己找一遍：`huggingface_hub.snapshot_download` 每次都会先去 HF 问
    「最新版本是哪个」，网络不通就整个加载失败 —— **哪怕权重一个字节都不缺**。
    实测过的表现是：三份权重大小分毫不差地躺在磁盘上，请求还是在 10.5 秒后
    返回 503 ProxyError。对一个以「本地推理」为卖点的服务来说这是不能接受的。

    直接把本地路径交给 laya_mlx 就能从根上绕开：它的 `resolve_model` 见到存在的
    路径就直接用，一个网络请求都不发。

    代价是**不会再去检查有没有新版本**。对按 repo id 固定的 checkpoint 来说这是
    好事（可复现）；要拉更新，用控制台的「加载」按钮配合 `hf_endpoint` 重下即可。
    """
    if "/" not in repo:
        return None
    repo_dir = _hf_cache_dir() / f"models--{repo.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    # 多个快照时取名字最大的那一个（HF 用 commit hash 命名，但排序只是为了稳定，
    # 不是真的在比版本新旧——同一个 repo 通常只有一个）。
    for candidate in sorted(snapshots.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        if all((candidate / name).is_file() for name in REQUIRED_CHECKPOINT_FILES):
            return candidate
    return None


def local_source(repo: str) -> Optional[Tuple[str, str]]:
    """权重能不能纯本地拿到。返回 `(路径, 来源说明)` 或 None。

    两种「本地」，都不需要联网：

      * `slots` 里直接写了一个本地目录 → 目录在就是有
      * Hugging Face 缓存里有完整快照 → 用快照路径

    这两种情况对 `laya_mlx.resolve_model` 来说都是「路径存在就直接用，
    不碰网络」。给不出路径才走在线下载。

    「本地有」和「能加载」在原实现里是两回事 —— 这里把它们合成一个判断，
    启动时的 `auto_load: local` 才有意义：只有真的能立刻读出来的才去读。
    """
    path = Path(repo).expanduser()
    if path.is_dir():
        return str(path), "本地目录"
    snapshot = local_snapshot(repo)
    if snapshot is not None:
        return str(snapshot), "本地缓存"
    return None


def describe_error(exc: BaseException) -> str:
    """把异常连同它的 cause 链一起写清楚。

    只取 `str(exc)` 会丢掉最有用的那一层。包装型异常（扩展加载失败、下载失败、
    HTTP 客户端的包装错误）外层文案往往很笼统 —— 比如 MLX 扩展初始化失败时外层
    只说 "Encountered an error while initializing the extension."，真正的原因
    （找不到 metallib、找不到 dylib）挂在 `__cause__` 上。打包排查时，这一层之差
    就是「一眼看出问题」和「完全不知道从哪下手」的区别。
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    seen = {id(exc)}
    cause = exc.__cause__ or exc.__context__
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        parts.append(f"（起因：{type(cause).__name__}: {cause}）")
        cause = cause.__cause__ or cause.__context__
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 答案归一化：把 laya 的输出裁剪成 Jev 的 answer 形状
# ---------------------------------------------------------------------------

def _as_text(value: Any) -> str:
    """Jev 的 legend 是 map<string, string>，但 criteria 允许对象/数组，这里统一成字符串。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _probability_map(raw: Any) -> Dict[str, float]:
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(raw))
    else:
        raise InferenceError(f"概率字段不是 map 也不是 list：{type(raw).__name__}")
    out: Dict[str, float] = {}
    for key, value in items:
        try:
            out[str(key)] = round(float(value), 4)
        except (TypeError, ValueError) as exc:
            raise InferenceError(f"概率值无法转成数字：{key}={value!r}") from exc
    return out


def to_jev_answers(
    raw_answers: Dict[str, Any],
    questions: Dict[str, Any],
    *,
    passthrough: bool = False,
) -> Dict[str, Any]:
    """把 laya 的 answers 裁剪成 Jev 的 answer 形状。

    laya 比 Jev 多两样东西：每个 answer 上的 `action.act_probability`，
    以及 noul answer 上的 `confidence`。默认丢掉，因为 Jev 的响应里没有——
    客户端只要多出来的字段就会被 `extra="forbid"` 之类的严格解析器拒掉。
    `passthrough=True`（debug 模式）时原地保留，方便排查。

    Score 的 `legend` 重新按请求里的 criteria 生成：laya 的 legend 直接回抄
    我们传进去的 criteria 原值，而 criteria 允许对象/数组，
    不转字符串就不是 Jev 文档里写的 map<string, string> 了。
    """
    answers: Dict[str, Any] = {}
    for qid, definition in questions.items():
        raw = raw_answers.get(qid)
        if not isinstance(raw, dict):
            # 静默漏答比报错更糟糕：下游会在几层之外拿到 undefined/None，
            # 然后在完全无关的地方炸掉。
            raise InferenceError(
                f"模型没有返回问题 {qid!r} 的答案"
                f"（响应里只有：{', '.join(map(str, raw_answers)) or '空'}）"
            )

        qtype = definition["type"]
        if qtype == "choice":
            probabilities = _probability_map(raw.get("probabilities"))
            if not probabilities:
                raise InferenceError(f"问题 {qid!r} 的 choice 答案没有概率分布")
            choice = raw.get("choice")
            if not isinstance(choice, str) or choice not in probabilities:
                # 只有两种情况能走到这里：模型没给 choice，或者给的选项不在它自己返回的
                # 概率表里 —— 两种都说明这对字段不自洽。Jev 的语义是 choice = argmax，
                # 所以按概率重算，保证 choice 一定落在 probabilities 的键里，
                # 而不是把一个不存在的选项名透给下游。自洽时一律保留模型的原值。
                choice = max(probabilities, key=lambda k: probabilities[k])
            answer: Dict[str, Any] = {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": round(float(raw.get("confidence") or 0.0), 4),
            }
        elif qtype == "score":
            probabilities = _probability_map(raw.get("probabilities"))
            levels = definition.get("criteria") or []
            legend = {
                str(i): _as_text(value) for i, value in enumerate(levels) if i < len(levels)
            }
            try:
                score = float(raw.get("score"))
            except (TypeError, ValueError) as exc:
                raise InferenceError(f"问题 {qid!r} 的 score 答案不可用：{raw.get('score')!r}") from exc
            answer = {
                "type": "score",
                "score": round(score, 4),
                "legend": legend,
                "probabilities": probabilities,
                "confidence": round(float(raw.get("confidence") or 0.0), 4),
            }
        elif qtype == "noul":
            value = raw.get("noul")
            if value is None:
                raise InferenceError(f"问题 {qid!r} 的 noul 答案缺失")
            answer = {"type": "noul", "noul": round(float(value), 4)}
        else:  # pragma: no cover - schema 已经挡住了
            raise InferenceError(f"未知问题类型：{qtype!r}")

        if passthrough:
            for key, value in raw.items():
                answer.setdefault(key, value)
        answers[qid] = answer

    return answers


# ---------------------------------------------------------------------------
# 后端协议
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """服务只依赖这个接口，不关心底下是 MLX 还是别的。"""

    def infer(
        self, state: Any, questions: Dict[str, Any], model: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """返回 (原始 laya 响应, 路由信息或 None)。"""

    def status(self) -> Dict[str, Any]: ...

    def preload(self, only_local: bool = False) -> List[str]:
        """加载槽位权重，返回成功的 repo 列表。

        `only_local=True` 表示「只加载本地已有的」，不产生任何网络流量 ——
        启动时的自动加载用的就是这个模式。
        """

    def unload(self, model: Optional[str] = None) -> None: ...

    def load(self, model: str) -> str:
        """加载**一个** checkpoint，返回解析后的 repo id。

        这是「不靠发推理请求，也能把权重拉下来」的入口：它和第一个请求走的是同一条
        路径（`_agent()` → `laya.load()` → snapshot_download），区别只是不需要编造
        一个假 state 去骗过 schema。控制台上的「加载」按钮就是调它。
        """

    def trim(self) -> None:
        """按当前的 max_loaded 立即淘汰。控制台把常驻上限调小时用。"""


# ---------------------------------------------------------------------------
# MLX 后端
# ---------------------------------------------------------------------------


class MlxBackend:
    """本地 laya-mlx 推理。模型按需下载、按需构建。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        #: repo_id → Agent。OrderedDict 用来记 LRU 顺序（最近用过的在末尾）。
        self._agents: "OrderedDict[str, Any]" = OrderedDict()
        #: 保护 _agents 这张表本身。**不覆盖推理**——推理只在 app 层的信号量里串行。
        self._registry_lock = threading.RLock()
        #: 每个 repo 一把锁，避免并发首请求把同一份权重构建两遍。
        self._build_locks: Dict[str, threading.Lock] = {}
        #: repo_id → 上次加载失败的说明。失败不缓存 Agent，但记住原因好让 503 有话说。
        self._errors: Dict[str, str] = {}
        self._laya = None  # 延迟 import，让不需要 MLX 的路径（测试 / echo 模式）能跑

    # ------------------------------------------------------------ 惰性依赖
    @property
    def laya(self):
        if self._laya is None:
            # 关键顺序：HF_ENDPOINT / HF_HUB_OFFLINE 必须在导入 laya_mlx（进而
            # 导入 huggingface_hub）**之前**写进环境变量。huggingface_hub 在导入期
            # 就把 endpoint 读成了模块级常量，晚一步设置就完全无效。
            apply_hf_env(self.settings)
            try:
                import laya_mlx
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise ModelLoadError(
                    "没装 laya-mlx。pip install 'laya-mlx>=0.2.0' 后再启动。"
                ) from exc
            self._laya = laya_mlx
        return self._laya

    @property
    def package_version(self) -> str:
        return str(getattr(self.laya, "__version__", "unknown"))

    # ------------------------------------------------------------ 解析
    def resolve(self, model: Optional[str]) -> Tuple[str, str]:
        """model 取值 → (kind, target)。

        kind 是 `slot` | `repo` | `auto`。三种来源的优先级写在 config 的注释里：
        填 Jev 别名走 slot，填 `auto` 交给语言路由，填 `org/name` 直接当权重路径。
        """
        requested = (model or self.settings.default_model).strip()
        if not requested:
            requested = self.settings.default_model

        if requested.lower() == AUTO_ALIAS:
            return "auto", ""

        slot = self.settings.aliases.get(requested)
        if slot is None:
            slot = self.settings.aliases.get(requested.lower())
        if slot is not None:
            return "slot", slot

        if "/" in requested and self.settings.allow_raw_checkpoint:
            return "repo", requested

        raise UnknownModelError(
            f"未知模型 {requested!r}。可用别名：{', '.join(sorted(self.settings.aliases))}"
            f"，或 {'、'.join(self.settings.slots)} 槽位，或直接给 checkpoint id（形如 org/name）。"
        )

    def slot_repo(self, slot: str) -> str:
        try:
            return self.settings.slots[slot]
        except KeyError as exc:
            raise UnknownModelError(f"未配置的路由槽位 {slot!r}") from exc

    def route(
        self, state: Any, requested: str
    ) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """返回 (repo_id, slot_name 或 None, 路由理由)。

        路由规则直接照搬 laya 自己的判定（`detect_language`）：拉丁字符且英文 →
        英文 checkpoint；其他情况 → 多语言 checkpoint。理由字符串会原样进 debug 块，
        因为「这条为什么走了多语言」是排查准确率问题时第一个要问的东西。
        """
        kind, target = self.resolve(requested)

        if kind == "repo":
            return target, None, {"reason": f"explicit checkpoint {target!r}"}

        if kind == "slot":
            repo = self.slot_repo(target)
            return repo, target, {"reason": f"alias {requested!r} → slot {target!r}"}

        detection = self.laya.detect_language(state)
        slot = "english" if detection.get("is_english") else "multilingual"
        if slot not in self.settings.slots:
            slot = self.settings.default_slot
        reason = self._route_reason(detection, slot)
        return self.slot_repo(slot), slot, {"reason": reason, "detection": detection}

    @staticmethod
    def _route_reason(detection: Dict[str, Any], slot: str) -> str:
        script = detection.get("script")
        if script == "unknown":
            return f"state 里没有字母，回退到 {slot!r}"
        if script != "latin":
            return (
                f"非拉丁文字（{script}，占字母 {100 * float(detection.get('non_latin_fraction') or 0):.0f}%），"
                "英文 checkpoint 读不了"
            )
        if not detection.get("is_english"):
            lang = detection.get("language")
            if lang:
                return f"拉丁文字但语言是 {lang!r}，不是英文"
            return (
                f"拉丁文字、语言未能识别，但有 {100 * float(detection.get('diacritic_rate') or 0):.0f}% 非英文字母"
            )
        return "英文拉丁文本"

    # ------------------------------------------------------------ 模型生命周期
    def _agent(self, repo: str) -> Any:
        with self._registry_lock:
            agent = self._agents.get(repo)
            if agent is not None:
                self._agents.move_to_end(repo)
                return agent
            build_lock = self._build_locks.setdefault(repo, threading.Lock())

        # 构建放在 registry 锁之外，否则一个慢加载会把其他 repo 的命中路径也堵住。
        with build_lock:
            with self._registry_lock:  # 双重检查：等锁期间别人可能已经建好了
                agent = self._agents.get(repo)
                if agent is not None:
                    self._agents.move_to_end(repo)
                    return agent

            settings = self.settings
            started = time.perf_counter()
            # 本地有就直接用本地路径，完全不碰网络。没有才交给 laya 去下。
            source = local_source(repo)
            target = source[0] if source else repo
            try:
                agent = self.laya.load(
                    target,
                    dtype=settings.dtype,
                    device=settings.device,
                    batch_size=settings.batch_size,
                    compile=settings.compile,
                    pad_to_multiple=settings.pad_to_multiple,
                    cache_prompts=settings.cache_prompts,
                )
            except Exception as exc:  # noqa: BLE001 —— 任何加载失败都要变成可读的 503
                message = describe_error(exc)
                if source is None:
                    message += (
                        f"（本地没有 {repo} 的完整权重，这次是去 Hugging Face 下的；"
                        "网络不通就先手动下，或把 hf_endpoint 指到镜像）"
                    )
                with self._registry_lock:
                    self._errors[repo] = message
                raise ModelLoadError(f"加载 {repo} 失败：{message}") from exc

            with self._registry_lock:
                self._agents[repo] = agent
                self._errors.pop(repo, None)
                self._evict_locked()
            elapsed = time.perf_counter() - started
            print(
                f"[laya-server] loaded {repo} from {source[1] if source else 'Hugging Face'}"
                f" in {elapsed:.2f}s",
                flush=True,
            )
            return agent

    def _evict_locked(self) -> None:
        """把超额的 Agent 移出表。

        正在推理的 Agent 被移出是安全的：调用栈里还持有它的强引用，
        Python 的引用计数会把它留到那次前向结束，之后才真正释放 MLX 缓冲。
        所以这里不需要「等推理结束再淘汰」那一套。
        """
        limit = max(1, int(self.settings.max_loaded))
        while len(self._agents) > limit:
            victim, _ = self._agents.popitem(last=False)
            print(f"[laya-server] evicted {victim} (max_loaded={limit})", flush=True)

    def unload(self, model: Optional[str] = None) -> None:
        with self._registry_lock:
            if model is None:
                self._agents.clear()
                return
            kind, target = self.resolve(model)
            repo = self.slot_repo(target) if kind == "slot" else target
            self._agents.pop(repo, None)

    def preload(self, only_local: bool = False) -> List[str]:
        """把槽位权重提前拉起来，返回真正加载成功的 repo 列表。

        `only_local=True` 时**只处理本地已经有完整权重的槽位**，一个网络请求都不发。
        这是启动时的默认行为：既让首个请求不必吃冷加载，又不会让「双击打开应用」
        变成「悄悄下载 2GB 流量」。

        单个槽位失败不影响其他槽位 —— 全部收集起来返回，原因留在 `load_errors`
        里给控制台展示。直接抛出去只会让「三个里有两个已经好了」这件事看不见。
        """
        loaded: List[str] = []
        for slot, repo in self.settings.slots.items():
            if only_local and local_source(repo) is None:
                print(f"[laya-server] 跳过 {slot}：{repo} 本地没有完整权重", flush=True)
                continue
            try:
                self._agent(repo)
            except BackendError as exc:
                print(f"[laya-server] 加载 {slot} 失败：{exc}", flush=True)
                continue
            loaded.append(repo)
        return loaded

    def trim(self) -> None:
        with self._registry_lock:
            self._evict_locked()

    def load(self, model: str) -> str:
        kind, target = self.resolve(model)
        repo = self.slot_repo(target) if kind == "slot" else target
        # auto 在这里没有意义：它要读 state 才能决定用哪个槽位，而这里没有 state。
        # 与其随便挑一个，不如让调用方把模型说清楚。
        if kind == "auto":
            raise UnknownModelError(
                "加载模型时不能填 auto —— 语言路由需要具体的一份 state 才能决定，"
                "这里没有。请指定具体别名或 checkpoint id。"
            )
        self._agent(repo)  # 首次会走 HF（或镜像）下载，并落进本地缓存
        return repo

    # ------------------------------------------------------------ 推理
    def infer(
        self, state: Any, questions: Dict[str, Any], model: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        repo, slot, routing = self.route(state, model)
        agent = self._agent(repo)
        try:
            raw = agent.predict(state, questions)
        except ValueError as exc:
            # laya 的 _to_internal 会为「选项太多塞不进 token 预算」这类结构问题抛 ValueError，
            # 那是请求的问题（422），不是服务的问题（500）。
            raise RequestError(str(exc)) from exc
        except FloatingPointError as exc:
            # laya 自己给的提示是换 float32。
            raise InferenceError(
                f"{exc}（建议把 dtype 改成 float32 再试）"
            ) from exc

        raw["model"] = repo
        if routing is not None:
            routing = dict(routing)
            routing["slot"] = slot
            routing["checkpoint"] = repo
        return raw, routing

    def status(self) -> Dict[str, Any]:
        with self._registry_lock:
            loaded = list(self._agents)
            errors = dict(self._errors)
        return {
            "backend": "mlx",
            "package": f"laya-mlx {self.package_version}",
            "loaded": loaded,
            "load_errors": errors,
        }


# ---------------------------------------------------------------------------
# Echo 后端：不加载任何权重，只回可预测的假答案。
# 用途是给客户端联调 / CI 契约测试，绝不能当生产后端——见 app.py 的启动告警。
# ---------------------------------------------------------------------------


def _fake_distribution(count: int, index: int, top: float = 0.6) -> List[float]:
    """给 echo 后端造一个「看起来像」的概率分布：指定项拿 top，其余均分，四舍五入后归一。

    最后那一步补差是必要的：舍入到 4 位之后和常常是 0.9999，客户端拿 `sum == 1`
    当断言就会莫名其妙地红。
    """
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    rest = (1.0 - top) / (count - 1)
    values = [round(top if i == index else rest, 4) for i in range(count)]
    drift = round(1.0 - sum(values), 4)
    if drift:
        values[index] = round(values[index] + drift, 4)
    return values


class EchoBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def resolve(self, model: Optional[str]) -> Tuple[str, str]:
        requested = (model or self.settings.default_model).strip()
        if requested.lower() == AUTO_ALIAS:
            return "auto", ""
        slot = self.settings.aliases.get(requested.lower())
        if slot is not None:
            return "slot", slot
        if "/" in requested and self.settings.allow_raw_checkpoint:
            return "repo", requested
        raise UnknownModelError(f"未知模型 {requested!r}")

    def infer(
        self, state: Any, questions: Dict[str, Any], model: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        kind, target = self.resolve(model)
        if kind == "slot":
            repo = self.settings.slots[target]
        elif kind == "repo":
            repo = target
        else:
            repo = self.settings.slots[self.settings.default_slot]

        def bucket(text: str) -> int:
            return sum(text.encode("utf-8")) + len(text)

        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        answers: Dict[str, Any] = {}
        for qid, definition in questions.items():
            seed = bucket(qid + state_text)
            qtype = definition["type"]
            if qtype == "choice":
                labels = list(definition["criteria"])
                index = seed % len(labels)
                values = _fake_distribution(len(labels), index)
                answers[qid] = {
                    "type": "choice",
                    "choice": labels[index],
                    "probabilities": {label: value for label, value in zip(labels, values)},
                    "confidence": 0.6,
                    "action": {"act_probability": 0.5},
                }
            elif qtype == "score":
                levels = list(definition["criteria"])
                index = seed % len(levels)
                values = _fake_distribution(len(levels), index)
                answers[qid] = {
                    "type": "score",
                    "score": float(index),
                    "legend": {str(i): _as_text(v) for i, v in enumerate(levels)},
                    "probabilities": {str(i): value for i, value in enumerate(values)},
                    "confidence": 0.6,
                    "action": {"act_probability": 0.5},
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round((seed % 100) / 100.0, 4),
                    "confidence": 0.6,
                    "action": {"act_probability": 0.5},
                }

        tokens = len(state_text) // 4 + sum(len(qid) for qid in questions)
        raw = {
            "model": repo,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
        routing = None
        if kind == "auto":
            routing = {"reason": "echo backend 不做语言检测", "slot": None, "checkpoint": repo}
        return raw, routing

    def status(self) -> Dict[str, Any]:
        return {"backend": "echo", "package": "n/a", "loaded": [], "load_errors": {}}

    def preload(self, only_local: bool = False) -> List[str]:  # pragma: no cover
        # 没有权重可加载。返回空列表而不是 None，调用方不用区分两种后端。
        return []

    def trim(self) -> None:  # pragma: no cover
        return None

    def load(self, model: str) -> str:
        kind, target = self.resolve(model)
        if kind == "auto":
            raise UnknownModelError("echo 后端不支持 auto")
        if kind == "slot":
            return self.settings.slots[target]
        return target

    def unload(self, model: Optional[str] = None) -> None:  # pragma: no cover
        return None


def build_backend(settings: Settings) -> Backend:
    if settings.backend == "echo":
        return EchoBackend(settings)
    return MlxBackend(settings)


__all__ = [
    "Backend",
    "BackendError",
    "EchoBackend",
    "InferenceError",
    "MlxBackend",
    "ModelLoadError",
    "RequestError",
    "REQUIRED_CHECKPOINT_FILES",
    "UnknownModelError",
    "build_backend",
    "describe_error",
    "local_snapshot",
    "local_source",
    "to_jev_answers",
]
