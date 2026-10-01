"""服务配置。环境变量 + 可选 JSON/YAML 配置文件，环境变量优先级更高。

所有可调项集中在这里：改模型别名表、并发、超时都不用碰服务代码。
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

ENV_PREFIX = "LAYA_SERVER_"

#: 用户级配置目录。控制台里改的配置写到这里，而不是写回工作目录的 server.json ——
#: 前者是「这台机器的偏好」，后者通常是仓库里的文件，跟着 git 走。
#: 用 LAYA_SERVER_HOME 可以整体挪走（打包成 .app 后会指向 Application Support）。
HOME_DIR = Path(os.environ.get(f"{ENV_PREFIX}HOME") or "~/.laya-server").expanduser()
CONFIG_FILENAME = "config.json"

#: 改了**立刻生效**的项。其余项写进配置、下次启动才生效，控制台会明确标出来。
#: 划分依据只有一条：这个值是在「每次请求」时读的，还是启动时读一次就固化的。
HOT_FIELDS = frozenset(
    {
        "api_key",
        "debug",
        "default_model",
        "max_concurrency",
        "queue_timeout_s",
        "max_state_chars",
        "max_questions",
        "max_loaded",
        "aliases",
        "slots",
        "allow_raw_checkpoint",
        "admin_local_only",
    }
)

#: 只在启动时读一次、改完必须重启的项。
COLD_FIELDS = frozenset(
    {
        "host",
        "port",
        "backend",
        "dtype",
        "device",
        "batch_size",
        "compile",
        "pad_to_multiple",
        "cache_prompts",
        "auto_load",
        "docs",
        "console",
        "stats_capacity",
        "hf_endpoint",
        "hf_offline",
    }
)

#: 控制台允许改的字段。没列进来的（比如 HOME_DIR）只能走环境变量。
EDITABLE_FIELDS = HOT_FIELDS | COLD_FIELDS

# ---------------------------------------------------------------------------
# 路由槽位 → 本地权重
#
# 三个槽位名沿用 laya 上游的叫法（english / multilingual / typed-decisions），
# 换成别的名字只会让「哪份权重读不了中文」这件事更难查。
#
# 用 HF 上预转换好的 FP16 checkpoint，而不是原始 convaiinnovations/*：
# 原始仓库要现场转换，首次启动会多花几十秒且更容易失败。
# ---------------------------------------------------------------------------
DEFAULT_SLOTS: Dict[str, str] = {
    "english": "aac6fef/laya-mlx",
    "multilingual": "aac6fef/laya-multilingual-mlx",
    "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
}

# ---------------------------------------------------------------------------
# 别名 → 槽位
#
# Jev 的 `model` 字段填的是 `jev-latest` / `jev-1.13.0` 这类别名。为了让按 Jev
# 写的客户端改个 base_url 就能连上来，这里把同一批别名指到本地槽位。
#
# 三个必须说清的事实：
#   1. 这是**别名 → 本地权重**的映射，不是"同一个模型"。本地跑的是 Laya 的 MLX
#      移植，与 TypeSafe 云端的 Jev 不是同一份权重，概率不会逐位相同，
#      按 Jev 标定过的 confidence 阈值必须重新标定。
#   2. `jev-preview` 没有本地对应物，保守地指向与 `jev-latest` 相同的槽位。
#      指向一个并不存在的"更新版本"比指向稳定版更容易在生产里咬人。
#   3. Jev 的训练主语言是英文，中日韩文准确率低；本地同理，中文流量请走
#      `laya-multilingual` 或 `model: auto`。
# ---------------------------------------------------------------------------
DEFAULT_ALIASES: Dict[str, str] = {
    "jev-latest": "english",
    "jev-1.13.0": "english",
    "jev-preview": "english",
    "laya": "english",
    "laya-mlx": "english",
    "english": "english",
    "laya-multilingual": "multilingual",
    "laya-multilingual-mlx": "multilingual",
    "multilingual": "multilingual",
    "laya-typed-decisions": "typed-decisions",
    "laya-typed-decisions-mlx": "typed-decisions",
    "typed-decisions": "typed-decisions",
}

DEFAULT_ALIAS = "jev-latest"

# 语言路由哨兵：model 填这个值时按 state 语言自动挑槽位。
AUTO_ALIAS = "auto"

#: `auto_load` 的三档取值。默认 `local` —— 这是「打开就能用」和「不要偷偷下 2GB」
#: 之间的那个平衡点。
AUTO_LOAD_MODES = ("off", "local", "all")


def _normalise_auto_load(value: Any) -> str:
    """把 `auto_load` 的各种写法收敛成三档之一。

    接受布尔值和 0/1/on/off 这些写法，是因为这个字段以前叫 `preload`（布尔），
    用户的配置文件和环境变量里可能还留着老值。让 `LAYA_SERVER_AUTO_LOAD=1`
    继续能工作，比让它在启动时报一句「只能是 off/local/all」友好得多。
    """
    if isinstance(value, bool):
        return "all" if value else "off"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "all"}:
        return "all"
    if text in {"0", "false", "no", "off", "none", ""}:
        return "off"
    return text  # local（或拼错的词，交给 resolved 里统一报错）


@dataclass(slots=True)
class Settings:
    # --- 网络 ---
    host: str = "127.0.0.1"
    port: int = 8077

    # --- 鉴权。为空表示不校验（本机服务默认）。---
    api_key: Optional[str] = None

    # --- 模型 ---
    slots: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SLOTS))
    aliases: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    default_model: str = DEFAULT_ALIAS
    #: 推理后端。mlx = 真实本地推理；echo = 不加载权重的假后端，仅供联调与测试。
    backend: str = "mlx"
    # 允许请求直接传 checkpoint id（形如 `org/name`）而不走别名表。
    allow_raw_checkpoint: bool = True
    # 常驻的权重数量上限。默认 3 = 三个槽位全驻留（约 1.16B 参数 / fp16 约 2.3GB），
    # 因为服务端最怕的是「中英文交替流量触发 LRU 反复重载」——冷加载是秒级的。
    # 内存吃紧就降到 1，代价是切语言时重载。
    max_loaded: int = 3
    # 是否在启动时预加载 default_model 所属槽位（避免首个请求吃几秒加载延迟）。
    #: 启动时自动加载哪些权重：
    #:
    #:   `off`   —— 什么都不加载，全部等第一个请求（最省内存）
    #:   `local` —— 只加载**本地已经有完整权重**的槽位，一个网络请求都不发（默认）
    #:   `all`   —— 三个槽位全加载，本地没有就去 Hugging Face 下（等于旧的 preload）
    #:
    #: 加载发生在后台线程里，不挡启动：控制台立刻能打开，模型状态会自己长出来。
    auto_load: str = "local"

    # --- 推理 ---
    dtype: str = "float16"
    device: Optional[str] = None  # None → MLX 默认设备
    batch_size: int = 16
    compile: bool = False
    pad_to_multiple: Optional[int] = None
    cache_prompts: bool = False

    # --- 权重下载 ---
    #: Hugging Face 端点。国内直连 huggingface.co 通常不通，
    #: 填 `https://hf-mirror.com` 即可（镜像与官方同构，只是换了域名）。
    #: 留空则用 HF_ENDPOINT 环境变量，再没有就走官方。
    hf_endpoint: Optional[str] = None
    #: 只用本地缓存，绝不联网。权重已经下过、或者机器完全离线时打开，
    #: 否则每次启动都要为「检查有没有新版本」付一次网络超时的代价。
    hf_offline: bool = False

    # --- 服务行为 ---
    # 同时进行的推理数。MLX 是单设备，>1 只对 IO 等待有意义，默认串行最稳。
    max_concurrency: int = 1
    # 等推理槽位的上限。超时返回 429 + Retry-After，对应 Jev 的 Too Many Requests。
    queue_timeout_s: float = 120.0
    # state 字符上限，0 = 不限制（checkpoint 自身的 512/1024 context 才是硬约束）。
    max_state_chars: int = 0
    # 单请求问题数上限。Jev 文档给的 Choice 选项上限是 255，问题条数未公开，
    # 这里沿用 255，超了直接 422 而不是让 checkpoint 静默截断。
    max_questions: int = 255
    # 是否允许响应里多出 Jev 没有的字段（debug / 错误详情）。默认关闭，保持逐字段兼容。
    debug: bool = False
    # 是否导出 OpenAPI 文档（/docs、/openapi.json）。
    docs: bool = True
    # 控制台在 `/` 上托管，管理接口在 `/admin/*`。
    console: bool = True
    # 管理接口是否只接受本机来源。默认 True —— 见 admin.py 里的说明。
    admin_local_only: bool = True
    # 请求明细的保留条数（环形缓冲）。
    stats_capacity: int = 1000

    @property
    def default_slot(self) -> str:
        """default_model 指向的槽位；别名没配或配歪了就地回退，别让服务起不来。"""
        return self.aliases.get(self.default_model) or next(iter(self.slots))

    @property
    def resolved(self) -> "Settings":
        """补齐派生值并做一次启动前自检——配置错在这里报，比在首个请求里报好得多。"""
        if not self.slots:
            self.slots = dict(DEFAULT_SLOTS)
        if not self.aliases:
            self.aliases = dict(DEFAULT_ALIASES)

        unknown = {alias: slot for alias, slot in self.aliases.items() if slot not in self.slots}
        if unknown:
            raise ValueError(
                f"aliases 指向了不存在的槽位：{unknown}；已定义的槽位是 {sorted(self.slots)}"
            )
        if self.default_model not in self.aliases:
            default_slot = self.default_slot
            self.aliases[self.default_model] = default_slot
        if self.backend not in {"mlx", "echo"}:
            raise ValueError(f"backend 只能是 mlx 或 echo，收到 {self.backend!r}")
        self.auto_load = _normalise_auto_load(self.auto_load)
        if self.auto_load not in AUTO_LOAD_MODES:
            raise ValueError(
                f"auto_load 只能是 {'/'.join(AUTO_LOAD_MODES)}，收到 {self.auto_load!r}"
            )
        return self


def _coerce(raw: str) -> Any:
    """把环境变量字符串转成 bool / int / float / JSON，转不动就当字符串。"""
    low = raw.strip().lower()
    if low in {"1", "true", "yes", "on"}:
        return True
    if low in {"0", "false", "no", "off"}:
        return False
    if low in {"none", "null", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.strip().startswith(("{", "[")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


# 环境变量名 → Settings 字段名。只映射基础类型字段；aliases 走配置文件。
_ENV_FIELDS = {
    "HOST": "host",
    "PORT": "port",
    "API_KEY": "api_key",
    "DEFAULT_MODEL": "default_model",
    "BACKEND": "backend",
    "ALLOW_RAW_CHECKPOINT": "allow_raw_checkpoint",
    "MAX_LOADED": "max_loaded",
    "AUTO_LOAD": "auto_load",
    "DTYPE": "dtype",
    "DEVICE": "device",
    "BATCH_SIZE": "batch_size",
    "COMPILE": "compile",
    "PAD_TO_MULTIPLE": "pad_to_multiple",
    "CACHE_PROMPTS": "cache_prompts",
    "MAX_CONCURRENCY": "max_concurrency",
    "QUEUE_TIMEOUT_S": "queue_timeout_s",
    "MAX_STATE_CHARS": "max_state_chars",
    "MAX_QUESTIONS": "max_questions",
    "DEBUG": "debug",
    "DOCS": "docs",
}

#: 走 Hugging Face **自己的**环境变量名，不加 LAYA_SERVER_ 前缀。
#: 照抄官方约定能让「我明明设了 HF_ENDPOINT，为什么没用」这类问题少一半 ——
#: 用户已经在别处设过这两个变量了，我们没道理要求他再学一套。
_HF_ENV_FIELDS = {
    "HF_ENDPOINT": "hf_endpoint",
    "HF_HUB_OFFLINE": "hf_offline",
}


#: 已废弃的字段名 → 新写法。见 `_migrate_legacy`。
_LEGACY_FIELDS = {"preload": "auto_load"}


def _migrate_legacy(data: Dict[str, Any]) -> Dict[str, Any]:
    """把废弃的字段名翻译成新的，而不是报「未知字段」。

    `preload: bool` 在老版本里表示「启动时把所有槽位拉起来」，现在换成了三档的
    `auto_load`，语义是它的超集，所以老配置能无损升级：

      * `preload: true`  → `auto_load: "all"`（确实是「全都要」的意思）
      * `preload: false` → 删掉这一项。false 在老版本里**本来就是默认值**，
        也就是说它没携带任何信息；删掉后落到新的默认 `local` 上 —— 本地有就
        加载，本地没有也绝不会因此产生流量。

    为什么不直接报错：升级之后「配置文件突然不被认了、服务起不来」是那种
    完全不知道该从哪查起的失败。迁移一个键的成本几乎为零。
    """
    legacy = {old: new for old, new in _LEGACY_FIELDS.items() if old in data}
    if not legacy:
        return data
    migrated = {key: value for key, value in data.items() if key not in legacy}
    for old, new in legacy.items():
        if new in migrated:
            continue  # 新字段已经写了，以它为准
        value = data[old]
        if value is False or value is None:
            # 丢掉，而不是翻译成 "off"。false 在老版本里就是默认值，没携带任何信息
            # ——多半只是有人照抄了示例配置。翻成 "off" 会把「他什么也没说」变成
            # 「他明确要求不要加载」，那是两回事。
            continue
        migrated[new] = "all" if value is True else value
    return migrated


def _load_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                f"{path} 是 YAML，但没装 PyYAML。改用 .json，或 pip install pyyaml。"
            ) from exc
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是对象：{path}")
    # 下划线开头的键当注释用（JSON 没有注释语法，但总得让人写点什么）。
    data = {key: value for key, value in data.items() if not str(key).startswith("_")}
    data = _migrate_legacy(data)
    known = set(Settings.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"配置文件里有未知字段：{sorted(unknown)}")
    return data


def config_path(config_path: Optional[str] = None) -> Optional[Path]:
    """决定这次启动读哪个配置文件。命令行 > 环境变量 > 工作目录 > 用户目录。"""
    if config_path:
        return Path(config_path).expanduser()
    env_path = os.environ.get(f"{ENV_PREFIX}CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    local = Path.cwd() / "server.json"
    if local.is_file():
        return local
    home = HOME_DIR / CONFIG_FILENAME
    return home if home.is_file() else None


def load_settings(path: Optional[str] = None) -> Settings:
    """读配置文件 → 叠加环境变量。"""
    settings = Settings()

    selected = config_path(path)
    if selected is not None:
        settings = replace(settings, **_load_file(selected))

    updates: Dict[str, Any] = {}
    for env_name, field_name in _ENV_FIELDS.items():
        value = os.environ.get(f"{ENV_PREFIX}{env_name}")
        if value is not None:
            updates[field_name] = _coerce(value)
    for env_name, field_name in _HF_ENV_FIELDS.items():
        # 官名优先，LAYA_SERVER_ 前缀的别名可以覆盖它（便于只给本服务开镜像）。
        for key in (env_name, f"{ENV_PREFIX}{env_name}"):
            value = os.environ.get(key)
            if value is not None:
                updates[field_name] = _coerce(value)

    # 老变量名 LAYA_SERVER_PRELOAD 继续有效（= auto_load: all）。它可能已经写在
    # 某个启动脚本或 launchd plist 里了 —— 让老变量**静默失效**比什么都糟：
    # 用户会以为「预加载还开着」，实际每次启动都在等第一次冷加载。
    legacy_preload = os.environ.get(f"{ENV_PREFIX}PRELOAD")
    if legacy_preload is not None and "auto_load" not in updates:
        updates["auto_load"] = _coerce(legacy_preload)

    if updates:
        settings = replace(settings, **updates)

    return settings.resolved


def apply_hf_env(settings: Settings) -> None:
    """把 HF 端点/离线开关写进 os.environ。

    **必须在 `import huggingface_hub` 之前调用**：那个包在导入期就把
    `constants.ENDPOINT` 读成模块级常量了，之后再改环境变量对它没有任何影响 ——
    这类「设了没用」的坑排查起来特别费时间，所以这个前置条件写在函数名旁边。
    """
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint
    if settings.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"


def settings_diff(settings: Settings, baseline: Optional[Settings] = None) -> Dict[str, Any]:
    """挑出与基线不同的项。基线默认是内置默认值。

    持久化必须存 diff 而不是整份 settings：存全量会把今天的默认值冻进配置文件，
    以后升级改了默认值，这些改动就对老用户永远不生效了。
    """
    base = baseline if baseline is not None else Settings()
    diff: Dict[str, Any] = {}
    for name in Settings.__dataclass_fields__:
        if getattr(settings, name) != getattr(base, name):
            diff[name] = getattr(settings, name)
    return diff


def save_settings(
    settings: Settings,
    baseline: Optional[Settings] = None,
    path: Optional[Path] = None,
) -> Path:
    """原子写配置。

    `baseline` 是**本次进程启动时的生效值**，不是内置默认值。这个区别很关键：
    启动值可能来自命令行或环境变量，而它们不该被写进配置文件 —— 否则
    `laya-server --backend echo` 跑一次，用户的 profile 里就永久多了一条
    `backend: echo`，下次他不加参数启动会拿到一个假后端，且完全不知道为什么。

    写之前先读回已有文件再合并：进程启动后才改的项覆盖进去，改回启动值的项删掉，
    其余原样保留 —— 否则每次保存都会把用户手写在文件里的其他配置抹掉。
    """
    target = path or (HOME_DIR / CONFIG_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if target.is_file():
        with contextlib.suppress(Exception):
            existing = _load_file(target)

    base = baseline if baseline is not None else Settings()
    defaults = Settings()
    for name in Settings.__dataclass_fields__:
        current = getattr(settings, name)
        default = getattr(defaults, name)
        started = getattr(base, name)
        if current == default and started != default:
            # 启动时是改过的，现在被改回了默认值 —— 这是「恢复默认」，
            # 要把这个覆盖项从文件里删掉，否则界面上点了没用。
            #
            # 判断里必须同时看两侧：只看「等于默认」会把用户在文件里手写的项
            # 也一并删掉（他写 `queue_timeout_s: 30` 时启动值就是 30，不是默认值，
            # 但另一个不相关的字段可能恰好等于默认值）。
            existing.pop(name, None)
        elif current != started:
            existing[name] = current
        # 其余情况与启动时一致：文件里原样保留（可能本来就写在里面）。

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target


def apply_updates(settings: Settings, updates: Dict[str, Any]) -> Settings:
    """把控制台提交的改动套上去。未知字段与启动时自检的规则在这里同样执行。"""
    updates = _migrate_legacy(dict(updates))
    unknown = set(updates) - EDITABLE_FIELDS
    if unknown:
        raise ValueError(
            f"这些字段不能改：{sorted(unknown)}；可改的是 {sorted(EDITABLE_FIELDS)}"
        )
    # 空 map 会被 resolved 悄悄换成默认值 —— 结果是「提交了但什么也没发生」，
    # 而调用方以为自己清空了配置。直接拒绝比默默还原好。
    for name in ("slots", "aliases"):
        if name in updates and not updates[name]:
            raise ValueError(f"{name} 不能清空；要恢复默认请用 DELETE /admin/config")
    # Settings 是 slots=True 的 dataclass，没有 __dict__，只能按字段名逐个取。
    merged = {name: getattr(settings, name) for name in Settings.__dataclass_fields__}
    merged.update(updates)
    return Settings(**merged).resolved
