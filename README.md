# laya-server

把 TypeSafe Jev 的 `POST /v1/systemone` 搬到本机。Apple Silicon 上用
[`laya-mlx`](https://pypi.org/project/laya-mlx/) 跑 Laya 的类型化决策模型，
**权重与推理全部在本机，0 次云端调用**。

```python
client = TypeSafeClient(base_url="http://127.0.0.1:8077", api_key="whatever")
```

改这一行，按 Jev 写的客户端就能继续跑。

---

## 为什么把这件事放到本机

| | 云端 Jev | 本机 laya-server |
| --- | --- | --- |
| 数据 | 原文要发到第三方 | **不出本机**。默认只监听 `127.0.0.1`，连局域网都不出 |
| 费用 | 按调用计费 | 一次性下 2.2GB 权重，之后免费 |
| 延迟 | 每次一个网络往返 | 热态 **约 120ms**（M1 实测，见下） |
| 离线 | 不可用 | **断网照常工作**，飞机上、内网里都能跑 |
| 客户端 | — | 零改动，只换 `base_url` |

除了「省事」，还有两件用起来才知道的事：

* **自带控制台**（`GET /`）。请求统计、模型装载、资源占用、改配置、优雅停机都在一页里，
  由服务自己托管 —— 不用装任何东西，也没有构建步骤。
* **macOS 上双击即用**。真的原生窗口（Dock 图标、菜单栏、⌘Q 都在），用系统自带的
  WebKit 渲染，**不需要装 Chrome / Safari**。

---

## ⚠️ 三条边界，先读

1. **这不是同一个模型。** 本机跑的是 Laya 的 MLX 移植（`aac6fef/laya-mlx` 等），
   与 TypeSafe 云端的 Jev **不是同一份权重**。JSON 形状逐字段一致，**概率不一致**。
   → 任何按 Jev 标定过的 `confidence` 阈值，**必须拿你自己的数据重新标定**。
   这不是保守猜测：`laya-mlx` 加载时就会打一条警告，说这份 checkpoint 的温度参数
   落在 `[0.5, 5]` 之外、会让部分桶的 confidence 失真，需要按「未标定」对待。
   响应里的 `model` 会如实回本地 checkpoint 的 id，不会伪装成 `jev-1.13.0`。
2. **中文要显式选模型。** Jev 和 Laya 的训练主语言都是英文，中日韩文准确率低
   （英文权重在非英语上不是平滑退化，是崩）。中文流量把 `"model"` 写成
   `laya-multilingual`，或者干脆写 `auto` 让它按 state 的语言自动挑。
3. **`output_tokens` 恒为 0。** 这是 System One 的定义，不是 bug。字段保留是为了让
   按 Jev 写的计量代码不至于 `KeyError`。

---

## 安装

```bash
cd laya_server
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

想要 **macOS 原生窗口**（不依赖浏览器）再多一步：

```bash
.venv/bin/pip install -e ".[native]"
```

---

## 启动

**macOS 双击**（不想碰命令行）：

```bash
packaging/make_macos_app.sh        # 生成 ~/Applications/LayaServer.app
```

双击它：一个真正的原生窗口打开，里面就是控制台，服务已经在后面跑起来。

**命令行**：

```bash
.venv/bin/laya-console                     # 起服务 + 开窗口（默认行为）
.venv/bin/laya-console --auto-load all     # 顺便把没有的权重也下下来
.venv/bin/laya-console --no-native         # 用浏览器代替原生窗口
.venv/bin/laya-console --no-browser        # 只起服务
.venv/bin/laya-server                      # 只要 API，不要界面
```

**只要接口、连界面都不要**：`laya-server --port 8077`。

### 窗口的行为（这是个服务端应用，不是网页应用）

| 操作 | 结果 |
| --- | --- |
| 关窗口（红点 / ⌘W） | 只是把界面收起来，**服务照常跑**，脚本继续调 API |
| 点 Dock 图标 | 窗口回来 |
| 菜单栏的 `laya` 图标 | 打开控制台 / 在浏览器中打开 / 停止服务并退出 |
| ⌘Q 或菜单「停止服务并退出」 | 真正停服务并退出 |

**再双击一次不会起第二个服务** —— 会认出已经在跑的那个，直接把窗口连上去。

---

## 发第一个请求

```bash
curl -s http://127.0.0.1:8077/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I have been trying to connect my Stripe account for 3 days and the integration keeps failing. I am losing sales. Please help ASAP.",
    "model": "jev-latest",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {
          "billing": "Payment or subscription issues",
          "technical": "Bugs or integration problems",
          "sales": "Pricing or account questions"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]
      },
      "is_urgent": {
        "type": "noul",
        "instructions": "The message conveys urgency or time-sensitivity"
      }
    }
  }'
```

响应（字段名与 Jev 文档逐字一致）：

```json
{
  "model": "aac6fef/laya-mlx",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {"billing": 0.15, "technical": 0.85, "sales": 0.0},
      "confidence": 0.78
    },
    "frustration": {
      "type": "score",
      "score": 1.0,
      "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"},
      "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
      "confidence": 1.0
    },
    "is_urgent": {"type": "noul", "noul": 1.0}
  },
  "usage": {"input_tokens": 392, "output_tokens": 0}
}
```

### 接现成客户端

```python
# typesafe_sdk：只改 base_url
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(base_url="http://127.0.0.1:8077", api_key="unused-or-whatever")
response = client.system_one(
    state="发票被重复扣款，请退款。",
    questions={"department": Choice(instructions="Which team should handle this",
                                    criteria=["billing", "technical", "sales"])},
)
print(response.answers["department"].choice)
```

更多例子在 `examples/`：`curl.sh`、`client.py`、`request.json`、`request_zh.json`。

---

## 该用哪个模型

| 槽位 | 默认 checkpoint | 用途 |
| --- | --- | --- |
| `english` | `aac6fef/laya-mlx` | 英文（Jev 别名都指到这里） |
| `multilingual` | `aac6fef/laya-multilingual-mlx` | 100+ 语言，**中文走这个** |
| `typed-decisions` | `aac6fef/laya-typed-decisions-mlx` | 特定工作流，**不会自动选中** |

`"model"` 可以填四种东西：

| 填什么 | 效果 |
| --- | --- |
| `jev-latest` / `jev-1.13.0` / `jev-preview` | 指向 `english`，兼容按 Jev 写的客户端 |
| `laya-multilingual` / `laya-typed-decisions` | 指向对应槽位 |
| `auto` | **按 state 的语言自动挑**：英文 → `english`，其余 → `multilingual` |
| `org/name` | 直接当 checkpoint id 加载 |

不确定就填 `auto`。想看它这次为什么这么选，启动时加 `--debug`，响应里会有 `debug.routing.reason`。

---

## 控制台

`GET /` 是一页自带的管理界面，服务自己托管，**零构建、零前端依赖**。

| 区块 | 能做什么 |
| --- | --- |
| **概览** | 总请求 / 成功率 / 吞吐 / p50·p99 延迟 / **纯推理耗时** / 已驻留模型数，附最近 30 分钟柱状图 |
| **资源占用** | CPU、内存、GPU 利用率、**MLX 显存**、服务进程内存，附走势图 |
| **模型** | 每个槽位的加载状态与别名；单个加载/卸载，或一键全加载/全卸载 |
| **最近请求** | 最近 1000 条：模型、问题数、答案类型、状态码、**总延迟与纯推理耗时分开列**、tokens、request id |
| **配置** | 表单由字段元数据生成；热改项立刻生效，冷改项写盘并标「重启生效」 |
| **进程** | 优雅停机 |

### 关于资源占用面板

几个数字值得说清楚，不然容易看错：

| 指标 | 含义 |
| --- | --- |
| **MLX 显存** | **模型真实占用了多少**。这是整个面板里最该看的一个数 —— 系统层面只看到进程 RSS，里面混着 Python 堆和各种缓存；MLX 记的是它实际持有的 Metal 缓冲 |
| GPU 利用率 | 读的是系统给 Metal 设备的统计，和「模型在不在跑」不是一回事（窗口合成也会算进去） |
| 内存 | 口径是「应用内存 + 有线内存 + 压缩内存」，与活动监视器的「已用内存」接近；不是简单的「总量 − 空闲」 |
| 服务进程 | 进程本身的内存。**它远小于 MLX 显存是正常的** —— 权重在统一内存里以 GPU 缓冲的形式存在，不完全计入进程 RSS |

Apple Silicon 是统一内存，**没有独立的「显存」**。面板里的「MLX 显存」和「内存」是同一块物理内存的不同归属，**不要相加**。

CPU 使用率是增量指标，采样需要一个时间窗（约 3 秒，和控制台的刷新间隔一致）。所以刚打开页面那一瞬间可能还没有数字，下一秒就有了。

---

## 配置

优先级：

```
命令行  >  环境变量（LAYA_SERVER_*）  >  工作目录 server.json  >  ~/.laya-server/config.json  >  默认值
```

`LAYA_SERVER_HOME` 换用户级目录；`LAYA_SERVER_CONFIG` 直接指定文件。
顺序是刻意的：仓库里的 `server.json` 属于「这个项目怎么跑」，用户级配置属于
「这台机器的偏好」，前者更具体所以优先级更高。

常用项：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8077` | 默认只对本机开放 |
| `--model` | `jev-latest` | 默认模型别名或 checkpoint id |
| `--api-key` | 无 | 设了就要求 `Authorization: Bearer <key>` |
| `--auto-load` | `local` | 启动时加载哪些权重，见下一节 |
| `--dtype` | `float16` | `float32` 数值更准、更慢更占内存 |
| `--batch-size` | `16` | 单次前向最多几个问题 |
| `--max-concurrency` | `1` | 同时推理数。MLX 是单设备，调高只在 IO 等待上有意义 |
| `--max-loaded` | `3` | 常驻权重数上限（LRU）。降到 1 省内存，代价是切语言时重载 |
| `--debug` | 关 | 响应附带路由、延迟等调试字段 |
| `--backend` | `mlx` | `echo` = 不加载权重的假后端，**只用于联调** |

`--preload` 是老写法，等价于 `--auto-load all`。

完整清单见 `server.example.json`，或启动后打开控制台的「配置」区 —— 那里每一项都带中文说明，
改完能直接看到「立即生效」还是「重启生效」。

用环境变量：

```bash
LAYA_SERVER_PORT=8077 LAYA_SERVER_AUTO_LOAD=all LAYA_SERVER_DTYPE=float32 .venv/bin/laya-server
```

需要改别名表或换本地权重目录时用配置文件（这两个是 map，环境变量表达不了）：

```json
{
  "port": 8077,
  "default_model": "jev-latest",
  "auto_load": "local",
  "max_loaded": 2,
  "aliases": {"jev-latest": "english", "laya-multilingual": "multilingual"},
  "slots": {
    "english": "aac6fef/laya-mlx",
    "multilingual": "aac6fef/laya-multilingual-mlx"
  }
}
```

配置会在启动时自检：别名指向不存在的槽位、`backend` 写错、`auto_load` 拼错，都会立刻报错，
不会拖到第一个请求才炸。

---

## 权重：下载、离线、启动加载

### 只下一次

权重落在 Hugging Face 缓存（`~/.cache/huggingface/hub`），三份合计约 2.2GB。
之后每次启动都是直接从磁盘读进内存，**网络不参与**。

| 场景 | 行为 |
| --- | --- |
| 有缓存 + 有网 | 直接用本地，不发网络请求 |
| 有缓存 + **没网** | 照常工作 ✓ |
| 没缓存 + 有网 | 正常下载 |
| 没缓存 + 没网 | 返回 503，报错里会写明「本地没有完整权重」并提示配镜像 |

代价是**不会自动检查有没有新版本**。要拉更新，在控制台点对应模型的「加载」即可。

不能直连 `huggingface.co` 时，配一个镜像端点：

```json
{"hf_endpoint": "https://hf-mirror.com"}
```

或者把权重放到本地目录，再把 `slots` 指过去（这种用法完全不涉及 Hugging Face）。

### 启动时自动加载本地已有的权重

`auto_load` 控制这件事，三档：

| 值 | 启动时做什么 | 适合 |
| --- | --- | --- |
| `local`（**默认**） | 把**本地已经有完整权重**的槽位读进内存，**一个网络请求都不发** | 绝大多数情况 |
| `all` | 三个槽位全加载，本地没有的就去下 | 第一次部署，想一次把环境准备好 |
| `off` | 什么都不加载，全部等第一个请求 | 内存紧张，或者这台机器只是偶尔被调用 |

默认选 `local` 的理由：既让第一个请求不必吃冷加载，又不会让「打开应用」变成
「悄悄下载 2GB 流量」。

**加载在后台进行，不挡启动** —— 窗口和服务立刻就绪，模型状态会自己在控制台里长出来。
三份权重全在本地时，实测约 **3.1 秒**全部就绪。

---

## 用自己的数据做后训练（微调）

出厂权重是通用英文场景。**你的分类法、你的置信度阈值，只能靠自己的数据喂出来**。
好消息是这件事不贵：出厂 checkpoint 本身就是一次 **1.96 小时 / 1 epoch / 单卡**
微调的产物（`rl_agent_config.json` 的 `training` 段写着），你的也不会贵多少。

`laya-mlx` 是纯推理库，训练不在它身上。两条路：

| | 路线 A：上游官方（PyTorch） | 路线 B：纯 MLX |
| --- | --- | --- |
| 工具 | 上游 [`laya`](https://github.com/NandhaKishorM/laya)（torch 2.x） | 本仓库现成的 `.venv` 就够 |
| 入口 | `notebooks/laya_finetune_typed_decisions_mps.py`（Apple GPU 可跑）<br>或 `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` | 自己写循环；模型类就是 `laya_mlx.model.DecisionModel`（标准 `nn.Module`） |
| 闭环 | 建数据集 → RLCD 训练 → 温度标定 → 评估 → 导出，全在 notebook 里 | 见下面「三件不能错的事」 |
| 转换 | `laya-mlx convert --model <微调产物> --output <目录>` | 不用转，直接按格式写盘 |

（RLCD 是 Laya 的训练法：用严格适当评分规则当奖励。落到答案头上，实质就是
正确选项的 log loss —— 路线 B 自己写就是这么简单。）

路线 B 的可行性已在本机实测（M1 / 16GB）：冻结编码器、只训决策 head（约 **27M** 可训
参数），`mlx.nn` 的 `freeze()` + `value_and_grad` 对这个模型完全可用，**单步 0.3–0.5s**。

### 训练完的东西怎么接进来

微调产物就是一个普通 checkpoint 目录，指到 `slots` 即可（写法见「配置」一节）：

```json
{
  "slots": {"english": "/path/to/你的微调产物"},
  "aliases": {"jev-latest": "english"}
}
```

### 三件不能错的事

1. **prompt 必须原样复用 `laya_mlx.common.build_sequence`**。序列格式是固定的
   `[CLS] type instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`，
   选项得分取 MASK 位置。自己拼字符串等于在另一个分布上训练，上线即失效。
2. **写回是四件套，`strict=True` 加载，参数名一个不能错**：`model.safetensors`
   （206 个张量，含一个 `temperature` 缓冲）+ `rl_agent_config.json` +
   `encoder/config.json` + `tokenizer/`。`laya_mlx/convert.py` 是现成的写盘参考。
3. **训完必须重新标定温度**。温度按桶（题型 × 选项数）写进 `rl_agent_config.json`
   的 `temperature_by_options`，推理时按 `logits / T` 生效。不标定，confidence
   就还是开头那条「未标定」警告说的事。

### 数据是唯一的瓶颈（实测）

用真实权重跑过一次完整验证：6 条标注 + 模型无法从文字推断的标签映射，训完——

```
训练集   loss 0.0001   acc 100%
留出集   acc 33% → 0%      ← 灾难性过拟合，比瞎猜还差
```

机制完全通，**6 条数据教不出任何东西**。量级：**500 条起步、几千条像样**，
并且**必须留出验证集**。数据不够时解冻编码器只会更糟 —— head 只有 ~27M 参数，
先把它喂饱。

---

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/systemone` | Jev 兼容主端点 |
| GET | `/` | 控制台页面 |
| GET | `/v1/models` | 本地别名 → 槽位 → checkpoint，以及哪些已驻留 |
| GET | `/healthz` | 存活 + 模型状态 + uptime + 累计请求数 |
| GET | `/docs` | OpenAPI（`--no-docs` 关掉） |
| GET/PUT/DELETE | `/admin/*` | 控制台用的管理接口，**默认只接受本机来源** |

额外响应头（Jev 没有，纯附加）：`X-Request-Id`、`X-Laya-Latency-Ms`、`X-Laya-Inference-Ms`。

### 错误码

| 状态 | 何时 | 错误码 |
| --- | --- | --- |
| 401 | 设了 `--api-key` 但 `Authorization: Bearer` 不对 | `unauthorized` |
| 422 | 请求体不合法（选项超 255、score 等级不在 2–10、`model` 不存在…） | `invalid_request` / `unknown_model` |
| 429 | 等推理槽位超时（`max_concurrency` 满了） | `too_many_requests`，带 `Retry-After` |
| 503 | 权重下载或构建失败 | `model_unavailable` |

错误体是 `{"error": {"code": "...", "message": "..."}, "detail": [...]}`，
`detail` 是逐字段的校验信息。

### 三种问题类型

| `type` | `criteria` | 答案字段 |
| --- | --- | --- |
| `choice` | 对象，键是选项名，值是描述（可为 `null`）。上限 **255** 项 | `choice`、`probabilities`、`confidence` |
| `score` | 有序数组，**2–10** 个等级 | `score`、`legend`、`probabilities`、`confidence` |
| `noul` | 可选 `{"true": ..., "false": ...}` | `noul`（0–1） |

`instructions` 和 `criteria` 的值可以是字符串、对象或数组，会原样透传给模型。

---

## 打包成 macOS App

两种产物，用途完全不同：

| | `make_macos_app.sh` | `build_macos.sh` |
| --- | --- | --- |
| 产物 | **薄壳**：20KB、6 个文件 | **独立包**：275MB，自带 Python |
| 给谁用 | 自己（指向本仓库的 `.venv`） | **别人**——换台机器照样跑 |
| 耗时 | 秒级 | 约 2 分钟 |
| 换机器 / 挪目录 | ❌ 立刻失效 | ✅ 能用（同架构） |

### 自己用：薄壳

```bash
packaging/make_macos_app.sh                 # → ~/Applications/LayaServer.app
packaging/make_macos_app.sh --port 9000     # 换默认端口
OUT_DIR=/tmp packaging/make_macos_app.sh    # 换输出目录
```

它的可执行文件就是一段 shell 脚本，`exec` 到本仓库的 `.venv`。所以**别把它拷给别人** ——
脚本里写死了生成时的仓库路径。对方双击会看到一个说明为什么打不开的弹窗，
并且 `~/.laya-server/app.log` 里会留一份详情。

### 给别人：独立包

```bash
.venv/bin/pip install -e ".[packaging,dev]"   # 需要 PyInstaller
packaging/build_macos.sh                      # → dist/LayaServer.app
packaging/build_macos.sh --dmg                # 顺带生成 .dmg
```

打包完的 `.app` 里有什么：

| | |
| --- | --- |
| Python 3.13 + 全部依赖 | 由 PyInstaller 打进去，对方**不需要装 Python** |
| `Contents/Resources/runtime/mlx` | 209MB，**原样拷贝**的 MLX（原因见下） |
| 控制台页面 | 已收进 `laya_server/static` |
| **模型权重** | **不在里面**。首次使用从 Hugging Face 下载约 2.2GB |

本机（M1）实测结果：

| 项 | 实测 |
| --- | --- |
| 产物大小 | 275MB（对比薄壳的 20KB） |
| 双击启动 → 服务就绪 | 约 0.5s |
| 三份权重读进内存 | 3.5s |
| 热态推理 | **0.12s**，与开发版**逐字段相同**的答案 |
| 推理时 GPU 利用率 | 70% |
| 依赖本机 Python / 本仓库 | **没有** |

**为什么 mlx 不交给 PyInstaller。** 这是个踩过才知道的坑，改了打包配置之前请先读
`packaging/laya_server.spec` 顶部那段注释。一句话版本：

`mlx/core.cpython-313-darwin.so` 的 rpath 是 `@loader_path/lib`，而 181MB 的
`mlx/lib/mlx.metallib`（Metal 着色器库）由 C++ 层按 dylib 所在目录查找，Python 层
没有任何接口能改它。交给 PyInstaller 之后它会把两者拆到两个目录、并改写 rpath ——
结果是 `import mlx` 直接失败。所以这个包是**原样拷贝**进 `Resources/runtime/mlx` 的，
一个字节都不动。

（一个容易搞混的地方：那种失败报的是
`ImportError: Encountered an error while initializing the extension.`，
**不是** `Failed to load the default metallib`。两者原因不同，别朝错的方向修。）

### 冒烟：确保它真的能干活

```bash
packaging/smoke_app.sh            # 快速：能起、能连、能停（不加载权重）
packaging/smoke_app.sh --mlx      # 真加载权重、真跑推理
```

`--mlx` 那一层是必须的。打包最容易出的问题就是 MLX 的原生件没收对，而那种情况下
服务**照样起得来**、`/healthz` 也可能通 —— 只有真跑一次推理才暴露。它还会卡一个
1 秒的延迟上限：MLX 找不到 Metal 内核时会**静默**退回 CPU，结果照样 200、答案照样
像模像样，只是慢十几倍。

### 要给别人用，还差一步

现在的产物是 **ad-hoc 签名**，本机双击没问题，但**通过网络传给别人会被 Gatekeeper 拦**，
提示「已损坏」（那其实是没签名，不是真坏了）。要让对方顺畅双击，需要 Apple
Developer ID 签名 + 公证：

```bash
MAC_SIGN_ID="Developer ID Application: 你的名字 (TEAMID)" \
NOTARY_PROFILE=你的钥匙串条目 \
packaging/build_macos.sh --dmg
```

`spctl --assess` 对 ad-hoc 签名的产物会报 `rejected`，这是正常的 —— 本机双击不受影响。

### 不管走哪条路，这三条都躲不掉

| 约束 | 说明 |
| --- | --- |
| **必须是 Apple Silicon** | MLX 只有 arm64 构建（`mlx/core.cpython-313-darwin.so` 是 arm64-only）。Intel Mac 装不上，也跑不了 |
| **macOS 14+** | `LSMinimumSystemVersion` 就是这个 |
| **还是要下 ~2.2GB 权重** | 权重不在仓库里、也不在 `.app` 里，首次使用从 Hugging Face 拿。国内基本要配 `hf_endpoint` 镜像，否则第一步就卡住 |

权重本身是 **Apache-2.0**（`aac6fef/laya-mlx`，非 gated），所以随包分发在许可上没问题 ——
但那样安装包会再大 2.2GB，通常不如让对方自己下。

---

## 常见问题

**第一次请求很慢？**
权重还没进内存。默认 `auto_load: local` 会在启动时就把本地已有的权重读起来，
后台约 3 秒完成；所以正常情况下第一个请求也是快的（实测 0.32s）。
如果本地还没有权重，第一次要下 2.2GB —— 这一步在控制台的「模型」区能看着它涨。

**空闲一会儿之后的第一发要 300–450ms，后面又回到 120ms？**
这是 **macOS 的内存压缩器**，不是服务的问题。系统会把进程里久没用到的冷页面压缩起来，
第一发请求要把它们解压回内存。

怎么判断是它：那一发的 **总延迟 ≈ 纯推理耗时**，说明没在排队。但要注意这两列**都**测不到
「等内存」—— 解压发生在推理调用内部，所以它会被一起算进「纯推理耗时」里，
看起来就像「推理突然变慢了」。证据要靠系统计数器（下面这张表）。

| 空闲 | 延迟 | 解压次数 | 压缩器页数变化 |
| --- | --- | --- | --- |
| 10s | 159ms | +70 | −7 |
| 25s | 142ms | 0 | +2313 |
| **40s** | **398ms** | **+71,356**（≈1.14 GB） | **−68,905** |
| **40s** | **376ms** | **+73,320**（≈1.12 GB） | — |
| **40s** | **406ms** | **+73,665**（≈1.12 GB） | — |

三轮都是「空闲 40 秒 → 400ms 上下 → 紧接着 120ms 上下」，可复现。
**不是 GPU 空闲降频** —— 那个单独测过，只值 20ms 左右；也不是排队（排队只占几毫秒）。

一次解压之后就恢复了，所以它只影响「隔了几十秒的第一发」。
真要压掉它，就把常驻内存降下来：`max_loaded` 改成 `1`（只留你在用的那个语种），
代价是切语言时要重新加载。

**返回 503 说是 `model_unavailable`？**
权重没下下来。报错信息里会区分「本地没有完整权重」和「下载本身失败」：
前者配 `hf_endpoint` 镜像或手动放权重，后者看网络。

**中文结果不准？**
把 `"model"` 改成 `laya-multilingual` 或 `auto`。别用 `jev-latest` 处理中文。

**能直接拿 `confidence` 做门控吗？**
**不能。** 见开头第 1 条 —— 这份权重与云端 Jev 不同，按 Jev 标定的阈值会失效；
`laya-mlx` 自己也会警告部分桶的 confidence 未标定。要用就先拿你的数据标一遍，
做法见「用自己的数据做后训练」。

**端口被占？**
`laya-console` 会自动往后找一个空闲端口并在日志里说明。想让它直接失败用 `--strict-port`。

**关掉窗口服务还在跑？**
这是设计：窗口只是界面，服务是服务（脚本也在用它）。要真停就 ⌘Q，或者用控制台的「停止服务」。

**双击 App 没反应？**
先看日志，它在 `~/.laya-server/app.log`（超过 2MB 自动轮转）。两种 `.app` 都把输出写在那儿 ——
GUI 应用双击时没有终端，不写日志就真的什么都看不到。
也可以直接跑可执行文件，错误会打在终端里而不是消失：

```bash
~/Applications/LayaServer.app/Contents/MacOS/launch    # 薄壳版
dist/LayaServer.app/Contents/MacOS/laya-console        # 独立包版
```

命令行起服务看全部日志：

```bash
cd .../laya_server && .venv/bin/laya-console
```

**`--host 0.0.0.0` 之后，隔壁同事能改我的配置吗？**
不能。`/admin/*` 默认只接受回环地址来的请求，管理面不会跟着监听地址一起暴露。
（能改配置意味着能改 API Key、能停机，这个方向上的失败不值得赌。）
真要放开就改 `admin_local_only`，但那等于把这些权限交出去。

**同事能直接用我机器上跑着的这个服务吗（而不是自己装一份）？**
可以，两步：你的服务监听局域网，并设一个 API Key ——

```bash
.venv/bin/laya-console --host 0.0.0.0 --api-key 换一个够长的随机串
```

默认只听 `127.0.0.1`，所以第一步必须显式打开。**不设 `--api-key` 就等于把模型开放给
整个网段**，`--host` 不是本机地址时启动日志里也会警告你这一点。

然后对方把 `base_url` 指到 `http://<你的局域网 IP>:8077` 即可，代码一行不用改。
管理面（`/admin/*`）仍然只接受你本机的请求，所以对方能用推理接口、改不了你的配置。
连不上先看 macOS 防火墙有没有拦入站连接。

---

## 性能参考

Apple M1 / 16GB / 三份权重全部驻留（约 2.2GB）实测：

| 项 | 数字 |
| --- | --- |
| 进程启动到 `/healthz` 可用 | 约 0.4s |
| 启动后台把三份本地权重读进内存 | 3.1s |
| 首个请求（权重已驻留） | 0.32s |
| **热态请求 p50（连续）** | **120ms** |
| 连续 60 发的分布 | 117–179ms，中位 122ms，无离群 |
| 空闲 40 秒后的第一发 | 380–450ms（macOS 解压 ~1.1GB 冷页面，见「常见问题」） |
| 常驻内存（3 份权重） | MLX 显存约 2.2GB |

延迟的两列在控制台里是分开的：**总延迟**和**纯推理耗时**。两者差距大说明在排队，
差距小说明时间都花在推理里 —— 这是判断「为什么变慢」的第一刀。

---

## 与 Jev 的差异

| 项 | Jev | 本服务 |
| --- | --- | --- |
| 后端 | 云端 | 本机 MLX |
| `model` 响应值 | `jev-1.13.0` 这类服务端版本 | 本地 checkpoint id |
| 权重 | 官方 Jev | Laya MLX 移植，**不同权重** |
| 单 answer 上的 `action.act_probability` | 无 | 默认剔除，`--debug` 时保留 |
| noul answer 上的 `confidence` | 无 | 默认剔除，`--debug` 时保留 |
| `/v1/models`、`/healthz` | 无 | 有（附加，不影响兼容） |
| 529 Overloaded | 有 | 归到 429（本地排队超时） |

默认响应**逐字段**与 Jev 一致：`{"model", "answers", "usage"}`，一个不多一个不少。
多出来的字段会让 `extra="forbid"` 的严格解析器直接拒掉，所以默认一律不带。

---

## 开发

```bash
.venv/bin/python -m pytest -q          # 158 个测试，离线可跑，不下载任何权重
```

模型准不准不在测试里量 —— 那要用真实权重和真实样本量去跑，而且得用你自己的数据。

---

## 许可

Apache-2.0。Laya 权重与上游 prompt 构造来自 Convai Innovations 及贡献者；
`laya-mlx` 是独立 MLX 移植，非 Convai 官方发布。
