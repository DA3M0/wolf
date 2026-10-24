# wolf

`wolf` 是一个 Rust 本地 GGUF 模型聊天和检查命令行工具。GGUF 对话推理由 llama.cpp（llama-cpp-4 绑定）提供；Safetensors 文件仅支持结构检查，不用于推理。最简单的用法是把模型文件路径作为唯一参数：

```sh
wolf ./models/llama-3-8b-instruct.Q4_K_M.gguf
```

模型加载完成后会出现 `>` 提示符。输入问题开始多轮对话，输入 `/exit`（或 `/quit`）结束；终端中也可以按 `Ctrl-D` 结束输入。

## 完整使用命令

### 1. 构建

在项目目录运行：

```sh
cargo build --release
```

生成的可执行文件为 `target/release/wolf`。以下命令假定它已加入 `PATH`；如果没有，请使用 `./target/release/wolf` 代替 `wolf`。

默认构建使用 CPU；项目发布的预编译版本也仅包含 CPU 后端。GPU 后端需在安装相应工具链/SDK 后自行编译。Vulkan 是推荐的跨平台 GPU 后端，使用 `vulkan` feature 显式启用。构建机需要 C/C++ 构建工具、CMake，以及 Vulkan SDK（包含 Vulkan headers、loader 和 `glslc` GLSL 编译器）。例如 Ubuntu/Debian 可安装：

```sh
sudo apt install build-essential cmake libvulkan-dev glslc
cargo build --release --features vulkan
```

Windows 请先安装 [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home)，并确保 MSVC C++ 构建工具可用。macOS 可使用 Metal feature；若使用 Vulkan，需安装 LunarG Vulkan SDK/MoltenVK 并设置 `VULKAN_SDK`。若构建时 macOS 上的 Vulkan SDK 不可用，底层依赖会回退到 Metal。其他 Linux 发行版请安装对应的 Vulkan 开发包和 `glslc`。

其他 GPU 后端同样需要在本机安装相应工具链后自行构建：

```sh
cargo build --release --features cuda   # NVIDIA CUDA
cargo build --release --features rocm   # AMD ROCm（底层 HIP）
cargo build --release --features metal  # Apple Metal（macOS）
```

运行时还需要兼容的 GPU 和已安装的驱动/运行库；启用 feature 不会自动提供 GPU 或驱动。具体 SDK 支持范围由 llama.cpp 后端决定。oneAPI/SYCL 暂不提供 feature：当前 `llama-cpp-4` 绑定尚未暴露该构建选项。

查询版本：

```sh
wolf version        # 或 wolf --version / -V
```

### 2. 加载模型并聊天

路径可以是相对路径或绝对路径。路径中有空格时用引号括起来：

```sh
# Linux/macOS，相对路径
wolf ./models/model.gguf

# 其他已支持架构同样直接传入 GGUF 文件路径
wolf ./models/Qwen3-8B-Instruct-Q4_K_M.gguf
wolf ./models/gemma-3-4b-it-Q4_K_M.gguf
wolf ./models/Phi-3-mini-4k-instruct.Q4_K_M.gguf

# 调整线程数、每轮回答长度
wolf ./models/model.gguf --threads 8 --max-tokens 128

# 调整采样：温度 0.7、top-k 50、固定随机种子（可复现）
wolf ./models/model.gguf --temperature 0.7 --top-k 50 --seed 42

# 确定性输出（贪心解码，等价于关闭采样）
wolf ./models/model.gguf --temperature 0

# 设置系统提示词（仅在对话首轮注入）
wolf ./models/model.gguf --system "你是一个简洁的中文助手"

# 保存和恢复命名上下文
# 在聊天提示符输入：/save work
wolf chat ./models/model.gguf --context work
wolf context list
wolf context delete work

# Linux/macOS，绝对路径
wolf /home/alice/models/model.gguf

# macOS 示例：路径有空格
wolf "/Users/alice/My Models/model.gguf"

# Windows PowerShell 示例：引号包住路径
wolf "D:\AI Models\model.gguf"
```

也可以显式使用 `chat` 子命令：

```sh
wolf chat /data/models/model.gguf
```

对话示例：

```text
正在加载模型：/data/models/model.gguf（3.8 GB）
已加载 qwen2 模型（CPU，用时 12.4 秒）。输入 /exit 退出。
> 你好，请介绍一下自己
你好！我是一个本地运行的语言模型……

> 用三点总结刚才的回答
……

> /exit
已退出。
```

### 3. 生成与采样选项

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--threads N` | llama.cpp 默认值 | GGUF 推理线程数 |
| `--max-tokens N` | 256 | 每轮最大生成 token 数 |
| `--temperature F` | 0（贪心） | 采样温度；大于 0 时开启随机采样，同一输入每次输出会不同 |
| `--top-p F` | 0.9 | 核采样概率阈值，仅在 `--temperature` > 0 时生效 |
| `--top-k N` | 关闭 | top-k 采样候选数 |
| `--repeat-penalty F` | 1（关闭） | 重复惩罚，作用于最近 64 个 token |
| `--seed N` | 随机 | 随机种子；固定后同一输入可复现 |
| `--system "..."` | 无 | 系统提示词，仅在对话首轮注入 |

默认使用确定性贪心解码（温度 0、无重复惩罚），同样的输入总是得到同样的回答，输出最稳定。想要更多样化的回答可以显式开启采样，例如 `--temperature 0.7 --top-p 0.9`；配合 `--seed` 可以复现某一次生成。

推理时会在 stderr 显示生成速度统计；回答文本输出到 stdout。

### 4. 对话内命令

| 命令 | 作用 |
| --- | --- |
| `/exit`、`/quit` | 退出对话 |
| `/clear` | 清空对话历史并重新加载模型（系统提示词会在下一轮重新注入） |
| `/save <名称>` | 保存或更新当前对话上下文 |
| `/help` | 显示对话内命令帮助 |

生成过程中按 `Ctrl-C` 只会中断当前这轮回答，会话保留，可以继续提问；在 `>` 提示符处按 `Ctrl-C` 会取消当前输入并给出新提示符。`Ctrl-D`（输入为空时按回车等同于 EOF）退出程序。

### 5. 上下文管理

聊天中使用 `/save <名称>` 保存完整对话；之后用 `wolf chat <模型路径> --context <名称>` 恢复。恢复时模型路径必须匹配；省略推理选项会自动使用存档设置，显式指定推理选项时必须与存档设置一致。`wolf context list` 列出存档，`wolf context delete <名称>` 删除存档。可使用 `--context-dir <路径>` 覆盖默认存储目录，该选项可用于聊天和上下文管理命令。

默认存档目录为 Linux 上的 `$XDG_DATA_HOME/wolf/contexts`（未设置时为 `~/.local/share/wolf/contexts`）、macOS 上的 `~/Library/Application Support/wolf/contexts`，以及 Windows 上的 `%LOCALAPPDATA%/wolf/contexts`（回退到 `%APPDATA%`）。存档使用带版本号的 JSON；GGUF 对话恢复时会按 llama.cpp 聊天模板重新构造对话。

上下文用量接近模型上限时，wolf 会自动截断最早的完整对话轮次，再由 llama.cpp 根据剩余历史重新构造提示。单轮输入加生成长度超过整个上下文窗口时，会提示缩短输入。`/clear` 清除对话历史及 llama.cpp 的 KV cache。

### 6. 检查模型文件

```sh
# 查看帮助和支持范围
wolf help
wolf formats

# 显示模型信息，默认最多显示 20 个张量
wolf load ./models/model.gguf
wolf load /data/models/model.safetensors

# 显示全部张量，或输出 JSON
wolf load ./models/model.gguf --all
wolf load ./models/model.safetensors --json
```

`inspect` 是 `load` 的别名，例如：

```sh
wolf inspect ./models/model.safetensors --json
```

检查会校验文件头部、张量数据范围，并检测张量数据是否互相重叠或越界（GGUF 与 Safetensors 均支持）。无法确定大小的 GGUF 量化类型会跳过重叠检测。

退出码约定：参数或用法错误为 `2`，其他运行时错误为 `1`，成功为 `0`。

### 7. 以本地 API 服务运行（wolf serve）

`wolf serve` 加载一个 GGUF 模型并对外提供 OpenAI 兼容的 HTTP 接口，让 Agent（Cline / Continue / Aider 等）可以用标准 `openai` 客户端调用本地模型：

```sh
wolf serve qwen2.5-0.5b --model-dir ./models --port 8080 --generate-key
```

启动后输出：

```text
已加载模型：./models/qwen2.5-0.5b.gguf
已启动服务: http://localhost:8080
API Key: wolf-8f3a1c...
```

Ctrl-C 停止服务。

#### 服务选项

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `<名称或路径>` | 必填 | 模型名称或 `.gguf` 文件路径；也兼容旧写法 `--model <名称或路径>` |
| `--model-dir <目录>` | `.` | 按名称搜索模型的目录 |
| `--host <地址>` | `127.0.0.1` | 监听地址 |
| `--port <端口>` | `8080` | 监听端口（`0` = 系统分配） |
| `--max-concurrent-generations <N>` | `1` | 同时执行模型推理的请求上限 |
| `--generate-key` | 关闭 | 自动生成 API Key 并打印 |
| `--api-key <KEY>` | 无 | 指定 API Key（与 `--generate-key` 互斥） |

服务选项可与对话选项（`--threads`、`--max-tokens`、`--temperature`、`--top-p`、`--top-k`、`--repeat-penalty`、`--seed`、`--system`）共用，用于设置服务的默认采样参数。请求中的同名字段会覆盖这些默认值。

生成请求按 HTTP 服务接收顺序进入队列，最多同时执行 `--max-concurrent-generations` 次模型推理；默认值 `1` 保持串行行为。增大上限可提高吞吐量，但会增加内存和计算资源占用。无效请求在校验阶段取消其队列位置，不会阻塞后续请求。

模型名或路径建议直接作为 `wolf serve` 后的第一个参数；已有脚本仍可使用 `--model <名称或路径>`。

未提供 `--generate-key` 或 `--api-key` 时，服务仅允许监听回环地址（`127.0.0.1` / `localhost` / `::1`）；若要监听非回环地址（如 `0.0.0.0`），必须显式启用鉴权，避免误暴露到局域网。

#### API 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | 对话补全，支持流式（SSE，`stream: true`） |
| `POST` | `/v1/completions` | 文本补全，`prompt` 作为单条用户消息（仅非流式） |
| `GET` | `/v1/models` | 列出已加载模型 |
| `GET` | `/v1/models/:id` | 获取单个模型 |
| `GET` | `/health` | 健康检查（免鉴权） |

所有请求（`/health` 除外）须携带 `Authorization: Bearer <API Key>` 头。错误响应使用 OpenAI 风格的 JSON 结构。

#### curl 示例

```sh
# 列出模型
curl -H "Authorization: Bearer wolf-8f3a1c..." http://localhost:8080/v1/models

# 非流式对话
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer wolf-8f3a1c..." \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":128}'

# 流式对话
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer wolf-8f3a1c..." \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

#### OpenAI 客户端示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="wolf-8f3a1c...",
)

response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

#### 支持的请求字段

`POST /v1/chat/completions` 支持：`model`、`messages`（`content` 可为字符串或文本数组）、`stream`、`temperature`、`top_p`、`top_k`、`max_tokens` / `max_completion_tokens`、`stop`（字符串或数组）、`seed`、`n`（仅 `1`）。角色支持 `system` / `developer`（映射为 `system`） / `user` / `assistant`。

`model` 字段可以省略；若填写，必须等于已加载模型的 ID（即 `/v1/models` 返回的 `id`，默认为文件名去掉 `.gguf`），否则返回 404。服务一次只加载一个模型，并发请求按到达顺序串行推理。

**不支持工具/函数调用**（`tools`、`tool_calls`）：引擎无 grammar/工具支持。多模型热加载、`/v1/embeddings`、`/v1/audio`、`/v1/images` 均不支持。

## 支持范围

- **对话推理：** 仅支持 GGUF，由 llama.cpp 推理，可使用其支持的 GGUF 架构与模型内置聊天模板；默认 CPU，可通过 Vulkan/CUDA/ROCm/Metal Cargo feature 自行构建 GPU 后端。oneAPI/SYCL 暂未支持。
- **文件检查：** GGUF v2/v3、Safetensors；展示元数据和张量信息（含 GGUF 张量字节大小估算），校验头部、数据范围与张量重叠。
- **暂不支持：** Safetensors 推理、ONNX、PyTorch `.bin`/pickle。GGUF 的具体架构和量化类型支持范围取决于 llama.cpp 版本；优先使用 instruction/chat 微调模型。

大型模型的加载和 CPU 推理需要足够内存，速度取决于模型大小和设备。量化模型通常更适合普通本地机器。

## 许可证

本项目以 [Apache-2.0](LICENSE) 许可证发布。

GGUF 推理由 llama.cpp 提供（MIT 许可证）；Rust 绑定 `llama-cpp-4` 使用 MIT 或 Apache-2.0 许可证。分发时还须保留相应上游版权和许可证声明。
