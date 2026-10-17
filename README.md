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

### 作为 Rust 库使用

`wolf` 同时提供可复用的 library crate。模型检查 API 返回结构化数据，不会向终端打印；上下文存档 API 也可独立用于应用程序：

```rust
use std::path::Path;
use wolf::{ChatOptions, ContextStore, ModelInfo, inspect_model};

fn inspect(path: &Path) -> Result<ModelInfo, String> {
    let info = inspect_model(path)?;
    println!("{}: {} 个张量", info.format, info.tensors.len());
    Ok(info)
}

fn main() -> Result<(), String> {
    let info = inspect(Path::new("./models/model.gguf"))?;
    let store = ContextStore::new(None);
    let saved_contexts = store.list()?;
    println!("{} 个已保存上下文", saved_contexts.len());
    Ok(())
}
```

聊天 API 可通过 `wolf::chat::chat` 启动交互式对话；使用 `ChatOptions`、`ContextArchive` 和 `ContextStore` 配置模型与上下文。默认选项可通过 `ChatOptions::default()` 获取：

```rust
use std::path::Path;
use wolf::{ChatOptions, ContextArchive, ContextStore};

fn main() -> Result<(), String> {
    let path = Path::new("./models/model.gguf");
    let options = ChatOptions::default();
    let archive = ContextArchive::new(path, options.clone())?;
    wolf::chat::chat(path, options, archive, ContextStore::new(None))?;
    Ok(())
}
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

## 支持范围

- **对话推理：** 仅支持 GGUF，由 llama.cpp 推理，可使用其支持的 GGUF 架构与模型内置聊天模板；默认 CPU，可通过 Vulkan/CUDA/ROCm/Metal Cargo feature 自行构建 GPU 后端。oneAPI/SYCL 暂未支持。
- **文件检查：** GGUF v2/v3、Safetensors；展示元数据和张量信息（含 GGUF 张量字节大小估算），校验头部、数据范围与张量重叠。
- **暂不支持：** Safetensors 推理、ONNX、PyTorch `.bin`/pickle。GGUF 的具体架构和量化类型支持范围取决于 llama.cpp 版本；优先使用 instruction/chat 微调模型。

大型模型的加载和 CPU 推理需要足够内存，速度取决于模型大小和设备。量化模型通常更适合普通本地机器。

## 许可证

本项目以 [Apache-2.0](LICENSE) 许可证发布。

GGUF 推理由 llama.cpp 提供（MIT 许可证）；Rust 绑定 `llama-cpp-4` 使用 MIT 或 Apache-2.0 许可证。分发时还须保留相应上游版权和许可证声明。
