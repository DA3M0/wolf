# wolf

`wolf` 是一个 Rust 本地模型加载和对话命令行工具。最简单的用法是把模型文件路径作为唯一参数：

```sh
wolf ./models/llama-3-8b-instruct.Q4_K_M.gguf
```

模型加载完成后会出现 `>` 提示符。输入问题开始多轮对话，输入 `/exit` 结束；终端中也可以按 `Ctrl-D` 结束输入。

## 完整使用命令

### 1. 构建

在项目目录运行：

```sh
cargo build --release
```

生成的可执行文件为 `target/release/wolf`。以下命令假定它已加入 `PATH`；如果没有，请使用 `./target/release/wolf` 代替 `wolf`。

### 2. 加载任意路径的 GGUF 模型并聊天

路径可以是相对路径或绝对路径。路径中有空格时用引号括起来：

```sh
# Linux/macOS，相对路径
wolf ./models/model.gguf

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
正在加载模型：/data/models/model.gguf
模型已加载（CPU）。输入 /exit 退出。
> 你好，请介绍一下自己
你好！我是一个本地运行的语言模型……

> 用三点总结刚才的回答
……

> /exit
已退出。
```

输入 `/exit` 是正常退出命令；其他输入（包括 `exit`）会当作普通问题发送给模型。

### 3. 检查模型文件

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

## 支持范围

- **对话推理：** GGUF 格式、LLaMA 系模型、CPU 推理。覆盖 GGUF GPT-2/BPE tokenizer 与 LLaMA SentencePiece tokenizer，并识别 Llama 3 与 `[INST]` 两类对话格式；模型架构及词表必须符合 GGUF 元数据约定。
- **文件检查：** GGUF v2/v3、Safetensors；展示元数据和张量信息，校验头部和数据范围。
- **暂不支持：** Safetensors 推理、GPU 推理、ONNX、PyTorch `.bin`/pickle。对话提示模板使用 LLaMA `[INST] ... [/INST]` 格式，因此优先使用 instruction/chat 微调模型；基础模型未必适合对话。

大型模型的加载和 CPU 推理需要足够内存，速度取决于模型大小和设备。量化模型通常更适合普通本地机器。
