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

### 2. 加载模型并聊天

路径可以是相对路径或绝对路径。路径中有空格时用引号括起来：

```sh
# Linux/macOS，相对路径
wolf ./models/model.gguf

# 其他已支持架构同样直接传入 GGUF 文件路径
wolf ./models/Qwen3-8B-Instruct-Q4_K_M.gguf
wolf ./models/gemma-3-4b-it-Q4_K_M.gguf
wolf ./models/Phi-3-mini-4k-instruct.Q4_K_M.gguf

# Gemma 4 使用包含配置、tokenizer 和 Safetensors 权重的 Hugging Face 模型目录
wolf ./models/gemma-4-E2B-it

# 按 CPU 核心数调线程，并限制每轮回答长度
wolf ./models/model.gguf --threads 8 --max-tokens 128

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
已加载 qwen2 模型（CPU）。输入 /exit 退出。
> 你好，请介绍一下自己
你好！我是一个本地运行的语言模型……

> 用三点总结刚才的回答
……

> /exit
已退出。
```

输入 `/exit` 是正常退出命令；其他输入（包括 `exit`）会当作普通问题发送给模型。推理会显示生成速度；可用 `--threads` 调整 CPU 线程数，用 `--max-tokens` 限制每轮回答长度。

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

- **对话推理：** GGUF CPU 推理支持 `llama`、`qwen2`、`qwen3`、`qwen3moe`、`gemma`、`gemma2`、`gemma3`、`phi2` 和 `phi3` 架构。Gemma 4 支持 Hugging Face **文本模型目录**：目录需包含 `config.json`、`tokenizer.json` 和 `.safetensors` 权重；支持分片 `model.safetensors.index.json`。Gemma 4 使用 Candle 0.11 文本模型实现和 CPU 推理，不启用视觉/音频输入。目录权重支持 `model.safetensors` 或索引文件 `model.safetensors.index.json` 指定的多个分片。GGUF 根据 `general.architecture` 选择对应 Candle 模型；支持 LLaMA SentencePiece 和 GGUF GPT-2/BPE tokenizer，以及对应的 LLaMA、Llama 3、ChatML、Gemma 和 Phi 对话提示格式。
- **文件检查：** GGUF v2/v3、Safetensors；展示元数据和张量信息，校验头部和数据范围。
- **暂不支持：** Gemma 4 GGUF、Gemma 4 图像/音频输入、尚未实现的 GGUF 架构（如 Mistral、DeepSeek、GLM）、其他模型的 Safetensors 推理、GPU 推理、ONNX、PyTorch `.bin`/pickle。仅支持的 GGUF 架构才可推理；架构虽然受支持，仍要求模型张量命名、量化类型和 tokenizer 元数据符合相应实现。优先使用 instruction/chat 微调模型。

大型模型的加载和 CPU 推理需要足够内存，速度取决于模型大小和设备。量化模型通常更适合普通本地机器。
