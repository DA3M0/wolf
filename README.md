# wolf

`wolf` 是一个 Rust 本地模型加载和对话命令行工具。最简单的用法是把模型文件路径作为唯一参数：

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

# Gemma 4 使用包含配置、tokenizer 和 Safetensors 权重的 Hugging Face 模型目录
wolf ./models/gemma-4-E2B-it

# 调整线程数、每轮回答长度
wolf ./models/model.gguf --threads 8 --max-tokens 128

# 调整采样：温度 0.7、top-k 50、固定随机种子（可复现）
wolf ./models/model.gguf --temperature 0.7 --top-k 50 --seed 42

# 确定性输出（贪心解码，等价于关闭采样）
wolf ./models/model.gguf --temperature 0

# 设置系统提示词（仅在对话首轮注入）
wolf ./models/model.gguf --system "你是一个简洁的中文助手"

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
| `--threads N` | Rayon 默认值 | CPU 推理线程数 |
| `--max-tokens N` | 256 | 每轮最大生成 token 数 |
| `--temperature F` | 0.8 | 采样温度；`0` 表示贪心确定性输出 |
| `--top-p F` | 0.9 | 核采样概率阈值 |
| `--top-k N` | 关闭 | top-k 采样候选数 |
| `--repeat-penalty F` | 1.1 | 重复惩罚，作用于最近 64 个 token；`1` 表示关闭 |
| `--seed N` | 随机 | 随机种子；固定后同一输入可复现 |
| `--system "..."` | 无 | 系统提示词，仅在对话首轮注入 |

默认使用轻度采样（温度 0.8、top-p 0.9、重复惩罚 1.1），回答更自然；需要完全可复现的输出时使用 `--temperature 0`，或 `--seed` 加相同参数。

推理时会在 stderr 显示生成速度统计；回答文本输出到 stdout。

### 4. 对话内命令

| 命令 | 作用 |
| --- | --- |
| `/exit`、`/quit` | 退出对话 |
| `/clear` | 清空对话历史并重新加载模型（系统提示词会在下一轮重新注入） |
| `/help` | 显示对话内命令帮助 |

生成过程中按 `Ctrl-C` 只会中断当前这轮回答，会话保留，可以继续提问；在 `>` 提示符处按 `Ctrl-C` 会取消当前输入并给出新提示符。`Ctrl-D`（输入为空时按回车等同于 EOF）退出程序。

### 5. 上下文管理

上下文用量达到模型上限的 80% 时会给出警告。继续对话导致当前轮放不下时，wolf 会自动截断最早的对话轮次、重新加载模型并把最近的历史一次性重放（重放结果与连续对话等价），不需要手动重启。单轮输入加生成长度超过整个上下文窗口时，会提示缩短输入。

截断重放会重新加载模型，大模型上会花费与启动时相当的加载时间；`/clear` 也走同样的重载逻辑。

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

- **对话推理：** GGUF CPU 推理支持 `llama`、`qwen2`、`qwen3`、`qwen3moe`、`gemma`、`gemma2`、`gemma3`、`phi2` 和 `phi3` 架构。Gemma 4 支持 Hugging Face **文本模型目录**：目录需包含 `config.json`、`tokenizer.json` 和 `.safetensors` 权重；支持分片 `model.safetensors.index.json`。Gemma 4 使用 Candle 0.11 文本模型实现和 CPU 推理，不启用视觉/音频输入。目录权重支持 `model.safetensors` 或索引文件 `model.safetensors.index.json` 指定的多个分片。GGUF 根据 `general.architecture` 选择对应 Candle 模型；支持 LLaMA SentencePiece 和 GGUF GPT-2/BPE tokenizer，以及对应的 LLaMA、Llama 3、ChatML、Gemma 和 Phi 对话提示格式。
- **文件检查：** GGUF v2/v3、Safetensors；展示元数据和张量信息（含 GGUF 张量字节大小估算），校验头部、数据范围与张量重叠。
- **暂不支持：** Gemma 4 GGUF、Gemma 4 图像/音频输入、尚未实现的 GGUF 架构（如 Mistral、DeepSeek、GLM）、其他模型的 Safetensors 推理、GPU 推理、ONNX、PyTorch `.bin`/pickle。仅支持的 GGUF 架构才可推理；架构虽然受支持，仍要求模型张量命名、量化类型和 tokenizer 元数据符合相应实现。优先使用 instruction/chat 微调模型。

大型模型的加载和 CPU 推理需要足够内存，速度取决于模型大小和设备。量化模型通常更适合普通本地机器。

## 许可证

本项目以 [Apache-2.0](LICENSE) 许可证发布。
