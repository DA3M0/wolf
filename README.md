# wolf

wolf is an ultra-lightweight, OpenAI-compatible local LLM gateway for AI Agents, built in Rust and C. It also provides a powerful CLI for chatting with, inspecting, and managing GGUF models. GGUF chat inference is provided by llama.cpp (through the `llama-cpp-4` bindings); Safetensors files are supported for structural inspection only, not inference. The simplest way to use wolf is to pass a model file path as its only argument:

```sh
wolf ./models/llama-3-8b-instruct.Q4_K_M.gguf
```

Once the model is loaded, a `>` prompt appears. Enter a question to start a multi-turn conversation. Enter `/exit` (or `/quit`) to quit; you can also press `Ctrl-D` in the terminal.

## Commands

### 1. Build

Run this from the project directory:

```sh
cargo build --release
```

The executable is generated at `target/release/wolf`. The commands below assume it is on your `PATH`; otherwise, use `./target/release/wolf` instead of `wolf`.

The default build uses the CPU with OpenMP enabled out-of-the-box. Prebuilt releases include only the CPU backend. To use a GPU backend, install its toolchain/SDK and build it yourself. Vulkan is the recommended cross-platform GPU backend and can be enabled explicitly with the `vulkan` feature. The build machine needs C/C++ build tools, CMake, and the Vulkan SDK (including Vulkan headers, a loader, and the `glslc` GLSL compiler). For example, on Ubuntu/Debian:

```sh
sudo apt install build-essential cmake libvulkan-dev glslc
cargo build --release --features vulkan
```

For maximum CPU performance on a specific machine, build with the optional `native` feature:

```sh
cargo build --release --features native
```

This applies maximum CPU-specific tuning through llama.cpp's auto-tuning (`GGML_NATIVE`), allowing it to use instruction sets supported by the CPU on the build machine (for example, AVX2 on x86-64 or NEON on ARM). SIMD-optimized kernels can improve inference speed, especially for CPU-bound workloads, though the actual gain depends on the model and hardware. Build on the machine that will run the binary: this option tunes for the build host, is not a cross-compilation target selector, and the resulting binary may not run on older CPUs or other machines without the same instruction-set support. Keep the default build for portable binaries and release artifacts. To combine CPU tuning with a GPU backend, enable both features, for example `cargo build --release --features "native,vulkan"`.

On Windows, install the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home) and ensure the MSVC C++ build tools are available. On macOS, you can use the Metal feature; to use Vulkan, install the LunarG Vulkan SDK/MoltenVK and set `VULKAN_SDK`. If the Vulkan SDK is unavailable when building on macOS, the underlying dependency falls back to Metal. On other Linux distributions, install the corresponding Vulkan development packages and `glslc`.

Other GPU backends also require their toolchains to be installed locally before building:

```sh
cargo build --release --features cuda   # NVIDIA CUDA
cargo build --release --features rocm   # AMD ROCm (HIP underneath)
cargo build --release --features metal  # Apple Metal (macOS)
```

At runtime, a compatible GPU and its drivers/runtime libraries are also required; enabling a feature does not install a GPU or its drivers. The supported SDK range depends on the llama.cpp backend. There is no oneAPI/SYCL feature yet because the current `llama-cpp-4` bindings do not expose that build option.

To check the version:

```sh
wolf version        # or wolf --version / -V
```

### 2. Load a model and chat

Paths can be relative or absolute. Enclose paths containing spaces in quotes:

```sh
# Linux/macOS, relative path
wolf ./models/model.gguf

# Pass GGUF files for other supported architectures in the same way
wolf ./models/Qwen3-8B-Instruct-Q4_K_M.gguf
wolf ./models/gemma-3-4b-it-Q4_K_M.gguf
wolf ./models/Phi-3-mini-4k-instruct.Q4_K_M.gguf

# Set the thread count and maximum response length per turn
wolf ./models/model.gguf --threads 8 --max-tokens 128

# Set the inference context window (default: 8192 tokens)
wolf ./models/model.gguf --ctx-size 16384

# Limit the serialized KV prefix cache to 128 MiB (0 disables prefix caching)
wolf ./models/model.gguf --max-cache-mb 128

# Adjust sampling: temperature 0.7, top-k 50, and a fixed, reproducible random seed
wolf ./models/model.gguf --temperature 0.7 --top-k 50 --seed 42

# Deterministic output (greedy decoding, equivalent to disabling sampling)
wolf ./models/model.gguf --temperature 0

# Set a system prompt (injected only at the start of a conversation)
wolf ./models/model.gguf --system "You are a concise assistant"

# Save and restore a named context
# At the chat prompt, enter: /save work
wolf chat ./models/model.gguf --context work
wolf context list
wolf context delete work

# Linux/macOS, absolute path
wolf /home/alice/models/model.gguf

# macOS, path containing spaces
wolf "/Users/alice/My Models/model.gguf"

# Windows PowerShell, enclose the path in quotes
wolf "D:\AI Models\model.gguf"
```

You can also invoke the `chat` subcommand explicitly:

```sh
wolf chat /data/models/model.gguf
```

Example conversation:

```text
Loading model: /data/models/model.gguf (3.8 GB)
Loaded qwen2 model (CPU, 12.4 seconds). Enter /exit to quit.
> Hello, please introduce yourself
Hello! I am a locally running language model...

> Summarize your answer above in three points
...

> /exit
Goodbye.
```

### 3. Generation and sampling options

| Option | Default | Description |
| --- | --- | --- |
| `--threads N` | llama.cpp default | Number of threads for GGUF inference |
| `--max-tokens N` | 256 | Maximum number of tokens to generate per turn |
| `--ctx-size N` | 8192 | Inference context window size in tokens |
| `--max-cache-mb N` | 256 | Maximum size in MiB of a saved KV prefix-cache snapshot; snapshots larger than the limit are not cached, and `0` disables prefix caching |
| `--temperature F` | 0 (greedy) | Sampling temperature; values greater than 0 enable sampling, so output may differ for the same input |
| `--top-p F` | 0.9 | Nucleus sampling threshold; applies only when `--temperature` > 0 |
| `--top-k N` | Disabled | Number of top-k sampling candidates |
| `--repeat-penalty F` | 1 (disabled) | Repetition penalty applied to the most recent 64 tokens |
| `--seed N` | Random | Random seed; setting a fixed value makes the same input reproducible |
| `--system "..."` | None | System prompt, injected only at the start of a conversation |

By default, wolf uses deterministic greedy decoding (temperature 0, no repetition penalty), so the same input always produces the same answer. To get more varied output, explicitly enable sampling, for example with `--temperature 0.7 --top-p 0.9`; use `--seed` to reproduce a particular generation.

Generation speed statistics are written to stderr; the response text is written to stdout.

`--max-cache-mb` limits the memory used by saved prefix-cache snapshots. It does not limit the inference context's KV cache or total GPU memory usage.

### 4. In-chat commands

| Command | Description |
| --- | --- |
| `/exit`, `/quit` | Exit the conversation |
| `/clear` | Clear the conversation history and reload the model (the system prompt is injected again on the next turn) |
| `/save <name>` | Save or update the current conversation context |
| `/help` | Show help for in-chat commands |

Pressing `Ctrl-C` while generating interrupts only the current response; the session remains active and you can continue asking questions. At the `>` prompt, `Ctrl-C` cancels the current input and displays a new prompt. Press `Ctrl-D` (or press Enter on an empty input) to exit.

### 5. Context management

During a chat, use `/save <name>` to save the full conversation. Restore it with `wolf chat <model-path> --context <name>`. The model path must match the saved context. If you omit inference options, the saved settings are used automatically; explicitly specified inference options must match the saved settings. Use `wolf context list` to list saved contexts and `wolf context delete <name>` to delete one. Use `--context-dir <path>` to override the default storage directory; this option is available for chat and context-management commands.

The default context directory is `$XDG_DATA_HOME/wolf/contexts` on Linux (or `~/.local/share/wolf/contexts` if unset), `~/Library/Application Support/wolf/contexts` on macOS, and `%LOCALAPPDATA%/wolf/contexts` on Windows (falling back to `%APPDATA%`). Contexts are stored as versioned JSON. When restoring a GGUF conversation, the prompt is reconstructed using the llama.cpp chat template.

As context usage approaches the model limit, wolf automatically truncates the oldest complete conversation turns, then llama.cpp reconstructs the prompt from the remaining history. If the input and generated output for a single turn exceed the entire context window, wolf prompts you to shorten the input. `/clear` clears the conversation history and llama.cpp's KV cache.

### 6. Inspect model files

```sh
# View help and supported formats
wolf help
wolf formats

# Inspect model information (up to 20 tensors by default)
wolf inspect ./models/model.gguf
wolf inspect /data/models/model.safetensors

# Show all tensors or output JSON
wolf inspect ./models/model.gguf --all
wolf inspect ./models/model.safetensors --json

# Write a text or JSON report to a file (overwrites an existing file and prints nothing to the terminal)
wolf inspect ./models/model.gguf --output report.txt
wolf inspect ./models/model.gguf --json -o report.json
```

`wolf load` is a compatibility alias for `wolf inspect`. Inspection validates the file header and tensor data ranges, and detects overlapping or out-of-bounds tensor data (for both GGUF and Safetensors). For GGUF quantization types whose sizes cannot be determined, overlap detection is skipped.

When `-o/--output` is specified, the report is written to a file in the selected format (text by default, JSON with `--json`) and is not printed to the terminal. An existing output file is overwritten. Without an output file, the report is displayed in the terminal.

Exit codes: `2` for argument or usage errors, `1` for other runtime errors, and `0` on success.

### 7. Run as a local API service (`wolf serve`)

#### ⚡ Why Wolf? (Unmatched Footprint)

Unlike traditional Python backends (like vLLM or Ollama), Wolf eliminates the Python interpreter and PyTorch overhead entirely. By combining a native C GGUF parser with zero-copy memory mapping (`mmap`) and a Rust async gateway:

- **Instant Model Loading:** Models are mapped into memory in milliseconds.
- **Edge-Ready RAM Footprint:** Running a 3B Q4 model consumes only **~529MB of process RAM** (loaded as OS Page Cache).
- **No Docker or Python:** A single binary, ready for Raspberry Pi, low-end VPS, and local dev machines.

`wolf serve` loads a GGUF model and provides an OpenAI-compatible HTTP API, so agents (Cline / Continue / Aider, etc.) can call the local model using a standard `openai` client:

```sh
wolf serve qwen2.5-0.5b --model-dir ./models --port 8080 --generate-key
```

On startup, it prints:

```text
Loaded model: ./models/qwen2.5-0.5b.gguf
Service started: http://localhost:8080
API Key: wolf-8f3a1c...
```

Press `Ctrl-C` to stop the service.

#### Service options

| Option | Default | Description |
| --- | --- | --- |
| `<name-or-path>` | Required | Model name or `.gguf` file path; the legacy syntax `--model <name-or-path>` is also supported |
| `--model-dir <directory>` | `.` | Directory to search for models by name |
| `--host <address>` | `127.0.0.1` | Address to listen on |
| `--port <port>` | `8080` | Port to listen on (`0` = assigned by the system) |
| `--max-concurrent-generations <N>` | `1` | Maximum number of concurrent model inference requests |
| `--generate-key` | Disabled | Generate an API key automatically and print it |
| `--api-key <KEY>` | None | Specify an API key (mutually exclusive with `--generate-key`) |

Service options can be combined with chat options (`--threads`, `--max-tokens`, `--ctx-size`, `--max-cache-mb`, `--temperature`, `--top-p`, `--top-k`, `--repeat-penalty`, `--seed`, `--system`) to set default inference and sampling parameters for the service. Request fields such as `max_tokens`, `temperature`, and `top_p` override their corresponding defaults; `--ctx-size` configures the context window when the service starts.

Generation requests are queued in the order they are received by the HTTP service. At most `--max-concurrent-generations` model inference requests run at once; the default value of `1` keeps requests serial. Increasing this limit may improve throughput but uses more memory and compute resources. **Warning:** each concurrent generation can significantly increase memory usage and cause memory pressure or out-of-memory failures, especially with large models or long contexts. Invalid requests are removed from the queue during validation and do not block subsequent requests.

It is recommended to pass the model name or path as the first argument after `wolf serve`; existing scripts can still use `--model <name-or-path>`.

If neither `--generate-key` nor `--api-key` is provided, the service can listen only on a loopback address (`127.0.0.1` / `localhost` / `::1`). Authentication must be explicitly enabled to listen on a non-loopback address (such as `0.0.0.0`), to avoid unintentionally exposing the service to the local network.

#### API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat completion, including streaming (SSE, `stream: true`) |
| `POST` | `/v1/completions` | Text completion; `prompt` is treated as a single user message (non-streaming only) |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/v1/models/:id` | Get a single model |
| `GET` | `/health` | Health check (no authentication required) |

Before inference, wolf tokenizes each rendered prompt (including chat-template and tool content) and checks that it plus the reserved `max_tokens` fits the configured context window. Older complete chat turns are truncated where possible; if the request still does not fit, the gateway returns HTTP 400 without passing it to llama.cpp.

All requests except `/health` must include an `Authorization: Bearer <API_KEY>` header. Error responses use an OpenAI-style JSON structure.

#### curl examples

```sh
# List models
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8080/v1/models

# Non-streaming chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'

# Streaming chat
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

#### OpenAI client example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="wolf-8f3a1c...",
)

response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

#### Function calling

`/v1/chat/completions` accepts OpenAI-style function tools and `tool_choice` values (`auto`, `none`, `required`, or a named function). It returns standard `message.tool_calls` for non-streaming requests and `tool_calls` deltas for SSE requests. Tool-enabled streams buffer the model output for classification while continuing to send SSE heartbeats; requests without tools retain the existing token-by-token stream. The calling application is responsible for executing the function and sending the assistant `tool_calls` plus `role: "tool"` result message in the next request. Function support and call formatting depend on the loaded model's llama.cpp chat template; if the output is not recognized as a valid declared tool call, wolf returns it as assistant text.

```python
response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls)
```

*Note: Local models (especially <7B) might struggle with complex tool schemas. For best results with Cline/Aider, we recommend Qwen2.5-Coder-7B or Llama-3.1-8B.*

#### Supported request fields

`POST /v1/chat/completions` supports: `model`, `messages` (`content` can be a string or an array of text), `stream`, `temperature`, `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `stop` (a string or array), `seed`, `n` (only `1`), `tools`, and `tool_choice`. Supported roles are `system` / `developer` (mapped to `system`) / `user` / `assistant` / `tool`.

The `model` field is optional. If provided, it must match the ID of the loaded model (the `id` returned by `/v1/models`, which defaults to the filename without `.gguf`); otherwise, the service returns 404. The service loads one model at a time, and concurrent requests are processed serially in arrival order.

Hot-loading multiple models, `/v1/embeddings`, `/v1/audio`, and `/v1/images` are unsupported.

## Supported formats and limitations

- **Chat inference:** GGUF only, powered by llama.cpp. Supported GGUF architectures and built-in chat templates depend on llama.cpp. The default backend is CPU; GPU backends can be built locally using the Vulkan/CUDA/ROCm/Metal Cargo features. oneAPI/SYCL is not supported yet.
- **File inspection:** GGUF v2/v3 and Safetensors. Displays metadata and tensor information (including estimated GGUF tensor sizes in bytes), and validates headers, data ranges, and overlapping tensor data.
- **Not supported:** Safetensors inference, ONNX, or PyTorch `.bin`/pickle. GGUF architecture and quantization support depends on the llama.cpp version; instruction/chat-tuned models are recommended.

Loading and running inference on large models requires sufficient memory. Speed depends on the model size and hardware. Quantized models are generally better suited to typical local machines.

## License

This project is distributed under the [Apache-2.0](LICENSE) license.

GGUF inference is provided by llama.cpp (MIT license); the Rust bindings `llama-cpp-4` are licensed under either MIT or Apache-2.0. When redistributing, retain the applicable upstream copyright and license notices.
