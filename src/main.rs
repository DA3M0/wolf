//! `wolf` command-line interface.

mod chat;
mod cli;
mod context;
mod inspect;
mod options;
mod serve;

use cli::{ContextCommand, RunError};
use context::{ContextArchive, ContextStore};
use inspect::ModelInfo;
use options::ChatOptions;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("错误：{error}");
            error.exit_code()
        }
    }
}

fn run(args: Vec<String>) -> Result<(), RunError> {
    let Some(command) = args.first().map(String::as_str) else {
        print_help();
        return Ok(());
    };

    match command {
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        "formats" => {
            print_formats();
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("wolf {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "load" | "inspect" => {
            let (path, json, show_all) = cli::parse_load_args(&args[1..])?;
            let info = runtime(inspect::inspect_model(&path))?;
            if json {
                print_json(&info)?;
            } else {
                print_model(&info, show_all);
            }
            Ok(())
        }
        "chat" => {
            let (path, options, context_options) = cli::parse_chat_args(&args[1..])?;
            run_chat(&path, options, context_options)
        }
        "context" => {
            let (operation, options) = cli::parse_context_args(&args[1..])?;
            run_context_command(operation, options.directory)
        }
        "serve" => {
            let (serve_options, chat_options) = cli::parse_serve_args(&args[1..])?;
            runtime(serve::serve(serve_options, chat_options))
        }
        value if !value.starts_with('-') => {
            let (path, options, context_options) = cli::parse_chat_args(&args)?;
            run_chat(&path, options, context_options)
        }
        other => Err(RunError::usage(format!(
            "未知命令“{other}”。运行 `wolf help` 查看用法。"
        ))),
    }
}

fn run_chat(
    path: &Path,
    requested_options: ChatOptions,
    context_options: cli::ContextCliOptions,
) -> Result<(), RunError> {
    let chat_options_specified = context_options.chat_options_specified;
    let store = ContextStore::new(context_options.directory);
    let archive = match context_options.name {
        Some(name) => runtime(store.load(&name))?,
        None => runtime(ContextArchive::new(path, requested_options.clone()))?,
    };
    let options = if archive.name.is_empty() {
        requested_options
    } else {
        runtime(archive.validate_for_launch(path, &requested_options, chat_options_specified))?
    };
    runtime(chat::chat(path, options, archive, store))
}

fn run_context_command(
    operation: ContextCommand,
    directory: Option<PathBuf>,
) -> Result<(), RunError> {
    let store = ContextStore::new(directory);
    match operation {
        ContextCommand::List => {
            for archive in runtime(store.list())? {
                println!("{:<24} {:>5} 轮", archive.name, archive.turns.len());
            }
            Ok(())
        }
        ContextCommand::Delete(name) => {
            runtime(store.delete(&name))?;
            println!("已删除上下文“{name}”。");
            Ok(())
        }
    }
}

fn runtime<T>(result: Result<T, String>) -> Result<T, RunError> {
    result.map_err(RunError::Runtime)
}

fn print_formats() {
    println!("当前支持的权重格式：");
    println!("  .gguf          llama.cpp 对话推理和模型信息检查（默认 CPU）");
    println!("  .safetensors   模型信息检查（不支持对话推理）");
}

fn print_help() {
    println!(
        "wolf - Rust 本地模型加载器与对话 CLI\n\
\n\
用法：\n\
  wolf <模型路径>                         加载 GGUF 模型并进入多轮对话\n\
  wolf chat <模型路径>                    同上\n\
  wolf serve --model <名称> [选项]        以 OpenAI 兼容 API 服务运行\n\
  wolf load <模型文件> [--all] [--json]   加载并显示模型信息\n\
  wolf inspect <模型文件> [--all] [--json]（load 的别名）\n\
  wolf context list [--context-dir <路径>]  列出已保存上下文\n\
  wolf context delete <名称> [--context-dir <路径>]  删除上下文\n\
  wolf formats                            列出支持的格式\n\
  wolf version                            显示版本（--version / -V）\n\
  wolf help                               显示帮助\n\
\n\
对话选项：\n\
  --threads N          推理线程数（默认由后端决定）\n\
  --max-tokens N       每轮最大生成 token 数（默认 256）\n\
  --temperature F      采样温度（默认 0 = 确定性贪心；0.8 左右输出更多样但每次不同）\n\
  --top-p F            核采样概率阈值（默认 0.9，仅在 temperature > 0 时生效）\n\
  --top-k N            top-k 采样候选数（默认关闭）\n\
  --repeat-penalty F   重复惩罚（默认 1 = 关闭）\n\
  --seed N             随机种子，用于复现同一次生成\n\
  --system \"...\"       系统提示词，仅在对话首轮注入\n\
  --context <名称>     恢复已保存的对话上下文\n\
  --context-dir <路径> 上下文存档目录（也可用于 /save）\n\
\n\
服务选项（wolf serve）：\n\
  --model <名称>        模型名称或 .gguf 文件路径（必填）\n\
  --model-dir <目录>    按名称搜索模型的目录（默认 .）\n\
  --host <地址>         监听地址（默认 127.0.0.1）\n\
  --port <端口>         监听端口（默认 8080，0 为系统分配）\n\
  --generate-key        自动生成 API Key 并打印\n\
  --api-key <KEY>       指定 API Key\n\
  【以上选项可与对话选项共用，设置服务默认采样参数】\n\
\n\
服务 API 端点：\n\
  POST /v1/chat/completions    支持流式（SSE）\n\
  POST /v1/completions         仅非流式，prompt 作为单条消息\n\
  GET  /v1/models              列出已加载模型\n\
  GET  /v1/models/:id          获取单个模型\n\
  GET  /health                 健康检查（免鉴权）\n\
\n\
示例：\n\
  wolf ./models/llama-2-7b-chat.Q4_K_M.gguf\n\
  wolf ./models/model.gguf --threads 8 --max-tokens 128\n\
  wolf ./models/model.gguf --temperature 0.7 --top-k 50 --seed 42\n\
  wolf ./models/model.gguf --system \"你是一个简洁的中文助手\"\n\
  wolf ./models/model.gguf --temperature 0\n\
  wolf serve --model qwen2.5-0.5b --model-dir ./models --port 8080 --generate-key\n\
  wolf serve --model qwen2.5-0.5b.gguf --port 8080 --api-key wolf-secret --temperature 0\n\
  wolf /data/models/llama.gguf\n\
  wolf chat \"C:\\\\models\\\\llama.gguf\"\n\
  wolf load ./model.safetensors\n\
  wolf load ./model.gguf --all\n\
  wolf inspect ./model.safetensors --json\n\
\n\
对话中输入 /exit 退出。GGUF 推理由 llama.cpp 提供。Safetensors 仅支持文件检查。\n\
服务模式下 Ctrl-C 停止服务；不支持工具/函数调用。"
    );
}

fn print_model(info: &ModelInfo, show_all: bool) {
    println!("模型：{}", info.path.display());
    println!("格式：{}", info.format);
    println!("文件大小：{}", format_size(info.file_size));
    println!("张量数量：{}", info.tensors.len());
    println!("状态：已映射文件并验证头部");

    if !info.metadata.is_empty() {
        println!("\n元数据：");
        for (key, value) in &info.metadata {
            println!("  {key}: {value}");
        }
    }
    if !info.tensors.is_empty() {
        println!("\n张量：");
        let count = if show_all {
            info.tensors.len()
        } else {
            info.tensors.len().min(DEFAULT_TENSOR_LIMIT)
        };
        for tensor in info.tensors.iter().take(count) {
            let bytes = tensor
                .bytes
                .map(|size| format!("  {}", format_size(size)))
                .unwrap_or_default();
            println!(
                "  {}  {:?}  {}{}",
                tensor.name, tensor.shape, tensor.dtype, bytes
            );
        }
        if count < info.tensors.len() {
            println!(
                "  …另有 {} 个张量未显示（使用 --all 查看）",
                info.tensors.len() - count
            );
        }
    }
}

fn print_json(info: &ModelInfo) -> Result<(), RunError> {
    let output = serde_json::json!({
        "path": info.path,
        "format": info.format,
        "file_size": info.file_size,
        "tensor_count": info.tensors.len(),
        "metadata": info.metadata,
        "tensors": info.tensors.iter().map(|tensor| serde_json::json!({
            "name": tensor.name,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "bytes": tensor.bytes,
        })).collect::<Vec<_>>(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| RunError::Runtime(format!("无法生成 JSON 输出：{error}")))?
    );
    Ok(())
}

fn format_size(size: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = size as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{size} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

const DEFAULT_TENSOR_LIMIT: usize = 20;
