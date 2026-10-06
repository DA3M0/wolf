//! wolf - Rust 本地模型加载器与对话 CLI。

mod chat;
mod cli;
mod inspect;

use cli::RunError;
use std::env;
use std::path::Path;
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
            runtime(inspect::inspect_model(&path, json, show_all))?;
            Ok(())
        }
        "chat" => {
            let (path, options) = cli::parse_chat_args(&args[1..])?;
            run_chat(&path, options)
        }
        value if !value.starts_with('-') => {
            let (path, options) = cli::parse_chat_args(&args)?;
            run_chat(&path, options)
        }
        other => Err(RunError::usage(format!(
            "未知命令“{other}”。运行 `wolf help` 查看用法。"
        ))),
    }
}

fn run_chat(path: &Path, options: cli::ChatOptions) -> Result<(), RunError> {
    runtime(cli::configure_threads(options.threads))?;
    runtime(chat::chat(path, options))
}

fn runtime<T>(result: Result<T, String>) -> Result<T, RunError> {
    result.map_err(RunError::Runtime)
}

fn print_formats() {
    println!("当前支持的权重格式：");
    println!("  .gguf              对话推理（LLaMA、Qwen、Gemma、Phi 系 CPU）和模型信息检查");
    println!("  Gemma 4 模型目录  CPU 文本推理（Safetensors 权重）");
    println!("  .safetensors       模型信息检查（其他模型暂不支持推理）");

    println!("\nGemma 4 目录需包含 config.json、tokenizer.json 和 Safetensors 权重。");
}

fn print_help() {
    println!(
        "wolf - Rust 本地模型加载器与对话 CLI\n\
\n\
用法：\n\
  wolf <模型路径>                         加载 GGUF 或 Gemma 4 目录并进入多轮对话\n\
  wolf chat <模型路径>                    同上\n\
  wolf load <模型文件> [--all] [--json]   加载并显示模型信息\n\
  wolf inspect <模型文件> [--all] [--json]（load 的别名）\n\
  wolf formats                            列出支持的格式\n\
  wolf version                            显示版本（--version / -V）\n\
  wolf help                               显示帮助\n\
\n\
对话选项：\n\
  --threads N          CPU 线程数（默认使用 Rayon 默认值）\n\
  --max-tokens N       每轮最大生成 token 数（默认 256）\n\
  --temperature F      采样温度（默认 0 = 确定性贪心；0.8 左右输出更多样但每次不同）\n\
  --top-p F            核采样概率阈值（默认 0.9，仅在 temperature > 0 时生效）\n\
  --top-k N            top-k 采样候选数（默认关闭）\n\
  --repeat-penalty F   重复惩罚（默认 1 = 关闭）\n\
  --seed N             随机种子，用于复现同一次生成\n\
  --system \"...\"       系统提示词，仅在对话首轮注入\n\
\n\
示例：\n\
  wolf ./models/llama-2-7b-chat.Q4_K_M.gguf\n\
  wolf ./models/model.gguf --threads 8 --max-tokens 128\n\
  wolf ./models/model.gguf --temperature 0.7 --top-k 50 --seed 42\n\
  wolf ./models/model.gguf --system \"你是一个简洁的中文助手\"\n\
  wolf ./models/model.gguf --temperature 0\n\
  wolf ./models/gemma-4-E2B-it/\n\
  wolf /data/models/llama.gguf\n\
  wolf chat \"C:\\\\models\\\\llama.gguf\"\n\
  wolf load ./model.safetensors\n\
  wolf load ./model.gguf --all\n\
  wolf inspect ./model.safetensors --json\n\
\n\
对话中输入 /exit 退出。支持 LLaMA、Qwen、Gemma、Phi 系 GGUF 和 Gemma 4 Safetensors 目录的 CPU 推理。"
    );
}
