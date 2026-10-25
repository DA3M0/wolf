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
            eprintln!("Error: {error}");
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
        "load" | "inspect" => run_inspect_command(&args[1..]),
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
            "Unknown command '{other}'. Run `wolf help` for usage."
        ))),
    }
}

fn run_inspect_command(args: &[String]) -> Result<(), RunError> {
    let options = cli::parse_inspect_args(args)?;
    let info = runtime(inspect::inspect_model(&options.path))?;
    let report = if options.json {
        render_json(&info)?
    } else {
        render_model(&info, options.show_all)
    };
    if let Some(path) = options.output {
        write_report(&path, &report)?;
    } else {
        println!("{report}");
    }
    Ok(())
}

fn write_report(path: &Path, report: &str) -> Result<(), RunError> {
    std::fs::write(path, format!("{report}\n")).map_err(|error| {
        RunError::Runtime(format!(
            "Failed to write inspection report '{}': {error}",
            path.display()
        ))
    })
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
                println!("{:<24} {:>5} turns", archive.name, archive.turns.len());
            }
            Ok(())
        }
        ContextCommand::Delete(name) => {
            runtime(store.delete(&name))?;
            println!("Deleted context '{name}'.");
            Ok(())
        }
    }
}

fn runtime<T>(result: Result<T, String>) -> Result<T, RunError> {
    result.map_err(RunError::Runtime)
}

fn print_formats() {
    println!("Supported model formats:");
    println!("  .gguf          llama.cpp chat inference and model inspection (CPU by default)");
    println!("  .safetensors   Model inspection only (chat inference is not supported)");
}

fn print_help() {
    println!(
        "wolf - Local LLM loader and chat CLI written in Rust\n\
\n\
Usage:\n\
  wolf <model-path>                         Load a GGUF model and start a chat\n\
  wolf chat <model-path>                    Same as above\n\
  wolf serve <name-or-path> [options]       Run as an OpenAI-compatible API server\n\
  wolf inspect <model-file> [options]       Inspect a GGUF or Safetensors file\n\
  wolf load <model-file> [options]           Compatibility alias for inspect\n\
  wolf context list [--context-dir <path>]   List saved contexts\n\
  wolf context delete <name> [--context-dir <path>]  Delete a context\n\
  wolf formats                               List supported formats\n\
  wolf version                               Show the version (--version / -V)\n\
  wolf help                                  Show this help\n\
\n\
Chat options:\n\
  --threads N          Number of inference threads (backend default)\n\
  --max-tokens N       Maximum tokens to generate per turn (default: 256)\n\
  --temperature F      Sampling temperature (default: 0, deterministic; around 0.8 for varied output)\n\
  --top-p F            Nucleus sampling threshold (default: 0.9; only used when temperature > 0)\n\
  --top-k N            Number of top-k sampling candidates (default: disabled)\n\
  --repeat-penalty F   Repetition penalty (default: 1, disabled)\n\
  --seed N             Random seed for reproducible generation\n\
  --system \"...\"       System prompt, included only in the first turn\n\
  --context <name>     Resume a saved chat context\n\
  --context-dir <path> Context archive directory (also used by /save)\n\
\n\
Server options (wolf serve):\n\
  --model <name>        Model name or .gguf file path (legacy syntax)\n\
  --model-dir <dir>     Directory to search for models by name (default: .)\n\
  --host <address>      Address to listen on (default: 127.0.0.1)\n\
  --port <port>         Port to listen on (default: 8080; 0 assigns a free port)\n\
  --max-concurrent-generations <N> Maximum concurrent inference requests (default: 1)\n\
  --generate-key        Generate and print an API key\n\
  --api-key <KEY>       Set the API key\n\
  These options can be combined with chat options to set server defaults.\n\
\n\
Inspection options (wolf inspect):\n\
  --all                 Show all tensors (default: at most 20)\n\
  --json                Output JSON\n\
  -o, --output <file>   Save the report to a file (overwrites; no terminal output)\n\
\n\
Server API endpoints:\n\
  POST /v1/chat/completions    Supports streaming (SSE)\n\
  POST /v1/completions         Non-streaming only; prompt is sent as one message\n\
  GET  /v1/models              List loaded models\n\
  GET  /v1/models/:id          Get a single model\n\
  GET  /health                 Health check (no authentication required)\n\
\n\
Examples:\n\
  wolf ./models/llama-2-7b-chat.Q4_K_M.gguf\n\
  wolf ./models/model.gguf --threads 8 --max-tokens 128\n\
  wolf ./models/model.gguf --temperature 0.7 --top-k 50 --seed 42\n\
  wolf ./models/model.gguf --system \"You are a concise assistant.\"\n\
  wolf ./models/model.gguf --temperature 0\n\
  wolf serve qwen2.5-0.5b --model-dir ./models --port 8080 --generate-key\n\
  wolf serve qwen2.5-0.5b.gguf --port 8080 --api-key wolf-secret --temperature 0\n\
  wolf /data/models/llama.gguf\n\
  wolf chat \"C:\\\\models\\\\llama.gguf\"\n\
  wolf inspect ./model.safetensors\n\
  wolf inspect ./model.gguf --all\n\
  wolf inspect ./model.safetensors --json\n\
  wolf inspect ./model.gguf --json --output report.json\n\
\n\
Enter /exit to leave a chat. GGUF inference is provided by llama.cpp. Safetensors files can only be inspected.\n\
Press Ctrl-C to stop the server. Tool/function calling is not supported."
    );
}

fn render_model(info: &ModelInfo, show_all: bool) -> String {
    let mut output = String::new();
    use std::fmt::Write as _;

    writeln!(&mut output, "Model: {}", info.path.display()).unwrap();
    writeln!(&mut output, "Format: {}", info.format).unwrap();
    writeln!(&mut output, "File size: {}", format_size(info.file_size)).unwrap();
    writeln!(&mut output, "Tensor count: {}", info.tensors.len()).unwrap();
    writeln!(&mut output, "Status: File mapped and header validated").unwrap();

    if !info.metadata.is_empty() {
        output.push_str("\nMetadata:\n");
        for (key, value) in &info.metadata {
            writeln!(&mut output, "  {key}: {value}").unwrap();
        }
    }
    if !info.tensors.is_empty() {
        output.push_str("\nTensors:\n");
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
            writeln!(
                &mut output,
                "  {}  {:?}  {}{}",
                tensor.name, tensor.shape, tensor.dtype, bytes
            )
            .unwrap();
        }
        if count < info.tensors.len() {
            writeln!(
                &mut output,
                "  Omitted tensors: {} (use --all to show all)",
                info.tensors.len() - count
            )
            .unwrap();
        }
    }
    output.trim_end().to_string()
}

fn render_json(info: &ModelInfo) -> Result<String, RunError> {
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
    serde_json::to_string_pretty(&output)
        .map_err(|error| RunError::Runtime(format!("Failed to generate JSON output: {error}")))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn inspect_text_export_limits_tensors_by_default() {
        let info = ModelInfo {
            path: PathBuf::from("model.gguf"),
            file_size: 1,
            format: "GGUF".to_string(),
            tensors: (0..21)
                .map(|index| inspect::TensorInfo {
                    name: format!("tensor_{index}"),
                    shape: vec![1],
                    dtype: "F32".to_string(),
                    bytes: Some(4),
                })
                .collect(),
            metadata: Vec::new(),
        };
        let report = render_model(&info, false);
        assert!(report.contains("tensor_19"));
        assert!(!report.contains("tensor_20"));
        assert!(report.contains("Omitted tensors: 1"));
        assert!(render_model(&info, true).contains("tensor_20"));
    }

    #[test]
    fn inspect_json_export_preserves_structured_schema() {
        let info = ModelInfo {
            path: PathBuf::from("model.gguf"),
            file_size: 1,
            format: "GGUF".to_string(),
            tensors: Vec::new(),
            metadata: Vec::new(),
        };
        let report: serde_json::Value = serde_json::from_str(&render_json(&info).unwrap()).unwrap();
        assert_eq!(report["tensor_count"], 0);
        assert_eq!(report["format"], "GGUF");
    }

    #[test]
    fn inspect_report_export_overwrites_existing_file() {
        let path = std::env::temp_dir().join(format!(
            "wolf-report-{}-{}.txt",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, "old report").unwrap();

        write_report(&path, "new report").unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new report\n");
        std::fs::remove_file(path).unwrap();
    }
}
