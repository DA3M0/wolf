//! Command-line argument parsing and runtime options.

use crate::options::{ChatOptions, DEFAULT_CHAT_OPTIONS};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextCliOptions {
    pub name: Option<String>,
    pub directory: Option<PathBuf>,
    pub chat_options_specified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextCommand {
    List,
    Delete(String),
}

/// `wolf serve` 的服务参数(模型解析、监听地址与鉴权)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeOptions {
    /// 模型名称或 GGUF 文件路径。
    pub model: String,
    /// 解析模型名称时搜索的目录。
    pub model_dir: PathBuf,
    pub host: String,
    /// 0 表示由系统分配空闲端口。
    pub port: u16,
    pub generate_key: bool,
    pub api_key: Option<String>,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            model: String::new(),
            model_dir: PathBuf::from("."),
            host: "127.0.0.1".to_string(),
            port: 8080,
            generate_key: false,
            api_key: None,
        }
    }
}

/// 运行期错误:参数/用法错误以退出码 2 报告,其余运行时错误以退出码 1 报告。
#[derive(Debug)]
pub enum RunError {
    Usage(String),
    Runtime(String),
}

impl RunError {
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage(message.into())
    }

    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Usage(_) => ExitCode::from(2),
            Self::Runtime(_) => ExitCode::FAILURE,
        }
    }
}

impl std::fmt::Display for RunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(message) | Self::Runtime(message) => formatter.write_str(message),
        }
    }
}

pub fn parse_chat_args(
    args: &[String],
) -> Result<(PathBuf, ChatOptions, ContextCliOptions), RunError> {
    let mut path = None;
    let mut options = DEFAULT_CHAT_OPTIONS;
    let mut context = ContextCliOptions::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--threads" => {
                index += 1;
                options.threads = Some(parse_positive_option(args, index, "--threads")?);
                context.chat_options_specified = true;
            }
            "--max-tokens" => {
                index += 1;
                options.max_tokens = parse_positive_option(args, index, "--max-tokens")?;
                context.chat_options_specified = true;
            }
            "--temperature" => {
                index += 1;
                options.temperature = parse_float_option(args, index, "--temperature", 0.0)?;
                context.chat_options_specified = true;
            }
            "--top-p" => {
                index += 1;
                options.top_p = parse_bounded_float_option(args, index, "--top-p", 0.0, 1.0)?;
                context.chat_options_specified = true;
            }
            "--top-k" => {
                index += 1;
                options.top_k = parse_nonnegative_option(args, index, "--top-k")?;
                context.chat_options_specified = true;
            }
            "--repeat-penalty" => {
                index += 1;
                options.repeat_penalty =
                    parse_bounded_float_option(args, index, "--repeat-penalty", 1.0, f64::MAX)?
                        as f32;
                context.chat_options_specified = true;
            }
            "--seed" => {
                index += 1;
                options.seed = Some(parse_u64_option(args, index, "--seed")?);
                context.chat_options_specified = true;
            }
            "--system" => {
                index += 1;
                options.system = Some(parse_text_option(args, index, "--system")?);
                context.chat_options_specified = true;
            }
            "--context" => {
                index += 1;
                let name = parse_text_option(args, index, "--context")?;
                if context.name.replace(name).is_some() {
                    return Err(RunError::usage("--context 只能指定一次。"));
                }
            }
            "--context-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--context-dir 缺少路径。"))?;
                if value.is_empty() {
                    return Err(RunError::usage("--context-dir 路径不能为空。"));
                }
                if context.directory.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("--context-dir 只能指定一次。"));
                }
            }
            option if option.starts_with('-') => {
                return Err(RunError::usage(format!(
                    "未知选项“{option}”。运行 `wolf help` 查看用法。"
                )));
            }
            value => {
                if path.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("只能指定一个模型路径。"));
                }
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| {
        RunError::usage(
            "请指定一个 GGUF 模型文件。用法：wolf <模型路径> [选项]，运行 `wolf help` 查看全部选项。",
        )
    })?;
    Ok((path, options, context))
}

pub fn parse_context_args(
    args: &[String],
) -> Result<(ContextCommand, ContextCliOptions), RunError> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(RunError::usage(
            "缺少上下文操作。用法：wolf context list | delete <名称> [--context-dir <路径>]",
        ));
    };
    let mut context = ContextCliOptions::default();
    let mut name = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--context-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--context-dir 缺少路径。"))?;
                if value.is_empty() {
                    return Err(RunError::usage("--context-dir 路径不能为空。"));
                }
                if context.directory.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("--context-dir 只能指定一次。"));
                }
            }
            value if value.starts_with('-') => {
                return Err(RunError::usage(format!("未知选项“{value}”。")));
            }
            value => {
                if name.replace(value.to_string()).is_some() {
                    return Err(RunError::usage("上下文操作只能指定一个名称。"));
                }
            }
        }
        index += 1;
    }
    let operation = match command {
        "list" if name.is_none() => ContextCommand::List,
        "delete" => ContextCommand::Delete(name.ok_or_else(|| {
            RunError::usage(
                "缺少上下文名称。用法：wolf context delete <名称> [--context-dir <路径>]",
            )
        })?),
        "list" => return Err(RunError::usage("wolf context list 不接受上下文名称。")),
        other => {
            return Err(RunError::usage(format!(
                "未知上下文操作“{other}”。可用操作：list、delete。"
            )));
        }
    };
    Ok((operation, context))
}

pub fn parse_serve_args(args: &[String]) -> Result<(ServeOptions, ChatOptions), RunError> {
    let mut serve = ServeOptions::default();
    let mut options = DEFAULT_CHAT_OPTIONS;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--model 缺少模型名称或路径。"))?;
                if value.is_empty() {
                    return Err(RunError::usage("--model 不能为空。"));
                }
                if !serve.model.is_empty() {
                    return Err(RunError::usage("--model 只能指定一次。"));
                }
                serve.model = value.clone();
            }
            "--model-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--model-dir 缺少路径。"))?;
                if value.is_empty() {
                    return Err(RunError::usage("--model-dir 路径不能为空。"));
                }
                serve.model_dir = PathBuf::from(value);
            }
            "--host" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--host 缺少地址。"))?;
                if value.is_empty() {
                    return Err(RunError::usage("--host 不能为空。"));
                }
                serve.host = value.clone();
            }
            "--port" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--port 缺少端口号。"))?;
                serve.port = value.parse::<u16>().map_err(|_| {
                    RunError::usage(format!("--port 需要 0-65535 之间的端口，收到“{value}”。"))
                })?;
            }
            "--generate-key" => {
                if serve.generate_key {
                    return Err(RunError::usage("--generate-key 只能指定一次。"));
                }
                serve.generate_key = true;
            }
            "--api-key" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--api-key 缺少密钥。"))?;
                if value.is_empty() {
                    return Err(RunError::usage("--api-key 不能为空。"));
                }
                if serve.api_key.replace(value.clone()).is_some() {
                    return Err(RunError::usage("--api-key 只能指定一次。"));
                }
            }
            "--threads" => {
                index += 1;
                options.threads = Some(parse_positive_option(args, index, "--threads")?);
            }
            "--max-tokens" => {
                index += 1;
                options.max_tokens = parse_positive_option(args, index, "--max-tokens")?;
            }
            "--temperature" => {
                index += 1;
                options.temperature = parse_float_option(args, index, "--temperature", 0.0)?;
            }
            "--top-p" => {
                index += 1;
                options.top_p = parse_bounded_float_option(args, index, "--top-p", 0.0, 1.0)?;
            }
            "--top-k" => {
                index += 1;
                options.top_k = parse_nonnegative_option(args, index, "--top-k")?;
            }
            "--repeat-penalty" => {
                index += 1;
                options.repeat_penalty =
                    parse_bounded_float_option(args, index, "--repeat-penalty", 1.0, f64::MAX)?
                        as f32;
            }
            "--seed" => {
                index += 1;
                options.seed = Some(parse_u64_option(args, index, "--seed")?);
            }
            "--system" => {
                index += 1;
                options.system = Some(parse_text_option(args, index, "--system")?);
            }
            option if option.starts_with('-') => {
                return Err(RunError::usage(format!(
                    "未知选项“{option}”。运行 `wolf help` 查看用法。"
                )));
            }
            value => {
                return Err(RunError::usage(format!(
                    "serve 不接受位置参数“{value}”；请用 --model 指定模型。"
                )));
            }
        }
        index += 1;
    }

    if serve.model.is_empty() {
        return Err(RunError::usage(
            "请用 --model 指定模型名称或路径。用法：wolf serve --model <名称或路径> [选项]",
        ));
    }
    if serve.generate_key && serve.api_key.is_some() {
        return Err(RunError::usage(
            "--generate-key 与 --api-key 不能同时使用。",
        ));
    }
    Ok((serve, options))
}

pub fn parse_load_args(args: &[String]) -> Result<(PathBuf, bool, bool), RunError> {
    let mut path = None;
    let mut json = false;
    let mut show_all = false;

    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "--all" => show_all = true,
            value if value.starts_with('-') => {
                return Err(RunError::usage(format!(
                    "未知选项“{value}”。运行 `wolf help` 查看用法。"
                )));
            }
            value => {
                if path.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("只能指定一个模型文件。"));
                }
            }
        }
    }

    let path = path.ok_or_else(|| {
        RunError::usage("缺少模型文件路径。用法：wolf load <模型文件> [--all] [--json]")
    })?;
    Ok((path, json, show_all))
}

pub fn parse_positive_option(
    args: &[String],
    index: usize,
    option: &str,
) -> Result<usize, RunError> {
    let value = args
        .get(index)
        .ok_or_else(|| RunError::usage(format!("{option} 缺少数值。")))?;
    let parsed = value
        .parse::<usize>()
        .map_err(|_| RunError::usage(format!("{option} 需要正整数，收到“{value}”。")))?;
    if parsed == 0 {
        return Err(RunError::usage(format!("{option} 必须大于 0。")));
    }
    Ok(parsed)
}

pub fn parse_nonnegative_option(
    args: &[String],
    index: usize,
    option: &str,
) -> Result<usize, RunError> {
    let value = args
        .get(index)
        .ok_or_else(|| RunError::usage(format!("{option} 缺少数值。")))?;
    value
        .parse::<usize>()
        .map_err(|_| RunError::usage(format!("{option} 需要非负整数，收到“{value}”。")))
}

fn parse_u64_option(args: &[String], index: usize, option: &str) -> Result<u64, RunError> {
    let value = args
        .get(index)
        .ok_or_else(|| RunError::usage(format!("{option} 缺少数值。")))?;
    value
        .parse::<u64>()
        .map_err(|_| RunError::usage(format!("{option} 需要非负整数，收到“{value}”。")))
}

fn parse_text_option(args: &[String], index: usize, option: &str) -> Result<String, RunError> {
    args.get(index).cloned().ok_or_else(|| {
        RunError::usage(format!(
            "{option} 缺少内容，例如 --system \"你是一个简洁的助手\"。"
        ))
    })
}

/// 解析下界为 `min` 的浮点数(拒绝 NaN)。
fn parse_float_option(
    args: &[String],
    index: usize,
    option: &str,
    min: f64,
) -> Result<f64, RunError> {
    let value = parse_float_raw(args, index, option)?;
    if value < min {
        return Err(RunError::usage(format!(
            "{option} 不能小于 {min}，收到“{value}”。"
        )));
    }
    Ok(value)
}

/// 解析介于 `min` 与 `max` 之间(含边界)的浮点数。
fn parse_bounded_float_option(
    args: &[String],
    index: usize,
    option: &str,
    min: f64,
    max: f64,
) -> Result<f64, RunError> {
    let value = parse_float_raw(args, index, option)?;
    if !(min..=max).contains(&value) {
        return Err(RunError::usage(format!(
            "{option} 需要介于 {min} 和 {max} 之间的数值，收到“{value}”。"
        )));
    }
    Ok(value)
}

fn parse_float_raw(args: &[String], index: usize, option: &str) -> Result<f64, RunError> {
    let value = args
        .get(index)
        .ok_or_else(|| RunError::usage(format!("{option} 缺少数值。")))?;
    let parsed = value
        .parse::<f64>()
        .map_err(|_| RunError::usage(format!("{option} 需要数值，收到“{value}”。")))?;
    if parsed.is_nan() {
        return Err(RunError::usage(format!("{option} 不能是 NaN。")));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::DEFAULT_CHAT_OPTIONS;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn parses_chat_speed_options() {
        let values = args(&["model.gguf", "--threads", "8", "--max-tokens", "64"]);
        let (path, options, context) = parse_chat_args(&values).unwrap();
        assert_eq!(path, PathBuf::from("model.gguf"));
        assert!(context.chat_options_specified);
        assert_eq!(
            options,
            ChatOptions {
                threads: Some(8),
                max_tokens: 64,
                ..DEFAULT_CHAT_OPTIONS
            }
        );
    }

    #[test]
    fn parses_sampling_and_system_options() {
        let values = args(&[
            "model.gguf",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "40",
            "--repeat-penalty",
            "1.3",
            "--seed",
            "7",
            "--system",
            "你是一个简洁的助手",
        ]);
        let (_, options, _) = parse_chat_args(&values).unwrap();
        assert_eq!(options.temperature, 0.0);
        assert_eq!(options.top_p, 1.0);
        assert_eq!(options.top_k, 40);
        assert_eq!(options.repeat_penalty, 1.3);
        assert_eq!(options.seed, Some(7));
        assert_eq!(options.system.as_deref(), Some("你是一个简洁的助手"));
    }

    #[test]
    fn defaults_are_deterministic() {
        assert_eq!(DEFAULT_CHAT_OPTIONS.temperature, 0.0);
        assert_eq!(DEFAULT_CHAT_OPTIONS.top_p, 0.9);
        assert_eq!(DEFAULT_CHAT_OPTIONS.top_k, 0);
        assert_eq!(DEFAULT_CHAT_OPTIONS.repeat_penalty, 1.0);
        assert_eq!(DEFAULT_CHAT_OPTIONS.seed, None);
        assert_eq!(DEFAULT_CHAT_OPTIONS.system, None);
    }

    #[test]
    fn rejects_out_of_range_sampling_values() {
        for invalid in [
            &["--temperature", "-0.1"][..],
            &["--top-p", "1.5"][..],
            &["--repeat-penalty", "0.9"][..],
            &["--temperature", "abc"][..],
            &["--top-k", "-1"][..],
            &["--system"][..],
        ] {
            let values = args(&[vec!["model.gguf"], invalid.to_vec()].concat());
            let error = parse_chat_args(&values).unwrap_err();
            assert!(matches!(error, RunError::Usage(_)), "case: {invalid:?}");
        }
    }

    #[test]
    fn rejects_zero_and_missing_numeric_options() {
        let values = args(&["model.gguf", "--threads", "0"]);
        assert!(
            parse_chat_args(&values)
                .unwrap_err()
                .to_string()
                .contains("必须大于 0")
        );
        let values = args(&["model.gguf", "--threads"]);
        assert!(
            parse_chat_args(&values)
                .unwrap_err()
                .to_string()
                .contains("缺少数值")
        );
    }

    #[test]
    fn rejects_missing_model_path() {
        assert!(
            parse_chat_args(&args(&["--max-tokens", "8"]))
                .unwrap_err()
                .to_string()
                .contains("请指定")
        );
    }

    #[test]
    fn reports_usage_errors_with_usage_variant() {
        let error = parse_chat_args(&args(&["--bogus"])).unwrap_err();
        assert!(matches!(error, RunError::Usage(_)));
    }

    #[test]
    fn parses_context_options_and_management_commands() {
        let (_, _, context) = parse_chat_args(&args(&[
            "model.gguf",
            "--context",
            "work",
            "--context-dir",
            "/tmp/wolf-contexts",
        ]))
        .unwrap();
        assert_eq!(context.name.as_deref(), Some("work"));
        assert_eq!(context.directory, Some(PathBuf::from("/tmp/wolf-contexts")));
        assert!(!context.chat_options_specified);
        let (_, _, context) =
            parse_chat_args(&args(&["model.gguf", "--max-tokens", "256"])).unwrap();
        assert!(context.chat_options_specified);

        let (operation, options) = parse_context_args(&args(&[
            "delete",
            "work",
            "--context-dir",
            "/tmp/wolf-contexts",
        ]))
        .unwrap();
        assert_eq!(operation, ContextCommand::Delete("work".into()));
        assert_eq!(options.directory, Some(PathBuf::from("/tmp/wolf-contexts")));
        assert_eq!(
            parse_context_args(&args(&["list"])).unwrap().0,
            ContextCommand::List
        );
    }

    #[test]
    fn parses_serve_defaults() {
        let (serve, options) = parse_serve_args(&args(&["--model", "qwen2.5-0.5b"])).unwrap();
        assert_eq!(serve.model, "qwen2.5-0.5b");
        assert_eq!(serve.model_dir, PathBuf::from("."));
        assert_eq!(serve.host, "127.0.0.1");
        assert_eq!(serve.port, 8080);
        assert!(!serve.generate_key);
        assert_eq!(serve.api_key, None);
        assert_eq!(options, DEFAULT_CHAT_OPTIONS);
    }

    #[test]
    fn parses_serve_options_and_generation_overrides() {
        let (serve, options) = parse_serve_args(&args(&[
            "--model",
            "qwen2.5-0.5b",
            "--model-dir",
            "./models",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--api-key",
            "wolf-secret",
            "--temperature",
            "0.7",
            "--max-tokens",
            "512",
        ]))
        .unwrap();
        assert_eq!(serve.model_dir, PathBuf::from("./models"));
        assert_eq!(serve.host, "0.0.0.0");
        assert_eq!(serve.port, 9000);
        assert_eq!(serve.api_key.as_deref(), Some("wolf-secret"));
        assert!(!serve.generate_key);
        assert_eq!(options.temperature, 0.7);
        assert_eq!(options.max_tokens, 512);
    }

    #[test]
    fn accepts_ephemeral_port_and_generate_key() {
        let (serve, _) = parse_serve_args(&args(&[
            "--model",
            "m.gguf",
            "--port",
            "0",
            "--generate-key",
        ]))
        .unwrap();
        assert_eq!(serve.port, 0);
        assert!(serve.generate_key);
    }

    #[test]
    fn serve_rejects_mutually_exclusive_keys() {
        let error = parse_serve_args(&args(&[
            "--model",
            "m.gguf",
            "--generate-key",
            "--api-key",
            "wolf-secret",
        ]))
        .unwrap_err();
        assert!(matches!(error, RunError::Usage(_)));
        assert!(error.to_string().contains("不能同时使用"));
    }

    #[test]
    fn serve_rejects_missing_model_and_invalid_port() {
        assert!(matches!(
            parse_serve_args(&args(&["--port", "8080"])).unwrap_err(),
            RunError::Usage(_)
        ));
        let error = parse_serve_args(&args(&["--model", "m.gguf", "--port", "70000"])).unwrap_err();
        assert!(matches!(error, RunError::Usage(_)));
        assert!(error.to_string().contains("0-65535"));
    }

    #[test]
    fn serve_rejects_positional_arguments() {
        let error = parse_serve_args(&args(&["--model", "m.gguf", "extra"]));
        assert!(error.is_err());
        assert!(error.unwrap_err().to_string().contains("位置参数"));
    }
}
