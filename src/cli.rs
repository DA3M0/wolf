//! 命令行参数解析与运行期选项。

use std::path::PathBuf;
use std::process::ExitCode;

pub const DEFAULT_MAX_TOKENS_PER_TURN: usize = 256;
pub const DEFAULT_CONTEXT_TOKENS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatOptions {
    pub threads: Option<usize>,
    pub max_tokens: usize,
}

pub const DEFAULT_CHAT_OPTIONS: ChatOptions = ChatOptions {
    threads: None,
    max_tokens: DEFAULT_MAX_TOKENS_PER_TURN,
};

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

pub fn parse_chat_args(args: &[String]) -> Result<(PathBuf, ChatOptions), RunError> {
    let mut path = None;
    let mut options = DEFAULT_CHAT_OPTIONS;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--threads" => {
                index += 1;
                options.threads = Some(parse_positive_option(args, index, "--threads")?);
            }
            "--max-tokens" => {
                index += 1;
                options.max_tokens = parse_positive_option(args, index, "--max-tokens")?;
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
            "请指定一个 GGUF 模型文件或 Gemma 4 模型目录。用法：wolf <模型路径> [--threads N] [--max-tokens N]",
        )
    })?;
    Ok((path, options))
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

pub fn configure_threads(threads: Option<usize>) -> Result<(), String> {
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("无法设置 CPU 推理线程数：{error}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn parses_chat_speed_options() {
        let values = args(&["model.gguf", "--threads", "8", "--max-tokens", "64"]);
        let (path, options) = parse_chat_args(&values).unwrap();
        assert_eq!(path, PathBuf::from("model.gguf"));
        assert_eq!(
            options,
            ChatOptions {
                threads: Some(8),
                max_tokens: 64
            }
        );
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
}
