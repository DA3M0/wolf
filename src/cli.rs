//! 命令行参数解析与运行期选项。

use std::path::PathBuf;
use std::process::ExitCode;

pub const DEFAULT_MAX_TOKENS_PER_TURN: usize = 256;
pub const DEFAULT_CONTEXT_TOKENS: usize = 4096;
/// 默认温度 0:确定性贪心输出;需要更多样化的回答时显式开启采样。
pub const DEFAULT_TEMPERATURE: f64 = 0.0;
pub const DEFAULT_TOP_P: f64 = 0.9;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatOptions {
    pub threads: Option<usize>,
    pub max_tokens: usize,
    /// 采样温度;0 表示关闭采样,使用确定性贪心。
    pub temperature: f64,
    /// 核采样概率阈值;1 表示对完整分布采样。
    pub top_p: f64,
    /// top-k 候选数;0 表示关闭。
    pub top_k: usize,
    /// 重复惩罚;1 表示关闭。
    pub repeat_penalty: f32,
    /// 随机种子;None 表示按时间生成。
    pub seed: Option<u64>,
    /// 系统提示词,仅在首轮注入。
    pub system: Option<String>,
}

pub const DEFAULT_CHAT_OPTIONS: ChatOptions = ChatOptions {
    threads: None,
    max_tokens: DEFAULT_MAX_TOKENS_PER_TURN,
    temperature: DEFAULT_TEMPERATURE,
    top_p: DEFAULT_TOP_P,
    top_k: 0,
    repeat_penalty: DEFAULT_REPEAT_PENALTY,
    seed: None,
    system: None,
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
                if path.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("只能指定一个模型路径。"));
                }
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| {
        RunError::usage(
            "请指定一个 GGUF 模型文件或 Gemma 4 模型目录。用法：wolf <模型路径> [选项]，运行 `wolf help` 查看全部选项。",
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
        let (_, options) = parse_chat_args(&values).unwrap();
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
}
