//! Shared options for GGUF chat generation.

use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_TOKENS_PER_TURN: usize = 256;
pub const DEFAULT_CONTEXT_TOKENS: usize = 8192;
/// 默认温度 0:确定性贪心输出;需要更多样化的回答时显式开启采样。
pub const DEFAULT_TEMPERATURE: f64 = 0.0;
pub const DEFAULT_TOP_P: f64 = 0.9;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.0;
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a concise assistant. Respond in English.";

// Serde uses this function when loading archives written before ctx_size was introduced.
// 反序列化缺少 ctx_size 的旧版会话归档时，Serde 会使用此默认值函数。
fn default_context_tokens() -> usize {
    DEFAULT_CONTEXT_TOKENS
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatOptions {
    pub threads: Option<usize>,
    #[serde(default = "default_context_tokens")]
    pub ctx_size: usize,
    #[serde(default)]
    pub max_cache_mb: Option<usize>,
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

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            system: Some(DEFAULT_SYSTEM_PROMPT.to_string()),
            ..DEFAULT_CHAT_OPTIONS
        }
    }
}

pub const DEFAULT_CHAT_OPTIONS: ChatOptions = ChatOptions {
    threads: None,
    ctx_size: DEFAULT_CONTEXT_TOKENS,
    max_cache_mb: None,
    max_tokens: DEFAULT_MAX_TOKENS_PER_TURN,
    temperature: DEFAULT_TEMPERATURE,
    top_p: DEFAULT_TOP_P,
    top_k: 0,
    repeat_penalty: DEFAULT_REPEAT_PENALTY,
    seed: None,
    system: None,
};
