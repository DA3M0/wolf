//! OpenAI-compatible request/response types and serialization helpers.

use crate::chat::ChatMessage;
use crate::options::ChatOptions;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::hash::{BuildHasher, Hasher};

/// 一个可渲染为 OpenAI 风格错误响应的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApiError {
    pub(crate) status: u16,
    pub(crate) message: String,
}

impl ApiError {
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum StopValue {
    Single(String),
    Many(Vec<String>),
}

/// `POST /v1/chat/completions` 的请求体。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct ChatCompletionRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<RequestMessage>,
    pub(crate) stream: bool,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<usize>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) max_completion_tokens: Option<usize>,
    pub(crate) stop: Option<StopValue>,
    pub(crate) seed: Option<u64>,
    pub(crate) n: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RequestMessage {
    pub(crate) role: String,
    pub(crate) content: Content,
}

/// `POST /v1/completions` 的请求体。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct CompletionRequest {
    pub(crate) model: String,
    pub(crate) prompt: PromptValue,
    pub(crate) stream: bool,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) stop: Option<StopValue>,
    pub(crate) seed: Option<u64>,
    pub(crate) echo: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum PromptValue {
    Text(String),
    Many(Vec<String>),
}

impl Default for PromptValue {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedChat {
    pub(crate) system: Option<String>,
    pub(crate) messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoleKind {
    System,
    User,
    Assistant,
}

fn role_kind(role: &str) -> Result<RoleKind, ApiError> {
    match role {
        "system" | "developer" => Ok(RoleKind::System),
        "user" => Ok(RoleKind::User),
        "assistant" => Ok(RoleKind::Assistant),
        other => Err(ApiError::bad_request(format!(
            "不支持的角色“{other}”：wolf 仅支持 system/developer/user/assistant，不支持工具调用。"
        ))),
    }
}

fn parse_content(content: &Content) -> Result<String, ApiError> {
    match content {
        Content::Text(text) => Ok(text.clone()),
        Content::Parts(parts) => {
            let mut joined = String::new();
            for part in parts {
                if part.kind != "text" {
                    return Err(ApiError::bad_request(format!(
                        "消息内容仅支持文本部分，收到类型“{}”。",
                        part.kind
                    )));
                }
                joined.push_str(&part.text);
            }
            Ok(joined)
        }
    }
}

pub(crate) fn build_chat_messages(
    request: &ChatCompletionRequest,
) -> Result<PreparedChat, ApiError> {
    let mut system_parts = Vec::new();
    let mut messages = Vec::new();
    for message in &request.messages {
        let content = parse_content(&message.content)?;
        match role_kind(&message.role)? {
            RoleKind::System => system_parts.push(content),
            RoleKind::User => messages.push(ChatMessage::new("user", content)),
            RoleKind::Assistant => messages.push(ChatMessage::new("assistant", content)),
        }
    }
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "messages 中至少需要一条 user 或 assistant 消息。",
        ));
    }
    let system = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));
    Ok(PreparedChat { system, messages })
}

pub(crate) fn build_completion_message(
    request: &CompletionRequest,
) -> Result<ChatMessage, ApiError> {
    let text = match &request.prompt {
        PromptValue::Text(text) => text.clone(),
        PromptValue::Many(parts) => parts.join("\n\n"),
    };
    if text.is_empty() {
        return Err(ApiError::bad_request("prompt 不能为空。"));
    }
    Ok(ChatMessage::new("user", text))
}

fn apply_sampling(
    options: &mut ChatOptions,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    max_tokens: Option<usize>,
    seed: Option<u64>,
) -> Result<(), ApiError> {
    if let Some(temperature) = temperature {
        if temperature < 0.0 {
            return Err(ApiError::bad_request("temperature 不能小于 0。"));
        }
        options.temperature = temperature;
    }
    if let Some(top_p) = top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(ApiError::bad_request("top_p 必须介于 0 和 1 之间。"));
        }
        options.top_p = top_p;
    }
    if let Some(top_k) = top_k {
        options.top_k = top_k;
    }
    if let Some(max_tokens) = max_tokens {
        options.max_tokens = max_tokens;
    }
    if let Some(seed) = seed {
        options.seed = Some(seed);
    }
    Ok(())
}

pub(crate) fn resolve_chat_options(
    defaults: &ChatOptions,
    request: &ChatCompletionRequest,
) -> Result<ChatOptions, ApiError> {
    let mut options = defaults.clone();
    let max_tokens = match (request.max_tokens, request.max_completion_tokens) {
        (Some(_), Some(_)) => {
            return Err(ApiError::bad_request(
                "max_tokens 与 max_completion_tokens 不能同时出现。",
            ));
        }
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    };
    apply_sampling(
        &mut options,
        request.temperature,
        request.top_p,
        request.top_k,
        max_tokens,
        request.seed,
    )?;
    if request.n.is_some_and(|n| n > 1) {
        return Err(ApiError::bad_request("wolf 仅支持 n=1。"));
    }
    Ok(options)
}

pub(crate) fn resolve_completion_options(
    defaults: &ChatOptions,
    request: &CompletionRequest,
) -> Result<ChatOptions, ApiError> {
    let mut options = defaults.clone();
    apply_sampling(
        &mut options,
        request.temperature,
        request.top_p,
        None,
        request.max_tokens,
        request.seed,
    )?;
    if request.echo {
        return Err(ApiError::bad_request("wolf 暂不支持 echo。"));
    }
    Ok(options)
}

pub(crate) fn stop_strings(stop: &Option<StopValue>) -> Vec<String> {
    match stop {
        Some(StopValue::Single(value)) => vec![value.clone()],
        Some(StopValue::Many(values)) => values.clone(),
        None => Vec::new(),
    }
}

/// 请求中携带了与已加载模型不一致的名称时返回 404。
pub(crate) fn ensure_model_matches(requested: &str, loaded: &str) -> Result<(), ApiError> {
    if !requested.is_empty() && requested != loaded {
        return Err(ApiError {
            status: 404,
            message: format!("模型“{requested}”不存在；当前服务加载的是“{loaded}”。"),
        });
    }
    Ok(())
}

pub(crate) fn error_kind(status: u16) -> &'static str {
    match status {
        400 | 404 | 405 => "invalid_request_error",
        401 => "authentication_error",
        500 => "server_error",
        _ => "server_error",
    }
}

pub(crate) fn error_body(status: u16, message: &str) -> String {
    serde_json::to_string(&json!({
        "error": {
            "message": message,
            "type": error_kind(status),
            "param": null,
            "code": null,
        }
    }))
    .unwrap_or_else(|_| {
        r#"{"error":{"message":"无法生成错误响应","type":"server_error","param":null,"code":null}}"#
            .to_string()
    })
}

// ---- 响应结构 ----

#[derive(Debug, Serialize)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) total_tokens: usize,
}

impl Usage {
    pub(crate) fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChatChoice>,
    pub(crate) usage: Usage,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatChoice {
    pub(crate) index: usize,
    pub(crate) message: ChoiceMessage,
    pub(crate) finish_reason: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChoiceMessage {
    pub(crate) role: &'static str,
    pub(crate) content: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionChunk {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChunkChoice {
    pub(crate) index: usize,
    pub(crate) delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct TextCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<TextChoice>,
    pub(crate) usage: Usage,
}

#[derive(Debug, Serialize)]
pub(crate) struct TextChoice {
    pub(crate) index: usize,
    pub(crate) text: String,
    pub(crate) finish_reason: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelEntry {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelList {
    pub(crate) object: &'static str,
    pub(crate) data: Vec<ModelEntry>,
}

pub(crate) fn completion_id() -> String {
    format!("chatcmpl-{}", random_hex(12))
}

pub(crate) fn text_completion_id() -> String {
    format!("cmpl-{}", random_hex(12))
}

pub(crate) fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

pub(crate) fn generate_api_key() -> String {
    format!("wolf-{}", random_hex(16))
}

/// 由操作系统熵播种的随机十六进制字符串(适合生成本地 API Key / 请求 ID)。
fn random_hex(bytes: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes * 2);
    for _ in 0..bytes {
        let value = random_u64() as u8;
        out.push(HEX[usize::from(value >> 4)] as char);
        out.push(HEX[usize::from(value & 0x0f)] as char);
    }
    out
}

fn random_u64() -> u64 {
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse<T: serde::de::DeserializeOwned>(json: &str) -> T {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parses_chat_request_with_content_array() {
        let request: ChatCompletionRequest = parse(
            r#"{
                "model": "qwen",
                "messages": [
                    {"role": "system", "content": "你是助手"},
                    {"role": "user", "content": [{"type": "text", "text": "你好"}]}
                ],
                "stream": true,
                "max_completion_tokens": 128
            }"#,
        );
        let prepared = build_chat_messages(&request).unwrap();
        assert_eq!(prepared.system.as_deref(), Some("你是助手"));
        assert_eq!(prepared.messages.len(), 1);
        assert_eq!(prepared.messages[0].content, "你好");
        assert_eq!(request.max_completion_tokens, Some(128));
        assert!(request.stream);
    }

    #[test]
    fn rejects_non_text_and_unknown_roles() {
        let request: ChatCompletionRequest = parse(
            r#"{
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
                ]
            }"#,
        );
        assert!(
            build_chat_messages(&request)
                .unwrap_err()
                .message
                .contains("仅支持文本")
        );

        let request: ChatCompletionRequest = parse(
            r#"{
                "messages": [{"role": "tool", "content": "result"}]
            }"#,
        );
        assert!(
            build_chat_messages(&request)
                .unwrap_err()
                .message
                .contains("不支持的角色")
        );
    }

    #[test]
    fn maps_developer_role_to_system() {
        let request: ChatCompletionRequest = parse(
            r#"{
                "messages": [
                    {"role": "developer", "content": "系统"},
                    {"role": "user", "content": "hi"}
                ]
            }"#,
        );
        let prepared = build_chat_messages(&request).unwrap();
        assert_eq!(prepared.system.as_deref(), Some("系统"));
    }

    #[test]
    fn normalizes_stop_values() {
        let stop = Some(StopValue::Single("END".to_string()));
        assert_eq!(stop_strings(&stop), vec!["END".to_string()]);
        let stop = Some(StopValue::Many(vec!["a".into(), "b".into()]));
        assert_eq!(stop_strings(&stop), vec!["a".to_string(), "b".to_string()]);
        assert!(stop_strings(&None).is_empty());
    }

    #[test]
    fn resolves_options_and_rejects_conflicts() {
        let defaults = ChatOptions::default();
        let request: ChatCompletionRequest = parse(
            r#"{
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_tokens": 300,
                "seed": 42
            }"#,
        );
        let options = resolve_chat_options(&defaults, &request).unwrap();
        assert_eq!(options.temperature, 0.7);
        assert_eq!(options.top_p, 0.8);
        assert_eq!(options.top_k, 40);
        assert_eq!(options.max_tokens, 300);
        assert_eq!(options.seed, Some(42));

        let request: ChatCompletionRequest =
            parse(r#"{"max_tokens": 10, "max_completion_tokens": 20}"#);
        assert!(resolve_chat_options(&defaults, &request).is_err());

        let request: ChatCompletionRequest = parse(r#"{"n": 2}"#);
        assert!(resolve_chat_options(&defaults, &request).is_err());
    }

    #[test]
    fn builds_completion_prompt() {
        let request: CompletionRequest = parse(r#"{"prompt": ["片段一", "片段二"], "echo": true}"#);
        let message = build_completion_message(&request).unwrap();
        assert_eq!(message.content, "片段一\n\n片段二");
        assert!(resolve_completion_options(&ChatOptions::default(), &request).is_err());
    }

    #[test]
    fn model_matching_and_error_body() {
        assert!(ensure_model_matches("other", "qwen").is_err());
        assert!(ensure_model_matches("", "qwen").is_ok());
        assert!(ensure_model_matches("qwen", "qwen").is_ok());

        let body = error_body(401, "缺少 API Key");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["type"], "authentication_error");
        assert_eq!(value["error"]["message"], "缺少 API Key");
    }

    #[test]
    fn sse_chunk_delta_omits_none_fields() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion.chunk",
            created: 1,
            model: "qwen".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains(r#""delta":{}"#), "{json}");
        assert!(json.contains(r#""finish_reason":"stop""#), "{json}");
    }

    #[test]
    fn api_key_uses_wolf_prefix_and_hex() {
        let key = generate_api_key();
        assert!(key.starts_with("wolf-"));
        assert_eq!(key.len(), "wolf-".len() + 32);
        assert!(key.chars().skip(5).all(|c| c.is_ascii_hexdigit()));
    }
}
