//! OpenAI-compatible request/response types and serialization helpers.

use crate::chat::ChatMessage;
use crate::options::ChatOptions;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    pub(crate) tools: Option<Vec<ToolDefinition>>,
    pub(crate) tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RequestMessage {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: Option<Content>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<RequestToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RequestToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ToolCallFunction {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ToolDefinition {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct FunctionDefinition {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum ToolChoice {
    Mode(String),
    Function(NamedToolChoice),
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct NamedToolChoice {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) function: NamedFunctionChoice,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct NamedFunctionChoice {
    pub(crate) name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResolvedToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

#[derive(Debug)]
pub(crate) enum ToolResponse {
    Text(String),
    Calls(Vec<ToolCall>),
}

#[derive(Debug, Deserialize)]
struct ParsedToolCall {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<Value>,
    #[serde(default)]
    function: Option<ParsedToolCallFunction>,
}

#[derive(Debug, Deserialize)]
struct ParsedToolCallFunction {
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct ParsedAssistantMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ParsedToolCall>>,
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
    Tool,
}

fn role_kind(role: &str) -> Result<RoleKind, ApiError> {
    match role {
        "system" | "developer" => Ok(RoleKind::System),
        "user" => Ok(RoleKind::User),
        "assistant" => Ok(RoleKind::Assistant),
        "tool" => Ok(RoleKind::Tool),
        other => Err(ApiError::bad_request(format!(
            "Unsupported role '{other}': wolf supports only system/developer/user/assistant/tool."
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
                        "Message content supports text parts only; received type '{}'.",
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
        if let Some(calls) = &message.tool_calls {
            if message.role != "assistant" {
                return Err(ApiError::bad_request(
                    "tool_calls can only appear in assistant messages.",
                ));
            }
            if calls.iter().any(|call| call.kind != "function") {
                return Err(ApiError::bad_request(
                    "Assistant history messages support only function-type tool_calls.",
                ));
            }
        }
        if message.role == "tool" && message.tool_call_id.as_deref().is_none_or(str::is_empty) {
            return Err(ApiError::bad_request(
                "Messages with role 'tool' must provide tool_call_id.",
            ));
        }
        let mut content = match &message.content {
            Some(content) => parse_content(content)?,
            None if message.role == "assistant" && message.tool_calls.is_some() => {
                serde_json::to_string(&message.tool_calls)
                    .map_err(|error| ApiError::bad_request(error.to_string()))?
            }
            None => {
                return Err(ApiError::bad_request(format!(
                    "Messages with role '{}' must provide content.",
                    message.role
                )));
            }
        };
        if message.role == "assistant"
            && message.content.is_some()
            && let Some(tool_calls) = &message.tool_calls
        {
            content.push('\n');
            content.push_str(
                &serde_json::to_string(tool_calls)
                    .map_err(|error| ApiError::bad_request(error.to_string()))?,
            );
        }
        match role_kind(&message.role)? {
            RoleKind::System => system_parts.push(content),
            RoleKind::User => messages.push(ChatMessage::new("user", content)),
            RoleKind::Assistant => messages.push(ChatMessage::new("assistant", content)),
            RoleKind::Tool => messages.push(ChatMessage::new("tool", content)),
        }
    }
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "messages must contain at least one user or assistant message.",
        ));
    }
    let system = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));
    Ok(PreparedChat { system, messages })
}

pub(crate) fn parse_tool_response(
    parsed: &str,
    raw_text: &str,
    tools: &[ToolDefinition],
    choice: &ResolvedToolChoice,
) -> ToolResponse {
    let parsed_message = serde_json::from_str::<ParsedAssistantMessage>(parsed).ok();
    let nested_message = parsed_message
        .as_ref()
        .and_then(|message| message.content.as_deref())
        .and_then(|content| serde_json::from_str::<ParsedAssistantMessage>(content).ok());
    let candidates = parsed_message
        .as_ref()
        .and_then(|message| message.tool_calls.as_deref())
        .or_else(|| {
            nested_message
                .as_ref()
                .and_then(|message| message.tool_calls.as_deref())
        });

    if let Some(calls) = candidates
        && !calls.is_empty()
        && let Some(calls) = convert_tool_calls(calls, tools, choice)
    {
        return ToolResponse::Calls(calls);
    }

    if let Ok(message) = serde_json::from_str::<ParsedAssistantMessage>(raw_text)
        && let Some(calls) = message.tool_calls
        && !calls.is_empty()
        && let Some(calls) = convert_tool_calls(&calls, tools, choice)
    {
        return ToolResponse::Calls(calls);
    }

    ToolResponse::Text(
        parsed_message
            .as_ref()
            .and_then(|message| message.content.clone())
            .or_else(|| nested_message.and_then(|message| message.content))
            .unwrap_or_else(|| raw_text.to_string()),
    )
}

pub(crate) fn parse_generated_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let json = extract_generated_tool_call_json(text)?;
    let message = serde_json::from_str::<ParsedAssistantMessage>(json).ok()?;
    let calls = message.tool_calls?;
    if calls.is_empty() {
        return None;
    }
    calls
        .iter()
        .map(|call| {
            if call.kind.as_deref().is_some_and(|kind| kind != "function") {
                return None;
            }
            let name = call
                .function
                .as_ref()
                .map(|function| &function.name)
                .or(call.name.as_ref())?;
            if name.is_empty() {
                return None;
            }
            let arguments = call
                .function
                .as_ref()
                .map(|function| &function.arguments)
                .or(call.arguments.as_ref())?;
            Some(ToolCall {
                id: tool_call_id(),
                kind: "function",
                function: ToolCallFunction {
                    name: name.clone(),
                    arguments: normalize_tool_arguments(arguments)?,
                },
            })
        })
        .collect()
}

fn extract_generated_tool_call_json(text: &str) -> Option<&str> {
    let mut text = text.trim();
    if let Some(body) = text.strip_prefix("```json") {
        text = body;
    } else if let Some(body) = text.strip_prefix("```") {
        text = body;
    }
    text = text.trim();
    if let Some(body) = text.strip_suffix("```") {
        text = body.trim();
    }

    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (start <= end).then(|| &text[start..=end])
}

fn convert_tool_calls(
    calls: &[ParsedToolCall],
    tools: &[ToolDefinition],
    choice: &ResolvedToolChoice,
) -> Option<Vec<ToolCall>> {
    calls
        .iter()
        .map(|call| {
            let name = call
                .function
                .as_ref()
                .map(|function| &function.name)
                .or(call.name.as_ref())?;
            if matches!(choice, ResolvedToolChoice::None)
                || call.kind.as_deref().is_some_and(|kind| kind != "function")
                || !tools.iter().any(|tool| tool.function.name == *name)
                || matches!(choice, ResolvedToolChoice::Named(selected) if selected != name)
            {
                return None;
            }
            let arguments = call
                .function
                .as_ref()
                .map(|function| &function.arguments)
                .or(call.arguments.as_ref())?;
            Some(ToolCall {
                id: tool_call_id(),
                kind: "function",
                function: ToolCallFunction {
                    name: name.clone(),
                    arguments: normalize_tool_arguments(arguments)?,
                },
            })
        })
        .collect()
}

fn normalize_tool_arguments(arguments: &Value) -> Option<String> {
    let arguments = match arguments {
        Value::String(value) => serde_json::from_str(value).ok()?,
        Value::Object(_) => arguments.clone(),
        _ => return None,
    };
    arguments.is_object().then(|| arguments.to_string())
}

pub(crate) fn tool_call_id() -> String {
    format!("call_{}", random_hex(12))
}

pub(crate) fn resolve_tool_choice(
    tools: &Option<Vec<ToolDefinition>>,
    tool_choice: &Option<ToolChoice>,
) -> Result<Option<ResolvedToolChoice>, ApiError> {
    let Some(tools) = tools else {
        return match tool_choice {
            None => Ok(None),
            Some(ToolChoice::Mode(value)) if value == "none" => Ok(None),
            Some(_) => Err(ApiError::bad_request(
                "tool_choice requires a non-empty tools list.",
            )),
        };
    };

    let mut names = std::collections::HashSet::new();
    for tool in tools {
        if tool.kind != "function" {
            return Err(ApiError::bad_request(format!(
                "Unsupported tool type '{}'; only function is supported.",
                tool.kind
            )));
        }
        let name = &tool.function.name;
        if name.is_empty() {
            return Err(ApiError::bad_request("Function tool name cannot be empty."));
        }
        if !names.insert(name) {
            return Err(ApiError::bad_request(format!(
                "Duplicate function tool name '{name}'."
            )));
        }
        if tool
            .function
            .parameters
            .as_ref()
            .is_some_and(|parameters| !parameters.is_object())
        {
            return Err(ApiError::bad_request(format!(
                "parameters for function '{name}' must be a JSON Schema object."
            )));
        }
    }

    if tools.is_empty() {
        return match tool_choice {
            None => Ok(None),
            Some(ToolChoice::Mode(value)) if value == "none" || value == "auto" => Ok(None),
            Some(_) => Err(ApiError::bad_request(
                "Cannot require tool calls when tools is empty.",
            )),
        };
    }

    let choice = match tool_choice {
        None => ResolvedToolChoice::Auto,
        Some(ToolChoice::Mode(value)) => match value.as_str() {
            "auto" => ResolvedToolChoice::Auto,
            "none" => ResolvedToolChoice::None,
            "required" => ResolvedToolChoice::Required,
            _ => {
                return Err(ApiError::bad_request(format!(
                    "Unsupported tool_choice value '{value}'."
                )));
            }
        },
        Some(ToolChoice::Function(value)) => {
            if value.kind != "function" {
                return Err(ApiError::bad_request(format!(
                    "Unsupported tool_choice type '{}'.",
                    value.kind
                )));
            }
            if !names.contains(&value.function.name) {
                return Err(ApiError::bad_request(format!(
                    "tool_choice specifies an undeclared function '{}'.",
                    value.function.name
                )));
            }
            ResolvedToolChoice::Named(value.function.name.clone())
        }
    };
    Ok(Some(choice))
}

pub(crate) fn build_completion_message(
    request: &CompletionRequest,
) -> Result<ChatMessage, ApiError> {
    let text = match &request.prompt {
        PromptValue::Text(text) => text.clone(),
        PromptValue::Many(parts) => parts.join("\n\n"),
    };
    if text.is_empty() {
        return Err(ApiError::bad_request("prompt cannot be empty."));
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
            return Err(ApiError::bad_request("temperature cannot be less than 0."));
        }
        options.temperature = temperature;
    }
    if let Some(top_p) = top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(ApiError::bad_request("top_p must be between 0 and 1."));
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
                "max_tokens and max_completion_tokens cannot both be provided.",
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
        return Err(ApiError::bad_request("wolf supports only n=1."));
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
        return Err(ApiError::bad_request("wolf does not support echo yet."));
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
            message: format!("Model '{requested}' does not exist; the loaded model is '{loaded}'."),
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
        r#"{"error":{"message":"Unable to generate error response","type":"server_error","param":null,"code":null}}"#
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
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) function: ToolCallFunction,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolCallDelta {
    pub(crate) index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub(crate) kind: Option<&'static str>,
    pub(crate) function: ToolCallFunctionDelta,
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arguments: Option<String>,
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
    fn parses_tool_calls_and_tool_result_history() {
        let request: ChatCompletionRequest = parse(
            r#"{
                "messages": [
                    {"role": "user", "content": "weather"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{\"city\":\"Tokyo\"}"}
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                }],
                "tool_choice": "auto"
            }"#,
        );
        let prepared = build_chat_messages(&request).unwrap();
        assert_eq!(prepared.messages.len(), 3);
        assert!(prepared.messages[1].content.contains("weather"));
        assert_eq!(prepared.messages[2].role, "tool");
        assert_eq!(
            resolve_tool_choice(&request.tools, &request.tool_choice).unwrap(),
            Some(ResolvedToolChoice::Auto)
        );
    }

    #[test]
    fn validates_tool_choice_and_tool_definitions() {
        let request: ChatCompletionRequest = parse(
            r#"{
                "tools": [
                    {"type": "function", "function": {"name": "weather"}},
                    {"type": "function", "function": {"name": "time"}}
                ],
                "tool_choice": {"type": "function", "function": {"name": "time"}}
            }"#,
        );
        assert_eq!(
            resolve_tool_choice(&request.tools, &request.tool_choice).unwrap(),
            Some(ResolvedToolChoice::Named("time".to_string()))
        );

        let invalid: ChatCompletionRequest = parse(
            r#"{
                "tools": [{"type": "function", "function": {"name": "weather"}}],
                "tool_choice": {"type": "function", "function": {"name": "missing"}}
            }"#,
        );
        assert!(resolve_tool_choice(&invalid.tools, &invalid.tool_choice).is_err());
        let invalid: ChatCompletionRequest = parse(r#"{"tool_choice": "required"}"#);
        assert!(resolve_tool_choice(&invalid.tools, &invalid.tool_choice).is_err());
    }

    #[test]
    fn parses_valid_tool_output_and_falls_back_for_invalid_output() {
        let request: ChatCompletionRequest = parse(
            r#"{
                "tools": [{
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                }]
            }"#,
        );
        let tools = request.tools.unwrap();
        let response = r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Tokyo\"}"}}]}"#;
        match parse_tool_response(
            response,
            "native model output",
            &tools,
            &ResolvedToolChoice::Auto,
        ) {
            ToolResponse::Calls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "weather");
                assert_eq!(calls[0].function.arguments, r#"{"city":"Tokyo"}"#);
            }
            ToolResponse::Text(text) => panic!("expected tool call, got {text}"),
        }

        let invalid = r#"{"tool_calls":[{"type":"function","function":{"name":"missing","arguments":"{}"}}]}"#;
        assert!(matches!(
            parse_tool_response(
                invalid,
                "plain answer",
                &tools,
                &ResolvedToolChoice::Auto
            ),
            ToolResponse::Text(text) if text == "plain answer"
        ));

        assert!(matches!(
            parse_tool_response(
                response,
                "plain answer",
                &tools,
                &ResolvedToolChoice::None
            ),
            ToolResponse::Text(text) if text == "plain answer"
        ));
        assert!(matches!(
            parse_tool_response(
                response,
                "plain answer",
                &tools,
                &ResolvedToolChoice::Named("other".to_string())
            ),
            ToolResponse::Text(text) if text == "plain answer"
        ));
    }

    #[test]
    fn parses_generated_tool_calls_in_compact_and_openai_shapes() {
        let compact = parse_generated_tool_calls(
            r#"{"tool_calls":[{"name":"weather","arguments":{"city":"Tokyo"}}]}"#,
        )
        .unwrap();
        let openai = parse_generated_tool_calls(
            r#"{"tool_calls":[{"type":"function","id":"ignored","function":{"name":"time","arguments":"{\"zone\":\"UTC\"}"}}]}"#,
        )
        .unwrap();

        assert_eq!(compact[0].function.name, "weather");
        assert_eq!(compact[0].function.arguments, r#"{"city":"Tokyo"}"#);
        assert_eq!(openai[0].function.name, "time");
        assert_eq!(openai[0].function.arguments, r#"{"zone":"UTC"}"#);
        assert!(compact[0].id.starts_with("call_"));
        assert!(openai[0].id.starts_with("call_"));
        assert_ne!(compact[0].id, openai[0].id);
    }

    #[test]
    fn parses_generated_tool_calls_from_fenced_json_and_surrounding_prose() {
        let fenced = parse_generated_tool_calls(
            "```json\n{\"tool_calls\":[{\"name\":\"weather\",\"arguments\":{\"city\":\"Tokyo\"}}]}\n```",
        )
        .unwrap();
        let verbose = parse_generated_tool_calls(
            "好的，我马上为你查询。\n{\"tool_calls\":[{\"name\":\"time\",\"arguments\":{\"zone\":\"UTC\"}}]}\n以上是结果。",
        )
        .unwrap();

        assert_eq!(fenced[0].function.name, "weather");
        assert_eq!(fenced[0].function.arguments, r#"{"city":"Tokyo"}"#);
        assert_eq!(verbose[0].function.name, "time");
        assert_eq!(verbose[0].function.arguments, r#"{"zone":"UTC"}"#);
    }

    #[test]
    fn rejects_invalid_generated_tool_calls_for_text_fallback() {
        for text in [
            "plain text",
            "This contains no JSON object.",
            r#"{"tool_calls":[]}"#,
            r#"{"tool_calls":[{"name":"weather","arguments":[]}]}"#,
            r#"{"tool_calls":[{"name":"","arguments":{}}]}"#,
            r#"{"tool_calls":[{"type":"other","name":"weather","arguments":{}}]}"#,
        ] {
            assert!(
                parse_generated_tool_calls(text).is_none(),
                "unexpected tool call parse for {text}"
            );
        }
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
                .contains("text parts only")
        );

        let request: ChatCompletionRequest = parse(
            r#"{
                "messages": [{"role": "function", "content": "result"}]
            }"#,
        );
        assert!(
            build_chat_messages(&request)
                .unwrap_err()
                .message
                .contains("Unsupported role")
        );

        let request: ChatCompletionRequest =
            parse(r#"{"messages": [{"role": "tool", "content": "result"}]}"#);
        assert!(
            build_chat_messages(&request)
                .unwrap_err()
                .message
                .contains("tool_call_id")
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

        let body = error_body(401, "Missing API key");
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["type"], "authentication_error");
        assert_eq!(value["error"]["message"], "Missing API key");
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
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains(r#""delta":{}"#), "{json}");
        assert!(json.contains(r#""finish_reason":"stop""#), "{json}");
    }

    #[test]
    fn serializes_tool_calls_and_omits_them_from_text_responses() {
        let tool_message = ChoiceMessage {
            role: "assistant",
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                kind: "function",
                function: ToolCallFunction {
                    name: "weather".to_string(),
                    arguments: r#"{"city":"Tokyo"}"#.to_string(),
                },
            }]),
        };
        let value: Value = serde_json::to_value(tool_message).unwrap();
        assert!(value["content"].is_null());
        assert_eq!(value["tool_calls"][0]["type"], "function");
        assert_eq!(
            value["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Tokyo"}"#
        );

        let text_message = ChoiceMessage {
            role: "assistant",
            content: Some("hello".to_string()),
            tool_calls: None,
        };
        let value = serde_json::to_value(text_message).unwrap();
        assert_eq!(value["content"], "hello");
        assert!(value.get("tool_calls").is_none());
    }

    #[test]
    fn api_key_uses_wolf_prefix_and_hex() {
        let key = generate_api_key();
        assert!(key.starts_with("wolf-"));
        assert_eq!(key.len(), "wolf-".len() + 32);
        assert!(key.chars().skip(5).all(|c| c.is_ascii_hexdigit()));
    }
}
