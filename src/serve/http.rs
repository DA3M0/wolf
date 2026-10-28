//! HTTP routing, authentication and streaming glue for `wolf serve`.

use crate::chat::{Engine, Generation, TextSink};
use crate::options::ChatOptions;
use crate::serve::openai::{self, *};
use crate::serve::queue::{QueueGuard, QueueTicket};
use bytes::Bytes;
use llama_cpp_4::chat::ToolChoice as LlamaToolChoice;
use llama_cpp_4::prelude::{ChatParams, LlamaToken};
use serde::Serialize;
use std::io::{self, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::time::Duration;
use tiny_http::{Header, Method, Request, Response, StatusCode};

const SSE_CHANNEL_CAPACITY: usize = 8;
const SSE_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(4);

pub(crate) struct ServerState {
    pub(crate) engine: Arc<Engine>,
    pub(crate) default_options: ChatOptions,
    pub(crate) model_id: String,
    pub(crate) api_key: Option<String>,
    pub(crate) generation_queue: Arc<super::queue::FifoQueue>,
    pub(crate) created: u64,
}

pub(crate) fn is_generation_request(request: &Request) -> bool {
    if request.method() != &Method::Post {
        return false;
    }
    let raw = request.url();
    let path = raw.split('?').next().unwrap_or(raw);
    let path = path.strip_suffix('/').unwrap_or(path);
    matches!(path, "/v1/chat/completions" | "/v1/completions")
}

pub(crate) fn handle(
    state: Arc<ServerState>,
    request: Request,
    generation_ticket: Option<QueueTicket>,
) {
    if request.method() == &Method::Options {
        let _ = request.respond(options_response());
        return;
    }

    let method = request.method().clone();
    let raw = request.url();
    let path = raw.split('?').next().unwrap_or(raw);
    let path = path.strip_suffix('/').unwrap_or(path);
    let path = if path.is_empty() { "/" } else { path };

    // /health is auth-free so probes and load balancers can check liveness.
    if method == Method::Get && path == "/health" {
        respond_json(request, StatusCode(200), health_body());
        return;
    }

    if !authorized(&request, state.api_key.as_deref()) {
        respond_error(
            request,
            401,
            "Missing or invalid API key; provide a Bearer token in the Authorization header.",
        );
        return;
    }

    match route(&method, path) {
        Ok(RouteEnum::Health) => respond_json(request, StatusCode(200), health_body()),
        Ok(RouteEnum::ListModels) => {
            let body = model_list_string(&state);
            respond_json(request, StatusCode(200), body);
        }
        Ok(RouteEnum::GetModel(id)) => {
            if id == state.model_id {
                let body = single_model_string(&state);
                respond_json(request, StatusCode(200), body);
            } else {
                respond_error(
                    request,
                    404,
                    &format!(
                        "Model '{id}' does not exist; the loaded model is '{}'.",
                        state.model_id
                    ),
                );
            }
        }
        Ok(RouteEnum::ChatCompletions) => handle_chat_completions(
            state,
            request,
            generation_ticket.expect("generation request receives a queue ticket"),
        ),
        Ok(RouteEnum::Completions) => handle_completions(
            state,
            request,
            generation_ticket.expect("generation request receives a queue ticket"),
        ),
        Err(status) => respond_error(
            request,
            status.0,
            if status.0 == 405 {
                "Method not allowed."
            } else {
                "Endpoint not found."
            },
        ),
    }
}

// ---- routing ----

#[derive(Debug, PartialEq, Eq)]
enum RouteEnum {
    Health,
    ListModels,
    GetModel(String),
    ChatCompletions,
    Completions,
}

fn route(method: &Method, path: &str) -> Result<RouteEnum, StatusCode> {
    if method == &Method::Get && path == "/health" {
        return Ok(RouteEnum::Health);
    }
    if method == &Method::Get && path == "/v1/models" {
        return Ok(RouteEnum::ListModels);
    }
    if method == &Method::Get
        && let Some(id) = path.strip_prefix("/v1/models/")
        && !id.is_empty()
        && !id.contains('/')
    {
        return Ok(RouteEnum::GetModel(id.to_string()));
    }
    if method == &Method::Post && path == "/v1/chat/completions" {
        return Ok(RouteEnum::ChatCompletions);
    }
    if method == &Method::Post && path == "/v1/completions" {
        return Ok(RouteEnum::Completions);
    }

    let known = path == "/health"
        || path == "/v1/models"
        || path.starts_with("/v1/models/")
        || path == "/v1/chat/completions"
        || path == "/v1/completions";
    if known {
        Err(StatusCode(405))
    } else {
        Err(StatusCode(404))
    }
}

// ---- auth ----

fn authorized(request: &Request, expected: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    let header = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"));
    let Some(header) = header else {
        return false;
    };
    let Some(token) = header.value.as_str().strip_prefix("Bearer ") else {
        return false;
    };
    constant_time_eq(token.as_bytes(), expected.as_bytes())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ---- helpers ----

fn health_body() -> String {
    r#"{"status":"ok"}"#.to_string()
}

fn model_list_string(state: &ServerState) -> String {
    let list = ModelList {
        object: "list",
        data: vec![ModelEntry {
            id: state.model_id.clone(),
            object: "model",
            created: state.created,
            owned_by: "wolf",
        }],
    };
    serde_json::to_string(&list).unwrap_or_default()
}

fn single_model_string(state: &ServerState) -> String {
    let entry = ModelEntry {
        id: state.model_id.clone(),
        object: "model",
        created: state.created,
        owned_by: "wolf",
    };
    serde_json::to_string(&entry).unwrap_or_default()
}

fn cors_headers() -> Vec<Header> {
    vec![
        Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).expect("cors header"),
        Header::from_bytes(
            &b"Access-Control-Allow-Methods"[..],
            &b"GET, POST, OPTIONS"[..],
        )
        .expect("cors methods"),
        Header::from_bytes(
            &b"Access-Control-Allow-Headers"[..],
            &b"Content-Type, Authorization"[..],
        )
        .expect("cors headers"),
    ]
}

fn options_response() -> Response<io::Empty> {
    let mut response = Response::new(StatusCode(204), cors_headers(), io::empty(), Some(0), None);
    response = response.with_header(
        Header::from_bytes(&b"Allow"[..], &b"GET, POST, OPTIONS"[..]).expect("allow header"),
    );
    response
}

fn json_headers() -> Vec<Header> {
    let mut headers = cors_headers();
    headers.push(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("ct"));
    headers
}

fn respond_json(request: Request, status: StatusCode, body: String) {
    let data = body.into_bytes();
    let len = data.len();
    let response = Response::new(
        status,
        json_headers(),
        io::Cursor::new(data),
        Some(len),
        None,
    );
    let _ = request.respond(response);
}

fn respond_error(request: Request, status: u16, message: &str) {
    let body = error_body(status, message);
    respond_json(request, StatusCode(status), body);
}

fn read_body(request: &mut Request) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    request.as_reader().read_to_end(&mut body)?;
    Ok(body)
}

// ---- chat completions ----

fn handle_chat_completions(
    state: Arc<ServerState>,
    request: Request,
    generation_ticket: QueueTicket,
) {
    let mut request = request;
    let body = match read_body(&mut request) {
        Ok(body) => body,
        Err(error) => {
            respond_error(
                request,
                400,
                &format!("Failed to read request body: {error}"),
            );
            return;
        }
    };
    let api_request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(error) => {
            respond_error(
                request,
                400,
                &format!("Request body is not valid JSON: {error}"),
            );
            return;
        }
    };
    if let Err(error) = ensure_model_matches(&api_request.model, &state.model_id) {
        respond_error(request, error.status, &error.message);
        return;
    }
    let tool_choice = match resolve_tool_choice(&api_request.tools, &api_request.tool_choice) {
        Ok(choice) => choice,
        Err(error) => {
            respond_error(request, error.status, &error.message);
            return;
        }
    };

    let prepared = match build_chat_messages(&api_request) {
        Ok(p) => p,
        Err(error) => {
            respond_error(request, error.status, &error.message);
            return;
        }
    };
    let options = match resolve_chat_options(&state.default_options, &api_request) {
        Ok(o) => o,
        Err(error) => {
            respond_error(request, error.status, &error.message);
            return;
        }
    };
    let system = prepared
        .system
        .clone()
        .or_else(|| state.default_options.system.clone());
    let stop = stop_strings(&api_request.stop);
    let model = if api_request.model.is_empty() {
        state.model_id.clone()
    } else {
        api_request.model.clone()
    };
    let completion_id = completion_id();
    let created = openai::now_unix();

    if let (Some(tool_choice), Some(tools)) = (tool_choice, api_request.tools.as_ref()) {
        let (messages_json, tools_json) = match build_tool_prompt_inputs(
            &api_request,
            tools,
            tool_choice.clone(),
            state.default_options.system.as_deref(),
        ) {
            Ok(inputs) => inputs,
            Err(error) => {
                respond_error(request, 400, &error);
                return;
            }
        };
        let guard = generation_ticket.acquire();
        let llama_choice = match tool_choice {
            ResolvedToolChoice::Auto => LlamaToolChoice::Auto,
            ResolvedToolChoice::None => LlamaToolChoice::None,
            ResolvedToolChoice::Required | ResolvedToolChoice::Named(_) => {
                LlamaToolChoice::Required
            }
        };
        let (tokens, chat_params) = match state.engine.prepare_tool_chat(
            &messages_json,
            &tools_json,
            llama_choice,
            options.max_tokens,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                respond_error(request, prompt_error_status(&error), &error);
                return;
            }
        };
        if api_request.stream {
            let heartbeat_frame =
                match sse_frame(&completion_id, created, &model, None, Some(""), None, None) {
                    Ok(frame) => frame,
                    Err(error) => {
                        respond_error(request, 500, &error);
                        return;
                    }
                };
            let (tx, rx) = mpsc::sync_channel::<Bytes>(SSE_CHANNEL_CAPACITY);
            let response = sse_response(rx, heartbeat_frame, None);
            let gen_state = Arc::clone(&state);
            let requested_tools = tools.clone();
            std::thread::spawn(move || {
                let result = generate_tool_chat(
                    &gen_state,
                    guard,
                    ToolChatInput {
                        tokens,
                        chat_params,
                        tools: &requested_tools,
                        choice: &tool_choice,
                        options: &options,
                        stop: &stop,
                    },
                );
                match result {
                    Ok((generation, ToolResponse::Text(content))) => {
                        let mut sink = SseSink::new(tx, completion_id, created, model);
                        let result = sink
                            .push(&content)
                            .and_then(|()| {
                                sink.send_chunk(
                                    None,
                                    None,
                                    None,
                                    Some(generation.finish_reason.as_str()),
                                )
                            })
                            .and_then(|()| sink.send_done());
                        if let Err(error) = result {
                            eprintln!("Failed to send tool-mode SSE response: {error}");
                        }
                    }
                    Ok((_, ToolResponse::Calls(calls))) => {
                        let mut sink = SseSink::new(tx, completion_id, created, model);
                        let result = sink
                            .send_tool_calls(&calls)
                            .and_then(|()| sink.send_chunk(None, None, None, Some("tool_calls")))
                            .and_then(|()| sink.send_done());
                        if let Err(error) = result {
                            eprintln!("Failed to send tool-call SSE response: {error}");
                        }
                    }
                    Err(error) => {
                        let sink = SseSink::new(tx, completion_id, created, model);
                        let _ = sink.send_chunk(None, None, None, Some("server_error"));
                        let _ = sink.send_done();
                        eprintln!("Tool-call generation failed: {error}");
                    }
                }
            });
            let _ = request.respond(response);
        } else {
            match generate_tool_chat(
                &state,
                guard,
                ToolChatInput {
                    tokens,
                    chat_params,
                    tools,
                    choice: &tool_choice,
                    options: &options,
                    stop: &stop,
                },
            ) {
                Ok((generation, result)) => {
                    let (content, calls, finish_reason) = match result {
                        ToolResponse::Text(content) => (
                            Some(content),
                            None,
                            generation.finish_reason.as_str().to_string(),
                        ),
                        ToolResponse::Calls(calls) => (None, Some(calls), "tool_calls".to_string()),
                    };
                    let body = serde_json::to_string(&ChatCompletionResponse {
                        id: completion_id,
                        object: "chat.completion",
                        created,
                        model,
                        choices: vec![ChatChoice {
                            index: 0,
                            message: ChoiceMessage {
                                role: "assistant",
                                content,
                                tool_calls: calls,
                            },
                            finish_reason,
                        }],
                        usage: Usage::new(generation.prompt_tokens, generation.generated_tokens),
                    })
                    .unwrap_or_default();
                    respond_json(request, StatusCode(200), body);
                }
                Err(error) => respond_error(request, 500, &error),
            }
        }
        return;
    }

    let guard = generation_ticket.acquire();
    let mut messages = prepared.messages;
    let tokens = match state
        .engine
        .prepare(&mut messages, system.as_deref(), options.max_tokens)
    {
        Ok((tokens, _)) => tokens,
        Err(error) => {
            respond_error(request, prompt_error_status(&error), &error);
            return;
        }
    };

    if api_request.stream {
        let heartbeat_frame =
            match sse_frame(&completion_id, created, &model, None, Some(""), None, None) {
                Ok(frame) => frame,
                Err(error) => {
                    respond_error(request, 500, &error);
                    return;
                }
            };
        let (tx, rx) = mpsc::sync_channel::<Bytes>(SSE_CHANNEL_CAPACITY);
        let disconnected = Arc::new(AtomicBool::new(false));
        let response = sse_response(rx, heartbeat_frame, Some(Arc::clone(&disconnected)));
        let gen_state = Arc::clone(&state);
        std::thread::spawn(move || {
            let mut sink =
                SseSink::new_with_disconnect(tx, completion_id, created, model, disconnected);
            let result = generate_chat(&gen_state, guard, tokens, &options, &stop, &mut sink);
            match result {
                Ok(generation) => {
                    if let Err(err) = sink
                        .send_chunk(None, None, None, Some(generation.finish_reason.as_str()))
                        .and_then(|()| sink.send_done())
                    {
                        eprintln!("Failed to send chat SSE response: {err}");
                    }
                }
                Err(err) => {
                    let _ = sink.send_chunk(None, None, None, Some("server_error"));
                    let _ = sink.send_done();
                    eprintln!("Generation failed: {err}");
                }
            }
        });
        let _ = request.respond(response);
    } else {
        let mut sink = NullSink;
        match generate_chat(&state, guard, tokens, &options, &stop, &mut sink) {
            Ok(generation) => {
                let (content, calls, finish_reason) =
                    chat_completion_parts(&generation.text, generation.finish_reason.as_str());
                let body = serde_json::to_string(&ChatCompletionResponse {
                    id: completion_id,
                    object: "chat.completion",
                    created,
                    model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChoiceMessage {
                            role: "assistant",
                            content,
                            tool_calls: calls,
                        },
                        finish_reason,
                    }],
                    usage: Usage::new(generation.prompt_tokens, generation.generated_tokens),
                })
                .unwrap_or_default();
                respond_json(request, StatusCode(200), body);
            }
            Err(err) => {
                respond_error(request, 500, &err);
            }
        }
    }
}

fn chat_completion_parts(
    text: &str,
    finish_reason: &str,
) -> (Option<String>, Option<Vec<ToolCall>>, String) {
    match parse_generated_tool_calls(text) {
        Some(calls) => (None, Some(calls), "tool_calls".to_string()),
        None => (Some(text.to_string()), None, finish_reason.to_string()),
    }
}

fn build_tool_prompt_inputs(
    request: &ChatCompletionRequest,
    tools: &[ToolDefinition],
    choice: ResolvedToolChoice,
    default_system: Option<&str>,
) -> Result<(String, String), String> {
    let mut messages = request.messages.clone();
    if !messages
        .iter()
        .any(|message| matches!(message.role.as_str(), "system" | "developer"))
        && let Some(system) = default_system
    {
        messages.insert(
            0,
            RequestMessage {
                role: "system".to_string(),
                content: Some(Content::Text(system.to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        );
    }
    let tools: Vec<_> = match choice {
        ResolvedToolChoice::Named(ref name) => tools
            .iter()
            .filter(|tool| tool.function.name == *name)
            .cloned()
            .collect(),
        _ => tools.to_vec(),
    };
    let messages_json = serde_json::to_string(&messages)
        .map_err(|error| format!("Failed to serialize messages: {error}"))?;
    let tools_json = serde_json::to_string(&tools)
        .map_err(|error| format!("Failed to serialize tool definitions: {error}"))?;
    Ok((messages_json, tools_json))
}

struct ToolChatInput<'a> {
    tokens: Vec<LlamaToken>,
    chat_params: ChatParams,
    tools: &'a [ToolDefinition],
    choice: &'a ResolvedToolChoice,
    options: &'a ChatOptions,
    stop: &'a [String],
}

fn generate_tool_chat(
    state: &ServerState,
    _guard: QueueGuard,
    input: ToolChatInput<'_>,
) -> Result<(Generation, ToolResponse), String> {
    let mut context = state.engine.new_context()?;
    let mut stop = input.stop.to_vec();
    let template_stops: Vec<String> =
        serde_json::from_str(&input.chat_params.additional_stops_json)
            .map_err(|error| format!("Failed to parse template stop markers: {error}"))?;
    stop.extend(template_stops);
    let mut sink = NullSink;
    let generation = state.engine.generate_with_chat_params(
        &mut context,
        &input.tokens,
        input.options,
        &stop,
        &mut sink,
        &input.chat_params,
    )?;
    let parsed = input
        .chat_params
        .parse(&generation.text, false)
        .unwrap_or_else(|_| generation.text.clone());
    let result = parse_tool_response(&parsed, &generation.text, input.tools, input.choice);
    Ok((generation, result))
}

// ---- completions (legacy) ----

fn handle_completions(state: Arc<ServerState>, request: Request, generation_ticket: QueueTicket) {
    let mut request = request;
    let body = match read_body(&mut request) {
        Ok(body) => body,
        Err(error) => {
            respond_error(
                request,
                400,
                &format!("Failed to read request body: {error}"),
            );
            return;
        }
    };
    let api_request: CompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(error) => {
            respond_error(
                request,
                400,
                &format!("Request body is not valid JSON: {error}"),
            );
            return;
        }
    };
    if let Err(error) = ensure_model_matches(&api_request.model, &state.model_id) {
        respond_error(request, error.status, &error.message);
        return;
    }
    if api_request.stream {
        respond_error(
            request,
            400,
            "Streaming for /v1/completions is not supported yet; use /v1/chat/completions.",
        );
        return;
    }

    let message = match build_completion_message(&api_request) {
        Ok(m) => m,
        Err(error) => {
            respond_error(request, error.status, &error.message);
            return;
        }
    };
    let options = match resolve_completion_options(&state.default_options, &api_request) {
        Ok(o) => o,
        Err(error) => {
            respond_error(request, error.status, &error.message);
            return;
        }
    };
    let stop = stop_strings(&api_request.stop);
    let model = if api_request.model.is_empty() {
        state.model_id.clone()
    } else {
        api_request.model.clone()
    };
    let system = state.default_options.system.clone();
    let guard = generation_ticket.acquire();
    let mut messages = vec![message];
    let tokens = match state
        .engine
        .prepare(&mut messages, system.as_deref(), options.max_tokens)
    {
        Ok((tokens, _)) => tokens,
        Err(error) => {
            respond_error(request, prompt_error_status(&error), &error);
            return;
        }
    };
    let completion_id = text_completion_id();
    let created = openai::now_unix();

    let mut sink = NullSink;
    match generate_chat(&state, guard, tokens, &options, &stop, &mut sink) {
        Ok(generation) => {
            let body = serde_json::to_string(&TextCompletionResponse {
                id: completion_id,
                object: "text_completion",
                created,
                model,
                choices: vec![TextChoice {
                    index: 0,
                    text: generation.text,
                    finish_reason: generation.finish_reason.as_str().to_string(),
                }],
                usage: Usage::new(generation.prompt_tokens, generation.generated_tokens),
            })
            .unwrap_or_default();
            respond_json(request, StatusCode(200), body);
        }
        Err(err) => {
            respond_error(request, 500, &err);
        }
    }
}

// ---- generation ----

fn generate_chat(
    state: &ServerState,
    _guard: QueueGuard,
    tokens: Vec<LlamaToken>,
    options: &ChatOptions,
    stop: &[String],
    sink: &mut dyn TextSink,
) -> Result<Generation, String> {
    let mut context = state.engine.new_context()?;
    state
        .engine
        .generate(&mut context, &tokens, options, stop, sink)
}

fn prompt_error_status(error: &str) -> u16 {
    if error.starts_with("Input plus reserved generation tokens exceeds the model context limit") {
        400
    } else {
        500
    }
}

// ---- sinks ----

struct NullSink;

impl TextSink for NullSink {
    fn push(&mut self, _text: &str) -> Result<(), String> {
        Ok(())
    }
}

/// 流式 SSE 输出目标:把每段文本封装成 OpenAI 格式的 `data:` 帧。
struct SseSink {
    tx: SyncSender<Bytes>,
    id: String,
    created: u64,
    model: String,
    first: bool,
    disconnected: Option<Arc<AtomicBool>>,
}

impl SseSink {
    fn new(tx: SyncSender<Bytes>, id: String, created: u64, model: String) -> Self {
        Self::new_inner(tx, id, created, model, None)
    }

    fn new_with_disconnect(
        tx: SyncSender<Bytes>,
        id: String,
        created: u64,
        model: String,
        disconnected: Arc<AtomicBool>,
    ) -> Self {
        Self::new_inner(tx, id, created, model, Some(disconnected))
    }

    fn new_inner(
        tx: SyncSender<Bytes>,
        id: String,
        created: u64,
        model: String,
        disconnected: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            tx,
            id,
            created,
            model,
            first: true,
            disconnected,
        }
    }

    fn send_chunk(
        &self,
        role: Option<&'static str>,
        content: Option<&str>,
        tool_calls: Option<Vec<ToolCallDelta>>,
        finish_reason: Option<&str>,
    ) -> Result<(), String> {
        self.tx
            .send(sse_frame(
                &self.id,
                self.created,
                &self.model,
                role,
                content,
                tool_calls,
                finish_reason,
            )?)
            .map_err(|_| "Client disconnected".to_string())
    }

    fn send_tool_calls(&mut self, calls: &[ToolCall]) -> Result<(), String> {
        let role = self.first.then_some("assistant");
        self.first = false;
        let calls = calls
            .iter()
            .enumerate()
            .map(|(index, call)| ToolCallDelta {
                index,
                id: Some(call.id.clone()),
                kind: Some(call.kind),
                function: ToolCallFunctionDelta {
                    name: Some(call.function.name.clone()),
                    arguments: Some(call.function.arguments.clone()),
                },
            })
            .collect();
        self.send_chunk(role, None, Some(calls), None)
    }

    fn send_done(&self) -> Result<(), String> {
        self.tx
            .send(Bytes::from_static(b"data: [DONE]\n\n"))
            .map_err(|_| "Client disconnected".to_string())
    }
}

#[derive(Serialize)]
struct SseChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [SseChunkChoice<'a>; 1],
}

#[derive(Serialize)]
struct SseChunkChoice<'a> {
    index: usize,
    delta: SseChunkDelta<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'a str>,
}

#[derive(Serialize)]
struct SseChunkDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

fn sse_frame(
    id: &str,
    created: u64,
    model: &str,
    role: Option<&'static str>,
    content: Option<&str>,
    tool_calls: Option<Vec<ToolCallDelta>>,
    finish_reason: Option<&str>,
) -> Result<Bytes, String> {
    let chunk = SseChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [SseChunkChoice {
            index: 0,
            delta: SseChunkDelta {
                role,
                content,
                tool_calls,
            },
            finish_reason,
        }],
    };
    let mut frame = Vec::with_capacity(192 + content.map_or(0, str::len));
    frame.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut frame, &chunk)
        .map_err(|error| format!("Failed to serialize SSE frame: {error}"))?;
    frame.extend_from_slice(b"\n\n");
    Ok(Bytes::from(frame))
}

impl TextSink for SseSink {
    fn push(&mut self, text: &str) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        if self
            .disconnected
            .as_ref()
            .is_some_and(|disconnected| disconnected.load(Ordering::Relaxed))
        {
            return Err("Client disconnected".to_string());
        }
        let role = self.first.then_some("assistant");
        self.first = false;
        self.send_chunk(role, Some(text), None, None)
    }

    fn capture_text(&self) -> bool {
        false
    }

    fn interrupted(&mut self) -> bool {
        self.disconnected
            .as_ref()
            .is_some_and(|disconnected| disconnected.load(Ordering::Relaxed))
    }
}

// ---- SSE reader ----

fn sse_response(
    receiver: Receiver<Bytes>,
    heartbeat_frame: Bytes,
    disconnected: Option<Arc<AtomicBool>>,
) -> Response<ChannelReader> {
    let mut headers = cors_headers();
    headers
        .push(Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).expect("ssect"));
    headers.push(Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).expect("ssecc"));
    let reader = match disconnected {
        Some(disconnected) => ChannelReader::new_with_disconnect(
            receiver,
            heartbeat_frame,
            SSE_HEARTBEAT_INTERVAL,
            disconnected,
        ),
        None => ChannelReader::new(receiver, heartbeat_frame, SSE_HEARTBEAT_INTERVAL),
    };
    Response::new(StatusCode(200), headers, reader, None, None)
}

struct ChannelReader {
    receiver: Receiver<Bytes>,
    heartbeat_frame: Bytes,
    heartbeat_interval: Duration,
    disconnected: Arc<AtomicBool>,
    current: Bytes,
    offset: usize,
}

impl ChannelReader {
    fn new(
        receiver: Receiver<Bytes>,
        heartbeat_frame: Bytes,
        heartbeat_interval: Duration,
    ) -> Self {
        Self::new_with_disconnect(
            receiver,
            heartbeat_frame,
            heartbeat_interval,
            Arc::new(AtomicBool::new(false)),
        )
    }

    fn new_with_disconnect(
        receiver: Receiver<Bytes>,
        heartbeat_frame: Bytes,
        heartbeat_interval: Duration,
        disconnected: Arc<AtomicBool>,
    ) -> Self {
        Self {
            receiver,
            heartbeat_frame,
            heartbeat_interval,
            disconnected,
            current: Bytes::new(),
            offset: 0,
        }
    }
}

impl Drop for ChannelReader {
    fn drop(&mut self) {
        self.disconnected.store(true, Ordering::Relaxed);
    }
}

impl Read for ChannelReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        while written < buf.len() {
            if self.offset >= self.current.len() {
                let next = if written == 0 {
                    match self.receiver.recv_timeout(self.heartbeat_interval) {
                        Ok(chunk) => Some(chunk),
                        Err(RecvTimeoutError::Timeout) => Some(self.heartbeat_frame.clone()),
                        Err(RecvTimeoutError::Disconnected) => return Ok(0),
                    }
                } else {
                    match self.receiver.try_recv() {
                        Ok(chunk) => Some(chunk),
                        Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => None,
                    }
                };
                let Some(next) = next else {
                    break;
                };
                self.current = next;
                self.offset = 0;
            }
            let remaining = &self.current[self.offset..];
            let n = remaining.len().min(buf.len() - written);
            buf[written..written + n].copy_from_slice(&remaining[..n]);
            self.offset += n;
            written += n;
        }
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_compares_bearer_tokens() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn prompt_context_overflow_maps_to_bad_request() {
        assert_eq!(
            prompt_error_status(
                "Input plus reserved generation tokens exceeds the model context limit of 64"
            ),
            400
        );
        assert_eq!(prompt_error_status("Failed to render prompt"), 500);
    }

    #[test]
    fn routes_known_paths() {
        assert!(matches!(
            route(&Method::Get, "/health"),
            Ok(RouteEnum::Health)
        ));
        assert!(matches!(
            route(&Method::Get, "/v1/models"),
            Ok(RouteEnum::ListModels)
        ));
        assert!(matches!(
            route(&Method::Get, "/v1/models/qwen2"),
            Ok(RouteEnum::GetModel(id)) if id == "qwen2"
        ));
        assert!(matches!(
            route(&Method::Post, "/v1/chat/completions"),
            Ok(RouteEnum::ChatCompletions)
        ));
        assert!(matches!(
            route(&Method::Post, "/v1/completions"),
            Ok(RouteEnum::Completions)
        ));
    }

    #[test]
    fn routes_return_405_wrong_method_and_404_unknown() {
        assert_eq!(route(&Method::Post, "/v1/models"), Err(StatusCode(405)));
        assert_eq!(
            route(&Method::Get, "/v1/chat/completions"),
            Err(StatusCode(405))
        );
        assert_eq!(route(&Method::Get, "/v1/unknown"), Err(StatusCode(404)));
    }

    #[test]
    fn channel_reader_eof_on_disconnect() {
        let (tx, rx) = mpsc::sync_channel(1);
        let mut reader =
            ChannelReader::new(rx, Bytes::from_static(b"heartbeat"), Duration::from_secs(1));
        let mut buf = [0u8; 16];
        // dropping tx → EOF
        drop(tx);
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn channel_reader_pumps_chunks() {
        let (tx, rx) = mpsc::sync_channel(1);
        let mut reader =
            ChannelReader::new(rx, Bytes::from_static(b"heartbeat"), Duration::from_secs(1));
        let mut buf = [0u8; 4];
        tx.send(Bytes::from_static(b"hello")).unwrap();
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"hell");
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"o");
    }

    #[test]
    fn channel_reader_batches_ready_chunks() {
        let (tx, rx) = mpsc::sync_channel(4);
        let mut reader =
            ChannelReader::new(rx, Bytes::from_static(b"heartbeat"), Duration::from_secs(1));
        tx.send(Bytes::from_static(b"one")).unwrap();
        tx.send(Bytes::from_static(b"two")).unwrap();
        let mut buf = [0u8; 16];

        let n = reader.read(&mut buf).unwrap();

        assert_eq!(&buf[..n], b"onetwo");
    }

    #[test]
    fn dropping_reader_unblocks_sse_sink() {
        let (tx, rx) = mpsc::sync_channel(1);
        tx.send(Bytes::from_static(b"queued")).unwrap();
        let disconnected = Arc::new(AtomicBool::new(false));
        let reader = ChannelReader::new_with_disconnect(
            rx,
            Bytes::from_static(b"heartbeat"),
            Duration::from_secs(1),
            Arc::clone(&disconnected),
        );
        let mut sink = SseSink::new_with_disconnect(
            tx,
            "test".to_string(),
            0,
            "test".to_string(),
            disconnected,
        );
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let sender = std::thread::spawn(move || {
            done_tx.send(sink.push("token").is_err()).unwrap();
        });

        drop(reader);
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap()
        );
        sender.join().unwrap();
    }

    #[test]
    fn sse_sink_stops_generation_after_reader_drop() {
        let (tx, rx) = mpsc::sync_channel(1);
        let disconnected = Arc::new(AtomicBool::new(false));
        let reader = ChannelReader::new_with_disconnect(
            rx,
            Bytes::from_static(b"heartbeat"),
            Duration::from_secs(1),
            Arc::clone(&disconnected),
        );
        let mut sink =
            SseSink::new_with_disconnect(tx, "test".into(), 0, "test".into(), disconnected);
        assert!(sink.push("token").is_ok());
        drop(reader);
        assert_eq!(sink.push("token"), Err("Client disconnected".to_string()));
    }

    #[test]
    fn channel_reader_sends_heartbeat_when_idle() {
        let (tx, rx) = mpsc::sync_channel(1);
        let heartbeat = sse_frame("test", 1, "test", None, Some(""), None, None).unwrap();
        let mut reader = ChannelReader::new(rx, heartbeat.clone(), Duration::from_millis(10));
        let mut buf = [0u8; 512];

        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], heartbeat.as_ref());
        assert!(heartbeat.starts_with(b"data: "));
        assert!(heartbeat.ends_with(b"\n\n"));
        let payload = std::str::from_utf8(&heartbeat[6..heartbeat.len() - 2]).unwrap();
        let chunk: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(chunk["choices"][0]["delta"]["content"], "");

        tx.send(Bytes::from_static(b"data: token\n\n")).unwrap();
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"data: token\n\n");
        drop(tx);
        assert_eq!(reader.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn sse_tool_call_delta_uses_openai_fields() {
        let frame = sse_frame(
            "chatcmpl-1",
            1,
            "qwen",
            Some("assistant"),
            None,
            Some(vec![ToolCallDelta {
                index: 0,
                id: Some("call_1".to_string()),
                kind: Some("function"),
                function: ToolCallFunctionDelta {
                    name: Some("weather".to_string()),
                    arguments: Some(r#"{"city":"Tokyo"}"#.to_string()),
                },
            }]),
            None,
        )
        .unwrap();
        let payload = std::str::from_utf8(&frame[6..frame.len() - 2]).unwrap();
        let chunk: serde_json::Value = serde_json::from_str(payload).unwrap();
        let delta = &chunk["choices"][0]["delta"];
        assert_eq!(delta["role"], "assistant");
        assert_eq!(delta["tool_calls"][0]["index"], 0);
        assert_eq!(delta["tool_calls"][0]["id"], "call_1");
        assert_eq!(delta["tool_calls"][0]["type"], "function");
        assert_eq!(
            delta["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Tokyo"}"#
        );
    }

    #[test]
    fn chat_stream_emits_each_generated_chunk_without_copying_text() {
        let (tx, rx) = mpsc::sync_channel(4);
        let mut sink = SseSink::new(tx, "chatcmpl-test".to_string(), 1, "test".to_string());
        assert!(!sink.capture_text());
        sink.push("plain ").unwrap();
        sink.push("response").unwrap();
        sink.send_chunk(None, None, None, Some("stop")).unwrap();
        sink.send_done().unwrap();

        let content_frame = rx.try_recv().unwrap();
        let content_json = std::str::from_utf8(&content_frame[6..content_frame.len() - 2]).unwrap();
        let content_chunk: serde_json::Value = serde_json::from_str(content_json).unwrap();
        assert_eq!(content_chunk["choices"][0]["delta"]["content"], "plain ");

        let next_content_frame = rx.try_recv().unwrap();
        let next_content_json =
            std::str::from_utf8(&next_content_frame[6..next_content_frame.len() - 2]).unwrap();
        let next_content_chunk: serde_json::Value =
            serde_json::from_str(next_content_json).unwrap();
        assert_eq!(
            next_content_chunk["choices"][0]["delta"]["content"],
            "response"
        );

        let finish_frame = rx.try_recv().unwrap();
        let finish_json = std::str::from_utf8(&finish_frame[6..finish_frame.len() - 2]).unwrap();
        let finish_chunk: serde_json::Value = serde_json::from_str(finish_json).unwrap();
        assert_eq!(finish_chunk["choices"][0]["finish_reason"], "stop");
        assert_eq!(rx.try_recv().unwrap().as_ref(), b"data: [DONE]\n\n");
    }

    #[test]
    fn non_stream_chat_converts_tool_json_and_falls_back_to_text() {
        let (content, calls, finish_reason) = chat_completion_parts(
            r#"{"tool_calls":[{"name":"weather","arguments":{"city":"Tokyo"}}]}"#,
            "stop",
        );
        assert!(content.is_none());
        assert_eq!(finish_reason, "tool_calls");
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Tokyo"}"#);

        let (content, calls, finish_reason) = chat_completion_parts("plain response", "length");
        assert_eq!(content.as_deref(), Some("plain response"));
        assert!(calls.is_none());
        assert_eq!(finish_reason, "length");
    }

    #[test]
    fn sse_sink_ignores_empty_generated_chunks() {
        let (tx, rx) = mpsc::sync_channel(2);
        let mut sink = SseSink {
            tx,
            id: "test".to_string(),
            created: 0,
            model: "test".to_string(),
            first: true,
            disconnected: None,
        };

        sink.push("").unwrap();
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        sink.push("token").unwrap();
        let chunk: serde_json::Value = serde_json::from_slice(
            &rx.try_recv()
                .unwrap()
                .strip_prefix(b"data: ")
                .unwrap()
                .strip_suffix(b"\n\n")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunk["choices"][0]["delta"]["content"], "token");
    }
}
