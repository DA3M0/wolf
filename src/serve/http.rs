//! HTTP routing, authentication and streaming glue for `wolf serve`.

use crate::chat::{ChatMessage, Engine, Generation, TextSink};
use crate::options::ChatOptions;
use crate::serve::openai::{self, *};
use std::io::{self, Read};
use std::sync::{Arc, Mutex, mpsc};
use tiny_http::{Header, Method, Request, Response, StatusCode};

pub(crate) struct ServerState {
    pub(crate) engine: Arc<Engine>,
    pub(crate) default_options: ChatOptions,
    pub(crate) model_id: String,
    pub(crate) api_key: Option<String>,
    pub(crate) generation_lock: Mutex<()>,
    pub(crate) created: u64,
}

pub(crate) fn handle(state: Arc<ServerState>, request: Request) {
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
        respond_json(request, StatusCode(200), &health_body());
        return;
    }

    if !authorized(&request, state.api_key.as_deref()) {
        respond_error(
            request,
            401,
            "缺少有效的 API Key；请在 Authorization 头中提供 Bearer Key。",
        );
        return;
    }

    match route(&method, path) {
        Ok(RouteEnum::Health) => respond_json(request, StatusCode(200), &health_body()),
        Ok(RouteEnum::ListModels) => {
            let body = model_list_string(&state);
            respond_json(request, StatusCode(200), &body);
        }
        Ok(RouteEnum::GetModel(id)) => {
            if id == state.model_id {
                let body = single_model_string(&state);
                respond_json(request, StatusCode(200), &body);
            } else {
                respond_error(
                    request,
                    404,
                    &format!("模型“{id}”不存在，当前服务加载的是“{}”。", state.model_id),
                );
            }
        }
        Ok(RouteEnum::ChatCompletions) => handle_chat_completions(state, request),
        Ok(RouteEnum::Completions) => handle_completions(state, request),
        Err(status) => respond_error(
            request,
            status.0,
            if status.0 == 405 {
                "方法不允许。"
            } else {
                "未找到该端点。"
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

fn respond_json(request: Request, status: StatusCode, body: &str) {
    let data = body.to_string().into_bytes();
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
    respond_json(request, StatusCode(status), &body);
}

fn read_body(request: &mut Request) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    request.as_reader().read_to_end(&mut body)?;
    Ok(body)
}

// ---- chat completions ----

fn handle_chat_completions(state: Arc<ServerState>, request: Request) {
    let mut request = request;
    let body = match read_body(&mut request) {
        Ok(body) => body,
        Err(error) => {
            respond_error(request, 400, &format!("读取请求体失败：{error}"));
            return;
        }
    };
    let api_request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(error) => {
            respond_error(request, 400, &format!("请求体不是有效的 JSON：{error}"));
            return;
        }
    };
    if let Err(error) = ensure_model_matches(&api_request.model, &state.model_id) {
        respond_error(request, error.status, &error.message);
        return;
    }

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
        .or_else(|| state.default_options.system.clone());
    let stop = stop_strings(&api_request.stop);
    let model = if api_request.model.is_empty() {
        state.model_id.clone()
    } else {
        api_request.model.clone()
    };
    let completion_id = completion_id();
    let created = openai::now_unix();

    if api_request.stream {
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let response = sse_response(rx);
        let gen_state = Arc::clone(&state);
        std::thread::spawn(move || {
            let mut sink = SseSink {
                tx,
                id: completion_id,
                created,
                model: model.clone(),
                first: true,
            };
            let result = generate_chat(
                &gen_state,
                system,
                prepared.messages,
                &options,
                &stop,
                &mut sink,
            );
            match result {
                Ok(generation) => {
                    let _ = sink.send_chunk(
                        None,
                        None,
                        Some(generation.finish_reason.as_str().to_string()),
                    );
                    let _ = sink.send_done();
                }
                Err(err) => {
                    let _ = sink.send_chunk(None, None, Some("server_error".to_string()));
                    let _ = sink.send_done();
                    eprintln!("生成过程中出错：{err}");
                }
            }
        });
        let _ = request.respond(response);
    } else {
        let mut sink = NullSink;
        match generate_chat(
            &state,
            system,
            prepared.messages,
            &options,
            &stop,
            &mut sink,
        ) {
            Ok(generation) => {
                let body = serde_json::to_string(&ChatCompletionResponse {
                    id: completion_id,
                    object: "chat.completion",
                    created,
                    model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChoiceMessage {
                            role: "assistant",
                            content: generation.text,
                        },
                        finish_reason: generation.finish_reason.as_str().to_string(),
                    }],
                    usage: Usage::new(generation.prompt_tokens, generation.generated_tokens),
                })
                .unwrap_or_default();
                respond_json(request, StatusCode(200), &body);
            }
            Err(err) => {
                respond_error(request, 500, &err);
            }
        }
    }
}

// ---- completions (legacy) ----

fn handle_completions(state: Arc<ServerState>, request: Request) {
    let mut request = request;
    let body = match read_body(&mut request) {
        Ok(body) => body,
        Err(error) => {
            respond_error(request, 400, &format!("读取请求体失败：{error}"));
            return;
        }
    };
    let api_request: CompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(error) => {
            respond_error(request, 400, &format!("请求体不是有效的 JSON：{error}"));
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
            "/v1/completions 流式暂不支持，请使用 /v1/chat/completions。",
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
    let completion_id = text_completion_id();
    let created = openai::now_unix();

    let mut sink = NullSink;
    match generate_chat(&state, system, vec![message], &options, &stop, &mut sink) {
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
            respond_json(request, StatusCode(200), &body);
        }
        Err(err) => {
            respond_error(request, 500, &err);
        }
    }
}

// ---- generation ----

fn generate_chat(
    state: &ServerState,
    system: Option<String>,
    mut messages: Vec<ChatMessage>,
    options: &ChatOptions,
    stop: &[String],
    sink: &mut dyn TextSink,
) -> Result<Generation, String> {
    let _guard = state
        .generation_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut context = state.engine.new_context()?;
    let (tokens, _dropped) =
        state
            .engine
            .prepare(&mut messages, system.as_deref(), options.max_tokens)?;
    state
        .engine
        .generate(&mut context, &tokens, options, stop, sink)
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
    tx: mpsc::Sender<Vec<u8>>,
    id: String,
    created: u64,
    model: String,
    first: bool,
}

impl SseSink {
    fn send_chunk(
        &self,
        role: Option<&'static str>,
        content: Option<String>,
        finish_reason: Option<String>,
    ) -> Result<(), String> {
        let chunk = ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta { role, content },
                finish_reason,
            }],
        };
        let json =
            serde_json::to_string(&chunk).map_err(|error| format!("SSE 帧序列化失败：{error}"))?;
        let frame = format!("data: {json}\n\n");
        self.tx
            .send(frame.into_bytes())
            .map_err(|_| "客户端已断开".to_string())
    }

    fn send_done(&self) -> Result<(), String> {
        self.tx
            .send(b"data: [DONE]\n\n".to_vec())
            .map_err(|_| "客户端已断开".to_string())
    }
}

impl TextSink for SseSink {
    fn push(&mut self, text: &str) -> Result<(), String> {
        let role = self.first.then_some("assistant");
        self.first = false;
        let content = Some(text.to_string());
        self.send_chunk(role, content, None)
    }
}

// ---- SSE reader ----

fn sse_response(receiver: mpsc::Receiver<Vec<u8>>) -> Response<ChannelReader> {
    let mut headers = cors_headers();
    headers
        .push(Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).expect("ssect"));
    headers.push(Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).expect("ssecc"));
    Response::new(
        StatusCode(200),
        headers,
        ChannelReader::new(receiver),
        None,
        None,
    )
}

struct ChannelReader {
    receiver: mpsc::Receiver<Vec<u8>>,
    current: Vec<u8>,
    offset: usize,
}

impl ChannelReader {
    fn new(receiver: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            current: Vec::new(),
            offset: 0,
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        while self.offset >= self.current.len() {
            match self.receiver.recv() {
                Ok(chunk) => {
                    self.current = chunk;
                    self.offset = 0;
                }
                Err(_) => return Ok(0),
            }
        }
        let remaining = &self.current[self.offset..];
        let n = remaining.len().min(buf.len());
        buf[..n].copy_from_slice(&remaining[..n]);
        self.offset += n;
        Ok(n)
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
        let (tx, rx) = mpsc::channel();
        let mut reader = ChannelReader::new(rx);
        let mut buf = [0u8; 16];
        // dropping tx → EOF
        drop(tx);
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn channel_reader_pumps_chunks() {
        let (tx, rx) = mpsc::channel();
        let mut reader = ChannelReader::new(rx);
        let mut buf = [0u8; 4];
        tx.send(b"hello".to_vec()).unwrap();
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"hell");
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"o");
    }
}
