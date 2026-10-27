//! End-to-end test for `wolf serve`.
//!
//! Requires a real GGUF model via the `WOLF_TEST_MODEL` environment variable.
//! CI does not set it, so the test is skipped and always passes in CI.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct ServerGuard(Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn serve_openai_compatible_api() {
    let Some(model_path) = std::env::var_os("WOLF_TEST_MODEL").map(PathBuf::from) else {
        eprintln!("Skipping serve e2e test: WOLF_TEST_MODEL is not set.");
        return;
    };
    if !model_path.is_file() {
        panic!(
            "WOLF_TEST_MODEL points to a file that does not exist: {}",
            model_path.display()
        );
    }

    let bin = std::env!("CARGO_BIN_EXE_wolf");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--model",
            &model_path.display().to_string(),
            "--port",
            "0",
            "--api-key",
            "wolf-test",
            "--max-tokens",
            "32",
            "--ctx-size",
            "64",
            "--temperature",
            "0",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start wolf serve");

    let stdout = child.stdout.take().expect("stdout");
    let reader = BufReader::new(stdout);
    let mut port: Option<u16> = None;
    for line in reader.lines() {
        let line = line.expect("Failed to read stdout");
        if let Some(url) = line.strip_prefix("Server started: ") {
            port = url.rsplit(':').next().and_then(|s| s.parse().ok());
            break;
        }
    }
    let port = port.expect("Failed to parse port from stdout");
    let guard = ServerGuard(child);

    let base = format!("http://127.0.0.1:{port}");
    wait_for_ready(&base);

    // health
    let (code, _) = http_get(&format!("{base}/health"), None);
    assert_eq!(code, 200, "/health should return 200");

    // auth failures
    let (code, _) = http_get(&format!("{base}/v1/models"), None);
    assert_eq!(code, 401, "Request without an API key should return 401");
    let (code, _) = http_get(&format!("{base}/v1/models"), Some("wrong"));
    assert_eq!(
        code, 401,
        "Request with an invalid API key should return 401"
    );

    // list models
    let (code, body) = http_get(&format!("{base}/v1/models"), Some("wolf-test"));
    assert_eq!(code, 200, "Request with the correct key should return 200");
    assert!(
        body.contains("\"object\":\"list\""),
        "Model list response: {body}"
    );
    let listing: serde_json::Value = serde_json::from_str(http_body(&body)).unwrap();
    let model_id = listing["data"][0]["id"]
        .as_str()
        .expect("Model list should contain an id")
        .to_string();

    // retrieve a single model
    let (code, body) = http_get(&format!("{base}/v1/models/{model_id}"), Some("wolf-test"));
    assert_eq!(
        code, 200,
        "Retrieving a single model should return 200: {body}"
    );
    assert!(
        body.contains("\"object\":\"model\""),
        "Single model response: {body}"
    );
    let (code, body) = http_get(&format!("{base}/v1/models/gpt-4"), Some("wolf-test"));
    assert_eq!(code, 404, "Unknown model should return 404: {body}");

    // unknown model name is rejected
    let req = serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16
    });
    let (code, body) = http_post(
        &format!("{base}/v1/chat/completions"),
        &req,
        Some("wolf-test"),
    );
    assert_eq!(code, 404, "Unknown model name should return 404: {body}");

    // chat completions non-streaming
    let req = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0
    });
    let (code, body) = http_post(
        &format!("{base}/v1/chat/completions"),
        &req,
        Some("wolf-test"),
    );
    assert_eq!(code, 200, "Non-streaming chat should return 200: {body}");
    assert!(
        body.contains("\"object\":\"chat.completion\""),
        "Non-streaming response: {body}"
    );
    assert!(
        body.contains("\"role\":\"assistant\""),
        "Non-streaming response should contain the assistant role: {body}"
    );

    // Oversized prompts are rejected before streaming starts or llama.cpp is called.
    let req = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "oversized prompt ".repeat(100)}],
        "max_tokens": 16,
        "stream": true
    });
    let (code, body) = http_post(
        &format!("{base}/v1/chat/completions"),
        &req,
        Some("wolf-test"),
    );
    assert_eq!(code, 400, "Oversized prompt should return 400: {body}");

    // chat completions streaming
    let req = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "stream": true,
        "temperature": 0
    });
    let (code, body) = http_post(
        &format!("{base}/v1/chat/completions"),
        &req,
        Some("wolf-test"),
    );
    assert_eq!(code, 200, "Streaming chat should return 200: {body}");
    assert!(
        body.contains("data: "),
        "Streaming response should contain a data: line: {body}"
    );
    assert!(
        body.contains("data: [DONE]"),
        "Streaming response should end with [DONE]: {body}"
    );

    // completions
    let req = serde_json::json!({
        "prompt": "hello",
        "max_tokens": 16,
        "temperature": 0
    });
    let (code, body) = http_post(&format!("{base}/v1/completions"), &req, Some("wolf-test"));
    assert_eq!(code, 200, "Completions should return 200: {body}");
    assert!(
        body.contains("\"object\":\"text_completion\""),
        "Completions response: {body}"
    );

    drop(guard);
}

fn wait_for_ready(base: &str) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if let Some(port) = base.rsplit(':').next().and_then(|s| s.parse::<u16>().ok()) {
            let addr: SocketAddr = ([127, 0, 0, 1], port).into();
            if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
                std::thread::sleep(Duration::from_millis(200));
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    panic!("Server did not become ready within 120 seconds");
}

fn http_get(url: &str, key: Option<&str>) -> (u16, String) {
    send_http("GET", url, key, None)
}

fn http_post(url: &str, body: &serde_json::Value, key: Option<&str>) -> (u16, String) {
    send_http("POST", url, key, Some(body))
}

fn send_http(
    method: &str,
    url: &str,
    key: Option<&str>,
    body: Option<&serde_json::Value>,
) -> (u16, String) {
    // Parse http://host:port/path
    let rest = &url["http://".len()..];
    let (host_port, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = host_port.rsplit_once(':').unwrap_or((host_port, "80"));
    let port = port.parse::<u16>().unwrap();

    let mut stream = TcpStream::connect((host, port)).expect("Failed to connect to server");
    let body_bytes = body.map(|b| b.to_string().into_bytes()).unwrap_or_default();
    let auth = key
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    let content_length = if body.is_some() {
        format!("Content-Length: {}\r\n", body_bytes.len())
    } else {
        String::new()
    };
    let content_type = if body.is_some() {
        "Content-Type: application/json\r\n"
    } else {
        ""
    };
    let request = format!(
        "{method} /{path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n{content_length}{content_type}{auth}\r\n",
        path = path
    );
    stream.write_all(request.as_bytes()).unwrap();
    if body.is_some() {
        stream.write_all(&body_bytes).unwrap();
    }
    stream.flush().unwrap();

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).unwrap();
    let text = String::from_utf8_lossy(&buf).into_owned();
    let status = text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (status, text)
}

/// Splits the body off a raw HTTP/1.1 response.
fn http_body(response: &str) -> &str {
    response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or(response)
}
