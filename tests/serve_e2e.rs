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
        eprintln!("跳过 serve e2e 测试：未设置 WOLF_TEST_MODEL。");
        return;
    };
    if !model_path.is_file() {
        panic!("WOLF_TEST_MODEL 指向的文件不存在：{}", model_path.display());
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
            "--temperature",
            "0",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("启动 wolf serve 失败");

    let stdout = child.stdout.take().expect("stdout");
    let reader = BufReader::new(stdout);
    let mut port: Option<u16> = None;
    for line in reader.lines() {
        let line = line.expect("读取 stdout");
        if let Some(url) = line.strip_prefix("已启动服务: ") {
            port = url.rsplit(':').next().and_then(|s| s.parse().ok());
            break;
        }
    }
    let port = port.expect("未从 stdout 解析出端口");
    let guard = ServerGuard(child);

    let base = format!("http://127.0.0.1:{port}");
    wait_for_ready(&base);

    // health
    let (code, _) = http_get(&format!("{base}/health"), None);
    assert_eq!(code, 200, "/health 应返回 200");

    // auth failures
    let (code, _) = http_get(&format!("{base}/v1/models"), None);
    assert_eq!(code, 401, "未带 API Key 应返回 401");
    let (code, _) = http_get(&format!("{base}/v1/models"), Some("wrong"));
    assert_eq!(code, 401, "错误 API Key 应返回 401");

    // list models
    let (code, body) = http_get(&format!("{base}/v1/models"), Some("wolf-test"));
    assert_eq!(code, 200, "携带正确 Key 应返回 200");
    assert!(body.contains("\"object\":\"list\""), "模型列表响应：{body}");
    let listing: serde_json::Value = serde_json::from_str(http_body(&body)).unwrap();
    let model_id = listing["data"][0]["id"]
        .as_str()
        .expect("模型列表应含 id")
        .to_string();

    // retrieve a single model
    let (code, body) = http_get(&format!("{base}/v1/models/{model_id}"), Some("wolf-test"));
    assert_eq!(code, 200, "获取单个模型应返回 200：{body}");
    assert!(
        body.contains("\"object\":\"model\""),
        "单个模型响应：{body}"
    );
    let (code, body) = http_get(&format!("{base}/v1/models/gpt-4"), Some("wolf-test"));
    assert_eq!(code, 404, "未知模型应返回 404：{body}");

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
    assert_eq!(code, 404, "未知模型名应返回 404：{body}");

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
    assert_eq!(code, 200, "非流式 chat 应返回 200：{body}");
    assert!(
        body.contains("\"object\":\"chat.completion\""),
        "非流式响应：{body}"
    );
    assert!(
        body.contains("\"role\":\"assistant\""),
        "非流式响应应含 assistant 角色：{body}"
    );

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
    assert_eq!(code, 200, "流式 chat 应返回 200：{body}");
    assert!(body.contains("data: "), "流式响应应含 data: 行：{body}");
    assert!(
        body.contains("data: [DONE]"),
        "流式响应应以 [DONE] 结束：{body}"
    );

    // completions
    let req = serde_json::json!({
        "prompt": "hello",
        "max_tokens": 16,
        "temperature": 0
    });
    let (code, body) = http_post(&format!("{base}/v1/completions"), &req, Some("wolf-test"));
    assert_eq!(code, 200, "completions 应返回 200：{body}");
    assert!(
        body.contains("\"object\":\"text_completion\""),
        "completions 响应：{body}"
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
    panic!("服务未能在 120 秒内就绪");
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

    let mut stream = TcpStream::connect((host, port)).expect("连接服务失败");
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
