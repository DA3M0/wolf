//! `wolf serve` — 启动 OpenAI 兼容的本地 API 服务。

mod http;
mod openai;
mod queue;

use crate::chat::{self, Engine};
use crate::cli::ServeOptions;
use crate::options::ChatOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

pub fn serve(options: ServeOptions, chat_options: ChatOptions) -> Result<(), String> {
    if options.api_key.is_none() && !options.generate_key && !is_loopback(&options.host) {
        return Err(format!(
            "When serving on '{}', authentication must be enabled with --generate-key or --api-key.\n\
             Do not expose the server on a non-loopback address without authentication. For local use only, set --host to 127.0.0.1.",
            options.host
        ));
    }

    let (model_path, model_id) = resolve_model(&options)?;
    if model_path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        return Err(format!(
            "Chat inference only supports GGUF model files: {}",
            model_path.display()
        ));
    }

    chat::configure_threads(chat_options.threads)?;
    eprintln!("Loading model: {}", model_path.display());

    let engine = Engine::load(
        &model_path,
        chat_options.threads,
        chat_options.max_cache_mb,
        chat_options.ctx_size,
    )?;
    let api_key = resolve_api_key(&options)?;

    let addr = format!("{}:{}", options.host, options.port);
    let server = tiny_http::Server::http(&addr)
        .map_err(|error| format!("Failed to listen on {addr}: {error}"))?;
    let local_addr = server.server_addr();
    let port = match local_addr {
        tiny_http::ListenAddr::IP(address) => address.port(),
        #[cfg(unix)]
        tiny_http::ListenAddr::Unix(_) => {
            return Err("Listening on Unix sockets is not supported yet.".to_string());
        }
    };

    println!("Loaded model: {}", model_path.display());
    println!("Server started: {}", display_url(&options.host, port));
    if let Some(key) = &api_key {
        println!("API Key: {key}");
    } else {
        println!(
            "API key authentication is disabled (server is listening on a loopback address only)."
        );
    }

    ctrlc::set_handler(|| SHUTDOWN.store(true, Ordering::SeqCst))
        .map_err(|error| format!("Failed to install Ctrl-C handler: {error}"))?;

    let created = openai::now_unix();
    let state = Arc::new(http::ServerState {
        engine: Arc::new(engine),
        default_options: chat_options,
        model_id,
        api_key,
        generation_queue: Arc::new(queue::FifoQueue::new(options.max_concurrent_generations)?),
        created,
    });

    loop {
        if SHUTDOWN.load(Ordering::SeqCst) {
            break;
        }
        match server.recv_timeout(Duration::from_millis(200)) {
            Ok(Some(request)) => {
                let state = Arc::clone(&state);
                let ticket = http::is_generation_request(&request)
                    .then(|| Arc::clone(&state.generation_queue).enqueue());
                std::thread::spawn(move || http::handle(state, request, ticket));
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!("Failed to receive connection: {error}");
            }
        }
    }

    println!("Server stopped.");
    Ok(())
}

// ---- model resolution ----

fn resolve_model(options: &ServeOptions) -> Result<(PathBuf, String), String> {
    let candidate = PathBuf::from(&options.model);
    if candidate.is_file() {
        return Ok((candidate.clone(), model_id_from_path(&candidate)));
    }

    let name = model_name_for_match(&options.model);
    let mut matches = Vec::new();
    if options.model_dir.is_dir() {
        match std::fs::read_dir(&options.model_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    if !entry.file_type().is_ok_and(|ft| ft.is_file()) {
                        continue;
                    }
                    let path = entry.path();
                    if path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                        && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                        && stem.eq_ignore_ascii_case(name)
                    {
                        matches.push(path);
                    }
                }
            }
            Err(error) => {
                return Err(format!(
                    "Failed to read model directory '{}': {error}",
                    options.model_dir.display()
                ));
            }
        }
    }

    let count = matches.len();
    let id = if count == 1 {
        let path = matches.remove(0);
        let id = model_id_from_path(&path);
        return Ok((path, id));
    } else if count == 0 {
        let listing = list_gguf_files(&options.model_dir);
        if listing.is_empty() {
            format!(
                "Model '{}' not found: directory '{}' contains no .gguf files.",
                options.model,
                options.model_dir.display()
            )
        } else {
            format!(
                "Model '{}' not found. Available GGUF models in '{}':\n{}",
                options.model,
                options.model_dir.display(),
                listing
            )
        }
    } else {
        let list = matches
            .iter()
            .map(|path| format!("  {}", path.display()))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Model name '{}' is ambiguous; found {} matches:\n{list}",
            options.model, count
        )
    };
    Err(id)
}

fn model_name_for_match(model: &str) -> &str {
    let name = Path::new(model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model);
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".gguf") {
        &name[..name.len() - 5]
    } else {
        name
    }
}

fn model_id_from_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string())
}

fn list_gguf_files(dir: &Path) -> String {
    if !dir.is_dir() {
        return String::new();
    }
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
            {
                names.push(format!("  {name}"));
            }
        }
    }
    names.sort();
    names.join("\n")
}

// ---- key / url helpers ----

fn resolve_api_key(options: &ServeOptions) -> Result<Option<String>, String> {
    if let Some(key) = &options.api_key {
        return Ok(Some(key.clone()));
    }
    if options.generate_key {
        return Ok(Some(openai::generate_api_key()));
    }
    Ok(None)
}

fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
}

fn display_url(host: &str, port: u16) -> String {
    let display = if is_loopback(host) {
        "localhost"
    } else if host.contains(':') {
        return format!("http://[{host}]:{port}");
    } else {
        host
    };
    format!("http://{display}:{port}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_strips_gguf() {
        assert_eq!(model_name_for_match("qwen2.5-0.5b"), "qwen2.5-0.5b");
        assert_eq!(model_name_for_match("qwen2.5-0.5b.GGUF"), "qwen2.5-0.5b");
        assert_eq!(
            model_name_for_match("path/qwen2.5-0.5b.gguf"),
            "qwen2.5-0.5b"
        );
        assert_eq!(
            model_name_for_match("./dir/qwen2.5-0.5b.GgUf"),
            "qwen2.5-0.5b"
        );
    }

    #[test]
    fn model_id_from_stem() {
        assert_eq!(
            model_id_from_path(Path::new("/data/qwen2.5-0.5b.gguf")),
            "qwen2.5-0.5b"
        );
        assert_eq!(model_id_from_path(Path::new("model.gguf")), "model");
    }

    #[test]
    fn loopback_checking() {
        assert!(is_loopback("127.0.0.1"));
        assert!(is_loopback("localhost"));
        assert!(is_loopback("::1"));
        assert!(!is_loopback("0.0.0.0"));
        assert!(!is_loopback("192.168.1.1"));
    }

    #[test]
    fn display_url_defaults_to_localhost() {
        assert_eq!(display_url("127.0.0.1", 8080), "http://localhost:8080");
        assert_eq!(display_url("::1", 9090), "http://localhost:9090");
        assert_eq!(display_url("0.0.0.0", 3000), "http://0.0.0.0:3000");
    }

    #[test]
    fn resolve_model_by_exact_file() {
        let temp = std::env::temp_dir().join("wolf_test_resolve_file");
        let _ = std::fs::create_dir_all(&temp);
        let file = temp.join("test.gguf");
        std::fs::write(&file, b"dummy").unwrap();

        let options = ServeOptions {
            model: file.display().to_string(),
            model_dir: PathBuf::from("."),
            ..Default::default()
        };
        let (path, id) = resolve_model(&options).unwrap();
        assert_eq!(path, file);
        assert_eq!(id, "test");
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn resolve_model_by_name_in_dir() {
        let temp = std::env::temp_dir().join("wolf_test_resolve_name");
        let _ = std::fs::create_dir_all(&temp);
        let file = temp.join("qwen2.5-0.5b.gguf");
        std::fs::write(&file, b"dummy").unwrap();

        let options = ServeOptions {
            model: "qwen2.5-0.5b".to_string(),
            model_dir: temp.clone(),
            ..Default::default()
        };
        let (path, id) = resolve_model(&options).unwrap();
        assert_eq!(path, file);
        assert_eq!(id, "qwen2.5-0.5b");
        std::fs::remove_file(&file).ok();
        std::fs::remove_dir(&temp).ok();
    }

    #[test]
    fn resolve_model_ambiguous_errors() {
        let temp = std::env::temp_dir().join("wolf_test_resolve_ambig");
        let _ = std::fs::create_dir_all(&temp);
        std::fs::write(temp.join("model-a.gguf"), b"").unwrap();
        std::fs::write(temp.join("model-a.bin"), b"").unwrap(); // not gguf
        std::fs::write(temp.join("MODEL-A.GGUF"), b"").unwrap(); // case-insensitive match

        let options = ServeOptions {
            model: "model-a".to_string(),
            model_dir: temp.clone(),
            ..Default::default()
        };
        let gguf_count = std::fs::read_dir(&temp)
            .unwrap()
            .flatten()
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            })
            .count();

        if gguf_count > 1 {
            let err = resolve_model(&options).unwrap_err();
            assert!(
                err.contains("ambiguous"),
                "expected ambiguity message, got: {err}"
            );
        } else {
            // Case-insensitive filesystems collapse these differently-cased names.
            let (_, id) = resolve_model(&options).unwrap();
            assert!(id.eq_ignore_ascii_case("model-a"), "got: {id}");
        }
        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn resolve_model_not_found_shows_listing() {
        let temp = std::env::temp_dir().join("wolf_test_resolve_missing");
        let _ = std::fs::create_dir_all(&temp);
        std::fs::write(temp.join("available.gguf"), b"").unwrap();

        let options = ServeOptions {
            model: "nonexistent".to_string(),
            model_dir: temp.clone(),
            ..Default::default()
        };
        let err = resolve_model(&options).unwrap_err();
        assert!(err.contains("available.gguf"), "got: {err}");
        std::fs::remove_dir_all(&temp).ok();
    }
}
