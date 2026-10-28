//! Multi-turn GGUF chat powered by llama.cpp.

mod llama;

pub(crate) use llama::{ChatMessage, Engine, Generation, TextSink};

use crate::context::{ContextArchive, ContextStore};
use crate::options::ChatOptions;
use std::path::Path;

pub fn chat(
    path: &Path,
    options: ChatOptions,
    archive: ContextArchive,
    store: ContextStore,
) -> Result<(), String> {
    // Inspection supports more formats, but inference is intentionally restricted to GGUF.
    // 模型检查支持更多格式，但推理入口有意仅接受 GGUF。
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return Err(format!(
            "Chat inference only supports GGUF model files: {}",
            path.display()
        ));
    }
    llama::chat(path, options, archive, store)
}

pub(crate) fn resolve_threads(threads: Option<usize>) -> usize {
    threads.unwrap_or_else(|| num_cpus::get_physical().max(1))
}
