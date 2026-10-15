//! Multi-turn GGUF chat powered by llama.cpp.

mod llama;

use crate::cli::ChatOptions;
use crate::context::{ContextArchive, ContextStore};
use std::path::Path;

pub fn chat(
    path: &Path,
    options: ChatOptions,
    archive: ContextArchive,
    store: ContextStore,
) -> Result<(), String> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return Err(format!("对话推理仅支持 GGUF 模型文件：{}", path.display()));
    }
    llama::chat(path, options, archive, store)
}
