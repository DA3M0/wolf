//! Multi-turn GGUF chat powered by llama.cpp.

mod llama;

use crate::context::{ContextArchive, ContextStore};
use crate::options::ChatOptions;
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
    configure_threads(options.threads)?;
    llama::chat(path, options, archive, store)
}

fn configure_threads(threads: Option<usize>) -> Result<(), String> {
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("无法设置 CPU 推理线程数：{error}"))?;
    }
    Ok(())
}
