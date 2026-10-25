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
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return Err(format!(
            "Chat inference only supports GGUF model files: {}",
            path.display()
        ));
    }
    configure_threads(options.threads)?;
    llama::chat(path, options, archive, store)
}

pub(crate) fn configure_threads(threads: Option<usize>) -> Result<(), String> {
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("Failed to set CPU inference thread count: {error}"))?;
    }
    Ok(())
}
