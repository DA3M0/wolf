//! Reusable local model chat, context management, and model inspection APIs.

pub mod chat;
pub mod context;
pub mod inspect;
pub mod options;

pub use context::{ContextArchive, ContextStore, ConversationTurn};
pub use inspect::{ModelInfo, TensorInfo, inspect_model};
pub use options::{
    ChatOptions, DEFAULT_CHAT_OPTIONS, DEFAULT_CONTEXT_TOKENS, DEFAULT_MAX_TOKENS_PER_TURN,
    DEFAULT_REPEAT_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
};
