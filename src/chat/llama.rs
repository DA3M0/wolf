//! GGUF chat powered by llama.cpp through llama-cpp-4.
//!
//! The reusable inference core ([`Engine`]) is shared by the interactive CLI
//! loop and the `wolf serve` HTTP API: the loop owns stdin/stdout, the engine
//! owns the model and hands generated text to a [`TextSink`].

use crate::context::{ContextArchive, ContextStore, ConversationTurn};
use crate::options::{ChatOptions, DEFAULT_CONTEXT_TOKENS};
use llama_cpp_4::prelude::*;
use std::io::{self, BufRead, Write};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
const BATCH_SIZE: usize = 512;
const CHATML_END_MARKER: &str = "<|im_end|>";

/// 一条对话消息(角色 + 文本内容)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    fn to_llama(&self) -> std::result::Result<LlamaChatMessage, String> {
        LlamaChatMessage::new(self.role.clone(), self.content.clone())
            .map_err(|error| format!("Failed to create '{}' message: {error}", self.role))
    }
}

/// 生成过程中可见文本的输出目标。
pub trait TextSink {
    /// 输出一段可见文本;返回 `Err` 表示应中止本轮生成(例如客户端已断开)。
    fn push(&mut self, text: &str) -> std::result::Result<(), String>;
    /// 本轮生成结束时的收尾(CLI 补换行,服务端补收尾分片)。
    fn finish(&mut self) -> std::result::Result<(), String> {
        Ok(())
    }
    /// 是否应中断本轮生成(CLI 的 Ctrl-C);服务端由 [`TextSink::push`] 报错表示断开。
    fn interrupted(&mut self) -> bool {
        false
    }
}

/// 本轮生成结束的原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// 一次完整生成的结果。
#[derive(Debug, Clone)]
pub struct Generation {
    /// 生成的可见文本(不含被拦截的结束标记)。
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub finish_reason: FinishReason,
}

/// 已加载的 GGUF 模型及其推理配置。
///
/// `LlamaModel` 是 `Send + Sync`,因此 `Engine` 可放进 `Arc` 供多个工作线程
/// 共享;但 `LlamaContext` 不是 `Send`,必须由各线程自行创建并独占使用。
pub struct Engine {
    /// 字段顺序即析构顺序:模型必须先于 backend 释放。
    templates: std::result::Result<ChatTemplates, String>,
    model: LlamaModel,
    backend: LlamaBackend,
    stop_markers: Vec<String>,
    context_size: u32,
    threads: Option<usize>,
}

impl Engine {
    /// 初始化 llama.cpp、加载模型,并从对话模板探测结束标记。
    pub fn load(path: &Path, threads: Option<usize>) -> std::result::Result<Self, String> {
        let mut backend = LlamaBackend::init()
            .map_err(|error| format!("Failed to initialize llama.cpp: {error}"))?;
        backend.void_logs();
        let model = LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default())
            .map_err(|error| format!("Failed to load GGUF model: {error}"))?;
        let default_context_size = u32::try_from(DEFAULT_CONTEXT_TOKENS)
            .map_err(|_| "Context length exceeds the range supported by llama.cpp.".to_string())?;
        let context_size = default_context_size.min(model.n_ctx_train()).max(1);
        let stop_markers = detect_stop_markers(&model)?;
        let templates = ChatTemplates::from_model(&model, None)
            .map_err(|error| format!("Failed to initialize the model chat template: {error}"));
        Ok(Self {
            model,
            templates,
            backend,
            stop_markers,
            context_size,
            threads,
        })
    }

    /// 为一次推理创建独立的 llama.cpp 上下文。
    pub fn new_context(&self) -> std::result::Result<LlamaContext<'_>, String> {
        let mut context_params =
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.context_size));
        if let Some(threads) = self.threads {
            let threads = i32::try_from(threads)
                .map_err(|_| "--threads exceeds the range supported by llama.cpp.".to_string())?;
            context_params = context_params.with_n_threads(threads);
        }
        self.model
            .new_context(&self.backend, context_params)
            .map_err(|error| format!("Failed to create llama.cpp inference context: {error}"))
    }

    /// 应用对话模板并编码。超出上下文上限时截断最早的完整轮次。
    ///
    /// 返回编码后的 token 与被截断的轮次数量(`messages` 已就地同步截断)。
    pub fn prepare(
        &self,
        messages: &mut Vec<ChatMessage>,
        system: Option<&str>,
        max_tokens: usize,
    ) -> std::result::Result<(Vec<LlamaToken>, usize), String> {
        let limit = usize::try_from(self.context_size).map_err(|_| {
            "Model context length exceeds the range supported by this platform.".to_string()
        })?;
        let mut dropped_turns = 0;
        loop {
            let mut prompt_messages =
                Vec::with_capacity(messages.len() + usize::from(system.is_some()));
            if let Some(system) = system {
                prompt_messages.push(
                    LlamaChatMessage::new("system".into(), system.into())
                        .map_err(|error| format!("Failed to create system message: {error}"))?,
                );
            }

            for message in messages.iter() {
                prompt_messages.push(message.to_llama()?);
            }
            let prompt = self
                .model
                .apply_chat_template(None, &prompt_messages, true)
                .map_err(|error| format!("Failed to apply GGUF chat template: {error}"))?;
            let tokens = self
                .model
                .str_to_token(&prompt, AddBos::Always)
                .map_err(|error| format!("Failed to encode input text: {error}"))?;
            if tokens.len().saturating_add(max_tokens) <= limit {
                return Ok((tokens, dropped_turns));
            }
            if messages.len() <= 1 {
                return Err(format!(
                    "Input plus reserved generation tokens exceeds the model context limit of {limit}; shorten the input or reduce --max-tokens."
                ));
            }
            messages.drain(..2);
            dropped_turns += 1;
            eprintln!("Context limit nearly reached; earlier conversation turns were truncated.");
        }
    }

    /// Render an OpenAI tool request using llama.cpp's model-aware chat parser.
    pub fn prepare_tool_chat(
        &self,
        messages_json: &str,
        tools_json: &str,
        tool_choice: ToolChoice,
        max_tokens: usize,
    ) -> std::result::Result<(Vec<LlamaToken>, ChatParams), String> {
        let templates = match &self.templates {
            Ok(templates) => templates,
            Err(error) => return Err(error.clone()),
        };
        let mut messages: Vec<serde_json::Value> = serde_json::from_str(messages_json)
            .map_err(|error| format!("Failed to parse tool request message: {error}"))?;
        let limit = usize::try_from(self.context_size).map_err(|_| {
            "Model context length exceeds the range supported by this platform.".to_string()
        })?;
        loop {
            let messages_json = serde_json::to_string(&messages)
                .map_err(|error| format!("Failed to encode tool request message: {error}"))?;
            let applied =
                Self::apply_tool_chat(templates, &messages_json, tools_json, tool_choice)?;
            let tokens = self
                .model
                .str_to_token(&applied.prompt, AddBos::Always)
                .map_err(|error| {
                    format!("Failed to encode input text containing tools: {error}")
                })?;
            if tokens.len().saturating_add(max_tokens) <= limit {
                return Ok((tokens, applied));
            }
            if !drop_oldest_tool_turn(&mut messages) {
                return Err(format!(
                    "Input plus reserved generation tokens exceeds the model context limit of {limit}; shorten the conversation or reduce --max-tokens."
                ));
            }
            eprintln!("Context limit nearly reached; earlier conversation turns were truncated.");
        }
    }

    fn apply_tool_chat(
        templates: &ChatTemplates,
        messages_json: &str,
        tools_json: &str,
        tool_choice: ToolChoice,
    ) -> std::result::Result<ChatParams, String> {
        if let Ok(params) = templates.apply(
            &ChatApplyParams::new(messages_json)
                .with_tools(tools_json)
                .with_tool_choice(tool_choice),
        ) && !params.grammar.is_empty()
        {
            return Ok(params);
        }
        let messages = Self::tool_prompt_fallback(messages_json, tools_json, tool_choice)?;
        templates
            .apply(&ChatApplyParams::new(messages))
            .map_err(|error| format!("Failed to render tool-call prompt: {error}"))
    }

    fn tool_prompt_fallback(
        messages_json: &str,
        tools_json: &str,
        choice: ToolChoice,
    ) -> std::result::Result<String, String> {
        let instruction = match choice {
            ToolChoice::None => format!(
                "Do not call any function. Respond only as a JSON object of this form: \
                 {{\"content\":\"your response\"}}. Available function definitions (do not call them): {tools_json}"
            ),
            ToolChoice::Required => format!(
                "Call one of the available functions. Respond only as a JSON object of this form: \
                 {{\"tool_calls\":[{{\"name\":\"function_name\",\"arguments\":{{}}}}]}}. \
                 Use valid JSON arguments matching the function schema. Functions: {tools_json}"
            ),
            ToolChoice::Auto => format!(
                "If a function is needed, respond only as a JSON object of this form: \
                 {{\"tool_calls\":[{{\"name\":\"function_name\",\"arguments\":{{}}}}]}}. \
                 Otherwise respond only as {{\"content\":\"your response\"}}. \
                 Use valid JSON arguments matching the function schema. Functions: {tools_json}"
            ),
        };
        let mut messages: Vec<serde_json::Value> = serde_json::from_str(messages_json)
            .map_err(|error| format!("Failed to parse tool prompt message: {error}"))?;
        messages.insert(
            0,
            serde_json::json!({"role": "system", "content": instruction}),
        );
        serde_json::to_string(&messages)
            .map_err(|error| format!("Failed to encode tool prompt messages: {error}"))
    }

    /// 执行一次生成,把可见文本交给 `sink`。
    pub fn generate(
        &self,
        context: &mut LlamaContext<'_>,
        prompt: &[LlamaToken],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
    ) -> std::result::Result<Generation, String> {
        self.generate_inner(context, prompt, options, extra_stop, sink, None)
    }

    /// Generate from a template-rendered request and constrain it with its grammar.
    pub fn generate_with_chat_params(
        &self,
        context: &mut LlamaContext<'_>,
        prompt: &[LlamaToken],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
        chat_params: &ChatParams,
    ) -> std::result::Result<Generation, String> {
        self.generate_inner(
            context,
            prompt,
            options,
            extra_stop,
            sink,
            Some(chat_params),
        )
    }

    fn generate_inner(
        &self,
        context: &mut LlamaContext<'_>,
        prompt: &[LlamaToken],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
        chat_params: Option<&ChatParams>,
    ) -> std::result::Result<Generation, String> {
        if prompt.is_empty() {
            return Err("Failed to convert input to model tokens; please try again.".into());
        }
        let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
        context.clear_kv_cache();
        for (chunk_index, chunk) in prompt.chunks(BATCH_SIZE).enumerate() {
            batch.clear();
            let chunk_start = chunk_index.saturating_mul(BATCH_SIZE);
            let last_position = i32::try_from(chunk_start + chunk.len() - 1).map_err(|_| {
                "Input token count exceeds the range supported by llama.cpp.".to_string()
            })?;
            for (offset, token) in chunk.iter().copied().enumerate() {
                let position = i32::try_from(chunk_start + offset).map_err(|_| {
                    "Input token count exceeds the range supported by llama.cpp.".to_string()
                })?;
                batch
                    .add(token, position, &[0], position == last_position)
                    .map_err(|error| format!("Failed to prepare input batch: {error}"))?;
            }
            context
                .decode(&mut batch)
                .map_err(|error| format!("Model inference failed: {error}"))?;
        }
        let mut sampler = make_sampler(&self.model, options, chat_params)?;
        if !chat_params.is_some_and(|params| !params.grammar.is_empty()) {
            for token in prompt.iter().copied() {
                sampler.accept(token);
            }
        }
        let mut stop_markers = self.stop_markers.clone();
        stop_markers.extend(extra_stop.iter().cloned());
        stop_markers.retain(|marker| !marker.is_empty());
        stop_markers.sort();
        stop_markers.dedup();
        let mut stop_sequence = StopSequence::new(stop_markers);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut answer = String::new();
        let mut finish_reason = FinishReason::Length;
        let mut position = i32::try_from(prompt.len()).map_err(|_| {
            "Input token count exceeds the range supported by llama.cpp.".to_string()
        })?;
        let mut generated = 0usize;

        for _ in 0..options.max_tokens {
            if sink.interrupted() {
                eprintln!("\n[Generation interrupted]");
                finish_reason = FinishReason::Stop;
                break;
            }
            let token = sampler.sample(context, batch.n_tokens() - 1);
            if is_stop_token(
                token,
                self.model.token_eot(),
                self.model.is_eog_token(token),
            ) {
                finish_reason = FinishReason::Stop;
                break;
            }
            sampler.accept(token);

            let bytes = self
                .model
                .token_to_bytes(token, Special::Tokenize)
                .map_err(|error| format!("Failed to decode model response: {error}"))?;
            let mut text = String::with_capacity(bytes.len().saturating_mul(3).max(32));
            let (decode_result, _, _) = decoder.decode_to_string(&bytes, &mut text, false);
            if matches!(decode_result, encoding_rs::CoderResult::OutputFull) {
                return Err(
                    "UTF-8 decoding buffer is too small for the model response.".to_string()
                );
            }
            let (text, stopped) = stop_sequence.push(&text);
            answer.push_str(&text);
            sink.push(&text)?;
            generated += 1;
            if stopped {
                finish_reason = FinishReason::Stop;
                break;
            }

            batch.clear();
            batch
                .add(token, position, &[0], true)
                .map_err(|error| format!("Failed to prepare generation batch: {error}"))?;
            context
                .decode(&mut batch)
                .map_err(|error| format!("Model inference failed: {error}"))?;
            position = position
                .checked_add(1)
                .ok_or_else(|| "Model context position overflow.".to_string())?;
            if u32::try_from(position).unwrap_or(u32::MAX) >= context.n_ctx() {
                break;
            }
        }

        let mut final_text = String::new();
        let (decode_result, _, _) = decoder.decode_to_string(&[], &mut final_text, true);
        if matches!(decode_result, encoding_rs::CoderResult::OutputFull) {
            return Err("UTF-8 decoding buffer is too small for the model response.".to_string());
        }
        let (final_text, _) = stop_sequence.push(&final_text);
        answer.push_str(&final_text);
        sink.push(&final_text)?;
        let remaining = stop_sequence.finish();
        answer.push_str(&remaining);
        sink.push(&remaining)?;
        sink.finish()?;

        Ok(Generation {
            text: answer,
            prompt_tokens: prompt.len(),
            generated_tokens: generated,
            finish_reason,
        })
    }
}

fn drop_oldest_tool_turn(messages: &mut Vec<serde_json::Value>) -> bool {
    let Some(start) = messages.iter().position(|message| {
        message.get("role").and_then(serde_json::Value::as_str) == Some("user")
    }) else {
        return false;
    };
    let Some(next_user) = messages.iter().skip(start + 1).position(|message| {
        message.get("role").and_then(serde_json::Value::as_str) == Some("user")
    }) else {
        return false;
    };
    messages.drain(start..start + 1 + next_user);
    true
}

pub fn chat(
    path: &Path,
    options: ChatOptions,
    archive: ContextArchive,
    store: ContextStore,
) -> std::result::Result<(), String> {
    let engine = Engine::load(path, options.threads)?;
    ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
        .map_err(|error| format!("Failed to install Ctrl-C handler: {error}"))?;
    let mut context = engine.new_context()?;
    println!("Loaded llama.cpp GGUF model. Enter /exit to quit.");
    chat_loop(&engine, &mut context, &options, archive, store)
}

/// CLI 的文本输出目标:直接写 stdout 并逐段刷新。
struct StdoutSink {
    output: io::StdoutLock<'static>,
}

impl StdoutSink {
    fn new() -> Self {
        Self {
            output: io::stdout().lock(),
        }
    }
}

impl TextSink for StdoutSink {
    fn push(&mut self, text: &str) -> std::result::Result<(), String> {
        self.output
            .write_all(text.as_bytes())
            .and_then(|()| self.output.flush())
            .map_err(|error| format!("Failed to write model response: {error}"))
    }

    fn finish(&mut self) -> std::result::Result<(), String> {
        writeln!(self.output)
            .and_then(|()| self.output.flush())
            .map_err(|error| format!("Failed to finish model response: {error}"))
    }

    fn interrupted(&mut self) -> bool {
        INTERRUPTED.swap(false, Ordering::SeqCst)
    }
}

fn chat_loop(
    engine: &Engine,
    context: &mut LlamaContext<'_>,
    options: &ChatOptions,
    mut archive: ContextArchive,
    store: ContextStore,
) -> std::result::Result<(), String> {
    let mut messages = Vec::with_capacity(archive.turns.len() * 2);
    for turn in &archive.turns {
        messages.push(ChatMessage::new("user", turn.user.clone()));
        messages.push(ChatMessage::new("assistant", turn.assistant.clone()));
    }
    let mut input = String::new();
    let stdin = io::stdin();
    let mut stdin = stdin.lock();

    loop {
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|error| format!("Failed to display input prompt: {error}"))?;
        input.clear();
        let bytes_read = stdin
            .read_line(&mut input)
            .map_err(|error| format!("Failed to read input: {error}"))?;
        if bytes_read == 0 {
            println!("\nExited.");
            return Ok(());
        }
        if INTERRUPTED.swap(false, Ordering::SeqCst) {
            println!("(Input cancelled; enter /exit or press Ctrl-D to quit)");
            continue;
        }

        let prompt = input.trim();
        match prompt {
            "/exit" | "/quit" => {
                println!("Exited.");
                return Ok(());
            }
            "/clear" => {
                messages.clear();
                archive.turns.clear();
                context.clear_kv_cache();
                println!("Conversation history cleared.");
                continue;
            }
            "/save" => {
                eprintln!(
                    "Usage: /save <name> (name may contain only ASCII letters, digits, periods, hyphens, and underscores)"
                );
                continue;
            }
            "/help" => {
                println!(
                    "Chat commands:\n  /save <name>  Save the current conversation\n  /clear        Clear conversation history\n  /help         Show this help\n  /exit         Quit (or use /quit or Ctrl-D)"
                );
                continue;
            }
            "" => continue,
            _ => {
                if let Some(name) = prompt.strip_prefix("/save ") {
                    let name = name.trim();
                    if name.is_empty() || name.chars().any(char::is_whitespace) {
                        eprintln!(
                            "Usage: /save <name> (name may contain only ASCII letters, digits, periods, hyphens, and underscores)"
                        );
                        continue;
                    }
                    archive.name = name.to_string();
                    match store.save(&archive) {
                        Ok(()) => println!("Saved context '{name}'."),
                        Err(error) => eprintln!("{error}"),
                    }
                    continue;
                }
            }
        }

        messages.push(ChatMessage::new("user", prompt));
        let (tokens, dropped_turns) =
            engine.prepare(&mut messages, options.system.as_deref(), options.max_tokens)?;
        archive.turns.drain(..dropped_turns);

        println!("Processing input and generating a response…");
        io::stdout()
            .flush()
            .map_err(|error| format!("Failed to flush output: {error}"))?;
        let generation_started = Instant::now();
        let mut sink = StdoutSink::new();
        let generation = engine.generate(context, &tokens, options, &[], &mut sink)?;
        messages.push(ChatMessage::new("assistant", generation.text.clone()));
        archive.turns.push(ConversationTurn {
            user: prompt.to_string(),
            assistant: generation.text,
        });
        let elapsed = generation_started.elapsed().as_secs_f64();
        if generation.generated_tokens > 0 && elapsed > 0.0 {
            eprintln!(
                "[Generated {} tokens at {:.1} tokens/s]",
                generation.generated_tokens,
                generation.generated_tokens as f64 / elapsed
            );
        }
    }
}

fn detect_stop_markers(model: &LlamaModel) -> std::result::Result<Vec<String>, String> {
    const PROBE: &str = "wolf_template_stop_probe_7f3a";

    let messages = [
        LlamaChatMessage::new("user".into(), "probe".into())
            .map_err(|error| format!("Failed to create template-probe user message: {error}"))?,
        LlamaChatMessage::new("assistant".into(), PROBE.into()).map_err(|error| {
            format!("Failed to create template-probe assistant message: {error}")
        })?,
    ];
    let rendered = model
        .apply_chat_template(None, &messages, false)
        .map_err(|error| {
            format!("Failed to detect end markers from the GGUF chat template: {error}")
        })?;

    let mut markers = template_ending(&rendered, PROBE)
        .into_iter()
        .collect::<Vec<_>>();
    let end_of_turn = model.token_eot();
    if end_of_turn.0 >= 0 {
        let bytes = model
            .token_to_bytes(end_of_turn, Special::Tokenize)
            .map_err(|error| format!("Failed to read model turn end marker: {error}"))?;
        let marker = String::from_utf8(bytes)
            .map_err(|error| format!("Model turn end marker is not valid UTF-8: {error}"))?;
        if !marker.is_empty() {
            markers.push(marker);
        }
    }
    markers.push(CHATML_END_MARKER.to_string());
    markers.sort();
    markers.dedup();
    Ok(markers)
}

fn template_ending(rendered: &str, probe: &str) -> Option<String> {
    let probe_end = rendered.rfind(probe)?.checked_add(probe.len())?;
    let ending = rendered[probe_end..].trim();
    (!ending.is_empty()).then(|| ending.to_string())
}

fn is_stop_token(token: LlamaToken, end_of_turn: LlamaToken, is_eog: bool) -> bool {
    is_eog || token == end_of_turn
}

struct StopSequence {
    markers: Vec<String>,
    pending: String,
}

impl StopSequence {
    fn new(markers: Vec<String>) -> Self {
        Self {
            markers: markers
                .into_iter()
                .filter(|marker| !marker.is_empty())
                .collect(),
            pending: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> (String, bool) {
        if self.markers.is_empty() {
            return (text.to_owned(), false);
        }
        self.pending.push_str(text);
        let stop = self
            .markers
            .iter()
            .filter_map(|marker| self.pending.find(marker).map(|index| (index, marker.len())))
            .min_by_key(|(index, length)| (*index, *length));
        if let Some((index, _)) = stop {
            let visible = self.pending[..index].to_owned();
            self.pending.clear();
            return (visible, true);
        }

        let mut held_len = 0;
        for marker in &self.markers {
            let max_prefix_len = self.pending.len().min(marker.len() - 1);
            for length in (1..=max_prefix_len).rev() {
                if marker
                    .get(..length)
                    .is_some_and(|prefix| self.pending.ends_with(prefix))
                {
                    held_len = held_len.max(length);
                    break;
                }
            }
        }
        let mut visible_end = self.pending.len() - held_len;
        while !self.pending.is_char_boundary(visible_end) {
            visible_end -= 1;
        }
        let visible = self.pending[..visible_end].to_owned();
        self.pending.drain(..visible_end);
        (visible, false)
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

fn make_sampler(
    model: &LlamaModel,
    options: &ChatOptions,
    chat_params: Option<&ChatParams>,
) -> std::result::Result<LlamaSampler, String> {
    let grammar = match chat_params {
        Some(params) => make_grammar_sampler(model, params)?,
        None => None,
    };
    if options.temperature < 1e-7 {
        let mut samplers = Vec::new();
        if options.repeat_penalty > 1.0 {
            samplers.push(LlamaSampler::penalties_simple(
                model.n_vocab(),
                64,
                options.repeat_penalty,
            ));
        }
        if let Some(grammar) = grammar {
            samplers.push(grammar);
        }
        samplers.push(LlamaSampler::greedy());
        return Ok(LlamaSampler::chain_simple(samplers));
    }

    let mut samplers = Vec::new();
    if options.repeat_penalty > 1.0 {
        samplers.push(LlamaSampler::penalties_simple(
            model.n_vocab(),
            64,
            options.repeat_penalty,
        ));
    }
    if let Some(grammar) = grammar {
        samplers.push(grammar);
    }
    if options.top_k > 0 {
        let top_k = i32::try_from(options.top_k)
            .map_err(|_| "--top-k exceeds the range supported by llama.cpp.".to_string())?;
        samplers.push(LlamaSampler::top_k(top_k));
    }
    samplers.push(LlamaSampler::top_p(options.top_p as f32, 1));
    samplers.push(LlamaSampler::temp(options.temperature as f32));
    let seed = options.seed.unwrap_or_else(random_seed);
    let seed = u32::try_from(seed % u64::from(u32::MAX))
        .map_err(|_| "Random seed exceeds the range supported by llama.cpp.".to_string())?;
    samplers.push(LlamaSampler::dist(seed));
    Ok(LlamaSampler::chain_simple(samplers))
}

fn make_grammar_sampler(
    model: &LlamaModel,
    params: &ChatParams,
) -> std::result::Result<Option<LlamaSampler>, String> {
    if params.grammar.is_empty() {
        return Ok(None);
    }
    let (patterns, tokens) = params.sampler_triggers();
    let lazy = params.grammar_lazy && !(patterns.is_empty() && tokens.is_empty());
    let mut sampler = if lazy {
        let patterns = patterns.iter().map(String::as_str).collect::<Vec<_>>();
        LlamaSampler::try_grammar_lazy_patterns(model, &params.grammar, "root", &patterns, &tokens)
            .map_err(|error| {
                format!("Failed to initialize tool-call grammar constraints: {error}")
            })?
    } else {
        LlamaSampler::try_grammar(model, &params.grammar, "root").map_err(|error| {
            format!("Failed to initialize tool-call grammar constraints: {error}")
        })?
    };

    if !lazy {
        let generation_prompt = params.generation_prompt();
        let tokens = model
            .str_to_token(generation_prompt, AddBos::Never)
            .map_err(|error| {
                format!("Failed to encode chat template generation prefix: {error}")
            })?;
        let starts_with_space = generation_prompt.starts_with(char::is_whitespace);
        for (index, token) in tokens.into_iter().enumerate() {
            if index == 0 && !starts_with_space {
                let piece = model
                    .token_to_str(token, Special::Tokenize)
                    .map_err(|error| format!("Failed to read generation prefix token: {error}"))?;
                if piece.starts_with(char::is_whitespace) {
                    continue;
                }
            }
            sampler.accept(token);
        }
    }
    Ok(Some(sampler))
}

fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0x5EED_1234_ABCD_EF01)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_on_model_end_of_turn_token_even_when_not_marked_eog() {
        let end_of_turn = LlamaToken::new(42);
        assert!(is_stop_token(end_of_turn, end_of_turn, false));
    }

    #[test]
    fn preserves_generic_end_of_generation_stopping() {
        let end_of_turn = LlamaToken::new(42);
        let eog = LlamaToken::new(43);
        assert!(is_stop_token(eog, end_of_turn, true));
    }

    #[test]
    fn tool_prompt_fallback_includes_json_contract_and_tools() {
        let messages = r#"[{"role":"user","content":"weather"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"weather"}}]"#;
        let prompt = Engine::tool_prompt_fallback(messages, tools, ToolChoice::Auto).unwrap();
        let messages: serde_json::Value = serde_json::from_str(&prompt).unwrap();
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .contains(r#"{"tool_calls""#)
        );
        assert_eq!(messages[1]["content"], "weather");
    }

    #[test]
    fn truncates_complete_tool_turns_without_dropping_system_or_latest_user() {
        let mut messages: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
                {"role":"system","content":"system"},
                {"role":"user","content":"old question"},
                {"role":"assistant","tool_calls":[{"id":"call_1"}]},
                {"role":"tool","tool_call_id":"call_1","content":"result"},
                {"role":"user","content":"latest question"}
            ]"#,
        )
        .unwrap();
        assert!(drop_oldest_tool_turn(&mut messages));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["content"], "latest question");
        assert!(!drop_oldest_tool_turn(&mut messages));
    }

    #[test]
    fn does_not_stop_on_ordinary_tokens() {
        let end_of_turn = LlamaToken::new(42);
        let ordinary = LlamaToken::new(44);
        assert!(!is_stop_token(ordinary, end_of_turn, false));
    }

    #[test]
    fn stops_when_model_end_marker_is_spelled_across_tokens() {
        let mut matcher = StopSequence::new(vec!["<|eot_id|>".into(), CHATML_END_MARKER.into()]);
        let (first, stopped) = matcher.push("Answer <|im_");
        assert_eq!(first, "Answer ");
        assert!(!stopped);

        let (second, stopped) = matcher.push("end|>user");
        assert_eq!(second, "");
        assert!(stopped);
        assert_eq!(matcher.finish(), "");
    }

    #[test]
    fn flushes_partial_marker_prefix_when_generation_finishes() {
        let mut matcher = StopSequence::new(vec![CHATML_END_MARKER.into()]);
        let (visible, stopped) = matcher.push("Answer <|im_");
        assert_eq!(visible, "Answer ");
        assert!(!stopped);
        assert_eq!(matcher.finish(), "<|im_");
    }

    #[test]
    fn preserves_utf8_text_before_a_partial_stop_marker() {
        let mut matcher = StopSequence::new(vec![CHATML_END_MARKER.into()]);
        let (visible, stopped) = matcher.push("你好!<");
        assert_eq!(visible, "你好!");
        assert!(!stopped);
        assert_eq!(matcher.finish(), "<");
    }

    #[test]
    fn extracts_non_llama_turn_ending_from_rendered_template() {
        let rendered = "<|user|>probe<|assistant|>wolf_template_stop_probe_7f3a<|end_of_turn|>\n";
        assert_eq!(
            template_ending(rendered, "wolf_template_stop_probe_7f3a").as_deref(),
            Some("<|end_of_turn|>")
        );
    }

    #[test]
    fn returns_no_template_stop_when_rendered_turn_has_no_suffix() {
        assert_eq!(
            template_ending(
                "assistant: wolf_template_stop_probe_7f3a",
                "wolf_template_stop_probe_7f3a"
            ),
            None
        );
    }

    #[test]
    fn reports_finish_reasons_in_openai_wording() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Length.as_str(), "length");
    }
}
