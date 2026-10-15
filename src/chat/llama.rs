//! GGUF chat powered by llama.cpp through llama-cpp-4.

use crate::cli::{ChatOptions, DEFAULT_CONTEXT_TOKENS};
use crate::context::{ContextArchive, ContextStore, ConversationTurn};
use llama_cpp_4::prelude::*;
use std::io::{self, BufRead, Write};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
const BATCH_SIZE: usize = 512;

pub fn chat(
    path: &Path,
    options: ChatOptions,
    archive: ContextArchive,
    store: ContextStore,
) -> Result<(), String> {
    let mut backend =
        LlamaBackend::init().map_err(|error| format!("无法初始化 llama.cpp：{error}"))?;
    backend.void_logs();
    let model = LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default())
        .map_err(|error| format!("无法加载 GGUF 模型：{error}"))?;
    let default_context_size = u32::try_from(DEFAULT_CONTEXT_TOKENS)
        .map_err(|_| "上下文长度超出 llama.cpp 支持范围。".to_string())?;
    let context_size = default_context_size.min(model.n_ctx_train()).max(1);
    let mut context_params =
        LlamaContextParams::default().with_n_ctx(NonZeroU32::new(context_size));
    if let Some(threads) = options.threads {
        let threads = i32::try_from(threads)
            .map_err(|_| "--threads 超出 llama.cpp 支持范围。".to_string())?;
        context_params = context_params.with_n_threads(threads);
    }
    let mut context = model
        .new_context(&backend, context_params)
        .map_err(|error| format!("无法创建 llama.cpp 推理上下文：{error}"))?;

    ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
        .map_err(|error| format!("无法安装 Ctrl-C 处理器：{error}"))?;
    println!("已加载 llama.cpp GGUF 模型。输入 /exit 退出。");
    chat_loop(&model, &mut context, &options, archive, store)
}

fn chat_loop(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    options: &ChatOptions,
    mut archive: ContextArchive,
    store: ContextStore,
) -> Result<(), String> {
    let mut messages = Vec::with_capacity(archive.turns.len() * 2);
    for turn in &archive.turns {
        messages.push(
            LlamaChatMessage::new("user".into(), turn.user.clone().into())
                .map_err(|error| format!("无法恢复上下文中的用户消息：{error}"))?,
        );
        messages.push(
            LlamaChatMessage::new("assistant".into(), turn.assistant.clone().into())
                .map_err(|error| format!("无法恢复上下文中的助手消息：{error}"))?,
        );
    }
    let mut input = String::new();
    let stdin = io::stdin();
    let mut stdin = stdin.lock();

    loop {
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|error| format!("无法显示输入提示：{error}"))?;
        input.clear();
        let bytes_read = stdin
            .read_line(&mut input)
            .map_err(|error| format!("读取输入失败：{error}"))?;
        if bytes_read == 0 {
            println!("\n已退出。");
            return Ok(());
        }
        if INTERRUPTED.swap(false, Ordering::SeqCst) {
            println!("（已取消输入；/exit 或 Ctrl-D 退出）");
            continue;
        }

        let prompt = input.trim();
        match prompt {
            "/exit" | "/quit" => {
                println!("已退出。");
                return Ok(());
            }
            "/clear" => {
                messages.clear();
                archive.turns.clear();
                context.clear_kv_cache();
                println!("已清空对话历史。");
                continue;
            }
            "/save" => {
                eprintln!(
                    "用法：/save <名称>（名称仅可包含 ASCII 字母、数字、点、短横线和下划线）"
                );
                continue;
            }
            "/help" => {
                println!(
                    "对话内命令：\n  /save <名称>  保存当前对话\n  /clear        清空对话历史\n  /help         显示本帮助\n  /exit         退出（或 /quit、Ctrl-D）"
                );
                continue;
            }
            "" => continue,
            _ => {
                if let Some(name) = prompt.strip_prefix("/save ") {
                    let name = name.trim();
                    if name.is_empty() || name.chars().any(char::is_whitespace) {
                        eprintln!(
                            "用法：/save <名称>（名称仅可包含 ASCII 字母、数字、点、短横线和下划线）"
                        );
                        continue;
                    }
                    archive.name = name.to_string();
                    match store.save(&archive) {
                        Ok(()) => println!("已保存上下文“{name}”。"),
                        Err(error) => eprintln!("{error}"),
                    }
                    continue;
                }
            }
        }

        let user_message = LlamaChatMessage::new("user".into(), prompt.into())
            .map_err(|error| format!("无法创建用户消息：{error}"))?;
        messages.push(user_message);
        let (tokens, dropped_turns) = fit_prompt(
            model,
            options.system.as_deref(),
            &mut messages,
            options.max_tokens,
            context.n_ctx(),
        )?;
        archive.turns.drain(..dropped_turns);

        println!("正在处理输入并生成回答…");
        io::stdout()
            .flush()
            .map_err(|error| format!("无法刷新输出：{error}"))?;
        let generation_started = Instant::now();
        let (answer, generated) = generate(model, context, &tokens, options)?;
        messages.push(
            LlamaChatMessage::new("assistant".into(), answer.clone())
                .map_err(|error| format!("无法创建助手消息：{error}"))?,
        );
        archive.turns.push(ConversationTurn {
            user: prompt.to_string(),
            assistant: answer,
        });
        let elapsed = generation_started.elapsed().as_secs_f64();
        if generated > 0 && elapsed > 0.0 {
            eprintln!(
                "[生成 {generated} tokens，{:.1} tokens/s]",
                generated as f64 / elapsed
            );
        }
    }
}

fn fit_prompt(
    model: &LlamaModel,
    system: Option<&str>,
    messages: &mut Vec<LlamaChatMessage>,
    max_tokens: usize,
    context_size: u32,
) -> Result<(Vec<LlamaToken>, usize), String> {
    let limit = usize::try_from(context_size)
        .map_err(|_| "模型上下文长度超出当前平台支持范围。".to_string())?;
    let mut dropped_turns = 0;
    loop {
        let mut prompt_messages =
            Vec::with_capacity(messages.len() + usize::from(system.is_some()));
        if let Some(system) = system {
            prompt_messages.push(
                LlamaChatMessage::new("system".into(), system.into())
                    .map_err(|error| format!("无法创建系统消息：{error}"))?,
            );
        }
        prompt_messages.extend(messages.iter().cloned());
        let prompt = model
            .apply_chat_template(None, &prompt_messages, true)
            .map_err(|error| format!("无法应用 GGUF 对话模板：{error}"))?;
        let tokens = model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|error| format!("无法编码输入文本：{error}"))?;
        if tokens.len().saturating_add(max_tokens) <= limit {
            return Ok((tokens, dropped_turns));
        }
        if messages.len() <= 1 {
            return Err(format!(
                "单轮输入加生成预留超过模型上下文上限 {limit} tokens，请缩短输入或减少 --max-tokens。"
            ));
        }
        messages.drain(..2);
        dropped_turns += 1;
        eprintln!("上下文接近上限：已截断较早的对话轮次。");
    }
}

fn generate(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    prompt: &[LlamaToken],
    options: &ChatOptions,
) -> Result<(String, usize), String> {
    if prompt.is_empty() {
        return Err("输入未能转换为模型 token，请重试。".into());
    }
    let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
    context.clear_kv_cache();
    for (chunk_index, chunk) in prompt.chunks(BATCH_SIZE).enumerate() {
        batch.clear();
        let chunk_start = chunk_index.saturating_mul(BATCH_SIZE);
        let last_position = i32::try_from(chunk_start + chunk.len() - 1)
            .map_err(|_| "输入 token 数超出 llama.cpp 支持范围。".to_string())?;
        for (offset, token) in chunk.iter().copied().enumerate() {
            let position = i32::try_from(chunk_start + offset)
                .map_err(|_| "输入 token 数超出 llama.cpp 支持范围。".to_string())?;
            batch
                .add(token, position, &[0], position == last_position)
                .map_err(|error| format!("无法准备输入批次：{error}"))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| format!("模型推理失败：{error}"))?;
    }
    let mut sampler = make_sampler(model, options)?;
    for token in prompt.iter().copied() {
        sampler.accept(token);
    }
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut answer = String::new();
    let mut output = io::stdout().lock();
    let mut position = i32::try_from(prompt.len())
        .map_err(|_| "输入 token 数超出 llama.cpp 支持范围。".to_string())?;
    let mut generated = 0usize;

    for _ in 0..options.max_tokens {
        if INTERRUPTED.swap(false, Ordering::SeqCst) {
            eprintln!("\n[已中断本轮生成]");
            break;
        }
        let token = sampler.sample(context, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }

        let bytes = model
            .token_to_bytes(token, Special::Tokenize)
            .map_err(|error| format!("无法解码模型回答：{error}"))?;
        let mut text = String::with_capacity(bytes.len().saturating_mul(3).max(32));
        decoder.decode_to_string(&bytes, &mut text, false);
        answer.push_str(&text);
        output
            .write_all(text.as_bytes())
            .map_err(|error| format!("无法输出模型回答：{error}"))?;
        output
            .flush()
            .map_err(|error| format!("无法刷新模型回答：{error}"))?;
        generated += 1;

        batch.clear();
        batch
            .add(token, position, &[0], true)
            .map_err(|error| format!("无法准备生成批次：{error}"))?;
        context
            .decode(&mut batch)
            .map_err(|error| format!("模型推理失败：{error}"))?;
        position = position
            .checked_add(1)
            .ok_or_else(|| "模型上下文位置溢出。".to_string())?;
        if u32::try_from(position).unwrap_or(u32::MAX) >= context.n_ctx() {
            break;
        }
    }

    let mut final_text = String::new();
    decoder.decode_to_string(&[], &mut final_text, true);
    answer.push_str(&final_text);
    output
        .write_all(final_text.as_bytes())
        .and_then(|()| writeln!(output))
        .and_then(|()| output.flush())
        .map_err(|error| format!("无法结束模型回答：{error}"))?;
    Ok((answer, generated))
}

fn make_sampler(model: &LlamaModel, options: &ChatOptions) -> Result<LlamaSampler, String> {
    if options.temperature < 1e-7 {
        let mut samplers = Vec::new();
        if options.repeat_penalty > 1.0 {
            samplers.push(LlamaSampler::penalties_simple(
                model.n_vocab(),
                64,
                options.repeat_penalty,
            ));
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
    if options.top_k > 0 {
        let top_k = i32::try_from(options.top_k)
            .map_err(|_| "--top-k 超出 llama.cpp 支持范围。".to_string())?;
        samplers.push(LlamaSampler::top_k(top_k));
    }
    samplers.push(LlamaSampler::top_p(options.top_p as f32, 1));
    samplers.push(LlamaSampler::temp(options.temperature as f32));
    let seed = options.seed.unwrap_or_else(random_seed);
    let seed = u32::try_from(seed % u64::from(u32::MAX))
        .map_err(|_| "随机种子超出 llama.cpp 支持范围。".to_string())?;
    samplers.push(LlamaSampler::dist(seed));
    Ok(LlamaSampler::chain_simple(samplers))
}

fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0x5EED_1234_ABCD_EF01)
}
