//! 多轮对话编排:加载模型并驱动生成循环。

pub mod format;
pub mod model;
pub mod sampling;

use crate::cli::ChatOptions;
use candle_core::{Device, Tensor};
use format::{build_stop_sequences, is_turn_start_marker, matching_stop_sequence};
use model::{ChatModel, ChatSource};
use sampling::select_next_token;
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

pub fn chat(path: &Path, options: ChatOptions) -> Result<(), String> {
    let source = ChatSource::from_path(path)?;
    let loaded = source.load(options)?;
    chat_loop(loaded.model, loaded.tokenizer, loaded.settings)
}

fn chat_loop(
    mut model: ChatModel,
    tokenizer: Tokenizer,
    settings: model::ChatSettings,
) -> Result<(), String> {
    let model::ChatSettings {
        bos,
        eos,
        format: chat_format,
        context_limit,
        add_bos,
        options,
    } = settings;
    let stop_sequences = build_stop_sequences(&tokenizer, eos);
    let max_stop_len = stop_sequences.iter().map(Vec::len).max().unwrap_or(1);
    let mut input = String::new();
    let mut position = 0usize;
    let mut first_turn = true;
    let mut previous_stop = Vec::new();
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
        let prompt = input.trim();
        if prompt == "/exit" {
            println!("已退出。");
            return Ok(());
        }
        if prompt.is_empty() {
            continue;
        }

        let mut tokens = Vec::new();
        if first_turn {
            if add_bos && let Some(bos) = bos {
                tokens.push(bos);
            }
            first_turn = false;
        } else {
            tokens.extend_from_slice(&previous_stop);
            if matches!(chat_format, format::ChatFormat::Llama2)
                && let Some(bos) = bos
            {
                tokens.push(bos);
            }
        }
        let formatted = chat_format.format_prompt(prompt);
        let encoded = tokenizer
            .encode(formatted, false)
            .map_err(|error| format!("无法编码输入文本：{error}"))?;
        tokens.extend_from_slice(encoded.get_ids());
        if tokens.is_empty() {
            eprintln!("输入未能转换为模型 token，请重试。");
            continue;
        }
        if position.saturating_add(tokens.len()) >= context_limit {
            eprintln!("已达到模型上下文上限（{context_limit} tokens）。请退出并重新启动对话。");
            continue;
        }
        previous_stop.clear();

        println!("正在处理输入并生成回答…");
        io::stdout()
            .flush()
            .map_err(|error| format!("无法刷新输出：{error}"))?;
        let mut logits = model
            .forward(
                &Tensor::new(tokens.as_slice(), &Device::Cpu)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|error| format!("无法创建输入张量：{error}"))?,
                position,
            )
            .map_err(|error| format!("模型推理失败：{error}"))?;
        position += tokens.len();

        let mut decoder = tokenizer.decode_stream(true);
        let mut pending_tokens = VecDeque::new();
        let mut response_ids = Vec::new();
        let mut output = io::stdout().lock();
        let generation_started = Instant::now();
        let mut generated_count = 0usize;
        let mut stopped_at_turn_boundary = false;
        let mut matched_stop = Vec::new();
        for _ in 0..options.max_tokens {
            let next = select_next_token(&logits)?;
            response_ids.push(next);
            pending_tokens.push_back(next);
            if let Some(stop_sequence) = matching_stop_sequence(&response_ids, &stop_sequences) {
                stopped_at_turn_boundary = true;
                matched_stop.clone_from(stop_sequence);
                break;
            }
            generated_count += 1;
            while pending_tokens.len() > max_stop_len {
                let token = pending_tokens.pop_front().expect("queue is not empty");
                if let Some(text) = decoder
                    .step(token)
                    .map_err(|error| format!("无法解码模型回答：{error}"))?
                {
                    output
                        .write_all(text.as_bytes())
                        .map_err(|error| format!("无法输出模型回答：{error}"))?;
                }
                if generated_count.is_multiple_of(8) {
                    output
                        .flush()
                        .map_err(|error| format!("无法刷新模型回答：{error}"))?;
                }
            }
            if position >= context_limit {
                break;
            }
            logits = model
                .forward(
                    &Tensor::new(&[next], &Device::Cpu)
                        .and_then(|tensor| tensor.unsqueeze(0))
                        .map_err(|error| format!("无法创建 token 张量：{error}"))?,
                    position,
                )
                .map_err(|error| format!("模型推理失败：{error}"))?;
            position += 1;
        }
        let stop_len = if stopped_at_turn_boundary {
            matched_stop.len()
        } else {
            0
        };
        while pending_tokens.len() > stop_len {
            let token = pending_tokens.pop_front().expect("queue is not empty");
            if let Some(text) = decoder
                .step(token)
                .map_err(|error| format!("无法解码模型回答：{error}"))?
            {
                output
                    .write_all(text.as_bytes())
                    .map_err(|error| format!("无法输出模型回答：{error}"))?;
            }
        }
        writeln!(output).map_err(|error| format!("无法结束模型回答：{error}"))?;
        output
            .flush()
            .map_err(|error| format!("无法刷新模型回答：{error}"))?;
        let elapsed = generation_started.elapsed().as_secs_f64();
        if generated_count > 0 && elapsed > 0.0 {
            eprintln!(
                "[生成 {generated_count} tokens，{:.1} tokens/s]",
                generated_count as f64 / elapsed
            );
        }
        if stopped_at_turn_boundary && !is_turn_start_marker(&tokenizer, &matched_stop) {
            previous_stop = matched_stop;
        }
    }
}
