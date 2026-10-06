//! 多轮对话编排:加载模型并驱动生成循环。

pub mod format;
pub mod model;
pub mod sampling;

use crate::cli::ChatOptions;
use candle_core::{Device, Tensor};
use format::{build_stop_sequences, is_turn_start_marker, matching_stop_sequence};
use model::{ChatSource, LoadedChat};
use sampling::Sampler;
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

pub fn chat(path: &Path, options: ChatOptions) -> Result<(), String> {
    let source = ChatSource::from_path(path)?;
    let loaded = source.load(options)?;
    chat_loop(source, loaded)
}

/// 截断历史并重载模型:candle 量化模型的 KV cache 没有公开的重置接口,
/// 无法原地丢弃旧上下文,因此通过"重载模型 + 重放保留的轮次"重建会话。
fn truncate_and_reload(
    source: &ChatSource,
    options: &ChatOptions,
    loaded: &mut LoadedChat,
    history: &mut Vec<Vec<u32>>,
    position: &mut usize,
    required: usize,
    context_limit: usize,
) -> Result<(), String> {
    if required > context_limit {
        return Err(format!(
            "单轮输入加生成长度需要 {required} tokens，超过模型上下文上限 {context_limit}，无法继续。"
        ));
    }
    let budget = context_limit * 3 / 5;
    let lens: Vec<usize> = history.iter().map(Vec::len).collect();
    let keep = kept_turns(&lens, required, budget, context_limit);
    let dropped = history.len() - keep;
    println!("上下文接近上限：截断最早 {dropped} 轮对话并重新加载模型…");
    let mut fresh = source.load(options.clone())?;
    let replay = history[dropped..].concat();
    if !replay.is_empty() {
        fresh
            .model
            .forward(
                &Tensor::new(replay.as_slice(), &Device::Cpu)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|error| format!("无法创建历史重放张量：{error}"))?,
                0,
            )
            .map_err(|error| format!("重放对话历史失败：{error}"))?;
    }
    *position = replay.len();
    history.drain(..dropped);
    *loaded = fresh;
    Ok(())
}

/// 计算截断后保留的最近轮次数:优先按 60% 预算保留完整轮次,
/// 若本轮输入(required = 输入 + 生成预留)仍放不下剩余空间,则继续舍弃。
/// 返回值 ≤ history_lens.len();即使为 0,调用方也需先确认 required ≤ context_limit。
fn kept_turns(
    history_lens: &[usize],
    required: usize,
    budget: usize,
    context_limit: usize,
) -> usize {
    let replay_len =
        |count: usize| -> usize { history_lens[history_lens.len() - count..].iter().sum() };
    let mut keep = history_lens.len();
    while keep > 0 && (replay_len(keep) > budget || required > context_limit - replay_len(keep)) {
        keep -= 1;
    }
    keep
}

fn print_chat_help() {
    println!(
        "对话内命令：\n  \
         /clear  清空对话历史并重新加载模型\n  \
         /help   显示本帮助\n  \
         /exit   退出（或 /quit、Ctrl-D）"
    );
}

fn chat_loop(source: ChatSource, mut loaded: LoadedChat) -> Result<(), String> {
    let options = loaded.settings.options.clone();
    let stop_sequences = build_stop_sequences(&loaded.tokenizer, loaded.settings.eos);
    let max_stop_len = stop_sequences.iter().map(Vec::len).max().unwrap_or(1);
    let mut sampler = Sampler::new(&options);
    let mut input = String::new();
    let mut position = 0usize;
    // 每轮实际喂入模型的 token(提示 + 生成的已喂入部分),用于截断后精确重放。
    let mut history: Vec<Vec<u32>> = Vec::new();
    let mut first_turn = true;
    let mut previous_stop: Vec<u32> = Vec::new();
    let mut usage_warned = false;
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
        if prompt == "/exit" || prompt == "/quit" {
            println!("已退出。");
            return Ok(());
        }
        if prompt == "/clear" {
            loaded = source.load(options.clone())?;
            position = 0;
            history.clear();
            first_turn = true;
            previous_stop.clear();
            usage_warned = false;
            println!("已清空对话历史。");
            continue;
        }
        if prompt == "/help" {
            print_chat_help();
            continue;
        }
        if prompt.is_empty() {
            continue;
        }

        let chat_format = loaded.settings.format;
        let context_limit = loaded.settings.context_limit;
        let add_bos = loaded.settings.add_bos;
        let bos = loaded.settings.bos;
        let reserve = options.max_tokens;

        let mut tokens = Vec::new();
        let is_first_turn = first_turn;
        if first_turn {
            if add_bos && let Some(bos) = bos {
                tokens.push(bos);
            }
            first_turn = false;
        } else {
            tokens.extend_from_slice(&previous_stop);
            if chat_format == format::ChatFormat::Llama2
                && let Some(bos) = bos
            {
                tokens.push(bos);
            }
        }
        let formatted = if is_first_turn && let Some(system) = options.system.as_deref() {
            chat_format.format_prompt_with_system(prompt, system)
        } else {
            chat_format.format_prompt(prompt)
        };
        let encoded = loaded
            .tokenizer
            .encode(formatted, false)
            .map_err(|error| format!("无法编码输入文本：{error}"))?;
        tokens.extend_from_slice(encoded.get_ids());
        if tokens.is_empty() {
            eprintln!("输入未能转换为模型 token，请重试。");
            continue;
        }

        if position + tokens.len() + reserve > context_limit {
            let required = tokens.len() + reserve;
            if let Err(message) = truncate_and_reload(
                &source,
                &options,
                &mut loaded,
                &mut history,
                &mut position,
                required,
                context_limit,
            ) {
                eprintln!("{message}");
                continue;
            }
            usage_warned = false;
        }
        if !usage_warned && (position + tokens.len() + reserve) * 10 > context_limit * 8 {
            usage_warned = true;
            eprintln!(
                "注意：上下文用量已超过 80%（上限 {context_limit} tokens），超限后较早的对话轮次会被自动截断。"
            );
        }
        previous_stop.clear();

        println!("正在处理输入并生成回答…");
        io::stdout()
            .flush()
            .map_err(|error| format!("无法刷新输出：{error}"))?;
        let mut logits = loaded
            .model
            .forward(
                &Tensor::new(tokens.as_slice(), &Device::Cpu)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|error| format!("无法创建输入张量：{error}"))?,
                position,
            )
            .map_err(|error| format!("模型推理失败：{error}"))?;
        position += tokens.len();
        let mut recent = tokens.clone();

        let mut decoder = loaded.tokenizer.decode_stream(true);
        let mut pending_tokens = VecDeque::new();
        let mut response_ids = Vec::new();
        let mut output = io::stdout().lock();
        let generation_started = Instant::now();
        let mut generated_count = 0usize;
        let mut stopped_at_turn_boundary = false;
        let mut matched_stop = Vec::new();
        for _ in 0..options.max_tokens {
            let next = sampler.select(&logits, &recent)?;
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
            recent.push(next);
            logits = loaded
                .model
                .forward(
                    &Tensor::new(&[next], &Device::Cpu)
                        .and_then(|tensor| tensor.unsqueeze(0))
                        .map_err(|error| format!("无法创建 token 张量：{error}"))?,
                    position,
                )
                .map_err(|error| format!("模型推理失败：{error}"))?;
            position += 1;
        }
        history.push(std::mem::take(&mut recent));
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
        if stopped_at_turn_boundary && !is_turn_start_marker(&loaded.tokenizer, &matched_stop) {
            previous_stop = matched_stop;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kept_turns;

    const CONTEXT_LIMIT: usize = 100;
    const BUDGET: usize = 60; // 3/5 上限

    #[test]
    fn keeps_recent_turns_within_budget() {
        // 5 轮 × 10 tokens = 50,整体在预算内。
        assert_eq!(kept_turns(&[10; 5], 20, BUDGET, CONTEXT_LIMIT), 5);
    }

    #[test]
    fn drops_oldest_turns_to_fit_budget() {
        // 3 轮 × 25 = 75 超出预算 60,保留最近 2 轮。
        assert_eq!(kept_turns(&[25, 25, 25], 20, BUDGET, CONTEXT_LIMIT), 2);
    }

    #[test]
    fn shrinks_further_when_incoming_needs_room() {
        // 保留 2 轮后剩余 50,能容纳 required=45。
        assert_eq!(kept_turns(&[25, 25, 25], 45, BUDGET, CONTEXT_LIMIT), 2);
        // required=55 时剩余 50 不够,再舍一轮。
        assert_eq!(kept_turns(&[25, 25, 25], 55, BUDGET, CONTEXT_LIMIT), 1);
    }

    #[test]
    fn may_drop_everything_for_oversized_history() {
        assert_eq!(kept_turns(&[70], 20, BUDGET, CONTEXT_LIMIT), 0);
    }
}
