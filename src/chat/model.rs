//! 模型加载：GGUF 对话模型与 Gemma 4 目录的构建。

use crate::cli::{ChatOptions, DEFAULT_CONTEXT_TOKENS};
use crate::inspect::format_size;
use candle_core::quantized::{gguf_file, tokenizer::TokenizerFromGguf};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    gemma4::{config::Gemma4TextConfig, text::TextModel},
    quantized_gemma3, quantized_llama, quantized_phi, quantized_phi3, quantized_qwen2,
    quantized_qwen3, quantized_qwen3_moe,
};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokenizers::AddedToken;
use tokenizers::Tokenizer;
use tokenizers::pre_tokenizers::metaspace::{Metaspace as MetaspacePreTokenizer, PrependScheme};
use tokenizers::{decoders::metaspace::Metaspace as MetaspaceDecoder, models::unigram::Unigram};

use super::format::ChatFormat;

/// Gemma 4 目录的 EOS 候选,按优先级排列。
const GEMMA4_EOS_CANDIDATES: &[&str] = &["<|im_end|>", "<end_of_turn>", "<|end_of_turn|>", "</s>"];

/// 模型加载期间的等待提示:每隔一秒在 stderr 上刷新一次已等待秒数。
struct LoadTicker {
    done: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl LoadTicker {
    fn start() -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let done_thread = done.clone();
        let handle = thread::spawn(move || {
            let started = Instant::now();
            while !done_thread.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(1));
                if done_thread.load(Ordering::Relaxed) {
                    break;
                }
                eprint!("\r  已等待 {} 秒…", started.elapsed().as_secs());
                let _ = io::stderr().flush();
            }
        });
        Self {
            done,
            handle: Some(handle),
        }
    }

    fn finish(mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        eprint!("\r{}", " ".repeat(40));
        let _ = io::stderr().flush();
        eprint!("\r");
        let _ = io::stderr().flush();
    }
}

pub struct ChatSettings {
    pub bos: Option<u32>,
    pub eos: u32,
    pub format: ChatFormat,
    pub context_limit: usize,
    pub add_bos: bool,
    pub options: ChatOptions,
}

pub enum ChatModel {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Qwen3Moe(quantized_qwen3_moe::GGUFQWenMoE),
    Gemma(quantized_gemma3::ModelWeights),
    Gemma4(TextModel),
    Phi2(quantized_phi::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
}

impl ChatModel {
    pub fn forward(&mut self, input: &Tensor, position: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(model) => model.forward(input, position),
            Self::Qwen2(model) => model.forward(input, position),
            Self::Qwen3(model) => model.forward(input, position),
            Self::Qwen3Moe(model) => model.forward(input, position),
            Self::Gemma(model) => model.forward(input, position),
            Self::Gemma4(model) => model.forward(input, position)?.squeeze(0)?.squeeze(0),
            Self::Phi2(model) => model.forward(input, position),
            Self::Phi3(model) => model.forward(input, position),
        }
    }
}

/// 一次加载完成的对话会话组件。
pub struct LoadedChat {
    pub model: ChatModel,
    pub tokenizer: Tokenizer,
    pub settings: ChatSettings,
    /// 加载来源的架构名,用于提示信息(如 "qwen3" 或 "Gemma 4 文本")。
    pub architecture: String,
}

/// 对话模型的来源:GGUF 文件或 Gemma 4 模型目录。
pub enum ChatSource {
    Gguf(PathBuf),
    Gemma4Directory(PathBuf),
}

impl ChatSource {
    pub fn from_path(path: &Path) -> Result<Self, String> {
        if path.is_dir() {
            return Ok(Self::Gemma4Directory(path.to_path_buf()));
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            return Err(format!(
                "对话推理支持 GGUF 文件，或包含 Gemma 4 配置、tokenizer 和 Safetensors 权重的模型目录。指定的路径：{}",
                path.display()
            ));
        }
        Ok(Self::Gguf(path.to_path_buf()))
    }

    pub fn load(&self, options: ChatOptions) -> Result<LoadedChat, String> {
        match self {
            Self::Gguf(path) => load_gguf(path, options),
            Self::Gemma4Directory(path) => load_gemma4_directory(path, options),
        }
    }
}

fn load_gguf(path: &Path, options: ChatOptions) -> Result<LoadedChat, String> {
    let file_size = fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    println!(
        "正在加载模型：{}（{}）",
        path.display(),
        format_size(file_size)
    );
    let started = Instant::now();
    let ticker = LoadTicker::start();
    let loaded = load_gguf_inner(path, options);
    ticker.finish();
    let loaded = loaded?;
    println!(
        "已加载 {} 模型（CPU，用时 {:.1} 秒）。输入 /exit 退出。",
        loaded.architecture,
        started.elapsed().as_secs_f64()
    );
    Ok(loaded)
}

fn load_gguf_inner(path: &Path, options: ChatOptions) -> Result<LoadedChat, String> {
    let file = File::open(path).map_err(|error| format!("无法打开模型文件：{error}"))?;
    let mut reader = BufReader::new(file);
    let content = gguf_file::Content::read(&mut reader)
        .map_err(|error| format!("无法读取 GGUF 模型：{error}"))?;
    let architecture = content
        .metadata
        .get("general.architecture")
        .and_then(|value| value.to_string().ok())
        .cloned()
        .ok_or_else(|| "GGUF 缺少 general.architecture 元数据。".to_string())?;
    if architecture == "gemma4" {
        return Err(
            "当前 Candle 的 Gemma 4 实现使用 Hugging Face Safetensors；请传入包含 config.json、tokenizer.json 和权重的模型目录。".into(),
        );
    }
    let template = content
        .metadata
        .get("tokenizer.chat_template")
        .and_then(|value| value.to_string().ok())
        .map(String::as_str)
        .unwrap_or_default();
    let chat_format = if template.contains("[USER]") && template.contains("[INST]") {
        ChatFormat::UserInst
    } else if architecture == "llama"
        && content
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|value| value.to_string().ok())
            .is_some_and(|value| value == "llama3")
    {
        ChatFormat::Llama3
    } else {
        ChatFormat::from_architecture(&architecture)?
    };
    let tokenizer = build_tokenizer(&content)?;
    let eos = metadata_token_id(&content, "tokenizer.ggml.eot_token_id")
        .or_else(|| metadata_token_id(&content, "tokenizer.ggml.eos_token_id"))
        .ok_or_else(|| "GGUF 模型缺少可用的对话结束 token ID。".to_string())?;
    let bos = metadata_token_id(&content, "tokenizer.ggml.bos_token_id");
    let add_bos = content
        .metadata
        .get("tokenizer.ggml.add_bos_token")
        .and_then(|value| value.to_bool().ok())
        .unwrap_or(matches!(
            chat_format,
            ChatFormat::Llama2 | ChatFormat::Llama3 | ChatFormat::Phi2
        ));
    let context_key = format!("{architecture}.context_length");
    let context_limit = content
        .metadata
        .get(&context_key)
        .and_then(|value| value.to_u32().ok())
        .map(|value| value as usize)
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_CONTEXT_TOKENS)
        .min(if architecture == "llama" {
            quantized_llama::MAX_SEQ_LEN
        } else {
            usize::MAX
        });
    let model = match architecture.as_str() {
        "llama" => ChatModel::Llama(
            quantized_llama::ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)
                .map_err(|error| format!("无法加载 LLaMA GGUF 模型：{error}"))?,
        ),
        "qwen2" => ChatModel::Qwen2(
            quantized_qwen2::ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)
                .map_err(|error| format!("无法加载 Qwen2 GGUF 模型：{error}"))?,
        ),
        "qwen3" => ChatModel::Qwen3(
            quantized_qwen3::ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)
                .map_err(|error| format!("无法加载 Qwen3 GGUF 模型：{error}"))?,
        ),
        "qwen3moe" => ChatModel::Qwen3Moe(
            quantized_qwen3_moe::GGUFQWenMoE::from_gguf(
                content,
                &mut reader,
                &Device::Cpu,
                DType::F32,
            )
            .map_err(|error| format!("无法加载 Qwen3-MoE GGUF 模型：{error}"))?,
        ),
        "gemma" | "gemma2" | "gemma3" => ChatModel::Gemma(
            quantized_gemma3::ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)
                .map_err(|error| format!("无法加载 Gemma GGUF 模型：{error}"))?,
        ),
        "phi2" => ChatModel::Phi2(
            quantized_phi::ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)
                .map_err(|error| format!("无法加载 Phi-2 GGUF 模型：{error}"))?,
        ),
        "phi3" => ChatModel::Phi3(
            quantized_phi3::ModelWeights::from_gguf(false, content, &mut reader, &Device::Cpu)
                .map_err(|error| format!("无法加载 Phi-3 GGUF 模型：{error}"))?,
        ),
        unsupported => {
            return Err(format!("GGUF 架构“{unsupported}”暂不支持对话推理。"));
        }
    };

    Ok(LoadedChat {
        model,
        tokenizer,
        settings: ChatSettings {
            bos,
            eos,
            format: chat_format,
            context_limit,
            add_bos,
            options,
        },
        architecture,
    })
}

fn load_gemma4_directory(path: &Path, options: ChatOptions) -> Result<LoadedChat, String> {
    println!("正在加载 Gemma 4 模型目录：{}", path.display());
    let started = Instant::now();
    let ticker = LoadTicker::start();
    let loaded = load_gemma4_directory_inner(path, options);
    ticker.finish();
    let loaded = loaded?;
    println!(
        "Gemma 4 文本模型已加载（CPU，用时 {:.1} 秒）。输入 /exit 退出。",
        started.elapsed().as_secs_f64()
    );
    Ok(loaded)
}

fn load_gemma4_directory_inner(path: &Path, options: ChatOptions) -> Result<LoadedChat, String> {
    let config_path = path.join("config.json");
    let config_data = fs::read(&config_path)
        .map_err(|error| format!("无法读取 {}：{error}", config_path.display()))?;
    let raw_config: Value = serde_json::from_slice(&config_data)
        .map_err(|error| format!("Gemma 4 config.json 无效：{error}"))?;
    let model_type = raw_config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model_type != "gemma4" {
        return Err(format!(
            "目录中的 model_type 是“{model_type}”，不是 gemma4。当前目录加载器仅用于 Gemma 4。"
        ));
    }
    let text_config_value = raw_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| raw_config.clone());
    let mut config: Gemma4TextConfig = serde_json::from_value(text_config_value)
        .map_err(|error| format!("Gemma 4 文本模型配置无效：{error}"))?;
    config.use_flash_attn = false;
    let context_limit = config.max_position_embeddings;

    let tokenizer_path = path.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("无法加载 {}：{error}", tokenizer_path.display()))?;
    let eos = GEMMA4_EOS_CANDIDATES
        .iter()
        .find_map(|token| tokenizer.token_to_id(token))
        .ok_or_else(|| {
            format!(
                "Gemma 4 tokenizer 中找不到对话结束 token（{}）。",
                GEMMA4_EOS_CANDIDATES.join("、")
            )
        })?;
    let bos = ["<bos>", "<s>"]
        .iter()
        .find_map(|token| tokenizer.token_to_id(token));
    if let Some(bos) = bos {
        let token = tokenizer
            .id_to_token(bos)
            .ok_or_else(|| "无法读取 Gemma 4 tokenizer 的 BOS token。".to_string())?;
        tokenizer.add_special_tokens(&[AddedToken::from(token, true)]);
    }

    let weights = gemma4_safetensor_paths(path)?;
    let total_weight_bytes: u64 = weights
        .iter()
        .filter_map(|weight| fs::metadata(weight).ok())
        .map(|meta| meta.len())
        .sum();
    let device = Device::Cpu;
    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &device)
            .map_err(|error| format!("无法映射 Gemma 4 Safetensors 权重：{error}"))?
    };
    let model = TextModel::new(&config, var_builder)
        .map_err(|error| format!("无法构建 Gemma 4 文本模型：{error}"))?;
    Ok(LoadedChat {
        model: ChatModel::Gemma4(model),
        tokenizer,
        settings: ChatSettings {
            bos,
            eos,
            format: ChatFormat::Gemma4,
            context_limit,
            add_bos: bos.is_some(),
            options,
        },
        architecture: format!("Gemma 4 文本（权重 {}）", format_size(total_weight_bytes)),
    })
}

fn gemma4_safetensor_paths(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let index_path = directory.join("model.safetensors.index.json");
    let mut paths = if index_path.is_file() {
        let index_data = fs::read(&index_path)
            .map_err(|error| format!("无法读取 {}：{error}", index_path.display()))?;
        let index: Value = serde_json::from_slice(&index_data)
            .map_err(|error| format!("Safetensors 权重索引无效：{error}"))?;
        let shard_names = gemma4_shard_names(&index)?;
        shard_names
            .into_iter()
            .map(|name| Ok(directory.join(name)))
            .collect::<Result<Vec<_>, String>>()?
    } else {
        let entries = fs::read_dir(directory)
            .map_err(|error| format!("无法读取模型目录 {}：{error}", directory.display()))?;
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| format!("读取模型目录条目失败：{error}"))?;
            let entry_path = entry.path();
            if entry_path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
                paths.push(entry_path);
            }
        }
        paths.sort();
        paths
    };
    if paths.is_empty() {
        return Err(format!(
            "模型目录中没有找到 Safetensors 权重：{}",
            directory.display()
        ));
    }
    for weight_path in &paths {
        if !weight_path.is_file() {
            return Err(format!(
                "找不到索引指定的权重分片：{}",
                weight_path.display()
            ));
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn gemma4_shard_names(index: &Value) -> Result<Vec<&str>, String> {
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| "Safetensors 权重索引中缺少 weight_map。".to_string())?;
    let mut shard_names = weight_map
        .values()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| "Safetensors 分片文件名必须是字符串。".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shard_names.is_empty() {
        return Err("Safetensors 权重索引中的 weight_map 为空。".into());
    }
    for name in &shard_names {
        let name_path = Path::new(name);
        if name_path.components().count() != 1
            || name_path.file_name().is_none()
            || name_path.extension().and_then(|ext| ext.to_str()) != Some("safetensors")
        {
            return Err(format!("权重索引中的分片文件名无效：{name}"));
        }
    }
    shard_names.sort_unstable();
    shard_names.dedup();
    Ok(shard_names)
}

fn build_tokenizer(content: &gguf_file::Content) -> Result<Tokenizer, String> {
    let model = content
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(|value| value.to_string().ok())
        .map(String::as_str)
        .ok_or_else(|| "GGUF 缺少 tokenizer.ggml.model 元数据。".to_string())?;

    if model != "llama" {
        return Tokenizer::from_gguf(content)
            .map_err(|error| format!("无法从 GGUF 加载 tokenizer（{model}）：{error}"));
    }

    let values = content
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|value| value.to_vec().ok())
        .ok_or_else(|| "GGUF 缺少有效的 tokenizer.ggml.tokens。".to_string())?;
    let tokens = values
        .iter()
        .map(|value| {
            value
                .to_string()
                .cloned()
                .map_err(|error| format!("GGUF tokenizer 词表无效：{error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scores = content
        .metadata
        .get("tokenizer.ggml.scores")
        .and_then(|value| value.to_vec().ok())
        .ok_or_else(|| "LLaMA GGUF 缺少 tokenizer.ggml.scores。".to_string())?;
    if tokens.len() != scores.len() {
        return Err("GGUF tokenizer 词表与分数数量不一致。".into());
    }
    let vocab = tokens
        .iter()
        .zip(scores)
        .map(|(token, score)| {
            score
                .to_f32()
                .map(|score| (token.clone(), score as f64))
                .map_err(|error| format!("GGUF tokenizer 分数无效：{error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let unk_id = metadata_token_id(content, "tokenizer.ggml.unk_token_id").map(|id| id as usize);
    let byte_fallback = content
        .metadata
        .get("tokenizer.ggml.byte_fallback")
        .and_then(|value| value.to_bool().ok())
        .unwrap_or(false);
    let unigram = Unigram::from(vocab, unk_id, byte_fallback)
        .map_err(|error| format!("无法构建 LLaMA tokenizer：{error}"))?;
    let mut tokenizer = Tokenizer::new(unigram);
    tokenizer.with_pre_tokenizer(Some(MetaspacePreTokenizer::new(
        '▁',
        PrependScheme::Always,
        true,
    )));
    tokenizer.with_decoder(Some(MetaspaceDecoder::new(
        '▁',
        PrependScheme::Always,
        true,
    )));

    let mut special_tokens = Vec::new();
    if let Some(types) = content
        .metadata
        .get("tokenizer.ggml.token_type")
        .and_then(|value| value.to_vec().ok())
    {
        for (id, value) in types.iter().enumerate() {
            if value.to_u32().is_ok_and(|kind| matches!(kind, 2..=5))
                && let Some(token) = tokens.get(id)
            {
                special_tokens.push(AddedToken::from(token.clone(), true));
            }
        }
    }
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.unk_token_id",
    ] {
        if let Some(token) = metadata_token_id(content, key).and_then(|id| tokens.get(id as usize))
        {
            special_tokens.push(AddedToken::from(token.clone(), true));
        }
    }
    tokenizer.add_special_tokens(&special_tokens);
    Ok(tokenizer)
}

fn metadata_token_id(content: &gguf_file::Content, key: &str) -> Option<u32> {
    content
        .metadata
        .get(key)
        .and_then(|value| value.to_u32().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_unique_gemma4_safetensors_shards() {
        let index = serde_json::json!({
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
                "layer.2": "model-00001-of-00002.safetensors"
            }
        });
        assert_eq!(
            gemma4_shard_names(&index).unwrap(),
            vec![
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
    }

    #[test]
    fn rejects_gemma4_safetensors_shard_path_traversal() {
        let index = serde_json::json!({
            "weight_map": {
                "layer.0": "../outside.safetensors"
            }
        });
        assert!(
            gemma4_shard_names(&index)
                .unwrap_err()
                .contains("文件名无效")
        );
    }
}
