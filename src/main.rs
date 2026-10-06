use candle_core::quantized::{gguf_file, tokenizer::TokenizerFromGguf};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    gemma4::{config::Gemma4TextConfig, text::TextModel},
    quantized_gemma3, quantized_llama, quantized_phi, quantized_phi3, quantized_qwen2,
    quantized_qwen3, quantized_qwen3_moe,
};
use memmap2::MmapOptions;
use serde_json::Value;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use tokenizers::AddedToken;
use tokenizers::Tokenizer;
use tokenizers::pre_tokenizers::metaspace::{Metaspace as MetaspacePreTokenizer, PrependScheme};
use tokenizers::{decoders::metaspace::Metaspace as MetaspaceDecoder, models::unigram::Unigram};

const MAX_HEADER_SIZE: usize = 100 * 1024 * 1024;
const DEFAULT_TENSOR_LIMIT: usize = 20;
const DEFAULT_MAX_TOKENS_PER_TURN: usize = 256;
const DEFAULT_CONTEXT_TOKENS: usize = 4096;
const DEFAULT_CHAT_OPTIONS: ChatOptions = ChatOptions {
    threads: None,
    max_tokens: DEFAULT_MAX_TOKENS_PER_TURN,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChatOptions {
    threads: Option<usize>,
    max_tokens: usize,
}

struct ChatSettings {
    bos: Option<u32>,
    eos: u32,
    format: ChatFormat,
    context_limit: usize,
    add_bos: bool,
    options: ChatOptions,
}

enum ChatModel {
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
    fn forward(&mut self, input: &Tensor, position: usize) -> candle_core::Result<Tensor> {
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

#[derive(Debug, Clone, Copy)]
enum ChatFormat {
    Llama2,
    Llama3,
    Qwen,
    Gemma,
    Gemma4,
    UserInst,
    Phi2,
    Phi3,
}

impl ChatFormat {
    fn from_architecture(architecture: &str) -> Result<Self, String> {
        match architecture {
            "llama" => Ok(Self::Llama2),
            "qwen2" | "qwen3" | "qwen3moe" => Ok(Self::Qwen),
            "gemma" | "gemma2" | "gemma3" => Ok(Self::Gemma),
            "gemma4" => Ok(Self::Gemma4),
            "phi2" => Ok(Self::Phi2),
            "phi3" => Ok(Self::Phi3),
            unsupported => Err(format!(
                "GGUF 架构“{unsupported}”暂不支持对话推理。当前支持：llama、qwen2、qwen3、qwen3moe、gemma、gemma2、gemma3、phi2、phi3。"
            )),
        }
    }

    fn format_prompt(self, prompt: &str) -> String {
        match self {
            Self::Llama2 => format!("[INST] {prompt} [/INST]"),
            Self::Llama3 => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::Qwen => format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
            Self::Gemma => {
                format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
            }
            Self::Gemma4 => {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
            Self::UserInst => format!("[USER] {prompt} [/USER]\n[INST]"),
            Self::Phi2 => format!("Instruct: {prompt}\nOutput:"),
            Self::Phi3 => format!("<|user|>\n{prompt}<|end|>\n<|assistant|>\n"),
        }
    }
}

#[derive(Debug)]
struct TensorInfo {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    bytes: Option<u64>,
}

#[derive(Debug)]
struct ModelInfo {
    format: &'static str,
    tensors: Vec<TensorInfo>,
    metadata: Vec<(String, String)>,
}

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("错误：{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let Some(command) = args.first().map(String::as_str) else {
        print_help();
        return Ok(());
    };

    match command {
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        "formats" => {
            print_formats();
            Ok(())
        }
        "load" | "inspect" => {
            let (path, json, show_all) = parse_load_args(&args[1..])?;
            let file = File::open(&path)
                .map_err(|error| format!("无法打开 {}：{error}", path.display()))?;
            let mmap = unsafe {
                MmapOptions::new()
                    .map(&file)
                    .map_err(|error| format!("无法映射 {}：{error}", path.display()))?
            };
            let info = parse_model(&mmap)?;
            if json {
                print_json(&path, mmap.len(), &info)?;
            } else {
                print_model(&path, mmap.len(), &info, show_all);
            }
            Ok(())
        }
        "chat" => {
            let (path, options) = parse_chat_args(&args[1..])?;
            configure_threads(options.threads)?;
            chat(&path, options)
        }
        value if !value.starts_with('-') => {
            let (path, options) = parse_chat_args(&args)?;
            configure_threads(options.threads)?;
            chat(&path, options)
        }
        other => Err(format!("未知命令“{other}”。运行 `wolf help` 查看用法。")),
    }
}

fn parse_chat_args(args: &[String]) -> Result<(PathBuf, ChatOptions), String> {
    let mut path = None;
    let mut options = DEFAULT_CHAT_OPTIONS;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--threads" => {
                index += 1;
                options.threads = Some(parse_positive_option(args, index, "--threads")?);
            }
            "--max-tokens" => {
                index += 1;
                options.max_tokens = parse_positive_option(args, index, "--max-tokens")?;
            }
            option if option.starts_with('-') => {
                return Err(format!("未知选项“{option}”。运行 `wolf help` 查看用法。"));
            }
            value => {
                if path.replace(PathBuf::from(value)).is_some() {
                    return Err("只能指定一个模型路径。".into());
                }
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| {
        "请指定一个 GGUF 模型文件或 Gemma 4 模型目录。用法：wolf <模型路径> [--threads N] [--max-tokens N]".to_string()
    })?;
    Ok((path, options))
}

fn parse_positive_option(args: &[String], index: usize, option: &str) -> Result<usize, String> {
    let value = args
        .get(index)
        .ok_or_else(|| format!("{option} 缺少数值。"))?;
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{option} 需要正整数，收到“{value}”。"))?;
    if parsed == 0 {
        return Err(format!("{option} 必须大于 0。"));
    }
    Ok(parsed)
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

fn chat(path: &Path, options: ChatOptions) -> Result<(), String> {
    if path.is_dir() {
        return chat_gemma4_directory(path, options);
    }
    if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        return Err(format!(
            "对话推理支持 GGUF 文件，或包含 Gemma 4 配置、tokenizer 和 Safetensors 权重的模型目录。指定的路径：{}",
            path.display()
        ));
    }

    println!("正在加载模型：{}", path.display());
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

    println!("已加载 {architecture} 模型（CPU）。输入 /exit 退出。");
    chat_loop(
        model,
        tokenizer,
        ChatSettings {
            bos,
            eos,
            format: chat_format,
            context_limit,
            add_bos,
            options,
        },
    )
}

fn chat_gemma4_directory(path: &Path, options: ChatOptions) -> Result<(), String> {
    println!("正在加载 Gemma 4 模型目录：{}", path.display());
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
    let eos = ["<|im_end|>", "<end_of_turn>", "<|end_of_turn|>", "</s>"]
        .iter()
        .find_map(|token| tokenizer.token_to_id(token))
        .ok_or_else(|| {
            "Gemma 4 tokenizer 中找不到对话结束 token（<|im_end|>、<end_of_turn>、<|end_of_turn|> 或 </s>）。".to_string()
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
    let device = Device::Cpu;
    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &device)
            .map_err(|error| format!("无法映射 Gemma 4 Safetensors 权重：{error}"))?
    };
    let model = TextModel::new(&config, var_builder)
        .map_err(|error| format!("无法构建 Gemma 4 文本模型：{error}"))?;
    println!("Gemma 4 文本模型已加载（CPU）。输入 /exit 退出。");
    chat_loop(
        ChatModel::Gemma4(model),
        tokenizer,
        ChatSettings {
            bos,
            eos,
            format: ChatFormat::Gemma4,
            context_limit,
            add_bos: bos.is_some(),
            options,
        },
    )
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

fn chat_loop(
    mut model: ChatModel,
    tokenizer: Tokenizer,
    settings: ChatSettings,
) -> Result<(), String> {
    let ChatSettings {
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
            if matches!(chat_format, ChatFormat::Llama2)
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
        let mut pending_tokens = std::collections::VecDeque::new();
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

fn select_next_token(logits: &Tensor) -> Result<u32, String> {
    let token_ids = logits
        .argmax(candle_core::D::Minus1)
        .and_then(|indices| indices.flatten_all())
        .and_then(|indices| indices.to_vec1::<u32>())
        .map_err(|error| format!("无法选择下一个 token：{error}"))?;
    match token_ids.as_slice() {
        [token] => Ok(*token),
        _ => Err(format!(
            "无法选择下一个 token：模型输出必须包含一个 batch/sequence 位置，实际 argmax 形状为 {:?}。",
            logits.dims()
        )),
    }
}

fn build_stop_sequences(tokenizer: &Tokenizer, eos_id: u32) -> Vec<Vec<u32>> {
    let mut sequences = vec![vec![eos_id]];
    let eos_markers = [
        "</s>",
        "<|eot_id|>",
        "<|im_end|>",
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<|end|>",
        "<|fim_suffix|>",
    ]
    .iter()
    .filter_map(|marker| tokenizer.token_to_id(marker))
    .map(|id| vec![id])
    .collect::<Vec<_>>();
    for sequence in eos_markers {
        if !sequences.contains(&sequence) {
            sequences.push(sequence);
        }
    }
    let markers = vec![
        "[USER]",
        " [USER]",
        "\n[USER]",
        "<|user|>",
        "<|im_start|>user",
        "<start_of_turn>user",
    ];
    for marker in markers {
        if let Ok(encoded) = tokenizer.encode(marker, false) {
            let ids = encoded.get_ids();
            if !ids.is_empty() && !sequences.iter().any(|sequence| sequence == ids) {
                sequences.push(ids.to_vec());
            }
        }
    }
    sequences
}

fn matching_stop_sequence<'a>(
    generated: &[u32],
    stop_sequences: &'a [Vec<u32>],
) -> Option<&'a Vec<u32>> {
    stop_sequences
        .iter()
        .filter(|sequence| ends_with(generated, sequence))
        .max_by_key(|sequence| sequence.len())
}

fn is_turn_start_marker(tokenizer: &Tokenizer, sequence: &[u32]) -> bool {
    [
        "[USER]",
        " [USER]",
        "\n[USER]",
        "<|user|>",
        "<|im_start|>user",
        "<start_of_turn>user",
    ]
    .iter()
    .filter_map(|marker| tokenizer.encode(*marker, false).ok())
    .any(|encoding| encoding.get_ids() == sequence)
}

fn ends_with(values: &[u32], suffix: &[u32]) -> bool {
    !suffix.is_empty() && values.ends_with(suffix)
}

fn parse_load_args(args: &[String]) -> Result<(PathBuf, bool, bool), String> {
    let mut path = None;
    let mut json = false;
    let mut show_all = false;

    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "--all" => show_all = true,
            value if value.starts_with('-') => {
                return Err(format!("未知选项“{value}”。运行 `wolf help` 查看用法。"));
            }
            value => {
                if path.replace(PathBuf::from(value)).is_some() {
                    return Err("只能指定一个模型文件。".to_string());
                }
            }
        }
    }

    let path = path.ok_or_else(|| {
        "缺少模型文件路径。用法：wolf load <模型文件> [--all] [--json]".to_string()
    })?;
    Ok((path, json, show_all))
}

fn parse_model(data: &[u8]) -> Result<ModelInfo, String> {
    if data.starts_with(b"GGUF") {
        parse_gguf(data)
    } else if data.len() >= 8 {
        let header_len = u64::from_le_bytes(data[..8].try_into().unwrap());
        if header_len as usize <= data.len().saturating_sub(8) {
            parse_safetensors(data)
        } else {
            Err("无法识别文件格式；支持 GGUF 和 Safetensors（.gguf、.safetensors）。".into())
        }
    } else {
        Err("文件过短或为空；支持 GGUF 和 Safetensors（.gguf、.safetensors）。".into())
    }
}

fn parse_safetensors(data: &[u8]) -> Result<ModelInfo, String> {
    if data.len() < 8 {
        return Err("Safetensors 文件缺少头部长度。".into());
    }
    let header_len = usize::try_from(u64::from_le_bytes(data[..8].try_into().unwrap()))
        .map_err(|_| "Safetensors 头部长度超出当前平台支持范围。")?;
    if header_len > MAX_HEADER_SIZE {
        return Err(format!(
            "Safetensors 头部过大（{header_len} 字节，上限 {MAX_HEADER_SIZE} 字节）。"
        ));
    }
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| "Safetensors 头部长度溢出。".to_string())?;
    let header = data
        .get(8..header_end)
        .ok_or_else(|| "Safetensors 文件头部不完整。".to_string())?;
    let value: Value = serde_json::from_slice(header)
        .map_err(|error| format!("Safetensors 头部 JSON 无效：{error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Safetensors 头部必须是 JSON 对象。".to_string())?;
    let payload_len = data.len() - header_end;
    let mut tensors = Vec::new();
    let mut metadata = Vec::new();
    let mut ranges = Vec::new();

    for (name, entry) in object {
        if name == "__metadata__" {
            if let Some(items) = entry.as_object() {
                metadata.extend(items.iter().map(|(key, value)| {
                    (
                        key.clone(),
                        value.as_str().unwrap_or("<非字符串>").to_string(),
                    )
                }));
            }
            continue;
        }
        let entry = entry
            .as_object()
            .ok_or_else(|| format!("张量“{name}”的描述无效。"))?;
        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("张量“{name}”缺少 dtype。"))?
            .to_string();
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("张量“{name}”缺少有效 shape。"))?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .ok_or_else(|| format!("张量“{name}”包含无效维度。"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| format!("张量“{name}”缺少有效 data_offsets。"))?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("张量“{name}”的起始偏移无效。"))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("张量“{name}”的结束偏移无效。"))?;
        if start > end || end > payload_len as u64 {
            return Err(format!(
                "张量“{name}”的数据范围 {start}..{end} 超出文件数据区（{payload_len} 字节）。"
            ));
        }
        ranges.push((start, end, name.clone()));
        tensors.push(TensorInfo {
            name: name.clone(),
            shape,
            dtype,
            bytes: Some(end - start),
        });
    }

    ranges.sort_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(format!(
                "张量“{}”和“{}”的数据范围重叠。",
                pair[0].2, pair[1].2
            ));
        }
    }
    Ok(ModelInfo {
        format: "Safetensors",
        tensors,
        metadata,
    })
}

fn parse_gguf(data: &[u8]) -> Result<ModelInfo, String> {
    let mut reader = Reader::new(data);
    if reader.take(4)? != b"GGUF" {
        return Err("GGUF 文件签名无效。".into());
    }
    let version = reader.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("不支持 GGUF 版本 {version}；当前支持版本 2 和 3。"));
    }
    let tensor_count = reader.count("张量")?;
    let metadata_count = reader.count("元数据")?;
    let mut metadata = Vec::new();
    let mut alignment = 32u64;

    for _ in 0..metadata_count {
        let key = reader.string()?;
        let value_type = reader.u32()?;
        let value = reader.skip_metadata_value(value_type)?;
        if key == "general.alignment"
            && let Some(value) = value.as_ref()
        {
            alignment = value
                .parse::<u64>()
                .map_err(|_| "GGUF general.alignment 值无效。")?;
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err("GGUF general.alignment 必须是 2 的幂。".into());
            }
        }
        if let Some(value) = value {
            metadata.push((key, value));
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(1_000_000));
    let mut offsets = Vec::with_capacity(tensor_count.min(1_000_000));
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let dimensions = reader.u32()? as usize;
        if dimensions > 4 {
            return Err(format!(
                "GGUF 张量“{name}”的维度数 {dimensions} 超出支持范围。"
            ));
        }
        let mut shape = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            shape.push(reader.u64()?);
        }
        let dtype = ggml_type_name(reader.u32()?);
        let offset = reader.u64()?;
        offsets.push((offset, name.clone()));
        tensors.push(TensorInfo {
            name,
            shape,
            dtype: dtype.to_string(),
            bytes: None,
        });
    }

    let data_start = reader.position();
    let data_start = align_up(data_start as u64, alignment)
        .ok_or_else(|| "GGUF 数据区偏移溢出。".to_string())?;
    if data_start > data.len() as u64 {
        return Err("GGUF 张量数据区超出文件范围。".into());
    }
    for (offset, name) in offsets {
        if offset > data.len() as u64 - data_start {
            return Err(format!("GGUF 张量“{name}”的数据偏移超出文件范围。"));
        }
    }
    Ok(ModelInfo {
        format: "GGUF",
        tensors,
        metadata,
    })
}

fn ggml_type_name(value: u32) -> &'static str {
    match value {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31 => "Q4_0_4_4",
        32 => "Q4_0_4_8",
        33 => "Q4_0_8_8",
        _ => "UNKNOWN",
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

struct Reader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| "GGUF 文件偏移溢出。".to_string())?;
        let value = self
            .data
            .get(self.position..end)
            .ok_or_else(|| "GGUF 文件意外结束，文件可能已损坏。".to_string())?;
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn count(&mut self, label: &str) -> Result<usize, String> {
        let count = usize::try_from(self.u64()?)
            .map_err(|_| format!("GGUF {label}数量超出当前平台支持范围。"))?;
        if count > 1_000_000 {
            return Err(format!("GGUF {label}数量异常：{count}。"));
        }
        Ok(count)
    }

    fn string(&mut self) -> Result<String, String> {
        let length =
            usize::try_from(self.u64()?).map_err(|_| "GGUF 字符串长度超出当前平台支持范围。")?;
        if length > MAX_HEADER_SIZE {
            return Err("GGUF 字符串长度超出安全上限。".into());
        }
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| "GGUF 文件包含无效 UTF-8 字符串。".into())
    }

    fn skip_metadata_value(&mut self, value_type: u32) -> Result<Option<String>, String> {
        let value = match value_type {
            0 => Some(self.u8()?.to_string()),
            1 => Some((self.u8()? as i8).to_string()),
            2 => Some(self.u16()?.to_string()),
            3 => Some((self.u16()? as i16).to_string()),
            4 => Some(self.u32()?.to_string()),
            5 => Some((self.u32()? as i32).to_string()),
            6 => Some(f32::from_le_bytes(self.take(4)?.try_into().unwrap()).to_string()),
            7 => Some((self.u8()? != 0).to_string()),
            8 => Some(self.string()?),
            9 => {
                let element_type = self.u32()?;
                let count = self.count("数组元素")?;
                for _ in 0..count {
                    self.skip_metadata_value(element_type)?;
                }
                Some(format!("[{count} 项]"))
            }
            10 => Some(self.u64()?.to_string()),
            11 => Some((self.u64()? as i64).to_string()),
            12 => Some(f64::from_le_bytes(self.take(8)?.try_into().unwrap()).to_string()),
            _ => return Err(format!("GGUF 元数据包含未知类型 {value_type}。")),
        };
        Ok(value)
    }
}

fn print_model(path: &Path, file_size: usize, info: &ModelInfo, show_all: bool) {
    println!("模型：{}", path.display());
    println!("格式：{}", info.format);
    println!("文件大小：{}", format_size(file_size as u64));
    println!("张量数量：{}", info.tensors.len());
    println!("状态：已映射文件并验证头部");

    if !info.metadata.is_empty() {
        println!("\n元数据：");
        for (key, value) in &info.metadata {
            println!("  {key}: {value}");
        }
    }
    if !info.tensors.is_empty() {
        println!("\n张量：");
        let count = if show_all {
            info.tensors.len()
        } else {
            info.tensors.len().min(DEFAULT_TENSOR_LIMIT)
        };
        for tensor in info.tensors.iter().take(count) {
            let bytes = tensor
                .bytes
                .map(|size| format!("  {}", format_size(size)))
                .unwrap_or_default();
            println!(
                "  {}  {:?}  {}{}",
                tensor.name, tensor.shape, tensor.dtype, bytes
            );
        }
        if count < info.tensors.len() {
            println!(
                "  …另有 {} 个张量未显示（使用 --all 查看）",
                info.tensors.len() - count
            );
        }
    }
}

fn print_json(path: &Path, file_size: usize, info: &ModelInfo) -> Result<(), String> {
    let output = serde_json::json!({
        "path": path,
        "format": info.format,
        "file_size": file_size,
        "tensor_count": info.tensors.len(),
        "metadata": info.metadata,
        "tensors": info.tensors.iter().map(|tensor| serde_json::json!({
            "name": tensor.name,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "bytes": tensor.bytes,
        })).collect::<Vec<_>>(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("无法生成 JSON 输出：{error}"))?
    );
    Ok(())
}

fn format_size(size: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = size as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{size} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn print_formats() {
    println!("当前支持的权重格式：");
    println!("  .gguf              对话推理（LLaMA、Qwen、Gemma、Phi 系 CPU）和模型信息检查");
    println!("  Gemma 4 模型目录  CPU 文本推理（Safetensors 权重）");
    println!("  .safetensors       模型信息检查（其他模型暂不支持推理）");

    println!("\nGemma 4 目录需包含 config.json、tokenizer.json 和 Safetensors 权重。");
}

fn print_help() {
    println!(
        "wolf - Rust 本地模型加载器与对话 CLI\n\
\n\
用法：\n\
  wolf <模型路径>                         加载 GGUF 或 Gemma 4 目录并进入多轮对话\n\
  wolf chat <模型路径>                    同上\n\
  wolf load <模型文件> [--all] [--json]   加载并显示模型信息\n\
  wolf inspect <模型文件> [--all] [--json]（load 的别名）\n\
  wolf formats                            列出支持的格式\n\
  wolf help                               显示帮助\n\
\n\
对话选项：\n\
  --threads N       CPU 线程数（默认使用 Rayon 默认值）\n\
  --max-tokens N    每轮最大生成 token 数（默认 256）\n\
\n\
示例：\n\
  wolf ./models/llama-2-7b-chat.Q4_K_M.gguf\n\
  wolf ./models/model.gguf --threads 8 --max-tokens 128\n\
  wolf ./models/gemma-4-E2B-it/\n\
  wolf /data/models/llama.gguf\n\
  wolf chat \"C:\\\\models\\\\llama.gguf\"\n\
  wolf load ./model.safetensors\n\
  wolf load ./model.gguf --all\n\
  wolf inspect ./model.safetensors --json\n\
\n\
对话中输入 /exit 退出。支持 LLaMA、Qwen、Gemma、Phi 系 GGUF 和 Gemma 4 Safetensors 目录的 CPU 推理。"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safetensors_file(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(payload);
        data
    }

    #[test]
    fn parses_valid_safetensors_header() {
        let file = safetensors_file(
            r#"{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#,
            &[0; 16],
        );
        let info = parse_model(&file).unwrap();
        assert_eq!(info.format, "Safetensors");
        assert_eq!(info.tensors[0].shape, vec![2, 2]);
        assert_eq!(info.tensors[0].bytes, Some(16));
    }

    #[test]
    fn rejects_safetensors_payload_out_of_bounds() {
        let file = safetensors_file(
            r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,16]}}"#,
            &[0; 4],
        );
        assert!(parse_model(&file).unwrap_err().contains("超出文件数据区"));
    }

    #[test]
    fn rejects_overlapping_safetensors_tensors() {
        let file = safetensors_file(
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#,
            &[0; 12],
        );
        assert!(parse_model(&file).unwrap_err().contains("重叠"));
    }

    #[test]
    fn parses_minimal_gguf() {
        let mut file = b"GGUF".to_vec();
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        file.resize(32, 0);
        let info = parse_model(&file).unwrap();
        assert_eq!(info.format, "GGUF");
        assert!(info.tensors.is_empty());
    }

    #[test]
    fn rejects_unsupported_format() {
        assert!(
            parse_model(b"not a model")
                .unwrap_err()
                .contains("无法识别")
        );
    }

    #[test]
    fn routes_supported_non_llama_architectures() {
        assert!(matches!(
            ChatFormat::from_architecture("qwen2"),
            Ok(ChatFormat::Qwen)
        ));
        assert!(matches!(
            ChatFormat::from_architecture("qwen3moe"),
            Ok(ChatFormat::Qwen)
        ));
        assert!(matches!(
            ChatFormat::from_architecture("gemma3"),
            Ok(ChatFormat::Gemma)
        ));
        assert!(matches!(
            ChatFormat::from_architecture("gemma2"),
            Ok(ChatFormat::Gemma)
        ));
        assert!(matches!(
            ChatFormat::from_architecture("gemma4"),
            Ok(ChatFormat::Gemma4)
        ));
        assert!(matches!(
            ChatFormat::from_architecture("phi2"),
            Ok(ChatFormat::Phi2)
        ));
        assert!(matches!(
            ChatFormat::from_architecture("phi3"),
            Ok(ChatFormat::Phi3)
        ));
    }

    #[test]
    fn reports_unsupported_architecture_without_panicking() {
        assert!(
            ChatFormat::from_architecture("mistral")
                .unwrap_err()
                .contains("暂不支持")
        );
    }

    #[test]
    fn formats_non_llama_chat_prompts() {
        assert_eq!(
            ChatFormat::Qwen.format_prompt("你好"),
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            ChatFormat::Gemma.format_prompt("你好"),
            "<start_of_turn>user\n你好<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(
            ChatFormat::Gemma4.format_prompt("你好"),
            "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            ChatFormat::Phi3.format_prompt("你好"),
            "<|user|>\n你好<|end|>\n<|assistant|>\n"
        );
        assert_eq!(
            ChatFormat::Phi2.format_prompt("你好"),
            "Instruct: 你好\nOutput:"
        );
    }

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

    #[test]
    fn selects_next_token_from_singleton_batch_and_sequence_ranks() {
        for shape in [&[3usize][..], &[1, 3], &[1, 1, 3]] {
            let logits = Tensor::from_vec(vec![0f32, 4.0, 1.0], shape, &Device::Cpu).unwrap();
            assert_eq!(select_next_token(&logits).unwrap(), 1);
        }
    }

    #[test]
    fn rejects_logits_with_multiple_batch_positions() {
        let logits =
            Tensor::from_vec(vec![0f32, 4.0, 1.0, 3.0, 0.0, 1.0], (2, 3), &Device::Cpu).unwrap();
        assert!(
            select_next_token(&logits)
                .unwrap_err()
                .contains("必须包含一个")
        );
    }

    #[test]
    fn stops_before_start_of_next_user_turn() {
        let stops = vec![vec![9], vec![3, 4, 5]];
        assert_eq!(
            matching_stop_sequence(&[1, 2, 3, 4, 5], &stops),
            Some(&stops[1])
        );
        assert_eq!(matching_stop_sequence(&[1, 2, 3, 4], &stops), None);
    }

    #[test]
    fn parses_chat_speed_options() {
        let args = vec![
            "model.gguf".to_string(),
            "--threads".to_string(),
            "8".to_string(),
            "--max-tokens".to_string(),
            "64".to_string(),
        ];
        let (path, options) = parse_chat_args(&args).unwrap();
        assert_eq!(path, PathBuf::from("model.gguf"));
        assert_eq!(
            options,
            ChatOptions {
                threads: Some(8),
                max_tokens: 64
            }
        );
    }
}
