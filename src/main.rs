use candle_core::quantized::{gguf_file, tokenizer::TokenizerFromGguf};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama;
use memmap2::MmapOptions;
use serde_json::Value;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tokenizers::AddedToken;
use tokenizers::Tokenizer;
use tokenizers::pre_tokenizers::metaspace::{Metaspace as MetaspacePreTokenizer, PrependScheme};
use tokenizers::{decoders::metaspace::Metaspace as MetaspaceDecoder, models::unigram::Unigram};

const MAX_HEADER_SIZE: usize = 100 * 1024 * 1024;
const DEFAULT_TENSOR_LIMIT: usize = 20;
const MAX_TOKENS_PER_TURN: usize = 512;
const MAX_CONTEXT_TOKENS: usize = quantized_llama::MAX_SEQ_LEN;

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
            let path = one_path(&args[1..])?;
            chat(&path)
        }
        value if !value.starts_with('-') => {
            let path = one_path(&args)?;
            chat(&path)
        }
        other => Err(format!("未知命令“{other}”。运行 `wolf help` 查看用法。")),
    }
}

fn one_path(args: &[String]) -> Result<PathBuf, String> {
    if args.len() != 1 {
        return Err("请指定一个 GGUF 模型文件路径。用法：wolf <模型路径>".into());
    }
    Ok(PathBuf::from(&args[0]))
}

fn chat(path: &Path) -> Result<(), String> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        return Err(format!(
            "对话推理目前只支持 GGUF 文件；Safetensors 仅支持检查。指定的文件：{}",
            path.display()
        ));
    }

    println!("正在加载模型：{}", path.display());
    let file = File::open(path).map_err(|error| format!("无法打开模型文件：{error}"))?;
    let mut reader = BufReader::new(file);
    let content = gguf_file::Content::read(&mut reader)
        .map_err(|error| format!("无法读取 GGUF 模型：{error}"))?;
    let tokenizer = build_tokenizer(&content)?;
    let llama3_chat = content
        .metadata
        .get("tokenizer.ggml.pre")
        .and_then(|value| value.to_string().ok())
        .is_some_and(|value| value == "llama3");
    let eos = metadata_token_id(&content, "tokenizer.ggml.eot_token_id")
        .or_else(|| metadata_token_id(&content, "tokenizer.ggml.eos_token_id"))
        .ok_or_else(|| "GGUF 模型缺少可用的对话结束 token ID。".to_string())?;
    let bos = metadata_token_id(&content, "tokenizer.ggml.bos_token_id");
    let model = quantized_llama::ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)
        .map_err(|error| format!("无法加载 LLaMA 系模型：{error}"))?;

    println!("模型已加载（CPU）。输入 /exit 退出。");
    chat_loop(model, tokenizer, bos, eos, llama3_chat)
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
    mut model: quantized_llama::ModelWeights,
    tokenizer: Tokenizer,
    bos: Option<u32>,
    eos: u32,
    llama3_chat: bool,
) -> Result<(), String> {
    let mut input = String::new();
    let mut position = 0usize;
    let mut first_turn = true;
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
            if let Some(bos) = bos {
                tokens.push(bos);
            }
            first_turn = false;
        } else if llama3_chat {
            tokens.push(eos);
        } else {
            tokens.push(eos);
            if let Some(bos) = bos {
                tokens.push(bos);
            }
        }
        let formatted = if llama3_chat {
            format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        } else {
            format!("[INST] {prompt} [/INST]")
        };
        let encoded = tokenizer
            .encode(formatted, false)
            .map_err(|error| format!("无法编码输入文本：{error}"))?;
        tokens.extend_from_slice(encoded.get_ids());
        if tokens.is_empty() {
            eprintln!("输入未能转换为模型 token，请重试。");
            continue;
        }
        if position.saturating_add(tokens.len()) >= MAX_CONTEXT_TOKENS {
            eprintln!("已达到模型 4096 token 上下文上限。请退出并重新启动对话。");
            continue;
        }

        let mut logits = model
            .forward(
                &Tensor::new(tokens.as_slice(), &Device::Cpu)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|error| format!("无法创建输入张量：{error}"))?,
                position,
            )
            .map_err(|error| format!("模型推理失败：{error}"))?;
        position += tokens.len();

        let mut generated = Vec::new();
        for _ in 0..MAX_TOKENS_PER_TURN {
            let next = logits
                .argmax(candle_core::D::Minus1)
                .and_then(|token| token.to_scalar::<u32>())
                .map_err(|error| format!("无法选择下一个 token：{error}"))?;
            if next == eos {
                break;
            }
            generated.push(next);
            if position >= MAX_CONTEXT_TOKENS {
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

        let answer = tokenizer
            .decode(&generated, true)
            .map_err(|error| format!("无法解码模型回答：{error}"))?;
        println!("{answer}\n");
    }
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
    println!("  .gguf         对话推理（LLaMA 系 CPU）和模型信息检查");
    println!("  .safetensors  模型信息检查（暂不支持推理）");

    println!("\nGGUF 对话推理使用 CPU；其他架构/设备请先确认模型兼容性。");
}

fn print_help() {
    println!(
        "wolf - Rust 本地模型加载器与对话 CLI\n\
\n\
用法：\n\
  wolf <模型路径>                         加载 GGUF 模型并进入多轮对话\n\
  wolf chat <模型路径>                    同上\n\
  wolf load <模型文件> [--all] [--json]   加载并显示模型信息\n\
  wolf inspect <模型文件> [--all] [--json]（load 的别名）\n\
  wolf formats                            列出支持的格式\n\
  wolf help                               显示帮助\n\
\n\
示例：\n\
  wolf ./models/llama-2-7b-chat.Q4_K_M.gguf\n\
  wolf /data/models/llama.gguf\n\
  wolf chat \"C:\\\\models\\\\llama.gguf\"\n\
  wolf load ./model.safetensors\n\
  wolf load ./model.gguf --all\n\
  wolf inspect ./model.safetensors --json\n\
\n\
对话中输入 /exit 退出。当前对话推理支持 LLaMA 系 GGUF，运行于 CPU；Safetensors 仅可检查。"
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
}
