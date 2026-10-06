//! 模型文件检查：GGUF 与 Safetensors 的解析、校验和展示。

use memmap2::MmapOptions;
use serde_json::Value;
use std::fs::File;
use std::path::Path;

const MAX_HEADER_SIZE: usize = 100 * 1024 * 1024;
const DEFAULT_TENSOR_LIMIT: usize = 20;

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

pub fn inspect_model(path: &Path, json: bool, show_all: bool) -> Result<(), String> {
    let file = File::open(path).map_err(|error| format!("无法打开 {}：{error}", path.display()))?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|error| format!("无法映射 {}：{error}", path.display()))?
    };
    let info = parse_model(&mmap)?;
    if json {
        print_json(path, mmap.len(), &info)?;
    } else {
        print_model(path, mmap.len(), &info, show_all);
    }
    Ok(())
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
    let mut ranges = Vec::with_capacity(tensor_count.min(1_000_000));
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
        let type_id = reader.u32()?;
        let dtype = ggml_type_name(type_id);
        let offset = reader.u64()?;
        let bytes = ggml_type_size(type_id).map(|(block_size, type_size)| {
            let elements = shape.iter().product::<u64>().max(1);
            elements.div_ceil(block_size) * type_size
        });
        if let Some(bytes) = bytes {
            ranges.push((offset, offset.checked_add(bytes), name.clone()));
        }
        tensors.push(TensorInfo {
            name,
            shape,
            dtype: dtype.to_string(),
            bytes,
        });
    }

    let data_start = reader.position();
    let data_start = align_up(data_start as u64, alignment)
        .ok_or_else(|| "GGUF 数据区偏移溢出。".to_string())?;
    if data_start > data.len() as u64 {
        return Err("GGUF 张量数据区超出文件范围。".into());
    }
    let data_len = data.len() as u64 - data_start;
    let mut absolute_ranges = Vec::with_capacity(ranges.len());
    for (offset, end, name) in ranges {
        if offset > data_len {
            return Err(format!("GGUF 张量“{name}”的数据偏移超出文件范围。"));
        }
        if let Some(end) = end {
            if end > data_len {
                return Err(format!("GGUF 张量“{name}”的数据超出文件范围。"));
            }
            absolute_ranges.push((offset, end, name));
        }
    }
    absolute_ranges.sort_by_key(|range| range.0);
    for pair in absolute_ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(format!(
                "GGUF 张量“{}”和“{}”的数据范围重叠。",
                pair[0].2, pair[1].2
            ));
        }
    }
    Ok(ModelInfo {
        format: "GGUF",
        tensors,
        metadata,
    })
}

/// GGML 类型名,与 GGUF 规范的类型编号对应。
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

/// GGML 类型的 (块大小, 每块字节数),依据 GGUF 规范;无法确定的类型返回 None,
/// 对应张量将跳过字节大小与重叠校验。
fn ggml_type_size(value: u32) -> Option<(u64, u64)> {
    match value {
        0 => Some((1, 4)),         // F32
        1 => Some((1, 2)),         // F16
        2 => Some((32, 18)),       // Q4_0
        3 => Some((32, 20)),       // Q4_1
        6 => Some((32, 22)),       // Q5_0
        7 => Some((32, 24)),       // Q5_1
        8 => Some((32, 34)),       // Q8_0
        9 => Some((32, 36)),       // Q8_1
        10 => Some((256, 84)),     // Q2_K
        11 => Some((256, 110)),    // Q3_K
        12 => Some((256, 144)),    // Q4_K
        13 => Some((256, 176)),    // Q5_K
        14 => Some((256, 210)),    // Q6_K
        16 => Some((256, 66)),     // IQ2_XXS
        17 => Some((256, 74)),     // IQ2_XS
        18 => Some((256, 98)),     // IQ3_XXS
        19 => Some((256, 50)),     // IQ1_S
        20 => Some((32, 20)),      // IQ4_NL
        21 => Some((256, 110)),    // IQ3_S
        22 => Some((256, 65)),     // IQ2_S
        23 => Some((256, 72)),     // IQ4_XS
        24 => Some((1, 1)),        // I8
        25 => Some((1, 2)),        // I16
        26 => Some((1, 4)),        // I32
        27 => Some((1, 8)),        // I64
        28 => Some((1, 8)),        // F64
        29 => Some((256, 56)),     // IQ1_M
        30 => Some((1, 2)),        // BF16
        31..=33 => Some((32, 18)), // Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8
        _ => None,                 // Q8_K 等内部或未知类型
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

pub(crate) fn format_size(size: u64) -> String {
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

    /// 构造一个最小 GGUF v3 文件,可附加元数据、张量描述和数据区。
    fn gguf_file(
        metadata: &[(&str, u32, String)],
        tensors: &[(&str, Vec<u64>, u32, u64)],
        data: &[u8],
    ) -> Vec<u8> {
        let write_string = |file: &mut Vec<u8>, value: &str| {
            file.extend_from_slice(&(value.len() as u64).to_le_bytes());
            file.extend_from_slice(value.as_bytes());
        };
        let mut file = b"GGUF".to_vec();
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        file.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value_type, value) in metadata {
            write_string(&mut file, key);
            file.extend_from_slice(&value_type.to_le_bytes());
            match value_type {
                10 => file.extend_from_slice(&value.parse::<u64>().unwrap().to_le_bytes()),
                4 => file.extend_from_slice(&value.parse::<u32>().unwrap().to_le_bytes()),
                _ => panic!("测试仅支持 u32/u64 元数据"),
            }
        }
        for (name, shape, type_id, offset) in tensors {
            write_string(&mut file, name);
            file.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dimension in shape {
                file.extend_from_slice(&dimension.to_le_bytes());
            }
            file.extend_from_slice(&type_id.to_le_bytes());
            file.extend_from_slice(&offset.to_le_bytes());
        }
        if !file.len().is_multiple_of(32) {
            file.resize(file.len().next_multiple_of(32), 0);
        }
        file.extend_from_slice(data);
        file
    }

    #[test]
    fn parses_minimal_gguf() {
        let file = gguf_file(&[], &[], &[]);
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
    fn reports_gguf_tensor_sizes() {
        // Q8_0:64*64 元素 / 32 * 34 字节 = 4352。
        let file = gguf_file(&[], &[("w", vec![64, 64], 8, 0)], &[0; 4352]);
        let info = parse_model(&file).unwrap();
        assert_eq!(info.format, "GGUF");
        assert_eq!(info.tensors[0].dtype, "Q8_0");
        assert_eq!(info.tensors[0].bytes, Some(4352));
    }

    #[test]
    fn rejects_overlapping_gguf_tensors() {
        // 两个 F32 张量各 4 元素(16 字节),第二个偏移与第一个重叠。
        let file = gguf_file(&[], &[("a", vec![4], 0, 0), ("b", vec![4], 0, 8)], &[0; 32]);
        assert!(parse_model(&file).unwrap_err().contains("重叠"));
    }

    #[test]
    fn accepts_non_overlapping_gguf_tensors() {
        let file = gguf_file(
            &[],
            &[("a", vec![4], 0, 0), ("b", vec![4], 0, 16)],
            &[0; 32],
        );
        let info = parse_model(&file).unwrap();
        assert_eq!(info.tensors.len(), 2);
    }

    #[test]
    fn rejects_gguf_tensor_data_out_of_bounds() {
        // 1024 个 F32 需要 4096 字节,数据区为空。
        let file = gguf_file(&[], &[("a", vec![1024], 0, 0)], &[]);
        assert!(parse_model(&file).unwrap_err().contains("超出文件范围"));
    }
}
