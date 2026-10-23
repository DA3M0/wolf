//! Model file parsing and validation for GGUF and Safetensors.

use serde::{Deserialize, Serialize};
use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor name from the model file.
    pub name: String,
    /// Tensor dimensions.
    pub shape: Vec<u64>,
    /// Tensor data type name.
    pub dtype: String,
    /// Estimated or declared tensor data size in bytes, when known.
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Path passed to [`inspect_model`].
    pub path: PathBuf,
    /// Total model file size in bytes.
    pub file_size: u64,
    /// Detected format (`GGUF` or `Safetensors`).
    pub format: String,
    /// Parsed model tensors.
    pub tensors: Vec<TensorInfo>,
    /// Parsed model metadata.
    pub metadata: Vec<(String, String)>,
}

#[derive(Debug, Deserialize)]
struct ParsedModel {
    format: String,
    tensors: Vec<TensorInfo>,
    metadata: Vec<(String, String)>,
}

unsafe extern "C" {
    fn wolf_parse_model(
        data: *const u8,
        length: usize,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn wolf_parser_free(pointer: *mut c_void);
}

struct CAllocation(*mut c_char);

impl Drop for CAllocation {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { wolf_parser_free(self.0.cast()) };
        }
    }
}

/// Validate a GGUF or Safetensors model file and return its parsed metadata.
pub fn inspect_model(path: &Path) -> Result<ModelInfo, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("无法打开 {}：{error}", path.display()))?;
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|error| format!("无法映射 {}：{error}", path.display()))?
    };
    let parsed = parse_model(&mmap)?;
    Ok(ModelInfo {
        path: path.to_path_buf(),
        file_size: u64::try_from(mmap.len())
            .map_err(|_| "模型文件大小超出当前平台支持范围。".to_string())?,
        format: parsed.format,
        tensors: parsed.tensors,
        metadata: parsed.metadata,
    })
}

fn parse_model(data: &[u8]) -> Result<ParsedModel, String> {
    let mut json = std::ptr::null_mut();
    let mut error = std::ptr::null_mut();
    let success = unsafe { wolf_parse_model(data.as_ptr(), data.len(), &mut json, &mut error) };
    let json = CAllocation(json);
    let error = CAllocation(error);

    if success == 0 {
        if error.0.is_null() {
            return Err("模型文件解析失败。".to_string());
        }
        return Err(unsafe { CStr::from_ptr(error.0) }
            .to_string_lossy()
            .into_owned());
    }
    if json.0.is_null() {
        return Err("模型文件解析器未返回结果。".to_string());
    }
    serde_json::from_slice(unsafe { CStr::from_ptr(json.0) }.to_bytes())
        .map_err(|error| format!("模型解析结果无效：{error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn safetensors_file(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(payload);
        data
    }

    #[test]
    fn inspect_model_maps_file_and_returns_metadata() {
        let path = std::env::temp_dir().join(format!(
            "wolf-inspect-{}-{}.safetensors",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let file = safetensors_file(
            r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        std::fs::write(&path, &file).unwrap();

        let info = inspect_model(&path).unwrap();

        assert_eq!(info.path, path);
        assert_eq!(info.file_size, file.len() as u64);
        assert_eq!(info.format, "Safetensors");
        assert_eq!(info.tensors[0].name, "weight");
        std::fs::remove_file(path).unwrap();
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
    fn safetensors_metadata_and_keys_remain_sorted_and_decoded() {
        let file = safetensors_file(
            r#"{"z":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F16","shape":[1],"data_offsets":[4,6]},"__metadata__":{"z":"last","a":"\u4e2d\u6587"}}"#,
            &[0; 6],
        );
        let info = parse_model(&file).unwrap();
        assert_eq!(info.tensors[0].name, "a");
        assert_eq!(
            info.metadata,
            [("a".into(), "中文".into()), ("z".into(), "last".into())]
        );
    }

    #[test]
    fn preserves_embedded_nuls_in_safetensors_strings() {
        let file = safetensors_file(
            r#"{"a\u0000b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"__metadata__":{"note":"m\u0000x"}}"#,
            &[0; 4],
        );
        let info = parse_model(&file).unwrap();
        assert_eq!(info.tensors[0].name, "a\0b");
        assert_eq!(info.metadata[0].1, "m\0x");
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
    fn accepts_adjacent_zero_length_safetensors_range() {
        let file = safetensors_file(
            r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        assert_eq!(parse_model(&file).unwrap().tensors.len(), 2);
    }

    #[test]
    fn rejects_invalid_safetensors_json() {
        let file = safetensors_file("{", &[]);
        assert!(parse_model(&file).unwrap_err().contains("JSON"));
    }

    #[test]
    fn rejects_overlapping_safetensors_tensors() {
        let file = safetensors_file(
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#,
            &[0; 12],
        );
        assert!(parse_model(&file).unwrap_err().contains("重叠"));
    }

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
                0 => file.push(value.parse::<u8>().unwrap()),
                1 => file.push(value.parse::<i8>().unwrap() as u8),
                2 => file.extend_from_slice(&value.parse::<u16>().unwrap().to_le_bytes()),
                3 => file.extend_from_slice(&value.parse::<i16>().unwrap().to_le_bytes()),
                10 => file.extend_from_slice(&value.parse::<u64>().unwrap().to_le_bytes()),
                4 => file.extend_from_slice(&value.parse::<u32>().unwrap().to_le_bytes()),
                5 => file.extend_from_slice(&value.parse::<i32>().unwrap().to_le_bytes()),
                6 => file.extend_from_slice(&value.parse::<f32>().unwrap().to_le_bytes()),
                7 => file.push(u8::from(value == "true")),
                8 => write_string(&mut file, value),
                11 => file.extend_from_slice(&value.parse::<i64>().unwrap().to_le_bytes()),
                12 => file.extend_from_slice(&value.parse::<f64>().unwrap().to_le_bytes()),
                _ => panic!("测试包含不支持的 GGUF 元数据类型"),
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
    fn rejects_unsupported_gguf_version() {
        let mut file = gguf_file(&[], &[], &[]);
        file[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse_model(&file).unwrap_err().contains("不支持 GGUF 版本"));
    }

    #[test]
    fn reports_gguf_tensor_sizes() {
        let file = gguf_file(&[], &[("w", vec![64, 64], 8, 0)], &[0; 4352]);
        let info = parse_model(&file).unwrap();
        assert_eq!(info.format, "GGUF");
        assert_eq!(info.tensors[0].dtype, "Q8_0");
        assert_eq!(info.tensors[0].bytes, Some(4352));
    }

    #[test]
    fn reads_gguf_metadata_and_unknown_tensor_types() {
        let file = gguf_file(
            &[
                ("general.alignment", 10, "64".into()),
                ("general.name", 4, "7".into()),
            ],
            &[("opaque", vec![8], 99, 0)],
            &[0; 8],
        );
        let info = parse_model(&file).unwrap();
        assert_eq!(
            info.metadata,
            [
                ("general.alignment".into(), "64".into()),
                ("general.name".into(), "7".into())
            ]
        );
        assert_eq!(info.tensors[0].dtype, "UNKNOWN");
        assert_eq!(info.tensors[0].bytes, None);
    }

    #[test]
    fn formats_gguf_scalar_metadata_values() {
        let file = gguf_file(
            &[
                ("bool", 7, "true".into()),
                ("f32", 6, "1.25".into()),
                ("f64", 12, "-2.5".into()),
                ("i8", 1, "-3".into()),
                ("i16", 3, "-4".into()),
                ("i32", 5, "-5".into()),
                ("i64", 11, "-6".into()),
                ("u8", 0, "7".into()),
                ("u16", 2, "8".into()),
                ("u32", 4, "9".into()),
                ("u64", 10, "10".into()),
            ],
            &[],
            &[],
        );
        let info = parse_model(&file).unwrap();
        assert_eq!(
            info.metadata,
            [
                ("bool".into(), "true".into()),
                ("f32".into(), "1.25".into()),
                ("f64".into(), "-2.5".into()),
                ("i8".into(), "-3".into()),
                ("i16".into(), "-4".into()),
                ("i32".into(), "-5".into()),
                ("i64".into(), "-6".into()),
                ("u8".into(), "7".into()),
                ("u16".into(), "8".into()),
                ("u32".into(), "9".into()),
                ("u64".into(), "10".into()),
            ]
        );
    }

    #[test]
    fn preserves_embedded_nuls_in_gguf_strings() {
        let file = gguf_file(&[("general.name", 8, "model\0name".into())], &[], &[]);
        let info = parse_model(&file).unwrap();
        assert_eq!(info.metadata[0].1, "model\0name");
    }

    #[test]
    fn rejects_overlapping_gguf_tensors() {
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
        let file = gguf_file(&[], &[("a", vec![1024], 0, 0)], &[]);
        assert!(parse_model(&file).unwrap_err().contains("超出文件范围"));
    }
}
