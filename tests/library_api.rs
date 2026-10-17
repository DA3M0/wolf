use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use wolf::inspect_model;

static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn inspect_model_returns_structured_safetensors_data() {
    let path = std::env::temp_dir().join(format!(
        "wolf-library-api-{}-{}.safetensors",
        std::process::id(),
        FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let header = r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&[0; 4]);
    fs::write(&path, file).unwrap();

    let info = inspect_model(&path).unwrap();
    fs::remove_file(&path).unwrap();

    assert_eq!(info.path, path);
    assert_eq!(info.format, "Safetensors");
    assert_eq!(info.file_size, (8 + header.len() + 4) as u64);
    assert_eq!(info.tensors.len(), 1);
    assert_eq!(info.tensors[0].name, "weight");
    assert_eq!(info.tensors[0].shape, vec![1]);
    assert_eq!(info.tensors[0].dtype, "F32");
    assert_eq!(info.tensors[0].bytes, Some(4));
}
