//! Named, portable chat context archives.

use crate::options::ChatOptions;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(unix)]
use std::{
    fs::Permissions,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
};

const SCHEMA_VERSION: u32 = 1;
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub user: String,
    pub assistant: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextArchive {
    pub schema_version: u32,
    pub name: String,
    pub model_path: String,
    pub options: ChatOptions,
    pub turns: Vec<ConversationTurn>,
    pub created_at: u64,
    pub updated_at: u64,
}

impl ContextArchive {
    pub fn new(model_path: &Path, options: ChatOptions) -> Result<Self, String> {
        ensure_gguf_path(model_path)?;
        let model_path = model_path
            .canonicalize()
            .map_err(|error| format!("无法解析模型路径“{}”：{error}", model_path.display()))?;
        let now = unix_timestamp()?;
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            name: String::new(),
            model_path: model_path.to_string_lossy().into_owned(),
            options,
            turns: Vec::new(),
            created_at: now,
            updated_at: now,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "上下文“{}”使用不支持的数据版本 {}（当前支持版本 {}）。",
                self.name, self.schema_version, SCHEMA_VERSION
            ));
        }
        validate_name(&self.name)?;
        if self.options.max_tokens == 0
            || self.options.threads == Some(0)
            || !self.options.temperature.is_finite()
            || self.options.temperature < 0.0
            || !(0.0..=1.0).contains(&self.options.top_p)
            || !self.options.repeat_penalty.is_finite()
            || self.options.repeat_penalty < 1.0
        {
            return Err(format!("上下文“{}”包含无效的推理选项。", self.name));
        }
        Ok(())
    }

    pub fn validate_for_launch(
        &self,
        model_path: &Path,
        requested_options: &ChatOptions,
        options_specified: bool,
    ) -> Result<ChatOptions, String> {
        ensure_gguf_path(model_path)?;
        let resolved_path = model_path
            .canonicalize()
            .map_err(|error| format!("无法解析模型路径“{}”：{error}", model_path.display()))?;
        if resolved_path.to_string_lossy().as_ref() != self.model_path.as_str() {
            return Err(format!(
                "上下文“{}”绑定到模型“{}”，不能用于“{}”。",
                self.name,
                self.model_path,
                resolved_path.display()
            ));
        }
        if options_specified && requested_options != &self.options {
            return Err(format!(
                "上下文“{}”保存的推理选项与命令行选项不匹配；请省略推理选项以恢复存档设置，或使用相同选项。",
                self.name
            ));
        }
        Ok(self.options.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ContextStore {
    directory: Option<PathBuf>,
}

impl ContextStore {
    pub fn new(directory: Option<PathBuf>) -> Self {
        Self { directory }
    }

    pub fn load(&self, name: &str) -> Result<ContextArchive, String> {
        let directory = self.directory()?;
        let path = Self::archive_path(&directory, name)?;
        let bytes = fs::read(&path)
            .map_err(|error| format!("无法读取上下文“{name}”（{}）：{error}", path.display()))?;
        let archive: ContextArchive = serde_json::from_slice(&bytes)
            .map_err(|error| format!("上下文“{name}”不是有效的 JSON 存档：{error}"))?;
        archive.validate()?;
        if archive.name != name {
            return Err(format!(
                "上下文文件“{}”中的名称与请求名称不匹配。",
                path.display()
            ));
        }
        Ok(archive)
    }

    pub fn save(&self, archive: &ContextArchive) -> Result<(), String> {
        let mut archive = archive.clone();
        archive.updated_at = unix_timestamp()?;
        archive.validate()?;
        let directory = self.directory()?;
        fs::create_dir_all(&directory)
            .map_err(|error| format!("无法创建上下文目录“{}”：{error}", directory.display()))?;
        #[cfg(unix)]
        if self.directory.is_none() {
            fs::set_permissions(&directory, Permissions::from_mode(0o700)).map_err(|error| {
                format!(
                    "无法限制默认上下文目录“{}”的访问权限：{error}",
                    directory.display()
                )
            })?;
        }
        let target = Self::archive_path(&directory, &archive.name)?;
        let bytes = serde_json::to_vec_pretty(&archive)
            .map_err(|error| format!("无法序列化上下文“{}”：{error}", archive.name))?;
        let (temp_path, mut file) = loop {
            let temp_path = Self::temporary_path(&directory, &archive.name);
            let mut open_options = OpenOptions::new();
            open_options.write(true).create_new(true);
            #[cfg(unix)]
            open_options.mode(0o600);
            match open_options.open(&temp_path) {
                Ok(file) => break (temp_path, file),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!(
                        "无法创建上下文临时文件“{}”：{error}",
                        temp_path.display()
                    ));
                }
            }
        };
        if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            let _ = fs::remove_file(&temp_path);
            return Err(format!("无法写入上下文“{}”：{error}", archive.name));
        }
        drop(file);
        if let Err(error) = fs::rename(&temp_path, &target) {
            let _ = fs::remove_file(&temp_path);
            return Err(format!(
                "无法更新上下文“{}”（{}）：{error}",
                archive.name,
                target.display()
            ));
        }
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<ContextArchive>, String> {
        let directory = self.directory()?;
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(format!(
                    "无法读取上下文目录“{}”：{error}",
                    directory.display()
                ));
            }
        };
        let mut archives = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| format!("无法读取上下文目录条目：{error}"))?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path)
                .map_err(|error| format!("无法读取上下文文件“{}”：{error}", path.display()))?;
            let archive: ContextArchive = serde_json::from_slice(&bytes).map_err(|error| {
                format!(
                    "上下文文件“{}”不是有效的 JSON 存档：{error}",
                    path.display()
                )
            })?;
            archive.validate()?;
            if path.file_stem().and_then(|stem| stem.to_str()) != Some(archive.name.as_str()) {
                return Err(format!(
                    "上下文文件“{}”的文件名与存档名称不匹配。",
                    path.display()
                ));
            }
            archives.push(archive);
        }
        archives.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(archives)
    }

    pub fn delete(&self, name: &str) -> Result<(), String> {
        let directory = self.directory()?;
        let path = Self::archive_path(&directory, name)?;
        fs::remove_file(&path)
            .map_err(|error| format!("无法删除上下文“{name}”（{}）：{error}", path.display()))
    }

    fn directory(&self) -> Result<PathBuf, String> {
        match &self.directory {
            Some(directory) => Ok(directory.clone()),
            None => default_context_directory(),
        }
    }

    fn archive_path(directory: &Path, name: &str) -> Result<PathBuf, String> {
        validate_name(name)?;
        Ok(directory.join(format!("{name}.json")))
    }

    fn temporary_path(directory: &Path, name: &str) -> PathBuf {
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        directory.join(format!(".{name}.{}.{}.tmp", std::process::id(), counter))
    }
}

fn ensure_gguf_path(path: &Path) -> Result<(), String> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return Err(format!(
            "对话和上下文仅支持 GGUF 模型文件：{}",
            path.display()
        ));
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.len() > 64
        || name == "."
        || name == ".."
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(
            "上下文名称必须为 1–64 个 ASCII 字母、数字、点、短横线或下划线，且不能是“.”或“..”。"
                .into(),
        );
    }
    Ok(())
}

fn default_context_directory() -> Result<PathBuf, String> {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var_os("LOCALAPPDATA")
            .or_else(|| std::env::var_os("APPDATA"))
            .ok_or_else(|| "无法确定用户数据目录；请使用 --context-dir 指定。".to_string())?;
        return Ok(PathBuf::from(base).join("wolf").join("contexts"));
    }

    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME")
            .ok_or_else(|| "无法确定用户数据目录；请使用 --context-dir 指定。".to_string())?;
        return Ok(PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("wolf")
            .join("contexts"));
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(base) = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from)
            && base.is_absolute()
        {
            return Ok(base.join("wolf").join("contexts"));
        }
        let home = std::env::var_os("HOME")
            .ok_or_else(|| "无法确定用户数据目录；请使用 --context-dir 指定。".to_string())?;
        return Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("wolf")
            .join("contexts"));
    }

    #[allow(unreachable_code)]
    Err("当前平台无法确定用户数据目录；请使用 --context-dir 指定。".into())
}

fn unix_timestamp() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| format!("系统时间早于 Unix epoch，无法创建上下文存档：{error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::DEFAULT_CHAT_OPTIONS;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_store() -> (ContextStore, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "wolf-context-test-{}-{}",
            std::process::id(),
            TEST_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        (ContextStore::new(Some(path.clone())), path)
    }

    fn test_archive(name: &str) -> ContextArchive {
        ContextArchive {
            schema_version: SCHEMA_VERSION,
            name: name.into(),
            model_path: "/models/test.gguf".into(),
            options: DEFAULT_CHAT_OPTIONS,
            turns: vec![ConversationTurn {
                user: "hello".into(),
                assistant: "world".into(),
            }],
            created_at: 1,
            updated_at: 1,
        }
    }

    #[test]
    fn rejects_unsafe_context_names() {
        for name in ["", ".", "..", "../outside", "a/b", "name with spaces", "é"] {
            assert!(validate_name(name).is_err(), "accepted {name:?}");
        }
        assert!(validate_name("valid_name-1.json").is_ok());
    }

    #[test]
    fn saves_lists_loads_and_replaces_contexts() {
        let (store, directory) = test_store();
        let mut archive = test_archive("session");
        store.save(&archive).unwrap();
        archive.turns[0].assistant = "updated".into();
        store.save(&archive).unwrap();
        let loaded = store.load("session").unwrap();
        assert_eq!(loaded.turns, archive.turns);
        assert!(loaded.updated_at >= archive.updated_at);
        assert_eq!(store.list().unwrap()[0].name, "session");
        store.delete("session").unwrap();
        assert!(store.list().unwrap().is_empty());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn reports_corrupt_and_unsupported_archives() {
        let (store, directory) = test_store();
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("broken.json"), b"{").unwrap();
        assert!(store.load("broken").unwrap_err().contains("有效的 JSON"));
        fs::remove_file(directory.join("broken.json")).unwrap();

        let mut archive = test_archive("unsupported");
        archive.schema_version = SCHEMA_VERSION + 1;
        fs::write(
            directory.join("unsupported.json"),
            serde_json::to_vec(&archive).unwrap(),
        )
        .unwrap();
        assert!(store.load("unsupported").unwrap_err().contains("不支持"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn empty_missing_directory_lists_no_contexts() {
        let (store, directory) = test_store();
        assert!(store.list().unwrap().is_empty());
        assert!(store.load("missing").is_err());
        assert!(store.delete("missing").is_err());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn validates_model_identity_and_saved_options() {
        let (_, directory) = test_store();
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("model.gguf");
        fs::write(&model_path, b"").unwrap();
        let mut archive = ContextArchive::new(&model_path, DEFAULT_CHAT_OPTIONS).unwrap();
        archive.name = "session".into();
        assert_eq!(
            archive
                .validate_for_launch(&model_path, &DEFAULT_CHAT_OPTIONS, false)
                .unwrap(),
            DEFAULT_CHAT_OPTIONS
        );

        let other_model = directory.join("other.gguf");
        fs::write(&other_model, b"").unwrap();
        assert!(
            archive
                .validate_for_launch(&other_model, &DEFAULT_CHAT_OPTIONS, false)
                .unwrap_err()
                .contains("不能用于")
        );
        let changed_options = ChatOptions {
            max_tokens: DEFAULT_CHAT_OPTIONS.max_tokens - 1,
            ..DEFAULT_CHAT_OPTIONS
        };
        assert!(
            archive
                .validate_for_launch(&model_path, &changed_options, true)
                .unwrap_err()
                .contains("推理选项")
        );

        archive.options.temperature = -1.0;
        assert!(archive.validate().unwrap_err().contains("无效"));
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn rejects_non_gguf_chat_models() {
        assert!(ContextArchive::new(Path::new("model.safetensors"), DEFAULT_CHAT_OPTIONS).is_err());
        assert!(ContextArchive::new(Path::new("model-directory"), DEFAULT_CHAT_OPTIONS).is_err());
    }
}
