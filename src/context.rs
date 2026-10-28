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
        let model_path = model_path.canonicalize().map_err(|error| {
            format!(
                "Failed to resolve model path '{}': {error}",
                model_path.display()
            )
        })?;
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
                "Context '{}' uses unsupported data version {} (currently supported: {}).",
                self.name, self.schema_version, SCHEMA_VERSION
            ));
        }
        validate_name(&self.name)?;
        if self.options.max_tokens == 0
            || self.options.ctx_size == 0
            || self.options.threads == Some(0)
            || !self.options.temperature.is_finite()
            || self.options.temperature < 0.0
            || !(0.0..=1.0).contains(&self.options.top_p)
            || !self.options.repeat_penalty.is_finite()
            || self.options.repeat_penalty < 1.0
        {
            return Err(format!(
                "Context '{}' contains invalid inference options.",
                self.name
            ));
        }
        Ok(())
    }

    pub fn validate_for_launch(
        &self,
        model_path: &Path,
        requested_options: &ChatOptions,
        options_specified: bool,
    ) -> Result<ChatOptions, String> {
        // Bind a saved conversation to the canonical model path to avoid silently resuming on another model.
        // 使用规范化后的模型路径绑定会话，避免无提示地在另一模型上恢复历史记录。
        ensure_gguf_path(model_path)?;
        let resolved_path = model_path.canonicalize().map_err(|error| {
            format!(
                "Failed to resolve model path '{}': {error}",
                model_path.display()
            )
        })?;
        if resolved_path.to_string_lossy().as_ref() != self.model_path.as_str() {
            return Err(format!(
                "Context '{}' is bound to model '{}', not '{}'.",
                self.name,
                self.model_path,
                resolved_path.display()
            ));
        }
        if options_specified && requested_options != &self.options {
            return Err(format!(
                "Inference options saved in context '{}' do not match the command-line options; omit inference options to restore the archived settings, or use the same options.",
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
        let bytes = fs::read(&path).map_err(|error| {
            format!(
                "Failed to read context '{name}' ({}): {error}",
                path.display()
            )
        })?;
        let archive: ContextArchive = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Context '{name}' is not a valid JSON archive: {error}"))?;
        archive.validate()?;
        if archive.name != name {
            return Err(format!(
                "Name in context file '{}' does not match the requested name.",
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
        fs::create_dir_all(&directory).map_err(|error| {
            format!(
                "Failed to create context directory '{}': {error}",
                directory.display()
            )
        })?;
        #[cfg(unix)]
        if self.directory.is_none() {
            fs::set_permissions(&directory, Permissions::from_mode(0o700)).map_err(|error| {
                format!(
                    "Failed to restrict permissions on the default context directory '{}': {error}",
                    directory.display()
                )
            })?;
        }
        let target = Self::archive_path(&directory, &archive.name)?;
        let bytes = serde_json::to_vec_pretty(&archive)
            .map_err(|error| format!("Failed to serialize context '{}': {error}", archive.name))?;
        // Stage the complete archive in a private sibling file before replacing the target.
        // 先将完整归档写入权限受限的同目录临时文件，再替换目标文件。
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
                        "Failed to create temporary context file '{}': {error}",
                        temp_path.display()
                    ));
                }
            }
        };
        if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            let _ = fs::remove_file(&temp_path);
            return Err(format!(
                "Failed to write context '{}': {error}",
                archive.name
            ));
        }
        drop(file);
        if let Err(error) = fs::rename(&temp_path, &target) {
            let _ = fs::remove_file(&temp_path);
            return Err(format!(
                "Failed to update context '{}' ({}): {error}",
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
                    "Failed to read context directory '{}': {error}",
                    directory.display()
                ));
            }
        };
        let mut archives = Vec::new();
        for entry in entries {
            let entry = entry
                .map_err(|error| format!("Failed to read context directory entry: {error}"))?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path).map_err(|error| {
                format!("Failed to read context file '{}': {error}", path.display())
            })?;
            let archive: ContextArchive = serde_json::from_slice(&bytes).map_err(|error| {
                format!(
                    "Context file '{}' is not a valid JSON archive: {error}",
                    path.display()
                )
            })?;
            archive.validate()?;
            if path.file_stem().and_then(|stem| stem.to_str()) != Some(archive.name.as_str()) {
                return Err(format!(
                    "Filename of context file '{}' does not match the archive name.",
                    path.display()
                ));
            }
            archives.push(archive);
        }
        // Stable ordering makes the CLI listing predictable across filesystem implementations.
        // 显式排序可让 CLI 列表在不同文件系统上保持一致。
        archives.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(archives)
    }

    pub fn delete(&self, name: &str) -> Result<(), String> {
        let directory = self.directory()?;
        let path = Self::archive_path(&directory, name)?;
        fs::remove_file(&path).map_err(|error| {
            format!(
                "Failed to delete context '{name}' ({}): {error}",
                path.display()
            )
        })
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
            "Chat and contexts only support GGUF model files: {}",
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
            "Context names must be 1-64 ASCII letters, digits, periods, hyphens, or underscores, and cannot be '.' or '..'."
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
            .ok_or_else(|| {
                "Unable to determine the user data directory; specify one with --context-dir."
                    .to_string()
            })?;
        return Ok(PathBuf::from(base).join("wolf").join("contexts"));
    }

    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME").ok_or_else(|| {
            "Unable to determine the user data directory; specify one with --context-dir."
                .to_string()
        })?;
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
        let home = std::env::var_os("HOME").ok_or_else(|| {
            "Unable to determine the user data directory; specify one with --context-dir."
                .to_string()
        })?;
        return Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("wolf")
            .join("contexts"));
    }

    #[allow(unreachable_code)]
    Err("Unable to determine the user data directory on this platform; specify one with --context-dir.".into())
}

fn unix_timestamp() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| format!("System time is earlier than the Unix epoch; unable to create context archive: {error}"))
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
        assert!(store.load("broken").unwrap_err().contains("valid JSON"));
        fs::remove_file(directory.join("broken.json")).unwrap();

        let mut archive = test_archive("unsupported");
        archive.schema_version = SCHEMA_VERSION + 1;
        fs::write(
            directory.join("unsupported.json"),
            serde_json::to_vec(&archive).unwrap(),
        )
        .unwrap();
        assert!(
            store
                .load("unsupported")
                .unwrap_err()
                .contains("unsupported")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn older_archives_default_context_size_to_8192() {
        let mut options = serde_json::to_value(DEFAULT_CHAT_OPTIONS).unwrap();
        options.as_object_mut().unwrap().remove("ctx_size");
        let restored: ChatOptions = serde_json::from_value(options).unwrap();
        assert_eq!(restored.ctx_size, crate::options::DEFAULT_CONTEXT_TOKENS);
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
                .contains("not")
        );
        let changed_options = ChatOptions {
            max_tokens: DEFAULT_CHAT_OPTIONS.max_tokens - 1,
            ..DEFAULT_CHAT_OPTIONS
        };
        assert!(
            archive
                .validate_for_launch(&model_path, &changed_options, true)
                .unwrap_err()
                .contains("Inference options")
        );

        archive.options.temperature = -1.0;
        assert!(archive.validate().unwrap_err().contains("invalid"));
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn rejects_non_gguf_chat_models() {
        assert!(ContextArchive::new(Path::new("model.safetensors"), DEFAULT_CHAT_OPTIONS).is_err());
        assert!(ContextArchive::new(Path::new("model-directory"), DEFAULT_CHAT_OPTIONS).is_err());
    }
}
