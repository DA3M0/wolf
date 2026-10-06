//! 对话提示模板与停止序列。

use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
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
    pub fn from_architecture(architecture: &str) -> Result<Self, String> {
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

    pub fn format_prompt(self, prompt: &str) -> String {
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

    /// 首轮提示:按各格式的惯例注入系统提示。没有系统角色实现的格式
    /// (Gemma、UserInst、Phi2)按官方/通用做法把系统提示并入首轮用户消息。
    pub fn format_prompt_with_system(self, prompt: &str, system: &str) -> String {
        match self {
            Self::Llama2 => format!("[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"),
            Self::Llama3 => format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::Qwen | Self::Gemma4 => format!(
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ),
            Self::Gemma => format!(
                "<start_of_turn>user\n{system}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            ),
            Self::UserInst => format!("[USER] {system}\n\n{prompt} [/USER]\n[INST]"),
            Self::Phi2 => format!("Instruct: {system}\n{prompt}\nOutput:"),
            Self::Phi3 => {
                format!("<|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n")
            }
        }
    }
}

/// 常见对话结束标记，用于从 tokenizer 中定位可用的停止 token。
pub const EOS_MARKER_TOKENS: &[&str] = &[
    "</s>",
    "<|eot_id|>",
    "<|im_end|>",
    "<end_of_turn>",
    "<|end_of_turn|>",
    "<|end|>",
    "<|fim_suffix|>",
];

/// 用户轮次开始标记；生成到这些序列时模型想把话头交还给用户，应停止本轮。
pub const TURN_START_MARKERS: &[&str] = &[
    "[USER]",
    " [USER]",
    "\n[USER]",
    "<|user|>",
    "<|im_start|>user",
    "<start_of_turn>user",
];

pub fn build_stop_sequences(tokenizer: &Tokenizer, eos_id: u32) -> Vec<Vec<u32>> {
    let mut sequences = vec![vec![eos_id]];
    for marker in EOS_MARKER_TOKENS {
        if let Some(id) = tokenizer.token_to_id(marker) {
            let sequence = vec![id];
            if !sequences.contains(&sequence) {
                sequences.push(sequence);
            }
        }
    }
    for marker in TURN_START_MARKERS {
        if let Ok(encoded) = tokenizer.encode(*marker, false) {
            let ids = encoded.get_ids();
            if !ids.is_empty() && !sequences.iter().any(|sequence| sequence == ids) {
                sequences.push(ids.to_vec());
            }
        }
    }
    sequences
}

pub fn matching_stop_sequence<'a>(
    generated: &[u32],
    stop_sequences: &'a [Vec<u32>],
) -> Option<&'a Vec<u32>> {
    stop_sequences
        .iter()
        .filter(|sequence| ends_with(generated, sequence))
        .max_by_key(|sequence| sequence.len())
}

pub fn is_turn_start_marker(tokenizer: &Tokenizer, sequence: &[u32]) -> bool {
    TURN_START_MARKERS
        .iter()
        .filter_map(|marker| tokenizer.encode(*marker, false).ok())
        .any(|encoding| encoding.get_ids() == sequence)
}

fn ends_with(values: &[u32], suffix: &[u32]) -> bool {
    !suffix.is_empty() && values.ends_with(suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn formats_first_turn_prompts_with_system() {
        assert_eq!(
            ChatFormat::Qwen.format_prompt_with_system("你好", "简洁作答"),
            "<|im_start|>system\n简洁作答<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            ChatFormat::Llama2.format_prompt_with_system("你好", "简洁作答"),
            "[INST] <<SYS>>\n简洁作答\n<</SYS>>\n\n你好 [/INST]"
        );
        assert_eq!(
            ChatFormat::Llama3.format_prompt_with_system("你好", "简洁作答"),
            "<|start_header_id|>system<|end_header_id|>\n\n简洁作答<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(
            ChatFormat::Gemma.format_prompt_with_system("你好", "简洁作答"),
            "<start_of_turn>user\n简洁作答\n\n你好<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(
            ChatFormat::Phi3.format_prompt_with_system("你好", "简洁作答"),
            "<|system|>\n简洁作答<|end|>\n<|user|>\n你好<|end|>\n<|assistant|>\n"
        );
        assert_eq!(
            ChatFormat::Phi2.format_prompt_with_system("你好", "简洁作答"),
            "Instruct: 简洁作答\n你好\nOutput:"
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
}
