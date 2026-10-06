//! 下一个 token 的选择策略:贪心(argmax)或温度采样,可选重复惩罚。

use crate::cli::ChatOptions;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;
use std::time::{SystemTime, UNIX_EPOCH};

/// 重复惩罚作用的最近上下文窗口长度。
const REPEAT_LAST_N: usize = 64;

pub struct Sampler {
    processor: LogitsProcessor,
    repeat_penalty: f32,
}

impl Sampler {
    pub fn new(options: &ChatOptions) -> Self {
        let seed = options.seed.unwrap_or_else(random_seed);
        let sampling = if options.temperature < 1e-7 {
            Sampling::ArgMax
        } else if options.top_k > 0 {
            Sampling::TopKThenTopP {
                k: options.top_k,
                p: options.top_p,
                temperature: options.temperature,
            }
        } else {
            Sampling::TopP {
                p: options.top_p,
                temperature: options.temperature,
            }
        };
        Self {
            processor: LogitsProcessor::from_sampling(seed, sampling),
            repeat_penalty: options.repeat_penalty,
        }
    }

    /// 依据当前 logits 选出下一个 token;`context` 是已喂入模型的最近 token
    /// 序列,用于计算重复惩罚。
    pub fn select(&mut self, logits: &Tensor, context: &[u32]) -> Result<u32, String> {
        let logits = logits
            .flatten_all()
            .map_err(|error| format!("无法整理模型输出：{error}"))?;
        let logits = if self.repeat_penalty > 1.0 && !context.is_empty() {
            let window_start = context.len().saturating_sub(REPEAT_LAST_N);
            apply_repeat_penalty(&logits, self.repeat_penalty, &context[window_start..])
                .map_err(|error| format!("无法应用重复惩罚：{error}"))?
        } else {
            logits
        };
        self.processor
            .sample(&logits)
            .map_err(|error| format!("无法选择下一个 token：{error}"))
    }
}

fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0x5EED_1234_ABCD_EF01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::DEFAULT_CHAT_OPTIONS;
    use candle_core::Device;

    fn options(overrides: impl FnOnce(&mut ChatOptions)) -> ChatOptions {
        let mut options = ChatOptions {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            ..DEFAULT_CHAT_OPTIONS
        };
        overrides(&mut options);
        options
    }

    #[test]
    fn greedy_sampling_picks_argmax() {
        let logits = Tensor::from_vec(vec![0f32, 4.0, 1.0], (1, 3), &Device::Cpu).unwrap();
        let mut sampler = Sampler::new(&options(|_| {}));
        assert_eq!(sampler.select(&logits, &[]).unwrap(), 1);
    }

    #[test]
    fn repeat_penalty_redirects_greedy_choice() {
        let logits = Tensor::from_vec(vec![3f32, 4.0], (1, 2), &Device::Cpu).unwrap();
        let mut sampler = Sampler::new(&options(|o| o.repeat_penalty = 2.0));
        assert_eq!(sampler.select(&logits, &[1]).unwrap(), 0);
        // 不在上下文中的 token 不受惩罚。
        assert_eq!(sampler.select(&logits, &[0]).unwrap(), 1);
    }

    #[test]
    fn same_seed_gives_same_sequence() {
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.4, 0.2, 0.3, 0.35, 0.25, 0.15, 0.45],
            (4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let run = |seed: u64| {
            let mut sampler = Sampler::new(&options(|o| {
                o.temperature = 1.0;
                o.top_p = 1.0;
                o.seed = Some(seed);
            }));
            (0..16)
                .map(|_| sampler.select(&logits, &[]).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42), run(43));
    }

    #[test]
    fn accepts_non_standard_logit_shapes() {
        for shape in [&[3usize][..], &[1, 3], &[1, 1, 3]] {
            let logits = Tensor::from_vec(vec![0f32, 4.0, 1.0], shape, &Device::Cpu).unwrap();
            let mut sampler = Sampler::new(&options(|_| {}));
            assert_eq!(sampler.select(&logits, &[]).unwrap(), 1);
        }
    }
}
