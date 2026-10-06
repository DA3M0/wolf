//! 下一个 token 的选择策略。当前为贪心(argmax);采样参数随后引入。

use candle_core::Tensor;

pub fn select_next_token(logits: &Tensor) -> Result<u32, String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
}
