use candle_core::{Tensor, Result, D};
use candle_nn::{VarBuilder, linear, Linear, rms_norm, RmsNorm};

/// Full Attention Residuals.
/// Aggregates all previous layer outputs using a learned pseudo-attention mechanism.
pub struct FullAttnRes {
    proj: Linear,
    norm: Option<RmsNorm>,
}

impl FullAttnRes {
    /// Creates a new `FullAttnRes` instance.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear(dim, 1, vb.pp("proj"))?; // Pseudo-query projection
        let norm = rms_norm(dim, 1e-5, vb.pp("norm")).ok();
        Ok(Self { proj, norm })
    }

    /// Performs the forward pass of the attention residual block.
    pub fn forward(&self, hidden_states: &[Tensor]) -> Result<Tensor> {
        if hidden_states.is_empty() {
            candle_core::bail!("hidden_states cannot be empty");
        }

        let l = hidden_states.len();

        // Compute attention weights over all previous layers
        // hidden_states is Vec<[T, D]>
        let mut state_stack = Vec::with_capacity(l);
        let mut weight_stack = Vec::with_capacity(l);
        for state in hidden_states {
            // [T, D] -> [1, T, D]
            state_stack.push(state.unsqueeze(0)?);
            // [T, D] -> [T, 1]
            weight_stack.push(state.apply(&self.proj)?);
        }

        // Stack all previous states: [L, T, D]
        let all_states = Tensor::cat(&state_stack, 0)?;
        // Concatenate weights: [T, L]
        let weights = Tensor::cat(&weight_stack, D::Minus1)?;
        // Normalize weights over layers: [T, L]
        let weights = candle_nn::ops::softmax(&weights, D::Minus1)?;

        // Vectorized aggregation using matmul
        // Reshape weights to [T, 1, L] for batch matmul over all_states [T, L, D]
        // But we have [L, T, D]. Let's transpose all_states to [T, L, D]
        let all_states_t = all_states.transpose(0, 1)?; // [T, L, D]
        let weights_unsz = weights.unsqueeze(1)?; // [T, 1, L]

        // [T, 1, L] @ [T, L, D] -> [T, 1, D] -> [T, D]
        let mut aggregated = weights_unsz.matmul(&all_states_t)?.squeeze(1)?;

        if let Some(norm) = &self.norm {
            aggregated = aggregated.apply(norm)?;
        }

        Ok(aggregated)
    }
}

/// Block Attention Residuals.
/// Partitions layers into blocks for memory efficiency.
pub struct BlockAttnRes {
    proj: Linear,
}

impl BlockAttnRes {
    /// Creates a new `BlockAttnRes` instance.
    pub fn new(dim: usize, _block_size: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear(dim, 1, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    /// Performs the forward pass of the block attention residual block.
    pub fn forward(&self, block_outputs: &[Tensor], current_state: &Tensor) -> Result<Tensor> {
        if block_outputs.is_empty() {
            return Ok(current_state.clone());
        }

        // Combine all previous outputs in the block and the current state
        let mut hidden_states = block_outputs.to_vec();
        hidden_states.push(current_state.clone());

        let l = hidden_states.len();
        let mut state_stack = Vec::with_capacity(l);
        let mut weight_stack = Vec::with_capacity(l);

        for state in &hidden_states {
            state_stack.push(state.unsqueeze(0)?);
            weight_stack.push(state.apply(&self.proj)?);
        }

        let all_states_t = Tensor::cat(&state_stack, 0)?.transpose(0, 1)?; // [T, L, D]
        let weights = Tensor::cat(&weight_stack, D::Minus1)?; // [T, L]
        let weights_unsz = candle_nn::ops::softmax(&weights, D::Minus1)?.unsqueeze(1)?; // [T, 1, L]

        // [T, 1, L] @ [T, L, D] -> [T, 1, D] -> [T, D]
        let aggregated = weights_unsz.matmul(&all_states_t)?.squeeze(1)?;

        Ok(aggregated)
    }
}
