use candle_core::{Tensor, Result, D};
use candle_nn::{VarBuilder, linear, Linear, rms_norm, RmsNorm};
use crate::rope::RotaryEmbedding;

/// Full Attention Residuals.
/// Aggregates all previous layer outputs using a learned pseudo-attention mechanism.
pub struct FullAttnRes {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    norm: Option<RmsNorm>,
}

impl FullAttnRes {
    /// Creates a new `FullAttnRes` instance with cross-attention logic.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;
        let norm = rms_norm(dim, 1e-5, vb.pp("norm")).ok();
        Ok(Self { q_proj, k_proj, v_proj, out_proj, norm })
    }

    /// Performs the forward pass using the last hidden state as a query over all previous states.
    pub fn forward(&self, hidden_states: &[Tensor], rope: &RotaryEmbedding, start_pos: usize) -> Result<Tensor> {
        if hidden_states.is_empty() {
            candle_core::bail!("hidden_states cannot be empty");
        }

        let l = hidden_states.len();
        let query_state = &hidden_states[l - 1]; // [T, D]

        // 1. Prepare Query
        let q = query_state.apply(&self.q_proj)?;
        let q = rope.apply(&q, start_pos)?; // [T, D]

        // 2. Prepare Keys and Values from all layers
        let mut k_stack = Vec::with_capacity(l);
        let mut v_stack = Vec::with_capacity(l);

        for state in hidden_states {
            let k = state.apply(&self.k_proj)?;
            let v = state.apply(&self.v_proj)?;

            // Each layer's output passed here was computed at its own "time"
            // For cross-layer attention, we can use 0 or some fixed position since it's per-token
            k_stack.push(rope.apply(&k, 0)?.unsqueeze(0)?); // [1, T, D]
            v_stack.push(v.unsqueeze(0)?); // [1, T, D]
        }

        let k_all = Tensor::cat(&k_stack, 0)?; // [L, T, D]
        let v_all = Tensor::cat(&v_stack, 0)?; // [L, T, D]

        // Transpose for cross-attention over layers per token
        let k_all = k_all.transpose(0, 1)?; // [T, L, D]
        let v_all = v_all.transpose(0, 1)?; // [T, L, D]

        // 3. Attention calculation [T, 1, D] @ [T, D, L] -> [T, 1, L]
        let q = q.unsqueeze(1)?; // [T, 1, D]
        let scale = (q.dim(D::Minus1)? as f64).sqrt();
        let scores = (q.matmul(&k_all.transpose(1, 2)?)? / scale)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?; // [T, 1, L]

        // 4. Aggregate [T, 1, L] @ [T, L, D] -> [T, 1, D]
        let context = attn.matmul(&v_all)?.squeeze(1)?; // [T, D]
        let mut aggregated = context.apply(&self.out_proj)?;

        if let Some(norm) = &self.norm {
            aggregated = aggregated.apply(norm)?;
        }

        Ok(aggregated)
    }
}

/// Block Attention Residuals.
/// Aggregates outputs from completed blocks and current local states.
pub struct BlockAttnRes {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl BlockAttnRes {
    /// Creates a new `BlockAttnRes` instance with cross-attention logic.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;
        Ok(Self { q_proj, k_proj, v_proj, out_proj })
    }

    /// Performs the forward pass using cross-attention over block outputs.
    pub fn forward(&self, states: &[Tensor], current: &Tensor, rope: &RotaryEmbedding, start_pos: usize) -> Result<Tensor> {
        if states.is_empty() {
            return Ok(current.clone());
        }

        let mut all_history = states.to_vec();
        all_history.push(current.clone());
        let l = all_history.len();

        let q = current.apply(&self.q_proj)?;
        let q = rope.apply(&q, start_pos)?.unsqueeze(1)?; // [T, 1, D]

        let mut k_stack = Vec::with_capacity(l);
        let mut v_stack = Vec::with_capacity(l);

        for state in &all_history {
            let k = state.apply(&self.k_proj)?;
            let v = state.apply(&self.v_proj)?;
            k_stack.push(rope.apply(&k, 0)?.unsqueeze(0)?);
            v_stack.push(v.unsqueeze(0)?);
        }

        let k_all = Tensor::cat(&k_stack, 0)?.transpose(0, 1)?; // [T, L, D]
        let v_all = Tensor::cat(&v_stack, 0)?.transpose(0, 1)?; // [T, L, D]

        let scale = (q.dim(D::Minus1)? as f64).sqrt();
        let scores = (q.matmul(&k_all.transpose(1, 2)?)? / scale)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let context = attn.matmul(&v_all)?.squeeze(1)?;
        context.apply(&self.out_proj)
    }
}
