use candle_core::{Tensor, Result, D};
use candle_nn::{VarBuilder, linear, Linear, rms_norm, RmsNorm};

pub struct FullAttnRes {
    proj: Linear,
    norm: Option<RmsNorm>,
    #[allow(dead_code)]
    dim: usize,
}

impl FullAttnRes {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear(dim, 1, vb.pp("proj"))?; // Pseudo-query projection
        let norm = rms_norm(dim, 1e-5, vb.pp("norm")).ok();
        Ok(Self { proj, norm, dim })
    }

    pub fn forward(&self, hidden_states: &[Tensor]) -> Result<Tensor> {
        if hidden_states.is_empty() {
             candle_core::bail!("hidden_states cannot be empty");
        }

        let l = hidden_states.len();
        let last_state = &hidden_states[l - 1];

        // Compute attention weights over all previous layers
        let mut weights = Vec::with_capacity(l);
        for state in hidden_states {
            let w = state.apply(&self.proj)?; // [B, T, 1]
            weights.push(w);
        }

        let weights = Tensor::cat(&weights, D::Minus1)?; // [B, T, L]
        let weights = candle_nn::ops::softmax(&weights, D::Minus1)?;

        // Aggregate
        let mut aggregated = last_state.zeros_like()?;
        for (i, state) in hidden_states.iter().enumerate() {
            let w_i = weights.narrow(D::Minus1, i, 1)?; // [B, T, 1]
            let weighted_state = state.broadcast_mul(&w_i)?;
            aggregated = (aggregated + weighted_state)?;
        }

        if let Some(norm) = &self.norm {
            aggregated = aggregated.apply(norm)?;
        }

        Ok(aggregated)
    }
}

/// Block Attention Residuals
/// Partitions layers into blocks for memory efficiency.
pub struct BlockAttnRes {
    proj: Linear,
    #[allow(dead_code)]
    block_size: usize,
}

impl BlockAttnRes {
    pub fn new(dim: usize, block_size: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear(dim, 1, vb.pp("proj"))?;
        Ok(Self { proj, block_size })
    }

    pub fn forward(&self, block_outputs: &[Tensor], current_state: &Tensor) -> Result<Tensor> {
        if block_outputs.is_empty() {
            return Ok(current_state.clone());
        }

        // Attend over completed block representations
        let mut states = block_outputs.to_vec();
        states.push(current_state.clone());

        let mut weights = Vec::with_capacity(states.len());
        for state in &states {
            weights.push(state.apply(&self.proj)?);
        }

        let weights = Tensor::cat(&weights, D::Minus1)?;
        let weights = candle_nn::ops::softmax(&weights, D::Minus1)?;

        let mut aggregated = current_state.zeros_like()?;
        for (i, state) in states.iter().enumerate() {
            let w_i = weights.narrow(D::Minus1, i, 1)?;
            aggregated = (aggregated + state.broadcast_mul(&w_i)?)?;
        }

        Ok(aggregated)
    }
}
