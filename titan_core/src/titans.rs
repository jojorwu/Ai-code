//! Titans Neural Memory: Learning to Memorize at Test Time.
//!
//! This module implements a matrix-based long-term associative memory.
//! It uses an iterative Delta-rule to update the memory matrix `M`
//! and a "surprise" gating mechanism to determine when to update.

use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear, ops};
use crate::rope::RotaryEmbedding;

/// Titans Long-Term Memory (Neural Memory).
///
/// This module implements a persistent memory that uses a matrix state `M`
/// for associative storage and retrieval. It uses an iterative Delta-rule update
/// with a "surprise" gating mechanism.
pub struct TitansMemory {
    key_proj: Linear,
    val_proj: Linear,
    gate_proj: Linear,
    out_proj: Linear,
    surprise_proj: Linear,
    decay_proj: Linear,
    use_turbo_quant: bool,
    eta: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl TitansMemory {
    /// Creates a new `TitansMemory` instance.
    pub fn new(dim: usize, num_heads: usize, use_turbo_quant: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;

        let key_proj = linear(dim, dim, vb.pp("key_proj"))?;
        let val_proj = linear(dim, dim, vb.pp("val_proj"))?;
        let gate_proj = linear(dim, dim, vb.pp("gate_proj"))?;
        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;
        let surprise_proj = linear(head_dim, 1, vb.pp("surprise_proj"))?;
        let decay_proj = linear(dim, num_heads, vb.pp("decay_proj"))?;

        // Per-head learnable parameters [H]
        let eta = vb.get((num_heads,), "eta")?;

        Ok(Self {
            key_proj,
            val_proj,
            gate_proj,
            out_proj,
            surprise_proj,
            decay_proj,
            use_turbo_quant,
            eta,
            num_heads,
            head_dim,
        })
    }

    /// Performs the forward pass of the memory module using vectorized head processing.
    pub fn forward(
        &self,
        x: &Tensor,
        memory_matrices: &Tensor,
        rope: &RotaryEmbedding,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        // x: [T, D], memory_matrices: [H, Hd, Hd]
        let (t_size, d_size) = x.dims2()?;

        let mut keys = x.apply(&self.key_proj)?; // [T, D]
        let mut vals = x.apply(&self.val_proj)?; // [T, D]

        if self.use_turbo_quant {
             // PQ-QJL Compression for Memory Keys/Values
             let (r_k, d_k) = crate::quant::QJL::compress_pq_qjl(&keys)?;
             keys = crate::quant::PolarQuant::decompress(&r_k, &d_k)?;

             let (r_v, d_v) = crate::quant::QJL::compress_pq_qjl(&vals)?;
             vals = crate::quant::PolarQuant::decompress(&r_v, &d_v)?;
        }

        // Apply SiLU gate to values (Gated Linear Unit like capacity)
        vals = candle_nn::ops::silu(&vals)?;

        // Apply RoPE with correct start position
        keys = rope.apply(&keys, start_pos)?;
        let gate = ops::sigmoid(&x.apply(&self.gate_proj)?)?; // [T, D]

        // Reshape for multi-head: [H, T, Hd]
        let keys = keys.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?;
        let vals = vals.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?;
        let gate = gate.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?;

        // Hyperparameters for the Delta-rule (learnable per-head, vectorized)
        let etas = (ops::sigmoid(&self.eta)? * 0.5)?; // [H]
        // Dynamic context-aware decay [H, T]
        let decays = (ops::sigmoid(&x.apply(&self.decay_proj)?)? * 0.1)?.transpose(0, 1)?;

        // Vectorized memory update across all heads
        let (output, updated_memory) = self.update_memory_vectorized(
            &keys, &vals, &gate, memory_matrices, &etas, &decays
        )?;

        // output: [H, T, Hd] -> [T, H, Hd] -> [T, D]
        let output = output.transpose(0, 1)?.reshape((t_size, d_size))?.apply(&self.out_proj)?;

        Ok((output, updated_memory))
    }

    /// Performs iterative memory updates vectorized across heads.
    ///
    /// The update rule follows the Least Mean Squares (LMS) / Delta-rule with dynamic decay:
    /// $$ M_t = (1 - \text{decay}_t) M_{t-1} + \eta \cdot \text{surprise}_t \cdot ((v_t - y_t) \otimes k_t) $$
    /// where:
    /// - $y_t = k_t M_{t-1}$ is the value retrieved from the associative memory.
    /// - $v_t - y_t$ is the prediction error (surprise vector).
    /// - $\eta$ is the base learning rate for memory updates.
    /// - $\text{surprise}_t$ is the learnable gating factor predicted from retrieval error.
    /// - $\text{decay}_t$ is the context-aware forgetting factor.
    fn update_memory_vectorized(
        &self,
        keys: &Tensor,      // [H, T, Hd]
        vals: &Tensor,      // [H, T, Hd]
        gate: &Tensor,      // [H, T, Hd]
        initial_m: &Tensor, // [H, Hd, Hd]
        etas: &Tensor,      // [H]
        decays: &Tensor,    // [H, T]
    ) -> Result<(Tensor, Tensor)> {
        let (h_size, t_size, _hd_size) = keys.dims3()?;
        let mut current_m = initial_m.clone();
        let mut outputs = Vec::with_capacity(t_size);

        let one_minus_decays = (decays.neg()?.affine(1.0, 1.0))?; // [H, T]
        let gate_means = gate.mean(candle_core::D::Minus1)?; // [H, T]

        for t in 0..t_size {
            let kt = keys.narrow(1, t, 1)?; // [H, 1, Hd]
            let vt = vals.narrow(1, t, 1)?; // [H, 1, Hd]

            // 1. Retrieve: y_t = k_t * M_{t-1}
            let yt = kt.matmul(&current_m)?; // [H, 1, Hd]
            outputs.push(yt.clone());

            // 2. Surprise calculation
            let diff = (vt - yt)?; // [H, 1, Hd]
            let surprise_score = diff.apply(&self.surprise_proj)?; // [H, 1, 1]
            let surprise_refined = surprise_score.tanh()?; // [H, 1, 1]

            let gt = gate_means.narrow(1, t, 1)?.unsqueeze(candle_core::D::Minus1)?; // [H, 1, 1]
            let total_surprise = gt.broadcast_mul(&surprise_refined)?; // [H, 1, 1]

            // 3. Delta update: ΔM = (v_t - y_t) ⊗ k_t
            let update = kt.transpose(1, 2)?.matmul(&diff)?; // [H, Hd, Hd]

            // 4. Update M
            let dt = one_minus_decays.narrow(1, t, 1)?.unsqueeze(candle_core::D::Minus1)?; // [H, 1, 1]
            let learning_rates = etas.reshape((h_size, 1, 1))?.broadcast_mul(&total_surprise)?; // [H, 1, 1]

            current_m = (current_m.broadcast_mul(&dt)? + update.broadcast_mul(&learning_rates)?)?;
        }

        let output = Tensor::cat(&outputs, 1)?; // [H, T, Hd]
        Ok((output, current_m))
    }
}
