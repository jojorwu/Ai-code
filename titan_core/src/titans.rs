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
    eta: Tensor,
    decay: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl TitansMemory {
    /// Creates a new `TitansMemory` instance.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let num_heads = 8; // Default to 8 heads
        let head_dim = dim / num_heads;

        let key_proj = linear(dim, dim, vb.pp("key_proj"))?;
        let val_proj = linear(dim, dim, vb.pp("val_proj"))?;
        let gate_proj = linear(dim, dim, vb.pp("gate_proj"))?;
        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;
        let surprise_proj = linear(head_dim, 1, vb.pp("surprise_proj"))?;

        // Per-head learnable parameters [H]
        let eta = vb.get((num_heads,), "eta")?;
        let decay = vb.get((num_heads,), "decay")?;

        Ok(Self {
            key_proj,
            val_proj,
            gate_proj,
            out_proj,
            surprise_proj,
            eta,
            decay,
            num_heads,
            head_dim,
        })
    }

    /// Performs the forward pass of the memory module.
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

        // Apply SiLU gate to values (Gated Linear Unit like capacity)
        vals = candle_nn::ops::silu(&vals)?;

        // Apply RoPE with correct start position
        keys = rope.apply(&keys, start_pos)?;
        let gate = ops::sigmoid(&x.apply(&self.gate_proj)?)?; // [T, D]

        // Reshape for multi-head: [T, H, Hd]
        let keys = keys.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?; // [H, T, Hd]
        let vals = vals.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?; // [H, T, Hd]
        let gate = gate.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?; // [H, T, Hd]

        // Hyperparameters for the Delta-rule (learnable per-head, vectorized)
        let etas = (ops::sigmoid(&self.eta)? * 0.5)?;
        let decays = (ops::sigmoid(&self.decay)? * 0.1)?;

        let mut new_m_list = Vec::with_capacity(self.num_heads);
        let mut final_head_outputs = Vec::with_capacity(self.num_heads);

        // Process each head in parallel (recurrent within each head)
        for h in 0..self.num_heads {
            let kh = keys.get(h)?; // [T, Hd]
            let vh = vals.get(h)?; // [T, Hd]
            let gh = gate.get(h)?; // [T, Hd]
            let mh = memory_matrices.get(h)?; // [Hd, Hd]

            let eta_h = etas.get(h)?; // [1]
            let decay_h = decays.get(h)?; // [1]

            let (y_h, m_h_new) = self.update_memory_loop(&kh, &vh, &gh, &mh, &eta_h, &decay_h)?;

            final_head_outputs.push(y_h.unsqueeze(1)?); // [T, 1, Hd]
            new_m_list.push(m_h_new.unsqueeze(0)?); // [1, Hd, Hd]
        }

        let output = Tensor::cat(&final_head_outputs, 1)?; // [T, H, Hd]
        let output = output.reshape((t_size, d_size))?.apply(&self.out_proj)?;

        let updated_memory = Tensor::cat(&new_m_list, 0)?; // [H, Hd, Hd]

        Ok((output, updated_memory))
    }

    /// Helper to perform iterative memory updates (Delta-rule) for a single head.
    fn update_memory_loop(
        &self,
        keys: &Tensor,
        vals: &Tensor,
        gate: &Tensor,
        initial_matrix: &Tensor,
        eta: &Tensor,
        decay: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (t_size, _hd_size) = keys.dims2()?;
        let mut current_m = initial_matrix.clone();
        let mut outputs = Vec::with_capacity(t_size);

        let one_minus_decay = (decay.neg()?.affine(1.0, 1.0))?;

        // Pre-convert eta to a scalar for faster application if possible
        let eta_val = eta.to_vec0::<f32>()?;

        // Pre-calculate per-token gates mean to avoid repeated calls in the loop
        let gate_means = gate.mean(1)?.to_vec1::<f32>()?;

        for t in 0..t_size {
            let kt = keys.narrow(0, t, 1)?; // [1, Hd]
            let vt = vals.narrow(0, t, 1)?; // [1, Hd]

            // 1. Retrieve from memory: y_t = k_t * M_{t-1}
            let yt = kt.matmul(&current_m)?;
            outputs.push(yt.clone());

            // 2. Compute surprise gate: learnable surprise from retrieval error
            let diff = (vt - yt)?;
            // surprise_score: [1, 1]
            let surprise_score = diff.apply(&self.surprise_proj)?;
            let surprise_val = surprise_score.sum_all()?.to_dtype(candle_core::DType::F32)?.to_vec0::<f32>()?;
            let surprise_refined = (surprise_val as f64).tanh();
            let gt = gate_means[t] as f64 * surprise_refined;

            // 3. Delta update: ΔM = (v_t - y_t) ⊗ k_t
            let update = kt.t()?.matmul(&diff)?;

            // 4. Update M: M_t = (1 - decay) * M_{t-1} + (eta * surprise) * ΔM
            let gated_eta_val = (eta_val as f64 * gt) as f32;
            current_m = current_m
                .broadcast_mul(&one_minus_decay)?
                .broadcast_add(&update.affine(gated_eta_val as f64, 0.0)?)?;
        }

        let output = Tensor::cat(&outputs, 0)?;
        Ok((output, current_m))
    }
}
