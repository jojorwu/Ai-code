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

        let eta = vb.get((1,), "eta")?;
        let decay = vb.get((1,), "decay")?;

        Ok(Self {
            key_proj,
            val_proj,
            gate_proj,
            out_proj,
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
        rope: &RotaryEmbedding
    ) -> Result<(Tensor, Tensor)> {
        // x: [T, D], memory_matrices: [H, Hd, Hd]
        let (t_size, d_size) = x.dims2()?;

        let mut keys = x.apply(&self.key_proj)?; // [T, D]
        let mut vals = x.apply(&self.val_proj)?; // [T, D]

        // Apply SiLU gate to values (Gated Linear Unit like capacity)
        vals = candle_nn::ops::silu(&vals)?;

        // Apply RoPE to keys
        keys = rope.apply(&keys)?;
        let gate = ops::sigmoid(&x.apply(&self.gate_proj)?)?; // [T, D]

        // Reshape for multi-head: [T, H, Hd]
        let keys = keys.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?; // [H, T, Hd]
        let vals = vals.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?; // [H, T, Hd]
        let gate = gate.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?; // [H, T, Hd]

        // Hyperparameters for the Delta-rule (learnable)
        let eta = ops::sigmoid(&self.eta)?.flatten_all()?.to_vec1::<f32>()?[0] as f64 * 0.5; // Scale to [0, 0.5]
        let decay = ops::sigmoid(&self.decay)?.flatten_all()?.to_vec1::<f32>()?[0] as f64 * 0.1; // Scale to [0, 0.1]

        let mut new_m_list = Vec::with_capacity(self.num_heads);
        let mut final_head_outputs = Vec::with_capacity(self.num_heads);

        // Process each head in parallel (recurrent within each head)
        for h in 0..self.num_heads {
            let kh = keys.get(h)?; // [T, Hd]
            let vh = vals.get(h)?; // [T, Hd]
            let gh = gate.get(h)?; // [T, Hd]
            let mh = memory_matrices.get(h)?; // [Hd, Hd]

            let (y_h, m_h_new) = self.update_memory_loop(&kh, &vh, &gh, &mh, eta, decay)?;

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
        eta: f64,
        decay: f64,
    ) -> Result<(Tensor, Tensor)> {
        let (t_size, hd_size) = keys.dims2()?;
        let mut current_m = initial_matrix.clone();
        let mut outputs = Vec::with_capacity(t_size);

        for t in 0..t_size {
            let kt = keys.get(t)?.reshape((1, hd_size))?;
            let vt = vals.get(t)?.reshape((1, hd_size))?;
            let gt = gate.get(t)?.mean_all()?.to_vec0::<f32>()? as f64;

            // y_t = k_t * M_{t-1}
            let yt = kt.matmul(&current_m)?;
            outputs.push(yt.clone());

            // ΔM = (v_t - y_t) ⊗ k_t
            let diff = (vt - yt)?;
            let update = kt.t()?.matmul(&diff)?;

            // M_t = (1 - decay) * M_{t-1} + eta * surprise_gate * ΔM
            current_m = ((current_m * (1.0 - decay))? + (update * (eta * gt))?)?;
        }

        let output = Tensor::cat(&outputs, 0)?;
        Ok((output, current_m))
    }
}
