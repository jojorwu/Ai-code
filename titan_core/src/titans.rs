use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear, ops};

/// Titans Long-Term Memory (Neural Memory).
///
/// This module implements a persistent memory that uses a matrix state `M`
/// for associative storage and retrieval. It uses an iterative Delta-rule update
/// with a "surprise" gating mechanism.
pub struct TitansMemory {
    key_proj: Linear,
    val_proj: Linear,
    gate_proj: Linear,
}

impl TitansMemory {
    /// Creates a new `TitansMemory` instance.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let key_proj = linear(dim, dim, vb.pp("key_proj"))?;
        let val_proj = linear(dim, dim, vb.pp("val_proj"))?;
        let gate_proj = linear(dim, dim, vb.pp("gate_proj"))?;
        Ok(Self {
            key_proj,
            val_proj,
            gate_proj,
        })
    }

    /// Performs the forward pass of the memory module.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `[T, D]`.
    /// * `memory_matrix` - The current persistent memory state of shape `[D, D]`.
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. The retrieved values from memory of shape `[T, D]`.
    /// 2. The updated memory matrix of shape `[D, D]`.
    pub fn forward(&self, x: &Tensor, memory_matrix: &Tensor) -> Result<(Tensor, Tensor)> {
        // x: [T, D], memory_matrix: [D, D]
        let keys = x.apply(&self.key_proj)?; // [T, D]
        let vals = x.apply(&self.val_proj)?; // [T, D]
        let gate = ops::sigmoid(&x.apply(&self.gate_proj)?)?; // [T, D]

        let mut current_m = memory_matrix.clone();
        let (t_size, d_size) = keys.dims2()?;
        let mut outputs = Vec::with_capacity(t_size);

        // Hyperparameters for the Delta-rule
        let eta = 0.1;
        let decay = 0.01;

        // Iteratively update memory for each token in the sequence.
        // The Delta rule update: M_t = (1 - decay) * M_{t-1} + eta * surprise_gate * ((v_t - M_{t-1}k_t) ⊗ k_t)
        for t in 0..t_size {
            let kt = keys.get(t)?.reshape((1, d_size))?;
            let vt = vals.get(t)?.reshape((1, d_size))?;
            let gt = gate.get(t)?.mean_all()?.to_vec0::<f32>()? as f64;

            // Retrieve from current state: y_t = k_t * M_{t-1}
            let yt = kt.matmul(&current_m)?;
            outputs.push(yt.clone());

            // Compute the error/surprise (v_t - y_t) and the outer product with k_t
            let diff = (vt - yt)?;
            let update = kt.t()?.matmul(&diff)?;

            // Update M with gated delta and decay
            current_m = ((current_m * (1.0 - decay))? + (update * (eta * gt))?)?;
        }

        let output = Tensor::cat(&outputs, 0)?;

        Ok((output, current_m))
    }
}
