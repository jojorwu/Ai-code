use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear, ops};

/// Titans Long-Term Memory (Neural Memory)
/// Implements a persistent memory module that uses a matrix state M for associative memory.
pub struct TitansMemory {
    key_proj: Linear,
    val_proj: Linear,
    gate_proj: Linear,
    #[allow(dead_code)]
    dim: usize,
}

impl TitansMemory {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let key_proj = linear(dim, dim, vb.pp("key_proj"))?;
        let val_proj = linear(dim, dim, vb.pp("val_proj"))?;
        let gate_proj = linear(dim, dim, vb.pp("gate_proj"))?;
        Ok(Self { key_proj, val_proj, gate_proj, dim })
    }

    pub fn forward(&self, x: &Tensor, memory_matrix: &Tensor) -> Result<(Tensor, Tensor)> {
        // x: [T, D], memory_matrix: [D, D]
        let keys = x.apply(&self.key_proj)?; // [T, D]
        let vals = x.apply(&self.val_proj)?; // [T, D]

        // Compute gated updates for the matrix
        let gate = ops::sigmoid(&x.apply(&self.gate_proj)?)?; // [T, D]
        let gate_avg = gate.mean_all()?.to_vec0::<f32>()?;

        // Associative update: ΔM = sum(keys^T * vals)
        // keys.t() is [D, T], vals is [T, D] -> update is [D, D]
        let update = keys.t()?.matmul(&vals)?;

        // Update rule: M = (1 - η) * M + η * ΔM
        let updated_memory = ((memory_matrix * ((1.0 - gate_avg) as f64))? + (&update * (gate_avg as f64))?)?;

        // Retrieve from memory: y = keys * M
        // keys is [T, D], M is [D, D] -> output is [T, D]
        let output = keys.matmul(&updated_memory)?;

        Ok((output, updated_memory))
    }
}
