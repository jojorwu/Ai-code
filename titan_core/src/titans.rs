use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear};

/// Titans Long-Term Memory (Neural Memory)
/// Learns to memorize historical context using a fast-update associative memory concept.
pub struct TitansMemory {
    key_proj: Linear,
    val_proj: Linear,
}

impl TitansMemory {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let key_proj = linear(dim, dim, vb.pp("key_proj"))?;
        let val_proj = linear(dim, dim, vb.pp("val_proj"))?;
        Ok(Self { key_proj, val_proj })
    }

    pub fn forward(&self, x: &Tensor, memory_state: &Tensor) -> Result<(Tensor, Tensor)> {
        // Fast-update associative memory logic:
        // memory_state = memory_state + key * val^T (conceptually)
        // Here, we simplify it by an additive update based on projections.
        let k = x.apply(&self.key_proj)?;
        let v = x.apply(&self.val_proj)?;

        let update = k.broadcast_mul(&v)?;
        let updated_memory = memory_state.broadcast_add(&update)?;
        let output = updated_memory.clone();
        Ok((output, updated_memory))
    }
}
