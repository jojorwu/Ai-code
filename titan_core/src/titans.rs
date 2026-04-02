use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear, ops};

/// Titans Long-Term Memory (Neural Memory)
/// Implements a persistent memory module that uses a "surprise" gated update.
pub struct TitansMemory {
    key_proj: Linear,
    val_proj: Linear,
    gate_proj: Linear,
}

impl TitansMemory {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let key_proj = linear(dim, dim, vb.pp("key_proj"))?;
        let val_proj = linear(dim, dim, vb.pp("val_proj"))?;
        let gate_proj = linear(dim, dim, vb.pp("gate_proj"))?;
        Ok(Self { key_proj, val_proj, gate_proj })
    }

    pub fn forward(&self, x: &Tensor, memory_state: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = x.apply(&self.key_proj)?;
        let v = x.apply(&self.val_proj)?;

        // k, v are [T, D]
        // Compute "surprise" gate: how much should we update the memory?
        let gate = ops::sigmoid(&x.apply(&self.gate_proj)?)?; // [T, D]

        // Conceptual associative update for each token in sequence
        // For simplicity in this demo, we aggregate the updates across the sequence T
        // update_seq = gate * (k * v)
        let update_seq = (k.broadcast_mul(&v)?).broadcast_mul(&gate)?;

        // Sum or average over T to get a single update for the persistent state [1, D]
        let update = update_seq.sum_keepdim(0)?; // [1, D]

        // Similarly for gate to decay old memory
        let gate_avg = gate.mean_keepdim(0)?; // [1, D]

        let term1 = memory_state.broadcast_mul(&(gate_avg.ones_like()? - &gate_avg)?)?;
        let term2 = update;

        let updated_memory = (term1 + term2)?;
        let output = updated_memory.clone();

        Ok((output, updated_memory))
    }
}
