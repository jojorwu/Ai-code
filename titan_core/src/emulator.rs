use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear};

/// Python Emulator Module.
///
/// This module models an internal program state (like a register or memory block)
/// that gets updated based on a sequence of inputs (instructions).
pub struct PythonEmulator {
    up_proj: Linear,
    down_proj: Linear,
    // GRU Gates
    update_gate: Linear,
    reset_gate: Linear,
    candidate_gate: Linear,
}

impl PythonEmulator {
    /// Creates a new `PythonEmulator` instance with a bottleneck GRU state update.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Bottleneck: compress information to captue higher-level instruction semantics
        let up_proj = linear(dim, dim / 4, vb.pp("up_proj"))?;
        let down_proj = linear(dim / 4, dim, vb.pp("down_proj"))?;

        let update_gate = linear(dim, dim, vb.pp("update_gate"))?;
        let reset_gate = linear(dim, dim, vb.pp("reset_gate"))?;
        let candidate_gate = linear(dim, dim, vb.pp("candidate_gate"))?;

        Ok(Self {
            up_proj,
            down_proj,
            update_gate,
            reset_gate,
            candidate_gate,
        })
    }

    /// Updates the persistent internal program state using a GRU mechanism.
    pub fn step(&self, instruction_rep: &Tensor, current_state: &Tensor) -> Result<Tensor> {
        // instruction_rep: [T, D]
        // Use MLP to process instructions
        let h = instruction_rep.apply(&self.up_proj)?;
        let h = candle_nn::ops::silu(&h)?;
        let x = h.apply(&self.down_proj)?; // [T, D]

        // Aggregate instructions [1, D]
        let x_agg = x.mean_keepdim(0)?;

        // GRU logic for state h
        // z = sigmoid(Wz * x + Uz * h)
        // r = sigmoid(Wr * x + Ur * h)
        // n = tanh(Wn * x + r * (Un * h))
        // h_new = (1 - z) * h + z * n

        let z = candle_nn::ops::sigmoid(&x_agg.apply(&self.update_gate)?)?;
        let r = candle_nn::ops::sigmoid(&x_agg.apply(&self.reset_gate)?)?;

        let gated_state = current_state.broadcast_mul(&r)?;
        let candidate = (x_agg.apply(&self.candidate_gate)? + gated_state)?.tanh()?;

        let one_minus_z = z.neg()?.affine(1.0, 1.0)?;
        let new_state = (current_state.broadcast_mul(&one_minus_z)? + candidate.broadcast_mul(&z)?)?;

        Ok(new_state)
    }
}
