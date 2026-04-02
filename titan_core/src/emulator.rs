use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear};

/// Python Emulator Module.
///
/// This module models an internal program state (like a register or memory block)
/// that gets updated based on a sequence of inputs (instructions).
pub struct PythonEmulator {
    up_proj: Linear,
    down_proj: Linear,
    gate_proj: Linear,
}

impl PythonEmulator {
    /// Creates a new `PythonEmulator` instance.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let up_proj = linear(dim, dim * 2, vb.pp("up_proj"))?;
        let down_proj = linear(dim * 2, dim, vb.pp("down_proj"))?;
        let gate_proj = linear(dim, dim, vb.pp("gate_proj"))?;
        Ok(Self {
            up_proj,
            down_proj,
            gate_proj,
        })
    }

    /// Updates the persistent internal program state with a new sequence of instructions.
    ///
    /// # Arguments
    /// * `instruction_rep` - Tensor of shape `[T, D]` representing the instructions.
    /// * `current_state` - Tensor of shape `[1, D]` representing the current state.
    ///
    /// # Returns
    /// The updated state as a tensor of shape `[1, D]`.
    pub fn step(&self, instruction_rep: &Tensor, current_state: &Tensor) -> Result<Tensor> {
        // Use MLP to process instructions: Linear -> SiLU -> Linear
        let h = instruction_rep.apply(&self.up_proj)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = h.apply(&self.down_proj)?;

        // Aggregate instructions from the whole sequence [1, D]
        let instruction_agg = h.mean_keepdim(0)?;

        // Compute a gating mechanism to determine how much of the old state to keep
        let gate = candle_nn::ops::sigmoid(&current_state.apply(&self.gate_proj)?)?;
        // Gated update: state = (1 - gate) * old_state + gate * instruction_agg
        let new_state = ((current_state * (1.0 - &gate)?)? + (instruction_agg * &gate)?)?;

        // Normalize state using tanh
        new_state.tanh()
    }
}
