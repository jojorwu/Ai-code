use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear};

/// Python Emulator Module
/// Models internal program state (e.g. registers, variables) update.
pub struct PythonEmulator {
    up_proj: Linear,
    down_proj: Linear,
    gate_proj: Linear,
}

impl PythonEmulator {
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

    pub fn step(&self, instruction_rep: &Tensor, current_state: &Tensor) -> Result<Tensor> {
        // instruction_rep is [T, D]
        // Use MLP to process instructions: Linear -> Silu -> Linear
        let h = instruction_rep.apply(&self.up_proj)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = h.apply(&self.down_proj)?;

        // Aggregate instructions [1, D]
        let instruction_agg = h.mean_keepdim(0)?;

        // Gated update for the persistent state
        let gate = candle_nn::ops::sigmoid(&current_state.apply(&self.gate_proj)?)?;
        let new_state = ((current_state * (1.0 - &gate)?)? + (instruction_agg * &gate)?)?;

        new_state.tanh()
    }
}
