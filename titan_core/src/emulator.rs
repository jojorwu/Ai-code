use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, conv1d, Conv1d, Conv1dConfig, Linear};

/// Python Emulator Module.
///
/// This module models an internal program state (like a register or memory block)
/// that gets updated based on a sequence of inputs (instructions).
pub struct PythonEmulator {
    up_proj: Linear,
    down_proj: Linear,
    conv: Conv1d,
    // GRU Gates
    update_gate: Linear,
    reset_gate: Linear,
    candidate_gate: Linear,
}

impl PythonEmulator {
    /// Creates a new `PythonEmulator` instance with a bottleneck GRU state update.
    ///
    /// The bottleneck design (dim -> dim/4 -> dim) ensures that the emulator
    /// extracts only the most salient "instruction" features from the hidden state,
    /// preventing the program state from being saturated by noise.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Bottleneck: compress information to captue higher-level instruction semantics
        let bottleneck_dim = dim / 4;
        let up_proj = linear(dim, bottleneck_dim, vb.pp("up_proj"))?;
        let down_proj = linear(bottleneck_dim, dim, vb.pp("down_proj"))?;

        let conv_cfg = Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: bottleneck_dim,
        };
        let conv = conv1d(bottleneck_dim, bottleneck_dim, 3, conv_cfg, vb.pp("conv"))?;

        let update_gate = linear(dim, dim, vb.pp("update_gate"))?;
        let reset_gate = linear(dim, dim, vb.pp("reset_gate"))?;
        let candidate_gate = linear(dim, dim, vb.pp("candidate_gate"))?;

        Ok(Self {
            up_proj,
            down_proj,
            conv,
            update_gate,
            reset_gate,
            candidate_gate,
        })
    }

    /// Updates the persistent internal program state using a Gated Recurrent Unit (GRU) mechanism.
    ///
    /// The state update follows:
    /// 1. Instruction extraction via bottleneck MLP.
    /// 2. Mean aggregation over the sequence dimension (T).
    /// 3. GRU state transition:
    ///    $$ z_t = \sigma(W_z x_t + U_z h_{t-1}) $$
    ///    $$ r_t = \sigma(W_r x_t + U_r h_{t-1}) $$
    ///    $$ \tilde{h}_t = \tanh(W_h x_t + r_t \odot (U_h h_{t-1})) $$
    ///    $$ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t $$
    pub fn step(&self, instruction_rep: &Tensor, current_state: &Tensor) -> Result<Tensor> {
        // instruction_rep: [T, D]
        // Use Bottleneck + Depthwise Conv to process instructions
        let h = instruction_rep.apply(&self.up_proj)?; // [T, Bd]

        // Local Context: Conv1D over sequence [1, Bd, T]
        let h_conv = h.t()?.unsqueeze(0)?;
        let h_conv = h_conv.apply(&self.conv)?;
        let h = h_conv.squeeze(0)?.t()?; // [T, Bd]

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
