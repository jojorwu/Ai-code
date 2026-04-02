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
        Ok(Self {
            key_proj,
            val_proj,
            gate_proj,
            dim,
        })
    }

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

        for t in 0..t_size {
            let kt = keys.get(t)?.reshape((1, d_size))?;
            let vt = vals.get(t)?.reshape((1, d_size))?;
            let gt = gate.get(t)?.mean_all()?.to_vec0::<f32>()? as f64;

            // Retrieve: y_t = kt * M_{t-1}
            let yt = kt.matmul(&current_m)?;
            outputs.push(yt.clone());

            // Delta update: ΔM = (vt - yt) ⊗ kt
            // vt - yt: [1, D], kt: [1, D] -> kt.t() @ (vt - yt): [D, D]
            let diff = (vt - yt)?;
            let update = kt.t()?.matmul(&diff)?;

            // M_t = (1 - decay) * M_{t-1} + eta * surprise_gate * ΔM
            current_m = ((current_m * (1.0 - decay))? + (update * (eta * gt))?)?;
        }

        let output = Tensor::cat(&outputs, 0)?;

        Ok((output, current_m))
    }
}
