use candle_core::{D, Device, Result, Tensor};

pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(candle_core::DType::F32)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        // Broadcast cos/sin to [max_seq_len, dim]
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;

        Ok(Self { sin, cos })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        // x: [..., T, D]
        let dims = x.dims();
        let t_size = dims[dims.len() - 2];
        let d_size = dims[dims.len() - 1];

        // Narrow RoPE to match input sequence length and dimension
        let cos = self.cos.narrow(0, 0, t_size)?.narrow(1, 0, d_size)?;
        let sin = self.sin.narrow(0, 0, t_size)?.narrow(1, 0, d_size)?;

        // x: [..., T, D], cos/sin: [T, D]
        // We need to ensure cos/sin are broadcastable to x's shape.
        // For [H, T, D], we need [1, T, D].
        let mut cos = cos;
        let mut sin = sin;
        for _ in 0..(dims.len() - 2) {
            cos = cos.unsqueeze(0)?;
            sin = sin.unsqueeze(0)?;
        }

        let x1 = x.narrow(D::Minus1, 0, d_size / 2)?;
        let x2 = x.narrow(D::Minus1, d_size / 2, d_size / 2)?;

        let rotated_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        let out = (x.broadcast_mul(&cos)? + rotated_x.broadcast_mul(&sin)?)?;
        Ok(out)
    }
}
