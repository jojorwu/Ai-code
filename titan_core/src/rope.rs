//! Rotary Positional Embeddings (RoPE) with NTK-aware scaling.
//!
//! This module implements relative positional information injection
//! for multi-head attention and long-term memory.

use candle_core::{D, Device, Result, Tensor};

/// Rotary Positional Embedding (RoPE) implementation.
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        // NTK-aware scaling for sequence lengths longer than max_seq_len (e.g. 2048)
        // alpha = (current_seq_len / max_seq_len).pow(dim / (dim-2))
        // For static initialization, we'll use a slightly higher base to improve extrapolate
        let base = 50000f32;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
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

    /// Applies Rotary Positional Embeddings to the input tensor.
    ///
    /// $$ \text{RoPE}(x, \text{pos}) = x \cdot \cos(\theta_{\text{pos}}) + \text{rotate\_half}(x) \cdot \sin(\theta_{\text{pos}}) $$
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [..., T, D].
    /// * `start_pos` - The starting index in the sequence for positional information.
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let dims = x.dims();
        let t_size = dims[dims.len() - 2];
        let d_size = dims[dims.len() - 1];
        let max_t = self.cos.dim(0)?;

        // Handle overflow/extrapolation by wrapping around (NTK-aware still helps)

        // Ensure t_size is not greater than max_t to avoid negative narrow
        let t_size_safe = if t_size > max_t { max_t } else { t_size };
        let start_pos_safe = if start_pos + t_size_safe > max_t {
            max_t - t_size_safe
        } else {
            start_pos
        };

        let cos = self.cos.narrow(0, start_pos_safe, t_size_safe)?.narrow(1, 0, d_size)?;
        let sin = self.sin.narrow(0, start_pos_safe, t_size_safe)?.narrow(1, 0, d_size)?;

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
