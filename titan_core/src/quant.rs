use candle_core::{Tensor, Result, D};

/// PolarQuant separates the magnitude and angle.
pub struct PolarQuant;

impl PolarQuant {
    pub fn compress(x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Compute the norm (radius) and normalized direction (angle-like)
        let radius = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        // Add epsilon to avoid division by zero
        let eps = Tensor::new(&[1e-8f32], x.device())?.broadcast_as(radius.shape())?;
        let radius_safe = radius.add(&eps)?;
        let direction = x.broadcast_div(&radius_safe)?;
        Ok((radius, direction))
    }

    pub fn decompress(radius: &Tensor, direction: &Tensor) -> Result<Tensor> {
        radius.broadcast_mul(direction)
    }
}

/// Quantized Johnson-Lindenstrauss (QJL)
pub struct QJL;

impl QJL {
    pub fn compress(x: &Tensor) -> Result<Tensor> {
        // One-bit quantization: sign(x)
        x.sign()
    }
}
