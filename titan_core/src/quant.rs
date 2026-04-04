//! Quantization utilities for the Titan model.
//!
//! Includes PolarQuant for magnitude/direction separation and QJL for 1-bit quantization.

use candle_core::{Tensor, Result, D};

/// PolarQuant separates magnitude (radius) and normalized direction.
pub struct PolarQuant;

impl PolarQuant {
    /// Compresses a tensor into its magnitude (radius) and normalized direction.
    ///
    /// $$ r = \|x\|_2, \quad \hat{x} = \frac{x}{r + \epsilon} $$
    /// Uses an epsilon to avoid division by zero when calculating the direction.
    pub fn compress(x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Compute the norm (radius) and normalized direction (angle-like)
        let radius = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        // Add epsilon to avoid division by zero
        let radius_safe = radius.affine(1.0, 1e-8)?;
        let direction = x.broadcast_div(&radius_safe)?;
        Ok((radius, direction))
    }

    /// Decompresses the magnitude and direction back into the original representation.
    pub fn decompress(radius: &Tensor, direction: &Tensor) -> Result<Tensor> {
        radius.broadcast_mul(direction)
    }
}

/// Quantized Johnson-Lindenstrauss (QJL).
/// Implements 1-bit quantization.
pub struct QJL;

impl QJL {
    /// Performs 1-bit quantization by taking the sign of each element.
    pub fn compress(x: &Tensor) -> Result<Tensor> {
        // One-bit quantization: sign(x)
        x.sign()
    }

    /// Combines PolarQuant and QJL approaches.
    /// It first calculates Polar magnitude and then applies 1-bit quantization to the direction.
    pub fn compress_pq_qjl(x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (radius, direction) = PolarQuant::compress(x)?;
        let quantized_direction = Self::compress(&direction)?;
        Ok((radius, quantized_direction))
    }
}
