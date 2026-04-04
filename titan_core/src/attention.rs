//! Multi-Head Attention implementation with Grouped-Query Attention (GQA) and Sliding Window.
//!
//! This module implements the attention mechanism used in Titan, featuring:
//! - GQA for efficient inference.
//! - Sliding Window Attention (SWA) for linear complexity relative to context window.
//! - KV-Cache with simulated 8-bit quantization.
//! - Query-Key Normalization (QK-Norm) for training stability.

use candle_core::{D, Result, Tensor};
use candle_nn::{linear, rms_norm, Linear, RmsNorm, VarBuilder, Init};
use crate::rope::RotaryEmbedding;

/// Multi-Head Attention block.
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
    is_global: bool,
    use_differential: bool,
    lambda_1: Option<Tensor>,
    lambda_2: Option<Tensor>,
    // MLA (Multi-Head Latent Attention)
    use_mla: bool,
    kv_down_proj: Option<Linear>,
    kv_up_proj: Option<Linear>,
    q_down_proj: Option<Linear>,
    q_up_proj: Option<Linear>,
    kv_norm: Option<RmsNorm>,
    q_norm_mla: Option<RmsNorm>,
}

impl MultiHeadAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        window_size: usize,
        is_global: bool,
        use_differential: bool,
        use_mla: bool,
        kv_lora_rank: usize,
        qk_lora_rank: usize,
        vb: VarBuilder
    ) -> Result<Self> {
        let head_dim = dim / num_heads;

        let (q_proj, k_proj, v_proj, kv_down_proj, kv_up_proj, q_down_proj, q_up_proj, kv_norm, q_norm_mla) = if use_mla {
             let q_down = linear(dim, qk_lora_rank, vb.pp("q_down_proj"))?;
             let q_up = linear(qk_lora_rank, dim, vb.pp("q_up_proj"))?;
             let kv_down = linear(dim, kv_lora_rank, vb.pp("kv_down_proj"))?;
             let kv_up = linear(kv_lora_rank, num_kv_heads * head_dim * 2, vb.pp("kv_up_proj"))?;
             let kv_norm = rms_norm(kv_lora_rank, 1e-5, vb.pp("kv_norm"))?;
             let q_norm_mla = rms_norm(qk_lora_rank, 1e-5, vb.pp("q_norm_mla"))?;

             // In MLA, we don't use direct QKV projs usually, but we keep structure
             let dummy_q = linear(dim, dim, vb.pp("q_proj"))?;
             let dummy_k = linear(dim, num_kv_heads * head_dim, vb.pp("k_proj"))?;
             let dummy_v = linear(dim, num_kv_heads * head_dim, vb.pp("v_proj"))?;

             (dummy_q, dummy_k, dummy_v, Some(kv_down), Some(kv_up), Some(q_down), Some(q_up), Some(kv_norm), Some(q_norm_mla))
        } else {
             let q_proj = linear(dim, dim, vb.pp("q_proj"))?;
             let k_proj = linear(dim, num_kv_heads * head_dim, vb.pp("k_proj"))?;
             let v_proj = linear(dim, num_kv_heads * head_dim, vb.pp("v_proj"))?;
             (q_proj, k_proj, v_proj, None, None, None, None, None, None)
        };

        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;

        let (q_norm, k_norm, lambda_1, lambda_2) = if use_differential {
             let d_d = head_dim / 2;
             let q_norm = rms_norm(d_d, 1e-5, vb.pp("q_norm"))?;
             let k_norm = rms_norm(d_d, 1e-5, vb.pp("k_norm"))?;
             let l1 = vb.get_with_hints((num_heads, 1, 1), "lambda_1", Init::Const(0.0))?;
             let l2 = vb.get_with_hints((num_heads, 1, 1), "lambda_2", Init::Const(0.0))?;
             (q_norm, k_norm, Some(l1), Some(l2))
        } else {
             let q_norm = rms_norm(head_dim, 1e-5, vb.pp("q_norm"))?;
             let k_norm = rms_norm(head_dim, 1e-5, vb.pp("k_norm"))?;
             (q_norm, k_norm, None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            window_size,
            is_global,
            use_differential,
            lambda_1,
            lambda_2,
            use_mla,
            kv_down_proj,
            kv_up_proj,
            q_down_proj,
            q_up_proj,
            kv_norm,
            q_norm_mla,
        })
    }

    /// Performs the forward pass with Grouped-Query Attention, RoPE, and Sliding Window.
    ///
    /// Implementation details:
    /// - **QK-Norm**: Stabilizes scores by applying RMSNorm to Query and Key heads.
    /// - **RoPE**: Injects relative positional information using NTK-aware scaling.
    /// - **KV-Cache**: Implements a sliding window of size $W$ by truncating history.
    /// - **Complexity**: Reduces attention complexity to $O(N \cdot W)$.
    /// - **Differential Attention**: If enabled, calculates two softmax maps and subtracts them.
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        kv_cache: Option<(Tensor, Tensor)>,
        start_pos: usize,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (t_size, _d_size) = x.dims2()?;

        let mut q: Tensor;
        let mut k: Tensor;
        let mut v: Tensor;

        if self.use_mla {
             // 1. Latent compression
             let kv_latent = x.apply(self.kv_down_proj.as_ref().unwrap())?.apply(self.kv_norm.as_ref().unwrap())?;
             let kv_full = kv_latent.apply(self.kv_up_proj.as_ref().unwrap())?; // [T, Hkv * Hd * 2]

             let split_size = self.num_kv_heads * self.head_dim;
             k = kv_full.narrow(D::Minus1, 0, split_size)?;
             v = kv_full.narrow(D::Minus1, split_size, split_size)?;

             let q_latent = x.apply(self.q_down_proj.as_ref().unwrap())?.apply(self.q_norm_mla.as_ref().unwrap())?;
             q = q_latent.apply(self.q_up_proj.as_ref().unwrap())?;
        } else {
             q = x.apply(&self.q_proj)?;
             k = x.apply(&self.k_proj)?;
             v = x.apply(&self.v_proj)?;
        }

        // Reshape for GQA: q: [H, T, Hd], k/v: [Hkv, T, Hd]
        q = q.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?;
        k = k.reshape((t_size, self.num_kv_heads, self.head_dim))?.transpose(0, 1)?;
        v = v.reshape((t_size, self.num_kv_heads, self.head_dim))?.transpose(0, 1)?;

        let d_d = if self.use_differential { self.head_dim / 2 } else { self.head_dim };

        // Apply QK Norm
        if self.use_differential {
             let (q1, q2) = (q.narrow(D::Minus1, 0, d_d)?, q.narrow(D::Minus1, d_d, d_d)?);
             let (k1, k2) = (k.narrow(D::Minus1, 0, d_d)?, k.narrow(D::Minus1, d_d, d_d)?);
             q = Tensor::cat(&[q1.apply(&self.q_norm)?, q2.apply(&self.q_norm)?], D::Minus1)?;
             k = Tensor::cat(&[k1.apply(&self.k_norm)?, k2.apply(&self.k_norm)?], D::Minus1)?;
        } else {
             q = q.apply(&self.q_norm)?;
             k = k.apply(&self.k_norm)?;
        }

        // Apply RoPE
        q = rope.apply(&q, start_pos)?;
        let k_rope = rope.apply(&k, start_pos)?;

        // Update KV cache
        if let Some((prev_k, prev_v)) = kv_cache {
            let prev_k = (prev_k.affine(127.0, 0.0)?.round()? / 127.0)?;
            let prev_v = (prev_v.affine(127.0, 0.0)?.round()? / 127.0)?;

            k = Tensor::cat(&[prev_k, k_rope], 1)?;
            v = Tensor::cat(&[prev_v, v], 1)?;

            let cur_kv_len = k.dim(1)?;
            if !self.is_global && cur_kv_len > self.window_size {
                 k = k.narrow(1, cur_kv_len - self.window_size, self.window_size)?;
                 v = v.narrow(1, cur_kv_len - self.window_size, self.window_size)?;
            }
        } else {
            k = k_rope;
        }

        let current_kv = (k.clone(), v.clone());

        // Repeat KV heads
        let k_rep = if self.num_heads != self.num_kv_heads { self.repeat_heads(&k)? } else { k };
        let v_rep = if self.num_heads != self.num_kv_heads { self.repeat_heads(&v)? } else { v };

        let seq_len = q.dim(1)?;
        let kv_len = k_rep.dim(1)?;
        let mask = if seq_len > 1 || kv_len > 1 {
             Some(self.get_causal_mask(seq_len, kv_len, x.device())?)
        } else {
             None
        };

        let context = if self.use_differential {
             let q1 = q.narrow(D::Minus1, 0, d_d)?;
             let q2 = q.narrow(D::Minus1, d_d, d_d)?;
             let k1 = k_rep.narrow(D::Minus1, 0, d_d)?;
             let k2 = k_rep.narrow(D::Minus1, d_d, d_d)?;

             let scale = (d_d as f64).sqrt();
             let mut s1 = (q1.matmul(&k1.transpose(D::Minus1, D::Minus2)?)? / scale)?;
             let mut s2 = (q2.matmul(&k2.transpose(D::Minus1, D::Minus2)?)? / scale)?;

             if let Some(m) = &mask {
                 s1 = s1.broadcast_add(m)?;
                 s2 = s2.broadcast_add(m)?;
             }

             let attn1 = candle_nn::ops::softmax(&s1, D::Minus1)?;
             let attn2 = candle_nn::ops::softmax(&s2, D::Minus1)?;

             let l1 = self.lambda_1.as_ref().unwrap();
             let l2 = self.lambda_2.as_ref().unwrap();
             let lambda = (l1.exp()?.broadcast_sub(&l2.exp()?)? + 0.8)?;

             let attn = attn1.broadcast_sub(&(attn2.broadcast_mul(&lambda)?))?;
             attn.matmul(&v_rep)?
        } else {
             let scale = (self.head_dim as f64).sqrt();
             let mut scores = (q.matmul(&k_rep.transpose(D::Minus1, D::Minus2)?)? / scale)?;
             if let Some(m) = &mask {
                 scores = scores.broadcast_add(m)?;
             }
             let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
             attn.matmul(&v_rep)?
        };

        let context = context.transpose(0, 1)?.reshape((t_size, self.num_heads * self.head_dim))?;
        Ok((context.apply(&self.out_proj)?, current_kv))
    }

    fn repeat_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (num_kv_heads, t_size, head_dim) = x.dims3()?;
        let ratio = self.num_heads / num_kv_heads;
        x.unsqueeze(1)?
            .expand((num_kv_heads, ratio, t_size, head_dim))?
            .reshape((self.num_heads, t_size, head_dim))
    }

    fn get_causal_mask(&self, q_len: usize, kv_len: usize, device: &candle_core::Device) -> Result<Tensor> {
        let mask: Vec<_> = (0..q_len)
            .flat_map(|i| {
                // Safely calculate i_abs to avoid subtraction overflow
                let i_abs = (i as i64 + kv_len as i64 - q_len as i64) as usize;
                (0..kv_len).map(move |j| {
                    // Causal mask: j > i_abs
                    // Sliding window: j < i_abs - window_size (only if not global)
                    let is_causal = j > i_abs;
                    let is_outside_window = !self.is_global && i_abs >= self.window_size && j < i_abs - self.window_size;

                    if is_causal || is_outside_window {
                         f32::NEG_INFINITY
                    } else {
                        0f32
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (q_len, kv_len), device)
    }
}
