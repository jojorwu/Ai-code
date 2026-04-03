use candle_core::{D, Result, Tensor};
use candle_nn::{linear, rms_norm, Linear, RmsNorm, VarBuilder};
use crate::rope::RotaryEmbedding;

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
}

impl MultiHeadAttention {
    pub fn new(dim: usize, num_heads: usize, num_kv_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q_proj = linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = linear(dim, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(dim, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;

        let q_norm = rms_norm(head_dim, 1e-5, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, 1e-5, vb.pp("k_norm"))?;

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
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        kv_cache: Option<(Tensor, Tensor)>
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (t_size, _d_size) = x.dims2()?;

        let q = x.apply(&self.q_proj)?;
        let k = x.apply(&self.k_proj)?;
        let v = x.apply(&self.v_proj)?;

        // Reshape for GQA: q: [H, T, Hd], k/v: [Hkv, T, Hd]
        let q = q.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?;
        let k = k.reshape((t_size, self.num_kv_heads, self.head_dim))?.transpose(0, 1)?;
        let mut v = v.reshape((t_size, self.num_kv_heads, self.head_dim))?.transpose(0, 1)?;

        // Apply QK Norm (improved training stability)
        let q = q.apply(&self.q_norm)?;
        let mut k = k.apply(&self.k_norm)?;

        // Apply RoPE
        let q = rope.apply(&q)?;
        let k_rope = rope.apply(&k)?;

        // Update KV cache if provided
        if let Some((prev_k, prev_v)) = kv_cache {
            k = Tensor::cat(&[prev_k, k_rope], 1)?;
            v = Tensor::cat(&[prev_v, v], 1)?;
        } else {
            k = k_rope;
        }

        let current_kv = (k.clone(), v.clone());

        // Repeat KV heads to match Q heads if needed
        let k_rep = if self.num_heads != self.num_kv_heads {
            self.repeat_heads(&k)?
        } else {
            k
        };
        let v_rep = if self.num_heads != self.num_kv_heads {
            self.repeat_heads(&v)?
        } else {
            v
        };

        // scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k_rep.transpose(D::Minus1, D::Minus2)?)? / scale)?;

        // causal mask (only if sequence length > 1)
        let seq_len = q.dim(1)?;
        let kv_len = k_rep.dim(1)?;
        if seq_len > 1 {
             let mask = self.get_causal_mask(seq_len, kv_len, x.device())?;
             let scores = scores.broadcast_add(&mask)?;
             let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
             let context = attn.matmul(&v_rep)?;
             let context = context.transpose(0, 1)?.reshape((t_size, self.num_heads * self.head_dim))?;
             Ok((context.apply(&self.out_proj)?, current_kv))
        } else {
             let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
             let context = attn.matmul(&v_rep)?;
             let context = context.transpose(0, 1)?.reshape((t_size, self.num_heads * self.head_dim))?;
             Ok((context.apply(&self.out_proj)?, current_kv))
        }
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
                let i_abs = i + kv_len - q_len;
                (0..kv_len).map(move |j| if j > i_abs { f32::NEG_INFINITY } else { 0f32 })
            })
            .collect();
        Tensor::from_slice(&mask, (q_len, kv_len), device)
    }
}
