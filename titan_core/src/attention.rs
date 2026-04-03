use candle_core::{D, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use crate::rope::RotaryEmbedding;

pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
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

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let (t_size, _d_size) = x.dims2()?;

        let q = x.apply(&self.q_proj)?;
        let k = x.apply(&self.k_proj)?;
        let v = x.apply(&self.v_proj)?;

        // Reshape for GQA: q: [H, T, Hd], k/v: [Hkv, T, Hd]
        let q = q.reshape((t_size, self.num_heads, self.head_dim))?.transpose(0, 1)?;
        let k = k.reshape((t_size, self.num_kv_heads, self.head_dim))?.transpose(0, 1)?;
        let v = v.reshape((t_size, self.num_kv_heads, self.head_dim))?.transpose(0, 1)?;

        // Apply RoPE
        let q = rope.apply(&q)?;
        let k = rope.apply(&k)?;

        // Repeat KV heads to match Q heads if needed
        let k = if self.num_heads != self.num_kv_heads {
            self.repeat_heads(&k)?
        } else {
            k
        };
        let v = if self.num_heads != self.num_kv_heads {
            self.repeat_heads(&v)?
        } else {
            v
        };

        // scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? / scale)?;

        // causal mask
        let mask = self.get_causal_mask(t_size, x.device())?;
        let scores = scores.broadcast_add(&mask)?;

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let context = attn.matmul(&v)?; // [H, T, Hd]

        // Reshape back
        let context = context.transpose(0, 1)?.reshape((t_size, self.num_heads * self.head_dim))?;
        context.apply(&self.out_proj)
    }

    fn repeat_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (num_kv_heads, t_size, head_dim) = x.dims3()?;
        let ratio = self.num_heads / num_kv_heads;
        x.unsqueeze(1)?
            .expand((num_kv_heads, ratio, t_size, head_dim))?
            .reshape((self.num_heads, t_size, head_dim))
    }

    fn get_causal_mask(&self, t: usize, device: &candle_core::Device) -> Result<Tensor> {
        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        Tensor::from_slice(&mask, (t, t), device)
    }
}
