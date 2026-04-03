use candle_core::{Tensor, Result};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use crate::attention::MultiHeadAttention;
use crate::attn_res::FullAttnRes;
use crate::titans::TitansMemory;
use crate::emulator::PythonEmulator;
use crate::quant::PolarQuant;
use crate::rope::RotaryEmbedding;

/// Main Titan Transformer model.
pub struct TitanTransformer {
    embedding: Embedding,
    layers: Vec<TitanLayer>,
    norm_final: RmsNorm,
    output: Linear,
    rope: RotaryEmbedding,
}

/// Feed-Forward Network (FFN) block using SwiGLU activation.
pub struct FFN {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FFN {
    /// Creates a new FFN block with SwiGLU.
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (dim * 8) / 3; // Standard for SwiGLU to keep params similar to 4x FFN
        let gate_proj = linear(dim, hidden_dim, vb.pp("gate_proj"))?;
        let up_proj = linear(dim, hidden_dim, vb.pp("up_proj"))?;
        let down_proj = linear(hidden_dim, dim, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Performs the forward pass of the FFN block.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&x.apply(&self.gate_proj)?)?;
        let up = x.apply(&self.up_proj)?;
        let h = (gate * up)?;
        h.apply(&self.down_proj)
    }
}

/// A single layer of the Titan Transformer.
pub struct TitanLayer {
    pub attention: MultiHeadAttention,
    pub memory: TitansMemory,
    pub emulator: PythonEmulator,
    pub attn_res: FullAttnRes,
    pub ffn: FFN,
    pub norm_1: RmsNorm,
    pub norm_2: RmsNorm,
}

impl TitanTransformer {
    /// Creates a new Titan Transformer model.
    pub fn new(vocab_size: usize, dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(vocab_size, dim, vb.pp("embedding"))?;
        let rope = RotaryEmbedding::new(dim, 2048, vb.device())?;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let vb_layer = vb.pp(format!("layer_{}", i));
            layers.push(TitanLayer {
                attention: MultiHeadAttention::new(dim, 8, 2, vb_layer.pp("attention"))?, // GQA with 2 KV heads
                memory: TitansMemory::new(dim, vb_layer.pp("memory"))?,
                emulator: PythonEmulator::new(dim, vb_layer.pp("emulator"))?,
                attn_res: FullAttnRes::new(dim, vb_layer.pp("attn_res"))?,
                ffn: FFN::new(dim, vb_layer.pp("ffn"))?,
                norm_1: rms_norm(dim, 1e-5, vb_layer.pp("norm_1"))?,
                norm_2: rms_norm(dim, 1e-5, vb_layer.pp("norm_2"))?,
            });
        }
        let norm_final = rms_norm(dim, 1e-5, vb.pp("norm_final"))?;
        let output = linear(dim, vocab_size, vb.pp("output"))?;
        Ok(Self {
            embedding,
            layers,
            norm_final,
            output,
            rope,
        })
    }

    /// Performs the forward pass of the model.
    pub fn forward(
        &self,
        x: &Tensor,
        memory_states: &mut [Tensor],
        program_states: &mut [Tensor],
    ) -> Result<Tensor> {
        let mut h = x.apply(&self.embedding)?;
        let mut layer_outputs = Vec::with_capacity(self.layers.len() + 1);
        layer_outputs.push(h.clone());

        for (i, layer) in self.layers.iter().enumerate() {
            // Parallel Block Architecture:
            // h = h + SelfAttn(Norm(h)) + Memory(Norm(h)) + Emulator(Norm(h))
            let h_norm = h.apply(&layer.norm_1)?;

            // 1. Self-Attention
            let attn_out = layer.attention.forward(&h_norm, &self.rope)?;

            // 2. Neural Memory
            let (radius, direction) = PolarQuant::compress(&h_norm)?;
            let h_quantized = PolarQuant::decompress(&radius, &direction)?;
            let (mem_out, new_mem) = layer.memory.forward(&h_quantized, &memory_states[i], &self.rope)?;
            memory_states[i] = new_mem;

            // 3. Program Emulator
            let new_prog = layer.emulator.step(&h_quantized, &program_states[i])?;
            program_states[i] = new_prog.clone();

            // Aggregate parallel components
            let mut parallel_out = (attn_out + mem_out)?;
            parallel_out = parallel_out.broadcast_add(&new_prog)?;

            // 4. Attention Residuals (Long-term cross-layer aggregation)
            layer_outputs.push(parallel_out.clone());
            let attn_res_out = layer.attn_res.forward(&layer_outputs, &self.rope)?;

            h = (h + attn_res_out)?;

            // Sequential Block: Feed-Forward
            let h_ffn_norm = h.apply(&layer.norm_2)?;
            h = (h + layer.ffn.forward(&h_ffn_norm)?)?;
        }

        h.apply(&self.norm_final)?.apply(&self.output)
    }
}
