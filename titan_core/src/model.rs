use candle_core::{Tensor, Result};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use crate::attn_res::FullAttnRes;
use crate::titans::TitansMemory;
use crate::emulator::PythonEmulator;
use crate::quant::PolarQuant;

pub struct TitanTransformer {
    embedding: Embedding,
    layers: Vec<TitanLayer>,
    norm_final: RmsNorm,
    output: Linear,
}

pub struct FFN {
    up_proj: Linear,
    down_proj: Linear,
}

impl FFN {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let up_proj = linear(dim, dim * 4, vb.pp("up_proj"))?;
        let down_proj = linear(dim * 4, dim, vb.pp("down_proj"))?;
        Ok(Self { up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.apply(&self.up_proj)?;
        let h = candle_nn::ops::silu(&h)?;
        h.apply(&self.down_proj)
    }
}

pub struct TitanLayer {
    pub memory: TitansMemory,
    pub emulator: PythonEmulator,
    pub attn_res: FullAttnRes,
    pub ffn: FFN,
    pub norm_1: RmsNorm,
    pub norm_2: RmsNorm,
}

impl TitanTransformer {
    pub fn new(vocab_size: usize, dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(vocab_size, dim, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let vb_layer = vb.pp(format!("layer_{}", i));
            layers.push(TitanLayer {
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
        Ok(Self { embedding, layers, norm_final, output })
    }

    pub fn forward(&self, x: &Tensor, memory_states: &mut [Tensor], program_states: &mut [Tensor]) -> Result<Tensor> {
        let mut h = x.apply(&self.embedding)?;
        let mut layer_outputs = Vec::with_capacity(self.layers.len() + 1);
        layer_outputs.push(h.clone());

        for (i, layer) in self.layers.iter().enumerate() {
            // Residual Block 1: Norm -> Memory/Emulator/AttnRes -> Add
            let h_norm = h.apply(&layer.norm_1)?;

            let (radius, direction) = PolarQuant::compress(&h_norm)?;
            let h_quantized = PolarQuant::decompress(&radius, &direction)?;

            // Memory update
            let (mem_out, new_mem) = layer.memory.forward(&h_quantized, &memory_states[i])?;
            memory_states[i] = new_mem;

            // Emulator update
            let new_prog = layer.emulator.step(&h_quantized, &program_states[i])?;
            program_states[i] = new_prog.clone();

            // Combined components
            let mut combined = mem_out.broadcast_add(&new_prog)?;

            // Attention Residuals
            layer_outputs.push(combined.clone());
            combined = layer.attn_res.forward(&layer_outputs)?;

            h = (h + combined)?;

            // Residual Block 2: Norm -> FFN -> Add
            let h_ffn = h.apply(&layer.norm_2)?;
            h = (h + layer.ffn.forward(&h_ffn)?)?;
        }

        h.apply(&self.norm_final)?.apply(&self.output)
    }
}
