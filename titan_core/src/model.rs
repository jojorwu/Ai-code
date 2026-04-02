use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, linear, Linear, embedding, Embedding};
use crate::attn_res::FullAttnRes;
use crate::titans::TitansMemory;
use crate::emulator::PythonEmulator;
use crate::quant::PolarQuant;

pub struct TitanTransformer {
    embedding: Embedding,
    layers: Vec<TitanLayer>,
    output: Linear,
}

pub struct TitanLayer {
    pub memory: TitansMemory,
    pub emulator: PythonEmulator,
    pub attn_res: FullAttnRes,
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
            });
        }
        let output = linear(dim, vocab_size, vb.pp("output"))?;
        Ok(Self { embedding, layers, output })
    }

    pub fn forward(&self, x: &Tensor, memory_states: &mut [Tensor], program_states: &mut [Tensor]) -> Result<Tensor> {
        let h_initial = x.apply(&self.embedding)?;
        let mut h = h_initial.clone();
        let mut layer_outputs = Vec::with_capacity(self.layers.len() + 1);
        layer_outputs.push(h_initial);

        for (i, layer) in self.layers.iter().enumerate() {
            let (radius, direction) = PolarQuant::compress(&h)?;
            let h_quantized = PolarQuant::decompress(&radius, &direction)?;

            // Memory update
            let (mem_out, new_mem) = layer.memory.forward(&h_quantized, &memory_states[i])?;
            memory_states[i] = new_mem;

            // Emulator update
            let new_prog = layer.emulator.step(&h_quantized, &program_states[i])?;
            program_states[i] = new_prog.clone();

            // Combine
            // mem_out and new_prog are [1, D]. h is [T, D].
            h = h.broadcast_add(&mem_out)?;
            h = h.broadcast_add(&new_prog)?;

            // Apply Attention Residuals over all previous layer outputs
            layer_outputs.push(h.clone());
            h = layer.attn_res.forward(&layer_outputs)?;
        }

        h.apply(&self.output)
    }
}
