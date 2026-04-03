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
    logit_cap: Tensor,
    memory_tokens: Tensor,
    block_attn_res: crate::attn_res::BlockAttnRes,
    block_size: usize,
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

/// Sparse Mixture of Experts (MoE) block with a Shared Expert.
pub struct MoE {
    router: Linear,
    experts: Vec<FFN>,
    shared_expert: FFN,
    num_experts: usize,
}

impl MoE {
    /// Creates a new MoE block with a Shared Expert.
    pub fn new(dim: usize, num_experts: usize, vb: VarBuilder) -> Result<Self> {
        let router = linear(dim, num_experts, vb.pp("router"))?;
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(FFN::new(dim, vb.pp(format!("expert_{}", i)))?);
        }
        let shared_expert = FFN::new(dim, vb.pp("shared_expert"))?;
        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts,
        })
    }

    /// Performs the forward pass of the MoE block with Top-1 routing and a Shared Expert.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let router_logits = x.apply(&self.router)?;
        let routing_weights = candle_nn::ops::softmax(&router_logits, candle_core::D::Minus1)?;

        // 1. Shared Expert processes all tokens
        let mut final_output = self.shared_expert.forward(x)?;

        // 2. Dynamic Experts (Top-1 routing)
        let expert_indices = router_logits.argmax(candle_core::D::Minus1)?;

        // Correct implementation for architectural demonstration:
        // We compute all experts and mask, OR we can process subset.
        // For standard Candle CPU, mask is robust.
        for i in 0..self.num_experts {
            let expert_out = self.experts[i].forward(x)?;

            // Mask: 1.0 if expert_indices == i, 0.0 otherwise
            let mask = expert_indices.eq(i as u32)?; // [T] (U8)

            // routing_weights has [T, E]
            let weight = routing_weights.narrow(candle_core::D::Minus1, i, 1)?.squeeze(candle_core::D::Minus1)?;

            // Combine mask and weight [T]
            // Standardize both to F32 for multiplication then back to original.
            let mask_f32 = mask.to_dtype(candle_core::DType::F32)?;
            let weight_f32 = weight.to_dtype(candle_core::DType::F32)?;
            let combined_gate = (mask_f32 * weight_f32)?;
            let combined_gate = combined_gate.to_dtype(x.dtype())?;

            // final_output: [T, D], expert_out: [T, D], combined_gate: [T]
            // We need to unsqueeze combined_gate to [T, 1] for broadcasting to [T, D]
            let combined_gate = combined_gate.unsqueeze(candle_core::D::Minus1)?;

            final_output = (final_output + expert_out.broadcast_mul(&combined_gate)?)?;
        }

        Ok(final_output)
    }
}

/// A single layer of the Titan Transformer.
pub struct TitanLayer {
    pub attention: MultiHeadAttention,
    pub memory: TitansMemory,
    pub emulator: PythonEmulator,
    pub moe: MoE,
    pub norm_1: RmsNorm,
    pub norm_2: RmsNorm,
    pub branch_gates: Tensor,
    pub layerscale_1: Tensor,
    pub layerscale_2: Tensor,
}

impl TitanTransformer {
    /// Creates a new Titan Transformer model.
    pub fn new(vocab_size: usize, dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(vocab_size, dim, vb.pp("embedding"))?;
        let rope = RotaryEmbedding::new(dim, 2048, vb.device())?;
        let mut layers = Vec::with_capacity(num_layers);
        let block_size = 4;
        for i in 0..num_layers {
            let vb_layer = vb.pp(format!("layer_{}", i));
            layers.push(TitanLayer {
                attention: MultiHeadAttention::new(dim, 8, 2, vb_layer.pp("attention"))?, // GQA with 2 KV heads
                memory: TitansMemory::new(dim, vb_layer.pp("memory"))?,
                emulator: PythonEmulator::new(dim, vb_layer.pp("emulator"))?,
                moe: MoE::new(dim, 4, vb_layer.pp("moe"))?, // 4 experts + shared expert
                norm_1: rms_norm(dim, 1e-5, vb_layer.pp("norm_1"))?,
                norm_2: rms_norm(dim, 1e-5, vb_layer.pp("norm_2"))?,
                branch_gates: vb_layer.get((3,), "branch_gates")?,
                layerscale_1: vb_layer.get((dim,), "layerscale_1")?,
                layerscale_2: vb_layer.get((dim,), "layerscale_2")?,
            });
        }
        let norm_final = rms_norm(dim, 1e-5, vb.pp("norm_final"))?;
        let logit_cap = vb.get((1,), "logit_cap")?;
        let memory_tokens = vb.get((8, dim), "memory_tokens")?; // 8 persistent memory tokens
        let block_attn_res = crate::attn_res::BlockAttnRes::new(dim, vb.pp("block_attn_res"))?;

        // Weight Tying: The output layer shares weights with the embedding layer
        let output_vb = vb.pp("output");
        let output_weights = embedding.embeddings().clone();
        let output = Linear::new(output_weights, None); // No bias for weight tying usually

        Ok(Self {
            embedding,
            layers,
            norm_final,
            output,
            rope,
            logit_cap,
            memory_tokens,
            block_attn_res,
            block_size,
        })
    }

    /// Performs the forward pass of the model with optional KV caching.
    pub fn forward(
        &self,
        x: &Tensor,
        memory_states: &mut [Tensor],
        program_states: &mut [Tensor],
        kv_caches: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let mut h = x.apply(&self.embedding)?;
        let dtype = h.dtype();
        let device = h.device();

        // Detect if we are in incremental generation mode (using KV-cache)
        let is_incremental = kv_caches.iter().any(|c| c.is_some());

        // Prepend persistent memory tokens ONLY if we are at the start of a sequence
        if !is_incremental {
             h = Tensor::cat(&[&self.memory_tokens, &h], 0)?;
        }

        let mut local_history = Vec::with_capacity(self.block_size);
        let mut block_summaries = Vec::with_capacity(self.layers.len() / self.block_size + 1);

        for (i, layer) in self.layers.iter().enumerate() {
            // Parallel Block Architecture:
            // h = h + SelfAttn(Norm(h)) + Memory(Norm(h)) + Emulator(Norm(h))
            let h_norm = h.apply(&layer.norm_1)?;

            // 1. Self-Attention with KV-Cache
            let (attn_out, new_kv) = layer.attention.forward(&h_norm, &self.rope, kv_caches[i].clone())?;
            kv_caches[i] = Some(new_kv);

            // 2. Neural Memory
            let (radius, direction) = PolarQuant::compress(&h_norm)?;
            let h_quantized = PolarQuant::decompress(&radius, &direction)?;
            let (mem_out, new_mem) = layer.memory.forward(&h_quantized, &memory_states[i], &self.rope)?;
            memory_states[i] = new_mem;

            // 3. Program Emulator
            let new_prog = layer.emulator.step(&h_quantized, &program_states[i])?;
            program_states[i] = new_prog.clone();

            // Aggregate parallel components with learnable gating (vectorized)
            let gates = candle_nn::ops::softmax(&layer.branch_gates, 0)?;
            let g_attn = gates.narrow(0, 0, 1)?;
            let g_mem = gates.narrow(0, 1, 1)?;
            let g_emu = gates.narrow(0, 2, 1)?;

            let mut parallel_out = attn_out.broadcast_mul(&g_attn)?;
            parallel_out = (parallel_out + mem_out.broadcast_mul(&g_mem)?)?;
            parallel_out = (parallel_out.broadcast_add(&new_prog.broadcast_mul(&g_emu)?))?;

            // 4. Block Attention Residuals
            let attn_res_out = self.block_attn_res.forward(&block_summaries, &parallel_out, &self.rope)?;

            h = (h + attn_res_out.broadcast_mul(&layer.layerscale_1)?)?;

            // Sequential Block: MoE Feed-Forward
            let h_ffn_norm = h.apply(&layer.norm_2)?;
            let moe_out = layer.moe.forward(&h_ffn_norm)?;
            h = (h + moe_out.broadcast_mul(&layer.layerscale_2)?)?;

            // Manage history for Block Attention Residuals
            local_history.push(h.clone());
            if local_history.len() == self.block_size {
                 // Block completed: summarize block (e.g. mean of local history)
                 let block_summary = (Tensor::stack(&local_history, 0)?.mean(0))?;
                 block_summaries.push(block_summary);
                 local_history.clear();
            }
        }

        let h = h.apply(&self.norm_final)?;

        // Skip memory tokens for output projection
        let m_size = self.memory_tokens.dim(0)?;
        let total_size = h.dim(0)?;
        let h_output = if total_size > m_size {
             h.narrow(0, m_size, total_size - m_size)?
        } else {
             h
        };

        let logits = h_output.apply(&self.output)?;

        // Learnable Logit Softcapping (vectorized)
        // softplus: ln(1 + exp(x))
        let cap = (self.logit_cap.exp()?.affine(1.0, 1.0)?.log()? + 1.0)?;

        let softcapped = logits.broadcast_div(&cap)?.tanh()?;
        softcapped.broadcast_mul(&cap)
    }
}
