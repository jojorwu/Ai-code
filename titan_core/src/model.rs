//! Main architecture definition for the Titan Transformer.
//!
//! Titan is a hybrid model combining Self-Attention, Neural Long-Term Memory (Titans),
//! and a Program Emulator in a parallel residual structure.

use candle_core::{Tensor, Result};
use candle_nn::{embedding, linear, rms_norm, conv1d, Conv1d, Conv1dConfig, Embedding, Linear, RmsNorm, VarBuilder, Init};
use serde::{Serialize, Deserialize};
use crate::attention::MultiHeadAttention;
use crate::titans::TitansMemory;
use crate::emulator::PythonEmulator;
use crate::quant::PolarQuant;
use crate::rope::RotaryEmbedding;

/// Architectural configuration for the Titan Transformer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub window_size: usize,
    pub block_size: usize,
    pub m_size: usize,
    pub num_experts: usize,
    pub global_attn_period: usize,
    pub use_weight_std: bool,
    pub use_turbo_quant: bool,
    pub drop_path_rate: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 4,
            window_size: 512,
            block_size: 4,
            m_size: 8,
            num_experts: 4,
            global_attn_period: 4,
            use_weight_std: true,
            use_turbo_quant: false,
            drop_path_rate: 0.1,
        }
    }
}

/// Main Titan Transformer model.
pub struct TitanTransformer {
    pub config: Config,
    embedding: Embedding,
    layers: Vec<TitanLayer>,
    norm_final: RmsNorm,
    output: Linear,
    rope: RotaryEmbedding,
    logit_cap: Tensor,
    memory_tokens: Tensor,
    block_attn_res: crate::attn_res::BlockAttnRes,
}

/// Helper for Stochastic Depth (DropPath).
pub fn drop_path(x: &Tensor, drop_prob: f32, is_training: bool) -> Result<Tensor> {
    if !is_training || drop_prob <= 0.0 {
        return Ok(x.clone());
    }
    let keep_prob = 1.0 - drop_prob;
    let shape = x.shape();
    let mut dims = vec![1; shape.dims().len()];
    dims[0] = shape.dims()[0]; // Batch or Sequence
    let rand = Tensor::rand(0f32, 1f32, dims, x.device())?;
    let mask = rand.ge(drop_prob as f64)?.to_dtype(x.dtype())?;
    x.broadcast_mul(&mask)? / keep_prob as f64
}

/// Helper for Weight Standardization.
/// Normalizes the weights of a linear layer.
pub fn weight_std(w: &Tensor) -> Result<Tensor> {
    let mean = w.mean_keepdim(1)?;
    let centered = w.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(1)?;
    let std = (var + 1e-5)?.sqrt()?;
    centered.broadcast_div(&std)
}

/// A standardized linear layer.
pub struct StdLinear {
    inner: Linear,
    use_std: bool,
}

impl StdLinear {
    pub fn new(in_dim: usize, out_dim: usize, use_std: bool, vb: VarBuilder) -> Result<Self> {
        let inner = linear(in_dim, out_dim, vb)?;
        Ok(Self { inner, use_std })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.use_std {
             let w = self.inner.weight();
             let w_std = weight_std(w)?;
             // We can't easily replace the weight in Linear, so we matmul manually
             let x = x.matmul(&w_std.t()?)?;
             if let Some(bias) = self.inner.bias() {
                 x.broadcast_add(bias)
             } else {
                 Ok(x)
             }
        } else {
             x.apply(&self.inner)
        }
    }
}

/// Feed-Forward Network (FFN) block using SwiGLU activation and depthwise Conv1D.
pub struct FFN {
    gate_proj: StdLinear,
    up_proj: StdLinear,
    down_proj: StdLinear,
    conv: Conv1d,
}

impl FFN {
    /// Creates a new FFN block with Conv-GLU structure.
    pub fn new(dim: usize, use_std: bool, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (dim * 8) / 3;
        let gate_proj = StdLinear::new(dim, hidden_dim, use_std, vb.pp("gate_proj"))?;
        let up_proj = StdLinear::new(dim, hidden_dim, use_std, vb.pp("up_proj"))?;
        let down_proj = StdLinear::new(hidden_dim, dim, use_std, vb.pp("down_proj"))?;

        // Depthwise convolution: in_channels == out_channels == groups
        let conv_cfg = Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: hidden_dim,
        };
        let conv = conv1d(hidden_dim, hidden_dim, 3, conv_cfg, vb.pp("conv"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            conv,
        })
    }

    /// Performs the forward pass of the FFN block with Conv-GLU.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [T, D]
        let gate = self.gate_proj.forward(x)?; // [T, H]

        // Apply 1D conv over sequence dimension: [1, H, T]
        let gate = gate.t()?.unsqueeze(0)?;
        let gate = gate.apply(&self.conv)?;
        let gate = gate.squeeze(0)?.t()?; // [T, H]

        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let h = (gate * up)?;
        self.down_proj.forward(&h)
    }
}

/// Sparse Mixture of Experts (MoE) block with a Shared Expert and Top-1 routing.
pub struct MoE {
    router: StdLinear,
    experts: Vec<FFN>,
    shared_expert: FFN,
    num_experts: usize,
}

impl MoE {
    /// Creates a new MoE block with a Shared Expert.
    pub fn new(dim: usize, num_experts: usize, use_std: bool, vb: VarBuilder) -> Result<Self> {
        let router = StdLinear::new(dim, num_experts, use_std, vb.pp("router"))?;
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(FFN::new(dim, use_std, vb.pp(format!("expert_{}", i)))?);
        }
        let shared_expert = FFN::new(dim, use_std, vb.pp("shared_expert"))?;
        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts,
        })
    }

    /// Performs the forward pass of the MoE block with Top-1 routing and a Shared Expert.
    /// Returns (output, auxiliary_loss)
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut router_logits = self.router.forward(x)?;

        // Noise-Gated Routing (Exploration)
        // Add small Gaussian noise to logits during training to prevent routing collapse
        if router_logits.device().is_cpu() {
             let noise = Tensor::randn(0f32, 0.01f32, router_logits.shape(), router_logits.device())?;
             router_logits = (router_logits + noise)?;
        }

        let routing_weights = candle_nn::ops::softmax(&router_logits, candle_core::D::Minus1)?;

        // 1. Shared Expert processes all tokens
        let mut final_output = self.shared_expert.forward(x)?;

        // 2. Dynamic Experts (Top-1 routing)
        let expert_indices = router_logits.argmax(candle_core::D::Minus1)?;

        // Implement Expert Routing logic (Top-1)
        // Selection: expert_i = argmax(router(x))
        // Final Output: y = shared_expert(x) + routing_weights[expert_i] * expert_i(x)
        for i in 0..self.num_experts {
            let expert_out = self.experts[i].forward(x)?;

            // Generate binary mask for expert selection [T, 1]
            let mask = expert_indices.eq(i as u32)?.unsqueeze(candle_core::D::Minus1)?;
            let weight = routing_weights.narrow(candle_core::D::Minus1, i, 1)?;

            // Apply gating: mask * softmax_weight
            let combined_gate = (mask.to_dtype(candle_core::DType::F32)?.broadcast_mul(&weight))?.to_dtype(x.dtype())?;

            final_output = (final_output + expert_out.broadcast_mul(&combined_gate)?)?;
        }

        // 3. Auxiliary Balancing Loss (Load Balancing)
        let mut aux_loss = Tensor::new(0f32, x.device())?;
        let t_size = routing_weights.dim(0)? as f32;
        let p_mean = routing_weights.mean(0)?;

        for i in 0..self.num_experts {
             let f_i = expert_indices.eq(i as u32)?.to_dtype(candle_core::DType::F32)?.sum_all()?.to_vec0::<f32>()? / t_size;
             let p_i = p_mean.get(i)?.to_vec0::<f32>()?;
             let term = f_i * p_i * (self.num_experts as f32);
             aux_loss = (aux_loss + Tensor::new(term, x.device())?)?;
        }

        Ok((final_output, aux_loss))
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
    pub gate_net: Linear,
    pub layerscale_1: Tensor,
    pub layerscale_2: Tensor,
}

impl TitanTransformer {
    /// Creates a new Titan Transformer model with a configuration struct.
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.dim, vb.pp("embedding"))?;
        let rope = RotaryEmbedding::new(config.dim, 8192, vb.device())?;
        let mut layers = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let is_global = i % config.global_attn_period == 0;
            let vb_layer = vb.pp(format!("layer_{}", i));
            layers.push(TitanLayer {
                attention: MultiHeadAttention::new(
                    config.dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.window_size,
                    is_global,
                    vb_layer.pp("attention")
                )?,
                memory: TitansMemory::new(config.dim, config.num_heads, config.use_turbo_quant, vb_layer.pp("memory"))?,
                emulator: PythonEmulator::new(config.dim, vb_layer.pp("emulator"))?,
                moe: MoE::new(config.dim, config.num_experts, config.use_weight_std, vb_layer.pp("moe"))?,
                norm_1: rms_norm(config.dim, 1e-5, vb_layer.pp("norm_1"))?,
                norm_2: rms_norm(config.dim, 1e-5, vb_layer.pp("norm_2"))?,
                gate_net: linear(config.dim, 3, vb_layer.pp("gate_net"))?,
                layerscale_1: vb_layer.get_with_hints((config.dim,), "layerscale_1", Init::Const(0.1))?,
                layerscale_2: vb_layer.get_with_hints((config.dim,), "layerscale_2", Init::Const(0.1))?,
            });
        }
        let norm_final = rms_norm(config.dim, 1e-5, vb.pp("norm_final"))?;
        let logit_cap = vb.get_with_hints((1,), "logit_cap", Init::Const(2.0))?;
        let memory_tokens = vb.get((config.m_size, config.dim), "memory_tokens")?;
        let block_attn_res = crate::attn_res::BlockAttnRes::new(config.dim, vb.pp("block_attn_res"))?;

        // Weight Tying: The output layer shares weights with the embedding layer
        let output_weights = embedding.embeddings().clone();
        let output = Linear::new(output_weights, None); // No bias for weight tying usually

        Ok(Self {
            config,
            embedding,
            layers,
            norm_final,
            output,
            rope,
            logit_cap,
            memory_tokens,
            block_attn_res,
        })
    }

    /// Performs the forward pass of the model with optional KV caching.
    /// Returns (logits, auxiliary_loss)
    pub fn forward(
        &self,
        x: &Tensor,
        memory_states: &mut [Tensor],
        program_states: &mut [Tensor],
        kv_caches: &mut [Option<(Tensor, Tensor)>],
        is_training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (mut h, start_pos) = self.prepare_input(x, kv_caches)?;
        let mut total_aux_loss = Tensor::new(0f32, h.device())?;

        let mut local_history = Vec::with_capacity(self.config.block_size);
        let mut block_summaries = Vec::with_capacity(self.layers.len() / self.config.block_size + 1);

        for (i, layer) in self.layers.iter().enumerate() {
            let (h_next, aux_loss) = self.apply_layer(
                layer,
                &h,
                &mut memory_states[i],
                &mut program_states[i],
                &mut kv_caches[i],
                &block_summaries,
                start_pos,
                is_training,
            )?;

            h = h_next;
            total_aux_loss = (total_aux_loss + aux_loss)?;

            // Manage block history
            local_history.push(h.clone());
            if local_history.len() == self.config.block_size {
                 let block_summary = (Tensor::stack(&local_history, 0)?.mean(0))?;
                 block_summaries.push(block_summary);
                 local_history.clear();
            }
        }

        self.post_process(&h, total_aux_loss)
    }

    /// Prepares input by applying embedding and prepending memory tokens if necessary.
    fn prepare_input(&self, x: &Tensor, kv_caches: &[Option<(Tensor, Tensor)>]) -> Result<(Tensor, usize)> {
        let mut h = x.apply(&self.embedding)?;
        let kv_len = if let Some(cache) = &kv_caches[0] {
            cache.0.dim(1)?
        } else {
            0
        };
        let is_incremental = kv_len > 0;

        if !is_incremental {
             h = Tensor::cat(&[&self.memory_tokens, &h], 0)?;
        }

        let start_pos = if is_incremental { kv_len } else { 0 };
        Ok((h, start_pos))
    }

    /// Applies a single Titan layer.
    fn apply_layer(
        &self,
        layer: &TitanLayer,
        h: &Tensor,
        memory_state: &mut Tensor,
        program_state: &mut Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        block_summaries: &[Tensor],
        start_pos: usize,
        is_training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let h_norm = h.apply(&layer.norm_1)?;

        // 1. Parallel Branches
        let (attn_out, new_kv) = layer.attention.forward(&h_norm, &self.rope, kv_cache.clone(), start_pos)?;
        *kv_cache = Some(new_kv);

        let (radius, direction) = PolarQuant::compress(&h_norm)?;
        let h_quantized = PolarQuant::decompress(&radius, &direction)?;
        let (mem_out, new_mem) = layer.memory.forward(&h_quantized, memory_state, &self.rope, start_pos)?;
        *memory_state = new_mem;

        let new_prog = layer.emulator.step(&h_quantized, program_state)?;
        *program_state = new_prog.clone();

        // 2. Dynamic Aggregation
        let gates = h_norm.apply(&layer.gate_net)?;
        let gates = candle_nn::ops::softmax(&gates, candle_core::D::Minus1)?;
        let g_attn = gates.narrow(candle_core::D::Minus1, 0, 1)?;
        let g_mem = gates.narrow(candle_core::D::Minus1, 1, 1)?;
        let g_emu = gates.narrow(candle_core::D::Minus1, 2, 1)?;

        let mut parallel_out = attn_out.broadcast_mul(&g_attn)?;
        parallel_out = (parallel_out + mem_out.broadcast_mul(&g_mem)?)?;
        parallel_out = (parallel_out + new_prog.broadcast_mul(&g_emu)?)?;

        // 3. Block Residuals
        let attn_res_out = self.block_attn_res.forward(block_summaries, &parallel_out, &self.rope, start_pos)?;
        let layer_out = drop_path(&attn_res_out.broadcast_mul(&layer.layerscale_1)?, self.config.drop_path_rate, is_training)?;
        let mut h = (h + layer_out)?;

        // 4. Sequential MoE
        let h_ffn_norm = h.apply(&layer.norm_2)?;
        let (moe_out, aux_loss) = layer.moe.forward(&h_ffn_norm)?;
        let moe_res = drop_path(&moe_out.broadcast_mul(&layer.layerscale_2)?, self.config.drop_path_rate, is_training)?;
        h = (h + moe_res)?;

        Ok((h, aux_loss))
    }

    /// Final normalization, softcapping, and output projection.
    fn post_process(&self, h: &Tensor, total_aux_loss: Tensor) -> Result<(Tensor, Tensor)> {
        let h = h.apply(&self.norm_final)?;

        let m_size = self.memory_tokens.dim(0)?;
        let total_size = h.dim(0)?;
        let h_output = if total_size > m_size {
             h.narrow(0, m_size, total_size - m_size)?
        } else {
             h
        };

        let logits = h_output.apply(&self.output)?;
        let cap = (self.logit_cap.exp()?.affine(1.0, 1.0)?.log()? + 1.0)?;
        let softcapped = logits.broadcast_div(&cap)?.tanh()?;
        Ok((softcapped.broadcast_mul(&cap)?, total_aux_loss))
    }
}
