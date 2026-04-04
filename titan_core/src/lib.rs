pub mod attn_res;
pub mod titans;
pub mod quant;
pub mod emulator;
pub mod rope;
pub mod attention;
pub mod model;

use candle_core::{Device, Tensor, DType, Shape};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use crate::model::{TitanTransformer, Config};
use pyo3::prelude::*;
use tokenizers::Tokenizer;

/// Helper to convert candle errors into PyValueError.
fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
}

#[pyclass]
#[doc = "The main entry point for the Titan Transformer in Python.\n\n\
         This class manages the lifecycle of the model, including state management\n\
         for persistent neural memory and KV-caching. It provides high-level methods\n\
         for forward inference and sequence training."]
pub struct PyTitanTransformer {
    inner: TitanTransformer,
    varmap: VarMap,
    config: Config,
    memory_states: Vec<Tensor>,
    program_states: Vec<Tensor>,
    kv_caches: Vec<Option<(Tensor, Tensor)>>,
    optimizer: Option<AdamW>,
}

#[pymethods]
impl PyTitanTransformer {
    #[new]
    #[pyo3(signature = (vocab_size, dim, num_layers, num_heads=None, num_kv_heads=None, window_size=None, block_size=None, m_size=None, num_experts=None))]
    #[doc = "Initializes a new Titan Transformer with advanced configuration."]
    fn new(
        vocab_size: usize,
        dim: usize,
        num_layers: usize,
        num_heads: Option<usize>,
        num_kv_heads: Option<usize>,
        window_size: Option<usize>,
        block_size: Option<usize>,
        m_size: Option<usize>,
        num_experts: Option<usize>,
    ) -> PyResult<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut config = Config::default();
        config.vocab_size = vocab_size;
        config.dim = dim;
        config.num_layers = num_layers;
        config.num_heads = num_heads.unwrap_or(8);
        config.num_kv_heads = num_kv_heads.unwrap_or(2);
        config.window_size = window_size.unwrap_or(512);
        if let Some(b) = block_size { config.block_size = b; }
        if let Some(m) = m_size { config.m_size = m; }
        if let Some(e) = num_experts { config.num_experts = e; }

        // Safety: Ensure dim is divisible by num_heads
        if config.dim % config.num_heads != 0 {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                 format!("Dimension {} must be divisible by num_heads {}", config.dim, config.num_heads)
             ));
        }

        let inner = TitanTransformer::new(config, vb).map_err(to_py_err)?;

        // Initialize states
        let head_dim = config.dim / config.num_heads;
        let mut memory_states = Vec::with_capacity(config.num_layers);
        let mut program_states = Vec::with_capacity(config.num_layers);
        let mut kv_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            // Multi-head memory state [H, Hd, Hd]
            memory_states.push(Tensor::zeros((config.num_heads, head_dim, head_dim), DType::F32, &device).map_err(to_py_err)?);
            program_states.push(Tensor::zeros((1, config.dim), DType::F32, &device).map_err(to_py_err)?);
            kv_caches.push(None);
        }

        Ok(Self {
            inner,
            varmap,
            config,
            memory_states,
            program_states,
            kv_caches,
            optimizer: None,
        })
    }

    #[doc = "Initializes the AdamW optimizer with a specific learning rate."]
    fn init_optimizer(&mut self, lr: f64) -> PyResult<()> {
        let params = ParamsAdamW {
            lr,
            ..Default::default()
        };
        let opt = AdamW::new(self.varmap.all_vars(), params).map_err(to_py_err)?;
        self.optimizer = Some(opt);
        Ok(())
    }

    #[doc = "Saves the model weights to the specified path in .safetensors format."]
    fn save_weights(&self, path: String) -> PyResult<()> {
        self.varmap.save(path).map_err(to_py_err)?;
        Ok(())
    }

    #[doc = "Loads the model weights from the specified .safetensors path."]
    fn load_weights(&mut self, path: String) -> PyResult<()> {
        self.varmap.load(path).map_err(to_py_err)?;
        Ok(())
    }

    #[doc = "Resets the persistent internal long-term memory, program states, and KV-cache to zero/empty."]
    fn reset_state(&mut self) -> PyResult<()> {
        let device = Device::Cpu;
        let head_dim = self.config.dim / self.config.num_heads;
        for i in 0..self.config.num_layers {
            self.memory_states[i] =
                Tensor::zeros((self.config.num_heads, head_dim, head_dim), DType::F32, &device).map_err(to_py_err)?;
            self.program_states[i] =
                Tensor::zeros((1, self.config.dim), DType::F32, &device).map_err(to_py_err)?;
            self.kv_caches[i] = None;
        }
        Ok(())
    }

    #[doc = "Performs a forward pass on a sequence of token IDs and returns the logits for each token."]
    fn forward(&mut self, x_ids: Vec<u32>) -> PyResult<Vec<f32>> {
        let device = Device::Cpu;
        let n = x_ids.len();

        // Safety guard: Prevent OOM on excessively long sequences
        const MAX_SEQ_LEN: usize = 8192;
        if n > MAX_SEQ_LEN {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                 format!("Input sequence length {} exceeds maximum allowed length of {}", n, MAX_SEQ_LEN)
             ));
        }

        if n == 0 {
            return Ok(vec![]);
        }

        // Ensure all persistent states are on the correct device and DType
        for state in &self.memory_states {
            if state.device().location() != device.location() || state.dtype() != DType::F32 {
                 return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Internal state device/dtype mismatch"));
            }
        }

        let x = Tensor::from_vec(x_ids, Shape::from(n), &device).map_err(to_py_err)?;

        let (out, _aux_loss) = self
            .inner
            .forward(&x, &mut self.memory_states, &mut self.program_states, &mut self.kv_caches)
            .map_err(to_py_err)?;

        let (rows, cols) = out.dims2().map_err(to_py_err)?;

        let flattened = out
            .reshape(rows * cols)
            .map_err(to_py_err)?
            .to_vec1::<f32>()
            .map_err(to_py_err)?;

        Ok(flattened)
    }

    #[doc = "Executes a single training step (forward pass, loss calculation, and backpropagation) and returns the loss value."]
    fn train_step(&mut self, x_ids: Vec<u32>, target_ids: Vec<u32>) -> PyResult<f32> {
        let device = Device::Cpu;
        let n = x_ids.len();

        const MAX_SEQ_LEN: usize = 4096; // Stricter for training due to gradients
        if n > MAX_SEQ_LEN {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                 format!("Training sequence length {} exceeds maximum allowed length of {}", n, MAX_SEQ_LEN)
             ));
        }

        if n == 0 {
            return Ok(0.0);
        }

        for state in &self.memory_states {
            if state.device().location() != device.location() || state.dtype() != DType::F32 {
                 return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Internal state device/dtype mismatch"));
            }
        }

        let x = Tensor::from_vec(x_ids, Shape::from(n), &device).map_err(to_py_err)?;
        let targets = Tensor::from_vec(target_ids, Shape::from(n), &device).map_err(to_py_err)?;

        // During training, we typically don't use KV cache or we want to reset it.
        // For simplicity, we'll reset it here.
        for i in 0..self.config.num_layers {
             self.kv_caches[i] = None;
        }

        let (logits, aux_loss) = self
            .inner
            .forward(&x, &mut self.memory_states, &mut self.program_states, &mut self.kv_caches)
            .map_err(to_py_err)?;

        let log_sm = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1).map_err(to_py_err)?;

        let nll_loss = candle_nn::loss::nll(&log_sm, &targets).map_err(to_py_err)?;

        // Total loss = NLL + 0.1 * Auxiliary MoE Balancing Loss
        let total_loss = (nll_loss + (aux_loss * 0.1).map_err(to_py_err)?) .map_err(to_py_err)?;

        if let Some(opt) = &mut self.optimizer {
            opt.backward_step(&total_loss).map_err(to_py_err)?;
        }

        let loss_val = total_loss.to_vec0::<f32>().map_err(to_py_err)?;

        Ok(loss_val)
    }
}

#[pyclass]
#[doc = "A wrapper around the tokenizers library for text encoding and decoding."]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[doc = "Loads a tokenizer from a JSON configuration file."]
    fn new(json_path: String) -> PyResult<Self> {
        let inner = Tokenizer::from_file(json_path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[doc = "Encodes a string of text into a list of token IDs."]
    fn encode(&self, text: String) -> PyResult<Vec<u32>> {
        let encoding = self.inner.encode(text, true).map_err(to_py_err)?;
        Ok(encoding.get_ids().to_vec())
    }

    #[doc = "Decodes a list of token IDs back into a human-readable string."]
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let text = self.inner.decode(&ids, true).map_err(to_py_err)?;
        Ok(text)
    }
}

#[pymodule]
fn _titan_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTitanTransformer>()?;
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
