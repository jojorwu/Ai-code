pub mod attn_res;
pub mod titans;
pub mod quant;
pub mod emulator;
pub mod rope;
pub mod attention;
pub mod model;

use candle_core::{Device, Tensor, DType, Shape};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use crate::model::TitanTransformer;
use pyo3::prelude::*;
use tokenizers::Tokenizer;

/// Helper to convert candle errors into PyValueError.
fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
}

#[pyclass]
#[doc = "Rust-based implementation of the Titan Transformer model."]
pub struct PyTitanTransformer {
    inner: TitanTransformer,
    varmap: VarMap,
    dim: usize,
    num_layers: usize,
    memory_states: Vec<Tensor>,
    program_states: Vec<Tensor>,
    optimizer: Option<AdamW>,
}

#[pymethods]
impl PyTitanTransformer {
    #[new]
    #[doc = "Initializes a new Titan Transformer with the given vocabulary size, dimension, and number of layers."]
    fn new(vocab_size: usize, dim: usize, num_layers: usize) -> PyResult<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let inner = TitanTransformer::new(vocab_size, dim, num_layers, vb).map_err(to_py_err)?;

        // Initialize states
        let num_heads = 8;
        let head_dim = dim / num_heads;
        let mut memory_states = Vec::with_capacity(num_layers);
        let mut program_states = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // Multi-head memory state [H, Hd, Hd]
            memory_states.push(Tensor::zeros((num_heads, head_dim, head_dim), DType::F32, &device).map_err(to_py_err)?);
            program_states.push(Tensor::zeros((1, dim), DType::F32, &device).map_err(to_py_err)?);
        }

        Ok(Self {
            inner,
            varmap,
            dim,
            num_layers,
            memory_states,
            program_states,
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

    #[doc = "Resets the persistent internal long-term memory and program states to zero."]
    fn reset_state(&mut self) -> PyResult<()> {
        let device = Device::Cpu;
        let num_heads = 8;
        let head_dim = self.dim / num_heads;
        for i in 0..self.num_layers {
            self.memory_states[i] =
                Tensor::zeros((num_heads, head_dim, head_dim), DType::F32, &device).map_err(to_py_err)?;
            self.program_states[i] =
                Tensor::zeros((1, self.dim), DType::F32, &device).map_err(to_py_err)?;
        }
        Ok(())
    }

    #[doc = "Performs a forward pass on a sequence of token IDs and returns the logits for each token."]
    fn forward(&mut self, x_ids: Vec<u32>) -> PyResult<Vec<f32>> {
        let device = Device::Cpu;
        let n = x_ids.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let x = Tensor::from_vec(x_ids, Shape::from(n), &device).map_err(to_py_err)?;

        let out = self
            .inner
            .forward(&x, &mut self.memory_states, &mut self.program_states)
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
        if n == 0 {
            return Ok(0.0);
        }
        let x = Tensor::from_vec(x_ids, Shape::from(n), &device).map_err(to_py_err)?;
        let targets = Tensor::from_vec(target_ids, Shape::from(n), &device).map_err(to_py_err)?;

        let logits = self
            .inner
            .forward(&x, &mut self.memory_states, &mut self.program_states)
            .map_err(to_py_err)?;

        let log_sm = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1).map_err(to_py_err)?;

        let loss = candle_nn::loss::nll(&log_sm, &targets).map_err(to_py_err)?;

        if let Some(opt) = &mut self.optimizer {
            opt.backward_step(&loss).map_err(to_py_err)?;
        }

        let loss_val = loss.to_vec0::<f32>().map_err(to_py_err)?;

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
