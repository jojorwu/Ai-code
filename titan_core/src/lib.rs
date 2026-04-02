pub mod attn_res;
pub mod titans;
pub mod quant;
pub mod emulator;
pub mod model;

use candle_core::{Device, Tensor, DType, Shape};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use crate::model::TitanTransformer;
use pyo3::prelude::*;
use tokenizers::Tokenizer;

#[pyclass]
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
    fn new(vocab_size: usize, dim: usize, num_layers: usize) -> PyResult<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let inner = TitanTransformer::new(vocab_size, dim, num_layers, vb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let memory_states = vec![Tensor::zeros((1, dim), DType::F32, &device).unwrap(); num_layers];
        let program_states = vec![Tensor::zeros((1, dim), DType::F32, &device).unwrap(); num_layers];

        Ok(Self { inner, varmap, dim, num_layers, memory_states, program_states, optimizer: None })
    }

    fn init_optimizer(&mut self, lr: f64) -> PyResult<()> {
        let params = ParamsAdamW {
            lr,
            ..Default::default()
        };
        let opt = AdamW::new(self.varmap.all_vars(), params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        self.optimizer = Some(opt);
        Ok(())
    }

    fn save_weights(&self, path: String) -> PyResult<()> {
        self.varmap.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(())
    }

    fn load_weights(&mut self, path: String) -> PyResult<()> {
        self.varmap.load(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(())
    }

    fn reset_state(&mut self) -> PyResult<()> {
        let device = Device::Cpu;
        self.memory_states = vec![Tensor::zeros((1, self.dim), DType::F32, &device).unwrap(); self.num_layers];
        self.program_states = vec![Tensor::zeros((1, self.dim), DType::F32, &device).unwrap(); self.num_layers];
        Ok(())
    }

    fn forward(&mut self, x_ids: Vec<u32>) -> PyResult<Vec<f32>> {
        let device = Device::Cpu;
        let n = x_ids.len();
        if n == 0 {
             return Ok(vec![]);
        }
        let x = Tensor::from_vec(x_ids, Shape::from(n), &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let out = self.inner.forward(&x, &mut self.memory_states, &mut self.program_states)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let flattened = out.flatten_all()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(flattened)
    }

    fn train_step(&mut self, x_ids: Vec<u32>, target_ids: Vec<u32>) -> PyResult<f32> {
        let device = Device::Cpu;
        let n = x_ids.len();
        if n == 0 {
             return Ok(0.0);
        }
        let x = Tensor::from_vec(x_ids, Shape::from(n), &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let targets = Tensor::from_vec(target_ids, Shape::from(n), &device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let logits = self.inner.forward(&x, &mut self.memory_states, &mut self.program_states)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let log_sm = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let loss = candle_nn::loss::nll(&log_sm, &targets)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        if let Some(opt) = &mut self.optimizer {
            opt.backward_step(&loss)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        }

        let loss_val = loss.to_vec0::<f32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(loss_val)
    }
}

#[pyclass]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(json_path: String) -> PyResult<Self> {
        let inner = Tokenizer::from_file(json_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(Self { inner })
    }

    fn encode(&self, text: String) -> PyResult<Vec<u32>> {
        let encoding = self.inner.encode(text, true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let text = self.inner.decode(&ids, true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(text)
    }
}

#[pymodule]
fn titan_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTitanTransformer>()?;
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
