pub mod attn_res;
pub mod titans;
pub mod quant;
pub mod emulator;
pub mod model;

use candle_core::{Device, Tensor, DType, Shape};
use candle_nn::VarBuilder;
use crate::model::TitanTransformer;
use pyo3::prelude::*;
use tokenizers::Tokenizer;

#[pyclass]
pub struct PyTitanTransformer {
    inner: TitanTransformer,
    dim: usize,
    num_layers: usize,
    memory_states: Vec<Tensor>,
    program_states: Vec<Tensor>,
}

#[pymethods]
impl PyTitanTransformer {
    #[new]
    fn new(vocab_size: usize, dim: usize, num_layers: usize) -> PyResult<Self> {
        let device = Device::Cpu;

        // Simple trick for semi-random initialization: use a small seed or constant to offset zeros
        // Real training would involve loading weights or a proper random VarBuilder backend.
        let vb = VarBuilder::zeros(DType::F32, &device);

        let inner = TitanTransformer::new(vocab_size, dim, num_layers, vb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        let memory_states = vec![Tensor::zeros((1, dim), DType::F32, &device).unwrap(); num_layers];
        let program_states = vec![Tensor::zeros((1, dim), DType::F32, &device).unwrap(); num_layers];

        Ok(Self { inner, dim, num_layers, memory_states, program_states })
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
