# Titan: Learning to Memorize

Titan is a neural network project implementing the "Titans" architecture, which incorporates a neural long-term memory module alongside standard attention and internal program emulation.

## Architecture

The Titan architecture consists of:
- **Neural Long-Term Memory (TitansMemory):** A persistent matrix-based memory using a Multi-Head Delta-rule update with learnable "surprise" gating and context-aware decay.
- **Sliding Window Attention (SWA):** Optimized Grouped-Query Attention (GQA) with a 512-token local window and 8-bit quantized KV-cache for linear context efficiency.
- **Sparse Mixture of Experts (MoE):** High-capacity FFN replacement featuring dynamic Top-1 routing and a Shared Expert for improved performance/parameter ratio.
- **Python Emulator (PythonEmulator):** Models internal program state updates using a bottleneck GRU mechanism for semantic instruction processing.
- **Block Attention Residuals:** Vectorized Mini-Cross-Attention mechanism to aggregate information across layer blocks.
- **Modern Refinements:** NTK-aware RoPE scaling (base 50,000), LayerScale, learnable Logit Softcapping, and weight tying.

## Components

### Rust Core (`titan_core`)
The core of the model is implemented in Rust using the `candle` framework for high performance.
- `src/titans.rs`: Neural memory implementation.
- `src/emulator.rs`: Program state emulator.
- `src/attn_res.rs`: Vectorized attention residuals.
- `src/model.rs`: Main Transformer and Layer definitions.
- `src/quant.rs`: Polar and QJL quantization.

### Python Bindings (`titan_py`)
The Rust core is exposed to Python via PyO3, providing a high-level `PyTitanTransformer` class.

## Usage

### Installation
1. Ensure you have Rust and Python installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Build the Rust extension:
   ```bash
   python setup.py build_rust --inplace
   ```

### Basic Example
```python
import titan_core as titan

# Initialize model
vocab_size = 1000
dim = 256
num_layers = 4
model = titan.TitanTransformer(vocab_size, dim, num_layers)

# Forward pass
input_ids = [1, 2, 3, 4, 5]
output = model.forward(input_ids)

# Reset internal state (memory and program states)
model.reset_state()
```

## Testing
Run the provided smoke tests and training examples:
```bash
python tests/smoke_test.py
python tests/train_example.py
python tests/train_complete.py
```
