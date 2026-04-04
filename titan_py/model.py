from typing import List, Optional
from ._titan_core import PyTitanTransformer, PyTokenizer

class TitanTransformer:
    """
    High-level wrapper for the Titan Transformer model.

    This class provides a user-friendly interface to the underlying Rust implementation,
    managing long-term memory states and KV-caches automatically.

    Args:
        vocab_size (int): Size of the vocabulary.
        dim (int): Hidden dimension of the model.
        num_layers (int): Number of Transformer layers.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        num_kv_heads (int, optional): Number of KV heads for GQA. Defaults to 2.
        window_size (int, optional): Size of the sliding attention window. Defaults to 512.
        block_size (int, optional): Number of layers per residual block. Defaults to 4.
        m_size (int, optional): Number of persistent memory tokens. Defaults to 8.
        num_experts (int, optional): Number of dynamic experts in MoE. Defaults to 4.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        window_size: Optional[int] = None,
        block_size: Optional[int] = None,
        m_size: Optional[int] = None,
        num_experts: Optional[int] = None,
    ):
        self._inner = PyTitanTransformer(
            vocab_size,
            dim,
            num_layers,
            num_heads,
            num_kv_heads,
            window_size,
            block_size,
            m_size,
            num_experts
        )

    def init_optimizer(self, lr: float) -> None:
        """Initializes the AdamW optimizer with the given learning rate."""
        self._inner.init_optimizer(lr)

    def save_weights(self, path: str) -> None:
        """Saves model weights to a .safetensors file."""
        self._inner.save_weights(path)

    def load_weights(self, path: str) -> None:
        """Loads model weights from a .safetensors file."""
        self._inner.load_weights(path)

    def reset_state(self) -> None:
        """Resets the internal long-term memory and program state."""
        self._inner.reset_state()

    def forward(self, input_ids: List[int]) -> List[float]:
        """Performs a forward pass given a list of input token IDs."""
        return self._inner.forward(input_ids)

    def train_step(self, input_ids: List[int], target_ids: List[int]) -> float:
        """Performs a single training step and returns the loss."""
        return self._inner.train_step(input_ids, target_ids)

class Tokenizer:
    """
    High-level wrapper for the project tokenizer.
    """
    def __init__(self, json_path: str):
        self._inner = PyTokenizer(json_path)

    def encode(self, text: str) -> List[int]:
        """Encodes text into a list of token IDs."""
        return self._inner.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs into text."""
        return self._inner.decode(ids)
