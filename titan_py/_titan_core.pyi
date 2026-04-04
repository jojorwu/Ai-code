from typing import List, Optional

class PyTitanTransformer:
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
        global_attn_period: Optional[int] = None,
        use_weight_std: Optional[bool] = None,
        use_turbo_quant: Optional[bool] = None,
        use_differential_attn: Optional[bool] = None,
        use_mla: Optional[bool] = None,
        kv_lora_rank: Optional[int] = None,
        qk_lora_rank: Optional[int] = None,
        use_aux_loss_free_lb: Optional[bool] = None,
        drop_path_rate: Optional[float] = None,
    ) -> None: ...
    def init_optimizer(self, lr: float) -> None: ...
    def save_weights(self, path: str) -> None: ...
    def load_weights(self, path: str) -> None: ...
    def reset_state(self) -> None: ...
    def forward(self, x_ids: List[int]) -> List[float]: ...
    def train_step(self, x_ids: List[int], target_ids: List[int]) -> float: ...

class PyTokenizer:
    def __init__(self, json_path: str) -> None: ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...
