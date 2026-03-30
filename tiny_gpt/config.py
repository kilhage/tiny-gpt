from dataclasses import dataclass


# -------------------------
# Config
# -------------------------
@dataclass
class GPTConfig:
    # data
    batch_size: int = 64
    block_size: int = 256  # context length
    # model
    n_layer: int = 6

    # normal multi-head attention
    n_head: int = 8

    # GQA
    n_q_head: int = 8
    n_kv_head: int = 4

    n_embd: int = 384
    dropout: float = 0.1
    # training
    max_steps: int = 3000
    eval_interval: int = 250
    eval_batches: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # runtime
    device: str = "auto"  # "auto" | "mps" | "cpu"
