import numpy as np
import torch

from tiny_gpt.config import GPTConfig


# -------------------------
# Utilities
# -------------------------
def get_device(cfg: GPTConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "mps":
        return torch.device("mps")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
