import torch

from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT, TextData


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def estimate_loss(model: GPT, data: TextData, cfg: GPTConfig, device: torch.device):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_batches)
        for i in range(cfg.eval_batches):
            xb, yb = data.get_batch(split, cfg.batch_size, cfg.block_size, device)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
