import time

import torch

from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT, CharTokenizer, TextData
from tiny_gpt.trainer import estimate_loss
from tiny_gpt.utils import get_device, set_seed


def main():
    cfg = GPTConfig()
    set_seed(1337)

    # read dataset
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = TextData(text, tokenizer)

    device = get_device(cfg)
    print(f"Device: {device} | vocab_size={tokenizer.vocab_size}")

    model = GPT(cfg, vocab_size=tokenizer.vocab_size).to(device)

    # AdamW optimizer (standard for transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    t0 = time.time()
    for step in range(1, cfg.max_steps + 1):
        xb, yb = data.get_batch("train", cfg.batch_size, cfg.block_size, device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping helps stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == 1:
            losses = estimate_loss(model, data, cfg, device)
            dt = time.time() - t0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | elapsed {dt:.1f}s"
            )

    # Save checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
    }
    torch.save(ckpt, "tiny_gpt.pt")
    print("Saved: tiny_gpt.pt")

    # Demo generation
    prompt = "Hello"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=400, temperature=0.9, top_k=50)[0].tolist()
    print("\n--- SAMPLE ---")
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
