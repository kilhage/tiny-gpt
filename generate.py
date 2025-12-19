import torch
import torch.nn.functional as F

# must match tiny_gpt.py class names, so simplest is: import them
from tiny_gpt import GPT, CharTokenizer, GPTConfig


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load("tiny_gpt.pt", map_location=device)
    cfg = GPTConfig(**ckpt["config"])

    tokenizer = CharTokenizer(" ")  # dummy init
    tokenizer.stoi = ckpt["stoi"]
    tokenizer.itos = ckpt["itos"]
    tokenizer.vocab_size = len(tokenizer.stoi)

    model = GPT(cfg, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt = "Once upon a time"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=400, temperature=0.9, top_k=50)[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
