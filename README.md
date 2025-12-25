# tiny-gpt

A minimal, educational **GPT-style (decoder-only) Transformer** you can train locally (including on **Apple Silicon via PyTorch MPS**) to learn the core mechanics:

- **Token + positional embeddings**
- **Causal self-attention** (masked self-attention)
- **Pre-LayerNorm Transformer blocks** with residual connections
- **MLP feed-forward** (4× expansion)
- **Next-token prediction** training loop + evaluation
- **Sampling** (temperature + top-k)

This repo intentionally stays small and readable rather than fast or feature-complete.

---

## Repo layout

- **`train.py`**: training script (reads `data.txt`, trains, saves `tiny_gpt.pt`).
- **`tiny_gpt/model.py`**: GPT model + tokenizer/data helpers + sampling (`generate`).
- **`tiny_gpt/trainer.py`**: evaluation helpers (loss estimation).
- **`tiny_gpt/generate.py`**: loads `tiny_gpt.pt` and generates text from a prompt.
- **`tiny_gpt/dataset.py`**: downloads a starter dataset (Tiny Shakespeare) to `data.txt`.
- **`data.txt`**: your training corpus (plain UTF-8 text).
- **`pyproject.toml` / `uv.lock`**: Python deps (works great with `uv`).

---

## Setup

This project targets **Python 3.11+**.

### Use `uv` (recommended)

```bash
cd tiny-gpt
uv sync
```

Run Python through `uv`:

```bash
uv run python -c "import torch; print('mps?', torch.backends.mps.is_available())"
```

If `mps? True` prints, PyTorch can use your Mac GPU via Metal.

### Troubleshooting (PyTorch)

If you see an error like:

```bash
ImportError: cannot import name '_initExtension' from 'torch._C'
```

it usually means your PyTorch install is **corrupted** (Python is importing `torch/_C/` as a package instead of the compiled `torch._C` extension).

The most reliable fix is to recreate the virtual environment:

```bash
cd tiny-gpt
rm -rf .venv
uv cache clean
uv sync
uv run python -c "import torch; print(torch.__version__); print('mps?', torch.backends.mps.is_available())"
```

If you want to double-check you’re not running under Rosetta, confirm:

```bash
uv run python -c "import platform; print(platform.machine())"
```

It should print `arm64` on Apple Silicon.

---

## Dataset (`data.txt`)

Training is **character-level**: every unique character in `data.txt` becomes a token in the vocabulary.

### Quick download (Tiny Shakespeare)

Fetch a starter corpus (~1 MB) to `data.txt`:

```bash
# uv (recommended)
uv run python dataset.py
```

Flags: `--force` re-downloads even if `data.txt` exists, `--output` changes the target path, `--url` lets you point at another text source.

### Bring your own text

- **Encoding**: make sure `data.txt` is UTF-8.
- **Size**: even a few hundred KB works; 1–5 MB is plenty for learning.
- **Quality matters**: coherent, consistently-formatted text yields much better generations.

---

## Train

`train.py` reads `data.txt`, trains for `GPTConfig.max_steps`, evaluates periodically, then saves a checkpoint to `tiny_gpt.pt` and prints a sample generation.

```bash
# uv
uv run python train.py
```

Output includes:

- chosen device (MPS if available, else CPU)
- periodic train/val loss
- a final saved checkpoint: **`tiny_gpt.pt`**
- a demo generation from the prompt `"Hello"`

---

## Generate from a saved checkpoint

After training, run:

```bash
# uv
uv run python tiny_gpt/generate.py -p "Hello"
```

`tiny_gpt/generate.py` loads:

- `model_state`: weights
- `config`: the `GPTConfig` values used for training
- `stoi`/`itos`: the character vocabulary

Then it samples tokens autoregressively using `GPT.generate(...)` with **temperature** and **top-k** filtering.

---

## How the model works (mental map)

### Data flow

- **Tokenizer**: `CharTokenizer` builds `stoi`/`itos` from the set of characters in `data.txt`.
- **Training objective**: next-token prediction.
  - inputs `x`: a sequence of length `block_size`
  - targets `y`: the same sequence shifted by 1 character

### Model architecture (decoder-only Transformer)

- **Embeddings**:
  - token embedding: `tok_emb`
  - positional embedding: `pos_emb`
  - summed + dropout
- **Transformer blocks** (`n_layer` times):
  - Pre-LN → masked multi-head self-attention → residual add
  - Pre-LN → MLP (4× hidden) → residual add
- **Final LayerNorm** → **LM head** → logits over vocab
- **Weight tying**: LM head weight is shared with token embeddings (`lm_head.weight = tok_emb.weight`).

### Causal self-attention

Attention uses a **lower-triangular (causal) mask** so token \(t\) can only attend to \(\le t\).

---

## Configuration tips (speed vs quality)

Edit `GPTConfig` in `tiny_gpt/config.py`.

- **Faster**:
  - reduce `block_size` (e.g. 256 → 128)
  - reduce `n_layer` (e.g. 6 → 4)
  - reduce `batch_size` (e.g. 64 → 32)
- **Better quality**:
  - increase `max_steps` (e.g. 10k–50k)
  - use more and better training text in `data.txt`

If you hit MPS memory issues, start by lowering `batch_size` and/or `block_size`.

---

## Next steps (high-leverage upgrades)

Below are realistic upgrades that keep the project educational, but move it closer to “real GPT”.

See [ROADMAP.md](docs/ROADMAP.md) for a focused roadmap moving towards GPT-4 capabilities.

### Use PyTorch scaled dot-product attention (cleaner + often faster)

PyTorch provides `torch.nn.functional.scaled_dot_product_attention(...)`, which can simplify attention code and may improve performance depending on backend.

- PyTorch docs: [Scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

### Training improvements that teach “real” tricks

- **Learning-rate schedules** (warmup + cosine decay)
- **Gradient accumulation** (simulate larger batch sizes)
- **Mixed precision** (where supported)
- **Checkpointing** and **resume training**
- **Logging** (TensorBoard / Weights & Biases)

### Better datasets (starter-friendly)

Small, classic, easy-to-use corpora:

- **Tiny Shakespeare** (classic for learning GPT mechanics): [karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
- **WikiText** (language modeling benchmark datasets): [WikiText on Hugging Face](https://huggingface.co/datasets/wikitext)

Larger web-scale datasets exist, but can be heavy and have licensing/usage considerations; start small and graduate intentionally.

### More learning resources / inspiration

- **“Attention Is All You Need”** (original Transformer paper): [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **The Illustrated Transformer** (excellent intuition): [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- **The Illustrated GPT-2**: [Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
- **Andrej Karpathy’s “Let’s build GPT”** (video + code walkthrough): [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)

---

## License / data note

Be mindful of what you put in `data.txt`. Use text you own, have rights to, or that is clearly licensed for your intended use.
