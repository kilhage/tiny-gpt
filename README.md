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

- **`tiny_gpt.py`**: the whole project in one file (model, data batching, training loop, sampling, checkpoint save).
- **`generate.py`**: loads `tiny_gpt.pt` and generates text from a prompt.
- **`data.txt`**: your training corpus (plain UTF-8 text).
- **`pyproject.toml` / `uv.lock`**: Python deps (works great with `uv`).
- **`main.py`**: currently a placeholder “hello world” script (not used for training).

---

## Setup

This project targets **Python 3.11+**.

### Option A: `uv` (recommended)

```bash
cd tiny-gpt
uv sync
```

Run Python through `uv`:

```bash
uv run python -c "import torch; print('mps?', torch.backends.mps.is_available())"
```

### Option B: venv + pip

```bash
cd tiny-gpt
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch numpy
python -c "import torch; print('mps?', torch.backends.mps.is_available())"
```

If `mps? True` prints, PyTorch can use your Mac GPU via Metal.

---

## Dataset (`data.txt`)

Training is **character-level**: every unique character in `data.txt` becomes a token in the vocabulary.

- **Encoding**: make sure `data.txt` is UTF-8.
- **Size**: even a few hundred KB works; 1–5 MB is plenty for learning.
- **Quality matters**: coherent, consistently-formatted text yields much better generations.

---

## Train

`tiny_gpt.py` reads `data.txt`, trains for `GPTConfig.max_steps`, evaluates periodically, then saves a checkpoint to `tiny_gpt.pt` and prints a sample generation.

```bash
# uv
uv run python tiny_gpt.py

# or venv/pip
python tiny_gpt.py
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
uv run python generate.py

# or venv/pip
python generate.py
```

`generate.py` loads:

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

Edit `GPTConfig` in `tiny_gpt.py`.

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

### Tokenization (move beyond character-level)

- **tiktoken** (fast BPE used by OpenAI-style tokenizers): [openai/tiktoken](https://github.com/openai/tiktoken)
- **SentencePiece** (BPE/unigram LM, widely used): [google/sentencepiece](https://github.com/google/sentencepiece)
- **Hugging Face tokenizers** (fast Rust tokenizers): [huggingface/tokenizers](https://github.com/huggingface/tokenizers)

### Faster generation: KV cache

Right now generation recomputes attention over the whole context each step. Adding a **KV cache** makes generation much faster by reusing previous keys/values.

Reference implementations:

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (clean, educational)
- [karpathy/minGPT](https://github.com/karpathy/minGPT) (even smaller / teaching-focused)

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
