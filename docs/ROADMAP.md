# tiny-gpt: Roadmap

## Milestone 1

### Tokenization (move beyond character-level)

WHY should we do this?

- **tiktoken** (fast BPE used by OpenAI-style tokenizers): [openai/tiktoken](https://github.com/openai/tiktoken)
- **SentencePiece** (BPE/unigram LM, widely used): [google/sentencepiece](https://github.com/google/sentencepiece)
- **Hugging Face tokenizers** (fast Rust tokenizers): [huggingface/tokenizers](https://github.com/huggingface/tokenizers)

### Faster generation: KV cache

Right now generation recomputes attention over the whole context each step. Adding a **KV cache** makes generation much faster by reusing previous keys/values.

Reference implementations:

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (clean, educational)
- [karpathy/minGPT](https://github.com/karpathy/minGPT) (even smaller / teaching-focused)
