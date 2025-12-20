# ROADMAP — From tiny-gpt to GPT-4-level system

> Goal: evolve a minimal GPT into a **production-grade, GPT-4–class LLM system** in stages, with clear capability inflection points.

This roadmap assumes:

- progressive scaling (not a single “big bang”)
- strong emphasis on **reproducibility, evals, and control**

---

## Phase 0 — Educational Baseline (current state)

**Status:** `tiny-gpt`

**Purpose:** Full architectural understanding, correctness, and debuggability.

Capabilities:

- Decoder-only Transformer
- Causal self-attention
- Char-level tokenization
- Single-node training
- Basic sampling

Missing (by design):

- Robust evals
- Tokenization realism
- Performance optimizations
- Data pipeline rigor

References (baseline reading & minimal refs):

- **Transformers**: “Attention Is All You Need” (Vaswani et al., 2017): [paper](https://arxiv.org/abs/1706.03762)
- **LayerNorm**: “Layer Normalization” (Ba et al., 2016): [paper](https://arxiv.org/abs/1607.06450)
- **Adam/AdamW**: “Adam” (Kingma & Ba, 2014): [paper](https://arxiv.org/abs/1412.6980), “Decoupled Weight Decay Regularization (AdamW)” (Loshchilov & Hutter, 2017): [paper](https://arxiv.org/abs/1711.05101)
- **Reference implementations**: [karpathy/minGPT](https://github.com/karpathy/minGPT), [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

Exit criteria:

- Model is fully understood end-to-end
- Deterministic training runs (seeded, pinned deps, captured configs)
- Loss curves and generations are explainable

---

## Phase 1 — Research-Grade GPT (≈ GPT-2 class)

**Goal:** Become a serious language model, still single-node.

### Architecture (Phase 1)

- **Tokenization**: BPE / Unigram tokenizer
  - [openai/tiktoken](https://github.com/openai/tiktoken) (fast BPE)
  - [google/sentencepiece](https://github.com/google/sentencepiece) (BPE/unigram LM)
  - [huggingface/tokenizers](https://github.com/huggingface/tokenizers) (fast Rust tokenizers)
- **Positional encodings**: RoPE or ALiBi
  - RoPE: “RoFormer: Enhanced Transformer with Rotary Position Embedding” (Su et al., 2021): [paper](https://arxiv.org/abs/2104.09864)
  - ALiBi: “Train Short, Test Long: Attention with Linear Biases…” (Press et al., 2021): [paper](https://arxiv.org/abs/2108.12409)
- **Faster attention**: Flash/SDPA
  - “FlashAttention” (Dao et al., 2022): [paper](https://arxiv.org/abs/2205.14135), [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
  - PyTorch SDPA: [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- **KV cache for inference**
  - Reference: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (clean inference loop patterns)
  - (Serving-oriented) [vllm-project/vllm](https://github.com/vllm-project/vllm) (PagedAttention / KV management)
- **MLP**: SwiGLU
  - “GLU Variants Improve Transformer” (Shazeer, 2020): [paper](https://arxiv.org/abs/2002.05202)

### Training (Phase 1)

- Learning-rate warmup + cosine decay
  - Cosine LR: “SGDR” (Loshchilov & Hutter, 2016): [paper](https://arxiv.org/abs/1608.03983)
- Gradient accumulation
- Mixed precision (bf16 / fp16)
  - PyTorch AMP: [torch.cuda.amp](https://pytorch.org/docs/stable/amp.html)
- Checkpoint + resume (including RNG state)

### Evaluation (Phase 1)

- Perplexity tracking on held-out splits
- Fixed prompt regression tests (“golden prompts”)
- Seeded generation snapshots (tracked over commits)

Reference target:

- **GPT-2**: “Language Models are Unsupervised Multitask Learners” (Radford et al., 2019): [report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Exit criteria:

- Stable multi-hour training
- Comparable behavior across runs
- Meaningful eval signal (not vibes)

---

## Phase 2 — Production-Grade Foundation Model (≈ GPT-3 class)

**Goal:** Scale, reliability, and data discipline.

### Architecture (Phase 2)

- 1–10B parameters (progressive scaling)
- Deeper networks (48–96 layers)
- Attention optimizations (FlashAttention v2+)
  - FlashAttention-2: “FlashAttention-2: Faster Attention with Better Parallelism…” (Dao, 2023): [paper](https://arxiv.org/abs/2307.08691)
- Activation checkpointing
  - PyTorch checkpointing: [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)
- Weight tying where appropriate

### Data & Training (Phase 2)

- Large-scale text pipeline (cleaning, dedup, filtering)
  - “The Pile” dataset: [EleutherAI/the-pile](https://github.com/EleutherAI/the-pile)
  - “C4” (Colossal Clean Crawled Corpus): [paper](https://arxiv.org/abs/1910.10683)
  - Dedup reference: “Deduplicating Training Data Mitigates Privacy Risks…” (Lee et al., 2022): [paper](https://arxiv.org/abs/2202.06539)
- Distributed training (DDP / FSDP / ZeRO)
  - PyTorch DDP: [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
  - FSDP: [torch.distributed.fsdp](https://pytorch.org/docs/stable/fsdp.html)
  - ZeRO: “ZeRO: Memory Optimizations…” (Rajbhandari et al., 2020): [paper](https://arxiv.org/abs/1910.02054), [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- Token-based batching + packing (better utilization)
- Training run manifests (full provenance): configs, data hashes, code revs, environment

### Evaluation (Phase 2)

- Standard benchmarks (LM + reasoning)
  - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  - HELM (Holistic Evaluation of Language Models): [paper](https://arxiv.org/abs/2211.09110), [crfm/helm](https://github.com/stanford-crfm/helm)
- Long-context stress tests (needle-in-haystack, recall, instruction retention)
- Overfitting & memorization checks
  - Data extraction / memorization risks: “Extracting Training Data from Large Language Models” (Carlini et al., 2021): [paper](https://arxiv.org/abs/2012.07805)

### Ops (Phase 2)

- Experiment tracking (e.g., W&B / MLflow)
  - [mlflow/mlflow](https://github.com/mlflow/mlflow)
- Model cards + documentation
  - “Model Cards for Model Reporting” (Mitchell et al., 2019): [paper](https://arxiv.org/abs/1810.03993)
- Deterministic inference modes (seeded sampling, stable kernels where possible)

Reference target:

- **GPT-3**: “Language Models are Few-Shot Learners” (Brown et al., 2020): [paper](https://arxiv.org/abs/2005.14165)

Exit criteria:

- Multi-node training is routine
- Models are reproducible and auditable
- Clear performance vs scale curves

---

## Phase 3 — GPT-4-Class System (this is the real jump)

> GPT-4 is not “just a bigger GPT-3”. It is a **system**, not only a model.

### Model Architecture

- Mixture-of-Experts (MoE) and/or sparse routing
  - Switch Transformer (MoE): [paper](https://arxiv.org/abs/2101.03961)
  - GShard (sparse scaling): [paper](https://arxiv.org/abs/2006.16668)
  - DeepSpeed-MoE: [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- 100B+ effective parameters (via MoE/sparsity, parallelism, scaling laws)
- Long-context support (32k–128k+)
  - Transformer-XL (recurrence / long dependencies): [paper](https://arxiv.org/abs/1901.02860)
  - RoPE scaling strategies (community + open implementations vary); track known approaches and test rigorously
- Possibly multimodal encoders (if desired)
  - CLIP: [paper](https://arxiv.org/abs/2103.00020)
  - Flamingo: [paper](https://arxiv.org/abs/2204.14198)
  - LLaVA (open multimodal): [paper](https://arxiv.org/abs/2304.08485), [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)

### Training Regime

- Massive curated datasets (text + code + structured data, with strict provenance)
- Curriculum learning (where it helps; avoid cargo-culting)
- Continual pretraining / refresh cycles
- Fine-grained loss shaping (domain mixes, length mixes, safety mixes)

### Post-Training (critical)

- Supervised fine-tuning (SFT)
  - InstructGPT (SFT + RLHF pipeline): [paper](https://arxiv.org/abs/2203.02155)
- Preference optimization (RLHF / DPO / RLAIF)
  - RLHF background: [paper](https://arxiv.org/abs/1706.03741) (Christiano et al., 2017)
  - DPO: “Direct Preference Optimization” (Rafailov et al., 2023): [paper](https://arxiv.org/abs/2305.18290)
  - RLAIF / Constitutional AI: [paper](https://arxiv.org/abs/2212.08073), [paper](https://arxiv.org/abs/2305.11206)
- Safety & refusal training + policy shaping (evaluate tradeoffs)
- Capability shaping (instruction-following, tool-use, style, verbosity controls)

### Evaluation (non-negotiable)

- Task-based evals (reasoning, coding, math)
  - MMLU: [paper](https://arxiv.org/abs/2009.03300)
  - GSM8K: [paper](https://arxiv.org/abs/2110.14168)
  - HumanEval: [paper](https://arxiv.org/abs/2107.03374)
- Red-team evals + adversarial testing
  - “Red Teaming Language Models…” (Ganguli et al., 2022): [paper](https://arxiv.org/abs/2202.03286)
- Regression detection (capability + safety) as a first-class CI gate
- Long-horizon coherence tests (agents, multi-step tasks, tool loops)

### Inference System

- Speculative decoding (reduce latency/cost)
  - “Speculative Decoding” (Leviathan et al., 2023): [paper](https://arxiv.org/abs/2211.17192)
- MoE-aware routing + batching
- Deterministic / auditable modes (traceability, stable decoding)
- Cost-aware serving (quotas, caching, batching, latency budgets)
  - vLLM: [vllm-project/vllm](https://github.com/vllm-project/vllm)
  - Hugging Face TGI: [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)
  - TensorRT-LLM: [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

### Governance & Control

- Training data lineage (what went in, when, and why)
  - “Datasheets for Datasets” (Gebru et al., 2018): [paper](https://arxiv.org/abs/1803.09010)
- Model versioning + system cards / release notes
- Capability gating (rollout controls, tiering, policy checks)
- Rollback & kill-switches (operational safety)

Exit criteria:

- The system is operationally safe (monitored, rate-limited, rollbackable)
- Behavior is shaped, not accidental
- Performance improvements are intentional and explainable

---
