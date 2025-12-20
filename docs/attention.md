# Transformer Architecture and Self-Attention

## Purpose & Scope

This document is written as a **conceptual guide to Transformer self-attention**, intended to build a durable mental model rather than provide implementation details or mathematical derivations.

The goal is to:

- understand why attention exists and how it evolved,
- grasp how self-attention works at a systems level,
- distinguish core attention mechanisms from engineering optimizations,
- and connect academic concepts to the architectures used in modern large language models (GPT-4/5, Claude, Gemini, LLaMA, Qwen, etc.).

This document intentionally prioritizes:

- intuition over equations,
- architectural tradeoffs over code,
- and real-world usage over exhaustive taxonomy.

It is meant to complement—not replace—primary research papers and implementations.

## Research

- [Understanding Transformers and Modern Attention Mechanisms](https://chatgpt.com/s/dr_6946dd97a00081918e9cbf892d61d0fe)

## Background

Attention began as a practical solution to a bottleneck in early neural sequence-to-sequence models: classic encoder–decoder RNNs had to compress an entire input (like a source sentence) into a single fixed-size vector, which degraded badly on long or information-dense sequences. The breakthrough was to let the decoder **dynamically “look back”** at the encoder’s hidden states at each generation step via a learned soft alignment. This idea first crystallized in **additive attention** (Bahdanau et al., 2015) and was soon streamlined into **dot-product / multiplicative attention** (Luong et al., 2015), which made the same core concept more hardware-friendly. In today’s language, these works established the key abstraction: compute relevance scores between a **query** and a set of **keys**, turn scores into weights (softmax), and use those weights to mix **values** into a context vector.
Papers: Bahdanau et al. ([https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)), Luong et al. ([https://arxiv.org/abs/1508.04025](https://arxiv.org/abs/1508.04025))

The Transformer reframed attention from “decoder attending to encoder” into **self-attention**, where each token in a sequence can attend to other tokens in the same sequence. In **Attention Is All You Need** (Vaswani et al., 2017), attention became the central compute primitive: **scaled dot-product attention** plus **multi-head attention** enabled parallel training (no recurrence), richer feature mixing (different heads can specialize), and a clean repeated building block that scaled well. Once this worked, the next wave of innovation focused on two pressure points: (1) **position**, because attention alone is order-agnostic (leading to relative position methods and later RoPE/ALiBi-style schemes), and (2) **cost**, because naïve attention is quadratic in sequence length.
Papers: Vaswani et al. ([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)), Shaw et al. relative positions ([https://arxiv.org/abs/1803.02155](https://arxiv.org/abs/1803.02155)), RoPE ([https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)), ALiBi ([https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409))

From ~2020 onward, the story splits into two tracks: research into **sparse/approximate attention** (Longformer, BigBird, Linformer, Performer, Reformer) to reduce the O(n²) wall for very long sequences, and—arguably more impactful for frontier LLMs—**engineering that keeps attention exact but makes it fast and memory-efficient**. Architectural tweaks like **Multi-Query Attention** reduce KV-cache overhead during generation; kernel-level advances like **FlashAttention / FlashAttention-2** make the same attention computation dramatically more IO-efficient on GPUs; and serving-time systems like **PagedAttention** manage KV cache memory to sustain long contexts and high concurrency. The net result is that modern LLMs still rely on the Transformer’s scaled dot-product attention at the core—but differentiate by _how efficiently they compute it_, _how they represent position_, and _how they make long context practical in training and serving_.
Papers: Longformer ([https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)), BigBird ([https://arxiv.org/abs/2007.14062](https://arxiv.org/abs/2007.14062)), Linformer ([https://arxiv.org/abs/2006.04768](https://arxiv.org/abs/2006.04768)), Performer ([https://arxiv.org/abs/2009.14794](https://arxiv.org/abs/2009.14794)), Reformer ([https://arxiv.org/abs/2001.04451](https://arxiv.org/abs/2001.04451)), MQA ([https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)), FlashAttention ([https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)), FlashAttention-2 ([https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)), PagedAttention/vLLM ([https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180))

## High-Level Overview

**Self-attention** is a mechanism that lets each token in a sequence decide which other tokens matter most when building its representation. Instead of processing tokens strictly left-to-right (like RNNs), the Transformer looks at the entire sequence at once and computes a new representation for each token by mixing information from other tokens, weighted by relevance. This is what gives Transformers their ability to model long-range dependencies, parallelize computation, and scale so effectively.

The core idea is simple: every token is projected into three vectors—**Query (Q), Key (K), and Value (V)**. You can think of the query as “what this token is looking for,” the key as “what this token offers,” and the value as “the information this token contains.” For a given token, its query is compared against the keys of all tokens in the sequence to compute relevance scores. These scores are normalized with a softmax to produce attention weights, which are then used to take a weighted sum of the value vectors. The result is a context-aware representation: each token becomes a blend of other tokens, emphasizing those that are most relevant. Importantly, this operation is **content-based**—tokens attend to others based on meaning, not distance alone.

Transformers apply this mechanism **in parallel for every token**, which is why it’s called self-attention. In practice, this happens across multiple **attention heads**. Each head learns a different way to compare queries and keys, allowing the model to attend to different patterns at the same time—such as syntax, long-range dependencies, or semantic similarity. The outputs of all heads are concatenated and mixed, giving the model a richer, more expressive representation than a single attention computation could provide. To make this work for ordered data like text, positional information (via positional embeddings or RoPE/ALiBi-style methods) is injected so attention can distinguish between “nearby” and “far away” tokens.

In **causal self-attention**, used by generative language models like GPT, an additional constraint is applied: each token is only allowed to attend to earlier tokens (and itself). This prevents the model from “seeing the future” during training or generation and ensures the model learns to predict the next token autoregressively. Aside from this masking, the mechanism is the same. A Transformer block then stacks this self-attention layer with a feed-forward network, residual connections, and normalization. Repeating this block many times allows information to be refined layer by layer—from local patterns in early layers to more abstract reasoning in later ones. At scale, this simple idea—repeated content-based mixing via self-attention—is what powers modern LLMs.

## Attention Variants & Optimizations

| Name                                                   | Description                                                                                                                                                                        | Benefits                                                                                       | When to Use                                                                                                            |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Scaled Dot-Product Self-Attention (Standard Attention) | The classic attention mechanism from "Attention Is All You Need"; computes dot products between queries and keys, applies a scaling factor, softmax, and weighted sum over values. | Simple, well-understood, direct baseline for research and education.                           | Default for educational, research, and small models; when transparency and debugging are priorities.                   |
| FlashAttention                                         | An optimized attention algorithm that uses tiling and kernel fusion to minimize memory reads/writes and exploit faster compute, usually on GPUs.                                   | Greatly reduces memory usage, speeds up attention computation, especially with long sequences. | For large models, long contexts, or when training/generating needs to be fast (requires compatible hardware/backends). |
| FlashAttention-2                                       | Improved version of FlashAttention with better performance and support for longer sequences and more model variants; further reduces memory bottlenecks.                           | Higher throughput, lower memory — supports even larger contexts, better hardware utilization.  | When maximum efficiency is needed in state-of-the-art training/inference, especially with long-context LLMs.           |
| PagedAttention                                         | A memory-efficient method for managing Key/Value caches, by organizing them into pages, often used for serving multiple prompts with long sequences.                               | Enables serving many concurrent users, very large contexts (up to millions of tokens).         | In production LLM serving, especially with multi-user, long-prompt cases (e.g. vLLM).                                  |
| Multi-Query Attention                                  | Uses a single set of key/value heads for all attention heads (instead of separate k/v heads per query head), reducing memory/computation.                                          | Reduces KV cache size and speeds up inference, with minimal quality loss in large models.      | When scaling up to billions of parameters, especially for LLM deployment and inference speed.                          |

## Attention Mechanisms in Major LLMs (Overview)

Most state-of-the-art language models—including OpenAI’s GPT-4/5, Anthropic’s Claude, Google DeepMind Gemini, Meta’s LLaMA, Alibaba’s Qwen, and others—are built around the core Transformer architecture and use scaled dot-product multi-head self-attention as their foundational mechanism. Recent advances primarily focus on improving efficiency, handling much longer contexts, and reducing memory/computation costs, rather than changing the underlying attention formula.

- **OpenAI GPT-4/GPT-5**: Use standard causal multi-head attention, likely with modern optimizations such as FlashAttention-style kernels and possibly Multi-Query or Grouped-Query Attention for faster inference. Context lengths have expanded dramatically—up to 32k/128k tokens—enabled by these optimizations and advanced positional encoding strategies. Retrieval and cross-attention may appear in future models, but core attention remains unchanged.

- **Anthropic Claude (e.g. Claude 2, 3, Opus, Sonnet)**: Also use Transformer attention; Claude’s standout feature is extremely large context windows (up to 100k tokens), which are probably realized using efficient attention kernels (FlashAttention or similar), memory management strategies (PagedAttention-like KV caches), and selective/sparse attention to older tokens for feasibility.

- **Google DeepMind Gemini**: Follows trends set by PaLM (which used Grouped Query Attention—GQA—for memory reduction and speed) and employs highly optimized attention kernels. Likely incorporates cross-attention for multimodal data and advanced positional encoding (e.g., RoPE, ALiBi) for handling long contexts.

- **Meta LLaMA 3/4**: Like earlier LLaMAs, expected to continue using causal multi-head attention but with upgrades such as FlashAttention, possibly GQA, and longer context support. Rotary position encodings (RoPE) and efficient training kernels are standard; overall architecture is conservative but incorporates efficiency improvements.

- **Alibaba Qwen 2.5/3**: Among the most innovative, with Grouped Query Attention, Dual Chunk Attention (hierarchical chunk-based long-range attention), adaptive RoPE for very long contexts (up to 128k tokens), and explicit use of FlashAttention kernels. Qwen’s techniques let it scale input length and maintain speed/efficiency.

- **Other models (e.g., DeepSeek)**: Also rely on Transformer attention with optimizations for efficiency, long-context, or retrieval-augmented strategies, but fundamentally retain the same scaled dot-product attention.

**Summary Table:**

| Model/Family | Attention Mechanism              | Key Optimizations / Features                | Max Context (public) | Notes                                                         |
| ------------ | -------------------------------- | ------------------------------------------- | -------------------- | ------------------------------------------------------------- |
| GPT-4/5      | Causal multi-head, SDPA          | FlashAttention(-2), MQA/GQA, long context   | 32k–128k             | Likely uses highly optimized kernels and MQA/GQA              |
| Claude       | Causal multi-head, SDPA          | FlashAttention, paging, possible sparsing   | 100k                 | Memory management for KV cache, efficient long-context tricks |
| Gemini       | Causal multi-head (+cross-modal) | GQA, FlashAttention, RoPE/ALiBi             | 32k+?                | Multi-modal (text-vision); cross-attention for modalities     |
| LLaMA 3/4    | Causal multi-head, SDPA          | FlashAttention, RoPE, possible GQA          | 8k–32k (est.)        | Stable, open; context extension possible via RoPE tunings     |
| Qwen 2.5/3   | GQA, Dual Chunk SDPA             | Dual Chunk Attention, adaptive RoPE         | 128k                 | Most aggressive context scaling, group heads for speed        |
| DeepSeek     | Causal multi-head, SDPA          | Likely FlashAttention, retrieval/cross-attn | 32k+ (est.)          | May use retrieval-augmented attention                         |

**Bottom line:**  
All major LLMs use scaled dot-product attention as the foundation, with mask-based causality for generation. They differentiate via engineering—faster kernels (FlashAttention), memory-efficient KV management (Paged/PagedAttention), more efficient head arrangements (MQA/GQA), smarter positional encodings, and context window scaling—rather than radical changes to attention itself. Mastering Transformer attention plus these enhancements covers >95% of what powers today’s top LLMs.
