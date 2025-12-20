# Transformer Architecture and Self-Attention

_A mental model_

---

## Purpose & Scope

This document is written as a **conceptual guide to Transformer self-attention**, intended to build a durable mental model rather than provide implementation details or mathematical derivations.

The goal is to:

- understand **why attention exists** and how it evolved,
- form an intuitive picture of **what self-attention is actually doing**,
- distinguish **core attention mechanics** from **engineering optimizations**,
- and connect research ideas to the architectures used in modern LLMs (GPT-4/5, Claude, Gemini, LLaMA, Qwen, etc.).

This document intentionally prioritizes:

- intuition over equations,
- architectural tradeoffs over code,
- and real-world usage over exhaustive taxonomy.

It is meant to complement—not replace—papers and code.

---

## The mental model

> **Self-attention is content-addressable memory lookup over the tokens in the context window.**

- Each token asks: _“Which other tokens are relevant to me right now?”_
- Attention computes **who to look at** and **how much to copy**.
- The output is a **weighted mix of information from other tokens**.

**Invariant (the most important thing):**

> **Attention does not create new information. It only routes and mixes information.**
> New information is created by the **MLP / feed-forward network**, not attention.

This is why Transformers alternate Attention (mix) and MLP (transform), layer after layer.

---

## The Transformer loop (keep this in mind)

**A Transformer repeatedly:**

1. **routes information** across tokens (attention),
2. **transforms information** locally (MLP),
3. **accumulates updates** in a persistent per-token state (residual stream).

Everything else in this document is an explanation of one part of this loop.

> - Attention decides **what to look at**,
> - MLP decides **how to change it**,
> - Residuals decide **what persists**.

---

## Background: why attention exists

Early encoder–decoder RNNs (Recurrent Neural Network) compressed an entire input sequence into a single vector. This created a severe bottleneck: long or information-dense inputs simply didn’t fit.

Attention fixed this by letting the decoder **dynamically look back** at relevant encoder states instead of relying on a single summary vector. Early forms (Bahdanau, Luong) introduced the core abstraction still used today:

- compute relevance scores between a **query** and a set of **keys**,
- normalize those scores into weights,
- use the weights to combine **values**.

The Transformer made a decisive leap: instead of “decoder attends to encoder”, **everything attends to everything**.
In _Attention Is All You Need_ (2017), attention became the **central compute primitive**, enabling:

- parallel processing (no recurrence),
- global information flow at every layer,
- clean scaling via repetition.

Two problems immediately emerged:

1. **Position** – attention is order-agnostic.
2. **Cost** – naïve attention is quadratic in sequence length.

Since then, progress has focused not on replacing attention, but on making it _practical at scale_.

---

## What self-attention does

Think of the sequence as a shared working memory. Each token produces:

- **Q (Query)**: what I’m looking for
- **K (Key)**: what I am / what I match on
- **V (Value)**: what I contain / what can be copied

For each token:

1. compare its **Q** against all **K** → relevance scores
2. normalize scores (softmax) → weights (a distribution over tokens)
3. take a weighted sum of **V** → a context-aware representation

**Result**: every token becomes a blend of other tokens’ information, based on learned relevance.

The working memory is read-only during attention; attention can only read and mix values. Only the residual stream can be updated.

---

## What is learned (why attention works)

The attention algorithm itself is fixed; the model learns the projection matrices that create Q, K, and V.

That means the model learns:

- what “matching” means (the Q·K similarity space),
- what content should be copied once a match is found (the V space),
- which relationships are useful for the task.

> A single attention head is best thought of as a learned **relation detector** plus **content copier**.

Heads specialize because they operate in **different learned similarity spaces**, with specialization emerging naturally during end-to-end training.

---

## Why multi-head attention exists

A single attention head must choose _one_ way to relate tokens.

Language requires many relationships to be considered simultaneously.

Multi-head attention provides:

- multiple independent match functions (Q/K spaces),
- multiple independent copy channels (V spaces).

> Multi-head attention is like running several different searches over the same sentence in parallel, then combining the evidence.

Heads are not explicitly assigned roles; specialization emerges naturally during end-to-end training.

---

## Position: attention needs order injected

Attention alone does not know _where_ tokens are.

Position is injected into the system via:

- absolute or relative positional embeddings,
- RoPE (Rotary Positional Embedding), ALiBi (Attention with Linear Biases), or similar schemes.

These methods don’t change attention itself; they change **what the queries and keys** (and sometimes values) encode, allowing distance and order to influence relevance.

---

## Causal self-attention and the KV-cache mental model

In generative models (GPT-style), attention is causal:

- a token may only attend to earlier tokens (and itself),
- enforced via a mask.

### Autoregressive generation mental model

- During training: all tokens attend in parallel (with masking).
- During inference: generation is **append-only**.

At each step:

1. Compute Q for the new token.
2. Reuse stored K/V for all previous tokens (the KV cache).
3. Attend to the past.

Past keys and values never change, so they can be cached and reused safely.

---

## The residual stream: the real “state” of the Transformer

Each token carries a vector that persists through the entire network.
This vector is the **residual stream** — the true state of the Transformer.

Crucially, layers do not replace this state.
They only **add small updates to it**.

Each layer performs:

1. **Attention** – reads from other tokens and proposes an update.
2. **MLP** – transforms the token’s own features and proposes another update.
3. **Residual addition** – adds both updates into the existing stream.

Mental picture:

> Each layer writes small edits into a running scratchpad for each token.
> Nothing is explicitly erased — information is continuously mixed, transformed, reweighted, and sometimes attenuated.

Depth = iterative refinement:
early layers write simple, local features;
later layers build more abstract, global structure on top.

The residual stream is the only thing that flows forward; everything else is just a function that reads from it and writes back into it.

All attention heads and MLPs read from the same residual stream, and their outputs are simply added back into it.

---

## What attention is good at (and not)

### Good at

- routing information across tokens,
- copying relevant context,
- resolving relationships (coreference, dependency).

### Not good at

- creating new features (that’s the MLP’s role),
- maintaining memory beyond the current context,
- performing exact, discrete algorithmic state updates.

This explains why:

- depth matters,
- MLPs matter,
- retrieval/memory systems appear alongside attention.

> Attention is powerful because it is flexible and soft — the same property that makes it weak at exact computation.

---

## Optimizations (same attention, computed differently)

Most “modern attention variants” in practice don’t change the meaning of attention; they change **how it’s computed or stored**:

- FlashAttention / FlashAttention-2: faster exact attention kernels (less memory traffic)
- MQA/GQA: shrink the KV cache by sharing K/V across heads/groups
- PagedAttention (vLLM): manage KV cache memory efficiently for serving many requests

Key takeaway:

> Frontier LLMs largely use the same scaled dot-product causal attention—what changes is efficiency, caching, and serving.

---

## The one loop to keep in mind

A Transformer repeatedly:

- Gather relevant information (attention)
- Transform it (MLP)
- Accumulate updates (residual stream)

Once that clicks, the Transformer stops feeling mysterious.

---

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

## Mental model in one minute

Self-attention is a _content-addressable read_ from a small “working memory” of tokens.

- Each token emits a **query** (what I need), and every token exposes a **key** (what I am) and **value** (what I can contribute).
- Attention computes: _for this token, which other tokens match what I’m looking for?_
- The output is a **mixture of values** — i.e., the token “pulls in” information from the most relevant tokens.

Then add the key invariant:

**Invariant**: Self-attention cannot create new information; it only **routes/mixes** information already present in the values. The **MLP/FFN** is the part that _creates_ new features (per token).
This single invariant explains why Transformers alternate **Attention (mix)** and **MLP (transform)**.

## What is learned in self-attention?

- The model learns the projection matrices that produce Q/K/V from token representations.
- This means the model learns **what “matching” means** (via Q·K) and **what content should be copied** (via V).
- Heads specialize because different projection matrices define different similarity spaces.

Useful intuition:

> A head is a learned “relation detector” + “content copier”.

Examples of relations (not claims, just intuitions):

- “this pronoun refers to that noun”
- “this verb attaches to that subject”
- “this line of code depends on that variable definition”

This helps because most confusion comes from thinking attention is a fixed algorithm. It’s not; the algorithm is fixed, but the spaces are learned.

## Attention Variants & Optimizations

These do **not** change what attention means — only _how it is computed or stored_.

| Name                                                   | Description                                                                                                                                                                        | Benefits                                                                                       | When to Use                                                                                                            |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Scaled Dot-Product Self-Attention (Standard Attention) | The classic attention mechanism from "Attention Is All You Need"; computes dot products between queries and keys, applies a scaling factor, softmax, and weighted sum over values. | Simple, well-understood, direct baseline for research and education.                           | Default for educational, research, and small models; when transparency and debugging are priorities.                   |
| FlashAttention                                         | An optimized attention algorithm that uses tiling and kernel fusion to minimize memory reads/writes and exploit faster compute, usually on GPUs.                                   | Greatly reduces memory usage, speeds up attention computation, especially with long sequences. | For large models, long contexts, or when training/generating needs to be fast (requires compatible hardware/backends). |
| FlashAttention-2                                       | Improved version of FlashAttention with better performance and support for longer sequences and more model variants; further reduces memory bottlenecks.                           | Higher throughput, lower memory — supports even larger contexts, better hardware utilization.  | When maximum efficiency is needed in state-of-the-art training/inference, especially with long-context LLMs.           |
| PagedAttention                                         | A memory-efficient method for managing Key/Value caches, by organizing them into pages, often used for serving multiple prompts with long sequences.                               | Enables serving many concurrent users, very large contexts (up to millions of tokens).         | In production LLM serving, especially with multi-user, long-prompt cases (e.g. vLLM).                                  |
| Multi-Query Attention                                  | Uses a single set of key/value heads for all attention heads (instead of separate k/v heads per query head), reducing memory/computation.                                          | Reduces KV cache size and speeds up inference, with minimal quality loss in large models.      | When scaling up to billions of parameters, especially for LLM deployment and inference speed.                          |

Key takeaway:

> Modern LLMs still use the same attention — they just compute it _far more efficiently_.

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

---

## Attention variants & optimizations (reframed)

These do **not** change what attention means — only _how it is computed or stored_.

| Category          | What changes       | Why it exists                   |
| ----------------- | ------------------ | ------------------------------- |
| Standard SDPA     | Core algorithm     | The foundation                  |
| FlashAttention    | Kernel execution   | Make exact attention fast       |
| FlashAttention-2  | Better parallelism | Use modern GPUs efficiently     |
| Multi-Query / GQA | Head structure     | Reduce KV cache size            |
| PagedAttention    | Memory management  | Serve long contexts efficiently |

Key takeaway:

> Modern LLMs still use the same attention — they just compute it _far more efficiently_.

---

## Attention in major LLMs (mental model view)

All frontier models:

- use **scaled dot-product attention**,
- apply **causal masking** for generation,
- rely on **engineering** to scale context and throughput.

Differences are almost entirely about:

- how Q/K/V are shared (MQA/GQA),
- how attention is computed (FlashAttention),
- how memory is managed (paging),
- how position is encoded.

No mainstream model has replaced attention.
