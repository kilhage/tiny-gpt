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

Early encoder–decoder RNNs compressed an entire input sequence into a single vector, creating a severe bottleneck: long or information-dense inputs simply didn’t fit.

Attention removed this bottleneck by letting the decoder **dynamically look back** at relevant encoder states instead of relying on a single summary. Early forms (Bahdanau, Luong) introduced the core abstraction still used today: compute relevance between a **query and keys**, turn it into **weights**, and use those weights to mix **values**.

The Transformer made a decisive leap. Instead of “decoder attends to encoder,” everything attends to everything. In _Attention Is All You Need_ (2017), attention became the central compute primitive, enabling parallel processing, global information flow at every layer, and clean scaling through repetition. This immediately exposed two new constraints—position (attention is order-agnostic) and cost (quadratic in sequence length)—and since then, progress has focused not on replacing attention, but on making it practical at scale.

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
- MQA (Multi-Query Attention) / GQA (Grouped-Query Attention): shrink the KV cache by sharing K/V across heads/groups
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
