import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_gpt.config import GPTConfig


# -------------------------
# Char-level tokenizer (simple + reliable)
# -------------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


# -------------------------
# Data loader (random batches)
# -------------------------
class TextData:
    def __init__(self, text: str, tokenizer: CharTokenizer, split: float = 0.9):
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(split * len(ids))
        self.train_ids = ids[:n]
        self.val_ids = ids[n:]

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ):
        data = self.train_ids if split == "train" else self.val_ids
        # random starting indices
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


# -------------------------
# Grouped-Query Attention (GQA)
# -------------------------
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    - n_q_head: number of query heads (like normal multi-head attention)
    - n_kv_head: number of key/value heads (smaller than n_q_head)
    - Each KV head is shared across (n_q_head / n_kv_head) Q heads.

    Input:  x (B, T, C)
    Output: y (B, T, C)
    """

    def __init__(
        self,
        cfg: GPTConfig,
    ):
        super().__init__()
        assert cfg.n_embd % cfg.n_q_head == 0, "n_embd must be divisible by n_q_head"
        assert (
            cfg.n_q_head % cfg.n_kv_head == 0
        ), "n_q_head must be divisible by n_kv_head"

        self.n_embd = cfg.n_embd
        self.n_q_head = cfg.n_q_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_q_head
        self.attn_dropout = cfg.dropout
        self.block_size = cfg.block_size
        self.group_size = self.n_q_head // self.n_kv_head

        # Output widths for fused projection
        self.q_out = cfg.n_q_head * self.head_dim
        self.kv_out = cfg.n_kv_head * self.head_dim

        assert self.q_out == self.n_embd
        assert self.kv_out == self.n_kv_head * self.head_dim

        # One projection for Q,K,V:
        # (B,T,C) -> (B,T, q_out + kv_out + kv_out)
        self.qkv_proj = nn.Linear(self.n_embd, self.q_out + 2 * self.kv_out, bias=False)

        # Final projection back to model dimension
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def _split_heads(self, x: torch.Tensor, n_head: int) -> torch.Tensor:
        """(B,T,n_head*head_dim) -> (B,n_head,T,head_dim)"""
        B, T, _ = x.shape
        return x.view(B, T, n_head, self.head_dim).transpose(1, 2)

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match Q heads without materializing repeats.
        kv: (B, n_kv, T, d) -> (B, n_q, T, d)
        """
        B, n_kv, T, d = kv.shape
        assert n_kv == self.n_kv_head
        assert d == self.head_dim

        kv = kv[:, :, None, :, :]  # (B, n_kv, 1, T, d)
        kv = kv.expand(B, n_kv, self.group_size, T, d)  # (B, n_kv, group, T, d) view
        return kv.reshape(B, n_kv * self.group_size, T, d)  # (B, n_q, T, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        assert C == self.n_embd, "Expected embedding dimension to match config."
        assert T <= self.block_size, "Sequence length exceeds configured block size."

        # Fused projection then split into Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_out, self.kv_out, self.kv_out], dim=-1)

        # Split into heads
        q = self._split_heads(q, self.n_q_head)  # (B, n_q,  T, d)
        k = self._split_heads(k, self.n_kv_head)  # (B, n_kv, T, d)
        v = self._split_heads(v, self.n_kv_head)  # (B, n_kv, T, d)

        # Expand K/V from n_kv_head -> n_q_head by repeating groups
        k = self._expand_kv(k)
        v = self._expand_kv(v)

        # Use fused SDPA (FlashAttention kernels on supported GPUs)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, SDPA returns (B,n_q,T,d))

        # Merge heads back: (B,T,C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


# -------------------------
# Causal self-attention using PyTorch SDPA
# -------------------------
class CausalSelfAttentionSDPA(nn.Module):
    """
    Multi-head causal self-attention using PyTorch SDPA.
    PyTorch will dispatch to FlashAttention/mem-efficient kernels when available.

    Input:  x (B, T, C)
    Output: y (B, T, C)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, n_head, T, head_dim)
        B, T, C = x.shape
        return x.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_head, T, head_dim) -> (B, T, C)
        B, nh, T, hd = x.shape
        return x.transpose(1, 2).contiguous().reshape(B, T, nh * hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # SDPA includes scaling, masking (if is_causal=True), softmax, and dropout.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = self._merge_heads(y)
        y = self.proj(y)
        y = self.resid_dropout(y)
        return y


# -------------------------
# Causal self-attention
# -------------------------
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Input:  x (B, T, C)
    Output: y (B, T, C)

    Causal = token t may only attend to tokens <= t.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head  # number of heads
        self.head_dim = cfg.n_embd // cfg.n_head  # dimension of each head
        self.dropout = cfg.dropout
        self.block_size = cfg.block_size

        # One projection for Q,K,V for efficiency: (B,T,C) -> (B,T,3C)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)

        # Final projection back to model dimension
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        # dropout for the attention weights
        self.attn_dropout = nn.Dropout(self.dropout)

        # dropout for the residual connections
        self.resid_dropout = nn.Dropout(self.dropout)

        # Boolean causal mask: True = allowed, False = masked out
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the channels (C) into multiple heads."""
        # (B,T,C) -> (B,n_head,T,head_dim)
        B, T, C = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal self-attention."""
        B, T, C = x.shape  # batch, time, channels
        assert T <= self.block_size, "Sequence length exceeds configured block size."

        # Project once, then split
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention scores: (B,n_head,T,T)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale

        # Mask future tokens (disallow attending to positions > t)
        att = att.masked_fill(~self.mask[:, :, :T, :T], torch.finfo(att.dtype).min)

        # Softmax -> dropout -> weighted sum
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B,n_head,T,head_dim)

        # Merge heads: (B,T,C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + residual dropout
        y = self.proj(y)
        y = self.resid_dropout(y)
        return y


# -------------------------
# MLP (feed-forward network)
# -------------------------
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden)
        self.proj = nn.Linear(hidden, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


# -------------------------
# Transformer block
# -------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = GroupedQueryAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # pre-LN + residual
        x = x + self.mlp(self.ln2(x))  # pre-LN + residual
        return x


# -------------------------
# GPT model
# -------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [Block(cfg) for _ in range(cfg.n_layer)]
        )  # transformer blocks
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # language modeling head
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        # weight tying (common in GPTs)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "Sequence too long for block_size"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            # flatten for cross-entropy
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]  # crop context
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / max(temperature, 1e-8)  # last step

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
