"""Cross-Modal Attention modules.

Implements bidirectional cross-modal attention:
  - Text queries attend to audio keys/values  (text2audio)
  - Audio queries attend to text keys/values  (audio2text)

Both directions are computed in a single CrossModalAttentionBlock and
optionally stacked for depth.

Mathematical formulation
------------------------
Given query Q ∈ R^{B×1×D} and context K,V ∈ R^{B×1×D}:

    Attention(Q, K, V) = softmax( Q K^T / √D ) V

We use single-token (pooled) representations, so the sequence length is 1.
The output is a residual-connected, layer-normalised vector.

For sequence-level attention (optional, when frame embeddings are available),
the same module handles variable-length padding via attention masks.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional additive mask."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,            # [B, Nq, D]
        key: torch.Tensor,              # [B, Nk, D]
        value: torch.Tensor,            # [B, Nk, D]
        mask: Optional[torch.Tensor] = None,  # [B, 1, Nq, Nk] additive (0/-inf)
    ) -> torch.Tensor:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, value)


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: query from modality A, key/value from B.

    Args:
        query_dim: dimension of query vectors (from modality A).
        kv_dim:    dimension of key/value vectors (from modality B).
        out_dim:   output projection dimension.
        num_heads: number of attention heads.
        dropout:   attention dropout.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert out_dim % num_heads == 0, (
            f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        self.q_proj = nn.Linear(query_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        query: torch.Tensor,            # [B, Nq, query_dim]
        key: torch.Tensor,              # [B, Nk, kv_dim]
        value: torch.Tensor,            # [B, Nk, kv_dim]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, Nk] bool (True=ignore)
    ) -> torch.Tensor:                  # [B, Nq, out_dim]
        B, Nq, _ = query.shape
        Nk = key.size(1)

        Q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, H, N, head_dim]

        attn_mask = None
        if key_padding_mask is not None:
            # Convert bool mask [B, Nk] -> additive mask [B, H, 1, Nk]
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Nk]
            attn_mask = attn_mask.float().masked_fill(attn_mask.bool(), float("-inf"))

        out = self.attn(Q, K, V, mask=attn_mask)  # [B, H, Nq, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.out_dim)
        return self.out_proj(out)


class CrossModalAttentionBlock(nn.Module):
    """One bidirectional cross-attention block.

    Applies:
      1. text2audio: text queries attend to audio → enriched text
      2. audio2text: audio queries attend to text → enriched audio
      3. Residual connections + LayerNorm for both outputs
      4. Feed-forward sub-layer for each output

    Args:
        text_dim:  width of text representation.
        audio_dim: width of audio representation.
        out_dim:   common output width.
        num_heads: attention heads.
        ff_mult:   feed-forward expansion factor.
        dropout:   dropout rate.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        out_dim: int,
        num_heads: int = 8,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim

        # Input projections (bring text and audio to common width)
        self.text_in = nn.Linear(text_dim, out_dim) if text_dim != out_dim else nn.Identity()
        self.audio_in = nn.Linear(audio_dim, out_dim) if audio_dim != out_dim else nn.Identity()

        # Cross-attention: text queries → audio keys/values
        self.text2audio = MultiHeadCrossAttention(
            query_dim=out_dim, kv_dim=out_dim, out_dim=out_dim,
            num_heads=num_heads, dropout=dropout,
        )
        # Cross-attention: audio queries → text keys/values
        self.audio2text = MultiHeadCrossAttention(
            query_dim=out_dim, kv_dim=out_dim, out_dim=out_dim,
            num_heads=num_heads, dropout=dropout,
        )

        # Layer norms
        self.norm_t2a = nn.LayerNorm(out_dim)
        self.norm_a2t = nn.LayerNorm(out_dim)
        self.norm_t_ff = nn.LayerNorm(out_dim)
        self.norm_a_ff = nn.LayerNorm(out_dim)

        # Feed-forward
        ff_dim = out_dim * ff_mult
        self.ff_text = nn.Sequential(
            nn.Linear(out_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, out_dim), nn.Dropout(dropout),
        )
        self.ff_audio = nn.Sequential(
            nn.Linear(out_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, out_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_emb: torch.Tensor,          # [B, 1, text_dim] or [B, Nt, text_dim]
        audio_emb: torch.Tensor,         # [B, 1, audio_dim] or [B, Na, audio_dim]
        text_mask: Optional[torch.Tensor] = None,   # [B, Nt] bool (True=pad)
        audio_mask: Optional[torch.Tensor] = None,  # [B, Na] bool (True=pad)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_out:  [B, Nt, out_dim]
            audio_out: [B, Na, out_dim]
        """
        # Project to common width
        t = self.text_in(text_emb)   # [B, Nt, out_dim]
        a = self.audio_in(audio_emb)  # [B, Na, out_dim]

        # text attends to audio
        t2a = self.text2audio(t, a, a, key_padding_mask=audio_mask)
        t = self.norm_t2a(t + self.dropout(t2a))

        # audio attends to text
        a2t = self.audio2text(a, t, t, key_padding_mask=text_mask)
        a = self.norm_a2t(a + self.dropout(a2t))

        # Feed-forward
        t = self.norm_t_ff(t + self.ff_text(t))
        a = self.norm_a_ff(a + self.ff_audio(a))

        return t, a


class StackedCrossModalAttention(nn.Module):
    """Stack of CrossModalAttentionBlocks.

    Args:
        text_dim:    input text width.
        audio_dim:   input audio width.
        out_dim:     output width (same for both modalities).
        num_layers:  number of blocks.
        num_heads:   attention heads per block.
        dropout:     dropout rate.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # After first layer both dims are out_dim
            t_dim = text_dim if i == 0 else out_dim
            a_dim = audio_dim if i == 0 else out_dim
            self.layers.append(
                CrossModalAttentionBlock(
                    text_dim=t_dim,
                    audio_dim=a_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

    def forward(
        self,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t, a = text_emb, audio_emb
        for block in self.layers:
            t, a = block(t, a, text_mask=text_mask, audio_mask=audio_mask)
        return t, a
