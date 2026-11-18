"""
Cross-Modal Attention Module for SarcasTone

Implements bidirectional cross-attention between text and audio modalities:
  - Text queries attending to audio keys/values
  - Audio queries attending to text keys/values

Uses Multi-Head Attention with proper masking for variable-length sequences.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class CrossModalAttentionLayer(nn.Module):
    """Single cross-modal attention layer with residual connection and layer norm."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Cross-attention: queries from one modality, keys/values from another
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_q, d_model) - from one modality
            key_value: (batch, seq_kv, d_model) - from the other modality
            key_padding_mask: (batch, seq_kv) - True for padded positions
            
        Returns:
            output: (batch, seq_q, d_model)
            attention_weights: (batch, n_heads, seq_q, seq_kv)
        """
        # Cross-attention with residual
        residual = query
        query_norm = self.norm1(query)
        
        attn_output, attn_weights = self.cross_attn(
            query=query_norm,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )
        query = residual + self.dropout(attn_output)
        
        # FFN with residual
        residual = query
        query = residual + self.ffn(self.norm2(query))
        
        return query, attn_weights


class BiDirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-modal attention between text and audio.
    
    Text attends to audio AND audio attends to text,
    allowing information flow in both directions.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Text-to-Audio attention layers (text queries, audio keys/values)
        self.text_to_audio_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Audio-to-Text attention layers (audio queries, text keys/values)  
        self.audio_to_text_layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norms
        self.text_output_norm = nn.LayerNorm(d_model)
        self.audio_output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        audio_padding_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            text_features: (batch, seq_text, d_model)
            audio_features: (batch, seq_audio, d_model)
            text_padding_mask: (batch, seq_text) - True for padded positions
            audio_padding_mask: (batch, seq_audio) - True for padded positions
            
        Returns:
            dict with keys:
                'text_attended': (batch, seq_text, d_model)
                'audio_attended': (batch, seq_audio, d_model)
                'text_to_audio_weights': list of attention weight tensors
                'audio_to_text_weights': list of attention weight tensors
        """
        text_out = text_features
        audio_out = audio_features
        
        t2a_weights = []
        a2t_weights = []
        
        for i in range(self.n_layers):
            # Text attends to audio
            text_out, t2a_w = self.text_to_audio_layers[i](
                query=text_out,
                key_value=audio_out,
                key_padding_mask=audio_padding_mask
            )
            t2a_weights.append(t2a_w)
            
            # Audio attends to text
            audio_out, a2t_w = self.audio_to_text_layers[i](
                query=audio_out,
                key_value=text_out,
                key_padding_mask=text_padding_mask
            )
            a2t_weights.append(a2t_w)
        
        return {
            'text_attended': self.text_output_norm(text_out),
            'audio_attended': self.audio_output_norm(audio_out),
            'text_to_audio_weights': t2a_weights,
            'audio_to_text_weights': a2t_weights
        }