import math

import torch
import torch.nn as nn

import hyperparams as hp


def _has_invalid_values(tensor):
    """Check if a tensor contains NaN or Inf values."""
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 120, debug: bool = False):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))

        if debug and _has_invalid_values(div_term):
            raise ValueError(f"Invalid div_term values: {div_term}")

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerBCE(nn.Module):
    """Transformer encoder with a CLS token for binary classification or regression.

    Uses a prepended learnable CLS token whose final representation is projected
    through a linear head to produce logits.
    """

    def __init__(
        self,
        feature_dim: int = 15,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = hp.DROPOUT,
        seq_len: int = hp.WINDOW_SIZE,
        head_size: int = 1,
        debug: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.debug = debug

        self.embed = nn.Linear(feature_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=seq_len + 1, debug=debug)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.input_norm = nn.LayerNorm(feature_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, head_size)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            Tensor of shape (batch_size, head_size) representing logits.
        """
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.size(1)}")

        bsz = x.size(0)
        tok = self.embed(self.input_norm(x))

        if self.debug and _has_invalid_values(tok):
            raise ValueError(
                f"NaN in embedded tokens. Input has NaN={torch.isnan(x).any()}, "
                f"Inf={torch.isinf(x).any()}, max_abs={torch.max(torch.abs(x))}"
            )

        cls = self.cls_token.expand(bsz, -1, -1)
        tokens = torch.cat([cls, tok], dim=1)
        tokens = self.positional_encoding(tokens)

        if self.debug and _has_invalid_values(tokens):
            raise ValueError("NaN in tokens after positional encoding")

        encoded = self.transformer(tokens)

        if self.debug and _has_invalid_values(encoded):
            raise ValueError("NaN in transformer output")

        cls_rep = encoded[:, 0]
        logits = self.head(self.dropout(cls_rep))
        return logits
