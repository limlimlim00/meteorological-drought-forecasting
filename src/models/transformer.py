import math
import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 512):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class Transformer(nn.Module):
    """
    Standard Transformer Encoder for time-series regression.

    Difference from TFT
    -------------------
    - No variable selection network
    - No gated residual networks
    - Pure Transformer encoder (multi-head self-attention + FFN) with
      sinusoidal positional encoding
    - Last time-step output → linear head

    Args
    ----
    input_size      : number of input features per time step
    d_model         : internal model dimension (embedding size)
    num_heads       : number of attention heads
    num_layers      : number of TransformerEncoderLayer stacks
    dropout         : dropout in attention and FFN
    dim_feedforward : FFN hidden dimension (default: 4 × d_model)
    """

    def __init__(self, input_size: int, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, dim_feedforward: int = None,
                 **kwargs):
        super().__init__()

        if d_model % num_heads != 0:
            # Round d_model up to nearest multiple of num_heads
            d_model = num_heads * math.ceil(d_model / num_heads)

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc      = nn.Linear(d_model, 1)
        self._last_attn_weights = None

    def forward(self, x):
        # x : (B, seq_len, n_feat)
        out = self.pos_enc(self.input_proj(x))
        out = self.encoder(out)
        return self.fc(out[:, -1, :]).squeeze(-1)

    @torch.no_grad()
    def compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        별도 순전파로 각 레이어의 attention weights 추출.
        Returns (B, heads, T, T) — 레이어 평균.
        """
        out = self.pos_enc(self.input_proj(x))
        attn_list = []
        for layer in self.encoder.layers:
            q = k = v = layer.norm1(out) if layer.norm_first else out
            _, w = layer.self_attn(q, k, v,
                                   need_weights=True,
                                   average_attn_weights=False)
            if w is not None:
                attn_list.append(w)
            out = layer(out)
        if attn_list:
            self._last_attn_weights = torch.stack(attn_list, 0).mean(0)
        return self._last_attn_weights

    def get_attention_weights(self):
        return self._last_attn_weights
