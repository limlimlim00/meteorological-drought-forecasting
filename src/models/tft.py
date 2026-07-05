import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _GRN(nn.Module):
    """
    Gated Residual Network (Lim et al., 2021).
    GLU-style gating with skip connection and LayerNorm.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.drop(h)
        out  = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.norm(gate * out + residual)


class _VSN(nn.Module):
    """
    Variable Selection Network.
    Learns a soft weight over input variables at each time step.
    """
    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        # Per-variable GRN
        self.var_grns = nn.ModuleList([
            _GRN(d_model, d_model, d_model, dropout) for _ in range(n_vars)
        ])
        # Softmax selection GRN: flat input → n_vars weights
        self.sel_grn = _GRN(n_vars * d_model, d_model, n_vars, dropout)

    def forward(self, x_emb):
        """
        x_emb : (B, T, n_vars, d_model)
        Returns
        -------
        context : (B, T, d_model)   weighted sum of variable embeddings
        weights : (B, T, n_vars)    variable selection weights (for XAI)
        """
        B, T, V, D = x_emb.shape

        # Per-variable transformations
        processed = torch.stack(
            [self.var_grns[i](x_emb[:, :, i, :]) for i in range(V)],
            dim=2
        )  # (B, T, V, D)

        # Selection weights
        flat    = x_emb.reshape(B, T, V * D)
        weights = torch.softmax(self.sel_grn(flat), dim=-1)  # (B, T, V)

        context = (processed * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)
        return context, weights


class _InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention (shared value projection across heads).
    Produces per-head attention weights usable for XAI.
    """
    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.h   = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)   # shared single value proj
        self.W_o = nn.Linear(self.d_k, d_model)   # d_k → d_model after head averaging
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        x : (B, T, d_model)
        Returns out (B, T, d_model) and attn_weights (B, h, T, T).
        """
        B, T, _ = x.shape
        h, d_k  = self.h, self.d_k

        Q = self.W_q(x).view(B, T, h, d_k).transpose(1, 2)  # (B, h, T, d_k)
        K = self.W_k(x).view(B, T, h, d_k).transpose(1, 2)
        V = self.W_v(x)                                       # (B, T, d_k) shared

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, h, T, T)
        attn   = self.drop(torch.softmax(scores, dim=-1))

        # Expand shared V to all heads
        V_exp  = V.unsqueeze(1).expand(-1, h, -1, -1)        # (B, h, T, d_k)
        ctx    = torch.matmul(attn, V_exp)                    # (B, h, T, d_k)
        ctx    = ctx.mean(dim=1)                              # (B, T, d_k) avg over heads
        out    = self.W_o(ctx)                                # (B, T, d_model)
        return out, attn


# ---------------------------------------------------------------------------
# Main TFT model
# ---------------------------------------------------------------------------

class TFT(nn.Module):
    """
    Temporal Fusion Transformer (simplified, time-series only variant).

    Pipeline
    --------
    1. Linear embedding per variable  → d_model
    2. Variable Selection Network     → weighted context vector
    3. LSTM encoder                   → temporal encoding
    4. Interpretable Multi-Head Attn  → captures long-range dependencies
    5. GRN + LayerNorm                → post-attention processing
    6. Linear output

    Args
    ----
    input_size  : number of input variables
    seq_len     : look-back window (used only for shape info)
    d_model     : internal model dimension
    num_layers  : number of LSTM layers
    num_heads   : attention heads
    dropout     : general dropout
    attn_dropout: attention weight dropout
    """

    def __init__(self, input_size: int, seq_len: int = 12,
                 d_model: int = 64, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1,
                 attn_dropout: float = 0.0, **kwargs):
        super().__init__()

        # 1. Per-variable linear embedding
        self.var_emb = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_size)
        ])

        # 2. Variable Selection Network
        self.vsn = _VSN(input_size, d_model, dropout)

        # 3. LSTM encoder
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Interpretable Multi-Head Attention
        self.attn = _InterpretableMultiHeadAttention(d_model, num_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # 5. Post-attention GRN
        self.post_grn = _GRN(d_model, d_model, d_model, dropout)

        # 6. Output
        self.fc = nn.Linear(d_model, 1)

        # Store last attention weights for XAI
        self._last_attn_weights = None
        self._last_vsn_weights  = None

    def forward(self, x):
        """
        x : (B, seq_len, n_feat)
        """
        B, T, V = x.shape

        # 1. Embed each variable
        x_emb = torch.stack(
            [self.var_emb[i](x[:, :, i:i+1]) for i in range(V)],
            dim=2
        )  # (B, T, V, d_model)

        # 2. Variable selection
        context, vsn_w = self.vsn(x_emb)    # (B, T, d_model), (B, T, V)
        self._last_vsn_weights = vsn_w.detach()

        # 3. LSTM
        lstm_out, _ = self.lstm(context)     # (B, T, d_model)

        # 4. Self-attention
        attn_out, attn_w = self.attn(lstm_out)
        self._last_attn_weights = attn_w.detach()
        lstm_out = self.attn_norm(lstm_out + attn_out)

        # 5. Post GRN on last time step
        out = self.post_grn(lstm_out[:, -1, :])  # (B, d_model)

        return self.fc(out).squeeze(-1)

    def get_attention_weights(self):
        """Return last forward-pass attention weights (B, h, T, T)."""
        return self._last_attn_weights

    def get_vsn_weights(self):
        """Return last forward-pass variable selection weights (B, T, V)."""
        return self._last_vsn_weights
