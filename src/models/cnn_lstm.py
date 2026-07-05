import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid model replicating Danandeh Mehr et al. (2023).

    Architecture
    ------------
    Conv1D layers (feature extraction)
      → LSTM layers (temporal sequence modeling)
      → Dense layers (output)

    Reference paper configuration (Table 5):
      - 2 × Conv1D (8 filters, kernel=5, linear activation, dropout=0.1)
      - 3 × LSTM   (9 units, linear activation)
      - 2 × Dense  (8 units → 1 unit)

    This implementation generalises the architecture via hyperparameters
    to support grid search, while preserving the structural design.

    Args
    ----
    input_size     : number of input features per time step
    cnn_channels   : output channels for each Conv1D layer (list)
    kernel_size    : convolution kernel width
    lstm_hidden    : hidden size for each LSTM layer
    lstm_layers    : number of stacked LSTM layers
    fc_hidden      : hidden size of the intermediate Dense layer
    dropout        : dropout applied after each Conv1D layer
    """

    def __init__(self, input_size: int,
                 cnn_channels: list = None,
                 kernel_size: int = 5,
                 lstm_hidden: int = 9,
                 lstm_layers: int = 3,
                 fc_hidden: int = 8,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [8, 8]   # reference paper default

        # ── CNN block ─────────────────────────────────────────────
        conv_layers = []
        in_ch = input_size
        for out_ch in cnn_channels:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size,
                          padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        # ── LSTM block ────────────────────────────────────────────
        self.lstm = nn.LSTM(
            in_ch, lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Dense output block ────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        # x : (B, seq_len, n_feat)
        # Conv1d expects (B, C, L)
        out = self.cnn(x.permute(0, 2, 1))      # (B, cnn_ch, seq_len)
        out = out.permute(0, 2, 1)               # (B, seq_len, cnn_ch)
        out, _ = self.lstm(out)                  # (B, seq_len, lstm_h)
        return self.fc(out[:, -1, :]).squeeze(-1)
