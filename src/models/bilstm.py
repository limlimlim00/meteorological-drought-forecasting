import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM — covers four variants from Natarajan et al. (2026):

      num_layers=1  → BiLSTM
      num_layers>1  → S-BiLSTM (Stacked BiLSTM)

    Existing LSTM model (unidirectional) covers:
      num_layers=1  → LSTM
      num_layers>1  → S-LSTM (Stacked LSTM)

    Args
    ----
    input_size  : number of input features per time step
    hidden_size : hidden state size per direction
                  (output dim = hidden_size × 2 due to bidirectionality)
    num_layers  : number of stacked BiLSTM layers
    dropout     : dropout between layers (only if num_layers > 1)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # forward + backward concatenation → hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x : (B, seq_len, n_feat)
        out, _ = self.lstm(x)          # (B, seq_len, hidden*2)
        return self.fc(out[:, -1, :]).squeeze(-1)
