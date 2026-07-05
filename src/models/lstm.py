import torch.nn as nn


class LSTM(nn.Module):
    """
    Long Short-Term Memory network.

    Args
    ----
    input_size  : number of input features per time step
    hidden_size : LSTM hidden state size
    num_layers  : number of stacked LSTM layers
    dropout     : dropout between layers (only if num_layers > 1)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x : (B, seq_len, n_feat)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
