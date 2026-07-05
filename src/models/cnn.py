import torch.nn as nn


class CNN(nn.Module):
    """
    Standard 1-D Convolutional Neural Network for time-series regression.

    Difference from TCN
    -------------------
    - No dilation / causal masking
    - No residual connections
    - Simple stack of Conv1d → ReLU → Dropout, then global average pooling

    Args
    ----
    input_size  : number of input features per time step
    hidden_size : number of convolutional channels (same for all layers)
    num_layers  : number of convolutional layers
    kernel_size : convolution kernel width
    dropout     : dropout probability after each conv layer
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, kernel_size: int = 3,
                 dropout: float = 0.1, **kwargs):
        super().__init__()

        layers = []
        in_ch = input_size
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_size,
                          kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = hidden_size

        self.conv = nn.Sequential(*layers)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x : (B, seq_len, n_feat) → Conv1d needs (B, C, L)
        out = self.conv(x.permute(0, 2, 1))   # (B, hidden, seq_len)
        out = out.mean(dim=-1)                 # global average pooling → (B, hidden)
        return self.fc(out).squeeze(-1)
