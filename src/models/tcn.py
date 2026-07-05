import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _Chomp1d(nn.Module):
    """Remove future time steps introduced by causal padding."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """
    One residual block: two dilated causal conv layers + residual connection.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1  = _Chomp1d(padding)
        self.relu1   = nn.ReLU()
        self.drop1   = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_ch, out_ch, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2  = _Chomp1d(padding)
        self.relu2   = nn.ReLU()
        self.drop2   = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        # 1×1 conv to match channel dims when needed
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network (Bai et al., 2018).

    Args
    ----
    input_size   : number of input channels (features)
    num_channels : list of output channels per TCN level, e.g. [64, 64, 64]
    kernel_size  : convolution kernel size
    dropout      : dropout probability in each TemporalBlock
    """

    def __init__(self, input_size: int, num_channels: list = None,
                 kernel_size: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__()

        if num_channels is None:
            num_channels = [64, 64, 64]

        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch    = input_size if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(_TemporalBlock(
                in_ch, out_ch, kernel_size,
                stride=1, dilation=dilation, dropout=dropout
            ))

        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x : (B, seq_len, n_feat) → Conv1d expects (B, C, L)
        out = self.tcn(x.permute(0, 2, 1))   # (B, ch, seq_len)
        return self.fc(out[:, :, -1]).squeeze(-1)
