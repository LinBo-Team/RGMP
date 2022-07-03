import torch
from torch import nn


class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 16, 3, 1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),

            nn.Conv1d(16, 16, 3, 1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 16, kernel_size=3, stride=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(7)
        )

        self.linea_layer = nn.Sequential(
            nn.Linear(16 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),    # 注意是状态类别数
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 16 * 7)
        x = self.linea_layer(x)

        return x