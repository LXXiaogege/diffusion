import torch
from torch import nn


class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2)
        ])

        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2)
        ])
        self.activate = nn.ReLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []  # 用list模拟一个队列，做残差连接
        for idx, layer in enumerate(self.down_layers):
            x = layer(x)
            x = self.activate(x)
            if idx < 2:
                h.append(x)
                x = self.downscale(x)
        for idx, layer in enumerate(self.up_layers):
            if idx > 0:
                x = self.upscale(x)
                x += h.pop()
            x = layer(x)
            x = self.activate(x)
        return x
