import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fno.layers import SpectralConv2d, DepthwiseSpectralConv2d


class FNO2dMultiGoal(nn.Module):
    def __init__(self, depth, padding, modes1, modes2, width, depthwise=False):
        super(FNO2dMultiGoal, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.depthwise = depthwise
        self.fc0 = nn.Linear(1, self.width)
        self.depth = depth
        self.fc1 = torch.nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
        )

        for i in range(self.depth):
            if depthwise:
                self.add_module('conv%d' % i,
                                DepthwiseSpectralConv2d(self.width, self.modes1, self.modes2))
            else:
                self.add_module('conv%d' % i,
                                SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.add_module('w%d' % i,
                            nn.Conv2d(self.width, self.width, 1))

    def forward(self, x, goal):
        for i in range(x.shape[0]):
            x[i][goal[i][1].long()][goal[i][0].long()] = -1

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.depth):
            x1 = self._modules['conv%d' % i](x)
            x2 = self._modules['w%d' % i](x)
            x = x1 + x2
            x = F.gelu(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FNO2dSDF(nn.Module):
    """FNO for geometry-to-SDF mapping.

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W)
    """

    def __init__(self, depth, padding, modes1, modes2, width, depthwise=False):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.depth = depth

        # Lift one channel geometry to hidden width.
        self.fc0 = nn.Linear(1, self.width)

        for i in range(self.depth):
            if depthwise:
                self.add_module(
                    f"conv{i}",
                    DepthwiseSpectralConv2d(self.width, self.modes1, self.modes2),
                )
            else:
                self.add_module(
                    f"conv{i}",
                    SpectralConv2d(self.width, self.width, self.modes1, self.modes2),
                )
            self.add_module(f"w{i}", nn.Conv2d(self.width, self.width, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # Accept BCHW or BHWC for convenience.
        if x.ndim != 4:
            raise ValueError("x must be 4D tensor")

        if x.shape[1] == 1:  # BCHW -> BHWC
            x = x.permute(0, 2, 3, 1)
        elif x.shape[-1] != 1:
            raise ValueError("Expected one input channel")

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.depth):
            x1 = self._modules[f"conv{i}"](x)
            x2 = self._modules[f"w{i}"](x)
            x = F.gelu(x1 + x2)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        return x
