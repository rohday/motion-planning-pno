"""
Exact replication of FNO2dMultiGoal from the official PNO repository.
Source: ExistentialRobotics/PNO - examples/models/fnoMultiGoal.py

Architecture:
  1. Lifting:        Linear(1, width)         -- maps input channel → width
  2. Fourier layers: u' = σ(K(u) + W(u))      -- K=spectral conv, W=1×1 conv
  3. Projection:     Linear(width, 128) → GELU → Linear(128, 1)

Input:  chi  (B, H, W, 1) -- smooth occupancy field, goal pixel marked as -1
Output: T    (B, H, W, 1) -- predicted value function (travel time)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_conv import SpectralConv2d


class FNO2dMultiGoal(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers, padding=9):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.padding = padding  # pad domain since input is non-periodic

        # Lifting: input has 1 channel (chi)
        self.fc0 = nn.Linear(1, self.width)

        # Fourier + residual conv layers
        for i in range(self.num_layers):
            self.add_module(f'conv{i}', SpectralConv2d(width, width, modes1, modes2))
            self.add_module(f'w{i}', nn.Conv2d(width, width, 1))

        # Projection
        self.fc1 = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, goal):
        """
        Args:
            x:    (B, H, W, 1)  -- smooth chi field
            goal: (B, 2)        -- goal pixel (row, col) per sample
        Returns:
            out:  (B, H, W, 1)
        """
        # Mark the goal cell with -1 in the input field
        x = x.clone()
        for i in range(x.shape[0]):
            row, col = int(goal[i, 0].item()), int(goal[i, 1].item())
            x[i, row, col, 0] = -1.0

        # Lift: (B, H, W, 1) → (B, H, W, width) → (B, width, H, W)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Pad for non-periodic domain
        x = F.pad(x, [0, self.padding, 0, self.padding])

        # Fourier layers
        for i in range(self.num_layers):
            x1 = self._modules[f'conv{i}'](x)
            x2 = self._modules[f'w{i}'](x)
            x = x1 + x2
            x = F.gelu(x)

        # Remove padding
        x = x[..., :-self.padding, :-self.padding]

        # Project: (B, width, H, W) → (B, H, W, width) → (B, H, W, 1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        return x

    def get_grid(self, shape, device):
        """Helper: create (B, H, W, 2) coordinate grid in [0, 1]²."""
        B, H, W = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, H, dtype=torch.float, device=device)
        gridy = torch.linspace(0, 1, W, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, H, 1, 1).expand(B, -1, W, 1)
        gridy = gridy.reshape(1, 1, W, 1).expand(B, H, -1, 1)
        return torch.cat([gridx, gridy], dim=-1)
