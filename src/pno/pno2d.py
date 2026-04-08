# pno model

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pno.layers import DAFNOBlock, DeepNormProjection


class PlanningNeuralOperator(nn.Module):
    def __init__(
        self,
        width: int = 48,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 4,
        padding: int = 9,
        beta: float = 5.0,
        deepnorm_hidden: int = 64,
    ):
        super().__init__()

        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.depth = depth
        self.padding = padding
        self.beta = beta

        # input lift
        self.fc0 = nn.Linear(3, width)

        # spectral blocks
        self.blocks = nn.ModuleList([
            DAFNOBlock(width, modes1, modes2) for _ in range(depth)
        ])

        # output head
        self.deepnorm = DeepNormProjection(width, hidden=deepnorm_hidden)

    # sifn mask
    def _build_sifn(self, sdf: torch.Tensor, raw_map: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.beta * sdf) * (raw_map - 0.5) + 0.5

    # goal channel
    def _build_goal_channel(
        self, shape: tuple, goal: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        B, _, H, W = shape
        g = torch.zeros(B, 1, H, W, device=device)
        gx = goal[:, 0].long().clamp(0, W - 1)
        gy = goal[:, 1].long().clamp(0, H - 1)
        g[torch.arange(B, device=device), 0, gy, gx] = -1.0
        return g

    # forward pass
    def forward(
        self,
        raw_map: torch.Tensor,
        sdf: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = raw_map.shape
        device = raw_map.device

        # sifn apply
        chi = self._build_sifn(sdf, raw_map)

        # stack input
        goal_ch = self._build_goal_channel(raw_map.shape, goal, device)
        x = torch.cat([raw_map, sdf, goal_ch], dim=1)
        x = x.permute(0, 2, 3, 1)

        # lift feature
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # pad field
        x = F.pad(x, [0, self.padding, 0, self.padding])
        chi_padded = F.pad(chi, [0, self.padding, 0, self.padding])

        # block stack
        for block in self.blocks:
            x = block(x, chi_padded)

        # crop field
        x = x[..., :H, :W]

        # project value
        gx = goal[:, 0].long().clamp(0, W - 1)
        gy = goal[:, 1].long().clamp(0, H - 1)
        out = self.deepnorm(x, gx, gy)

        return out
