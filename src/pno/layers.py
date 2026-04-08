# pno layers

import torch
import torch.nn as nn
import torch.nn.functional as F


# spectral conv

class DAFNOSpectralConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               modes1, modes2, dtype=torch.cfloat)
        )

    # complex mul
    @staticmethod
    def _cmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    # spectral pass
    def forward(self, v: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
        v_masked = v * chi

        # fft forward
        v_ft = torch.fft.rfft2(v_masked)

        B = v.shape[0]
        H, W = v.shape[2], v.shape[3]
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=v.device)

        # low modes
        out_ft[:, :, :self.modes1, :self.modes2] = self._cmul(
            v_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # neg modes
        out_ft[:, :, -self.modes1:, :self.modes2] = self._cmul(
            v_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # fft inverse
        return torch.fft.irfft2(out_ft, s=(H, W))


class DAFNOBlock(nn.Module):

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = DAFNOSpectralConv2d(width, width, modes1, modes2)
        self.local_conv = nn.Conv2d(width, width, kernel_size=1)

    # block step
    def forward(self, v: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral(v, chi)
        x2 = self.local_conv(v)
        return F.gelu(x1 + x2)


# deepnorm head

class DeepNormProjection(nn.Module):

    def __init__(self, width: int, hidden: int = 64):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(hidden, width) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hidden))
        self.weight2 = nn.Parameter(torch.randn(1, hidden) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(1))

    # head pass
    def forward(self, v: torch.Tensor,
                goal_x: torch.Tensor, goal_y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = v.shape

        # goal feat
        gx = goal_x.long()
        gy = goal_y.long()
        v_goal = v[torch.arange(B, device=v.device), :, gy, gx]

        # abs delta
        delta = torch.abs(v - v_goal[:, :, None, None])

        # flatten map
        delta_flat = delta.permute(0, 2, 3, 1).reshape(-1, C)

        # positive mlp
        w1_pos = F.softplus(self.weight1)
        h = F.gelu(delta_flat @ w1_pos.t() + self.bias1)

        w2_pos = F.softplus(self.weight2)
        out = h @ w2_pos.t() + self.bias2

        return out.reshape(B, H, W, 1).permute(0, 3, 1, 2)
