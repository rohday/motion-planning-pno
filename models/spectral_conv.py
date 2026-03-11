"""
Exact replication of SpectralConv2d from the official PNO repository.
Source: ExistentialRobotics/PNO - examples/models/spectralLayer.py

No norm, torch.rand init, scale = 1 / (in_channels * out_channels).
"""

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → linear transform on low-freq modes → IFFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Official init: uniform rand, not randn
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2,
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2,
                dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_ch, x, y), (in_ch, out_ch, x, y) -> (batch, out_ch, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # No norm kwarg — matches official code exactly
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
