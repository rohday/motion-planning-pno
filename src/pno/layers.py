# pno layers — paper-faithful DAFNO + DeepNormMetric

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constrained linear ────────────────────────────────────────────────────────
# Keeps weights positive via min(w^2, |w|) — required for DeepNorm triangle
# inequality guarantee.

class ConstrainedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, F.softplus(self.weight))


# ── Activations ───────────────────────────────────────────────────────────────

class MaxReLUPairwiseActivation(nn.Module):
    """
    Pools adjacent pairs: max-pool + ReLU-scaled avg-pool.
    Preserves feature dimension (F//2 + F//2 = F), so no dim change.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                                               # (N, 1, F)
        max_part  = F.max_pool1d(x, 2)                                   # (N, 1, F//2)
        relu_part = F.avg_pool1d(F.relu(x * F.softplus(self.weights)), 2)# (N, 1, F//2)
        return torch.cat((max_part, relu_part), dim=-1).squeeze(1)       # (N, F)


class ConcaveActivation(nn.Module):
    """
    Learnable concave activation: computes min over a set of affine functions.
    Preserves num_features dimension.
    """
    def __init__(self, num_features: int, concave_activation_size: int):
        super().__init__()
        assert concave_activation_size > 1
        self.bs_nonzero = nn.Parameter(
            1e-3 * torch.randn(1, num_features, concave_activation_size - 1) - 1
        )
        self.register_buffer("bs_zero", torch.zeros(1, num_features, 1))
        self.ms = nn.Parameter(
            1e-3 * torch.randn(1, num_features, concave_activation_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = torch.cat((F.softplus(self.bs_nonzero), self.bs_zero), dim=-1)
        ms = 2 * torch.sigmoid(self.ms)
        x = x.unsqueeze(-1) * ms + bs          # (N, F, concave_size)
        return x.min(-1)[0]                     # (N, F)


class ReduceMetric(nn.Module):
    def __init__(self, mode: str = "avg"):
        super().__init__()
        assert mode in ("avg", "max"), f"Unknown mode: {mode}"
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(-1) if self.mode == "avg" else x.max(-1)[0]


# ── DeepNorm metric (paper output head) ──────────────────────────────────────

class DeepNormMetric(nn.Module):
    """
    Paper-faithful DeepNormMetric from TrainPNO2D.py.

    Maps (x_features, goal_features) → scalar distance field.

    With symmetric=True (paper setting):
        h = _asym_fwd(x - goal) + _asym_fwd(goal - x)
    which enforces d(x, g) = d(g, x).

    ConstrainedLinear in Ws ensures weights stay non-negative so the
    network can represent a proper norm.
    """
    def __init__(
        self,
        num_features: int,
        layers: tuple,
        activation,
        concave_activation_size: int = None,
        mode: str = "avg",
        symmetric: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.symmetric = symmetric

        assert len(layers) >= 2

        # Us[i]: skip-connection projections from raw difference h
        # Ws[i]: constrained residual weights
        self.Us = nn.ModuleList([nn.Linear(num_features, layers[0], bias=False)])
        self.Ws = nn.ModuleList([])

        for in_f, out_f in zip(layers[:-1], layers[1:]):
            self.Us.append(nn.Linear(num_features, out_f, bias=False))
            self.Ws.append(ConstrainedLinear(in_f, out_f, bias=False))

        self.activation = activation()
        self.output_activation = (
            ConcaveActivation(layers[-1], concave_activation_size)
            if concave_activation_size else nn.Identity()
        )
        self.reduce_metric = ReduceMetric(mode)

    def _asym_fwd(self, h: torch.Tensor) -> torch.Tensor:
        h1 = self.Us[0](h)
        for U, W in zip(self.Us[1:], self.Ws):
            h1 = self.activation(W(h1) + U(h))
        return h1

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = x - y
        if self.symmetric:
            h = self._asym_fwd(h) + self._asym_fwd(-h)
        else:
            h = self._asym_fwd(-h)
        h = self.activation(h)
        h = self.output_activation(h)
        return self.reduce_metric(h)


# ── Standard Spectral Conv 2D (no chi masking inside) ────────────────────────
# The DAFNO identity applies chi in the block-level formula, NOT inside the FFT.

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        assert in_channels == out_channels, "Depthwise SpectralConv2d requires in_channels == out_channels"
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / in_channels
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _cmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, C, X, Y), w: (C, X, Y) -> output: (B, C, X, Y)
        return torch.einsum("bcxy,cxy->bcxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self._cmul(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._cmul(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


# ── DAFNO Block (paper-faithful formula) ─────────────────────────────────────

class DAFNOBlock(nn.Module):
    """
    Paper DAFNO recurrence (from TrainPNO2D.py, line 292-298):

        conv_chi   = K(chi)
        conv_chix  = K(chi * x)
        xconv_chi  = x * K(chi)
        wx         = W(x)
        x_new = chi * (conv_chix - xconv_chi + wx)

    This is algebraically equivalent to the DAFNO identity:
        K_chi(x) = chi * (K(chi*x) - x*K(chi) + W(x))
    which enforces x_new = 0 wherever chi = 0 (inside obstacles).
    """
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral   = SpectralConv2d(width, width, modes1, modes2)
        self.local_conv = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
        # chi: (B, 1, H, W) → broadcast to (B, width, H, W)
        chi_w = chi.expand_as(x)

        conv_chix = self.spectral(chi_w * x)    # K(chi * x)
        wx        = self.local_conv(x)          # W(x)

        return chi_w * (conv_chix + wx)
