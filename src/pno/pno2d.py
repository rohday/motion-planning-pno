# pno model — hybrid: 3-channel lift + paper-faithful DAFNO + DeepNormMetric

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pno.layers import DAFNOBlock, DeepNormMetric, MaxReLUPairwiseActivation


class PlanningNeuralOperator(nn.Module):
    def __init__(
        self,
        width: int = 48,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 4,
        padding: int = 9,
        beta: float = 5.0,
        deepnorm_hidden: int = 128,
    ):
        super().__init__()

        self.width   = width
        self.modes1  = modes1
        self.modes2  = modes2
        self.depth   = depth
        self.padding = padding
        self.beta    = beta

        # input lift: 3 channels (raw_map, sdf, goal_channel) → width
        self.fc0 = nn.Linear(3, width)

        # paper-faithful DAFNO spectral blocks
        self.blocks = nn.ModuleList([
            DAFNOBlock(width, modes1, modes2) for _ in range(depth)
        ])

        # paper-faithful output head: DeepNormMetric(width, (H, H), ...)
        # symmetric=True enforces d(x,g) = d(g,x); ConstrainedLinear
        # keeps the learned metric non-negative.
        self.deep_norm = DeepNormMetric(
            num_features=width,
            layers=(deepnorm_hidden, deepnorm_hidden),
            activation=lambda: MaxReLUPairwiseActivation(deepnorm_hidden),
            concave_activation_size=20,
            mode="avg",
            symmetric=True,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_sifn(self, sdf: torch.Tensor, raw_map: torch.Tensor) -> torch.Tensor:
        """Smooth indicator function: tanh(β·sdf)·(mask-0.5)+0.5"""
        return torch.tanh(self.beta * sdf) * (raw_map - 0.5) + 0.5

    def _build_goal_channel(
        self, shape: tuple, goal: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        B, _, H, W = shape
        g  = torch.zeros(B, 1, H, W, device=device)
        gx = goal[:, 0].long().clamp(0, W - 1)
        gy = goal[:, 1].long().clamp(0, H - 1)
        g[torch.arange(B, device=device), 0, gy, gx] = -1.0
        return g

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        raw_map: torch.Tensor,  # (B, 1, H, W)  binary occupancy
        sdf: torch.Tensor,      # (B, 1, H, W)  signed distance
        goal: torch.Tensor,     # (B, 2)         [col, row] pixel indices
    ) -> torch.Tensor:          # (B, 1, H, W)  value function

        B, _, H, W = raw_map.shape
        device = raw_map.device

        # 1. Smooth obstacle indicator (chi)
        chi = self._build_sifn(sdf, raw_map)           # (B, 1, H, W)

        # 2. Encode goal as a pixel-level channel
        goal_ch = self._build_goal_channel(raw_map.shape, goal, device)

        # 3. Lift [raw_map, sdf, goal_ch] → feature field
        x = torch.cat([raw_map, sdf, goal_ch], dim=1)  # (B, 3, H, W)
        x = x.permute(0, 2, 3, 1)                      # (B, H, W, 3)
        x = self.fc0(x)                                 # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)                      # (B, width, H, W)

        # 4. Pad (needed for spectral conv boundary artefacts)
        x     = F.pad(x,   [0, self.padding, 0, self.padding])
        chi_p = F.pad(chi, [0, self.padding, 0, self.padding])

        # 5. DAFNO blocks — GELU between layers, omitted after the last
        #    (matches paper: `if i < self.nlayers - 1: x = F.gelu(x)`)
        for i, block in enumerate(self.blocks):
            x = block(x, chi_p)
            if i < self.depth - 1:
                x = F.gelu(x)

        # 6. Crop padding
        x = x[..., :H, :W]                             # (B, width, H, W)

        # 7. Output head: DeepNormMetric
        #    Extract goal feature vector and compute learned norm field.
        x_feat = x.permute(0, 2, 3, 1)                 # (B, H, W, width)

        gx = goal[:, 0].long().clamp(0, W - 1)
        gy = goal[:, 1].long().clamp(0, H - 1)

        # goal feature: one vector per batch item → broadcast over spatial dims
        g_feat = x_feat[torch.arange(B, device=device), gy, gx, :]  # (B, width)
        g_feat = g_feat[:, None, None, :].expand_as(x_feat)          # (B, H, W, width)

        # flatten spatial → run metric
        x_flat = x_feat.reshape(-1, self.width)         # (B*H*W, width)
        g_flat = g_feat.reshape(-1, self.width)         # (B*H*W, width)

        out = self.deep_norm(x_flat, g_flat)            # (B*H*W,)
        return out.reshape(B, 1, H, W)
