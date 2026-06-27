import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNormProjection(nn.Module):
    def __init__(self, width: int, hidden: int = 64):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(hidden, width) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hidden))
        self.weight2 = nn.Parameter(torch.randn(1, hidden) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(1))

    def forward(self, v_flat, g_flat):
        # The new pno2d.py passes x_flat, g_flat
        # v_flat: (B*H*W, C)
        # g_flat: (B*H*W, C)
        
        delta = torch.abs(v_flat - g_flat)
        w1_pos = F.softplus(self.weight1)
        h = F.gelu(delta @ w1_pos.t() + self.bias1)
        
        w2_pos = F.softplus(self.weight2)
        out = h @ w2_pos.t() + self.bias2
        
        return out.squeeze(-1) # -> (B*H*W,)
