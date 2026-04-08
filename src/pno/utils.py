# pno utils

import torch
import torch.nn as nn


class EikonalLoss(nn.Module):

    def __init__(self, dx: float = 1.0):
        super().__init__()
        self.dx = dx

    # pde loss
    def forward(self, V: torch.Tensor, raw_map: torch.Tensor) -> torch.Tensor:
        # central diff
        dVdy = (V[:, :, 2:, 1:-1] - V[:, :, :-2, 1:-1]) / (2 * self.dx)
        dVdx = (V[:, :, 1:-1, 2:] - V[:, :, 1:-1, :-2]) / (2 * self.dx)

        grad_mag = torch.sqrt(dVdx ** 2 + dVdy ** 2 + 1e-8)

        # free mask
        free = raw_map[:, :, 1:-1, 1:-1]

        # pde residual
        residual = (grad_mag - 1.0) ** 2

        # masked mean
        n_free = free.sum().clamp(min=1.0)
        return (residual * free).sum() / n_free


def save_checkpoint(model, optimizer, epoch, path, **extra):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
    }
    payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("epoch", 0)
    else:
        model.load_state_dict(ckpt)
        return 0
