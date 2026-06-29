# Super-resolution inference utilities for PNO.
#
# The PNO (and its upstream SDF-FNO) are trained at a fixed resolution
# (typically 64×64) but can run at arbitrary resolutions thanks to the
# resolution-invariant properties of Fourier Neural Operators.
#
# Three quantities need explicit scaling at inference time:
#   1. Goal coordinates — must be in target-resolution pixel space.
#   2. SDF magnitude   — FNO outputs training-res magnitudes; multiply by scale.
#   3. Output values    — PNO outputs unit-domain distances; multiply by scale
#                         to get pixel-distance heuristics suitable for A*.

import torch
import torch.nn as nn


def scale_sdf(sdf: torch.Tensor, target_res: int, train_res: int = 64) -> torch.Tensor:
    """Scale FNO-predicted SDF from training-resolution magnitudes to
    target-resolution magnitudes.

    The SDF FNO learns to output distance values calibrated to the training
    grid spacing.  At higher resolution the same physical distance spans
    more pixels, so we multiply by the resolution ratio.
    """
    return sdf * (target_res / train_res)


def scale_value(value: torch.Tensor, target_res: int, train_res: int = 64) -> torch.Tensor:
    """Scale PNO output from unit-domain distances to pixel distances.

    The PNO value function is learned on a normalised [0,1]² domain.  To
    use it as an A* heuristic at a different resolution, multiply by the
    resolution ratio so that the numeric magnitudes match actual pixel
    distances in the search grid.
    """
    return value * (target_res / train_res)


def scale_goal(goal: torch.Tensor, target_res: int, train_res: int = 64) -> torch.Tensor:
    """Scale goal coordinates from training-resolution pixel space to
    target-resolution pixel space.

    Args:
        goal: (B, 2)  goal coordinates in *training* resolution pixels.
        target_res:    spatial size of the inference map.
        train_res:     spatial size used during training (default 64).

    Returns:
        (B, 2) goal coordinates in *target* resolution pixels.
    """
    return goal * (target_res / train_res)


class SuperResolutionPNO(nn.Module):
    """Convenience wrapper that runs the full FNO → PNO pipeline with
    correct resolution scaling.

    Usage::

        sr = SuperResolutionPNO(fno, pno, train_res=64)
        sr.eval()

        # raw_map: (B, 1, 256, 256)  binary map at target resolution
        # goal:    (B, 2)            goal in *target* resolution pixel coords
        heuristic = sr(raw_map, goal)   # (B, 1, 256, 256) pixel-distance values
    """

    def __init__(
        self,
        fno: nn.Module,
        pno: nn.Module,
        train_res: int = 64,
        fno_x_mean: float = 0.0,
        fno_x_std: float = 1.0,
        fno_y_mean: float = 0.0,
        fno_y_std: float = 1.0,
        fno_normalize_input: bool = False,
        fno_normalize_target: bool = False,
    ):
        super().__init__()
        self.fno = fno
        self.pno = pno
        self.train_res = train_res
        self.fno_x_mean = fno_x_mean
        self.fno_x_std = max(fno_x_std, 1e-6)
        self.fno_y_mean = fno_y_mean
        self.fno_y_std = max(fno_y_std, 1e-6)
        self.fno_normalize_input = fno_normalize_input
        self.fno_normalize_target = fno_normalize_target

    @torch.no_grad()
    def forward(
        self,
        raw_map: torch.Tensor,
        goal: torch.Tensor,
        return_sdf: bool = False,
    ) -> torch.Tensor:
        """Run full pipeline with correct super-resolution scaling.

        Args:
            raw_map: (B, 1, H, W) binary occupancy map at target resolution.
            goal:    (B, 2) goal coordinates in *target-resolution* pixel space.
            return_sdf: if True, return (value, sdf) tuple.

        Returns:
            (B, 1, H, W) value function scaled to pixel distances,
            suitable for direct use as an A* heuristic.
        """
        target_res = raw_map.shape[-1]
        scale = target_res / self.train_res

        # --- SDF prediction with scaling ---
        x_in = raw_map.clone()
        if self.fno_normalize_input:
            x_in = (x_in - self.fno_x_mean) / self.fno_x_std

        sdf = self.fno(x_in)

        if self.fno_normalize_target:
            sdf = sdf * self.fno_y_std + self.fno_y_mean

        # Scale SDF to target-resolution magnitudes
        sdf = sdf * scale

        # --- PNO value prediction with scaling ---
        value = self.pno(raw_map, sdf, goal)

        # Scale output to pixel distances
        value = value * scale

        if return_sdf:
            return value, sdf
        return value
