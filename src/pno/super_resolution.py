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

        # Feed unscaled SDF to PNO to prevent fc0 explosions
        value = self.pno(raw_map, sdf, goal)

        # Scale output to pixel distances
        value = value * scale

        if return_sdf:
            # Return properly scaled SDF magnitudes if requested
            return value, sdf * scale
        return value


class HierarchicalSuperResolutionPNO(nn.Module):
    """
    Hierarchical inference wrapper to bypass the Spectral Resolution Limit.
    
    When running zero-shot super-resolution on high-resolution maps (e.g. 256x256),
    the fixed `modes=12` acts as a severe low-pass filter, blurring out 1-pixel walls
    and narrow corridors, which incorrectly blocks paths.
    
    This wrapper:
    1. Downsamples the target map to the network's native 64x64 using Max Pooling
       (to preserve free space corridors).
    2. Runs the FNO and PNO natively at 64x64, ensuring the network stays perfectly
       in distribution (no frequency shifting, no SDF explosions).
    3. Upsamples the predicted 64x64 value function back to target resolution and
       scales the magnitudes to match target pixel distances.
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
        target_res = raw_map.shape[-1]
        scale = target_res / self.train_res
        pool_stride = int(scale)

        # 1. Downsample raw map using Max Pooling
        # Free space is > 0.5, obstacles are 0. Max pooling ensures that if ANY
        # pixel in the pool window is free, the downsampled pixel is free.
        # This keeps narrow corridors open.
        import torch.nn.functional as F
        raw_map_64 = F.max_pool2d(raw_map, kernel_size=pool_stride, stride=pool_stride)
        
        # 2. Downsample goal to 64x64 coordinate space
        goal_64 = goal / scale
        
        # 3. FNO SDF (Native 64x64)
        x_in = raw_map_64.clone()
        if self.fno_normalize_input:
            x_in = (x_in - self.fno_x_mean) / self.fno_x_std

        sdf_64 = self.fno(x_in)

        if self.fno_normalize_target:
            sdf_64 = sdf_64 * self.fno_y_std + self.fno_y_mean
            
        # 4. PNO Native Evaluation
        val_64 = self.pno(raw_map_64, sdf_64, goal_64)
        
        # 5. Upsample and Scale
        val_target = F.interpolate(val_64, scale_factor=scale, mode='bilinear', align_corners=False)
        val_target = val_target * scale
        
        if return_sdf:
            sdf_target = F.interpolate(sdf_64, scale_factor=scale, mode='bilinear', align_corners=False)
            sdf_target = sdf_target * scale
            return val_target, sdf_target
            
        return val_target
