#!/usr/bin/env python3
# cli: python -m src.data_generation.generate_dataset [--n_samples 10000] [--resolution 64] [--output_dir data/pno_10k] [--seed 42] [--workers 8] [--min_free_ratio 0.55] [--max_free_ratio 0.75]
# NOTE: This version generates CLUTTERED / JAGGED / FRAGMENTED maps
#       (high-frequency obstacles). See generate_dataset_smooth_backup.py
#       for the original smooth-blob pipeline.

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import skfmm
from scipy.ndimage import (
    binary_closing,
    distance_transform_edt,
    label,
    zoom,
)


def generate_random_map(size, rng, min_free=0.55, max_free=0.75):
    """Generate a cluttered 2-D occupancy map with jagged, fragmented obstacles.

    Pipeline
    --------
    1. Generate white noise at a *low* resolution (1/scale of target),
       threshold it, then upscale with nearest-neighbour interpolation.
       This produces blocky / fractal-like obstacle edges instead of smooth
       Gaussian blobs.
    2. Optionally layer a second, sparser noise pass at a different scale to
       add additional clutter diversity.
    3. Apply a minimal binary_closing (2×2 kernel) only to ensure narrow
       1-pixel gaps don't completely isolate free regions.  NO binary_opening
       is applied — small obstacle specks are intentionally preserved.
    4. Keep the largest free-space connected component, add boundary walls,
       and enforce the free-ratio constraint.
    """
    for _ in range(300):
        # --- 1. Blocky low-res noise → nearest-neighbour upscale -----------
        #  scale_factor in [3, 6]: lower → blockier / larger features
        scale_factor = rng.integers(3, 7)  # 3..6 inclusive
        lo_size = max(4, size // scale_factor)

        noise_lo = rng.standard_normal((lo_size, lo_size))
        threshold = rng.uniform(-0.25, 0.35)
        mask_lo = (noise_lo > threshold)  # True = free, False = obstacle

        # Up-scale to full resolution (order=0 → nearest-neighbour)
        zoom_factor = size / lo_size
        mask = zoom(mask_lo.astype(np.float64), zoom_factor, order=0) > 0.5

        # --- 2. Optional second noise layer at finer scale for extra scatter
        if rng.random() < 0.6:
            fine_factor = rng.integers(2, 4)  # 2..3
            fine_size = max(8, size // fine_factor)
            noise_fine = rng.standard_normal((fine_size, fine_size))
            fine_thresh = rng.uniform(-0.1, 0.4)
            mask_fine = zoom(
                (noise_fine > fine_thresh).astype(np.float64),
                size / fine_size, order=0,
            ) > 0.5
            # Combine: a cell is free only if BOTH layers say free
            mask = mask & mask_fine

        # --- 3. Minimal closing (2×2) to seal hairline gaps ----------------
        struct_small = np.ones((2, 2), dtype=bool)
        mask = binary_closing(mask, structure=struct_small)

        # --- 4. Largest free-space component + boundary walls --------------
        free_labeled, n_free = label(mask)
        if n_free == 0:
            continue
        free_sizes = np.bincount(free_labeled.ravel())
        if len(free_sizes) <= 1:
            continue
        largest_free_id = int(np.argmax(free_sizes[1:]) + 1)
        mask = (free_labeled == largest_free_id)

        mask = mask.astype(np.float64)
        mask[0, :] = 0.0
        mask[-1, :] = 0.0
        mask[:, 0] = 0.0
        mask[:, -1] = 0.0

        free_ratio = mask.mean()
        if not (min_free <= free_ratio <= max_free):
            continue
        return mask
    return None


def compute_sdf(mask):
    return distance_transform_edt(mask != 0) + distance_transform_edt(mask == 0)


def compute_value_function(mask, sdf, goal):
    size = mask.shape[0]
    speed = np.clip(sdf, 0, 1) * mask

    phi = np.ones((size, size), dtype=np.float64)
    phi[goal[0], goal[1]] = -1

    obstacle_mask = speed < 1e-10
    phi = np.ma.MaskedArray(phi, obstacle_mask)
    speed_safe = speed.copy()
    speed_safe[obstacle_mask] = 1.0

    try:
        tt = skfmm.travel_time(phi, speed_safe)
        result = np.array(tt.filled(0.0), dtype=np.float64)
    except Exception:
        return None

    result[result > 1e6] = 0.0
    return result


def pick_goal(mask, sdf, rng):
    free = np.argwhere(mask > 0.5)
    if len(free) < 10:
        return None
    valid = free[sdf[free[:, 0], free[:, 1]] > 1.0]
    if len(valid) < 5:
        valid = free
    return valid[rng.integers(len(valid))]


def generate_one_sample(args):
    seed, size, min_free, max_free = args
    rng = np.random.default_rng(seed)

    for _ in range(10):
        mask = generate_random_map(size, rng, min_free, max_free)
        if mask is None:
            continue

        sdf = compute_sdf(mask)
        goal = pick_goal(mask, sdf, rng)
        if goal is None:
            continue

        value = compute_value_function(mask, sdf, goal)
        if value is None or np.all(value == 0):
            continue

        high = value > 1000
        if high.any():
            value[high] = 0.0
            mask[high] = 0.0
            sdf = compute_sdf(mask)

        if np.any(value > 1000):
            continue

        goal_xy = np.array([goal[1], goal[0]], dtype=np.int64)
        return mask, sdf, goal_xy, value
    return None


def generate_dataset(n_samples, resolution, output_dir, seed, workers,
                     min_free, max_free):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 2**31, size=n_samples * 3).tolist()

    print(f"Generating {n_samples} samples at {resolution}x{resolution} with {workers} workers")
    t0 = time.time()

    all_masks, all_sdfs, all_goals, all_values = [], [], [], []
    generated = 0
    seed_idx = 0

    while generated < n_samples:
        batch = min(n_samples - generated, max(workers * 4, 200))
        task_args = [(seeds[seed_idx + i], resolution, min_free, max_free)
                     for i in range(batch)]
        seed_idx += batch
        if seed_idx >= len(seeds):
            extra = base_rng.integers(0, 2**31, size=n_samples).tolist()
            seeds.extend(extra)

        with mp.Pool(workers) as pool:
            results = pool.map(generate_one_sample, task_args)

        for r in results:
            if r is None or generated >= n_samples:
                continue
            mask, sdf, goal, value = r
            all_masks.append(mask)
            all_sdfs.append(sdf)
            all_goals.append(goal)
            all_values.append(value)
            generated += 1

        elapsed = time.time() - t0
        rate = generated / max(elapsed, 0.1)
        print(f"  {generated}/{n_samples} ({rate:.1f} samples/s)")

    masks_np = np.stack(all_masks[:n_samples])    # (N, H, W) float64
    sdfs_np = np.stack(all_sdfs[:n_samples])      # (N, H, W) float64
    goals_np = np.stack(all_goals[:n_samples])    # (N, 2)    int64
    values_np = np.stack(all_values[:n_samples])  # (N, H, W) float64

    np.save(str(out / "mask.npy"), masks_np)
    np.save(str(out / "dist_in.npy"), sdfs_np)
    np.save(str(out / "goal.npy"), goals_np)
    np.save(str(out / "output.npy"), values_np)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({n_samples / elapsed:.1f} samples/s)")
    print(f"Saved to {out.resolve()}/")
    print(f"  mask.npy:    {masks_np.shape}  {masks_np.dtype}")
    print(f"  dist_in.npy: {sdfs_np.shape}  {sdfs_np.dtype}")
    print(f"  goal.npy:    {goals_np.shape}  {goals_np.dtype}")
    print(f"  output.npy:  {values_np.shape}  {values_np.dtype}")
    print(f"  Free ratio: {masks_np.mean():.3f}")
    print(f"  SDF range: [{sdfs_np.min():.1f}, {sdfs_np.max():.1f}]")
    fv = values_np[values_np > 0]
    print(f"  Value range: [{fv.min():.2f}, {fv.max():.2f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PNO training dataset")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="data/pno_10k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--min_free_ratio", type=float, default=0.55)
    parser.add_argument("--max_free_ratio", type=float, default=0.75)
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        resolution=args.resolution,
        output_dir=args.output_dir,
        seed=args.seed,
        workers=args.workers,
        min_free=args.min_free_ratio,
        max_free=args.max_free_ratio,
    )
