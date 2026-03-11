#!/usr/bin/env python3
"""
Generate synthetic 2D motion planning dataset matching the official PNO format.

Pipeline (matches official generator):
  1. Generate random obstacle maps (binary mask)
  2. Compute SDF via scipy.ndimage.distance_transform_edt
  3. Compute velocity field: clip(SDF, 0, 1) * mask
  4. Solve Eikonal equation via scikit-fmm for random goal positions
  5. Cleanup: unreachable cells (travel_time > threshold) → 0

Output files (same format as HuggingFace dataset):
  mask.npy    — (N, H, W) float64, binary {0, 1}
  dist_in.npy — (N, H, W) float64, SDF values
  output.npy  — (N, H, W) float64, value function (0 at goal)
  goal.npy    — (N, 2)    int64,   goal coordinates [x, y]

Usage:
  python data_generation/generate.py --n_samples 5000 --resolution 64 --output_dir data/data_64x64_5k
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
from tqdm import tqdm

# Try pykonal first (used by official code), fallback to scikit-fmm
try:
    import pykonal
    SOLVER = 'pykonal'
except ImportError:
    try:
        import skfmm
        SOLVER = 'skfmm'
    except ImportError:
        print("ERROR: Need either pykonal or scikit-fmm. Install via:")
        print("  pip install pykonal   OR   pip install scikit-fmm")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Obstacle generation
# ---------------------------------------------------------------------------

def generate_random_obstacles(size, min_free_ratio=0.55, max_free_ratio=0.85):
    """
    Generate a random binary occupancy map with irregular blob-shaped obstacles.
    1 = free space, 0 = obstacle.
    
    Uses thresholded smoothed random noise to produce organic shapes 
    matching the official synthetic dataset's style.
    """
    from scipy.ndimage import gaussian_filter, label

    while True:
        # Random noise → smooth → threshold produces organic blob shapes
        noise = np.random.randn(size, size)
        # Sigma controls blob size: smaller = more scattered small blobs
        sigma = np.random.uniform(1.5, 3.5)
        smooth = gaussian_filter(noise, sigma=sigma)
        
        # Threshold to get binary map
        # Adjust threshold to control obstacle density
        threshold = np.random.uniform(-0.3, 0.3)
        mask = (smooth > threshold).astype(np.float64)

        # Ensure borders are free (no obstacles at edges)
        mask[0, :] = 1
        mask[-1, :] = 1
        mask[:, 0] = 1
        mask[:, -1] = 1

        free_ratio = np.mean(mask)
        if min_free_ratio <= free_ratio <= max_free_ratio:
            # Check that free space is mostly connected
            labeled, n_components = label(mask > 0.5)
            if n_components > 0:
                # Find largest connected component
                component_sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
                largest = max(component_sizes)
                # At least 80% of free space should be in largest component
                if largest / np.sum(mask > 0.5) > 0.8:
                    return mask


def calculate_signed_distance(mask):
    """Compute unsigned distance from obstacle boundary (matches official)."""
    return distance_transform_edt(mask != 0)


def solve_eikonal_pykonal(velocity, goal, size):
    """Solve Eikonal equation using pykonal (same as official code)."""
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = 1, 1, 1
    solver.velocity.npts = size, size, 1
    solver.velocity.values = velocity.reshape(size, size, 1)

    src_idx = int(goal[0]), int(goal[1]), 0
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()

    return solver.traveltime.values[:, :, 0]


def solve_eikonal_skfmm(velocity, goal, size):
    """Solve Eikonal equation using scikit-fmm.
    
    Uses skfmm.travel_time with proper phi setup.
    Obstacles (velocity=0) are masked in phi so FMM doesn't traverse them.
    """
    # phi: signed distance from goal. Negative = inside (goal), positive = outside
    phi = np.ones((size, size))
    phi[int(goal[0]), int(goal[1])] = -1

    # Mask obstacles in phi (FMM won't propagate through masked cells)
    obstacle_mask = velocity < 1e-10
    phi = np.ma.MaskedArray(phi, obstacle_mask)

    # Speed field (only in free space)
    speed = velocity.copy()
    speed[obstacle_mask] = 1.0  # dummy value — masked region anyway

    try:
        travel_time = skfmm.travel_time(phi, speed)
        result = np.array(travel_time.filled(0.0))
        return result
    except Exception:
        return None


def solve_eikonal(velocity, goal, size):
    if SOLVER == 'pykonal':
        return solve_eikonal_pykonal(velocity, goal, size)
    else:
        return solve_eikonal_skfmm(velocity, goal, size)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_sample(size, max_attempts=20):
    """Generate a single (mask, dist_in, output, goal) sample."""
    for _ in range(max_attempts):
        mask = generate_random_obstacles(size)
        sdf = calculate_signed_distance(mask)

        # Velocity field: clip SDF to [0, 1] and zero in obstacles
        velocity = np.clip(sdf, 0, 1) * mask

        # Pick a random goal in free space (not too close to boundary)
        free_cells = np.argwhere(mask > 0.5)
        if len(free_cells) < 10:
            continue

        # Filter goals that are away from obstacles (sdf > 1)
        valid_goals = free_cells[sdf[free_cells[:, 0], free_cells[:, 1]] > 1]
        if len(valid_goals) < 5:
            valid_goals = free_cells

        idx = np.random.randint(len(valid_goals))
        goal = valid_goals[idx]  # (row, col) = (y, x)

        # Solve Eikonal
        travel_time = solve_eikonal(velocity, goal, size)
        if travel_time is None:
            continue

        # Cleanup: unreachable cells → 0, also update mask
        high_mask = travel_time > 1000
        travel_time[high_mask] = 0.0

        input_mask = (velocity == 0)
        if (high_mask != input_mask).any():
            mask[high_mask] = 0.0
            sdf = calculate_signed_distance(mask)

        # Validate
        if np.all(travel_time == 0):
            continue
        if np.any(travel_time > 1000):
            continue

        # Goal convention: [x, y] = [col, row] (matching official)
        goal_xy = np.array([goal[1], goal[0]], dtype=np.int64)

        return mask, sdf, travel_time, goal_xy

    return None


def generate_dataset(n_samples, size, output_dir):
    """Generate the full dataset."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    masks = np.zeros((n_samples, size, size), dtype=np.float64)
    dists = np.zeros((n_samples, size, size), dtype=np.float64)
    outputs = np.zeros((n_samples, size, size), dtype=np.float64)
    goals = np.zeros((n_samples, 2), dtype=np.int64)

    generated = 0
    pbar = tqdm(total=n_samples, desc="Generating samples")

    while generated < n_samples:
        result = generate_sample(size)
        if result is None:
            continue

        mask, sdf, travel_time, goal = result
        masks[generated] = mask
        dists[generated] = sdf
        outputs[generated] = travel_time
        goals[generated] = goal
        generated += 1
        pbar.update(1)

    pbar.close()

    # Save
    np.save(str(out / 'mask.npy'), masks)
    np.save(str(out / 'dist_in.npy'), dists)
    np.save(str(out / 'output.npy'), outputs)
    np.save(str(out / 'goal.npy'), goals)

    print(f"\nDataset saved to {out.resolve()}")
    print(f"  mask.npy:    {masks.shape}  {masks.dtype}")
    print(f"  dist_in.npy: {dists.shape}  {dists.dtype}")
    print(f"  output.npy:  {outputs.shape}  {outputs.dtype}")
    print(f"  goal.npy:    {goals.shape}  {goals.dtype}")

    # Print stats
    free_ratios = np.mean(masks > 0.5, axis=(1, 2))
    max_vals = np.array([outputs[i][masks[i] > 0.5].max() for i in range(n_samples)])
    print(f"\n  Free space: {free_ratios.mean()*100:.1f}% ± {free_ratios.std()*100:.1f}%")
    print(f"  Output max: {max_vals.mean():.1f} ± {max_vals.std():.1f}")
    print(f"  Goal range: x=[{goals[:,0].min()}, {goals[:,0].max()}], "
          f"y=[{goals[:,1].min()}, {goals[:,1].max()}]")

    # Save preview image
    save_preview(masks, dists, outputs, goals, out, n_preview=5)


def save_preview(masks, dists, outputs, goals, out_dir, n_preview=5):
    """Save a preview image showing n_preview samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = min(n_preview, len(masks))
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row in range(n):
        mask = masks[row]
        val = np.where(mask > 0.5, outputs[row], np.nan)
        g = goals[row]

        ax = axes[row, 0]
        ax.imshow(mask, origin='lower', cmap='gray_r')
        ax.plot(g[0], g[1], 'r*', markersize=12)
        ax.axis('off')
        if row == 0: ax.set_title('Mask + Goal', fontsize=12)

        ax = axes[row, 1]
        ax.imshow(dists[row], origin='lower', cmap='viridis')
        ax.axis('off')
        if row == 0: ax.set_title('SDF', fontsize=12)

        ax = axes[row, 2]
        vmax = np.nanmax(val) if not np.all(np.isnan(val)) else 1
        ax.imshow(val, origin='lower', cmap='plasma', vmin=0, vmax=vmax)
        ax.plot(g[0], g[1], 'r*', markersize=12)
        ax.axis('off')
        if row == 0: ax.set_title('Value Function', fontsize=12)

    fig.suptitle(f'Dataset Preview  ({len(masks)} samples, {masks.shape[1]}×{masks.shape[2]})',
                 fontsize=13)
    plt.tight_layout()
    preview_path = Path(out_dir) / 'preview.png'
    plt.savefig(str(preview_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Preview: {preview_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic PNO dataset")
    parser.add_argument('--n_samples',  type=int, default=5000)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='data/data_64x64_5k')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"Generating {args.n_samples} samples at {args.resolution}×{args.resolution}")
    print(f"Eikonal solver: {SOLVER}")
    generate_dataset(args.n_samples, args.resolution, args.output_dir)
