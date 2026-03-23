#!/usr/bin/env python3
# cli: python data_generation/generate.py [--n_samples] [--resolution] [--output_dir] [--seed]

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

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

def generate_random_obstacles(size, min_free_ratio=0.65, max_free_ratio=0.73):
    """Generate random obstacle masks with controlled free-space ratio."""
    from scipy.ndimage import gaussian_filter, label

    while True:
        noise = np.random.randn(size, size)
        sigma = np.random.uniform(1.8, 3.0)
        smooth = gaussian_filter(noise, sigma=sigma)
        threshold = np.random.uniform(-0.2, 0.2)
        mask = (smooth > threshold).astype(np.float64)

        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0

        free_ratio = np.mean(mask)
        if min_free_ratio <= free_ratio <= max_free_ratio:
            labeled, n_components = label(mask > 0.5)
            if n_components > 0:
                component_sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
                largest = max(component_sizes)
                if largest / np.sum(mask > 0.5) > 0.8:
                    return mask


def calculate_signed_distance(mask):
    return distance_transform_edt(mask != 0)


def solve_eikonal_pykonal(velocity, goal, size):
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
    phi = np.ones((size, size))
    phi[int(goal[0]), int(goal[1])] = -1

    obstacle_mask = velocity < 1e-10
    phi = np.ma.MaskedArray(phi, obstacle_mask)

    speed = velocity.copy()
    speed[obstacle_mask] = 1.0

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

def generate_sample(size, max_attempts=20):
    for _ in range(max_attempts):
        mask = generate_random_obstacles(size)
        sdf = calculate_signed_distance(mask)

        velocity = np.clip(sdf, 0, 1) * mask

        free_cells = np.argwhere(mask > 0.5)
        if len(free_cells) < 10:
            continue

        valid_goals = free_cells[sdf[free_cells[:, 0], free_cells[:, 1]] > 1]
        if len(valid_goals) < 5:
            valid_goals = free_cells

        idx = np.random.randint(len(valid_goals))
        goal = valid_goals[idx]

        travel_time = solve_eikonal(velocity, goal, size)
        if travel_time is None:
            continue

        high_mask = travel_time > 1000
        travel_time[high_mask] = 0.0

        input_mask = (velocity == 0)
        if (high_mask != input_mask).any():
            mask[high_mask] = 0.0
            sdf = calculate_signed_distance(mask)

        if np.all(travel_time == 0):
            continue
        if np.any(travel_time > 1000):
            continue

        goal_xy = np.array([goal[1], goal[0]], dtype=np.int64)

        return mask, sdf, travel_time, goal_xy

    return None


def generate_dataset(n_samples, size, output_dir):
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

    np.save(str(out / 'mask.npy'), masks)
    np.save(str(out / 'dist_in.npy'), dists)
    np.save(str(out / 'output.npy'), outputs)
    np.save(str(out / 'goal.npy'), goals)

    print(f"\nDataset saved to {out.resolve()}")
    print(f"  mask.npy:    {masks.shape}  {masks.dtype}")
    print(f"  dist_in.npy: {dists.shape}  {dists.dtype}")
    print(f"  output.npy:  {outputs.shape}  {outputs.dtype}")
    print(f"  goal.npy:    {goals.shape}  {goals.dtype}")

    free_ratios = np.mean(masks > 0.5, axis=(1, 2))
    max_vals = np.array([outputs[i][masks[i] > 0.5].max() for i in range(n_samples)])
    print(f"\n  Free space: {free_ratios.mean()*100:.1f}% ± {free_ratios.std()*100:.1f}%")
    print(f"  Output max: {max_vals.mean():.1f} ± {max_vals.std():.1f}")
    print(f"  Goal range: x=[{goals[:,0].min()}, {goals[:,0].max()}], "
          f"y=[{goals[:,1].min()}, {goals[:,1].max()}]")

    save_preview(masks, dists, outputs, goals, out, n_preview=5)


def save_preview(masks, dists, outputs, goals, out_dir, n_preview=5):
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
