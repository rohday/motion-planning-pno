#!/usr/bin/env python3
"""Path planning demo with a trained FNO model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fno.fno2d import FNO2dMultiGoal
from data.loader import PNODataset
from evaluate import load_model
from src.pno.utils.path_extraction import extract_path


def run_path_planning(model, dataset, device, indices, step_size=0.8):
    """Run inference + path extraction on selected samples."""
    results = []
    model.eval()

    for idx in indices:
        chi, mask, y_true, goal = dataset[idx]
        chi_b = chi.unsqueeze(0).to(device)
        goal_b = goal.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(chi_b.clone(), goal_b)

        pred_np = pred[0, :, :, 0].cpu().numpy()
        mask_np = mask[:, :, 0].numpy()
        y_np = y_true[:, :, 0].numpy()
        goal_px = goal.numpy()

        pred_np = pred_np * mask_np
        goal_yx = (int(goal_px[1]), int(goal_px[0]))

        chi_np = chi[:, :, 0].numpy()
        free_cells = np.argwhere(mask_np > 0.5)
        interior = free_cells[
            (free_cells[:, 0] > 2) & (free_cells[:, 0] < mask_np.shape[0] - 3) &
            (free_cells[:, 1] > 2) & (free_cells[:, 1] < mask_np.shape[1] - 3)
        ]
        if len(interior) < 5:
            interior = free_cells
        dists = np.sqrt((interior[:, 0] - goal_yx[0])**2 +
                        (interior[:, 1] - goal_yx[1])**2)
        start_idx = np.argmax(dists)
        start_yx = tuple(interior[start_idx])

        path_pred, reached_pred = extract_path(
            pred_np, start_yx, goal_yx, mask_np,
            step_size=step_size, max_steps=500,
        )

        path_gt, reached_gt = extract_path(
            y_np, start_yx, goal_yx, mask_np,
            step_size=step_size, max_steps=500,
        )

        results.append({
            'idx': idx,
            'mask': mask_np,
            'pred_v': pred_np,
            'true_v': y_np,
            'goal_yx': goal_yx,
            'start_yx': start_yx,
            'path_pred': path_pred,
            'path_gt': path_gt,
            'reached_pred': reached_pred,
            'reached_gt': reached_gt,
        })

    return results


def plot_results(results, save_path):
    """Plot mask, predicted paths, and value fields."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, r in enumerate(results):
        mask = r['mask']
        pred_v = np.where(mask > 0.5, r['pred_v'], np.nan)
        true_v = np.where(mask > 0.5, r['true_v'], np.nan)
        sy, sx = r['start_yx']
        gy, gx = r['goal_yx']

        ax = axes[row, 0]
        ax.imshow(mask, origin='lower', cmap='gray_r', alpha=0.4)
        if r['path_gt']:
            gt_pts = np.array(r['path_gt'])
            ax.plot(gt_pts[:, 1], gt_pts[:, 0], 'b-', linewidth=1.5,
                    alpha=0.6, label='GT path')
        if r['path_pred']:
            pr_pts = np.array(r['path_pred'])
            ax.plot(pr_pts[:, 1], pr_pts[:, 0], 'r-', linewidth=2,
                    label='FNO path')
        ax.plot(gx, gy, 'g*', markersize=14, markeredgecolor='white',
                label='Goal')
        ax.plot(sx, sy, 'ms', markersize=10, markeredgecolor='white',
                label='Start')
        status = '✓' if r['reached_pred'] else '✗'
        ax.set_title(f"#{r['idx']}  {status}  steps={len(r['path_pred'])}",
                     fontsize=10)
        ax.axis('off')
        if row == 0:
            ax.legend(fontsize=7, loc='upper left')

        ax = axes[row, 1]
        vmax = np.nanmax(true_v) if not np.all(np.isnan(true_v)) else 1
        ax.imshow(pred_v, origin='lower', cmap='plasma', vmin=0, vmax=vmax)
        ax.plot(gx, gy, 'r*', markersize=10, markeredgecolor='white')
        ax.axis('off')
        if row == 0:
            ax.set_title('Predicted V', fontsize=12)

        ax = axes[row, 2]
        ax.imshow(true_v, origin='lower', cmap='plasma', vmin=0, vmax=vmax)
        ax.plot(gx, gy, 'r*', markersize=10, markeredgecolor='white')
        ax.axis('off')
        if row == 0:
            ax.set_title('Ground Truth V', fontsize=12)

    n_reached = sum(1 for r in results if r['reached_pred'])
    fig.suptitle(f'PNO Path Planning — {n_reached}/{n} paths reached goal',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PNO Path Planning Demo")
    parser.add_argument('--checkpoint', type=str,
                        default='allmodels/fno_m12w32d4/model_best.ckpt')
    parser.add_argument('--data', type=str, default='data/data_64x64')
    parser.add_argument('--num_samples', type=int, default=6)
    parser.add_argument('--step_size', type=float, default=0.8)
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (default: next to checkpoint)')
    parser.add_argument('--smooth_coef', type=float, default=5.0)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'])

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)

    ds = PNODataset(args.data, smooth_coef=args.smooth_coef)
    N = len(ds)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)

    if args.split == 'train':
        indices = list(range(0, n_train))
    elif args.split == 'val':
        indices = list(range(n_train, n_train + n_val))
    elif args.split == 'test':
        indices = list(range(n_train + n_val, N))
    else:
        indices = list(range(N))

    n = min(args.num_samples, len(indices))
    selected = np.random.choice(indices, n, replace=False)
    selected.sort()
    print(f"Running path planning on {n} samples from {args.split} split...")

    results = run_path_planning(model, ds, device, selected, args.step_size)

    n_reached = sum(1 for r in results if r['reached_pred'])
    n_reached_gt = sum(1 for r in results if r['reached_gt'])
    print(f"\nResults:")
    print(f"  FNO paths reached goal:  {n_reached}/{n}")
    print(f"  GT paths reached goal:   {n_reached_gt}/{n}")

    for r in results:
        status = '✓' if r['reached_pred'] else '✗'
        print(f"  #{r['idx']:3d}: {status}  steps={len(r['path_pred']):3d}  "
              f"start={r['start_yx']}  goal={r['goal_yx']}")

    out_path = args.output or str(Path(args.checkpoint).parent / 'path_planning.png')
    plot_results(results, out_path)


if __name__ == '__main__':
    main()
