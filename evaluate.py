#!/usr/bin/env python3
"""
Evaluate a trained FNO2dMultiGoal on the official PNO synthetic dataset.

Usage:
    python evaluate.py \\
        --checkpoint checkpoints/fno/model_best.ckpt \\
        --data_dir   data/synthetic \\
        --ntrain 500 --ntest 50 \\
        --subsample 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.fno2d import FNO2dMultiGoal
from data.loader import PNODataset


class LpLoss:
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x, y):
        n = x.size(0)
        diff = torch.norm(x.reshape(n, -1) - y.reshape(n, -1), self.p, dim=1)
        base = torch.norm(y.reshape(n, -1), self.p, dim=1)
        return (diff / (base + 1e-12)).mean()


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load model ---------------------------------------------------------
    model = FNO2dMultiGoal(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        num_layers=args.num_layers,
        padding=args.padding,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- Load test data -----------------------------------------------------
    full_ds = PNODataset(
        args.data_dir,
        smooth_coef=args.smooth_coef,
        subsample=args.subsample,
        max_samples=args.ntrain + args.ntest,
    )
    test_indices = list(range(args.ntrain, args.ntrain + args.ntest))
    test_ds = torch.utils.data.Subset(full_ds, test_indices)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    # ---- Compute loss -------------------------------------------------------
    loss_fn = LpLoss()
    all_losses, all_preds, all_chi, all_mask, all_y, all_goals = [], [], [], [], [], []

    with torch.no_grad():
        for chi, mask, y, goals in test_loader:
            chi, mask, y, goals = (chi.to(device), mask.to(device),
                                   y.to(device), goals.to(device))
            pred = model(chi, goals)
            pred_m = pred * mask
            y_m    = y    * mask
            l = loss_fn(pred_m, y_m)
            all_losses.append(l.item())
            all_preds.append(pred.cpu())
            all_chi.append(chi.cpu())
            all_mask.append(mask.cpu())
            all_y.append(y.cpu())
            all_goals.append(goals.cpu())

    mean_loss = float(np.mean(all_losses))
    print(f"\nTest relative L2 loss: {mean_loss:.5f}  ({(1-mean_loss)*100:.2f}% accuracy)")

    # Save stats
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'eval_stats.txt', 'w') as f:
        f.write(f"Checkpoint:       {args.checkpoint}\n")
        f.write(f"Data:             {args.data_dir}\n")
        f.write(f"Subsample:        {args.subsample}\n")
        f.write(f"N test samples:   {args.ntest}\n\n")
        f.write(f"Relative L2 loss: {mean_loss:.5f}\n")
        f.write(f"Accuracy:         {(1-mean_loss)*100:.2f}%\n")
    print(f"Saved: {out}/eval_stats.txt")

    # ---- Visualize ----------------------------------------------------------
    preds = torch.cat(all_preds)   # (N, H, W, 1)
    chis  = torch.cat(all_chi)
    masks = torch.cat(all_mask)
    ys    = torch.cat(all_y)
    goals_all = torch.cat(all_goals)

    n_vis = min(args.num_samples, len(preds))
    fig, axes = plt.subplots(n_vis, 3, figsize=(12, 4 * n_vis))
    if n_vis == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_vis):
        chi_np  = chis[row, :, :, 0].numpy()
        pred_np = preds[row, :, :, 0].numpy()
        y_np    = ys[row, :, :, 0].numpy()
        mask_np = masks[row, :, :, 0].numpy()
        goal    = goals_all[row].numpy()

        # Mask out obstacles for visualization
        pred_vis = np.where(mask_np > 0.5, pred_np, np.nan)
        y_vis    = np.where(mask_np > 0.5, y_np, np.nan)

        vmax = np.nanmax(y_vis) if not np.all(np.isnan(y_vis)) else 1.0

        for col, (img, title, cmap) in enumerate([
            (chi_np,   'Input chi',       'viridis'),
            (pred_vis, 'Predicted value', 'plasma'),
            (y_vis,    'Ground truth',    'plasma'),
        ]):
            ax = axes[row, col]
            im = ax.imshow(img, origin='lower', cmap=cmap,
                           vmin=0, vmax=vmax if col > 0 else None)
            plt.colorbar(im, ax=ax, shrink=0.75)
            if row == 0:
                ax.set_title(title, fontsize=11)
            # Mark goal
            ax.plot(int(goal[1]), int(goal[0]), 'r*', markersize=10, markeredgecolor='white')
            ax.axis('off')

    plt.suptitle(f"FNO Evaluation  |  L2={mean_loss:.4f}  ({(1-mean_loss)*100:.1f}%)",
                 fontsize=13)
    plt.tight_layout()
    plot_path = out / 'eval_result.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--data_dir',    type=str, required=True)
    parser.add_argument('--output_dir',  type=str, default=None)
    parser.add_argument('--ntrain',      type=int, default=500)
    parser.add_argument('--ntest',       type=int, default=50)
    parser.add_argument('--subsample',   type=int, default=1)
    parser.add_argument('--smooth_coef', type=float, default=5.0)
    parser.add_argument('--modes',       type=int, default=12)
    parser.add_argument('--width',       type=int, default=32)
    parser.add_argument('--num_layers',  type=int, default=4)
    parser.add_argument('--padding',     type=int, default=9)
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=4)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)
    evaluate(args)
