#!/usr/bin/env python3
# cli: python evaluate.py [--checkpoint] [--data_root] [--output_dir] [--smooth_coef] [--batch_size] [--max_samples] [--num_samples] [--split] [--modes] [--width] [--depth] [--padding] [--depthwise]

import argparse
import json
import os
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

class LpLoss:
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x, y):
        n = x.size(0)
        diff = torch.norm(x.reshape(n, -1) - y.reshape(n, -1), self.p, dim=1)
        base = torch.norm(y.reshape(n, -1), self.p, dim=1)
        return diff / (base + 1e-12)

def load_model(checkpoint_path, device, cli_args=None):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    width = state['fc0.weight'].shape[0]

    depth = sum(1 for k in state if k.startswith('conv') and k.endswith('.weights1'))

    w1_shape = state['conv0.weights1'].shape
    if len(w1_shape) == 3:
        depthwise = True
        modes = w1_shape[1]
    else:
        depthwise = False
        modes = w1_shape[2]

    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / 'model_config.json'
    padding = 9
    if config_path.exists():
        with open(config_path) as f:
            padding = json.load(f).get('padding', 9)

    cfg = {
        'modes': modes, 'width': width,
        'depth': depth, 'padding': padding,
        'depthwise': depthwise,
    }

    dw_str = ' (depthwise)' if depthwise else ''
    print(f"Inferred config: modes={modes}, width={width}, "
          f"layers={depth}{dw_str}")

    model = FNO2dMultiGoal(
        depth=depth,
        padding=padding,
        modes1=modes,
        modes2=modes,
        width=width,
        depthwise=depthwise,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {checkpoint_path}  |  {n_params:,} params")

    return model, cfg

def evaluate_dataset(model, data_dir, device, smooth_coef=5.0,
                     max_samples=None, batch_size=20, split='val'):
    full_ds = PNODataset(data_dir, smooth_coef=smooth_coef, max_samples=max_samples)
    N = len(full_ds)
    n_train = int(N * 0.8)
    n_val   = int(N * 0.1)

    if split == 'train':
        indices = range(0, n_train)
    elif split == 'val':
        indices = range(n_train, n_train + n_val)
    elif split == 'test':
        indices = range(n_train + n_val, N)
    else:
        indices = range(N)

    ds = torch.utils.data.Subset(full_ds, indices)
    print(f"  Split: {split}  ({len(ds)}/{N} samples)")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    loss_fn = LpLoss()

    all_losses, all_preds, all_chi, all_mask, all_y, all_goals = [], [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for chi, mask, y, goals in loader:
            chi, mask, y, goals = (chi.to(device), mask.to(device),
                                   y.to(device), goals.to(device))
            pred = model(chi, goals)
            pred_m = pred * mask
            y_m    = y    * mask
            losses = loss_fn(pred_m, y_m)

            all_losses.append(losses.cpu())
            all_preds.append(pred.cpu())
            all_chi.append(chi.cpu())
            all_mask.append(mask.cpu())
            all_y.append(y.cpu())
            all_goals.append(goals.cpu())

    return {
        'losses': torch.cat(all_losses),
        'preds':  torch.cat(all_preds),
        'chis':   torch.cat(all_chi),
        'masks':  torch.cat(all_mask),
        'ys':     torch.cat(all_y),
        'goals':  torch.cat(all_goals),
        'mean_loss': torch.cat(all_losses).mean().item(),
        'resolution': torch.cat(all_chi).shape[1],
    }

def plot_samples(results, title, save_path, num_samples=4):
    n = min(num_samples, len(results['preds']))
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    mean_loss = results['mean_loss']

    for row in range(n):
        chi_np  = results['chis'][row, :, :, 0].numpy()
        pred_np = results['preds'][row, :, :, 0].numpy()
        y_np    = results['ys'][row, :, :, 0].numpy()
        mask_np = results['masks'][row, :, :, 0].numpy()
        goal    = results['goals'][row].numpy()
        sample_loss = results['losses'][row].item()

        pred_vis = np.where(mask_np > 0.5, pred_np, np.nan)
        y_vis    = np.where(mask_np > 0.5, y_np, np.nan)
        vmax = np.nanmax(y_vis) if not np.all(np.isnan(y_vis)) else 1.0

        for col, (img, cmap, col_title) in enumerate([
            (chi_np,   'viridis', 'Input chi'),
            (pred_vis, 'plasma',  'Predicted'),
            (y_vis,    'plasma',  'Ground Truth'),
        ]):
            ax = axes[row, col]
            kwargs = dict(origin='lower', cmap=cmap)
            if col > 0:
                kwargs.update(vmin=0, vmax=vmax)
            im = ax.imshow(img, **kwargs)
            plt.colorbar(im, ax=ax, shrink=0.75)
            if row == 0:
                ax.set_title(col_title, fontsize=11)
            if col == 0:
                ax.set_ylabel(f'#{row}  L2={sample_loss:.3f}', fontsize=9)
            ax.plot(int(goal[1]), int(goal[0]), 'r*', markersize=8,
                    markeredgecolor='white')
            ax.axis('off')

    fig.suptitle(f'{title}\nL2 = {mean_loss:.5f}  |  Accuracy = {(1-mean_loss)*100:.2f}%',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {save_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model, cfg = load_model(args.checkpoint, device, cli_args=args)

    data_root = Path(args.data_root)
    data_dirs = sorted([d for d in data_root.iterdir()
                        if d.is_dir() and d.name.startswith('data_')])

    if not data_dirs:
        print(f"No data_* directories found in {data_root}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for data_dir in data_dirs:
        res_name = data_dir.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {res_name}")
        print(f"{'='*60}")

        results = evaluate_dataset(
            model, str(data_dir), device,
            smooth_coef=args.smooth_coef,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            split=args.split,
        )

        res = results['resolution']
        ml = results['mean_loss']
        acc = (1 - ml) * 100
        print(f"  Resolution: {res}x{res}")
        print(f"  Mean L2:    {ml:.5f}")
        print(f"  Accuracy:   {acc:.2f}%")

        all_stats.append({
            'dataset': res_name,
            'resolution': res,
            'mean_l2': ml,
            'accuracy_pct': acc,
            'n_samples': len(results['losses']),
        })

        plot_path = out_dir / f'eval_result_{res_name}.png'
        plot_samples(results, f'FNO @ {res}x{res}', plot_path,
                     num_samples=args.num_samples)

    stats_path = out_dir / 'eval_stats.txt'
    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"Model:      modes={cfg['modes']}, width={cfg['width']}, "
        f"layers={cfg['depth']}, depthwise={cfg.get('depthwise', False)}",
        f"Params:     {sum(p.numel() for p in model.parameters()):,}",
        "",
        f"{'Dataset':<20} {'Resolution':>10} {'L2 Loss':>10} {'Accuracy':>10} {'Samples':>8}",
    ]
    for s in all_stats:
        lines.append(
            f"{s['dataset']:<20} {s['resolution']:>10} "
            f"{s['mean_l2']:>10.5f} {s['accuracy_pct']:>9.2f}% {s['n_samples']:>8}"
        )

    with open(stats_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved stats: {stats_path}")

    print("\nSUMMARY")
    for s in all_stats:
        print(f"  {s['dataset']:20s}  L2={s['mean_l2']:.5f}  ({s['accuracy_pct']:.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate FNO on all available datasets")
    parser.add_argument('--checkpoint',  type=str, default='checkpoints/fno/model_best.ckpt')
    parser.add_argument('--data_root',   type=str, default='data')
    parser.add_argument('--output_dir',  type=str, default=None,
                        help="Where to save results (default: same dir as checkpoint)")
    parser.add_argument('--smooth_coef', type=float, default=5.0)
    parser.add_argument('--batch_size',  type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--split',       type=str, default='val',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to evaluate (80/10/10)')

    parser.add_argument('--modes',       type=int, default=12)
    parser.add_argument('--width',       type=int, default=32)
    parser.add_argument('--depth',  type=int, default=4)
    parser.add_argument('--padding',     type=int, default=9)
    parser.add_argument('--depthwise',   action='store_true')

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)

    main(args)
