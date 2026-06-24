#!/usr/bin/env python3
# cli: python evaluate_pno.py [--checkpoint] [--cache] [--output_dir] [--batch_size] [--max_samples] [--num_samples] [--split]

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pno import PlanningNeuralOperator


def _split_indices(n: int, split: str):
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    if split == 'train':
        return range(0, n_train)
    if split == 'val':
        return range(n_train, n_train + n_val)
    if split == 'test':
        return range(n_train + n_val, n)
    return range(n)


def _to_nchw(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32))
    if t.ndim == 3:
        return t.unsqueeze(1)
    if t.ndim == 4 and t.shape[1] == 1:
        return t
    if t.ndim == 4 and t.shape[-1] == 1:
        return t.permute(0, 3, 1, 2)
    raise ValueError(f"Expected (N,H,W), (N,1,H,W) or (N,H,W,1), got {tuple(t.shape)}")


def _infer_model_config(state: dict, checkpoint_path: str):
    width = int(state['fc0.weight'].shape[0])
    depth = len({k.split('.')[1] for k in state if k.startswith('blocks.') and k.endswith('.spectral.weights1')})
    modes = int(state['blocks.0.spectral.weights1'].shape[2])
    deepnorm_hidden = int(state['deepnorm.weight1'].shape[0])

    cfg = {
        'width': width,
        'depth': depth,
        'modes': modes,
        'padding': 9,
        'beta': 5.0,
        'deepnorm_hidden': deepnorm_hidden,
        'cache': 'data/cache_64x64/pno_cache.npz',
    }

    config_path = Path(checkpoint_path).parent / 'model_config.json'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        cfg['padding'] = int(saved.get('padding', cfg['padding']))
        cfg['beta'] = float(saved.get('beta', cfg['beta']))
        cfg['cache'] = str(saved.get('cache', cfg['cache']))
        if 'width' in saved:
            cfg['width'] = int(saved['width'])
        if 'depth' in saved:
            cfg['depth'] = int(saved['depth'])
        if 'modes' in saved:
            cfg['modes'] = int(saved['modes'])
        if 'deepnorm_hidden' in saved:
            cfg['deepnorm_hidden'] = int(saved['deepnorm_hidden'])

    return cfg


def load_model(checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = _infer_model_config(state, checkpoint_path)

    model = PlanningNeuralOperator(
        width=cfg['width'],
        modes1=cfg['modes'],
        modes2=cfg['modes'],
        depth=cfg['depth'],
        padding=cfg['padding'],
        beta=cfg['beta'],
        deepnorm_hidden=cfg['deepnorm_hidden'],
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded: {checkpoint_path} | {n_params:,} params | "
        f"modes={cfg['modes']} width={cfg['width']} depth={cfg['depth']}"
    )
    return model, cfg


def load_eval_tensors(cache_path: str, split: str, max_samples=None):
    z = np.load(cache_path)
    if 'value' not in z.files:
        raise ValueError(f"Cache at {cache_path} has no 'value' target to compare against.")

    raw_map = _to_nchw(z['raw_map'])
    sdf = _to_nchw(z['sdf'])
    value = _to_nchw(z['value'])
    goal = torch.from_numpy(z['goal'].astype(np.float32))

    if max_samples is not None:
        n = min(int(max_samples), raw_map.shape[0])
        raw_map = raw_map[:n]
        sdf = sdf[:n]
        goal = goal[:n]
        value = value[:n]

    idx = list(_split_indices(raw_map.shape[0], split))
    return raw_map[idx], sdf[idx], goal[idx], value[idx]


def evaluate_dataset(model, raw_map, sdf, goal, value, device, batch_size=32):
    ds = TensorDataset(raw_map, sdf, goal, value)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    abs_sum = sq_sum = y_sq_sum = 0.0
    abs_sum_free = sq_sum_free = y_sq_sum_free = 0.0
    count = count_free = 0

    samples = {'raw_map': [], 'pred': [], 'y': []}

    model.eval()
    with torch.no_grad():
        for raw_b, sdf_b, goal_b, y_b in loader:
            raw_b = raw_b.to(device)
            sdf_b = sdf_b.to(device)
            goal_b = goal_b.to(device)
            y_b = y_b.to(device)

            pred_b = model(raw_b, sdf_b, goal_b)
            err = pred_b - y_b

            abs_sum += err.abs().sum().item()
            sq_sum += err.pow(2).sum().item()
            y_sq_sum += y_b.pow(2).sum().item()
            count += y_b.numel()

            free_mask = (raw_b > 0.5).float()
            abs_sum_free += (err.abs() * free_mask).sum().item()
            sq_sum_free += (err.pow(2) * free_mask).sum().item()
            y_sq_sum_free += (y_b.pow(2) * free_mask).sum().item()
            count_free += int(free_mask.sum().item())

            if len(samples['raw_map']) < 32:
                samples['raw_map'].append(raw_b.cpu())
                samples['pred'].append(pred_b.cpu())
                samples['y'].append(y_b.cpu())

    out = {
        'mae': abs_sum / max(1, count),
        'rmse': (sq_sum / max(1, count)) ** 0.5,
        'rel_l2': (sq_sum ** 0.5) / (y_sq_sum ** 0.5 + 1e-12),
        'mae_free': abs_sum_free / max(1, count_free),
        'rmse_free': (sq_sum_free / max(1, count_free)) ** 0.5,
        'rel_l2_free': (sq_sum_free ** 0.5) / (y_sq_sum_free ** 0.5 + 1e-12),
        'n_samples': len(ds),
        'resolution': int(raw_map.shape[-1]),
    }
    if samples['raw_map']:
        out['samples'] = {
            'raw_map': torch.cat(samples['raw_map'], dim=0),
            'pred': torch.cat(samples['pred'], dim=0),
            'y': torch.cat(samples['y'], dim=0),
        }
    return out


def plot_samples(results, save_path, num_samples=4):
    if 'samples' not in results:
        return

    raw_map = results['samples']['raw_map']
    pred = results['samples']['pred']
    y = results['samples']['y']

    n = min(num_samples, len(pred))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4.0 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        m_np = raw_map[i, 0].numpy()
        p_np = pred[i, 0].numpy()
        y_np = y[i, 0].numpy()
        e_np = np.abs(p_np - y_np)

        plots = [
            (m_np, 'gray', 'Occupancy map'),
            (p_np, 'plasma', 'Pred value'),
            (y_np, 'plasma', 'GT value'),
            (e_np, 'magma', '|Error|'),
        ]
        for j, (img, cmap, title) in enumerate(plots):
            ax = axes[i, j]
            im = ax.imshow(img, origin='lower', cmap=cmap)
            plt.colorbar(im, ax=ax, shrink=0.75)
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.axis('off')

    fig.suptitle(
        f"PNO Eval | MAE={results['mae']:.6f} | RMSE={results['rmse']:.6f} | relL2={results['rel_l2']:.6f}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model, cfg = load_model(args.checkpoint, device)

    cache_path = args.cache or cfg.get('cache')
    if cache_path is None:
        raise ValueError('No cache path available. Provide --cache.')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_map, sdf, goal, value = load_eval_tensors(
        cache_path=cache_path,
        split=args.split,
        max_samples=args.max_samples,
    )
    print(f"Split: {args.split} ({raw_map.shape[0]} samples) from {cache_path}")

    results = evaluate_dataset(
        model=model,
        raw_map=raw_map,
        sdf=sdf,
        goal=goal,
        value=value,
        device=device,
        batch_size=args.batch_size,
    )

    print(f"\nResolution: {results['resolution']}x{results['resolution']}")
    print(f"Samples:    {results['n_samples']}")
    print(f"MAE:        {results['mae']:.8f}")
    print(f"RMSE:       {results['rmse']:.8f}")
    print(f"Rel L2:     {results['rel_l2']:.8f}")
    print(f"MAE free:   {results['mae_free']:.8f}")
    print(f"RMSE free:  {results['rmse_free']:.8f}")
    print(f"RelL2 free: {results['rel_l2_free']:.8f}")

    plot_samples(results, out_dir / 'eval_result_pno.png', num_samples=args.num_samples)

    stats_path = out_dir / 'eval_stats_pno.txt'
    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"Cache:      {cache_path}",
        f"Split:      {args.split}",
        f"Model:      modes={cfg['modes']}, width={cfg['width']}, depth={cfg['depth']}, beta={cfg['beta']}",
        f"Params:     {sum(p.numel() for p in model.parameters()):,}",
        "",
        f"MAE:        {results['mae']:.8f}",
        f"RMSE:       {results['rmse']:.8f}",
        f"RelativeL2: {results['rel_l2']:.8f}",
        "",
        f"MAE_free:   {results['mae_free']:.8f}",
        f"RMSE_free:  {results['rmse_free']:.8f}",
        f"RelL2_free: {results['rel_l2_free']:.8f}",
    ]
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved stats: {stats_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PNO against value-function targets')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pno/model_best.ckpt')
    parser.add_argument('--cache', type=str, default=None,
                        help='Path to PNO cache (.npz/.pt) with raw_map, sdf, goal, value')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save results (default: checkpoint directory)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test', 'all'])

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)

    main(args)
