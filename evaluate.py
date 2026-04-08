#!/usr/bin/env python3
# cli: python evaluate.py [--checkpoint] [--data_dir] [--output_dir] [--batch_size] [--max_samples] [--num_samples] [--split]

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fno.fno2d import FNO2dMultiGoal, FNO2dSDF


def _find_file(data_dir: str, base_name: str) -> str:
    exact = Path(data_dir) / f"{base_name}.npy"
    if exact.exists():
        return str(exact)
    matches = glob.glob(str(Path(data_dir) / f"{base_name}*.npy"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find {base_name}*.npy in {data_dir}")


def _infer_model_config(state: dict, checkpoint_path: str):
    width = state['fc0.weight'].shape[0]
    depth = sum(1 for k in state if k.startswith('conv') and k.endswith('.weights1'))

    w1_shape = state['conv0.weights1'].shape
    if len(w1_shape) == 3:
        depthwise = True
        modes = w1_shape[1]
    else:
        depthwise = False
        modes = w1_shape[2]

    cfg = {
        'task': 'legacy_multigoal',
        'modes': int(modes),
        'width': int(width),
        'depth': int(depth),
        'padding': 9,
        'depthwise': bool(depthwise),
        'normalization': {
            'x_mean': 0.0,
            'x_std': 1.0,
            'y_mean': 0.0,
            'y_std': 1.0,
            'normalize_input': False,
            'normalize_target': False,
        },
    }

    config_path = Path(checkpoint_path).parent / 'model_config.json'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        cfg['padding'] = int(saved.get('padding', cfg['padding']))
        cfg['depthwise'] = bool(saved.get('depthwise', cfg['depthwise']))
        cfg['task'] = str(saved.get('task', cfg['task']))
        if 'normalization' in saved:
            cfg['normalization'].update(saved['normalization'])

    return cfg


def load_model(checkpoint_path, device, cli_args=None):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = _infer_model_config(state, checkpoint_path)

    dw_str = ' (depthwise)' if cfg['depthwise'] else ''
    print(
        f"Inferred config: modes={cfg['modes']}, width={cfg['width']}, "
        f"layers={cfg['depth']}{dw_str}"
    )

    if cfg['task'] == 'geometry_to_sdf':
        model = FNO2dSDF(
            depth=cfg['depth'],
            padding=cfg['padding'],
            modes1=cfg['modes'],
            modes2=cfg['modes'],
            width=cfg['width'],
            depthwise=cfg['depthwise'],
        ).to(device)
    else:
        model = FNO2dMultiGoal(
            depth=cfg['depth'],
            padding=cfg['padding'],
            modes1=cfg['modes'],
            modes2=cfg['modes'],
            width=cfg['width'],
            depthwise=cfg['depthwise'],
        ).to(device)

    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {checkpoint_path}  |  {n_params:,} params")

    return model, cfg


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


def _normalize_input(x, norm_cfg):
    x_mean = float(norm_cfg.get('x_mean', 0.0))
    x_std = float(norm_cfg.get('x_std', 1.0))

    do_x = bool(norm_cfg.get('normalize_input', False))

    if do_x:
        x = (x - x_mean) / max(x_std, 1e-6)
    return x


def _denormalize_pred(pred_norm, norm_cfg):
    do_y = bool(norm_cfg.get('normalize_target', False))
    if not do_y:
        return pred_norm
    y_mean = float(norm_cfg.get('y_mean', 0.0))
    y_std = float(norm_cfg.get('y_std', 1.0))
    return pred_norm * max(y_std, 1e-6) + y_mean


def evaluate_dataset(model, data_dir, device, norm_cfg,
                     max_samples=None, batch_size=32, split='val', train_resolution=64):
    x_np = np.load(_find_file(data_dir, 'mask')).astype(np.float32)
    y_np = np.load(_find_file(data_dir, 'dist_in')).astype(np.float32)

    if max_samples is not None:
        n = min(max_samples, len(x_np))
        x_np = x_np[:n]
        y_np = y_np[:n]

    x = torch.from_numpy(x_np).unsqueeze(1)
    y = torch.from_numpy(y_np).unsqueeze(1)

    target_resolution = int(y.shape[-1])
    scale_factor = float(target_resolution) / float(train_resolution)

    idx = list(_split_indices(len(x), split))
    x = x[idx]
    y = y[idx]

    x_norm = _normalize_input(x.clone(), norm_cfg)

    ds = TensorDataset(x_norm, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    print(f"  Split: {split}  ({len(ds)} samples)")

    abs_sum = 0.0
    sq_sum = 0.0
    diff_sq_sum = 0.0
    y_sq_sum = 0.0
    count = 0

    samples = {'x': [], 'pred': [], 'y': []}

    model.eval()
    with torch.no_grad():
        for xb_norm, yb_raw in loader:
            xb_norm = xb_norm.to(device)
            yb_raw = yb_raw.to(device)

            pred_norm = model(xb_norm)
            pred_raw = _denormalize_pred(pred_norm, norm_cfg)
            pred_raw = pred_raw * scale_factor

            err = pred_raw - yb_raw
            abs_sum += err.abs().sum().item()
            sq_sum += err.pow(2).sum().item()
            diff_sq_sum += err.pow(2).sum().item()
            y_sq_sum += yb_raw.pow(2).sum().item()
            count += yb_raw.numel()

            if len(samples['x']) < 32:
                samples['x'].append(xb_norm.cpu())
                samples['pred'].append(pred_raw.cpu())
                samples['y'].append(yb_raw.cpu())

    mae = abs_sum / max(1, count)
    rmse = (sq_sum / max(1, count)) ** 0.5
    rel_l2 = (diff_sq_sum ** 0.5) / (y_sq_sum ** 0.5 + 1e-12)

    out = {
        'mae': mae,
        'rmse': rmse,
        'rel_l2': rel_l2,
        'scale_factor': scale_factor,
        'train_resolution': int(train_resolution),
        'target_resolution': target_resolution,
        'n_samples': len(ds),
        'resolution': int(x.shape[-1]),
    }

    if samples['x']:
        out['samples'] = {
            'x': torch.cat(samples['x'], dim=0),
            'pred': torch.cat(samples['pred'], dim=0),
            'y': torch.cat(samples['y'], dim=0),
        }
    return out


def plot_samples(results, save_path, num_samples=4):
    if 'samples' not in results:
        return

    x = results['samples']['x']
    pred = results['samples']['pred']
    y = results['samples']['y']

    n = min(num_samples, len(pred))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4.0 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        x_np = x[i, 0].numpy()
        p_np = pred[i, 0].numpy()
        y_np = y[i, 0].numpy()
        e_np = np.abs(p_np - y_np)

        plots = [
            (x_np, 'gray', 'Input mask (normalized)'),
            (p_np, 'plasma', 'Pred SDF'),
            (y_np, 'plasma', 'GT SDF'),
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
        f"SDF Eval | scale={results['scale_factor']:.3f} | MAE={results['mae']:.6f} | RMSE={results['rmse']:.6f} | relL2={results['rel_l2']:.6f}",
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

    if cfg.get('task') != 'geometry_to_sdf':
        raise ValueError(
            f"Checkpoint task='{cfg.get('task')}' is not geometry_to_sdf. "
            "Use an SDF-trained checkpoint from train_fno.py."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_dataset(
        model=model,
        data_dir=args.data_dir,
        device=device,
        norm_cfg=cfg.get('normalization', {}),
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        split=args.split,
        train_resolution=args.train_resolution,
    )

    print(f"\nResolution: {results['resolution']}x{results['resolution']}")
    print(f"Samples:    {results['n_samples']}")
    print(f"Scale:      {results['target_resolution']}/{results['train_resolution']} = {results['scale_factor']:.6f}")
    print(f"MAE:        {results['mae']:.8f}")
    print(f"RMSE:       {results['rmse']:.8f}")
    print(f"Rel L2:     {results['rel_l2']:.8f}")

    plot_samples(results, out_dir / 'eval_result_sdf.png', num_samples=args.num_samples)

    stats_path = out_dir / 'eval_stats.txt'
    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"Data:       {args.data_dir}",
        f"Split:      {args.split}",
        f"Model:      modes={cfg['modes']}, width={cfg['width']}, layers={cfg['depth']}, depthwise={cfg['depthwise']}",
        f"Params:     {sum(p.numel() for p in model.parameters()):,}",
        f"Scale:      {results['target_resolution']}/{results['train_resolution']} = {results['scale_factor']:.8f}",
        "",
        f"MAE:        {results['mae']:.8f}",
        f"RMSE:       {results['rmse']:.8f}",
        f"RelativeL2: {results['rel_l2']:.8f}",
    ]
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved stats: {stats_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FNO geometry->SDF model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fno_sdf/model_best.ckpt')
    parser.add_argument('--data_dir', type=str, default='data/data_64x64')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save results (default: checkpoint directory)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_resolution', type=int, default=64,
                        help='Resolution used during model training (for SDF scale correction).')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test', 'all'])

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)

    main(args)
