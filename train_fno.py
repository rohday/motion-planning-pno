#!/usr/bin/env python3
# cli: python train_fno.py [--data] [--modes] [--width] [--depth] [--padding] [--depthwise/--no-depthwise] [--epochs] [--batch_size] [--learning_rate] [--weight_decay] [--scheduler_step] [--scheduler_gamma] [--early_stop] [--checkevery] [--num_workers] [--output_dir]

import argparse
import json
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fno.fno2d import FNO2dSDF
from data.loader import SDFDataset

def evaluate_rel_l2(model, loader, device):
    model.eval()
    diff_sq_sum = 0.0
    y_sq_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            diff_sq_sum += (pred - y).pow(2).sum().item()
            y_sq_sum += y.pow(2).sum().item()

    return (diff_sq_sum ** 0.5) / (y_sq_sum ** 0.5 + 1e-12)

def train(args):
    if args.checkevery < 1:
        raise ValueError("--checkevery must be >= 1")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f">> Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f">> Device: {device}")

    print("Loading SDF dataset...")
    t1 = default_timer()

    full_ds = SDFDataset(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        normalize_input=not args.no_normalize_input,
        normalize_target=not args.no_normalize_target,
    )
    n_total = len(full_ds)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())

    t2 = default_timer()
    h, w = full_ds.x.shape[-2], full_ds.x.shape[-1]
    print(f">> Data ready in {t2-t1:.2f}s  |  resolution: {h}x{w}")
    print(f">> Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")

    model = FNO2dSDF(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        depth=args.depth,
        padding=args.padding,
        depthwise=args.depthwise,
    ).to(device)

    dw_str = ' (depthwise)' if args.depthwise else ''

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">> Model parameters: {n_params:,}{dw_str}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / 'model_best.ckpt'

    best_val_l2 = float('inf')
    best_epoch = 0
    early_stop = 0

    train_log, val_log, test_log = [], [], []

    print("-" * 80)
    for ep in range(args.epochs):
        t1 = default_timer()

        model.train()
        total_mse = 0.0
        diff_sq_sum = 0.0
        y_sq_sum = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out, y, reduction='mean')
            loss.backward()
            optimizer.step()
            bs = x.shape[0]
            total_mse += loss.item() * bs
            diff_sq_sum += (out - y).pow(2).sum().item()
            y_sq_sum += y.pow(2).sum().item()
            n_seen += bs

        scheduler.step()

        train_mse = total_mse / max(1, n_seen)
        train_l2 = (diff_sq_sum ** 0.5) / (y_sq_sum ** 0.5 + 1e-12)
        val_l2 = evaluate_rel_l2(model, val_loader, device)
        train_log.append([ep, train_l2])
        val_log.append([ep, val_l2])

        t2 = default_timer()

        if val_l2 < best_val_l2:
            early_stop = 0
            best_val_l2 = val_l2
            best_epoch = ep
            torch.save(model.state_dict(), model_path)
            config = {
                'task': 'geometry_to_sdf',
                'input': 'mask.npy',
                'target': 'dist_in.npy',
                'modes': args.modes,
                'width': args.width,
                'depth': args.depth,
                'padding': args.padding,
                'depthwise': args.depthwise,
                'normalization': full_ds.norm,
            }
            with open(out_dir / 'model_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}] "
                  f"runtime: {t2-t1:.2f}s  "
                  f"train_l2: {train_l2:.6f}  val_l2: {val_l2:.6f}  train_mse: {train_mse:.6f}  <- best")
        else:
            early_stop += 1
            print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}](best:{best_epoch+1}) "
                  f"runtime: {t2-t1:.2f}s  "
                  f"train_l2: {train_l2:.6f}  val_l2: {val_l2:.6f}  train_mse: {train_mse:.6f}")

        if (ep + 1) % args.checkevery == 0:
            test_l2_ep = evaluate_rel_l2(model, test_loader, device)
            test_log.append([ep, test_l2_ep])
            print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}] "
                f"periodic test_l2: {test_l2_ep:.6f}")

        if args.early_stop > 0 and early_stop > args.early_stop:
            print(f"Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_l2 = evaluate_rel_l2(model, test_loader, device)

    np.savetxt(str(out_dir / 'loss_train.txt'), train_log)
    np.savetxt(str(out_dir / 'loss_val.txt'),   val_log)
    np.savetxt(str(out_dir / 'loss_test.txt'),  test_log)

    print("-" * 80)
    print(f">> Best val L2:     {best_val_l2:.6f}")
    print(f">> Test L2:         {test_l2:.6f}  (held-out)")
    print(f">> Best epoch:      {best_epoch + 1}")
    print(f">> Model saved to:  {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FNO as geometry->SDF mapper")

    parser.add_argument('--data_dir',        type=str, default='data/data_64x64',
                        help="Dir with mask.npy and dist_in.npy")
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--no-normalize-input', action='store_true')
    parser.add_argument('--no-normalize-target', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--modes',       type=int, default=12)
    parser.add_argument('--width',       type=int, default=32)
    parser.add_argument('--depth',  type=int, default=4)
    parser.add_argument('--padding',     type=int, default=9)
    parser.add_argument('--depthwise',   action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use depthwise spectral conv (default: on, --no-depthwise to disable)")

    parser.add_argument('--epochs',          type=int,   default=401)
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--learning_rate',   type=float, default=5e-3)
    parser.add_argument('--weight_decay',    type=float, default=3e-6)
    parser.add_argument('--scheduler_step',  type=int,   default=20)
    parser.add_argument('--scheduler_gamma', type=float, default=0.7)
    parser.add_argument('--early_stop',      type=int,   default=100,
                        help="Stop if no improvement for N evals (0=disabled)")
    parser.add_argument('--checkevery',      type=int,   default=10,
                        help="Check test loss every N epochs (default: 10)")
    parser.add_argument('--num_workers',     type=int,   default=0)

    parser.add_argument('--output_dir',  type=str, default='checkpoints/fno_sdf')

    args = parser.parse_args()
    train(args)
