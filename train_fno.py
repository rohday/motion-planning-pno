#!/usr/bin/env python3
"""
Train FNO2dMultiGoal on the official HuggingFace dataset.
python train_fno.py --data data/data_64x64 \\
                  --ntrain 500 --ntest 50 --batch_size 5 --epochs 401
"""

import argparse
import json
import os
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make sure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fno.fno2d import FNO2dMultiGoal
from data.loader import PNODataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class LpLoss:
    """
    Relative Lp loss, matching the official implementation.
    size_average=False → returns sum over the batch (caller divides by N).
    size_average=True  → returns mean over the batch.
    """

    def __init__(self, p=2, size_average=False):
        self.p = p
        self.size_average = size_average

    def rel(self, x, y):
        n = x.size(0)
        diff = torch.norm(x.reshape(n, -1) - y.reshape(n, -1), self.p, dim=1)
        base = torch.norm(y.reshape(n, -1), self.p, dim=1)
        loss = (diff / (base + 1e-12)).sum()
        if self.size_average:
            loss = loss / n
        return loss

    def __call__(self, x, y):
        return self.rel(x, y)


# ---------------------------------------------------------------------------
# LR schedule (official: step decay)
# ---------------------------------------------------------------------------

def lr_schedule(base_lr, step, scheduler_step, scheduler_gamma):
    return base_lr * (scheduler_gamma ** (step // scheduler_step))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f">> Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f">> Device: {device}")

    # ---- Data ---------------------------------------------------------------
    # 80/10/10 train/val/test split (ratio-based, works with any dataset size)
    print("Loading data...")
    t1 = default_timer()

    full_ds = PNODataset(
        args.data,
        smooth_coef=args.smooth_coef,
    )
    N = len(full_ds)
    n_train = int(N * 0.8)
    n_val   = int(N * 0.1)
    n_test  = N - n_train - n_val  # remaining ~10%

    train_ds = torch.utils.data.Subset(full_ds, range(0, n_train))
    val_ds   = torch.utils.data.Subset(full_ds, range(n_train, n_train + n_val))
    test_ds  = torch.utils.data.Subset(full_ds, range(n_train + n_val, N))

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
    H = full_ds.chi.shape[1]
    print(f">> Preprocessing done in {t2-t1:.2f}s  |  resolution: {H}×{H}")
    print(f">> Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}  (80/10/10)")

    # ---- Model --------------------------------------------------------------
    model = FNO2dMultiGoal(
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

    # ---- Training setup -----------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = LpLoss(size_average=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / 'model_best.ckpt'

    # ---- Training loop (matches official exactly) ---------------------------
    best_train_loss = float('inf')
    best_val_loss   = float('inf')
    best_epoch = 0
    early_stop = 0

    train_log, val_log = [], []

    print("-" * 80)
    for ep in range(args.epochs):
        t1 = default_timer()

        # Update LR (step decay, not cosine)
        current_lr = lr_schedule(args.learning_rate, ep, args.scheduler_step, args.scheduler_gamma)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        # --- train ---
        model.train()
        train_l2 = 0.0
        for chi, mask, y, goals in train_loader:
            chi, mask, y, goals = (chi.to(device), mask.to(device),
                                   y.to(device), goals.to(device))

            optimizer.zero_grad()
            out = model(chi, goals)

            # Mask: only compute loss on free-space cells
            out_m = out * mask
            y_m   = y   * mask

            loss = loss_fn(out_m, y_m)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        train_l2 /= n_train
        train_log.append([ep, train_l2])

        t2 = default_timer()

        # Evaluate on val set when train improved
        if train_l2 < best_train_loss:
            model.eval()
            val_l2 = 0.0
            with torch.no_grad():
                for chi, mask, y, goals in val_loader:
                    chi, mask, y, goals = (chi.to(device), mask.to(device),
                                           y.to(device), goals.to(device))
                    out = model(chi, goals)
                    out_m = out * mask
                    y_m   = y   * mask
                    val_l2 += loss_fn(out_m, y_m).item()

            val_l2 /= n_val
            val_log.append([ep, val_l2])

            if val_l2 < best_val_loss:
                early_stop = 0
                best_train_loss = train_l2
                best_val_loss   = val_l2
                best_epoch = ep
                torch.save(model.state_dict(), model_path)
                # Save model config so evaluate.py can auto-detect architecture
                config = {
                    'modes': args.modes, 'width': args.width,
                    'depth': args.depth, 'padding': args.padding,
                    'depthwise': args.depthwise,
                }
                with open(out_dir / 'model_config.json', 'w') as f:
                    json.dump(config, f, indent=2)
                print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}] "
                      f"runtime: {t2-t1:.2f}s  "
                      f"train: {train_l2:.5f}  val: {val_l2:.5f}  ← best")
            else:
                early_stop += 1
                print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}](best:{best_epoch+1}) "
                      f"runtime: {t2-t1:.2f}s  "
                      f"train: {train_l2:.5f}  "
                      f"(best train/val: {best_train_loss:.5f}/{best_val_loss:.5f})")
        else:
            early_stop += 1
            print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}](best:{best_epoch+1}) "
                  f"runtime: {t2-t1:.2f}s  "
                  f"train: {train_l2:.5f}  "
                  f"(best train/val: {best_train_loss:.5f}/{best_val_loss:.5f})")

        if args.early_stop > 0 and early_stop > args.early_stop:
            print(f"Early stopping at epoch {ep+1}")
            break

    # ---- Final test evaluation -----------------------------------------------
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for chi, mask, y, goals in test_loader:
            chi, mask, y, goals = (chi.to(device), mask.to(device),
                                   y.to(device), goals.to(device))
            out = model(chi, goals)
            out_m = out * mask
            y_m   = y   * mask
            test_l2 += loss_fn(out_m, y_m).item()
    test_l2 /= n_test

    # ---- Save logs ----------------------------------------------------------
    np.savetxt(str(out_dir / 'loss_train.txt'), train_log)
    np.savetxt(str(out_dir / 'loss_val.txt'),   val_log)

    print("-" * 80)
    print(f">> Best train loss: {best_train_loss:.5f}")
    print(f">> Best val loss:   {best_val_loss:.5f}")
    print(f">> Test loss:       {test_l2:.5f}  (held-out)")
    print(f">> Best epoch:      {best_epoch + 1}")
    print(f">> Model saved to:  {model_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FNO2dMultiGoal (official PNO replication)")

    # Data
    parser.add_argument('--data',        type=str, default='data/data_64x64',
                        help="Dir with mask.npy, dist_in.npy, output.npy, goal.npy")
    # Split is auto: 80/10/10 train/val/test from all samples in --data
    parser.add_argument('--smooth_coef', type=float, default=5.0,
                        help="Smooth chi coefficient (paper uses 5.0)")

    # Model
    parser.add_argument('--modes',       type=int, default=12)
    parser.add_argument('--width',       type=int, default=32)
    parser.add_argument('--depth',  type=int, default=4)
    parser.add_argument('--padding',     type=int, default=9)
    parser.add_argument('--depthwise',   action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use depthwise spectral conv (default: on, --no-depthwise to disable)")

    # Training
    parser.add_argument('--epochs',          type=int,   default=401)
    parser.add_argument('--batch_size',      type=int,   default=20)
    parser.add_argument('--learning_rate',   type=float, default=5e-3)
    parser.add_argument('--weight_decay',    type=float, default=3e-6)
    parser.add_argument('--scheduler_step',  type=int,   default=100)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--early_stop',      type=int,   default=400,
                        help="Stop if no improvement for N evals (0=disabled)")
    parser.add_argument('--num_workers',     type=int,   default=0)

    # Output
    parser.add_argument('--output_dir',  type=str, default='checkpoints/fno')

    args = parser.parse_args()
    train(args)

    # ---- Auto-evaluate on val split ------------------------------------------
    print("\n" + "=" * 80)
    print("Running evaluation on val split...")
    print("=" * 80 + "\n")
    import subprocess
    eval_cmd = [
        sys.executable, 'evaluate.py',
        '--checkpoint', str(Path(args.output_dir) / 'model_best.ckpt'),
        '--data_root', str(Path(args.data).parent),
        '--split', 'val',
        '--smooth_coef', str(args.smooth_coef),
    ]
    subprocess.run(eval_cmd)
