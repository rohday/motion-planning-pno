#!/usr/bin/env python3
"""
Train FNO2dMultiGoal on the official PNO synthetic dataset.

Replicates the training loop from:
  ExistentialRobotics/PNO - 2D_Neural_Heuristics/train/TrainPNO2D.py

Key details that match the official code:
  - LpLoss(size_average=False) — sum over batch, divide by N manually
  - Step-decay LR schedule: lr * gamma^(step // step_size)
  - Loss computed only on free-space cells: out * mask, y * mask
  - Best model saved by test loss (not train loss)
  - No data augmentation

Usage:
  # Download the dataset first (see data/download.py)
  python train.py --train_dir data/synthetic --test_dir data/synthetic \\
                  --ntrain 500 --ntest 50 --batch_size 5 --epochs 401
"""

import argparse
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

from models.fno2d import FNO2dMultiGoal
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
    # The official code uses the same file directory for train+test,
    # splits by index: first ntrain samples are train, last ntest are test.
    print("Loading data...")
    t1 = default_timer()

    full_ds = PNODataset(
        args.train_dir,
        smooth_coef=args.smooth_coef,
        subsample=args.subsample,
        max_samples=args.ntrain + args.ntest,
    )

    train_ds = torch.utils.data.Subset(full_ds, range(args.ntrain))
    test_ds  = torch.utils.data.Subset(full_ds, range(args.ntrain, args.ntrain + args.ntest))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())

    t2 = default_timer()
    H = full_ds.chi.shape[1]
    print(f">> Preprocessing done in {t2-t1:.2f}s  |  resolution: {H}×{H}")
    print(f">> Train: {len(train_ds)} samples  |  Test: {len(test_ds)} samples")

    # ---- Model --------------------------------------------------------------
    model = FNO2dMultiGoal(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        num_layers=args.num_layers,
        padding=args.padding,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">> Model parameters: {n_params:,}")

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
    best_train_loss = best_test_loss = 1e8
    best_epoch = 0
    early_stop = 0

    train_log, test_log = [], []

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

        train_l2 /= args.ntrain
        train_log.append([ep, train_l2])

        t2 = default_timer()

        # Official: only evaluate test when train improved
        if train_l2 < best_train_loss:
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

            test_l2 /= args.ntest
            test_log.append([ep, test_l2])

            if test_l2 < best_test_loss:
                early_stop = 0
                best_train_loss = train_l2
                best_test_loss  = test_l2
                best_epoch = ep
                torch.save(model.state_dict(), model_path)
                print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}] "
                      f"runtime: {t2-t1:.2f}s  "
                      f"train: {train_l2:.5f}  test: {test_l2:.5f}  ← best")
            else:
                early_stop += 1
                print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}](best:{best_epoch+1}) "
                      f"runtime: {t2-t1:.2f}s  "
                      f"train: {train_l2:.5f}  "
                      f"(best train/test: {best_train_loss:.5f}/{best_test_loss:.5f})")
        else:
            early_stop += 1
            print(f">> ep [{ep+1:>{len(str(args.epochs))}d}/{args.epochs}](best:{best_epoch+1}) "
                  f"runtime: {t2-t1:.2f}s  "
                  f"train: {train_l2:.5f}  "
                  f"(best train/test: {best_train_loss:.5f}/{best_test_loss:.5f})")

        if args.early_stop > 0 and early_stop > args.early_stop:
            print(f"Early stopping at epoch {ep+1}")
            break

    # ---- Save logs ----------------------------------------------------------
    np.savetxt(str(out_dir / 'loss_train.txt'), train_log)
    np.savetxt(str(out_dir / 'loss_test.txt'),  test_log)

    print("-" * 80)
    print(f">> Best train loss: {best_train_loss:.5f}")
    print(f">> Best test loss:  {best_test_loss:.5f}")
    print(f">> Best epoch:      {best_epoch + 1}")
    print(f">> Model saved to:  {model_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FNO2dMultiGoal (official PNO replication)")

    # Data
    parser.add_argument('--train_dir',   type=str, required=True,
                        help="Dir with mask.npy, dist_in.npy, output.npy, goals.npy")
    parser.add_argument('--test_dir',    type=str, default=None,
                        help="Test data dir (defaults to train_dir)")
    parser.add_argument('--ntrain',      type=int, default=500)
    parser.add_argument('--ntest',       type=int, default=50)
    parser.add_argument('--subsample',   type=int, default=1,
                        help="Spatial stride for downsampling (e.g. 4 for 256→64)")
    parser.add_argument('--smooth_coef', type=float, default=5.0,
                        help="Smooth chi coefficient (paper uses 5.0)")

    # Model
    parser.add_argument('--modes',       type=int, default=12)
    parser.add_argument('--width',       type=int, default=32)
    parser.add_argument('--num_layers',  type=int, default=4)
    parser.add_argument('--padding',     type=int, default=9)

    # Training
    parser.add_argument('--epochs',          type=int,   default=401)
    parser.add_argument('--batch_size',      type=int,   default=5)
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
    if args.test_dir is None:
        args.test_dir = args.train_dir

    train(args)
