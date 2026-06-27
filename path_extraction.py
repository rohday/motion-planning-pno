#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the astar directory to path so we can import it
sys.path.append(str(Path(__file__).parent / 'generalizableMotionPlanning' / '2D_Neural_Heuristics'))

from astar.astar import AStar
from astar.environment_simple import Environment2D
from evaluate_pno import load_model, load_eval_tensors


import time

def plot_path(ax, path_coords, color='r', label='Path'):
    if path_coords:
        xs = [c[1] for c in path_coords] # col is x
        ys = [c[0] for c in path_coords] # row is y
        ax.plot(xs, ys, color=color, linewidth=2, label=label)

def extract_path(env, start_coord):
    t0 = time.perf_counter()
    try:
        path_cost, path, action_idx, expands, sss = AStar.plan(start_coord, env, eps=1.0)
    except Exception as e:
        print(f"A* failed: {e}")
        return math.inf, [], 0, 0.0
    t1 = time.perf_counter()
    return path_cost, path, expands, (t1 - t0)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)

    cache_path = args.cache or cfg.get('cache')
    if cache_path is None:
        raise ValueError('No cache path available. Provide --cache.')

    # Load data
    raw_map, sdf, goal, value = load_eval_tensors(
        cache_path=cache_path,
        split=args.split,
        max_samples=args.max_samples,
    )
    
    n_samples = raw_map.shape[0]
    num_to_plot = min(args.num_samples, n_samples)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_samples, size=num_to_plot, replace=False)
    
    fig, axes = plt.subplots(num_to_plot, 4, figsize=(20, 5 * num_to_plot))
    if num_to_plot == 1:
        axes = axes[np.newaxis, :]
        
    for row, idx in enumerate(indices):
        raw_b = raw_map[idx:idx+1].to(device)
        sdf_b = sdf[idx:idx+1].to(device)
        goal_b = goal[idx:idx+1].to(device)
        gt_value = value[idx:idx+1]
        
        with torch.no_grad():
            pred_b = model(raw_b, sdf_b, goal_b)
            
        m_np = raw_b[0, 0].cpu().numpy()
        p_np = pred_b[0, 0].cpu().numpy()
        gt_np = gt_value[0, 0].cpu().numpy()
        
        # goal_b has shape (1, 2)
        gx, gy = goal_b[0].cpu().numpy()
        goal_coord = np.array([int(round(gy)), int(round(gx))]) # (row, col)
        
        # Pick a random start coordinate in free space
        free_coords = np.argwhere(m_np > 0.5)
        # remove the goal coord from start coords
        valid_mask = ~np.all(free_coords == goal_coord, axis=1)
        free_coords = free_coords[valid_mask]
        
        if len(free_coords) == 0:
            print(f"Sample {idx}: No valid start coordinates found.")
            continue
            
        start_coord = free_coords[rng.choice(len(free_coords))]
        
        # The Environment2D expects obstacle map where 0 is FREE and 1 is OBSTACLE
        cmap = (m_np <= 0.5).astype(int)
        
        # Compute Euclidean heuristic array
        h_dim, w_dim = m_np.shape
        y_idx, x_idx = np.indices((h_dim, w_dim))
        euc_np = np.sqrt((y_idx - goal_coord[0])**2 + (x_idx - goal_coord[1])**2)
        
        # 1. Dijkstra (No heuristic)
        env_dijkstra = Environment2D(goal_coord, cmap, valuefunction=None)
        cost_dij, path_dij, exp_dij, t_dij = extract_path(env_dijkstra, start_coord)
        
        # 2. A* with Euclidean Heuristic
        env_euc = Environment2D(goal_coord, cmap, valuefunction=euc_np)
        cost_euc, path_euc, exp_euc, t_euc = extract_path(env_euc, start_coord)
        
        # 3. A* with Ground Truth Heuristic
        env_gt = Environment2D(goal_coord, cmap, valuefunction=gt_np)
        cost_gt, path_gt, exp_gt, t_gt = extract_path(env_gt, start_coord)
        
        # 4. A* with PNO Predicted Heuristic
        env_pno = Environment2D(goal_coord, cmap, valuefunction=p_np)
        cost_pno, path_pno, exp_pno, t_pno = extract_path(env_pno, start_coord)
        
        print(f"Sample {idx}: Start={start_coord}, Goal={goal_coord}")
        print(f"  Dijkstra: Expands={exp_dij:4d}, Cost={cost_dij:5.2f}, Time={t_dij*1000:6.2f} ms")
        print(f"  A* (Euc): Expands={exp_euc:4d}, Cost={cost_euc:5.2f}, Time={t_euc*1000:6.2f} ms")
        print(f"  A* (GT):  Expands={exp_gt:4d}, Cost={cost_gt:5.2f}, Time={t_gt*1000:6.2f} ms")
        print(f"  A* (PNO): Expands={exp_pno:4d}, Cost={cost_pno:5.2f}, Time={t_pno*1000:6.2f} ms")
        print("-" * 50)
        
        # Plotting
        ax1 = axes[row, 0]
        ax1.imshow(m_np, origin='lower', cmap='gray_r')
        ax1.plot(gx, gy, 'g*', markersize=12, label='Goal')
        ax1.plot(start_coord[1], start_coord[0], 'bo', markersize=8, label='Start')
        plot_path(ax1, path_pno, color='r', label='PNO Path')
        if row == 0:
            ax1.set_title("Occupancy Map & PNO Path")
        ax1.legend()
        ax1.axis('off')
        
        ax2 = axes[row, 1]
        im2 = ax2.imshow(euc_np, origin='lower', cmap='plasma')
        ax2.plot(gx, gy, 'g*', markersize=12)
        ax2.plot(start_coord[1], start_coord[0], 'bo', markersize=8)
        plot_path(ax2, path_euc, color='cyan', label='Euc Path')
        if row == 0:
            ax2.set_title(f"Euclidean Heuristic\nExpands: {exp_euc}")
        else:
            ax2.set_title(f"Expands: {exp_euc}")
        plt.colorbar(im2, ax=ax2, shrink=0.7)
        ax2.axis('off')

        ax3 = axes[row, 2]
        im3 = ax3.imshow(gt_np, origin='lower', cmap='plasma')
        ax3.plot(gx, gy, 'g*', markersize=12)
        ax3.plot(start_coord[1], start_coord[0], 'bo', markersize=8)
        plot_path(ax3, path_gt, color='cyan', label='GT Path')
        if row == 0:
            ax3.set_title(f"GT Value Function\nExpands: {exp_gt}")
        else:
            ax3.set_title(f"Expands: {exp_gt}")
        plt.colorbar(im3, ax=ax3, shrink=0.7)
        ax3.axis('off')
        
        ax4 = axes[row, 3]
        im4 = ax4.imshow(p_np, origin='lower', cmap='plasma')
        ax4.plot(gx, gy, 'g*', markersize=12)
        ax4.plot(start_coord[1], start_coord[0], 'bo', markersize=8)
        plot_path(ax4, path_pno, color='cyan', label='PNO Path')
        if row == 0:
            ax4.set_title(f"PNO Value Function\nExpands: {exp_pno} (Dij: {exp_dij})")
        else:
            ax4.set_title(f"Expands: {exp_pno} (Dij: {exp_dij})")
        plt.colorbar(im4, ax=ax4, shrink=0.7)
        ax4.axis('off')

    plt.tight_layout()
    save_path = out_dir / 'path_extraction.png'
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A* Path Extraction using PNO Heuristic')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pno/model_best.ckpt')
    parser.add_argument('--cache', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='data/visualizations')
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)

    main(args)
