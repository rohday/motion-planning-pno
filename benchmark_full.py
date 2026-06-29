#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent / 'generalizableMotionPlanning' / '2D_Neural_Heuristics'))
from astar.astar import AStar
from astar.environment_simple import Environment2D
from evaluate_pno import load_model, load_eval_tensors

def extract_path(env, start_coord):
    t0 = time.perf_counter()
    try:
        path_cost, path, action_idx, expands, sss = AStar.plan(start_coord, env, eps=1.0)
    except Exception as e:
        return float('inf'), [], 0, 0.0
    t1 = time.perf_counter()
    return path_cost, path, expands, (t1 - t0)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    cache_path = args.cache or cfg.get('cache')
    
    raw_map, sdf, goal, value = load_eval_tensors(
        cache_path=cache_path,
        split=args.split,
        max_samples=1000,
    )
    
    n_samples = raw_map.shape[0]
    num_to_eval = min(args.num_samples, n_samples)
    
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_samples, size=num_to_eval, replace=False)
    
    metrics = {
        'dij': {'expands': [], 'time': []},
        'euc': {'expands': [], 'time': []},
        'gt':  {'expands': [], 'time': []},
        'pno': {'expands': [], 'time': []}
    }
    
    for i, idx in enumerate(indices):
        raw_b = raw_map[idx:idx+1].to(device)
        sdf_b = sdf[idx:idx+1].to(device)
        goal_b = goal[idx:idx+1].to(device)
        gt_value = value[idx:idx+1]
        
        with torch.no_grad():
            pred_b = model(raw_b, sdf_b, goal_b)
            
        m_np = raw_b[0, 0].cpu().numpy()
        p_np = pred_b[0, 0].cpu().numpy()
        gt_np = gt_value[0, 0].cpu().numpy()
        
        gx, gy = goal_b[0].cpu().numpy()
        goal_coord = np.array([int(round(gy)), int(round(gx))])
        
        free_coords = np.argwhere(m_np > 0.5)
        valid_mask = ~np.all(free_coords == goal_coord, axis=1)
        free_coords = free_coords[valid_mask]
        
        if len(free_coords) == 0:
            continue
            
        start_coord = free_coords[rng.choice(len(free_coords))]
        cmap = (m_np <= 0.5).astype(int)
        
        h_dim, w_dim = m_np.shape
        y_idx, x_idx = np.indices((h_dim, w_dim))
        euc_np = np.sqrt((y_idx - goal_coord[0])**2 + (x_idx - goal_coord[1])**2)
        
        # 1. Dijkstra
        env_dijkstra = Environment2D(goal_coord, cmap, valuefunction=None)
        _, _, exp_dij, t_dij = extract_path(env_dijkstra, start_coord)
        metrics['dij']['expands'].append(exp_dij)
        metrics['dij']['time'].append(t_dij)
        
        # 2. Euclidean
        env_euc = Environment2D(goal_coord, cmap, valuefunction=euc_np)
        _, _, exp_euc, t_euc = extract_path(env_euc, start_coord)
        metrics['euc']['expands'].append(exp_euc)
        metrics['euc']['time'].append(t_euc)
        
        # 3. Ground Truth
        env_gt = Environment2D(goal_coord, cmap, valuefunction=gt_np)
        _, _, exp_gt, t_gt = extract_path(env_gt, start_coord)
        metrics['gt']['expands'].append(exp_gt)
        metrics['gt']['time'].append(t_gt)
        
        # 4. PNO
        env_pno = Environment2D(goal_coord, cmap, valuefunction=p_np)
        _, _, exp_pno, t_pno = extract_path(env_pno, start_coord)
        metrics['pno']['expands'].append(exp_pno)
        metrics['pno']['time'].append(t_pno)
        
        if (i+1) % 10 == 0:
            print(f"Evaluated {i+1}/{num_to_eval} samples...")

    print(f"\n--- Benchmark Results over {len(metrics['pno']['expands'])} Samples ---")
    
    labels = [
        ('Dijkstra (No Heuristic)', 'dij'),
        ('A* (Euclidean)', 'euc'),
        ('A* (Ground Truth)', 'gt'),
        ('A* (PNO Compressed)', 'pno')
    ]
    
    print(f"{'Method':<25} | {'Avg Expands':<12} | {'Avg Time (ms)':<15} | {'Speedup vs Euc':<15}")
    print("-" * 75)
    
    avg_euc_time = np.mean(metrics['euc']['time']) * 1000
    
    for name, key in labels:
        avg_exp = np.mean(metrics[key]['expands'])
        avg_time = np.mean(metrics[key]['time']) * 1000
        speedup = avg_euc_time / avg_time if avg_time > 0 else 0
        print(f"{name:<25} | {avg_exp:<12.1f} | {avg_time:<15.2f} | {speedup:<15.2f}x")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cache', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    main(args)
