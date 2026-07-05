#!/usr/bin/env python3
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent / 'generalizableMotionPlanning' / '2D_Neural_Heuristics'))
from astar.astar import AStar
from astar.environment_simple import Environment2D

from src.fno.fno2d import FNO2dSDF
from src.pno import PlanningNeuralOperator, HierarchicalSuperResolutionPNO
import src.pno.layers as layers

# --- Legacy SpectralConv2d for Old PNO (2.67M params) ---
class LegacySpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _cmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self._cmul(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._cmul(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W))

def load_fno(checkpoint_dir, device):
    with open(Path(checkpoint_dir) / 'model_config.json') as f:
        cfg = json.load(f)
    fno = FNO2dSDF(
        modes1=cfg['modes'], modes2=cfg['modes'],
        width=cfg['width'], depth=cfg['depth'], padding=cfg['padding'],
        depthwise=cfg.get('depthwise', False)
    )
    ckpt = torch.load(Path(checkpoint_dir) / 'model_best.ckpt', map_location='cpu', weights_only=True)
    if 'model_state_dict' in ckpt:
        fno.load_state_dict(ckpt['model_state_dict'])
    else:
        fno.load_state_dict(ckpt)
    fno.to(device).eval()
    return fno, cfg

def load_pno(checkpoint_dir, device, is_legacy=False):
    with open(Path(checkpoint_dir) / 'model_config.json') as f:
        cfg = json.load(f)
        
    # Patch layers.SpectralConv2d if legacy
    orig_spectral = layers.SpectralConv2d
    if is_legacy:
        layers.SpectralConv2d = LegacySpectralConv2d

    pno = PlanningNeuralOperator(
        width=cfg.get('width', 48),
        modes1=cfg.get('modes1', 12),
        modes2=cfg.get('modes2', 12),
        depth=cfg.get('depth', 4),
        padding=cfg.get('padding', 9),
        beta=cfg.get('beta', 5.0),
        deepnorm_hidden=cfg.get('deepnorm_hidden', 128),
        concave_activation_size=cfg.get('concave_activation_size', 20),
    )
    
    ckpt = torch.load(Path(checkpoint_dir) / 'model_best.ckpt', map_location='cpu', weights_only=True)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Handle old shape mismatch for deepnorm concave activation if any
    if 'deep_norm.output_activation.ms' in state:
        ms_shape = state['deep_norm.output_activation.ms'].shape
        if ms_shape[-1] != pno.deep_norm.output_activation.ms.shape[-1]:
            # Re-init deepnorm with correct size
            pno = PlanningNeuralOperator(
                width=cfg.get('width', 48),
                modes1=cfg.get('modes1', 12),
                modes2=cfg.get('modes2', 12),
                depth=cfg.get('depth', 4),
                padding=cfg.get('padding', 9),
                beta=cfg.get('beta', 5.0),
                deepnorm_hidden=cfg.get('deepnorm_hidden', 128),
                concave_activation_size=ms_shape[-1],
            )
            
    pno.load_state_dict(state)
    pno.to(device).eval()
    
    # Restore original class
    if is_legacy:
        layers.SpectralConv2d = orig_spectral
        
    return pno

def extract_path(env, start_coord):
    t0 = time.perf_counter()
    try:
        path_cost, path, action_idx, expands, sss = AStar.plan(start_coord, env, eps=1.0)
    except Exception as e:
        return float('inf'), [], 0, 0.0
    t1 = time.perf_counter()
    return path_cost, path, expands, (t1 - t0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading models...")
    # Load FNO
    fno, fno_cfg = load_fno('checkpoints/fno_sdf_10k', device)
    fno_norm = fno_cfg.get('normalization', {})
    
    # Load New PNO
    new_pno = load_pno('checkpoints/pno_10k_compressed', device, is_legacy=False)
    
    # Load Old PNO
    try:
        old_pno = load_pno('checkpoints/pno_10k', device, is_legacy=True)
    except Exception as e:
        print(f"Could not load old PNO: {e}")
        old_pno = None

    # Wrappers
    sr_new = HierarchicalSuperResolutionPNO(
        fno, new_pno, train_res=64,
        fno_normalize_input=bool(fno_norm.get('normalize_input', False)),
        fno_normalize_target=bool(fno_norm.get('normalize_target', False)),
        fno_x_mean=float(fno_norm.get('x_mean', 0.0)),
        fno_x_std=float(fno_norm.get('x_std', 1.0)),
        fno_y_mean=float(fno_norm.get('y_mean', 0.0)),
        fno_y_std=float(fno_norm.get('y_std', 1.0))
    ).to(device).eval()
    
    if old_pno:
        sr_old = HierarchicalSuperResolutionPNO(
            fno, old_pno, train_res=64,
            fno_normalize_input=bool(fno_norm.get('normalize_input', False)),
            fno_normalize_target=bool(fno_norm.get('normalize_target', False)),
            fno_x_mean=float(fno_norm.get('x_mean', 0.0)),
            fno_x_std=float(fno_norm.get('x_std', 1.0)),
            fno_y_mean=float(fno_norm.get('y_mean', 0.0)),
            fno_y_std=float(fno_norm.get('y_std', 1.0))
        ).to(device).eval()

    print("Loading IRL data...")
    mask_arr = np.load('data/irl_data/mask.npy')
    goal_arr = np.load('data/irl_data/goal.npy')
    output_arr = np.load('data/irl_data/output.npy')
    
    n_samples = mask_arr.shape[0]
    
    metrics = {
        'dij': {'expands': [], 'time': []},
        'euc': {'expands': [], 'time': []},
        'gt':  {'expands': [], 'time': []},
        'pno_new': {'expands': [], 'time': []},
    }
    if old_pno:
        metrics['pno_old'] = {'expands': [], 'time': []}
        
    rng = np.random.default_rng(42)
    
    print(f"Evaluating {n_samples} samples at 256x256 resolution...")
    
    for i in range(n_samples):
        m_np = mask_arr[i]
        g_np = goal_arr[i]
        gt_np = output_arr[i]
        
        gx, gy = int(g_np[0]), int(g_np[1])
        goal_coord = np.array([gy, gx])
        
        # Free coords to sample start from
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
        
        # Tensors for model
        raw_b = torch.tensor(m_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        goal_b = torch.tensor([[gx, gy]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pno_new_pred = sr_new(raw_b, goal_b)[0, 0].cpu().numpy()
            if old_pno:
                pno_old_pred = sr_old(raw_b, goal_b)[0, 0].cpu().numpy()
                
        # Run A*
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
        
        # 4. New PNO
        env_pno_new = Environment2D(goal_coord, cmap, valuefunction=pno_new_pred)
        _, _, exp_pno_new, t_pno_new = extract_path(env_pno_new, start_coord)
        metrics['pno_new']['expands'].append(exp_pno_new)
        metrics['pno_new']['time'].append(t_pno_new)
        
        # 5. Old PNO
        if old_pno:
            env_pno_old = Environment2D(goal_coord, cmap, valuefunction=pno_old_pred)
            _, _, exp_pno_old, t_pno_old = extract_path(env_pno_old, start_coord)
            metrics['pno_old']['expands'].append(exp_pno_old)
            metrics['pno_old']['time'].append(t_pno_old)
            
        if (i+1) % 10 == 0:
            print(f"Evaluated {i+1}/{n_samples} samples...")

    print(f"\n--- Benchmark Results over {len(metrics['pno_new']['expands'])} IRL Samples (256x256) ---")
    
    labels = [
        ('Dijkstra (No Heuristic)', 'dij'),
        ('A* (Euclidean)', 'euc'),
        ('A* (Ground Truth)', 'gt'),
    ]
    if old_pno:
        labels.append(('A* (PNO Old - 2.67M)', 'pno_old'))
    labels.append(('A* (PNO Compressed - 76k)', 'pno_new'))
    
    print(f"{'Method':<30} | {'Avg Expands':<12} | {'Avg Time (ms)':<15} | {'Speedup vs Euc':<15}")
    print("-" * 80)
    
    avg_euc_time = np.mean(metrics['euc']['time']) * 1000
    
    for name, key in labels:
        avg_exp = np.mean(metrics[key]['expands'])
        avg_time = np.mean(metrics[key]['time']) * 1000
        speedup = avg_euc_time / avg_time if avg_time > 0 else 0
        print(f"{name:<30} | {avg_exp:<12.1f} | {avg_time:<15.2f} | {speedup:<15.2f}x")

if __name__ == '__main__':
    main()
