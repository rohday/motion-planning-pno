import argparse
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt

def compute_sdf(mask):
    return distance_transform_edt(mask != 0) + distance_transform_edt(mask == 0)

def compute_value_function(mask, sdf, goal):
    size = mask.shape[0]
    speed = np.clip(sdf, 0, 1) * mask
    phi = np.ones((size, size), dtype=np.float64)
    phi[goal[0], goal[1]] = -1
    obstacle_mask = speed < 1e-10
    phi = np.ma.MaskedArray(phi, obstacle_mask)
    speed_safe = speed.copy()
    speed_safe[obstacle_mask] = 1.0
    try:
        tt = skfmm.travel_time(phi, speed_safe)
        result = np.array(tt.filled(0.0), dtype=np.float64)
    except Exception:
        return None
    result[result > 1e6] = 0.0
    return result

def pick_goals(mask, sdf, rng, n=10):
    free = np.argwhere(mask > 0.5)
    if len(free) < n:
        return None
    valid = free[sdf[free[:, 0], free[:, 1]] > 1.0]
    if len(valid) < n:
        valid = free
    if len(valid) < n:
        return None
    
    # Pick n goals without replacement
    idxs = rng.choice(len(valid), size=n, replace=False)
    return valid[idxs]

def process_one_map(args):
    idx, mask_in, seed = args
    rng = np.random.default_rng(seed)
    
    mask = mask_in.copy()
    sdf = compute_sdf(mask)
    goals = pick_goals(mask, sdf, rng, n=10)
    
    if goals is None:
        return None
        
    results = []
    for g in goals:
        value = compute_value_function(mask, sdf, g)
        if value is None or np.all(value == 0) or np.any(value > 1000):
            continue
            
        goal_xy = np.array([g[1], g[0]], dtype=np.int64)
        results.append((mask.copy(), sdf.copy(), goal_xy, value))
        
    if len(results) < 10:
        return None
        
    return results[:10]

def main():
    in_path = Path("data/data_64x64_orig/mask.npy")
    out_dir = Path("data/data_10k_from_orig")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading original maps from {in_path}...")
    orig_masks = np.load(in_path).astype(np.float64)
    n_maps = len(orig_masks)
    
    print(f"Generating 10 goals for each of the {n_maps} maps...")
    
    task_args = [(i, orig_masks[i], 42 + i) for i in range(n_maps)]
    
    all_masks = []
    all_sdfs = []
    all_goals = []
    all_values = []
    
    t0 = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, res in enumerate(pool.imap(process_one_map, task_args)):
            if res is not None:
                for r in res:
                    m, s, g, v = r
                    all_masks.append(m)
                    all_sdfs.append(s)
                    all_goals.append(g)
                    all_values.append(v)
            if (i+1) % 100 == 0:
                print(f"  Processed {i+1}/{n_maps} maps...")
                
    elapsed = time.time() - t0
    
    masks_np = np.stack(all_masks)
    sdfs_np = np.stack(all_sdfs)
    goals_np = np.stack(all_goals)
    values_np = np.stack(all_values)
    
    print(f"Saving {len(masks_np)} samples...")
    np.save(out_dir / "mask.npy", masks_np)
    np.save(out_dir / "dist_in.npy", sdfs_np)
    np.save(out_dir / "goal.npy", goals_np)
    np.save(out_dir / "output.npy", values_np)
    
    print(f"Done in {elapsed:.1f}s")
    
if __name__ == "__main__":
    main()
