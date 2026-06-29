# Planning Neural Operator (PNO)

A continuous, generalizable neural operator framework for robotic motion planning, providing an optimal value function (cost-to-go) heuristic for graph search algorithms like A*.

## Overview

The Planning Neural Operator (PNO) predicts the shortest path distance from every free-space coordinate to a specified goal coordinate in a 2D environment. Unlike standard Convolutional Neural Networks, PNO uses Fourier Neural Operators (FNO) to solve the Eikonal Partial Differential Equation (PDE) in a resolution-invariant, continuous domain. This value function is strictly admissible and acts as a highly efficient heuristic for A*, significantly reducing node expansions compared to standard Euclidean heuristics.

## System Architecture

The pipeline consists of two models trained sequentially: the SDF-FNO and the PNO.

### 1. SDF-FNO (Geometry to SDF)
Occupancy maps are binary and non-differentiable at boundaries. To enable smooth gradient-based operations for the main operator, the binary map is converted into a Continuous Unsigned Distance Field (SDF).
- **Input:** Binary occupancy map $m(x)$ (1 for free space, 0 for obstacles).
- **Output:** True unsigned distance field $SDF(x)$ representing the absolute distance to the nearest boundary from both inside and outside the obstacles.
- **Model:** Standard Fourier Neural Operator (FNO2d).

### 2. Planning Neural Operator (PNO)
The main operator takes the geometry and goal, and outputs the optimal value function.
- **Inputs:** A 3-channel tensor containing the raw occupancy map, the predicted continuous SDF, and a one-hot goal channel.
- **Masking Mechanism (SIFN):** A Smoothed Indicator Function computes a continuous mask from the SDF: $\chi(x) = \tanh(\beta \cdot SDF(x)) \cdot (m(x) - 0.5) + 0.5$. This mask evaluates to 0 inside obstacles and 1 in free space, with a smooth differentiable transition exactly at the boundary.
- **DAFNO Backbone:** 4 Domain-Agnostic Fourier Neural Operator (DAFNO) blocks propagate global topological information. To ensure information does not bleed through obstacles, the spectral convolution is strictly constrained by the mask: $x_{l+1} = \chi \cdot (\mathcal{K}(\chi \cdot x_l) + \mathcal{W}(x_l))$.
- **Metric Projection (DeepNorm):** The final feature tensor is projected into a valid metric space. A non-negative constrained network (using Softplus weights and Concave Activations) enforces the triangle inequality. The strictly asymmetric form $f_\theta(\phi(x) - \phi(g))$ guarantees that the predicted cost-to-go is a valid, monotonically increasing distance metric.

## Optimization & Loss Functions

PNO is optimized using a dual-objective loss function computed exclusively over free-space coordinates:
1. **Supervised Loss:** Mean Squared Error (MSE) against the Ground Truth value function (computed via Fast Marching Method).
2. **PDE Loss:** An Eikonal loss residual $(||\nabla V|| - 1)^2$ computed via central finite differences.

## Repository Structure

```text
motion-planning-pno/
├── data/
│   ├── data_10k_from_orig/    # Preprocessed dataset (mask, dist_in, goal, output)
│   ├── cache_10k/             # Cached FNO-predicted SDFs for PNO training
│   └── visualizations_10k/    # A* path extraction plots
├── checkpoints/               # Trained FNO and PNO model weights
├── src/
│   ├── fno/                   # Standard FNO architecture for SDF generation
│   ├── pno/                   # PNO, DAFNO blocks, and DeepNorm metric head
│   └── data_generation/       # Parallelized data generation and FMM solvers
├── train_fno.py               # Training script for SDF-FNO
├── train_pno.py               # Training script for PNO
├── evaluate_fno.py            # Evaluation and metric calculation for SDF-FNO
├── evaluate_pno.py            # Evaluation and metric calculation for PNO
└── path_extraction.py         # A* Benchmarking against Dijkstra and Euclidean
```

## Execution Pipeline

1. **Dataset Generation:** Generate randomized obstacle maps and exact Fast Marching Method (FMM) value functions.
   `python src/data_generation/generate_10k_from_orig.py`

2. **Train SDF-FNO:** Train the initial operator to predict continuous unsigned distance fields.
   `python train_fno.py --data_dir data/data_10k_from_orig --output_dir checkpoints/fno_sdf_10k`

3. **Train PNO:** Train the main operator. The script automatically predicts and caches SDFs using the trained FNO before commencing training.
   `python train_pno.py --data_dir data/data_10k_from_orig --cache data/cache_10k/pno_cache.npz --fno_checkpoint checkpoints/fno_sdf_10k/model_best.ckpt --fno_config checkpoints/fno_sdf_10k/model_config.json --output_dir checkpoints/pno_10k`

4. **A* Benchmarking:** Evaluate the trained neural heuristic on a large test set (100 samples) to calculate average node expansions and search times.
   `python benchmark_full.py --checkpoint checkpoints/pno_10k/model_best.ckpt --cache data/cache_10k/pno_cache.npz --num_samples 100`

## Benchmark Results

Evaluated over 100 random test samples on the 10k dataset (`data/cache_10k/pno_cache.npz`). The **Depthwise Compressed PNO (76k params)** perfectly preserves the massive structural capacity of the original paper (`width=48`, `modes=12`) while radically reducing the parameter footprint.

| Method | Avg Nodes Expanded | Avg Time (ms) | Speedup vs Euc |
| :--- | :--- | :--- | :--- |
| **Dijkstra (No Heuristic)** | 1453.2 | 31.17 ms | 0.19x |
| **A\* (Euclidean)** | 261.1 | 5.86 ms | **1.00x** (Baseline) |
| **A\* (PNO Compressed)** | **206.4** | **4.82 ms** | **1.22x** |
| **A\* (Ground Truth)** | 183.9 | 4.27 ms | 1.37x |