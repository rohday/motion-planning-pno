# Planning Neural Operator (PNO) for Motion Planning

---

## 📖 Overview

The **Planning Neural Operator (PNO)** is a novel architecture designed to learn and compute optimal value functions (cost-to-go) from 2D obstacle environments to any specified goal position. The value function provides the shortest distance to reach a goal from every spatial location, effectively serving as an $\epsilon$-consistent heuristic for pathfinding algorithms like A*.

Traditional planning algorithms rely on discrete graph searches, while standard CNNs struggle with strict physical constraints. PNO uses **Fourier Neural Operators (FNO)** to operate in continuous space, efficiently capturing global topological information while satisfying the Eikonal Partial Differential Equation (PDE).

### Key Features
- **Generalization**: Resolves the PDE in a resolution-invariant manner.
- **Hardware-Agnostic**: Implemented efficiently using PyTorch.
- **Physical Consistency**: DeepNorm projection head enforces triangle inequalities.
- **End-to-End**: Learns from raw binary occupancy maps via an intermediate Continuous Signed Distance Field (SDF).

---

## 🏗 System Architecture

The pipeline consists of two primary models:

### 1. SDF-FNO (Geometry to SDF)
Binary occupancy maps are non-differentiable at boundaries. To enable smooth gradient-based operations, the binary map is first converted to a Continuous Signed Distance Field (SDF) using an FNO model. 
- **Input**: Binary occupancy map $m(x)$
- **Output**: Signed Distance Field $\text{SDF}(x)$

### 2. Planning Neural Operator (PNO)
The PNO learns the Eikonal solution directly by jointly conditioning on three inputs: the raw binary map, the continuous SDF, and the goal location.
- **Input**: Concatenated tensor $[m(x), \text{SDF}(x), \mathbf{g}(x)]$
- **Core Engine**: $4$ Domain-Agnostic Fourier Neural Operator (DAFNO) Blocks with a Smoothed Indicator Function (SIFN) mask.
- **Output**: Value Function field $V(x, g)$.
- **Projection**: A DeepNorm head guarantees that cost monotonically increases with distance.

---

## 📂 Repository Structure

```
motion-planning-pno/
├── data/                      # Dataset arrays (.npy/.npz)
├── checkpoints/               # Trained model weights and configs
├── src/
│   ├── fno/                   # FNO2dSDF model architecture
│   ├── pno/                   # PNO, DAFNO blocks, and DeepNorm layers
│   └── data_generation/       # Data preprocessing and caching utils
├── train_fno.py               # Training script for SDF-FNO
├── train_pno.py               # Training script for PNO
├── evaluate_fno.py            # Evaluation script for SDF-FNO
├── evaluate_pno.py            # Evaluation script for PNO
├── visualize.py               # Data and result visualization tool
└── requirements.txt           # Python dependencies
```

---

## 📈 Current Progress

### Phase 1: Repository Audit & Core Pipeline Assembly
- **Architecture Validation**: Audited the existing PNO repository and aligned the codebase with efficient, hardware-agnostic design principles observed in reference implementations.
- **Pipeline Completion**: Built missing components to establish a functional end-to-end evaluation pipeline (`evaluate_pno.py`, `evaluate_fno.py`).
- **Bug Fixes**: Addressed critical bugs in the path extraction logic (`path_extraction.py`) and verified DAFNO forward pass formulas.

### Phase 2: Data Generation & Physical Consistency
- **SDF Generation Pipeline**: Improved the data generation scripts to produce highly accurate Signed Distance Functions (SDFs) and Eikonal-based value functions.
- **Boundary Handling**: Resolved normalization inconsistencies and handled obstacle boundaries dynamically by coupling the raw map and SDF inputs correctly.
- **DeepNorm Implementation**: Corrected the DeepNorm projection head to enforce physical constraints (monotonic cost increase with distance) according to the paper.

### Phase 3: Training Stability & Metric Optimization
- **Data Pruning**: Analyzed the `data_new_10k` dataset distributions to identify and remove pathological outliers that were stalling training.
- **Regularization & Hyperparameters**: Applied robust regularization (increased weight decay) and tuned hyperparameters (learning rate schedule, early stopping) to mitigate overfitting.
- **Baseline Establishment**: Executed controlled training experiments on subset data to set solid performance baselines against the original reported results.

### Next Steps
- Finalize the spectral attention mechanisms to evaluate any remaining differences with the paper's original DAFNO design.
- Scale training to the full dataset using optimized hyperparameter sweeps.

