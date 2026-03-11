# Motion Planning via Operator Learning (PNO)

Replication of [Generalizable Motion Planning via Operator Learning (ICLR 2025)](papers/generalizable_motion_planning_via_operator.pdf).

## Project Structure

```
motion-planning-pno/
│
├── src/
│   ├── fno/                        # FNO architecture (copy your FNO model files here)
│   │   ├── __init__.py
│   │   ├── config.py               # FNOConfig
│   │   ├── layers.py               # SpectralConv2d, DepthwiseSpectralConv2d
│   │   ├── fno2d.py                # FNO2d model
│   │   └── utils.py                # GaussianNormalizer, save/load checkpoint
│   │
│   └── pno/                        # PNO system
│       ├── data_generation/        # Dataset generation
│       │   ├── config.py           # PNODataGenConfig
│       │   ├── generate_occupancy.py
│       │   ├── generate_sdf.py
│       │   ├── generate_value_function.py
│       │   └── generate_fno_dataset.py   # Data for training FNO
│       ├── models/                 # PNO model components (Phase 3)
│       │   ├── layers.py           # DAFNO spectral layers
│       │   ├── deepnorm.py         # Triangle-inequality projection
│       │   └── pno.py              # Full PNO model
│       └── utils/
│           └── path_extraction.py  # Gradient descent on value function
│
├── train_fno.py                    # Step 1: Train FNO (Occupancy, Goal) → SDF
├── train_pno.py                    # Step 2: Train PNO (Occupancy, SDF, Goal) → V
├── evaluate.py                     # Evaluate + visualize either model
│
├── data/                           # Generated datasets (gitignored)
│   ├── fno/                        # FNO training data
│   └── pno/                        # PNO training data (generated via FNO)
│
├── checkpoints/                    # Saved models (gitignored)
│   ├── fno/
│   └── pno/
│
├── papers/                         # Reference papers
├── requirements.txt
└── .gitignore
```

## Pipeline

```
Step 1 — Generate data & train FNO:
  python src/pno/data_generation/generate_fno_dataset.py   # occupancy + goal → SDF pairs
  python train_fno.py                                       # trains FNO2d on (occ, goal) → SDF

Step 2 — Generate PNO dataset using trained FNO:
  (script coming in next phase)

Step 3 — Train PNO:
  (script coming in next phase)
```

## CLI Commands (DataGen + FNO Training)

Run all commands from project root after activating `.venv`.

### 1) Generate FNO dataset

Default command:

```bash
python src/pno/data_generation/generate_fno_dataset.py
```

Recommended custom command:

```bash
python src/pno/data_generation/generate_fno_dataset.py \
  --train_maps 3000 \
  --val_maps 500 \
  --test_maps 500 \
  --grid_size 64 \
  --num_goals 5 \
  --method mixed \
  --seed 42 \
  --output data/fno
```

What this writes:
- `data/fno/train.npz`
- `data/fno/val.npz`
- `data/fno/test.npz`
- `data/fno/config.yaml`

Show all dataset flags:

```bash
python src/pno/data_generation/generate_fno_dataset.py -h
```

### 2) Train FNO (PNO prerequisite)

Default command:

```bash
python train_fno.py
```

Recommended training command:

```bash
python train_fno.py \
  --data_path data/fno/train.npz \
  --val_path data/fno/val.npz \
  --output_dir checkpoints/fno \
  --epochs 500 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --modes 16 \
  --width 64 \
  --depth 4 \
  --num_workers 4 \
  --eval_every 10 \
  --patience 50 \
  --grad_clip 0.25 \
  --auto_eval
```

Useful variants:

```bash
# quick sanity check with synthetic tensors
python train_fno.py --dry_run

# live dashboard during training
python train_fno.py --plot

# run evaluation on test split after training
python train_fno.py --auto_eval --eval_data data/fno/test.npz
```

Show all training flags:

```bash
python train_fno.py -h
```

Training outputs are saved in `checkpoints/fno/`:
- `fno_best.pth`
- `fno_latest.pth`
- `history.json`
- `metadata.json`
- `eval_stats.json` and `eval_visualization.png` (if `--auto_eval`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After activating your venv, copy your FNO model files into `src/fno/`:
```bash
cp FNO_wrapper/model/{config,layers,fno2d,utils}.py src/fno/
```
