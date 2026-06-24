# Contributing to Planning Neural Operator (PNO)

First off, thank you for considering contributing to the Planning Neural Operator (PNO) project! We welcome contributions from everyone—whether it's fixing bugs, improving documentation, submitting feature requests, or adding new model capabilities.

This document outlines the process for contributing to the repository, how to set up your development environment, and the coding standards we follow.

---

## 🛠 Setting Up for Development

To contribute effectively, you'll want to set up the repository from scratch on your local machine.

### 1. Clone the Repository
Fork the repository to your own GitHub account and clone your fork locally:
```bash
git clone https://github.com/YOUR-USERNAME/motion-planning-pno.git
cd motion-planning-pno
```

### 2. Environment Setup
We highly recommend using a virtual environment (`venv` or `conda`) to keep your dependencies isolated.

**Using venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

**Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Your Setup
Run the `visualize.py` script to ensure you can load data properly, assuming you have some dummy or provided data in `data/data_64x64/`.

```bash
python visualize.py --data_dir data/data_64x64/ --num_samples 1
```
If the script succeeds and generates a PNG file in `data/visualizations/`, you are ready to start coding!

---

## 🔍 Codebase Architecture Overview

Before contributing code, it's helpful to understand the separation of concerns within `src/`:

- `src/fno/`: Contains the base Fourier Neural Operator architectures. If you want to tweak the spectral convolutions or the 2D SDF conversions, look here.
- `src/pno/`: Contains the domain-specific Planning Neural Operator code.
  - `pno2d.py`: The high-level model connecting the blocks.
  - `layers.py`: The `DAFNO` (Domain-Agnostic FNO) and `DeepNorm` projection heads. If you are modifying physical constraints or masking behavior, this is the place.
- `src/data_generation/`: Everything related to parsing, caching, and batching arrays for training.
- **Root Scripts**: The root directory holds entry points (`train_pno.py`, `evaluate_fno.py`, etc.). Try to keep the heavy logic inside `src/` and only parsing/looping in the root scripts.

---

## 👩‍💻 How to Contribute

### Reporting Bugs
If you find a bug, please create an Issue and include:
- A clear, descriptive title.
- Steps to reproduce the bug.
- Expected behavior vs. actual behavior.
- Details about your environment (OS, Python version, PyTorch version, CUDA version).

### Submitting Pull Requests (PRs)
1. **Create a branch:** `git checkout -b feature/my-awesome-feature`
2. **Make your changes:** Keep commits focused and atomic.
3. **Format your code:** We recommend adhering to PEP-8 standards. Ensure your imports are organized.
4. **Test your code:** If you are adding a new feature or modifying the model architecture, make sure to run a short training loop to ensure gradients flow correctly and no shapes mismatch:
   ```bash
   python train_pno.py --epochs 2 --batch_size 4 --data_dir data/data_64x64
   ```
5. **Commit your changes:**
   ```bash
   git commit -m "Add feature X"
   ```
6. **Push to your fork:**
   ```bash
   git push origin feature/my-awesome-feature
   ```
7. **Open a Pull Request** against the `main` branch of this repository.

### Updating Documentation
Good documentation is just as important as good code. If you make architectural changes, please update the corresponding `.tex` files in `documentation/tex/` and any relevant sections in the `README.md`. 

---

## 🧪 Experimentation Guidelines
If you are contributing new model variants (e.g., trying a different activation function or loss formulation):
1. Use command-line arguments (e.g., via `argparse` in the root scripts) to toggle your new feature. This ensures backward compatibility.
2. Log your metrics thoroughly (loss, PDE loss, supervision loss).
3. If proposing a permanent change to the default model architecture, provide evaluation metrics (MAE, RMSE) comparing your change against the baseline.

## 💬 Code of Conduct
Please be respectful and constructive in your communications in issues and PRs. We are all here to build great motion planning models together. 

Thank you for contributing!
