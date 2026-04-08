"""
  python visualize.py --data_dir data/data_64x64 --num_samples 10
"""

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KEY_ALIASES = {
    "raw_map": ["raw_map", "mask"],
    "sdf": ["sdf", "dist_in"],
    "value": ["value", "output"],
    "goal": ["goal"],
}


def _load_array_file(path: Path, expected_aliases: List[str]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)

    if path.suffix.lower() == ".npz":
        z = np.load(path)
        # Prefer matching key name inside .npz, else fall back to first key.
        for k in expected_aliases:
            if k in z.files:
                return z[k]
        if len(z.files) == 1:
            return z[z.files[0]]
        raise ValueError(
            f"NPZ file {path} has keys {z.files}, none match {expected_aliases}."
        )

    raise ValueError(f"Unsupported file type: {path}")


def _find_and_load(data_dir: Path, aliases: List[str]) -> Tuple[np.ndarray, Path]:
    for alias in aliases:
        exact_npy = data_dir / f"{alias}.npy"
        if exact_npy.exists():
            return _load_array_file(exact_npy, aliases), exact_npy

        exact_npz = data_dir / f"{alias}.npz"
        if exact_npz.exists():
            return _load_array_file(exact_npz, aliases), exact_npz

        # wildcard fallback (e.g., dist_in_256.npy)
        matches = sorted(glob.glob(str(data_dir / f"{alias}*.npy")))
        if matches:
            p = Path(matches[0])
            return _load_array_file(p, aliases), p

        matches = sorted(glob.glob(str(data_dir / f"{alias}*.npz")))
        if matches:
            p = Path(matches[0])
            return _load_array_file(p, aliases), p

    raise FileNotFoundError(
        f"Could not find data file for any of aliases {aliases} in {data_dir}"
    )


def load_dataset(data_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Path]]:
    arrays: Dict[str, np.ndarray] = {}
    sources: Dict[str, Path] = {}

    for key, aliases in KEY_ALIASES.items():
        arr, src = _find_and_load(data_dir, aliases)
        arrays[key] = arr
        sources[key] = src

    n = arrays["raw_map"].shape[0]
    for k, arr in arrays.items():
        if arr.shape[0] != n:
            raise ValueError(
                f"Sample count mismatch: raw_map has {n}, but {k} has {arr.shape[0]}"
            )

    if arrays["goal"].ndim != 2 or arrays["goal"].shape[1] != 2:
        raise ValueError("goal array must have shape (N, 2)")

    return arrays, sources


def _as_image(arr: np.ndarray, idx: int) -> np.ndarray:
    x = arr[idx]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def visualize_samples(
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    output_path: Path,
    title: str,
):
    raw_map = arrays["raw_map"]
    sdf = arrays["sdf"]
    value = arrays["value"]
    goal = arrays["goal"]

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(16, max(3 * n, 8)))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, i in enumerate(indices):
        m = _as_image(raw_map, i)
        s = _as_image(sdf, i)
        v = _as_image(value, i)
        gx, gy = goal[i]

        # 1) Map + goal
        ax = axes[row, 0]
        ax.imshow(m, origin="lower", cmap="gray_r")
        ax.plot(gx, gy, "r*", markersize=10)
        ax.set_title("raw_map / mask" if row == 0 else "")
        ax.axis("off")

        # 2) SDF
        ax = axes[row, 1]
        im = ax.imshow(s, origin="lower", cmap="viridis")
        ax.plot(gx, gy, "r*", markersize=8)
        ax.set_title("sdf / dist_in" if row == 0 else "")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # 3) Value/output
        ax = axes[row, 2]
        im = ax.imshow(v, origin="lower", cmap="plasma")
        ax.plot(gx, gy, "r*", markersize=8)
        ax.set_title("value / output" if row == 0 else "")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # 4) Goal-only panel
        ax = axes[row, 3]
        h, w = m.shape[-2], m.shape[-1]
        goal_map = np.zeros((h, w), dtype=np.float32)
        gx_i = int(np.clip(round(float(gx)), 0, w - 1))
        gy_i = int(np.clip(round(float(gy)), 0, h - 1))
        goal_map[gy_i, gx_i] = 1.0
        ax.imshow(goal_map, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title("goal" if row == 0 else "")
        ax.text(0.02, 0.98, f"(x={gx_i}, y={gy_i})", transform=ax.transAxes,
                va="top", ha="left", color="white", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"))
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 10 random samples from dataset arrays (4 files)."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing dataset files")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/visualizations/newvis.png",
        help="Output image path",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    arrays, sources = load_dataset(data_dir)

    n = arrays["raw_map"].shape[0]
    k = min(max(1, args.num_samples), n)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n, size=k, replace=False)

    title = (
        f"Random dataset samples (n={k})\n"
        f"raw_map: {sources['raw_map'].name} | "
        f"sdf: {sources['sdf'].name} | "
        f"value: {sources['value'].name} | "
        f"goal: {sources['goal'].name}"
    )

    output_path = Path(args.output)
    visualize_samples(arrays, indices, output_path, title)

    print(f"Saved visualization: {output_path.resolve()}")
    print(f"Sample indices: {indices.tolist()}")


if __name__ == "__main__":
    main()
