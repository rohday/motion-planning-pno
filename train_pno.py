#!/usr/bin/env python3
# cli: python train_pno.py [--data_dir] [--cache] [--output_dir] [--cpu] [--fno_checkpoint] [--fno_config] [--fno_batch_size] [--use_fno_sdf / --no-use_fno_sdf] [--refresh_cache_from_fno / --no-refresh_cache_from_fno] [--width] [--modes] [--depth] [--beta] [--deepnorm_hidden] [--epochs] [--batch_size] [--lr] [--weight_decay] [--lr_step] [--lr_gamma] [--grad_clip] [--early_stop] [--lambda_sup] [--lambda_pde] [--train_ratio] [--val_ratio] [--seed] [--num_workers]

import argparse
import glob
import json
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
import torch.nn.functional as F

from data.pno_loader import build_pno_dataloaders, PNOCachedDataset
from src.fno.fno2d import FNO2dSDF
from src.pno import EikonalLoss, PlanningNeuralOperator


def evaluate(model, loader, device, pde_loss_fn, lambda_sup=1.0, lambda_pde=1.0):
	model.eval()
	total, sup_total, pde_total = 0.0, 0.0, 0.0
	n = 0

	with torch.no_grad():
		for batch in loader:
			raw_map = batch["raw_map"].to(device)
			sdf = batch["sdf"].to(device)
			goal = batch["goal"].to(device)

			pred = model(raw_map, sdf, goal)

			loss_sup = torch.zeros((), device=device)
			if "value" in batch:
				target = batch["value"].to(device)
				free_bool = raw_map > 0.5
				loss_sup = F.mse_loss(pred[free_bool], target[free_bool])

			loss_pde = pde_loss_fn(pred, raw_map)
			loss = lambda_sup * loss_sup + lambda_pde * loss_pde

			bs = raw_map.shape[0]
			n += bs
			total += loss.item() * bs
			sup_total += loss_sup.item() * bs
			pde_total += loss_pde.item() * bs

	return {
		"loss": total / max(1, n),
		"sup": sup_total / max(1, n),
		"pde": pde_total / max(1, n),
	}


def _find_npy(data_dir: str, base: str) -> str:
	exact = Path(data_dir) / f"{base}.npy"
	if exact.exists():
		return str(exact)
	matches = glob.glob(str(Path(data_dir) / f"{base}*.npy"))
	if matches:
		return matches[0]
	raise FileNotFoundError(f"Cannot find {base}*.npy in {data_dir}")


def _rebuild_pno_cache_from_fno(args, device):
	cache_path = Path(args.cache)
	cache_path.parent.mkdir(parents=True, exist_ok=True)

	fno_cfg_path = Path(args.fno_config)
	fno_ckpt_path = Path(args.fno_checkpoint)
	if not fno_cfg_path.exists():
		raise FileNotFoundError(f"FNO config not found: {fno_cfg_path}")
	if not fno_ckpt_path.exists():
		raise FileNotFoundError(f"FNO checkpoint not found: {fno_ckpt_path}")

	with open(fno_cfg_path, "r", encoding="utf-8") as f:
		fno_cfg = json.load(f)
	if fno_cfg.get("task") != "geometry_to_sdf":
		raise ValueError("FNO config task must be 'geometry_to_sdf' for PNO input generation.")

	state = torch.load(fno_ckpt_path, map_location=device, weights_only=True)
	fno = FNO2dSDF(
		depth=int(fno_cfg["depth"]),
		padding=int(fno_cfg["padding"]),
		modes1=int(fno_cfg["modes"]),
		modes2=int(fno_cfg["modes"]),
		width=int(fno_cfg["width"]),
		depthwise=bool(fno_cfg.get("depthwise", False)),
	).to(device)
	fno.load_state_dict(state)
	fno.eval()

	norm = fno_cfg.get("normalization", {})
	do_x_norm = bool(norm.get("normalize_input", False))
	do_y_denorm = bool(norm.get("normalize_target", False))
	x_mean = float(norm.get("x_mean", 0.0))
	x_std = max(float(norm.get("x_std", 1.0)), 1e-6)
	y_mean = float(norm.get("y_mean", 0.0))
	y_std = max(float(norm.get("y_std", 1.0)), 1e-6)

	raw_map = np.load(_find_npy(args.data_dir, "mask")).astype(np.float32)
	goal = np.load(_find_npy(args.data_dir, "goal")).astype(np.float32)
	value = np.load(_find_npy(args.data_dir, "output")).astype(np.float32)

	x = torch.from_numpy(raw_map).unsqueeze(1)
	if do_x_norm:
		x = (x - x_mean) / x_std

	all_sdf = []
	with torch.no_grad():
		for i in range(0, x.shape[0], args.fno_batch_size):
			xb = x[i:i + args.fno_batch_size].to(device)
			pred = fno(xb)
			if do_y_denorm:
				pred = pred * y_std + y_mean
			all_sdf.append(pred.squeeze(1).cpu())

	sdf = torch.cat(all_sdf, dim=0).numpy().astype(np.float32)

	np.savez_compressed(
		cache_path,
		raw_map=raw_map,
		sdf=sdf,
		goal=goal,
		value=value,
	)
	print(f"[cache] Rebuilt {cache_path} using FNO SDF predictions from {fno_ckpt_path}")


def _cache_matches_fno_probe(args, device, probe_n: int = 16, atol: float = 1e-4):
	cache_path = Path(args.cache)
	if not cache_path.exists():
		return False

	with open(args.fno_config, "r", encoding="utf-8") as f:
		fno_cfg = json.load(f)
	state = torch.load(args.fno_checkpoint, map_location=device, weights_only=True)
	fno = FNO2dSDF(
		depth=int(fno_cfg["depth"]),
		padding=int(fno_cfg["padding"]),
		modes1=int(fno_cfg["modes"]),
		modes2=int(fno_cfg["modes"]),
		width=int(fno_cfg["width"]),
		depthwise=bool(fno_cfg.get("depthwise", False)),
	).to(device)
	fno.load_state_dict(state)
	fno.eval()

	norm = fno_cfg.get("normalization", {})
	do_x_norm = bool(norm.get("normalize_input", False))
	do_y_denorm = bool(norm.get("normalize_target", False))
	x_mean = float(norm.get("x_mean", 0.0))
	x_std = max(float(norm.get("x_std", 1.0)), 1e-6)
	y_mean = float(norm.get("y_mean", 0.0))
	y_std = max(float(norm.get("y_std", 1.0)), 1e-6)

	raw_map = np.load(_find_npy(args.data_dir, "mask")).astype(np.float32)[:probe_n]
	cache = np.load(cache_path)
	cache_sdf = cache["sdf"][:probe_n].astype(np.float32)

	x = torch.from_numpy(raw_map).unsqueeze(1)
	if do_x_norm:
		x = (x - x_mean) / x_std

	with torch.no_grad():
		pred = fno(x.to(device))
		if do_y_denorm:
			pred = pred * y_std + y_mean
	pred = pred.squeeze(1).cpu().numpy().astype(np.float32)

	return np.allclose(pred, cache_sdf, atol=atol, rtol=1e-4)


def train(args):
	device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	print(f"Device: {device}")

	if args.use_fno_sdf:
		if args.refresh_cache_from_fno or not _cache_matches_fno_probe(args, device):
			_rebuild_pno_cache_from_fno(args, device)
		else:
			print(f"[cache] Using existing FNO-consistent cache: {args.cache}")

	train_loader, val_loader, test_loader = build_pno_dataloaders(
		cache_path=args.cache,
		batch_size=args.batch_size,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		num_workers=args.num_workers,
		seed=args.seed,
	)

	ds = PNOCachedDataset(args.cache)
	has_supervision = ds.value is not None
	if not has_supervision:
		print("[warn] No value target in cache; supervised term disabled.")

	model = PlanningNeuralOperator(
		width=args.width,
		modes1=args.modes,
		modes2=args.modes,
		depth=args.depth,
		beta=args.beta,
		deepnorm_hidden=args.deepnorm_hidden,
	).to(device)

	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameters: {n_params:,}")

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.lr,
		weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer,
		step_size=args.lr_step,
		gamma=args.lr_gamma,
	)
	pde_loss_fn = EikonalLoss()

	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	ckpt_path = out_dir / "model_best.ckpt"

	history = []
	best_val = float("inf")
	stale = 0

	for ep in range(1, args.epochs + 1):
		t0 = default_timer()
		model.train()

		ep_total, ep_sup, ep_pde, n = 0.0, 0.0, 0.0, 0
		for batch in train_loader:
			raw_map = batch["raw_map"].to(device)
			sdf = batch["sdf"].to(device)
			goal = batch["goal"].to(device)

			pred = model(raw_map, sdf, goal)

			loss_sup = torch.zeros((), device=device)
			if has_supervision and "value" in batch:
				target = batch["value"].to(device)
				free_bool = raw_map > 0.5
				loss_sup = F.mse_loss(pred[free_bool], target[free_bool])

			loss_pde = pde_loss_fn(pred, raw_map)
			loss = args.lambda_sup * loss_sup + args.lambda_pde * loss_pde

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
			optimizer.step()

			bs = raw_map.shape[0]
			n += bs
			ep_total += loss.item() * bs
			ep_sup += loss_sup.item() * bs
			ep_pde += loss_pde.item() * bs

		scheduler.step()

		train_stats = {
			"loss": ep_total / max(1, n),
			"sup": ep_sup / max(1, n),
			"pde": ep_pde / max(1, n),
		}
		val_stats = evaluate(
			model,
			val_loader,
			device,
			pde_loss_fn,
			lambda_sup=args.lambda_sup,
			lambda_pde=args.lambda_pde,
		)

		t1 = default_timer()
		row = {
			"epoch": ep,
			"train_loss": train_stats["loss"],
			"train_sup": train_stats["sup"],
			"train_pde": train_stats["pde"],
			"val_loss": val_stats["loss"],
			"val_sup": val_stats["sup"],
			"val_pde": val_stats["pde"],
			"lr": optimizer.param_groups[0]["lr"],
			"sec": t1 - t0,
		}
		history.append(row)

		is_best = row["val_loss"] < best_val
		if is_best:
			best_val = row["val_loss"]
			stale = 0
			torch.save(model.state_dict(), ckpt_path)
			print(
				f"ep {ep:4d}/{args.epochs} | {t1-t0:6.2f}s | "
				f"train={row['train_loss']:.6f} (sup={row['train_sup']:.6f}, pde={row['train_pde']:.6f}) | "
				f"val={row['val_loss']:.6f}  <-- best"
			)
		else:
			stale += 1
			print(
				f"ep {ep:4d}/{args.epochs} | {t1-t0:6.2f}s | "
				f"train={row['train_loss']:.6f} (sup={row['train_sup']:.6f}, pde={row['train_pde']:.6f}) | "
				f"val={row['val_loss']:.6f}"
			)

		if args.early_stop > 0 and stale >= args.early_stop:
			print(f"Early stopping at epoch {ep}")
			break

	model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
	test_stats = evaluate(
		model,
		test_loader,
		device,
		pde_loss_fn,
		lambda_sup=args.lambda_sup,
		lambda_pde=args.lambda_pde,
	)

	(out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
	np.savetxt(
		out_dir / "loss_train_val.txt",
		np.array([[r["epoch"], r["train_loss"], r["val_loss"], r["train_pde"], r["val_pde"]] for r in history], dtype=np.float32),
		header="epoch train_loss val_loss train_pde val_pde",
	)

	config = {
		"cache": args.cache,
		"data_dir": args.data_dir,
		"use_fno_sdf": args.use_fno_sdf,
		"fno_checkpoint": args.fno_checkpoint,
		"fno_config": args.fno_config,
		"width": args.width,
		"modes": args.modes,
		"depth": args.depth,
		"beta": args.beta,
		"deepnorm_hidden": args.deepnorm_hidden,
		"lambda_sup": args.lambda_sup,
		"lambda_pde": args.lambda_pde,
		"best_val": best_val,
		"test": test_stats,
	}
	(out_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

	print("-" * 80)
	print(f"Best val loss: {best_val:.6f}")
	print(f"Test loss:     {test_stats['loss']:.6f} (sup={test_stats['sup']:.6f}, pde={test_stats['pde']:.6f})")
	print(f"Saved model:   {ckpt_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train Planning Neural Operator (DAFNO + DeepNorm)")

	parser.add_argument("--data_dir", type=str, default="data/data_64x64")
	parser.add_argument("--cache", type=str, default="data/cache_64x64/pno_cache.npz")
	parser.add_argument("--output_dir", type=str, default="checkpoints/pno")
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--fno_checkpoint", type=str, default="checkpoints/fno_sdf/model_best.ckpt")
	parser.add_argument("--fno_config", type=str, default="checkpoints/fno_sdf/model_config.json")
	parser.add_argument("--fno_batch_size", type=int, default=128)
	parser.add_argument(
		"--use_fno_sdf",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Use SDF predicted by the configured FNO checkpoint as PNO input (default: on)",
	)
	parser.add_argument(
		"--refresh_cache_from_fno",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Force rebuilding cache from FNO before training (default: off; auto-rebuild on mismatch)",
	)

	parser.add_argument("--width", type=int, default=48)
	parser.add_argument("--modes", type=int, default=12)
	parser.add_argument("--depth", type=int, default=4)
	parser.add_argument("--beta", type=float, default=5.0)
	parser.add_argument("--deepnorm_hidden", type=int, default=64)

	parser.add_argument("--epochs", type=int, default=300)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=2e-3)
	parser.add_argument("--weight_decay", type=float, default=1e-6)
	parser.add_argument("--lr_step", type=int, default=100)
	parser.add_argument("--lr_gamma", type=float, default=0.5)
	parser.add_argument("--grad_clip", type=float, default=1.0)
	parser.add_argument("--early_stop", type=int, default=50)

	parser.add_argument("--lambda_sup", type=float, default=1.0)
	parser.add_argument("--lambda_pde", type=float, default=0.1)

	parser.add_argument("--train_ratio", type=float, default=0.8)
	parser.add_argument("--val_ratio", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num_workers", type=int, default=0)

	train(parser.parse_args())
