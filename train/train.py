"""
Phase 1 training loop: Track B (20m -> 10m, real ground truth).

Local runs (this machine, CPU) are for smoke-testing correctness only --
loss going down over a few hundred steps on a handful of chips. Real
training (many thousands of steps, larger model) happens on Kaggle T4s;
see the config block below for what to scale up there.

Usage:
    python -m train.train --smoke        # ~30s correctness check, no real training
    python -m train.train                # local dev run, small dataset
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.swinir_lite import SwinIRLite
from losses.sam_loss import SAMLoss
from losses.index_loss import IndexLoss
from data.track_b_dataset import build_dataset_from_mpc, RED_IDX, NIR_IDX

PUNJAB_TRAIN_WINDOWS = [
    (75.75, 30.55, 75.80, 30.60),
    (75.80, 30.60, 75.85, 30.65),
    (75.70, 30.50, 75.75, 30.55),
    (75.85, 30.55, 75.90, 30.60),
]
DATE_RANGE = "2023-11-01/2023-12-15"


def build_model(device):
    # Local/smoke config: small enough for CPU. On Kaggle, scale up:
    # embed_dim=60->96+, depths=(2,2,2,2)->(6,6,6,6), num_heads accordingly --
    # this is the "RSTB depth 4, embed 60, window 8" spec from
    # DEVELOPMENT_PLAN.md Phase 1, sized for a first correctness pass here.
    model = SwinIRLite(in_chans=4, out_chans=4, scale=2, embed_dim=60,
                        depths=(2, 2, 2, 2), num_heads=6, window_size=8)
    return model.to(device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke:
        # No network calls -- pure synthetic tensors, just proves the training
        # graph (model + losses + optimizer step) is wired correctly.
        print("[SMOKE TEST] Using synthetic data, no MPC fetch.")
        lr = torch.rand(4, 4, 32, 32)
        hr = torch.rand(4, 4, 64, 64)
        dataset = [(lr[i], hr[i]) for i in range(4)]
        n_steps = 20
    else:
        print("Fetching real Sentinel-2 chips for training set...")
        dataset = build_dataset_from_mpc(
            PUNJAB_TRAIN_WINDOWS, DATE_RANGE, scale=2, chip_size=64
        )
        print(f"Dataset size: {len(dataset)} chips")
        n_steps = args.steps

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    l1_loss = torch.nn.L1Loss()
    sam_loss = SAMLoss()
    idx_loss = IndexLoss(nir_idx=NIR_IDX, red_idx=RED_IDX)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss weights per DEVELOPMENT_PLAN.md Phase 1: L = L1 + lambda_sam*SAM + lambda_idx*|dNDVI|
    lambda_sam = args.lambda_sam
    lambda_idx = args.lambda_idx

    model.train()
    step = 0
    t0 = time.time()
    history = []

    while step < n_steps:
        for lr_batch, hr_batch in loader:
            if step >= n_steps:
                break
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

            optimizer.zero_grad()
            pred = model(lr_batch)

            loss_l1 = l1_loss(pred, hr_batch)
            loss_sam = sam_loss(pred, hr_batch)
            loss_idx = idx_loss(pred, hr_batch)
            loss = loss_l1 + lambda_sam * loss_sam + lambda_idx * loss_idx

            loss.backward()
            optimizer.step()

            history.append({
                "step": step, "loss": loss.item(), "l1": loss_l1.item(),
                "sam_rad": loss_sam.item(), "sam_deg": math.degrees(loss_sam.item()),
                "ndvi_l1": loss_idx.item(),
            })

            if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
                print(f"step {step:4d}/{n_steps} | loss {loss.item():.4f} | "
                      f"L1 {loss_l1.item():.4f} | SAM {math.degrees(loss_sam.item()):.2f}deg | "
                      f"NDVI-L1 {loss_idx.item():.4f}")

            step += 1

    elapsed = time.time() - t0
    print(f"\nDone: {step} steps in {elapsed:.1f}s ({elapsed/step:.3f}s/step)")

    if len(history) >= 2:
        first, last = history[0], history[-1]
        print(f"Loss: {first['loss']:.4f} -> {last['loss']:.4f} "
              f"({'decreasing (good sign)' if last['loss'] < first['loss'] else 'NOT decreasing -- check setup'})")

    ckpt_dir = _REPO_ROOT / "train" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ("smoke_test.pt" if args.smoke else "track_b_local.pt")
    torch.save({"model_state_dict": model.state_dict(), "history": history}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Synthetic-data correctness check only")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-sam", type=float, default=0.1)
    parser.add_argument("--lambda-idx", type=float, default=0.1)
    args = parser.parse_args()
    train(args)
