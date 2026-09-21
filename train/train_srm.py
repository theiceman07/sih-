"""
SRM head training loop: 40m -> 10m land-cover fractions (FOUR_DAY_PLAN.md
section 2B). Mirrors train/train.py's structure (smoke test vs local dev
run vs Kaggle-scale config), for the land-cover half of the pipeline that
train.py's Track B SR model does not cover.

Usage:
    python -m train.train_srm --smoke        # ~30s correctness check, no real training
    python -m train.train_srm                # local dev run, small dataset
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.srm_head import SwinIRSRM
from losses.srm_loss import SRMLoss
from losses.sam_loss import SAMLoss
from losses.index_loss import IndexLoss
from data.srm_dataset import build_dataset_from_mpc
from data.worldcover_fetch import CLASS_CODES

RED_IDX, NIR_IDX = 2, 6  # project's standard 10-band order: B04=idx2, B08=idx6

# Covers all three api/main.py DEMO_REGIONS so the live jury demo has
# actually seen training data from (near) each bbox it will be run on.
TRAIN_WINDOWS = [
    (75.75, 30.55, 75.80, 30.60),   # punjab_wheat
    (75.80, 30.60, 75.85, 30.65),   # punjab_wheat_2
    (78.55, 18.75, 78.60, 18.80),   # kudaliar_telangana
]
DATE_RANGE = "2023-11-01/2023-12-15"


def build_model(device):
    # Local/smoke config, same sizing rationale as train/train.py:build_model
    # -- small enough for CPU correctness checks; scale up embed_dim/depths
    # on Kaggle GPUs for the real run.
    model = SwinIRSRM(in_chans=10, n_classes=len(CLASS_CODES), scale=4,
                       embed_dim=60, depths=(2, 2, 2, 2), num_heads=6, window_size=8)
    return model.to(device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_classes = len(CLASS_CODES)

    if args.smoke:
        print("[SMOKE TEST] Using synthetic data, no MPC/WorldCover fetch.")
        lr = torch.rand(4, 10, 16, 16)
        hr = torch.rand(4, 10, 64, 64)
        fractions = torch.softmax(torch.rand(4, n_classes, 64, 64), dim=1)
        dataset = [(lr[i], hr[i], fractions[i]) for i in range(4)]
        n_steps = 20
    else:
        print("Fetching real Sentinel-2 chips + WorldCover labels...")
        dataset = build_dataset_from_mpc(
            TRAIN_WINDOWS, DATE_RANGE, scale=4, chip_size=args.chip_size
        )
        print(f"Dataset size: {len(dataset)} chips")
        n_steps = args.steps

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    srm_loss_fn = SRMLoss(lambda_sum=args.lambda_sum)
    l1_loss_fn = torch.nn.L1Loss()
    sam_loss_fn = SAMLoss()
    idx_loss_fn = IndexLoss(nir_idx=NIR_IDX, red_idx=RED_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    t0 = time.time()
    history = []

    while step < n_steps:
        for lr_batch, hr_batch, frac_batch in loader:
            if step >= n_steps:
                break
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            frac_batch = frac_batch.to(device)

            optimizer.zero_grad()
            sr_pred, frac_pred = model(lr_batch)

            # Both heads now get real gradient: SR head against the real
            # un-degraded 10m chip (L1 + spectral + NDVI, same recipe as
            # Track B/train.py), SRM head against real WorldCover fractions.
            # Previously only the SRM loss was computed here, so the SR
            # head's weights never trained at all -- see data/srm_dataset.py.
            loss_sr = (l1_loss_fn(sr_pred, hr_batch)
                       + args.lambda_sam * sam_loss_fn(sr_pred, hr_batch)
                       + args.lambda_sam * idx_loss_fn(sr_pred, hr_batch))
            loss_srm = srm_loss_fn(frac_pred, frac_batch)
            loss = loss_sr + args.lambda_srm * loss_srm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                pred_hard = frac_pred.argmax(dim=1)
                target_hard = frac_batch.argmax(dim=1)
                acc = (pred_hard == target_hard).float().mean().item()

            history.append({"step": step, "loss": loss.item(), "loss_sr": loss_sr.item(),
                             "loss_srm": loss_srm.item(), "pixel_acc": acc})

            if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
                print(f"step {step:4d}/{n_steps} | loss {loss.item():.4f} "
                      f"| SR {loss_sr.item():.4f} | SRM {loss_srm.item():.4f} | pixel_acc {acc:.3f}")

            step += 1

    elapsed = time.time() - t0
    print(f"\nDone: {step} steps in {elapsed:.1f}s ({elapsed/step:.3f}s/step)")

    if len(history) >= 2:
        first, last = history[0], history[-1]
        print(f"Loss: {first['loss']:.4f} -> {last['loss']:.4f} "
              f"({'decreasing (good sign)' if last['loss'] < first['loss'] else 'NOT decreasing -- check setup'})")

    ckpt_dir = _REPO_ROOT / "train" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ("srm_smoke_test.pt" if args.smoke else "srm_local.pt")
    torch.save({"model_state_dict": model.state_dict(), "history": history}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Synthetic-data correctness check only")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-sum", type=float, default=0.1)
    parser.add_argument("--lambda-sam", type=float, default=0.1)
    parser.add_argument("--lambda-srm", type=float, default=0.5)
    parser.add_argument("--chip-size", type=int, default=32,
                         help="native-resolution chip edge length (smaller = faster CPU iteration)")
    args = parser.parse_args()
    train(args)
