"""
Real-GT SAM-loss ablation on Kudaliar (FOUR_DAY_PLAN.md section 2A):
lambda_sam=0 (L1 only) vs lambda_sam=0.1 (L1+SAM+NDVI), trained and
evaluated on REAL measured Sentinel-2/VENuS pairs -- not the simulated
MTF-degraded chips used elsewhere (data/track_b_dataset.py,
data/srm_dataset.py). This is the plan's core scientific claim (does the
SAM loss actually help spectral fidelity) tested on the one dataset in the
project with zero synthetic-degradation assumption in the label.

CPU-only local run: short enough to show a directional trend, not a fully
converged model (see train/train.py's own docstring about this same
local-vs-Kaggle split). Scale up steps/model size on a Kaggle GPU for the
real ablation numbers that go in the evidence pack.

Usage:
    python -m train.train_kudaliar --steps 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.swinir_lite import SwinIRLite
from losses.sam_loss import SAMLoss
from losses.index_loss import IndexLoss
from data.sen2venus_kudaliar import train_test_split_by_date
from eval.metrics import psnr, ssim, sam, ergas, compute_ndvi, ndvi_delta as ndvi_delta_fn

RED_IDX, NIR_IDX = 2, 3  # Kudaliar's 10m_b2b3b4b8 order: B02,B03,B04,B08


def build_model():
    return SwinIRLite(in_chans=4, out_chans=4, scale=2, embed_dim=60,
                       depths=(2, 2, 2, 2), num_heads=6, window_size=8)


def train_one(lambda_sam, train_ds, steps, batch_size, lr, seed=0):
    torch.manual_seed(seed)
    # NOTE: batch_size>1 at Kudaliar's native 128x128 LR resolution (256
    # attention windows/sample) triggers a Windows access violation in
    # torch's batched matmul on this dev machine (confirmed via faulthandler
    # and reproducible with synthetic data too -- an environment/MKL
    # threading issue at that size, not a data or model-logic bug). Using
    # data/sen2venus_kudaliar.py's lr_crop_size to train at 32x32 avoids it
    # entirely (verified) and is fast enough for local CPU iteration.
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = build_model()
    l1_loss = torch.nn.L1Loss()
    sam_loss_fn = SAMLoss()
    idx_loss_fn = IndexLoss(nir_idx=NIR_IDX, red_idx=RED_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    step = 0
    history = []
    while step < steps:
        for lr_batch, hr_batch in loader:
            if step >= steps:
                break
            optimizer.zero_grad()
            pred = model(lr_batch)

            loss_l1 = l1_loss(pred, hr_batch)
            loss_sam = sam_loss_fn(pred, hr_batch)
            loss_idx = idx_loss_fn(pred, hr_batch)
            loss = loss_l1 + lambda_sam * (loss_sam + loss_idx)

            loss.backward()
            # Gradient clipping: an earlier batch_size=1 run on real,
            # high-contrast (field-boundary-heavy) Kudaliar data diverged
            # without it; kept as a safety margin now that batch_size=4
            # (via the lr_crop_size=32 workaround above) provides its own
            # batch-averaging stability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            history.append(loss.item())
            step += 1

    return model, history


def evaluate_on_test(model, test_ds):
    """Frozen eval/metrics.py suite, computed per-patch then averaged -- real held-out dates."""
    model.eval()
    rows = {"psnr": [], "ssim": [], "sam": [], "ergas": [], "ndvi_delta": []}
    with torch.no_grad():
        for i in range(len(test_ds)):
            lr, hr = test_ds[i]
            pred = model(lr.unsqueeze(0)).squeeze(0).clamp(0, 1).numpy()
            hr_np = hr.numpy()

            rows["psnr"].append(psnr(hr_np, pred))
            rows["ssim"].append(ssim(hr_np, pred))
            rows["sam"].append(sam(hr_np, pred))
            rows["ergas"].append(ergas(hr_np, pred))

            ref_ndvi = compute_ndvi(hr_np[NIR_IDX], hr_np[RED_IDX])
            pred_ndvi = compute_ndvi(pred[NIR_IDX], pred[RED_IDX])
            rows["ndvi_delta"].append(ndvi_delta_fn(ref_ndvi, pred_ndvi))

    return {k: float(np.mean(v)) for k, v in rows.items()}


def main(args):
    print("Loading Kudaliar real ground-truth (10m -> 5m, x2)...")
    crop = args.lr_crop_size if args.lr_crop_size else None
    train_ds, test_ds, train_dates, test_dates = train_test_split_by_date(
        ratio="10m_5m", n_test_dates=2, max_patches_per_acquisition=args.max_patches,
        lr_crop_size=crop)
    print(f"train={len(train_ds)} patches ({len(train_dates)} dates), "
          f"test={len(test_ds)} patches ({test_dates})")

    results = {}
    for lambda_sam in (0.0, args.lambda_sam):
        label = f"lambda_sam={lambda_sam}"
        print(f"\n--- Training {label} ---")
        model, history = train_one(lambda_sam, train_ds, args.steps, args.batch_size, args.lr)
        checkpoints = [history[i] for i in range(0, len(history), max(1, len(history) // 5))]
        print(f"  loss trajectory: {[round(v, 4) for v in checkpoints]} -> {history[-1]:.4f}")
        metrics = evaluate_on_test(model, test_ds)
        print(f"  test metrics: {metrics}")
        results[label] = metrics

    print("\n" + "=" * 60)
    print("SAM-loss ablation on REAL Kudaliar ground truth (held-out dates)")
    print("=" * 60)
    for label, m in results.items():
        print(f"{label:20s} PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} "
              f"SAM={m['sam']:.3f}deg ERGAS={m['ergas']:.3f} dNDVI={m['ndvi_delta']:.4f}")

    out_dir = _REPO_ROOT / "phase0_results"
    out_dir.mkdir(exist_ok=True)
    import csv
    out_path = out_dir / "kudaliar_sam_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "psnr", "ssim", "sam", "ergas", "ndvi_delta"])
        writer.writeheader()
        for label, m in results.items():
            writer.writerow({"config": label, **m})
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-sam", type=float, default=0.1)
    parser.add_argument("--max-patches", type=int, default=24,
                         help="patches per acquisition date (memory-bounded, see data/sen2venus_kudaliar.py)")
    parser.add_argument("--lr-crop-size", type=int, default=32,
                         help="center-crop LR patches to this size for fast local CPU runs (None/0 = full 128x128)")
    args = parser.parse_args()
    main(args)
