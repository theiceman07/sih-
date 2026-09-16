"""
PHASE 0: Baseline Bank for PS 26142
Run on Kaggle (2x T4, 30 GPU-hr/week)

1. Download one clean Sentinel-2 L2A per zone (Punjab wheat)
2. Extract 2000+ chips
3. Run baselines: bicubic, Lanczos, Real-ESRGAN
4. Compute frozen metrics (PSNR, SSIM, SAM, NDVI delta)
5. Save results to CSV

Exit gate: Table with ESRGAN SAM error ~0.13 (proves deck claim)
Timeline: 8 hours on Kaggle
"""

import sys
import io

# Force UTF-8 stdout so ✓ etc. render on Windows consoles (cp1252 default).
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
from scipy.ndimage import zoom

# Make the repo root importable whether run as `python notebooks/phase0_baseline_bank.py`
# (this script's parent dir) or on Kaggle (mounted dataset dir).
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), "/kaggle/input/srm-sentinel"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.metrics import evaluate, compute_ndvi
from data.mpc_fetch import search_scenes, fetch_scene_bands, stack_to_chips

# Punjab wheat-belt windows (~2-3km each) used for Phase 0. Small windows =
# fast HTTP range reads (seconds, not minutes) instead of full-tile downloads.
PUNJAB_WINDOWS = [
    (75.75, 30.55, 75.80, 30.60),
    (75.80, 30.60, 75.85, 30.65),
    (75.70, 30.50, 75.75, 30.55),
    (75.85, 30.55, 75.90, 30.60),
]
DATE_RANGE = "2023-11-01/2023-12-15"


def bicubic_upsample(chip, scale=2):
    """Bicubic 2x upsampling."""
    _, h, w = chip.shape
    result = np.zeros((chip.shape[0], h*scale, w*scale), dtype=chip.dtype)
    for c in range(chip.shape[0]):
        result[c] = cv2.resize(chip[c], (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    return result


def lanczos_upsample(chip, scale=2):
    """Lanczos 2x upsampling."""
    _, h, w = chip.shape
    result = np.zeros((chip.shape[0], h*scale, w*scale), dtype=chip.dtype)
    for c in range(chip.shape[0]):
        result[c] = cv2.resize(chip[c], (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)
    return result


_ESRGAN_MODEL = None


def _get_esrgan_model():
    """Lazy-load the real RealESRGAN_x4plus weights (downloaded once, ~64MB)."""
    global _ESRGAN_MODEL
    if _ESRGAN_MODEL is None:
        import torch
        from models.rrdbnet import load_pretrained
        weights_path = _REPO_ROOT / "models" / "weights" / "RealESRGAN_x4plus.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"{weights_path} not found. Download it with:\n"
                f'  curl -L -o "{weights_path}" '
                f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            )
        _ESRGAN_MODEL = load_pretrained(str(weights_path), device="cpu")
    return _ESRGAN_MODEL


def esrgan_upsample(chip, scale=4):
    """
    Real Real-ESRGAN (RRDBNet, RealESRGAN_x4plus, official pretrained weights).
    The model is RGB-only (trained on natural photos), so it's applied
    per-band, each band replicated into a fake-RGB triplet and averaged back
    to one channel on output. This is exactly the failure mode the deck
    describes: a generic photo-SR model has no cross-band spectral awareness,
    which is why it distorts indices like NDVI even though PSNR looks fine.
    """
    import torch

    assert scale == 4, "RealESRGAN_x4plus is a fixed 4x model"
    model = _get_esrgan_model()
    c, h, w = chip.shape

    # Batch all bands into one forward pass instead of looping (C, h, w) ->
    # (C, 3, h, w) fake-RGB batch -> single RRDBNet call -> (C, 3, 4h, 4w).
    # Far faster on CPU than C sequential calls (one conv-graph launch, not C).
    band_rgb = np.repeat(chip[:, None, :, :], 3, axis=1)  # (C, 3, h, w)
    x = torch.from_numpy(band_rgb).float()

    with torch.no_grad():
        y = model(x).clamp(0, 1).numpy()  # (C, 3, 4h, 4w)

    return y.mean(axis=1)  # (C, 4h, 4w)


def degrade_chip(chip, scale=4):
    """Degrade chip to 64x64 (Wald protocol)."""
    _, h, w = chip.shape
    degraded = np.zeros((chip.shape[0], h//scale, w//scale), dtype=chip.dtype)
    for c in range(chip.shape[0]):
        degraded[c] = cv2.resize(chip[c], (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
    return degraded


def run_baseline_bank():
    """Main Phase 0 pipeline."""

    print("\n" + "="*60)
    print("PHASE 0: BASELINE BANK")
    print("="*60 + "\n")

    # Real Sentinel-2 L2A chips from Microsoft Planetary Computer (public STAC,
    # no login needed). Small geographic windows -> fast HTTP range reads.
    print("[1/5] Fetching real Sentinel-2 chips (Punjab wheat belt)...")
    real_chips = []
    for bbox in PUNJAB_WINDOWS:
        items = search_scenes(bbox, DATE_RANGE, cloud_cover_max=10, limit=1)
        if not items:
            print(f"   (no scene found for {bbox}, skipping)")
            continue
        item = items[0]
        stack, scl, _, _ = fetch_scene_bands(item, bbox)
        chips = stack_to_chips(stack, scl, chip_size=64, stride=64, cloud_threshold=0.95)
        real_chips.extend(chips)
        print(f"   {item.id[:40]}... cloud={item.properties.get('eo:cloud_cover', 0):.1f}% "
              f"-> {len(chips)} chips")

    if len(real_chips) == 0:
        raise RuntimeError("No real chips extracted -- check network access / MPC availability.")

    # Phase 0 is a fast sanity check, not the final validation set (that's
    # Phase 4, 5-zone report). Cap it so a CPU run finishes in a couple of
    # minutes; bump MAX_CHIPS (or set to None) for a fuller run once this
    # looks right. Fixed seed -> reproducible subset choice.
    MAX_CHIPS = None  # use all extracted chips for this fuller run
    rng = np.random.default_rng(42)
    if MAX_CHIPS is not None and len(real_chips) > MAX_CHIPS:
        idx = rng.choice(len(real_chips), size=MAX_CHIPS, replace=False)
        demo_chips = [real_chips[i] for i in idx]
    else:
        demo_chips = real_chips
    n_chips = len(demo_chips)
    print(f"   ✓ {len(real_chips)} real chips extracted, using {n_chips} "
          f"(64x64, 10 bands, normalized DN)\n")

    # Step 2: Run baselines
    print("[2/5] Running baseline methods...")

    results = []

    for i, chip in enumerate(tqdm(demo_chips, desc="Baseline evaluation")):
        ref = chip
        degraded = degrade_chip(chip, scale=4)

        # Bicubic
        pred_bicubic = bicubic_upsample(degraded, scale=4)
        metrics_bc = evaluate(ref, np.clip(pred_bicubic, 0, 1))
        results.append({'method': 'bicubic', 'chip_id': i, **metrics_bc})

        # Lanczos
        pred_lanczos = lanczos_upsample(degraded, scale=4)
        metrics_lc = evaluate(ref, np.clip(pred_lanczos, 0, 1))
        results.append({'method': 'lanczos', 'chip_id': i, **metrics_lc})

        # Real Real-ESRGAN (RealESRGAN_x4plus, applied per-band -- see esrgan_upsample docstring)
        pred_esrgan = esrgan_upsample(degraded, scale=4)
        metrics_es = evaluate(ref, np.clip(pred_esrgan, 0, 1))
        results.append({'method': 'esrgan', 'chip_id': i, **metrics_es})

    df_results = pd.DataFrame(results)
    print(f"   ✓ Evaluated {len(results)} results\n")

    # Step 3: Aggregated metrics
    print("[3/5] Aggregating metrics...")

    metric_cols = ['psnr', 'ssim', 'sam', 'ergas']
    if 'ndvi_delta' in df_results.columns:
        metric_cols.append('ndvi_delta')
    summary = df_results.groupby('method')[metric_cols].agg(['mean', 'std'])

    print("\n" + summary.to_string())
    print()

    # Step 4: Save results
    print("[4/5] Saving results...")

    output_dir = Path('phase0_results')
    output_dir.mkdir(exist_ok=True)

    df_results.to_csv(output_dir / 'baseline_metrics.csv', index=False)
    summary.to_csv(output_dir / 'baseline_summary.csv')

    print(f"   ✓ Saved to {output_dir}/\n")

    # Step 5: Print exit gate
    print("[5/5] Exit gate check...")

    mean_sam_esrgan = df_results[df_results['method'] == 'esrgan']['sam'].mean()
    mean_sam_lanczos = df_results[df_results['method'] == 'lanczos']['sam'].mean()

    print(f"   ESRGAN mean SAM: {mean_sam_esrgan:.3f}°")
    print(f"   Lanczos mean SAM: {mean_sam_lanczos:.3f}°")

    print("\n" + "="*60)
    print("PHASE 0 COMPLETE")
    print("="*60 + "\n")

    return df_results, summary


if __name__ == "__main__":
    df_results, summary = run_baseline_bank()
