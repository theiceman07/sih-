"""
Shared inference pipeline: fetch real Sentinel-2 -> tile the SRM model
across the scene -> gate -> (optionally) TTA confidence -> return results.

Both api/main.py (the live FastAPI job queue) and scripts/prebake_demo.py
(the offline cache-generation script for the jury demo) call this same
function, so there is exactly one place that can have a pipeline bug --
this file exists specifically because api/main.py's model-config mismatch
bug (depths=(2,2) vs the trained checkpoint's (2,2,2,2)) was only caught by
manually testing a live API call. A single shared function makes that class
of bug testable once, not per-caller.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2

from data.mpc_fetch import search_scenes, fetch_scene_bands
from data.worldcover_fetch import CLASS_NAMES
from models.srm_head import SwinIRSRM
from infer.tiler import tile_infer
from infer.gate import release_gate
from infer.confidence import tta_confidence

_REPO_ROOT = Path(__file__).resolve().parent.parent

SCALE = 4
TILE_SIZE = 32  # small enough to run the current small demo model fast, tile-by-tile
BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

_model = None


def get_model():
    global _model
    if _model is None:
        # Must match train/train_srm.py:build_model()'s architecture exactly
        # -- this loads that script's checkpoint (srm_local.pt) by state_dict,
        # which requires identical layer shapes/counts.
        m = SwinIRSRM(in_chans=10, n_classes=len(CLASS_NAMES), scale=SCALE,
                       embed_dim=60, depths=(2, 2, 2, 2), num_heads=6, window_size=8)
        ckpt_path = _REPO_ROOT / "train" / "checkpoints" / "srm_local.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            m.load_state_dict(state["model_state_dict"])
        m.eval()
        _model = m
    return _model


@dataclass
class PipelineResult:
    sr: np.ndarray                 # (n_bands, H, W) reflectance, gate-passed model output
    fallback_used: bool            # True if gate failed and sr is the bicubic fallback
    class_idx: np.ndarray          # (H, W) int32 land-cover class indices
    confidence: Optional[np.ndarray]  # (H, W) float32 TTA std-dev, or None if use_tta=False
    lr_norm: np.ndarray             # (n_bands, h, w) the original input, normalized [0,1]
    transform: object
    crs: object
    gate_report: dict = field(default_factory=dict)


def run_pipeline(bbox, date_range="2023-11-01/2023-12-15", cloud_cover_max=10, use_tta=False):
    """
    The full fetch -> tile -> gate pipeline, shared by the live API and the
    prebake script. use_tta=True is slower (4x forward passes per tile) but
    also produces a real per-pixel confidence map -- worth it for a one-time
    offline prebake, usually not worth it for an interactive live demo call.
    """
    items = search_scenes(bbox, date_range, cloud_cover_max=cloud_cover_max, limit=1)
    if not items:
        raise RuntimeError(f"No Sentinel-2 scene found for bbox={bbox}, date_range={date_range}")

    stack, scl, transform, crs = fetch_scene_bands(items[0], bbox)
    lr_norm = np.clip(stack / 10000.0, 0.0, 1.0).astype(np.float32)

    model = get_model()
    n_bands = lr_norm.shape[0]
    n_classes = len(CLASS_NAMES)

    def base_infer_fn(tile):
        x = torch.from_numpy(tile).float().unsqueeze(0)
        with torch.no_grad():
            sr, fractions = model(x)
        return torch.cat([sr.squeeze(0), fractions.squeeze(0)], dim=0).numpy()

    if use_tta:
        def infer_fn(tile):
            mean_out, confidence = tta_confidence(tile, base_infer_fn)
            return np.concatenate([mean_out, confidence[None]], axis=0)
    else:
        infer_fn = base_infer_fn

    combined = tile_infer(lr_norm, infer_fn, scale=SCALE, tile_size=TILE_SIZE, overlap=8)
    sr = np.clip(combined[:n_bands], 0, 1)
    fractions = combined[n_bands:n_bands + n_classes]
    class_idx = fractions.argmax(axis=0).astype(np.int32)
    confidence = combined[n_bands + n_classes] if use_tta else None

    gate_result = release_gate(lr_norm, sr, scale=SCALE)

    fallback_used = not gate_result.passed
    if fallback_used:
        # Gate failed: ship bicubic instead of an unverifiable SR tile
        # (FOUR_DAY_PLAN.md section 3B) -- never serve a silently-wrong output.
        sr = np.stack([
            cv2.resize(lr_norm[c], (lr_norm.shape[2] * SCALE, lr_norm.shape[1] * SCALE),
                       interpolation=cv2.INTER_CUBIC)
            for c in range(n_bands)
        ])
        sr = np.clip(sr, 0, 1)

    return PipelineResult(
        sr=sr, fallback_used=fallback_used, class_idx=class_idx, confidence=confidence,
        lr_norm=lr_norm, transform=transform, crs=crs, gate_report=gate_result.as_dict(),
    )
