"""
Frozen metric suite for PS 26142 evaluation.
DO NOT MODIFY after baseline runs.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, pred):
    """Peak Signal-to-Noise Ratio."""
    return peak_signal_noise_ratio(ref, pred, data_range=ref.max() - ref.min())


def ssim(ref, pred):
    """Structural Similarity Index."""
    return structural_similarity(ref, pred, data_range=ref.max() - ref.min(), channel_axis=0)


def sam(ref, pred, eps=1e-8):
    """Spectral Angle Mapper (degrees).
    ref, pred: shape (C, H, W) normalized to [0, 1]
    Returns mean SAM in degrees.
    """
    # Reshape to (C, -1)
    ref_flat = ref.reshape(ref.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)

    # Compute spectral angles
    dots = np.sum(ref_flat * pred_flat, axis=0)
    norms_ref = np.sqrt(np.sum(ref_flat ** 2, axis=0))
    norms_pred = np.sqrt(np.sum(pred_flat ** 2, axis=0))

    cos_angles = dots / (norms_ref * norms_pred + eps)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles)

    return np.degrees(np.mean(angles))


def ndvi_delta(ref_ndvi, pred_ndvi):
    """Mean absolute error in NDVI.
    ref_ndvi, pred_ndvi: shape (H, W), normalized to [-1, 1]
    """
    return np.mean(np.abs(ref_ndvi - pred_ndvi))


def ergas(ref, pred):
    """Erreur Relative Globale Adimensionnelle de Synthèse.
    ref, pred: shape (C, H, W)
    """
    mse = np.mean((ref - pred) ** 2, axis=(1, 2))
    mean_ref = np.mean(ref, axis=(1, 2))
    return 100 * np.sqrt(np.mean(mse / (mean_ref ** 2 + 1e-8)))


def compute_ndvi(nir, red):
    """NDVI = (NIR - RED) / (NIR + RED).
    nir, red: shape (H, W), float in [0, 1]
    """
    return (nir - red) / (nir + red + 1e-8)


def evaluate(ref, pred):
    """
    Compute full metric suite.
    ref, pred: shape (C, H, W), float in [0, 1]
    Assumes bands: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
    Band 4 = RED, Band 8 = NIR for NDVI.
    """
    results = {
        'psnr': psnr(ref, pred),
        'ssim': ssim(ref, pred),
        'sam': sam(ref, pred),
        'ergas': ergas(ref, pred),
    }

    # NDVI if we have enough bands
    if ref.shape[0] >= 9:  # Has B4 (idx 2) and B8 (idx 6)
        ref_ndvi = compute_ndvi(ref[6], ref[2])
        pred_ndvi = compute_ndvi(pred[6], pred[2])
        results['ndvi_delta'] = ndvi_delta(ref_ndvi, pred_ndvi)

    return results
