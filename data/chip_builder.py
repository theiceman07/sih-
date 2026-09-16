"""
Build Sentinel-2 training chips: 256x256 10m-resolution with cloud masking.
"""
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import zipfile
import json


def load_band_10m(band_path):
    """Load a 10m band (B2, B3, B4, B8) from GeoTIFF."""
    with rasterio.open(band_path) as src:
        return src.read(1).astype(np.float32)


def resample_to_10m(band_path, reference_shape):
    """Resample 20m band to 10m via bilinear interpolation."""
    with rasterio.open(band_path) as src:
        data = src.read(1).astype(np.float32)

    from scipy.ndimage import zoom
    scale = reference_shape[0] / data.shape[0]
    resampled = zoom(data, scale, order=1)
    return resampled[:reference_shape[0], :reference_shape[1]]


def load_scl(scl_path):
    """
    Load Scene Classification map and return a valid-pixel mask.
    SCL classes: 0=nodata 1=saturated 2=dark-area 3=cloud-shadow 4=vegetation
    5=bare-soil 6=water 7/8/9=cloud(low/med/high) 10=thin-cirrus 11=snow.
    "Valid" excludes cloud/shadow/nodata/saturated -- vegetation (4) is NOT
    the only acceptable class, otherwise non-vegetated scenes (bare soil,
    water, urban) get entirely masked out even at 0% cloud cover.
    """
    CLOUD_OR_INVALID = {0, 1, 3, 7, 8, 9, 10}
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
    return np.isin(scl, list(CLOUD_OR_INVALID), invert=True).astype(np.float32)


def extract_chips(l2a_dir, bands=['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
                  chip_size=256, stride=256, cloud_threshold=0.95):
    """
    Extract non-overlapping 256x256 chips from one L2A scene.
    Returns list of chips, each shape (10, 256, 256).
    """
    l2a_dir = Path(l2a_dir)

    # Find band files
    band_paths = {}
    for band in bands:
        jp2_files = list(l2a_dir.glob(f"*_{band}_*.jp2"))
        if not jp2_files:
            raise ValueError(f"Band {band} not found")
        band_paths[band] = str(jp2_files[0])

    # Load 10m reference
    ref_band = load_band_10m(band_paths[bands[0]])
    h, w = ref_band.shape

    # Load all bands, resampling 20m to 10m
    data = {}
    for band in bands:
        if band in ['B02', 'B03', 'B04', 'B08']:
            data[band] = load_band_10m(band_paths[band])
        else:
            data[band] = resample_to_10m(band_paths[band], (h, w))

    # Load SCL
    scl_files = list(l2a_dir.glob("*_SCL_*.jp2"))
    if scl_files:
        scl = load_scl(scl_files[0])
    else:
        scl = np.ones((h, w))

    # Stack into (C, H, W)
    stack = np.stack([data[b] for b in bands], axis=0)

    # Normalize to [0, 1]
    stack = stack / 10000.0
    stack = np.clip(stack, 0, 1)

    # Extract chips
    chips = []
    for y in range(0, h - chip_size + 1, stride):
        for x in range(0, w - chip_size + 1, stride):
            chip = stack[:, y:y+chip_size, x:x+chip_size]
            scl_chip = scl[y:y+chip_size, x:x+chip_size]

            if scl_chip.mean() >= cloud_threshold:
                chips.append(chip)

    return chips


def save_chips(chips, output_dir, prefix="chip"):
    """Save chips to .npy files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, chip in enumerate(chips):
        path = output_dir / f"{prefix}_{i:06d}.npy"
        np.save(path, chip.astype(np.float32))

    return output_dir
