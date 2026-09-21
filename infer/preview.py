"""
Renders multi-band reflectance arrays and land-cover class maps as PNGs a
browser can display directly -- GeoTIFFs need a tile server (titiler etc)
to view in-browser, which is out of scope for the hackathon demo. This is
what makes the jury demo show actual before/after imagery instead of only
metric tables.
"""
import io

import numpy as np
from PIL import Image

from data.worldcover_fetch import CLASS_CODES


def _percentile_stretch(band, low=2, high=98):
    """Contrast-stretch one band to 0-255 using percentile clipping (standard
    true-color rendering practice -- raw reflectance is too dark/flat otherwise)."""
    lo, hi = np.percentile(band, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    stretched = np.clip((band - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def _downscale(img, max_size):
    """Cap the longer edge at max_size -- full-resolution scenes (thousands
    of px) rendered as PNG were 10MB+ each, too slow to load in a browser
    demo and needlessly large to commit; QGIS work should use the GeoTIFF,
    not this preview."""
    if max_size is None:
        return img
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def true_color_png(stack, red_idx=2, green_idx=1, blue_idx=0, max_size=800):
    """
    stack: (C, H, W) float32 in [0, 1] (or any consistent scale), project's
    standard band order [B02,B03,B04,...] by default (red_idx=2 -> B04).
    max_size: cap the longer output edge in pixels (None = full resolution).
    Returns PNG bytes.
    """
    r = _percentile_stretch(stack[red_idx])
    g = _percentile_stretch(stack[green_idx])
    b = _percentile_stretch(stack[blue_idx])
    rgb = np.stack([r, g, b], axis=-1)
    img = _downscale(Image.fromarray(rgb, mode="RGB"), max_size)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# Same palette as infer/export_geotiff.py's colour table, for visual consistency
# between the GeoTIFF opened in QGIS and the PNG preview shown in the browser.
_PALETTE_RGB = [
    (0, 100, 0), (255, 187, 34), (255, 255, 76), (240, 150, 255),
    (250, 0, 0), (180, 180, 180), (240, 240, 240), (0, 100, 200),
    (0, 150, 160), (0, 207, 117), (250, 230, 160),
]


def landcover_png(class_idx, max_size=800):
    """class_idx: (H, W) int, values 0..len(CLASS_CODES)-1. Returns PNG bytes."""
    h, w = class_idx.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(CLASS_CODES)):
        rgb[class_idx == i] = _PALETTE_RGB[i % len(_PALETTE_RGB)]
    img = _downscale(Image.fromarray(rgb, mode="RGB"), max_size)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def confidence_png(confidence, cmap_low=(20, 20, 90), cmap_high=(255, 60, 60), max_size=800):
    """
    confidence: (H, W) float, higher = less trustworthy (e.g. TTA std-dev).
    Renders a simple two-colour heat gradient (low=blue, high=red) rather
    than pulling in matplotlib just for a colormap.
    """
    lo, hi = np.percentile(confidence, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((confidence - lo) / (hi - lo), 0, 1)[..., None]
    low = np.array(cmap_low, dtype=np.float32)
    high = np.array(cmap_high, dtype=np.float32)
    rgb = (low * (1 - t) + high * t).astype(np.uint8)
    img = _downscale(Image.fromarray(rgb, mode="RGB"), max_size)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    import cv2

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.4, size=(10, 64, 64)).astype(np.float32)
    stack = np.stack([cv2.GaussianBlur(raw[c], (5, 5), 1.5) for c in range(10)])

    png = true_color_png(stack)
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a valid PNG"
    print(f"true_color_png: {len(png)} bytes")

    class_idx = rng.integers(0, 11, size=(64, 64)).astype(np.int32)
    png_lc = landcover_png(class_idx)
    assert png_lc[:8] == b"\x89PNG\r\n\x1a\n"
    print(f"landcover_png: {len(png_lc)} bytes")

    conf = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)
    png_conf = confidence_png(conf)
    assert png_conf[:8] == b"\x89PNG\r\n\x1a\n"
    print(f"confidence_png: {len(png_conf)} bytes")

    print("Preview rendering sanity checks passed.")
