"""
ESA WorldCover 10m v200 fetch -- SRM (land-cover) labels. See FOUR_DAY_PLAN.md
section 1B and 2B.

WorldCover is served as a STAC collection on the same Microsoft Planetary
Computer catalog as our Sentinel-2 fetch (data/mpc_fetch.py), so this reuses
that pattern: windowed HTTP range reads onto Sentinel-2's 10m reference grid,
no full-tile download.

These labels are a REAL, independently-measured product (not a downsampled
copy of our Sentinel-2 input), which is what makes them usable as SRM
supervision -- see DEVELOPMENT_PLAN.md / FOUR_DAY_PLAN.md section 2B.
"""
import numpy as np
import rasterio
import pystac_client
import planetary_computer as pc
from rasterio.enums import Resampling
from rasterio.warp import reproject

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "esa-worldcover"

# WorldCover v200 (2021) class codes -> the 11-class legend used throughout
# the SRM head. Codes are the raw pixel values in the "map" asset.
CLASS_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
CLASS_NAMES = [
    "tree_cover", "shrubland", "grassland", "cropland", "built_up",
    "bare_sparse_veg", "snow_ice", "water", "herbaceous_wetland",
    "mangrove", "moss_lichen",
]
CODE_TO_INDEX = {code: i for i, code in enumerate(CLASS_CODES)}


def search_worldcover(bbox, version="v200"):
    """
    bbox: (west, south, east, north)
    Returns the single best-matching WorldCover STAC item (there is one
    global product per version, tiled in 3x3 degree cells).
    """
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = catalog.search(collections=[COLLECTION], bbox=bbox, limit=10)
    items = [it for it in search.item_collection() if version in it.id]
    if not items:
        raise RuntimeError(f"No WorldCover {version} tile found for bbox={bbox}")
    return items[0]


def fetch_worldcover_labels(bbox, ref_shape, ref_transform, ref_crs, version="v200"):
    """
    Read WorldCover class codes for bbox, resampled (nearest-neighbour --
    labels must not be interpolated) onto the given Sentinel-2 10m reference
    grid (same grid produced by data.mpc_fetch.fetch_scene_bands).

    Returns:
        class_idx: (H, W) int32, values 0..len(CLASS_CODES)-1
    """
    item = search_worldcover(bbox, version=version)
    href = item.assets["map"].href

    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds, transform as win_transform

    with rasterio.open(href) as src:
        west, south, east, north = bbox
        if src.crs.to_epsg() != 4326:
            west, south, east, north = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
        window = from_bounds(west, south, east, north, transform=src.transform).round_lengths().round_offsets()
        win_tf = win_transform(window, src.transform)
        raw = src.read(1, window=window, out_dtype="float32", boundless=True, fill_value=0)
        src_crs = src.crs

    if raw.shape != ref_shape:
        out = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            raw, out,
            src_transform=win_tf, src_crs=src_crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.nearest,  # labels: never interpolate classes
        )
        raw = out

    codes = np.round(raw).astype(int)
    class_idx = np.zeros_like(codes, dtype=np.int32)
    for code, idx in CODE_TO_INDEX.items():
        class_idx[codes == code] = idx
    return class_idx


def labels_to_fractions(class_idx, coarsen_factor):
    """
    Aggregate a fine-grid class map into per-class abundance fractions at a
    coarser grid -- the sub-pixel mapping ground truth (FOUR_DAY_PLAN.md 2B):
    each coarse cell's fraction vector sums to 1 and states, e.g., "60%
    cropland / 40% built-up" instead of a single hard label.

    class_idx: (H, W) int32, H and W divisible by coarsen_factor
    Returns: (n_classes, H//f, W//f) float32 fraction map
    """
    n_classes = len(CLASS_CODES)
    h, w = class_idx.shape
    assert h % coarsen_factor == 0 and w % coarsen_factor == 0, \
        f"shape {class_idx.shape} not divisible by {coarsen_factor}"
    oh, ow = h // coarsen_factor, w // coarsen_factor

    fractions = np.zeros((n_classes, oh, ow), dtype=np.float32)
    for c in range(n_classes):
        mask = (class_idx == c).astype(np.float32)
        blocks = mask.reshape(oh, coarsen_factor, ow, coarsen_factor)
        fractions[c] = blocks.mean(axis=(1, 3))
    return fractions


if __name__ == "__main__":
    from data.mpc_fetch import search_scenes, fetch_scene_bands

    bbox = (75.75, 30.55, 75.80, 30.60)  # Punjab wheat belt smoke test
    items = search_scenes(bbox, "2023-11-01/2023-12-15", cloud_cover_max=10, limit=1)
    stack, scl, ref_transform, ref_crs = fetch_scene_bands(items[0], bbox)
    ref_shape = stack.shape[1:]

    labels = fetch_worldcover_labels(bbox, ref_shape, ref_transform, ref_crs)
    print(f"S2 stack: {stack.shape}, WorldCover labels: {labels.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {CLASS_NAMES[u]:20s} {c:6d} px ({100*c/labels.size:.1f}%)")

    h, w = labels.shape
    cropped = labels[:h - h % 4, :w - w % 4]
    fractions = labels_to_fractions(cropped, coarsen_factor=4)
    print(f"Fraction map (coarsen=4): {fractions.shape}, sums to 1: "
          f"{np.allclose(fractions.sum(axis=0), 1.0)}")
