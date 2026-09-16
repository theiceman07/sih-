"""
Microsoft Planetary Computer Sentinel-2 L2A fetch — no login required.
Public STAC API + signed COG reads via the `planetary-computer` SAS signer.

This is the Phase 0 fast path (no CDSE account needed). Swap to
data/cdse_fetch.py once CDSE credentials are set up, if you want the
CDSE archive instead (same band set, same downstream code).
"""
import numpy as np
import rasterio
import pystac_client
import planetary_computer as pc
from rasterio.enums import Resampling
from rasterio.warp import reproject

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Bands we use throughout the project: 4x 10m + 6x 20m -> all resampled to 10m.
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
NATIVE_10M = {"B02", "B03", "B04", "B08"}


def search_scenes(bbox, date_range, cloud_cover_max=10, limit=5):
    """
    Query Sentinel-2 L2A over a bbox from the public MPC STAC catalog.

    bbox: (west, south, east, north)
    date_range: "YYYY-MM-DD/YYYY-MM-DD"
    Returns a list of pystac Items sorted by cloud cover (ascending).
    """
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        limit=limit,
    )
    items = list(search.item_collection())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items


def _window_for_bbox(src, bbox_geo):
    """Convert a lon/lat bbox to a pixel Window + its geotransform in src's CRS."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds, transform as win_transform

    west, south, east, north = bbox_geo
    if src.crs.to_epsg() != 4326:
        west, south, east, north = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
    window = from_bounds(west, south, east, north, transform=src.transform).round_lengths().round_offsets()
    transform = win_transform(window, src.transform)
    return window, transform


def _read_band(item, band, bbox_geo, ref_shape=None, ref_transform=None, ref_crs=None):
    """
    Read a small windowed crop of one signed band asset, cropped to bbox_geo
    (much faster than downloading + reprojecting the full ~110x110km tile).
    Reprojects/resamples onto the reference 10m grid if one is given (20m bands).
    """
    href = item.assets[band].href
    with rasterio.open(href) as src:
        window, win_tf = _window_for_bbox(src, bbox_geo)
        data = src.read(1, window=window, out_dtype="float32", boundless=True, fill_value=0)
        src_crs = src.crs

    if ref_shape is None:
        return data, win_tf, src_crs

    if data.shape == ref_shape:
        return data.astype(np.float32), ref_transform, ref_crs

    # Resample this band's window onto the reference (10m) grid.
    out = np.zeros(ref_shape, dtype=np.float32)
    reproject(
        data, out,
        src_transform=win_tf, src_crs=src_crs,
        dst_transform=ref_transform, dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )
    return out, ref_transform, ref_crs


def fetch_scene_bands(item, bbox_geo, bands=BANDS):
    """
    Read a small windowed crop (bbox_geo, lon/lat) of all bands for one STAC
    item, resampled onto the 10m grid. Downloads only the pixels needed via
    HTTP range requests -- seconds, not minutes, for a chip-sized area.

    Returns:
        stack: (len(bands), H, W) float32, DN units (0-10000ish, NOT yet normalized)
        scl:   (H, W) float32, 1.0 = clear (SCL class 4), 0.0 = cloud/shadow/other
        transform, crs: for georeferencing (needed later for GeoTIFF export)
    """
    # Establish the 10m reference grid from B04 (red, native 10m).
    ref_data, ref_transform, ref_crs = _read_band(item, "B04", bbox_geo)
    ref_shape = ref_data.shape

    data = {"B04": ref_data}
    for band in bands:
        if band == "B04":
            continue
        band_data, _, _ = _read_band(item, band, bbox_geo, ref_shape, ref_transform, ref_crs)
        data[band] = band_data

    stack = np.stack([data[b] for b in bands], axis=0)

    # Scene Classification Layer for cloud/shadow masking (20m native -> reproject to 10m).
    # SCL classes: 0=nodata 1=saturated 2=dark-area 3=cloud-shadow 4=vegetation
    # 5=bare-soil 6=water 7/8/9=cloud(low/med/high) 10=thin-cirrus 11=snow.
    # "Valid" = not cloud/shadow/nodata/saturated; vegetation is NOT the only
    # acceptable class (bare soil, water, dark areas are legitimate surface).
    CLOUD_OR_INVALID = {0, 1, 3, 7, 8, 9, 10}
    if "SCL" in item.assets:
        scl_raw, _, _ = _read_band(item, "SCL", bbox_geo, ref_shape, ref_transform, ref_crs)
        scl_class = np.round(scl_raw).astype(int)
        scl = np.isin(scl_class, list(CLOUD_OR_INVALID), invert=True).astype(np.float32)
    else:
        scl = np.ones(ref_shape, dtype=np.float32)

    return stack, scl, ref_transform, ref_crs


def stack_to_chips(stack, scl, chip_size=256, stride=256, cloud_threshold=0.95, dn_scale=10000.0):
    """
    Slice a (C, H, W) DN stack into normalized [0,1] float32 chips, dropping cloudy ones.
    """
    c, h, w = stack.shape
    norm = np.clip(stack / dn_scale, 0.0, 1.0)

    chips = []
    for y in range(0, h - chip_size + 1, stride):
        for x in range(0, w - chip_size + 1, stride):
            scl_chip = scl[y:y + chip_size, x:x + chip_size]
            if scl_chip.mean() < cloud_threshold:
                continue
            chips.append(norm[:, y:y + chip_size, x:x + chip_size])

    return chips


if __name__ == "__main__":
    # Quick smoke test: small ~5x5km window over Punjab wheat belt (dry season,
    # low cloud cover likely). Small bbox = fast windowed HTTP range reads.
    bbox = (75.75, 30.55, 75.80, 30.60)
    items = search_scenes(bbox, "2023-11-01/2023-12-15", cloud_cover_max=10, limit=5)
    print(f"Found {len(items)} scenes")
    if items:
        item = items[0]
        print(f"Using {item.id}, cloud_cover={item.properties.get('eo:cloud_cover'):.1f}%")
        stack, scl, transform, crs = fetch_scene_bands(item, bbox)
        print(f"Stack shape: {stack.shape}, valid fraction: {scl.mean():.2%}")
        chips = stack_to_chips(stack, scl, chip_size=64, stride=64)
        print(f"Extracted {len(chips)} clean 64x64 chips")
