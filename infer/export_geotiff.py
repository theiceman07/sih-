"""
Cloud-Optimized GeoTIFF export (FOUR_DAY_PLAN.md section 3A).

Writes a super-resolved array back out with correct georeferencing: the
input CRS is preserved, the affine transform is rescaled by the SR factor
(pixel size shrinks, origin stays anchored to the same real-world corner),
and band descriptions are set so the file is self-documenting when opened
directly in QGIS/ArcGIS -- no separate metadata sidecar needed.
"""
import numpy as np
import rasterio
from rasterio.transform import Affine


def rescale_transform(transform, scale):
    """
    Scale a rasterio Affine transform by `scale` (pixel size shrinks by
    1/scale), keeping the same top-left corner anchored in real-world
    coordinates -- this is what "same place, finer pixels" means for a
    GeoTIFF's geotransform.
    """
    return Affine(transform.a / scale, transform.b, transform.c,
                   transform.d, transform.e / scale, transform.f)


def write_cog(path, array, transform, crs, band_names=None, nodata=None, dtype="float32"):
    """
    array: (C, H, W) -- already at the SR (fine) resolution
    transform: the already-rescaled (fine-resolution) rasterio Affine transform
    crs: rasterio/pyproj CRS (or anything rasterio.open accepts as crs=)
    band_names: optional list of length C, written as each band's description
    """
    c, h, w = array.shape
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": c,
        "height": h,
        "width": w,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2 if dtype != "float32" else 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        # COG-style overview-friendly layout; a true .tif COG additionally
        # wants an internal overview pyramid, generated separately by
        # `gdaladdo` (or GDAL's COG driver, if the build supports it) once
        # the tiled+compressed base file exists here.
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype))
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)


def export_sr_geotiff(path, sr_array, orig_transform, orig_crs, scale, band_names=None):
    """Convenience wrapper: rescales the ORIGINAL (native-resolution) transform by `scale` and writes."""
    fine_transform = rescale_transform(orig_transform, scale)
    write_cog(path, sr_array, fine_transform, orig_crs, band_names=band_names)


def export_landcover_geotiff(path, class_idx, orig_transform, orig_crs, scale, class_names):
    """
    Writes a single-band land-cover class map with a GDAL colour table, so
    it renders with a legend directly in QGIS without a manual style file.
    class_idx: (H, W) int, values 0..len(class_names)-1
    """
    fine_transform = rescale_transform(orig_transform, scale)
    h, w = class_idx.shape

    # A fixed, readable palette (matches ESA WorldCover's own official
    # colours where the class names line up, for a viewer's muscle memory).
    palette_rgb = [
        (0, 100, 0), (255, 187, 34), (255, 255, 76), (240, 150, 255),
        (250, 0, 0), (180, 180, 180), (240, 240, 240), (0, 100, 200),
        (0, 150, 160), (0, 207, 117), (250, 230, 160),
    ]

    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "height": h, "width": w, "crs": orig_crs, "transform": fine_transform,
        "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(class_idx.astype(np.uint8), 1)
        colormap = {i: (*palette_rgb[i % len(palette_rgb)], 255) for i in range(len(class_names))}
        dst.write_colormap(1, colormap)
        dst.set_band_description(1, "land_cover_class")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from rasterio.crs import CRS

    orig_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3400000.0)
    orig_crs = CRS.from_epsg(32643)  # UTM 43N, plausible for Punjab

    sr = np.random.rand(4, 64, 64).astype(np.float32)
    class_idx = np.random.randint(0, 11, size=(64, 64)).astype(np.int32)

    with tempfile.TemporaryDirectory() as tmp:
        sr_path = Path(tmp) / "sr_test.tif"
        lc_path = Path(tmp) / "landcover_test.tif"

        export_sr_geotiff(str(sr_path), sr, orig_transform, orig_crs, scale=4,
                           band_names=["B02", "B03", "B04", "B08"])
        export_landcover_geotiff(str(lc_path), class_idx, orig_transform, orig_crs,
                                  scale=4, class_names=[f"class_{i}" for i in range(11)])

        with rasterio.open(sr_path) as src:
            print(f"SR GeoTIFF: {src.count} bands, {src.shape}, transform={src.transform}, "
                  f"pixel size={src.transform.a}m (expected {orig_transform.a/4}m)")
            assert abs(src.transform.a - orig_transform.a / 4) < 1e-9
            assert src.descriptions == ("B02", "B03", "B04", "B08")

        with rasterio.open(lc_path) as src:
            print(f"Land-cover GeoTIFF: {src.count} band, {src.shape}, "
                  f"has colormap: {src.colormap(1) is not None}")
            assert src.colormap(1) is not None

    print("GeoTIFF export sanity checks passed.")
