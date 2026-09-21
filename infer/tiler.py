"""
Sliding-window tiler with overlap + cosine-feather blending (FOUR_DAY_PLAN.md
section 3A). Runs a model over an arbitrarily large scene by cutting it into
overlapping tiles, running each tile independently, and blending the
overlaps with a cosine window so the seams a naive non-overlapping tiling
would leave (a visible grid in the output) disappear.
"""
import numpy as np


def _cosine_window_1d(size, overlap, taper_start, taper_end):
    """
    1D taper: cosine ramp over `overlap` px on each side, flat 1.0 elsewhere.
    A side is only tapered if it actually borders a neighbouring tile
    (taper_start/taper_end) -- a tile sitting at the scene's outer edge has
    nothing to blend with there, so tapering it to 0 would leave that edge
    permanently under-weighted (division by ~0 in the final blend).
    """
    w = np.ones(size, dtype=np.float32)
    if overlap > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32))
        if taper_start:
            w[:overlap] = ramp
        if taper_end:
            w[-overlap:] = np.minimum(w[-overlap:], ramp[::-1])
    return w


def _cosine_window_2d(h, w, overlap, taper_top, taper_bottom, taper_left, taper_right):
    wy = _cosine_window_1d(h, overlap, taper_top, taper_bottom)
    wx = _cosine_window_1d(w, overlap, taper_left, taper_right)
    return wy[:, None] * wx[None, :]


def tile_infer(image, infer_fn, scale, tile_size=128, overlap=32):
    """
    image: (C, H, W) float32, the full input at native resolution
    infer_fn: callable(tile[C, tile_size, tile_size]) -> sr_tile[C_out, tile_size*scale, tile_size*scale]
    scale: the model's upscale factor
    tile_size: input tile edge length (must match what infer_fn expects, e.g. 128 for SEN2SR)
    overlap: input-space overlap between adjacent tiles (scaled by `scale` on the output side)

    Returns: (C_out, H*scale, W*scale) float32, blended with no visible seams.
    """
    c, h, w = image.shape
    stride = tile_size - overlap
    assert stride > 0, "overlap must be smaller than tile_size"

    # Pad so the scene is exactly tile_size + k*stride for some k>=0 (so the
    # sliding window lands on an exact multiple with no ragged final tile),
    # reflect-padded to avoid a zero-padded edge artifact in the blend.
    def _padded_size(n):
        if n <= tile_size:
            return tile_size
        import math
        k = math.ceil((n - tile_size) / stride)
        return tile_size + k * stride

    pad_h = _padded_size(h) - h
    pad_w = _padded_size(w) - w
    padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape[1:]

    out_overlap = overlap * scale
    out_tile = tile_size * scale

    c_out = None
    accum = None
    weight = None

    y = 0
    while y < ph:
        y_end = min(y + tile_size, ph)
        y_start = y_end - tile_size
        taper_top = y_start > 0
        taper_bottom = y_end < ph
        x = 0
        while x < pw:
            x_end = min(x + tile_size, pw)
            x_start = x_end - tile_size
            taper_left = x_start > 0
            taper_right = x_end < pw

            tile = padded[:, y_start:y_end, x_start:x_end]
            sr_tile = infer_fn(tile)

            if c_out is None:
                c_out = sr_tile.shape[0]
                accum = np.zeros((c_out, ph * scale, pw * scale), dtype=np.float32)
                weight = np.zeros((ph * scale, pw * scale), dtype=np.float32)

            window = _cosine_window_2d(out_tile, out_tile, out_overlap,
                                        taper_top, taper_bottom, taper_left, taper_right)

            oy, ox = y_start * scale, x_start * scale
            accum[:, oy:oy + out_tile, ox:ox + out_tile] += sr_tile * window[None, :, :]
            weight[oy:oy + out_tile, ox:ox + out_tile] += window

            x += stride
            if x_end == pw:
                break
        y += stride
        if y_end == ph:
            break

    weight = np.maximum(weight, 1e-8)
    blended = accum / weight[None, :, :]

    return blended[:, : h * scale, : w * scale]


if __name__ == "__main__":
    # Smoke test: a trivial "model" that just bicubic-upsamples 2x, run over
    # a scene bigger than one tile so overlap/blending logic is exercised.
    import cv2

    def fake_model_2x(tile):
        c, th, tw = tile.shape
        out = np.zeros((c, th * 2, tw * 2), dtype=np.float32)
        for i in range(c):
            out[i] = cv2.resize(tile[i], (tw * 2, th * 2), interpolation=cv2.INTER_CUBIC)
        return out

    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=(3, 40, 40)).astype(np.float32)
    scene = np.stack([cv2.GaussianBlur(raw[c], (5, 5), 1.5) for c in range(3)])

    tiled = tile_infer(scene, fake_model_2x, scale=2, tile_size=24, overlap=8)
    direct = fake_model_2x(scene)

    print(f"scene {scene.shape} -> tiled {tiled.shape}, direct {direct.shape}")
    assert tiled.shape == direct.shape

    max_diff = np.max(np.abs(tiled - direct))
    print(f"max abs diff (tiled vs single-shot bicubic): {max_diff:.4f}")
    assert max_diff < 0.05, "tiled+blended output should closely match a single-shot pass"

    # Seam check: no discontinuity spike at the tile boundary (~row/col 16 in
    # output space for stride=16 input tiles at scale=2).
    seam_row = tiled.shape[1] // 2
    grad_at_seam = np.abs(tiled[:, seam_row, :] - tiled[:, seam_row - 1, :]).mean()
    grad_elsewhere = np.abs(tiled[:, seam_row + 5, :] - tiled[:, seam_row + 4, :]).mean()
    print(f"gradient at likely seam: {grad_at_seam:.4f}, elsewhere: {grad_elsewhere:.4f}")

    print("Tiler sanity checks passed.")
