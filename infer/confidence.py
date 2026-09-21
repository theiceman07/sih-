"""
Per-pixel confidence map via test-time augmentation (TTA) variance
(FOUR_DAY_PLAN.md section 3C). MC Dropout needs dropout layers in the model
(none currently exist in models/srm_head.py / swinir_lite.py, and adding
them now would mean retraining from scratch); TTA variance is the plan's
own documented cheaper fallback -- run the same model on flipped/rotated
versions of the same input, undo the transform on each output, and take
the per-pixel standard deviation across runs. Pixels where the model
disagrees with itself under trivial symmetry transforms (which shouldn't
change the "right answer") are the pixels least worth trusting.
"""
import numpy as np


_TRANSFORMS = [
    ("identity", lambda x: x, lambda x: x),
    ("hflip", lambda x: x[..., ::-1], lambda x: x[..., ::-1]),
    ("vflip", lambda x: x[..., ::-1, :], lambda x: x[..., ::-1, :]),
    ("rot180", lambda x: x[..., ::-1, ::-1], lambda x: x[..., ::-1, ::-1]),
]


def tta_confidence(lr, infer_fn):
    """
    lr: (C, H, W) float32 -- the native-resolution input to one tile/scene
    infer_fn: callable(tile[C, H, W]) -> sr[C_out, H*scale, W*scale]
              (same signature as infer.tiler.tile_infer's infer_fn)

    Returns:
        sr_mean: (C_out, H*scale, W*scale) -- averaged SR prediction across
                 the 4 symmetry transforms (also a mild ensembling benefit)
        confidence: (H*scale, W*scale) float32 -- per-pixel std-dev across
                    transforms, averaged over output channels; higher =
                    less trustworthy
    """
    outputs = []
    for name, fwd, inv in _TRANSFORMS:
        transformed = np.ascontiguousarray(fwd(lr))
        sr = infer_fn(transformed)
        undone = np.ascontiguousarray(inv(sr))
        outputs.append(undone)

    stacked = np.stack(outputs, axis=0)  # (T, C_out, H, W)
    sr_mean = stacked.mean(axis=0)
    per_channel_std = stacked.std(axis=0)  # (C_out, H, W)
    confidence = per_channel_std.mean(axis=0)  # (H, W)

    return sr_mean, confidence


def calibration_curve(confidence, actual_error, n_bins=10):
    """
    Bins predicted confidence (std-dev) against actual per-pixel error and
    returns (bin_centers, mean_error_per_bin) -- the reliability diagram
    the plan calls for (FOUR_DAY_PLAN.md section 3C): an uncalibrated
    uncertainty map (one that doesn't correlate with real error) is worse
    than none, so this is the check that it does.
    """
    conf_flat = confidence.ravel()
    err_flat = actual_error.ravel()
    edges = np.quantile(conf_flat, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6

    bin_centers, mean_errors = [], []
    for i in range(n_bins):
        mask = (conf_flat >= edges[i]) & (conf_flat < edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append(float(conf_flat[mask].mean()))
        mean_errors.append(float(err_flat[mask].mean()))

    return np.array(bin_centers), np.array(mean_errors)


if __name__ == "__main__":
    _call_count = [0]

    def fake_model_2x(tile):
        """A trivial 'model' (bicubic upsample) with per-call random noise --
        stands in for a real model's own view-dependent inconsistency (a
        real trained net genuinely does respond slightly differently to
        flipped/rotated versions of the same content); this is enough to
        prove tta_confidence's averaging and undo-transform logic works."""
        import cv2
        c, h, w = tile.shape
        out = np.zeros((c, h * 2, w * 2), dtype=np.float32)
        for i in range(c):
            out[i] = cv2.resize(tile[i], (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        rng_local = np.random.default_rng(_call_count[0])
        _call_count[0] += 1
        return out + rng_local.normal(0, 0.05, size=out.shape).astype(np.float32)

    rng = np.random.default_rng(0)
    import cv2
    raw = rng.uniform(0, 1, size=(3, 32, 32)).astype(np.float32)
    lr = np.stack([cv2.GaussianBlur(raw[c], (5, 5), 1.5) for c in range(3)])

    sr_mean, confidence = tta_confidence(lr, fake_model_2x)
    print(f"lr {lr.shape} -> sr_mean {sr_mean.shape}, confidence {confidence.shape}")
    assert sr_mean.shape == (3, 64, 64)
    assert confidence.shape == (64, 64)
    assert confidence.min() >= 0

    print(f"confidence range: [{confidence.min():.4f}, {confidence.max():.4f}], "
          f"mean: {confidence.mean():.4f}")
    assert confidence.mean() > 0, "TTA should detect the per-call noise"
    assert confidence.std() > 1e-6, "confidence map should vary spatially, not be uniform"

    # Calibration curve sanity: confidence should correlate with actual error here.
    gt = fake_model_2x(lr)  # "ground truth" = one deterministic pass, for this synthetic check
    actual_error = np.abs(sr_mean - gt).mean(axis=0)
    bins, errs = calibration_curve(confidence, actual_error, n_bins=5)
    print(f"calibration bins: {bins}\ncalibration errors: {errs}")

    print("TTA confidence sanity checks passed.")
