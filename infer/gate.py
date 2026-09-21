"""
Ground-truth-free release gate (FOUR_DAY_PLAN.md section 3B).

The actual differentiator in the plan: at 2.5m there is no ground truth,
anywhere, ever (that's the whole reason Track B exists at a smaller ratio
where real GT does exist -- see data/track_b_dataset.py). So the one thing
that CAN be checked at inference time, on any tile anywhere on Earth, is
consistency: re-degrade the SR output back down to the input's resolution
and confirm it still agrees with what the sensor actually measured. If it
doesn't, the tile is refused -- flagged and served as bicubic instead of a
model output nobody can verify.

This reuses the frozen eval/metrics.py functions (sam, compute_ndvi) so the
gate's spectral checks are numerically identical to the ones already used
throughout Phase 0 evaluation, and reuses mtf_degrade_stack (the same
physically-motivated blur+decimate used for Track A/B/SRM training data)
so the gate's outlook on "resembles the sensor" matches how the model was
actually trained -- a gate built on a different degradation than training
would fail tiles unfairly or pass tiles it shouldn't, on the mismatch alone.
"""
import numpy as np

from eval.metrics import sam, compute_ndvi
from data.srm_dataset import mtf_degrade_stack


class GateResult:
    def __init__(self, passed, reason, spectral_deg, reflectance_l1, ndvi_delta):
        self.passed = passed
        self.reason = reason
        self.spectral_deg = spectral_deg
        self.reflectance_l1 = reflectance_l1
        self.ndvi_delta = ndvi_delta

    def as_dict(self):
        return {
            "passed": self.passed,
            "reason": self.reason,
            "spectral_deg": self.spectral_deg,
            "reflectance_l1": self.reflectance_l1,
            "ndvi_delta": self.ndvi_delta,
        }

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return (f"GateResult({status}: SAM={self.spectral_deg:.2f}deg, "
                f"reflectance_L1={self.reflectance_l1:.4f}, "
                f"ndvi_delta={self.ndvi_delta:.4f}"
                + (f", reason={self.reason!r}" if not self.passed else "") + ")")


def release_gate(lr, sr, scale, sam_threshold_deg=5.0, reflectance_threshold=0.05,
                  ndvi_threshold=0.05, red_idx=2, nir_idx=6):
    """
    lr: (C, h, w) float32 in [0, 1] -- the real sensor input at native resolution
    sr: (C, h*scale, w*scale) float32 in [0, 1] -- the model's super-resolved output
    scale: the SR factor (must match lr/sr shape ratio)

    Re-degrades sr back to lr's resolution (same MTF degradation used in
    training) and checks three GT-free consistency conditions:
      spectral      -- SAM(sr_down, lr) < sam_threshold_deg
      reflectance   -- L1(sr_down, lr) < reflectance_threshold
      ndvi          -- |NDVI(sr_down) - NDVI(lr)| < ndvi_threshold  (if red/nir bands present)

    Returns a GateResult. FAIL on ANY condition -- ship bicubic instead,
    never a silently-wrong SR tile (see FOUR_DAY_PLAN.md section 3B).
    """
    assert sr.shape[-1] == lr.shape[-1] * scale, \
        f"sr/lr shape mismatch for scale={scale}: sr={sr.shape}, lr={lr.shape}"

    sr_down = mtf_degrade_stack(sr, scale)

    spectral_deg = float(sam(lr, sr_down))
    reflectance_l1 = float(np.mean(np.abs(lr - sr_down)))

    ndvi_delta = 0.0
    if lr.shape[0] > max(red_idx, nir_idx):
        ndvi_lr = compute_ndvi(lr[nir_idx], lr[red_idx])
        ndvi_sr = compute_ndvi(sr_down[nir_idx], sr_down[red_idx])
        ndvi_delta = float(np.mean(np.abs(ndvi_lr - ndvi_sr)))

    failures = []
    if spectral_deg >= sam_threshold_deg:
        failures.append(f"SAM {spectral_deg:.2f}deg >= {sam_threshold_deg}deg")
    if reflectance_l1 >= reflectance_threshold:
        failures.append(f"reflectance L1 {reflectance_l1:.4f} >= {reflectance_threshold}")
    if ndvi_delta >= ndvi_threshold:
        failures.append(f"NDVI delta {ndvi_delta:.4f} >= {ndvi_threshold}")

    passed = len(failures) == 0
    reason = "; ".join(failures) if failures else ""
    return GateResult(passed, reason, spectral_deg, reflectance_l1, ndvi_delta)


if __name__ == "__main__":
    import cv2

    rng = np.random.default_rng(0)

    # A "good" SR output: same content, mild upsampling noise -> should pass.
    # Real satellite imagery is spatially smooth (fields, not i.i.d. noise),
    # so the synthetic lr here is smoothed too -- an i.i.d.-noise lr would
    # make even a perfect re-degradation disagree with itself once blurred,
    # which is a synthetic-test artifact, not a gate bug.
    raw = rng.uniform(0.1, 0.6, size=(4, 16, 16)).astype(np.float32)
    lr = np.stack([cv2.GaussianBlur(raw[c], (5, 5), 1.5) for c in range(4)])
    sr_good = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)
    sr_good += rng.normal(0, 0.01, size=sr_good.shape).astype(np.float32)
    sr_good = np.clip(sr_good, 0, 1)

    result_good = release_gate(lr, sr_good, scale=4, red_idx=2, nir_idx=3)
    print("Good SR:", result_good)
    assert result_good.passed, "a faithful upsample should pass the gate"

    # A "hallucinated" SR output: spectrally scrambled -> should fail.
    sr_bad = rng.uniform(0.1, 0.6, size=(4, 64, 64)).astype(np.float32)
    result_bad = release_gate(lr, sr_bad, scale=4, red_idx=2, nir_idx=3)
    print("Bad SR:", result_bad)
    assert not result_bad.passed, "spectrally scrambled output should fail the gate"

    print("Gate sanity checks passed.")
