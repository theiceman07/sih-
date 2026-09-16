"""
Spectral Angle Mapper (SAM) loss -- differentiable version of eval/metrics.py's
sam() function, for use as a training objective.

Kruse et al. 1993. Measures the angle between the predicted and reference
spectral vectors at each pixel, independent of magnitude -- this is what
penalizes a model for distorting relative band ratios (which is what breaks
NDVI/spectral indices) even when brightness/contrast look fine.
"""
import torch
import torch.nn as nn


class SAMLoss(nn.Module):
    """
    pred, target: (B, C, H, W), values in [0, 1] (or any consistent scale --
    SAM is scale-invariant per pixel, only the angle between band vectors
    matters).
    Returns mean spectral angle in radians (differentiable, matches
    eval/metrics.sam up to the degrees conversion -- keep this loss in
    radians for training, convert with `math.degrees` only when logging).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_flat = pred.flatten(2)      # B, C, HW
        target_flat = target.flatten(2)  # B, C, HW

        dot = (pred_flat * target_flat).sum(dim=1)                     # B, HW
        pred_norm = torch.sqrt((pred_flat ** 2).sum(dim=1) + self.eps)
        target_norm = torch.sqrt((target_flat ** 2).sum(dim=1) + self.eps)

        cos_angle = dot / (pred_norm * target_norm + self.eps)
        cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)  # acos domain safety
        angle = torch.acos(cos_angle)  # B, HW, radians

        return angle.mean()


if __name__ == "__main__":
    import math
    loss_fn = SAMLoss()
    a = torch.rand(2, 10, 16, 16)

    # Identical spectra -> zero angle (up to the acos-domain clamp's float32
    # floor, ~5e-4 rad from the 1-1e-7 clamp -- not exactly 0).
    assert loss_fn(a, a).item() < 1e-3, "SAM(x, x) should be ~0"

    # Scaled but same direction -> same (near-zero) angle: scale invariance.
    assert abs(loss_fn(a, a).item() - loss_fn(a, a * 2.0).item()) < 1e-6, \
        "SAM should be scale-invariant"

    # Orthogonal-ish spectra -> large angle.
    b = torch.rand(2, 10, 16, 16)
    angle = loss_fn(a, b)
    print(f"SAM(random, random) = {angle.item():.4f} rad = {math.degrees(angle.item()):.2f} deg")
    print("Sanity checks passed.")
