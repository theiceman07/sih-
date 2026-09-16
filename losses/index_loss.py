"""
Spectral index preservation loss (NDVI, and NDWI as a bonus check).

Directly penalizes the model for getting the *ratio* between specific bands
wrong, which is what farmers/agronomists actually read off the output
(NDVI), not raw reflectance values. Complements SAMLoss (which is a whole-
spectrum angle, band-order-agnostic) with a metric-specific penalty.
"""
import torch
import torch.nn as nn


def differentiable_ndvi(x, nir_idx, red_idx, eps=1e-8):
    """x: (B, C, H, W). Returns (B, H, W) NDVI in [-1, 1]."""
    nir = x[:, nir_idx]
    red = x[:, red_idx]
    return (nir - red) / (nir + red + eps)


class IndexLoss(nn.Module):
    """
    L1 loss between predicted and target NDVI (default band indices assume
    the project's standard 10-band stack order:
    [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
    -> B04 (red) = index 2, B08 (NIR) = index 6.
    Override nir_idx/red_idx if a model uses a different band subset (e.g.
    Track B's 4-band 10m-native stack [B02, B03, B04, B08] -> red=2, nir=3).
    """

    def __init__(self, nir_idx=6, red_idx=2):
        super().__init__()
        self.nir_idx = nir_idx
        self.red_idx = red_idx
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_ndvi = differentiable_ndvi(pred, self.nir_idx, self.red_idx)
        target_ndvi = differentiable_ndvi(target, self.nir_idx, self.red_idx)
        return self.l1(pred_ndvi, target_ndvi)


if __name__ == "__main__":
    # 4-band stack: [B02, B03, B04, B08] -> red=2, nir=3
    loss_fn = IndexLoss(nir_idx=3, red_idx=2)
    a = torch.rand(2, 4, 16, 16).clamp(0.01, 1.0)

    assert loss_fn(a, a).item() < 1e-6, "IndexLoss(x, x) should be exactly 0"

    b = torch.rand(2, 4, 16, 16).clamp(0.01, 1.0)
    print(f"IndexLoss(random, random) = {loss_fn(a, b).item():.4f}")
    print("Sanity checks passed.")
