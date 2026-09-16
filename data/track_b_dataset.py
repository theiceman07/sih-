"""
Track B dataset: real ground-truth pairs for the 20m->10m super-resolution
model (Phase 1). See DEVELOPMENT_PLAN.md section 0.

The trick: Sentinel-2 already gives us 4 bands natively at 10m (B02, B03,
B04, B08). We degrade *those real 10m pixels* down to 20m (matching how the
genuinely-20m bands B05/B06/B07/B8A/B11/B12 are sampled), and train the
model to reconstruct the real 10m pixels from that 20m input. This is REAL
supervision -- the label is measured reflectance, not a synthetic guess --
unlike Track A's Wald protocol (10m -> 40m -> 10m, same synthetic-degradation
assumption end to end).

Once trained on this real pair, the model is applied at inference time to
the genuinely-20m bands (transfer/generalization -- there's no ground truth
there by construction, that's the whole reason we needed Track B first).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from data.mpc_fetch import search_scenes, fetch_scene_bands, stack_to_chips

# The 4 natively-10m bands, in stack order. Track B trains on these only.
NATIVE_10M_BANDS = ["B02", "B03", "B04", "B08"]
RED_IDX = 2   # B04
NIR_IDX = 3   # B08


def mtf_degrade(band_10m, scale=2):
    """
    Degrade a real 10m band to 20m, approximating Sentinel-2's own MTF
    (modulation transfer function) with a Gaussian blur matched to the
    sensor's approx. relative cutoff frequency before decimation, rather
    than a naive resize (which has no physical basis and is the #1 reason
    SR models trained on synthetic degradation fail to transfer to real
    sensor data -- see DEVELOPMENT_PLAN.md Phase 2 notes).

    band_10m: (H, W) float32
    Returns: (H//scale, W//scale) float32
    """
    # Sentinel-2's 10m bands have an MTF at Nyquist of roughly 0.2-0.3;
    # a Gaussian with sigma ~= scale/2.2 approximates the equivalent
    # pre-decimation blur (this constant is a reasonable literature-typical
    # value, not a measured per-mission calibration -- flag as an assumption
    # to validate in Phase 4 against real 20m bands, not just this synthetic
    # pair).
    sigma = scale / 2.2
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    blurred = cv2.GaussianBlur(band_10m, (ksize, ksize), sigma)
    return blurred[::scale, ::scale]


class TrackBDataset(Dataset):
    """
    Yields (lr, hr) pairs:
      lr: (4, h, w)   -- the 4 native bands degraded to "20m" (real pixels, blurred+decimated)
      hr: (4, 2h, 2w) -- the same 4 bands at their real, actually-measured 10m resolution
    """

    def __init__(self, chips, scale=2):
        """chips: list of (10, H, W) normalized [0,1] arrays from stack_to_chips (full 10-band stack)."""
        self.scale = scale
        # Native-10m bands are indices 0,1,2,6 in the project's standard
        # 10-band order: [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12].
        native_idx = [0, 1, 2, 6]
        self.hr_chips = [chip[native_idx] for chip in chips]

    def __len__(self):
        return len(self.hr_chips)

    def __getitem__(self, idx):
        hr = self.hr_chips[idx]  # (4, H, W)
        lr = np.stack([mtf_degrade(hr[b], self.scale) for b in range(hr.shape[0])], axis=0)
        return torch.from_numpy(lr).float(), torch.from_numpy(hr).float()


def build_dataset_from_mpc(bboxes, date_range, scale=2, chip_size=64, cloud_cover_max=10):
    """Fetch real Sentinel-2 chips from MPC and wrap them as a TrackBDataset."""
    all_chips = []
    for bbox in bboxes:
        items = search_scenes(bbox, date_range, cloud_cover_max=cloud_cover_max, limit=1)
        if not items:
            continue
        stack, scl, _, _ = fetch_scene_bands(items[0], bbox)
        chips = stack_to_chips(stack, scl, chip_size=chip_size, stride=chip_size, cloud_threshold=0.95)
        all_chips.extend(chips)
    return TrackBDataset(all_chips, scale=scale)


if __name__ == "__main__":
    bboxes = [(75.75, 30.55, 75.80, 30.60), (75.80, 30.60, 75.85, 30.65)]
    ds = build_dataset_from_mpc(bboxes, "2023-11-01/2023-12-15", scale=2, chip_size=64)
    print(f"Dataset size: {len(ds)}")
    lr, hr = ds[0]
    print(f"lr: {tuple(lr.shape)}, hr: {tuple(hr.shape)}")
    assert hr.shape[-1] == lr.shape[-1] * 2
    print("Track B dataset smoke test passed.")
