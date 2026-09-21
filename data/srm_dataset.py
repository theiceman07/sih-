"""
SRM training dataset: degraded Sentinel-2 -> real WorldCover land-cover
fractions. See FOUR_DAY_PLAN.md section 2B.

Supervision, and why it's stronger than image-only Wald degradation:
  input:  Sentinel-2 10-band stack, MTF-degraded 10m -> 40m (same physically
          -motivated blur+decimate as Track B's mtf_degrade, reused here)
  label:  ESA WorldCover class fractions at the *real* 10m grid (an
          independent measurement, not a downsampled copy of the S2 input)

Train 40m -> 10m land-cover fractions; the same learned x4 operator is then
applied at 10m -> 2.5m at inference (no ground truth exists at 2.5m, by
construction -- that's the whole reason Track B proves the SR/SAM approach
on a real-GT ratio first before trusting the extrapolated scale here too).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from data.mpc_fetch import search_scenes, fetch_scene_bands, stack_to_chips
from data.worldcover_fetch import fetch_worldcover_labels, labels_to_fractions, CLASS_CODES


def mtf_degrade_stack(stack, scale):
    """Per-band MTF-matched degrade (see data/track_b_dataset.py:mtf_degrade)."""
    sigma = scale / 2.2
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    c, h, w = stack.shape
    out = np.zeros((c, h // scale, w // scale), dtype=np.float32)
    for i in range(c):
        blurred = cv2.GaussianBlur(stack[i], (ksize, ksize), sigma)
        out[i] = blurred[::scale, ::scale]
    return out


class SRMDataset(Dataset):
    """
    Yields (lr, hr, fractions) triples:
      lr:        (10, h, w)              -- S2 bands degraded to "40m" (real pixels, blurred+decimated)
      hr:        (10, h*scale, w*scale)  -- the real, un-degraded 10m chip -- the SR head's target.
                 Without this, only the SRM (land-cover) head has a training
                 signal; the SR head's weights never receive gradient at all
                 (a real bug found via a failed live API demo -- see
                 train/train_srm.py's loss computation, which now uses this).
      fractions: (n_classes, h*scale, w*scale) -- real WorldCover class fractions at the fine (10m) grid
    """

    def __init__(self, chips_and_labels, scale=4):
        """chips_and_labels: list of (s2_chip[10,H,W], class_idx[H,W]) real pairs."""
        self.scale = scale
        self.pairs = []
        for chip, class_idx in chips_and_labels:
            h, w = class_idx.shape
            ch, cw = h - h % scale, w - w % scale
            self.pairs.append((chip[:, :ch, :cw], class_idx[:ch, :cw]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        chip, class_idx = self.pairs[idx]
        lr = mtf_degrade_stack(chip, self.scale)
        fractions = labels_to_fractions(class_idx, coarsen_factor=1)  # per-pixel one-hot at full res
        return torch.from_numpy(lr).float(), torch.from_numpy(chip).float(), torch.from_numpy(fractions).float()


def build_dataset_from_mpc(bboxes, date_range, scale=4, chip_size=64, cloud_cover_max=10):
    """Fetch real Sentinel-2 chips + real WorldCover labels, wrap as SRMDataset."""
    chips_and_labels = []
    for bbox in bboxes:
        items = search_scenes(bbox, date_range, cloud_cover_max=cloud_cover_max, limit=1)
        if not items:
            continue
        stack, scl, ref_transform, ref_crs = fetch_scene_bands(items[0], bbox)
        ref_shape = stack.shape[1:]
        labels = fetch_worldcover_labels(bbox, ref_shape, ref_transform, ref_crs)

        norm_stack = np.clip(stack / 10000.0, 0.0, 1.0).astype(np.float32)

        h, w = ref_shape
        for y in range(0, h - chip_size + 1, chip_size):
            for x in range(0, w - chip_size + 1, chip_size):
                scl_chip = scl[y:y + chip_size, x:x + chip_size]
                if scl_chip.mean() < 0.95:
                    continue
                s2_chip = norm_stack[:, y:y + chip_size, x:x + chip_size]
                label_chip = labels[y:y + chip_size, x:x + chip_size]
                chips_and_labels.append((s2_chip, label_chip))

    return SRMDataset(chips_and_labels, scale=scale)


if __name__ == "__main__":
    bboxes = [(75.75, 30.55, 75.80, 30.60)]
    ds = build_dataset_from_mpc(bboxes, "2023-11-01/2023-12-15", scale=4, chip_size=64)
    print(f"Dataset size: {len(ds)}")
    lr, fractions = ds[0]
    print(f"lr: {tuple(lr.shape)}, fractions: {tuple(fractions.shape)}")
    assert fractions.shape[-1] == lr.shape[-1] * 4
    assert fractions.shape[0] == len(CLASS_CODES)
    print(f"fractions sum to 1: {torch.allclose(fractions.sum(dim=0), torch.ones(fractions.shape[1:]), atol=1e-5)}")
    print("SRM dataset smoke test passed.")
