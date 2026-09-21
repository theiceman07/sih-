"""
Real Indian ground-truth dataset: SEN2VENuS Kudaliar site (Telangana),
20 real same-day Sentinel-2/VENuS acquisition pairs across 2 tiles
(44QKE, 44QKF). See FOUR_DAY_PLAN.md section 0.2 / 1A.

Unlike Track B (data/track_b_dataset.py, which SIMULATES a 20m input by
degrading real 10m pixels) or the SRM dataset (which uses MTF-simulated
degradation of live-fetched chips), this is genuinely measured ground
truth: VENuS acquired the SAME scene on the SAME day at 5m, independently
of Sentinel-2's own 10m/20m measurement. This is the one place in the
whole project with zero synthetic-degradation assumption baked into the
label -- see FOUR_DAY_PLAN.md section 0.2 for why that matters (it is
what makes "India-validated" a measured claim rather than an aspiration).

Two real pairs exist per acquisition:
  10m -> 5m (x2): Sentinel-2's native 10m bands (B02,B03,B04,B08) vs their
                  true VENuS 5m counterpart
  20m -> 5m (x4): Sentinel-2's 20m-native bands vs their true VENuS 5m
                  counterpart (note the x4 ratio here matches the eventual
                  10m->2.5m product target, on REAL data, at a coarser
                  starting resolution -- see FOUR_DAY_PLAN.md section 1A
                  about the x2-real-GT vs x4-deployed distinction)

Held out by ACQUISITION DATE, not by patch -- patches from the same date
share spatial context (adjacent tiles of one scene), so splitting by patch
index would leak information between train and test.
"""
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data_raw" / "kudaliar" / "KUDALIAR"

_FNAME_RE = re.compile(r"KUDALIAR_(?P<tile>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<res>05m|10m|20m)_(?P<bands>\w+)\.pt")


def _discover_acquisitions(root):
    """Group the flat .pt file list into {(tile, date): {res_bands: path}}."""
    acquisitions = {}
    for p in Path(root).glob("*.pt"):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        key = (m.group("tile"), m.group("date"))
        acquisitions.setdefault(key, {})[f"{m.group('res')}_{m.group('bands')}"] = p
    return acquisitions


class KudaliarDataset(Dataset):
    """
    Yields (lr, hr) patch pairs, real measured reflectance, normalized to
    roughly [0, 1] (DN / 10000, matching this project's convention
    elsewhere -- see data/mpc_fetch.py's dn_scale).

    ratio="10m_5m": lr (4, 128, 128) native-10m bands, hr (4, 256, 256), scale=2
    ratio="20m_5m": lr (4, 64, 64) native-20m bands,  hr (4, 256, 256), scale=4
    """

    RATIO_KEYS = {
        "10m_5m": ("10m_b2b3b4b8", "05m_b2b3b4b8", 2),
        "20m_5m": ("20m_b4b5b6b8a", "05m_b4b5b6b8a", 4),
    }

    def __init__(self, root=DEFAULT_ROOT, ratio="10m_5m", dates=None, dn_scale=10000.0,
                 max_patches_per_acquisition=32, seed=0, lr_crop_size=None):
        """
        max_patches_per_acquisition bounds memory: this machine has very
        little free RAM (observed ~70MB free of ~8GB total), and each
        acquisition's full 05m tensor is ~350MB (343 patches x 4 bands x
        256x256 x int16) -- loading all ~20 dates x 2 tiles eagerly caused
        an out-of-memory segfault (torch.load's C-level allocator crashes
        rather than raising a catchable Python MemoryError). Loading each
        file fully, slicing a fixed random subset of patches, then dropping
        the rest keeps peak memory to one file at a time regardless of how
        many acquisitions are requested.

        lr_crop_size: if set, center-crops each LR patch (and the matching
        HR region) down to this size -- Kudaliar's native 128x128 LR patches
        make CPU training/local iteration very slow (16x more attention
        windows than the 32x32 chips used in train.py's own smoke tests);
        cropping to e.g. 32 keeps the real-GT data but at a fast, correctness
        -check-sized resolution.
        """
        assert ratio in self.RATIO_KEYS, f"ratio must be one of {list(self.RATIO_KEYS)}"
        self.scale = self.RATIO_KEYS[ratio][2]
        self.dn_scale = dn_scale
        self.lr_crop_size = lr_crop_size

        acquisitions = _discover_acquisitions(root)
        lr_key, hr_key = self.RATIO_KEYS[ratio][0], self.RATIO_KEYS[ratio][1]
        rng = torch.Generator().manual_seed(seed)

        lr_patches, hr_patches = [], []
        for (tile, date), files in sorted(acquisitions.items()):
            if dates is not None and date not in dates:
                continue
            if lr_key not in files or hr_key not in files:
                continue

            lr_all = torch.load(files[lr_key], weights_only=False)  # (N, 4, h, w)
            n = lr_all.shape[0]
            k = min(max_patches_per_acquisition, n)
            idx = torch.randperm(n, generator=rng)[:k]
            lr_patches.append(lr_all[idx].clone())
            del lr_all  # drop the full (unsampled) tensor before loading hr

            hr_all = torch.load(files[hr_key], weights_only=False)  # (N, 4, h*scale, w*scale)
            hr_patches.append(hr_all[idx].clone())
            del hr_all

        self._lr = torch.cat(lr_patches, dim=0) if lr_patches else torch.empty(0, 4, 1, 1)
        self._hr = torch.cat(hr_patches, dim=0) if hr_patches else torch.empty(0, 4, 1, 1)

    @staticmethod
    def available_dates(root=DEFAULT_ROOT):
        return sorted({date for (_tile, date) in _discover_acquisitions(root).keys()})

    def __len__(self):
        return self._lr.shape[0]

    def __getitem__(self, idx):
        lr = (self._lr[idx].float() / self.dn_scale).clamp(0.0, 1.0)
        hr = (self._hr[idx].float() / self.dn_scale).clamp(0.0, 1.0)

        if self.lr_crop_size is not None and self.lr_crop_size < lr.shape[-1]:
            lh, lw = lr.shape[-2:]
            cs = self.lr_crop_size
            y0 = (lh - cs) // 2
            x0 = (lw - cs) // 2
            lr = lr[:, y0:y0 + cs, x0:x0 + cs]
            hy0, hx0 = y0 * self.scale, x0 * self.scale
            hcs = cs * self.scale
            hr = hr[:, hy0:hy0 + hcs, hx0:hx0 + hcs]

        return lr, hr


def train_test_split_by_date(root=DEFAULT_ROOT, ratio="10m_5m", n_test_dates=2, dn_scale=10000.0,
                              max_patches_per_acquisition=32, lr_crop_size=None):
    """
    Splits by acquisition DATE (not patch) so no spatial leakage between
    train/test -- see module docstring. The last n_test_dates (by string
    sort, which sorts chronologically for YYYY-MM-DD) become the test set.
    """
    all_dates = KudaliarDataset.available_dates(root)
    assert len(all_dates) > n_test_dates, f"only {len(all_dates)} dates available"
    train_dates = all_dates[:-n_test_dates]
    test_dates = all_dates[-n_test_dates:]

    train_ds = KudaliarDataset(root=root, ratio=ratio, dates=set(train_dates), dn_scale=dn_scale,
                                max_patches_per_acquisition=max_patches_per_acquisition,
                                lr_crop_size=lr_crop_size)
    test_ds = KudaliarDataset(root=root, ratio=ratio, dates=set(test_dates), dn_scale=dn_scale,
                               max_patches_per_acquisition=max_patches_per_acquisition,
                               lr_crop_size=lr_crop_size)
    return train_ds, test_ds, train_dates, test_dates


if __name__ == "__main__":
    dates = KudaliarDataset.available_dates()
    print(f"Found {len(dates)} acquisition dates: {dates}")

    for ratio in ("10m_5m", "20m_5m"):
        train_ds, test_ds, train_dates, test_dates = train_test_split_by_date(
            ratio=ratio, n_test_dates=2, max_patches_per_acquisition=8)
        print(f"\nratio={ratio}: train={len(train_ds)} patches ({len(train_dates)} dates), "
              f"test={len(test_ds)} patches ({test_dates})")
        lr, hr = train_ds[0]
        print(f"  lr {tuple(lr.shape)} range=[{lr.min():.3f},{lr.max():.3f}], "
              f"hr {tuple(hr.shape)} range=[{hr.min():.3f},{hr.max():.3f}]")
        assert hr.shape[-1] == lr.shape[-1] * train_ds.scale

    print("\nKudaliar dataset smoke test passed.")
