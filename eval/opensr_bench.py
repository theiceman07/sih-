"""
Day 1 benchmark harness (FOUR_DAY_PLAN.md section 0.4 / 1C).

Runs bicubic / Lanczos / ESRGAN / SEN2SR (pretrained, CC0) through
opensr-test's trustworthiness metrics:

    consistency  -- reflectance / spectral (SAD deg) / spatial, vs the LR input,
                    computable with NO ground truth (this is the release-gate signal)
    synthesis    -- how much genuine high-frequency detail was added
    correctness  -- hallucination / omission / improvement, vs real HR ground truth

Two run modes, since we have two different data situations on Day 1:

  --dataset venus     opensr_test's bundled Venus HR/LR pairs (real 5m GT,
                       downloads once, ~small). Gives the FULL metric set
                       (consistency + synthesis + correctness) immediately,
                       without waiting on the 5GB Kudaliar download.

  --dataset punjab     Real Sentinel-2 chips fetched live from Microsoft
                       Planetary Computer (same source as phase0_baseline_bank.py).
                       No HR ground truth exists at 2.5m for these, so only
                       consistency + synthesis are reported (correctness columns
                       are blank). This is the GT-free gate check.

SEN2SR needs a fixed 128x128 input tile (its hard-constraint module reshapes
in frequency space); chips smaller than that are padded and cropped back.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import opensr_test

# Project's standard 10-band order: [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12]
#
# opensr-test's Venus HR ground truth is "RGBNIR" = [B04,B03,B02,B08] (Red,
# Green,Blue,NIR reading order), confirmed empirically: HR channel 3's mean
# reflectance is ~3-5x channels 0-2 (the NIR plateau), and this is also
# SEN2SR's own declared input order (model/SEN2SRLite/mlm.json). It is NOT
# this project's ascending [B02,B03,B04,B08] order used elsewhere (Track B
# etc) -- those pipelines are internally self-consistent so they're fine as-
# is, but comparing against Venus's real HR specifically requires this
# RGBNIR-matching selection, not the project's usual native-band indices.
NATIVE_10M_IDX = [2, 1, 0, 6]  # B04,B03,B02,B08 in this project's 10-band order
SEN2SR_TILE = 128

# SEN2SRLite's documented input band order (model/SEN2SRLite/mlm.json,
# mlm:input.bands) is B04,B03,B02,B08,B05,B06,B07,B8A,B11,B12 -- NOT this
# project's standard order above. Feeding bands in the wrong order silently
# runs the model on the wrong physical bands (e.g. treating B04-as-red data
# as if it were B02-as-blue), so this reorder is required, not cosmetic.
_PROJECT_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
_SEN2SR_ORDER = ["B04", "B03", "B02", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
_TO_SEN2SR_IDX = [_PROJECT_ORDER.index(b) for b in _SEN2SR_ORDER]
_FROM_SEN2SR_IDX = [_SEN2SR_ORDER.index(b) for b in _PROJECT_ORDER]


def _cv2_resize_stack(chip, out_hw, interp):
    c = chip.shape[0]
    out = np.zeros((c, out_hw[0], out_hw[1]), dtype=np.float32)
    for i in range(c):
        out[i] = cv2.resize(chip[i], (out_hw[1], out_hw[0]), interpolation=interp)
    return out


def bicubic_sr(lr_np, scale):
    _, h, w = lr_np.shape
    return _cv2_resize_stack(lr_np, (h * scale, w * scale), cv2.INTER_CUBIC)


def lanczos_sr(lr_np, scale):
    _, h, w = lr_np.shape
    return _cv2_resize_stack(lr_np, (h * scale, w * scale), cv2.INTER_LANCZOS4)


_ESRGAN_MODEL = None


def esrgan_sr(lr_np, scale):
    """
    Real RealESRGAN_x4plus, applied per-band (see notebooks/phase0_baseline_bank.py).
    The pretrained weights are a fixed 4x model; for a non-4x target scale
    (e.g. Venus's native x2 real-GT task) we run the fixed 4x pass and then
    bicubic-resample its output down to the requested scale, rather than
    skip ESRGAN on that benchmark.
    """
    global _ESRGAN_MODEL
    if _ESRGAN_MODEL is None:
        from models.rrdbnet import load_pretrained
        weights_path = _REPO_ROOT / "models" / "weights" / "RealESRGAN_x4plus.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"{weights_path} not found. Download:\n"
                f'  curl -L -o "{weights_path}" '
                f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            )
        _ESRGAN_MODEL = load_pretrained(str(weights_path), device="cpu")

    band_rgb = np.repeat(lr_np[:, None, :, :], 3, axis=1)
    x = torch.from_numpy(band_rgb).float()
    with torch.no_grad():
        y = _ESRGAN_MODEL(x).clamp(0, 1).numpy()
    out_4x = y.mean(axis=1)  # (C, 4h, 4w)

    if scale == 4:
        return out_4x
    _, h4, w4 = out_4x.shape
    target_h, target_w = h4 * scale // 4, w4 * scale // 4
    return _cv2_resize_stack(out_4x, (target_h, target_w), cv2.INTER_CUBIC)


_SEN2SR_MODEL = None


def _get_sen2sr_model():
    global _SEN2SR_MODEL
    if _SEN2SR_MODEL is None:
        import mlstac
        model_dir = _REPO_ROOT / "model" / "SEN2SRLite"
        if not model_dir.exists():
            mlstac.download(
                file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
                output_dir=str(model_dir),
            )
        _SEN2SR_MODEL = mlstac.load(str(model_dir)).compiled_model(device="cpu")
    return _SEN2SR_MODEL


def sen2sr_sr(lr_10band_np, scale):
    """
    SEN2SR pretrained (CC0-1.0, ESA OpenSR). Needs all 10 S2 bands, a fixed
    128x128 input tile, and returns a fixed 4x (512x512) output -- pad/crop
    to fit non-128 chips rather than change the harness's chip size.
    """
    assert scale == 4, "SEN2SRLite is a fixed 4x model"
    model = _get_sen2sr_model()
    c, h, w = lr_10band_np.shape
    assert c == 10, f"SEN2SR needs all 10 bands, got {c}"

    # lr_10band_np is in this project's standard band order -- reorder to
    # SEN2SR's documented input order before feeding the model.
    reordered = lr_10band_np[_TO_SEN2SR_IDX]

    pad_h, pad_w = max(0, SEN2SR_TILE - h), max(0, SEN2SR_TILE - w)
    padded = np.pad(reordered, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    padded = padded[:, :SEN2SR_TILE, :SEN2SR_TILE]

    x = torch.from_numpy(padded).float().unsqueeze(0)
    with torch.no_grad():
        y = model(x).squeeze(0).numpy()  # (10, 512, 512), SEN2SR band order

    out_h, out_w = h * scale, w * scale
    y = y[:, :out_h, :out_w]
    return y[_FROM_SEN2SR_IDX]  # back to project's standard band order


METHODS_4BAND_ONLY = {"bicubic": bicubic_sr, "lanczos": lanczos_sr, "esrgan": esrgan_sr}


def bench_venus():
    """Full metric set (consistency + synthesis + correctness) on real HR."""
    print("Loading opensr-test Venus benchmark (real 5m HR, downloads once)...")
    d = opensr_test.load(dataset="venus")
    n = d["HR"].shape[0]
    print(f"  {n} scenes")

    # L2A band order here is the standard ascending Sentinel-2 L2A order:
    # B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12 (12 bands; confirmed
    # against the opensr-test/HF dataset card's documented L2A-index table).
    # Full 10-band project-order subset (skips B01 idx0, B09 idx9):
    L2A_TO_PROJECT_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    # HR-matching 4-band (B04,B03,B02,B08) directly from L2A indices:
    L2A_HR_ORDER_IDX = [3, 2, 1, 7]

    rows = []
    for i in range(n):
        lr_full = d["L2A"][i].astype(np.float32) / 10000.0  # (12, 128, 128)
        hr = torch.from_numpy(d["HR"][i].astype(np.float32) / 10000.0)  # (4, 256, 256), RGBNIR
        lr_4band = lr_full[L2A_HR_ORDER_IDX]  # (4, 128, 128), matches HR's B04,B03,B02,B08 order
        lr_10band = lr_full[L2A_TO_PROJECT_IDX]  # (10, 128, 128), project order, for SEN2SR
        scale = hr.shape[-1] // lr_4band.shape[-1]

        for name, fn in METHODS_4BAND_ONLY.items():
            try:
                sr_np = fn(lr_4band, scale)
            except Exception as e:
                print(f"  [{name}] scene {i} failed: {e}")
                continue
            sr = torch.from_numpy(np.clip(sr_np, 0, 1)).float()
            lr_t = torch.from_numpy(lr_4band).float()
            m = opensr_test.Metrics()
            res = m.compute(lr=lr_t, sr=sr, hr=hr)
            rows.append({"method": name, "scene": i, "scale": scale, **res})

        # SEN2SR is a fixed 4x, 128-input model -- scale here is only 2x
        # (Venus HR/LR ratio), so pad the 128x128 LR to run it, then
        # downsample its 4x output back to the actual 2x target before
        # comparing against HR (same "run at native scale, resample to the
        # benchmark's ratio" approach used for ESRGAN's fixed-4x model above).
        try:
            sr10_4x = sen2sr_sr(lr_10band, scale=4)  # (10, 512, 512), project order
            sr_hr_order_4x = sr10_4x[NATIVE_10M_IDX]  # (4, 512, 512), B04,B03,B02,B08
            target_hw = hr.shape[-2:]
            sr_np = _cv2_resize_stack(sr_hr_order_4x, target_hw, cv2.INTER_CUBIC)
            sr = torch.from_numpy(np.clip(sr_np, 0, 1)).float()
            lr_t = torch.from_numpy(lr_4band).float()
            m = opensr_test.Metrics()
            res = m.compute(lr=lr_t, sr=sr, hr=hr)
            rows.append({"method": "sen2sr", "scene": i, "scale": scale, **res})
        except Exception as e:
            print(f"  [sen2sr] scene {i} failed: {e}")

    df = pd.DataFrame(rows)
    return df


def bench_punjab(n_scenes_max=4):
    """Consistency + synthesis only (no HR at 2.5m exists) on live Sentinel-2 chips."""
    from data.mpc_fetch import search_scenes, fetch_scene_bands, stack_to_chips

    windows = [
        (75.75, 30.55, 75.80, 30.60),
        (75.80, 30.60, 75.85, 30.65),
    ]
    date_range = "2023-11-01/2023-12-15"

    chips = []
    for bbox in windows[:n_scenes_max]:
        items = search_scenes(bbox, date_range, cloud_cover_max=10, limit=1)
        if not items:
            continue
        stack, scl, _, _ = fetch_scene_bands(items[0], bbox)
        chips.extend(stack_to_chips(stack, scl, chip_size=128, stride=128, cloud_threshold=0.95))

    print(f"  {len(chips)} real Punjab chips (128x128, 10-band)")
    rows = []
    for i, chip in enumerate(chips):
        lr_10 = chip.astype(np.float32)  # (10, 128, 128), already the "LR" -- no degrade
        lr_4band = lr_10[NATIVE_10M_IDX]  # (4, 128, 128)
        scale = 4

        for name, fn in {**METHODS_4BAND_ONLY}.items():
            try:
                sr_np = fn(lr_4band, scale)
            except Exception as e:
                print(f"  [{name}] chip {i} failed: {e}")
                continue
            sr = torch.from_numpy(np.clip(sr_np, 0, 1)).float()
            lr_t = torch.from_numpy(lr_4band).float()
            m = opensr_test.Metrics()
            cons = m.consistency(lr=lr_t, sr=sr)
            syn = m.synthesis(lr=lr_t, sr=sr, hr=None)
            rows.append({"method": name, "chip": i, "scale": scale, **cons, **syn})

        # SEN2SR: full 10-band native path
        try:
            sr10_np = sen2sr_sr(lr_10, scale=4)
            sr10 = torch.from_numpy(np.clip(sr10_np[NATIVE_10M_IDX], 0, 1)).float()
            lr_t = torch.from_numpy(lr_4band).float()
            m = opensr_test.Metrics()
            cons = m.consistency(lr=lr_t, sr=sr10)
            syn = m.synthesis(lr=lr_t, sr=sr10, hr=None)
            rows.append({"method": "sen2sr", "chip": i, "scale": scale, **cons, **syn})
        except Exception as e:
            print(f"  [sen2sr] chip {i} failed: {e}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["venus", "punjab"], default="venus")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / "phase0_results"
    out_dir.mkdir(exist_ok=True)

    if args.dataset == "venus":
        df = bench_venus()
        out_path = out_dir / "opensr_bench_venus.csv"
    else:
        df = bench_punjab()
        out_path = out_dir / "opensr_bench_punjab.csv"

    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}\n")

    metric_cols = [c for c in df.columns if c not in ("method", "scene", "chip", "scale")]
    print(df.groupby("method")[metric_cols].mean().to_string())
