# Model weights

Pretrained weight files (`.pth`) are gitignored — they're large binaries that
don't belong in git history. Download them with:

```bash
mkdir -p models/weights
curl -L -o models/weights/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

Used by `models/rrdbnet.py` as the Phase 0 baseline (real, official
pretrained Real-ESRGAN — see `notebooks/phase0_baseline_bank.py`).

Our own trained checkpoints (SwinIR-lite, `train/checkpoints/*.pt`) are
produced locally by `train/train.py` and are also gitignored for the same
reason; push them to Hugging Face Hub or Kaggle Datasets instead once
training moves off this machine (see `DEVELOPMENT_PLAN.md` Phase 1).
