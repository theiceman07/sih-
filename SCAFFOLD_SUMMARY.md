# Repo Scaffold Complete ✓

**Created:** 2026-09-16  
**Status:** Ready for Phase 0 (1 day) on Kaggle

---

## Files Ready Now

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned dependencies |
| `eval/metrics.py` | **FROZEN** metric suite — do not modify |
| `data/cdse_fetch.py` | CDSE Sentinel-2 download client |
| `data/chip_builder.py` | Extract 256×256 chips with cloud masking |
| `notebooks/phase0_baseline_bank.py` | Phase 0 baseline script |
| `README.md` | Quick-start guide |
| `DEVELOPMENT_PLAN.md` | 10-week full roadmap |
| `.gitignore` | Standard git ignores |

---

## Run Phase 0 in 1 Day

```bash
pip install -r requirements.txt
python notebooks/phase0_baseline_bank.py
```

**Output:** `phase0_results/baseline_summary.csv` with metrics proving ESRGAN breaks NDVI (SAM > 5°).

---

## Key Points

1. **`eval/metrics.py` is LOCKED** — all future runs must use this exact code.
2. **CDSE is free** — register at https://dataspace.copernicus.eu/, get API credentials.
3. **Kaggle is your compute** — 2× T4, 30 GPU-hr/week free.
4. **Phase 1 starts with Track B** — build 20m→10m real-GT model to prove SAM loss works.

---

## Next (Weeks 2–3)

Start Phase 1 after baseline gate passes:
- Build `models/swinir.py` (Transformer backbone)
- Write `losses/sam_loss.py` (Spectral Angle Mapper)
- Create `train/train.py` main loop
- Train on Kaggle, checkpoint to Hugging Face every epoch

All details in `DEVELOPMENT_PLAN.md`.

---

**See files for full implementation details.**
