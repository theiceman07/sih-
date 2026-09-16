# PS 26142: Spectral-Safe Super-Resolution for Sentinel-2

A 10-week development plan to build a free, India-validated super-resolution engine for Sentinel-2 satellite imagery.

**Phase 0 (1 day):** Baseline bank proving ESRGAN breaks spectral fidelity.

---

## Quick Start: Phase 0

```bash
pip install -r requirements.txt
python notebooks/phase0_baseline_bank.py
```

**Output:** `phase0_results/baseline_summary.csv` with PSNR/SSIM/SAM/ERGAS metrics.

---

## Files Created in This Scaffold

- `requirements.txt` — Pinned dependencies (PyTorch, timm, rasterio, etc).
- `eval/metrics.py` — **FROZEN** metric suite (PSNR, SSIM, SAM, NDVI delta).
- `data/cdse_fetch.py` — CDSE (Copernicus) API client for free Sentinel-2 download.
- `data/chip_builder.py` — Extract 256×256 chips with cloud masking.
- `notebooks/phase0_baseline_bank.py` — Phase 0 baseline comparison script.
- `README.md` — This file.

---

## Next: Phase 1–6 Implementation

See `DEVELOPMENT_PLAN.md` for the complete 10-week roadmap, team split, and risk register.

**Phases:**
1. **Phase 1 (Weeks 2–3):** Track B real-GT model (20m → 10m).
2. **Phase 2 (Weeks 3–5):** Track A 4× Wald model (10m → 2.5m).
3. **Phase 3 (Weeks 5–6):** MC Dropout uncertainty + geospatial.
4. **Phase 4 (Weeks 6–7):** India multi-zone validation.
5. **Phase 5 (Weeks 7–9):** FastAPI + HF Space product.
6. **Phase 6 (Weeks 9–10):** Model card + evidence pack.

---

## Data Sources (100% Free)

- **Sentinel-2 L2A:** Copernicus Data Space Ecosystem (CDSE, free account)
- **Validation:** ISRO Bhoonidhi LISS-IV 5.8m (free after registration)
- **Compute:** Kaggle (2× T4, 30 GPU-hr/week), Google Colab

---

## Key Metrics (Frozen in Phase 0)

- **PSNR** — Peak Signal-to-Noise Ratio (higher is better)
- **SSIM** — Structural Similarity (higher is better)
- **SAM** — Spectral Angle Mapper in degrees (lower is better, goal: SAM < 5°)
- **NDVI Delta** — Mean absolute NDVI error (lower is better, goal: ≤ 0.05)
- **ERGAS** — Relative Dimensionless Global Error (lower is better)

**Do not modify `eval/metrics.py` after baseline runs.**

---

## Repository Structure

```
srm-sentinel/
  data/          # CDSE fetch, chipping, degradation
  models/        # SwinIR, losses, MC Dropout
  train/         # Training loop, configs
  eval/          # FROZEN metrics, zone reports
  infer/         # Tiling, COG export, ONNX
  api/           # FastAPI service
  web/           # MapLibre UI
  notebooks/     # Phase 0 baseline, EDA
```

---

## Team Roles (6 people)

1. **Data pipeline** — CDSE, chipping, cloud masking
2. **Model & training** — SwinIR, loss design, Kaggle
3. **Metrics & validation** — Baselines, zone reports, calibration
4. **Geospatial & inference** — Tiling, COG, ONNX, QGIS
5. **Backend/API** — FastAPI, job queue, HF Space
6. **Frontend & evidence** — MapLibre, model card, demo video

---

## References

- Copernicus Data Space: https://dataspace.copernicus.eu/
- SwinIR (Liang et al. 2021): https://arxiv.org/abs/2108.10257
- Spectral Angle Mapper: Kruse et al. 1993
- Wald Protocol: Wald et al. 1997
- ISRO Bhoonidhi: https://bhoonidhi.nrsc.gov.in/

---

## License

MIT
