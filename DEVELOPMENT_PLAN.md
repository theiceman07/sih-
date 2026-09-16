# PS 26142 — Spectral-Safe Super-Resolution for Sentinel-2
## Development Plan (100% free tooling)

Target: 10m Sentinel-2 → ~2.5–4m GeoTIFF, preserving spectral truth.
Gates: PSNR > 30 dB, SSIM > 0.90, **SAM < 5°**, |ΔNDVI| ≤ 0.05, inference ≤ 3 s/tile (512×512), plus a per-pixel confidence map.

---

## 0. The one decision that makes or breaks this

There is **no free 4m multispectral ground truth over India**. So you cannot naively train 10m → 4m. Solve it with a **three-track supervision strategy** (all free), in this order:

| Track | Supervision | Why it matters |
|---|---|---|
| **A. Wald protocol (primary training)** | Degrade 10m → 40m, learn 40m → 10m, apply the learned 4× operator to 10m → 2.5m | Standard, defensible, infinite free data. This is how the SR remote-sensing literature does it. |
| **B. Real paired GT (spectral proof)** | Sentinel-2 native 20m bands (B5, B6, B7, B8A, B11, B12) → 10m, guided by the real 10m bands (B2, B3, B4, B8) | **Real** ground truth, no synthetic assumption. This is your headline scientific evidence that SAM loss works. |
| **C. Cross-sensor India validation** | ISRO Bhoonidhi LISS-IV (5.8m, free after registration); NAIP 1m (free, US) for ablation only | Turns "synthetic benchmark" into "India-validated". This is the jury differentiator. |

Track B is your safety net: even if Track A's scale-invariance assumption gets challenged, B gives you real-GT numbers. Build B **first** — it is cheap and it directly de-risks the "SAM loss preserves spectral fidelity — RISKY" assumption on slide 4.

---

## 1. Free stack (nothing paid)

**Data**
- Copernicus Data Space Ecosystem (CDSE) — free account, STAC + OData + S3 access to the full Sentinel-2 L2A archive. Use `openeo` or direct S3 via `boto3`. Do not build on the retired SciHub.
- Microsoft Planetary Computer — free STAC for Sentinel-2 L2A and NAIP, no egress cost.
- ISRO Bhoonidhi (bhoonidhi.nrsc.gov.in) — free LISS-IV 5.8m for Indian validation tiles.

**Compute (training)**
- Kaggle Notebooks: 2× T4, 30 GPU-hours/week, free — this is your main trainer.
- Google Colab free T4 — quick experiments only (session caps, preemptible).
- Lightning AI Studio free tier / Paperspace free GPU — overflow capacity.
- Checkpoint to Hugging Face Hub every epoch so a killed session never costs you work.

**Libraries**: PyTorch, `timm`, the SwinIR reference implementation (Apache-2.0), `rasterio`/GDAL, `xarray` + `rioxarray`, `stackstac`, `spectral`, `albumentations`, `torchmetrics`.

**Tracking**: Weights & Biases free personal tier, or TensorBoard plus a CSV. MLflow locally as a fallback.

**Serving / demo**
- Hugging Face Spaces (free CPU, ZeroGPU if granted) running Gradio — the model demo.
- FastAPI + ONNX Runtime (CPU, INT8/FP16) for the tile API.
- Frontend: React + Vite + MapLibre GL (or Leaflet) on Vercel/Netlify free tier; COG tiles served via titiler.
- Storage: Hugging Face Datasets (free, versioned) for chips and weights. SQLite or DuckDB for job metadata.

All of the above covers a hackathon and a pilot on free tiers. No PlanetLabs, no paid cloud.

---

## 2. Repo layout

```
srm-sentinel/
  data/          # cdse_fetch.py, bhoonidhi_ingest.py, chip_builder.py, degrade.py (Wald)
  models/        # swinir.py, heads.py, mc_dropout.py
  losses/        # l1.py, sam_loss.py, index_loss.py (NDVI/NDWI/EVI), perceptual.py
  train/         # train.py, config/*.yaml, sched.py, resume.py
  eval/          # metrics.py (PSNR/SSIM/SAM/ERGAS/dNDVI), zone_report.py, ablation.py
  infer/         # tiler.py (overlap + blend), export_geotiff.py, uncertainty.py, onnx_export.py
  api/           # FastAPI app, job queue, COG writer
  web/           # React + MapLibre, before/after swipe, confidence overlay
  notebooks/     # kaggle_train.ipynb, eda.ipynb
```

---

## 3. Phases

### Phase 0 — Foundations (Week 1)
1. CDSE and Bhoonidhi accounts; verify programmatic download of one L2A scene per target zone.
2. `chip_builder.py`: read L2A, resample all bands onto a common grid, cloud/shadow mask from the SCL band, reject chips with >5% cloud, write 256×256 × N-band float32 chips to `.npy` or zarr.
3. **Baseline bank** — do this before any training, it is your evidence: bicubic, Lanczos, and off-the-shelf Real-ESRGAN on the same chips. Measure PSNR / SSIM / SAM / ΔNDVI for each.
   - Deliverable: a table reproducing ESRGAN's ~0.13 NDVI error from slide 4 with **your own** rerunnable numbers.
4. Freeze `eval/metrics.py` now. Metrics must not change after results start appearing.

**Exit gate:** baseline table committed; 2000+ clean chips for one zone.

### Phase 1 — Track B real-GT model (Weeks 2–3)
- Task: 20m bands → 10m, conditioned on the 10m bands (pan-sharpening-style guidance).
- Model: SwinIR-lite (RSTB depth 4, embed dim 60, window 8) — fits a T4 comfortably.
- Loss: `L = L1 + λ_sam·SAM + λ_idx·|ΔNDVI| + λ_grad·edge`.
  - Start at `λ_sam = 0.1`; sweep {0.03, 0.1, 0.3, 1.0}. Log the SAM-vs-PSNR trade-off curve — that curve is a slide.
- **Ablation matrix (mandatory — this is the scientific contribution):** {L1} vs {L1+SAM} vs {L1+SAM+NDVI} vs {L1+perceptual}. Report all four across all metrics.

**Exit gate:** on **real** GT, SAM < 5° and L1+SAM beats L1-only on SAM by a clear margin while losing < 0.5 dB PSNR.

### Phase 2 — Track A 4× Wald model (Weeks 3–5)
- Build the degradation properly: MTF-aware Gaussian blur matched to Sentinel-2's per-band MTF, then anti-alias-correct decimation 10m → 40m. Do **not** use a plain `cv2.resize` — a naive kernel is the number-one reason SR models fail to transfer to real data.
- Train full SwinIR (embed 96–180 depending on VRAM), 4× upscale, all 10 usable bands as channels.
- Mixed precision, gradient accumulation, cosine LR, ~300–500k iterations. Resume-from-HF-checkpoint loop to survive Kaggle's session cap.
- Scale-transfer check: apply the learned operator at native scale and compare against LISS-IV 5.8m (Track C).

**Exit gate:** PSNR > 30 dB, SSIM > 0.90, SAM < 5° on the held-out Wald test set; qualitative agreement with LISS-IV.

### Phase 3 — Uncertainty and geospatial correctness (Weeks 5–6)
- MC Dropout (p = 0.1 in the RSTB MLPs), N = 10 stochastic passes → per-pixel std → confidence map. Then **calibrate**: bin predicted std against actual error and plot a reliability diagram. An uncalibrated uncertainty map is worse than none — show the calibration plot.
- Cheaper alternative if MC Dropout is too slow: a 3-model deep ensemble, or test-time-augmentation variance.
- `export_geotiff.py`: preserve CRS, transform (scaled 4×), nodata, band descriptions, DN scaling. Write a Cloud-Optimized GeoTIFF. Verify the round trip in QGIS.
- Tiling with 32 px overlap and cosine feather blending so tile seams disappear.

**Exit gate:** a QGIS-openable COG with correct georeferencing and a matching confidence band, plus the calibration plot.

### Phase 4 — India multi-zone validation (Weeks 6–7)
The five zones from slide 13: Punjab wheat, Karnataka coffee, Tamil Nadu rice, Maharashtra cotton, Himalayan sparse vegetation.
- Per zone: ≥ 20 scenes across ≥ 2 seasons.
- Report metrics **per zone and per crop**, not pooled. Pooled numbers hide exactly the generalization failure the jury will ask about.
- Report failure cases honestly — slide 4 already flags "SwinIR generalizes across crops — UNPROVEN". A named failure mode scores higher than a hidden one.

**Exit gate:** `zone_report.md` with a PSNR/SSIM/SAM/ΔNDVI table across 5 zones and an explicit failure-mode list.

### Phase 5 — Product (Weeks 7–9)
- ONNX export plus INT8 quantization; target ≤ 3 s/tile on free CPU, and actually measure it.
- FastAPI: `POST /jobs {bbox, date_range}` → async worker pulls from CDSE, runs inference, writes a COG → `GET /jobs/{id}` returns tile URL, confidence URL, and a metrics JSON.
- Web: MapLibre, draw a bbox, before/after swipe slider, NDVI toggle, confidence heat overlay, GeoTIFF download.
- A Hugging Face Space with 3 pre-baked demo regions so judges never wait on a live download.

### Phase 6 — Evidence pack (Weeks 9–10)
- One-command reproduction script with fixed seeds.
- Model card — the limitations section is not optional.
- 3-minute demo video; benchmark table vs bicubic / Real-ESRGAN / SwinIR-base; the λ_sam trade-off curve; the calibration plot; the 5-zone table.

---

## 4. Risk register (the honest version)

| Risk | Reality | Mitigation |
|---|---|---|
| Wald scale-invariance may not hold at native 10m | Genuine, known limitation of all SR remote-sensing work | Track B real-GT results plus the LISS-IV cross-check; state the limitation explicitly |
| SR "detail" is plausible, not measured | You cannot create information the sensor never captured | Frame the work as *detail-consistent enhancement with spectral guarantees*, never as "true 4m imaging". The confidence map is the honesty mechanism |
| Kaggle 30 h/week cap | A real constraint on a 4× transformer | Checkpoint to HF every epoch; small model first; scale up only after Phase 1 proves the loss design |
| Cloud cover in monsoon zones | Blocks Tamil Nadu and Karnataka data windows | SCL masking plus temporal stacking of 2–3 revisits |
| Bhoonidhi access friction | Registration and approval delays | Open the account on day 1; NAIP as the ablation fallback |

One framing note, worth getting right before the pitch: the deck's ₹24 Cr freemium projection and the 3–5K month-1 user target sit awkwardly next to a research prototype. The defensible claim is the **spectral-fidelity guarantee at zero marginal cost**, not the revenue model. Lead with SAM < 5° on real ground truth.

---

## 5. Six-person split (matches the slide 15 roster)

1. **Data pipeline** — CDSE/Bhoonidhi ingestion, chipping, cloud masking, degradation kernels.
2. **Model and training** — SwinIR, loss design, ablations, Kaggle orchestration.
3. **Metrics and validation** — the frozen metric suite, baselines, 5-zone reports, calibration.
4. **Geospatial and inference** — tiling, COG export, ONNX, QGIS verification.
5. **Backend/API** — FastAPI, job queue, HF Space.
6. **Frontend and evidence** — MapLibre UI, demo video, model card, deck updates.

---

## 6. First five concrete actions

1. Create CDSE, Bhoonidhi, Hugging Face, and Kaggle accounts (Bhoonidhi first — approval takes the longest).
2. Script one Sentinel-2 L2A download for Punjab and open it in QGIS.
3. Write and freeze `eval/metrics.py` (PSNR, SSIM, SAM, ERGAS, ΔNDVI).
4. Run the baseline bank and publish the ESRGAN-breaks-NDVI table.
5. Build Track B (20m → 10m real GT) chips and train the lite model — first real SAM number within ~10 days.
