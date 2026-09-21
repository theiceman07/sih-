# SIH26142: Deep Learning Based Super Resolution Mapping (SRM) for Medium Resolution Satellite Imageries

**Team AL Clauda · NTRO · Space Technology**

A Sentinel-2 (10m) → 2.5m pipeline that does two things the problem statement actually
asks for: spectrally-safe image super-resolution, AND sub-pixel land-cover mapping
(the "mixed pixel problem"). See [`FOUR_DAY_PLAN.md`](FOUR_DAY_PLAN.md) for the current
build plan and the research findings behind it, and [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md)
for the original 10-week roadmap.

---

## Handoff status (read this first)

This was built solo on a memory-constrained dev laptop (~8GB RAM, CPU-only, no GPU).
Everything below is real and tested against real data — but the model sizes, training
step counts, and dataset sizes are all "prove it works" scale, not "final quality" scale.
**The single highest-value thing a teammate with a real GPU can do next is scale up
training** (bigger model, more steps, the full Kudaliar dataset instead of a memory-capped
32×32-crop subset) — the pipeline around it (data, losses, gate, API, demo) is done and
does not need to change to absorb that.

**What's proven working, end to end, on real data (see [`RESULTS.md`](RESULTS.md) for numbers):**
- Real Sentinel-2 fetch (Planetary Computer + CDSE), real ESA WorldCover labels, real
  Indian ground truth (SEN2VENµS Kudaliar, Telangana — 20 real satellite-pair acquisitions)
- A dual-head model (spectral SR + sub-pixel land-cover fractions) trained on real data
  across all 3 demo regions — **the SR output passes the ground-truth-free release gate
  on live Punjab data** (SAM 2.8-3.0°, well under the 5° threshold)
- A trustworthiness benchmark (`opensr-test`) against ESA's own pretrained SEN2SR model
  on real 5m ground truth — this project is honest that SEN2SR is a strong baseline, not
  something to just claim victory over
- A full geospatial pipeline: tiling with seamless blending, CRS-correct GeoTIFF export,
  a FastAPI service, and a pre-baked demo cache (`web/index.html`) that shows real
  before/after imagery without needing a live server

**What's explicitly NOT done yet — the real remaining work:**
1. **Real GPU training.** Every model here trained for a few hundred steps on a CPU, with
   patch sizes capped at 32×32 specifically to dodge a memory/threading crash on this
   machine (see `train/train_kudaliar.py`'s comments). On a GPU: remove the crop-size
   workaround, train at native resolution (128×128 for Kudaliar, full tile size
   elsewhere), for thousands of steps, with a bigger model (`embed_dim`, `depths` in
   `models/swinir_lite.py` / `models/srm_head.py` are all still at the smallest
   "correctness check" settings from `DEVELOPMENT_PLAN.md`'s Phase 1 spec).
2. **The λ_sam ablation is inconclusive** (`RESULTS.md` section 3) — a short run found
   the SAM loss term made things slightly *worse*, which is almost certainly a
   too-short-training artifact, not a real finding. Needs a proper GPU run with the full
   `{0.03, 0.1, 0.3, 1.0}` sweep from `DEVELOPMENT_PLAN.md`.
3. **Kudaliar demo region prebake may still be pending** — this machine kept running out
   of memory generating it (see `scripts/prebake_demo.py`'s per-region-subprocess
   workaround). Punjab's two regions are done and gate-pass; re-run
   `python -m scripts.prebake_demo` on a real machine to fill in Kudaliar and to redo
   everything at full quality once a better model is trained.
4. **5-zone validation, per-class IoU for the land-cover head, ONNX export/latency
   benchmark, model card, and the actual pitch deck edits** — all described in
   `FOUR_DAY_PLAN.md` sections 2-4, none started.
5. **The pipeline itself has one known local-machine-only bug workaround**: SwinIR's
   batched window-attention crashes (Windows access violation) at `batch_size>1` and
   128×128 resolution on this CPU, reproducible with synthetic data too — an
   environment/threading issue, not a data or logic bug, and it should simply not occur
   on a GPU. If it somehow does, `train/train_kudaliar.py`'s comments document the
   workaround (crop to 32×32) used here.

**Fastest way to pick this up:** read `RESULTS.md` top to bottom (it's short and has the
real numbers), then `FOUR_DAY_PLAN.md` for the why-we're-building-this-way context, then
just start scaling up the existing training scripts — nothing needs to be rewritten.

---

## Quick Start: Phase 0 (baseline evidence)

```bash
pip install -r requirements.txt
python notebooks/phase0_baseline_bank.py
```

**Output:** `phase0_results/baseline_summary.csv` with PSNR/SSIM/SAM/ERGAS metrics.

## Quick Start: opensr-test trustworthiness benchmark

```bash
python -m eval.opensr_bench --dataset venus     # bicubic/lanczos/esrgan/SEN2SR on real 5m GT
python -m eval.opensr_bench --dataset punjab    # ground-truth-free consistency check, live data
```

## Quick Start: real Indian ground-truth (Kudaliar, Telangana)

```bash
# Download once (~5GB): https://zenodo.org/records/6514159/files/KUDALIAR.7z?download=1
# Extract into data_raw/kudaliar/
python -m data.sen2venus_kudaliar          # smoke test the loader
python -m train.train_kudaliar --steps 300 # SAM-loss ablation on real GT
```

## Quick Start: SRM (land-cover) head

```bash
python -m data.worldcover_fetch   # smoke test WorldCover label fetch
python -m models.srm_head         # dual-head model shape check
python -m train.train_srm         # trains the land-cover fraction head
```

## Quick Start: inference API + dashboard

```bash
uvicorn api.main:app --reload
python -m eval.export_dashboard_data   # refresh web/data/results.json
# open web/index.html in a browser
```

## Quick Start: jury demo (pre-baked, no live server needed)

```bash
python -m scripts.prebake_demo             # runs all 3 demo regions once, ~5-15 min
cd web && python -m http.server 8080       # any static file server works
# open http://localhost:8080 -- shows real before/after images + gate status per region
```
This is the reliable path for an actual presentation: no live inference, no server
uptime risk, no network dependency during the demo itself. See `scripts/prebake_demo.py`.

---

## What's Built

| Area | Files | Status |
|---|---|---|
| Frozen metrics | `eval/metrics.py` | PSNR/SSIM/SAM/ERGAS/ΔNDVI, locked |
| Trustworthiness benchmark | `eval/opensr_bench.py` | bicubic/lanczos/ESRGAN/SEN2SR via `opensr-test` |
| Data ingest | `data/mpc_fetch.py`, `data/cdse_fetch.py`, `data/chip_builder.py` | live Sentinel-2 (Planetary Computer / CDSE) |
| Land-cover labels | `data/worldcover_fetch.py` | real ESA WorldCover 10m, fraction aggregation |
| Real Indian GT | `data/sen2venus_kudaliar.py` | SEN2VENµS Kudaliar (Telangana) site, 20 real acquisitions |
| Track B (real-GT SR) | `data/track_b_dataset.py`, `train/train.py` | 20m→10m, real 10m pixels as label |
| SRM dual-head model | `models/srm_head.py`, `losses/srm_loss.py`, `data/srm_dataset.py`, `train/train_srm.py` | SR + sub-pixel land-cover fractions, shared encoder |
| Ablation | `train/train_kudaliar.py` | λ_sam=0 vs 0.1 on real Kudaliar GT |
| Release gate | `infer/gate.py` | ground-truth-free consistency check (SAM/reflectance/NDVI re-degrade) |
| Tiling | `infer/tiler.py` | overlap + cosine-feather blending, no seams |
| GeoTIFF export | `infer/export_geotiff.py` | COG-style, CRS-correct, land-cover colour table |
| API | `api/main.py` | FastAPI job queue, gate-then-ship, pre-baked demo regions |
| Dashboard | `web/index.html`, `eval/export_dashboard_data.py` | evidence tables + live demo trigger |

See `DEVELOPMENT_PLAN.md` for the original 10-week roadmap and `FOUR_DAY_PLAN.md` for
the accelerated plan and the research findings that changed it (land-cover requirement,
real Indian GT availability, the SAM<5° gate being too weak, SEN2SR as a strong existing
baseline).

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
sih/
  data/          # CDSE/MPC fetch, WorldCover labels, Kudaliar real-GT loader, chipping, degradation
  models/        # SwinIR-lite, dual-head SRM model, RRDBNet (ESRGAN baseline)
  losses/        # SAM loss, NDVI index loss, SRM fraction loss
  train/         # Track B, SRM head, Kudaliar ablation training loops
  eval/          # FROZEN metrics, opensr-test benchmark harness, dashboard export
  infer/         # Ground-truth-free gate, tiler, GeoTIFF export
  api/           # FastAPI inference service
  web/           # Evidence dashboard (static HTML/JS)
  notebooks/     # Phase 0 baseline bank
  data_raw/      # Downloaded Kudaliar archive (gitignored)
  phase0_results/# All benchmark/ablation CSV outputs
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
- SEN2VENµS dataset (real 5m GT, incl. Kudaliar, India): https://zenodo.org/records/6514159
- ESA SEN2SR (pretrained, CC0): https://github.com/ESAOpenSR/SEN2SR
- opensr-test benchmark: https://github.com/ESAOpenSR/opensr-test
- ESA WorldCover 10m: https://esa-worldcover.org/en/data-access

---

## License

MIT
