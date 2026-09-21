# SIH26142 — 4-Day Build Plan

**Team AL Clauda · NTRO · Space Technology**
**Written 2026-09-21. Build window: Sept 21–24. Nomination deadline: Sept 30.**

---

## 0. Five research findings that change the plan

Before the schedule, read this. Five things came out of the research round that materially
change what we should build. Ignoring any of them costs us the round.

### 0.1 The problem statement asks for land-cover MAPS, not just sharper pixels

The official NTRO text:

> Medium-resolution satellite imagery (10m–30m, e.g., Sentinel-2, Landsat-8/9) offers excellent
> temporal revisit times and global coverage, but lacks spatial clarity for micro-land cover
> analysis, urban monitoring, and disaster damage assessment. [...] implement a Geospatial Deep
> Learning Pipeline for **Super Resolution Mapping (SRM)**, converting medium-resolution
> multi-spectral bands into **high-resolution land cover maps** and enhanced spatial imagery
> while **solving the mixed pixel problem**.

"Super Resolution Mapping" is a term of art in remote sensing. It means **sub-pixel land-cover
mapping** — predicting the class distribution *inside* a coarse mixed pixel — which is a
different task from image super-resolution. Our current deck (slide 3) delivers only the image
SR half: "10m → 4m + confidence map." We deliver zero land-cover output and never mention
mixed pixels.

A judge from NTRO reading our deck against their own problem statement will notice.

**Action: add a land-cover head.** This is non-negotiable and it is the single highest-value
change in this plan. Detail in Day 2.

### 0.2 There is real 5 m ground truth over India, and it is free

SEN2VENµS is a dataset of Sentinel-2 patches paired with same-day VENµS satellite acquisitions
at 5 m. 29 sites, 132,955 patches. One of those sites is **KUDALIAR — Telangana, India**:
7,269 patches, 20 acquisition pairs, 5.0 GB.

This directly kills two "UNPROVEN/RISKY" flags on our own slide 4:

| Slide 4 challenge | Status after Kudaliar |
|---|---|
| "Scarce high-res ground truth" | Solved — real 5 m GT, no Wald assumption |
| "Not India-validated" (vs ESA OpenSR) | Solved — measured numbers on Indian farmland |

Telangana farmland is also genuinely representative of the small-holding problem we pitch
(slide 4: 1.08 ha average). This is our headline evidence. Get it on Day 1.

```
https://zenodo.org/records/6514159/files/KUDALIAR.7z?download=1     # 5.0 GB
```

### 0.3 Our own baseline numbers weaken the "SAM < 5°" pitch — fix the claim

From `phase0_results/baseline_summary_n30.csv`, our real measured numbers:

| Method | PSNR | SSIM | SAM° | ΔNDVI |
|---|---|---|---|---|
| bicubic | 31.47 | 0.882 | **1.00** | **0.018** |
| lanczos | 31.34 | 0.877 | **1.02** | **0.019** |
| esrgan | 23.29 | 0.507 | 2.59 | 0.034 |

**Bicubic already passes SAM < 5° and ΔNDVI ≤ 0.05.** So does Lanczos. So does ESRGAN, on our
data. Our advertised spectral gate is a bar that a 1980s interpolation kernel clears with 5×
margin. If a judge asks "what does your model do that bicubic doesn't?" and our answer is
"SAM < 5°", we lose.

Also note slide 2 claims ESRGAN distorts NDVI by **0.13**. Our own measurement says **0.034**.
Do not present a number we cannot reproduce — replace it with our measured 0.034 or drop it.

**The honest and much stronger claim** is a three-way separation that no single metric shows:

| | Adds real detail | Spectrally safe |
|---|---|---|
| Bicubic | ✗ none | ✓ |
| Generic SR (ESRGAN) | ✓ but fabricated | ✗ |
| **Ours** | ✓ **and verified** | ✓ |

That middle column needs a metric we don't currently have. Which brings us to:

### 0.4 opensr-test gives us the metric that proves "real detail, not hallucination"

ESA's `opensr-test` (pip installable) is a benchmark built specifically for trustworthy
Sentinel-2 SR. It computes three families:

- **Consistency** — reflectance / spectral (SAD°) / spatial alignment vs the LR input
- **Synthesis** — how much genuine high-frequency detail was added beyond interpolation
- **Correctness** — decomposed into **hallucination** (detail in SR absent from HR),
  **omission** (detail in HR missed), **improvement** (detail correctly recovered)

This is exactly the missing column. Bicubic will score ~0 synthesis. ESRGAN will score high
synthesis *and* high hallucination. If we land high synthesis + high improvement + low
hallucination, that is a defensible, third-party-benchmarked scientific result — not a
self-graded metric.

It also ships a Venµs ×2 benchmark split, which pairs naturally with our Kudaliar work.

```bash
pip install opensr-test
```

### 0.5 The competition is real, public, and already building

Two public GitHub repos are on this exact problem statement:

- **`shyam0github/DrishtiSR`** — substantially mature. 10m→2.5m ×4, heteroscedastic
  per-pixel uncertainty, ONNX INT8, ≤1M params, 453 tests, React + MapLibre UI, Kaggle
  training. Roughly our whole deck, already built.
- **`TIRUNARA/sih-26142-super-resolution-mapping`** — early stage (23 commits), but note
  their framing: "converting medium-resolution multi-spectral bands into high-resolution
  **land cover maps**." They read the PS correctly. We didn't.

Meanwhile **ESA SEN2SR** is pip-installable, CC0-1.0 (public domain), pretrained, and does
10m→2.5m on all 10 bands today. Anyone can `pip install sen2sr` and beat a from-scratch
4-day model.

**Therefore we do not compete on "we built an SR model."** We compete on the three things
none of them have:

1. **Land-cover SRM output with sub-pixel fractions** — what the PS actually asked for
2. **Real Indian 5 m ground truth validation** (Kudaliar) — including a head-to-head vs SEN2SR
3. **A release gate that refuses to ship a tile** when consistency fails — trust as a product
   feature, not a metric in a table

---

## 1. Revised one-line pitch

> Free Sentinel-2 10 m → 2.5 m imagery **and a 2.5 m land-cover map with sub-pixel class
> fractions**, validated against real 5 m satellite ground truth over Indian farmland,
> with every tile gated on spectral consistency and a per-pixel confidence map.

Changes vs current deck: land-cover map added, "India-validated" upgraded from aspiration to
measured, scale corrected to 2.5 m (matches SEN2SR and the ×4 literature; our own
`phase0` and deck disagree — deck says 4 m/2.5×, plan says 2.5 m/4×. **Pick 2.5 m ×4** and
make it consistent everywhere).

---

## 2. Team split (6 people, per slide 1 roster)

| # | Owner | Track | Day 1 | Day 2 | Day 3 | Day 4 |
|---|---|---|---|---|---|---|
| 1 | Data | Kudaliar + WorldCover ingest | ██ | ░ | | |
| 2 | Model — SR | SwinIR-lite train | ░ | ██ | ░ | |
| 3 | Model — SRM | Land-cover head | ░ | ██ | ░ | |
| 4 | Eval | opensr-test harness + SEN2SR baseline | ██ | ░ | ██ | ░ |
| 5 | Backend | Inference, COG, gate, API | | ░ | ██ | ░ |
| 6 | Frontend | MapLibre UI, demo, deck | | ░ | ░ | ██ |

██ = primary load, ░ = support. Everyone commits to `main` via their own branch daily.

---

## 3. Day-by-day

### DAY 1 (Sept 21) — Real Indian ground truth on disk, baselines honest

**Goal by end of day: we can measure anything against real 5 m Indian GT.**

**1A · Kudaliar ingest** (Owner 1)
```bash
# Run on Kaggle (fast pipe, free disk) not on the laptop
wget -O KUDALIAR.7z "https://zenodo.org/records/6514159/files/KUDALIAR.7z?download=1"
7z x KUDALIAR.7z
```
Deliver `data/sen2venus_kudaliar.py`: a `Dataset` yielding `(lr_10m[4,128,128],
hr_5m[4,256,256])` pairs — that is a native **×2** real-GT task. Hold out 2 of the 20
acquisition pairs as test, by *date* not by patch, so there is no spatial leakage.

> Note the resolution arithmetic: Kudaliar gives real GT at ×2 (10→5 m), while the product
> target is ×4 (10→2.5 m). Do not paper over this. Report ×2 as *measured on real GT* and ×4
> as *the deployed operator, validated by consistency + opensr-test*. Stating this distinction
> plainly is worth more to a technical judge than hiding it.

**1B · Land-cover labels** (Owner 1)
ESA WorldCover 10 m v200, free, 11 classes, full India coverage, COGs on AWS Open Data.
Pull the tiles covering Kudaliar + our 5 zones. These are our SRM labels and they are
*real measured labels*, not synthetic — which makes the SRM task's supervision stronger
than the image SR task's.

**1C · Benchmark harness** (Owner 4) — the critical path item
```bash
pip install opensr-test sen2sr mlstac
```
- Wire `eval/opensr_bench.py` around `opensr_test.Metrics()`.
- Run **SEN2SR pretrained** over our Kudaliar test split. This is the number to beat and we
  must know it on Day 1, not Day 4.
```python
import mlstac
mlstac.download(
    file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
    output_dir="model/SEN2SRLite",
)
model = mlstac.load("model/SEN2SRLite").compiled_model(device=device)
```
- Re-run bicubic + ESRGAN through opensr-test too, so Day 1 ends with a 3-row honest table:
  synthesis / hallucination / improvement / consistency for bicubic, ESRGAN, SEN2SR.

**Gate to pass before sleeping:** `phase0_results/opensr_bench_day1.csv` exists with real
numbers for ≥3 methods on real Kudaliar data. If SEN2SR beats everything by a mile, that is
fine and useful — we pivot to "India-adapted SEN2SR + SRM head" and say so openly.

---

### DAY 2 (Sept 22) — Train both heads on Kaggle

**Goal: two trained models, ablation evidence, checkpoints on HF.**

**2A · SR model** (Owner 2)
- `SwinIRLite`, `in_chans=10`, scale ×2, train on Kudaliar real pairs.
- Loss `L = L1 + λ_sam·SAM + λ_idx·|ΔNDVI| + λ_grad·edge`, `λ_sam ∈ {0, 0.1, 0.3}`.
- **Mandatory ablation: λ_sam = 0 vs 0.1.** If SAM loss doesn't measurably help on real GT,
  we must know before we put it on a slide. Our whole thesis is that it does.
- Checkpoint to Hugging Face every epoch — Kaggle sessions die.

**2B · The SRM / land-cover head** (Owner 3) — *this is the new work, protect it*

Architecture: shared encoder, two heads.
```
S2 10m (10 bands) ──► SwinIR-lite encoder ──┬──► SR head    ──► reflectance ×2/×4
                                            └──► SRM head   ──► 11-class fractions @ ×4
```
The SRM head outputs **per-class abundance fractions** (softmax over 11 WorldCover classes)
at the fine grid, plus an argmax hard map. Fractions are literally the mixed-pixel solution
the PS asks for — a 10 m pixel that is 60 % cropland / 40 % built-up comes out as a fraction
vector, then as 16 sub-pixels at 2.5 m.

Supervision, and why it is defensible:
- Input: Sentinel-2 degraded 10 m → 40 m (MTF-matched, reuse `mtf_degrade`)
- Label: **real** ESA WorldCover at 10 m
- Train 40 m → 10 m land cover, apply the learned ×4 operator at 10 m → 2.5 m

The label here is a genuine independent measurement, not a downsampled copy of the input.
That makes this supervision materially stronger than image-Wald, and it's a good slide.

Losses: cross-entropy on fractions + an **abundance-sum constraint** (fractions within a
coarse pixel must sum to the coarse pixel's true fraction). That constraint is the classical
sub-pixel mapping correctness condition and citing it shows we know the SRM literature.

Metrics: per-class IoU, overall accuracy, and fraction RMSE vs WorldCover.

**2C** (Owner 4) Fold the trained checkpoints into the Day 1 harness as they land.

**Gate:** SR model beats bicubic on synthesis with hallucination below ESRGAN's, on real
Kudaliar test. SRM head beats a "nearest-neighbour upsampled WorldCover" baseline on IoU.
Both numbers committed to the repo.

---

### DAY 3 (Sept 23) — Make it a product, not a notebook

**3A · Inference + geospatial** (Owner 5)
- `infer/tiler.py`: 32 px overlap, cosine feather blend, no visible seams.
- `infer/export_geotiff.py`: Cloud-Optimized GeoTIFF, CRS preserved, transform scaled ×4,
  band descriptions set, nodata honoured. **Open it in QGIS and screenshot it** — that
  screenshot is a slide.
- Two outputs per job: reflectance COG + land-cover COG (with a colour table).

**3B · The release gate** (Owner 5 + 4) — our actual differentiator, build it properly
```
gate(tile):
    sr_down = MTF_degrade(sr, 4)              # re-degrade the output
    spectral  = SAD(sr_down, lr)      < 5°    # spectral consistency
    reflect   = L1(sr_down, lr)       < τ     # radiometric consistency
    ndvi      = |ΔNDVI(sr_down, lr)|  < 0.05
    → PASS = ship | FAIL = flag, return bicubic + reason, never ship silently
```
The gate is checkable **without ground truth**, at inference time, anywhere on Earth. That is
the property that makes it a deployable trust mechanism rather than a benchmark number. Log
every gate decision — a "12 of 400 tiles refused, here's why" stat is a strong demo moment.

**3C · Confidence map** (Owner 2)
MC Dropout p=0.1, N=10 passes → per-pixel σ. Then **calibrate**: bin predicted σ against
actual error on the Kudaliar test set and plot the reliability diagram. An uncalibrated
uncertainty map is worse than none. If MC Dropout is too slow, use TTA variance (flips +
rotations) — cheaper and honest.

**3D · API** (Owner 5)
FastAPI, matching the existing `fastapi` project setup:
- `POST /jobs {bbox, date_range}` → job id
- `GET /jobs/{id}` → `{reflectance_cog, landcover_cog, confidence_cog, metrics, gate_report}`
- Pre-bake 3 demo regions so the finale demo never waits on a Copernicus download.

**Gate:** one bbox in → two COGs + confidence + gate report out, verified open in QGIS.

---

### DAY 4 (Sept 24) — Demo, evidence, deck

**4A · Frontend** (Owner 6)
React + MapLibre. Before/after swipe. Land-cover layer toggle with legend. Confidence heat
overlay. Gate badge per tile (PASS green / FAIL amber with reason). GeoTIFF download.
Deploy free on Vercel; model on a Hugging Face Space with Gradio as the fallback demo.

**4B · Evidence pack** (Owner 4)
- `RESULTS.md` — the full table: bicubic / Lanczos / ESRGAN / **SEN2SR** / ours, across
  PSNR / SSIM / SAM / ΔNDVI / synthesis / hallucination / improvement, on **real Kudaliar GT**
- λ_sam ablation curve
- Calibration / reliability diagram
- SRM confusion matrix + per-class IoU
- Model card **including the limitations section** — ×2 real-GT vs ×4 deployed, the single
  Indian site, WorldCover's own 76.7 % accuracy ceiling
- One-command repro script, fixed seeds

**4C · Deck rewrite** (Owner 6) — see §4

**Gate:** a stranger can clone the repo, run one command, and reproduce the headline table.

---

## 4. Deck edits (do these, they're cheap and they matter)

| Slide | Change | Why |
|---|---|---|
| 2 | ESRGAN NDVI error `0.13` → **`0.034`** | We measured it. Never show a number we can't reproduce. |
| 3 | Add land-cover map as a **second output** alongside imagery | The PS asks for it. Currently absent. |
| 3 | Add "mixed pixel" / sub-pixel fractions to "How It Works" | Uses the PS's own vocabulary. |
| 3 | Pick **2.5 m ×4** and use it everywhere | Deck says 4 m/2.5×, plan says 2.5 m/4×. Inconsistency reads as sloppiness. |
| 4 | "Scarce high-res ground truth" → **solved, Kudaliar 5 m** | Turns a risk into evidence. |
| 4 | "Overlap with ESA OpenSR" → **head-to-head table vs SEN2SR on India** | Confronts the strongest objection with data. |
| 5 | Add synthesis/hallucination as the headline metric, SAM as a floor | SAM < 5° is cleared by bicubic; it cannot be the differentiator. |
| 6 | Competitive table: add **"Land-cover SRM output"** row | It's the row where we're alone. |
| 6 | Soften ₹24 Cr freemium, lead with the public-good core | A research prototype with a revenue projection invites the wrong question. |

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| SEN2SR beats our model outright | **High** | Plan for it. Pivot to "India-adapted SEN2SR + SRM head + gate" and say so openly. Fine-tuning a CC0 model is legitimate engineering; pretending we didn't isn't. |
| 5 GB download + training exceeds Kaggle 30 h/week | Medium | Start the download Day 1 morning. Train ×2 only. Subsample Kudaliar to ~20 k patches. |
| Land-cover head doesn't converge in one day | Medium | Fallback: fractions-only regression head (no hard classification). Still answers the mixed-pixel ask. |
| Only one Indian validation site | **Certain** | State it plainly as a limitation. Supplement with the 5-zone consistency-gate results (no GT needed) — that's the whole point of a GT-free gate. |
| MC Dropout too slow for the 3 s/tile target | Medium | TTA variance fallback, decided Day 3 morning. |
| Confusing SR with SRM in the pitch | Medium | Say both names explicitly on slide 3. "We do SRM — the land-cover mapping — *and* the imagery." |

---

## 6. What to cut if we fall behind

Cut in this order. Protect the top of the list at all costs.

1. ~~Frontend polish~~ — a Gradio Space is an acceptable demo
2. ~~MC Dropout~~ — TTA variance is fine
3. ~~ONNX / INT8~~ — measure PyTorch CPU latency and report it honestly
4. ~~5-zone validation~~ — one zone + Kudaliar is enough for nomination
5. ~~×4 model~~ — ship ×2 on real GT and say the ×4 is next

**Never cut:** Kudaliar validation (§0.2), the land-cover head (§0.1), the opensr-test
table (§0.4), the release gate (Day 3B). Those four *are* the submission.

---

## 7. Day 1 first commands

```bash
pip install opensr-test sen2sr mlstac tacoreader
```

```bash
wget -O KUDALIAR.7z "https://zenodo.org/records/6514159/files/KUDALIAR.7z?download=1"
```

---

## Sources

- [SEN2VENµS on Zenodo](https://zenodo.org/records/6514159) · [on Hugging Face](https://huggingface.co/datasets/tacofoundation/sen2venus)
- [ESA SEN2SR](https://github.com/ESAOpenSR/SEN2SR) · [opensr-model](https://github.com/ESAOpenSR/opensr-model) · [opensr-utils](https://github.com/ESAOpenSR/opensr-utils)
- [opensr-test benchmark](https://github.com/ESAOpenSR/opensr-test) · [docs](https://esaopensr.github.io/opensr-test/index.html) · [PyPI](https://pypi.org/project/opensr-test/)
- [ESA WorldCover 10 m](https://esa-worldcover.org/en/data-access) · [AWS Open Data](https://registry.opendata.aws/esa-worldcover-vito/) · [v200 on Zenodo](https://zenodo.org/records/7254221)
- [Bhoonidhi (ISRO LISS-IV)](https://bhoonidhi.nrsc.gov.in/) · [API spec](https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/)
- [A radiometrically and spatially consistent SR framework for Sentinel-2 (RSE 2025)](https://www.sciencedirect.com/science/article/pii/S0034425725006261)
- [DiffFuSR: SR of all Sentinel-2 bands using diffusion](https://arxiv.org/pdf/2506.11764)
- [Comparative validation of 10 m global land cover maps (RSE 2024)](https://www.sciencedirect.com/science/article/pii/S0034425724003341)
- Competing SIH26142 entries: [DrishtiSR](https://github.com/shyam0github/DrishtiSR) · [TIRUNARA](https://github.com/TIRUNARA/sih-26142-super-resolution-mapping)
- [SIH 2026 problem statement catalogue](https://github.com/NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements)
