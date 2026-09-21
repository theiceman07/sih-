"""
Compiles the evidence pack's CSVs into a single JSON for the web dashboard
(FOUR_DAY_PLAN.md section 4A/4B). Run after any benchmark/ablation script
that writes into phase0_results/, before opening web/index.html.
"""
import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "phase0_results"
OUT_PATH = _REPO_ROOT / "web" / "data" / "results.json"


def _safe_read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs) if path.exists() else None


def build_dashboard_data():
    data = {}

    # Phase 0 baseline bank: bicubic/lanczos/esrgan on real Punjab Sentinel-2
    # (frozen eval/metrics.py suite -- PSNR/SSIM/SAM/ERGAS/dNDVI).
    p = RESULTS_DIR / "baseline_summary_n30.csv"
    df = _safe_read_csv(p, header=[0, 1], index_col=0)
    if df is not None:
        data["phase0_baseline"] = {
            method: {metric: float(df.loc[method, (metric, "mean")]) for metric in ["psnr", "ssim", "sam", "ergas", "ndvi_delta"]}
            for method in df.index
        }

    # opensr-test trustworthiness benchmark on real Venus 5m ground truth
    # (bicubic/lanczos/esrgan/sen2sr -- consistency/synthesis/correctness).
    p = RESULTS_DIR / "opensr_bench_venus.csv"
    df = _safe_read_csv(p)
    if df is not None:
        metric_cols = ["reflectance", "spectral", "synthesis", "ha_metric", "om_metric", "im_metric"]
        grouped = df.groupby("method")[metric_cols].mean()
        data["opensr_bench_venus"] = {
            method: {col: float(grouped.loc[method, col]) for col in metric_cols}
            for method in grouped.index
        }

    # SAM-loss ablation on real Kudaliar ground truth (lambda_sam=0 vs 0.1).
    p = RESULTS_DIR / "kudaliar_sam_ablation.csv"
    df = _safe_read_csv(p)
    if df is not None:
        data["kudaliar_ablation"] = df.set_index("config").to_dict(orient="index")

    return data


if __name__ == "__main__":
    data = build_dashboard_data()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    for section, content in data.items():
        print(f"  {section}: {list(content.keys())}")
