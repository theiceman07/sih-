"""
Runs the full pipeline (api/pipeline.py) once for each of api/main.py's
DEMO_REGIONS and caches everything a jury demo needs as static files under
web/demo_cache/<region>/ -- PNGs the browser can show directly, GeoTIFFs
for a QGIS demo, and a JSON report with the gate result and metrics.

This is the reliable path for a live jury presentation: no server, no live
inference, no dependency on network/CPU/memory conditions during the actual
demo. Run this once ahead of time (and again whenever the model checkpoint
changes), then open web/index.html -- it loads from this cache automatically.

Usage:
    python -m scripts.prebake_demo
    python -m scripts.prebake_demo --skip-tta   # faster, no confidence map
"""
import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.pipeline import run_pipeline, BAND_NAMES, SCALE
from api.main import DEMO_REGIONS
from data.worldcover_fetch import CLASS_NAMES
from infer.export_geotiff import export_sr_geotiff, export_landcover_geotiff
from infer.preview import true_color_png, landcover_png, confidence_png

CACHE_DIR = _REPO_ROOT / "web" / "demo_cache"


def prebake_region(name, bbox, use_tta=True):
    print(f"\n=== {name} {bbox} ===")
    t0 = time.time()
    result = run_pipeline(bbox, use_tta=use_tta)
    elapsed = time.time() - t0
    print(f"  pipeline done in {elapsed:.1f}s | gate_passed={not result.fallback_used} "
          f"| {result.gate_report}")

    out_dir = CACHE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "lr.png").write_bytes(true_color_png(result.lr_norm))
    (out_dir / "reflectance.png").write_bytes(true_color_png(result.sr))
    (out_dir / "landcover.png").write_bytes(landcover_png(result.class_idx))
    if result.confidence is not None:
        (out_dir / "confidence.png").write_bytes(confidence_png(result.confidence))

    sr_name = "reflectance_bicubic_fallback.tif" if result.fallback_used else "reflectance.tif"
    export_sr_geotiff(str(out_dir / sr_name), result.sr, result.transform, result.crs,
                       scale=SCALE, band_names=BAND_NAMES)
    export_landcover_geotiff(str(out_dir / "landcover.tif"), result.class_idx, result.transform,
                              result.crs, scale=SCALE, class_names=CLASS_NAMES)

    report = {
        "region": name,
        "bbox": list(bbox),
        "gate_passed": not result.fallback_used,
        "gate_report": result.gate_report,
        "output_shape": list(result.sr.shape),
        "scale": SCALE,
        "class_names": CLASS_NAMES,
        "elapsed_seconds": round(elapsed, 1),
        "has_confidence": result.confidence is not None,
    }
    if result.confidence is not None:
        report["confidence_mean"] = float(result.confidence.mean())
        report["confidence_max"] = float(result.confidence.max())

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"  saved to {out_dir}")
    return report


def _load_index():
    idx_path = CACHE_DIR / "index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            return json.load(f)
    return {"regions": list(DEMO_REGIONS.keys()), "reports": {}}


def main(args):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.region:
        # Single-region mode: this machine has very little free RAM (as low
        # as ~65MB observed), and running all 3 regions in one long-lived
        # process accumulated enough allocator/cache overhead to OOM on
        # region 2-3 even though each region alone succeeds. A fresh
        # process per region guarantees the OS reclaims everything between
        # runs -- see the run-all branch below, which shells out per region
        # instead of looping in-process.
        index = _load_index()
        try:
            index["reports"][args.region] = prebake_region(
                args.region, DEMO_REGIONS[args.region], use_tta=not args.skip_tta)
        except Exception as e:
            print(f"  FAILED: {e}")
            index["reports"][args.region] = {"region": args.region, "error": str(e)}
        with open(CACHE_DIR / "index.json", "w") as f:
            json.dump(index, f, indent=2)
        return

    # Run-all mode: one subprocess per region (see note above).
    import subprocess
    for name in DEMO_REGIONS:
        print(f"\n>>> subprocess for {name}")
        cmd = [sys.executable, "-u", "-m", "scripts.prebake_demo", "--region", name]
        if args.skip_tta:
            cmd.append("--skip-tta")
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)

    index = _load_index()
    n_ok = sum(1 for r in index["reports"].values() if "error" not in r)
    print(f"\nDone. Cache index: {CACHE_DIR / 'index.json'}")
    print(f"{n_ok}/{len(DEMO_REGIONS)} regions prebaked successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tta", action="store_true", help="Skip TTA confidence map (faster)")
    parser.add_argument("--region", choices=list(DEMO_REGIONS.keys()), default=None,
                         help="Run only this region (used internally for per-region subprocess isolation)")
    args = parser.parse_args()
    main(args)
