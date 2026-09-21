"""
FastAPI inference service (FOUR_DAY_PLAN.md section 3D).

    POST /jobs {bbox, date_range}  -> {job_id}
    GET  /jobs/{job_id}            -> {status, reflectance_png/cog, landcover_png/cog,
                                        gate_report}

Runs asynchronously in a background thread (in-process job queue -- no
Redis/Celery needed for a hackathon-scale demo). The actual fetch/tile/gate
logic lives in api/pipeline.py, shared with scripts/prebake_demo.py so the
live demo and the offline demo cache can never silently disagree.

For the jury demo itself, prefer web/index.html's pre-baked cache (see
scripts/prebake_demo.py) over triggering a live job here -- live inference
on a large scene is slow on CPU and this machine has been memory-
constrained during development. This endpoint is real and works, but it is
the "bonus/backup" path, not the primary demo path.
"""
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from data.worldcover_fetch import CLASS_NAMES
from api.pipeline import run_pipeline, BAND_NAMES, SCALE
from infer.export_geotiff import export_sr_geotiff, export_landcover_geotiff
from infer.preview import true_color_png, landcover_png, confidence_png

_REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = _REPO_ROOT / "api" / "job_outputs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-bake these at deploy time (run once, cache the bbox->job mapping) so
# the finale demo never blocks on a live Copernicus/MPC download -- see
# scripts/prebake_demo.py.
DEMO_REGIONS = {
    "punjab_wheat": (75.75, 30.55, 75.80, 30.60),
    "punjab_wheat_2": (75.80, 30.60, 75.85, 30.65),
    "kudaliar_telangana": (78.55, 18.75, 78.60, 18.80),
}

app = FastAPI(title="SRM Inference API", description="Spectral-safe Sentinel-2 super-resolution + land-cover mapping")
_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class JobRecord:
    job_id: str
    status: str = "pending"  # pending | running | done | failed
    error: Optional[str] = None
    reflectance_path: Optional[str] = None
    landcover_path: Optional[str] = None
    reflectance_png: Optional[bytes] = None
    landcover_png: Optional[bytes] = None
    lr_png: Optional[bytes] = None
    metrics: dict = field(default_factory=dict)
    gate_report: dict = field(default_factory=dict)


_JOBS: dict[str, JobRecord] = {}


class JobRequest(BaseModel):
    bbox: Optional[tuple[float, float, float, float]] = None
    demo_region: Optional[str] = None
    date_range: str = "2023-11-01/2023-12-15"
    cloud_cover_max: int = 10


def _run_job(job_id: str, bbox, date_range, cloud_cover_max):
    record = _JOBS[job_id]
    try:
        record.status = "running"
        result = run_pipeline(bbox, date_range=date_range, cloud_cover_max=cloud_cover_max, use_tta=False)

        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        sr_name = "reflectance_bicubic_fallback.tif" if result.fallback_used else "reflectance.tif"
        sr_path = job_dir / sr_name
        export_sr_geotiff(str(sr_path), result.sr, result.transform, result.crs, scale=SCALE,
                           band_names=BAND_NAMES)
        record.reflectance_path = str(sr_path)

        lc_path = job_dir / "landcover.tif"
        export_landcover_geotiff(str(lc_path), result.class_idx, result.transform, result.crs,
                                  scale=SCALE, class_names=CLASS_NAMES)
        record.landcover_path = str(lc_path)

        record.lr_png = true_color_png(result.lr_norm)
        record.reflectance_png = true_color_png(result.sr)
        record.landcover_png = landcover_png(result.class_idx)

        record.gate_report = result.gate_report
        record.metrics = {
            "gate_passed": not result.fallback_used,
            "n_bands": result.sr.shape[0],
            "output_shape": list(result.sr.shape),
            "scale": SCALE,
        }
        record.status = "done"
    except Exception as e:
        record.status = "failed"
        record.error = str(e)


@app.post("/jobs")
def create_job(req: JobRequest):
    if req.demo_region:
        if req.demo_region not in DEMO_REGIONS:
            raise HTTPException(400, f"Unknown demo_region. Options: {list(DEMO_REGIONS)}")
        bbox = DEMO_REGIONS[req.demo_region]
    elif req.bbox:
        bbox = req.bbox
    else:
        raise HTTPException(400, "Provide either bbox or demo_region")

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = JobRecord(job_id=job_id)
    _executor.submit(_run_job, job_id, bbox, req.date_range, req.cloud_cover_max)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    record = _JOBS.get(job_id)
    if record is None:
        raise HTTPException(404, "job not found")
    return {
        "job_id": record.job_id,
        "status": record.status,
        "error": record.error,
        "lr_png": f"/jobs/{job_id}/lr.png" if record.lr_png else None,
        "reflectance_png": f"/jobs/{job_id}/reflectance.png" if record.reflectance_png else None,
        "landcover_png": f"/jobs/{job_id}/landcover.png" if record.landcover_png else None,
        "reflectance_cog": f"/jobs/{job_id}/reflectance.tif" if record.reflectance_path else None,
        "landcover_cog": f"/jobs/{job_id}/landcover.tif" if record.landcover_path else None,
        "metrics": record.metrics,
        "gate_report": record.gate_report,
    }


@app.get("/jobs/{job_id}/lr.png")
def get_lr_png(job_id: str):
    record = _JOBS.get(job_id)
    if record is None or not record.lr_png:
        raise HTTPException(404, "not ready")
    return Response(record.lr_png, media_type="image/png")


@app.get("/jobs/{job_id}/reflectance.png")
def get_reflectance_png(job_id: str):
    record = _JOBS.get(job_id)
    if record is None or not record.reflectance_png:
        raise HTTPException(404, "not ready")
    return Response(record.reflectance_png, media_type="image/png")


@app.get("/jobs/{job_id}/landcover.png")
def get_landcover_png(job_id: str):
    record = _JOBS.get(job_id)
    if record is None or not record.landcover_png:
        raise HTTPException(404, "not ready")
    return Response(record.landcover_png, media_type="image/png")


@app.get("/jobs/{job_id}/reflectance.tif")
def get_reflectance(job_id: str):
    record = _JOBS.get(job_id)
    if record is None or not record.reflectance_path:
        raise HTTPException(404, "not ready")
    return FileResponse(record.reflectance_path, media_type="image/tiff")


@app.get("/jobs/{job_id}/landcover.tif")
def get_landcover(job_id: str):
    record = _JOBS.get(job_id)
    if record is None or not record.landcover_path:
        raise HTTPException(404, "not ready")
    return FileResponse(record.landcover_path, media_type="image/tiff")


@app.get("/demo_regions")
def list_demo_regions():
    return DEMO_REGIONS


@app.get("/health")
def health():
    return {"status": "ok"}
