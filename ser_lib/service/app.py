"""供 Tauri/Electron 壳启动的本地 FastAPI 服务。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from ser_lib.data.manifest import DatasetManifest
from ser_lib.service.catalog import check_compatibility, component_catalog
from ser_lib.service.jobs import JobManager

class CompatibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: dict[str, Any]
    model_name: str
    model_params: dict[str, Any] = Field(default_factory=dict)

class ManifestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest_path: Path
    split: str | None = None
    limit: int = Field(default=20, ge=1, le=200)

def create_app() -> FastAPI:
    app = FastAPI(title="SER Desktop Local Service", version="0.1.0")
    jobs = JobManager(max_workers=2)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/components")
    def components():
        return component_catalog()

    @app.post("/api/compatibility/check")
    def compatibility(request: CompatibilityRequest):
        return check_compatibility(request.data, request.model_name, request.model_params)

    @app.post("/api/datasets/validate")
    def validate_dataset(request: ManifestRequest):
        try:
            manifest = DatasetManifest.load(request.manifest_path)
            records = manifest.get_records(request.split)
            return {"valid": True, "stats": manifest.stats(), "selected_count": len(records)}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/jobs/dataset-validate")
    def submit_dataset_validation(request: ManifestRequest):
        def task(context):
            context.report(0.1, "正在读取数据集")
            manifest = DatasetManifest.load(request.manifest_path)
            context.raise_if_cancelled()
            records = manifest.get_records(request.split)
            context.report(0.9, "正在生成统计")
            return {"stats": manifest.stats(), "selected_count": len(records)}
        return {"job_id": jobs.submit("dataset_validate", task)}

    @app.get("/api/jobs")
    def list_jobs():
        return {"items": jobs.list()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="任务不存在") from None

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return {"accepted": jobs.cancel(job_id)}
        except KeyError:
            raise HTTPException(status_code=404, detail="任务不存在") from None

    @app.post("/api/datasets/preview")
    def preview_dataset(request: ManifestRequest):
        try:
            manifest = DatasetManifest.load(request.manifest_path)
            records = manifest.get_records(request.split)[:request.limit]
            return {"items": [
                {
                    "uid": record.uid,
                    "audio_path": str(manifest.resolve_audio_path(record)),
                    "label": record.label,
                    "speaker_id": record.speaker_id,
                    "start_ms": record.start_ms,
                    "end_ms": record.end_ms,
                    "metadata": dict(record.metadata),
                }
                for record in records
            ]}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app

app = create_app()
