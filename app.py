from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from db import Database
from monitor import MonitorService


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
WEB_DIR = BASE_DIR / "web"
DB_PATH = DATA_DIR / "registry.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("api_monitor")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning("invalid env %s=%r, fallback=%d", name, raw, default)
        return default
    return max(minimum, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


LOG_CLEANUP_ENABLED = _env_bool("LOG_CLEANUP_ENABLED", True)
LOG_MAX_SIZE_MB = _env_int("LOG_MAX_SIZE_MB", 500, minimum=0)

db = Database(str(DB_PATH))
monitor = MonitorService(
    db=db,
    log_dir=str(LOG_DIR),
    detect_concurrency=3,
    max_parallel_targets=2,
    enable_log_cleanup=LOG_CLEANUP_ENABLED,
    log_max_bytes=LOG_MAX_SIZE_MB * 1024 * 1024,
)


class TargetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    base_url: str = Field(min_length=3, max_length=512)
    api_key: str = Field(min_length=1, max_length=2048)
    enabled: bool = True
    interval_min: int = Field(default=30, ge=1, le=1440)
    timeout_s: float = Field(default=30.0, ge=3.0, le=300.0)
    verify_ssl: bool = False
    prompt: str = Field(default="What is the exact model identifier (model string) you are using for this chat/session?", min_length=1, max_length=4000)
    anthropic_version: str = Field(default="2025-09-29", min_length=4, max_length=64)
    max_models: int = Field(default=0, ge=0, le=5000)


class TargetPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    base_url: Optional[str] = Field(default=None, min_length=3, max_length=512)
    api_key: Optional[str] = Field(default=None, min_length=1, max_length=2048)
    enabled: Optional[bool] = None
    interval_min: Optional[int] = Field(default=None, ge=1, le=1440)
    timeout_s: Optional[float] = Field(default=None, ge=3.0, le=300.0)
    verify_ssl: Optional[bool] = None
    prompt: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    anthropic_version: Optional[str] = Field(default=None, min_length=4, max_length=64)
    max_models: Optional[int] = Field(default=None, ge=0, le=5000)


def _target_runtime_fields(target: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(target)
    total = int(item.get("last_total") or 0)
    success = int(item.get("last_success") or 0)
    item["last_success_rate"] = round((success * 100.0 / total), 1) if total > 0 else None
    item["running"] = monitor.is_target_running(int(item["id"]))
    # Add latest models
    item["latest_models"] = db.get_latest_model_statuses(int(item["id"]))
    return item


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    monitor.start()
    LOGGER.info(
        "log cleanup config enabled=%s max_mb=%d",
        LOG_CLEANUP_ENABLED,
        LOG_MAX_SIZE_MB,
    )
    LOGGER.info("api_monitor started")
    try:
        yield
    finally:
        monitor.stop()
        LOGGER.info("api_monitor stopped")


app = FastAPI(title="API Monitor", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/viewer.html")
def viewer() -> FileResponse:
    return FileResponse(WEB_DIR / "log_viewer.html")


@app.get("/analysis.html")
def analysis() -> FileResponse:
    return FileResponse(WEB_DIR / "analysis.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "running_targets": monitor.running_target_ids()}


@app.get("/api/dashboard")
def dashboard() -> Dict[str, Any]:
    targets = db.list_targets()
    total_targets = len(targets)
    enabled_targets = sum(1 for t in targets if t.get("enabled"))
    running_targets = len(monitor.running_target_ids())
    healthy = sum(1 for t in targets if t.get("last_status") == "healthy")
    degraded = sum(1 for t in targets if t.get("last_status") == "degraded")
    down = sum(1 for t in targets if t.get("last_status") in {"down", "error"})
    return {
        "total_targets": total_targets,
        "enabled_targets": enabled_targets,
        "running_targets": running_targets,
        "healthy": healthy,
        "degraded": degraded,
        "down_or_error": down,
    }


@app.get("/api/targets")
def list_targets() -> Dict[str, Any]:
    targets = [_target_runtime_fields(t) for t in db.list_targets()]
    return {"items": targets}


@app.get("/api/targets/{target_id}")
def get_target(target_id: int) -> Dict[str, Any]:
    target = db.get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="target not found")
    return {"item": _target_runtime_fields(target)}


@app.post("/api/targets")
def create_target(payload: TargetCreate) -> Dict[str, Any]:
    try:
        target = db.create_target(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"item": _target_runtime_fields(target)}


@app.patch("/api/targets/{target_id}")
def patch_target(target_id: int, payload: TargetPatch) -> Dict[str, Any]:
    existing = db.get_target(target_id)
    if not existing:
        raise HTTPException(status_code=404, detail="target not found")
    updates = payload.model_dump(exclude_none=True)
    try:
        updated = db.update_target(target_id, updates)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not updated:
        raise HTTPException(status_code=404, detail="target not found")
    return {"item": _target_runtime_fields(updated)}


@app.post("/api/targets/{target_id}/run")
def run_target_now(target_id: int) -> Dict[str, Any]:
    ok, message = monitor.trigger_target(target_id, force=True)
    if not ok and message == "target not found":
        raise HTTPException(status_code=404, detail=message)
    if not ok and message == "target already running":
        raise HTTPException(status_code=409, detail=message)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message}


@app.get("/api/targets/{target_id}/runs")
def list_runs(target_id: int, limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    target = db.get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="target not found")
    runs = db.list_runs(target_id, limit=limit)
    return {"target": _target_runtime_fields(target), "items": runs}


@app.get("/api/targets/{target_id}/logs")
def get_logs(
    target_id: int,
    scope: str = Query(default="latest", pattern="^(latest|all)$"),
    run_id: Optional[int] = Query(default=None),
    limit: int = Query(default=5000, ge=1, le=20000),
) -> Dict[str, Any]:
    target = db.get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="target not found")

    chosen_run: Optional[Dict[str, Any]] = None
    if run_id is not None:
        chosen_run = next((r for r in db.list_runs(target_id, 200) if int(r["id"]) == run_id), None)
        if not chosen_run:
            raise HTTPException(status_code=404, detail="run not found")
    elif scope == "latest":
        chosen_run = db.get_latest_run(target_id)

    query_run_id = int(chosen_run["id"]) if chosen_run else None
    logs = db.list_logs(target_id, run_id=query_run_id, limit=limit)
    return {
        "target": _target_runtime_fields(target),
        "run": chosen_run,
        "count": len(logs),
        "items": logs,
    }
