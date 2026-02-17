# TODO: API Proxy feature
#   - POST /v1/chat/completions       (OpenAI Chat API)
#   - POST /v1/messages                (Anthropic Claude Messages API)
#   - POST /v1/responses               (OpenAI Responses API)
#   - POST /v1beta/models/{model}:generateContent       (Google Gemini API)
#   - POST /v1beta/models/{model}:streamGenerateContent  (Google Gemini Streaming)
#   - Proxy key management (create/revoke, per-channel/model access control)
#   - Forward requests to upstream targets using stored api_key
#   - Docs page at /docs/proxy
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.responses import FileResponse, StreamingResponse
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


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

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
API_TOKEN = os.getenv("API_MONITOR_TOKEN", "").strip()

db = Database(str(DB_PATH))
monitor = MonitorService(
    db=db,
    log_dir=str(LOG_DIR),
    detect_concurrency=3,
    max_parallel_targets=2,
    enable_log_cleanup=LOG_CLEANUP_ENABLED,
    log_max_bytes=LOG_MAX_SIZE_MB * 1024 * 1024,
)


# ---------------------------------------------------------------------------
# SSE event bus
# ---------------------------------------------------------------------------

class EventBus:
    """Simple in-process SSE broadcast."""

    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, event: str, data: str) -> None:
        async with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(f"event: {event}\ndata: {data}\n\n")
                except asyncio.QueueFull:
                    pass  # drop for slow consumers

event_bus = EventBus()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_token(request: Request) -> None:
    """Check Bearer token if API_MONITOR_TOKEN is set."""
    if not API_TOKEN:
        return  # auth disabled
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {API_TOKEN}":
        return
    # Also accept ?token= query param (for SSE / EventSource)
    if request.query_params.get("token") == API_TOKEN:
        return
    raise HTTPException(status_code=401, detail="unauthorized")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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
    source_url: Optional[str] = Field(default=None, max_length=1024)


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
    source_url: Optional[str] = Field(default=None, max_length=1024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_runtime_fields(target: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(target)
    total = int(item.get("last_total") or 0)
    success = int(item.get("last_success") or 0)
    item["last_success_rate"] = round((success * 100.0 / total), 1) if total > 0 else None
    item["running"] = monitor.is_target_running(int(item["id"]))
    item["latest_models"] = db.get_latest_model_statuses(int(item["id"]))
    return item


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    # Inject SSE callback into monitor
    monitor.set_event_callback(_on_monitor_event)
    monitor.start()
    LOGGER.info(
        "log cleanup config enabled=%s max_mb=%d",
        LOG_CLEANUP_ENABLED,
        LOG_MAX_SIZE_MB,
    )
    LOGGER.info("api_monitor started (auth=%s)", "enabled" if API_TOKEN else "disabled")
    try:
        yield
    finally:
        monitor.stop()
        LOGGER.info("api_monitor stopped")


def _on_monitor_event(event_type: str, data: str) -> None:
    """Called from monitor thread – schedule coroutine on the main loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(event_bus.publish(event_type, data), loop)
    except RuntimeError:
        pass


app = FastAPI(title="API Monitor", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# Static pages (no auth)
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/viewer.html")
def viewer() -> FileResponse:
    return FileResponse(WEB_DIR / "log_viewer.html")


@app.get("/analysis.html")
def analysis() -> FileResponse:
    return FileResponse(WEB_DIR / "analysis.html")


# ---------------------------------------------------------------------------
# Health (no auth)
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "running_targets": monitor.running_target_ids()}


# ---------------------------------------------------------------------------
# SSE (auth via query param)
# ---------------------------------------------------------------------------

@app.get("/api/events")
async def sse_events(request: Request, _: None = Depends(verify_token)):
    q = await event_bus.subscribe()

    async def stream():
        try:
            # Send initial heartbeat
            yield "event: connected\ndata: ok\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield msg
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
                # Check if client disconnected
                if await request.is_disconnected():
                    break
        finally:
            await event_bus.unsubscribe(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# API endpoints (protected)
# ---------------------------------------------------------------------------

@app.get("/api/dashboard", dependencies=[Depends(verify_token)])
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


@app.get("/api/targets", dependencies=[Depends(verify_token)])
def list_targets() -> Dict[str, Any]:
    targets = [_target_runtime_fields(t) for t in db.list_targets()]
    return {"items": targets}


@app.get("/api/targets/{target_id}", dependencies=[Depends(verify_token)])
def get_target(target_id: int) -> Dict[str, Any]:
    target = db.get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="target not found")
    return {"item": _target_runtime_fields(target)}


@app.post("/api/targets", dependencies=[Depends(verify_token)])
def create_target(payload: TargetCreate) -> Dict[str, Any]:
    try:
        target = db.create_target(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"item": _target_runtime_fields(target)}


@app.patch("/api/targets/{target_id}", dependencies=[Depends(verify_token)])
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


@app.delete("/api/targets/{target_id}", dependencies=[Depends(verify_token)])
def delete_target(target_id: int) -> Dict[str, Any]:
    existing = db.get_target(target_id)
    if not existing:
        raise HTTPException(status_code=404, detail="target not found")
    success = db.delete_target(target_id)
    if not success:
        raise HTTPException(status_code=404, detail="target not found")
    return {"ok": True}


@app.post("/api/targets/{target_id}/run", dependencies=[Depends(verify_token)])
def run_target_now(target_id: int) -> Dict[str, Any]:
    ok, message = monitor.trigger_target(target_id, force=True)
    if not ok and message == "target not found":
        raise HTTPException(status_code=404, detail=message)
    if not ok and message == "target already running":
        raise HTTPException(status_code=409, detail=message)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message}


@app.get("/api/targets/{target_id}/runs", dependencies=[Depends(verify_token)])
def list_runs(target_id: int, limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    target = db.get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="target not found")
    runs = db.list_runs(target_id, limit=limit)
    return {"target": _target_runtime_fields(target), "items": runs}


@app.get("/api/targets/{target_id}/logs", dependencies=[Depends(verify_token)])
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
