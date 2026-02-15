# TODO: Refactor to async architecture
#   - Replace urllib with httpx.AsyncClient (connection pooling, HTTP/2, retries)
#   - Replace ThreadPoolExecutor with asyncio.Semaphore
#   - Replace BackgroundScheduler with AsyncIOScheduler
#   - Convert http_json / _detect_one / _get_models / _run_target to async def
from __future__ import annotations

import json
import logging
import re
import socket
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import httpx
from apscheduler.schedulers.background import BackgroundScheduler

from db import Database


LOGGER = logging.getLogger("api_monitor")

ROUTE_RULES: Tuple[Tuple[str, str], ...] = (
    (r"claude", "anthropic"),
    (r"gemini", "gemini"),
    (r"codex", "responses"),
    (r"gpt-5\.[123]", "responses"),
)


class ApiMonitorError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    text: str
    json_body: Optional[Any]
    elapsed_ms: int


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return normalized
    parts = urlsplit(normalized)
    path = parts.path.rstrip("/")
    if path.lower().endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment)).rstrip("/")


def build_ssl_context(verify_ssl: bool) -> ssl.SSLContext:
    if verify_ssl:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "api-monitor/1.0",
        "Accept": "application/json",
    }


def _check_response_body_for_error(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None

    if "error" in body and body["error"]:
        err = body["error"]
        if isinstance(err, str):
            return err
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg:
                return msg
            return json.dumps(err, ensure_ascii=False)[:500]
        return str(err)[:500]

    if body.get("success") is False and isinstance(body.get("message"), str):
        return body["message"]

    code = body.get("code")
    if (
        isinstance(code, (int, float))
        and code not in (0, 200)
        and isinstance(body.get("message"), str)
    ):
        return f"[{code}] {body['message']}"
    return None


def _extract_text_from_chat(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = c0.get("message") if isinstance(c0.get("message"), dict) else None
    if msg:
        for key in ("content", "reasoning_content", "refusal"):
            val = msg.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:500]
    text = c0.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()[:500]
    return None


def _extract_text_from_anthropic(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    content = body.get("content")
    if not isinstance(content, list) or not content:
        return None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()[:500]
    return None


def _extract_text_from_gemini(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    c0 = candidates[0] if isinstance(candidates[0], dict) else {}
    content = c0.get("content") if isinstance(c0.get("content"), dict) else {}
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return None
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            thought = part.get("thought")
            if isinstance(text, str) and text.strip() and not thought:
                return text.strip()[:500]
    return None


def _extract_text_from_responses(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    output = body.get("output")
    if not isinstance(output, list) or not output:
        return None
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    text = c.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()[:500]
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()[:500]
    return None


def http_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[Dict[str, Any]],
    timeout_s: float,
    ssl_context: ssl.SSLContext,
) -> HttpResult:
    req_headers = dict(headers)
    payload = None
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req = Request(url=url, method=method.upper(), headers=req_headers, data=payload)

    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            parsed = None
            try:
                parsed = json.loads(text) if text else None
            except json.JSONDecodeError:
                parsed = None
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return HttpResult(
                status_code=int(resp.status),
                text=text,
                json_body=parsed,
                elapsed_ms=elapsed_ms,
            )
    except HTTPError as e:
        raw = b""
        try:
            raw = e.read()
        except Exception:
            pass
        text = raw.decode("utf-8", errors="replace")
        parsed = None
        try:
            parsed = json.loads(text) if text else None
        except json.JSONDecodeError:
            parsed = None
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return HttpResult(
            status_code=int(e.code),
            text=text,
            json_body=parsed,
            elapsed_ms=elapsed_ms,
        )
    except (TimeoutError, socket.timeout) as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        raise ApiMonitorError(
            f"HTTP {method.upper()} {url} timeout({timeout_s}s, {elapsed_ms}ms): {e}"
        ) from e
    except URLError as e:
        raise ApiMonitorError(
            f"HTTP {method.upper()} {url} network failure: {e}"
        ) from e


class MonitorService:
    def __init__(
        self,
        *,
        db: Database,
        log_dir: str,
        detect_concurrency: int = 3,
        max_parallel_targets: int = 2,
        enable_log_cleanup: bool = True,
        log_max_bytes: int = 500 * 1024 * 1024,
    ) -> None:
        self.db = db
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.detect_concurrency = max(1, int(detect_concurrency))
        self.enable_log_cleanup = bool(enable_log_cleanup)
        self.log_max_bytes = max(0, int(log_max_bytes))
        self._target_executor = ThreadPoolExecutor(max_workers=max_parallel_targets)
        self._running_targets: set[int] = set()
        self._lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._active_log_files: set[str] = set()
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._started = False
        self._event_callback: Optional[Callable[[str, str], None]] = None

    def set_event_callback(self, callback: Callable[[str, str], None]) -> None:
        self._event_callback = callback

    def _emit_event(self, event_type: str, data: str) -> None:
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception:
                pass

    def start(self) -> None:
        if self._started:
            return
        self._scheduler.add_job(
            self.scan_due_targets,
            trigger="interval",
            minutes=1,
            id="scan_due_targets",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        self._started = True
        LOGGER.info("scheduler started")

    def stop(self) -> None:
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._target_executor.shutdown(wait=False, cancel_futures=False)
        self._started = False
        LOGGER.info("scheduler stopped")

    def running_target_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._running_targets)

    def is_target_running(self, target_id: int) -> bool:
        with self._lock:
            return target_id in self._running_targets

    def scan_due_targets(self) -> None:
        now_ts = time.time()
        due_targets = self.db.list_due_targets(now_ts)
        for target in due_targets:
            self.trigger_target(int(target["id"]), force=False)

    def trigger_target(self, target_id: int, force: bool = True) -> Tuple[bool, str]:
        target = self.db.get_target(target_id)
        if not target:
            return False, "target not found"
        if not force and not target.get("enabled", False):
            return False, "target disabled"

        with self._lock:
            if target_id in self._running_targets:
                return False, "target already running"
            self._running_targets.add(target_id)

        self._target_executor.submit(self._run_target_safe, target)
        return True, "target started"

    def _run_target_safe(self, target: Dict[str, Any]) -> None:
        target_id = int(target["id"])
        try:
            self._run_target(target)
        finally:
            with self._lock:
                self._running_targets.discard(target_id)

    def _run_target(self, target: Dict[str, Any]) -> None:
        target_id = int(target["id"])
        started_at = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
        log_file = str((self.log_dir / f"target_{target_id}_{ts}.jsonl").resolve())
        with self._lock:
            self._active_log_files.add(log_file)
        run_id = self.db.create_run(target_id=target_id, started_at=started_at, log_file=log_file)

        LOGGER.info("run start target=%s id=%d", target.get("name"), target_id)
        rows: List[Dict[str, Any]] = []
        try:
            models = self._get_models(target)
            max_models = int(target.get("max_models") or 0)
            if max_models > 0:
                models = models[:max_models]

            with open(log_file, "a", encoding="utf-8") as fp:
                with ThreadPoolExecutor(max_workers=self.detect_concurrency) as pool:
                    futures = [pool.submit(self._detect_one, target, model_id) for model_id in models]
                    for future in as_completed(futures):
                        row = future.result()
                        row["target_id"] = target_id
                        row["run_id"] = run_id
                        row["target_name"] = target.get("name")
                        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)

            self.db.insert_model_rows(run_id, target_id, rows)
            total = len(rows)
            success = sum(1 for item in rows if item.get("success"))
            fail = total - success
            if total == 0:
                target_status = "no_models"
            elif fail == 0:
                target_status = "healthy"
            elif success == 0:
                target_status = "down"
            else:
                target_status = "degraded"

            ended_at = time.time()
            self.db.finish_run(
                run_id,
                status="completed",
                finished_at=ended_at,
                total=total,
                success=success,
                fail=fail,
                error=None,
            )
            self.db.update_target_after_run(
                target_id,
                last_run_at=ended_at,
                last_status=target_status,
                last_total=total,
                last_success=success,
                last_fail=fail,
                last_log_file=log_file,
                last_error=None,
            )
            LOGGER.info(
                "run finished target=%s id=%d status=%s total=%d success=%d fail=%d",
                target.get("name"),
                target_id,
                target_status,
                total,
                success,
                fail,
            )
            self._emit_event("run_completed", json.dumps({
                "target_id": target_id,
                "target_name": target.get("name"),
                "status": target_status,
                "total": total,
                "success": success,
                "fail": fail,
            }))
        except Exception as e:
            ended_at = time.time()
            err = f"{type(e).__name__}: {e}"
            self.db.finish_run(
                run_id,
                status="error",
                finished_at=ended_at,
                total=0,
                success=0,
                fail=0,
                error=err,
            )
            self.db.update_target_after_run(
                target_id,
                last_run_at=ended_at,
                last_status="error",
                last_total=0,
                last_success=0,
                last_fail=0,
                last_log_file=log_file,
                last_error=err,
            )
            LOGGER.exception("run failed target=%s id=%d", target.get("name"), target_id)
        finally:
            with self._lock:
                self._active_log_files.discard(log_file)
            self._cleanup_data_logs()

    def _cleanup_data_logs(self) -> None:
        if not self.enable_log_cleanup or self.log_max_bytes <= 0:
            return

        with self._cleanup_lock:
            with self._lock:
                active_files = set(self._active_log_files)

            entries: List[Dict[str, Any]] = []
            for path in self.log_dir.glob("*.jsonl"):
                try:
                    resolved = str(path.resolve())
                except OSError:
                    resolved = str(path)
                if resolved in active_files:
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                entries.append(
                    {
                        "path": path,
                        "mtime": float(st.st_mtime),
                        "size": int(st.st_size),
                    }
                )

            entries.sort(key=lambda item: item["mtime"], reverse=True)

            total_bytes = sum(int(item["size"]) for item in entries)
            if total_bytes <= self.log_max_bytes:
                return

            to_delete: List[Dict[str, Any]] = []
            for item in reversed(entries):
                if total_bytes <= self.log_max_bytes:
                    break
                to_delete.append(item)
                total_bytes -= int(item["size"])

            deleted_files, deleted_bytes = self._delete_log_entries(to_delete)
            if deleted_files > 0:
                LOGGER.info(
                    "cleanup data/logs removed files=%d reclaimed=%.2fMB (max_mb=%d)",
                    deleted_files,
                    (deleted_bytes / 1024.0 / 1024.0),
                    int(self.log_max_bytes / 1024 / 1024),
                )

    @staticmethod
    def _delete_log_entries(entries: List[Dict[str, Any]]) -> Tuple[int, int]:
        deleted_files = 0
        deleted_bytes = 0
        for item in entries:
            path = item["path"]
            size = int(item.get("size") or 0)
            try:
                path.unlink()
                deleted_files += 1
                deleted_bytes += size
            except FileNotFoundError:
                continue
            except OSError as e:
                LOGGER.warning("cleanup skip file %s: %s", path, e)
        return deleted_files, deleted_bytes

    def _get_models(self, target: Dict[str, Any]) -> List[str]:
        base_url = normalize_base_url(str(target["base_url"]))
        timeout_s = float(target.get("timeout_s", 30.0))
        ssl_context = build_ssl_context(bool(target.get("verify_ssl", False)))
        url = f"{base_url}/v1/models"
        res = http_json(
            "GET",
            url,
            auth_headers(str(target["api_key"])),
            body=None,
            timeout_s=timeout_s,
            ssl_context=ssl_context,
        )
        if res.status_code != 200:
            msg = _check_response_body_for_error(res.json_body) or res.text[:500] or "unknown error"
            raise ApiMonitorError(f"GET /v1/models failed: HTTP {res.status_code} - {msg}")
        if not isinstance(res.json_body, dict):
            raise ApiMonitorError("models response must be JSON object")
        data = res.json_body.get("data")
        if not isinstance(data, list):
            raise ApiMonitorError("models response missing data[]")
        models = [item.get("id") for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
        if not models:
            raise ApiMonitorError("models list is empty")
        return models

    def _choose_route(self, model_id: str) -> str:
        actual = model_id.split("/", 1)[-1].lower()
        for pattern, route in ROUTE_RULES:
            if re.search(pattern, actual):
                return route
        return "chat"

    @staticmethod
    def _route_to_protocol(route: str) -> str:
        if route in ("chat", "responses"):
            return "openai"
        return route

    def _detect_one(self, target: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        route = self._choose_route(model_id)
        base_url = normalize_base_url(str(target["base_url"]))
        timeout_s = float(target.get("timeout_s", 30.0))
        ssl_context = build_ssl_context(bool(target.get("verify_ssl", False)))
        headers = auth_headers(str(target["api_key"]))
        prompt = str(target.get("prompt", "What is the exact model identifier (model string) you are using for this chat/session?"))
        anthropic_version = str(target.get("anthropic_version", "2025-09-29"))

        try:
            if route == "chat":
                url = f"{base_url}/v1/chat/completions"
                body = {
                    "model": model_id,
                    "stream": False,
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": prompt}],
                }
                res = http_json("POST", url, headers, body, timeout_s, ssl_context)
                return self._validate_detection_result(model_id, route, "chat", res, _extract_text_from_chat)

            if route == "responses":
                url = f"{base_url}/v1/responses"
                body = {
                    "model": model_id,
                    "stream": False,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                }
                res = http_json("POST", url, headers, body, timeout_s, ssl_context)
                return self._validate_detection_result(
                    model_id, route, "responses", res, _extract_text_from_responses
                )

            if route == "anthropic":
                url = f"{base_url}/v1/messages"
                ext_headers = dict(headers)
                ext_headers["anthropic-version"] = anthropic_version
                body = {
                    "model": model_id,
                    "stream": False,
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": prompt}],
                }
                res = http_json("POST", url, ext_headers, body, timeout_s, ssl_context)
                return self._validate_detection_result(
                    model_id, route, "messages", res, _extract_text_from_anthropic
                )

            if route == "gemini":
                segments = model_id.split("/")
                quoted_segments = [quote(seg, safe="-_.~") for seg in segments[:-1]]
                last = quote(f"{segments[-1]}:generateContent", safe="-_.~:")
                path = "/".join(quoted_segments + [last])
                url = f"{base_url}/v1beta/models/{path}"
                body = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 10},
                }
                res = http_json("POST", url, headers, body, timeout_s, ssl_context)
                return self._validate_detection_result(
                    model_id, route, "gemini", res, _extract_text_from_gemini
                )

            return self._build_fail_row(
                model_id=model_id,
                route=route,
                endpoint="unknown",
                message=f"unknown route: {route}",
                duration_s=0.0,
                status_code=None,
                transport_success=False,
            )
        except ApiMonitorError as e:
            return self._build_fail_row(
                model_id=model_id,
                route=route,
                endpoint=route,
                message=str(e),
                duration_s=0.0,
                status_code=None,
                transport_success=False,
            )
        except Exception as e:
            return self._build_fail_row(
                model_id=model_id,
                route=route,
                endpoint=route,
                message=f"unexpected error: {type(e).__name__}: {e}",
                duration_s=0.0,
                status_code=None,
                transport_success=False,
            )

    def _validate_detection_result(
        self,
        model_id: str,
        route: str,
        endpoint: str,
        res: HttpResult,
        extractor: Callable[[Any], Optional[str]],
    ) -> Dict[str, Any]:
        duration_s = max(0.0, res.elapsed_ms / 1000.0)
        if res.status_code != 200:
            msg = _check_response_body_for_error(res.json_body) or res.text[:500] or "unknown error"
            return self._build_fail_row(
                model_id=model_id,
                route=route,
                endpoint=endpoint,
                message=f"HTTP {res.status_code}: {msg}",
                duration_s=duration_s,
                status_code=res.status_code,
                transport_success=True,
            )

        body_error = _check_response_body_for_error(res.json_body)
        if body_error:
            return self._build_fail_row(
                model_id=model_id,
                route=route,
                endpoint=endpoint,
                message=f"response error: {body_error}",
                duration_s=duration_s,
                status_code=res.status_code,
                transport_success=True,
            )

        content = extractor(res.json_body)
        if not content:
            return self._build_fail_row(
                model_id=model_id,
                route=route,
                endpoint=endpoint,
                message="response parse failed: no readable text",
                duration_s=duration_s,
                status_code=res.status_code,
                transport_success=True,
            )

        return {
            "protocol": self._route_to_protocol(route),
            "model": model_id,
            "stream": False,
            "duration": duration_s,
            "success": True,
            "transport_success": True,
            "tool_calls_count": 0,
            "tool_calls": [],
            "tool_calls_json": "[]",
            "content": content,
            "timestamp": time.time(),
            "error": None,
            "status_code": res.status_code,
            "route": route,
            "endpoint": endpoint,
        }

    def _build_fail_row(
        self,
        *,
        model_id: str,
        route: str,
        endpoint: str,
        message: str,
        duration_s: float,
        status_code: Optional[int],
        transport_success: bool,
    ) -> Dict[str, Any]:
        return {
            "protocol": self._route_to_protocol(route),
            "model": model_id,
            "stream": False,
            "duration": max(0.0, duration_s),
            "success": False,
            "transport_success": transport_success,
            "tool_calls_count": 0,
            "tool_calls": [],
            "tool_calls_json": "[]",
            "content": "",
            "timestamp": time.time(),
            "error": message,
            "status_code": status_code,
            "route": route,
            "endpoint": endpoint,
        }
