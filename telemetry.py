"""In-process telemetry: request IDs, RED metrics and LLM call records.

Questions this module exists to answer (on-call / admin dashboard):
  1. Is the API healthy right now? (request rate, 5xx rate, p95 latency per route)
  2. Is the LLM provider slow or failing? (latency, time-to-first-token, error rate)
  3. Which model tier is doing the work, and how much output is it producing?
  4. What happened in one specific request? (request_id on every log line)

Storage is bounded in-memory ring buffers, so memory use is fixed and nothing
leaves the process. Numbers reset on restart; this is a live operational view,
not a long-term store. Labels are restricted to bounded sets (route template,
method, status class, task, tier, model) to keep cardinality fixed.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------------
# Correlation ID
# --------------------------------------------------------------------------

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Attach the current request_id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "request_id"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra fields passed via ``extra={...}`` are
    included, so call sites can log ``event=...`` plus structured context."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [%(request_id)s] %(message)s")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class RequestRecord:
    ts: float
    method: str
    route: str
    status: int
    duration_ms: float


@dataclass
class LLMCallRecord:
    ts: float
    task: str
    tier: str
    model: str
    status: str  # ok | error | cancelled
    latency_ms: float
    ttft_ms: float | None = None
    output_chars: int = 0
    output_tokens: int | None = None
    tokens_estimated: bool = True
    error_type: str | None = None
    request_id: str = field(default_factory=lambda: request_id_var.get())


MAX_REQUESTS = 5000
MAX_LLM_CALLS = 1000

_lock = threading.Lock()
_requests: deque[RequestRecord] = deque(maxlen=MAX_REQUESTS)
_llm_calls: deque[LLMCallRecord] = deque(maxlen=MAX_LLM_CALLS)
_started_at = time.time()


def record_request(method: str, route: str, status: int, duration_ms: float) -> None:
    with _lock:
        _requests.append(RequestRecord(time.time(), method, route, status, duration_ms))


def record_llm_call(rec: LLMCallRecord) -> None:
    with _lock:
        _llm_calls.append(rec)


def reset() -> None:
    """Test helper."""
    with _lock:
        _requests.clear()
        _llm_calls.clear()


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100). None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return round(ordered[k], 1)


def _latency_summary(values: list[float]) -> dict:
    return {"p50": percentile(values, 50), "p95": percentile(values, 95), "p99": percentile(values, 99)}


def _status_class(status: int) -> str:
    return f"{status // 100}xx"


def _minute_series(timestamps_and_flags: list[tuple[float, bool]], now: float, minutes: int) -> list[dict]:
    """Per-minute buckets for the last ``minutes`` minutes: total and error counts."""
    start = int(now // 60) - minutes + 1
    buckets = {start + i: [0, 0] for i in range(minutes)}
    for ts, is_error in timestamps_and_flags:
        minute = int(ts // 60)
        if minute in buckets:
            buckets[minute][0] += 1
            if is_error:
                buckets[minute][1] += 1
    return [{"t": m * 60, "count": c, "errors": e} for m, (c, e) in sorted(buckets.items())]


def summary(window_minutes: int = 60) -> dict:
    """Aggregate metrics over a trailing window for the admin dashboard."""
    now = time.time()
    cutoff = now - window_minutes * 60
    with _lock:
        reqs = [r for r in _requests if r.ts >= cutoff]
        calls = [c for c in _llm_calls if c.ts >= cutoff]

    # ---- HTTP (RED) ----
    errors = [r for r in reqs if r.status >= 500]
    by_route: dict[tuple[str, str], list[RequestRecord]] = {}
    for r in reqs:
        by_route.setdefault((r.method, r.route), []).append(r)
    routes = []
    for (method, route), items in by_route.items():
        durations = [i.duration_ms for i in items]
        routes.append(
            {
                "method": method,
                "route": route,
                "count": len(items),
                "errors": sum(1 for i in items if i.status >= 500),
                "client_errors": sum(1 for i in items if 400 <= i.status < 500),
                "latency_ms": _latency_summary(durations),
            }
        )
    routes.sort(key=lambda x: x["count"], reverse=True)

    status_classes: dict[str, int] = {}
    for r in reqs:
        key = _status_class(r.status)
        status_classes[key] = status_classes.get(key, 0) + 1

    # ---- LLM ----
    def group(key: str) -> list[dict]:
        groups: dict[str, list[LLMCallRecord]] = {}
        for c in calls:
            groups.setdefault(getattr(c, key), []).append(c)
        out = []
        for name, items in groups.items():
            ok = [i for i in items if i.status == "ok"]
            out.append(
                {
                    key: name,
                    "count": len(items),
                    "errors": sum(1 for i in items if i.status == "error"),
                    "latency_ms": _latency_summary([i.latency_ms for i in ok]),
                    "output_tokens": sum(i.output_tokens or 0 for i in items),
                }
            )
        out.sort(key=lambda x: x["count"], reverse=True)
        return out

    ok_calls = [c for c in calls if c.status == "ok"]
    ttfts = [c.ttft_ms for c in ok_calls if c.ttft_ms is not None]

    return {
        "generated_at": now,
        "window_minutes": window_minutes,
        "uptime_seconds": round(now - _started_at),
        "http": {
            "requests": len(reqs),
            "errors": len(errors),
            "error_rate": round(len(errors) / len(reqs), 4) if reqs else 0.0,
            "rate_per_min": round(len(reqs) / window_minutes, 2),
            "latency_ms": _latency_summary([r.duration_ms for r in reqs]),
            "status_classes": status_classes,
            "routes": routes[:20],
            "series": _minute_series([(r.ts, r.status >= 500) for r in reqs], now, min(window_minutes, 60)),
        },
        "llm": {
            "calls": len(calls),
            "errors": sum(1 for c in calls if c.status == "error"),
            "cancelled": sum(1 for c in calls if c.status == "cancelled"),
            "error_rate": round(sum(1 for c in calls if c.status == "error") / len(calls), 4) if calls else 0.0,
            "latency_ms": _latency_summary([c.latency_ms for c in ok_calls]),
            "ttft_ms": _latency_summary(ttfts),
            "output_tokens": sum(c.output_tokens or 0 for c in calls),
            "tokens_estimated": any(c.tokens_estimated for c in calls),
            "by_tier": group("tier"),
            "by_task": group("task"),
            "by_model": group("model"),
            "series": _minute_series([(c.ts, c.status == "error") for c in calls], now, min(window_minutes, 60)),
            "recent": [asdict(c) for c in list(calls)[-25:][::-1]],
        },
    }
