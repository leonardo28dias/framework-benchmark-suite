"""
Structured logging for the Benchmark Suite pipeline.

Human-readable progress output keeps using print() on stdout; every lifecycle
event and every failure is additionally emitted as one compact JSON object per
line on stderr. Each record carries the run id, the active stage and the
affected collection/dataset, so a failure can be traced back to:

    which run failed -> run_id
    which collection failed -> collection_id
    which stage failed -> stage
    why did it fail -> operation + outcome + error

Run ids are generated once per pipeline execution (see configure_logging) and
propagated through every stage that runs inside that execution. Stages write
no secrets: only identifiers, counters and error types/messages are logged.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator

LOGGER_NAME = "benchmark_suite"

_run_id_var: ContextVar[str | None] = ContextVar("benchmark_run_id", default=None)
_stage_var: ContextVar[str | None] = ContextVar("benchmark_stage", default=None)

# LogRecord attributes that belong to the logging protocol itself and must not
# be duplicated into the JSON payload.
_RESERVED_RECORD_FIELDS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName",
})


class JsonFormatter(logging.Formatter):
    """Formats records as one compact JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "run_id": getattr(record, "run_id", None),
            "stage": getattr(record, "stage", None),
            "event": getattr(record, "event", None),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_FIELDS or key in payload or value is None:
                continue
            payload[key] = value
        if record.exc_info is not None and record.exc_info[0] is not None:
            payload.setdefault("error", f"{record.exc_info[0].__name__}: {record.exc_info[1]}")
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Adds the active run id and stage to every record logged in this process."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "run_id", None) is None:
            record.run_id = _run_id_var.get()
        if getattr(record, "stage", None) is None:
            record.stage = _stage_var.get()
        return True


class JsonStreamHandler(logging.StreamHandler):
    """Marker subclass so the handler owned by this module can be replaced.

    When no explicit stream is given, the current ``sys.stderr`` is resolved on
    every emit, so swapping ``sys.stderr`` (as test harnesses do) keeps working.
    """

    def __init__(self, stream: Any = None) -> None:
        self._dynamic_stream = stream is None
        super().__init__(stream)

    def emit(self, record: logging.LogRecord) -> None:
        if self._dynamic_stream:
            self.stream = sys.stderr
        super().emit(record)



def new_run_id() -> str:
    """Generates one short random identifier for a single pipeline execution."""
    return uuid.uuid4().hex[:12]


def get_run_id() -> str | None:
    """Returns the run id active in the current execution context."""
    return _run_id_var.get()


def get_stage() -> str | None:
    """Returns the pipeline stage active in the current execution context."""
    return _stage_var.get()


def get_logger() -> logging.Logger:
    """Returns the single shared logger every stage logs through."""
    return logging.getLogger(LOGGER_NAME)


def configure_logging(
    run_id: str | None = None,
    *,
    level: str | None = None,
    stream: Any = None,
) -> str:
    """Installs the JSON log handler (idempotent) and activates a run id.

    A run id is generated only when none is already active in the current
    context, so an orchestrator can generate one run id and every stage it
    calls keeps logging under it. Standalone stage executions get a fresh id.
    """
    logger = get_logger()
    logger.setLevel((level or os.environ.get("LOG_LEVEL", "INFO")).upper())
    logger.propagate = False

    for existing in list(logger.handlers):
        if isinstance(existing, JsonStreamHandler):
            logger.removeHandler(existing)
            existing.close()
    for existing in list(logger.filters):  # type: ignore
        if isinstance(existing, ContextFilter):
            logger.removeFilter(existing)

    handler = JsonStreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.addFilter(ContextFilter())

    active = run_id or _run_id_var.get() or new_run_id()
    _run_id_var.set(active)
    return active


def reset_logging() -> None:
    """Detaches owned handlers/filters and clears context (used by tests)."""
    logger = get_logger()
    for handler in list(logger.handlers):
        if isinstance(handler, JsonStreamHandler):
            logger.removeHandler(handler)
            handler.close()
    for existing in list(logger.filters):
        if isinstance(existing, ContextFilter):
            logger.removeFilter(existing)
    _run_id_var.set(None)
    _stage_var.set(None)


def log_event(event: str, *, level: int = logging.INFO, message: str | None = None, **fields: Any) -> None:
    """Emits one structured record; ``fields`` become additional JSON fields."""
    get_logger().log(level, message or event, extra={"event": event, **fields})


class StageOutcome:
    """Lets a stage report a handled failure that must not raise."""

    def __init__(self) -> None:
        self.failed = False
        self._context_fields: dict[str, Any] = {}

    def _set_context_fields(self, **fields: Any) -> None:
        self._context_fields = fields

    def fail(self, message: str, **fields: Any) -> None:
        """Marks the stage as failed and logs why.

        Any fields set on the stage context (e.g. collection_id) are automatically
        included so callers do not need to repeat them.
        """
        self.failed = True
        merged = {**self._context_fields, **fields}
        log_event("stage_failed", level=logging.ERROR, message=message, failure_reason=message, **merged)


@contextlib.contextmanager
def stage_context(stage: str, **fields: Any) -> Iterator[StageOutcome]:
    """Runs a block as a pipeline stage, logging start/success/failure.

    Fields passed here are attached to every lifecycle event for this stage
    (start, success, failure), so contextual information like collection_id
    propagates automatically to fail() calls within the block.
    """
    token = _stage_var.set(stage)
    outcome = StageOutcome()
    outcome._set_context_fields(**fields)
    log_event("stage_started", stage=stage, **fields)
    try:
        yield outcome
    except BaseException as exc:
        if not outcome.failed:
            log_event(
                "stage_failed",
                level=logging.ERROR,
                stage=stage,
                outcome="failure",
                error=f"{type(exc).__name__}: {exc}",
                **fields,
            )
        raise
    else:
        if not outcome.failed:
            log_event("stage_completed", stage=stage, outcome="success", **fields)
    finally:
        _stage_var.reset(token)
