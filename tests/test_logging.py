"""
Tests for structured JSON logging: run ID propagation, stage context, and
the contextual fields that make failures traceable.
"""
from __future__ import annotations

import io
import json
import logging

import pytest

from logging_utils import (
    configure_logging,
    get_run_id,
    log_event,
    reset_logging,
    stage_context,
    new_run_id,
)


@pytest.fixture
def json_log_stream():
    """Configures logging into a fresh buffer and returns it."""
    reset_logging()
    buffer = io.StringIO()
    run_id = configure_logging(stream=buffer)
    return buffer, run_id


def test_configure_logging_returns_a_run_id(json_log_stream):
    buffer, run_id = json_log_stream
    assert len(run_id) == 12


def test_log_event_emits_json_with_run_id(json_log_stream):
    buffer, run_id = json_log_stream
    log_event("test_event", collection_id="my_col", dataset_id="ds_1")

    record = json.loads(buffer.getvalue().strip().split("\n")[-1])
    assert record["run_id"] == run_id
    assert record["event"] == "test_event"
    assert record["level"] == "INFO"
    assert record["collection_id"] == "my_col"
    assert record["dataset_id"] == "ds_1"
    assert record["stage"] is None


def test_stage_context_propagates_stage_and_run_id(json_log_stream):
    buffer, run_id = json_log_stream

    with stage_context("download", collection_id="my_col"):
        log_event("dataset_skipped", dataset_id="ds_1", outcome="skipped")

    lines = buffer.getvalue().strip().split("\n")
    started = json.loads(lines[0])
    event = json.loads(lines[1])
    completed = json.loads(lines[2])

    assert started["event"] == "stage_started"
    assert started["stage"] == "download"
    assert started["collection_id"] == "my_col"
    assert started["run_id"] == run_id

    assert event["event"] == "dataset_skipped"
    assert event["stage"] == "download"
    assert event["run_id"] == run_id

    assert completed["event"] == "stage_completed"
    assert completed["stage"] == "download"
    assert completed["outcome"] == "success"
    assert completed["run_id"] == run_id


def test_stage_context_logs_failure_on_exception(json_log_stream):
    buffer, run_id = json_log_stream

    with pytest.raises(RuntimeError, match="boom"):
        with stage_context("catalog"):
            raise RuntimeError("boom")

    lines = buffer.getvalue().strip().split("\n")
    started = json.loads(lines[0])
    failed = json.loads(lines[1])

    assert started["event"] == "stage_started"
    assert started["stage"] == "catalog"
    assert failed["event"] == "stage_failed"
    assert failed["outcome"] == "failure"
    assert "RuntimeError: boom" in failed["error"]
    assert failed["run_id"] == run_id


def test_stage_outcome_fail_logs_and_marks_failed(json_log_stream):
    buffer, run_id = json_log_stream

    with stage_context("validate", collection_id="col") as stage:
        stage.fail("dataset rejected", dataset_id="bad_ds")

    lines = buffer.getvalue().strip().split("\n")
    started = json.loads(lines[0])
    failed = json.loads(lines[1])

    assert started["event"] == "stage_started"
    assert failed["event"] == "stage_failed"
    assert failed["collection_id"] == "col"
    assert failed["dataset_id"] == "bad_ds"
    assert "dataset rejected" in failed["message"]


def test_get_run_id_is_none_without_configure():
    reset_logging()
    assert get_run_id() is None


def test_reset_logging_clears_context():
    reset_logging()
    run_id = configure_logging()
    assert get_run_id() == run_id
    reset_logging()
    assert get_run_id() is None


def test_error_records_include_exception_info(json_log_stream):
    buffer, run_id = json_log_stream

    logger = logging.getLogger("benchmark_suite")
    try:
        raise ValueError("test error detail")
    except ValueError:
        logger.error("something went wrong", exc_info=True, extra={"event": "op_failed", "collection_id": "col"})

    record = json.loads(buffer.getvalue().strip().split("\n")[-1])
    assert record["event"] == "op_failed"
    assert record["level"] == "ERROR"
    assert "ValueError" in record["error"]
    assert "test error detail" in record["error"]


def test_run_id_is_deterministic_per_session(json_log_stream):
    buffer1, run_id1 = json_log_stream
    reset_logging()

    buffer2 = io.StringIO()
    run_id2 = configure_logging(stream=buffer2)

    assert run_id1 != run_id2


def test_new_run_id_is_short_and_unique():
    id1 = new_run_id()
    id2 = new_run_id()
    assert id1 != id2
    assert len(id1) == 12
    assert len(id2) == 12
