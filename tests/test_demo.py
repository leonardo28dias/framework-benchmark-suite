"""
Test that the offline demo runs successfully end-to-end.
"""
from __future__ import annotations

import subprocess
import sys


def test_demo_completes_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.demo"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Demo failed:\n{result.stderr}"
    assert "Demo complete" in result.stdout


def test_demo_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.demo", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "offline demo" in result.stdout.lower()
