"""
Tests for the ``benchmark-framework`` console entry point.
"""
from __future__ import annotations

import subprocess


def test_cli_help() -> None:
    result = subprocess.run(
        ["benchmark-framework", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Declarative, validation-first" in result.stdout
    for cmd in ("define", "download", "catalog", "validate", "export", "run", "demo"):
        assert cmd in result.stdout, f"Missing subcommand: {cmd}"


def test_cli_demo_help() -> None:
    result = subprocess.run(
        ["benchmark-framework", "demo", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--keep" in result.stdout


def test_cli_define_help() -> None:
    result = subprocess.run(
        ["benchmark-framework", "define", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--filter" in result.stdout


def test_cli_unknown_command_returns_error() -> None:
    result = subprocess.run(
        ["benchmark-framework", "doesnotexist"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
