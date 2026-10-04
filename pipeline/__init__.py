"""High-level package for the Benchmark Suite framework.

The package exposes a CLI entry point (``benchmark-framework``) and an offline
demo module. The pipeline stages themselves live in :mod:`scripts` and keep
their flat, import-by-module-name structure.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
