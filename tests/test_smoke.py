"""Smoke tests that run without the GPU simulator or the custom SB3/ManiSkill2
forks installed.

The heavy RL training/eval paths depend on ManiSkill2 + Sapien + a CUDA GPU
and cannot realistically run in CI, so these tests only verify that the
top-level package is structured correctly.
"""

from __future__ import annotations

import importlib


def test_package_import() -> None:
    """The top-level package imports without requiring ML dependencies."""
    mod = importlib.import_module("rma4rma")
    assert hasattr(mod, "__version__")
    assert isinstance(mod.__version__, str)
