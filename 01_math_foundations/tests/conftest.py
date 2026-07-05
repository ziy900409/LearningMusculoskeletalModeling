"""Pytest configuration for the Stage-1 test suite.

Puts ``../src`` on the import path so tests can ``import dynamics_utils`` without
the folder being installed as a package.
"""
from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
