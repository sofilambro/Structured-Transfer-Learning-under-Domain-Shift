"""
Pytest configuration.

Puts ``src/`` on the path so the tests run from a bare checkout without
``pip install -e .`` first. Installing the package is still the recommended
workflow -- this only removes a stumbling block for a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
