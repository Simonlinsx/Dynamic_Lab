"""SimToolReal tasks for IsaacLab."""

from __future__ import annotations

import os

import toml

SIMTOOLREAL_LAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
SIMTOOLREAL_LAB_METADATA = toml.load(os.path.join(SIMTOOLREAL_LAB_EXT_DIR, "config", "extension.toml"))
__version__ = SIMTOOLREAL_LAB_METADATA["package"]["version"]

from . import tasks  # noqa: E402,F401
