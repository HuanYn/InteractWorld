"""Portable data-root selection without relaxing the pinned model revision."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_DATA_ROOT = "/path/to/interactworld"
PINNED_MODEL_DIR = "Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"


def _absolute_path_parts(value: str):
    windows = PureWindowsPath(value)
    return windows if windows.drive else PurePosixPath(value)


def project_root(value: str | Path | None = None) -> Path:
    """Explicit path wins over INTERACTWORLD_ROOT, then the private legacy root.

    This validates a path, not a physical mount. Operators must select a data
    disk with adequate space; a root directory or relative path is rejected.
    """
    raw = str(value if value is not None else os.environ.get("INTERACTWORLD_ROOT", DEFAULT_DATA_ROOT))
    pure = _absolute_path_parts(raw)
    if not pure.is_absolute() or len(pure.parts) <= 1 or ".." in pure.parts:
        raise ValueError("project root must be an explicit absolute non-root directory on the data disk")
    return Path(raw)


def pinned_base_model_path(root: str | Path | None = None) -> Path:
    return project_root(root) / "models" / PINNED_MODEL_DIR


def is_pinned_base_model(value: str | Path) -> bool:
    """Allow relocation, but never a different model/revision or relative name."""
    pure = _absolute_path_parts(str(value))
    return pure.is_absolute() and ".." not in pure.parts and pure.name == PINNED_MODEL_DIR
