"""Profile and subdirectory path resolution for the runner service.

The runner reads skills from and writes runs under
``~/Library/Application Support/Trace[-dev]/``. Tests override the root via
the ``TRACE_PROFILE_ROOT`` environment variable so no real user data is
touched. The dev profile suffix (``Trace-dev``) is the Ralph default; the prod
profile is selected when ``TRACE_PROFILE`` is set to ``"prod"``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROFILE_ROOT_ENV_VAR: Final[str] = "TRACE_PROFILE_ROOT"
PROFILE_ENV_VAR: Final[str] = "TRACE_PROFILE"

_PROD_DIRNAME: Final[str] = "Trace"
_DEV_DIRNAME: Final[str] = "Trace-dev"


def profile_root() -> Path:
    """Return the profile root directory.

    Precedence: explicit ``TRACE_PROFILE_ROOT`` > dev/prod default derived
    from ``TRACE_PROFILE``. ``TRACE_PROFILE=prod`` selects ``Trace``;
    anything else (including unset) selects ``Trace-dev``.
    """
    explicit = os.environ.get(PROFILE_ROOT_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    dirname = _PROD_DIRNAME if os.environ.get(PROFILE_ENV_VAR) == "prod" else _DEV_DIRNAME
    return Path.home() / "Library" / "Application Support" / dirname


def runs_root() -> Path:
    return profile_root() / "runs"


def skills_root() -> Path:
    return profile_root() / "skills"


def trajectories_root() -> Path:
    return profile_root() / "trajectories"


def costs_path() -> Path:
    return profile_root() / "costs.jsonl"


__all__ = [
    "PROFILE_ENV_VAR",
    "PROFILE_ROOT_ENV_VAR",
    "costs_path",
    "profile_root",
    "runs_root",
    "skills_root",
    "trajectories_root",
]
