"""Filesystem layout helpers for the Recorder.

Trajectories live under
``~/Library/Application Support/Trace[-dev]/trajectories/<uuid>/``.

The ``Trace`` vs ``Trace-dev`` choice is governed by the ``TRACE_DEV_MODE``
environment variable (set to ``"1"`` for Ralph iterations and developer
machines so the production directory is never touched accidentally — see
``CLAUDE.md`` Safety Invariants).  Tests point at a temporary directory by
setting ``TRACE_PROFILE_DIR`` directly.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path

__all__ = [
    "DEV_PROFILE",
    "PROD_PROFILE",
    "default_trajectories_root",
    "ensure_trajectories_root",
    "trajectory_dir",
]

PROD_PROFILE = "Trace"
DEV_PROFILE = "Trace-dev"


def default_trajectories_root() -> Path:
    """Return the trajectories root directory for the current process.

    Resolution order:

    1. ``TRACE_PROFILE_DIR`` — explicit override (used by tests to point at
       a ``tmp_path``).  The value is treated as the *profile* directory;
       trajectories live in ``<value>/trajectories/``.
    2. ``TRACE_DEV_MODE=1`` — points at ``~/Library/Application Support/Trace-dev``
       (Ralph + developer machines).
    3. Default — ``~/Library/Application Support/Trace`` (production).
    """
    override = os.environ.get("TRACE_PROFILE_DIR")
    if override:
        return Path(override) / "trajectories"
    profile = DEV_PROFILE if os.environ.get("TRACE_DEV_MODE") == "1" else PROD_PROFILE
    return Path.home() / "Library" / "Application Support" / profile / "trajectories"


def ensure_trajectories_root(root: Path) -> Path:
    """Create the trajectories root with directory perms ``0700``.

    Safe to call repeatedly.  Returns the root for chaining.
    """
    root.mkdir(parents=True, exist_ok=True)
    # chmod is best-effort: on some filesystems (FAT, network mounts) the bits
    # don't stick — we still want the directory to exist so capture can
    # proceed, even though the safety invariant is weakened.
    with contextlib.suppress(OSError):
        os.chmod(root, 0o700)
    return root


def trajectory_dir(root: Path, trajectory_id: str) -> Path:
    """Return the on-disk directory for ``trajectory_id`` under ``root``.

    This does NOT validate that the directory exists; callers should check
    with :py:meth:`pathlib.Path.is_dir`.
    """
    return root / trajectory_id


def remove_trajectory(root: Path, trajectory_id: str) -> bool:
    """Remove a trajectory directory.  Returns ``True`` if it existed."""
    target = trajectory_dir(root, trajectory_id)
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True
