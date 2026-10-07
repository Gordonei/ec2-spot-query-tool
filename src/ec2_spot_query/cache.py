"""On-disk TTL cache for EC2 API results."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

CACHE_FILENAME = "ec2-spot-cache.json"
SPOT_TTL_SECONDS = 3600  # 1 hour
INSTANCE_TTL_SECONDS = 31536000  # 1 year (365 days)


def get_cache_paths() -> list[Path]:
    """Return ordered list of candidate cache file paths.

    Tries ``~/.cache/ec2-spot-cache.json`` first, then
    ``./.cache/ec2-spot-cache.json``. Falls back to the home path if
    neither is writable.
    """
    candidates: list[Path] = []
    home_dir = Path.home() / ".cache"
    if _is_writable_or_can_become(home_dir):
        candidates.append(home_dir / CACHE_FILENAME)
    project_dir = Path.cwd() / ".cache"
    if _is_writable_or_can_become(project_dir):
        candidates.append(project_dir / CACHE_FILENAME)
    if not candidates:
        candidates.append(home_dir / CACHE_FILENAME)
    return candidates


def _is_writable_or_can_become(path: Path) -> bool:
    """Return True if *path* exists and is writable, or can be created."""
    if path.exists():
        return os.access(path, os.W_OK)
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".test_writable"
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


def load_cache(key: str, ttl_seconds: float = SPOT_TTL_SECONDS) -> Any | None:
    """Load a cached value by *key*.

    Returns ``None`` if the key is not found or the entry has expired.
    """
    for path in get_cache_paths():
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if key in data:
                    entry = data[key]
                    if time.time() < entry.get("expires_at", 0):
                        return entry["data"]
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return None


def save_cache(key: str, data: Any, ttl_seconds: float = SPOT_TTL_SECONDS) -> None:
    """Save *data* to disk under *key* with the given TTL."""
    entry = {
        "data": data,
        "expires_at": time.time() + ttl_seconds,
    }
    for path in get_cache_paths():
        try:
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)
            existing: dict = {}
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, ValueError):
                    existing = {}
            existing[key] = entry
            path.write_text(json.dumps(existing, default=str), encoding="utf-8")
            return
        except (OSError, PermissionError, TypeError, ValueError):
            continue


def clear_cache() -> None:
    """Remove the cache file at the first writable path."""
    for path in get_cache_paths():
        try:
            if path.exists():
                path.unlink()
            return
        except (OSError, PermissionError):
            continue
