"""On-disk TTL cache for EC2 API results."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CACHE_FILENAME = "ec2-spot-cache.json"
SPOT_TTL_SECONDS = 3600  # 1 hour
INSTANCE_TTL_SECONDS = 31536000  # 1 year (365 days)
REGIONS_TTL_SECONDS = 2592000  # 1 month (30 days)


def get_cache_path() -> Path:
    """Return the single cache file path.

    Tries ``~/.cache/ec2-spot-cache.json`` first, then
    ``./.cache/ec2-spot-cache.json``. Falls back to the home path if
    neither is writable.
    """
    home_dir = Path.home() / ".cache"
    if _is_writable_or_can_become(home_dir):
        return home_dir / CACHE_FILENAME
    project_dir = Path.cwd() / ".cache"
    if _is_writable_or_can_become(project_dir):
        return project_dir / CACHE_FILENAME
    return home_dir / CACHE_FILENAME


@contextmanager
def _open_cache(path: Path, read_only: bool = True):
    """Context manager for cache file I/O.

    Yields the parsed JSON dict, or ``{}`` when the file is missing or
    contains corrupt JSON.  When *read_only* is False, modifications to
    the yielded dict are persisted to disk on exit (the parent directory
    is created automatically).
    """
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            data = {}
    yield data
    if not read_only:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, default=str), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            pass


def _is_writable_or_can_become(path: Path) -> bool:
    """Return True if *path* exists and is writable, or can be created."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(path, os.W_OK)


def load_cache(key: str, ttl_seconds: float = SPOT_TTL_SECONDS) -> Any | None:
    """Load a cached value by *key*.

    Returns ``None`` if the key is not found or the entry has expired.
    """
    with _open_cache(get_cache_path(), read_only=True) as data:
        entry = data.get(key)
        if entry is not None and time.time() < entry.get("expires_at", 0):
            return entry["data"]
    return None


def save_cache(key: str, data: Any, ttl_seconds: float = SPOT_TTL_SECONDS) -> None:
    """Save *data* to disk under *key* with the given TTL."""
    entry = {
        "data": data,
        "expires_at": time.time() + ttl_seconds,
    }
    with _open_cache(get_cache_path(), read_only=False) as store:
        store[key] = entry


def make_spot_cache_key(instance_types: list[str], regions: list[str]) -> str:
    """Build a deterministic cache key from instance types and regions.

    Both lists are sorted so that different orderings produce the same key.
    """
    sorted_instances = ",".join(sorted(instance_types))
    sorted_regions = ",".join(sorted(regions))
    return f"spot:{sorted_instances}:{sorted_regions}"
