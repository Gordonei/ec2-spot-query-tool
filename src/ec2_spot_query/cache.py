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


@contextmanager
def _open_cache(path: Path, mode: str = "r"):
    """Context manager for cache file I/O.

    In *read* mode (``"r"``), yields the parsed JSON dict or ``None`` if
    the file does not exist or contains corrupt JSON.

    In *write* mode (``"w"``), yields an empty dict or the existing data
    dict.  Modifications to the yielded dict are persisted to disk when
    the context exits (parent directory is created automatically).
    """
    if mode == "r":
        if not path.exists():
            yield None
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            yield data
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            yield None
    elif mode == "w":
        parent = path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            yield {}
            return
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError, TypeError):
                existing = {}
        yield existing
        try:
            path.write_text(json.dumps(existing, default=str), encoding="utf-8")
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
    for path in get_cache_paths():
        with _open_cache(path, mode="r") as data:
            if data is not None and key in data:
                entry = data[key]
                if time.time() < entry.get("expires_at", 0):
                    return entry["data"]
    return None


def save_cache(key: str, data: Any, ttl_seconds: float = SPOT_TTL_SECONDS) -> None:
    """Save *data* to disk under *key* with the given TTL."""
    entry = {
        "data": data,
        "expires_at": time.time() + ttl_seconds,
    }
    for path in get_cache_paths():
        with _open_cache(path, mode="w") as existing:
            existing[key] = entry
        return


def clear_cache() -> None:
    """Remove the cache file at the first writable path."""
    for path in get_cache_paths():
        try:
            if path.exists():
                path.unlink()
            return
        except (OSError, PermissionError):
            continue


def make_spot_cache_key(instance_types: list[str], regions: list[str]) -> str:
    """Build a deterministic cache key from instance types and regions.

    Both lists are sorted so that different orderings produce the same key.
    """
    sorted_instances = ",".join(sorted(instance_types))
    sorted_regions = ",".join(sorted(regions))
    return f"spot:{sorted_instances}:{sorted_regions}"
