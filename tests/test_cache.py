"""Tests for ec2_spot_query.cache — on-disk TTL cache."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ec2_spot_query import cache


class TestCachePaths:
    """Test cache path discovery logic."""

    def test_paths_return_home_first(self):
        """get_cache_paths() returns home path before project path."""
        paths = cache.get_cache_paths()
        home_path = Path.home() / ".cache" / cache.CACHE_FILENAME
        project_path = Path.cwd() / ".cache" / cache.CACHE_FILENAME
        assert len(paths) >= 1
        if len(paths) >= 2:
            assert paths[0] == home_path
            assert paths[1] == project_path

    def test_paths_has_at_least_one_entry(self):
        """get_cache_paths() always returns at least one path."""
        paths = cache.get_cache_paths()
        assert len(paths) >= 1

    def test_paths_have_correct_filename(self):
        """All paths end with the expected cache filename."""
        paths = cache.get_cache_paths()
        for p in paths:
            assert p.name == cache.CACHE_FILENAME


class TestCacheRoundtrip:
    """Test save and load with mocked filesystem."""

    def _make_mock_cache_dir(self, content: dict | None = None) -> Path:
        """Create a temporary directory with an optional pre-populated cache file."""
        tmpdir = Path(os.environ.get("PYTEST_CURRENT_TEST", "/tmp")) / ".test_cache_tmp"
        tmpdir.mkdir(exist_ok=True)
        if content is not None:
            (tmpdir / cache.CACHE_FILENAME).write_text(json.dumps(content))
        return tmpdir

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_cache and load_cache round-trip preserves data."""
        cache_file = tmp_path / "ec2-spot-cache.json"
        mock_paths = [cache_file]

        data = {"key": "value", "number": 42}
        with patch("ec2_spot_query.cache.get_cache_paths", return_value=mock_paths):
            cache.save_cache("test_key", data, ttl_seconds=3600)
            result = cache.load_cache("test_key")

        assert result == data

    def test_cache_expired_ttl(self, tmp_path):
        """Expired cache entries are not returned."""
        cache_file = tmp_path / "ec2-spot-cache.json"
        mock_paths = [cache_file]

        data = {"key": "value"}
        expired_at = time.time() - 7200  # 2 hours ago
        with patch("ec2_spot_query.cache.get_cache_paths", return_value=mock_paths):
            cache.save_cache("test_key", data, ttl_seconds=3600)

        # Manually corrupt the expiry to simulate expiration
        content = json.loads(cache_file.read_text())
        content["test_key"]["expires_at"] = expired_at
        cache_file.write_text(json.dumps(content))

        with patch("ec2_spot_query.cache.get_cache_paths", return_value=mock_paths):
            result = cache.load_cache("test_key")

        assert result is None

    def test_cache_valid_ttl(self, tmp_path):
        """Valid (non-expired) cache entries are returned."""
        cache_file = tmp_path / "ec2-spot-cache.json"
        mock_paths = [cache_file]

        data = {"key": "value"}
        with patch("ec2_spot_query.cache.get_cache_paths", return_value=mock_paths):
            cache.save_cache("test_key", data, ttl_seconds=3600)
            result = cache.load_cache("test_key")

        assert result == data

    def test_cache_miss_returns_none(self, tmp_path):
        """Loading a non-existent key returns None."""
        cache_file = tmp_path / "ec2-spot-cache.json"
        mock_paths = [cache_file]

        with patch("ec2_spot_query.cache.get_cache_paths", return_value=mock_paths):
            result = cache.load_cache("nonexistent_key")

        assert result is None

    def test_clear_cache_removes_file(self, tmp_path):
        """clear_cache() removes the cache file."""
        cache_file = tmp_path / "ec2-spot-cache.json"
        mock_paths = [cache_file]

        data = {"key": "value"}
        with patch("ec2_spot_query.cache.get_cache_paths", return_value=mock_paths):
            cache.save_cache("test_key", data, ttl_seconds=3600)
            assert cache_file.exists()
            with patch("ec2_spot_query.cache.get_cache_paths", return_value=mock_paths):
                cache.clear_cache()
            assert not cache_file.exists()
