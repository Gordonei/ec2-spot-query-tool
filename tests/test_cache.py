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


class TestCacheContextManager:
    """Test the _open_cache context manager."""

    def test_open_cache_valid_json(self, tmp_path):
        """_open_cache yields parsed JSON dict for valid file."""
        cache_file = tmp_path / cache.CACHE_FILENAME
        cache_file.write_text(json.dumps({"existing": "data"}), encoding="utf-8")

        with cache._open_cache(cache_file, mode="r") as data:
            assert data == {"existing": "data"}

    def test_open_cache_missing_file(self, tmp_path):
        """_open_cache yields None when file does not exist."""
        cache_file = tmp_path / cache.CACHE_FILENAME
        # Don't create the file

        with cache._open_cache(cache_file, mode="r") as data:
            assert data is None

    def test_open_cache_corrupt_json(self, tmp_path):
        """_open_cache yields None when file contains corrupt JSON."""
        cache_file = tmp_path / cache.CACHE_FILENAME
        cache_file.write_text("not valid json{{{", encoding="utf-8")

        with cache._open_cache(cache_file, mode="r") as data:
            assert data is None

    def test_open_cache_write_mode_creates_dir(self, tmp_path):
        """_open_cache write mode creates parent directory if needed."""
        nested = tmp_path / "deep" / "nested" / cache.CACHE_FILENAME
        assert not nested.parent.exists()

        with cache._open_cache(nested, mode="w") as data:
            assert isinstance(data, dict)
            assert nested.parent.exists()

    def test_open_cache_write_mode_persists_data(self, tmp_path):
        """_open_cache write mode persists modified data after context exit."""
        cache_file = tmp_path / cache.CACHE_FILENAME

        with cache._open_cache(cache_file, mode="w") as data:
            data["new_key"] = "new_value"

        content = json.loads(cache_file.read_text(encoding="utf-8"))
        assert content == {"new_key": "new_value"}

    def test_open_cache_write_mode_preserves_existing(self, tmp_path):
        """_open_cache write mode preserves existing entries."""
        cache_file = tmp_path / cache.CACHE_FILENAME
        cache_file.write_text(json.dumps({"existing": "data"}), encoding="utf-8")

        with cache._open_cache(cache_file, mode="w") as data:
            data["new_key"] = "new_value"

        content = json.loads(cache_file.read_text(encoding="utf-8"))
        assert content == {"existing": "data", "new_key": "new_value"}


class TestCacheWritable:
    """Test _is_writable_or_can_become."""

    def test_writable_existing_path(self, tmp_path):
        """Existing writable path returns True."""
        writable_dir = tmp_path / "writable"
        writable_dir.mkdir()
        assert cache._is_writable_or_can_become(writable_dir) is True

    def test_writable_nonexistent_creates_dir(self, tmp_path):
        """Non-existent path creates directory and returns True when writable."""
        new_path = tmp_path / "new_dir"
        assert not new_path.exists()
        result = cache._is_writable_or_can_become(new_path)
        assert result is True
        assert new_path.exists()

    def test_writable_nonexistent_no_permission(self, tmp_path):
        """Non-existent path that cannot be created returns False."""
        parent = tmp_path / "parent"
        parent.mkdir()
        # Make parent read-only so we can't create inside it
        parent.chmod(0o444)
        path_to_test = parent / "should_fail"
        try:
            result = cache._is_writable_or_can_become(path_to_test)
            assert result is False
        finally:
            parent.chmod(0o755)  # Restore for cleanup


class TestCacheKey:
    """Test cache key includes region."""

    def test_spot_cache_key_includes_regions(self):
        """Cache key format includes regions."""
        key = cache.make_spot_cache_key(["t3.micro", "t3.large"], ["us-east-1", "eu-west-1"])
        assert "us-east-1" in key
        assert "eu-west-1" in key
        assert "t3.micro" in key
        assert "t3.large" in key

    def test_cache_key_sorted_regions(self):
        """Cache key regions are sorted."""
        key1 = cache.make_spot_cache_key(["t3.micro"], ["eu-west-1", "us-east-1"])
        key2 = cache.make_spot_cache_key(["t3.micro"], ["us-east-1", "eu-west-1"])
        assert key1 == key2

    def test_cache_key_sorted_instances(self):
        """Cache key instance types are sorted."""
        key1 = cache.make_spot_cache_key(["t3.large", "t3.micro"], ["us-east-1"])
        key2 = cache.make_spot_cache_key(["t3.micro", "t3.large"], ["us-east-1"])
        assert key1 == key2
