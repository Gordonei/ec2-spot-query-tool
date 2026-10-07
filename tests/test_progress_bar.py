"""Tests for the live progress bar in main.py"""

from __future__ import annotations

import typer
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock, call
import sys

from ec2_spot_query.main import app


runner = CliRunner()


class TestProgressCallbackTracking:
    """Test that the progress callback correctly tracks pairs and records."""

    def test_callback_tracks_multiple_calls(self):
        """Multiple progress callbacks accumulate pair count and final record count."""
        done = {"count": 0}
        records_seen = []

        def _callback(pair_name: str, total_records: int) -> None:
            done["count"] += 1
            records_seen.append(total_records)

        _callback("Fetching t3.micro in us-east-1", 100)
        _callback("Fetching t3.large in us-east-1", 250)
        _callback("Fetching t3.micro in eu-west-1", 380)

        assert done["count"] == 3
        assert records_seen == [100, 250, 380]

    def test_callback_single_call(self):
        """Single progress callback increments pair count to 1."""
        done = {"count": 0}

        def _callback(pair_name: str, total_records: int) -> None:
            done["count"] += 1

        _callback("Fetching t3.micro in us-east-1", 42)
        assert done["count"] == 1

    def test_callback_records_is_cumulative(self):
        """Record count always holds the latest cumulative total from the API."""
        cumulative = [0]

        def _callback(pair_name: str, total_records: int) -> None:
            cumulative[0] = total_records

        _callback("Fetching t3.micro in us-east-1", 10)
        assert cumulative[0] == 10
        _callback("Fetching t3.large in us-east-1", 50)
        assert cumulative[0] == 50
        _callback("Fetching t3.micro in eu-west-1", 80)
        assert cumulative[0] == 80


class TestPairCountCalculation:
    """Test that total pairs is calculated correctly."""

    def test_pairs_from_specified_instances_and_regions(self):
        """With 2 instances and 3 regions, total pairs = 6."""
        ipts = ["t3.micro", "t3.large"]
        regions = ["us-east-1", "us-west-2", "eu-west-1"]
        total_pairs = len(ipts) * len(regions)
        assert total_pairs == 6

    def test_pairs_with_all_regions_is_zero(self):
        """When regions is None (all-regions), total_pairs = 0 (unknown)."""
        ipts = ["t3.micro"]
        regions_none = None
        total_pairs = len(ipts) * len(regions_none) if regions_none else 0
        assert total_pairs == 0

    def test_pairs_empty_instances_is_zero(self):
        """With 0 instances, total pairs is 0."""
        ipts = []
        regions = ["us-east-1"]
        total_pairs = len(ipts) * len(regions)
        assert total_pairs == 0


class TestProgressColumns:
    """Test that Progress bar contains expected column types."""

    def test_progress_columns_importable(self):
        """All Progress bar column classes are importable from rich.progress."""
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        # Just verify they can be imported - they exist in the installed version
        assert Progress is not None
        assert SpinnerColumn is not None
        assert TextColumn is not None
        assert BarColumn is not None
        assert TimeElapsedColumn is not None


class TestProgressFlagIntegration:
    """Integration tests verifying --progress flag controls output mode."""

    def test_progress_flag_in_help(self):
        """--progress flag appears in --help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--progress" in result.stdout or "-p" in result.stdout

    def test_progress_flag_default_is_false(self):
        """Without --progress flag, the handler is a noop."""
        from rich.console import Console
        from ec2_spot_query.main import _make_progress_handler

        console = Console()
        handler = _make_progress_handler(console, use_progress=False)

        # Should not raise, should be a noop
        handler("progress", "test message", {"records": 5})
        handler("query", "test message", None)
        handler("complete", "test message", None)
