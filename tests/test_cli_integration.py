"""Integration tests for the CLI — full workflow with core mocked."""

from __future__ import annotations

import datetime as dt
import json

from typer.testing import CliRunner

from ec2_spot_query.cli import app
from unittest.mock import patch

runner = CliRunner()


def _spot_records() -> list[dict]:
    """One default spot record for t3.micro in us-east-1."""
    return [
        {
            "Region": "us-east-1",
            "InstanceId": "us-east-1:us-east-1a",
            "AvailabilityZone": "us-east-1a",
            "InstanceType": "t3.micro",
            "SpotPrice": "0.012",
            "Timestamp": dt.datetime(2025, 10, 1, 0, 0, tzinfo=dt.timezone.utc),
        }
    ]


class TestFullWorkflow:
    """Test complete CLI workflow with core mocked."""

    def test_full_flow_with_specified_instances(self, mocker):
        """Full pipeline: resolve instances → fetch → compute → render."""
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        mocker.patch(
            "ec2_spot_query.core.resolve_instance_types", return_value=["t3.micro"]
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "us-east-1", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        assert "EC2 Spot Instance Rankings" in result.stdout
        mock_fetch.assert_called_once()


class TestAllRegions:
    """Test --all-regions flag."""

    def test_all_regions_flag_invokes_list_regions(self, mocker):
        """--all-regions calls core.list_regions for discovery."""
        mock_list = mocker.patch(
            "ec2_spot_query.core.list_regions", return_value=["us-east-1", "eu-west-1"]
        )
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--all-regions", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        mock_list.assert_called_once()


class TestCacheHitFlow:
    """Test cache hit path — skip fetch calls."""

    def test_cache_hit_skips_fetch(self, mocker, tmp_path):
        """Cached spot data skips the fetch call."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cached_data = {
            "spot:t3.micro:us-east-1": {
                "data": [
                    {
                        "Region": "us-east-1",
                        "InstanceId": "us-east-1:us-east-1a",
                        "AvailabilityZone": "us-east-1a",
                        "InstanceType": "t3.micro",
                        "SpotPrice": "0.012",
                        "Timestamp": "2025-10-01T00:00:00+00:00",
                    }
                ],
                "expires_at": 9999999999,
            }
        }
        cache_file.write_text(json.dumps(cached_data), encoding="utf-8")
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app,
                ["--instance-types", "t3.micro", "--regions", "us-east-1"],
            )

        assert result.exit_code == 0, result.stdout
        mock_fetch.assert_not_called()


class TestCacheMissFlow:
    """Test cache miss path — fetch and save."""

    def test_cache_miss_fetches_and_saves(self, mocker, tmp_path):
        """Cache miss triggers the fetch call and saves the result."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("{}", encoding="utf-8")
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app,
                ["--instance-types", "t3.micro", "--regions", "us-east-1"],
            )

        assert result.exit_code == 0, result.stdout
        mock_fetch.assert_called_once()
        saved = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "spot:t3.micro:us-east-1" in saved


class TestFlags:
    """Test --progress and --debug flags."""

    def test_progress_flag_in_help(self):
        """--progress flag appears in --help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--progress" in result.stdout

    def test_progress_flag_default_is_false(self, mocker):
        """Without --progress the CLI still runs (handler is a noop)."""
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app, ["--instance-types", "t3.micro", "--regions", "us-east-1", "--no-cache"]
        )
        assert result.exit_code == 0

    def test_debug_flag_in_help(self):
        """--debug flag appears in --help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_debug_configures_logging(self, mocker):
        """--debug enables detailed logging without breaking the run."""
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "us-east-1", "--debug", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
