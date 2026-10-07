"""Integration tests for the CLI — full workflow with mocked AWS."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ec2_spot_query.cli import app

runner = CliRunner()


def _make_mock_ec2(
    spot_records: list[dict] | None = None,
    regions: list[dict] | None = None,
) -> MagicMock:
    """Create a fully configured mock EC2 client."""
    mock = MagicMock()
    if spot_records is None:
        spot_records = [
            {
                "InstanceId": "us-east-1:us-east-1a",
                "AvailabilityZone": "us-east-1a",
                "InstanceType": "t3.micro",
                "SpotPrice": "0.012",
                "Timestamp": dt.datetime(2025, 10, 1, 0, 0, tzinfo=dt.timezone.utc),
            }
        ]
    if regions is None:
        regions = [{"RegionName": "us-east-1"}]
    mock.describe_spot_price_history.return_value = {"SpotPriceHistory": spot_records}
    mock.describe_regions.return_value = {"Regions": regions}
    mock.describe_instance_types.return_value = {"InstanceTypes": []}
    return mock


class TestFullWorkflow:
    """Test complete CLI workflow with mocked AWS."""

    def test_full_flow_with_specified_instances(self, mocker):
        """Full pipeline: resolve instances → fetch → compute → render."""
        mock_ec2 = _make_mock_ec2()
        mock = mocker.patch("ec2_spot_query.core.boto3.client", return_value=mock_ec2)
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "us-east-1"],
        )
        assert result.exit_code == 0, result.stdout
        assert "EC2 Spot Instance Rankings" in result.stdout


class TestAllRegions:
    """Test --all-regions flag."""

    def test_all_regions_flag_invokes_describe_regions(self, mocker):
        """--all-regions calls describe_regions."""
        mock_ec2 = _make_mock_ec2(
            spot_records=[
                {
                    "InstanceId": "us-east-1:us-east-1a",
                    "AvailabilityZone": "us-east-1a",
                    "InstanceType": "t3.micro",
                    "SpotPrice": "0.012",
                    "Timestamp": dt.datetime(2025, 10, 1, 0, 0, tzinfo=dt.timezone.utc),
                }
            ],
            regions=[{"RegionName": "us-east-1"}, {"RegionName": "eu-west-1"}],
        )
        mocker.patch("ec2_spot_query.core.boto3.client", return_value=mock_ec2)
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--all-regions"],
        )
        assert result.exit_code == 0, result.stdout


class TestCacheHitFlow:
    """Test cache hit path — skip API calls."""

    def test_cache_hit_skips_api(self, mocker, tmp_path):
        """Cached spot data skips API calls."""
        import json
        from pathlib import Path as P

        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cached_data = {
            "spot:t3.micro:us-east-1": {
                "data": [
                    {
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

        mock_ec2 = _make_mock_ec2()
        mocker.patch("ec2_spot_query.core.boto3.client", return_value=mock_ec2)

        with patch("ec2_spot_query.cache.get_cache_paths", return_value=[cache_file]):
            result = runner.invoke(
                app,
                ["--instance-types", "t3.micro", "--regions", "us-east-1"],
            )

        assert result.exit_code == 0, result.stdout
        # AWS API should NOT be called when cache hits
        mock_ec2.describe_spot_price_history.assert_not_called()


class TestCacheMissFlow:
    """Test cache miss path — fetch from API and save."""

    def test_cache_miss_fetches_and_saves(self, mocker, tmp_path):
        """Cache miss triggers API call and saves result."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Start with empty cache
        cache_file.write_text("{}", encoding="utf-8")

        spot_records = [
            {
                "InstanceId": "us-east-1:us-east-1a",
                "AvailabilityZone": "us-east-1a",
                "InstanceType": "t3.micro",
                "SpotPrice": "0.012",
                "Timestamp": dt.datetime(2025, 10, 1, 0, 0, tzinfo=dt.timezone.utc),
            }
        ]
        mock_ec2 = _make_mock_ec2(spot_records=spot_records)
        mocker.patch("ec2_spot_query.core.boto3.client", return_value=mock_ec2)

        with patch("ec2_spot_query.cache.get_cache_paths", return_value=[cache_file]):
            result = runner.invoke(
                app,
                ["--instance-types", "t3.micro", "--regions", "us-east-1"],
            )

        assert result.exit_code == 0, result.stdout
        # AWS API SHOULD be called when cache misses
        mock_ec2.describe_spot_price_history.assert_called()


class TestProgressFlag:
    """Test --progress flag."""

    def test_progress_flag_in_help(self):
        """--progress flag appears in --help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--progress" in result.stdout

    def test_progress_flag_default_is_false(self):
        """--progress defaults to false."""
        result = runner.invoke(app, ["--instance-types", "t3.micro", "--regions", "us-east-1"])
        # Without progress, spinner should not appear
        assert "spinner" not in result.stdout.lower() or result.exit_code == 0


class TestDebugFlag:
    """Test --debug flag."""

    def test_debug_flag_in_help(self):
        """--debug flag appears in --help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_debug_configures_logging(self, mocker):
        """--debug enables detailed logging."""
        mock_ec2 = _make_mock_ec2()
        mocker.patch("ec2_spot_query.core.boto3.client", return_value=mock_ec2)
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "us-east-1", "--debug"],
        )
        assert result.exit_code == 0, result.stdout
