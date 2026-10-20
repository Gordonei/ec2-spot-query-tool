"""Tests for Spot Placement Score integration — TDD cycle."""

from __future__ import annotations

pytest_plugins = ("tests.test_utils",)

import datetime as dt
import io
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from rich.console import Console
from typer.testing import CliRunner

from ec2_spot_query import cache, cli, core

runner = CliRunner()


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLICLIIntegration:
    """Test CLI integration of placement score flags."""

    def test_placement_score_flag_in_help(self):
        """--placement-score appears in --help output."""
        result = runner.invoke(cli.app, ["--help"])
        assert result.exit_code == 0
        assert "--placement-score" in result.stdout

    def test_min_placement_score_flag_in_help(self):
        """--min-placement-score appears in --help output."""
        result = runner.invoke(cli.app, ["--help"])
        assert result.exit_code == 0
        assert "--min-placement-score" in result.stdout

    def test_target_capacity_flag_in_help(self):
        """--target-capacity appears in --help output."""
        result = runner.invoke(cli.app, ["--help"])
        assert result.exit_code == 0
        assert "--target-capacity" in result.stdout

    def test_render_table_with_score_column(self):
        """render_table with with_score=True shows Score column."""
        df = pd.DataFrame([{
            "region_az": "us-east-1/us-east-1a",
            "region": "us-east-1",
            "instance_type": "t3.micro",
            "current_price": 0.012,
            "1h_mean": 0.013, "1h_vol": 0.0010,
            "6h_mean": 0.012, "6h_vol": 0.0020,
            "12h_mean": 0.011, "12h_vol": 0.0010,
            "1d_mean": 0.012, "1d_vol": 0.0030,
            "1w_mean": 0.010, "1w_vol": 0.0020,
            "1m_mean": 0.015, "1m_vol": 0.0040,
            "placement_score": 8,
        }])
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        with patch("ec2_spot_query.cli.Console", return_value=console):
            cli.render_table(df, with_score=True)
        output = buf.getvalue()
        assert "Score" in output
        assert "8" in output

    def test_render_table_without_score_column(self):
        """render_table with with_score=False does not show Score column."""
        df = pd.DataFrame([{
            "region_az": "us-east-1/us-east-1a",
            "region": "us-east-1",
            "instance_type": "t3.micro",
            "current_price": 0.012,
            "1h_mean": 0.013, "1h_vol": 0.0010,
            "6h_mean": 0.012, "6h_vol": 0.0020,
            "12h_mean": 0.011, "12h_vol": 0.0010,
            "1d_mean": 0.012, "1d_vol": 0.0030,
            "1w_mean": 0.010, "1w_vol": 0.0020,
            "1m_mean": 0.015, "1m_vol": 0.0040,
            "placement_score": 8,
        }])
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        with patch("ec2_spot_query.cli.Console", return_value=console):
            cli.render_table(df, with_score=False)
        output = buf.getvalue()
        assert "Score" not in output

    def test_render_table_score_na_for_none(self):
        """render_table shows N/A when placement_score is None."""
        df = pd.DataFrame([{
            "region_az": "us-east-1/us-east-1a",
            "region": "us-east-1",
            "instance_type": "t3.micro",
            "current_price": 0.012,
            "1h_mean": 0.013, "1h_vol": 0.0010,
            "6h_mean": 0.012, "6h_vol": 0.0020,
            "12h_mean": 0.011, "12h_vol": 0.0010,
            "1d_mean": 0.012, "1d_vol": 0.0030,
            "1w_mean": 0.010, "1w_vol": 0.0020,
            "1m_mean": 0.015, "1m_vol": 0.0040,
            "placement_score": None,
        }])
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        with patch("ec2_spot_query.cli.Console", return_value=console):
            cli.render_table(df, with_score=True)
        output = buf.getvalue()
        assert "N/A" in output


def test_cli_invoke_with_placement_score(mock_ec2, tmp_path):
    """Invoking with --placement-score exits cleanly with mocked data."""
    cache_file = tmp_path / cache.CACHE_FILENAME
    mock_ec2.describe_spot_price_history.return_value = {
        "SpotPriceHistory": [
            {
                "InstanceId": "us-east-1:us-east-1a",
                "AvailabilityZone": "us-east-1a",
                "SpotPrice": "0.012",
                "Timestamp": "2025-10-01T00:00:00Z",
            }
        ]
    }
    mock_ec2.describe_availability_zones.return_value = {
        "AvailabilityZones": [
            {"ZoneName": "us-east-1a", "ZoneId": "use1-az1"},
        ]
    }
    mock_ec2.get_spot_placement_scores.return_value = {
        "SpotPlacementScores": [
            {"Region": "us-east-1",
                    "AvailabilityZoneId": "use1-az1", "Score": 7},
        ]
    }
    mock_ec2.describe_regions.return_value = {
        "Regions": [{"RegionName": "us-east-1"}],
    }

    with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
        result = runner.invoke(
            cli.app,
            ["--instance-types", "t3.micro", "--regions", "us-east-1",
             "--no-cache", "--placement-score"],
        )
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AZ ID mapping tests
# ---------------------------------------------------------------------------

class TestAZMapping:
    """Test AZ ID resolution and caching."""

    def test_get_az_id_mapping_returns_dict(self, mock_ec2, tmp_path):
        """describe_availability_zones result is resolved to name->id mapping."""
        cache_file = tmp_path / "ec2-spot-cache.json"
        mock_ec2.describe_availability_zones.return_value = {
            "AvailabilityZones": [
                {"ZoneName": "us-east-1a", "ZoneId": "use1-az1"},
                {"ZoneName": "us-east-1b", "ZoneId": "use1-az2"},
            ]
        }
        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = core.get_az_id_mapping("us-east-1")
        assert isinstance(result, dict)
        assert result == {"us-east-1a": "use1-az1", "us-east-1b": "use1-az2"}

    def test_get_az_id_mapping_uses_cached_result(self, mock_ec2, tmp_path):
        """Cached AZ mapping is returned without another API call."""
        cache_file = tmp_path / "ec2-spot-cache.json"

        mock_ec2.describe_availability_zones.return_value = {
            "AvailabilityZones": [
                {"ZoneName": "us-east-1a", "ZoneId": "use1-az1"},
            ]
        }

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            # First call — populates cache
            result1 = core.get_az_id_mapping("us-east-1")
            assert result1 == {"us-east-1a": "use1-az1"}

            # Second call — should use cache (mock not called again)
            mock_ec2.describe_availability_zones.reset_mock()
            result2 = core.get_az_id_mapping("us-east-1")
            assert result2 == {"us-east-1a": "use1-az1"}
            # describe_availability_zones should NOT be called again
            mock_ec2.describe_availability_zones.assert_not_called()

    def test_get_az_id_mapping_cache_miss_fallback(self, mock_ec2, tmp_path):
        """A region not in cache triggers a real API call with per-region client."""
        cache_file = tmp_path / "ec2-spot-cache.json"

        mock_ec2.describe_availability_zones.return_value = {
            "AvailabilityZones": [
                {"ZoneName": "eu-west-1a", "ZoneId": "euw1-az1"},
                {"ZoneName": "eu-west-1b", "ZoneId": "euw1-az2"},
            ]
        }

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = core.get_az_id_mapping("eu-west-1")
        assert result == {"eu-west-1a": "euw1-az1", "eu-west-1b": "euw1-az2"}
        # Now uses a per-region client (no Filters param needed)
        mock_ec2.describe_availability_zones.assert_called_once()

    def test_get_az_id_mapping_api_error_returns_empty(self, mock_ec2, tmp_path):
        """API errors return empty dict without crashing."""
        from botocore.exceptions import ClientError

        cache_file = tmp_path / cache.CACHE_FILENAME
        mock_ec2.describe_availability_zones.side_effect = ClientError(
            error_response={"Error": {"Code": "AuthFailure", "Message": "Unauthorized"}},
            operation_name="DescribeAvailabilityZones",
        )
        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = core.get_az_id_mapping("us-east-1")
        assert result == {}


# ---------------------------------------------------------------------------
# Spot Placement Score tests
# ---------------------------------------------------------------------------

class TestSpotPlacementScores:
    """Test get_spot_placement_scores batched fetching."""

    def _make_mock_ec2_client(self, region: str) -> MagicMock:
        """Create a mock EC2 client for a given region."""
        client = MagicMock()
        client.get_spot_placement_scores.return_value = {
            "SpotPlacementScores": [
                {
                    "Region": "us-east-1",
                    "AvailabilityZoneId": "use1-az1",
                    "Score": 8,
                },
                {
                    "Region": "us-east-1",
                    "AvailabilityZoneId": "use1-az2",
                    "Score": 5,
                },
            ]
        }
        return client

    def test_get_spot_placement_scores_returns_dict(self, mock_ec2, tmp_path):
        """Returns {region: {instance_type: {az: score}}}."""
        cache_file = tmp_path / cache.CACHE_FILENAME
        mock_ec2.describe_availability_zones.return_value = {
            "AvailabilityZones": [
                {"ZoneName": "us-east-1a", "ZoneId": "use1-az1"},
                {"ZoneName": "us-east-1b", "ZoneId": "use1-az2"},
            ]
        }
        mock_ec2.get_spot_placement_scores.return_value = {
            "SpotPlacementScores": [
                {"Region": "us-east-1",
                    "AvailabilityZoneId": "use1-az1", "Score": 8},
                {"Region": "us-east-1",
                    "AvailabilityZoneId": "use1-az2", "Score": 5},
            ]
        }

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = core.get_spot_placement_scores(
                ["t3.micro"],
                ["us-east-1"],
                target_capacity=1,
            )
        assert isinstance(result, dict)
        assert "us-east-1" in result
        assert "t3.micro" in result["us-east-1"]
        assert result["us-east-1"]["t3.micro"] == {"us-east-1a": 8, "us-east-1b": 5}

    def test_get_spot_placement_scores_az_name_resolution(self, mock_ec2, tmp_path):
        """AZ IDs in response are resolved to human-readable names."""
        cache_file = tmp_path / cache.CACHE_FILENAME
        mock_ec2.describe_availability_zones.return_value = {
            "AvailabilityZones": [
                {"ZoneName": "us-east-1a", "ZoneId": "use1-az1"},
                {"ZoneName": "us-east-1b", "ZoneId": "use1-az2"},
            ]
        }
        mock_ec2.get_spot_placement_scores.return_value = {
            "SpotPlacementScores": [
                {"Region": "us-east-1",
                    "AvailabilityZoneId": "use1-az1", "Score": 7},
            ]
        }
        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = core.get_spot_placement_scores(
                ["t3.micro"],
                ["us-east-1"],
                target_capacity=1,
            )
        assert "us-east-1a" in result["us-east-1"]["t3.micro"]
        assert "use1-az1" not in result["us-east-1"]["t3.micro"]

    def test_get_spot_placement_scores_single_az_mode(self, mock_ec2, tmp_path):
        """SingleAvailabilityZone=True is passed to the API."""
        cache_file = tmp_path / cache.CACHE_FILENAME
        mock_ec2.describe_availability_zones.return_value = {"AvailabilityZones": []}
        mock_ec2.get_spot_placement_scores.return_value = {
            "SpotPlacementScores": [
                {"Region": "us-east-1",
                    "AvailabilityZoneId": "use1-az1", "Score": 9},
            ]
        }

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            core.get_spot_placement_scores(
                ["t3.micro"],
                ["us-east-1"],
                target_capacity=2,
            )

        _, kwargs = mock_ec2.get_spot_placement_scores.call_args
        assert kwargs["SingleAvailabilityZone"] is True
        assert kwargs["TargetCapacity"] == 2

    def test_get_spot_placement_scores_region_batching(self, mock_ec2, tmp_path):
        """Regions are batched into groups of 10 per API call."""
        cache_file = tmp_path / cache.CACHE_FILENAME
        mock_ec2.describe_availability_zones.return_value = {"AvailabilityZones": []}
        mock_ec2.get_spot_placement_scores.return_value = {
            "SpotPlacementScores": [
                {"Region": "us-east-1",
                    "AvailabilityZoneId": "use1-az1", "Score": 6},
            ]
        }

        regions = [f"us-east-{i}" for i in range(1, 13)]  # 12 regions
        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            core.get_spot_placement_scores(
                ["t3.micro"],
                regions,
                target_capacity=1,
            )

        assert mock_ec2.get_spot_placement_scores.call_count == 2
        # First call should have 10 regions, second call 2
        first_call_regions = mock_ec2.get_spot_placement_scores.call_args_list[0][1]["RegionNames"]
        second_call_regions = mock_ec2.get_spot_placement_scores.call_args_list[1][1]["RegionNames"]
        assert len(first_call_regions) == 10
        assert len(second_call_regions) == 2

    def test_get_spot_placement_scores_api_error_is_ignored(self, mock_ec2, tmp_path):
        """API errors for individual batches are logged and skipped."""
        from botocore.exceptions import ClientError

        cache_file = tmp_path / cache.CACHE_FILENAME
        mock_ec2.describe_availability_zones.return_value = {"AvailabilityZones": []}
        mock_ec2.get_spot_placement_scores.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException", "Message": "Forbidden"}},
            operation_name="GetSpotPlacementScores",
        )
        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file), \
             patch("ec2_spot_query.core.logger") as mock_logger:
            result = core.get_spot_placement_scores(
                ["t3.micro"],
                ["us-east-1"],
                target_capacity=1,
            )
        # API error returns initialized dict with empty score dicts
        assert result == {"us-east-1": {"t3.micro": {}}}
        mock_logger.warning.assert_called()


# ---------------------------------------------------------------------------
# Filter by placement score tests
# ---------------------------------------------------------------------------

class TestFilterByPlacementScore:
    """Test placement score filtering logic."""

    def test_filter_removes_low_scores(self):
        """Rows with placement_score < min_score are dropped."""
        df = pd.DataFrame([
            {
                "region_az": "us-east-1/us-east-1a",
                "region": "us-east-1",
                "instance_type": "t3.micro",
                "current_price": 0.012,
                "1d_mean": 0.013, "1d_vol": 0.002,
                "placement_score": 8,
            },
            {
                "region_az": "us-east-1/us-east-1b",
                "region": "us-east-1",
                "instance_type": "t3.micro",
                "current_price": 0.010,
                "1d_mean": 0.011, "1d_vol": 0.001,
                "placement_score": 3,
            },
        ])
        scores = {
            "us-east-1": {
                "t3.micro": {"us-east-1a": 8, "us-east-1b": 3},
            }
        }
        result = core.filter_by_placement_score(df, min_score=5, scores=scores)
        assert len(result) == 1
        assert result.iloc[0]["placement_score"] == 8

    def test_filter_keeps_none_scores(self):
        """Rows with placement_score=None are kept (no score data)."""
        df = pd.DataFrame([
            {
                "region_az": "us-east-1/us-east-1a",
                "region": "us-east-1",
                "instance_type": "t3.micro",
                "current_price": 0.012,
                "1d_mean": 0.013, "1d_vol": 0.002,
                "placement_score": None,
            },
        ])
        scores: dict[str, dict[str, dict[str, int]]] = {}
        result = core.filter_by_placement_score(df, min_score=5, scores=scores)
        assert len(result) == 1
        assert result.iloc[0]["placement_score"] is None

    def test_filter_boundary_score_kept(self):
        """Rows with score == min_score are kept (strictly lower than dropped)."""
        df = pd.DataFrame([
            {
                "region_az": "us-east-1/us-east-1a",
                "region": "us-east-1",
                "instance_type": "t3.micro",
                "current_price": 0.012,
                "1d_mean": 0.013, "1d_vol": 0.002,
                "placement_score": 5,
            },
        ])
        scores = {
            "us-east-1": {
                "t3.micro": {"us-east-1a": 5},
            }
        }
        result = core.filter_by_placement_score(df, min_score=5, scores=scores)
        assert len(result) == 1

    def test_filter_boundary_score_dropped(self):
        """Rows with score == min_score - 1 are dropped."""
        df = pd.DataFrame([
            {
                "region_az": "us-east-1/us-east-1a",
                "region": "us-east-1",
                "instance_type": "t3.micro",
                "current_price": 0.012,
                "1d_mean": 0.013, "1d_vol": 0.002,
                "placement_score": 4,
            },
        ])
        scores = {
            "us-east-1": {
                "t3.micro": {"us-east-1a": 4},
            }
        }
        result = core.filter_by_placement_score(df, min_score=5, scores=scores)
        assert len(result) == 0

    def test_filter_empty_df_returns_empty(self):
        """Empty DataFrame passes through unchanged."""
        empty_df = pd.DataFrame(columns=["region_az", "region", "instance_type",
                                          "current_price", "placement_score"])
        result = core.filter_by_placement_score(empty_df, min_score=5, scores={})
        assert result.empty

    def test_filter_partial_coverage_keeps_missing(self):
        """Rows with no score data in the scores dict are kept."""
        df = pd.DataFrame([
            {
                "region_az": "us-east-1/us-east-1a",
                "region": "us-east-1",
                "instance_type": "t3.micro",
                "current_price": 0.012,
                "1d_mean": 0.013, "1d_vol": 0.002,
                "placement_score": 9,
            },
            {
                "region_az": "us-west-2/us-west-2a",
                "region": "us-west-2",
                "instance_type": "t3.micro",
                "current_price": 0.018,
                "1d_mean": 0.019, "1d_vol": 0.003,
                "placement_score": 7,
            },
        ])
        # Only us-east-1 has scores; us-west-2 does not
        scores = {
            "us-east-1": {
                "t3.micro": {"us-east-1a": 9},
            }
        }
        result = core.filter_by_placement_score(df, min_score=5, scores=scores)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Column schema tests
# ---------------------------------------------------------------------------

class TestColumnSchema:
    """Test that placement_score is in the expected column positions."""

    def test_compute_metrics_with_placement_score_column(self):
        """compute_metrics returns placement_score column in result."""
        result = core.compute_metrics([])
        assert "placement_score" in result.columns

    def test_compute_metrics_placement_score_is_none(self):
        """New rows start with placement_score=None."""
        result = core.compute_metrics([])
        if not result.empty:
            for val in result["placement_score"]:
                assert val is None
        else:
            # Empty df — verify column exists
            assert list(result["placement_score"]) == []
