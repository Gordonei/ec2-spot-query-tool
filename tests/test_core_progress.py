"""Tests for core.py progress callback in fetch_spot_prices."""

from __future__ import annotations

import datetime as dt
import threading

import pytest
from unittest.mock import MagicMock, patch, call

from ec2_spot_query import core


def _make_spot_record(
    instance_type: str = "t3.micro",
    az: str = "us-east-1a",
    price: float = 0.012,
) -> dict:
    return {
        "InstanceId": f"us-east-1:{az}",
        "AvailabilityZone": az,
        "InstanceType": instance_type,
        "SpotPrice": str(price),
        "Timestamp": dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc),
        "ProductDescription": "Linux/UNIX",
    }


def test_progress_callback_receives_updates(mock_ec2):
    """Callback receives (pair_name, total_records) tuples."""
    mock_ec2.describe_spot_price_history.return_value = {
        "SpotPriceHistory": [
            _make_spot_record("t3.micro", "us-east-1a", 0.010),
            _make_spot_record("t3.micro", "us-east-1a", 0.012),
        ]
    }
    mock_ec2.describe_regions.return_value = {
        "Regions": [{"RegionName": "us-east-1"}]
    }

    received: list[tuple[str, int]] = []
    callback = lambda pair, total: received.append((pair, total))

    result = core.fetch_spot_prices(
        ["t3.micro"],
        regions=["us-east-1"],
        progress_callback=callback,
    )

    assert len(result) == 2
    assert len(received) == 1  # one (instance_type, region) pair
    assert received[0][0] == "Fetching t3.micro in us-east-1"
    assert received[0][1] == 2


def test_progress_callback_none_is_noop(mock_ec2):
    """progress_callback=None raises no errors."""
    mock_ec2.describe_spot_price_history.return_value = {
        "SpotPriceHistory": [
            _make_spot_record("t3.large", "us-west-2a", 0.048),
        ]
    }
    mock_ec2.describe_regions.return_value = {
        "Regions": [{"RegionName": "us-west-2"}]
    }

    result = core.fetch_spot_prices(
        ["t3.large"],
        regions=["us-west-2"],
        progress_callback=None,
    )

    assert len(result) == 1


def test_progress_callback_format(mock_ec2):
    """Callback receives sensible values: name contains instance+region, total increases."""
    mock_ec2.describe_spot_price_history.side_effect = [
        {
            "SpotPriceHistory": [
                _make_spot_record("t3.micro", "us-east-1a", 0.010),
                _make_spot_record("t3.micro", "us-east-1a", 0.011),
                _make_spot_record("t3.micro", "us-east-1a", 0.012),
            ]
        },
        {
            "SpotPriceHistory": [
                _make_spot_record("t3.micro", "us-east-1b", 0.013),
                _make_spot_record("t3.micro", "us-east-1b", 0.014),
            ]
        },
    ]
    mock_ec2.describe_regions.return_value = {
        "Regions": [
            {"RegionName": "us-east-1"},
            {"RegionName": "us-east-2"},
        ]
    }

    received: list[tuple[str, int]] = []
    callback = lambda pair, total: received.append((pair, total))

    result = core.fetch_spot_prices(
        ["t3.micro"],
        regions=["us-east-1", "us-east-2"],
        progress_callback=callback,
    )

    assert len(result) == 5
    assert len(received) == 2
    # Total should be monotonically increasing
    totals = [r[1] for r in received]
    assert totals[0] < totals[1]
    assert totals[-1] == 5  # all records


def test_progress_includes_all_pairs(mock_ec2):
    """Every (instance_type, region) pair triggers at least one callback."""
    mock_ec2.describe_spot_price_history.return_value = {
        "SpotPriceHistory": [
            _make_spot_record("t3.micro", "us-east-1a", 0.010),
            _make_spot_record("t3.large", "us-west-2a", 0.048),
        ]
    }
    mock_ec2.describe_regions.return_value = {
        "Regions": [
            {"RegionName": "us-east-1"},
            {"RegionName": "us-west-2"},
        ]
    }

    received: list[tuple[str, int]] = []
    callback = lambda pair, total: received.append((pair, total))

    core.fetch_spot_prices(
        ["t3.micro", "t3.large"],
        regions=["us-east-1", "us-west-2"],
        progress_callback=callback,
    )

    # 2 instance types × 2 regions = 4 pairs
    pairs_seen = [r[0] for r in received]
    assert len(pairs_seen) == 4
    assert any("t3.micro" in p and "us-east-1" in p for p in pairs_seen)
    assert any("t3.micro" in p and "us-west-2" in p for p in pairs_seen)
    assert any("t3.large" in p and "us-east-1" in p for p in pairs_seen)
    assert any("t3.large" in p and "us-west-2" in p for p in pairs_seen)


def test_no_progress_unchanged_behavior(mock_ec2):
    """Existing tests pass unmodified — default progress_callback=None is invisible."""
    mock_ec2.describe_spot_price_history.return_value = {
        "SpotPriceHistory": [
            _make_spot_record("t2.nano", "us-east-1a", 0.002),
        ]
    }
    mock_ec2.describe_regions.return_value = {
        "Regions": [{"RegionName": "us-east-1"}]
    }

    result = core.fetch_spot_prices(["t2.nano"], regions=["us-east-1"])
    assert len(result) == 1
    assert result[0]["InstanceType"] == "t2.nano"
