"""Shared fixtures for core tests."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from unittest.mock import MagicMock, patch

from ec2_spot_query import core


@pytest.fixture
def mock_ec2(mocker):
    """Provide a mocked boto3 EC2 client used across core tests."""
    mock = MagicMock()
    with patch("ec2_spot_query.core.boto3.client", return_value=mock):
        yield mock


# Raw data returned by boto3 describe_spot_price_history
# Format: one dict per API result line
def _make_spot_record(
    instance_type: str = "t3.micro",
    region: str = "us-east-1",
    az: str = "us-east-1a",
    price: float = 0.012,
    ts: dt.datetime | None = None,
) -> dict[str, Any]:
    ts = ts or dt.datetime(2025, 10, 1, 0, 0, tzinfo=dt.timezone.utc)
    return {
        "InstanceId": f"{region}:{az}",
        "AvailabilityZone": az,
        "InstanceType": instance_type,
        "SpotPrice": str(price),
        "Timestamp": ts,
        "ProductDescription": "Linux/UNIX",
    }


@pytest.fixture
def raw_spot_data() -> list[dict[str, Any]]:
    """A list of spot price records spanning multiple days and two AZs."""
    base = dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc)
    records = []
    for delta_hours in range(0, 48, 6):  # 06:12:18:24:30:36:42:18
        hour_offset = delta_hours
        records.append(_make_spot_record("t3.micro", "us-east-1", "us-east-1a", 0.010, base + dt.timedelta(hours=hour_offset)))
        records.append(_make_spot_record("t3.micro", "us-east-1", "us-east-1b", 0.015, base + dt.timedelta(hours=hour_offset)))
    return records


@pytest.fixture
def raw_spot_flat_price() -> list[dict[str, Any]]:
    """Records where price never changes (tests stddev=0 handling)."""
    base = dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc)
    return [
        _make_spot_record("t3.small", "us-west-2", "us-west-2a", 0.020, base + dt.timedelta(hours=h))
        for h in range(0, 25, 3)
    ]


@pytest.fixture
def raw_spot_multi_instance() -> list[dict[str, Any]]:
    """Records for multiple instance types in same region."""
    base = dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc)
    records = []
    for inst, price in [("t3.micro", 0.010), ("t3.large", 0.048)]:
        for az, az_price in [("us-east-1a", price), ("us-east-1b", price + 0.003)]:
            for h in range(0, 25, 5):
                records.append(_make_spot_record(inst, "us-east-1", az, az_price, base + dt.timedelta(hours=h)))
    return records
