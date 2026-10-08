"""Tests for ec2_spot_query.core — driven by TDD cycle."""

from __future__ import annotations

pytest_plugins = ("tests.test_utils",)

import datetime as dt

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from ec2_spot_query import core


# ---------------------------------------------------------------------------
# Stub-passing tests (Phase 1: stubs return empty/no-op)
# ---------------------------------------------------------------------------

def test_compute_metrics_empty_input(raw_spot_data):
    """Even with non-empty raw data, the stub returns an empty df with cols."""
    df = core.compute_metrics([])
    assert df.empty


def test_compute_metrics_has_correct_columns():
    """Result DataFrame has all expected column names."""
    result = core.compute_metrics([])
    expected_cols = [
        "region_az", "region", "instance_type", "current_price",
        "1h_mean", "1h_vol",
        "6h_mean", "6h_vol",
        "12h_mean", "12h_vol",
        "1d_mean", "1d_vol",
        "1w_mean", "1w_vol",
        "1m_mean", "1m_vol",
    ]
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# Real implementation tests (Phase 2+)
# ---------------------------------------------------------------------------

def test_compute_metrics_with_data_has_rows(raw_spot_data):
    """When fed real data, result has at least one row."""
    df = core.compute_metrics(raw_spot_data)
    assert len(df) > 0


def test_compute_metrics_sort_by_1d_mean(raw_spot_data):
    """Rows are sorted descending by the sort_by column."""
    df = core.compute_metrics(raw_spot_data, sort_by="1d_mean")
    means = df["1d_mean"].dropna()
    if len(means) >= 2:
        for i in range(len(means) - 1):
            assert means.iloc[i] >= means.iloc[i + 1]


def test_compute_metrics_multiple_windows_all_present(raw_spot_data):
    """All 6 time windows produce exactly 2 columns (mean + vol) each."""
    df = core.compute_metrics(raw_spot_data)
    windows = ["1h", "6h", "12h", "1d", "1w", "1m"]
    for w in windows:
        assert f"{w}_mean" in df.columns, f"Missing {w}_mean"
        assert f"{w}_vol" in df.columns, f"Missing {w}_vol"


def test_compute_metrics_grouped_by_az(raw_spot_data):
    """Same InstanceType in different AZs produces separate rows."""
    df = core.compute_metrics(raw_spot_data)
    group_labels = df["region_az"].unique()
    assert len(group_labels) >= 2  # us-east-1a and us-east-1b
    assert "us-east-1/us-east-1a" in group_labels
    assert "us-east-1/us-east-1b" in group_labels


def test_compute_metrics_has_region_column(raw_spot_data):
    """Output DataFrame includes a separate region column."""
    df = core.compute_metrics(raw_spot_data)
    assert "region" in df.columns
    assert df["region"].str.contains("us-east-1").all()


def test_compute_metrics_sort_by_6h_mean(raw_spot_data):
    """Sorting by a different metric also works."""
    df = core.compute_metrics(raw_spot_data, sort_by="6h_mean")
    means = df["6h_mean"].dropna()
    if len(means) >= 2:
        for i in range(len(means) - 1):
            assert means.iloc[i] >= means.iloc[i + 1]


def test_compute_metrics_flat_price_stddev_zero(raw_spot_flat_price):
    """When all prices are identical, volatility is 0.0 (not NaN)."""
    df = core.compute_metrics(raw_spot_flat_price)
    vol_cols = [c for c in df.columns if c.endswith("_vol")]
    for col in vol_cols:
        assert not df[col].isna().any(), f"{col} has NaN values"
        assert (df[col] == 0.0).any(), f"{col} should contain 0.0 for flat prices"


def test_compute_metrics_multi_instance_separate_rows(raw_spot_multi_instance):
    """Different instance types appear as separate rows."""
    df = core.compute_metrics(raw_spot_multi_instance)
    instances = df["instance_type"].unique()
    assert "t3.micro" in instances
    assert "t3.large" in instances


def test_fetch_spot_prices_returns_list(mock_ec2):
    """When instance_types and regions are provided, returns a list."""
    mock_ec2.describe_spot_price_history.return_value = {
        "SpotPriceHistory": [
            {
                "Region": "us-east-1",
                "InstanceId": "us-east-1:us-east-1a",
                "AvailabilityZone": "us-east-1a",
                "InstanceType": "t3.micro",
                "SpotPrice": "0.012",
                "Timestamp": dt.datetime.now(dt.timezone.utc),
            }
        ]
    }
    mock_ec2.describe_regions.return_value = {"Regions": [{"RegionName": "us-east-1"}]}
    result = core.fetch_spot_prices(["t3.micro"], regions=["us-east-1"], days=30, product_description="Linux/UNIX")
    assert isinstance(result, list)



def test_resolve_instance_types_provided():
    """When instances are provided directly, they are returned as-is."""
    result = core.resolve_instance_types(
        instance_types=["t3.micro", "t3.large"],
        min_vcpu=0,
        min_ram_gb=0.5,
        min_gpu=0,
        max_vcpu=64,
        max_ram_gb=128,
        max_gpu=8,
        min_instance_storage_gb=0,
        max_instance_storage_gb=256,
    )
    assert isinstance(result, list)
    assert result == ["t3.micro", "t3.large"]


def test_resolve_instance_types_filters_instances():
    """When no instances provided, describe_instance_types is called with correct filters."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "t3.micro", "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 1024}},
            {"InstanceType": "t2.micro", "VCpuInfo": {"DefaultVCpus": 1}, "MemoryInfo": {"SizeInMiB": 1024}},
        ]
    }
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(
            instance_types=[],
            min_vcpu=2,
            min_ram_gb=0.5,
            min_gpu=0,
            max_vcpu=64,
            max_ram_gb=128,
            max_gpu=8,
            min_instance_storage_gb=0,
            max_instance_storage_gb=256,
        )
    mock_client.describe_instance_types.assert_called_once()
    assert "t3.micro" in result
    assert "t2.micro" not in result


def test_list_regions(mock_ec2):
    """list_regions returns region names from describe_regions."""
    mock_ec2.describe_regions.return_value = {
        "Regions": [{"RegionName": "us-east-1"}, {"RegionName": "eu-west-1"}]
    }
    assert core.list_regions() == ["us-east-1", "eu-west-1"]


def test_aggregate_by_region_selects_best_az():
    """For each (instance_type, region), the AZ with min current_price is selected."""
    df = pd.DataFrame([{
        "region_az": "us-east-1/us-east-1a",
        "instance_type": "t3.micro",
        "current_price": 0.015,
        "1d_mean": 0.014, "1d_vol": 0.002,
    }, {
        "region_az": "us-east-1/us-east-1b",
        "instance_type": "t3.micro",
        "current_price": 0.012,
        "1d_mean": 0.013, "1d_vol": 0.003,
    }])
    result = core.aggregate_by_region(df)
    assert len(result) == 1
    assert result.iloc[0]["region_az"] == "us-east-1"
    assert result.iloc[0]["current_price"] == 0.012


def test_aggregate_by_region_preserves_metrics():
    """Mean and vol columns are preserved for the selected AZ."""
    df = pd.DataFrame([{
        "region_az": "us-east-1/us-east-1a",
        "instance_type": "t3.micro",
        "current_price": 0.015,
        "1d_mean": 0.014, "1d_vol": 0.002,
        "1m_mean": 0.016, "1m_vol": 0.004,
    }, {
        "region_az": "us-east-1/us-east-1b",
        "instance_type": "t3.micro",
        "current_price": 0.012,
        "1d_mean": 0.013, "1d_vol": 0.003,
        "1m_mean": 0.015, "1m_vol": 0.005,
    }])
    result = core.aggregate_by_region(df)
    assert result.iloc[0]["1d_mean"] == 0.013
    assert result.iloc[0]["1d_vol"] == 0.003
    assert result.iloc[0]["1m_mean"] == 0.015
    assert result.iloc[0]["1m_vol"] == 0.005


def test_aggregate_by_region_no_duplicate_instance_types():
    """Same instance type across multiple regions produces separate rows."""
    df = pd.DataFrame([{
        "region_az": "us-east-1/us-east-1a",
        "instance_type": "t3.micro",
        "current_price": 0.012,
        "1d_mean": 0.013, "1d_vol": 0.001,
    }, {
        "region_az": "us-west-2/us-west-2a",
        "instance_type": "t3.micro",
        "current_price": 0.018,
        "1d_mean": 0.019, "1d_vol": 0.002,
    }])
    result = core.aggregate_by_region(df)
    assert len(result) == 2
    regions = result["region_az"].tolist()
    assert "us-east-1" in regions
    assert "us-west-2" in regions


def test_aggregate_by_region_empty():
    """Empty DataFrame returns empty DataFrame."""
    result = core.aggregate_by_region(pd.DataFrame(columns=["region_az", "instance_type", "current_price"]))
    assert result.empty


def test_resolve_max_vcpu_excludes():
    """max_vcpu filters out instances exceeding the threshold."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "p2.xlarge", "VCpuInfo": {"DefaultVCpus": 4}, "MemoryInfo": {"SizeInMiB": 61440}},
            {"InstanceType": "p3.2xlarge", "VCpuInfo": {"DefaultVCpus": 8}, "MemoryInfo": {"SizeInMiB": 61440}},
        ]
    }
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(instance_types=[], min_vcpu=1, min_ram_gb=0.5, min_gpu=0, max_vcpu=5, max_ram_gb=128, max_gpu=8, min_instance_storage_gb=0, max_instance_storage_gb=256)
    mock_client.describe_instance_types.assert_called_once()
    assert "p2.xlarge" in result
    assert "p3.2xlarge" not in result


def test_resolve_max_ram_excludes():
    """max_ram_gb filters out instances exceeding the threshold."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "t3.micro", "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 1024}},
            {"InstanceType": "p3.2xlarge", "VCpuInfo": {"DefaultVCpus": 8}, "MemoryInfo": {"SizeInMiB": 629145}},
        ]
    }
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(instance_types=[], min_vcpu=1, min_ram_gb=0.5, min_gpu=0, max_vcpu=64, max_ram_gb=128, max_gpu=8, min_instance_storage_gb=0, max_instance_storage_gb=256)
    assert "t3.micro" in result
    assert "p3.2xlarge" not in result


def test_resolve_max_gpu_excludes():
    """max_gpu filters out instances exceeding the threshold."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "t3.micro", "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 1024}, "GpuInfo": {}},
            {"InstanceType": "p4d.24xlarge", "VCpuInfo": {"DefaultVCpus": 96}, "MemoryInfo": {"SizeInMiB": 11534336}, "GpuInfo": {"Gpus": [{"Count": 8}]}},
        ]
    }
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(instance_types=[], min_vcpu=1, min_ram_gb=0.5, min_gpu=0, max_vcpu=64, max_ram_gb=128, max_gpu=4, min_instance_storage_gb=0, max_instance_storage_gb=256)
    assert "t3.micro" in result
    assert "p4d.24xlarge" not in result


def test_resolve_storage_filter_excludes_high():
    """max_instance_storage_gb filters out instances with too much storage."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "m5.large", "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 8192}, "InstanceStorageInfo": {"TotalSizeInGB": 50}},
            {"InstanceType": "d3.2xlarge", "VCpuInfo": {"DefaultVCpus": 8}, "MemoryInfo": {"SizeInMiB": 65536}, "InstanceStorageInfo": {"TotalSizeInGB": 8000}},
        ]
    }
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(instance_types=[], min_vcpu=1, min_ram_gb=0.5, min_gpu=0, max_vcpu=64, max_ram_gb=128, max_gpu=8, min_instance_storage_gb=1, max_instance_storage_gb=100)
    assert "m5.large" in result
    assert "d3.2xlarge" not in result


def test_resolve_storage_filter_excludes_low():
    """min_instance_storage_gb filters out instances with too little storage."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "m5.large", "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 8192}, "InstanceStorageInfo": None},
            {"InstanceType": "d3.2xlarge", "VCpuInfo": {"DefaultVCpus": 8}, "MemoryInfo": {"SizeInMiB": 65536}, "InstanceStorageInfo": {"TotalSizeInGB": 8000}},
        ]
    }
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(instance_types=[], min_vcpu=1, min_ram_gb=0.5, min_gpu=0, max_vcpu=64, max_ram_gb=128, max_gpu=8, min_instance_storage_gb=1000, max_instance_storage_gb=10000)
    assert "m5.large" not in result
    assert "d3.2xlarge" in result


def test_resolve_storage_multi_disk():
    """InstanceStorageInfo with TotalSizeInGB sums correctly."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "hlo1.24xl", "VCpuInfo": {"DefaultVCpus": 96}, "MemoryInfo": {"SizeInMiB": 786432}, "InstanceStorageInfo": {"TotalSizeInGB": 1000}},
        ]
    }
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(instance_types=[], min_vcpu=1, min_ram_gb=0.5, min_gpu=0, max_vcpu=128, max_ram_gb=1024, max_gpu=8, min_instance_storage_gb=1, max_instance_storage_gb=900)
    assert "hlo1.24xl" not in result  # 1000 > 900
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(instance_types=[], min_vcpu=1, min_ram_gb=0.5, min_gpu=0, max_vcpu=128, max_ram_gb=1024, max_gpu=8, min_instance_storage_gb=1, max_instance_storage_gb=1100)
    assert "hlo1.24xl" in result  # 1000 <= 1100


def test_resolve_max_vcpu_less_than_min_raises():
    """ValueError raised when max_vcpu < min_vcpu."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        with pytest.raises(ValueError, match="max_vcpu.*min_vcpu"):
            core.resolve_instance_types(instance_types=[], min_vcpu=8, min_ram_gb=0.5, min_gpu=0, max_vcpu=4, max_ram_gb=128, max_gpu=8, min_instance_storage_gb=0, max_instance_storage_gb=256)


def test_resolve_max_ram_less_than_min_raises():
    """ValueError raised when max_ram_gb < min_ram_gb."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        with pytest.raises(ValueError, match="max_ram_gb.*min_ram_gb"):
            core.resolve_instance_types(instance_types=[], min_vcpu=0, min_ram_gb=64, min_gpu=0, max_vcpu=64, max_ram_gb=32, max_gpu=8, min_instance_storage_gb=0, max_instance_storage_gb=256)


def test_resolve_max_gpu_less_than_min_raises():
    """ValueError raised when max_gpu < min_gpu."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        with pytest.raises(ValueError, match="max_gpu.*min_gpu"):
            core.resolve_instance_types(instance_types=[], min_vcpu=0, min_ram_gb=0.5, min_gpu=4, max_vcpu=64, max_ram_gb=128, max_gpu=2, min_instance_storage_gb=0, max_instance_storage_gb=256)


def test_resolve_max_storage_less_than_min_raises():
    """ValueError raised when max_instance_storage_gb < min_instance_storage_gb."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        with pytest.raises(ValueError, match="max_instance_storage_gb.*min_instance_storage_gb"):
            core.resolve_instance_types(instance_types=[], min_vcpu=0, min_ram_gb=0.5, min_gpu=0, max_vcpu=64, max_ram_gb=128, max_gpu=8, min_instance_storage_gb=200, max_instance_storage_gb=100)


def test_negative_min_vcpu_triggers_warning(mock_ec2):
    """Negative min_vcpu triggers a warning log."""
    mock_ec2.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core.logger") as mock_logger:
        result = core.resolve_instance_types(
            instance_types=[],
            min_vcpu=-1,
            min_ram_gb=0.5,
            min_gpu=0,
            max_vcpu=64,
            max_ram_gb=128,
            max_gpu=8,
            min_instance_storage_gb=0,
            max_instance_storage_gb=256,
        )
    mock_logger.warning.assert_any_call("min_vcpu is negative (%d); treated as unbounded", -1)


def test_negative_max_ram_gb_triggers_warning(mock_ec2):
    """Negative max_ram_gb triggers a warning log."""
    mock_ec2.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core.logger") as mock_logger:
        result = core.resolve_instance_types(
            instance_types=[],
            min_vcpu=0,
            min_ram_gb=-1.0,
            min_gpu=0,
            max_vcpu=64,
            max_ram_gb=-0.5,
            max_gpu=8,
            min_instance_storage_gb=0,
            max_instance_storage_gb=256,
        )
    mock_logger.warning.assert_any_call("max_ram_gb is negative (%f); treated as unbounded", -0.5)
    mock_logger.warning.assert_any_call("min_ram_gb is negative (%f); treated as unbounded", -1.0)


def test_fetch_region_prices_consumes_multiple_pages(mock_ec2):
    """Multi-page responses are fully consumed."""
    mock_ec2.describe_spot_price_history.side_effect = [
        {
            "SpotPriceHistory": [
                {
                    "Region": "us-east-1",
                    "InstanceId": "us-east-1:us-east-1a",
                    "AvailabilityZone": "us-east-1a",
                    "InstanceType": "t3.micro",
                    "SpotPrice": "0.010",
                    "Timestamp": dt.datetime(2025, 10, 1, 0, 0, tzinfo=dt.timezone.utc),
                },
            ],
            "NextToken": "page2",
        },
        {
            "SpotPriceHistory": [
                {
                    "Region": "us-east-1",
                    "InstanceId": "us-east-1:us-east-1a",
                    "AvailabilityZone": "us-east-1a",
                    "InstanceType": "t3.micro",
                    "SpotPrice": "0.011",
                    "Timestamp": dt.datetime(2025, 10, 1, 1, 0, tzinfo=dt.timezone.utc),
                },
            ],
        },
    ]
    result = core._fetch_region_prices(
        "us-east-1",
        ["t3.micro"],
        dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2025, 10, 2, tzinfo=dt.timezone.utc),
        "Linux/UNIX",
    )
    assert len(result) == 2
    assert mock_ec2.describe_spot_price_history.call_count == 2
