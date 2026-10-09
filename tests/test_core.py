"""Tests for ec2_spot_query.core — driven by TDD cycle."""

from __future__ import annotations

pytest_plugins = ("tests.test_utils",)

import datetime as dt

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

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
    assert "region" in result.columns
    assert result.iloc[0]["region"] == "us-east-1"


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
    region_cols = result["region"].tolist()
    assert "us-east-1" in region_cols
    assert "us-west-2" in region_cols


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


# ---------------------------------------------------------------------------
# Progress callback tests (folded from test_core_progress.py)
# ---------------------------------------------------------------------------

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
        days=30,
        progress_callback=callback,
        product_description="Linux/UNIX",
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
        days=30,
        progress_callback=None,
        product_description="Linux/UNIX",
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
        days=30,
        progress_callback=callback,
        product_description="Linux/UNIX",
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
        days=30,
        progress_callback=callback,
        product_description="Linux/UNIX",
    )

    # 2 instance types × 2 regions = 4 pairs
    pairs_seen = [r[0] for r in received]
    assert len(pairs_seen) == 4
    assert any("t3.micro" in p and "us-east-1" in p for p in pairs_seen)
    assert any("t3.micro" in p and "us-west-2" in p for p in pairs_seen)
    assert any("t3.large" in p and "us-east-1" in p for p in pairs_seen)
    assert any("t3.large" in p and "us-west-2" in p for p in pairs_seen)


def test_progress_callback_records_offset(mock_ec2):
    """records_offset seeds the running total so cached records are counted."""
    mock_ec2.describe_spot_price_history.side_effect = [
        {
            "SpotPriceHistory": [
                _make_spot_record("t3.micro", "us-east-1a", 0.010),
                _make_spot_record("t3.micro", "us-east-1a", 0.011),
            ]
        },
        {
            "SpotPriceHistory": [
                _make_spot_record("t3.micro", "us-east-1b", 0.012),
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
        days=30,
        progress_callback=callback,
        product_description="Linux/UNIX",
        records_offset=100,
    )

    assert len(result) == 3
    assert len(received) == 2
    totals = [t for _, t in received]
    assert all(t >= 100 for t in totals)
    assert totals == sorted(totals), "running total must be monotonic"
    assert totals[-1] == 103, "offset (100) + fetched (3)"


# ---------------------------------------------------------------------------
# Architecture filter tests
# ---------------------------------------------------------------------------

def test_map_architectures_none():
    """None input means no architecture filter."""
    assert core.map_architectures(None) is None


def test_map_architectures_empty_input():
    """Empty list / empty string input means no architecture filter."""
    assert core.map_architectures([]) is None
    assert core.map_architectures("") is None
    assert core.map_architectures(["  ", ""]) is None


def test_map_architectures_x86_aliases():
    """All x86-family aliases map to x86_64."""
    for alias in ("x86", "x86_64", "intel", "amd"):
        assert core.map_architectures(alias) == ["x86_64"]


def test_map_architectures_arm_aliases():
    """All arm-family aliases map to arm64."""
    for alias in ("arm", "arm64", "graviton", "aarch64"):
        assert core.map_architectures(alias) == ["arm64"]


def test_map_architectures_mac_expands():
    """mac expands to both mac architecture values."""
    assert core.map_architectures("mac") == ["arm64_mac", "x86_64_mac"]


def test_map_architectures_case_insensitive():
    """Input tokens are matched case-insensitively."""
    assert core.map_architectures("ARM") == ["arm64"]
    assert core.map_architectures("Graviton") == ["arm64"]
    assert core.map_architectures(["X86", "ARM"]) == ["x86_64", "arm64"]


def test_map_architectures_official_passthrough():
    """Official AWS values pass through unchanged."""
    assert core.map_architectures("riscv64") == ["riscv64"]
    assert core.map_architectures("x86_64_mac") == ["x86_64_mac"]
    assert core.map_architectures("arm64_mac") == ["arm64_mac"]


def test_map_architectures_deduplicates():
    """Equivalent aliases collapse to one value, preserving first-seen order."""
    assert core.map_architectures(["arm", "graviton", "aarch64"]) == ["arm64"]
    assert core.map_architectures(["x86", "arm", "intel"]) == ["x86_64", "arm64"]


def test_map_architectures_strips_whitespace():
    """Surrounding whitespace on tokens is ignored."""
    assert core.map_architectures(["  arm  "]) == ["arm64"]


def test_map_architectures_unknown_raises():
    """Unknown tokens raise ValueError."""
    with pytest.raises(ValueError, match="Unknown architecture"):
        core.map_architectures("sparc")
    with pytest.raises(ValueError, match="Unknown architecture"):
        core.map_architectures(["arm", "bogus"])


def test_resolve_arch_filter_passed_to_api():
    """architectures are passed as a processor-info.supported-architecture filter."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "t3.micro", "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 1024}},
        ]
    }
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(
            instance_types=[], min_vcpu=0, min_ram_gb=0.5, min_gpu=0,
            max_vcpu=64, max_ram_gb=128, max_gpu=8,
            min_instance_storage_gb=0, max_instance_storage_gb=256,
            architectures=["arm64"],
        )
    _, kwargs = mock_client.describe_instance_types.call_args
    assert kwargs["Filters"] == [
        {"Name": "processor-info.supported-architecture", "Values": ["arm64"]},
    ]
    assert result == ["t3.micro"]


def test_resolve_arch_multiple_values_passed():
    """Multiple architecture values are all passed in the filter."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        core.resolve_instance_types(
            instance_types=[], min_vcpu=0, min_ram_gb=0.5, min_gpu=0,
            max_vcpu=64, max_ram_gb=128, max_gpu=8,
            min_instance_storage_gb=0, max_instance_storage_gb=256,
            architectures=["arm64_mac", "x86_64_mac"],
        )
    _, kwargs = mock_client.describe_instance_types.call_args
    assert kwargs["Filters"] == [
        {"Name": "processor-info.supported-architecture", "Values": ["arm64_mac", "x86_64_mac"]},
    ]


def test_resolve_no_architectures_omits_filters():
    """Without architectures the describe call carries no Filters (unchanged behavior)."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        core.resolve_instance_types(
            instance_types=[], min_vcpu=0, min_ram_gb=0.5, min_gpu=0,
            max_vcpu=64, max_ram_gb=128, max_gpu=8,
            min_instance_storage_gb=0, max_instance_storage_gb=256,
        )
    _, kwargs = mock_client.describe_instance_types.call_args
    assert "Filters" not in kwargs


def test_resolve_architectures_ignored_when_types_provided():
    """Explicit instance types short-circuit; the API filter is never applied."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        result = core.resolve_instance_types(
            instance_types=["t3.micro"], min_vcpu=0, min_ram_gb=0.5, min_gpu=0,
            max_vcpu=64, max_ram_gb=128, max_gpu=8,
            min_instance_storage_gb=0, max_instance_storage_gb=256,
            architectures=["arm64"],
        )
    assert result == ["t3.micro"]
    mock_client.describe_instance_types.assert_not_called()


def test_resolve_invalid_architecture_raises():
    """Non-official architecture values raise ValueError."""
    mock_client = MagicMock()
    mock_client.describe_instance_types.return_value = {"InstanceTypes": []}
    with patch("ec2_spot_query.core._DEFAULT_CLIENT", mock_client):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            core.resolve_instance_types(
                instance_types=[], min_vcpu=0, min_ram_gb=0.5, min_gpu=0,
                max_vcpu=64, max_ram_gb=128, max_gpu=8,
                min_instance_storage_gb=0, max_instance_storage_gb=256,
                architectures=["sparc"],
            )
    mock_client.describe_instance_types.assert_not_called()


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

    result = core.fetch_spot_prices(["t2.nano"], regions=["us-east-1"], days=30, product_description="Linux/UNIX")
    assert len(result) == 1
    assert result[0]["InstanceType"] == "t2.nano"


# ---------------------------------------------------------------------------
# Glob expansion tests
# ---------------------------------------------------------------------------

_REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "eu-west-2", "eu-central-1", "eu-north-1", "ap-southeast-2"]
_TYPES = ["t3.micro", "t3.large", "m7g.xlarge", "m7g.2xlarge", "m7i.xlarge", "inf2.xlarge", "inf2.2xlarge", "inf2.48xlarge"]


def test_has_glob_true():
    """Patterns containing *, ? or [ are glob patterns."""
    assert core.has_glob("eu-*")
    assert core.has_glob("inf2.*")
    assert core.has_glob("us-ea?t-1")
    assert core.has_glob("m7[gr].large")
    assert core.has_glob("a[!b]c")


def test_has_glob_false():
    """Plain names without glob characters are not glob patterns."""
    assert not core.has_glob("t3.micro")
    assert not core.has_glob("eu-west-1")
    assert not core.has_glob("")


def test_match_names_star():
    """Star pattern matches all names with the prefix."""
    assert core.match_names(_REGIONS, "eu-*") == ["eu-central-1", "eu-north-1", "eu-west-1", "eu-west-2"]


def test_match_names_question_mark():
    """Question mark matches exactly one character."""
    assert core.match_names(_TYPES, "t3.mi?ro") == ["t3.micro"]
    assert core.match_names(_TYPES, "t3.mi?o") == []


def test_match_names_bracket():
    """Bracket expression matches any character in the set (and negation)."""
    assert core.match_names(_TYPES, "m7[gi].xlarge") == ["m7g.xlarge", "m7i.xlarge"]
    assert core.match_names(_TYPES, "m7[!i].xlarge") == ["m7g.xlarge"]


def test_match_names_no_match():
    """A pattern matching nothing returns an empty list."""
    assert core.match_names(_REGIONS, "zzz-*") == []


def test_match_names_sorted():
    """Results are returned sorted regardless of input order."""
    shuffled = list(reversed(_TYPES))
    assert core.match_names(shuffled, "inf2.*") == ["inf2.2xlarge", "inf2.48xlarge", "inf2.xlarge"]


def test_match_names_case_sensitive():
    """Matching is case-sensitive: uppercase patterns match nothing."""
    assert core.match_names(_TYPES, "INF2.*") == []
    assert core.match_names(_TYPES, "Inf2.*") == []


def test_expand_names_literal_passthrough():
    """Patterns without glob characters pass through as-is, even if absent from names."""
    assert core.expand_names(_REGIONS, ["us-east-1"]) == ["us-east-1"]
    assert core.expand_names(_REGIONS, ["us-east-1", "us-iso-east-1"]) == ["us-east-1", "us-iso-east-1"]


def test_expand_names_glob_only():
    """Glob patterns expand against the name list."""
    assert core.expand_names(_REGIONS, ["eu-*"]) == ["eu-central-1", "eu-north-1", "eu-west-1", "eu-west-2"]


def test_expand_names_mixed_literals_and_globs():
    """Literals and globs are unioned."""
    result = core.expand_names(_TYPES, ["t3.micro", "inf2.*"])
    assert result == ["inf2.2xlarge", "inf2.48xlarge", "inf2.xlarge", "t3.micro"]


def test_expand_names_deduplicates():
    """Overlapping patterns produce no duplicates."""
    result = core.expand_names(_TYPES, ["inf2.*", "inf2.xlarge", "t3.*", "t3.micro"])
    assert result == ["inf2.2xlarge", "inf2.48xlarge", "inf2.xlarge", "t3.large", "t3.micro"]
    assert len(result) == len(set(result))


def test_expand_names_no_match():
    """All-glob input with no matches returns an empty list."""
    assert core.expand_names(_REGIONS, ["zzz-*"]) == []


def test_expand_names_mixed_no_match_keeps_literals():
    """A glob with no matches does not discard literal patterns."""
    assert core.expand_names(_REGIONS, ["zzz-*", "eu-west-1"]) == ["eu-west-1"]


def test_list_instance_types(mock_ec2):
    """list_instance_types returns all instance type names from describe_instance_types."""
    mock_ec2.describe_instance_types.return_value = {
        "InstanceTypes": [
            {"InstanceType": "t3.micro"},
            {"InstanceType": "inf2.xlarge"},
            {"InstanceType": "inf2.2xlarge"},
        ]
    }
    assert core.list_instance_types() == ["t3.micro", "inf2.xlarge", "inf2.2xlarge"]
    mock_ec2.describe_instance_types.assert_called_once()
