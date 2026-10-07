"""Core logic for EC2 spot price analysis."""

from __future__ import annotations

import datetime as dt
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import boto3
import botocore.config
import pandas as pd

# Adaptive retry config to handle RequestLimitExceeded / ThrottlingException
_EC2_CONFIG = botocore.config.Config(
    retries={'max_attempts': 10, 'mode': 'adaptive'},
)

logger = logging.getLogger(__name__)


def _get_retry_count(resp_meta: dict | None) -> int:
    """Extract retry count from response metadata, safe for MagicMock in tests."""
    if resp_meta is None:
        return 0
    val = resp_meta.get("RetryAttempts")
    return val if isinstance(val, int) else 0

# Time windows in hours
_WINDOWS = {
    "1h": pd.Timedelta(hours=1),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(days=7),
    "1m": pd.Timedelta(days=30),
}


def resolve_instance_types(
    instance_types: list[str] | None,
    min_vcpu: int,
    min_ram_gb: float,
    min_gpu: int,
    region: str,
    max_vcpu: int,
    max_ram_gb: float,
    max_gpu: int,
    min_instance_storage_gb: float,
    max_instance_storage_gb: float,
) -> list[str]:
    """Resolve target instance types from input or EC2 describe_instance_types call.

    If *instance_types* is non-empty it is returned as-is (minus empty strings).
    Otherwise ``describe_instance_types`` is queried and filtered by hardware
    constraints (min and max for vCPUs, RAM, GPU, and instance storage).

    Raises:
        ValueError: If any max parameter is less than its corresponding min parameter.
    """
    if instance_types:
        return [t for t in instance_types if t]

    # Log warnings for negative values (interpreted as unbounded)
    if min_vcpu < 0:
        logger.warning("min_vcpu is negative (%d); treated as unbounded", min_vcpu)
    if min_ram_gb < 0:
        logger.warning("min_ram_gb is negative (%f); treated as unbounded", min_ram_gb)
    if min_gpu < 0:
        logger.warning("min_gpu is negative (%d); treated as unbounded", min_gpu)
    if min_instance_storage_gb < 0:
        logger.warning("min_instance_storage_gb is negative (%f); treated as unbounded", min_instance_storage_gb)
    if max_vcpu < 0:
        logger.warning("max_vcpu is negative (%d); treated as unbounded", max_vcpu)
    if max_ram_gb < 0:
        logger.warning("max_ram_gb is negative (%f); treated as unbounded", max_ram_gb)
    if max_gpu < 0:
        logger.warning("max_gpu is negative (%d); treated as unbounded", max_gpu)
    if max_instance_storage_gb < 0:
        logger.warning("max_instance_storage_gb is negative (%f); treated as unbounded", max_instance_storage_gb)

    # Validate min <= max constraints
    if max_vcpu < min_vcpu:
        raise ValueError(f"max_vcpu ({max_vcpu}) cannot be less than min_vcpu ({min_vcpu})")
    if max_ram_gb < min_ram_gb:
        raise ValueError(f"max_ram_gb ({max_ram_gb}) cannot be less than min_ram_gb ({min_ram_gb})")
    if max_gpu < min_gpu:
        raise ValueError(f"max_gpu ({max_gpu}) cannot be less than min_gpu ({min_gpu})")
    if max_instance_storage_gb < min_instance_storage_gb:
        raise ValueError(f"max_instance_storage_gb ({max_instance_storage_gb}) cannot be less than min_instance_storage_gb ({min_instance_storage_gb})")

    client = boto3.client("ec2", region_name=region, config=_EC2_CONFIG)
    t0 = time.perf_counter()
    resp = client.describe_instance_types()
    elapsed = time.perf_counter() - t0
    record_count = len(resp.get("InstanceTypes", []))
    resp_meta = resp.get("ResponseMetadata", {})
    retries = _get_retry_count(resp_meta)
    parts = [f"-> {record_count} types in {elapsed*1000:.0f}ms"]
    if retries > 0:
        parts.append(f"{retries} retries")
    logger.debug("EC2 describe_instance_types (%s) %s", region, ", ".join(parts))

    result: list[str] = []
    for inst in resp.get("InstanceTypes", []):
        vcpus = inst.get("VCpuInfo", {}).get("DefaultVCpus", 0)
        mem_mi = inst.get("MemoryInfo", {}).get("SizeInMiB", 0)
        mem_gb = mem_mi / 1024
        gpus = inst.get("GpuInfo", {}).get("Gpus", [{}])[0].get("Count", 0)
        storage_info = inst.get("InstanceStorageInfo")
        if isinstance(storage_info, dict):
            storage_gb = storage_info.get("TotalSizeInGB", 0)
        elif isinstance(storage_info, list):
            storage_gb = sum(d.get("TotalSizeInGB", 0) for d in storage_info)
        else:
            storage_gb = 0
        if (vcpus >= min_vcpu and vcpus <= max_vcpu and
            mem_gb >= min_ram_gb and mem_gb <= max_ram_gb and
            gpus >= min_gpu and gpus <= max_gpu and
            storage_gb >= min_instance_storage_gb and storage_gb <= max_instance_storage_gb):
            result.append(inst["InstanceType"])
    return list(dict.fromkeys(result))  # deduplicate while preserving order


def _fetch_region_prices(
    region: str,
    instance_types: list[str],
    start: dt.datetime,
    end: dt.datetime,
    product_description: str,
) -> list[dict[str, Any]]:
    """Fetch spot price history for a batch of instance types in one region.

    Handles API pagination internally via ``NextToken`` — all pages are
    consumed before returning.
    """
    client = boto3.client("ec2", region_name=region, config=_EC2_CONFIG)
    records: list[dict[str, Any]] = []
    total_page = 0
    total_records = 0
    first_it = instance_types[0]

    spot_history_kwargs: dict[str, Any] = {
        "StartTime": start,
        "EndTime": end,
        "ProductDescriptions": [product_description],
        "Filters": [
            {"Name": "instance-type", "Values": instance_types},
        ],
    }

    while True:
        t0 = time.perf_counter()
        resp = client.describe_spot_price_history(**spot_history_kwargs)
        page_records = 0
        for item in resp.get("SpotPriceHistory", []):
            ts = item.get("Timestamp")
            if isinstance(ts, str):
                ts = dt.datetime.fromisoformat(ts)
            if isinstance(ts, dt.datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            records.append({
                "Region": region,
                "InstanceId": f"{region}:{item.get('AvailabilityZone', '')}",
                "AvailabilityZone": item.get("AvailabilityZone", ""),
                "InstanceType": item.get("InstanceType", first_it),
                "SpotPrice": float(item.get("SpotPrice", 0)),
                "Timestamp": ts or dt.datetime.now(dt.timezone.utc),
                "ProductDescription": item.get("ProductDescription", product_description),
            })
            page_records += 1

        elapsed = time.perf_counter() - t0
        total_page += 1
        total_records += page_records
        next_token = resp.get("NextToken")
        if not next_token:
            break
        spot_history_kwargs["NextToken"] = next_token

    # Log concise summary after all pages are fetched
    parts = [f"-> {total_records} records in {elapsed*1000:.0f}ms"]
    if total_page > 1:
        parts.append(f"{total_page} pages")
    resp_meta = resp.get("ResponseMetadata", {})
    retries = _get_retry_count(resp_meta)
    if retries > 0:
        parts.append(f"{retries} retries")
    logger.debug("EC2 describe_spot_price_history (%s, %d types) %s", region, len(instance_types), ", ".join(parts))

    return records


def fetch_spot_prices(
    instance_types: list[str],
    regions: list[str],
    days: int,
    progress_callback: Callable[[str, int], None] | None = None,
    product_description: str = "Linux/UNIX",
) -> list[dict[str, Any]]:
    """Fetch spot price history concurrently across (instance_type, region) pairs.

    Instance types are batched into chunks of up to 50 per API call to reduce
    the number of describe_spot_price_history requests.  The thread pool is
    capped at 4 workers to stay under AWS service-level rate limits.

    Args:
        progress_callback: Optional callable pair(name, total_records) -> None.
            Called after each (instance_type, region) fetch completes.
        product_description: Product description filter for spot price queries.
            Defaults to "Linux/UNIX".
    """
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=days)
    all_records: list[dict[str, Any]] = []

    # Batch instance types into chunks of up to 50
    BATCH_SIZE = 50
    batches: list[list[str]] = []
    for i in range(0, len(instance_types), BATCH_SIZE):
        batches.append(instance_types[i:i + BATCH_SIZE])

    # Build futures: one per (region, batch) pair
    max_workers = min(4, len(batches) * len(regions)) if batches and regions else 4
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for reg in regions:
            for batch_idx, batch in enumerate(batches):
                fut = pool.submit(
                    _fetch_region_prices, reg, batch, start, now, product_description,
                )
                futures[fut] = (reg, batch_idx, batch)

        total_so_far = 0
        for future in as_completed(futures):
            reg, batch_idx, batch_types = futures[future]
            batch_records = future.result()
            all_records.extend(batch_records)
            total_so_far += len(batch_records)

            # Fire per-pair progress callbacks for backward compatibility
            if progress_callback:
                for itype in batch_types:
                    progress_callback(f"Fetching {itype} in {reg}", total_so_far)
    return all_records


def _compute_group_metrics(grp_sorted: pd.DataFrame, itype: str, region_az: str) -> dict[str, Any]:
    """Compute metric row for one (instance_type, region_az) group."""
    last_price = float(grp_sorted["SpotPrice"].iloc[-1])
    row: dict[str, Any] = {
        "region_az": region_az,
        "region": region_az.rsplit("/", 1)[0],
        "instance_type": itype,
        "current_price": last_price,
    }

    for wname, wdelta in _WINDOWS.items():
        if len(grp_sorted) < 2:
            mean_val = float(grp_sorted["SpotPrice"].mean())
            row[f"{wname}_mean"] = mean_val
            row[f"{wname}_vol"] = 0.0
            continue

        # Resample into the window size
        resampled = grp_sorted["SpotPrice"].resample(wdelta)

        # Collect all window-bucket means
        bucket_means: list[float] = []
        for _, bucket in resampled:
            if len(bucket) > 0:
                bucket_means.append(float(bucket.mean()))
                bucket_std = float(bucket.std())
                if pd.isna(bucket_std) or bucket_std == 0.0:
                    bucket_std = 0.0
                row.setdefault(f"{wname}_vol_buckets", []).append(bucket_std)

        if bucket_means:
            row[f"{wname}_mean"] = float(pd.Series(bucket_means).mean())
            vol_series = pd.Series(row.pop(f"{wname}_vol_buckets", [0.0]))
            overall_vol = float(vol_series.std())
            if pd.isna(overall_vol) or overall_vol == 0.0:
                overall_vol = 0.0
            row[f"{wname}_vol"] = overall_vol
        else:
            row[f"{wname}_mean"] = mean_val
            row[f"{wname}_vol"] = 0.0

    return row


def compute_metrics(
    raw_data: list[dict[str, Any]],
    sort_by: str = "1d_mean",
) -> pd.DataFrame:
    """Load raw spot-price data into a DataFrame and compute metrics.

    For every (InstanceType, AZ) group, mean and standard deviation are
    computed for each of the 6 time windows (1 h … 1 m).  Flat price
    histories get a volatility of ``0.0`` instead of ``NaN``.

    The returned DataFrame includes a ``region`` column alongside the
    existing ``region_az`` column.
    """
    if not raw_data:
        return _empty_result()

    df = pd.DataFrame(raw_data)
    df["SpotPrice"] = pd.to_numeric(df["SpotPrice"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values("Timestamp")

    # Build group key: region/az + instance_type
    df["region_az"] = df["Region"] + "/" + df["AvailabilityZone"]

    def _row_generator():
        for (itype, group_key), grp in df.groupby(["InstanceType", "region_az"]):
            grp_sorted = grp.set_index("Timestamp").sort_index()
            yield _compute_group_metrics(grp_sorted, itype, group_key)

    result = pd.DataFrame(list(_row_generator()))

    # Sort descending by the requested metric
    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=False).reset_index(drop=True)

    return result


def _empty_result() -> pd.DataFrame:
    """Return an empty DataFrame with the expected column schema."""
    ncols = [
        "region_az", "region", "instance_type", "current_price",
        "1h_mean", "1h_vol", "6h_mean", "6h_vol",
        "12h_mean", "12h_vol", "1d_mean", "1d_vol",
        "1w_mean", "1w_vol", "1m_mean", "1m_vol",
    ]
    return pd.DataFrame(columns=ncols)


def aggregate_by_region(
    df: pd.DataFrame,
    sort_by: str = "1d_mean",
) -> pd.DataFrame:
    """Collapse per-AZ metrics to per-region metrics.

    For each (instance_type, region), select the AZ with the minimum
    current_price and return its metrics.
    """
    if df.empty:
        return df

    df = df.copy()
    if "region" not in df.columns:
        df["region"] = df["region_az"].str.split("/").str[0]

    result = []
    for (itype, region), group in df.groupby(["instance_type", "region"]):
        best = group.loc[group["current_price"].idxmin()].copy()
        row: dict[str, Any] = {"region_az": region, "instance_type": itype}
        for col in df.columns:
            if col in ("region_az", "region", "instance_type"):
                continue
            row[col] = best[col]
        result.append(row)
    result_df = pd.DataFrame(result)

    if sort_by in result_df.columns and not result_df.empty:
        result_df = result_df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    return result_df
