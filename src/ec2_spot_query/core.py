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
import botocore.exceptions
import pandas as pd

from ec2_spot_query import cache, utils

# Adaptive retry config to handle RequestLimitExceeded / ThrottlingException
_EC2_CONFIG = botocore.config.Config(
    retries={'max_attempts': 10, 'mode': 'adaptive'},
)

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"
_DEFAULT_CLIENT = boto3.client("ec2", region_name=DEFAULT_REGION, config=_EC2_CONFIG)
BATCH_SIZE = 50

# Time windows in hours
_WINDOWS = {
    "1h": pd.Timedelta(hours=1),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(days=7),
    "1m": pd.Timedelta(days=30),
}


def _filter_instance(
    inst: dict[str, Any],
    min_vcpu: int,
    max_vcpu: int,
    min_ram_gb: float,
    max_ram_gb: float,
    min_gpu: int,
    max_gpu: int,
    min_instance_storage_gb: float,
    max_instance_storage_gb: float,
) -> bool:
    """Return True if *inst* satisfies all hardware constraints."""
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
    return (
        min_vcpu <= vcpus <= max_vcpu and
        min_ram_gb <= mem_gb <= max_ram_gb and
        min_gpu <= gpus <= max_gpu and
        min_instance_storage_gb <= storage_gb <= max_instance_storage_gb
    )


def resolve_instance_types(
    instance_types: list[str] | None,
    min_vcpu: int,
    min_ram_gb: float,
    min_gpu: int,
    max_vcpu: int,
    max_ram_gb: float,
    max_gpu: int,
    min_instance_storage_gb: float,
    max_instance_storage_gb: float,
    architectures: list[str] | None = None,
    regions: list[str] | None = None,
) -> list[str]:
    """Resolve target instance types from input or EC2 describe_instance_types call.

    If *instance_types* is non-empty it is returned as-is (minus empty strings).
    Otherwise ``describe_instance_types`` is queried and filtered by hardware
    constraints (min and max for vCPUs, RAM, GPU, and instance storage).

    When *architectures* is non-empty, a ``processor-info.supported-architecture``
    filter is passed to ``describe_instance_types`` so that unwanted
    architectures never reach the spot-price fetch stage.

    When *regions* is provided (non-empty list), ``describe_instance_types`` is
    called from **each** region and results are unioned.  This is needed because
    the API is regional — a single region's catalog may not list instance types
    that are available in other regions.

    When *regions* is ``None`` (the default), the shared ``_DEFAULT_CLIENT``
    (us-east-1) is used — a single call.  This preserves backward compatibility
    for callers that don't specify regions.

    Raises:
        ValueError: If any max parameter is less than its corresponding min
            parameter, or if *architectures* contains a value that is not an
            official AWS architecture.
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

    if architectures:
        for arch in architectures:
            if arch not in utils.OFFICIAL_ARCHITECTURES:
                valid = ", ".join(sorted(utils.OFFICIAL_ARCHITECTURES))
                raise ValueError(f"Unsupported architecture {arch!r}. Valid values: {valid}")

    # Build the describe call kwargs (shared across all regions)
    describe_kwargs: dict[str, Any] = {}
    if architectures:
        describe_kwargs["Filters"] = [
            {"Name": "processor-info.supported-architecture", "Values": list(architectures)},
        ]

    if regions:
        # Multi-region: query each region and union results
        all_result: set[str] = set()
        total_count = 0
        for reg in regions:
            try:
                client = boto3.client("ec2", region_name=reg, config=_EC2_CONFIG)
                t0 = time.perf_counter()
                resp = client.describe_instance_types(**describe_kwargs)
                elapsed = time.perf_counter() - t0
                record_count = len(resp.get("InstanceTypes", []))
                resp_meta = resp.get("ResponseMetadata", {})
                retries = utils.get_retry_count(resp_meta)
                parts = [f"-> {record_count} types in {elapsed*1000:.0f}ms"]
                if retries > 0:
                    parts.append(f"{retries} retries")
                if architectures:
                    parts.append(f"arch[{','.join(architectures)}]")
                logger.debug("EC2 describe_instance_types (%s) %s", reg, ", ".join(parts))

                for inst in resp.get("InstanceTypes", []):
                    if _filter_instance(
                        inst, min_vcpu, max_vcpu,
                        min_ram_gb, max_ram_gb,
                        min_gpu, max_gpu,
                        min_instance_storage_gb, max_instance_storage_gb,
                    ):
                        all_result.add(inst["InstanceType"])
                total_count += record_count
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                if isinstance(e, botocore.exceptions.ClientError):
                    code = e.response["Error"].get("Code", "Unknown")
                    logger.warning("describe_instance_types (%s): %s (region skipped)", reg, code)
                else:
                    logger.warning("describe_instance_types (%s): %s (region skipped)", reg, e)

        logger.info("EC2 describe_instance_types (%d regions): %d unique types across %d total responses",
                     len(regions), len(all_result), total_count)
        return sorted(all_result)

    else:
        # Fallback: single call from _DEFAULT_CLIENT (us-east-1)
        client = _DEFAULT_CLIENT
        t0 = time.perf_counter()
        resp = client.describe_instance_types(**describe_kwargs)
        elapsed = time.perf_counter() - t0
        record_count = len(resp.get("InstanceTypes", []))
        resp_meta = resp.get("ResponseMetadata", {})
        retries = utils.get_retry_count(resp_meta)
        parts = [f"-> {record_count} types in {elapsed*1000:.0f}ms"]
        if retries > 0:
            parts.append(f"{retries} retries")
        if architectures:
            parts.append(f"arch[{','.join(architectures)}]")
        logger.debug("EC2 describe_instance_types (%s) %s", DEFAULT_REGION, ", ".join(parts))

        result: list[str] = []
        for inst in resp.get("InstanceTypes", []):
            if _filter_instance(
                inst, min_vcpu, max_vcpu,
                min_ram_gb, max_ram_gb,
                min_gpu, max_gpu,
                min_instance_storage_gb, max_instance_storage_gb,
            ):
                result.append(inst["InstanceType"])
        return list(set(result))


def list_regions() -> list[str]:
    """Return all AWS region names.

    ``describe_regions`` is a global call — the result is identical from
    any region, so the shared default-region client is used.
    """
    resp = _DEFAULT_CLIENT.describe_regions()
    return [r["RegionName"] for r in resp.get("Regions", [])]


def list_instance_types(regions: list[str] | None = None) -> list[str]:
    """Return all EC2 instance type names.

    When *regions* is provided, ``describe_instance_types`` is queried from
    each region and results are unioned (deduplicated).  This is needed because
    the API is regional — a single region's catalog may not list all types.

    When *regions* is ``None``, the shared ``_DEFAULT_CLIENT`` (us-east-1) is
    used.  Used to expand glob patterns such as ``inf2.*``.
    """
    if regions:
        all_names: set[str] = set()
        for reg in regions:
            try:
                client = boto3.client("ec2", region_name=reg, config=_EC2_CONFIG)
                resp = client.describe_instance_types()
                all_names.update(inst["InstanceType"] for inst in resp.get("InstanceTypes", []))
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                if isinstance(e, botocore.exceptions.ClientError):
                    code = e.response["Error"].get("Code", "Unknown")
                    logger.warning("describe_instance_types (%s): %s (region skipped)", reg, code)
                else:
                    logger.warning("describe_instance_types (%s): %s (region skipped)", reg, e)
        return sorted(all_names)

    resp = _DEFAULT_CLIENT.describe_instance_types()
    return [inst["InstanceType"] for inst in resp.get("InstanceTypes", [])]


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
        try:
            resp = client.describe_spot_price_history(**spot_history_kwargs)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            if isinstance(e, botocore.exceptions.ClientError):
                code = e.response["Error"]["Code"]
                message = e.response["Error"].get("Message", "")
                logger.warning(
                    "Skipping %s: %s - %s (region may not support spot pricing)",
                    region, code, message,
                )
            else:
                logger.warning(
                    "Skipping %s: %s (region may be unreachable or not support spot pricing)",
                    region, e,
                )
            return []
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
    retries = utils.get_retry_count(resp_meta)
    if retries > 0:
        parts.append(f"{retries} retries")
    logger.debug("EC2 describe_spot_price_history (%s, %d types) %s", region, len(instance_types), ", ".join(parts))

    return records


def fetch_spot_prices(
    instance_types: list[str],
    regions: list[str],
    days: int,
    product_description: str,
    progress_callback: Callable[[str, int], None] | None = None,
    records_offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch spot price history concurrently across (instance_type, region) pairs.

    Instance types are batched into chunks of up to 50 per API call to reduce
    the number of describe_spot_price_history requests.  The thread pool is
    capped at 4 workers to stay under AWS service-level rate limits.

    Args:
        product_description: Product description filter for spot price queries.
        progress_callback: Optional callable pair(name, total_records) -> None.
            Called after each (instance_type, region) fetch completes.
        records_offset: Number of records already accounted for before this
            call (e.g. records served from the cache). The reported
            ``total_records`` in each callback is seeded from this value so
            callers can maintain a single running total across fetches.
    """
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=days)
    all_records: list[dict[str, Any]] = []

    # Batch instance types into chunks of up to 50
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

        total_so_far = records_offset
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


def _compute_group_metrics(
    grp_sorted: pd.DataFrame,
    itype: str,
    region_az: str,
    max_window: str = "1w",
) -> dict[str, Any]:
    """Compute metric row for one (instance_type, region_az) group."""
    last_price = float(grp_sorted["SpotPrice"].iloc[-1])
    # Read the region straight from the group data rather than re-parsing it
    # out of the region_az key.
    region = str(grp_sorted["Region"].iloc[0])
    row: dict[str, Any] = {
        "region_az": region_az,
        "region": region,
        "instance_type": itype,
        "current_price": last_price,
        "placement_score": None,
    }

    window_keys = list(_WINDOWS.keys())
    max_idx = window_keys.index(max_window)
    active_windows = dict(list(_WINDOWS.items())[:max_idx + 1])

    for wname, wdelta in active_windows.items():
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
    max_window: str = "1w",
) -> pd.DataFrame:
    """Load raw spot-price data into a DataFrame and compute metrics.

    For every (InstanceType, AZ) group, mean and standard deviation are
    computed for each time window up to and including *max_window*.
    Flat price histories get a volatility of ``0.0`` instead of ``NaN``.

    The returned DataFrame includes a ``region`` column alongside the
    existing ``region_az`` column.

    Args:
        max_window: Maximum time window to compute (e.g. "1w", "1m").
            Only windows up to and including this value are included.
    """
    window_keys = list(_WINDOWS.keys())
    if max_window not in _WINDOWS:
        raise ValueError(f"Unknown time window {max_window!r}. Valid values: {', '.join(window_keys)}")
    if not raw_data:
        return _empty_result(max_window)

    df = pd.DataFrame(raw_data)
    df["SpotPrice"] = pd.to_numeric(df["SpotPrice"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values("Timestamp")

    # Build group key: region/az + instance_type
    df["region_az"] = df["Region"] + "/" + df["AvailabilityZone"]

    def _row_generator():
        for (itype, group_key), grp in df.groupby(["InstanceType", "region_az"]):
            grp_sorted = grp.set_index("Timestamp").sort_index()
            yield _compute_group_metrics(grp_sorted, itype, group_key, max_window)

    result = pd.DataFrame(list(_row_generator()))

    # Sort descending by the requested metric
    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=False).reset_index(drop=True)

    return result


def _empty_result(max_window: str = "1w") -> pd.DataFrame:
    """Return an empty DataFrame with the expected column schema."""
    window_keys = list(_WINDOWS.keys())
    if max_window not in _WINDOWS:
        raise ValueError(f"Unknown time window {max_window!r}. Valid values: {', '.join(window_keys)}")
    max_idx = window_keys.index(max_window)
    active_keys = window_keys[:max_idx + 1]
    ncols = ["region_az", "region", "instance_type", "current_price"]
    for k in active_keys:
        ncols.extend([f"{k}_mean", f"{k}_vol"])
    ncols.append("placement_score")
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
        row: dict[str, Any] = {"region_az": region, "region": region, "instance_type": itype}
        for col in df.columns:
            if col in ("region_az", "region", "instance_type"):
                continue
            row[col] = best[col]
        result.append(row)
    result_df = pd.DataFrame(result)

    if sort_by in result_df.columns and not result_df.empty:
        result_df = result_df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    return result_df


# ---------------------------------------------------------------------------
# Spot Placement Score helpers
# ---------------------------------------------------------------------------

_PLACEMENT_SCORE_REGION_BATCH = 10


def get_az_id_mapping(region: str) -> dict[str, str]:
    """Resolve AZ names to AZ IDs for the given region.

    Returns a mapping ``{"us-east-1a": "eun1-az2", ...}``.
    Results are cached on disk (1-year TTL) since AZ IDs are stable per
    account/region.
    """
    # Check cache first
    az_map_key = cache.make_az_map_cache_key(region)
    cached = cache.load_cache(az_map_key, ttl_seconds=cache.AZ_MAP_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        # Create a per-region client because describe_availability_zones
        # with a region-name filter only works from within that region
        client = boto3.client("ec2", region_name=region, config=_DEFAULT_CLIENT.meta.config)
        resp = client.describe_availability_zones()
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        if isinstance(e, botocore.exceptions.ClientError):
            code = e.response["Error"]["Code"]
            message = e.response["Error"].get("Message", "")
            logger.warning(
                "describe_availability_zones (%s): %s - %s",
                region, code, message,
            )
        else:
            logger.warning("describe_availability_zones (%s): %s", region, e)
        return {}

    mapping: dict[str, str] = {
        zone["ZoneName"]: zone["ZoneId"]
        for zone in resp.get("AvailabilityZones", [])
    }
    # Cache with long TTL (AZ IDs rarely change)
    cache.save_cache(az_map_key, mapping, ttl_seconds=cache.AZ_MAP_TTL_SECONDS)
    return mapping


def get_spot_placement_scores(
    instance_types: list[str],
    regions: list[str],
    target_capacity: int = 1,
) -> dict[str, dict[str, dict[str, int]]]:
    """Fetch Spot Placement Scores across regions.

    Returns a nested dict: ``{region: {instance_type: {az_name: score}}}``
    where *score* is an integer 1–10.

    The AWS API returns scores per AZ (same for all instance types in that AZ),
    so scores are replicated across all requested instance types.

    Regions are batched in chunks of 10 per API call.
    AZ IDs in the response are resolved back to human-readable names.
    """
    # Build AZ ID → (az_name, region) lookup
    az_id_to_info: dict[str, tuple[str, str]] = {}
    for reg in regions:
        az_mapping = get_az_id_mapping(reg)
        id_to_name: dict[str, str] = {v: k for k, v in az_mapping.items()}
        for az_id, az_name in id_to_name.items():
            az_id_to_info[az_id] = (az_name, reg)

    result: dict[str, dict[str, dict[str, int]]] = {}
    for reg in regions:
        result[reg] = {}
        for itype in instance_types:
            result[reg][itype] = {}

    # Batch regions into groups of 10 per API call
    for i in range(0, len(regions), _PLACEMENT_SCORE_REGION_BATCH):
        batch = regions[i:i + _PLACEMENT_SCORE_REGION_BATCH]
        try:
            resp = _DEFAULT_CLIENT.get_spot_placement_scores(
                InstanceTypes=instance_types,
                RegionNames=batch,
                SingleAvailabilityZone=True,
                TargetCapacity=target_capacity,
            )
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            if isinstance(e, botocore.exceptions.ClientError):
                code = e.response["Error"]["Code"]
                message = e.response["Error"].get("Message", "")
                logger.warning(
                    "get_spot_placement_scores (%s): %s - %s",
                    ", ".join(batch), code, message,
                )
            else:
                logger.warning(
                    "get_spot_placement_scores (%s): %s", ", ".join(batch), e,
                )
            continue

        for score_entry in resp.get("SpotPlacementScores", []):
            az_id = score_entry.get("AvailabilityZoneId", "")
            score = score_entry.get("Score", 0)

            info = az_id_to_info.get(az_id)
            if info is None:
                continue

            az_name, reg = info

            # Score is per AZ — replicate across all instance types
            for itype in instance_types:
                result[reg][itype][az_name] = score

    return result


def filter_by_placement_score(
    df: pd.DataFrame,
    min_score: int,
    scores: dict[str, dict[str, dict[str, int]]],
) -> pd.DataFrame:
    """Drop rows whose (instance_type, region, az) placement score is < min_score.

    If no placement score is available for a row, it is kept (no filtering).
    """
    if df.empty:
        return df

    df = df.copy()
    mask = pd.Series(True, index=df.index)

    for _, row in df.iterrows():
        region = row.get("region", "")
        itype = row.get("instance_type", "")
        region_az = row.get("region_az", "")

        score = row.get("placement_score")
        if score is None:
            # No score data — keep row
            continue

        # Look up the score from the placement scores dict
        expected_score = scores.get(region, {}).get(itype, {}).get(region_az.rsplit("/", 1)[-1], None)

        if expected_score is None:
            # No score data available for this row — keep it
            continue

        if expected_score < min_score:
            mask.loc[row.name] = False

    return df.loc[mask].reset_index(drop=True)
