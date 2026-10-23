"""CLI entry point, rendering, and progress handling."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable

import pandas as pd
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    Table,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ec2_spot_query import cache, core, utils

_DAYS_FOR_WINDOW: dict[str, int] = {
    "1h": 1,
    "6h": 1,
    "12h": 1,
    "1d": 1,
    "1w": 7,
    "1m": 30,
}


app = typer.Typer(
    name="ec2-spot-query",
    help="Analyse and rank AWS Spot Market instances across multiple time horizons.",
    invoke_without_command=True,
)


# ---------------------------------------------------------------------------
# Progress / logging helpers
# ---------------------------------------------------------------------------

def _make_progress_handler(console: Console, use_progress: bool) -> Callable[[str, str, dict | None], None]:
    """Create a logging callback for rich console output.

    Returns callable(status, message, meta) -> None.
    """
    if not use_progress:
        stderr_console = Console(stderr=True)

        def _noop(status: str, message: str, meta: dict | None = None) -> None:
            if status == "error":
                stderr_console.print(f"[red]✗ {message}[/red]")
            elif status == "warn":
                stderr_console.print(f"[yellow]⚠ {message}[/yellow]")

        return _noop

    stderr_console = Console(stderr=True)
    status_ref: dict[str, object] = {"live": None}

    def _log(status: str, message: str, meta: dict | None = None) -> None:
        """Log a status message to the console."""
        records = meta.get("records", 0) if meta else 0
        records_str = f"[{records:,} records]" if records else ""
        suffix = f" {records_str}" if records_str else ""

        status_map: dict[str, Callable[[], None]] = {
            "init": lambda: console.rule(f"[cyan]{message}[/cyan]"),
            "query": lambda: (
                status_ref["live"].update(f"[blue]● {message}{suffix}[/blue]")
                if status_ref["live"]
                else console.print(f"[blue]● {message}{suffix}[/blue]")
            ),
            "progress": lambda: (
                status_ref["live"].update(f"[green]● {message}{suffix}[/green]")
                if status_ref["live"]
                else console.print(f"[green]● {message}{suffix}[/green]")
            ),
            "complete": lambda: console.log(f"[green]✓ {message}[/green]"),
            "error": lambda: stderr_console.print(f"[red]✗ {message}[/red]"),
            "warn": lambda: stderr_console.print(f"[yellow]⚠ {message}[/yellow]"),
            "info": lambda: stderr_console.print(f"[dim]{message}[/dim]") if records > 0 else None,
        }
        handler = status_map.get(status)
        if handler:
            handler()

    return _log


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_table(result_df: pd.DataFrame, sort_by: str = "1d_mean", limit: int = -1, per_az: bool = False, with_score: bool = False, max_window: str = "1w") -> None:
    """Render result DataFrame as a rich table to stdout."""
    console = Console()

    if result_df.empty:
        console.print("[yellow]No matching instances found.[/yellow]")
        return

    if limit > 0:
        result_df = result_df.sort_values(sort_by, ascending=True).head(limit).reset_index(drop=True)

    table = Table(title="EC2 Spot Instance Rankings")
    table.add_column("Region / AZ" if per_az else "Region")
    table.add_column("Instance Type")
    table.add_column("Current Price")

    _WINDOW_LABELS = {
        "1h": "1h Avg",
        "6h": "6h Avg",
        "12h": "12h Avg",
        "1d": "1d Avg",
        "1w": "1w Avg",
        "1m": "1m Avg",
    }

    window_keys = list(_DAYS_FOR_WINDOW.keys())
    if max_window not in _DAYS_FOR_WINDOW:
        max_window = "1w"
    max_idx = window_keys.index(max_window)
    active_windows = window_keys[:max_idx + 1]

    for w in active_windows:
        table.add_column(f"{_WINDOW_LABELS[w]}")
        table.add_column(f"{w} Vol")

    if with_score:
        table.add_column("Score")

    for _, row in result_df.iterrows():
        score_val = row.get("placement_score")
        score_str = str(score_val) if score_val is not None else "N/A"
        cells = [
            str(row["region_az"] if per_az else row["region"]),
            str(row["instance_type"]),
            f"${row['current_price']:.3f}",
        ]
        for w in active_windows:
            cells.append(f"{row[f'{w}_mean']:.3f}")
            cells.append(f"{row[f'{w}_vol']:.4f}")
        cells.append(score_str)
        table.add_row(*cells)

    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    instance_types: list[str] = typer.Option(None, "--instance-type", "-i", help="Instance type to query. Supports glob patterns (*, ?, [seq]), e.g. 'inf2.*'"),
    min_vcpu: int = typer.Option(0, "--min-vcpu", help="Minimum vCPUs"),
    min_ram: float = typer.Option(0.5, "--min-ram", help="Minimum RAM in GB"),
    min_gpu: int = typer.Option(0, "--min-gpu", help="Minimum GPU count"),
    max_vcpu: int = typer.Option(64, "--max-vcpu", help="Maximum vCPUs"),
    max_ram: float = typer.Option(128, "--max-ram", help="Maximum RAM in GB"),
    max_gpu: int = typer.Option(0, "--max-gpu", help="Maximum GPU count (default: 0 = CPU-only; use e.g. 8 or -1 to include GPUs)"),
    min_instance_storage: float = typer.Option(0, "--min-instance-storage", help="Minimum instance storage in GB"),
    max_instance_storage: float = typer.Option(256, "--max-instance-storage", help="Maximum instance storage in GB"),
    regions: list[str] = typer.Option(None, "--region", "-r", help="Region to query. Supports glob patterns (*, ?, [seq]), e.g. 'eu-*'"),
    all_regions: bool = typer.Option(False, "--all-regions", help="Query all regions"),
    sort_by: str = typer.Option("1d_mean", "--sort-by", help="Sort metric (1h_mean, 6h_mean, 12h_mean, 1d_mean, 1w_mean, 1m_mean)"),
    limit: int = typer.Option(10, "--limit", help="Number of rows to display (-1 for all)"),
    per_az: bool = typer.Option(True, "--per-az", help="Show per-AZ breakdown (default on; use --no-per-az for per-region)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache (skip load and save)"),
    no_cache_lookup: bool = typer.Option(False, "--no-cache-lookup", help="Fetch fresh data but still write cache"),
    progress: bool = typer.Option(False, "--progress", "-p", help="Show progress with rich spinner"),
    product_description: str = typer.Option("Linux/UNIX", "--product-description", help="Product description filter for spot price queries (e.g. Linux/UNIX, Windows, SUSE Linux)"),
    debug: bool = typer.Option(False, "--debug", help="Show detailed EC2 API request/response logs"),
    arch: list[str] | None = typer.Option(None, "--arch", "--architecture", help="Filter by processor architecture. Aliases: x86, arm, graviton, mac. Official values: x86_64, arm64, riscv64, x86_64_mac, arm64_mac. Repeat the flag for multiple architectures."),
    placement_score: bool = typer.Option(True, "--placement-score", help="Query AWS Spot Placement Scores for capacity availability (default on; use --no-placement-score to disable)"),
    min_placement_score: int = typer.Option(5, "--min-placement-score", help="Minimum placement score threshold (1-10, default: 5)"),
    target_capacity: int = typer.Option(1, "--target-capacity", help="Target capacity for placement score queries (default: 1)"),
    max_days: str = typer.Option("1w", "--max-days", help="Maximum time window to display (1h, 6h, 12h, 1d, 1w, 1m)"),
) -> None:
    """Analyse and rank AWS EC2 Spot instance prices."""
    if ctx.invoked_subcommand is not None:
        return

    sort_by = sort_by or "1d_mean"

    # Configure DEBUG logging for API call tracing
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
        logging.getLogger("botocore").setLevel(logging.INFO)
        logging.getLogger("botocore.endpoint").setLevel(logging.INFO)
        logging.getLogger("botocore.parsers").setLevel(logging.INFO)

    # CLI mode
    console = Console()
    _log = _make_progress_handler(console, progress)

    # Counter for skipped regions (logged as "Skipping ..." by core.logger)
    _skip_count: dict[str, int] = {"count": 0}

    class _SkipCountHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING and "Skipping" in record.getMessage():
                _skip_count["count"] += 1

    _skip_handler = _SkipCountHandler()
    core.logger.addHandler(_skip_handler)

    try:
        _log("init", "EC2 Spot Query")
        _log("info", "Validating AWS credentials...")

        # Map architecture aliases to official values (fails fast on unknown values)
        mapped_archs = utils.map_architectures(arch)

        # Resolve regions first (needed for multi-region instance type catalog)
        regions_need_expansion = bool(regions) and any(utils.has_glob(p) for p in regions)
        regions_to_use = None if (all_regions or regions_need_expansion) else regions
        if not regions_to_use:
            _log("query", "Fetching region list from AWS")
            regions_to_use = core.list_regions(
                no_cache=no_cache, no_cache_lookup=no_cache_lookup,
            )
            if not regions_to_use:
                raise typer.Exit(code=1)
            _log("info", f"Found {len(regions_to_use)} regions")
            if regions_need_expansion:
                for p in regions:
                    if utils.has_glob(p) and not utils.match_names(regions_to_use, p):
                        _log("warn", f"No regions match pattern '{p}'")
                regions_to_use = utils.expand_names(regions_to_use, regions)
                _log("complete", f"Expanded region patterns to {len(regions_to_use)} regions: {', '.join(regions_to_use)}")

        regions_list = sorted(regions_to_use)

        # Resolve instance types (with cache)
        ipts = instance_types or []
        if ipts:
            if any(utils.has_glob(p) for p in ipts):
                all_types: list[str] | None = None
                if not no_cache:
                    all_types = cache.load_cache("all_instance_types", ttl_seconds=cache.INSTANCE_TTL_SECONDS)
                    if all_types:
                        _log("info", f"Using cached instance type catalog ({len(all_types)} types)", {"types": all_types})
                if all_types is None or no_cache_lookup:
                    all_types = core.list_instance_types(regions=regions_list)
                    if not all_types:
                        raise typer.Exit(code=1)
                    if not no_cache:
                        cache.save_cache("all_instance_types", all_types, ttl_seconds=cache.INSTANCE_TTL_SECONDS)
                    _log("info", f"Found {len(all_types)} instance types")
                for p in ipts:
                    if utils.has_glob(p) and not utils.match_names(all_types, p):
                        _log("warn", f"No instance types match pattern '{p}'")
                ipts = utils.expand_names(all_types, ipts)
                _log("complete", f"Expanded instance type patterns to {len(ipts)} types: {', '.join(ipts)}")
            _log("query", f"Using {len(ipts)} specified instance types")
            if mapped_archs:
                _log("info", "Note: --arch is ignored when explicit --instance-type values are given")
        else:
            arch_desc = ",".join(mapped_archs) if mapped_archs else "any"
            _log(
                "query",
                f"Resolving instances (arch[{arch_desc}], vcpu[{min_vcpu}-{max_vcpu}], ram[{min_ram}-{max_ram}] GB, gpu[{min_gpu}-{max_gpu}], storage[{min_instance_storage}-{max_instance_storage}] GB)",
            )
            resolve_cache_key = f"resolve:vcpu{min_vcpu}-{max_vcpu}:ram{min_ram}-{max_ram}:gpu{min_gpu}-{max_gpu}:storage{min_instance_storage}-{max_instance_storage}"
            if mapped_archs:
                resolve_cache_key += f":arch{','.join(mapped_archs)}"
            if regions_list:
                resolve_cache_key += f":regions{','.join(regions_list)}"
            ipts = None
            if not no_cache:
                ipts = cache.load_cache(resolve_cache_key, ttl_seconds=cache.INSTANCE_TTL_SECONDS)
                if ipts:
                    _log("info", f"Using cached instance types ({len(ipts)} types)", {"types": ipts})
            if ipts is None or no_cache_lookup:
                ipts = core.resolve_instance_types(
                    instance_types=[],
                    min_vcpu=min_vcpu,
                    min_ram_gb=min_ram,
                    min_gpu=min_gpu,
                    max_vcpu=max_vcpu,
                    max_ram_gb=max_ram,
                    max_gpu=max_gpu,
                    min_instance_storage_gb=min_instance_storage,
                    max_instance_storage_gb=max_instance_storage,
                    architectures=mapped_archs,
                    regions=regions_list,
                )
                if not no_cache:
                    cache.save_cache(resolve_cache_key, ipts, ttl_seconds=cache.INSTANCE_TTL_SECONDS)
            _log("complete", f"Found {len(ipts)} types: {', '.join(ipts)}", {"types": ipts})

        total_pairs = len(ipts) * len(regions_list)

        if not regions_need_expansion:
            _log("complete", f"Found {len(regions_list)} regions: {', '.join(regions_list)}", {"regions": regions_list})

        if total_pairs == 0:
            _log("warn", "No instance×region pairs to query (0 types × 0 regions)")
            _log("complete", "Nothing to fetch")
        else:
            _log("query", f"{len(ipts)} instances across {len(regions_list)} regions = {total_pairs} pairs")

        # Fetch spot prices (per-instance-type cache)
        raw: list[dict] = []
        if total_pairs > 0:
            done: dict[str, int] = {"count": 0}

            def _wrap_progress_callback(pair_name: str, total_records: int) -> None:
                done["count"] += 1
                progress.update(  # type: ignore[call-arg]
                    task,
                    advance=1,
                    done=done["count"],
                    records=total_records,
                    description=f"Fetching {pair_name}...",
                    refresh=True,
                )

            t0 = time.time()
            with Progress(
                SpinnerColumn(),
                TextColumn("[blue]Fetching spot prices...[/blue]"),
                BarColumn(),
                TextColumn(
                    "[dim]{task.fields[records]} records ({task.fields[done]}/{task.fields[total_pairs]} pairs)[/dim]"
                ),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "fetching", total=total_pairs, done=0, records=0, total_pairs=total_pairs
                )

                for itype in ipts:
                    spot_cache_key = cache.make_spot_cache_key(itype, regions_list)
                    cached = None
                    if not no_cache:
                        cached = cache.load_cache(spot_cache_key, ttl_seconds=cache.SPOT_TTL_SECONDS)
                    if cached and not no_cache_lookup:
                        raw.extend(cached)
                        for _ in regions_list:
                            done["count"] += 1
                        progress.update(
                            task,
                            advance=len(regions_list),
                            done=done["count"],
                            records=len(raw),
                            refresh=True,
                        )
                        _log("info", f"Using cached spot prices for {itype} ({len(cached):,} records)", {"records": len(cached)})
                        continue
                    batch = core.fetch_spot_prices(
                        [itype],
                        regions=regions_to_use,
                        days=_DAYS_FOR_WINDOW.get(max_days, 7),
                        progress_callback=_wrap_progress_callback,
                        product_description=product_description,
                        records_offset=len(raw),
                    )
                    if not no_cache:
                        cache.save_cache(spot_cache_key, batch, ttl_seconds=cache.SPOT_TTL_SECONDS)
                    raw.extend(batch)

            elapsed = time.time() - t0
            _log("complete", f"Fetched {len(raw):,} records in {elapsed:.1f}s", {"records": len(raw)})

            if not raw and _skip_count["count"] > 0:
                _log("error", f"All {_skip_count['count']} region queries failed — check logs for details")

        # Fetch placement scores (if enabled)
        placement_scores: dict[str, dict[str, dict[str, int]]] = {}
        if placement_score:
            _log("query", f"Fetching Spot Placement Scores for {len(ipts)} instance types across {len(regions_list)} regions (target_capacity={target_capacity})")
            placement_scores = core.get_spot_placement_scores(
                ipts,
                regions_list,
                target_capacity=target_capacity,
            )
            scored_regions = len(placement_scores)
            scored_types = sum(len(rt) for rt in placement_scores.values())
            _log("complete", f"Retrieved placement scores for {scored_regions} region(s), {scored_types} instance type(s)")

        # Compute metrics
        _log("query", f"Computing metrics for {len(raw):,} price points across time windows up to {max_days}")
        result = core.compute_metrics(raw, sort_by=sort_by, max_window=max_days)

        # Inject placement scores and filter (before aggregation for per-region output)
        if placement_scores:
            # Map placement scores back to the result DataFrame
            def _assign_score(row):
                region = row.get("region")
                itype = row.get("instance_type")
                region_az = row.get("region_az")
                # Guard against NaN values (empty DataFrame case in pandas 3.x)
                if pd.isna(region) or pd.isna(itype) or pd.isna(region_az):
                    return None
                region = str(region) or ""
                itype = str(itype) or ""
                region_az = str(region_az) or ""
                az_name = region_az.rsplit("/", 1)[-1] if "/" in region_az else region_az
                az_scores = placement_scores.get(region, {}).get(itype, {})
                return az_scores.get(az_name)

            result = result.copy()
            scores_series = result.apply(_assign_score, axis=1)
            result["placement_score"] = scores_series
            _log("info", f"Assigned placement scores to {result['placement_score'].notna().sum()} of {len(result)} rows")

            # Apply min placement score filter (before aggregation so the best AZ is from filtered set)
            if min_placement_score > 1:
                before = len(result)
                result = core.filter_by_placement_score(result, min_score=min_placement_score, scores=placement_scores)
                dropped = before - len(result)
                if dropped > 0:
                    _log("info", f"Filtered out {dropped} rows below min placement score {min_placement_score}")

        if not per_az:
            _log("info", "Aggregating results to region level (lowest price per AZ per region)")
            result = core.aggregate_by_region(result, sort_by=sort_by)

        _log("complete", f"Computed metrics for {len(result)} instance×region pairs (sorted by {sort_by})", {"rows": len(result)})

        # Render
        _log("info", f"Rendering {len(result)} rows")
        render_table(result, sort_by=sort_by, limit=limit, per_az=per_az, with_score=placement_score, max_window=max_days)

    except Exception as e:
        _log("error", str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
