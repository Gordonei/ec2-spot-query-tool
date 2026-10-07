"""CLI entry point, rendering, and progress handling."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable

import boto3
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

from ec2_spot_query import cache, core


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

def render_table(result_df: pd.DataFrame, sort_by: str = "1d_mean", limit: int = -1, per_az: bool = False) -> None:
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
    table.add_column("1h Avg")
    table.add_column("1h Vol")
    table.add_column("6h Avg")
    table.add_column("6h Vol")
    table.add_column("12h Avg")
    table.add_column("12h Vol")
    table.add_column("1d Avg")
    table.add_column("1d Vol")
    table.add_column("1w Avg")
    table.add_column("1w Vol")
    table.add_column("1m Avg")
    table.add_column("1m Vol")

    for _, row in result_df.iterrows():
        table.add_row(
            str(row["region_az"]),
            str(row["instance_type"]),
            f"${row['current_price']:.3f}",
            f"{row['1h_mean']:.3f}",
            f"{row['1h_vol']:.4f}",
            f"{row['6h_mean']:.3f}",
            f"{row['6h_vol']:.4f}",
            f"{row['12h_mean']:.3f}",
            f"{row['12h_vol']:.4f}",
            f"{row['1d_mean']:.3f}",
            f"{row['1d_vol']:.4f}",
            f"{row['1w_mean']:.3f}",
            f"{row['1w_vol']:.4f}",
            f"{row['1m_mean']:.3f}",
            f"{row['1m_vol']:.4f}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    instance_types: list[str] = typer.Option(None, "--instance-types", "-i", help="Instance types to query"),
    min_vcpu: int = typer.Option(0, "--min-vcpu", help="Minimum vCPUs"),
    min_ram: float = typer.Option(0.5, "--min-ram", help="Minimum RAM in GB"),
    min_gpu: int = typer.Option(0, "--min-gpu", help="Minimum GPU count"),
    max_vcpu: int = typer.Option(64, "--max-vcpu", help="Maximum vCPUs"),
    max_ram: float = typer.Option(128, "--max-ram", help="Maximum RAM in GB"),
    max_gpu: int = typer.Option(8, "--max-gpu", help="Maximum GPU count"),
    min_instance_storage: float = typer.Option(0, "--min-instance-storage", help="Minimum instance storage in GB"),
    max_instance_storage: float = typer.Option(256, "--max-instance-storage", help="Maximum instance storage in GB"),
    regions: list[str] = typer.Option(None, "--regions", "-r", help="Regions to query"),
    all_regions: bool = typer.Option(False, "--all-regions", help="Query all regions"),
    sort_by: str = typer.Option("1d_mean", "--sort-by", help="Sort metric (1h_mean, 6h_mean, 12h_mean, 1d_mean, 1w_mean, 1m_mean)"),
    limit: int = typer.Option(10, "--limit", help="Number of rows to display (-1 for all)"),
    per_az: bool = typer.Option(False, "--per-az", help="Show per-AZ breakdown instead of per-region"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache (skip load and save)"),
    no_cache_lookup: bool = typer.Option(False, "--no-cache-lookup", help="Fetch fresh data but still write cache"),
    progress: bool = typer.Option(False, "--progress", "-p", help="Show progress with rich spinner"),
    product_description: str = typer.Option("Linux/UNIX", "--product-description", help="Product description filter for spot price queries (e.g. Linux/UNIX, Windows, SUSE Linux)"),
    debug: bool = typer.Option(False, "--debug", help="Show detailed EC2 API request/response logs"),
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

    try:
        _log("init", "EC2 Spot Query")
        _log("info", "Validating AWS credentials...")

        # Resolve instance types (with cache)
        ipts = instance_types or []
        if ipts:
            _log("query", f"Using {len(ipts)} specified instance types")
        else:
            _log(
                "query",
                f"Resolving instances (vcpu[{min_vcpu}-{max_vcpu}], ram[{min_ram}-{max_ram}] GB, gpu[{min_gpu}-{max_gpu}], storage[{min_instance_storage}-{max_instance_storage}] GB)",
            )
            resolve_cache_key = f"resolve:vcpu{min_vcpu}-{max_vcpu}:ram{min_ram}-{max_ram}:gpu{min_gpu}-{max_gpu}:storage{min_instance_storage}-{max_instance_storage}"
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
                    region="us-east-1",
                    min_instance_storage_gb=min_instance_storage,
                    max_instance_storage_gb=max_instance_storage,
                )
                if not no_cache:
                    cache.save_cache(resolve_cache_key, ipts, ttl_seconds=cache.INSTANCE_TTL_SECONDS)
            _log("complete", f"Found {len(ipts)} types: {', '.join(ipts)}", {"types": ipts})

        # Resolve regions (discover from AWS if --all-regions)
        regions_to_use = None if all_regions else regions
        if not regions_to_use:
            _log("query", "Fetching region list from AWS")
            ec2_client = boto3.client("ec2", region_name="us-east-1", config=core._EC2_CONFIG)
            resp = ec2_client.describe_regions()
            regions_to_use = [r["RegionName"] for r in resp.get("Regions", [])]
            if not regions_to_use:
                raise typer.Exit(code=1)
            _log("info", f"Found {len(regions_to_use)} regions")

        regions_list = sorted(regions_to_use)
        regions_str = ",".join(regions_list)
        total_pairs = len(ipts) * len(regions_list)

        if total_pairs == 0:
            _log("warn", "No instance×region pairs to query (0 types × 0 regions)")
            _log("complete", "Nothing to fetch")
        else:
            _log("info", f"Querying {total_pairs} instance×region pairs ({len(ipts)} types × {len(regions_list)} regions)")

        # Fetch spot prices (with cache)
        spot_cache_key = cache.make_spot_cache_key(ipts, regions_list)
        raw = None
        if not no_cache:
            raw = cache.load_cache(spot_cache_key, ttl_seconds=cache.SPOT_TTL_SECONDS)
            if raw:
                _log("info", f"Using cached spot prices ({len(raw):,} records)", {"records": len(raw)})

        if raw is None or no_cache_lookup:
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
                    raw = core.fetch_spot_prices(
                        ipts,
                        regions=regions_to_use,
                        days=30,
                        progress_callback=_wrap_progress_callback,
                        product_description=product_description,
                    )
                    elapsed = time.time() - t0

                    if not no_cache:
                        cache.save_cache(spot_cache_key, raw, ttl_seconds=cache.SPOT_TTL_SECONDS)
                    _log("complete", f"Fetched {len(raw):,} records in {elapsed:.1f}s", {"records": len(raw)})
        else:
            _log("info", f"Using cached spot prices ({len(raw):,} records)", {"records": len(raw)})

        # Compute metrics
        _log("query", f"Computing metrics for {len(raw):,} price points across 6 time windows")
        result = core.compute_metrics(raw, sort_by=sort_by)
        if not per_az:
            _log("info", "Aggregating results to region level (lowest price per AZ per region)")
            result = core.aggregate_by_region(result, sort_by=sort_by)
        _log("complete", f"Computed metrics for {len(result)} instance×region pairs (sorted by {sort_by})", {"rows": len(result)})

        # Render
        _log("info", f"Rendering {len(result)} rows")
        render_table(result, sort_by=sort_by, limit=limit, per_az=per_az)

    except Exception as e:
        _log("error", str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
