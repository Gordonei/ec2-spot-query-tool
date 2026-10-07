"""CLI output rendering with rich tables."""

from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table


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
