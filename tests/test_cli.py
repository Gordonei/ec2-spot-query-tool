"""Tests for ec2_spot_query.cli — table rendering."""

from __future__ import annotations

import io
import re

import pandas as pd
import pytest
from rich.console import Console
from unittest.mock import patch

from ec2_spot_query import cli


def _sample_df() -> pd.DataFrame:
    """Return a DataFrame shaped like compute_metrics output."""
    return pd.DataFrame([{
        "region_az": "us-east-1/us-east-1a",
        "instance_type": "t3.micro",
        "current_price": 0.012,
        "1h_mean": 0.013, "1h_vol": 0.0010,
        "6h_mean": 0.012, "6h_vol": 0.0020,
        "12h_mean": 0.011, "12h_vol": 0.0010,
        "1d_mean": 0.012, "1d_vol": 0.0030,
        "1w_mean": 0.010, "1w_vol": 0.0020,
        "1m_mean": 0.015, "1m_vol": 0.0040,
    }, {
        "region_az": "us-east-1/us-east-1b",
        "instance_type": "t3.large",
        "current_price": 0.048,
        "1h_mean": 0.049, "1h_vol": 0.0020,
        "6h_mean": 0.047, "6h_vol": 0.0030,
        "12h_mean": 0.050, "12h_vol": 0.0010,
        "1d_mean": 0.048, "1d_vol": 0.0050,
        "1w_mean": 0.045, "1w_vol": 0.0030,
        "1m_mean": 0.050, "1m_vol": 0.0050,
    }])


def test_render_table_no_data_prints_message(capsys):
    """Empty DataFrame prints an informational message."""
    cli.render_table(pd.DataFrame())
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "no matching instance" in combined.lower()


def test_render_table_draws_columns(capsys):
    """Table header contains the expected column names."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=200)
    with patch("ec2_spot_query.cli.Console", return_value=console):
        cli.render_table(_sample_df())
    output = buf.getvalue().lower()
    for keyword in ["region", "instance", "price", "avg", "vol"]:
        assert keyword in output, f"Missing keyword '{keyword}' in table output"


def test_render_table_with_data_prints_content(capsys):
    """Non-empty DataFrame produces output with instance types and AZ data."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=200)
    with patch("ec2_spot_query.cli.Console", return_value=console):
        cli.render_table(_sample_df())
    output = buf.getvalue()
    assert "t3.micro" in output
    assert "t3.large" in output


def test_render_table_currency_format(capsys):
    """Prices are formatted as dollar amounts (e.g. $0.012)."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=200)
    with patch("ec2_spot_query.cli.Console", return_value=console):
        cli.render_table(_sample_df())
    output = buf.getvalue()

    # Find all $-prefixed values in output
    prices = re.findall(r"\$\d+\.\d+", output)
    assert len(prices) >= 2  # one row per sample dataframe row
    for p in prices:
        assert len(p.split(".")[1]) == 3  # 3 decimal places


def test_render_table_volatility_format(capsys):
    """Volatility values have fixed decimal precision (4 decimals)."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=200)
    with patch("ec2_spot_query.cli.Console", return_value=console):
        cli.render_table(_sample_df())
    output = buf.getvalue()

    # Volatility values have 4 decimal places (e.g. 0.0010)
    vols = re.findall(r"\b\d+\.\d{4}\b", output)
    assert len(vols) >= 2  # multiple volatility values expected


def test_render_table_with_limit(capsys):
    """Limit parameter renders only the specified number of rows."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=200)
    with patch("ec2_spot_query.cli.Console", return_value=console):
        cli.render_table(_sample_df(), limit=1)
    output = buf.getvalue()
    assert "t3.micro" in output
    assert "t3.large" not in output


def test_render_table_without_limit(capsys):
    """limit=-1 renders all rows (no limit applied)."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=200)
    with patch("ec2_spot_query.cli.Console", return_value=console):
        cli.render_table(_sample_df(), limit=-1)
    output = buf.getvalue()
    assert "t3.micro" in output
    assert "t3.large" in output
