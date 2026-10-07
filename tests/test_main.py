"""Tests for ec2_spot_query.cli — Typer CLI entry point."""

from __future__ import annotations

import typer
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from ec2_spot_query.cli import app


runner = CliRunner()


def test_help_flag():
    """--help exits 0 and lists expected options."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    for flag in ["--instance-types", "--sort-by", "--regions", "--debug"]:
        assert flag in out


def test_cli_invoke_no_errors(mock_ec2):
    """Invoking with instance types exits cleanly."""
    mock_ec2.describe_spot_price_history.return_value = {
        "SpotPriceHistory": [
            {
                "InstanceId": "us-east-1:us-east-1a",
                "AvailabilityZone": "us-east-1a",
                "InstanceType": "t3.micro",
                "SpotPrice": "0.012",
                "Timestamp": "2025-10-01T00:00:00Z",
            }
        ]
    }
    result = runner.invoke(app, ["--instance-types", "t3.micro", "--regions", "us-east-1"])
    assert result.exit_code == 0


def test_limit_flag_in_help():
    """--limit flag appears in --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--limit" in result.stdout


def test_per_az_flag_in_help():
    """--per-az flag appears in --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--per-az" in result.stdout


def test_no_cache_flag_in_help():
    """--no-cache flag appears in --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--no-cache" in result.stdout


def test_no_cache_lookup_flag_in_help():
    """--no-cache-lookup flag appears in --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--no-cache-lookup" in result.stdout


def test_max_vcpu_flag_in_help():
    """--max-vcpu flag appears in --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--max-vcpu" in result.stdout


def test_max_ram_flag_in_help():
    """--max-ram flag appears in --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--max-ram" in result.stdout


def test_max_gpu_flag_in_help():
    """--max-gpu flag appears in --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--max-gpu" in result.stdout


def test_instance_storage_flags_in_help():
    """--min-instance-storage and --max-instance-storage appear in --help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--min-instance-storage" in result.stdout
    assert "--max-instance-storage" in result.stdout
