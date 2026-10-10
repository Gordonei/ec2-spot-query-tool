"""Integration tests for the CLI — full workflow with core mocked."""

from __future__ import annotations

import datetime as dt
import json

from typer.testing import CliRunner

from ec2_spot_query.cli import app
from unittest.mock import MagicMock, patch

runner = CliRunner()


def _spot_records() -> list[dict]:
    """One default spot record for t3.micro in us-east-1."""
    return [
        {
            "Region": "us-east-1",
            "InstanceId": "us-east-1:us-east-1a",
            "AvailabilityZone": "us-east-1a",
            "InstanceType": "t3.micro",
            "SpotPrice": "0.012",
            "Timestamp": dt.datetime(2025, 10, 1, 0, 0, tzinfo=dt.timezone.utc),
        }
    ]


class TestFullWorkflow:
    """Test complete CLI workflow with core mocked."""

    def test_full_flow_with_specified_instances(self, mocker):
        """Full pipeline: resolve instances → fetch → compute → render."""
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        mocker.patch(
            "ec2_spot_query.core.resolve_instance_types", return_value=["t3.micro"]
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "us-east-1", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        assert "EC2 Spot Instance Rankings" in result.stdout
        mock_fetch.assert_called_once()


class TestAllRegions:
    """Test --all-regions flag."""

    def test_all_regions_flag_invokes_list_regions(self, mocker):
        """--all-regions calls core.list_regions for discovery."""
        mock_list = mocker.patch(
            "ec2_spot_query.core.list_regions", return_value=["us-east-1", "eu-west-1"]
        )
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--all-regions", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        mock_list.assert_called_once()


class TestCacheHitFlow:
    """Test cache hit path — skip fetch calls."""

    def test_cache_hit_skips_fetch(self, mocker, tmp_path):
        """Cached spot data skips the fetch call."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cached_data = {
            "spot:t3.micro:us-east-1": {
                "data": [
                    {
                        "Region": "us-east-1",
                        "InstanceId": "us-east-1:us-east-1a",
                        "AvailabilityZone": "us-east-1a",
                        "InstanceType": "t3.micro",
                        "SpotPrice": "0.012",
                        "Timestamp": "2025-10-01T00:00:00+00:00",
                    }
                ],
                "expires_at": 9999999999,
            }
        }
        cache_file.write_text(json.dumps(cached_data), encoding="utf-8")
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app,
                ["--instance-types", "t3.micro", "--regions", "us-east-1"],
            )

        assert result.exit_code == 0, result.stdout
        mock_fetch.assert_not_called()


class TestCacheMissFlow:
    """Test cache miss path — fetch and save."""

    def test_cache_miss_fetches_and_saves(self, mocker, tmp_path):
        """Cache miss triggers the fetch call and saves the result."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("{}", encoding="utf-8")
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app,
                ["--instance-types", "t3.micro", "--regions", "us-east-1"],
            )

        assert result.exit_code == 0, result.stdout
        mock_fetch.assert_called_once()
        saved = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "spot:t3.micro:us-east-1" in saved


class TestArchFilter:
    """Test --arch / --architecture option wiring."""

    def test_arch_alias_passed_to_resolve(self, mocker):
        """--arch graviton maps to arm64 and reaches resolve_instance_types."""
        mock_resolve = mocker.patch(
            "ec2_spot_query.core.resolve_instance_types", return_value=["c6g.large"]
        )
        mocker.patch("ec2_spot_query.core.fetch_spot_prices", return_value=[])
        result = runner.invoke(
            app,
            ["--arch", "graviton", "--regions", "us-east-1", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        assert mock_resolve.call_args.kwargs["architectures"] == ["arm64"]

    def test_arch_repeated_flags_build_list(self, mocker):
        """Multiple --arch occurrences produce a multi-value filter list."""
        mock_resolve = mocker.patch(
            "ec2_spot_query.core.resolve_instance_types", return_value=["t3.micro"]
        )
        mocker.patch("ec2_spot_query.core.fetch_spot_prices", return_value=[])
        result = runner.invoke(
            app,
            ["--arch", "x86", "--architecture", "arm", "--regions", "us-east-1", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        assert mock_resolve.call_args.kwargs["architectures"] == ["x86_64", "arm64"]

    def test_no_arch_passes_none(self, mocker):
        """Without --arch, resolve_instance_types receives architectures=None."""
        mock_resolve = mocker.patch(
            "ec2_spot_query.core.resolve_instance_types", return_value=["t3.micro"]
        )
        mocker.patch("ec2_spot_query.core.fetch_spot_prices", return_value=[])
        result = runner.invoke(
            app, ["--regions", "us-east-1", "--no-cache"]
        )
        assert result.exit_code == 0, result.stdout
        assert mock_resolve.call_args.kwargs["architectures"] is None

    def test_arch_ignored_with_explicit_instance_types(self, mocker):
        """Explicit --instance-types short-circuits resolution; no API filter needed."""
        mock_resolve = mocker.patch(
            "ec2_spot_query.core.resolve_instance_types", return_value=["t3.micro"]
        )
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            [
                "--instance-types", "t3.micro", "--arch", "arm",
                "--regions", "us-east-1", "--no-cache",
            ],
        )
        assert result.exit_code == 0, result.stdout
        mock_resolve.assert_not_called()

    def test_unknown_arch_exits_nonzero(self, mocker):
        """Unknown architecture aliases fail fast with a clear error."""
        mock_resolve = mocker.patch(
            "ec2_spot_query.core.resolve_instance_types", return_value=["t3.micro"]
        )
        result = runner.invoke(
            app,
            ["--arch", "sparc", "--regions", "us-east-1", "--no-cache"],
        )
        assert result.exit_code == 1
        combined = result.stdout + (result.stderr if result.stderr_bytes else "")
        assert "Unknown architecture" in combined
        mock_resolve.assert_not_called()

    def test_arch_cache_key_is_arch_specific(self, mocker, tmp_path):
        """Resolve cache keys embed the mapped arch so filters never share entries."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "resolve:vcpu0-64:ram0.5-128.0:gpu0-8:storage0.0-256.0:archarm64": {
                "data": ["c6g.large"],
                "expires_at": 9999999999,
            }
        }), encoding="utf-8")
        mock_resolve = mocker.patch(
            "ec2_spot_query.core.resolve_instance_types", return_value=["c6g.large"]
        )
        mocker.patch("ec2_spot_query.core.fetch_spot_prices", return_value=[])

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app, ["--arch", "arm", "--regions", "us-east-1"]
            )
        assert result.exit_code == 0, result.stdout
        mock_resolve.assert_not_called()  # served from the arch-specific cache entry

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(app, ["--regions", "us-east-1"])
        assert result.exit_code == 0, result.stdout
        mock_resolve.assert_called_once()  # no-arch key differs, so it must resolve


class TestGlobExpansion:
    """Test glob pattern support in --regions and --instance-types."""

    def test_region_glob_expands(self, mocker):
        """-r 'eu-*' expands against the discovered region list."""
        mocker.patch(
            "ec2_spot_query.core.list_regions",
            return_value=["us-east-1", "eu-west-1", "eu-central-1", "ap-southeast-2"],
        )
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "eu-*", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        assert mock_fetch.call_args.kwargs["regions"] == ["eu-central-1", "eu-west-1"]

    def test_region_literal_fast_path_skips_list_regions(self, mocker):
        """Literal-only --regions never calls list_regions (fast path preserved)."""
        mock_list = mocker.patch("ec2_spot_query.core.list_regions")
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "us-east-1", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        mock_list.assert_not_called()
        assert mock_fetch.call_args.kwargs["regions"] == ["us-east-1"]

    def test_instance_glob_expands(self, mocker):
        """-i 'inf2.*' expands against the full instance type catalog."""
        mocker.patch(
            "ec2_spot_query.core.list_instance_types",
            return_value=["t3.micro", "inf2.xlarge", "inf2.2xlarge"],
        )
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "inf2.*", "--regions", "us-east-1", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        fetched = [c.args[0] for c in mock_fetch.call_args_list]
        assert fetched == [["inf2.2xlarge"], ["inf2.xlarge"]]

    def test_mixed_literals_and_globs(self, mocker):
        """-i 't3.micro' -i 'inf2.*' keeps literals and expands globs."""
        mocker.patch(
            "ec2_spot_query.core.list_instance_types",
            return_value=["t3.micro", "inf2.xlarge", "c7g.large"],
        )
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            [
                "--instance-types", "t3.micro", "--instance-types", "inf2.*",
                "--regions", "us-east-1", "--no-cache",
            ],
        )
        assert result.exit_code == 0, result.stdout
        fetched = [c.args[0] for c in mock_fetch.call_args_list]
        assert fetched == [["inf2.xlarge"], ["t3.micro"]]

    def test_region_glob_zero_match_warns_and_continues(self, mocker):
        """A region pattern matching nothing warns and exits cleanly without fetching."""
        mocker.patch(
            "ec2_spot_query.core.list_regions",
            return_value=["us-east-1", "eu-west-1"],
        )
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "zzz-*", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        mock_fetch.assert_not_called()
        combined = result.stdout + (result.stderr if result.stderr_bytes else "")
        assert "zzz-*" in combined

    def test_instance_glob_zero_match_warns_and_continues(self, mocker):
        """An instance pattern matching nothing warns and exits cleanly without fetching."""
        mocker.patch(
            "ec2_spot_query.core.list_instance_types",
            return_value=["t3.micro", "c7g.large"],
        )
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "zzz.*", "--regions", "us-east-1", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout
        mock_fetch.assert_not_called()
        combined = result.stdout + (result.stderr if result.stderr_bytes else "")
        assert "zzz.*" in combined

    def test_region_glob_uses_cached_region_list(self, mocker, tmp_path):
        """A cached region list serves glob expansion without calling list_regions."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "regions": {
                "data": ["us-east-1", "eu-west-1", "eu-central-1"],
                "expires_at": 9999999999,
            }
        }), encoding="utf-8")
        mock_list = mocker.patch("ec2_spot_query.core.list_regions")
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app,
                ["--instance-types", "t3.micro", "--regions", "eu-*"],
            )

        assert result.exit_code == 0, result.stdout
        mock_list.assert_not_called()
        assert mock_fetch.call_args.kwargs["regions"] == ["eu-central-1", "eu-west-1"]

    def test_instance_glob_uses_cached_catalog(self, mocker, tmp_path):
        """A cached instance type catalog serves expansion without list_instance_types."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "all_instance_types": {
                "data": ["t3.micro", "inf2.xlarge", "inf2.2xlarge"],
                "expires_at": 9999999999,
            }
        }), encoding="utf-8")
        mock_types = mocker.patch("ec2_spot_query.core.list_instance_types")
        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app,
                ["--instance-types", "inf2.*", "--regions", "us-east-1"],
            )

        assert result.exit_code == 0, result.stdout
        mock_types.assert_not_called()
        fetched = [c.args[0] for c in mock_fetch.call_args_list]
        assert fetched == [["inf2.2xlarge"], ["inf2.xlarge"]]

    def test_instance_glob_saves_catalog_to_cache(self, mocker, tmp_path):
        """A catalog miss saves the fetched type list under the cache key."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("{}", encoding="utf-8")
        mocker.patch(
            "ec2_spot_query.core.list_instance_types",
            return_value=["t3.micro", "inf2.xlarge"],
        )
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app,
                ["--instance-types", "inf2.*", "--regions", "us-east-1"],
            )

        assert result.exit_code == 0, result.stdout
        saved = json.loads(cache_file.read_text(encoding="utf-8"))
        assert saved["all_instance_types"]["data"] == ["t3.micro", "inf2.xlarge"]

    def test_help_mentions_glob_patterns(self):
        """--help documents glob support for --regions and --instance-types."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "glob" in result.stdout.lower()


class TestFlags:
    """Test --progress and --debug flags."""

    def test_progress_flag_in_help(self):
        """--progress flag appears in --help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--progress" in result.stdout

    def test_progress_flag_default_is_false(self, mocker):
        """Without --progress the CLI still runs (handler is a noop)."""
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app, ["--instance-types", "t3.micro", "--regions", "us-east-1", "--no-cache"]
        )
        assert result.exit_code == 0

    def test_debug_flag_in_help(self):
        """--debug flag appears in --help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_debug_configures_logging(self, mocker):
        """--debug enables detailed logging without breaking the run."""
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            ["--instance-types", "t3.micro", "--regions", "us-east-1", "--debug", "--no-cache"],
        )
        assert result.exit_code == 0, result.stdout


class TestPairLogging:
    """Test the region listing and the 'instances across regions = pairs' line."""

    def test_lists_regions_and_pairs_line(self, mocker):
        """CLI lists the regions and logs the 'X instances across Y regions = Z pairs' line."""
        mock_log = MagicMock()
        mocker.patch("ec2_spot_query.cli._make_progress_handler", return_value=mock_log)
        mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", return_value=_spot_records()
        )
        result = runner.invoke(
            app,
            [
                "--instance-types", "t3.micro",
                "--regions", "us-east-1", "--regions", "eu-west-1",
                "--no-cache", "--progress",
            ],
        )
        assert result.exit_code == 0, result.stdout
        messages = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        assert any(
            "2 regions" in m and "us-east-1" in m and "eu-west-1" in m
            for m in messages
        ), f"no region listing line in {messages}"
        assert any(
            "1 instances across 2 regions = 2 pairs" in m for m in messages
        ), f"no pairs line in {messages}"


class TestRunningTotal:
    """Progress-bar records field is a global running total that includes cached records."""

    def test_records_running_total_includes_cached(self, mocker, tmp_path):
        """Cached records seed the fetch offset and the bar's running total."""
        cache_file = tmp_path / ".cache" / "ec2-spot-cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # t3.micro is cached (3 records); t3.large is a cache miss (fetched, 2 records)
        cache_file.write_text(json.dumps({
            "spot:t3.micro:us-east-1": {
                "data": [
                    {
                        "Region": "us-east-1",
                        "InstanceId": "us-east-1:us-east-1a",
                        "AvailabilityZone": "us-east-1a",
                        "InstanceType": "t3.micro",
                        "SpotPrice": "0.012",
                        "Timestamp": "2025-10-01T00:00:00+00:00",
                    }
                    for _ in range(3)
                ],
                "expires_at": 9999999999,
            }
        }), encoding="utf-8")

        def fake_fetch(itypes, regions=None, days=30, product_description=None,
                       progress_callback=None, records_offset=0):
            recs = [{
                "Region": "us-east-1",
                "InstanceId": "us-east-1:us-east-1b",
                "AvailabilityZone": "us-east-1b",
                "InstanceType": itypes[0],
                "SpotPrice": "0.048",
                "Timestamp": dt.datetime(2025, 10, 1, 0, 0, tzinfo=dt.timezone.utc),
            } for _ in range(2)]
            if progress_callback is not None:
                progress_callback(
                    f"Fetching {itypes[0]} in us-east-1", records_offset + len(recs)
                )
            return recs

        mock_fetch = mocker.patch(
            "ec2_spot_query.core.fetch_spot_prices", side_effect=fake_fetch
        )

        mock_progress = MagicMock()
        mock_progress.__enter__.return_value = mock_progress
        mock_progress.add_task.return_value = MagicMock(name="task")
        mocker.patch("ec2_spot_query.cli.Progress", return_value=mock_progress)
        mocker.patch(
            "ec2_spot_query.cli._make_progress_handler", return_value=MagicMock()
        )

        with patch("ec2_spot_query.cache.get_cache_path", return_value=cache_file):
            result = runner.invoke(
                app,
                [
                    "--instance-types", "t3.micro", "--instance-types", "t3.large",
                    "--regions", "us-east-1",
                    "--progress",
                ],
            )

        assert result.exit_code == 0, result.stdout
        # The fetch must be offset by the 3 cached records.
        assert mock_fetch.call_count == 1
        assert mock_fetch.call_args.kwargs["records_offset"] == 3

        records_values = [
            c.kwargs["records"]
            for c in mock_progress.update.call_args_list
            if "records" in c.kwargs
        ]
        assert records_values, "expected the progress task to track a running total"
        assert records_values == sorted(records_values), "running total must be monotonic"
        assert records_values[-1] == 5, "final = 3 cached + 2 fetched"
