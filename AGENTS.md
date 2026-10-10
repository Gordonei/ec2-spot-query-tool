# AGENTS.md

## Commands

Dependency management is `uv` (Python >= 3.12; dev deps in `[dependency-groups]`):

```bash
uv sync                  # install deps (incl. pytest, pytest-mock)
uv run python -m pytest  # run tests — NOT `uv run pytest` (see below)
uv run python -m pytest tests/test_core.py::test_compute_metrics_empty_input  # single test
uv run ec2-spot-query --help
```

- `uv run pytest` (bare) FAILS collection with `No module named 'tests'`: test files declare `pytest_plugins = ("tests.test_utils",)` and `tests/` is only importable as a namespace package when the repo root is on `sys.path`, which `python -m` adds. Always use `uv run python -m pytest`.
- Real CLI runs need AWS credentials with EC2 read access; the test suite is fully offline (boto3 is mocked).

## Workflow: TDD

Make changes in this loop:

1. Structural change first — add function/method stubs with the intended signatures and docstrings.
2. Write tests covering the intended behavior: happy cases and the obvious unhappy paths (invalid input, empty results, error conditions).
3. Implement until the tests pass.

Run `uv run python -m pytest` (full suite, fast — ~1s) after each change.

## Style

- PEP 8 strictly.
- No lint/typecheck tooling is configured in this repo; PEP 8 compliance is the bar.
- Existing code uses `from __future__ import annotations`, type hints throughout, and `X | None` unions — match that.

## Git

- Commit messages: `<area>: <imperative summary>` — area is one of `cli`, `core`, `cache`, `tests`, `docs`, `refactor`, `feat`; append `(GH-n)` when referencing an issue.
- Keep commits semantically coherent: each commit contains only the changes relevant to that one logical change. Never bundle unrelated edits (code + unrelated tests, drive-by cleanups, etc.) into one commit.
- Stage only intended files — check `git status` / `git diff` before committing; never commit secrets.
- Authorship: commits are authored as the coding agent, not the human:
  - `user.name`: `<coding agent>/<model identifier string>`, e.g. `opencode/qwen3.8-27b-q4`
  - `user.email`: `<human email name part>+<model identifier string>@<human email domain>`, e.g. `gordon.e.inggs+qwen3.8-27b-q4@ieee.org`
  - The default git identity is the human (`Gordon Inggs <gordon.e.inggs@ieee.org>`); override per commit with `git -c user.name="..." -c user.email="..." commit ...` or repo-local config.

## Architecture

Single package, src layout. All logic in `src/ec2_spot_query/`:

- `cli.py` — Typer app (`app`), rich table rendering, progress/logging handlers. Entry point: `ec2-spot-query = ec2_spot_query.cli:app`.
- `core.py` — EC2 calls and metric computation. `resolve_instance_types` / `list_regions` use a module-level `_DEFAULT_CLIENT` (eu-west-1); spot-price fetches create per-region clients, batched 50 types/call, capped at 4 threads, adaptive retries. `compute_metrics` produces one row per (instance_type, region/az) with 6 time windows (1h–1m mean + vol); `aggregate_by_region` picks the cheapest AZ per (instance_type, region).
- `cache.py` — single JSON file at `~/.cache/ec2-spot-cache.json` (fallback `./.cache/`). TTLs: spot 1h, instance types 1y, regions 30d. Keys: `spot:{type}:{sorted regions}`, `resolve:...`, `regions`.

Tests mirror modules in `tests/`. `tests/test_utils.py` holds shared fixtures (notably `mock_ec2`, which patches both `core.boto3.client` and `core._DEFAULT_CLIENT`) and is loaded only via the `pytest_plugins` line in each test file — keep it named `test_utils.py` and keep those lines, or the fixtures vanish.

## Gotchas

- Stale `__pycache__`/`.pytest_cache` entries may reference deleted modules (wui, main, progress_bar) — ignore them.
- `compute_metrics`/`aggregate_by_region` must return the full column schema even for empty input (`_empty_result`); rendering in `cli.py` reads columns by name.
