# ec2-spot-query-tool

Analyze and rank AWS EC2 Spot instance prices across multiple time horizons.

## Prerequisites

AWS credentials configured via `aws configure` with EC2 read permissions.

## Install & Run

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync
uv run ec2-spot-query            # query all regions, cheapest 1d_mean instances
uv run ec2-spot-query --help
```

Or with pip:

```bash
pip install .
ec2-spot-query
```

## Usage

### Find cheapest price for a specific instance across regions

```bash
ec2-spot-query -i m7g.xlarge
```

### Find cheapest instances with resource constraints (any region)

```bash
ec2-spot-query --min-vcpu 1 --max-vcpu 4 --min-ram 32 --max-ram 128
```

### Other common options

| Flag | Description | Example |
|------|-------------|---------|
| `-i, --instance-types` | Specific instance types | `-i m7g.xlarge,c7g.2xlarge` |
| `-r, --regions` | Limit to specific regions | `-r us-west-2,eu-west-1` |
| `--sort-by` | Sort metric | `--sort-by 1m_mean` |
| `--limit` | Rows to display | `--limit 20` |
| `--per-az` | Per-AZ breakdown | `--per-az` |
| `-p, --progress` | Progress spinner | `-p` |
| `--no-cache` | Skip cache entirely | `--no-cache` |

Sort options: `1h_mean`, `6h_mean`, `12h_mean`, `1d_mean`, `1w_mean`, `1m_mean`

## Project Structure

```
src/ec2_spot_query/
├── main.py      # CLI entry point (typer)
├── core.py      # Spot price fetching & metric computation
├── cli.py       # Rich table rendering
├── cache.py     # On-disk TTL cache
└── __init__.py  # Package metadata
tests/           # pytest test suite
```

## Development

```bash
uv sync          # install deps
uv run pytest    # run tests
```

## License

MIT — see LICENSE file.
