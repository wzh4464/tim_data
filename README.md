# Tim Data

## Quickstart

1. Install uv

```bash
brew install uv
# 或
pipx install uv
```

2. Create venv

```bash
uv venv
```

3. Activate venv

```bash
source .venv/bin/activate
```

4. Sync dependencies

```bash
uv sync
```

5. Run scripts

```bash
uv run plot_precision
uv run plot_loss
uv run cal_corelation
```

## Usage

Generate precision comparison plot:

```bash
uv run plot_precision
```

Prepare precision data only (no plot):

```bash
uv run python src/plot_precision.py --no-plot
```

Plot using existing prepared CSV:

```bash
uv run python src/plot_precision.py --no-prepare-data
```
