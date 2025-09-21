# Tim Data

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
