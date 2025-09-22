import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import plot_style as ps
import pandas as pd


def _resolve_prepared_data_path() -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "precision")
    return os.path.abspath(os.path.join(base_dir, "precision_prepared.csv"))


def _summarize_precision(data_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    required_columns = {"method", "interval_index", "precision"}
    if not required_columns.issubset(data_df.columns):
        missing = required_columns - set(data_df.columns)
        raise ValueError(f"Prepared data missing required columns: {missing}")

    for method, group in data_df.groupby("method"):
        ordered = group.sort_values("interval_index")
        precision_series = ordered["precision"].dropna()
        if precision_series.empty:
            summary_rows.append(
                {
                    "method": method,
                    "mean_precision": float("nan"),
                    "median_precision": float("nan"),
                    "min_precision": float("nan"),
                    "max_precision": float("nan"),
                    "last_interval_precision": float("nan"),
                }
            )
            continue

        summary_rows.append(
            {
                "method": method,
                "mean_precision": precision_series.mean(),
                "median_precision": precision_series.median(),
                "min_precision": precision_series.min(),
                "max_precision": precision_series.max(),
                "last_interval_precision": precision_series.iloc[-1],
            }
        )

    if not summary_rows:
        return pd.DataFrame(
            columns=
            [
                "method",
                "mean_precision",
                "median_precision",
                "min_precision",
                "max_precision",
                "last_interval_precision",
            ]
        )

    return pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)


def _load_prepared_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prepared data file not found: {path}")
    df = pd.read_csv(path)
    required_columns = {"method", "interval_index", "epoch_end", "interval_label", "precision"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"{path} missing required columns: {missing}")

    df["interval_index"] = df["interval_index"].astype(int)
    return df


def _plot_precision(data_df: pd.DataFrame, output_path: str) -> None:
    if data_df.empty:
        raise ValueError("Prepared data is empty; nothing to plot.")

    ps.apply_global_style()
    grouped = data_df.groupby("method")
    anchor_name = grouped["interval_index"].max().idxmax()
    anchor_df = data_df[data_df["method"] == anchor_name].sort_values("interval_index")
    x_vals = anchor_df["epoch_end"].tolist()
    x_labels = anchor_df["interval_label"].tolist()

    plt.figure(figsize=ps.default_figsize())
    markers = ps.MARKERS
    iclr_colors = ps.PALETTE_ICLR

    for idx, (method, group) in enumerate(sorted(grouped, key=lambda item: item[0])):
        marker = markers[idx % len(markers)]
        color = iclr_colors[idx % len(iclr_colors)]
        ordered = group.sort_values("interval_index")
        linestyle = "--" if method.lower() == "origin" else "-"
        color = ps.PALETTE_PRIMARY["grey"] if method.lower() == "origin" else color
        plt.plot(
            ordered["epoch_end"],
            ordered["precision"],
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=method.upper(),
            linewidth=2,
            markersize=6,
            zorder=2,
        )

    plt.xlabel("Epoch Intervals")
    plt.ylabel("Precision")
    # no title per requirement
    plt.xticks(x_vals, x_labels, rotation=0)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    ps.enable_axes_grid(plt.gca())
    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    ps.savefig(output_path)
    plt.close()


def main() -> None:
    prepared_path = _resolve_prepared_data_path()
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots"))
    output_plot = os.path.join(plots_dir, "precision.png")

    os.makedirs(plots_dir, exist_ok=True)

    data_df = _load_prepared_data(prepared_path)
    _plot_precision(data_df, output_plot)
    summary_df = _summarize_precision(data_df)
    print("Precision summary stats:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
