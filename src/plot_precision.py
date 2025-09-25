import os
import argparse
from functools import lru_cache
from typing import Dict, List, Optional

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


@lru_cache(maxsize=None)
def _load_influence_matrix(filename: str, data_dir: str) -> pd.DataFrame:
    base_dir = os.path.abspath(data_dir)
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Influence file not found: {path}")
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col != "sample_idx":
        df = df.rename(columns={first_col: "sample_idx"})
    return df.set_index("sample_idx")


@lru_cache(maxsize=1)
def _load_loo_matrix(data_dir: str) -> pd.DataFrame:
    base_dir = os.path.abspath(data_dir)
    path = os.path.join(base_dir, "loo_valuation_matrix.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LOO valuation matrix not found: {path}")
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col != "sample_idx":
        df = df.rename(columns={first_col: "sample_idx"})
    return df.set_index("sample_idx")


def _compute_precision_from_influence(method: str, data_dir: str, drop_k: int) -> pd.DataFrame:
    method_to_file: Dict[str, str] = {
        "icml": "infl_icml_all_epochs_relabel_000_pct_042.csv",
        "lava": "infl_lava_all_epochs_relabel_000_pct_042.csv",
        "tim": "infl_tim_all_epochs_relabel_000_pct_042.csv",
        "dve": "infl_dve_all_epochs_relabel_000_pct_042.csv",
    }
    if method not in method_to_file:
        raise ValueError(f"Unknown method '{method}'")

    infl_df = _load_influence_matrix(method_to_file[method], data_dir)
    loo_df = _load_loo_matrix(data_dir)

    common_index = infl_df.index.intersection(loo_df.index)
    if common_index.empty:
        raise ValueError(f"No overlapping sample indices for method '{method}'")

    infl_df = infl_df.loc[common_index]
    loo_df = loo_df.loc[common_index]

    effective_k = min(int(drop_k), len(common_index))
    rows: List[Dict[str, float]] = []
    for epoch in range(loo_df.shape[1]):
        infl_col = f"influence_epoch_{epoch}"
        loo_col = f"epoch_{epoch}"
        if infl_col not in infl_df.columns or loo_col not in loo_df.columns:
            continue

        infl_top = set(infl_df[infl_col].astype(float).nlargest(effective_k).index)
        loo_top = set(loo_df[loo_col].astype(float).nlargest(effective_k).index)
        overlap = len(infl_top & loo_top)

        rows.append(
            {
                "epoch": epoch,
                "num_dropped": effective_k,
                "num_overlap_dropped": overlap,
                "precision": overlap / effective_k if effective_k else float("nan"),
            }
        )

    if not rows:
        raise ValueError(f"No overlapping epochs computed for method '{method}'")

    return pd.DataFrame(rows)


def _collect_prepared_from_matrices(data_dir: str, methods: List[str], drop_k: int) -> pd.DataFrame:
    data_rows: List[Dict[str, object]] = []
    for method in methods:
        df = _compute_precision_from_influence(method, data_dir, drop_k)
        epochs = df["epoch"].to_numpy()
        if len(epochs) < 2:
            continue
        x = df["epoch"].iloc[1:].reset_index(drop=True)
        labels = [f"[{epochs[i - 1]},{epochs[i]}]" for i in range(1, len(epochs))]
        precision = df["precision"].iloc[1:].reset_index(drop=True)

        for interval_idx, (epoch_end, label, precision_val) in enumerate(zip(x, labels, precision)):
            data_rows.append(
                {
                    "method": method,
                    "interval_index": interval_idx,
                    "epoch_end": epoch_end,
                    "interval_label": label,
                    "precision": precision_val,
                }
            )

    if not data_rows:
        raise RuntimeError("No prepared data could be derived from matrices.")

    return pd.DataFrame(data_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot precision and optionally prepare data from matrices.")
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_dir = os.path.join(project_dir, "data", "mnist_256")
    default_plots_dir = os.path.join(project_dir, "plots")
    default_prepared_path = _resolve_prepared_data_path()
    default_plot_path = os.path.join(default_plots_dir, "precision.png")

    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Enable preparing precision CSV from mnist_256 influence and LOO matrices (default off).",
    )
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help="Directory containing mnist_256 matrices (infl_*.csv and loo_valuation_matrix.csv).",
    )
    parser.add_argument(
        "--prepared-data-path",
        default=default_prepared_path,
        help="Path to write/read the prepared precision CSV.",
    )
    parser.add_argument(
        "--plot-output-path",
        default=default_plot_path,
        help="Path to write the precision plot PNG.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["icml", "lava", "tim", "dve"],
        help="Methods to include when preparing data from matrices.",
    )
    parser.add_argument(
        "--drop-k",
        type=int,
        default=205,
        help="Top-k to consider when deriving precision overlap from matrices.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip generating the precision plot.",
    )
    parser.set_defaults(plot=True)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.plot_output_path), exist_ok=True)

    if args.prepare_data:
        data_df = _collect_prepared_from_matrices(args.data_dir, args.methods, args.drop_k)
        out_dir = os.path.dirname(args.prepared_data_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        data_df.to_csv(args.prepared_data_path, index=False)

    data_df: Optional[pd.DataFrame]
    data_df = _load_prepared_data(args.prepared_data_path)

    if args.plot:
        _plot_precision(data_df, args.plot_output_path)

    summary_df = _summarize_precision(data_df)
    print("Precision summary stats:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
