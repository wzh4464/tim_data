import argparse
import glob
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FALLBACK_INFLUENCE_FILES = {
    "icml": "infl_icml_all_epochs_relabel_000_pct_042.csv",
}

FALLBACK_DROP_COUNT = {
    "icml": 205,
}

IGNORE_DATASETS = {"tim_2"}


@lru_cache(maxsize=None)
def _load_influence_matrix(filename: str) -> pd.DataFrame:
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "mnsit_256")
    path = os.path.join(base_dir, filename)
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col != "sample_idx":
        df = df.rename(columns={first_col: "sample_idx"})
    return df.set_index("sample_idx")


@lru_cache(maxsize=1)
def _load_loo_matrix() -> pd.DataFrame:
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "mnsit_256")
    path = os.path.join(base_dir, "loo_valuation_matrix.csv")
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col != "sample_idx":
        df = df.rename(columns={first_col: "sample_idx"})
    return df.set_index("sample_idx")


def _compute_precision_from_influence(method: str) -> Tuple[pd.DataFrame, str]:
    if method not in FALLBACK_INFLUENCE_FILES:
        raise ValueError(f"No fallback influence matrix registered for method '{method}'")

    infl_df = _load_influence_matrix(FALLBACK_INFLUENCE_FILES[method])
    loo_df = _load_loo_matrix()

    common_index = infl_df.index.intersection(loo_df.index)
    if common_index.empty:
        raise ValueError(f"No overlapping sample indices found for method '{method}'")

    infl_df = infl_df.loc[common_index]
    loo_df = loo_df.loc[common_index]

    drop_k = min(FALLBACK_DROP_COUNT.get(method, len(common_index)), len(common_index))
    rows: List[Dict[str, float]] = []
    for epoch in range(loo_df.shape[1]):
        infl_col = f"influence_epoch_{epoch}"
        loo_col = f"epoch_{epoch}"
        if infl_col not in infl_df.columns or loo_col not in loo_df.columns:
            continue

        infl_top = set(infl_df[infl_col].astype(float).nlargest(drop_k).index)
        loo_top = set(loo_df[loo_col].astype(float).nlargest(drop_k).index)
        overlap = len(infl_top & loo_top)

        rows.append(
            {
                "epoch": epoch,
                "num_dropped": drop_k,
                "num_overlap_dropped": overlap,
                "precision": overlap / drop_k if drop_k else float("nan"),
            }
        )

    if not rows:
        raise ValueError(f"No overlapping epochs found for method '{method}'")

    df = pd.DataFrame(rows)
    note = f"Derived {method} precision via LOO overlap (top-{drop_k})."
    return df, note


def _prepare_precision(name: str, path: str) -> Tuple[Dict[str, pd.Series], Optional[str]]:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        if name in FALLBACK_INFLUENCE_FILES:
            df, note = _compute_precision_from_influence(name)
        else:
            raise ValueError(f"{path} is empty")
    else:
        note = None
    if "epoch" not in df.columns:
        raise ValueError(f"{path} does not include an 'epoch' column")

    df = df.sort_values("epoch").reset_index(drop=True)
    if "precision" not in df.columns:
        required = {"num_overlap_dropped", "num_dropped"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} missing columns to compute precision: {required}")
        df["precision"] = df["num_overlap_dropped"] / df["num_dropped"].replace(0, pd.NA)

    if len(df) < 2:
        raise ValueError(f"{path} needs at least two rows to form intervals")

    epochs = df["epoch"].to_numpy()
    x = df["epoch"].iloc[1:].reset_index(drop=True)
    labels = [f"[{epochs[i - 1]},{epochs[i]}]" for i in range(1, len(epochs))]
    precision = df["precision"].iloc[1:].reset_index(drop=True)

    return (
        {
            "x": x,
            "labels": labels,
            "precision": precision,
            "raw": df,
        },
        note,
    )


def _collect_prepared_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    csv_files = sorted(
        f for f in glob.glob(os.path.join(data_dir, "*.csv")) if os.path.isfile(f)
    )

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    prepared: Dict[str, Dict[str, pd.Series]] = {}
    errors: List[str] = []
    notes: List[str] = []

    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        if name in IGNORE_DATASETS:
            continue
        try:
            prepared[name], note = _prepare_precision(name, csv_path)
            if note:
                notes.append(note)
        except ValueError as exc:
            errors.append(f"Skipping {name}: {exc}")

    if not prepared:
        err_msg = "\n".join(errors) if errors else "No usable CSV files found."
        raise RuntimeError(err_msg)

    data_rows: List[Dict[str, object]] = []
    for method, content in prepared.items():
        for interval_idx, (epoch_end, label, precision_val) in enumerate(
            zip(content["x"], content["labels"], content["precision"])
        ):
            data_rows.append(
                {
                    "method": method,
                    "interval_index": interval_idx,
                    "epoch_end": epoch_end,
                    "interval_label": label,
                    "precision": precision_val,
                }
            )

    data_df = pd.DataFrame(data_rows)
    summary_df = _summarize_precision(data_df)
    return data_df, summary_df, errors, notes


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

    grouped = data_df.groupby("method")
    anchor_name = grouped["interval_index"].max().idxmax()
    anchor_df = data_df[data_df["method"] == anchor_name].sort_values("interval_index")
    x_vals = anchor_df["epoch_end"].tolist()
    x_labels = anchor_df["interval_label"].tolist()

    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]

    for idx, (method, group) in enumerate(sorted(grouped, key=lambda item: item[0])):
        marker = markers[idx % len(markers)]
        ordered = group.sort_values("interval_index")
        plt.plot(
            ordered["epoch_end"],
            ordered["precision"],
            f"{marker}-",
            label=method.upper(),
            linewidth=2,
            markersize=6,
        )

    plt.xlabel("Epoch Intervals")
    plt.ylabel("Precision")
    plt.title("Precision Comparison Across Methods")
    plt.xticks(x_vals, x_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare precision data and plot comparisons.")
    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    default_plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    default_prepared_path = os.path.join(default_plots_dir, "precision_prepared.csv")
    default_plot_path = os.path.join(default_plots_dir, "precision.png")

    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help="Directory containing raw precision CSV files.",
    )
    parser.add_argument(
        "--prepared-data-path",
        default=default_prepared_path,
        help="Path to write the prepared precision CSV.",
    )
    parser.add_argument(
        "--plot-output-path",
        default=default_plot_path,
        help="Path to write the precision plot PNG.",
    )
    parser.add_argument(
        "--no-prepare-data",
        dest="prepare_data",
        action="store_false",
        help="Skip preparing the precision data CSV.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip generating the precision plot.",
    )
    parser.set_defaults(prepare_data=True, plot=True)

    args = parser.parse_args()

    if not (args.prepare_data or args.plot):
        raise ValueError("Nothing to do. Enable data preparation and/or plotting.")

    data_df: Optional[pd.DataFrame] = None
    summary_df: Optional[pd.DataFrame] = None
    errors: List[str] = []
    notes: List[str] = []

    if args.prepare_data:
        data_df, summary_df, errors, notes = _collect_prepared_data(args.data_dir)
        prepared_dir = os.path.dirname(args.prepared_data_path)
        if prepared_dir:
            os.makedirs(prepared_dir, exist_ok=True)
        data_df.to_csv(args.prepared_data_path, index=False)

    if args.plot:
        if data_df is None:
            data_df = _load_prepared_data(args.prepared_data_path)
            summary_df = _summarize_precision(data_df)
        _plot_precision(data_df, args.plot_output_path)
        if summary_df is None:
            summary_df = _summarize_precision(data_df)
        print("Precision summary stats:")
        print(summary_df.to_string(index=False))

    if errors:
        print("\nWarnings:")
        for message in errors:
            print(f"- {message}")
    if notes:
        print("\nNotes:")
        for message in notes:
            print(f"- {message}")


if __name__ == "__main__":
    main()
