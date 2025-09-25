import os
import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import plot_style as ps
import pandas as pd


def _resolve_prepared_data_path() -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "mut_precision")
    return os.path.abspath(os.path.join(base_dir, "precision_prepared.csv"))


def _extract_keep_ratio_and_method(dir_path: str) -> Tuple[int, str]:
    """Extract keep_ratio and method from directory name like sentiment_bert_relabel_50_keep_ratio_80_dve"""
    dir_name = os.path.basename(dir_path)

    # Extract keep_ratio
    ratio_match = re.search(r'keep_ratio_(\d+)', dir_name)
    if not ratio_match:
        raise ValueError(f"Could not extract keep_ratio from {dir_name}")
    keep_ratio = int(ratio_match.group(1))

    # Extract method (last part after final underscore)
    method = dir_name.split('_')[-1]

    return keep_ratio, method


def _load_relabel_overlap_data(file_path: str) -> pd.DataFrame:
    """Load relabel_overlap_042.csv file and return relevant columns, excluding epoch 0"""
    if not os.path.exists(file_path):
        return pd.DataFrame()  # Return empty df for missing files

    df = pd.read_csv(file_path)
    required_columns = {'epoch', 'precision'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"File {file_path} missing required columns: {missing}")

    # Filter out epoch 0
    df = df[df['epoch'] != 0]
    return df[['epoch', 'precision']]


def _collect_prepared_data(data_dir: str) -> pd.DataFrame:
    """Collect all relabel_overlap_042.csv data into a prepared format"""
    data_rows: List[Dict[str, object]] = []

    # Find all directories with keep_ratio pattern
    pattern = os.path.join(data_dir, "*keep_ratio_*")
    directories = glob.glob(pattern)

    for dir_path in directories:
        if not os.path.isdir(dir_path):
            continue

        try:
            keep_ratio, method = _extract_keep_ratio_and_method(dir_path)
            overlap_file = os.path.join(dir_path, "relabel_overlap_042.csv")
            data = _load_relabel_overlap_data(overlap_file)

            if data.empty:
                continue

            # Add keep_ratio and method to the data
            for _, row in data.iterrows():
                data_rows.append({
                    'keep_ratio': keep_ratio,
                    'method': method,
                    'epoch': row['epoch'],
                    'precision': row['precision']
                })

        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Skipping {dir_path}: {e}")
            continue

    if not data_rows:
        raise RuntimeError("No valid data found for preparation")

    return pd.DataFrame(data_rows)


def _load_prepared_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prepared data file not found: {path}")
    df = pd.read_csv(path)
    required_columns = {"keep_ratio", "method", "epoch", "precision"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def _plot_precision_by_keep_ratio(keep_ratio: int, methods_data: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """Plot precision vs epoch for all methods at a given keep_ratio"""
    if not methods_data:
        print(f"No data available for keep_ratio {keep_ratio}")
        return

    ps.apply_global_style()
    plt.figure(figsize=ps.default_figsize())

    markers = ps.MARKERS
    colors = ps.PALETTE_ICLR

    for idx, (method, data) in enumerate(sorted(methods_data.items())):
        if data.empty:
            continue

        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]

        # Handle missing data gracefully
        epochs = data['epoch'].values
        precision = data['precision'].values

        plt.plot(
            epochs,
            precision,
            marker=marker,
            color=color,
            linestyle="-",
            label=method.upper(),
            linewidth=2,
            markersize=6,
            zorder=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    # No title per requirement
    plt.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    ps.enable_axes_grid(plt.gca())
    plt.tight_layout()

    # Calculate check_ratio (100 - keep_ratio) for filename
    check_ratio = 100 - keep_ratio
    output_path = os.path.join(output_dir, f"precision_check_ratio_{check_ratio:02d}.png")
    ps.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")


def _plot_from_prepared_data(prepared_df: pd.DataFrame, output_dir: str) -> None:
    """Generate plots from prepared data"""
    os.makedirs(output_dir, exist_ok=True)

    # Group by keep_ratio
    for keep_ratio, ratio_group in prepared_df.groupby('keep_ratio'):
        methods_data = {}
        for method, method_group in ratio_group.groupby('method'):
            methods_data[method] = method_group[['epoch', 'precision']].copy()

        _plot_precision_by_keep_ratio(keep_ratio, methods_data, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot precision vs epoch for each keep_ratio")
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_dir = os.path.join(project_dir, "data", "mut_precision")
    default_plots_dir = os.path.join(project_dir, "plots")
    default_prepared_path = _resolve_prepared_data_path()

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare intermediate CSV data from raw files"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots from prepared data"
    )
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help="Directory containing mut_precision data subdirectories"
    )
    parser.add_argument(
        "--prepared-data-path",
        default=default_prepared_path,
        help="Path to write/read the prepared precision CSV"
    )
    parser.add_argument(
        "--output-dir",
        default=default_plots_dir,
        help="Directory to save plots"
    )

    args = parser.parse_args()

    # Default behavior: plot only if neither --prepare nor --plot specified
    if not args.prepare and not args.plot:
        args.plot = True

    if args.prepare:
        print("Preparing data...")
        prepared_df = _collect_prepared_data(args.data_dir)
        out_dir = os.path.dirname(args.prepared_data_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        prepared_df.to_csv(args.prepared_data_path, index=False)
        print(f"Prepared data saved to: {args.prepared_data_path}")

    if args.plot:
        print("Generating plots...")
        prepared_df = _load_prepared_data(args.prepared_data_path)
        _plot_from_prepared_data(prepared_df, args.output_dir)
        keep_ratios = sorted(prepared_df['keep_ratio'].unique())
        check_ratios = [100 - kr for kr in keep_ratios]
        print(f"Generated plots for check_ratios: {check_ratios}")


if __name__ == "__main__":
    main()