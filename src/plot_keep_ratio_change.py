"""Plot keep ratio change curves from CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import plot_style as ps
from matplotlib.ticker import MultipleLocator
import pandas as pd


DEFAULT_EXCLUDED_EPOCHS = (9,)


def _load_curves(
    data_dir: Path, excluded_epochs: Iterable[int]
) -> Dict[str, Dict[str, Sequence[float]]]:
    data: Dict[str, Dict[str, Sequence[float]]] = {}
    skip_epochs = {int(e) for e in excluded_epochs}

    for csv_path in sorted(data_dir.glob("*.csv")):
        method = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue

        required = {"epoch", "test_accuracy"}
        if missing := required - set(df.columns):
            raise ValueError(f"{csv_path} missing required columns: {missing}")

        df["epoch"] = df["epoch"].astype(int)
        df = df[~df["epoch"].isin(skip_epochs)]
        df = df.sort_values("epoch")
        if df.empty:
            continue

        data[method] = {
            "epoch": df["epoch"].tolist(),
            "test_accuracy": df["test_accuracy"].tolist(),
        }

    if not data:
        raise ValueError(f"No usable CSV files found in {data_dir}")

    return data


def _plot_curves(
    curves: Dict[str, Dict[str, Sequence[float]]],
    output_path: Path,
    check_ratio: str,
) -> None:
    methods = sorted(curves)
    markers = ps.MARKERS
    iclr_colors = ps.PALETTE_ICLR
    ps.apply_global_style()

    # Prepare full epoch range for ticks (accuracy uses shifted epochs e+1)
    epochs_all = sorted({e for m in methods for e in curves[m]["epoch"]})
    epochs_all_shifted = sorted({e + 1 for e in epochs_all})
    max_epoch_shifted = epochs_all_shifted[-1] if epochs_all_shifted else None

    # Test accuracy plot
    fig_acc, ax_acc = plt.subplots(figsize=ps.default_figsize())
    for idx, method in enumerate(methods):
        curve = curves[method]
        marker = markers[idx % len(markers)]
        label = method.upper()
        color = iclr_colors[idx % len(iclr_colors)]
        ls = "--" if method.lower() == "origin" else "-"
        if method.lower() == "origin":
            color = ps.PALETTE_PRIMARY["grey"]
        # accuracy: shift epochs by +1, then drop the last shifted point
        shifted_pairs = [(e + 1, a) for e, a in zip(curve["epoch"], curve["test_accuracy"]) ]
        if max_epoch_shifted is not None:
            end_tick = max_epoch_shifted - 1
            shifted_pairs = [(se, a) for se, a in shifted_pairs if se <= end_tick and se >= 0]
        if shifted_pairs:
            xs, ys = zip(*shifted_pairs)
        else:
            xs, ys = [], []
        ax_acc.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            linestyle=ls,
            label=label,
            zorder=2,
        )

    # no title per requirement
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.5, 0.95)
    ps.enable_axes_grid(ax_acc)
    # x-axis ticks every epoch (step=1)
    if epochs_all_shifted:
        start_tick = 0
        end_tick = epochs_all_shifted[-1] - 1 if len(epochs_all_shifted) > 0 else 0
        if end_tick < start_tick:
            end_tick = start_tick
        ax_acc.set_xlim(start_tick - 0.5, end_tick + 0.5)
        ax_acc.xaxis.set_major_locator(MultipleLocator(1))
        ax_acc.set_xticks(list(range(start_tick, end_tick + 1)))
        ax_acc.tick_params(axis="x", rotation=0)
    ax_acc.legend(loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_acc.tight_layout()
    ps.savefig(str(output_path))
    plt.close(fig_acc)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    default_data_dir = project_root / "data" / "keep_ratio_change"

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing keep_ratio_change CSV files.",
    )
    parser.add_argument(
        "--exclude-epochs",
        nargs="*",
        type=int,
        default=list(DEFAULT_EXCLUDED_EPOCHS),
        help="Epoch numbers to drop before plotting.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Group files by check ratio (the suffix after _)
    csv_files = list(args.data_dir.glob("*.csv"))
    check_ratios = set()
    for csv_file in csv_files:
        parts = csv_file.stem.split("_")
        if len(parts) >= 2:
            check_ratios.add(parts[-1])

    project_root = Path(__file__).resolve().parent.parent
    plots_dir = project_root / "plots"

    for check_ratio in sorted(check_ratios):
        # Create a temporary directory for this check ratio's files
        temp_dir = args.data_dir / f"temp_{check_ratio}"
        temp_dir.mkdir(exist_ok=True)

        # Copy files matching this check ratio to temp directory
        matching_files = [f for f in csv_files if f.stem.endswith(f"_{check_ratio}")]
        for src_file in matching_files:
            # Create new filename without the check ratio suffix
            method_name = "_".join(src_file.stem.split("_")[:-1])
            dst_file = temp_dir / f"{method_name}.csv"
            import shutil
            shutil.copy2(src_file, dst_file)

        if matching_files:
            curves = _load_curves(temp_dir, args.exclude_epochs)
            output_path = plots_dir / f"keep_ratio_change_{check_ratio}.png"
            _plot_curves(curves, output_path, check_ratio)
            rel_path = os.path.relpath(output_path, start=os.getcwd())
            print(f"Wrote keep ratio change plot (check ratio {check_ratio}) to {rel_path}")

        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()