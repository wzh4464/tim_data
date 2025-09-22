"""Plot training loss and test accuracy curves from converge CSV files."""

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


DEFAULT_EXCLUDED_EPOCHS = (-1, 9)


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

        required = {"epoch", "train_loss"}
        if missing := required - set(df.columns):
            raise ValueError(f"{csv_path} missing required columns: {missing}")

        if "test_accuracy" in df.columns:
            accuracy_col = "test_accuracy"
        elif "test_accuracy" in df.columns:
            accuracy_col = "test_accuracy"
        else:
            raise ValueError(
                f"{csv_path} needs a 'val_accuracy' or 'test_accuracy' column"
            )

        df["epoch"] = df["epoch"].astype(int)
        df = df[~df["epoch"].isin(skip_epochs)]
        df = df.sort_values("epoch")
        if df.empty:
            continue

        data[method] = {
            "epoch": df["epoch"].tolist(),
            "test_accuracy": df[accuracy_col].tolist(),
            "train_loss": df["train_loss"].tolist(),
        }

    if not data:
        raise ValueError(f"No usable CSV files found in {data_dir}")

    return data


def _plot_curves(
    curves: Dict[str, Dict[str, Sequence[float]]],
    accuracy_path: Path,
    loss_path: Path,
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
            shifted_pairs = [(se, a) for se, a in shifted_pairs if se <= end_tick and se >= 1]
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
    ax_acc.set_ylim(0.5, 0.9)
    ps.enable_axes_grid(ax_acc)
    # x-axis ticks every epoch (step=1)
    if epochs_all_shifted:
        start_tick = 1
        end_tick = epochs_all_shifted[-1] - 1 if len(epochs_all_shifted) > 0 else 1
        if end_tick < start_tick:
            end_tick = start_tick
        ax_acc.set_xlim(start_tick - 0.5, end_tick + 0.5)
        ax_acc.xaxis.set_major_locator(MultipleLocator(1))
        ax_acc.set_xticks(list(range(start_tick, end_tick + 1)))
        ax_acc.tick_params(axis="x", rotation=0)
    ax_acc.legend(loc="lower right")

    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    fig_acc.tight_layout()
    ps.savefig(str(accuracy_path))
    plt.close(fig_acc)

    # Training loss plot
    fig_loss, ax_loss = plt.subplots(figsize=ps.default_figsize())
    for idx, method in enumerate(methods):
        curve = curves[method]
        marker = markers[idx % len(markers)]
        label = method.upper()
        color = iclr_colors[idx % len(iclr_colors)]
        ls = "--" if method.lower() == "origin" else "-"
        if method.lower() == "origin":
            color = ps.PALETTE_PRIMARY["grey"]
        ax_loss.plot(
            curve["epoch"],
            curve["train_loss"],
            marker=marker,
            color=color,
            linestyle=ls,
            label=label,
            zorder=2,
        )

    # no title per requirement
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ps.enable_axes_grid(ax_loss)
    # x-axis ticks every epoch (step=1)
    if epochs_all:
        min_e, max_e = epochs_all[0], epochs_all[-1]
        ax_loss.set_xlim(min_e - 0.5, max_e + 0.5)
        ax_loss.xaxis.set_major_locator(MultipleLocator(1))
        ax_loss.set_xticks(list(range(min_e, max_e + 1)))
        ax_loss.tick_params(axis="x", rotation=0)
    ax_loss.legend(loc=0)

    loss_path.parent.mkdir(parents=True, exist_ok=True)
    fig_loss.tight_layout()
    ps.savefig(str(loss_path))
    plt.close(fig_loss)

    # moved tick setup earlier before saving figures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    default_data_dir = project_root / "data" / "converge"
    default_accuracy_path = project_root / "plots" / "test_accuracy.png"
    default_loss_path = project_root / "plots" / "train_loss.png"

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing converge CSV files.",
    )
    parser.add_argument(
        "--accuracy-output",
        type=Path,
        default=default_accuracy_path,
        help="Path to write the test accuracy plot.",
    )
    parser.add_argument(
        "--loss-output",
        type=Path,
        default=default_loss_path,
        help="Path to write the training loss plot.",
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
    curves = _load_curves(args.data_dir, args.exclude_epochs)
    _plot_curves(curves, args.accuracy_output, args.loss_output)
    rel_acc = os.path.relpath(args.accuracy_output, start=os.getcwd())
    rel_loss = os.path.relpath(args.loss_output, start=os.getcwd())
    print(f"Wrote test accuracy plot to {rel_acc}")
    print(f"Wrote training loss plot to {rel_loss}")


if __name__ == "__main__":
    main()
