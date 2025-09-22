"""Plot training loss and validation accuracy curves from converge CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

        if "val_accuracy" in df.columns:
            accuracy_col = "val_accuracy"
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
            "val_accuracy": df[accuracy_col].tolist(),
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
    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]

    # Validation accuracy plot
    fig_acc, ax_acc = plt.subplots(figsize=(6, 4.5))
    for idx, method in enumerate(methods):
        curve = curves[method]
        marker = markers[idx % len(markers)]
        label = method.upper()
        ax_acc.plot(curve["epoch"], curve["val_accuracy"], marker=marker, label=label)

    ax_acc.set_title("Validation Accuracy by Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.5, 1.0)
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend(loc="lower right")

    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    fig_acc.tight_layout()
    fig_acc.savefig(accuracy_path, dpi=200)
    plt.close(fig_acc)

    # Training loss plot
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4.5))
    for idx, method in enumerate(methods):
        curve = curves[method]
        marker = markers[idx % len(markers)]
        label = method.upper()
        ax_loss.plot(curve["epoch"], curve["train_loss"], marker=marker, label=label)

    ax_loss.set_title("Training Loss by Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend()

    loss_path.parent.mkdir(parents=True, exist_ok=True)
    fig_loss.tight_layout()
    fig_loss.savefig(loss_path, dpi=200)
    plt.close(fig_loss)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    default_data_dir = project_root / "data" / "converge"
    default_accuracy_path = project_root / "plots" / "val_accuracy.png"
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
        help="Path to write the validation accuracy plot.",
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
    print(f"Wrote validation accuracy plot to {rel_acc}")
    print(f"Wrote training loss plot to {rel_loss}")


if __name__ == "__main__":
    main()
