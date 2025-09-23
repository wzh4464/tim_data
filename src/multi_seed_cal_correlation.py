from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .cal_corelation import (
    _read_with_index,
    _extract_epoch_map,
    _safe_corr_pearson,
    _safe_corr_spearman,
    _safe_corr_kendall,
    _jaccard_top_k,
    compute_epoch_correlations,
    compute_total_correlation,
)


def extract_seed_from_dirname(dirname: str) -> int:
    """Extract seed number from directory name like 'loo_similar_256_seed_1'"""
    match = re.search(r'seed_(\d+)', dirname)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract seed number from directory: {dirname}")


def find_multi_seed_directories(base_path: str) -> Dict[int, Dict[str, str]]:
    """
    Find all multi-seed directories and categorize them by seed and method.

    Returns:
        Dict mapping seed number to dict of method names to directory paths
    """
    base_dir = os.path.join(base_path, "data", "mutli_seed")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Multi-seed directory not found: {base_dir}")

    seed_data = {}

    for dirname in os.listdir(base_dir):
        dirpath = os.path.join(base_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        try:
            seed = extract_seed_from_dirname(dirname)
        except ValueError:
            continue

        # Extract method from directory name
        # Patterns: loo_similar_256_seed_X_method or loo_similar_256_seed_X
        if dirname.endswith(f"_seed_{seed}"):
            method = "base"
        else:
            method_part = dirname.split(f"_seed_{seed}_")
            if len(method_part) == 2:
                method = method_part[1]
            else:
                continue

        if seed not in seed_data:
            seed_data[seed] = {}
        seed_data[seed][method] = dirpath

    return seed_data


def get_correlation_files_for_seed(seed_dir: str, method: str) -> Tuple[str, str]:
    """
    Get the appropriate influence and LOO files for a seed/method combination.

    Returns:
        Tuple of (influence_file_path, loo_file_path)
    """
    # Extract seed number to construct proper file patterns
    seed = extract_seed_from_dirname(os.path.basename(seed_dir))
    seed_suffix = f"{seed:03d}"  # Format as 3-digit number (001, 002, etc.)

    # Map method names to their influence file patterns
    infl_file_map = {
        "dve": f"infl_dve_all_epochs_relabel_000_pct_{seed_suffix}.csv",
        "tim": f"infl_tim_all_epochs_relabel_000_pct_{seed_suffix}.csv",
        "lava": f"infl_lava_all_epochs_relabel_000_pct_{seed_suffix}.csv",
        "icml": f"infl_icml_all_epochs_relabel_000_pct_{seed_suffix}.csv",
        "loo": f"infl_loo_all_epochs_relabel_000_pct_{seed_suffix}.csv",
    }

    # For other methods, find the influence file and corresponding LOO file
    if method in infl_file_map:
        infl_file = os.path.join(seed_dir, infl_file_map[method])
        if os.path.exists(infl_file):
            # Find LOO file - look in sibling directories
            parent_dir = os.path.dirname(seed_dir)
            seed_base = os.path.basename(seed_dir).split('_' + method)[0]
            loo_dir = os.path.join(parent_dir, seed_base + "_loo")
            loo_file = os.path.join(loo_dir, f"infl_loo_all_epochs_relabel_000_pct_{seed_suffix}.csv")

            if os.path.exists(loo_file):
                return infl_file, loo_file

    raise FileNotFoundError(f"Could not find correlation files for {seed_dir}, method {method}")


def compute_epoch_correlations_custom(
    infl_csv: str,
    loo_csv: str,
    invert_loo: bool = True,
    invert_infl: bool = False,
) -> pd.DataFrame:
    """
    Custom correlation computation that handles LOO vs LOO comparison properly.
    """
    infl_df = _read_with_index(infl_csv)
    loo_df = _read_with_index(loo_csv)

    # Handle different column prefixes
    infl_epochs = _extract_epoch_map(infl_df, prefix="influence_epoch_")
    loo_epochs = _extract_epoch_map(loo_df, prefix="influence_epoch_")  # LOO files also use influence_epoch_

    # If no epochs found with influence_epoch_, try epoch_ prefix
    if not loo_epochs:
        loo_epochs = _extract_epoch_map(loo_df, prefix="epoch_")

    common_epochs = sorted(set(infl_epochs.keys()) & set(loo_epochs.keys()))

    # Align rows by intersection of sample indices
    common_index = infl_df.index.intersection(loo_df.index)

    results: List[Dict[str, float]] = []
    for e in common_epochs:
        s1 = infl_df.loc[common_index, infl_epochs[e]].astype(float)
        if invert_infl:
            s1 = -s1
        s2 = loo_df.loc[common_index, loo_epochs[e]].astype(float)
        if invert_loo:
            s2 = -s2

        x = s1.to_numpy()
        y = s2.to_numpy()

        pear = _safe_corr_pearson(x, y)
        spear = _safe_corr_spearman(x, y)
        kend = _safe_corr_kendall(x, y)
        j30 = _jaccard_top_k(s1, s2, frac=0.3)

        results.append(
            {
                "epoch": e,
                "pearson": pear,
                "spearman": spear,
                "kendall": kend,
                "jaccard_top30pct": j30,
            }
        )

    out_df = pd.DataFrame(results).sort_values("epoch").reset_index(drop=True)
    return out_df


def compute_total_correlation_custom(
    infl_csv: str,
    loo_csv: str,
    invert_loo: bool = True,
    invert_infl: bool = False,
) -> Dict[str, float]:
    """
    Custom total correlation computation that handles LOO vs LOO comparison properly.
    """
    infl_df = _read_with_index(infl_csv)
    loo_df = _read_with_index(loo_csv)

    # Handle different column prefixes
    infl_epochs = _extract_epoch_map(infl_df, prefix="influence_epoch_")
    loo_epochs = _extract_epoch_map(loo_df, prefix="influence_epoch_")  # LOO files also use influence_epoch_

    # If no epochs found with influence_epoch_, try epoch_ prefix
    if not loo_epochs:
        loo_epochs = _extract_epoch_map(loo_df, prefix="epoch_")

    common_epochs = sorted(set(infl_epochs.keys()) & set(loo_epochs.keys()))
    if not common_epochs:
        raise ValueError("No common epochs found between influence data and LOO valuations.")

    common_index = infl_df.index.intersection(loo_df.index)
    if common_index.empty:
        raise ValueError("No overlapping sample indices between influence data and LOO valuations.")

    infl_cols = [infl_epochs[e] for e in common_epochs]
    loo_cols = [loo_epochs[e] for e in common_epochs]

    infl_total = infl_df.loc[common_index, infl_cols].astype(float).sum(axis=1)
    if invert_infl:
        infl_total = -infl_total

    loo_total = loo_df.loc[common_index, loo_cols].astype(float).sum(axis=1)
    if invert_loo:
        loo_total = -loo_total

    x = infl_total.to_numpy()
    y = loo_total.to_numpy()

    pear = _safe_corr_pearson(x, y)
    spear = _safe_corr_spearman(x, y)
    kend = _safe_corr_kendall(x, y)
    j30 = _jaccard_top_k(infl_total, loo_total, frac=0.3)

    return {
        "pearson": pear,
        "spearman": spear,
        "kendall": kend,
        "jaccard_top30pct": j30,
    }


def compute_multi_seed_correlations(base_path: str = ".") -> pd.DataFrame:
    """
    Compute correlations for all seeds and methods, returning a combined DataFrame.
    """
    seed_data = find_multi_seed_directories(base_path)

    all_results = []

    for seed, methods in seed_data.items():
        for method, method_dir in methods.items():
            if method == "base":
                continue  # Skip base directories without specific method

            try:
                infl_file, loo_file = get_correlation_files_for_seed(method_dir, method)

                # Determine inversion settings based on method
                invert_loo = True  # Always invert LOO as in original
                invert_infl = method in ["lava", "icml", "loo"]  # Invert for these methods

                # Compute epoch correlations
                epoch_corr = compute_epoch_correlations_custom(
                    infl_csv=infl_file,
                    loo_csv=loo_file,
                    invert_loo=invert_loo,
                    invert_infl=invert_infl
                )

                # Add seed and method columns
                epoch_corr.insert(0, "seed", seed)
                epoch_corr.insert(1, "method", method)

                # Compute total correlation
                total_corr = compute_total_correlation_custom(
                    infl_csv=infl_file,
                    loo_csv=loo_file,
                    invert_loo=invert_loo,
                    invert_infl=invert_infl
                )

                # Add total correlation as a special epoch (-1)
                total_row = {
                    "seed": seed,
                    "method": method,
                    "epoch": -1,  # Special marker for total correlation
                    **total_corr
                }

                all_results.append(epoch_corr)
                all_results.append(pd.DataFrame([total_row]))

            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Skipping seed {seed}, method {method}: {e}")
                continue

    if not all_results:
        raise ValueError("No valid correlation data found")

    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


def compute_statistics_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and variance of correlation metrics across seeds for each method and epoch.
    """
    # Exclude total correlations (epoch == -1) for now, handle separately
    epoch_df = df[df["epoch"] != -1].copy()
    total_df = df[df["epoch"] == -1].copy()

    stats_results = []

    # Group by method and epoch
    for (method, epoch), group in epoch_df.groupby(["method", "epoch"]):
        stats = {
            "method": method,
            "epoch": epoch,
            "seed_count": len(group),
        }

        # Compute mean and std for each correlation metric
        for metric in ["pearson", "spearman", "kendall", "jaccard_top30pct"]:
            values = group[metric].dropna()
            if len(values) > 0:
                stats[f"{metric}_mean"] = np.mean(values)
                stats[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            else:
                stats[f"{metric}_mean"] = np.nan
                stats[f"{metric}_std"] = np.nan

        stats_results.append(stats)

    # Handle total correlations separately
    for method, group in total_df.groupby("method"):
        stats = {
            "method": method,
            "epoch": -1,
            "seed_count": len(group),
        }

        for metric in ["pearson", "spearman", "kendall", "jaccard_top30pct"]:
            values = group[metric].dropna()
            if len(values) > 0:
                stats[f"{metric}_mean"] = np.mean(values)
                stats[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            else:
                stats[f"{metric}_mean"] = np.nan
                stats[f"{metric}_std"] = np.nan

        stats_results.append(stats)

    stats_df = pd.DataFrame(stats_results)
    return stats_df.sort_values(["method", "epoch"]).reset_index(drop=True)


def main() -> None:
    """Main function to compute multi-seed correlations and statistics."""
    try:
        # Compute correlations for all seeds
        print("Computing multi-seed correlations...")
        all_correlations = compute_multi_seed_correlations()

        # Compute statistics across seeds
        print("Computing statistics across seeds...")
        stats_df = compute_statistics_across_seeds(all_correlations)

        # Create output directory
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
        os.makedirs(out_dir, exist_ok=True)

        # Save results
        all_corr_path = os.path.join(out_dir, "multi_seed_all_correlations.csv")
        stats_path = os.path.join(out_dir, "multi_seed_correlation_statistics.csv")

        # Extract and save sum correlations (epoch == -1)
        sum_correlations = all_correlations[all_correlations["epoch"] == -1].copy()
        sum_correlations = sum_correlations.drop(columns=["epoch"])
        sum_corr_path = os.path.join(out_dir, "sum_multi_seed_correlations.csv")

        # Extract sum statistics
        sum_stats = stats_df[stats_df["epoch"] == -1].copy()
        sum_stats = sum_stats.drop(columns=["epoch"])
        sum_stats_path = os.path.join(out_dir, "sum_multi_seed_correlation_statistics.csv")

        all_correlations.to_csv(all_corr_path, index=False)
        stats_df.to_csv(stats_path, index=False)
        sum_correlations.to_csv(sum_corr_path, index=False)
        sum_stats.to_csv(sum_stats_path, index=False)

        # Display results
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", None)

        print("\nStatistics across seeds (mean ± std):")
        print(stats_df.to_string(index=False))

        print(f"\nSaved all correlations to: {all_corr_path}")
        print(f"Saved statistics to: {stats_path}")
        print(f"Saved sum correlations to: {sum_corr_path}")
        print(f"Saved sum statistics to: {sum_stats_path}")

        # Print summary by method for total correlations
        print("\nSummary of total correlations across seeds:")
        total_stats = stats_df[stats_df["epoch"] == -1].copy()
        for _, row in total_stats.iterrows():
            method = row["method"]
            count = row["seed_count"]
            print(f"\n{method.upper()} (n={count} seeds):")
            for metric in ["pearson", "spearman", "kendall", "jaccard_top30pct"]:
                mean_val = row[f"{metric}_mean"]
                std_val = row[f"{metric}_std"]
                print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()