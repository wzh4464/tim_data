from __future__ import annotations

import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


INFL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "mnsit_256",
        "infl_dve_all_epochs_relabel_000_pct_042.csv",
    )
)

TIM_INFL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "mnsit_256",
        "infl_tim_all_epochs_relabel_000_pct_042.csv",
    )
)

LOO_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "mnsit_256",
        "loo_valuation_matrix.csv",
    )
)

LAVA_INFL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "mnsit_256",
        "infl_lava_all_epochs_relabel_000_pct_042.csv",
    )
)

ICML_INFL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "mnsit_256",
        "infl_icml_all_epochs_relabel_000_pct_042.csv",
    )
)


def _read_with_index(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    if first_col in ("", None) or str(first_col).startswith("Unnamed"):
        df = df.rename(columns={first_col: "sample_idx"})
        first_col = "sample_idx"
    if first_col != "sample_idx":
        # Keep original name but use it as index to align rows
        df = df.rename(columns={first_col: "sample_idx"})
    df = df.set_index("sample_idx")
    return df


def _extract_epoch_map(df: pd.DataFrame, prefix: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for col in df.columns:
        if col.startswith(prefix):
            try:
                epoch_id = int(col[len(prefix) :])
            except ValueError:
                continue
            mapping[epoch_id] = col
    return mapping


def _safe_corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    r, _ = pearsonr(x, y)
    return float(r)


def _safe_corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def _safe_corr_kendall(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    r, _ = kendalltau(x, y)
    return float(r)


def _jaccard_top_k(x: pd.Series, y: pd.Series, frac: float = 0.3) -> float:
    if len(x) == 0:
        return float("nan")
    k = max(1, int(math.ceil(frac * len(x))))
    top_x = set(x.nlargest(k).index)
    top_y = set(y.nlargest(k).index)
    inter = len(top_x & top_y)
    union = len(top_x | top_y)
    if union == 0:
        return float("nan")
    return inter / union


def compute_epoch_correlations(
    infl_csv: str = INFL_PATH,
    loo_csv: str = LOO_PATH,
    invert_loo: bool = True,
    invert_infl: bool = False,
) -> pd.DataFrame:
    infl_df = _read_with_index(infl_csv)
    loo_df = _read_with_index(loo_csv)

    infl_epochs = _extract_epoch_map(infl_df, prefix="influence_epoch_")
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


def main() -> None:
    # DVE vs LOO (invert LOO)
    df_dve = compute_epoch_correlations(infl_csv=INFL_PATH, loo_csv=LOO_PATH, invert_loo=True)
    df_dve.insert(0, "source", "dve")

    # TIM vs LOO (invert LOO)
    df_tim = compute_epoch_correlations(infl_csv=TIM_INFL_PATH, loo_csv=LOO_PATH, invert_loo=True)
    df_tim.insert(0, "source", "tim")

    # LAVA vs LOO (invert LOO and LAVA)
    df_lava = compute_epoch_correlations(infl_csv=LAVA_INFL_PATH, loo_csv=LOO_PATH, invert_loo=True, invert_infl=True)
    df_lava.insert(0, "source", "lava")

    # ICML vs LOO (invert LOO and ICML)
    df_icml = compute_epoch_correlations(infl_csv=ICML_INFL_PATH, loo_csv=LOO_PATH, invert_loo=True, invert_infl=True)
    df_icml.insert(0, "source", "icml")

    # Combined
    df_all = pd.concat([df_dve, df_tim, df_lava, df_icml], ignore_index=True)

    # Print nicely
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df_all.to_string(index=False))

    # Save to CSV under project ./output directory
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(out_dir, exist_ok=True)

    out_dve = os.path.join(out_dir, "epoch_correlation_dve.csv")
    out_tim = os.path.join(out_dir, "epoch_correlation_tim.csv")
    out_lava = os.path.join(out_dir, "epoch_correlation_lava.csv")
    out_icml = os.path.join(out_dir, "epoch_correlation_icml.csv")
    out_all = os.path.join(out_dir, "epoch_correlation_all.csv")

    df_dve.drop(columns=["source"]).to_csv(out_dve, index=False)
    df_tim.drop(columns=["source"]).to_csv(out_tim, index=False)
    df_lava.drop(columns=["source"]).to_csv(out_lava, index=False)
    df_icml.drop(columns=["source"]).to_csv(out_icml, index=False)
    df_all.to_csv(out_all, index=False)
    print(f"Saved results to: {out_all}")


if __name__ == "__main__":
    main()
