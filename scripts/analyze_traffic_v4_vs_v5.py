
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATASET = "./dataset/traffic/traffic.csv"

V4_SELECTION = "./dataset/traffic/traffic_v4_selection.txt"
V5_SELECTION = "./dataset/traffic/traffic_v5_selection.txt"

TARGET = "OT"

# Same train split used by your experiments / V4-V5 selectors.
TRAIN_RATIO = 0.70

# For temporal analysis.
LAGS = [1, 2, 3, 6, 12, 24, 48, 96]

# To keep plots readable.
TOP_K = 30

# Output directory.
OUTPUT_DIR = "./dataset/traffic/selection_analysis"


# ============================================================
# Helpers
# ============================================================

def parse_selection_file(path):
    """
    Reads lines such as:

        0001: 669
        0002: 667
        ...

    and returns the original variable indices.
    """
    indices = []

    pattern = re.compile(r"^\s*\d+\s*:\s*(\d+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                indices.append(int(match.group(1)))

    if not indices:
        raise ValueError(
            f"No selected indices found in: {path}"
        )

    return np.asarray(indices, dtype=np.int64)


def normalize_minmax(x):
    x = np.asarray(x, dtype=float)

    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x)

    result = np.zeros_like(x)
    xmin = np.min(x[finite])
    xmax = np.max(x[finite])

    if xmax - xmin < 1e-12:
        return result

    result[finite] = (
        (x[finite] - xmin)
        / (xmax - xmin)
    )

    return result


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = (
        np.sqrt(
            np.sum(x * x)
            * np.sum(y * y)
        )
        + 1e-12
    )

    value = np.sum(x * y) / denom

    return float(value) if np.isfinite(value) else 0.0


def lagged_corr_scores(X, y, lags):
    """
    For every variable i:

        score_i =
            mean_k |corr(X_i[t-k], y[t])|

    This is forecasting-oriented and uses only training data.
    """
    n, p = X.shape

    scores_per_lag = []

    for lag in lags:

        if lag >= n - 2:
            continue

        x_lag = X[:-lag]
        y_future = y[lag:]

        y_centered = (
            y_future
            - np.mean(y_future)
        )

        x_centered = (
            x_lag
            - np.mean(x_lag, axis=0)
        )

        numerator = np.sum(
            x_centered
            * y_centered[:, None],
            axis=0
        )

        denominator = (
            np.sqrt(
                np.sum(
                    x_centered ** 2,
                    axis=0
                )
                * np.sum(
                    y_centered ** 2
                )
            )
            + 1e-12
        )

        corr = np.abs(
            numerator / denominator
        )

        scores_per_lag.append(corr)

    if not scores_per_lag:
        return np.zeros(p)

    return np.mean(
        np.stack(scores_per_lag),
        axis=0
    )


def spectral_signature(X, n_bands=8):
    """
    Normalized band-energy signature for each variable.
    """
    X = X - np.mean(
        X,
        axis=0,
        keepdims=True
    )

    fft = np.fft.rfft(
        X,
        axis=0
    )

    power = np.abs(fft) ** 2

    # Remove DC.
    power = power[1:]

    if len(power) == 0:
        return np.zeros(
            (X.shape[1], n_bands)
        )

    power /= (
        np.sum(
            power,
            axis=0,
            keepdims=True
        )
        + 1e-12
    )

    n_freq = power.shape[0]

    edges = np.linspace(
        0,
        n_freq,
        n_bands + 1,
        dtype=int
    )

    features = []

    for b in range(n_bands):

        start = edges[b]
        end = edges[b + 1]

        if end <= start:
            band = np.zeros(
                X.shape[1]
            )
        else:
            band = np.sum(
                power[start:end],
                axis=0
            )

        features.append(band)

    return np.stack(
        features,
        axis=1
    )


def spectral_target_similarity(X, y, n_bands=8):
    """
    Cosine similarity between each variable's
    frequency-energy profile and OT's profile.
    """
    X_spec = spectral_signature(
        X,
        n_bands=n_bands
    )

    y = y - np.mean(y)

    y_power = (
        np.abs(
            np.fft.rfft(y)
        ) ** 2
    )

    y_power = y_power[1:]

    if len(y_power) == 0:
        return np.zeros(X.shape[1])

    y_power /= (
        np.sum(y_power)
        + 1e-12
    )

    n_freq = len(y_power)

    edges = np.linspace(
        0,
        n_freq,
        n_bands + 1,
        dtype=int
    )

    y_bands = []

    for b in range(n_bands):

        start = edges[b]
        end = edges[b + 1]

        if end <= start:
            value = 0.0
        else:
            value = np.sum(
                y_power[start:end]
            )

        y_bands.append(value)

    y_bands = np.asarray(
        y_bands
    )

    numerator = np.sum(
        X_spec
        * y_bands[None, :],
        axis=1
    )

    denominator = (
        np.sqrt(
            np.sum(
                X_spec ** 2,
                axis=1
            )
        )
        * np.sqrt(
            np.sum(
                y_bands ** 2
            )
        )
        + 1e-12
    )

    return np.maximum(
        numerator / denominator,
        0.0
    )


def autocorrelation_at_lag(X, lag):
    if lag >= len(X) - 1:
        return np.zeros(X.shape[1])

    a = X[:-lag]
    b = X[lag:]

    a = a - np.mean(
        a,
        axis=0
    )

    b = b - np.mean(
        b,
        axis=0
    )

    numerator = np.sum(
        a * b,
        axis=0
    )

    denominator = (
        np.sqrt(
            np.sum(a * a, axis=0)
            * np.sum(b * b, axis=0)
        )
        + 1e-12
    )

    return numerator / denominator


def mean_pairwise_redundancy(X):
    """
    Mean absolute pairwise correlation.
    Lower is more diverse.

    This is computed by a correlation matrix.
    """
    Xz = (
        X - np.mean(
            X,
            axis=0,
            keepdims=True
        )
    ) / (
        np.std(
            X,
            axis=0,
            keepdims=True
        )
        + 1e-8
    )

    corr = (
        Xz.T @ Xz
    ) / max(
        len(Xz) - 1,
        1
    )

    corr = np.clip(
        corr,
        -1.0,
        1.0
    )

    abs_corr = np.abs(corr)

    np.fill_diagonal(
        abs_corr,
        np.nan
    )

    return np.nanmean(
        abs_corr
    )


def max_pairwise_redundancy_distribution(X):
    """
    For each variable, its maximum absolute correlation
    with any other selected variable.

    Lower generally means fewer near-duplicates.
    """
    Xz = (
        X - np.mean(
            X,
            axis=0,
            keepdims=True
        )
    ) / (
        np.std(
            X,
            axis=0,
            keepdims=True
        )
        + 1e-8
    )

    corr = (
        Xz.T @ Xz
    ) / max(
        len(Xz) - 1,
        1
    )

    corr = np.clip(
        corr,
        -1.0,
        1.0
    )

    corr = np.abs(corr)

    np.fill_diagonal(
        corr,
        0.0
    )

    return np.max(
        corr,
        axis=1
    )


# ============================================================
# Load
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("Traffic V4 vs V5 Selection Analysis")
    print("=" * 70)

    df = pd.read_csv(DATASET)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found."
        )

    feature_names = [
        c for c in df.columns
        if c not in {"date", TARGET}
    ]

    X_all = (
        df[feature_names]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .values
        .astype(float)
    )

    y_all = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce"
        )
        .values
        .astype(float)
    )

    train_end = int(
        len(df)
        * TRAIN_RATIO
    )

    X = X_all[:train_end]
    y = y_all[:train_end]

    # Fill missing values using training medians only.
    medians = np.nanmedian(
        X,
        axis=0
    )

    for j in range(X.shape[1]):
        mask = ~np.isfinite(X[:, j])
        if np.any(mask):
            X[mask, j] = medians[j]

    y_median = np.nanmedian(y)
    y[~np.isfinite(y)] = y_median

    v4 = parse_selection_file(
        V4_SELECTION
    )

    v5 = parse_selection_file(
        V5_SELECTION
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    assert len(v4) == len(
        np.unique(v4)
    )

    assert len(v5) == len(
        np.unique(v5)
    )

    assert np.all(
        (v4 >= 0)
        & (v4 < len(feature_names))
    )

    assert np.all(
        (v5 >= 0)
        & (v5 < len(feature_names))
    )

    print()
    print(
        f"Training rows : {len(y)}"
    )

    print(
        f"Variables     : {len(feature_names)}"
    )

    print(
        f"V4 size       : {len(v4)}"
    )

    print(
        f"V5 size       : {len(v5)}"
    )

    # --------------------------------------------------------
    # Set comparison
    # --------------------------------------------------------

    set_v4 = set(v4.tolist())
    set_v5 = set(v5.tolist())

    intersection = (
        set_v4
        & set_v5
    )

    union = (
        set_v4
        | set_v5
    )

    only_v4 = (
        set_v4
        - set_v5
    )

    only_v5 = (
        set_v5
        - set_v4
    )

    jaccard = (
        len(intersection)
        / len(union)
    )

    overlap_v4 = (
        len(intersection)
        / len(set_v4)
    )

    overlap_v5 = (
        len(intersection)
        / len(set_v5)
    )

    print()
    print("=" * 70)
    print("Subset overlap")
    print("=" * 70)

    print(
        f"Common variables : {len(intersection)}"
    )

    print(
        f"Only V4           : {len(only_v4)}"
    )

    print(
        f"Only V5           : {len(only_v5)}"
    )

    print(
        f"Jaccard           : {jaccard:.4f}"
    )

    print(
        f"V4 overlap        : {overlap_v4:.4f}"
    )

    print(
        f"V5 overlap        : {overlap_v5:.4f}"
    )

    # --------------------------------------------------------
    # Scores for ALL variables
    # --------------------------------------------------------

    print()
    print(
        "Computing forecasting-oriented scores..."
    )

    lag_score = lagged_corr_scores(
        X,
        y,
        LAGS
    )

    print(
        "Computing spectral scores..."
    )

    spectral_score = (
        spectral_target_similarity(
            X,
            y
        )
    )

    # --------------------------------------------------------
    # Build per-variable score table
    # --------------------------------------------------------

    stats = pd.DataFrame({
        "index": np.arange(
            len(feature_names)
        ),
        "name": feature_names,
        "lag_score": lag_score,
        "spectral_score": spectral_score,
    })

    stats["lag_norm"] = (
        normalize_minmax(
            stats["lag_score"]
        )
    )

    stats["spectral_norm"] = (
        normalize_minmax(
            stats["spectral_score"]
        )
    )

    stats["combined_score"] = (
        0.60
        * stats["lag_norm"]
        + 0.40
        * stats["spectral_norm"]
    )

    stats["set"] = "Not selected"

    stats.loc[
        stats["index"].isin(set_v4),
        "set"
    ] = "V4"

    stats.loc[
        stats["index"].isin(set_v5),
        "set"
    ] = "V5"

    stats.loc[
        stats["index"].isin(
            set_v4 & set_v5
        ),
        "set"
    ] = "Both"

    # --------------------------------------------------------
    # Per-set summary
    # --------------------------------------------------------

    summary_rows = []

    for label, indices in [
        ("V4", v4),
        ("V5", v5),
    ]:

        selected = stats[
            stats["index"].isin(
                indices
            )
        ].copy()

        X_selected = X[
            :,
            indices
        ]

        max_red = (
            max_pairwise_redundancy_distribution(
                X_selected
            )
        )

        row = {
            "set": label,
            "n": len(indices),

            "mean_lag_score":
                selected["lag_score"].mean(),

            "median_lag_score":
                selected["lag_score"].median(),

            "mean_spectral_score":
                selected["spectral_score"].mean(),

            "median_spectral_score":
                selected["spectral_score"].median(),

            "mean_combined_score":
                selected["combined_score"].mean(),

            "median_combined_score":
                selected["combined_score"].median(),

            "mean_abs_pairwise_corr":
                mean_pairwise_redundancy(
                    X_selected
                ),

            "median_max_pairwise_corr":
                np.median(max_red),

            "p90_max_pairwise_corr":
                np.percentile(
                    max_red,
                    90
                ),
        }

        # OT-only lag relevance.
        for lag in LAGS:

            if lag >= len(y):
                continue

            x_lag = X_selected[:-lag]
            y_future = y[lag:]

            correlations = []

            for j in range(
                X_selected.shape[1]
            ):

                correlations.append(
                    abs(
                        safe_corr(
                            x_lag[:, j],
                            y_future
                        )
                    )
                )

            row[
                f"mean_lag_{lag}"
            ] = np.mean(
                correlations
            )

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 70)
    print("V4 vs V5 quality summary")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Unique V4 vs V5 variables
    # --------------------------------------------------------

    unique_v4_stats = stats[
        stats["index"].isin(
            only_v4
        )
    ]

    unique_v5_stats = stats[
        stats["index"].isin(
            only_v5
        )
    ]

    print()
    print("=" * 70)
    print("Variables unique to each selector")
    print("=" * 70)

    print(
        "V4-only mean combined:",
        unique_v4_stats[
            "combined_score"
        ].mean()
    )

    print(
        "V5-only mean combined:",
        unique_v5_stats[
            "combined_score"
        ].mean()
    )

    # --------------------------------------------------------
    # Rank analysis
    # --------------------------------------------------------

    stats["rank_combined"] = (
        stats["combined_score"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    for label, indices in [
        ("V4", v4),
        ("V5", v5),
    ]:

        selected = stats[
            stats["index"].isin(indices)
        ]

        print()
        print(
            f"{label} rank statistics:"
        )

        print(
            f"  mean rank   : "
            f"{selected['rank_combined'].mean():.2f}"
        )

        print(
            f"  median rank : "
            f"{selected['rank_combined'].median():.2f}"
        )

        print(
            f"  top-100     : "
            f"{np.sum(selected['rank_combined'] <= 100)}"
        )

        print(
            f"  top-200     : "
            f"{np.sum(selected['rank_combined'] <= 200)}"
        )

    # --------------------------------------------------------
    # Save CSVs
    # --------------------------------------------------------

    stats_file = os.path.join(
        OUTPUT_DIR,
        "v4_v5_variable_scores.csv"
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "v4_v5_summary.csv"
    )

    stats.to_csv(
        stats_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Plot 1: score distributions
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        stats[
            stats["index"].isin(v4)
        ]["combined_score"],
        bins=30,
        alpha=0.55,
        label="V4"
    )

    plt.hist(
        stats[
            stats["index"].isin(v5)
        ]["combined_score"],
        bins=30,
        alpha=0.55,
        label="V5"
    )

    plt.xlabel(
        "Combined temporal-spectral relevance"
    )

    plt.ylabel(
        "Number of variables"
    )

    plt.title(
        "V4 vs V5 selected-variable relevance"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "01_relevance_distribution.png"
        ),
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: ranks
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        np.sort(
            stats[
                stats["index"].isin(v4)
            ]["rank_combined"].values
        ),
        label="V4"
    )

    plt.plot(
        np.sort(
            stats[
                stats["index"].isin(v5)
            ]["rank_combined"].values
        ),
        label="V5"
    )

    plt.xlabel(
        "Selected-variable order"
    )

    plt.ylabel(
        "Global relevance rank"
    )

    plt.title(
        "Global relevance ranks of selected variables"
    )

    plt.gca().invert_yaxis()

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "02_relevance_ranks.png"
        ),
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3: unique-variable score comparison
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        [
            unique_v4_stats[
                "combined_score"
            ].values,

            unique_v5_stats[
                "combined_score"
            ].values,
        ],
        labels=[
            "V4-only",
            "V5-only"
        ]
    )

    plt.ylabel(
        "Combined relevance"
    )

    plt.title(
        "Relevance of variables selected uniquely by each version"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "03_unique_variable_relevance.png"
        ),
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 4: selected-variable index distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        v4,
        bins=30,
        alpha=0.55,
        label="V4"
    )

    plt.hist(
        v5,
        bins=30,
        alpha=0.55,
        label="V5"
    )

    plt.xlabel(
        "Original variable index"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Where do V4 and V5 select variables?"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "04_variable_index_distribution.png"
        ),
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 5: lag profile
    # --------------------------------------------------------

    v4_lag = []
    v5_lag = []

    for lag in LAGS:

        if lag >= len(y):
            continue

        x_lag = X[:-lag]
        y_future = y[lag:]

        vals_v4 = [
            abs(
                safe_corr(
                    x_lag[:, i],
                    y_future
                )
            )
            for i in v4
        ]

        vals_v5 = [
            abs(
                safe_corr(
                    x_lag[:, i],
                    y_future
                )
            )
            for i in v5
        ]

        v4_lag.append(
            np.mean(vals_v4)
        )

        v5_lag.append(
            np.mean(vals_v5)
        )

    plt.figure(
        figsize=(10, 6)
    )

    valid_lags = [
        lag
        for lag in LAGS
        if lag < len(y)
    ]

    plt.plot(
        valid_lags,
        v4_lag,
        marker="o",
        label="V4"
    )

    plt.plot(
        valid_lags,
        v5_lag,
        marker="o",
        label="V5"
    )

    plt.xlabel(
        "Lag"
    )

    plt.ylabel(
        "Mean absolute correlation with future OT"
    )

    plt.title(
        "Forecasting-oriented lag relevance"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "05_lag_relevance.png"
        ),
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 6: redundancy
    # --------------------------------------------------------

    v4_max_red = (
        max_pairwise_redundancy_distribution(
            X[:, v4]
        )
    )

    v5_max_red = (
        max_pairwise_redundancy_distribution(
            X[:, v5]
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        v4_max_red,
        bins=30,
        alpha=0.55,
        label="V4"
    )

    plt.hist(
        v5_max_red,
        bins=30,
        alpha=0.55,
        label="V5"
    )

    plt.xlabel(
        "Maximum absolute correlation with another selected variable"
    )

    plt.ylabel(
        "Number of variables"
    )

    plt.title(
        "Within-subset redundancy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "06_redundancy_distribution.png"
        ),
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # Top variables
    # --------------------------------------------------------

    for label, indices in [
        ("V4", v4),
        ("V5", v5),
    ]:

        top = (
            stats[
                stats["index"].isin(indices)
            ]
            .sort_values(
                "combined_score",
                ascending=False
            )
            .head(TOP_K)
        )

        print()
        print("=" * 70)
        print(
            f"Top {TOP_K} variables according to common analysis metric: {label}"
        )
        print("=" * 70)

        print(
            top[
                [
                    "index",
                    "name",
                    "lag_score",
                    "spectral_score",
                    "combined_score",
                    "rank_combined",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save textual interpretation
    # --------------------------------------------------------

    v4_mean = summary.loc[
        summary["set"] == "V4",
        "mean_combined_score"
    ].iloc[0]

    v5_mean = summary.loc[
        summary["set"] == "V5",
        "mean_combined_score"
    ].iloc[0]

    v4_red = summary.loc[
        summary["set"] == "V4",
        "mean_abs_pairwise_corr"
    ].iloc[0]

    v5_red = summary.loc[
        summary["set"] == "V5",
        "mean_abs_pairwise_corr"
    ].iloc[0]

    if v4_mean > v5_mean and v4_red <= v5_red:
        conclusion = (
            "V4 is stronger under the common analysis metrics: "
            "higher relevance and no worse average redundancy."
        )
    elif v5_mean > v4_mean and v5_red <= v4_red:
        conclusion = (
            "V5 is stronger under the common analysis metrics: "
            "higher relevance and no worse average redundancy."
        )
    else:
        conclusion = (
            "There is a relevance/diversity trade-off between V4 and V5. "
            "Use the common metrics together with the iTransformer result."
        )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "interpretation.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "V4 vs V5 Selection Analysis\n"
        )

        f.write(
            "===========================\n\n"
        )

        f.write(
            f"V4/V5 Jaccard overlap: {jaccard:.6f}\n"
        )

        f.write(
            f"Common variables: {len(intersection)}\n"
        )

        f.write(
            f"V4-only variables: {len(only_v4)}\n"
        )

        f.write(
            f"V5-only variables: {len(only_v5)}\n\n"
        )

        f.write(
            conclusion
        )

    print()
    print("=" * 70)
    print("Analysis finished")
    print("=" * 70)

    print(
        f"Results written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
