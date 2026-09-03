import os
import sys

import numpy as np
import pandas as pd

# Allow importing from project root.
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from utils.target_aware_frequency_selector import (
    TargetAwareFrequencySelector
)


INPUT_PATH = "./dataset/traffic/traffic.csv"
OUTPUT_PATH = "./dataset/traffic/traffic_v3.csv"

TARGET = "OT"

# First experiment:
# 862 -> 720
NUM_VARIABLES = 720


def main():

    print()
    print("=" * 70)
    print("Creating Traffic v3")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Original shape: {df.shape}"
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    if "date" not in df.columns:
        raise ValueError(
            "Expected 'date' column."
        )

    # ---------------------------------------------------------
    # Fixed schema:
    #
    # date, col1, ..., colN, OT
    #
    # date and OT are preserved.
    # ---------------------------------------------------------

    feature_columns = [
        c
        for c in df.columns
        if c not in ("date", TARGET)
    ]

    print(
        f"Original variables: "
        f"{len(feature_columns)}"
    )

    X = (
        df[feature_columns]
        .values
        .astype(np.float64)
    )

    y = (
        df[TARGET]
        .values
        .astype(np.float64)
    )

    # ---------------------------------------------------------
    # Selector
    # ---------------------------------------------------------

    selector = TargetAwareFrequencySelector(
        num_variables=NUM_VARIABLES,

        # IMPORTANT:
        # only the first 70% is used for selection.
        train_ratio=0.70,

        random_state=2023,

        # Traffic/high-frequency oriented.
        lags=(
            1,
            3,
            6,
            12,
            24,
            48,
            96,
        ),

        mi_samples=12000,

        coherence_nperseg=256,
        coherence_noverlap=128,

        high_frequency_ratio=0.50,

        # Main relevance signal.
        weight_lag=0.40,
        weight_mi=0.35,
        weight_frequency=0.25,

        # Prevent selecting many near-duplicates.
        redundancy_weight=0.30,
    )

    selector.fit(
        X,
        y
    )

    selected_indices = (
        selector.get_indices()
    )

    selected_columns = [
        feature_columns[i]
        for i in selected_indices
    ]

    print()
    print("=" * 70)
    print("Selection finished")
    print("=" * 70)

    print(
        f"Selected variables: "
        f"{len(selected_columns)}"
    )

    print()
    print("Selected columns:")

    for i, column in enumerate(
        selected_columns,
        start=1
    ):
        print(
            f"{i:4d}: {column}"
        )

    # ---------------------------------------------------------
    # Build new dataset.
    #
    # IMPORTANT:
    # We do NOT modify values.
    # We only remove unselected columns.
    # ---------------------------------------------------------

    output_columns = (
        ["date"]
        + selected_columns
        + [TARGET]
    )

    df_v3 = df[
        output_columns
    ].copy()

    # ---------------------------------------------------------
    # Save dataset
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df_v3.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Save selector metadata as well.
    selector.save(
        "./dataset/traffic/traffic_v3_selector.npz"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Original shape : {df.shape}"
    )

    print(
        f"New shape      : {df_v3.shape}"
    )

    print(
        f"Output         : {OUTPUT_PATH}"
    )

    print(
        f"Selector       : "
        f"./dataset/traffic/traffic_v3_selector.npz"
    )


if __name__ == "__main__":
    main()