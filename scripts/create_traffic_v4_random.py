import os
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "./dataset/traffic/traffic.csv"
OUTPUT_DIR = "./dataset/traffic"

N_FIRST_VARS = 600


# ============================================================
# Main
# ============================================================

def create_first_k_subset(n_vars: int = N_FIRST_VARS):
    print("=" * 60)
    print(f"Creating First-{n_vars} subset")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print(f"Original shape: {df.shape}")

    # --------------------------------------------------------
    # Validate date and target variables
    # --------------------------------------------------------
    if "date" not in df.columns:
        raise ValueError("Column 'date' not found.")

    if "OT" not in df.columns:
        raise ValueError("Column 'OT' not found.")

    date_col = "date"
    target_col = "OT"

    # All candidate variables in original order (excluding date and OT)
    candidate_cols = [
        c for c in df.columns
        if c not in {date_col, target_col}
    ]

    print(f"Candidate variables available: {len(candidate_cols)}")

    if len(candidate_cols) < n_vars:
        raise ValueError(
            f"Only {len(candidate_cols)} candidate variables available, "
            f"but {n_vars} requested."
        )

    # --------------------------------------------------------
    # Sequential Selection (First N variables)
    # --------------------------------------------------------
    selected_cols = candidate_cols[:n_vars]

    # --------------------------------------------------------
    # Construct final column order: date -> first N vars -> OT
    # --------------------------------------------------------
    output_cols = [date_col] + selected_cols + [target_col]

    df_out = df[output_cols].copy()

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"traffic_v4_first_{n_vars}.csv"
    )

    metadata_file = os.path.join(
        OUTPUT_DIR,
        f"traffic_v4_first_{n_vars}_selection.txt"
    )

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------
    df_out.to_csv(output_file, index=False)

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write("Sequential Traffic Variable Selection (First N)\n")
        f.write("==============================================\n\n")
        f.write(f"Requested variables: {n_vars}\n")
        f.write(f"Original shape: {df.shape}\n")
        f.write(f"New shape: {df_out.shape}\n\n")

        f.write("Selected variables:\n")
        for i, col in enumerate(selected_cols, start=1):
            f.write(f"{i}: {col}\n")

    print(f"Selected variables: {len(selected_cols)}")
    print(f"New shape: {df_out.shape}")
    print(f"Saved dataset: {output_file}")
    print(f"Saved metadata: {metadata_file}")
    print()

    return output_file


if __name__ == "__main__":
    create_first_k_subset(N_FIRST_VARS)

    print("=" * 60)
    print("Dataset created successfully.")
    print("=" * 60)

# import os
# import numpy as np
# import pandas as pd
#
#
# # ============================================================
# # Configuration
# # ============================================================
#
# INPUT_FILE = "./dataset/traffic/traffic.csv"
# OUTPUT_DIR = "./dataset/traffic"
#
# N_RANDOM_VARS = 600
#
# # Two independent random selections
# SEEDS = [42]
#
#
# # ============================================================
# # Main
# # ============================================================
#
# def create_random_subset(seed: int):
#     print("=" * 60)
#     print(f"Creating Random-400 subset | seed={seed}")
#     print("=" * 60)
#
#     df = pd.read_csv(INPUT_FILE)
#
#     print(f"Original shape: {df.shape}")
#
#     # --------------------------------------------------------
#     # Separate date and numeric variables
#     # --------------------------------------------------------
#     if "date" not in df.columns:
#         raise ValueError("Column 'date' not found.")
#
#     if "OT" not in df.columns:
#         raise ValueError("Column 'OT' not found.")
#
#     date_col = "date"
#     target_col = "OT"
#
#     # All candidate variables except date and OT
#     candidate_cols = [
#         c for c in df.columns
#         if c not in {date_col, target_col}
#     ]
#
#     print(f"Candidate variables: {len(candidate_cols)}")
#
#     if len(candidate_cols) < N_RANDOM_VARS:
#         raise ValueError(
#             f"Only {len(candidate_cols)} candidate variables available, "
#             f"but {N_RANDOM_VARS} requested."
#         )
#
#     # --------------------------------------------------------
#     # Reproducible random selection
#     # --------------------------------------------------------
#     rng = np.random.default_rng(seed)
#
#     selected_cols = rng.choice(
#         candidate_cols,
#         size=N_RANDOM_VARS,
#         replace=False
#     ).tolist()
#
#     # --------------------------------------------------------
#     # Keep original column order?
#     #
#     # We put selected variables first, OT last,
#     # exactly like V4.
#     # --------------------------------------------------------
#     output_cols = [date_col] + selected_cols + [target_col]
#
#     df_out = df[output_cols].copy()
#
#     # --------------------------------------------------------
#     # Output paths
#     # --------------------------------------------------------
#     output_file = os.path.join(
#         OUTPUT_DIR,
#         f"traffic_v4_random_{N_RANDOM_VARS}.csv"
#     )
#
#     metadata_file = os.path.join(
#         OUTPUT_DIR,
#         f"traffic_v4_random_{N_RANDOM_VARS}_selection.txt"
#     )
#
#     # --------------------------------------------------------
#     # Save dataset
#     # --------------------------------------------------------
#     df_out.to_csv(output_file, index=False)
#
#     # --------------------------------------------------------
#     # Save metadata
#     # --------------------------------------------------------
#     with open(metadata_file, "w", encoding="utf-8") as f:
#         f.write("Random Traffic Variable Selection\n")
#         f.write("=================================\n\n")
#         f.write(f"Seed: {seed}\n")
#         f.write(f"Requested variables: {N_RANDOM_VARS}\n")
#         f.write(f"Original shape: {df.shape}\n")
#         f.write(f"New shape: {df_out.shape}\n\n")
#
#         f.write("Selected variables:\n")
#         for i, col in enumerate(selected_cols, start=1):
#             f.write(f"{i}: {col}\n")
#
#     print(f"Selected variables: {len(selected_cols)}")
#     print(f"New shape: {df_out.shape}")
#     print(f"Saved dataset: {output_file}")
#     print(f"Saved metadata: {metadata_file}")
#     print()
#
#     return output_file
#
#
# if __name__ == "__main__":
#     for seed in SEEDS:
#         create_random_subset(seed)
#
#     print("=" * 60)
#     print("All Random-400 datasets created successfully.")
#     print("=" * 60)