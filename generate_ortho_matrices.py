import os
import numpy as np
import pandas as pd
from numpy.linalg import eigh


# ============================================================
# CONFIG
# ============================================================

DATASETS = {
    "ETTh1": r"data/ETT-small/ETTh1.csv",
    "ETTh2": r"data/ETT-small/ETTh2.csv",
    "ETTm1": r"data/ETT-small/ETTm1.csv",
    "ETTm2": r"data/ETT-small/ETTm2.csv",
    "ECL": r"data/electricity/electricity.csv",
    "Exchange": r"data/exchange_rate/exchange_rate.csv",
    "Traffic": r"data/traffic/traffic.csv",
    "Weather": r"data/weather/weather.csv",
    "ILI": r"data/illness/national_illness.csv",
}

# Horizons used by iTransformer
LAGS = [96, 192, 336, 720]

# OLinear repository notes:
# ETT / PEMS use 0.6, others normally 0.7
TRAIN_RATIOS = {
    "ETTh1": 0.6,
    "ETTh2": 0.6,
    "ETTm1": 0.6,
    "ETTm2": 0.6,
    "Traffic": 0.7,
    "ECL": 0.7,
    "Exchange": 0.7,
    "Weather": 0.7,
    "ILI": 0.7,
}

SAVE_ROOT = "data/ortho"


# ============================================================
# GENERATE ONE Q MATRIX
# ============================================================

def generate_q_matrix(file_path, dataset_name, time_lag):

    print("=" * 70)
    print(f"Dataset : {dataset_name}")
    print(f"Lag     : {time_lag}")
    print(f"File    : {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    data = pd.read_csv(
        file_path,
        header=0
    )

    data = data.dropna(
        axis=1,
        how="all"
    )

    data = data.values

    train_ratio = TRAIN_RATIOS[dataset_name]

    train_length = int(
        data.shape[0] * train_ratio
    )

    # Same convention as OLinear for CSV:
    # first column is date/time
    A = data[
        train_length - int(data.shape[0] * train_ratio):
        train_length,
        1:
    ].astype(np.float32)

    print(f"Train portion : {A.shape}")

    sigma_list = []

    for feature_idx in range(A.shape[1]):

        lagged_matrix = np.array([
            A[
                i:
                A.shape[0] - time_lag + i + 1,
                feature_idx
            ]
            for i in range(time_lag)
        ])

        if np.isnan(lagged_matrix).any():
            lagged_matrix = np.nan_to_num(
                lagged_matrix
            )

        cov_matrix = np.cov(
            lagged_matrix
        )

        diag_vec = np.diag(
            cov_matrix
        )

        if (diag_vec < 1e-4).any():
            continue

        cov_matrix = (
            cov_matrix / diag_vec
        )

        sigma_list.append(
            np.asarray(
                cov_matrix,
                dtype=np.float32
            )
        )

    if not sigma_list:
        raise RuntimeError(
            f"No valid covariance matrices for "
            f"{dataset_name}, lag={time_lag}"
        )

    Sigma = np.mean(
        sigma_list,
        axis=0
    )

    # Eigen decomposition
    eigenvalues, eigenvectors = eigh(
        Sigma
    )

    # Same as OLinear
    Q = np.flip(
        eigenvectors.T,
        axis=0
    ).astype(np.float32)

    output_dir = os.path.join(
        SAVE_ROOT,
        dataset_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        f"Q_{time_lag}.npy"
    )

    np.save(
        output_file,
        Q
    )

    print(
        f"Saved: {output_file}"
    )

    print(
        f"Shape: {Q.shape}"
    )


# ============================================================
# MAIN
# ============================================================
# Q-matrix lengths required per dataset
DATASET_LAGS = {
    "ETTh1": [96, 192, 336, 720],
    "ETTh2": [96, 192, 336, 720],
    "ETTm1": [96, 192, 336, 720],
    "ETTm2": [96, 192, 336, 720],
    "ECL": [96, 192, 336, 720],
    "Exchange": [96, 192, 336, 720],
    "Traffic": [96, 192, 336, 720],
    "Weather": [96, 192, 336, 720],

    # ILI has shorter forecasting lengths
    "ILI": [12, 24, 36, 48],
}


if __name__ == "__main__":

    for dataset_name, file_path in DATASETS.items():

        for lag in DATASET_LAGS[dataset_name]:
            generate_q_matrix(
                file_path,
                dataset_name,
                lag
            )

    print()
    print("=" * 70)
    print("ALL Q MATRICES GENERATED SUCCESSFULLY")
    print("=" * 70)