import os
import numpy as np
import pandas as pd
import time

from sklearn.feature_selection import mutual_info_regression

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "./dataset/traffic/traffic.csv"
OUTPUT_FILE = "./dataset/traffic/traffic_v5.csv"

TARGET = "OT"

# Number of selected input variables.
N_SELECTED = 720

RANDOM_STATE = 2023

# Total training samples used for feature analysis.
ANALYSIS_SAMPLES = 4096

# IMPORTANT:
# Analysis is performed on contiguous temporal blocks.
N_BLOCKS = 4

# Lag configuration.
LAGS = [1, 2, 3, 6, 12, 24]

# Frequency representation.
N_FREQ_BANDS = 8

# MI configuration.
MI_NEIGHBORS = 5

# Candidate pool.
CANDIDATE_SIZE = 820

# Relevance weights.
W_LAG = 0.30
W_MI = 0.25
W_SPECTRAL = 0.20

# Diversity weight.
W_DIVERSITY = 0.25

# Additional redundancy penalty.
REDUNDANCY_POWER = 1.0


# ============================================================
# Utilities
# ============================================================

def minmax_normalize(x):
    x = np.asarray(x, dtype=np.float64)

    minimum = np.nanmin(x)
    maximum = np.nanmax(x)

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return np.zeros_like(x)

    if maximum - minimum < 1e-12:
        return np.zeros_like(x)

    return (x - minimum) / (maximum - minimum)


def zscore_columns(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)

    return (x - mean) / (std + 1e-8)


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - np.mean(x)
    y = y - np.mean(y)

    denominator = (
            np.sqrt(np.sum(x * x) * np.sum(y * y))
            + 1e-12
    )

    value = np.sum(x * y) / denominator

    if not np.isfinite(value):
        return 0.0

    return float(value)


# ============================================================
# Selector
# ============================================================

class DiversityAwareTemporalSelectorV5:

    def __init__(
            self,
            n_selected,
            candidate_size,
            analysis_samples,
            n_blocks,
            random_state=2023,
    ):
        self.n_selected = n_selected
        self.candidate_size = candidate_size
        self.analysis_samples = analysis_samples
        self.n_blocks = n_blocks
        self.random_state = random_state

        self.selected_indices_ = None
        self.scores_ = None
        self.feature_names_ = None

    # ========================================================
    # Temporal block construction
    # ========================================================

    def _build_blocks(self, x, y):

        n = len(y)

        total_samples = min(
            self.analysis_samples,
            n
        )

        block_length = total_samples // self.n_blocks

        if block_length < max(LAGS) + 2:
            raise ValueError(
                "Analysis blocks are too short for the requested lags."
            )

        usable_samples = (
                block_length * self.n_blocks
        )

        # Spread blocks across the training interval.
        #
        # This avoids concentrating all analysis
        # on the beginning of the training set.
        max_start = max(
            0,
            n - usable_samples
        )

        if self.n_blocks == 1:
            starts = [0]
        else:
            starts = np.linspace(
                0,
                max_start,
                self.n_blocks
            ).astype(int)

        blocks = []

        for start in starts:

            end = start + block_length

            if end > n:
                end = n
                start = end - block_length

            block_x = x[start:end]
            block_y = y[start:end]

            blocks.append(
                (block_x, block_y)
            )

        return blocks

    # ========================================================
    # Lag relevance
    # ========================================================

    def _lag_relevance(self, blocks):
        t_start = time.time()
        n_variables = blocks[0][0].shape[1]

        scores = np.zeros(
            n_variables,
            dtype=np.float64
        )

        valid_blocks = 0

        print(
            "Calculating lagged temporal relevance..."
        )

        for lag in LAGS:

            lag_values = []

            for block_x, block_y in blocks:

                if lag >= len(block_y):
                    continue

                x_lag = block_x[:-lag]
                y_future = block_y[lag:]

                y_centered = (
                        y_future
                        - np.mean(y_future)
                )

                x_centered = (
                        x_lag
                        - np.mean(
                    x_lag,
                    axis=0
                )
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

                lag_values.append(corr)

            if lag_values:
                lag_mean = np.mean(
                    np.stack(lag_values),
                    axis=0
                )

                scores += lag_mean
                valid_blocks += 1

            print(
                f"  lag={lag:>3} completed"
            )

        if valid_blocks > 0:
            scores /= (
                valid_blocks
            )

        print(f"  [Duration: {time.time() - t_start:.2f}s]")
        return minmax_normalize(
            scores
        )

    # ========================================================
    # Lagged mutual information
    # ========================================================

    def _lagged_mutual_information(self, blocks):
        t_start = time.time()
        n_variables = blocks[0][0].shape[1]

        scores = np.zeros(
            n_variables,
            dtype=np.float64
        )

        lag_count = 0

        print(
            "Calculating lagged mutual information..."
        )

        for lag in LAGS:

            lag_values = []

            for block_x, block_y in blocks:

                if lag >= len(block_y):
                    continue

                x_lag = block_x[:-lag]
                y_future = block_y[lag:]

                block_scores = np.zeros(
                    n_variables,
                    dtype=np.float64
                )

                for i in range(n_variables):

                    xi = x_lag[:, i]

                    if np.std(xi) < 1e-12:
                        continue

                    try:

                        mi = mutual_info_regression(
                            xi.reshape(-1, 1),
                            y_future,
                            n_neighbors=MI_NEIGHBORS,
                            random_state=self.random_state,
                        )

                        block_scores[i] = mi[0]

                    except Exception:

                        block_scores[i] = 0.0

                lag_values.append(
                    block_scores
                )

            if lag_values:
                lag_mean = np.mean(
                    np.stack(lag_values),
                    axis=0
                )

                scores += lag_mean
                lag_count += 1

            print(
                f"  lag={lag:>3} completed"
            )

        if lag_count > 0:
            scores /= lag_count

        print(f"  [Duration: {time.time() - t_start:.2f}s]")
        return minmax_normalize(
            scores
        )

    # ========================================================
    # Frequency representation
    # ========================================================

    def _spectral_features(self, x):

        x = x - np.mean(
            x,
            axis=0,
            keepdims=True
        )

        spectrum = np.fft.rfft(
            x,
            axis=0
        )

        power = np.abs(
            spectrum
        ) ** 2

        # Remove DC.
        power = power[1:]

        if len(power) == 0:
            return np.zeros(
                (
                    x.shape[1],
                    N_FREQ_BANDS
                )
            )

        # Normalize energy per variable.
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
            N_FREQ_BANDS + 1,
            dtype=int
        )

        features = []

        for band_idx in range(
                N_FREQ_BANDS
        ):

            start = edges[
                band_idx
            ]

            end = edges[
                band_idx + 1
                ]

            if end <= start:

                band = np.zeros(
                    x.shape[1]
                )

            else:

                band = np.sum(
                    power[start:end],
                    axis=0
                )

            features.append(
                band
            )

        return np.stack(
            features,
            axis=1
        )

    # ========================================================
    # Spectral relevance
    # ========================================================

    def _spectral_relevance(self, blocks):
        t_start = time.time()
        n_variables = blocks[0][0].shape[1]

        block_scores = []

        print(
            "Calculating spectral relevance..."
        )

        for block_x, block_y in blocks:

            x_spec = (
                self._spectral_features(
                    block_x
                )
            )

            y = (
                    block_y
                    - np.mean(block_y)
            )

            y_spec = (
                    np.abs(
                        np.fft.rfft(y)
                    ) ** 2
            )

            y_spec = y_spec[1:]

            if len(y_spec) == 0:
                continue

            y_spec /= (
                    np.sum(y_spec)
                    + 1e-12
            )

            n_freq = len(y_spec)

            edges = np.linspace(
                0,
                n_freq,
                N_FREQ_BANDS + 1,
                dtype=int
            )

            y_bands = []

            for band_idx in range(
                    N_FREQ_BANDS
            ):

                start = edges[
                    band_idx
                ]

                end = edges[
                    band_idx + 1
                    ]

                if end <= start:

                    value = 0.0

                else:

                    value = np.sum(
                        y_spec[
                        start:end
                        ]
                    )

                y_bands.append(
                    value
                )

            y_bands = np.asarray(
                y_bands
            )

            numerator = np.sum(
                x_spec
                * y_bands[None, :],
                axis=1
            )

            denominator = (
                    np.sqrt(
                        np.sum(
                            x_spec ** 2,
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

            similarity = (
                    numerator
                    / denominator
            )

            similarity = np.maximum(
                similarity,
                0.0
            )

            block_scores.append(
                similarity
            )

        if not block_scores:
            return np.zeros(
                n_variables
            )

        scores = np.mean(
            np.stack(
                block_scores
            ),
            axis=0
        )

        print(f"  [Duration: {time.time() - t_start:.2f}s]")
        return minmax_normalize(
            scores
        )

    # ========================================================
    # Temporal-frequency signature
    # ========================================================

    def _build_signature(self, blocks):
        t_start = time.time()
        signatures = []

        print(
            "Building temporal-frequency signatures..."
        )

        for block_x, _ in blocks:

            x_norm = zscore_columns(
                block_x
            )

            spectral = (
                self._spectral_features(
                    x_norm
                )
            )

            autocorr_features = []

            for lag in [
                1,
                2,
                6,
                12,
                24
            ]:

                if lag >= len(x_norm):
                    autocorr_features.append(
                        np.zeros(
                            x_norm.shape[1]
                        )
                    )

                    continue

                a = x_norm[:-lag]
                b = x_norm[lag:]

                numerator = np.sum(
                    a * b,
                    axis=0
                )

                denominator = (
                        np.sqrt(
                            np.sum(
                                a ** 2,
                                axis=0
                            )
                            * np.sum(
                                b ** 2,
                                axis=0
                            )
                        )
                        + 1e-12
                )

                corr = (
                        numerator
                        / denominator
                )

                autocorr_features.append(
                    corr
                )

            autocorr = np.stack(
                autocorr_features,
                axis=1
            )

            signature = np.concatenate(
                [
                    spectral,
                    autocorr
                ],
                axis=1
            )

            signatures.append(
                signature
            )

        # Average the signatures of the
        # contiguous temporal blocks.
        signature = np.mean(
            np.stack(signatures),
            axis=0
        )

        signature = zscore_columns(
            signature
        )

        print(f"  [Duration: {time.time() - t_start:.2f}s]")
        return signature.astype(
            np.float32
        )

    # ========================================================
    # Candidate pool
    # ========================================================

    def _build_candidate_pool(
            self,
            lag_score,
            mi_score,
            spectral_score
    ):

        initial_score = (
                W_LAG * lag_score
                + W_MI * mi_score
                + W_SPECTRAL
                * spectral_score
        )

        candidate_size = min(
            self.candidate_size,
            len(initial_score)
        )

        order = np.argsort(
            initial_score
        )[::-1]

        candidates = order[
                     :candidate_size
                     ]

        return (
            candidates,
            initial_score
        )

    # ========================================================
    # Diversity-aware selection
    # ========================================================

    def _diversity_selection(
            self,
            x,
            candidates,
            relevance,
            signatures
    ):
        t_start = time.time()
        print()
        print(
            "Starting diversity-aware selection..."
        )

        candidate_x = (
            x[:, candidates]
        )

        candidate_x = zscore_columns(
            candidate_x
        )

        # ----------------------------------------------------
        # Statistical redundancy
        # ----------------------------------------------------

        print(
            "Calculating candidate correlation matrix..."
        )

        corr_matrix = (
                              candidate_x.T
                              @ candidate_x
                      ) / max(
            len(candidate_x) - 1,
            1
        )

        corr_matrix = np.clip(
            corr_matrix,
            -1.0,
            1.0
        )

        abs_corr = np.abs(
            corr_matrix
        )

        np.fill_diagonal(
            abs_corr,
            0.0
        )

        # ----------------------------------------------------
        # Temporal-frequency signature redundancy
        # ----------------------------------------------------

        candidate_signature = (
            signatures[candidates]
        )

        candidate_signature = (
            zscore_columns(
                candidate_signature
            )
        )

        signature_norm = (
            np.linalg.norm(
                candidate_signature,
                axis=1,
                keepdims=True
            )
        )

        signature_norm[
            signature_norm < 1e-12
            ] = 1.0

        candidate_signature = (
                candidate_signature
                / signature_norm
        )

        signature_similarity = (
                candidate_signature
                @ candidate_signature.T
        )

        signature_similarity = np.clip(
            signature_similarity,
            -1.0,
            1.0
        )

        signature_similarity = np.abs(
            signature_similarity
        )

        np.fill_diagonal(
            signature_similarity,
            0.0
        )

        # ----------------------------------------------------
        # Greedy selection
        # ----------------------------------------------------

        selected = []

        remaining = np.ones(
            len(candidates),
            dtype=bool
        )

        first = np.argmax(
            relevance[candidates]
        )

        selected.append(
            first
        )

        remaining[first] = False

        print(
            f"Selected 1/{self.n_selected} "
            f"| variable={candidates[first]} "
            f"| relevance="
            f"{relevance[candidates[first]]:.6f}"
        )

        while (
                len(selected)
                < self.n_selected
        ):

            remaining_indices = np.where(
                remaining
            )[0]

            if len(remaining_indices) == 0:
                break

            selected_indices = np.asarray(
                selected,
                dtype=np.int64
            )

            # Strongest redundancy with ANY
            # already selected variable.
            redundancy = np.max(
                abs_corr[
                    np.ix_(
                        remaining_indices,
                        selected_indices
                    )
                ],
                axis=1
            )

            # Strongest temporal-frequency
            # similarity with ANY selected variable.
            signature_redundancy = np.max(
                signature_similarity[
                    np.ix_(
                        remaining_indices,
                        selected_indices
                    )
                ],
                axis=1
            )

            diversity = (
                    1.0
                    - 0.5 * redundancy
                    - 0.5 * signature_redundancy
            )

            diversity = np.clip(
                diversity,
                0.0,
                1.0
            )

            candidate_relevance = (
                relevance[
                    candidates[
                        remaining_indices
                    ]
                ]
            )

            score = (
                    (1.0 - W_DIVERSITY)
                    * candidate_relevance
                    + W_DIVERSITY
                    * diversity
            )

            score -= (
                    0.05
                    * (
                            redundancy
                            ** REDUNDANCY_POWER
                    )
            )

            best_position = np.argmax(
                score
            )

            best = remaining_indices[
                best_position
            ]

            selected.append(
                best
            )

            remaining[best] = False

            current = len(selected)

            if (
                    current % 25 == 0
                    or current <= 5
            ):
                print(
                    f"Selected "
                    f"{current}/{self.n_selected} "
                    f"| variable="
                    f"{candidates[best]} "
                    f"| score="
                    f"{score[best_position]:.6f} "
                    f"| relevance="
                    f"{candidate_relevance[best_position]:.6f} "
                    f"| redundancy="
                    f"{redundancy[best_position]:.4f}"
                )

        selected = np.asarray(
            selected,
            dtype=np.int64
        )

        print(f"  [Selection Duration: {time.time() - t_start:.2f}s]")
        return candidates[
            selected
        ]

    # ========================================================
    # Fit
    # ========================================================

    def fit(
            self,
            x_train,
            y_train,
            feature_names
    ):

        print()
        print("=" * 70)
        print(
            "DiversityAwareTemporalSelectorV5"
        )
        print("=" * 70)

        print(
            f"Training rows : {len(y_train)}"
        )

        print(
            f"Variables     : {x_train.shape[1]}"
        )

        print(
            f"Requested     : {self.n_selected}"
        )

        # ----------------------------------------------------
        # Build contiguous blocks
        # ----------------------------------------------------

        blocks = self._build_blocks(
            x_train,
            y_train
        )

        print(
            f"Analysis blocks : {len(blocks)}"
        )

        print(
            f"Block length    : "
            f"{len(blocks[0][1])}"
        )

        # ----------------------------------------------------
        # Feature relevance
        # ----------------------------------------------------

        lag_score = (
            self._lag_relevance(
                blocks
            )
        )

        mi_score = (
            self._lagged_mutual_information(
                blocks
            )
        )

        spectral_score = (
            self._spectral_relevance(
                blocks
            )
        )

        # ----------------------------------------------------
        # Combined relevance
        # ----------------------------------------------------

        relevance = (
                W_LAG * lag_score
                + W_MI * mi_score
                + W_SPECTRAL
                * spectral_score
        )

        relevance = minmax_normalize(
            relevance
        )

        # ----------------------------------------------------
        # Candidate pool
        # ----------------------------------------------------

        candidates, initial_score = (
            self._build_candidate_pool(
                lag_score,
                mi_score,
                spectral_score
            )
        )

        print()
        print(
            f"Candidate pool: "
            f"{len(candidates)}"
        )

        # ----------------------------------------------------
        # Signatures
        # ----------------------------------------------------

        signatures = (
            self._build_signature(
                blocks
            )
        )

        # ----------------------------------------------------
        # Diversity-aware selection
        # ----------------------------------------------------

        selected = (
            self._diversity_selection(
                blocks[0][0]
                if len(blocks) == 1
                else np.concatenate(
                    [
                        block_x
                        for block_x, _
                        in blocks
                    ],
                    axis=0
                ),
                candidates,
                relevance,
                signatures
            )
        )

        self.selected_indices_ = (
            selected
        )

        self.scores_ = {
            "lag": lag_score,
            "mi": mi_score,
            "spectral": spectral_score,
            "relevance": relevance,
            "initial": initial_score,
        }

        self.feature_names_ = [
            feature_names[i]
            for i in selected
        ]

        return self.feature_names_


# ============================================================
# Main
# ============================================================

def main():
    t_global_start = time.time()  # زمان شروع کل فرآیند

    print("=" * 70)
    print(
        "Creating Traffic V5"
    )
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            INPUT_FILE
        )

    t_read = time.time()
    df = pd.read_csv(
        INPUT_FILE
    )
    print(f"Data loading duration: {time.time() - t_read:.2f}s")

    print(
        f"Original shape: {df.shape}"
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found."
        )

    if "date" not in df.columns:
        raise ValueError(
            "Dataset must contain 'date'."
        )

    feature_names = [
        c
        for c in df.columns
        if c not in {
            "date",
            TARGET
        }
    ]

    print(
        f"Original variables: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_df = df[
        feature_names + [TARGET]
        ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Standard Traffic split
    # --------------------------------------------------------

    n_rows = len(df)

    train_end = int(
        n_rows * 0.70
    )

    train_numeric = (
        numeric_df.iloc[
        :train_end
        ].copy()
    )

    train_medians = (
        train_numeric.median()
    )

    numeric_df = (
        numeric_df.fillna(
            train_medians
        )
    )

    # --------------------------------------------------------
    # Training data only
    # --------------------------------------------------------

    x_train = (
        numeric_df[
            feature_names
        ]
        .iloc[:train_end]
        .values
        .astype(np.float64)
    )

    y_train = (
        numeric_df[
            TARGET
        ]
        .iloc[:train_end]
        .values
        .astype(np.float64)
    )

    # --------------------------------------------------------
    # Remove pathological variables
    # --------------------------------------------------------

    valid_mask = np.isfinite(
        x_train
    ).all(axis=0)

    feature_names = [
        feature_names[i]
        for i in range(
            len(feature_names)
        )
        if valid_mask[i]
    ]

    x_train = x_train[
              :,
              valid_mask
              ]

    print(
        f"Valid variables: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Selector
    # --------------------------------------------------------
    t_fit_start = time.time()
    selector = (
        DiversityAwareTemporalSelectorV5(
            n_selected=min(
                N_SELECTED,
                len(feature_names)
            ),
            candidate_size=min(
                CANDIDATE_SIZE,
                len(feature_names)
            ),
            analysis_samples=ANALYSIS_SAMPLES,
            n_blocks=N_BLOCKS,
            random_state=RANDOM_STATE,
        )
    )

    selected_features = (
        selector.fit(
            x_train,
            y_train,
            feature_names
        )
    )
    print(f"\nTotal Selection Process Duration: {time.time() - t_fit_start:.2f}s")

    selected_features = list(
        dict.fromkeys(
            selected_features
        )
    )

    print()
    print("=" * 70)
    print(
        "Selection completed"
    )
    print("=" * 70)

    print(
        f"Selected variables: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------
    # Create V5 dataset
    # --------------------------------------------------------

    output_columns = (
            ["date"]
            + selected_features
            + [TARGET]
    )

    df_v5 = df[
        output_columns
    ].copy()

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    t_save = time.time()
    df_v5.to_csv(
        OUTPUT_FILE,
        index=False
    )
    print(f"File saving duration: {time.time() - t_save:.2f}s")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_file = (
        OUTPUT_FILE.replace(
            ".csv",
            "_selection.txt"
        )
    )

    with open(
            metadata_file,
            "w",
            encoding="utf-8"
    ) as f:

        f.write(
            "Traffic V5 variable selection\n"
        )

        f.write(
            "========================================\n"
        )

        f.write(
            f"Original variables: "
            f"{len(feature_names)}\n"
        )

        f.write(
            f"Selected variables: "
            f"{len(selected_features)}\n"
        )

        f.write(
            f"Training rows: "
            f"{train_end}\n"
        )

        f.write(
            f"Analysis samples: "
            f"{ANALYSIS_SAMPLES}\n"
        )

        f.write(
            f"Temporal blocks: "
            f"{N_BLOCKS}\n"
        )

        f.write(
            f"Candidate pool: "
            f"{CANDIDATE_SIZE}\n"
        )

        f.write(
            "\nSelected variables:\n"
        )

        for i, name in enumerate(
                selected_features,
                start=1
        ):
            f.write(
                f"{i:04d}: {name}\n"
            )

    print()
    print(
        f"Saved dataset: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Saved metadata: "
        f"{metadata_file}"
    )

    print()
    print(
        f"Original shape : "
        f"{df.shape}"
    )

    print(
        f"New shape      : "
        f"{df_v5.shape}"
    )

    print(f"\nTotal script execution time: {time.time() - t_global_start:.2f}s")
    print("Done.")


if __name__ == "__main__":
    main()
