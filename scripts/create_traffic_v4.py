import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "./dataset/traffic/traffic.csv"
OUTPUT_FILE = "./dataset/traffic/traffic_v4.csv"

TARGET = "OT"

# Traffic original:
# date + 861 variables + OT
#
# We keep OT as target and select this many input variables.
N_SELECTED = 600

RANDOM_STATE = 2023

# Number of temporal samples used during feature analysis.
# This does NOT modify the final dataset.
ANALYSIS_SAMPLES = 4096

# Candidate pool before diversity-aware selection.
#
# We deliberately keep this larger than N_SELECTED so that
# diversity selection has room to replace target-similar variables.
CANDIDATE_SIZE = 820

# Lag configuration.
LAGS = [1, 2, 3, 6, 12, 24]

# Number of frequency bands.
N_FREQ_BANDS = 8

# MI estimation parameters.
MI_NEIGHBORS = 5

# Score weights.
#
# These are intentionally moderate.
# No single target-correlation criterion dominates selection.
W_LAG = 0.30
W_MI = 0.25
W_SPECTRAL = 0.20
W_DIVERSITY = 0.25

# Redundancy penalty.
REDUNDANCY_POWER = 1.0


# ============================================================
# Utility
# ============================================================

def minmax_normalize(x):
    x = np.asarray(x, dtype=np.float64)

    minimum = np.nanmin(x)
    maximum = np.nanmax(x)

    if maximum - minimum < 1e-12:
        return np.zeros_like(x)

    return (x - minimum) / (maximum - minimum)


def zscore_columns(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)

    return (x - mean) / (std + 1e-8)


def safe_corr(x, y):
    """
    Correlation between one variable x and y.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - np.mean(x)
    y = y - np.mean(y)

    denominator = (
        np.sqrt(np.sum(x * x) * np.sum(y * y))
        + 1e-12
    )

    return np.sum(x * y) / denominator


# ============================================================
# Selector
# ============================================================

class DiversityAwareTemporalSelector:

    def __init__(
        self,
        n_selected,
        candidate_size,
        analysis_samples,
        random_state=2023,
    ):

        self.n_selected = n_selected
        self.candidate_size = candidate_size
        self.analysis_samples = analysis_samples
        self.random_state = random_state

        self.selected_indices_ = None
        self.scores_ = None
        self.feature_names_ = None

    # --------------------------------------------------------
    # Analysis sampling
    # --------------------------------------------------------

    def _sample_training_data(self, x, y):

        n = len(y)

        if n <= self.analysis_samples:
            return x, y

        rng = np.random.RandomState(
            self.random_state
        )

        # Random temporal points are used for MI/statistics.
        # We use a fixed random seed for reproducibility.
        indices = rng.choice(
            n,
            size=self.analysis_samples,
            replace=False
        )

        indices.sort()

        return x[indices], y[indices]

    # --------------------------------------------------------
    # Lag relevance
    # --------------------------------------------------------

    def _lag_relevance(self, x, y):

        n, p = x.shape

        scores = np.zeros(p, dtype=np.float64)

        valid_count = 0

        print("Calculating temporal lag relevance...")

        for lag in LAGS:

            if lag >= n:
                continue

            x_lag = x[:-lag]
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
                x_centered * y_centered[:, None],
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

            scores += corr

            valid_count += 1

            print(
                f"  lag={lag:>3} completed"
            )

        if valid_count > 0:
            scores /= valid_count

        return minmax_normalize(scores)

    # --------------------------------------------------------
    # Nonlinear target relevance
    # --------------------------------------------------------

    def _mutual_information(self, x, y):

        n, p = x.shape

        scores = np.zeros(p, dtype=np.float64)

        print("Calculating nonlinear mutual information...")

        for i in range(p):

            xi = x[:, i]

            # Constant variables are useless.
            if np.std(xi) < 1e-12:
                scores[i] = 0.0
                continue

            try:

                mi = mutual_info_regression(
                    xi.reshape(-1, 1),
                    y,
                    n_neighbors=MI_NEIGHBORS,
                    random_state=self.random_state
                )

                scores[i] = mi[0]

            except Exception:

                scores[i] = 0.0

            if (i + 1) % 50 == 0:
                print(
                    f"  MI: {i + 1}/{p}"
                )

        return minmax_normalize(scores)

    # --------------------------------------------------------
    # Spectral representation
    # --------------------------------------------------------

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

        power = np.abs(spectrum) ** 2

        # Remove DC.
        power = power[1:]

        if len(power) == 0:
            return np.zeros(
                (x.shape[1], N_FREQ_BANDS)
            )

        # Normalize each variable's total energy.
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

        for b in range(N_FREQ_BANDS):

            start = edges[b]
            end = edges[b + 1]

            if end <= start:
                band = np.zeros(
                    x.shape[1]
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

    # --------------------------------------------------------
    # Spectral relevance to target
    # --------------------------------------------------------

    def _spectral_relevance(self, x, y):

        print(
            "Calculating spectral relevance..."
        )

        x_spec = self._spectral_features(x)

        y = y - np.mean(y)

        y_spec = np.abs(
            np.fft.rfft(y)
        ) ** 2

        y_spec = y_spec[1:]

        if len(y_spec) == 0:
            return np.zeros(x.shape[1])

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

        for b in range(N_FREQ_BANDS):

            start = edges[b]
            end = edges[b + 1]

            if end <= start:
                value = 0.0
            else:
                value = np.sum(
                    y_spec[start:end]
                )

            y_bands.append(value)

        y_bands = np.asarray(
            y_bands
        )

        # Cosine similarity between the
        # spectral profile of each variable
        # and OT.
        numerator = np.sum(
            x_spec * y_bands[None, :],
            axis=1
        )

        denominator = (
            np.sqrt(
                np.sum(x_spec ** 2, axis=1)
            )
            * np.sqrt(
                np.sum(y_bands ** 2)
            )
            + 1e-12
        )

        similarity = (
            numerator / denominator
        )

        similarity = np.maximum(
            similarity,
            0.0
        )

        return minmax_normalize(
            similarity
        )

    # --------------------------------------------------------
    # Build temporal/frequency representation
    # --------------------------------------------------------

    def _build_signature(self, x):

        print(
            "Building temporal-frequency signatures..."
        )

        # Normalize each variable.
        x_norm = zscore_columns(x)

        # Spectral representation.
        spectral = self._spectral_features(
            x_norm
        )

        # Add temporal autocorrelation characteristics.
        autocorr_features = []

        for lag in [1, 2, 6, 12, 24]:

            if lag >= len(x_norm):
                autocorr_features.append(
                    np.zeros(x_norm.shape[1])
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
                    np.sum(a ** 2, axis=0)
                    * np.sum(b ** 2, axis=0)
                )
                + 1e-12
            )

            corr = numerator / denominator

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

        signature = zscore_columns(
            signature
        )

        return signature.astype(
            np.float32
        )

    # --------------------------------------------------------
    # Candidate screening
    # --------------------------------------------------------

    def _build_candidate_pool(
        self,
        lag_score,
        mi_score,
        spectral_score
    ):

        initial_score = (
            W_LAG * lag_score
            + W_MI * mi_score
            + W_SPECTRAL * spectral_score
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

        return candidates, initial_score

    # --------------------------------------------------------
    # Diversity-aware greedy selection
    # --------------------------------------------------------

    def _diversity_selection(
        self,
        x,
        candidates,
        relevance,
        signatures
    ):

        print()
        print(
            "Starting diversity-aware selection..."
        )
        print(
            f"Candidates : {len(candidates)}"
        )
        print(
            f"Target     : {self.n_selected}"
        )
        print()

        candidate_x = x[:, candidates]

        # Standardize candidate variables.
        candidate_x = zscore_columns(
            candidate_x
        )

        # Correlation matrix between candidates.
        #
        # This is calculated ONCE.
        #
        # This is the major performance improvement over
        # recomputing pairwise correlations inside the loop.
        print(
            "Calculating candidate correlation matrix..."
        )

        corr_matrix = (
            candidate_x.T @ candidate_x
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

        print(
            "Correlation matrix ready."
        )

        # ----------------------------------------------------
        # Frequency/temporal signature similarity
        # ----------------------------------------------------

        candidate_signature = signatures[
            candidates
        ]

        candidate_signature = zscore_columns(
            candidate_signature
        )

        signature_norm = np.linalg.norm(
            candidate_signature,
            axis=1,
            keepdims=True
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
        # Selection
        # ----------------------------------------------------

        selected = []

        remaining = np.ones(
            len(candidates),
            dtype=bool
        )

        # First feature:
        # choose strongest target relevance.
        first = np.argmax(
            relevance[candidates]
        )

        selected.append(first)
        remaining[first] = False

        print(
            f"Selected 1/{self.n_selected} "
            f"| variable={candidates[first]} "
            f"| relevance="
            f"{relevance[candidates[first]]:.6f}"
        )

        # ----------------------------------------------------
        # Greedy selection
        # ----------------------------------------------------

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

            # Maximum redundancy with selected variables.
            #
            # We use maximum rather than mean because:
            #
            # if a candidate is almost an exact duplicate
            # of ANY selected variable, it should be penalized.
            redundancy = np.max(
                abs_corr[
                    np.ix_(
                        remaining_indices,
                        selected_indices
                    )
                ],
                axis=1
            )

            # Maximum signature similarity with selected
            # temporal/frequency patterns.
            signature_redundancy = np.max(
                signature_similarity[
                    np.ix_(
                        remaining_indices,
                        selected_indices
                    )
                ],
                axis=1
            )

            # Diversity reward.
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

            # Final score.
            #
            # Relevance remains important,
            # but diversity explicitly protects
            # complementary variables.
            score = (
                (1.0 - W_DIVERSITY)
                * candidate_relevance
                + W_DIVERSITY
                * diversity
            )

            # Small additional penalty for extreme redundancy.
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

            selected.append(best)

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

        return candidates[selected]

    # --------------------------------------------------------
    # Main fit
    # --------------------------------------------------------

    def fit(
        self,
        x_train,
        y_train,
        feature_names
    ):

        print()
        print("=" * 70)
        print("DiversityAwareTemporalSelector")
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

        # --------------------------------------------
        # Sample analysis data
        # --------------------------------------------

        x_analysis, y_analysis = (
            self._sample_training_data(
                x_train,
                y_train
            )
        )

        print(
            f"Analysis rows : {len(y_analysis)}"
        )

        # --------------------------------------------
        # Feature scores
        # --------------------------------------------

        lag_score = self._lag_relevance(
            x_analysis,
            y_analysis
        )

        mi_score = self._mutual_information(
            x_analysis,
            y_analysis
        )

        spectral_score = (
            self._spectral_relevance(
                x_analysis,
                y_analysis
            )
        )

        # --------------------------------------------
        # Combined relevance
        # --------------------------------------------

        relevance = (
            W_LAG * lag_score
            + W_MI * mi_score
            + W_SPECTRAL * spectral_score
        )

        relevance = minmax_normalize(
            relevance
        )

        # --------------------------------------------
        # Candidate pool
        # --------------------------------------------

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

        # --------------------------------------------
        # Temporal-frequency signature
        # --------------------------------------------

        signatures = self._build_signature(
            x_analysis
        )

        # --------------------------------------------
        # Diversity-aware selection
        # --------------------------------------------

        selected = self._diversity_selection(
            x_analysis,
            candidates,
            relevance,
            signatures
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

    print("=" * 70)
    print("Creating Traffic v4")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Original shape: {df.shape}"
    )

    if TARGET not in df.columns:

        raise ValueError(
            f"Target '{TARGET}' "
            f"not found in dataset."
        )

    if "date" not in df.columns:

        raise ValueError(
            "Dataset must contain "
            "'date' column."
        )

    # --------------------------------------------------------
    # Identify variables
    # --------------------------------------------------------

    feature_names = [
        c
        for c in df.columns
        if c not in ["date", TARGET]
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

    # Fill missing values using TRAIN ONLY.
    # This avoids using future/test information.
    #
    # Train boundary follows the standard iTransformer
    # Traffic split.
    # 70% / 10% / 20%
    # --------------------------------------------------------

    n_rows = len(df)

    train_end = int(
        n_rows * 0.70
    )

    train_numeric = numeric_df.iloc[
        :train_end
    ].copy()

    train_medians = (
        train_numeric.median()
    )

    numeric_df = numeric_df.fillna(
        train_medians
    )

    # --------------------------------------------------------
    # Extract train data
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

    valid_feature_names = [
        feature_names[i]
        for i in range(
            len(feature_names)
        )
        if valid_mask[i]
    ]

    x_train = x_train[
        :, valid_mask
    ]

    feature_names = (
        valid_feature_names
    )

    print(
        f"Valid variables: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Selector
    # --------------------------------------------------------

    selector = (
        DiversityAwareTemporalSelector(
            n_selected=min(
                N_SELECTED,
                len(feature_names)
            ),
            candidate_size=min(
                CANDIDATE_SIZE,
                len(feature_names)
            ),
            analysis_samples=ANALYSIS_SAMPLES,
            random_state=RANDOM_STATE,
        )
    )

    selected_features = selector.fit(
        x_train,
        y_train,
        feature_names
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    selected_features = list(
        dict.fromkeys(
            selected_features
        )
    )

    print()
    print("=" * 70)
    print("Selection completed")
    print("=" * 70)

    print(
        f"Selected variables: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------
    # Create v4 dataset
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We DO NOT replace values with processed values.
    #
    # We simply select the original columns.
    #
    # Therefore:
    #
    # traffic.csv
    #      ↓
    # select variables
    #      ↓
    # traffic_v4.csv
    #
    # All rows remain unchanged.
    #
    # Only the number of variables decreases.
    # --------------------------------------------------------

    output_columns = (
        ["date"]
        + selected_features
        + [TARGET]
    )

    df_v4 = df[
        output_columns
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    df_v4.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save selection metadata
    # --------------------------------------------------------

    metadata_file = (
        OUTPUT_FILE
        .replace(
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
            "Traffic v4 variable selection\n"
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
            f"Training rows used: "
            f"{train_end}\n"
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
    print("=" * 70)
    print("Output")
    print("=" * 70)

    print(
        f"Saved dataset:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Saved metadata:"
    )

    print(
        metadata_file
    )

    print()

    print(
        f"Original shape : "
        f"{df.shape}"
    )

    print(
        f"New shape      : "
        f"{df_v4.shape}"
    )

    print()
    print(
        "First columns:"
    )

    print(
        df_v4.columns[:10].tolist()
    )

    print()
    print(
        "Last columns:"
    )

    print(
        df_v4.columns[-10:].tolist()
    )

    print()
    print(
        "Done."
    )


if __name__ == "__main__":
    main()