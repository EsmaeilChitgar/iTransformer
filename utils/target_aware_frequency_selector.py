import os
import numpy as np
import pandas as pd

from scipy.signal import coherence
from sklearn.feature_selection import mutual_info_regression


class TargetAwareFrequencySelector:
    """
    Target-aware variable selection for high-dimensional MTS forecasting.

    Selection is performed ONLY on the training portion.

    Relevance:
        1. Lagged correlation with future target
        2. Mutual information with future target
        3. Frequency-domain coherence with target

    Redundancy:
        Mean absolute correlation with already selected variables.

    The selector returns ORIGINAL column indices.
    """

    def __init__(
        self,
        num_variables=720,
        train_ratio=0.70,
        random_state=2023,

        # lagged relevance
        lags=(1, 6, 12, 24, 48, 96),

        # MI
        mi_samples=12000,

        # frequency
        coherence_nperseg=256,
        coherence_noverlap=128,
        high_frequency_ratio=0.50,

        # relevance weights
        weight_lag=0.40,
        weight_mi=0.35,
        weight_frequency=0.25,

        # redundancy
        redundancy_weight=0.30,
    ):
        self.num_variables = num_variables
        self.train_ratio = train_ratio
        self.random_state = random_state

        self.lags = tuple(lags)
        self.mi_samples = mi_samples

        self.coherence_nperseg = coherence_nperseg
        self.coherence_noverlap = coherence_noverlap
        self.high_frequency_ratio = high_frequency_ratio

        self.weight_lag = weight_lag
        self.weight_mi = weight_mi
        self.weight_frequency = weight_frequency

        self.redundancy_weight = redundancy_weight

        self.indices_ = None
        self.scores_ = None
        self.relevance_ = None

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float64)

        minimum = np.min(x)
        maximum = np.max(x)

        if maximum - minimum < 1e-12:
            return np.zeros_like(x)

        return (x - minimum) / (maximum - minimum)

    def _safe_corr(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if len(x) < 3:
            return 0.0

        sx = np.std(x)
        sy = np.std(y)

        if sx < 1e-12 or sy < 1e-12:
            return 0.0

        value = np.corrcoef(x, y)[0, 1]

        if not np.isfinite(value):
            return 0.0

        return float(value)

    # ---------------------------------------------------------
    # Lagged relevance
    # ---------------------------------------------------------

    def _lagged_correlation_score(self, x, target):
        """
        Measures dependency between past values of x
        and future/current target.

        For lag L:

            x[t-L]  <->  target[t]

        The maximum absolute correlation across lags is used.
        """

        scores = []

        n = len(target)

        for lag in self.lags:

            if lag >= n - 2:
                continue

            x_lagged = x[:-lag]
            y_future = target[lag:]

            corr = self._safe_corr(
                x_lagged,
                y_future
            )

            scores.append(abs(corr))

        if not scores:
            return 0.0

        # We use the strongest predictive lag.
        return float(np.max(scores))

    # ---------------------------------------------------------
    # Mutual information
    # ---------------------------------------------------------

    def _mutual_information_score(self, x, target):
        """
        MI between lagged candidate and future target.

        Multiple lagged versions are evaluated and the maximum
        normalized MI is retained.
        """

        rng = np.random.RandomState(self.random_state)

        scores = []

        n = len(target)

        for lag in self.lags:

            if lag >= n - 2:
                continue

            x_lagged = x[:-lag]
            y_future = target[lag:]

            if len(x_lagged) > self.mi_samples:

                indices = rng.choice(
                    len(x_lagged),
                    size=self.mi_samples,
                    replace=False
                )

                x_sample = x_lagged[indices]
                y_sample = y_future[indices]

            else:
                x_sample = x_lagged
                y_sample = y_future

            # sklearn expects [samples, features]
            X = x_sample.reshape(-1, 1)

            try:
                mi = mutual_info_regression(
                    X,
                    y_sample,
                    random_state=self.random_state
                )[0]

            except Exception:
                mi = 0.0

            if np.isfinite(mi):
                scores.append(float(mi))

        if not scores:
            return 0.0

        return float(np.max(scores))

    # ---------------------------------------------------------
    # Frequency-domain relevance
    # ---------------------------------------------------------

    def _frequency_coherence_score(self, x, target):
        """
        Compute average high-frequency coherence between x and OT.

        Coherence:

            Cxy(f) = |Pxy(f)|^2 / (Pxx(f) Pyy(f))

        The score focuses on the high-frequency portion because
        Traffic contains substantial high-frequency dynamics.
        """

        nperseg = min(
            self.coherence_nperseg,
            len(target)
        )

        if nperseg < 16:
            return 0.0

        noverlap = min(
            self.coherence_noverlap,
            nperseg // 2
        )

        try:

            frequencies, coh = coherence(
                x,
                target,
                fs=1.0,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant"
            )

        except Exception:
            return 0.0

        if len(coh) < 3:
            return 0.0

        # Ignore DC.
        frequencies = frequencies[1:]
        coh = coh[1:]

        if len(coh) == 0:
            return 0.0

        cutoff = int(
            len(coh) * self.high_frequency_ratio
        )

        cutoff = max(
            0,
            min(cutoff, len(coh) - 1)
        )

        high_frequency_coherence = coh[cutoff:]

        if len(high_frequency_coherence) == 0:
            return 0.0

        score = np.mean(
            np.clip(
                high_frequency_coherence,
                0.0,
                1.0
            )
        )

        return float(score)

    # ---------------------------------------------------------
    # Relevance
    # ---------------------------------------------------------

    def _calculate_relevance(self, X, target):

        n_variables = X.shape[1]

        lag_scores = np.zeros(n_variables)
        mi_scores = np.zeros(n_variables)
        freq_scores = np.zeros(n_variables)

        print("Calculating lagged relevance...")

        for i in range(n_variables):

            lag_scores[i] = (
                self._lagged_correlation_score(
                    X[:, i],
                    target
                )
            )

            if (i + 1) % 50 == 0:
                print(
                    f"  Lag relevance: "
                    f"{i + 1}/{n_variables}"
                )

        print("Calculating mutual information...")

        for i in range(n_variables):

            mi_scores[i] = (
                self._mutual_information_score(
                    X[:, i],
                    target
                )
            )

            if (i + 1) % 50 == 0:
                print(
                    f"  MI: "
                    f"{i + 1}/{n_variables}"
                )

        print("Calculating frequency coherence...")

        for i in range(n_variables):

            freq_scores[i] = (
                self._frequency_coherence_score(
                    X[:, i],
                    target
                )
            )

            if (i + 1) % 50 == 0:
                print(
                    f"  Frequency: "
                    f"{i + 1}/{n_variables}"
                )

        # Normalize each criterion separately.
        lag_norm = self._normalize(lag_scores)
        mi_norm = self._normalize(mi_scores)
        freq_norm = self._normalize(freq_scores)

        relevance = (
            self.weight_lag * lag_norm
            + self.weight_mi * mi_norm
            + self.weight_frequency * freq_norm
        )

        return (
            relevance,
            lag_norm,
            mi_norm,
            freq_norm
        )

    # ---------------------------------------------------------
    # Redundancy
    # ---------------------------------------------------------

    def _calculate_redundancy(
        self,
        candidate,
        selected,
        X
    ):
        """
        Mean absolute correlation between candidate and
        already selected variables.
        """

        if not selected:
            return 0.0

        correlations = []

        x = X[:, candidate]

        for selected_index in selected:

            y = X[:, selected_index]

            corr = self._safe_corr(x, y)

            correlations.append(
                abs(corr)
            )

        if not correlations:
            return 0.0

        return float(
            np.mean(correlations)
        )

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    def fit(self, X, target):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        target = np.asarray(
            target,
            dtype=np.float64
        ).reshape(-1)

        if X.ndim != 2:
            raise ValueError(
                "X must have shape [T, N]"
            )

        if len(X) != len(target):
            raise ValueError(
                "X and target must have the same number of rows"
            )

        # -----------------------------------------------------
        # IMPORTANT:
        # Selection uses TRAINING data only.
        # -----------------------------------------------------

        train_end = int(
            len(X) * self.train_ratio
        )

        X_train = X[:train_end]
        y_train = target[:train_end]

        print()
        print("=" * 70)
        print("TargetAwareFrequencySelector")
        print("=" * 70)
        print(
            f"Total rows:   {len(X)}"
        )
        print(
            f"Train rows:   {len(X_train)}"
        )
        print(
            f"Variables:    {X.shape[1]}"
        )
        print(
            f"Requested:    {self.num_variables}"
        )
        print("=" * 70)
        print()

        relevance, lag, mi, freq = (
            self._calculate_relevance(
                X_train,
                y_train
            )
        )

        self.relevance_ = relevance

        n_variables = X.shape[1]

        k = min(
            self.num_variables,
            n_variables
        )

        # -----------------------------------------------------
        # Greedy mRMR-style selection.
        #
        # score(candidate | S)
        #
        # = relevance(candidate)
        #   - lambda * redundancy(candidate, S)
        # -----------------------------------------------------

        selected = []

        remaining = set(
            range(n_variables)
        )

        print()
        print("Greedy relevance/redundancy selection...")
        print()

        for step in range(k):

            best_candidate = None
            best_score = -np.inf

            for candidate in remaining:

                redundancy = (
                    self._calculate_redundancy(
                        candidate,
                        selected,
                        X_train
                    )
                )

                score = (
                    relevance[candidate]
                    - self.redundancy_weight
                    * redundancy
                )

                if score > best_score:

                    best_score = score
                    best_candidate = candidate

            selected.append(
                best_candidate
            )

            remaining.remove(
                best_candidate
            )

            if (
                (step + 1) % 25 == 0
                or step == 0
                or step + 1 == k
            ):

                print(
                    f"Selected "
                    f"{step + 1}/{k} "
                    f"| variable={best_candidate} "
                    f"| score={best_score:.6f}"
                )

        self.indices_ = np.asarray(
            selected,
            dtype=np.int64
        )

        # Preserve original column order.
        self.indices_ = np.sort(
            self.indices_
        )

        self.scores_ = {
            "relevance": relevance,
            "lag": lag,
            "mi": mi,
            "frequency": freq,
        }

        return self

    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------

    def save(self, path):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        np.savez(
            path,
            indices=self.indices_,
            relevance=self.relevance_,
            lag=self.scores_["lag"],
            mi=self.scores_["mi"],
            frequency=self.scores_["frequency"],
        )

    def load(self, path):

        data = np.load(path)

        self.indices_ = data["indices"]

        self.relevance_ = data["relevance"]

        self.scores_ = {
            "lag": data["lag"],
            "mi": data["mi"],
            "frequency": data["frequency"],
        }

        return self

    def get_indices(self):

        if self.indices_ is None:
            raise RuntimeError(
                "Selector has not been fitted."
            )

        return self.indices_