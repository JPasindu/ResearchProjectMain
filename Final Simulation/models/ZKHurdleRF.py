import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor


class ZKHurdleRF:
    """
    Zero-and-k Hurdle Model with Random Forest for positive counts

    Stage 1: Multinomial logit for {0, k, other}
    Stage 2: RandomForestRegressor (Poisson loss) for counts conditional on 'other'
    """

    def __init__(
        self,
        k,
        n_estimators=300,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ):
        self.k = k
        self.hurdle_model = None
        self.rf_model = None

        self.rf_params = dict(
            n_estimators=n_estimators,
            criterion="poisson",
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(self, X, y):
        y = np.asarray(y)

        # ---------- Stage 1: Hurdle ----------
        categories = np.zeros(len(y), dtype=int)
        categories[y == self.k] = 1
        categories[(y != 0) & (y != self.k)] = 2

        self.hurdle_model = sm.MNLogit(categories, X).fit(disp=False)

        # ---------- Stage 2: RF on "other" ----------
        mask_other = (categories == 2)

        X_other = X[mask_other]
        y_other = y[mask_other]

        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.rf_model.fit(X_other, y_other)

        return self

    # --------------------------------------------------
    # Probabilities
    # --------------------------------------------------
    def predict_probs(self, X):
        """
        Returns p0, pk, po
        """
        probs = self.hurdle_model.predict(X)
        p0 = probs.iloc[:, 0].values
        pk = probs.iloc[:, 1].values
        po = probs.iloc[:, 2].values
        return p0, pk, po

    # --------------------------------------------------
    # Mean prediction
    # --------------------------------------------------
    def predict_mean(self, X):
        """
        E[Y | X]
        """
        _, pk, po = self.predict_probs(X)

        mu_other = self.rf_model.predict(X)
        mu_other = np.maximum(mu_other, 0.0)  # safety

        return self.k * pk + po * mu_other
