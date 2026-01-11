import numpy as np
import statsmodels.api as sm
from scipy.special import gammaln

class ZKHurdlePoisson:
    """
    Zero-and-k Hurdle Poisson Model (Option A)

    Stage 1: Multinomial logit for {0, k, other}
    Stage 2: Poisson for counts conditional on 'other'
    """

    def __init__(self, k):
        self.k = k
        self.hurdle_model = None
        self.poisson_model = None

    def fit(self, X, y):
        y = np.asarray(y)

        # ----- Stage 1: Hurdle (0 / k / other) -----
        categories = np.zeros(len(y), dtype=int)
        categories[y == self.k] = 1
        categories[(y != 0) & (y != self.k)] = 2

        self.hurdle_model = sm.MNLogit(categories, X).fit(disp=False)

        # ----- Stage 2: Poisson on "other" only -----
        mask_other = (categories == 2)
        X_other = X[mask_other]
        y_other = y[mask_other]

        self.poisson_model = sm.GLM(
            y_other,
            X_other,
            family=sm.families.Poisson()
        ).fit()

        return self

    # --------------------------------------------------
    # Probabilities
    # --------------------------------------------------
    def predict_probs(self, X):
        """
        Returns p0, pk, po
        """
        probs = self.hurdle_model.predict(X)
        p0 = probs.iloc[:, 0]
        pk = probs.iloc[:, 1]
        po = probs.iloc[:, 2]
        return p0, pk, po

    # --------------------------------------------------
    # Mean prediction
    # --------------------------------------------------
    def predict_mean(self, X):
        p0, pk, po = self.predict_probs(X)
        mu = np.exp(X @ self.poisson_model.params)
        
        return self.k * pk + po * mu

    # --------------------------------------------------
    # Log-likelihood (FULL, CORRECT)
    # --------------------------------------------------
    def loglikelihood(self, X, y):
        y = np.asarray(y)
        p0, pk, po = self.predict_probs(X)
        mu = np.exp(X @ self.poisson_model.params)

        ll = np.zeros_like(y, dtype=float)

        mask0 = (y == 0)
        maskk = (y == self.k)
        masko = ~(mask0 | maskk)

        ll[mask0] = np.log(p0[mask0])
        ll[maskk] = np.log(pk[maskk])

        ll[masko] = (
            np.log(po[masko])
            + y[masko] * np.log(mu[masko])
            - mu[masko]
            - gammaln(y[masko] + 1)
        )

        return ll.sum()

