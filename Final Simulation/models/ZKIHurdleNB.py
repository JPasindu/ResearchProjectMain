import numpy as np
import statsmodels.api as sm
from scipy.special import gammaln


class ZKHurdleNB:
    """
    Zero-and-k Hurdle Negative Binomial Model (Option A)

    Stage 1: Multinomial logit for {0, k, other}
    Stage 2: NB2 (variance = mu + alpha * mu^2) for counts conditional on 'other'
    """

    def __init__(self, k, alpha):
        self.k = k
        self.hurdle_model = None
        self.nb_model = None
        self.alpha = alpha # dispersion

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

        # ---------- Stage 2: NB on "other" ----------
        mask_other = (categories == 2)
        X_other = X[mask_other]
        y_other = y[mask_other]

        self.nb_model = sm.GLM(
            y_other,
            X_other,
            family=sm.families.NegativeBinomial(alpha=self.alpha)
        ).fit()

        self.alpha = self.nb_model.scale  # NB2 dispersion

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
        """
        E[Y | X]
        """
        p0, pk, po = self.predict_probs(X)
        mu = np.exp(X @ self.nb_model.params)

        return self.k * pk + po * mu

    # --------------------------------------------------
    # NB log PMF (NB2 parameterization)
    # --------------------------------------------------
    def _nb_logpmf(self, y, mu, alpha):
        """
        NB2 log PMF:
        Var(Y) = mu + alpha * mu^2
        """
        r = 1.0 / alpha
        p = r / (r + mu)

        return (
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1)
            + r * np.log(p)
            + y * np.log(1 - p)
        )

    # --------------------------------------------------
    # Full log-likelihood
    # --------------------------------------------------
    def loglikelihood(self, X, y):
        y = np.asarray(y)
        p0, pk, po = self.predict_probs(X)
        mu = np.exp(X @ self.nb_model.params)
        alpha = self.alpha

        ll = np.zeros_like(y, dtype=float)

        mask0 = (y == 0)
        maskk = (y == self.k)
        masko = ~(mask0 | maskk)

        ll[mask0] = np.log(p0[mask0])
        ll[maskk] = np.log(pk[maskk])

        ll[masko] = (
            np.log(po[masko])
            + self._nb_logpmf(y[masko], mu[masko], alpha)
        )

        return ll.sum()

    # --------------------------------------------------
    # AIC
    # --------------------------------------------------
    def aic(self, X, y):
        ll = self.loglikelihood(X, y)
        k_params = (
            len(self.hurdle_model.params.flatten())
            + len(self.nb_model.params)
            + 1  # alpha
        )
        return -2 * ll + 2 * k_params
