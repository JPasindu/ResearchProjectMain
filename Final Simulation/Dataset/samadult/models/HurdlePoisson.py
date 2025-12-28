import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import gammaln

# problem with ll and the method(no EM?)

# ==============================================================
# Zero-Truncated Poisson (ZTP) MLE
# ==============================================================

class ZeroKTruncatedPoisson(GenericLikelihoodModel):
    def nloglikeobs(self, params):
        X = self.exog
        y = self.endog
        
        mu = np.exp(X @ params)
        k = self.k  # Pass k to the model
        
        # Poisson probabilities
        log_p0 = -mu  # P(Y=0)
        log_pk = k * np.log(mu) - mu - gammaln(k + 1)  # P(Y=k)
        
        # Log-likelihood for {0,k}-truncated Poisson
        # = log(Poisson) - log(1 - P(0) - P(k))

        eps = 1e-12
        Z = 1 - np.exp(log_p0) - np.exp(log_pk)
        Z = np.clip(Z, eps, None)

        ll = (
            y * np.log(mu)
            - mu
            - gammaln(y + 1)
            - np.log(Z)
        )

        return -ll


# ==============================================================
# Full Hurdle Model: Logit + ZTP
# ==============================================================


class ZeroKInflatedPoisson:
    """
    Zero & K-Inflated Poisson using single multinomial logit
    """
    
    def __init__(self, k):
        self.k = k
        self.infl_model = None  # Multinomial for 0/k/other
        self.ztp_model = None   # Zero-truncated Poisson for "other"
    
    def fit(self, X, y):
        y = np.array(y)
        
        # Create categories: 0=zero, 1=k, 2=other
        categories = np.zeros(len(y))
        categories[y == 0] = 0
        categories[y == self.k] = 1
        categories[(y > 0) & (y != self.k)] = 2
        
        # Multinomial logit (0/k/other)
        self.infl_model = sm.MNLogit(categories, X).fit(disp=False)
        
        # Truncated Poisson for "other" category
        mask_other = (categories == 2)
        if mask_other.sum() > 0:
            X_other = X[mask_other]
            y_other = y[mask_other]
            self.ztp_model = ZeroKTruncatedPoisson(y_other, X_other, k=self.k).fit(disp=False)        
        return self
    
    def predict_probs(self, X):
        probs = self.infl_model.predict(X)  # Returns matrix [P(0), P(k), P(other)]
        return probs.iloc[:, 0], probs.iloc[:, 1], probs.iloc[:, 2]
    
    def predict_mean(self, X):
        p_zero, p_k, p_other = self.predict_probs(X)
        
        if self.ztp_model:
            mu = np.exp(X @ self.ztp_model.params)
            p0 = np.exp(-mu)
            pk = np.exp(self.k * np.log(mu) - mu - gammaln(self.k + 1))

            mean_other = (mu - self.k * pk) / (1 - p0 - pk)

        else:
            mean_other = 0
        
        return self.k * p_k + mean_other * p_other

    def loglikelihood(self, X, y):
        probs = self.infl_model.predict(X)
        p0 = probs.iloc[:, 0].values
        pk = probs.iloc[:, 1].values
        po = probs.iloc[:, 2].values

        mu = np.exp(X @ self.ztp_model.params)

        log_p0_pois = -mu
        log_pk_pois = self.k * np.log(mu) - mu - gammaln(self.k + 1)

        ll = np.zeros_like(y, dtype=float)

        # y = 0
        mask0 = (y == 0)
        ll[mask0] = np.log(p0[mask0])

        # y = k
        maskk = (y == self.k)
        ll[maskk] = np.log(pk[maskk])

        # y ∉ {0,k}
        masko = ~(mask0 | maskk)
        ll[masko] = (
            np.log(po[masko])
            + y[masko] * np.log(mu[masko])
            - mu[masko]
            - gammaln(y[masko] + 1)
            - np.log(1 - np.exp(log_p0_pois[masko]) - np.exp(log_pk_pois[masko]))
        )

        return ll.sum()
