import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import gammaln
from scipy.special import logsumexp
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
        ll = (
            y * np.log(mu)
            - mu
            - gammaln(y + 1)
            - np.log(1 - np.exp(log_p0) - np.exp(log_pk))
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
        X = sm.add_constant(X)
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
        X = sm.add_constant(X)
        probs = self.infl_model.predict(X)  # Returns matrix [P(0), P(k), P(other)]
        return probs[:, 0], probs[:, 1], probs[:, 2]
    
    def predict_mean(self, X):
        p_zero, p_k, p_other = self.predict_probs(X)
        X = sm.add_constant(X)
        
        if self.ztp_model:
            mu = np.exp(X @ self.ztp_model.params)
            mean_other = mu / (1 - np.exp(-mu))
        else:
            mean_other = 0
        
        return self.k * p_k + mean_other * p_other