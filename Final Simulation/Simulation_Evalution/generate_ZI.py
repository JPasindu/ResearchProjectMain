import numpy as np
import pandas as pd
from scipy.stats import nbinom, bernoulli, norm
from scipy.special import expit, gammaln
from scipy.stats import shapiro
from statsmodels.base.model import GenericLikelihoodModel

def generate_ZI(n, k, beta0, beta1, gamma0, gamma1, alpha0, alpha1, r, cov_type="binary"):
    
    # Covariate generation (matching paper)
    if cov_type == "binary":
        x = bernoulli.rvs(0.5, size=n)
    else:
        x = np.random.normal(0, 1, n)
        x = (x-np.mean(x))/np.var(x)
    
    # Logistic zero-probabilities
    p0 = expit(beta0 + beta1 * x)
    pk = expit(gamma0 + gamma1 * x)
    # Log-linear mean
    mu = np.exp(alpha0 + alpha1 * x)
    
    y = np.zeros(n, dtype=int)
    w = np.zeros(n, dtype=int)

    for i in range(n):
        z = np.random.rand()

        if z < p0[i]:
            y[i] = 0
            w[i] = 0
        elif z < p0[i] + pk[i]:  # Direct probability sum
            y[i] = k
            w[i] = k

        else:
            # Generate from standard NB (no truncation)
            y[i] = nbinom.rvs(r, r / (r + mu[i])) #What r Controls:
            # r → ∞: Variance = μ (converges to Poisson)
            # r → 0: Variance → ∞ (highly overdispersed)
            # typical values: r between 0.1 and 10 for count data
            w[i] = k+1
                    
    return pd.DataFrame({"y": y, "x": x, "w":w})