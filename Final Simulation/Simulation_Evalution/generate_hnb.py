import numpy as np
import pandas as pd
from scipy.stats import nbinom, bernoulli, norm
from scipy.special import expit, gammaln
from scipy.stats import shapiro
from statsmodels.base.model import GenericLikelihoodModel

def generate_hnb(n, k, beta0, beta1, gamma0, gamma1, alpha0, alpha1, r, cov_type="binary"):
    
    # Covariate generation (matching paper)
    if cov_type == "binary":
        x = bernoulli.rvs(0.5, size=n)
    else:
        x = np.random.normal(0, 1, n)
    
    # Logistic zero-probabilities
    p0 = expit(beta0 + beta1 * x)
    pk = expit(gamma0 + gamma1 * x)
    # Log-linear mean
    mu = np.exp(alpha0 + alpha1 * x)
    
    y = np.zeros(n, dtype=int)
    
    j=0

    for i in range(n):
       z = np.random.rand()

       if z < p0[i]:
        y[i] = 0
       elif z < p0[i] + pk[i]:  # Direct probability sum
        y[i] = k
        
       else:
        # Generate from truncated NB (positive, ≠k)
        while True:
            y[i] = nbinom.rvs(r, r / (r + mu[i]))
            j += 1
            if y[i] > 0 and y[i] != k:
                break
                    
    return pd.DataFrame({"y": y, "x": x})
