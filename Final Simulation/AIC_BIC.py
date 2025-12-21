import numpy as np

def calculate_aic_bic(n, log_likelihood, k):
    """
    n: number of observations
    log_likelihood: maximized log-likelihood of the model
    k: number of parameters
    """
    aic = 2*k - 2*log_likelihood
    bic = k * np.log(n) - 2*log_likelihood
    return aic, bic
