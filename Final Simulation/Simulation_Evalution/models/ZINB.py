import numpy as np
from scipy.special import expit, gammaln
from numpy.linalg import solve

# problem with alpha

# -------------------------------------------------------------
# IRLS for Negative Binomial regression (NB2)
# -------------------------------------------------------------
def nb_irls(X, y, w, alpha, max_iter=50):
    beta = np.zeros(X.shape[1])

    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)

        # Variance: mu + alpha*mu^2
        var = mu + alpha * mu**2

        z = eta + (y - mu) / mu
        W = w * mu**2 / var

        WX = X * W[:, None]
        beta_new = solve(WX.T @ X, WX.T @ z)

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            break
        beta = beta_new

    return beta


# -------------------------------------------------------------
# IRLS for logistic regression (unchanged)
# -------------------------------------------------------------
def logit_irls(X, y, w, max_iter=50):
    gamma = np.zeros(X.shape[1])

    for _ in range(max_iter):
        eta = X @ gamma
        eta = np.clip(eta, -20, 20)
        p = expit(eta)

        z = eta + (y - p) / (p * (1 - p))
        W = w * p * (1 - p)

        WX = X * W[:, None]
        gamma_new = solve(WX.T @ X, WX.T @ z)

        if np.max(np.abs(gamma_new - gamma)) < 1e-6:
            break
        gamma = gamma_new

    return gamma


# ------------------------------------------------------------------
# ZINB EM Algorithm (Lambert-style extension)
# ------------------------------------------------------------------
def ZINB_EM(y, B, G, alpha, max_iter=100, tol=1e-5):
    n = len(y)

    beta = np.zeros(B.shape[1])
    gamma = np.zeros(G.shape[1])

    for iteration in range(max_iter):

        # ------------------ E-step ------------------
        mu = np.exp(B @ beta)
        p = expit(G @ gamma)

        # NB zero probability
        nb_zero = (1 + alpha * mu) ** (-1 / alpha)

        Z = np.zeros(n)
        mask0 = (y == 0)
        Z[mask0] = p[mask0] / (p[mask0] + (1 - p[mask0]) * nb_zero[mask0])

        # ------------------ M-step ------------------
        # NB regression for beta
        w_nb = 1 - Z
        beta_new = nb_irls(B, y, w_nb, alpha)

        # Logistic regression for gamma
        gamma_new = logit_irls(G, Z, np.ones(n))

        # Convergence check
        if (np.max(np.abs(beta_new - beta)) < tol and
            np.max(np.abs(gamma_new - gamma)) < tol):
            beta, gamma = beta_new, gamma_new
            break

        beta, gamma = beta_new, gamma_new
        
    final_ll=final_loglik(y, B, G, beta, gamma, alpha)

    return beta, gamma, final_ll


# -------------------------------------------------------------
# PREDICTION FUNCTIONS
# -------------------------------------------------------------
def predict_mu(B_new, beta):
    return np.exp(B_new @ beta)


def predict_p(G_new, gamma):
    return expit(G_new @ gamma)


def predict_mean(B_new, G_new, beta, gamma):
    mu = predict_mu(B_new, beta)
    p = predict_p(G_new, gamma)
    return (1 - p) * mu


def predict_prob_zero(B_new, G_new, beta, gamma, alpha):
    mu = predict_mu(B_new, beta)
    p = predict_p(G_new, gamma)
    nb_zero = (1 + alpha * mu) ** (-1 / alpha)
    return p + (1 - p) * nb_zero


def predict_pmf(k, B_new, G_new, beta, gamma, alpha):
    mu = predict_mu(B_new, beta)
    p = predict_p(G_new, gamma)

    r = 1 / alpha
    prob = r / (r + mu)

    if k == 0:
        return p + (1 - p) * prob**r
    else:
        nb_pmf = (
            gammaln(k + r) - gammaln(r) - gammaln(k + 1)
            + r * np.log(prob) + k * np.log(1 - prob)
        )
        return (1 - p) * np.exp(nb_pmf)


def final_loglik(y, B, G, beta, gamma, alpha):
    """
    Compute the final log-likelihood of the ZINB model.
    
    Parameters
    ----------
    y : array_like
        Observed counts (n,)
    B : array_like
        Design matrix for NB component (n, p)
    G : array_like
        Design matrix for zero-inflation component (n, q)
    beta : array_like
        Estimated coefficients for NB component (p,)
    gamma : array_like
        Estimated coefficients for zero-inflation component (q,)
    alpha : float
        Estimated dispersion parameter
    
    Returns
    -------
    loglik : float
        Final log-likelihood value
    """
    n = len(y)
    
    # Compute NB mean and zero-inflation probability
    mu = np.exp(B @ beta)
    p = expit(G @ gamma)
    
    # Compute NB part parameters
    r = 1 / alpha
    prob = r / (r + mu)
    
    # Compute NB zero probabilities
    nb_zero = prob ** r
    
    # Initialize log-likelihood array
    loglik = np.zeros(n)
    
    # Mask for zeros and non-zeros
    zero_mask = (y == 0)
    nonzero_mask = ~zero_mask
    
    # For zero counts
    if np.any(zero_mask):
        zinb_zero = p[zero_mask] + (1 - p[zero_mask]) * nb_zero[zero_mask]
        zinb_zero = np.clip(zinb_zero, 1e-15, 1 - 1e-15)
        loglik[zero_mask] = np.log(zinb_zero)
    
    # For non-zero counts
    if np.any(nonzero_mask):
        y_nz = y[nonzero_mask]
        prob_nz = prob[nonzero_mask]
        p_nz = p[nonzero_mask]
        
        # NB log-PMF for non-zero counts
        nb_logpmf = (
            gammaln(y_nz + r) - gammaln(r) - gammaln(y_nz + 1)
            + r * np.log(prob_nz) + y_nz * np.log(1 - prob_nz)
        )
        
        # ZINB PMF for non-zero counts
        zinb_pmf = (1 - p_nz) * np.exp(nb_logpmf)
        zinb_pmf = np.clip(zinb_pmf, 1e-15, 1 - 1e-15)
        
        loglik[nonzero_mask] = np.log(zinb_pmf)
    
    return np.sum(loglik)