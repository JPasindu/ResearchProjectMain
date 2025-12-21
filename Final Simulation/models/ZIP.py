import numpy as np
from scipy.special import expit, gammaln
from numpy.linalg import solve

# _Lambert_1992

# -------------------------------------------------------------
# IRLS for Poisson regression
# -------------------------------------------------------------
def poisson_irls(X, y, w, max_iter=50):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)

        z = eta + (y - mu) / mu
        W = w * mu  # IRLS weight

        WX = X * W[:, None]
        beta_new = solve(WX.T @ X, WX.T @ z)

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            break
        beta = beta_new
    return beta


# -------------------------------------------------------------
# IRLS for logistic regression
# -------------------------------------------------------------
def logit_irls(X, y, w, max_iter=50):
    gamma = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ gamma
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
# ZIP EM Algorithm (Lambert, 1992)
# ------------------------------------------------------------------
def ZIP_EM(y, B, G, max_iter=100, tol=1e-5):
    n = len(y)

    # initialize using Poisson MLE for β (simple)
    beta = np.zeros(B.shape[1])
    gamma = np.zeros(G.shape[1])

    for iteration in range(max_iter):

        # E-step -----------------------------------------------------
        lambda_i = np.exp(B @ beta)
        p_i = expit(G @ gamma)

        # Posterior probability Z_i = P(“perfect zero” | y_i==0)
        Z = np.zeros(n)
        mask0 = (y == 0)
        Z[mask0] = p_i[mask0] / (p_i[mask0] + (1 - p_i[mask0]) * np.exp(-lambda_i[mask0]))

        # M-step for β (Poisson) ------------------------------------
        w_poisson = 1 - Z
        beta_new = poisson_irls(B, y, w_poisson)

        # M-step for γ (Logistic) -----------------------------------
        y_logit = Z          # response is Z_i
        w_logit = np.ones(n)
        gamma_new = logit_irls(G, y_logit, w_logit)

        # Check convergence
        if (np.max(np.abs(beta_new - beta)) < tol) and \
           (np.max(np.abs(gamma_new - gamma)) < tol):
            beta, gamma = beta_new, gamma_new
            final_ll=final_loglik(beta, gamma, B, G, y)
            break

        
        beta, gamma = beta_new, gamma_new

    return beta, gamma, final_ll


# -------------------------------------------------------------
# PREDICTION FUNCTIONS
# -------------------------------------------------------------
def predict_lambda(B_new, beta):
    """Predicted Poisson mean λ."""
    return np.exp(B_new @ beta)


def predict_p(G_new, gamma):
    """Predicted zero-inflation probability p."""
    return expit(G_new @ gamma)


def predict_mean(B_new, G_new, beta, gamma):
    """ZIP Expected value: E[Y] = (1-p)*λ."""
    lam = predict_lambda(B_new, beta)
    p = predict_p(G_new, gamma)
    return (1 - p) * lam


def predict_prob_zero(B_new, G_new, beta, gamma):
    """ZIP probability P(Y=0)."""
    lam = predict_lambda(B_new, beta)
    p = predict_p(G_new, gamma)
    return p + (1 - p) * np.exp(-lam)


def predict_pmf(k, B_new, G_new, beta, gamma):
    """
    ZIP probability mass function for integer k.
    Returns vector of probabilities for each observation.
    """
    lam = predict_lambda(B_new, beta)
    p = predict_p(G_new, gamma)

    if k == 0:
        return p + (1 - p) * np.exp(-lam)
    else:
        return (1 - p) * np.exp(-lam) * (lam**k) / np.exp(gammaln(k + 1))


def final_loglik(beta, gamma, B, G, y):
    """
    Calculate the log-likelihood for a Zero-Inflated Poisson (ZIP) model.
    
    Parameters:
    -----------
    beta : array_like
        Estimated coefficients for the Poisson component
    gamma : array_like
        Estimated coefficients for the logistic (zero-inflation) component
    B : array_like
        Design matrix for Poisson component (covariates for λ)
    G : array_like
        Design matrix for logistic component (covariates for p)
    y : array_like
        Observed count responses
    
    Returns:
    --------
    float
        Log-likelihood of the ZIP model given the parameters
    """
    lambda_i = np.exp(B @ beta)
    p_i = expit(G @ gamma)
    
    # Poisson log-probabilities for all y
    poisson_log_prob = -lambda_i + y * np.log(lambda_i) - gammaln(y + 1)
    
    # Zero-inflated log-likelihood
    loglik_zero = np.log(p_i + (1 - p_i) * np.exp(-lambda_i))
    loglik_pos = np.log(1 - p_i) + poisson_log_prob
    
    # Combine using masks
    mask_zero = (y == 0)
    loglik = np.sum(loglik_zero[mask_zero]) + np.sum(loglik_pos[~mask_zero])
    
    return loglik
