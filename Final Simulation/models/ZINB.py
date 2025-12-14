import numpy as np
from scipy.special import expit, gammaln
from numpy.linalg import solve

# -------------------------------------------------------------
# IRLS for Negative Binomial regression (NB2)
# -------------------------------------------------------------
def nb_irls(X, y, w, alpha, max_iter=50):
    beta = np.zeros(X.shape[1])

    for _ in range(max_iter):
        eta = X @ beta
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

    return beta, gamma


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
