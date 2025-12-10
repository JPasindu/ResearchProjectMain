import numpy as np
from scipy.stats import nbinom
from scipy.optimize import minimize


class ZkINB:
    """
    Zero and k-Inflated Negative Binomial Distribution
    Based strictly on "On Zero and k Inflated Negative Binomial for Count Data with Inflated Frequencies"
    """

    def __init__(self, phi=None, psi=None, mu=None, alpha=None, k=None):
        self.phi = phi    # inflation at zero
        self.psi = psi    # inflation at k
        self.mu = mu      # NB mean
        self.alpha = alpha  # NB dispersion
        self.k = k        # inflated count value

    # -----------------------------
    # NB PMF (mean μ, dispersion α)
    # -----------------------------
    def _nb_pmf(self, y):
        """
        NB parameterization:
        mean = μ
        variance = μ + α μ^2  → α >= 0
        
        Convert to (r, p):
            r = 1/alpha
            p = r / (r + μ)
        """
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0 for NB distribution")

        r = 1 / self.alpha
        p = r / (r + self.mu)

        return nbinom.pmf(y, r, p)

    # -----------------------------
    # ZkINB PMF (Equation 7)
    # -----------------------------
    def pmf(self, y):
        y = np.asarray(y)

        nb_p = self._nb_pmf(y)

        out = np.zeros_like(y, dtype=float)

        # Case y = 0
        mask0 = (y == 0)
        out[mask0] = self.phi + (1 - self.phi - self.psi) * self._nb_pmf(0)

        # Case y = k
        maskk = (y == self.k)
        out[maskk] = self.psi + (1 - self.phi - self.psi) * self._nb_pmf(self.k)

        # Other values
        mask_other = ~(mask0 | maskk)
        out[mask_other] = (1 - self.phi - self.psi) * nb_p[mask_other]

        return out

    # -----------------------------
    # Log-likelihood
    # -----------------------------
    def loglik(self, data):
        p = self.pmf(data)
        return np.sum(np.log(p + 1e-12))  # numerical stability

    # -----------------------------
    # Fit model via MLE
    # -----------------------------
    def fit(self, data, k, init_params=None):
        """
        Fit ZkINB to data using MLE.
        Parameters to estimate: phi, psi, mu, alpha
        k must be supplied (inflated count).
        """

        self.k = k
        data = np.asarray(data)

        if init_params is None:
            # crude starting values
            init_params = np.array([0.1, 0.1, np.mean(data), 0.1])

        # Transform parameters to ensure constraints:
        # phi ∈ (0,1), psi ∈ (0,1), phi+psi < 1
        # mu > 0, alpha > 0
        def transform(params):
            phi = 1 / (1 + np.exp(-params[0]))               # sigmoid
            psi = (1 - phi) * (1 / (1 + np.exp(-params[1]))) # ensures phi+psi < 1
            mu = np.exp(params[2])
            alpha = np.exp(params[3])
            return phi, psi, mu, alpha

        def neg_loglik(raw_params):
            phi, psi, mu, alpha = transform(raw_params)
            self.phi, self.psi, self.mu, self.alpha = phi, psi, mu, alpha
            return -self.loglik(data)

        res = minimize(neg_loglik, init_params, method="L-BFGS-B")

        # Final parameters
        self.phi, self.psi, self.mu, self.alpha = transform(res.x)

        return res

    # -----------------------------
    # Mean & variance (Propositions 3.5)
    # -----------------------------
    def mean(self):
        return (1 - self.phi - self.psi) * self.mu + self.psi * self.k

    def variance(self):
        """Equation (15)"""
        term1 = (1 - self.phi - self.psi) * self.mu * (1 + self.alpha * self.mu)
        term2 = self.psi * (self.k - self.mu)**2
        term3 = self.phi * (self.mu**2)
        return term1 + term2 + term3

    # -----------------------------
    # Generate random samples
    # -----------------------------
    def rvs(self, n):
        u = np.random.rand(n)

        out = np.zeros(n, dtype=int)

        # Process 1: structural zeros
        mask0 = u < self.phi
        out[mask0] = 0

        # Process 2: structural k
        maskk = (u >= self.phi) & (u < self.phi + self.psi)
        out[maskk] = self.k

        # Process 3: NB-generated values
        mask_nb = ~(mask0 | maskk)
        out[mask_nb] = nbinom.rvs(1/self.alpha, (1/self.alpha)/(self.mu + 1/self.alpha), size=np.sum(mask_nb))

        return out
