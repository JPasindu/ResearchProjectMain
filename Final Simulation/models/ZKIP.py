import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

class ZKIP_EM:
    """
    Zero- and k-Inflated Poisson Regression Model
    Based on Arora and Chaganty (2021)
    """
    
    def __init__(self, k_inflated=1, max_iter=1000, tol=1e-6):
        self.k_inflated = k_inflated
        self.max_iter = max_iter
        self.tol = tol
        self.is_fitted = False
        self.pi1=0
        self.pi2=0
        self.pi3=0
        
    def _compute_poisson_prob(self, y, lambd):
        """Compute Poisson probability P(Y=y|lambda)"""
        return poisson.pmf(y, lambd)
    
    def _e_step(self, y, X, beta, gamma, delta):
        """E-step: Compute expected values of latent variables"""
        n = len(y)
        lambd = np.exp(X @ beta)
        
        p0 = self._compute_poisson_prob(0, lambd)
        pk = self._compute_poisson_prob(self.k_inflated, lambd)
        
        # Compute pi values from gamma, delta
        pi3 = 1 / (1 + np.exp(gamma) + np.exp(delta))
        pi1 = np.exp(gamma) * pi3
        pi2 = np.exp(delta) * pi3
        
        # Initialize expected values
        z1_hat = np.zeros(n)
        z2_hat = np.zeros(n)
        z3_hat = np.zeros(n)
        
        # Compute conditional expectations
        mask_0 = (y == 0)
        mask_k = (y == self.k_inflated)
        mask_other = ~mask_0 & ~mask_k
        
        # For y = 0
        if np.any(mask_0):
            denominator = np.exp(gamma) + p0[mask_0]
            z1_hat[mask_0] = np.exp(gamma) / denominator
            z3_hat[mask_0] = p0[mask_0] / denominator
        
        # For y = k
        if np.any(mask_k):
            denominator = np.exp(delta) + pk[mask_k]
            z2_hat[mask_k] = np.exp(delta) / denominator
            z3_hat[mask_k] = pk[mask_k] / denominator
        
        # For y ≠ 0, k
        z3_hat[mask_other] = 1
        
        return z1_hat, z2_hat, z3_hat, lambd
    
    def _m_step(self, X, y, z1_hat, z2_hat, z3_hat):
        """M-step: Update parameters"""
        n = len(y)
        
        # Update gamma and delta (mixing parameters)
        gamma_new = np.log(np.sum(z1_hat) / np.sum(z3_hat)) if np.sum(z3_hat) > 0 else self.gamma
        delta_new = np.log(np.sum(z2_hat) / np.sum(z3_hat)) if np.sum(z3_hat) > 0 else self.delta
        
        # Update beta using weighted Poisson regression
        def neg_log_likelihood_beta(beta):
            lambd = np.exp(X @ beta)
            return -np.sum(z3_hat * (y * np.log(lambd) - lambd))
        
        # Initialize beta with current values or zeros
        beta_init = self.beta if hasattr(self, 'beta') else np.zeros(X.shape[1])
        result = minimize(neg_log_likelihood_beta, beta_init, method='BFGS')
        beta_new = result.x
        
        return beta_new, gamma_new, delta_new
    
    def fit(self, X, y, verbose=False):
        """Fit ZKIP model using EM algorithm"""
        n, p = X.shape
        
        # Add intercept if not present
        # if not np.all(X[:, 0] == 1):
        #    X = np.column_stack([np.ones(n), X])
        
        # Initialize parameters
        self.beta = np.zeros(X.shape[1])
        self.gamma = 0.0
        self.delta = 0.0
        
        # Initial values based on observed proportions
        prop_0 = np.mean(y == 0)
        prop_k = np.mean(y == self.k_inflated)
        prop_other = 1 - prop_0 - prop_k
        
        if prop_other > 0:
            self.gamma = np.log(prop_0 / prop_other)
            self.delta = np.log(prop_k / prop_other)
        
        # EM algorithm
        for iteration in range(self.max_iter):
            # E-step
            z1_hat, z2_hat, z3_hat, lambd = self._e_step(y, X, self.beta, self.gamma, self.delta)
            
            # M-step
            beta_new, gamma_new, delta_new = self._m_step(X, y, z1_hat, z2_hat, z3_hat)
            
            # Check convergence
            beta_diff = np.max(np.abs(beta_new - self.beta))
            gamma_diff = np.abs(gamma_new - self.gamma)
            delta_diff = np.abs(delta_new - self.delta)
            
            # Update parameters
            self.beta = beta_new
            self.gamma = gamma_new
            self.delta = delta_new
            
            if verbose and iteration % 100 == 0:
                log_lik = self._log_likelihood(X, y)
                print(f"Iteration {iteration}: Log-likelihood = {log_lik:.4f}")
            
            if max(beta_diff, gamma_diff, delta_diff) < self.tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        # Compute final parameters
        self.pi3 = 1 / (1 + np.exp(self.gamma) + np.exp(self.delta))
        self.pi1 = np.exp(self.gamma) * self.pi3
        self.pi2 = np.exp(self.delta) * self.pi3
        
        self.is_fitted = True
        return self
    
    def _log_likelihood(self, X, y):
        """Compute observed data log-likelihood"""
        if not hasattr(self, 'beta'):
            return -np.inf
        
        lambd = np.exp(X @ self.beta)
        p0 = self._compute_poisson_prob(0, lambd)
        pk = self._compute_poisson_prob(self.k_inflated, lambd)
        
        # Log-likelihood components
        ll_0 = np.sum(np.log(self.pi1 + self.pi3 * p0[y == 0]))
        ll_k = np.sum(np.log(self.pi2 + self.pi3 * pk[y == self.k_inflated]))
        ll_other = np.sum(np.log(self.pi3 * self._compute_poisson_prob(y[(y != 0) & (y != self.k_inflated)], 
                                                                      lambd[(y != 0) & (y != self.k_inflated)])))
        
        return ll_0 + ll_k + ll_other
    
    def predict_proba(self, X, y_values=None):
        """Predict probability for each count value"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Add intercept if not present
        if not np.all(X[:, 0] == 1):
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        lambd = np.exp(X @ self.beta)
        n = len(lambd)
        
        if y_values is None:
            max_y = int(np.max([10, np.max(lambd) * 3, self.k_inflated + 5]))
            y_values = np.arange(0, max_y + 1)
        
        probabilities = np.zeros((n, len(y_values)))
        
        for i, y_val in enumerate(y_values):
            if y_val == 0:
                p0 = self._compute_poisson_prob(0, lambd)
                probabilities[:, i] = self.pi1 + self.pi3 * p0
            elif y_val == self.k_inflated:
                pk = self._compute_poisson_prob(self.k_inflated, lambd)
                probabilities[:, i] = self.pi2 + self.pi3 * pk
            else:
                probabilities[:, i] = self.pi3 * self._compute_poisson_prob(y_val, lambd)
        
        return probabilities, y_values
    
    def predict_expected(self, X):
        """Predict expected count E[Y|X]"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Add intercept if not present
        if not np.all(X[:, 0] == 1):
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        lambd = np.exp(X @ self.beta)
        return self.k_inflated * self.pi2 + self.pi3 * lambd
    
    def predict_mode(self, X):
        """
        Predict most likely count (mode)
        (according to probabilities(from pmf) what is the most likely count)
        """
        probabilities, y_values = self.predict_proba(X)
        return y_values[np.argmax(probabilities, axis=1)]
    
    def get_parameters(self):
        """Get model parameters"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'pi1': self.pi1,
            'pi2': self.pi2,
            'pi3': self.pi3,
            'k_inflated': self.k_inflated
        }
    
