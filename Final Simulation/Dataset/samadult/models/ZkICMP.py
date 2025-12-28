import numpy as np
from math import lgamma
from scipy.optimize import minimize
from functools import lru_cache
import warnings

# no EM algorithm
class ZkICMP:
    """
    Zero-and-k Inflated Conway-Maxwell-Poisson (ZkICMP) Model
    
    A flexible count data model that handles inflation at zero and a specific value k,
    using the Conway-Maxwell-Poisson distribution for the count process.
    
    Parameters
    ----------
    k : int
        The specific count value that is inflated (in addition to zero)
    max_iter : int, default=2000
        Maximum iterations for CMP normalizing constant calculation
    tol : float, default=1e-12
        Tolerance for CMP normalizing constant calculation
    cache_size : int, default=4096
        Size of cache for CMP normalizing constant
    """
    
    def __init__(self, k, max_iter=2000, tol=1e-12, cache_size=4096):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.cache_size = cache_size
        self._cached_Z = lru_cache(maxsize=cache_size)(self._cmp_Z_uncached)
        
        # Model parameters (set after fitting)
        self.beta_ = None
        self.gamma_ = None
        self.delta_ = None
        self.eta_ = None
        self.nu_ = None
        self.params_ = None
        self.fitted_ = False
        
    def _cmp_Z(self, lambda_, nu):
        """CMP normalizing constant Z(λ, ν) by truncation"""
        return self._cached_Z(float(lambda_), float(nu))
    
    def _cmp_Z_uncached(self, lambda_, nu):
        """Uncached version of CMP normalizing constant calculation"""
        if lambda_ <= 0:
            return 1.0  # Z=1 when lambda=0 (only y=0 possible)
        
        log_lambda = np.log(lambda_)
        terms = [0.0]  # log(1) for j=0
        j = 0
        
        while j < self.max_iter:
            j += 1
            log_term = j * log_lambda - nu * lgamma(j + 1) #------------RuntimeWarning: invalid value encountered in scalar subtract log_term = j * log_lambda - nu * lgamma(j + 1)
            if np.exp(log_term - max(terms)) < self.tol: #--------------RuntimeWarning: invalid value encountered in scalar subtract if np.exp(log_term - max(terms)) < self.tol:
                terms.append(log_term)
                break
            terms.append(log_term)
        
        a = np.array(terms)
        a_max = a.max()
        Z = np.exp(a_max) * np.sum(np.exp(a - a_max)) #----------------RuntimeWarning: invalid value encountered in subtract Z = np.exp(a_max) * np.sum(np.exp(a - a_max))
        return Z
    
    def _log_cmp_pmf(self, y, lambda_, nu):
        """log pmf for CMP distribution"""
        if lambda_ <= 0:
            return -np.inf if y > 0 else 0.0
        logZ = np.log(self._cmp_Z(lambda_, nu))
        return y * np.log(lambda_) - nu * lgamma(y + 1) - logZ #---------RuntimeWarning: invalid value encountered in scalar multiply return y * np.log(lambda_) - nu * lgamma(y + 1) - logZ
    
    def _cmp_pmf(self, y, lambda_, nu):
        """PMF for CMP distribution"""
        return np.exp(self._log_cmp_pmf(y, lambda_, nu))
    
    def _negloglik(self, params, X, y):
        """Negative log-likelihood for ZkICMP"""
        p = X.shape[1]
        beta = params[:p]
        gamma = params[p]
        delta = params[p + 1]
        eta = params[p + 2]
        nu = np.exp(eta) #----------------------------------RuntimeWarning: overflow encountered in exp nu = np.exp(eta)
        lambdas = np.exp(X.dot(beta)) #---------------------RuntimeWarning: overflow encountered in exp lambdas = np.exp(X.dot(beta))
        
        eg = np.exp(gamma)
        ed = np.exp(delta)
        denom = 1.0 + eg + ed
        pi1 = eg / denom
        pi2 = ed / denom
        pi3 = 1.0 / denom

        ll = 0.0
        n = len(y)
        for i in range(n):
            yi = int(y[i])
            lam = float(lambdas[i])
            if yi == 0:
                p0_cmp = self._cmp_pmf(0, lam, nu)
                prob = pi1 + pi3 * p0_cmp
                ll += np.log(max(prob, 1e-300))
            elif yi == self.k:
                pk_cmp = self._cmp_pmf(self.k, lam, nu)
                prob = pi2 + pi3 * pk_cmp
                ll += np.log(max(prob, 1e-300))
            else:
                py_cmp = self._cmp_pmf(yi, lam, nu)
                prob = pi3 * py_cmp
                ll += np.log(max(prob, 1e-300))
        return -ll
    
    def fit(self, X, y, init=None, method='L-BFGS-B', verbose=False):
        """
        Fit the ZkICMP model to data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Count response variable
        init : array-like, optional
            Initial parameter values [beta, gamma, delta, eta]
        method : str, default='L-BFGS-B'
            Optimization method
        verbose : bool, default=False
            Whether to print optimization messages
            
        Returns
        -------
        self : object
            Fitted model
        """
        X = np.array(X)
        y = np.array(y)
        n, p = X.shape
        
        if init is None:
            beta0 = np.zeros(p)
            if p == 1 and np.allclose(X[:, 0], 1.0):
                beta0[0] = np.log(max(np.mean(y[y != 0]), 1e-3))
            gamma0 = np.log(max(np.mean(y == 0), 1e-3) / max((1.0 - np.mean(y == 0) - np.mean(y == self.k)), 1e-6))
            delta0 = np.log(max(np.mean(y == self.k), 1e-3) / max((1.0 - np.mean(y == 0) - np.mean(y == self.k)), 1e-6))
            eta0 = 0.0
            init = np.concatenate([beta0, [gamma0, delta0, eta0]])
        
        bnds = [(None, None)] * (p + 3)
        res = minimize(self._negloglik, init, args=(X, y), method=method, bounds=bnds,
                      options={'maxiter': 1000, 'disp': verbose})
        
        if not res.success:
            warnings.warn(f"Optimization did not converge: {res.message}")
        
        # Store fitted parameters
        self.params_ = res.x
        self.beta_ = res.x[:p]
        self.gamma_ = res.x[p]
        self.delta_ = res.x[p + 1]
        self.eta_ = res.x[p + 2]
        self.nu_ = np.exp(self.eta_)
        self.fitted_ = True
        self.final_loglik = self._negloglik(self.params_, X, y)
        return self
    
    def predict(self, X, max_count=50):
        """
        Predict counts using the fitted model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        max_count : int, default=50
            Maximum count to consider in prediction
            
        Returns
        -------
        predictions : ndarray, shape (n_samples,)
            Predicted counts (modes of distributions)
        probabilities : ndarray, shape (n_samples, max_count+1)
            Probability distributions over counts
        lambdas : ndarray, shape (n_samples,)
            Predicted lambda values
        counts_range : ndarray, shape (max_count+1,)
            The count values corresponding to probability columns
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.array(X)
        p = X.shape[1]
        
        # Calculate lambdas
        lambdas = np.exp(X.dot(self.beta_))
        
        # Calculate probabilities
        eg = np.exp(self.gamma_)
        ed = np.exp(self.delta_)
        denom = 1.0 + eg + ed
        pi1 = eg / denom
        pi2 = ed / denom
        pi3 = 1.0 / denom
        
        # Predict counts (mode of distribution for each observation)
        predictions = []
        all_probabilities = []
        expected_values = []  # E[Y]

        # Use fixed count range for all observations to ensure consistent array shapes
        counts_range = np.arange(0, max_count + 1)
        
        for lam in lambdas:
            probs = np.zeros(len(counts_range))
            
            for i, count in enumerate(counts_range):
                if count == 0:
                    probs[i] = pi1 + pi3 * self._cmp_pmf(0, lam, self.nu_)
                elif count == self.k:
                    probs[i] = pi2 + pi3 * self._cmp_pmf(self.k, lam, self.nu_)
                else:
                    probs[i] = pi3 * self._cmp_pmf(count, lam, self.nu_)
            
            # Normalize probabilities
            probs = probs / np.sum(probs)
            
            # Predict as mode (most probable count)
            pred_count = counts_range[np.argmax(probs)]
            predictions.append(pred_count)
            all_probabilities.append(probs)
            expected_values.append((probs * counts_range).sum())
        
        return (np.array(predictions),  np.array(all_probabilities), 
                np.array(expected_values), counts_range)
    
    def predict_proba(self, X, max_count=50):
        """
        Predict probability distributions.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        max_count : int, default=50
            Maximum count to consider
            
        Returns
        -------
        probabilities : ndarray, shape (n_samples, max_count+1)
            Probability distributions over counts
        counts_range : ndarray, shape (max_count+1,)
            The count values corresponding to probability columns
        """
        _, probabilities, _, counts_range = self.predict(X, max_count)
        return probabilities, counts_range
    
    def get_params(self):
        """Get model parameters"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet")
        return {
            'beta': self.beta_,
            'gamma': self.gamma_,
            'delta': self.delta_,
            'eta': self.eta_,
            'nu': self.nu_,
            'all_params': self.params_
        }
    
    def score(self, X, y):
        """
        Compute the mean log-likelihood of the data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Count response variable
            
        Returns
        -------
        score : float
            Mean log-likelihood
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before scoring")
        neg_ll = self._negloglik(self.params_, X, y)
        return -neg_ll / len(y)

