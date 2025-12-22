# ZkINB regression (EM) implementation
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize

class ZkINB_EM:
    """
    Zero- and k-inflated Negative Binomial regression fitted by EM.

    Model:
      For each observation i with predictors:
        X_infl_i  -> predictors for inflation (phi_i, psi_i, p_nb_i via softmax)
        X_count_i -> predictors for NB mean mu_i via log-link

      Component logits (linear predictors):
        eta0_i = X_infl_i @ gamma0    # for structural-zero
        etak_i = X_infl_i @ gammak    # for structural-k
        etanb_i = 0                   # reference -> NB component logit is 0

      probs by softmax:
        [phi_i, psi_i, pnb_i] = softmax([eta0_i, etak_i, 0])

      NB:
        log(mu_i) = X_count_i @ beta
        alpha (dispersion) is scalar > 0
        NB pmf uses r = 1/alpha, p = r/(r+mu_i)

    Main methods:
      fit_em(y, X_infl, X_count, k, ...) -> runs EM and updates parameters
      predict_infl(X_infl) -> returns phi, psi, pnb
      predict_mu(X_count) -> returns mu
      pmf(y, X_infl, X_count) -> returns pmf
    """
    def __init__(self):
        # parameters (to be set by fit_em)
        self.gamma0 = None  # coefficients for structural-zero logits
        self.gammak = None  # coefficients for structural-k logits
        self.beta = None    # coefficients for log(mu)
        self.alpha = None   # scalar dispersion
        self.k = None

    # --------- numeric helpers ----------
    def _nb_logpmf_vec(self, y, mu, alpha):
        """Vectorized log PMF for NB with mean mu (array) and scalar alpha."""
        y = np.asarray(y)
        mu = np.asarray(mu)
        eps = 1e-12
        if alpha <= 0:
            return -np.inf * np.ones_like(y)
        r = 1.0 / alpha
        p = r / (r + mu + eps)
        # gammaln(y + r) - gammaln(r) - gammaln(y+1) + r*log(p) + y*log(1-p)
        return gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(np.maximum(p, eps)) + y * np.log(np.maximum(1 - p, eps))

    def _softmax_three(self, eta0, etak):
        # stack logits: [0, eta0, etak]
        logits = np.vstack([np.zeros_like(eta0), eta0, etak])
        lse = logsumexp(logits, axis=0)

        log_phi = eta0 - lse
        log_psi = etak - lse
        log_pnb = -lse

        phi = np.exp(log_phi)
        psi = np.exp(log_psi)
        pnb = np.exp(log_pnb)

        # clipping to avoid warnings
        eps = 1e-12
        return (
            np.clip(phi, eps, 1 - eps),
            np.clip(psi, eps, 1 - eps),
            np.clip(pnb, eps, 1 - eps),
        )

    # --------- prediction helpers ----------
    def predict_infl(self, X_infl):
        """Return (phi, psi, pnb) for design matrix X_infl (n x p_infl)."""
        X = np.asarray(X_infl)
        eta0 = X @ self.gamma0
        etak = X @ self.gammak
        return self._softmax_three(eta0, etak)

    def predict_mu(self, X_count):
        """Return mu_i = exp(X_count @ beta)."""
        return np.exp(np.asarray(X_count) @ self.beta)

    # --------- EM fit ----------
    def fit_em(self, y, X_infl, X_count, k,
               gamma0_init=None, gammak_init=None, beta_init=None, alpha_init=None,
               tol=1e-6, max_iter=200, verbose=False, reg=1e-6):
        """
        Fit the regression ZkINB via EM.
        Arguments:
          y : 1-d array of integer counts (n,)
          X_infl : design matrix for inflation (n, p_infl)
          X_count: design matrix for NB mean (n, p_count)
          k : integer inflated count
          *_init : optional initial arrays (matching shapes)
          tol, max_iter : EM stopping criteria
          verbose : show progress
          reg : small regularizer used in updates to avoid exact zeros
        Returns:
          result dict with history and final params.
        """
        y = np.asarray(y)
        n = y.size
        self.k = int(k)
        X_infl = np.asarray(X_infl)
        X_count = np.asarray(X_count)
        p_infl = X_infl.shape[1]
        p_count = X_count.shape[1]

        # init parameters if None
        if gamma0_init is None:
            # intercept-only guess for gamma0: logit(proportion_of_zero / proportion_of_nb) approx
            prop0 = np.mean(y == 0)
            gamma0_init = np.zeros(p_infl)
            gamma0_init[0] = np.log(max(prop0, 1e-6) / max(1 - prop0, 1e-6))
        if gammak_init is None:
            propk = np.mean(y == k)
            gammak_init = np.zeros(p_infl)
            gammak_init[0] = np.log(max(propk, 1e-6) / max(1 - propk, 1e-6))
        if beta_init is None:
            # intercept-only initial: log(mean(y_noninfl)+eps)
            mean_y = np.mean(y)
            beta_init = np.zeros(p_count)
            beta_init[0] = np.log(max(mean_y, 1e-2))
        if alpha_init is None:
            s = np.var(y, ddof=0)
            mom_alpha = max((s - mean_y) / (mean_y**2 + 1e-12), 1e-3)
            alpha_init = max(mom_alpha, 1e-3)

        self.gamma0 = gamma0_init.astype(float).copy()
        self.gammak = gammak_init.astype(float).copy()
        self.beta = beta_init.astype(float).copy()
        self.alpha = float(alpha_init)

        history = {"ll": [], "gamma0": [], "gammak": [], "beta": [], "alpha": []}
        prev_ll = -np.inf

        for it in range(1, max_iter + 1):
            # --- E-step ---
            # compute current probs and NB pmf
            eta0 = X_infl @ self.gamma0
            etak = X_infl @ self.gammak
            phi, psi, pnb = self._softmax_three(eta0, etak)
            # clipping to avoid warnings
            eta = X_count @ self.beta
            eta = np.clip(eta, -20, 20)
            mu = np.exp(eta)

            nb_logpmf = self._nb_logpmf_vec(y, mu, self.alpha)
            nb_p = np.exp(nb_logpmf)

            # responsibilities tau0, tauk, taunb
            tau0 = np.zeros(n)
            tauk = np.zeros(n)
            taunb = np.zeros(n)

            mask0 = (y == 0)
            denom0 = phi[mask0] + (pnb[mask0] * (1 - phi[mask0] - psi[mask0]))
            denom0 = np.maximum(denom0, 1e-300)
            tau0[mask0] = phi[mask0] / denom0
            taunb[mask0] = (1 - phi[mask0] - psi[mask0]) * nb_p[mask0] / denom0

            maskk = (y == self.k)
            denomk = psi[maskk] + (pnb[maskk] * (1 - phi[maskk] - psi[maskk]))
            denomk = np.maximum(denomk, 1e-300)
            tauk[maskk] = psi[maskk] / denomk
            taunb[maskk] = (1 - phi[maskk] - psi[maskk]) * nb_p[maskk] / denomk

            mask_other = ~(mask0 | maskk)
            taunb[mask_other] = 1.0

            # normalization safety
            ssum = tau0 + tauk + taunb
            ssum = np.maximum(ssum, 1e-300)
            tau0 /= ssum; tauk /= ssum; taunb /= ssum

            # --- M-step ---
            # Update gamma0 and gammak by maximizing weighted multinomial loglik:
            # max_{gamma} sum_i [ tau0_i * log phi_i + tauk_i * log psi_i + taunb_i * log pnb_i ]
            # where phi, psi, pnb = softmax([X_infl @ gamma0, X_infl @ gammak, 0])
            # We optimize gamma vectorized: concatenate gamma0 and gammak into one vector
            W0 = tau0
            Wk = tauk
            Wnb = taunb

            def weighted_multinomial_negloglik(gvec):
                g0 = gvec[:p_infl]
                gk = gvec[p_infl:]
                a = np.clip(X_infl @ g0, -20, 20)
                b = np.clip(X_infl @ gk, -20, 20)
                S = np.vstack([a, b, np.zeros_like(a)])  # (3, n)
                lse = logsumexp(S, axis=0)
                # log probs
                logphi = a - lse
                logpsi = b - lse
                logpnb = -lse
                # weighted negative loglik
                return -np.sum(W0 * logphi + Wk * logpsi + Wnb * logpnb)

            # initial g vector
            g0_init = np.concatenate([self.gamma0, self.gammak])
            res_g = minimize(weighted_multinomial_negloglik, g0_init, method="L-BFGS-B",
                             options={"ftol":1e-9, "gtol":1e-6, "maxiter":200})
            if res_g.success:
                self.gamma0 = res_g.x[:p_infl]
                self.gammak = res_g.x[p_infl:]
            else:
                # if optimization fails keep old values (rare)
                pass

            # Update beta and alpha by maximizing weighted NB log likelihood:
            # maximize sum_i taunb_i * log NB(y_i | mu_i, alpha)
            # where mu_i = exp(X_count @ beta)
            def weighted_nb_negloglik(params):
                # params = [beta (p_count,), log_alpha]
                beta = params[:p_count]
                log_alpha = params[-1]
                alpha = np.exp(log_alpha)
                # clipping to avoid warnings
                eta = X_count @ beta
                eta = np.clip(eta, -20, 20)
                mu_i = np.exp(eta)

                lp = self._nb_logpmf_vec(y, mu_i, alpha)
                return -np.sum(Wnb * lp)

            # initial params
            log_alpha0 = np.log(max(self.alpha, 1e-8))
            beta0 = self.beta.copy()
            x0 = np.concatenate([beta0, [log_alpha0]])
            bnds = [(-5, 5)] * p_count + [(-10, 3)]
            res_nb = minimize(weighted_nb_negloglik, x0, method="L-BFGS-B", bounds=bnds,
                              options={"ftol":1e-9, "gtol":1e-6, "maxiter":200})
            if res_nb.success:
                self.beta = res_nb.x[:p_count]
                self.alpha = float(np.exp(res_nb.x[-1]))
            else:
                # keep previous values on failure
                pass

            # compute observed-data log-likelihood for monitoring
            # pmf_i = phi_i * 1(y_i==0) + psi_i * 1(y_i==k) + (1-phi-psi) * NB(y_i)
            # but easier: use responsibilities? we compute exact observed pmf:
            eta0 = X_infl @ self.gamma0
            etak = X_infl @ self.gammak
            phi, psi, pnb = self._softmax_three(eta0, etak)
            mu = np.exp(X_count @ self.beta)
            nb_logpmf = self._nb_logpmf_vec(y, mu, self.alpha)
            nb_p = np.exp(nb_logpmf)
            pmf = np.zeros(n)
            pmf[mask0] = phi[mask0] + (1 - phi[mask0] - psi[mask0]) * nb_p[mask0]
            pmf[maskk] = psi[maskk] + (1 - phi[maskk] - psi[maskk]) * nb_p[maskk]
            pmf[mask_other] = (1 - phi[mask_other] - psi[mask_other]) * nb_p[mask_other]
            pmf = np.maximum(pmf, 1e-300)
            ll = np.sum(np.log(pmf))

            history["ll"].append(ll)
            history["gamma0"].append(self.gamma0.copy())
            history["gammak"].append(self.gammak.copy())
            history["beta"].append(self.beta.copy())
            history["alpha"].append(self.alpha)

            if verbose:
                phi_mean = np.mean(phi); psi_mean = np.mean(psi)
                print(f"iter {it:3d} ll={ll:.6f} phi_mean={phi_mean:.4f} psi_mean={psi_mean:.4f} mu_mean={np.mean(mu):.4f} alpha={self.alpha:.6f}")

            # convergence check
            if it > 1:
                rel = abs((ll - prev_ll) / (abs(prev_ll) + 1e-12))
                if rel < tol:
                    if verbose:
                        print(f"Converged at iter {it} (rel change {rel:.2e})")
                    break
            prev_ll = ll
        
        result = {
            "n_iter": it,
            "converged": (it < max_iter or (it == max_iter and rel < tol)),
            "history": history,
            "gamma0": self.gamma0,
            "gammak": self.gammak,
            "beta": self.beta,
            "alpha": self.alpha,
            "k": self.k,
            "final_loglik": prev_ll
        }
        return result

    # --------- convenience functions ---------
    def pmf_given_X(self, y, X_infl, X_count):
        eta0 = X_infl @ self.gamma0
        etak = X_infl @ self.gammak
        phi, psi, pnb = self._softmax_three(eta0, etak)

        eta = np.clip(X_count @ self.beta, -20, 20)
        mu = np.exp(eta)

        log_nb = self._nb_logpmf_vec(y, mu, self.alpha)

        log_phi = np.log(phi)
        log_psi = np.log(psi)
        log_pnb = np.log(pnb)

        out = np.zeros_like(y, dtype=float)

        mask0 = (y == 0)
        maskk = (y == self.k)
        masko = ~(mask0 | maskk)

        out[mask0] = np.exp(
            logsumexp([log_phi[mask0], log_pnb[mask0] + log_nb[mask0]], axis=0)
        )
        out[maskk] = np.exp(
            logsumexp([log_psi[maskk], log_pnb[maskk] + log_nb[maskk]], axis=0)
        )
        out[masko] = np.exp(log_pnb[masko] + log_nb[masko])

        return np.clip(out, 1e-300, None)


    def predict_infl_probs(self, X_infl):
        return self.predict_infl(X_infl)  # phi, psi, pnb

    def predict_mu_vals(self, X_count):
        return self.predict_mu(X_count)

    def predict(self, X_infl, X_count):
        """Predict expected counts E[Y] = psi * k + pnb * mu"""
        phi, psi, pnb = self.predict_infl(X_infl)
        mu = self.predict_mu(X_count)
        return psi*self.k + pnb * mu
