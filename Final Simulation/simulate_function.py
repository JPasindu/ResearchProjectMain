import numpy as np
from generate_hnb import *
from itertools import product

# -------------------------------------------------------------
#  SIMULATION PLAN (MATCHES FENG 2021)
# -------------------------------------------------------------
def simulation_plan(
        B=200,                  # replications per setting (paper uses 200)
        n=300,                  # sample size (paper uses 300)
        k=2,                    # inflated point in your generator
        r=1,                    # NB dispersion (r=1 NB-like in paper)
        cov_type="binary",      # binary or continuous x
        beta0=0, gamma0=0, alpha0=0,   # intercepts fixed
        # parameter grids (paper varies α1 and β1 between -2 to +2)
        beta1_vals=np.linspace(-2, 2, 9),  
        gamma1_vals=np.linspace(-2, 2, 9),
        alpha1_vals=np.linspace(-2, 2, 9)
    ):
    
    results = []

    # Loop over the parameter grid exactly as the paper
    for beta1, gamma1, alpha1 in product(beta1_vals, gamma1_vals, alpha1_vals):

        print(f"Generating: β1={beta1}, γ1={gamma1}, α1={alpha1}")

        for rep in range(B):

            df = generate_hnb(
                n=n,
                k=k,
                beta0=beta0, beta1=beta1,
                gamma0=gamma0, gamma1=gamma1,
                alpha0=alpha0, alpha1=alpha1,
                r=r,
                cov_type=cov_type
            )

            p_0 = df['y'].value_counts().get(0, 0) / len(df)
            p_k = df['y'].value_counts().get(k, 0) / len(df)
            p_p = max(1 - p_0 - p_k, 0)
            y_mean = df['y'].mean()
            y_std = df['y'].std()

            results.append({
                "data": df,
                "params": {
                    #"rep": rep,
                    #"n": n, "k": k,
                    "beta0": beta0, "beta1": beta1,
                    "gamma0": gamma0, "gamma1": gamma1,
                    "alpha0": alpha0, "alpha1": alpha1,
                    #"r": r,
                    #"cov_type": cov_type
                    'n_unique': len(df['y'].value_counts()), 
                    'p_0': p_0,
                    'p_k': p_k,
                    'p_p': p_p,
                    'y_mean': y_mean,
                    'y_std': y_std
                }
            })

    return results