import numpy as np
import pandas as pd

def generate_zkip_data(n=1000, k_inflated=3, true_beta=None, true_gamma=None, true_delta=None, 
                      pi1=None, pi2=None, pi3=None, seed=42):
    """
    Generate synthetic ZKIP (Zero-K Inflated Poisson) data with flexible probability control
    
    Parameters:
    -----------
    n : int
        Number of samples
    k_inflated : int
        The value at which inflation occurs (default: 3)
    true_beta : array-like
        Coefficients for Poisson component [intercept, x1, x2, ...]
    true_gamma : float
        Log-odds parameter for zero-inflation (alternative to pi1)
    true_delta : float
        Log-odds parameter for k-inflation (alternative to pi2)
    pi1 : float
        Probability of zero-inflation class (if specified, overrides true_gamma)
    pi2 : float
        Probability of k-inflation class (if specified, overrides true_delta)
    pi3 : float
        Probability of Poisson class (if specified, must satisfy pi1 + pi2 + pi3 = 1)
    seed : int
        Random seed
    
    Returns:
    --------
    X_data : array
        Covariate matrix (without intercept)
    y : array
        Generated counts
    metadata : dict
        Dictionary with true parameters and probabilities
    """
    np.random.seed(seed)
    
    # Default beta coefficients
    if true_beta is None:
        true_beta = np.array([0.5, -0.3, 0.8])  # intercept, x1, x2
    
    # Generate covariates
    X = np.column_stack([
        np.ones(n),
        np.random.normal(0, 1, n),  # x1
        np.random.normal(0, 1, n)   # x2
    ])
    
    # Handle probability specification
    if pi1 is not None and pi2 is not None and pi3 is not None:
        # Direct probability specification
        if not np.isclose(pi1 + pi2 + pi3, 1.0):
            raise ValueError("pi1 + pi2 + pi3 must sum to 1")
        true_pi1, true_pi2, true_pi3 = pi1, pi2, pi3
        # Back-calculate gamma and delta for consistency
        true_gamma = np.log(pi1 / pi3)
        true_delta = np.log(pi2 / pi3)
    
    elif pi1 is not None and pi2 is not None:
        # pi1 and pi2 specified, pi3 = 1 - pi1 - pi2
        if pi1 + pi2 >= 1.0:
            raise ValueError("pi1 + pi2 must be less than 1")
        true_pi3 = 1 - pi1 - pi2
        true_pi1, true_pi2 = pi1, pi2
        true_gamma = np.log(pi1 / true_pi3)
        true_delta = np.log(pi2 / true_pi3)
    
    elif true_gamma is not None and true_delta is not None:
        # Gamma/delta specification (original method)
        true_pi3 = 1 / (1 + np.exp(true_gamma) + np.exp(true_delta))
        true_pi1 = np.exp(true_gamma) * true_pi3
        true_pi2 = np.exp(true_delta) * true_pi3
    
    else:
        # Default values
        true_gamma = 0.5
        true_delta = 0.5
        true_pi3 = 1 / (1 + np.exp(true_gamma) + np.exp(true_delta))
        true_pi1 = np.exp(true_gamma) * true_pi3
        true_pi2 = np.exp(true_delta) * true_pi3
    
    # Calculate lambda for Poisson component
    lambd = np.exp(X @ true_beta)
    
    # Generate latent class indicators
    z = np.random.choice([0, 1, 2], size=n, p=[true_pi1, true_pi2, true_pi3])
    
    # Generate counts based on latent class
    y = np.zeros(n)
    
    # Class 0: Degenerate at 0
    mask_0 = (z == 0)
    y[mask_0] = 0
    
    # Class 1: Degenerate at k
    mask_k = (z == 1)
    y[mask_k] = k_inflated
    
    # Class 2: Poisson
    mask_poisson = (z == 2)
    y[mask_poisson] = np.random.poisson(lambd[mask_poisson])
    
    # Remove intercept from X for modeling
    X_data = X[:, 1:]
    
    return X_data, y, {
        'true_beta': true_beta,
        'true_gamma': true_gamma,
        'true_delta': true_delta,
        'true_pi1': true_pi1,
        'true_pi2': true_pi2,
        'true_pi3': true_pi3,
        'k_inflated': k_inflated,
        'class_counts': np.bincount(z),
        'class_proportions': np.bincount(z) / n
    }

def data_sets(n=1000, k_inflated=3, true_beta=None, true_gamma=None, true_delta=None, 
                      pi1=None, pi2=None, pi3=None, seed=42):
    # Step 1: Generate Synthetic Data
    print("\n1. GENERATING SYNTHETIC DATA")
    print("-" * 30)

    X, y, true_params = generate_zkip_data(n=n, k_inflated=k_inflated, true_beta=true_beta, true_gamma=None, true_delta=None, 
                      pi1=pi1, pi2=pi2, pi3=pi3, seed=42)

    # Create train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train_df=pd.DataFrame(X_train, columns=['X1', 'X2'])
    X_test_df=pd.DataFrame(X_test, columns=['X1', 'X2'])
    y_train_df=pd.DataFrame(y_train, columns=['Y'])
    y_test_df=pd.DataFrame(y_test, columns=['Y'])

    X_train_name=f'DataSets/X_train_0k_inflated_{X.shape}_({pi1:.2f},{pi2:.2f}).csv'
    X_test_name=f'DataSets/X_test_0k_inflated_{X.shape}_({pi1:.2f},{pi2:.2f}).csv'
    y_train_df_name=f'DataSets/y_train {X.shape}_({pi1:.2f},{pi2:.2f}).csv'
    y_test_df_name=f'DataSets/y_test {X.shape}_({pi1:.2f},{pi2:.2f}).csv'
    
    X_train_df.to_csv(X_train_name, index=False)
    X_test_df.to_csv(X_test_name, index=False)
    y_train_df.to_csv(y_train_df_name, index=False)
    y_test_df.to_csv(y_test_df_name, index=False)
    
    print(X_train_name)
    print(X_test_name)
    print(y_train_df_name)
    print(y_test_df_name)
    
    print(f"Data shape: {X.shape}")