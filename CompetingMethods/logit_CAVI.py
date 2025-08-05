# =============================================================================
# Copyright 2025. Somjit Roy and Pritam Dey. 
# This program implements mean-field variational inference (CAVI) for Bayesian
# logistic regression (following Durante and Rigon, 2019 <https://doi.org/10.1214/19-STS712>)
# in order to compare with TAVIE against the TAVIE algorithm 
# as developed in:
# Roy, S., Dey, P., Pati, D., and Mallick, B.K.
# 'A Generalized Tangent Approximation Framework for Strongly Super-Gaussian Likelihoods',
# arXiv:2504.05431 <https://arxiv.org/abs/2504.05431>.
#
# Authors:
#   Somjit Roy <sroy_123@tamu.edu> and Pritam Dey <pritam.dey@tamu.edu>
# =============================================================================


# Required imports
import numpy as np
from scipy.special import expit  
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

def elbo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    xi: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    mu0: np.ndarray,
    P: np.ndarray,
    logdetP: float
) -> float:
    """
    Compute the variational lower bound (ELBO) for Bayesian logistic regression.

    Parameters
    ----------
    mu : ndarray, shape (p,)
        Variational mean.
    Sigma : ndarray, shape (p, p)
        Variational covariance.
    xi : ndarray, shape (n,)
        Variational scale parameters.
    X : ndarray, shape (n, p)
        Design matrix.
    y : ndarray, shape (n,)
        Binary responses.
    mu0 : ndarray, shape (p,)
        Prior mean.
    P : ndarray, shape (p, p)
        Prior precision matrix (inverse of prior covariance).
    logdetP : float
        Log-determinant of P.

    Returns
    -------
    float
        The ELBO value.
    """
    p = mu.shape[0]
    eta = X @ mu
    # log σ(xi) = -log(1 + exp(-xi)) = -logaddexp(0, -xi)
    logsig = -np.logaddexp(0, -xi)

    term1 = 0.5 * p + 0.5 * np.linalg.slogdet(Sigma)[1] + 0.5 * logdetP
    term2 = -0.5 * (mu - mu0) @ (P @ (mu - mu0))
    term3 = np.sum((y - 0.5) * eta + logsig - 0.5 * xi)
    term4 = -0.5 * np.trace(P @ Sigma)

    return term1 + term2 + term3 + term4
    
def logit_cavi(
        X: np.ndarray,
        y: np.ndarray, 
        prior_params: Dict[str, np.ndarray], 
        tol: float=1e-16, 
        maxiter: int=10000,
        verbose: bool=True
        ) -> Dict[str, np.ndarray]:
    """
    Coordinate Ascent VI (CAVI) for Bayesian logistic regression.
    
    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    y : ndarray, shape (n,)
        Binary responses (0/1).
    prior : dict
        Must have keys 'Sigma' (pxp prior covariance) and 'mu' (p-vector).
    tol : float
        Convergence tolerance on the ELBO difference.
    maxiter : int
        Maximum number of iterations.
    
    Returns
    -------
    mu_vb : ndarray, shape (p,)
        Variational mean.
    Sigma_vb : ndarray, shape (p, p)
        Variational covariance.
    conv : ndarray, shape (T, 2)
        Iteration and ELBO history.
    xi : ndarray, shape (n,)
        Last variational scale parameters.
    """
    
    # --- input validation ---
    # X must be a 2‑D NumPy array (matrix)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("'X' must be a 2-D NumPy array (matrix) of shape (n, p)")
    n, p = X.shape
    
    # y must be a 1‑D array of length n
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != n:
        raise ValueError("'y' must be a 1-D NumPy array of length equal to X.shape[0]")
    
    # Validate prior contains Sigma and mu with correct shapes
    Sigma0 = prior_params.get('Sigma', None)
    mu0    = prior_params.get('mu',    None)
    if Sigma0 is None or mu0 is None:
        raise ValueError("`prior_params` must be a dict with keys 'Sigma' and 'mu'")
    if not isinstance(Sigma0, np.ndarray) or Sigma0.ndim != 2 or Sigma0.shape != (p, p):
        raise ValueError(f"'Sigma' must be a NumPy array of shape ({p}, {p}); got {None if Sigma0 is None else Sigma0.shape}")
    if not isinstance(mu0, np.ndarray) or mu0.ndim != 1 or mu0.shape[0] != p:
        raise ValueError(f"'mu' must be a NumPy array of length {p}; got {None if mu0 is None else mu0.shape}")

    # precision and precision×mean
    P      = np.linalg.inv(Sigma0)
    Pmu    = P @ mu0
    # log‑determinant of P
    signP, logdetP = np.linalg.slogdet(P)
    if signP <= 0:
        raise np.linalg.LinAlgError("Prior precision is not positive‑definite")
    
    # storage
    lb_hist = np.zeros(maxiter)
    
    #--- initialization (omega = 1/4) ---
    omega = np.full(n, 0.25)
    Pv    = X.T @ (omega[:,None] * X) + P
    Sigma = np.linalg.inv(Pv)
    mu    = Sigma @ (X.T @ (y - 0.5) + Pmu)
    
    eta   = X @ mu
    # xi_i = sqrt(eta_i^2 + Var[f_i] ), Var[f_i] = x_i^T Σ x_i
    xi    = np.sqrt(eta**2 + np.sum(X @ Sigma * X, axis=1))
    omega = np.tanh(xi/2) / (2*xi)
    omega = np.where(np.isnan(omega), 0.25, omega)
    
    lb_hist[0] = elbo(mu, Sigma, xi, X, y, mu0, P, logdetP)
    
    #--- CAVI iterations ---
    for t in range(1, maxiter):
        # update precision
        Pv    = X.T @ (omega[:,None] * X) + P
        Sigma = np.linalg.inv(Pv)
        mu    = Sigma @ (X.T @ (y - 0.5) + Pmu)
        
        eta   = X @ mu
        xi    = np.sqrt(eta**2 + np.sum(X @ Sigma * X, axis=1))
        omega = np.tanh(xi/2) / (2*xi)
        omega = np.where(np.isnan(omega), 0.25, omega)
        
        lb_hist[t] = elbo(mu, Sigma, xi, X, y, mu0, P, logdetP)
        if abs(lb_hist[t] - lb_hist[t-1]) < tol:
            iters = t + 1
            if(verbose):
                print(f"Converged after {iters} iterations "
                      f"(ELBO change = {abs(lb_hist[t] - lb_hist[t-1]):.2e}).")
            break
    else:
        raise RuntimeError("logit_cavi did not converge within maxiter")
    
    # pack convergence history
    conv = np.vstack((np.arange(iters), lb_hist[:iters])).T
    
    return {
        'mu'          : mu,        # p-vector
        'Sigma'       : Sigma,     # pxp matrix
        'convergence' : conv,      # array([[iter, elbo], ...])
        'xi'          : xi         # n-vector
    }

# =============================================================================
# Testing logit_cavi just for a particular choice of sample size
# =============================================================================

def run_logit_cavi_test():
    """
    Runs a test of Coordinate Ascent Variational Inference (CAVI)
    for Bayesian logistic regression using synthetic data.

    The model:
        y_i ~ Bernoulli(sigmoid(x_i^T β))
        β ~ N(μ₀, Σ₀)

    This function:
    - Generates synthetic logistic data with a known β
    - Constructs a design matrix with an intercept
    - Specifies Gaussian priors on β
    - Runs CAVI using `logit_cavi`
    - Prints the variational estimates and compares to truth

    Returns
    -------
    None
        Prints the results to stdout.

    Notes
    -----
    - The prior used is N(0, I).
    - The function assumes the existence of a valid `logit_cavi(X, y, prior_params)` function,
      as above.
    - You can uncomment lines to sample posterior draws from the variational distribution.
    """
    # True coefficients
    beta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = 5
    
    # Prior hyperparameters
    prior = {
        'mu'   : np.zeros(p),
        'Sigma': np.eye(p)
    }

    # Generate data
    n = 10000
    np.random.seed(123)
    x = np.random.uniform(-2, 2, size=(n, p-1))
    # adding an intercept term of 1
    X = np.column_stack((np.ones(n), x))
    prob = expit(X @ beta)
    y    = np.random.binomial(1, prob, size=n)

    # Run CAVI
    result = logit_cavi(X, y, prior_params=prior)
    mu_cavi    = result['mu']
    #Sigma_cavi = result['Sigma']

    # Draw posterior samples
    #np.random.seed(1010)
    # drawing from intercept distribution from CAVI 
    #beta0_samples = np.random.normal(mu_cavi[0], np.sqrt(Sigma_cavi[0,0]), size=10_000)
    # drawing from slope distribution from CAVI
    #beta1_samples = np.random.normal(mu_cavi[1], np.sqrt(Sigma_cavi[1,1]), size=10_000)
    
    print(f"Sample size: {n}")
    print(f"True coefficients: {beta}")
    print(f"CAVI variational estimates, E(beta): {mu_cavi}")
    #print(f"Mean of CAVI-estimated intercept: {np.mean(beta0_samples)}")
    #print(f"Mean of CAVI-estimated slope: {np.mean(beta1_samples)}")

if __name__ == "__main__":
    run_logit_cavi_test()

# =============================================================================
# Testing logit_cavi for different repetitions of sample sizes and plotting 
# the L2 error between true and CAVI estimated coefficients
# =============================================================================
def plot_cavi_true_l2_error(
    beta: np.ndarray,
    prior: Dict[str, np.ndarray],
    sample_sizes: list[int],
    n_reps: int,
    seed: int = 123
) -> None:
    """
    For each sample size in `sample_sizes`, run `n_reps` repetitions of:
      1. Generate an (n x p) design matrix with intercept + Uniform(-2,2) covariates.
      2. Simulate binary responses via logistic model with true coefficients `beta`.
      3. Run `logit_cavi` to get variational mean mu_cavi.
      4. Compute L2 error ||mu_cavi - beta||.
    Finally, display a boxplot of L2 errors versus sample sizes.

    Parameters
    ----------
    beta : ndarray, shape (p,)
        True regression coefficients.
    prior : dict
        Contains 'mu' (p-vector) and 'Sigma' (p x p covariance).
    sample_sizes : list of int
        Sample sizes to evaluate.
    n_reps : int
        Number of repetitions per sample size.
    seed : int, default=123
        Random seed for reproducibility.

    Returns
    -------
    None
        Shows a boxplot of L2 errors.
    """
    errors = {n: [] for n in sample_sizes}
    np.random.seed(seed)
    p = len(beta)
    
    for n in tqdm(sample_sizes, desc="Sample sizes"):
        for _ in tqdm(range(n_reps), desc=f" Reps (n={n})", leave=False):
            # generate design matrix
            x = np.random.uniform(-2, 2, size=(n, p-1))
            X = np.column_stack((np.ones(n), x))
            # simulate responses
            prob = expit(X @ beta)
            y    = np.random.binomial(1, prob, size=n)
            # run CAVI and compute error
            result = logit_cavi(X, y, prior_params=prior, verbose=False)
            err = np.linalg.norm(result['mu'] - beta)
            errors[n].append(err)

    # 4) plot boxplot
    data = [errors[n] for n in sample_sizes]
    plt.figure()
    plt.boxplot(data, labels=sample_sizes)
    plt.xlabel('Sample Size')
    plt.ylabel(r'$L_2$ Error')
    plt.title(r'CAVI/True $L_2$ Error vs. Sample Size')
    plt.tight_layout()
    plt.show()

beta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
prior = {'mu': np.zeros_like(beta), 'Sigma': np.eye(len(beta))}
plot_cavi_true_l2_error(beta, prior, sample_sizes=[200,500,1000,5000,10000,20000], n_reps=200)