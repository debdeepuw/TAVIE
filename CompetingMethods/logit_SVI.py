# =============================================================================
# Copyright 2025. Somjit Roy and Pritam Dey. 
# This program implements stochastic variational inference (SVI) for Bayesian
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


def logit_svi(
    X: np.ndarray,
    y: np.ndarray,
    prior_params: Dict[str, np.ndarray],
    n_iter: int,
    tau: float,
    kappa: float,
    seed: int = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Stochastic VI (SVI) for Bayesian logistic regression.
    
    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    y : ndarray, shape (n,)
        Binary responses (0 or 1).
    prior_params : dict
        'mu'    : ndarray, shape (p,)     — prior mean vector
        'Sigma' : ndarray, shape (p, p)   — prior covariance matrix
    n_iter : int
        Number of SVI iterations.
    tau : float
        Delay parameter in the learning rate schedule.
    kappa : float
        Forgetting-rate exponent (0.5 < kappa ≤ 1).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, prints progress every 100 iterations.

    Returns
    -------
    result : dict
        'mu'    : ndarray, shape (p,)   — variational mean
        'Sigma' : ndarray, shape (p, p) — variational covariance
    """
    # --- input validation ---
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("'X' must be a 2-D NumPy array of shape (n, p)")
    n, p = X.shape

    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != n:
        raise ValueError("'y' must be a 1-D NumPy array of length equal to X.shape[0]")

    Sigma0 = prior_params.get('Sigma')
    mu0    = prior_params.get('mu')
    if Sigma0 is None or mu0 is None:
        raise ValueError("`prior_params` must contain 'Sigma' and 'mu'")
    if not (isinstance(Sigma0, np.ndarray) and Sigma0.shape == (p, p)):
        raise ValueError(f"'Sigma' must be a NumPy array of shape ({p},{p})")
    if not (isinstance(mu0, np.ndarray) and mu0.shape == (p,)):
        raise ValueError(f"'mu' must be a NumPy array of length {p}")

    if not (isinstance(n_iter, int) and n_iter > 0):
        raise ValueError("'n_iter' must be a positive integer")
    if tau < 0 or kappa <= 0:
        raise ValueError("'tau' must be non-negative and 'kappa' positive")

    # --- set up RNG ---
    rng = np.random.default_rng(seed)

    # --- prior precision and natural parameters ---
    P    = np.linalg.inv(Sigma0)
    Pmu  = P @ mu0

    # natural parameters
    Eta1_out = Pmu.copy()      # vector, shape (p,)
    Eta2_out = P.copy()        # matrix, shape (p,p)

    # SVI loop
    for t in range(1, n_iter + 1):
        # 1) sample one data point
        i    = rng.integers(n)
        x_i  = X[i]              # shape (p,)
        y_i  = y[i]

        # 2) compute local VB solution
        Sigma_vb = np.linalg.inv(Eta2_out)
        mu_vb    = Sigma_vb @ Eta1_out

        # 3) local variational parameter
        eta_i = x_i @ mu_vb
        xi_i  = np.sqrt(eta_i**2 + (x_i @ Sigma_vb * x_i).sum())
        omega = np.tanh(xi_i / 2) / (2 * xi_i)

        # 4) compute new natural parameters
        Eta1 = n * x_i * (y_i - 0.5) + Pmu                     # shape (p,)
        Eta2 = n * np.outer(x_i * omega, x_i) + P              # shape (p,p)

        # 5) global update with step size rho_t
        rho = (t + tau) ** (-kappa)
        Eta1_out = (1 - rho) * Eta1_out + rho * Eta1
        Eta2_out = (1 - rho) * Eta2_out + rho * Eta2

        if verbose and (t % 100 == 0 or t == n_iter):
            print(f"[SVI] Iter {t}/{n_iter} — ρ={rho:.3e}")

    # --- final variational parameters ---
    Sigma_final = np.linalg.inv(Eta2_out)
    mu_final    = Sigma_final @ Eta1_out

    return {
        'mu'    : mu_final,
        'Sigma' : Sigma_final
    }

# =============================================================================
# Testing logit_svi just for a particular choice of sample size 
# =============================================================================

def run_logit_svi_test():
    """
    Runs a single test of Stochastic Variational Inference (SVI)
    for Bayesian logistic regression using synthetic data.

    The model:
        y_i ~ Bernoulli(sigmoid(x_i^T β))
        β ~ N(μ₀, Σ₀)

    This function:
    - Simulates logistic data with known true coefficients
    - Uses a Gaussian prior N(0, I) for β
    - Runs stochastic variational inference using `logit_svi`
    - Prints the variational posterior mean estimates and compares them to the true β

    Returns
    -------
    None
        Prints the results to stdout.

    Notes
    -----
    - This test uses 10,000 observations and 5 coefficients.
    - It assumes that `logit_svi` is defined and accepts arguments like:
        - `X`, `y`, `prior_params`, `n_iter`, `tau`, `kappa`, `verbose`, `seed`.
    - The SVI learning rate follows: ηₜ = (τ + t)^(-κ)
    """
    # True coefficients
    beta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = beta.size

    # Prior hyperparameters
    prior = {
        'mu'   : np.zeros(p),
        'Sigma': np.eye(p)
    }

    # Generate data
    n = 10_000
    np.random.seed(123)
    x = np.random.uniform(-2, 2, size=(n, p-1))
    X = np.column_stack((np.ones(n), x))
    prob = expit(X @ beta)
    y    = np.random.binomial(1, prob, size=n)

    # Run SVI
    result = logit_svi(
        X,
        y,
        prior_params=prior,
        n_iter=10_000,
        tau=1.0,
        kappa=0.75,
        verbose=True,
        seed=123
    )
    mu_svi = result['mu']

    # Print comparison
    print(f"Sample size: {n}")
    print(f"True coefficients:      {beta}")
    print(f"SVI variational means estimates:  {mu_svi}")

if __name__ == "__main__":
    run_logit_svi_test()
    
# =============================================================================
# Testing logit_svi for different repetitions of sample sizes and plotting 
# the L2 error between true and SVI estimated coefficients
# =============================================================================

def plot_svi_true_l2_error(
    beta: np.ndarray,
    prior: Dict[str, np.ndarray],
    sample_sizes: list[int],
    n_reps: int,
    n_iter: int,
    tau: float,
    kappa: float,
    seed: int = 123
) -> None:
    """
    Evaluates the accuracy of logit SVI across varying sample sizes by 
    computing the L2 error between the variational posterior mean and 
    the true coefficient vector.

    For each sample size `n` in `sample_sizes`, this function:
        1. Repeats the experiment `n_reps` times:
            a. Simulates synthetic logistic data with `n` observations
            b. Runs logit SVI
            c. Computes L2 error: ||μ_svi - β_true||₂
        2. Stores the errors and plots boxplots by sample size.

    Parameters
    ----------
    beta : np.ndarray of shape (p,)
        True regression coefficients used for simulation.
    prior : dict with keys 'mu' and 'Sigma'
        Prior mean and covariance for β.
    sample_sizes : list of int
        List of sample sizes to evaluate.
    n_reps : int
        Number of repetitions for each sample size.
    n_iter : int
        Number of iterations for SVI in each run.
    tau : float
        Learning rate schedule parameter (offset).
    kappa : float
        Learning rate decay exponent (in (0.5, 1]).
    seed : int, optional
        Random seed for reproducibility. Default is 123.

    Returns
    -------
    None
        Displays a boxplot of L2 errors across sample sizes.

    Notes
    -----
    - Requires a working `logit_svi` function.
    - Adds intercept to design matrix automatically.
    - Uses `tqdm` progress bars for tracking repetitions.
    """
    errors: Dict[int, list[float]] = {n: [] for n in sample_sizes}
    rng = np.random.default_rng(seed)
    p = beta.size

    for n in tqdm(sample_sizes, desc="Sample sizes"):
        for _ in tqdm(range(n_reps), desc=f" Reps (n={n})", leave=False):
            # simulate design and response
            x = rng.uniform(-2, 2, size=(n, p-1))
            X = np.column_stack((np.ones(n), x))
            y = rng.binomial(1, expit(X @ beta))
            # run SVI
            result = logit_svi(
                X,
                y,
                prior_params=prior,
                n_iter=n_iter,
                tau=tau,
                kappa=kappa,
                verbose=False,
                seed=int(rng.integers(1e9))
            )
            err = np.linalg.norm(result['mu'] - beta)
            errors[n].append(err)

    # boxplot
    data = [errors[n] for n in sample_sizes]
    plt.figure()
    plt.boxplot(data, labels=sample_sizes)
    plt.xlabel('Sample Size')
    plt.ylabel(r'$L_2$ Error')
    plt.title('SVI/True $L_2$ Error vs. Sample Size')
    plt.tight_layout()
    plt.show()

beta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
prior = {'mu': np.zeros_like(beta), 'Sigma': np.eye(len(beta))}
plot_svi_true_l2_error(beta, prior, sample_sizes=[200,500,1000,5000,10000,20000], 
                       n_reps=200, n_iter=10_000, tau=1.0,
                       kappa=0.75, seed=123)