# =============================================================================
# Copyright 2025. Somjit Roy and Pritam Dey. 
# This program implements mean-field variational inference (MFVI) for 
# Student's-t likelihood (following Wand et al., 2011 <doi:10.1214/11-BA631>) 
# and Bayesian logistic regression (following Durante and Rigon, 2019 
# <https://doi.org/10.1214/19-STS712>) in order to compare with TAVIE against 
# the TAVIE algorithm as developed in:
# Roy, S., Dey, P., Pati, D., and Mallick, B.K.
# 'A Generalized Tangent Approximation Framework for Strongly Super-Gaussian Likelihoods',
# arXiv:2504.05431 <https://arxiv.org/abs/2504.05431>.
#
# Authors:
#   Somjit Roy <sroy_123@tamu.edu> and Pritam Dey <pritam.dey@tamu.edu>
# =============================================================================

# Required imports
import numpy as np
from scipy.special import digamma, gammaln
from scipy.special import expit  
from typing import Dict

# =============================================================================
# Helper function to do MFVB for the Student's t model
# =============================================================================
def _compute_nu_moments(n, C1, nu_min, nu_max, num_grid=500):
    """
    Computes the zeroth and first moments (F0 and F1) of a non-normalized 
    weight function w(ν) over a grid of ν values in [nu_min, nu_max], using 
    a numerically stable log-domain Riemann sum approximation.

    The integrals approximated are:
        F0 = ∫ w(ν) dν,
        F1 = ∫ ν·w(ν) dν,

    where:
        log w(ν) = n * [ (ν/2) * log(ν/2) - log Γ(ν/2) ] - (ν/2) * C1

    Parameters
    ----------
    n : int
        Number of observations (used to scale log-likelihood term).
    C1 : float
        Data-dependent constant (typically a function of the sufficient statistics).
    nu_min : float
        Minimum value of ν to consider in the grid.
    nu_max : float
        Maximum value of ν to consider in the grid.
    num_grid : int, optional
        Number of evenly spaced ν points in the grid (default is 500).

    Returns
    -------
    F0 : float
        Zeroth moment: approximated ∫ w(ν) dν
    F1 : float
        First moment: approximated ∫ ν·w(ν) dν

    Notes
    -----
    - The function uses a log-domain evaluation with max-shift stabilization to 
      prevent numerical underflow or overflow in the exponential of log weights.
    - The returned ratio F1 / F0 is the expected value of ν under the normalized w(ν).

    """
    # 1) grid
    nus = np.linspace(nu_min, nu_max, num_grid)
    # 2) log-weights
    logw = n * (nus/2*np.log(nus/2) - gammaln(nus/2)) - (nus/2)*C1
    # 3) shift to avoid underflow
    logw -= logw.max()
    w = np.exp(logw)
    # 4) Riemann-step
    h = (nu_max - nu_min) / (num_grid - 1)
    F0 = w.sum() * h
    F1 = (nus * w).sum() * h
    return F0, F1

# =============================================================================
# MFVB for the student's t model
# =============================================================================

def MFVI_Student(X, y,
                      mu_beta, Sigma_beta,
                      A, B,
                      nu_min, nu_max,
                      max_iter=500, tol=1e-6, nu_grid=500,
                      verbose=True):
    """
    MFVB for Student-t linear regression.
    Inputs
    ------
      X : (n×p) design matrix
      y : (n,)    responses
      mu_beta : (p,)    prior mean
      Sigma_beta: (p×p) prior cov
      A,B      : IG(A,B) prior on sigma^2
      nu_min,nu_max : support for uniform prior on nu
    Returns
    -------
      beta_hat : posterior mean of β
      sigma2_hat : posterior mean of σ^2
      nu_hat     : posterior mean of ν
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n,p = X.shape

    # Precompute precision of prior on beta
    Sigma_beta_inv = np.linalg.inv(Sigma_beta)

    # Initialize VB parameters
    beta_q      = np.linalg.solve(X.T@X, X.T@y)        # OLS start
    V_beta_q    = Sigma_beta                # start with prior
    sigma2_inv_q = 1.0/np.var(y)            # E[1/σ²]
    nu_q        = 0.5*(nu_min+nu_max)
    a_q         = np.ones(n)                # E[a_i]
    loga_q      = np.zeros(n)               # log-rate term

    # For convergence checking
    prev = np.hstack([beta_q, [1/sigma2_inv_q, nu_q]])

    for it in range(1, max_iter+1):
        # 1) Update local a_i's
        #    shape = (nu_q + 1)/2,  rate = (nu_q + E[(y-Xβ)^2]/σ²)/2
        resid_mean = y - X@beta_q
        # E[(y - x^T β)^2] = resid_mean^2 + diag(X Vβ X^T)
        VXt = X @ V_beta_q
        resid_var = np.sum(VXt * X, axis=1)
        shape_ai = 0.5*(nu_q + 1)
        rate_ai  = 0.5*(nu_q + sigma2_inv_q*(resid_mean**2 + resid_var))
        a_q      = shape_ai / rate_ai
        loga_q   = np.log(rate_ai) - digamma(shape_ai)

        # 2) Update q(β) = N(beta_q, V_beta_q)
        W = sigma2_inv_q * a_q                  # weights for each obs.
        XtW = X.T * W                          # p×n matrix
        V_beta_q = np.linalg.inv(Sigma_beta_inv + XtW @ X)
        beta_q   = V_beta_q @ (Sigma_beta_inv@mu_beta + X.T@(W*y))

        # 3) Update q(ν) via grid
        #    C1 = ∑[loga_q + a_q]   (matches scalar case)
        C1 = np.sum(loga_q + a_q)
        F0,F1 = _compute_nu_moments(n, C1, nu_min, nu_max, num_grid=nu_grid)
        nu_q = F1 / F0

        # 4) Update q(σ²) = IG(A + n/2,  B + 0.5∑a_i E[(y−Xβ)^2])
        shape_s   = A + 0.5*n
        rate_s    = B + 0.5*np.sum(a_q*(resid_mean**2 + resid_var))
        sigma2_inv_q = shape_s / rate_s

        # 5) Check convergence (β-vector, σ², ν)
        curr = np.hstack([beta_q, [1/sigma2_inv_q, nu_q]])
        rel   = np.abs(curr - prev)/(np.abs(prev)+1e-12)
        if np.max(rel) < tol:
            if(verbose):
                print(f"Converged in {it} iters.")
            break
        prev = curr.copy()
    else:
        print("Warning: max_iter reached without full convergence.")

    sigma2_q = 1.0/sigma2_inv_q
    return beta_q, sigma2_q, nu_q

# =============================================================================
# MFVB for the logistic regression model
# =============================================================================

# =============================================================================
# Variational inference for Bayesian logistic regression using CAVI algorithm
# =============================================================================

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
# Variational inference for Bayesian logistic regression using SVI algorithm
# =============================================================================

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