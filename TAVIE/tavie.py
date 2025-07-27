# =============================================================================
# Copyright 2025. Somjit Roy and Pritam Dey. 
# This program implements the TAVIE algorithm as developed in:
# Roy, S., Dey, P., Pati, D., and Mallick, B.K.
# 'A Generalized Tangent Approximation Framework for Strongly Super-Gaussian Likelihoods',
# arXiv:2504.05431 <https://arxiv.org/abs/2504.05431>.
#
# Authors:
#   Somjit Roy <sroy_123@tamu.edu> and Pritam Dey <pritam.dey@tamu.edu>
# =============================================================================

# Required imports
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln
from .utils import *

def TAVIE_ls(
    X: np.ndarray,
    y: np.ndarray,
    A_func,
    cfunc=None,
    V0: np.ndarray = None,
    m0: np.ndarray = None,
    a0: float = 0.05,
    b0: float = 0.05,
    alpha: float = 1.0,
    maxiter: int = 1000,
    tol: float = 1e-9,
    verbose = True,
    **kwargs
) -> dict:
    """
    Performs tangent approximation based variational inference for 
    location-scale strongly super-Gaussian likelihoods.
    
    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix.
    y : np.ndarray of shape (n,)
        Response vector.
    A_func : callable
        Function of xi used to define the tangent approximation matrix. 
        Should return an array of shape (n,).
    cfunc : callable, optional
        Function for computing ELBO correction term. If provided, ELBO will be recorded.
    V0 : np.ndarray of shape (p, p), optional
        Prior covariance matrix for the regression coefficients. Defaults to identity.
    m0 : np.ndarray of shape (p,), optional
        Prior mean for the regression coefficients. Defaults to zero vector.
    a0 : float, optional
        Shape parameter of the inverse-gamma prior on scale. Default is 0.05.
    b0 : float, optional
        Rate parameter of the inverse-gamma prior on scale. Default is 0.05.
    alpha : float, optional
        Data-fidelity scaling factor in the variational objective. Default is 1.0.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Tolerance for convergence based on change in `xi`. Default is 1e-9.
    verbose : bool, optional
        If True, prints convergence info. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to A_func and cfunc.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - 'm': posterior mean of beta (np.ndarray of shape (p,))
        - 'V': posterior covariance of beta (np.ndarray of shape (p, p))
        - 'a': updated shape parameter of the inverse-gamma (float)
        - 'b': updated rate parameter of the inverse-gamma (float)
        - 'elbo': list of ELBO values during iterations (only if `cfunc` is provided)

    Notes
    -----
    The variational family is:
        q(β, τ²) = N(β | m, V) × Inv-Gamma(τ² | a, b)

    The Evidence Lower Bound (ELBO) has the closed-form expression:
        ELBO = 
            - (a / 2) * log(b / 2)
            + log Γ(a / 2)
            + α * Σ c(ξ_i)
            + (1/2) * log|V|

    where:
        a = a₀ + n·α,
        b = b₀ - 2·α·Σ A(ξ_i)·y_i² + m₀ᵀ V₀⁻¹ m₀ - mᵀ V_xi⁻¹ m,
        c(ξ) is a convex correction term defined by the likelihood,
        and A(ξ) is the negative curvature approximation of the log-likelihood.

    This framework supports Laplace, Student's-t, and other super-Gaussian likelihoods 
    by appropriately choosing A_func and cfunc.
    """
    
    n, p = X.shape
    
    # default priors
    if V0 is None:
        V0 = np.eye(p)
    if m0 is None:
        m0 = np.zeros(p)
    
    # precompute prameters used several times
    V0_inv = stablesolve(V0)
    V0_inv_m0 = V0_inv @ m0
    m0_V0_inv_m0 = m0 @ V0_inv_m0
    anew = a0 + (n * alpha)
    
    # initialization
    iter_count = 0
    xi = np.ones_like(y)
    m_xi = m0.copy()
    V_xi = V0.copy()
    b_xi = b0

    elbo = []
    
    while True:
        xi_prev = xi.copy()

        # xi update
        k1 = (anew / b_xi) * (y - X @ m_xi) ** 2
        k2 =  np.einsum('ij,ij->i', X @ V_xi, X)
        xi = np.sqrt( k1 + k2)

        # A_xi update
        A_xi = A_func(xi, **kwargs)
        
        # V_xi update
        V_xi_inv = V0_inv - 2 * alpha * X.T @ (X * A_xi[:, None])
        V_xi = stablesolve(V_xi_inv)

        # m_xi update
        m_xi = V_xi @ (V0_inv_m0 - 2 * alpha * (X.T * A_xi).dot(y))

        # b_xi update
        b_xi = b0 - 2 * alpha * A_xi.dot(y ** 2) + m0_V0_inv_m0 - m_xi @ V_xi_inv @ m_xi

        logdetV_xi = logdet(V_xi)
        if cfunc is not None:
            elbo.append(-(anew / 2) * np.log(b_xi / 2) + gammaln(anew / 2) + alpha * np.sum(cfunc(xi, **kwargs)) + 0.5 * logdetV_xi)

        delta_xi = np.linalg.norm(xi - xi_prev)
        if delta_xi < tol:
            if(verbose):
                print(f"Converged in {iter_count} iterations.")
            break
        
        iter_count += 1
        if iter_count >= maxiter:
            print("Warning: reached maximum iterations before convergence.")
            break

    if cfunc is None:
        return {'m': m_xi, 'V': V_xi, 'b': b_xi, 'a': anew}
    else:
        return {'m': m_xi, 'V': V_xi, 'b': b_xi, 'a': anew, 'elbo': elbo}


def A_func_laplace(xi: np.ndarray):
    """
    Computes the A(xi) function for the Laplace likelihood under the 
    tangent approximation framework used in TAVIE.

    This function is used in the local quadratic approximation of the 
    log-likelihood of the Laplace distribution, and corresponds to the 
    negative second derivative approximation term:
        A(xi) = -1 / (2 * xi)

    Parameters
    ----------
    xi : np.ndarray of shape (n,)
        Current variational scaling variables (one per observation), 
        updated iteratively in the TAVIE algorithm.

    Returns
    -------
    np.ndarray of shape (n,)
        The evaluated A(xi) values used in the variational update of the 
        covariance matrix.
    
    Notes
    -----
    - This formulation corresponds to the log-likelihood of the Laplace 
      distribution: log p(y | Xβ, τ) ∝ -|y - Xβ| / τ.
    - The negative Hessian approximation of the log-likelihood w.r.t. β 
      gives rise to this A(xi) form in the variational updates.
    """
    return -1 / (2 * xi)

def cfunc_laplace(xi: np.ndarray):
    """
    Computes the correction term c(xi) for the Evidence Lower Bound (ELBO)
    when using a Laplace (double exponential) likelihood in the TAVIE framework.
 
    This function captures the part of the Laplace log-likelihood that is not 
    accounted for by the local quadratic (second-order) approximation. It ensures 
    that the ELBO remains a valid lower bound during optimization.
 
    Mathematically, for the Laplace likelihood:
        log p(y | Xβ, τ) ≈ quadratic surrogate + c(xi),
    where:
        c(xi) = -xi / 2
 
    Parameters
    ----------
    xi : np.ndarray of shape (n,)
        Current variational scaling variables used in the local variational approximation.
 
    Returns
    -------
    np.ndarray of shape (n,)
        The correction term values c(xi) = -xi / 2 for each xi.
 
    Notes
    -----
    - This function is used when computing the ELBO term in the TAVIE algorithm
      when a Laplace likelihood is assumed.
    """
    return -xi / 2

def A_func_student(xi: np.ndarray, 
                   nu: float):
    """
    Computes the A(xi) function for the Student's-t likelihood under the 
    tangent variational approximation in the TAVIE framework.

    This function provides a local negative curvature (second-derivative) 
    approximation to the log-likelihood of the Student's-t distribution.

    Parameters
    ----------
    xi : np.ndarray of shape (n,)
        Variational local scale variables.
    nu : float
        Degrees of freedom for the Student's-t distribution. Must be positive.

    Returns
    -------
    np.ndarray of shape (n,)
        Values of A(xi) = -0.5 * (ν + 1) / (ν + xi^2)

    Notes
    -----
    - The Student's-t distribution log-likelihood is approximated by a 
      quadratic lower bound in the TAVIE framework, with A(xi) capturing
      the curvature of the surrogate.
    """
    return -0.5 * (nu + 1)/(nu + xi ** 2)

def cfunc_student(xi: np.ndarray,
                 nu:float):
    """
    Computes the ELBO correction term c(xi) for the Student's-t likelihood 
    in the TAVIE variational framework.

    This function captures the difference between the true log-likelihood 
    and its quadratic surrogate, ensuring the ELBO remains a valid bound.

    Parameters
    ----------
    xi : np.ndarray of shape (n,)
        Variational local scale variables.
    nu : float
        Degrees of freedom of the Student's-t distribution. Must be positive.

    Returns
    -------
    np.ndarray of shape (n,)
        The correction term used in the ELBO:
            c(xi) = -0.5 * (ν + 1) * [log(1 + xi² / ν) + xi² / (xi² + ν)]

    Notes
    -----
    - This function is used when computing the ELBO term in the TAVIE algorithm
      when a Student's-t likelihood is assumed.
    """
    res1 = -0.5 * (nu + 1) * np.log1p((xi ** 2) / nu)
    res2 = -0.5 * (nu + 1) * ((xi ** 2) / ((xi ** 2) + nu))
    
    return res1 + res2


def TAVIE_qr(
    X: np.ndarray,
    y: np.ndarray,
    V0: np.ndarray = None,
    m0: np.ndarray = None,
    u: float = 0.5,
    alpha_tau0: float = 1.0, # Here alpha and tau0 always occur as product
    maxiter: int = 1000,
    tol: float = 1e-9,
    verbose: bool = True
) -> dict:
    """
    Performs tangent approximation based variational inference for quantile regression 
    using the TAVIE framework.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix.
    y : np.ndarray of shape (n,)
        Response vector.
    V0 : np.ndarray of shape (p, p), optional
        Prior covariance matrix for regression coefficients. Defaults to identity.
    m0 : np.ndarray of shape (p,), optional
        Prior mean vector. Defaults to zero vector.
    u : float, optional
        Quantile level (e.g., 0.5 for median). Default is 0.5.
    alpha_tau0 : float, optional
        Product of data-fidelity weight α and precision of scale (τ₀). Default is 1.0.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Tolerance for convergence based on change in ξ. Default is 1e-9.
    verbose : bool, optional
        If True, prints convergence status. Default is True.

    Returns
    -------
    dict
        A dictionary containing:
        - 'm'    : np.ndarray of shape (p,), variational posterior mean of β
        - 'V'    : np.ndarray of shape (p, p), variational posterior covariance of β
        - 'elbo' : list of float, Evidence Lower Bound (ELBO) values across iterations

    Notes
    -----
    This algorithm approximates the asymmetric Laplace log-likelihood using a 
    quadratic surrogate derived from the convex dual form:
        log p(y_i | Xβ) ≥ -|y_i - x_iᵀβ| / τ + constant

    The auxiliary variable ξ allows a closed-form quadratic approximation, leading to
    tractable updates of the posterior mean and covariance.

    The ELBO for this model is given by:
        ELBO = 
            + 0.5 · mᵀ V⁻¹ m 
            + 0.5 · log|V|
            + ατ₀ · Σ c(ξ)
            + ατ₀ · Σ A(ξ) · y²

    where:
        - A(ξ) = -1 / (2ξ)
        - c(ξ) = -ξ / 2
        - m, V are the variational parameters for β
        - ατ₀ = alpha_tau0 is the product of fidelity and scale precision

    This formulation supports different quantile levels by adjusting `u`.
    """
    n, p = X.shape
    
    # default priors
    if V0 is None:
        V0 = np.eye(p)
    if m0 is None:
        m0 = np.zeros(p)
    
    # precompute prameters used several times
    V0_inv = stablesolve(V0)
    V0_inv_m0 = V0_inv @ m0
    bu = 2 * u - 1
    
    # initialization
    iter_count = 0
    xi = np.ones_like(y)
    m_xi = m0.copy()
    V_xi = V0.copy()

    elbo = []
    while True:
        xi_prev = xi.copy()

        # xi update
        k1 =  (y - X @ m_xi) ** 2
        k2 =  np.einsum('ij,ij->i', X @ V_xi, X)
        xi = np.sqrt( k1 + k2)

        # A_xi update
        A_xi = -1 / (2 * xi)
        
        # V_xi update
        V_xi_inv = V0_inv - 2 * alpha_tau0 * X.T @ (X * A_xi[:, None])
        V_xi = stablesolve(V_xi_inv)

        # m_xi update
        m_xi = V_xi @ (V0_inv_m0 - 2 * alpha_tau0 * (X.T * A_xi).dot(y) + alpha_tau0 * bu * X.sum(axis = 0))

        cvec = alpha_tau0 * cfunc_laplace(xi)
        elbo1 = 0.5 * m_xi @ (V_xi_inv @ m_xi)
        elbo2 = 0.5 * logdet(V_xi)
        elbo3 = np.sum(cvec)
        elbo4 = alpha_tau0 * np.sum((y ** 2) * A_xi)
        
        elbo.append(elbo1 + elbo2 + elbo3 + elbo4)
        
        delta_xi = np.linalg.norm(xi - xi_prev)
        if delta_xi < tol:
            if verbose:
                print(f"Converged in {iter_count} iterations.")
            break
        
        iter_count += 1
        if iter_count >= maxiter:
            print("Warning: reached maximum iterations before convergence.")
            break
    
    return {'m': m_xi, 'V': V_xi, 'elbo': elbo}

def TAVIE_bern(
    X: np.ndarray,
    avec: np.ndarray,
    bvec: np.ndarray,
    V0: np.ndarray = None,
    m0: np.ndarray = None,
    alpha: float = 1.0,
    maxiter: int = 1000,
    tol: float = 1e-9,
    verbose: bool = True
) -> dict:
    """
    Performs tangent approximation based variational inference for logistic/Bernoulli 
    regression using the TAVIE framework.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix.
    avec : np.ndarray of shape (n,)
        Vector of response indicators (0 or 1). Appears in the log-likelihood.
    bvec : np.ndarray of shape (n,)
        Scaling weights (e.g., 1 for standard logistic).
    V0 : np.ndarray of shape (p, p), optional
        Prior covariance matrix for the coefficients. Defaults to identity.
    m0 : np.ndarray of shape (p,), optional
        Prior mean vector. Defaults to zero vector.
    alpha : float, optional
        Data fidelity scaling factor. Default is 1.0.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Convergence tolerance. Default is 1e-9.
    verbose : bool, optional
        Whether to print convergence information. Default is True.

    Returns
    -------
    dict
        Dictionary with:
        - 'm': Variational mean of regression coefficients (np.ndarray of shape (p,))
        - 'V': Variational covariance (np.ndarray of shape (p, p))
        - 'elbo' : list of float, Evidence Lower Bound (ELBO) values across iterations

    Notes
    -----
    - The ELBO is computed and stored at each iteration:
        ELBO = 0.5 * mᵀ V⁻¹ m + 0.5 * log|V| + Σᵢ bᵢ * c(ξᵢ),
      where c(ξᵢ) = ξᵢ/4 * tanh(ξᵢ/2) - log(2cosh(ξᵢ/2)).
    """
    n, p = X.shape
    
    # default priors
    if V0 is None:
        V0 = np.eye(p)
    if m0 is None:
        m0 = np.zeros(p)
    
    # precompute prameters used several times
    V0_inv = stablesolve(V0)
    V0_inv_m0 = V0_inv @ m0
    var_inv_mean = V0_inv @ m0 + alpha * X.T @ (avec - bvec/2)
    
    # initialization
    iter_count = 0
    xi = np.ones_like(avec)
    m_xi = m0.copy()
    V_xi = V0.copy()

    elbo = []
    
    while True:
        xi_prev = xi.copy()

        # xi update
        k1 = (X @ m_xi) ** 2
        k2 = np.einsum('ij,ij->i', X @ V_xi, X)
        xi = np.sqrt( k1 + k2)

        # A_xi update
        A_xi = -bvec * np.tanh(xi / 2) / (4 * xi)
        
        # V_xi update
        V_xi_inv = V0_inv - 2 * alpha * X.T @ (X * A_xi[:, None])
        V_xi = stablesolve(V_xi_inv)

        # m_xi update
        m_xi = V_xi @ var_inv_mean

        cvec = (xi / 4) * np.tanh(xi / 2) - np.log(2 * np.cosh(xi / 2))
        elbo1 = 0.5 * m_xi @ (V_xi_inv @ m_xi)
        elbo2 = 0.5 * logdet(V_xi)
        elbo3 = np.sum(bvec * cvec)
        
        elbo.append(elbo1 + elbo2 + elbo3)
        
        delta_xi = np.linalg.norm(xi - xi_prev)
        if delta_xi < tol:
            if(verbose):
                print(f"Converged in {iter_count} iterations.")
            break
        
        iter_count += 1
        if iter_count >= maxiter:
            print("Warning: reached maximum iterations before convergence.")
            break
    
    return {'m': m_xi, 'V': V_xi, 'elbo': elbo}
