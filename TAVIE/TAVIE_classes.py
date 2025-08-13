# =============================================================================
# Copyright 2025. Somjit Roy and Pritam Dey. 
# This program implements different classes for the TAVIE algorithm as developed in:
# Roy, S., Dey, P., Pati, D., and Mallick, B.K.
# 'A Generalized Tangent Approximation Framework for Strongly Super-Gaussian Likelihoods',
# arXiv:2504.05431 <https://arxiv.org/abs/2504.05431>.
#
# Authors:
#   Somjit Roy <sroy_123@tamu.edu> and Pritam Dey <pritam.dey@tamu.edu>
# =============================================================================

# Required imports
from .tavie import *
import numpy as np
import pandas as pd
import rich
from sklearn.preprocessing import scale
from IPython.display import display, Latex
import matplotlib.pyplot as plt

# Valdating prior parameters for the location-scale family
def validate_prior_params_loc_scale(prior_params, dim):
    """
    Validate and unpack prior parameters for location-scale models.

    This function checks the correctness of prior parameters provided in the form:
        prior_params = [m0, V0, a0, b0],
    where:
        - m0: prior mean vector of β,
        - V0: prior covariance matrix of β,
        - a0: shape parameter of the inverse-gamma prior on scale (τ²),
        - b0: rate parameter of the inverse-gamma prior on scale (τ²).

    Parameters
    ----------
    prior_params : list or tuple of length 4
        The prior parameter set: [m0, V0, a0, b0].
    dim : int
        Expected dimension of m0 and V0 (number of regression coefficients).

    Returns
    -------
    m0 : np.ndarray of shape (dim,)
        Validated prior mean vector.
    V0 : np.ndarray of shape (dim, dim)
        Validated prior covariance matrix (must be symmetric positive definite).
    a0 : float
        Shape parameter (must be positive).
    b0 : float
        Rate parameter (must be positive).

    Raises
    ------
    ValueError
        If the input is malformed, dimensions mismatch, or parameters are invalid.
    """

    if not isinstance(prior_params, (list, tuple)) or len(prior_params) != 4:
        raise ValueError("prior_params must be a list of four elements: [m0, V0, a0, b0]")
        
    m0, V0, a0, b0 = prior_params

    m0 = np.asarray(m0)
    V0 = np.asarray(V0)

    # Check shapes
    if m0.ndim not in (1, 2) or m0.shape[0] != dim:
        raise ValueError(f"m0 must be a vector of length {dim}")
    if V0.ndim != 2 or V0.shape != (dim, dim):
        raise ValueError(f"V0 must be a square matrix of shape ({dim},{dim})")

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(V0)
    if np.any(eigvals <= 0):
        raise ValueError("V0 must be positive definite")

    # Scalars
    if not (isinstance(a0, (int, float)) and a0 > 0):
        raise ValueError("a0 must be a positive scalar")
    if not (isinstance(b0, (int, float)) and b0 > 0):
        raise ValueError("b0 must be a positive scalar")

    return m0, V0, float(a0), float(b0)

# Valdating prior parameters for the Type II family
def validate_prior_params_type_II(prior_params, dim):
    """
    Validate and unpack prior parameters for type-II SSG models.

    Parameters
    ----------
    prior_params : list or tuple of length 2
        The prior parameter set: [m0, V0], where:
        - m0 is the prior mean vector of β,
        - V0 is the prior covariance matrix of β.
    dim : int
        Expected dimension of m0 and V0.

    Returns
    -------
    m0 : np.ndarray of shape (dim,)
        Validated prior mean vector.
    V0 : np.ndarray of shape (dim, dim)
        Validated prior covariance matrix (must be symmetric positive definite).

    Raises
    ------
    ValueError
        If the input is malformed, dimensions mismatch, or the matrix is not positive definite.
    """

    if not isinstance(prior_params, (list, tuple)) or len(prior_params) != 2:
        raise ValueError("prior_params must be a list of two elements: [m0, V0]")
        
    m0, V0 = prior_params

    m0 = np.asarray(m0)
    V0 = np.asarray(V0)

    # Check shapes
    if m0.ndim not in (1, 2) or m0.shape[0] != dim:
        raise ValueError(f"m0 must be a vector of length {dim}")
    if V0.ndim != 2 or V0.shape != (dim, dim):
        raise ValueError(f"V0 must be a square matrix of shape ({dim},{dim})")

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(V0)
    if np.any(eigvals <= 0):
        raise ValueError("V0 must be positive definite")

    return m0, V0

# =============================================================================
# The TAVIE class for Quantile Regression
# =============================================================================
class TAVIE_QR:
    """
    TAVIE for Quantile Regression (QR).

    This class provides a high-level interface to fit the TAVIE model for asymmetric
    Laplace likelihoods used in quantile regression. It supports scaling, intercept
    addition, ELBO tracking, and getting both the TAVIE means & variational estimates.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term in the design matrix.

    scale_X : bool, default=False
        If True, standardizes X before fitting.

    scale_y : bool, default=False
        If True, standardizes y before fitting.

    Attributes
    ----------
    is_fitted : bool
        Indicates whether the model has been fitted.
    fitted_values : dict
        Stores the posterior mean, covariance, and ELBO from the fitted model.
    """
    def __init__(self, 
                 fit_intercept: bool = True, 
                 scale_X: bool = False,
                 scale_y: bool=False):
        """
        Initialize the TAVIE model parameters.
        """
        
        self.fit_intercept = fit_intercept
        self.scale_X = scale_X
        self.scale_y = scale_y
        self.is_fitted = False

    def fit(self, 
            X: np.ndarray,
            y: np.ndarray,
            quantile: float = 0.5,
            prior_params=None,
            alpha: float = 1.0,
            maxiter: int = 1000,
            tol: float = 1e-9,
            verbose = True):
        """
        Fit the TAVIE model to the input data for quantile regression.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
            Input design matrix.
        y : np.ndarray of shape (n,)
            Response vector.
        quantile : float, default=0.5
            Desired quantile level (e.g., 0.5 for median regression).
        prior_params : list or tuple, optional
            Prior parameters [m0, V0] for the Gaussian prior on coefficients.
            If None, uses zero-mean and identity covariance.
        alpha : float, default=1.0
            Product of data fidelity weight and scale precision.
        maxiter : int, default=1000
            Maximum number of iterations for optimization.
        tol : float, default=1e-9
            Tolerance threshold for convergence.
        verbose : bool, default=True
            Whether to print convergence information.
        """
        self.X = X
        self.y = y
        self.n, self.p = self.X.shape
        self.design_matrix = np.copy(self.X)
        self.alpha = alpha
        self.quantile = quantile

        if self.scale_y:
            self.y = scale(self.y, with_mean=True, with_std=True)
        
        if self.scale_X:
            self.design_matrix = scale(self.design_matrix, with_mean=True, with_std=True)
            
        if self.fit_intercept:
            self.design_matrix = np.column_stack((np.ones(self.n), self.design_matrix))
            self.p += 1
            
        ################################################################
        if verbose:
            try:
                from rich.console import Console
                from rich.panel import Panel
                RICH_AVAILABLE = True
                console = Console()
            except ImportError:
                RICH_AVAILABLE = False
                
            if RICH_AVAILABLE:
                console.print(
                    Panel(
                        "[bold green] Starting TAVIE fit![/]",
                        title=f"[bold blue]TAVIE Fit for Quantile Regression[/]",
                        border_style="magenta",
                        expand=False
                    )
                )
            else:
                print(f"Starting TAVIE fit for Quantile Regression!")
        ################################################################

        ################################################################
        ### prior parameter validation for quantile regression
        ################################################################
        
        if(prior_params == None):
            m0 = np.zeros(self.p)
            V0 = np.eye(self.p)
        else:
            m0, V0 = validate_prior_params_type_II(prior_params=prior_params, dim=self.p)

        ################################################################
        ### TAVIE for quantile regression
        ################################################################
        self.fitted_values = TAVIE_qr(X = self.design_matrix, y = self.y, 
                                        V0 = V0, m0 = m0, u=self.quantile,
                                        alpha_tau0 = alpha, 
                                        maxiter = maxiter, tol = tol, 
                                        verbose = verbose)
        
        self.is_fitted = True

    def get_variational_estimates(self):
        """
        Returns the variational estimates of model parameters.

        Returns
        -------
        dict
            Dictionary containing:
            - 'm_xi': Posterior mean of coefficients.
            - 'V_xi': Posterior covariance matrix.
        """
        return {'m_xi': self.fitted_values['m'], 'V_xi': self.fitted_values['V']}
    
    def get_elbo(self):
        """
        Returns the Evidence Lower Bound (ELBO) trajectory during training.

        Returns
        -------
        np.ndarray
            Array of ELBO values at each iteration.
        """
        ELBO = self.fitted_values['elbo']
        return np.array(ELBO)

    def get_TAVIE_means(self, verbose=True):
        """
        Display and return the posterior mean of the regression coefficients.

        Returns
        -------
        np.ndarray
            Posterior mean vector of regression coefficients.

        Raises
        ------
        Exception
            If the model has not been fitted.
        """
        if self.is_fitted == False:
            raise Exception("TAVIE model is not trained yet. Call fit() first.")

        res = self.fitted_values
        beta = res['m']
        if verbose:
            pass#display(Latex(f'$E(\\beta) = {np.array2string(beta, separator=', ')}$'))
        return beta

# =============================================================================
# The TAVIE class for SSG location-scale family
# =============================================================================
class TAVIE_loc_scale:
    """
    TAVIE for SSG Location-Scale Families.

    This class implements variational inference under the TAVIE framework 
    for a variety of location-scale likelihoods:
      - Laplace likelihood
      - Student's t likelihood (with user-specified degrees of freedom `nu`)
      - Generic strongly super-Gaussian likelihoods via user-defined `A_func` and `cfunc`

    It performs iterative updates for variational parameters (m, V, a, b)
    and optionally tracks the Evidence Lower Bound (ELBO). Also obtains the TAVIE means &
    variational estimates.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term in the design matrix.
    scale_X : bool, default=False
        Whether to standardize the features X before fitting.
    scale_y : bool, default=False
        Whether to standardize the response vector y before fitting.
    family : {'laplace', 'student', 'loc_scale'}, required
        Specifies which likelihood family to use.
        - 'laplace' : Laplace likelihood.
        - 'student' : Student's t likelihood.
        - 'loc_scale' : Custom location-scale likelihood (requires `afunc`).
    afunc : callable, optional
        Function that computes A(ξ) for custom location-scale families. 
        Required if family='loc_scale'.
    cfunc : callable, optional
        Function that computes the c(ξ) term for ELBO in custom families.
        Required if family='loc_scale'.
    """
    def __init__(self, 
                 fit_intercept: bool = True, 
                 scale_X: bool = False,
                 scale_y: bool = False,
                 family: str = None,
                 afunc=None, cfunc=None):
        """
        Initialize the TAVIE model parameters.
        """
        if (family not in ['laplace', 'student', 'loc_scale']):
            raise ValueError("select valid family")

        if family == 'loc_scale' and not callable(afunc):
            raise ValueError("`afunc` must be a callable function")
               
        self.fit_intercept = fit_intercept
        self.scale_X = scale_X
        self.scale_y = scale_y
        self.family = family
        self.is_fitted = False

    def fit(self, 
            X: np.ndarray,
            y: np.ndarray,
            prior_params=None,
            alpha: float = 1.0,
            maxiter: int = 1000,
            tol: float = 1e-9,
            verbose = True, **kwargs):
        """
        Fit the TAVIE model to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
            Input design matrix.
        y : np.ndarray of shape (n,)
            Response vector.
        prior_params : list or tuple, optional
            Prior parameters [m0, V0, a0, b0] for the Gaussian-Inverse-Gamma prior.
            If None, defaults to m0=0, V0=I, a0=0.05, b0=0.05.
        alpha : float, default=1.0
            Data fidelity scaling factor.
        maxiter : int, default=1000
            Maximum number of iterations for the TAVIE algorithm.
        tol : float, default=1e-9
            Convergence tolerance on auxiliary variable updates.
        verbose : bool, default=True
            Whether to print convergence information.
        **kwargs : dict
            Additional parameters:
            - For 'student' family, must provide `nu` (degrees of freedom).
            - For custom families, additional parameters passed to `afunc` and `cfunc`.

        Raises
        ------
        ValueError
            If `family` is invalid or `nu` is not provided for Student's t likelihood.
        """
        
        self.X = X
        self.y = y
        self.n, self.p = self.X.shape
        self.design_matrix = np.copy(self.X)
        self.alpha = alpha

        if self.scale_y:
            self.y = scale(self.y, with_mean=True, with_std=True)
        
        if self.scale_X:
            self.design_matrix = scale(self.design_matrix, with_mean=True, with_std=True)
            
        if self.fit_intercept:
            self.design_matrix = np.column_stack((np.ones(self.n), self.design_matrix))
            self.p += 1
            
        ################################################################
        if verbose:
            try:
                from rich.console import Console
                from rich.panel import Panel
                RICH_AVAILABLE = True
                console = Console()
            except ImportError:
                RICH_AVAILABLE = False
                
            if RICH_AVAILABLE:
                console.print(
                    Panel(
                        "[bold green] Starting TAVIE fit![/]",
                        title=f"[bold blue]TAVIE Fit for {self.family}[/]",
                        border_style="magenta",
                        expand=False
                    )
                )
            else:
                print(f"Starting TAVIE fit for {self.family}!")
        ################################################################

        ################################################################
        ### prior parameter validation for location-scale family
        ################################################################
        
        if(prior_params == None):
            m0 = np.zeros(self.p)
            V0 = np.eye(self.p)
            a0 = 0.05
            b0 = 0.05
        else:
            m0, V0, a0, b0 = validate_prior_params_loc_scale(prior_params=prior_params, dim=self.p)

        ################################################################
        ### TAVIE for laplace likelihood
        ################################################################
        if(self.family == "laplace"):
            self.fitted_values = TAVIE_ls(X = self.design_matrix, y = self.y, 
                                            A_func = A_func_laplace, cfunc = cfunc_laplace,
                                            V0 = V0, m0 = m0,
                                            a0 = a0, b0 = b0, alpha = alpha, 
                                            maxiter = maxiter, tol = tol, 
                                            verbose = verbose)
                
        ################################################################
        ### TAVIE for Student's t likelihood
        ################################################################   
        elif(self.family == "student"):
            if('nu' in kwargs.keys()):
                if not (isinstance(kwargs['nu'], (int, float)) and kwargs['nu'] >= 1):
                    raise ValueError("nu must be a scalar and greater than equal to 1")
                else:
                    nu = kwargs['nu']
            else:
                raise ValueError("nu (degrees of freedom for Student's t) is not provided")
                
            self.fitted_values = TAVIE_ls(X = self.design_matrix, y = self.y, 
                                            A_func = A_func_student, cfunc = cfunc_student,
                                            V0 = V0, m0 = m0,
                                            a0 = a0, b0 = b0, alpha = alpha, 
                                            maxiter = maxiter, tol = tol, 
                                            verbose = verbose, nu = nu)

        ################################################################
        ### TAVIE for General location-scale likelihood
        ################################################################   
        elif(self.family == "loc_scale"):
            self.fitted_values = TAVIE_ls(X = self.design_matrix, y = self.y, 
                                            A_func = afunc, cfunc = cfunc,
                                            V0 = V0, m0 = m0,
                                            a0 = a0, b0 = b0, alpha = alpha, 
                                            maxiter = maxiter, tol = tol, 
                                            verbose = verbose, **kwargs)
        
        self.is_fitted = True

    def get_variational_estimates(self):
        """
        Retrieve the variational posterior estimates.

        Returns
        -------
        dict
            A dictionary containing:
            - 'm_xi': np.ndarray of shape (p,), posterior mean of β.
            - 'V_xi': np.ndarray of shape (p, p), posterior covariance of β.
            - 'a_xi': float, shape parameter of the inverse-gamma posterior on τ².
            - 'b_xi': float, rate parameter of the inverse-gamma posterior on τ².
        """
        return {'m_xi': self.fitted_values['m'], 'V_xi': self.fitted_values['V'],
               'a_xi': self.fitted_values['a'], 'b_xi': self.fitted_values['b']}

    def get_elbo(self):
        """
        Retrieve the Evidence Lower Bound (ELBO) values recorded during training.

        Returns
        -------
        np.ndarray
            ELBO values over iterations.
        """
        ELBO = self.fitted_values['elbo']
        return np.array(ELBO)

    def get_TAVIE_means(self, verbose=True):
        """
        Display and return posterior means of model parameters.

        Returns
        -------
        beta : np.ndarray of shape (p,)
            Posterior mean of regression coefficients.
        tau2 : float
            Posterior mean of the scale parameter τ².

        Raises
        ------
        Exception
            If the model is not yet fitted.
        """
        if self.is_fitted == False:
            raise Exception("TAVIE model is not trained yet. Call fit() first.")
            
        res = self.fitted_values
        beta = res['m']
        tau2 = res['a'] / res['b']
        if verbose:
            pass#display(Latex(f'$E(\\beta) = {np.array2string(beta, separator=', ')}$'))
            #display(Latex(f'$E(\\tau^2) = {tau2}$'))
        return beta, tau2

# =============================================================================
# The TAVIE class for Type-II family
# =============================================================================
class TAVIE_type_II:
    """
    TAVIE for Type II Exponential Family Likelihoods.

    Supports generalized linear models (GLMs) with:
      - Binomial likelihood (for binary classification or binomial counts)
      - Negative Binomial likelihood (for overdispersed count data)

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept in the design matrix.
    scale_X : bool, default=False
        Whether to standardize the design matrix columns before fitting.
    family : {'binomial', 'negbin'}
        Specifies the exponential family likelihood:
        - 'binomial' for Binomial/Bernoulli
        - 'negbin' for Negative Binomial
    """
    def __init__(self, 
                 fit_intercept: bool = True, 
                 scale_X: bool = False, 
                 family: str = None):
        """
        Initialize the TAVIE model parameters.
        """
        if (family not in ['binomial', 'negbin']):
            raise ValueError("select valid family")
               
        self.fit_intercept = fit_intercept
        self.scale_X = scale_X
        self.family = family
        self.is_fitted = False

    def fit(self, 
            X: np.ndarray,
            y: np.ndarray,
            r=5.0,
            prior_params=None,
            alpha: float = 1.0,
            maxiter: int = 1000,
            tol: float = 1e-9,
            verbose = True):
        """
        Fit the TAVIE model on training data.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
            Input design matrix.
        y : np.ndarray of shape (n,)
            Response vector.
        r : float or np.ndarray, default=5.0
            - For Binomial: number of trials.
            - For Negative Binomial: dispersion parameter.
            If scalar, broadcasted to all observations.
        prior_params : list or tuple, optional
            Prior parameters [m0, V0] for the Gaussian prior on coefficients.
            If None, defaults to m0=0 and V0=I.
        alpha : float, default=1.0
            Data fidelity scaling factor in the variational objective.
        maxiter : int, default=1000
            Maximum number of coordinate ascent iterations.
        tol : float, default=1e-9
            Convergence tolerance on auxiliary variable updates.
        verbose : bool, default=True
            If True, print convergence messages.
        """
        
        self.X = X
        self.y = y
        self.n, self.p = self.X.shape
        self.design_matrix = np.copy(self.X)
        self.alpha = alpha

        if np.isscalar(r):
            self.r = np.full(self.n, r)
        else:
            self.r = np.asarray(r)
            if self.r.shape[0] != self.n:
                raise ValueError(f"r must be a scalar or a vector of length {self.n}")
        
        if self.scale_X:
            self.design_matrix = scale(self.design_matrix, with_mean=True, with_std=True)
        if self.fit_intercept:
            self.design_matrix = np.column_stack((np.ones(self.n), self.design_matrix))
            self.p += 1
            
        ################################################################
        if verbose:
            try:
                from rich.console import Console
                from rich.panel import Panel
                RICH_AVAILABLE = True
                console = Console()
            except ImportError:
                RICH_AVAILABLE = False
                
            if RICH_AVAILABLE:
                console.print(
                    Panel(
                        "[bold green] Starting TAVIE fit![/]",
                        title=f"[bold blue]TAVIE Fit for {self.family}[/]",
                        border_style="magenta",
                        expand=False
                    )
                )
            else:
                print(f"Starting TAVIE fit for {self.family}!")
        ################################################################

        ################################################################
        ### prior parameter validation for location-scale family
        ################################################################
        
        if(prior_params == None):
            m0 = np.zeros(self.p)
            V0 = np.eye(self.p)
        else:
            m0, V0 = validate_prior_params_type_II(prior_params=prior_params, dim=self.p)

        ################################################################
        ### TAVIE for binomial likelihood
        ################################################################
        if(self.family == "binomial"):
            avec = self.y
            bvec = self.r
            self.fitted_values = TAVIE_bern(X = self.design_matrix, avec = avec, bvec = bvec, 
                                            V0 = V0, m0 = m0,
                                            alpha = alpha, 
                                            maxiter = maxiter, tol = tol, 
                                            verbose = verbose)
                
        ################################################################
        ### TAVIE for Negative-Binomial likelihood
        ################################################################   
        elif(self.family == "negbin"):
            avec = self.r
            bvec = self.r + self.y
            self.fitted_values = TAVIE_bern(X = self.design_matrix, avec = avec, bvec = bvec, 
                                            V0 = V0, m0 = m0,
                                            alpha = alpha, 
                                            maxiter = maxiter, tol = tol, 
                                            verbose = verbose)
        
        self.is_fitted = True

    def get_variational_estimates(self):
        """
        Retrieve the variational posterior estimates.

        Returns
        -------
        dict
            Dictionary containing:
            - 'm_xi': np.ndarray, posterior mean of β.
            - 'V_xi': np.ndarray, posterior covariance of β.
        """
        return {'m_xi': self.fitted_values['m'], 'V_xi': self.fitted_values['V']}

    def get_elbo(self):
        """
        Retrieve the Evidence Lower Bound (ELBO) values recorded during optimization.

        Returns
        -------
        np.ndarray
            Array of ELBO values per iteration.
        """
        ELBO = self.fitted_values['elbo']
        return np.array(ELBO)

    def get_TAVIE_means(self, verbose=True):
        """
        Display and return posterior means of model coefficients.

        Returns
        -------
        beta : np.ndarray
            Posterior mean of regression coefficients.

        Raises
        ------
        Exception
            If model has not been fitted yet.
        """
        if self.is_fitted == False:
            raise Exception("TAVIE model is not trained yet. Call fit() first.")

        res = self.fitted_values
        beta = res['m']
        if verbose:
            pass#display(Latex(f'$E(\\beta) = {np.array2string(beta, separator=', ')}$'))
        return beta