# =============================================================================
# Copyright 2025. Somjit Roy and Pritam Dey. 
# This program implements helper functions for the TAVIE algorithm as developed in:
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


def logdet(A):
    """
    Computes the log-determinant of a positive definite matrix A using its Cholesky decomposition.
    
    This function is numerically more stable than computing the determinant directly and 
    then taking the logarithm, especially for high-dimensional matrices.
    
    Parameters
    ----------
    A : ndarray of shape (n, n)
        A real symmetric positive definite matrix.

    Returns
    -------
    float
        The natural logarithm of the determinant of matrix A.

    """
    L = np.linalg.cholesky(A)        # A = L @ L.T
    return 2 * np.sum(np.log(np.diag(L)))

def stablesolve(A: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a symmetric positive definite matrix A using
    Cholesky decomposition. If the decomposition fails due to numerical
    instability, a small jitter is added to the diagonal for stabilization.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        A real symmetric positive definite matrix to invert.

    Returns
    -------
    res : np.ndarray of shape (n, n)
        The inverse of matrix A.
    """
    c, lower = cho_factor(A, overwrite_a=False, check_finite=True)
    try:
        c, lower = cho_factor(A, overwrite_a=False, check_finite=True)
        res = cho_solve((c, lower), np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        print(1)
        A += np.eye(A.shape[0]) * 1e-8
        c, lower = cho_factor(A, overwrite_a=False, check_finite=True)
        res = cho_solve((c, lower), np.eye(A.shape[0]))
    return res