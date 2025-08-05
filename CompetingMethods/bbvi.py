# =============================================================================
# Copyright 2025. Somjit Roy and Pritam Dey. 
# This program implements black-box variational inference (BBVI) for comparison
# against the TAVIE algorithm as developed in:
# Roy, S., Dey, P., Pati, D., and Mallick, B.K.
# 'A Generalized Tangent Approximation Framework for Strongly Super-Gaussian Likelihoods',
# arXiv:2504.05431 <https://arxiv.org/abs/2504.05431>.
#
# Authors:
#   Somjit Roy <sroy_123@tamu.edu> and Pritam Dey <pritam.dey@tamu.edu>
# =============================================================================

# Required imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Gamma

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Gamma

def BBVI_Laplace_fullcov(
    X: np.ndarray,
    y: np.ndarray,
    Sigma_prior: np.ndarray,
    a: float,
    b: float,
    lr: float = 1e-2,
    max_iters: int = 10000,
    tol: float = 1e-8,
    patience: int = 500
):
    """
    BBVI for Bayesian linear regression with Laplace likelihood,
    prior β | τ² ~ N(0, Σ_prior/τ²) and τ² ~ Gamma(shape=a0, rate=b0),
    using a full‐covariance Gaussian variational posterior q(β) and
    early stopping on ELBO “patience.”
    """
    # 1) Data & prior setup
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t       = torch.from_numpy(X).float().to(device)              # (n, p)
    y_t       = torch.from_numpy(y).float().to(device)              # (n,)
    Sigma_t   = torch.from_numpy(Sigma_prior).float().to(device)    # (p, p)
    Sigma_inv = torch.inverse(Sigma_t)                              # (p, p)
    shape0    = a
    rate0     = b

    # 2) Variational parameters
    q_mu             = nn.Parameter(torch.zeros(p, device=device))
    # unconstrained lower‐triangular matrix for Cholesky factor
    L_unconstrained  = nn.Parameter(torch.eye(p, device=device))
    q_alpha          = nn.Parameter(torch.tensor(0.0, device=device))
    q_log_b          = nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = optim.Adam([q_mu, L_unconstrained, q_alpha, q_log_b], lr=lr)

    def get_scale_tril():
        # build a valid lower‐triangular L with positive diag
        L = torch.tril(L_unconstrained)
        diag = torch.diagonal(L, 0)
        L = L.clone()
        L[range(p), range(p)] = torch.exp(diag)
        return L  # L @ L.T is the full cov

    def log_joint(beta, tau2):
        # 1) Laplace log‐likelihood (w.r.t. precision τ = sqrt(τ2))
        tau    = torch.sqrt(tau2)
        resid  = y_t - X_t @ beta
        log_lik = n * torch.log(tau) - tau * torch.sum(torch.abs(resid))

        # 2) Prior β | τ² ~ N(0, Σ_prior/τ²)
        quad    = beta @ (Sigma_inv @ beta)
        log_p_b = (p/2) * torch.log(tau2) - 0.5 * tau2 * quad

        # 3) Prior τ² ~ Gamma(shape0, rate0)
        log_p_t = (shape0 - 1) * torch.log(tau2) - rate0 * tau2

        return log_lik + log_p_b + log_p_t

    # 3) Early‐stopping bookkeeping
    best_elbo  = -float("inf")
    no_improve = 0
    elbo_hist  = []

    # 4) BBVI loop
    for it in range(max_iters):
        optimizer.zero_grad()

        # 4.1) Sample β ∼ q(β) via MultivariateNormal
        L       = get_scale_tril()
        q_beta  = MultivariateNormal(loc=q_mu, scale_tril=L)
        beta_samp = q_beta.rsample()

        # 4.2) Sample τ² ∼ q(τ²)
        alpha   = torch.nn.functional.softplus(q_alpha)
        rate    = torch.nn.functional.softplus(q_log_b)
        q_tau   = Gamma(concentration=alpha, rate=rate)
        tau2_samp = q_tau.rsample()

        # 4.3) Compute ELBO
        logp      = log_joint(beta_samp, tau2_samp)
        logq_b    = q_beta.log_prob(beta_samp)
        logq_t    = q_tau.log_prob(tau2_samp)
        elbo      = logp - (logq_b + logq_t)

        # 4.4) Gradient step
        (-elbo).backward()
        optimizer.step()

        # 4.5) Record and check patience
        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        if elbo_val > best_elbo + tol:
            best_elbo  = elbo_val
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    # 5) Return variational posterior means
    beta_mean = q_mu.detach().cpu().numpy()
    tau2_mean = (alpha / rate).item()

    return {
        "beta_mean": beta_mean,
        "tau2_mean": tau2_mean,
        "elbo_hist": elbo_hist
    }


def BBVI_Laplace_patience(X, y,
                 lr: float = 1e-2,
                 max_iters: int = 10000,
                 tol: float = 1e-8,
                 patience: int = 500):
    """
    Performs BBVI for Bayesian linear regression with Laplace likelihood using
    a Normal-Gamma variational family and early stopping on ELBO “patience.”

    The model:
        y | X, β, τ ~ Laplace(Xβ, 1/τ)
        β | τ² ~ N(μ₀, I / τ²)
        τ² ~ Gamma(a₀, b₀)

    The variational approximation:
        q(β, τ²) = Normal(q_μ, diag(exp(2 * q_log_s))) × Gamma(softplus(q_α), softplus(q_log_b))

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix.
    y : np.ndarray of shape (n,)
        Response vector.
    lr : float
        Learning rate for Adam optimizer.
    max_iters : int
        Maximum number of BBVI iterations.
    tol : float
        Minimum ELBO improvement to reset patience.
    patience : int
        Number of consecutive iterations with < tol ELBO gain before stopping.

    Returns
    -------
    dict
        - 'beta_mean' : np.ndarray of shape (p,), the variational mean of β
        - 'tau2_mean' : float, the expected value of τ² under q
        - 'elbo_hist' : list of floats, ELBO values over training iterations
    """
    n, p = X.shape

    # Move data to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X).float().to(device)  # (n, p)
    y_t = torch.from_numpy(y).float().to(device)  # (n,)

    # Prior hyperparameters
    mu0 = torch.zeros(p, device=device)       # prior mean for β
    # Sigma0 = I  →  inverse is I
    a0, b0 = 0.05, 0.05                       # Gamma(shape=a0, rate=b0)

    # Variational parameters
    q_mu    = nn.Parameter(torch.zeros(p, device=device))
    q_log_s = nn.Parameter(torch.zeros(p, device=device))
    q_alpha = nn.Parameter(torch.tensor(0.0, device=device))
    q_log_b = nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = optim.Adam([q_mu, q_log_s, q_alpha, q_log_b], lr=lr)

    def log_joint(beta, tau2):
        # Laplace log-likelihood (drop constant)
        tau = torch.sqrt(tau2)
        resid = y_t - X_t @ beta
        log_lik = n * torch.log(tau) - tau * torch.sum(torch.abs(resid))

        # Gaussian prior on β | τ² (Σ₀ = I)
        delta = beta - mu0
        quad = delta @ delta
        log_p_beta = (p/2) * torch.log(tau2) - 0.5 * tau2 * quad

        # Gamma prior on τ²
        log_p_tau = (a0 - 1) * torch.log(tau2) - b0 * tau2

        return log_lik + log_p_beta + log_p_tau

    best_elbo = -float('inf')
    no_improve = 0
    elbo_hist = []

    for it in range(max_iters):
        optimizer.zero_grad()

        # Sample β ~ q(β)
        eps = torch.randn(p, device=device)
        s = torch.exp(q_log_s)
        beta_samp = q_mu + s * eps

        # Sample τ² ~ q(τ²)
        alpha = torch.nn.functional.softplus(q_alpha)
        rate  = torch.nn.functional.softplus(q_log_b)
        gamma_dist = Gamma(concentration=alpha, rate=rate)
        tau2_samp  = gamma_dist.rsample()

        # Compute log joint and variational log-density
        logp     = log_joint(beta_samp, tau2_samp)
        logq_beta = Normal(q_mu, s).log_prob(beta_samp).sum()
        logq_tau  = gamma_dist.log_prob(tau2_samp)
        elbo     = logp - (logq_beta + logq_tau)

        # Gradient step
        loss = -elbo
        loss.backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        # Early stopping on ELBO “patience”
        if elbo_val > best_elbo + tol:
            best_elbo = elbo_val
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    # Extract final variational means
    beta_mean = q_mu.detach().cpu().numpy()
    tau2_mean = (alpha / rate).item()

    return {
        'beta_mean': beta_mean,
        'tau2_mean': tau2_mean,
        'elbo_hist': elbo_hist
    }

def BBVI_Laplace(X, y):
    """
    Performs BBVI for Bayesian linear regression with Laplace likelihood using 
    a Normal-Gamma variational family.

    The model:
        y | X, β, τ ~ Laplace(Xβ, 1/τ)
        β | τ² ~ N(μ₀, I / τ²)
        τ² ~ Gamma(a₀, b₀)

    The variational approximation:
        q(β, τ²) = Normal(q_μ, diag(exp(2 * q_log_s))) × Gamma(softplus(q_α), softplus(q_log_b))

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix.
        y : np.ndarray of shape (n,)
        Response vector.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - 'beta_mean' : np.ndarray of shape (p,), the variational mean of β
        - 'tau2_mean' : float, the expected value of τ² under q
        - 'elbo_hist' : list of floats, ELBO values over training iterations
    
    Notes
    -------
    - BBVI uses the reparameterization trick to estimate gradients.
    - The ELBO is optimized using the Adam optimizer.
    """
    n, p = X.shape
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X).float().to(device)  # shape (n, p)
    y_t = torch.from_numpy(y).float().to(device)  # shape (n,)
    
    # --- 2) Hyperparameters -------------------------------------------------------
    mu0 = torch.zeros(p, device=device)           # prior mean for β
    Sigma0_inv = torch.eye(p, device=device)      # assume identity prior
    a0, b0 = 0.05, 0.05                           # Gamma(shape=a0, rate=b0)
    
    # --- 3) Variational parameters (Normal-Gamma) --------------------------------
    q_mu    = nn.Parameter(torch.zeros(p, device=device))  # mean of q(β)
    q_log_s = nn.Parameter(torch.zeros(p, device=device))  # log std of q(β)
    q_alpha = nn.Parameter(torch.tensor(0.0, device=device))  # unconstrained for Gamma shape
    q_log_b = nn.Parameter(torch.tensor(0.0, device=device))  # unconstrained for Gamma rate
    
    optimizer = optim.Adam([q_mu, q_log_s, q_alpha, q_log_b], lr=1e-2)
    
    # --- 4) Log‑joint function ---------------------------------------------------
    def log_joint(beta, tau2):
        # Laplace log-likelihood (drop constant -n*log2)
        tau = torch.sqrt(tau2)
        resid = y_t - X_t @ beta
        log_lik = n * torch.log(tau) - tau * torch.sum(torch.abs(resid))
    
        # Gaussian prior on β | tau2
        delta = beta - mu0
        quad = delta @ delta  # Sigma0 = I
        log_p_beta = (p/2) * torch.log(tau2) - 0.5 * tau2 * quad
    
        # Gamma prior on tau2 (shape=a0, rate=b0)
        log_p_tau = (a0 - 1) * torch.log(tau2) - b0 * tau2
        return log_lik + log_p_beta + log_p_tau
    
    # --- 5) BBVI loop -------------------------------------------------------------
    elbo_hist = []
    num_iters = 10000
    for it in range(num_iters):
        optimizer.zero_grad()
    
        # 5.1) Sample β ~ q(β)
        eps = torch.randn(p, device=device)
        s = torch.exp(q_log_s)
        beta_samp = q_mu + s * eps
    
        # 5.2) Sample τ2 ~ q(τ2)
        alpha = torch.nn.functional.softplus(q_alpha)
        rate = torch.nn.functional.softplus(q_log_b)
        gamma_dist = Gamma(concentration=alpha, rate=rate)
        tau2_samp = gamma_dist.rsample()
    
        # 5.3) Compute densities
        logp = log_joint(beta_samp, tau2_samp)
        logq_beta = Normal(q_mu, s).log_prob(beta_samp).sum()
        logq_tau  = gamma_dist.log_prob(tau2_samp)
        logq = logq_beta + logq_tau
    
        # 5.4) ELBO, gradient step
        elbo = logp - logq
        loss = -elbo
        loss.backward()
        optimizer.step()
    
        # record and print progress
        elbo_hist.append(elbo.item())
        #if it % 500 == 0:
        #    print(f"Iter {it:4d}  ELBO = {elbo.item():.3f}")

    beta_mean = q_mu.detach().cpu().numpy()
    tau2_mean = (alpha / rate).item()

    return {'beta_mean': beta_mean, 'tau2_mean':tau2_mean, 'elbo_hist':elbo_hist}


def BBVI_student_fullcov(
    X: np.ndarray,
    y: np.ndarray,
    nu: float,
    Sigma_prior: np.ndarray,
    a0: float,
    b0: float,
    num_iters: int = 10000,
    lr: float = 1e-2,
    tol: float = 1e-8,
    patience: int = 500
):
    """
    BBVI for Student’s-t regression with prior β|τ² ~ N(0, Σ_prior/τ²),
    τ² ~ Gamma(a0, b0), full‐cov Gaussian variational posterior, and ELBO patience.

    Model:
      y | X, β, τ² ~ Student-t(Xβ, df=ν, scale=1/√τ²)
      β | τ² ~ N(0, Σ_prior/τ²)
      τ² ~ Gamma(shape=a0, rate=b0)

    Variational q:
      q(β) = MVN(q_mu, L Lᵀ)
      q(τ²)= Gamma(softplus(q_alpha), softplus(q_log_b))

    Early-stopping:
      stop if ELBO improvement < tol for >= patience iters.
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move data + prior
    X_t       = torch.from_numpy(X).float().to(device)
    y_t       = torch.from_numpy(y).float().to(device)
    nu_t      = torch.tensor(nu, dtype=torch.float32, device=device)
    Sigma_t   = torch.from_numpy(Sigma_prior).float().to(device)  # (p, p)
    Sigma_inv = torch.inverse(Sigma_t)

    # variational parameters
    q_mu            = nn.Parameter(torch.zeros(p, device=device))
    L_unconstrained = nn.Parameter(torch.eye(p, device=device))
    q_alpha         = nn.Parameter(torch.tensor(0.0, device=device))
    q_log_b         = nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = optim.Adam([q_mu, L_unconstrained, q_alpha, q_log_b], lr=lr)

    def get_scale_tril():
        L = torch.tril(L_unconstrained)
        diag = torch.diagonal(L, 0)
        L = L.clone()
        L[range(p), range(p)] = torch.exp(diag)
        return L

    def log_joint(beta, tau2):
        resid = y_t - X_t @ beta
        tau   = torch.sqrt(tau2)

        # Student-t log-likelihood (up to const)
        const = (
            torch.lgamma((nu_t + 1) / 2)
            - torch.lgamma(nu_t / 2)
            - 0.5 * torch.log(nu_t * torch.tensor(np.pi, device=device))
        )
        log_lik = n * (const + torch.log(tau)) \
                  - ((nu_t + 1) / 2) * torch.sum(torch.log1p((tau2 * resid**2) / nu_t))

        # prior β|τ² ~ N(0, Σ_prior/τ²)
        quad    = beta @ (Sigma_inv @ beta)
        log_p_b = (p / 2) * torch.log(tau2) - 0.5 * tau2 * quad

        # prior τ² ~ Gamma(a0, b0)
        log_p_t = (a0 - 1) * torch.log(tau2) - b0 * tau2

        return log_lik + log_p_b + log_p_t

    best_elbo  = -float("inf")
    no_improve = 0
    elbo_hist  = []

    for it in range(num_iters):
        optimizer.zero_grad()

        # sample β ∼ q(β)
        L       = get_scale_tril()
        q_beta  = MultivariateNormal(loc=q_mu, scale_tril=L)
        beta_samp = q_beta.rsample()

        # sample τ² ∼ q(τ²)
        alpha     = torch.nn.functional.softplus(q_alpha)
        rate      = torch.nn.functional.softplus(q_log_b)
        q_tau2    = Gamma(concentration=alpha, rate=rate)
        tau2_samp = q_tau2.rsample()

        # ELBO estimate
        logp      = log_joint(beta_samp, tau2_samp)
        logq_b    = q_beta.log_prob(beta_samp)
        logq_t    = q_tau2.log_prob(tau2_samp)
        elbo      = logp - (logq_b + logq_t)

        # gradient step
        (-elbo).backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        # patience check
        if elbo_val > best_elbo + tol:
            best_elbo  = elbo_val
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    beta_mean = q_mu.detach().cpu().numpy()
    alpha     = torch.nn.functional.softplus(q_alpha)
    rate      = torch.nn.functional.softplus(q_log_b)
    tau2_mean = (alpha / rate).item()

    return {
        'beta_mean': beta_mean,
        'tau2_mean': tau2_mean,
        'elbo_hist': elbo_hist
    }


def BBVI_student_patience(X, y, nu,
                  num_iters: int = 10000,
                  lr: float = 1e-2,
                  tol: float = 1e-8,
                  patience: int = 500):
    """
    Performs BBVI for Bayesian linear regression with a Student's-t likelihood 
    using a Normal-Gamma variational family, with early stopping on ELBO plateau.

    The model:
        y | X, β, τ² ~ Student-t(Xβ, df=ν, scale=1/√τ²)
        β | τ² ~ N(0, I / τ²)
        τ² ~ Gamma(a₀, b₀)

    Variational approximation:
        q(β, τ²) = Normal(q_μ, diag(exp(2*q_log_s))) × Gamma(softplus(q_α), softplus(q_log_b))

    Early stopping:
        Stops if ELBO does not improve by ≥ tol for `patience` consecutive iters.
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    # Convert nu to a tensor for use in torch.lgamma
    nu_t = torch.tensor(nu, dtype=torch.float32, device=device)

    # Prior hyperparameters
    mu0 = torch.zeros(p, device=device)
    a0, b0 = 0.05, 0.05

    # Variational parameters
    q_mu    = nn.Parameter(torch.zeros(p, device=device))
    q_log_s = nn.Parameter(torch.zeros(p, device=device))
    q_alpha = nn.Parameter(torch.tensor(0.0, device=device))
    q_log_b = nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = optim.Adam([q_mu, q_log_s, q_alpha, q_log_b], lr=lr)

    def log_joint(beta, tau2):
        resid = y_t - X_t @ beta
        tau   = torch.sqrt(tau2)

        # Student-t constants all lifted to tensors
        const = (
            torch.lgamma((nu_t + 1) / 2)
            - torch.lgamma(nu_t / 2)
            - 0.5 * torch.log(nu_t * torch.tensor(np.pi, device=device))
        )

        log_lik = n * (const + torch.log(tau)) \
                  - ((nu_t + 1) / 2) * torch.sum(torch.log1p((tau2 * resid**2) / nu_t))

        # Gaussian prior on β | τ²
        delta = beta - mu0
        quad  = delta @ delta
        log_p_beta = (p / 2) * torch.log(tau2) - 0.5 * tau2 * quad

        # Gamma prior on τ²
        log_p_tau = (a0 - 1) * torch.log(tau2) - b0 * tau2

        return log_lik + log_p_beta + log_p_tau

    best_elbo = -float('inf')
    no_improve = 0
    elbo_hist = []

    for it in range(num_iters):
        optimizer.zero_grad()

        # Sample β ∼ q(β)
        eps       = torch.randn(p, device=device)
        scale     = torch.exp(q_log_s)
        beta_samp = q_mu + scale * eps

        # Sample τ² ∼ q(τ²)
        alpha = torch.nn.functional.softplus(q_alpha)
        rate  = torch.nn.functional.softplus(q_log_b)
        q_tau2 = Gamma(concentration=alpha, rate=rate)
        tau2_samp = q_tau2.rsample()

        # Compute ELBO
        logp      = log_joint(beta_samp, tau2_samp)
        logq_beta = Normal(q_mu, scale).log_prob(beta_samp).sum()
        logq_tau  = q_tau2.log_prob(tau2_samp)
        elbo      = logp - (logq_beta + logq_tau)

        # Gradient step
        (-elbo).backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        # Early-stopping “patience” check
        if elbo_val > best_elbo + tol:
            best_elbo = elbo_val
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    # Extract posterior means
    beta_mean = q_mu.detach().cpu().numpy()
    alpha = torch.nn.functional.softplus(q_alpha)
    rate  = torch.nn.functional.softplus(q_log_b)
    tau2_mean = (alpha / rate).item()

    return {
        'beta_mean': beta_mean,
        'tau2_mean': tau2_mean,
        'elbo_hist': elbo_hist
    }

def BBVI_student(X, y, nu, num_iters=10000, lr=1e-2):
    """
    Performs BBVI for Bayesian linear regression with a Student's-t likelihood 
    using a Normal-Gamma variational family.

    The model:
        y | X, β, τ² ~ Student-t(Xβ, df=ν, scale=1/√τ²)
        β | τ² ~ N(μ₀, I / τ²)
        τ² ~ Gamma(a₀, b₀)

    The variational approximation:
        q(β, τ²) = Normal(q_μ, diag(exp(2 * q_log_s))) × Gamma(softplus(q_α), softplus(q_log_b))

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix.
    y : np.ndarray of shape (n,)
        Response vector.
    nu : float
        Degrees of freedom for the Student's-t distribution (controls tail heaviness).
    num_iters : int, optional
        Number of BBVI training iterations. Default is 10000.
    lr : float, optional
        Learning rate for the Adam optimizer. Default is 1e-2.

    Returns
    -------
    dict
        A dictionary containing:
        - 'beta_mean' : np.ndarray of shape (p,), mean of variational posterior q(β)
        - 'tau2_mean' : float, expected value of τ² under q
        - 'elbo_hist' : list of float, ELBO values over training iterations
    
    Notes
    -----
    - Variational approximation is performed using reparameterization trick for both β and τ².
    - Priors are:
        β | τ² ~ N(0, I/τ²)
        τ² ~ Gamma(0.05, 0.05)
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move data
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    # --- Hyperparameters for priors --------------------------------------------
    mu0 = torch.zeros(p, device=device)
    Sigma0_inv = torch.eye(p, device=device)   # prior precision on β
    a0, b0 = 0.05, 0.05                        # Gamma(shape, rate) on τ²

    # --- Variational parameters ------------------------------------------------
    q_mu    = nn.Parameter(torch.zeros(p, device=device))
    q_log_s = nn.Parameter(torch.zeros(p, device=device))
    q_alpha = nn.Parameter(torch.tensor(0.0, device=device))
    q_log_b = nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = optim.Adam([q_mu, q_log_s, q_alpha, q_log_b], lr=lr)

    # --- log‑joint under Student's‑t likelihood -------------------------------
    def log_joint(beta, tau2):
        # residuals
        resid = y_t - X_t @ beta  # (n,)

        # Student‑t log‑likelihood:
        #   p(resid) ∝ Γ((ν+1)/2) / [Γ(ν/2) * sqrt(νπ)] * τ * (1 + τ² resid²/ν)^(-(ν+1)/2)
        #   where τ = sqrt(tau2) is the precision parameter.
        #   lgamma = torch.lgamma
        #   const = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * torch.log(nu * torch.tensor(np.pi, device=device))
        const=0
        log_lik = n * (const + 0.5 * torch.log(tau2)) \
                  - ((nu + 1) / 2) * torch.sum(torch.log1p((tau2 * resid**2) / nu))

        # Gaussian prior on β | τ²  ~ N(μ0, (τ² I)^(-1))
        delta = beta - mu0
        quad = (delta @ delta)  # since Σ0 = I
        log_p_beta = (p / 2) * torch.log(tau2) - 0.5 * tau2 * quad

        # Gamma prior on τ²
        log_p_tau = (a0 - 1) * torch.log(tau2) - b0 * tau2

        return log_lik + log_p_beta + log_p_tau

    # --- BBVI -------------------------------------------------------------------
    elbo_hist = []
    for it in range(num_iters):
        optimizer.zero_grad()

        # draw β ~ q(β)
        eps = torch.randn(p, device=device)
        s   = torch.exp(q_log_s)
        beta_samp = q_mu + s * eps

        # draw τ² ~ q(τ²)
        alpha = torch.nn.functional.softplus(q_alpha)
        rate  = torch.nn.functional.softplus(q_log_b)
        q_tau2 = Gamma(concentration=alpha, rate=rate)
        tau2_samp = q_tau2.rsample()

        # joint & variational densities
        logp      = log_joint(beta_samp, tau2_samp)
        logq_beta = Normal(q_mu, s).log_prob(beta_samp).sum()
        logq_tau  = q_tau2.log_prob(tau2_samp)
        elbo      = logp - (logq_beta + logq_tau)

        # gradient step
        (-elbo).backward()
        optimizer.step()

        elbo_hist.append(elbo.item())

    # Return posterior means
    beta_mean = q_mu.detach().cpu().numpy()
    tau2_mean = (alpha / rate).item()

    return {'beta_mean': beta_mean,
            'tau2_mean': tau2_mean,
            'elbo_hist': elbo_hist}

def BBVI_Logistic(X, y, num_iters=10000, lr=1e-2):
    """
    Performs BBVI for Bayesian logistic regression with a standard Gaussian 
    prior and a Normal variational posterior.

    The model:
        y_i | x_i, β ~ Bernoulli(sigmoid(x_i^T β))
        β ~ N(0, I)

    The variational approximation:
        q(β) = Normal(q_μ, diag(exp(2 * q_log_s)))

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix.
    y : np.ndarray of shape (n,)
        Binary response vector (0 or 1).
    num_iters : int, optional
        Number of BBVI iterations. Default is 10000.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-2.

    Returns
    -------
    dict
        Dictionary containing:
        - 'beta_mean' : np.ndarray of shape (p,), variational mean of β
        - 'elbo_hist' : list of float, ELBO values across iterations

    Notes
    -----
    - Uses the reparameterization trick for gradient estimation.
    - Assumes standard normal prior and independent normal variational posterior.
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data → torch
    X_t = torch.from_numpy(X).float().to(device)  # (n, p)
    y_t = torch.from_numpy(y).float().to(device)  # (n,)

    # prior mean
    mu0 = torch.zeros(p, device=device)

    # variational params
    q_mu    = nn.Parameter(torch.zeros(p, device=device))
    q_log_s = nn.Parameter(torch.zeros(p, device=device))  # log‐std

    optimizer = optim.Adam([q_mu, q_log_s], lr=lr)

    def log_joint(beta):
        # 1) logistic log‐likelihood
        Xbeta  = X_t @ beta
        log_lik = torch.sum(y_t * Xbeta) \
                  - torch.sum(torch.log1p(torch.exp(Xbeta)))

        # 2) Gaussian prior N(0, I)
        delta = beta - mu0
        quad  = torch.dot(delta, delta)
        log_p  = -0.5 * quad

        return log_lik + log_p

    elbo_hist = []
    for it in range(num_iters):
        optimizer.zero_grad()

        # sample β ~ q(β)
        eps       = torch.randn(p, device=device)
        scale     = torch.exp(q_log_s)
        beta_samp = q_mu + scale * eps

        # compute joint & q‐density
        logp      = log_joint(beta_samp)
        logq_beta = Normal(q_mu, scale).log_prob(beta_samp).sum()

        # ELBO estimate & step
        elbo = logp - logq_beta
        (-elbo).backward()
        optimizer.step()

        elbo_hist.append(elbo.item())

    return {
        'beta_mean': q_mu.detach().cpu().numpy(),
        'elbo_hist': elbo_hist
    }

def BBVI_Logistic_patience(X, y,
                  num_iters: int = 10000,
                  lr: float = 1e-2,
                  tol: float = 1e-8,
                  patience: int = 500):
    """
    Performs BBVI for Bayesian logistic regression with a standard Gaussian 
    prior and a Normal variational posterior, with early stopping on ELBO plateau.

    The model:
        y_i | x_i, β ~ Bernoulli(sigmoid(x_i^T β))
        β ~ N(0, I)

    Variational approximation:
        q(β) = Normal(q_μ, diag(exp(2 * q_log_s)))

    Early stopping:
        Stops if ELBO does not improve by ≥ tol for `patience` consecutive iters.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
    y : np.ndarray of shape (n,)
    num_iters : int
        Maximum BBVI iterations.
    lr : float
        Learning rate for Adam.
    tol : float
        Minimum ELBO improvement to reset patience.
    patience : int
        Number of consecutive iters without ≥ tol ELBO gain before stopping.

    Returns
    -------
    dict
        - 'beta_mean': np.ndarray of shape (p,), variational mean of β
        - 'elbo_hist' : list of float, ELBO values over iterations
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # to torch
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    # prior mean
    mu0 = torch.zeros(p, device=device)

    # variational params
    q_mu    = nn.Parameter(torch.zeros(p, device=device))
    q_log_s = nn.Parameter(torch.zeros(p, device=device))

    optimizer = optim.Adam([q_mu, q_log_s], lr=lr)

    def log_joint(beta):
        # logistic log-likelihood
        Xbeta  = X_t @ beta
        log_lik = (y_t * Xbeta).sum() \
                  - torch.log1p(torch.exp(Xbeta)).sum()
        # N(0, I) prior
        delta = beta - mu0
        log_p  = -0.5 * (delta @ delta)
        return log_lik + log_p

    best_elbo = -float('inf')
    no_improve = 0
    elbo_hist = []

    for it in range(num_iters):
        optimizer.zero_grad()

        # sample β ~ q(β)
        eps       = torch.randn(p, device=device)
        scale     = torch.exp(q_log_s)
        beta_samp = q_mu + scale * eps

        # compute ELBO
        logp      = log_joint(beta_samp)
        logq_beta = Normal(q_mu, scale).log_prob(beta_samp).sum()
        elbo      = logp - logq_beta

        # gradient step
        (-elbo).backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        # early stopping on ELBO “patience”
        if elbo_val > best_elbo + tol:
            best_elbo = elbo_val
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    return {
        'beta_mean': q_mu.detach().cpu().numpy(),
        'elbo_hist': elbo_hist
    }

def BBVI_Logistic_fullcov(
    X: np.ndarray,
    y: np.ndarray,
    Sigma_prior: np.ndarray,
    num_iters: int = 10000,
    lr: float = 1e-2,
    tol: float = 1e-8,
    patience: int = 500
):
    """
    BBVI for Bayesian logistic regression with prior β ~ N(0, Σ_prior),
    full‐covariance Gaussian variational posterior q(β), and ELBO‐patience stopping.

    Model:
        y_i | x_i, β ~ Bernoulli(sigmoid(x_i^T β))
        β ~ N(0, Σ_prior)

    Variational approximation:
        q(β) = MVN(q_mu, L L^T)  (L lower‐triangular with positive diag)

    Early stopping:
        stop if ELBO improvement < tol for ≥ patience iters.
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move data and prior to device
    X_t       = torch.from_numpy(X).float().to(device)             # (n, p)
    y_t       = torch.from_numpy(y).float().to(device)             # (n,)
    Sigma_t   = torch.from_numpy(Sigma_prior).float().to(device)   # (p, p)
    Sigma_inv = torch.inverse(Sigma_t)                             # (p, p)

    # variational parameters
    q_mu            = nn.Parameter(torch.zeros(p, device=device))
    L_unconstrained = nn.Parameter(torch.eye(p, device=device))

    optimizer = optim.Adam([q_mu, L_unconstrained], lr=lr)

    def get_scale_tril():
        L = torch.tril(L_unconstrained)
        diag = torch.diagonal(L, 0)
        L = L.clone()
        L[range(p), range(p)] = torch.exp(diag)
        return L  # L @ L.T is full cov

    def log_joint(beta):
        # 1) logistic log‐likelihood
        Xbeta  = X_t @ beta
        log_lik = (y_t * Xbeta).sum() - torch.log1p(torch.exp(Xbeta)).sum()

        # 2) Gaussian prior N(0, Σ_prior) (up to const)
        quad    = beta @ (Sigma_inv @ beta)
        log_p   = -0.5 * quad

        return log_lik + log_p

    best_elbo  = -float("inf")
    no_improve = 0
    elbo_hist  = []

    for it in range(num_iters):
        optimizer.zero_grad()

        # sample β ∼ q(β)
        L       = get_scale_tril()
        q_beta  = MultivariateNormal(loc=q_mu, scale_tril=L)
        beta_samp = q_beta.rsample()

        # compute ELBO
        logp      = log_joint(beta_samp)
        logq_b    = q_beta.log_prob(beta_samp)
        elbo      = logp - logq_b

        # gradient step
        (-elbo).backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        # patience early‐stop check
        if elbo_val > best_elbo + tol:
            best_elbo  = elbo_val
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    # return variational posterior mean and ELBO trace
    return {
        'beta_mean': q_mu.detach().cpu().numpy(),
        'elbo_hist': elbo_hist
    }


def BBVI_NegBin(X, y, r, num_iters=10000, lr=1e-2):
    """
    Performs BBVI for Bayesian negative-binomial regression with a Gaussian 
    prior and variational approximation.

    The model:
        y_i ~ NegBinom(mean = r * exp(x_i^T β), dispersion = r)
        β ~ N(0, I)

    The variational approximation:
        q(β) = Normal(q_μ, diag(exp(2 * q_log_s)))

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix.
    y : np.ndarray of shape (n,)
        Count response vector (non-negative integers).
    r : float
        Dispersion parameter of the negative binomial distribution (fixed).
    num_iters : int, optional
        Number of BBVI iterations. Default is 10000.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-2.

    Returns
    -------
    dict
        Dictionary containing:
        - 'beta_mean' : np.ndarray of shape (p,), variational mean of β
        - 'elbo_hist' : list of float, ELBO values across iterations

    Notes
    -----
    - This implementation assumes canonical form of NB log-likelihood:
        log p(y | β) = ∑ [r x_i^T β - (r + y_i) * log(1 + exp(x_i^T β))]
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data → torch
    X_t = torch.from_numpy(X).float().to(device)  # (n, p)
    y_t = torch.from_numpy(y).float().to(device)  # (n,)

    # prior mean
    mu0 = torch.zeros(p, device=device)

    # variational params
    q_mu    = nn.Parameter(torch.zeros(p, device=device))
    q_log_s = nn.Parameter(torch.zeros(p, device=device))  # log‐std

    optimizer = optim.Adam([q_mu, q_log_s], lr=lr)

    def log_joint(beta):
        # 1) negative-binomial log‐likelihood
        Xbeta  = X_t @ beta
        log_lik = torch.sum(r * Xbeta) - torch.sum((r + y_t) * torch.log1p(torch.exp(Xbeta)))

        # 2) Gaussian prior N(0, I)
        delta = beta - mu0
        quad  = torch.dot(delta, delta)
        log_p  = -0.5 * quad

        return log_lik + log_p

    elbo_hist = []
    for it in range(num_iters):
        optimizer.zero_grad()

        # sample β ~ q(β)
        eps       = torch.randn(p, device=device)
        scale     = torch.exp(q_log_s)
        beta_samp = q_mu + scale * eps

        # compute joint & q‐density
        logp      = log_joint(beta_samp)
        logq_beta = Normal(q_mu, scale).log_prob(beta_samp).sum()

        # ELBO estimate & step
        elbo = logp - logq_beta
        (-elbo).backward()
        optimizer.step()

        elbo_hist.append(elbo.item())

    return {
        'beta_mean': q_mu.detach().cpu().numpy(),
        'elbo_hist': elbo_hist
    }

def BBVI_NegBin_patience(X, y, r,
                num_iters: int = 10000,
                lr: float = 1e-2,
                tol: float = 1e-8,
                patience: int = 500):
    """
    Performs BBVI for Bayesian negative-binomial regression with a Gaussian 
    prior and variational approximation, with early stopping on ELBO plateau.

    The model:
        y_i ~ NegBinom(mean = r * exp(x_i^T β), dispersion = r)
        β ~ N(0, I)

    Variational approximation:
        q(β) = Normal(q_μ, diag(exp(2 * q_log_s)))

    Early stopping:
        Stops if ELBO does not improve by ≥ tol for `patience` consecutive iters.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
    y : np.ndarray of shape (n,)
        Count response vector.
    r : float
        Dispersion parameter of the negative binomial (fixed).
    num_iters : int
        Maximum BBVI iterations.
    lr : float
        Learning rate for Adam optimizer.
    tol : float
        Minimum ELBO improvement to reset patience.
    patience : int
        Number of consecutive iterations without ≥ tol ELBO gain before stopping.

    Returns
    -------
    dict
        - 'beta_mean': np.ndarray of shape (p,), variational mean of β
        - 'elbo_hist' : list of float, ELBO values over iterations
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move data to device
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    if isinstance(r, np.ndarray):
        r = torch.from_numpy(r).float().to(device)
    else:
        r = torch.tensor(r, dtype=torch.float32, device=device)

    # prior mean
    mu0 = torch.zeros(p, device=device)

    # variational parameters
    q_mu    = nn.Parameter(torch.zeros(p, device=device))
    q_log_s = nn.Parameter(torch.zeros(p, device=device))

    optimizer = optim.Adam([q_mu, q_log_s], lr=lr)

    def log_joint(beta):
        # negative-binomial log-likelihood (canonical form)
        Xbeta  = X_t @ beta
        log_lik = torch.sum(r * Xbeta) \
                  - torch.sum((r + y_t) * torch.log1p(torch.exp(Xbeta)))

        # Gaussian prior N(0, I)
        delta = beta - mu0
        quad  = torch.dot(delta, delta)
        log_p = -0.5 * quad

        return log_lik + log_p

    best_elbo = -float('inf')
    no_improve = 0
    elbo_hist = []

    for it in range(num_iters):
        optimizer.zero_grad()

        # sample β ~ q(β)
        eps       = torch.randn(p, device=device)
        scale     = torch.exp(q_log_s)
        beta_samp = q_mu + scale * eps

        # compute ELBO
        logp      = log_joint(beta_samp)
        logq_beta = Normal(q_mu, scale).log_prob(beta_samp).sum()
        elbo      = logp - logq_beta

        # gradient step
        (-elbo).backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        # early stopping on ELBO “patience”
        if elbo_val > best_elbo + tol:
            best_elbo = elbo_val
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    return {
        'beta_mean': q_mu.detach().cpu().numpy(),
        'elbo_hist': elbo_hist
    }

def BBVI_NegBin_fullcov(
    X: np.ndarray,
    y: np.ndarray,
    r,
    Sigma_prior: np.ndarray,
    num_iters: int = 10000,
    lr: float = 1e-2,
    tol: float = 1e-8,
    patience: int = 500
):
    """
    BBVI for negative‐binomial regression with prior β ~ N(0, Σ_prior),
    full‐covariance Gaussian variational posterior q(β), and ELBO‐patience stopping.

    Model:
      y_i ~ NegBinom(mean = r * exp(x_i^T β), dispersion = r)
      β ~ N(0, Σ_prior)

    Variational:
      q(β) = MVN(q_mu, L L^T)
      Stops if ELBO improvement < tol for ≥ patience iters.
    """
    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move data and prior to device
    X_t     = torch.from_numpy(X).float().to(device)
    y_t     = torch.from_numpy(y).float().to(device)
    if isinstance(r, np.ndarray):
        r_t = torch.from_numpy(r).float().to(device)
    else:
        r_t = torch.tensor(r, dtype=torch.float32, device=device)

    Sigma_t   = torch.from_numpy(Sigma_prior).float().to(device)  # (p, p)
    Sigma_inv = torch.inverse(Sigma_t)

    # variational params
    q_mu            = nn.Parameter(torch.zeros(p, device=device))
    L_unconstrained = nn.Parameter(torch.eye(p, device=device))
    optimizer = optim.Adam([q_mu, L_unconstrained], lr=lr)

    def get_scale_tril():
        L = torch.tril(L_unconstrained)
        diag = torch.diagonal(L, 0)
        L = L.clone()
        L[range(p), range(p)] = torch.exp(diag)
        return L

    def log_joint(beta):
        Xbeta  = X_t @ beta
        log_lik = torch.sum(r_t * Xbeta) \
                  - torch.sum((r_t + y_t) * torch.log1p(torch.exp(Xbeta)))

        # prior β ~ N(0, Σ_prior)
        quad    = beta @ (Sigma_inv @ beta)
        log_p   = -0.5 * quad

        return log_lik + log_p

    best_elbo  = -float("inf")
    no_improve = 0
    elbo_hist  = []

    for it in range(num_iters):
        optimizer.zero_grad()

        # sample β ∼ q(β)
        L       = get_scale_tril()
        q_beta  = MultivariateNormal(loc=q_mu, scale_tril=L)
        beta_samp = q_beta.rsample()

        # compute ELBO
        logp      = log_joint(beta_samp)
        logq_beta = q_beta.log_prob(beta_samp)
        elbo      = logp - logq_beta

        (-elbo).backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        if elbo_val > best_elbo + tol:
            best_elbo  = elbo_val
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    return {
        'beta_mean': q_mu.detach().cpu().numpy(),
        'elbo_hist': elbo_hist
    }