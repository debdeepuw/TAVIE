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