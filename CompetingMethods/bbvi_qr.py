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
from torch.distributions import Normal, Independent
from torch.distributions import MultivariateNormal
from sklearn.linear_model import LogisticRegression
from numpy.linalg import solve
import torch.nn.functional as F

def BBVI_QR_fr(
    X: np.ndarray,
    y: np.ndarray,
    Sigma_prior: np.ndarray,
    quant: float = 0.5,                # quantile in (0,1)
    lr: float = 1e-3,
    max_iters: int = 20000,
    tol: float = 1e-8,
    patience: int = 1000,
    weight_decay: float = 1e-2,
    verbose: bool = True,
    n_mc: int = 1,                   # MC samples per iter
    learn_sigma: bool = False,        # learn ALD scale
    sigma_init: float = 1.0,
    sigma_min: float = 1e-6,
    seed: int = 123
):
    """
    Bayesian Quantile Regression with full-rank Gaussian variational family.
    Prior: beta ~ N(0, Sigma_prior)
    Likelihood: Asymmetric Laplace with quantile quant and (optional) scale sigma.

    If learn_sigma=False, sigma term is omitted (equiv. to unscaled pinball loss).
    """
    assert 0.0 < quant < 1.0, "quant must be in (0,1)."
    torch.manual_seed(seed)

    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    Sigma_t = torch.from_numpy(Sigma_prior).float().to(device)
    Sigma_inv = torch.inverse(Sigma_t)

    # variational params
    q_mu = nn.Parameter(torch.zeros(p, device=device))
    # warm start at LS solution
    with torch.no_grad():
        # robust solve in case XtX is ill-conditioned
        q_mu.copy_(torch.linalg.lstsq(X_t, y_t).solution)

    L_unconstrained = nn.Parameter(torch.zeros((p, p), device=device))

    # optional ALD scale (global)
    if learn_sigma:
        rho = nn.Parameter(torch.tensor(np.log(np.expm1(sigma_init)), dtype=torch.float32, device=device))
        # sigma = softplus(rho) ensures positivity
        def get_sigma():
            return torch.clamp(F.softplus(rho), min=sigma_min)
        params = [{"params": [q_mu], "weight_decay": 0.0},
                  {"params": [L_unconstrained], "weight_decay": weight_decay},
                  {"params": [rho], "weight_decay": 0.0}]
    else:
        def get_sigma():
            return torch.tensor(1.0, device=device)
        params = [{"params": [q_mu], "weight_decay": 0.0},
                  {"params": [L_unconstrained], "weight_decay": weight_decay}]

    optimizer = optim.AdamW(params, lr=lr)

    def get_scale_tril():
        # lower-tri with softplus on diag for PD
        L = torch.tril(L_unconstrained)
        diag = torch.diagonal(L, 0)
        L = L.clone()
        L[range(p), range(p)] = torch.clamp(F.softplus(diag), min=1e-6)
        return L

    def pinball(r, quant):
        # rho_quant(r) = (quant - 1_{r<0}) * r
        return torch.where(r >= 0, quant * r, (quant - 1.0) * r)

    best_elbo = -float("inf")
    no_improve = 0
    elbo_hist = []
    best_state = None

    for it in range(max_iters):
        optimizer.zero_grad()

        L = get_scale_tril()
        q = MultivariateNormal(loc=q_mu, scale_tril=L)

        elbo_batch = 0.0
        for _ in range(n_mc):
            beta_samp = q.rsample()

            r = y_t - X_t @ beta_samp
            if learn_sigma:
                sigma = get_sigma()
                # Asymmetric Laplace loglik (up to additive const): - sum rho_quant(r)/sigma - n*log(sigma)
                log_lik = -(pinball(r, quant).sum() / sigma) - n * torch.log(sigma)
            else:
                # unscaled pinball (proportional to ALD with fixed sigma)
                log_lik = -pinball(r, quant).sum()

            # Gaussian prior log-density (up to const)
            log_p_b = -0.5 * (beta_samp @ (Sigma_inv @ beta_samp))

            logq = q.log_prob(beta_samp)

            elbo_batch = elbo_batch + (log_lik + log_p_b - logq)

        elbo = elbo_batch / n_mc
        (-elbo).backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        if elbo_val > best_elbo + tol:
            best_elbo = elbo_val
            no_improve = 0
            # save best variational params
            best_state = {
                "q_mu": q_mu.detach().clone(),
                "L_unconstrained": L_unconstrained.detach().clone()
            }
            if learn_sigma:
                best_state["rho"] = rho.detach().clone()
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    # restore best
    if best_state is not None:
        with torch.no_grad():
            q_mu.copy_(best_state["q_mu"])
            L_unconstrained.copy_(best_state["L_unconstrained"])
            if learn_sigma:
                rho.copy_(best_state["rho"])

    L = get_scale_tril().detach().cpu()
    beta_mean = q_mu.detach().cpu().numpy()
    Sigma_q = (L @ L.T).numpy()
    out = {
        "beta_mean": beta_mean,
        "Sigma_mean": Sigma_q,
        "elbo_hist": elbo_hist,
        "quant": quant
    }
    if learn_sigma:
        out["sigma"] = float(get_sigma().detach().cpu().item())

    return out


def BBVI_QR_mf(
    X: np.ndarray,
    y: np.ndarray,
    Sigma_prior: np.ndarray,
    quant: float = 0.5,                # quantile in (0,1)
    lr: float = 1e-3,
    max_iters: int = 20000,
    tol: float = 1e-8,
    patience: int = 1000,
    weight_decay: float = 1e-2,      # applied to variational std params
    verbose: bool = True,
    n_mc: int = 1,                   # MC samples per iter
    learn_sigma: bool = False,        # learn ALD scale
    sigma_init: float = 1.0,
    sigma_min: float = 1e-6,
    seed: int = 123
):
    """
    Bayesian Quantile Regression with mean-field (diagonal) Gaussian variational family.
    Prior: beta ~ N(0, Sigma_prior)  (Sigma_prior can be dense; only used via Sigma_inv).
    Likelihood: Asymmetric Laplace (pinball) with quantile quant and optional scale sigma.

    Returns:
        {
          "beta_mean": (p,) posterior mean,
          "Sigma_mean": (p,p) diag covariance of q,
          "elbo_hist": list,
          "quant": quant,
          "sigma": float (if learn_sigma)
        }
    """
    assert 0.0 < quant < 1.0, "quant must be in (0,1)."
    torch.manual_seed(seed)

    n, p = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    Sigma_t = torch.from_numpy(Sigma_prior).float().to(device)
    Sigma_inv = torch.inverse(Sigma_t)

    # variational params: mean and (unconstrained) stds
    q_mu = nn.Parameter(torch.zeros(p, device=device))
    with torch.no_grad():
        q_mu.copy_(torch.linalg.lstsq(X_t, y_t).solution)  # warm start

    # s_unconstrained -> std = softplus(s_unconstrained)
    s_unconstrained = nn.Parameter(torch.zeros(p, device=device))

    # optional ALD scale (global)
    if learn_sigma:
        rho = nn.Parameter(torch.tensor(np.log(np.expm1(sigma_init)), dtype=torch.float32, device=device))
        def get_sigma():
            return torch.clamp(F.softplus(rho), min=sigma_min)
        params = [
            {"params": [q_mu], "weight_decay": 0.0},
            {"params": [s_unconstrained], "weight_decay": weight_decay},
            {"params": [rho], "weight_decay": 0.0},
        ]
    else:
        def get_sigma():
            return torch.tensor(1.0, device=device)
        params = [
            {"params": [q_mu], "weight_decay": 0.0},
            {"params": [s_unconstrained], "weight_decay": weight_decay},
        ]

    optimizer = optim.AdamW(params, lr=lr)

    def q_dist():
        # diagonal stds, strictly positive
        std = torch.clamp(F.softplus(s_unconstrained), min=1e-6)
        base = Normal(loc=q_mu, scale=std)
        return Independent(base, 1), std

    def pinball(r, quant):
        return torch.where(r >= 0, quant * r, (quant - 1.0) * r)

    best_elbo = -float("inf")
    no_improve = 0
    elbo_hist = []
    best_state = None

    for it in range(max_iters):
        optimizer.zero_grad()

        q, std = q_dist()

        elbo_batch = 0.0
        for _ in range(n_mc):
            beta_samp = q.rsample()  # reparameterized sample: (p,)

            r = y_t - X_t @ beta_samp
            if learn_sigma:
                sigma = get_sigma()
                log_lik = -(pinball(r, quant).sum() / sigma) - n * torch.log(sigma)
            else:
                log_lik = -pinball(r, quant).sum()

            # Gaussian prior (0, Sigma_prior)
            log_p_b = -0.5 * (beta_samp @ (Sigma_inv @ beta_samp))

            logq = q.log_prob(beta_samp)  # sum over dims due to Independent

            elbo_batch = elbo_batch + (log_lik + log_p_b - logq)

        elbo = elbo_batch / n_mc
        (-elbo).backward()
        optimizer.step()

        elbo_val = elbo.item()
        elbo_hist.append(elbo_val)

        if elbo_val > best_elbo + tol:
            best_elbo = elbo_val
            no_improve = 0
            best_state = {
                "q_mu": q_mu.detach().clone(),
                "s_unconstrained": s_unconstrained.detach().clone()
            }
            if learn_sigma:
                best_state["rho"] = rho.detach().clone()
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at iter {it}: no ELBO gain ≥ {tol} for {patience} iters.")
            break

    # restore best
    if best_state is not None:
        with torch.no_grad():
            q_mu.copy_(best_state["q_mu"])
            s_unconstrained.copy_(best_state["s_unconstrained"])
            if learn_sigma:
                rho.copy_(best_state["rho"])

    # package outputs
    with torch.no_grad():
        std = torch.clamp(F.softplus(s_unconstrained), min=1e-6)
        var = std**2
        beta_mean = q_mu.detach().cpu().numpy()
        Sigma_diag = var.detach().cpu().numpy()
        Sigma_mean = np.diag(Sigma_diag)

    out = {
        "beta_mean": beta_mean,
        "Sigma_mean": Sigma_mean,   # diagonal covariance matrix
        "elbo_hist": elbo_hist,
        "quant": quant
    }
    if learn_sigma:
        out["sigma"] = float(get_sigma().detach().cpu().item())

    return out
