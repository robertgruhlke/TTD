"""Score functions for the time-diffused target density, computed by Legendre or
Gauss-Hermite quadrature over the OU transition kernel rather than by an xTT fit;
used as an independent, non-learned reference for validation.
"""

import math
from itertools import product

import torch
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

torch.set_default_dtype(torch.float64)


def compute_score_by_legendre_quadrature(problem, nP, L=10.0):
    """
    Computes the score function ∇ log π_t(x) via Legendre quadrature on [-L,L]^d.
    
    Args:
        problem: An object with attributes 
                 - dim (int): dimension
                 - target(x): potential Φ(x), takes x: (N,d) → (N,)
        nP (int): Number of quadrature points per dimension
        L (float): Domain limit for Legendre quadrature (default: [-L, L])
    
    Returns:
        pi_t(t, x): Function returning π_t(x) for given t, x
    """

    dim = problem.dim
    Phi = problem.target

    # ---- Legendre nodes/weights on [-1,1]
    x1d_np, w1d_np = leggauss(nP)
    # scale to [-L, L]
    x1d = 0.5 * (x1d_np + 1.0) * (2*L) - L   # map [-1,1] → [-L,L]
    w1d = 0.5 * (2*L) * w1d_np               # scale weights

    x1d = torch.tensor(x1d, dtype=torch.float32)
    w1d = torch.tensor(w1d, dtype=torch.float32)

    # ---- build grid (Nq, d) and weights (Nq,)
    W = torch.zeros([nP] * dim)
    X = torch.zeros([nP] * dim + [dim])
    for alpha in product(range(nP), repeat=dim):
        W[alpha] = torch.prod(torch.tensor([w1d[a] for a in alpha]))
        X[alpha] = torch.tensor([x1d[a] for a in alpha])

    X_flat = X.view(-1, dim)   # (Nq, d)
    W_flat = W.view(-1)        # (Nq,)

    # π*(x0) ∝ exp(-Φ(x0))
    pi_ast = torch.exp(-Phi(X_flat)).flatten()  # (Nq,)

    def M(t, x0): return torch.exp(-t) * x0

    def S(t):
        return (1 - torch.exp(-2 * t)) * torch.eye(dim)

    def transition_density(t, x, x0):
        # x: (B, d), x0: (N, d)
        Sigma = S(t)
        Sigma_inv = torch.linalg.inv(Sigma)
        det_Sigma = torch.det(Sigma)
        diff = x.unsqueeze(1) - M(t, x0).unsqueeze(0)  # (B, N, d)
        exponents = torch.einsum("bni,ij,bnj->bn", diff, Sigma_inv, diff)  # (B, N)
        norm_const = torch.sqrt((2 * math.pi) ** dim * det_Sigma)
        return torch.exp(-0.5 * exponents) / norm_const  # (B, N)

    def pi_t(t, x):
        # x: (B, d)
        tdens = transition_density(t, x, X_flat)   # (B, Nq)
        weights = W_flat * pi_ast                  # (Nq,)
        return torch.sum(tdens * weights[None, :], dim=1)  # (B,)

    return pi_t


def compute_score_by_hermite_quadrature(problem, nP):
    """
    Computes the score function ∇ log π_t(x) via Gaussian Hermite quadrature.

    Args:
        problem: An object with methods `dim()` and `target`, where `target(x)` returns Φ(x)
        nP (int): Number of quadrature points per dimension

    Returns:
        score(t, x): Function returning score ∇ log π_t(x) for given t, x
    """
    dim = problem.dim
    Phi = problem.target

    x1d_np, weights1D_np = hermgauss(nP)  # from numpy
    x1d = torch.tensor(x1d_np, dtype=torch.float32)
    weights1D = torch.tensor(weights1D_np, dtype=torch.float32)

    # Precompute multidimensional grid
    W = torch.zeros([nP] * dim)
    X = torch.zeros([nP] * dim + [dim])

    for alpha in product(range(nP), repeat=dim):
        W[alpha] = torch.prod(torch.tensor([weights1D[a] for a in alpha]))
        X[*alpha, :] = torch.tensor([x1d[a] for a in alpha])

    # Flatten for efficient vectorization later
    X_flat = X.view(-1, dim)
    W_flat = W.view(-1)

    # Define π_*(x) = 1/Z * exp(-Φ(x))
    Z = 1.0  # if pi_ast is left unnormalized, Z = 1 is fine since normalization cancels out in the score
    pi_ast = lambda x: (1. / Z) * torch.exp(-Phi(x))  # x: (N, d) → (N,)

    def M(t, x0): return torch.exp(-t) * x0
    def S(t):
        return (1 - torch.exp(-2 * t)) * torch.eye(dim)

    def transition_density(t, x, x0):
        # x: (B, d), x0: (N, d)
        # returns: (B, N)
        Sigma = S(t)
        Sigma_inv = torch.linalg.inv(Sigma)
        det_Sigma = torch.det(Sigma)
        diff = x.unsqueeze(1) - M(t, x0).unsqueeze(0)  # (B, N, d)
        exponents = torch.einsum("bni,ij,bnj->bn", diff, Sigma_inv, diff)  # (B, N)
        norm_const = torch.sqrt((2 * math.pi) ** dim * det_Sigma)
        return torch.exp(-0.5 * exponents) / norm_const

    def pi_t(t, x):
        # x: (B, d)
        x0 = X_flat  # (N, d)
        B = x.shape[0]
        N = x0.shape[0]
        tdens = transition_density(t, x, x0)  # (B, N)
        pi_star = pi_ast(x0).flatten()  # (N,)
        exp_weights = torch.exp(torch.norm(x0, dim=1) ** 2)  # (N,)
        weights = W_flat * pi_star * exp_weights  # (N,)

        return torch.sum(tdens * weights[None, :], dim=1)  # (B,)

    def I(t, x):
        # x: (B, d)
        x0 = X_flat  # (N, d)
        Mtx = M(t, x0)  # (N, d)
        B, d = x.shape
        N = x0.shape[0]
        diff = x.unsqueeze(1) - Mtx.unsqueeze(0)  # (B, N, d)
        tdens = transition_density(t, x, x0)  # (B, N)
        pi_star = pi_ast(x0)  # (N,)
        exp_weights = torch.exp(torch.norm(x0, dim=1) ** 2)  # (N,)
        weights = W_flat * pi_star * exp_weights  # (N,)
        integrand = diff * weights[None, :, None] * tdens.unsqueeze(-1)  # (B, N, d)
        return torch.sum(integrand, dim=1)  # (B, d)

    def score(t, x):
        Sigma_inv = torch.linalg.inv(S(t))  # (d, d)
        pi_val = pi_t(t, x).unsqueeze(1)  # (B, 1)
        I_val = I(t, x)  # (B, d)
        return -torch.matmul(I_val, Sigma_inv.T) / pi_val  # (B, d)

    return pi_t, score
