"""Drift/diffusion definitions for the controlled SDE, and `simulate_SDE`, which
integrates it forward via Euler-Maruyama and returns the full sample trajectory.
"""

from copy import deepcopy
from math import sqrt

import torch


class Drift(object):

    def __init__(self):
        pass

    def __call__(self, x, t):
        pass

    def div(self, x, t):
        pass


class Langevin(object):
    def __init__(self, target):
        self.target = target

    def run(self, X, N, T):
        tau = T / N
        U = deepcopy(X)
        for k in range(N):
            U = U - tau * self.target.grad(U) + sqrt(2 * tau) * torch.randn(X.shape)
        return U


class GaussianDrift(Drift):
    def __call__(self, x, t):
        """
        We define the (negative, time-reversed) drift of the forward process, i.e. f such that
            dY_s = -time_rev(f)(Y_s, s) ds + time_rev(sigma)(s) dW_s,  Y_0 ~ p_target
        Args:
            x (torch.tensor): Input of shape batch_size x dimension
            t (float): time

        Returns:
            f(x,t) (torch.tensor): shape batch_size x dimension
        """
        return x

    def div(self, x, t):
        return (torch.ones(x.shape[0], device=x.device) * x.shape[1])


def sigma_func(x, t, scale=1.):
    """
    We define the (time-reversed) diffusion of the forward process, i.e. sigma such that
            dY_s = -time_rev(f)(Y_s,s)ds + time_rev(sigma)(s)dW_s,  Y_0 ~ p_target
    Args:
        x (torch.tensor): Input of shape batch_size x dimension
        t (float): time

    Returns:
        sigma(x,t) (torch.tensor): shape batch_size x dimension x dimension
    """
    b = x.shape[0]
    d = x.shape[1]
    sig = scale * sqrt(2) * torch.eye(d, device=x.device)
    return sig


def sigma_func_inverse(x, t, scale=1.):
    """we define the (time-reversed) diffusion of the forward process, i.e. sigma such that
            dY_s = -time_rev(f)(Y_s,s)ds + time_rev(sigma)(s)dW_s,  Y_0 ~ p_target
    Args:
        x (torch.tensor): Input of shape batch_size x dimension
        t (float): time

    Returns:
        sigma(x,t) (torch.tensor): shape batch_size x dimension x dimension
    """
    b = x.shape[0]
    d = x.shape[1]
    sig = 1. / scale * 1. / sqrt(2) * torch.eye(d, device=x.device)
    return sig


def simulate_SDE(problem, u, stepSizes, batch_size, simulation_device="cpu", computation_device="cpu"):

    f = problem.f
    sigma = sigma_func
    N = len(stepSizes)
    b = batch_size
    d = problem.dim

    X_u = torch.zeros(N + 1, b, d, device=simulation_device)
    xi = torch.randn(N + 1, b, d, device=simulation_device)
    tp = torch.zeros(N + 1, device=simulation_device)
    # TODO: avoid saving sig_u
    sig_u = torch.zeros(N + 1, d, d, device=simulation_device)
    # TODO: only save u_val when requested
    u_val = torch.zeros(N + 1, b, d, device=simulation_device)

    X_u[0] = problem.start.sample(batch_size).to(simulation_device)

    t = torch.tensor(0., device=computation_device)
    for n in range(N):
        tau = stepSizes[n]
        tp[n] = t
        sig_u[n] = sigma(X_u[n], t)
        u_val[n] = u(X_u[n].to(computation_device), t).reshape(-1, d).to(simulation_device)
        X_u[n + 1] = X_u[n] + tau * (f(X_u[n], t) + u_val[n] @ sig_u[n].T) + sqrt(tau) * xi[n + 1] @ sig_u[n].T
        t += tau

    # TODO: check that t == T
    sig_u[N] = sigma(X_u[N], t)
    u_val[N] = u(X_u[N].to(computation_device), t).reshape(-1, d).to(simulation_device)
    tp[N] = t

    return X_u, xi, sig_u, u_val, tp


def forward_SDE(score, N, T, X0, stepSizes=None):
    """
    Simulate the forward SDE:
        X_{n+1} = X_n + (f + sigma * u)(X_n, t_n) * stepSize + sigma(X_n, t_n) * xi_{n+1} * sqrt(stepSize)

    Args:
        score (function): score function
        N (int): number of steps
        T (float): final time
        X0 (torch.Tensor): initial state, shape (batch_size, dim)
        stepSizes (list or tensor, optional): list of step sizes

    Returns:
        X_u (torch.Tensor): Trajectory of shape (N + 1, batch_size, dim)
    """
    b, d = X0.shape
    X_u = torch.zeros(N, b, d)
    X_u[0] = X0

    t = torch.tensor(0.0)

    for n in range(N - 1):
        tau = T / N if stepSizes is None else stepSizes[n]
        t_next = t + tau
        xi = torch.randn(b, d)
        drift = X_u[n] + 2 * score(T - t_next, X_u[n])
        noise = sqrt(2.) * xi
        X_u[n + 1] = X_u[n] + tau * drift + sqrt(tau) * noise
        t = t_next

    return X_u