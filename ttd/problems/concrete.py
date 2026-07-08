"""
Each problem class merges its target's potential math (__call__, .grad, .sample) with
the tensor-train discretization it's solved with (basis, ranks, time stepping). The
class sets self.target = self in its constructor, so `problem.target(x)` /
`problem.target.grad(x)` keep working for callers that expect a separate target object
(Girsanov-weight evaluation, initial-control policies, …).
"""

import numpy as np
import torch
from scipy import integrate

from ttd.problems.base import (
    GeneralizedProblem,
    const_regularization,
    get_Basis,
    getTimes,
    handle_rank,
    scheduling,
)

torch.set_default_dtype(torch.float64)


class GaussianProblem(GeneralizedProblem):
    """Gaussian target, discretized with a functional tensor train."""

    def __init__(self, mean, cov, basis_info, rank, timeSteps=1000, batch_size=1000, T=3, reg=0.):
        assert cov.shape[0] == cov.shape[1]
        self.precision = torch.linalg.inv(cov)
        self.cov = cov
        self.mean = mean
        self.L = torch.linalg.cholesky(cov)

        super().__init__(self, cov.shape[0])

        ## TT architecture ##
        self.domain, self.basis = get_Basis(basis_info)
        self.ranks = handle_rank(rank, self.dim)
        self.fdims = self.basis.ndofs

        ## learning sample size ##
        self.batch_size = batch_size

        ## ALS ##
        self.reg_params = const_regularization(reg)

        ## Time stepping ##
        self.timeSteps = timeSteps
        self.finalTime = T
        self.stepSizes = scheduling(self.timeSteps, self.finalTime, "const")
        self.times = getTimes(self.stepSizes)
        self.times[-1] = T

    def __call__(self, x):
        return (torch.einsum("bi,ij,bj -> b", (x - self.mean), self.precision, (x - self.mean)) / 2).reshape(-1, 1)

    def grad(self, x):
        return torch.einsum("ij,bj -> bi", self.precision, (x - self.mean))

    def sample(self, N, scale):
        return scale * torch.einsum("ij, bj->bi", self.L, torch.randn(N, self.dim)) + self.mean


class Multiwell(GeneralizedProblem):
    """Separable double-well + Gaussian target, discretized with a functional tensor train.

    Called "Multiwell" in the paper (https://openreview.net/pdf?id=DDQX97Xi1Z). Despite
    the "symmetric" look of the potential, this need not be symmetric once tilt or
    x_shift are nonzero.
    """

    def __init__(self, dim, n_double_wells, basis_info, rank=2, timeSteps=1000, batch_size=1000, T=3, reg=0.,
                 alpha=1.0, delta=2.0, x_shift=0.0, tilt=0.0):
        self.n_double_wells = n_double_wells
        self.alpha = alpha
        self.delta = delta
        self.x_shift = x_shift
        self.tilt = tilt

        super().__init__(self, dim)

        ## TT architecture ##
        self.domain, self.basis = get_Basis(basis_info)
        self.ranks = handle_rank(rank, self.dim)
        self.fdims = self.basis.ndofs

        ## learning sample size ##
        self.batch_size = batch_size

        ## ALS ##
        self.reg = reg

        ## Time stepping ##
        self.timeSteps = timeSteps
        self.finalTime = T
        self.stepSizes = scheduling(self.timeSteps, self.finalTime, "const")
        self.times = getTimes(self.stepSizes)
        self.times[-1] = T

    def __call__(self, x):
        shifted_x = x - self.x_shift
        return self.alpha * (torch.sum((shifted_x[:, :self.n_double_wells] ** 2 - self.delta) ** 2
                                        + self.tilt * shifted_x[:, :self.n_double_wells], 1, keepdim=True)
                              + torch.sum(0.5 * shifted_x[:, self.n_double_wells:] ** 2, 1, keepdim=True))

    def grad(self, x):
        shifted_x = x - self.x_shift
        return self.alpha * torch.cat([
            4 * shifted_x[:, :self.n_double_wells] ** 3 - 4 * self.delta * shifted_x[:, :self.n_double_wells]
            + self.tilt * torch.ones_like(x)[:, :self.n_double_wells],
            shifted_x[:, self.n_double_wells:],
        ], 1)

    def reg_params(self, t):
        return self.reg

    def analytic_reference(self):
        """Exact normalizing constant Z and E[||X_T||^2] for this (separable) double-well
        target. Only supports x_shift=0 and tilt=0, the only cases with a closed-form
        per-coordinate density.
        """
        if self.x_shift != 0.0 or self.tilt != 0.0:
            raise NotImplementedError("analytic_reference only supports x_shift=0, tilt=0")

        alpha, delta, n_double_wells, d = self.alpha, self.delta, self.n_double_wells, self.dim

        def multi_well(x):
            return np.exp(-alpha * (x ** 2 - delta) ** 2)

        Z = integrate.quad(lambda x: multi_well(x), -np.inf, np.inf)[0]
        expectation_norm_double_well = integrate.quad(lambda x: x ** 2 * multi_well(x) / Z, -np.inf, np.inf)[0]
        expectation_norm = n_double_wells * expectation_norm_double_well + (d - n_double_wells)
        Z_reference = Z ** n_double_wells * (2 * np.pi / alpha) ** ((d - n_double_wells) / 2)
        return Z_reference, expectation_norm


class Funnel(GeneralizedProblem):
    """Funnel target, discretized with a functional tensor train."""

    def __init__(self, dim, basis_info, rank=2, timeSteps=1000, batch_size=1000, T=3, reg=0.):
        self.nu = 3.0

        super().__init__(self, dim)

        self.domain, self.basis = get_Basis(basis_info)
        self.ranks = handle_rank(rank, self.dim)
        self.fdims = self.basis.ndofs
        self.batch_size = batch_size
        self.reg = reg
        self.timeSteps = timeSteps
        self.finalTime = T
        self.stepSizes = scheduling(self.timeSteps, self.finalTime, "const")
        self.times = getTimes(self.stepSizes)
        self.times[-1] = T

    def __call__(self, x):
        # negative log-target
        return x[:, 0:1] ** 2 / (2 * self.nu ** 2) + torch.sum(x[:, 1:] ** 2 / (2 * torch.exp(x[:, 0:1])), 1, keepdim=True)

    def grad(self, x):
        return torch.cat([
            x[:, 0:1] / (self.nu ** 2) - torch.sum(x[:, 1:] ** 2 / (2 * torch.exp(x[:, 0:1])), 1, keepdim=True),
            x[:, 1:] / (2 * torch.exp(x[:, 0:1])),
        ], 1)

    def reg_params(self, x):
        return self.reg


class GinzburgLandau(GeneralizedProblem):
    """Ginzburg-Landau energy target — double-well potential on the first
    n_double_wells coordinates, harmonic on the rest, plus nearest-neighbor coupling
    between all coordinates — discretized with a functional tensor train.
    """

    def __init__(self, dim, basis_info, n_double_wells, rank=2, timeSteps=1000, batch_size=1000, T=3, reg=0.,
                 beta=1.0, kappa=1.0, delta=1.0):
        self.beta = beta
        self.kappa = kappa
        self.delta = delta
        self.n_double_wells = n_double_wells

        super().__init__(self, dim)

        self.domain, self.basis = get_Basis(basis_info)
        self.ranks = handle_rank(rank, self.dim)
        self.fdims = self.basis.ndofs
        self.batch_size = batch_size
        self.reg = reg
        self.timeSteps = timeSteps
        self.finalTime = T
        self.stepSizes = scheduling(self.timeSteps, self.finalTime, "const")
        self.times = getTimes(self.stepSizes)
        self.times[-1] = T

    def __call__(self, x):
        x_dw = x[:, :self.n_double_wells]
        x_ho = x[:, self.n_double_wells:]
        V_dw = torch.sum((x_dw ** 2 - self.delta) ** 2, dim=1, keepdim=True)
        V_ho = torch.sum(x_ho ** 2, dim=1, keepdim=True)
        diffs = x[:, 1:] - x[:, :-1]
        interaction = self.kappa * torch.sum(diffs ** 2, dim=1, keepdim=True)
        energy = V_dw + V_ho + interaction
        return self.beta * energy

    def grad(self, x):
        grad_local = torch.zeros_like(x)
        if self.n_double_wells > 0:
            x_dw = x[:, :self.n_double_wells]
            grad_local[:, :self.n_double_wells] = 4.0 * x_dw * (x_dw ** 2 - self.delta)
        if self.n_double_wells < x.shape[-1]:
            x_ho = x[:, self.n_double_wells:]
            grad_local[:, self.n_double_wells:] = 2.0 * x_ho

        grad_interaction = torch.zeros_like(x)
        grad_interaction[:, 1:] += 2 * self.kappa * (x[:, 1:] - x[:, :-1])
        grad_interaction[:, :-1] += 2 * self.kappa * (x[:, :-1] - x[:, 1:])

        return self.beta * (grad_local + grad_interaction)

    def reg_params(self, t):
        return self.reg


class Kitagawa(GeneralizedProblem):
    """Modified Kitagawa nonlinear state-space model (Kitagawa, 1996), as described in
    the paper (https://openreview.net/pdf?id=DDQX97Xi1Z, "Additional experiment:
    Kitagawa nonlinear state space model"): nonlinear transition

        x_n = 0.5*x_{n-1} + gamma*x_{n-1}/(1 + x_{n-1}^2) + v_n,

    with a *linear* Gaussian observation model y_n = x_n + w_n (chosen, unlike the
    standard benchmark, to break the symmetry and give a unimodal posterior while
    keeping the challenging high-curvature correlations). Fixed initial state x_0 = 0.
    Samples the full latent trajectory x_1..x_M jointly, so `dim` should equal `T_model`
    (=M in the paper). The paper uses sigma_v=sigma_w=1, dim=10, and varies the
    nonlinearity strength gamma in {0.5, 1.0, 1.5, 2.0}.
    """

    def __init__(self, dim, basis_info, rank=2, timeSteps=1000, batch_size=1000, T=3, reg=0.,
                 T_model=20, sigma_v=1.0, sigma_w=1.0, nonlinear_strength=0.3, data_seed=42, device="cuda"):
        self.T_model = T_model
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.gamma = nonlinear_strength
        self.data_device = device
        self.x0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        self._generate_synthetic_data(seed=data_seed)

        super().__init__(self, dim)

        self.domain, self.basis = get_Basis(basis_info)
        self.ranks = handle_rank(rank, self.dim)
        self.fdims = self.basis.ndofs
        self.batch_size = batch_size
        self.reg = reg
        self.timeSteps = timeSteps
        self.finalTime = T
        self.stepSizes = scheduling(self.timeSteps, self.finalTime, "const")
        self.times = getTimes(self.stepSizes)
        self.times[-1] = T

    def _generate_synthetic_data(self, seed):
        g = torch.Generator(device=self.data_device)
        g.manual_seed(seed)

        x_true = torch.zeros(self.T_model, device=self.data_device)
        y_generated = torch.zeros(self.T_model, device=self.data_device)
        x_prev = self.x0
        for t in range(self.T_model):
            term_dyn = 0.5 * x_prev + self.gamma * x_prev / (1 + x_prev ** 2)
            x_true[t] = term_dyn + torch.randn(1, generator=g, device=self.data_device) * self.sigma_v
            y_generated[t] = x_true[t] + torch.randn(1, generator=g, device=self.data_device) * self.sigma_w
            x_prev = x_true[t]

        self.x_true = x_true
        self.y_obs = y_generated - y_generated.mean()

    def __call__(self, x):
        if x.device != self.y_obs.device:
            x = x.to(self.y_obs.device)
        B, T = x.shape

        x0_batch = self.x0.expand(B, 1)
        x_prevs = torch.cat([x0_batch, x[:, :-1]], dim=1)
        mu_trans = (0.5 * x_prevs) + (self.gamma * x_prevs / (1 + x_prevs ** 2))
        log_prob_trans = -0.5 * torch.sum((x - mu_trans) ** 2, dim=1, keepdim=True) / (self.sigma_v ** 2)

        log_prob_obs = -0.5 * torch.sum((self.y_obs - x) ** 2, dim=1, keepdim=True) / (self.sigma_w ** 2)

        return -log_prob_trans - log_prob_obs

    def grad(self, x):
        x_in = x.detach().requires_grad_(True)
        loss = torch.sum(self.__call__(x_in))
        loss.backward()
        return x_in.grad

    def reg_params(self, t):
        return self.reg
