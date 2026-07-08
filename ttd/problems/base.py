"""`Target` (potential + gradient + optional sample) and `GeneralizedProblem`, which
pairs a target with the TT discretization it's solved with. Also bare Target
implementations with no TT discretization (`StandardNormal`, `GaussianMixture`,
`SymmetricGaussianMixture2D`) and shared helpers (basis construction, rank handling,
regularization/time-stepping schedules).
"""

from math import sqrt

import numpy as np
import torch
from scipy.optimize import minimize

from ttd.bases.fourier import TensorExtendedFourierBasis
from ttd.bases.legendre import TensorLegendreBasis
from ttd.bases.spline import TensorSplineBasis, TensorSplineBasis_Equidistant
from ttd.sde import GaussianDrift, sigma_func_inverse
from ttd.utils.sampling import rejectionSampler

torch.set_default_dtype(torch.float64)


class Target(object):
    """A Target is a function class with a __call__ routine and a .grad routine.
       Using those two routines, the targetVector routine computes
            f(x) + sigma grad_f(x),
        where f is the target. This is used in the fit when the target is added as a degree of freedom to
        to the FTT basis.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        pass

    def grad(self, x):
        pass

    def unnormalizedDensity(self, x):
        return torch.exp(-self(x))

    def targetVector(self, sigma_vals, x):
        return (self(x) + torch.einsum("bd,bd->b", sigma_vals, self.grad(x)))

    def sample(self, N):
        raise NotImplementedError("Sampling has not been implemented for this Target object")


class StandardNormal(Target):

    def __init__(self, dim=2):
        self.dim = dim

    def __call__(self, x):
        return (torch.einsum("bi,bi -> b", x, x) / 2).reshape(-1, 1)

    def grad(self, x):
        return x

    def sample(self, N):
        return torch.randn(N, self.dim)


class GaussianMixture(Target):
    def __init__(self, means, covs):
        self.dim = len(means[0])
        for mean in means:
            assert len(mean) == self.dim
        for cov in covs:
            assert cov.shape[0] == cov.shape[1] and cov.shape[0] == self.dim

        self.covs = covs
        self.precs = [torch.linalg.inv(cov) for cov in covs]
        self.means = means
        self.numPoints = len(means)
        self.weights = [1. / self.numPoints * 1. / ((2 * torch.pi) ** (self.dim / 2) * torch.det(cov)) for cov in self.covs]

    def normalizedDensity(self, x):
        val = sum(w * torch.exp(-torch.einsum("bi,ij, bj -> b", x - mu, P, x - mu) / 2.) for w, mu, P in zip(self.weights, self.means, self.precs))
        return val

    def __call__(self, x):
        return -torch.log(self.normalizedDensity(x))

    def grad(self, x):
        grad = torch.zeros_like(x)
        for w, mu, P in zip(self.weights, self.means, self.precs):
            inner_grad = -torch.einsum("ij, bj->bi", P, x - mu)  # (P @ (x - mu).T).T
            val = w * torch.exp(-torch.einsum("bi,ij, bj -> b", x - mu, P, x - mu) / 2.)
            grad += torch.einsum("bd, b-> bd", inner_grad, val)
        # D log(f(x)) = 1/f(x) * Df(x), where Df(x) = grad
        return -torch.einsum("bd, b-> bd", grad, 1. / self.normalizedDensity(x))


class SymmetricGaussianMixture2D(Target):
    def __init__(self, means=[torch.tensor([sqrt(2.), sqrt(2.)])], vars=[1.]):
        self.dim = len(means[0])
        self.vars = [var for var in vars]
        self.means = [mean for mean in means]
        self.numPoints = len(means)
        self.weights = [1. / self.numPoints * var / (2 * torch.pi) for var in self.vars]

    def __call__(self, x):
        return -torch.log(self.normalizedDensity(x))

    def grad(self, x):
        grad = torch.zeros_like(x)
        for i in range(self.numPoints):
            inner_grad = -(x - self.means[i]) * self.vars[i]  # nabla f_i(x) in exp(f_i(x) shape)
            val = self.weights[i] * torch.exp(-torch.einsum("bi,bi -> b", (x - self.means[i]), (x - self.means[i])) * self.vars[i] / 2)
            grad += torch.einsum("bd, b-> bd", inner_grad, val)

        # D log(f(x)) = 1/f(x) * Df(x), where Df(x) = grad
        return -torch.einsum("bd, b-> bd", grad, 1. / self.normalizedDensity(x))

    def autograd(self, x):
        xx = x.clone().detach().requires_grad_(True)
        log_density = self.__call__(xx)
        log_density.backward()
        return xx.grad

    def normalizedDensity(self, x):
        val = sum(self.weights[i] * torch.exp(-torch.einsum("bi,bi -> b", (x - self.means[i]), (x - self.means[i])) * self.vars[i] / 2) for i in range(self.numPoints))
        return val

    def sample(self, N):
        """generate N samples from the posterior exp(-Phi) defined by the potential.

        Args:
            N (int): number of samples

        Returns:
            samples (torch.tensor): sample matrix of shape (D,N)
        """

        samples = torch.zeros((N, 2))
        # uniform distribution over the individual Gaussians
        indices = torch.randint(low=0, high=self.numPoints, size=(N,))
        counts = torch.bincount(indices).tolist()
        counts += [0] * (self.numPoints - len(counts))
        counts = torch.tensor(counts)
        last_count = 0
        count = 0
        for i in range(self.numPoints):
            count += counts[i]
            # samples[:,last_count:count] = gaussians[i].sample((2,counts[i]))
            samples[last_count:count, :] = self.means[i][None, :] + 1. / sqrt(self.vars[i]) * torch.randn((counts[i], 2))
            last_count += counts[i]

        return samples

    def sample(self, N):

        raise NotImplementedError("Target samples not implemented")

        assert self.dim == 2

        # Z = get_normalization(self, xlim=[-3.5, 3.5], ylim=[-3.5, 3.5])
        # Z = 6.426226281490308
        # Z = 5356.1648857703985
        Z = 5356.1647880005085172525412011550335662616899287735394296674624909  # credits wolfram alpha
        target_density = lambda x: 1 / Z * self.unnormalizedDensity(x)
        proposal_density = SymmetricGaussianMixture2D(means=[torch.tensor([sqrt(2.), sqrt(2.)]),
                                                               torch.tensor([sqrt(2.), -sqrt(2.)]),
                                                               torch.tensor([-sqrt(2.), sqrt(2.)]),
                                                               torch.tensor([-sqrt(2.), -sqrt(2.)])],
                                                    vars=[4., 4., 4., 4.])

        def quotient(x):
            x_torch = torch.from_numpy(x[None, :])
            res = (target_density(x_torch) / proposal_density.normalizedDensity(x_torch)).squeeze()
            return res.item()

        def log_quotient(x):
            x_torch = torch.from_numpy(x[None, :])
            res = (-self(x_torch) - np.log(proposal_density.normalizedDensity(x_torch))).squeeze()
            return res.item()

        if self.cached_M is None:
            K = 100
            Ms = []
            for k in range(K):
                x0 = 2 * torch.rand((2,))

                res = minimize(lambda x: -log_quotient(x), x0=x0, method="BFGS", tol=1e-8)
                Ms.append(quotient(res.x))

            x0 = torch.tensor([sqrt(2.), sqrt(2.)])
            res = minimize(lambda x: -log_quotient(x), x0=x0, method="BFGS", tol=1e-8)
            print("x0_sqrt = ", quotient(res.x))
            print(max(Ms))
            Ms.append(quotient(res.x))
            M = max(Ms)
            self.cached_M = M
        else:
            M = self.cached_M

        # for numerical instabilities in optimization
        M *= (1 + 1e-3)

        # M = 1.2555450895224476 + 1e-4
        return rejectionSampler(N, target_density, proposal_density, M)


def scheduling(nTimeSteps, T, schedule):
    if schedule == "const":
        stepSize = T / nTimeSteps
        stepSizes = [stepSize] * nTimeSteps

    elif schedule == "linear":
        c = T / (nTimeSteps * (nTimeSteps - 1) / 2)
        stepSizes = [c * (i + 1) for i in range(nTimeSteps)]
        stepSizes.reverse()

    elif schedule == "quadratic":
        # c * sum_{i=0}^{nTimeSteps} i^2 = finalTime
        c = T / (nTimeSteps * (nTimeSteps + 1) * (2 * nTimeSteps + 1) / 6.)
        stepSizes = [c * (i + 1) ** 2 for i in range(nTimeSteps)]
        stepSizes.reverse()

    else:
        raise NotImplementedError("Other schedules are not implemented")

    return stepSizes


def handle_rank(rank, d):
    ranks = None  # TT rank in format [1, r_1, ..., r_{d-1}, 1]
    if isinstance(rank, int):
        ranks = [1] + [rank] * (d - 1) + [1]
    elif isinstance(rank, (list, tuple)):
        if len(rank) == d + 1:
            assert rank[0] == 1 and rank[-1] == 1
            ranks = rank
        elif len(rank) == d - 1:
            ranks = [1] + rank + [1]
        else:
            raise ValueError("Given rank is not in correct length or format: list/tuple of length d+1/d-1. : ", rank)
    else:
        raise ValueError("Given rank is not in correct format: list/tuple or int: ", rank)
    return ranks


def absshape_reguarlization(finaltime, left, mid, right):
    #            .....right
    # left..      /
    #       \   /
    #        \./..... mid
    def reg(t):
        if t <= 0.5 * finaltime:
            return left - (left - mid) / (0.5 * finaltime) * t
        else:
            return mid + (right - mid) / (0.5 * finaltime) * (t - 0.5 * finaltime)
    return reg


def const_regularization(const_val):
    return lambda t: const_val


def get_Basis(basis_info):
    device = basis_info["device"]
    domain = basis_info["lims"]
    if basis_info["type"] == "TensorSplineBasis":
        grid_points = [torch.linspace(domain[i][0], domain[i][1], basis_info["nknots"][i]) for i in range(len(domain))]
        basis = TensorSplineBasis(grid_points, basis_info["deg"], basis_info["s"], device, orthonormalize=basis_info["orthonormalize"])
    elif basis_info["type"] == "TensorLegendreBasis":
        basis = TensorLegendreBasis(domain, basis_info["deg"], basis_info["orthonormalize"], device=device)
    elif basis_info["type"] == "TensorExtendedFourierBasis":
        basis = TensorExtendedFourierBasis(domain, basis_info["n_basis"], include_linear=basis_info["include_linear"], include_quadratic=basis_info["include_quadratic"], orthonormalize=basis_info["orthonormalize"], device=device)
    elif basis_info["type"] == "TensorSplineBasis_Equidistant":
        basis = TensorSplineBasis_Equidistant(domain, basis_info["nknots"], basis_info["deg"], basis_info["s"], device, orthonormalize=basis_info["orthonormalize"])

    return domain, basis


def get_basis_info(basis):
    if isinstance(basis, TensorSplineBasis):
        lims = [[knots[0], knots[-1]] for knots in basis.knots_list]
        nknots = [len(knots) for knots in basis.knots_list]
        basis_info = {"type": "TensorSplineBasis",
                      "device": basis.device,
                      "lims": lims,
                      "nknots": nknots,
                      "deg": basis.p_list,
                      "s": basis.s_list,
                      "orthonormalize": basis.orthonormalize}
    elif isinstance(basis, TensorLegendreBasis):
        lims = [[knots[0], knots[-1]] for knots in basis.domain_list]
        basis_info = {"type": "TensorLegendreBasis",
                      "lims": lims,
                      "deg": basis.deg_list,
                      "orthonormalize": basis.orthonormalize,
                      "device": basis.device}
    elif isinstance(basis, TensorExtendedFourierBasis):
        lims = [[a, b] for a, b in zip(*basis.domain_bounds)]
        basis_info = {"type": "TensorExtendedFourierBasis",
                      "device": basis.device,
                      "lims": lims,
                      "n_basis": basis.ndofs,
                      "orthonormalize": basis.orthonormalize,
                      "include_linear": basis.include_linear,
                      "include_quadratic": basis.include_quadratic}
    elif isinstance(basis, TensorSplineBasis_Equidistant):
        lims = [[a, b] for a, b in zip(*basis.domain_bounds)]
        nknots = basis.nknots
        basis_info = {"type": "TensorSplineBasis_Equidistant",
                      "device": basis.device,
                      "lims": lims,
                      "nknots": nknots,
                      "deg": basis.p_list,
                      "s": basis.s_list,
                      "orthonormalize": basis.orthonormalize}
    else:
        raise NotImplementedError("method not implemented for basis of instance", type(basis))

    return basis_info


class GeneralizedProblem(Target):
    """Base class for a solvable problem: a target (potential + gradient, inherited from
    Target) plus the tensor-train discretization it's solved with. Concrete subclasses
    implement __call__/.grad themselves and set self.target = self, so callers that
    expect a separate target object (problem.target(x), problem.target.grad(x)) keep
    working.
    """
    def __init__(self, target, dim):
        self.target = target
        self.start = StandardNormal(dim)
        self.f = GaussianDrift()
        self.dim = dim

    def u0(self, x, t):
        """Initial control corresponding to unadjusted Langevin dynamics.

        Args:
            x (torch.tensor): Input of shape batch_size x dimension
            t (float): time

        Returns:
            u (torch.tensor): shape batch_size x dimension
        """
        v = self.f(x, t) + self.target.grad(x)
        return -torch.einsum("ij, bj -> bi", sigma_func_inverse(x, t), v)

    def getTTparams(self, basis_info=None):
        if basis_info is not None:
            self.domain, self.basis = get_Basis(basis_info)
        return self.basis, self.ranks


def getTimes(stepsizes_list):
    """Build cumulative time points [t_0=0, ..., t_N] from a list of step sizes tau_0..tau_{N-1}."""
    stepsizes = torch.tensor(stepsizes_list)
    t_list = torch.cat([torch.zeros(1, dtype=stepsizes.dtype, device=stepsizes.device),
                        torch.cumsum(stepsizes, dim=0)])
    return t_list
