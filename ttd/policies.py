"""Initial-control policies for outer iteration 1 (`Langevin_policy`,
`annealed_Langevin`, `quadratic_annealed_Langevin`), plus `Linear_Extension` and its
subclasses, which extend a control linearly beyond the domain it was trained on.
"""

import torch

from ttd.sde import sigma_func_inverse


class annealed_Langevin(object):
    def __init__(self, t, problem):
        self.log_prior = problem.start
        self.log_target = problem.target
        self.t = t
        self.f = problem.f
        self.problem = problem

    def alpha(self, t):
        return t / self.problem.finalTime

    # def __call__(self, x):
    #     return ((1 - self.alpha(self.t)) * self.log_prior(x) + self.alpha(self.t) * self.log_target(x)).reshape(-1, self.ndofs)

    def grad(self, x):
        annealed_log_target = ((1 - self.alpha(self.t)) * self.log_prior.grad(x) + self.alpha(self.t) * self.log_target.grad(x))
        result = torch.einsum("ij, bj -> bi", sigma_func_inverse(x, self.t), self.f(x, self.t) + annealed_log_target)
        return torch.einsum("ji, bj -> bi", sigma_func_inverse(x, self.t), result)


class quadratic_annealed_Langevin(object):
    def __init__(self, t, problem):
        self.log_prior = problem.start
        self.log_target = problem.target
        self.t = t
        self.f = problem.f
        self.problem = problem

    def alpha(self, t):
        return (t / self.problem.finalTime) ** 2

    def grad(self, x):
        annealed_log_target = ((1 - self.alpha(self.t)) * self.log_prior.grad(x) + self.alpha(self.t) * self.log_target.grad(x))
        result = torch.einsum("ij, bj -> bi", sigma_func_inverse(x, self.t), self.f(x, self.t) + annealed_log_target)
        return torch.einsum("ji, bj -> bi", sigma_func_inverse(x, self.t), result)


class Langevin_policy(object):
    def __init__(self, t, problem):
        self.t = t
        self.f = problem.f
        self.target = problem.target

    def grad(self, x):
        result = torch.einsum("ij, bj -> bi", sigma_func_inverse(x, self.t), self.f(x, self.t) + self.target.grad(x))
        return torch.einsum("ji, bj -> bi", sigma_func_inverse(x, self.t), result)


class Linear_Extension(object):
    def __init__(self, ndofs):
        self.ndofs = ndofs

    def __call__(self, x, t):
        """
        Returns a tensor of shape b x ndofs, where x.shape = (b,d)
        """
        pass

    def grad(self, x, t):
        """
        Returns a tensor of shape b x d x ndofs, where x.shape = (b,d)
        """
        pass


class Interpolating(Linear_Extension):
    def __init__(self, alpha, log_prior, log_target):
        self.alpha = alpha
        self.log_prior = log_prior
        self.log_target = log_target
        self.ndofs = 1

    def __call__(self, x, t):
        return ((1 - self.alpha(t)) * self.log_prior(x) + self.alpha(t) * self.log_target(x)).reshape(-1, self.ndofs)

    def grad(self, x, t):
        return ((1 - self.alpha(t)) * self.log_prior.grad(x) + self.alpha(t) * self.log_target.grad(x)).reshape(-1, x.shape[1], self.ndofs)


class Dictionary(Linear_Extension):
    def __init__(self, basis_list):
        self.ndofs = len(basis_list)
        self.basis = basis_list

    def __call__(self, x, t):
        return torch.stack([b(x).reshape(-1, 1) for b in self.basis], dim=2).squeeze(1)  # b x ndofs

    def grad(self, x, t):
        return torch.stack([b.grad(x).reshape(-1, x.shape[1]) for b in self.basis], dim=2)  # b x d x ndofs
