"""`xFTT`: the per-problem container of one `xFTT_t` (value function + control) per
timestep, used to simulate and update the controlled SDE throughout training.
"""

import torch

from ttd.policies import Langevin_policy
from ttd.sde import sigma_func
from ttd.tt.extended import Extended_TensorTrain
from ttd.utils.numerical import nearest_index


class xFTT(object):
    def __init__(self, problem, linear_extension, default_policy=None, initial_policy=Langevin_policy, p_gradient_extension=0.15, device="cpu", max_rank=10):
        """_summary_

        Args:
            times (_type_): _description_
            problem (_type_): _description_
            alpha (_type_): _description_
        """
        self.logs = {
            "loss": [],
            "sweeps": [],
            "time": [],
            "singular_values": []
        }
        self.reg_value = [None] * problem.timeSteps
        self.times = problem.times  # sorted torch tensor of times, i.e. [0 = t_0, t_1, ..., t_N = T]
        assert(len(self.times) > 0)
        assert self.times[0] == 0 and self.times[-1] == problem.finalTime

        self.linear_extension = linear_extension
        self.p_gradient_extension = p_gradient_extension
        self.xFTT_list = None
        self.device = device

        self.problem = problem

        self.default_policy = default_policy

        # Init process corresponding to a Langevin dynamic
        self.xFTT_list = [xFTT_t(t, xTT=None, default_policy=initial_policy(t, problem), p_gradient_extension=self.p_gradient_extension, device=self.device) for t in self.times]
        # TODO: generalize init process
        # if linear_extension:
        #     self.xFTT_list = [xFTT_t(t, Zero(), 1., problem, alpha, linear_extension) for t in self.times]

    def update(self, n, xTT, c_linear):
        """
        if linear_extension is False, the value of c_linear will be ignored in the evaluation process
        """
        if self.linear_extension is not None:
            assert c_linear.dim() == 1
        t = self.times[n]
        if self.default_policy is None:
            self.xFTT_list[n] = xFTT_t(t, xTT, c_linear, self.linear_extension, p_gradient_extension=self.p_gradient_extension)
        else:
            # in case of non trivial default_policy, the value of p_gradient_extension is irrelevant hence set to 0.
            self.xFTT_list[n] = xFTT_t(t, xTT, c_linear, self.linear_extension, default_policy=self.default_policy(t, self.problem), p_gradient_extension=0.)

    def u(self, x, t):
        n_indices = nearest_index(self.times, t).tolist()
        if type(n_indices) == int:
            n_indices = [n_indices]
        # x: [b, d], t: scalar or tensor
        # n_indices: list of indices in self.times closest to t
        # Output: [b, d, len(t)] tensor, where each slice is u(x) for each time index, and [b, d] if len(t) = 1
        return torch.stack([self.xFTT_list[n].u(x) for n in n_indices], dim=-1).squeeze(2)

    def V(self, x, t):
        n_indices = nearest_index(self.times, t).tolist()
        if type(n_indices) == int:
            n_indices = [n_indices]
        return torch.stack([self.xFTT_list[n].V(x) for n in n_indices], dim=-1).squeeze(2)


class xFTT_t(object):
    def __init__(self, t, xTT=None, c_linear=0., linear_extension=None, default_policy=None, p_gradient_extension=0.15, device="cpu"):
        self.t = t
        self.xTT = xTT
        self.p_gradient_extension = p_gradient_extension
        self.device = device

        self.linear_extension = linear_extension
        # for convenience set c_linear to 0 if no linear_extension is required
        self.c_linear = 0.

        self.default_policy = default_policy
        # if self.default_policy is None:
        #     assert p_gradient_extension == 0.

        if linear_extension is not None:
            assert len(c_linear) == linear_extension.ndofs
            self.c_linear = c_linear

    def u(self, x):

        if self.xTT is not None:

            if self.default_policy is None:
                grad_part = torch.zeros_like(x)
                if self.linear_extension is not None:
                    linear_grad = torch.einsum("bdn,n->bd", self.linear_extension.grad(x, self.t), self.c_linear)
                    grad_part += linear_grad
                if isinstance(self.xTT, Extended_TensorTrain):
                    grad_part += self.xTT.grad_extended(x, const=False, p=self.p_gradient_extension)
                else:
                    grad_part += self.xTT.grad(x)

                return -torch.einsum("dt,bd->bt", sigma_func(x, self.t), grad_part)

            else:
                # in the case of a default policy the value of p_gradient_extension is not relevant
                lower_domain_bounds, upper_domain_bounds = self.xTT.tensor_basis_functions.domain_bounds
                # p = 0.05
                lower_domain_bounds = lower_domain_bounds.to(x.device)
                upper_domain_bounds = upper_domain_bounds.to(x.device)
                # x is in IR^d
                x_proj = torch.clip(x, lower_domain_bounds, upper_domain_bounds)
                # x_proj i in Tensordomain
                mask = (x != x_proj).any(dim=1)
                x_outer = x[mask]
                x_inner = x[~mask]
                res = torch.zeros_like(x)
                # outer policy values
                res[mask] = -torch.einsum("dt,bd->bt", sigma_func(x_outer, self.t), self.default_policy.grad(x_outer))

                # inner policy values
                grad_part = self.xTT.grad(x_inner)
                if self.linear_extension is not None:
                    linear_grad = torch.einsum("bdn,n->bd", self.linear_extension.grad(x_inner, self.t), self.c_linear)
                    grad_part += linear_grad
                res[~mask] = -torch.einsum("dt,bd->bt", sigma_func(x_inner, self.t), grad_part)

                return res

        else:  # self.xTT is None!
            assert self.default_policy is not None
            return -torch.einsum("dt,bd->bt", sigma_func(x, self.t), self.default_policy.grad(x))

    def V(self, x):
        if self.linear_extension is not None:
            return self.xTT(x) + (self.linear_extension(x, self.t) @ self.c_linear).reshape(-1, 1)
        return self.xTT(x)
