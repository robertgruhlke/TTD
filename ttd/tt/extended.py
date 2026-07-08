"""`Extended_TensorTrain` (xTT): a `TensorTrain` core paired with a tensor basis,
giving a function of continuous inputs with fast evaluation, gradient, and Hessian.
Also `projected_presentation_change`, which re-projects an xTT's coefficients onto a
different basis (e.g. after the training domain shifts between timesteps).
"""

import torch

from ttd.bases.fourier import TensorExtendedFourierBasis, compute_basis_transform
from ttd.bases.legendre import TensorLegendreBasis, compute_basis_transform_Legendre
from ttd.bases.spline import TensorSplineBasis, compute_basis_transform_spline
from ttd.tt.core import TensorTrain


class Extended_TensorTrain(object):
    def __init__(self, tensor_basis_functions, ranks, comps=None, eps=1., device="cpu"):
        """
        tensor_basis_functions should be a function returning evaluations of feature functions given a data
        batch as argument, i.e. tensor_basis_functions(x), where x is an lb.array of size (batch_size, n_comps),
        is a list of lb.arrays such that tensor_basis_functions(x)[i][j,k] is the k-th feature function (in that
        dimension) evaluated at the j-th sample's i-th component.
        """
        self.tensor_basis_functions = tensor_basis_functions
        self.d = self.tensor_basis_functions.d
        self.device = device

        # TODO: also allow ranks of length d + 1, with shape [1] + [...] + [1]
        assert (len(ranks) == self.tensor_basis_functions.d + 1)
        self.rank = ranks

        self.tt = TensorTrain(tensor_basis_functions.ndofs, device=self.device)
        if comps is None:
            self.tt.fill_random(self.rank, eps)
        else:
            # TODO: allow ranks of length d + 1
            for pos in range(self.tensor_basis_functions.d - 1):
                assert (comps[pos].shape[2] == ranks[pos + 1])
            self.tt.set_components(comps, True)
        self.tt.rank = self.rank

    def save(self, path, to_cpu=False):
        state_basis = self.tensor_basis_functions.prepare_save_state(to_cpu)

        state_tt = {
            "tt_comps": self.tt.comps if not to_cpu else [c.cpu() for c in self.tt.comps],
            "rank": self.rank,
            "core_position": self.tt.core_position,
            "device": self.device,
        }
        torch.save(state_basis | state_tt, path)

    @classmethod
    def load(cls, path, map_location="cpu", set_core_position_to=0):
        state = torch.load(path, map_location=map_location, weights_only=True)
        # create an empty object without recomputation
        obj = cls.__new__(cls)
        if state["basis_type"] == "TensorLegendreBasis":
            obj.tensor_basis_functions = TensorLegendreBasis.load_from_state(state)
        elif state["basis_type"] == "TensorExtendedFourierBasis":
            obj.tensor_basis_functions = TensorExtendedFourierBasis.load_from_state(state)
        else:
            raise NotImplementedError("Other basis types not supported")

        obj.d = obj.tensor_basis_functions.d
        obj.tt = TensorTrain(obj.tensor_basis_functions.ndofs, comp_list=state["tt_comps"], device="cpu" if map_location == "cpu" else state["device"])
        obj.device = "cpu" if map_location == "cpu" else state["device"]
        obj.rank = state["rank"]
        obj.tt.rank = obj.rank

        if set_core_position_to == state["core_position"]:
            obj.tt.core_position = state["core_position"]
        else:
            obj.tt.set_core(set_core_position_to)

        return obj

    def __call__(self, x, use_einsum=True):
        assert (x.shape[1] == self.d)
        u = self.tensor_basis_functions(x)
        return self.tt.dot_rank_one_new(u, use_einsum)

    def refresh_rank(self):
        # updates self.rank according to the currently held underlying tt object
        self.rank = self.tt.rank

    def set_ranks(self, ranks):
        self.tt.retract(self, ranks, verbose=False)

    def grad_extended(self, x, const=True, p=0.15):
        # linear extension:
        # f_out = f(x_proj) + grad_inside @ (x - x_proj)
        # grad_out = grad_inside  (the gradient stays constant)

        lower_domain_bounds, upper_domain_bounds = self.tensor_basis_functions.domain_bounds
        lower_domain_bounds = lower_domain_bounds.to(x.device)
        upper_domain_bounds = upper_domain_bounds.to(x.device)
        correction = p * (upper_domain_bounds - lower_domain_bounds)  # [d]
        x_proj = torch.clip(x, lower_domain_bounds + correction, upper_domain_bounds - correction)
        grad_inside = self.grad(x_proj)

        if not const:
            # mask of points that were actually projected
            mask = (x != x_proj).any(dim=1)  # [B] bool
            if mask.any():
                idx = mask.nonzero(as_tuple=False).squeeze(-1)  # indices where projection occurred
                # compute the Hessian only for these subsamples
                H = self.hessian(x_proj[idx])  # [B_mask, d, d]
                dx = (x[idx] - x_proj[idx])  # [B_mask, d]
                add = torch.einsum("bij, bj -> bi", H, dx)  # [B_mask, d]
                grad_inside[idx] += add  # add only there

        return grad_inside


    def grad(self, x=None, grad_data=None, sample_size=None, use_stacks=True):
        """
        Computes the analytical gradient of the forward pass.

        Parameters
        ----------
        x : lb.tensor
            input of shape (batch_size, input_dim)

        Returns
        -------
        gradient : lb.tensor
            gradient of the forward pass. Shape (batch_size, input_dim)
        """
        assert (x is not None or (grad_data is not None and isinstance(sample_size, int)))

        if self.tt.core_position == 0 and use_stacks:
            return self.grad_with_stacks(x, grad_data, sample_size)
        else:
            print("WARNING: Fall back to possible slow gradient evaluation code.")
            if use_stacks:
                print("Speed up possible if core position of underlying algebraic tensor train is mu = 0.")

        if x is not None:
            assert (x.shape[1] == self.d)
            # initialize gradient
            gradient = torch.zeros((x.shape[0], self.d), device=self.device)

            # lift data to feature space and feature-derivative space
            embedded_data = self.tensor_basis_functions(x)
            embedded_data_grad = self.tensor_basis_functions.grad(x)

            for mu in range(0, self.d):
                data = embedded_data[:mu] + [embedded_data_grad[mu]] + embedded_data[mu + 1:]
                gradient[:, mu] = torch.squeeze(self.tt.dot_rank_one_new(data, use_einsum=True))
        else:
            gradient = torch.zeros((sample_size, self.d), device=self.device)
            for mu in range(0, self.d):
                gradient[:, mu] = torch.squeeze(self.tt.dot_rank_one_new(grad_data[mu], use_einsum=True))
        return gradient

    def grad_with_stacks(self, x=None, grad_data=None, sample_size=None):
        """
        Computes the analytical gradient of the forward pass.

        Parameters
        ----------
        x : lb.tensor
            input of shape (batch_size, input_dim)

        Returns
        -------
        gradient : lb.tensor
            gradient of the forward pass. Shape (batch_size, input_dim)
        """
        assert (x is not None or (grad_data is not None and isinstance(sample_size, int)))

        if x is not None:
            assert (x.shape[1] == self.d)
            # initialize gradient
            assert self.tt.core_position == 0
            b, d = x.shape

            # lift data to feature space and feature-derivative space
            if d > 1:
                # in 1d no basis evaluations are relevant, only their derivatives
                embedded_data = self.tensor_basis_functions(x)
            embedded_data_grad = self.tensor_basis_functions.grad(x)

            Rstacks = [None] * (d - 1) + [torch.ones(b, 1, device=self.device, dtype=torch.float64)]

            # build right stacks for embedded_data
            for mu in range(d - 1):
                mu_r = d - 1 - mu
                Rstacks[mu_r - 1] = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu_r], embedded_data[mu_r], Rstacks[mu_r])

            if d > 1:
                basis_val_contract_1 = torch.einsum("rms, bm -> brs", self.tt.comps[1], embedded_data[1])
                shifted_L_val_stacks = [None] * 2 + [basis_val_contract_1] + [None] * (d - 2)
                for mu in range(2, d):
                    shifted_L_val_stacks[mu + 1] = torch.einsum("brs, smR, bm -> brR", shifted_L_val_stacks[mu], self.tt.comps[mu], embedded_data[mu])

            gradient = torch.zeros((x.shape[0], self.d), device=self.device)
            # compute gradient components using R_stacks,
            # grad_mu = val_0 * val_1 * ... * derive_mu * val_{mu+1} * ... * val_d
            #         = val_0 * val_1 * ... * derive_mu * Rstacks[mu]
            for mu in range(self.d):
                # grad_mu finishes D_1 v_2 * ... * v_d at this point = entry for mu = 0
                grad_mu = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu], embedded_data_grad[mu], Rstacks[mu])

                if mu > 1:
                    # right_contract = v_{2} * .... v_{mu-1} * D_{mu} * v_{mu+1} * ... v_{d}
                    right_contract = torch.einsum("brs, bs-> br", shifted_L_val_stacks[mu], grad_mu)  # TODO: rename
                    # finally contract the core contribution v_{1} to right_contract to obtain
                    # grad_mu = v_{1} * v_{2} * .... v_{mu-1} * D_{mu} * v_{mu+1} * ... v_{d}
                    grad_mu = torch.einsum("rms, bm, bs -> br", self.tt.comps[0], embedded_data[0], right_contract)
                elif mu == 1:
                    # just add the core contribution, since already
                    # grad_mu = D_{2} v_{3} * ... v_{d}, so only v_{1} is needed from the left
                    grad_mu = torch.einsum("rms, bm, bs -> br", self.tt.comps[0], embedded_data[0], grad_mu)
                gradient[:, mu] = torch.squeeze(grad_mu)
        else:
            print("WARNING: untested code path")
            gradient = torch.zeros((sample_size, self.d), device=self.device)
            for mu in range(0, self.d):
                gradient[:, mu] = torch.squeeze(self.tt.dot_rank_one_new(grad_data[mu], use_einsum=True))
        return gradient


    def hessian_with_double_stacks(self, x):
        """
        Computes the full Hessian matrix of the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            input of shape (batch_size, d)

        Returns
        -------
        H : torch.Tensor
            Hessian matrix of shape (batch_size, d, d)
        """
        assert x.shape[1] == self.d
        assert self.tt.core_position == 0
        b, d = x.shape

        # lift data to feature space and feature-derivative space
        if d > 1:
            # in d = 1 no basis-gradient evaluations are relevant, only its 2nd derivatives
            embedded_data = self.tensor_basis_functions(x)
            embedded_data_grad = self.tensor_basis_functions.grad(x)  # list of [B, n_basis_j]
        embedded_data_D2 = self.tensor_basis_functions.D2(x)  # list of [B, n_basis_j]

        H = torch.zeros(b, self.d, self.d, device=x.device, dtype=torch.float64)

        if d == 1:
            # special case to avoid any overhead
            mu = 0
            # ranks r and s are 1, so squeeze to obtain a b-shaped tensor
            H_mu_mu = torch.einsum("rms, bm -> brs", self.tt.comps[mu], embedded_data_D2[mu])
            H[:, mu, mu] = torch.squeeze(H_mu_mu)
            return H

        # stacks of basis-value-evaluation contractions from the right
        R_val_stacks = [None] * (d - 1) + [torch.ones(b, 1, device=self.device, dtype=torch.float64)]
        # build right stacks for possible embedded_data; note for d = 1, nothing happens here
        for mu in range(d - 1, 0, -1):
            R_val_stacks[mu - 1] = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu], embedded_data[mu], R_val_stacks[mu])

        if d > 1:
            basis_val_contract_1 = torch.einsum("rms, bm -> brs", self.tt.comps[1], embedded_data[1])
            shifted_L_val_stacks = [None] * 2 + [basis_val_contract_1] + [None] * (d - 2)

            for mu in range(2, d):
                shifted_L_val_stacks[mu + 1] = torch.einsum("brs, smR, bm -> brR", shifted_L_val_stacks[mu], self.tt.comps[mu], embedded_data[mu])

            R_vDv_stacks = [None] * d
            # right stacks for first-derivative basis evals, followed by R-stack contractions of the remaining basis evals
            R_D1_stacks = [None] * (d - 1) + [torch.ones(b, 1, device=self.device, dtype=torch.float64)]
            for mu in range(d - 1):
                mu_r = d - 1 - mu
                R_D1_stacks[mu_r - 1] = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu_r], embedded_data_grad[mu_r], R_val_stacks[mu_r])

                R_vDv_stacks[mu_r - 1] = [None] * (mu_r - 1) + [R_D1_stacks[mu_r - 1]]

                for k in range(mu_r - 2, -1, -1):
                    R_vDv_stacks[mu_r - 1][k] = torch.einsum("rms, bm, bs -> br", self.tt.comps[k + 1], embedded_data[k + 1], R_vDv_stacks[mu_r - 1][k + 1])

        # diagonal entries of the Hessian
        for mu in range(self.d):
            # already finished the entries for H_11; otherwise sets D_{mu}^2 v_{mu+1} * ... v_{d}
            H_mu_mu = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu], embedded_data_D2[mu], R_val_stacks[mu])
            # contract remaining parts to the left
            if mu > 1:
                # right_contract = v_{2} * .... v_{mu-1} * D_{mu}^2 v_{mu+1} * ... v_{d}
                right_contract = torch.einsum("brs, bs-> br", shifted_L_val_stacks[mu], H_mu_mu)  # TODO: rename
                # finally contract the core contribution v_{1} to right_contract to obtain
                # H_mu_mu = v_{1} * v_{2} * .... v_{mu-1} * D_{mu}^2 v_{mu+1} * ... v_{d}
                H_mu_mu = torch.einsum("rms, bm, bs -> br", self.tt.comps[0], embedded_data[0], right_contract)
            elif mu == 1:
                # just add the core contribution, since already
                # H_mu_mu = D_{2}^2 v_{3} * ... v_{d}, so only v_{1} is needed from the left
                H_mu_mu = torch.einsum("rms, bm, bs -> br", self.tt.comps[0], embedded_data[0], H_mu_mu)

            H[:, mu, mu] = torch.squeeze(H_mu_mu)

        # off-diagonal entries
        for mu in range(self.d - 1):
            for nu in range(mu + 1, self.d):
                # case of nu > mu
                # R_vDv_stacks[nu-1][mu] = v_{mu+1} * ... * v_{nu-1} D_nu * val_{nu+1} * ... val_{d} already
                H_mu_nu = R_vDv_stacks[nu - 1][mu]

                # add the basis-derivative contribution: H_mu_nu = D_mu * v_{mu+1} * ... * v_{nu-1} D_nu * val_{nu+1} * ... val_{d}
                H_mu_nu = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu], embedded_data_grad[mu], H_mu_nu)
                # the mu == 0 case entry is already finished at this point

                # contract the remaining v_{1} * ... v_{mu-1} into H_mu_nu to finally obtain
                # H_mu_nu = v_{1} * ... v_{mu-1} * D_mu * v_{mu+1} * ... * v_{nu-1} D_nu * val_{nu+1} * ... val_{d}
                if mu == 1:
                    # only the core v_{1}
                    H_mu_nu = torch.einsum("rms, bm, bs -> br", self.tt.comps[0], embedded_data[0], H_mu_nu)
                elif mu > 1:
                    # right_contract = v_{2} * .... v_{mu-1} * D_mu * v_{mu+1} * ... * v_{nu-1} D_nu * val_{nu+1} * ... val_{d}
                    right_contract = torch.einsum("brs, bs-> br", shifted_L_val_stacks[mu], H_mu_nu)  # TODO: rename
                    # finally contract the core contribution v_{1} to right_contract to obtain
                    # H_mu_mu = v_{1} * v_{2} * .... v_{mu-1} * D_mu * v_{mu+1} * ... * v_{nu-1} D_nu * val_{nu+1} * ... val_{d}
                    H_mu_nu = torch.einsum("rms, bm, bs -> br", self.tt.comps[0], embedded_data[0], right_contract)

                H_mu_nu = torch.squeeze(H_mu_nu)
                # set symmetric entries
                H[:, mu, nu] = H_mu_nu
                H[:, nu, mu] = H_mu_nu
        return H

    def hessian_with_stacks(self, x):
        """
        Computes the full Hessian matrix of the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            input of shape (batch_size, d)

        Returns
        -------
        H : torch.Tensor
            Hessian matrix of shape (batch_size, d, d)
        """
        assert x.shape[1] == self.d
        assert self.tt.core_position == 0
        b, d = x.shape


        # lift data to feature space and feature-derivative space
        if d > 1:
            # in d = 1 no basis-gradient evaluations are relevant, only its 2nd derivatives
            embedded_data = self.tensor_basis_functions(x)
            embedded_data_grad = self.tensor_basis_functions.grad(x)  # list of [B, n_basis_j]
        embedded_data_D2 = self.tensor_basis_functions.D2(x)  # list of [B, n_basis_j]

        # stacks of basis-value-evaluation contractions from the right
        R_val_stacks = [None] * (d - 1) + [torch.ones(b, 1, device=self.device, dtype=torch.float64)]

        # build right stacks for possible embedded_data; note for d = 1, nothing happens here
        for mu in range(d - 1):
            mu_r = d - 1 - mu
            R_val_stacks[mu_r - 1] = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu_r], embedded_data[mu_r], R_val_stacks[mu_r])

        if d > 1:
            # right stacks for first-derivative basis evals, followed by R-stack contractions of the remaining basis evals
            R_D1_stacks = [None] * (d - 1) + [torch.ones(b, 1, device=self.device, dtype=torch.float64)]
            for mu in range(d - 1):
                mu_r = d - 1 - mu
                R_D1_stacks[mu_r - 1] = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu_r], embedded_data_grad[mu_r], R_val_stacks[mu_r])

        H = torch.zeros(b, self.d, self.d, device=x.device, dtype=torch.float64)
        for mu in range(self.d):
            for nu in range(mu, self.d):
                if mu == nu:  # diagonal entries of the Hessian
                    H_mu_mu = torch.einsum("rms, bm, bs -> br", self.tt.comps[mu], embedded_data_D2[mu], R_val_stacks[mu])
                    # contract remaining parts to the left
                    for k in range(mu - 1, -1, -1):
                        H_mu_mu = torch.einsum("rms, bm, bs -> br", self.tt.comps[k], embedded_data[k], H_mu_mu)
                    H[:, mu, nu] = torch.squeeze(H_mu_mu)
                else:
                    # case of nu > mu
                    H_mu_nu = R_D1_stacks[nu - 1]  # R_D1_stacks[nu] already contains D_nu * val_{nu+1} * ... val_{d}
                    for k in range(nu - 1, -1, -1):
                        if k == mu:
                            # add the basis-derivative contribution
                            H_mu_nu = torch.einsum("rms, bm, bs -> br", self.tt.comps[k], embedded_data_grad[k], H_mu_nu)
                        else:
                            # just add the basis-value contribution
                            H_mu_nu = torch.einsum("rms, bm, bs -> br", self.tt.comps[k], embedded_data[k], H_mu_nu)
                    H_mu_nu = torch.squeeze(H_mu_nu)
                    # set symmetric entries
                    H[:, mu, nu] = H_mu_nu
                    H[:, nu, mu] = H_mu_nu
        return H

    def hessian(self, x, use_stacks=True):
        """
        Computes the full Hessian matrix of the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            input of shape (batch_size, d)

        Returns
        -------
        H : torch.Tensor
            Hessian matrix of shape (batch_size, d, d)
        """
        assert x.shape[1] == self.d

        if self.tt.core_position == 0 and use_stacks:
            return self.hessian_with_double_stacks(x)
        else:
            print("WARNING: Fall back to possible slow Hessian evaluation code")
            if use_stacks:
                print("Speed up possible if core position of underlying algebraic tensor train is mu = 0.")

        B = x.shape[0]
        H = torch.zeros(B, self.d, self.d, device=x.device, dtype=torch.float64)

        # prepare basis and derivative values
        embedded_data = self.tensor_basis_functions(x)  # list of [B, n_basis_j]
        embedded_data_grad = self.tensor_basis_functions.grad(x)  # list of [B, n_basis_j]
        embedded_data_D2 = self.tensor_basis_functions.D2(x)  # list of [B, n_basis_j]

        # TODO: use symmetry in the for loop
        for mu in range(self.d):
            for nu in range(self.d):
                if mu == nu:
                    # second derivative in the same direction
                    data = embedded_data[:mu] + [embedded_data_D2[mu]] + embedded_data[mu + 1:]
                    H[:, mu, nu] = torch.squeeze(self.tt.dot_rank_one_new(data, True))
                else:
                    # cross derivative: grad_mu * grad_nu
                    data = []
                    for j in range(self.d):
                        if j == mu:
                            data.append(embedded_data_grad[j])
                        elif j == nu:
                            data.append(embedded_data_grad[j])
                        else:
                            data.append(embedded_data[j])
                    H[:, mu, nu] = torch.squeeze(self.tt.dot_rank_one_new(data, True))

        return H

    def modify_ranks(self, rule, verbose):
        rank_info = self.tt.modify_ranks(rule, verbose)
        self.rank = self.tt.rank
        return rank_info


def projected_presentation_change(xtt_previous, xtt_next, inner_univariate_product="H2", n_quadrature=100):
    """
    Adapts the coefficients of xtt_previous to a new basis via local L2/H1/H2 projections.

    Args:
        xtt_previous: an Extended_TensorTrain with fixed components.
        xtt_next: an Extended_TensorTrain with different basis functions (possibly defined on
            another domain), for which the coefficients of xtt_previous are adapted via local
            "L2", "H1", "H2" projections.

    Returns:
        None. xtt_next.tt.comps is updated in place, and its core is set to position 0.
    """
    assert xtt_previous.d == xtt_next.d
    for r1, r2 in zip(xtt_previous.rank, xtt_next.rank):
        if not r1 == r2:
            raise NotImplementedError("Projected representation_change for rank change not implemented yet.")
    if isinstance(xtt_previous.tensor_basis_functions, TensorExtendedFourierBasis) and isinstance(xtt_next.tensor_basis_functions, TensorExtendedFourierBasis):
        for k, comp in enumerate(xtt_previous.tt.comps):
            T_k = compute_basis_transform(xtt_previous.tensor_basis_functions.bases[k], xtt_next.tensor_basis_functions.bases[k], inner_univariate_product, n_quadrature, device=xtt_previous.device)
            xtt_next.tt.comps[k] = torch.einsum("ij, rjs->ris", T_k, comp)

    if isinstance(xtt_previous.tensor_basis_functions, TensorSplineBasis) and isinstance(xtt_next.tensor_basis_functions, TensorSplineBasis):
        for k, comp in enumerate(xtt_previous.tt.comps):
            T_k = compute_basis_transform_spline(xtt_previous.tensor_basis_functions.splines[k], xtt_next.tensor_basis_functions.splines[k], inner_univariate_product, n_quadrature, device=xtt_previous.device)
            xtt_next.tt.comps[k] = torch.einsum("ij, rjs->ris", T_k, comp)

    elif isinstance(xtt_previous.tensor_basis_functions, TensorLegendreBasis) and isinstance(xtt_next.tensor_basis_functions, TensorLegendreBasis):
        T_list = compute_basis_transform_Legendre(xtt_previous.tensor_basis_functions, xtt_next.tensor_basis_functions, inner_univariate_product, n_quadrature, device=xtt_previous.device)
        for k, comp in enumerate(xtt_previous.tt.comps):
            T_k = T_list[k]
            xtt_next.tt.comps[k] = torch.einsum("ij, rjs->ris", T_k, comp)

    xtt_next.tt.set_core(0)
