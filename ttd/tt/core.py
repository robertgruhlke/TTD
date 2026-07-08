"""Algebraic tensor-train core: the `TensorTrain` class (orthogonalization, rounding,
rank truncation/modification) and free functions for combining tensor trains
(addition, Hadamard product).
"""

import torch

torch.set_default_dtype(torch.float64)
import time
from copy import deepcopy

from colorama import Fore, Style


def common_field_dtype(tensor_list1, tensor_list2):
    for t in tensor_list1:
        assert t.dtype in [torch.cfloat, torch.get_default_dtype()]
        if t.dtype == torch.cfloat:
            return torch.cfloat

    for t in tensor_list2:
        assert t.dtype in [torch.cfloat, torch.get_default_dtype()]
        if t.dtype == torch.cfloat:
            return torch.cfloat

    return torch.get_default_dtype()


def get_ranks(TT_list):
    return [1] + [C.shape[2] for C in TT_list[:-1]] + [1]


def laplace_like_sum(TT_list, TT2_list):
    # @pre each item of the TT_list is given as:
    #  LaplaceLikeOperator = sum_mu : I o I ... o L_mu o ... I    with L_mu at the mu-th position
    # in the implementation LLO = [L_1, ..., L_d]

    # the resulting sum of d tensor trains with this operator action then defines a tensor train of rank 2r
    assert len(TT_list) == len(TT2_list)
    for mu in range(len(TT_list)):
        assert TT_list[mu].shape == TT2_list[mu].shape

    comps = []

    # set first component
    L = TT2_list[0]
    C = TT_list[0]
    data = torch.cat([deepcopy(L[0, :, :]), C[0, :, :]], axis=1)
    data = data.reshape(1, data.shape[0], data.shape[1])
    comps.append(data)

    # set middle components
    for C, L in zip(TT_list[1:-1], TT2_list[1:-1]):
        rp1, _, rpp1 = C.shape
        rp2, _, rpp2 = L.shape

        ldtype = common_field_dtype([C], [L])

        data = torch.zeros((rp1 + rp2, C.shape[1], rpp1 + rpp2), dtype=ldtype)
        data[:rp1, :, :rpp1] = deepcopy(C)
        data[rp1:, :, :rpp1] = deepcopy(L)
        data[rp1:, :, rpp1:] = deepcopy(C)
        comps.append(data)

    C, L = TT_list[-1], TT2_list[-1]
    data = torch.cat([deepcopy(C)[:, :, 0], L[:, :, 0]], axis=0)
    data = data.reshape(data.shape[0], data.shape[1], 1)
    comps.append(data)

    return comps


def TT_add(TT_1, TT_2, a=1., b=1.):
    TT1 = TT_1.comps
    TT2 = TT_2.comps
    assert len(TT1) == len(TT2)
    assert TT_1.dims == TT_2.dims
    for mu in range(len(TT1)):
        assert TT1[mu].shape[1] == TT2[mu].shape[1]
    TT = []
    # set first component
    data = torch.cat([deepcopy(TT1[0][0, :, :]), deepcopy(TT2[0][0, :, :])], axis=1)
    data = data.reshape(1, data.shape[0], data.shape[1])
    TT.append(data)

    # set middle components
    for p in range(1, len(TT1) - 1):
        # r_{i}^1  di  r_{i+1}^1
        c1, c2 = TT1[p], TT2[p]
        rp1, _, rpp1 = c1.shape
        # r_{i}^2  di  r_{i+1}^2
        rp2, _, rpp2 = c2.shape
        data = torch.zeros((rp1 + rp2, c1.shape[1], rpp1 + rpp2))
        data[:rp1, :, :rpp1] = deepcopy(c1)
        data[rp1:, :, rpp1:] = deepcopy(c2)
        TT.append(data)

    # set last component
    data = torch.cat([deepcopy(a * TT1[-1][:, :, 0]), deepcopy(b * TT2[-1][:, :, 0])], axis=0)
    data = data.reshape(data.shape[0], data.shape[1], 1)
    TT.append(data)

    Added_TT = TensorTrain(TT_1.dims, TT)

    return Added_TT


def prod(list):
    res = list[0]
    for i in range(1, len(list)):
        res = res * list[i]
    return res


# upper bound for ranks
def max_ranks(degrees):
    dofs = [degree + 1 for degree in degrees]
    max_rank = [1] \
                    + [min(prod(dofs[:k + 1]), prod(dofs[k + 1:])) for k in range(len(dofs) - 1)] \
                    + [1]
    return max_rank


class Threshold(object):
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, u, sigma, v, pos):
        return max([torch.sum(sigma > self.delta), 1])


def rankstepsizecontrol(ETT, F, tau, truncation_rank, rtol, max_rank, verbose=False, bisection_steps=5):
    def get_Atrun_A(dt):
        A_trun = TT_add(ETT, F, a=1., b=dt)
        A = deepcopy(A_trun)
        A_trun.rank_truncation(truncation_rank)
        return A_trun, A

    def relative_truncation_error(dt):
        A_trun, A = get_Atrun_A(dt)
        diff = TT_add(A_trun, A, a=1., b=-1)
        diff.rank_truncation(max_rank)
        return TensorTrain.frob_norm(diff).item() / TensorTrain.frob_norm(A).item()

    # init the tau_rank test
    tau_rank = tau

    # halve tau_rank until the relative error requirement is met
    while relative_truncation_error(tau_rank) > rtol:
        tau_rank /= 2.
        if verbose:
            print("tau_rank halved")

    # if no halving step was performed, take the proposed tau as the candidate
    if tau_rank == tau:
        return tau

    # otherwise perform a bisection search to potentially increase tau_rank
    tau_rank_bounds = [tau_rank, 2 * tau_rank]
    for k in range(bisection_steps):
        tau_bound_candidate = 0.5 * sum(tau_rank_bounds)
        if relative_truncation_error(tau_bound_candidate) < rtol:
            tau_rank_bounds[0] = tau_bound_candidate
        else:
            tau_rank_bounds[1] = tau_bound_candidate

        if verbose:
            print("{k+1}th bisection step : ", tau_rank_bounds)

    # the lower tau_rank bound is a lower bound for the optimal tau_rank
    tau_rank = tau_rank_bounds[0]
    return tau_rank


def left_unfolding(order3tensor):
    s = order3tensor.shape
    return order3tensor.reshape(s[0] * s[1], s[2])


def right_unfolding(order3tensor):
    s = order3tensor.shape
    return order3tensor.reshape(s[0], s[1] * s[2])

class TensorTrain(object):
    def __init__(self, dims, comp_list=None, device="cpu", deep_copy=False):
        self.n_comps = len(dims)
        self.dims = dims
        self.comps = [None] * self.n_comps
        self.device = device

        self.rank = None
        self.core_position = None

        # upper bound for ranks
        self.uranks = [1] + [min(prod(dims[:k + 1]), prod(dims[k + 1:])) for k in range(len(dims) - 1)] + [1]

        if comp_list is not None:
            self.set_components(comp_list, deep_copy)

    @staticmethod
    def hadamard_product(A, B):
        """Computes <A,B> = AoB with o being the Hadamard product."""
        if isinstance(A, TensorTrain) and isinstance(B, TensorTrain):
            assert len(A.dims) == len(B.dims)
            for d in range(len(A.dims)):
                assert A.dims[d] == B.dims[d]

            n_comps = len(A.dims)
            d = A.dims[0]

            v = sum(torch.kron(A.comps[0][:, i, :], B.comps[0][:, i, :]) for i in range(d))

            for pos in range(1, n_comps):
                d = A.dims[pos]
                rA = A.comps[pos].shape[0]
                rB = B.comps[pos].shape[0]
                v = sum(v @ torch.einsum("ij,kl -> ikjl", A.comps[pos][:, i, :], B.comps[pos][:, i, :]).reshape(rA * rB, -1) for i in range(d))
            return v

        elif isinstance(A, list) and isinstance(B, list):
            assert len(A) == len(B)
            for c_A, c_B in zip(A, B):
                assert c_A.shape[1] == c_B.shape[1]

            n_comps = len(A)
            d = A[0].shape[1]
            v = sum(torch.kron(A[0][:, i, :], B[0][:, i, :]) for i in range(d))

            for mu in range(1, n_comps):
                d = A[mu].shape[1]
                rA = A[mu].shape[0]
                rB = B[mu].shape[0]
                v = sum(v @ torch.einsum("ij,kl -> ikjl", A[mu][:, i, :], B[mu][:, i, :]).reshape(rA * rB, -1) for i in range(d))
            return v

        else:
            raise NotImplementedError("Only TensorTrain/TensorTrain or component_list/component_list implemented.")

    # TODO: rename
    @staticmethod
    def skp(A, B):
        return TensorTrain.hadamard_product(A, B)

    @staticmethod
    def frob_norm(A):
        if A.core_position is not None:
            non_orth_comp = A.comps[A.core_position]
            return torch.sqrt((non_orth_comp * non_orth_comp).sum())
        return torch.sqrt(TensorTrain.skp(A, A))

    @staticmethod
    def frob_norm_squared(A):
        if A.core_position is not None:
            non_orth_comp = A.comps[A.core_position]
            return (non_orth_comp * non_orth_comp).sum()
        return TensorTrain.skp(A, A)

    @staticmethod
    def hsvd(A_full, ranks=None):
        """Obtains a TensorTrain from a full tensor via higher-order SVD."""
        d = len(A_full.shape)
        shapes = A_full.shape
        A_mat = A_full

        # if no ranks are provided, choose the maximum possible ranks
        if ranks is None:
            ranks = [1] + [min(prod(shapes[:mu + 1]), prod(shapes[mu + 1:])) for mu in range(d - 1)] + [1]

        comps = []
        for mu in range(d - 1):
            A_mat = A_mat.reshape((ranks[mu] * shapes[mu], -1))
            u, sigma, vt = torch.linalg.svd(A_mat)
            # truncation:
            u, sigma, vt = u[:, :ranks[mu + 1]], sigma[:ranks[mu + 1]], vt[:ranks[mu + 1], :]

            u_comp = u.reshape(ranks[mu], shapes[mu], ranks[mu + 1])
            comps.append(u_comp)

            A_mat = torch.diag(sigma) @ vt

        comps.append(A_mat.unsqueeze(2))
        return comps

    def set_components(self, comp_list, deep_copy):
        """
        @param comp_list: list of order-3 tensors representing the component tensors
                           = [C1, ..., Cd] with shape
                           Ci.shape = (ri, self.dims[i], ri+1)

                           with convention r0 = rd = 1
        """
        # the length of the component list has to match
        assert (len(comp_list) == self.n_comps)

        # each component must be an order-3 tensor object
        for pos in range(self.n_comps):
            assert (len(comp_list[pos].shape) == 3)

        # the given components' inner dimension must match the predefined fixed dimensions
        for pos in range(self.n_comps):
            assert (comp_list[pos].shape[1] == self.dims[pos])

        # neighboring rank sizes must match
        for pos in range(self.n_comps - 1):
            assert (comp_list[pos].shape[2] == comp_list[pos + 1].shape[0])

        # set the components
        for pos in range(self.n_comps):
            if deep_copy:
                self.comps[pos] = deepcopy(comp_list[pos])
            else:
                self.comps[pos] = comp_list[pos]

    def fill_random(self, ranks, eps):
        """
        Fills the TensorTrain with random elements for a given rank structure.
        If entries in the TensorTrain object have been set previously, they are overwritten
        regardless of the existing rank structure.

        @param ranks: list
        """
        self.rank = ranks

        for pos in range(self.n_comps):
            self.comps[pos] = eps * torch.randn(self.rank[pos], self.dims[pos], self.rank[pos + 1], device=self.device)


    def full(self):
        """
        Build the full tensor from a list of TT cores.

        comps: list of TT cores
            Each core has shape (r_{k-1}, n_k, r_k).

        Returns
        -------
        full : torch.Tensor
            The full tensor of shape (n1, n2, ..., nd).
        """
        d = len(self.comps)

        # start with the first core: (1, n1, r1) -> (n1, r1)
        res = self.comps[0][0, :, :]  # shape (n1, r1)
        for k in range(1, d):
            C = self.comps[k]  # shape (r_{k-1}, n_k, r_k)

            # contract res (..., r_{k-1}) with C (r_{k-1}, n_k, r_k)
            # result: (..., n_k, r_k)
            res = torch.einsum("...r,rns->...ns", res, C)
        # at the end, the last rank axis r_d = 1 must vanish
        res = res.squeeze(-1)  # shape (n1, n2, ..., nd)
        return res

    def __shift_to_right(self, pos, variant):
        c = self.comps[pos]
        s = c.shape
        c = left_unfolding(c)
        if variant == "qr":
            q, r = torch.linalg.qr(c)
            self.comps[pos] = q.reshape(s[0], s[1], q.shape[1])
            self.comps[pos + 1] = torch.einsum("ij, jkl->ikl ", r, self.comps[pos + 1])
        else:  # variant == "svd"
            u, S, vh = torch.linalg.svd(c, full_matrices=False)
            u, S, vh = u[:, :len(S)], S[:len(S)], vh[:len(S), :]

            # store orthonormal part at the current position
            self.comps[pos] = u.reshape(s[0], s[1], u.shape[1])
            self.comps[pos + 1] = torch.einsum("ij, jkl->ikl ", torch.diag(S) @ vh, self.comps[pos + 1])

    def __shift_to_left(self, pos, variant):
        c = self.comps[pos]

        s = c.shape
        c = right_unfolding(c)
        if variant == "qr":
            q, r = torch.linalg.qr(torch.transpose(c, 1, 0))
            qT = torch.transpose(q, 1, 0)
            self.comps[pos] = qT.reshape(qT.shape[0], s[1], s[2])  # refolding
            self.comps[pos - 1] = torch.einsum("ijk, kl->ijl ", self.comps[pos - 1], torch.transpose(r, 1, 0))

        else:  # perform svd
            u, S, vh = torch.linalg.svd(c, full_matrices=False)
            # store orthonormal part at the current position
            self.comps[pos] = vh.reshape(vh.shape[0], s[1], s[2])
            self.comps[pos - 1] = torch.einsum("ijk, kl->ijl ", self.comps[pos - 1], u @ torch.diag(S))

    def set_core(self, mu, variant="qr"):
        cc = []  # changed components

        if self.core_position is None:
            assert (variant in ["qr", "svd"])
            self.core_position = mu
            # left-to-right shift of the non-orthogonal component
            for pos in range(0, mu):
                self.__shift_to_right(pos, variant)
            # right-to-left shift of the non-orthogonal component
            for pos in range(self.n_comps - 1, mu, -1):
                self.__shift_to_left(pos, variant)

            cc = list(range(self.n_comps))

        else:
            while self.core_position > mu:
                cc.append(self.core_position)
                self.shift_core("left")
            while self.core_position < mu:
                cc.append(self.core_position)
                self.shift_core("right")

            cc.append(mu)

        assert (self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)

        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps)]
        return cc

    def shift_core(self, direction, variant="qr"):
        assert (direction in [-1, 1, "left", "right"])
        assert (self.core_position is not None)

        if direction == "left":
            shift = -1
        elif direction == "right":
            shift = 1
        else:
            shift = direction
        # current core position
        mu = self.core_position
        if shift == 1:
            self.__shift_to_right(mu, variant)
        else:
            self.__shift_to_left(mu, variant)

        self.core_position += shift

    def dot_rank_one(self, rank1obj):
        """
        Implements the multidimensional contraction of the underlying tensor train object
        with a rank-1 object being the product of vectors of sizes di.
        @param rank1obj: a list of vectors [vi, i = 0, ..., modes-1] with len(vi)=di
                          vi is of shape (b,di) with bi > 0
        """
        # the number of vectors must match the component number
        assert (len(rank1obj) == self.n_comps)
        for pos in range(0, self.n_comps):
            # inner dimension must match the respective vector size
            assert (self.comps[pos].shape[1] == rank1obj[pos].shape[1])
            # vectors must be 2d objects
            assert (len(rank1obj[pos].shape) == 2)

        G = [torch.einsum("ijk, bj->ibk", c, v) for c, v in zip(self.comps, rank1obj)]
        res = G[-1]
        # contract from right to left; TODO: here we assume row-wise memory allocation of matrices in G
        for pos in range(self.n_comps - 2, -1, -1):
            # contraction w.r.t. the 3d coordinate of G[pos]
            res = torch.einsum("ibj, jbk -> ibk", G[pos], res)  # k = 1 only
        # res is of shape b x 1
        return res.reshape(res.shape[1], res.shape[2])

    def dot_rank_one_new(self, rank1obj, use_einsum):
        """
        Implements the multidimensional contraction of the underlying tensor train object
        with a rank-1 object being the product of vectors of sizes di.
        @param rank1obj: a list of vectors [vi, i = 0, ..., modes-1] with len(vi)=di
                          vi is of shape (b,di) with bi > 0
        """
        b = rank1obj[0].shape[0]  # batch size should be equal for all rank-1 object components
        mu = self.core_position
        L = torch.ones(b, 1, device=self.device)  # left orthogonal stack
        R = torch.ones(b, 1, device=self.device)  # right orthogonal stack
        if use_einsum:
            for pos in range(mu):
                L = torch.einsum("bs, smR, bm -> bR", L, self.comps[pos], rank1obj[pos])
            for pos in range(self.n_comps - 1, mu, -1):
                R = torch.einsum("rms, bm, bs -> br", self.comps[pos], rank1obj[pos], R)
            # contract the non-orthogonal component last with L and R to avoid numerical instabilities
            return torch.einsum("br, rms, bm, bs -> b", L, self.comps[mu], rank1obj[mu], R).reshape(b, 1)
        else:
            for pos in range(mu):
                tmp = torch.tensordot(rank1obj[pos], self.comps[pos], dims=([1], [1]))  # (b,s,R)
                L = (tmp * L[:, :, None]).sum(dim=1)  # (b,R)

            for pos in range(self.n_comps - 1, mu, -1):
                tmp = torch.tensordot(rank1obj[pos], self.comps[pos], dims=([1], [1]))  # (b,r,s)
                R = (tmp * R[:, None, :]).sum(dim=-1)

            # step 1: contract over m
            tmp = torch.tensordot(rank1obj[mu], self.comps[mu], dims=([1], [1]))  # (b,r,s)

            # step 2: contract over s
            tmp2 = (tmp * R[:, None, :]).sum(dim=-1)  # (b,r)

            # step 3: contract over r
            out = (tmp2 * L).sum(dim=-1, keepdim=True)  # (b,1)
            return out


    def rank_truncation(self, max_ranks):
        if self.core_position != 0:
            self.set_core(0)

        for pos in range(self.n_comps - 1):
            c = self.comps[pos]
            s = c.shape

            c = c.reshape(s[0] * s[1], s[2])
            u, sigma, vt = torch.linalg.svd(c, full_matrices=False)
            new_rank = max_ranks[pos + 1]
            k = u.shape[1]

            # update information
            u, sigma, vt = u[:, :new_rank], sigma[:new_rank], vt[:new_rank, :]

            new_shape = (s[0], s[1], min(new_rank, k))

            self.comps[pos] = u.reshape(new_shape)

            self.comps[pos + 1] = torch.einsum("ir, rkl->ikl ", torch.matmul(torch.diag(sigma), vt), self.comps[pos + 1])

        self.core_position = self.n_comps - 1
        assert (self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)
        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps - 1)] + [1]

    def round(self, delta, verbose=False):
        rank_changed = False

        self.set_core(0)
        rule = Threshold(delta)
        for pos in range(self.n_comps - 1):
            c = self.comps[pos]
            s = c.shape
            c = c.reshape(s[0] * s[1], s[2])

            # QR decomposition
            q, r = torch.linalg.qr(c)
            # SVD on the small matrix
            u_r, s_r, v_r = torch.linalg.svd(r, full_matrices=False)
            u = q @ u_r
            sigma = s_r
            vt = v_r

            new_rank = rule(u, sigma, vt, pos)

            # update information
            u, sigma, vt = u[:, :new_rank], sigma[:new_rank], vt[:new_rank, :]
            new_shape = (s[0], s[1], min(new_rank, s[2]))
            self.comps[pos] = u.reshape(new_shape)

            ldtype = common_field_dtype([self.comps[pos + 1]], [sigma, vt])
            self.comps[pos + 1] = torch.einsum("ir, rkl->ikl ", torch.diag(sigma).type(ldtype) @ vt.type(ldtype), self.comps[pos + 1].type(ldtype))

        self.core_position = self.n_comps - 1
        assert (self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)

        if verbose and self.rank is not None:
            for mu, c in enumerate(self.comps[:-1]):
                if self.rank[mu + 1] > c.shape[2]:
                    print(f" {Fore.GREEN} A rank changed : {Style.RESET_ALL}  \
                                {Fore.BLUE} r_{mu} :  {self.rank[mu+1]} -> {c.shape[2]}{Style.RESET_ALL}")
                    rank_changed = True
                    time.sleep(1)

        # update the rank
        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps - 1)] + [1]
        if verbose:
            print("New rank is ", self.rank)

        return rank_changed

    def modify_ranks(self, rule, verbose=False):
        old_rank = self.rank
        new_rank = None

        # TODO: handle the case where the core is at the last position

        if self.core_position != 0:
            self.set_core(0)

        # possibly modify ranks r2, ..., rM-2
        for pos in range(self.n_comps - 1):
            c = self.comps[pos]
            s = c.shape
            c = c.reshape(s[0] * s[1], s[2])

            u, sigma, v = torch.linalg.svd(c, full_matrices=False)
            # obtain the possible new rank according to the truncation/retraction rule
            new_rank = rule(u, sigma, v, pos)
            if verbose:
                print("{c}Update{r} : rank r{p} = {c1}{rank}{r} -> {c}{rankn}{r}".format(p=pos + 1, rank=self.rank[pos + 1], rankn=new_rank, c1=Fore.RED, c=Fore.GREEN, r=Style.RESET_ALL))
                print("sing. vals for r{p} :\n".format(p=pos + 1), sigma)

            if new_rank > len(sigma):
                # u, sigma, v = svd(c) with c = self.comps[pos]
                k = new_rank - len(sigma)

                # 1. add k new columns to the left unfolding of u:
                #    - leftunfold(u) is an M x r matrix
                #    - add k orthogonal columns called u_k to u to obtain u_pk of shape M x (r + k)
                #    - undo the left unfolding w.r.t. M and store self.comps[pos] = u_pk
                # "add" random vectors from the kernel of u^T as an orthogonal projection of random vectors
                u_k = torch.rand(u.shape[0], k, device=u.device)
                u_k -= (u @ u.T) @ u_k

                # enlarged u plus k columns
                u_pk = torch.cat([u, u_k], dim=1)
                self.comps[pos] = u_pk.reshape(s[0], s[1], u_pk.shape[1])

                # 2. enlarge the singular values s with k new, very small entries
                s_pk = torch.cat([sigma, torch.tensor([1e-16] * k, device=u.device)])

                # 3. K = v * self.comps[pos+1] w.r.t. the 3rd and right unfolding
                #    yields an r x N orthogonal matrix. Add k orthogonal rows K_k
                #    to obtain an (r+k) x N orthogonal matrix K_pk
                K = torch.einsum("ir, rkl->ikl ", v, self.comps[pos + 1])
                s = K.shape
                K = K.reshape(s[0], s[1] * s[2])

                assert (abs(K.shape[0] - K.shape[1]) >= k)
                # get randomized orthogonal rows
                K_k = torch.rand(k, K.shape[1], device=K.device)
                K_k -= K_k @ (K.T @ K)
                K_pk = torch.cat([K, K_k])

                # 4. undo the unfolding of K_pk and scale it with the enlarged singular values
                #    to define the new right component self.comps[pos+1]
                K_pk = K_pk.reshape(K_pk.shape[0], s[1], s[2])
                self.comps[pos + 1] = torch.einsum("ij,jkl->ikl", torch.diag(s_pk), K_pk)

            else:
                # update information
                u, sigma, v = u[:, :new_rank], sigma[:new_rank], v[:new_rank, :]

                new_shape = (s[0], s[1], new_rank)
                self.comps[pos] = u.reshape(new_shape)

                self.comps[pos + 1] = torch.einsum("ir, rkl->ikl ", torch.diag(sigma) @ v, self.comps[pos + 1])

        self.core_position = self.n_comps - 1
        assert (self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)
        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps)]

        return old_rank, self.rank
