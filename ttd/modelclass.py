"""`Custom_Polynom_Arithmetic`: orthonormalized-polynomial coefficient arithmetic (used
by `ttd.bases.legendre`), and `Potential`/`Custom_Polynomial`, model wrappers for a
polynomial potential and its derivatives.
"""

import torch

torch.set_default_dtype(torch.float64)
import copy
from itertools import product
from math import sqrt

from numpy.polynomial.chebyshev import Chebyshev, chebder
from numpy.polynomial.hermite_e import HermiteE
from numpy.polynomial.legendre import Legendre, legder
from numpy.polynomial.polynomial import Polynomial, polyder
from scipy.optimize import minimize

PolynomClass = Legendre
PolyDeriviate = legder

# PolynomClass = Polynomial
# PolyDeriviate = polyder

if PolynomClass is Chebyshev:
    assert PolyDeriviate is chebder
elif PolynomClass is Legendre:
    assert PolyDeriviate is legder
elif PolynomClass is Polynomial:
    assert PolyDeriviate is polyder


def prod(list_of_arrays):
    res = list_of_arrays[0]
    for i in range(1, len(list_of_arrays)):
        res = res * list_of_arrays[i]
    return res


class Custom_Polynom_Arithmetic(object):

    def __init__(self, maximal_degrees, domain):

        self.maximal_degrees = maximal_degrees
        self.domain = domain

        self.dim = len(maximal_degrees)

        # changing from polyclass to monomial basis and back
        self.to_monomial_mats = []
        self.to_polyclass_mats = []

        # changing the domain
        self.to_oneone_mats = []
        self.to_domain_mats = []

        # changing from polyclass to hermite and back
        self.to_hermite_mats = []
        self.from_hermite_mats = []

        # changing from polyclass to H2-orthogonal and back
        self.to_h2_mats = []
        self.from_h2_mats = []

        # changing from orthogonal legendre to orthonormal legendre
        self.to_onb_mats = []
        self.from_onb_mats = []
        for d in range(self.dim):
            # current dofs
            n = maximal_degrees[d] + 1
            self.to_monomial_mats.append(torch.zeros(n, n))
            self.to_polyclass_mats.append(torch.zeros(n, n))

            self.to_domain_mats.append(torch.zeros(n, n))
            self.to_oneone_mats.append(torch.zeros(n, n))

            self.to_hermite_mats.append(torch.zeros(n, n))
            self.from_hermite_mats.append(torch.zeros(n, n))

            self.from_onb_mats.append(sqrt(domain[d][1] - domain[d][0]) * torch.diag(torch.tensor([sqrt(1 / (1 + 2 * i)) for i in range(n)])))
            self.to_onb_mats.append(1 / sqrt(domain[d][1] - domain[d][0]) * torch.diag(torch.tensor([sqrt((1 + 2 * i)) for i in range(n)])))

            # orth_coeff = torch.tensor(orth_coeffs(n, a=self.domain[d][0], b=self.domain[d][1], norm="H2")).T
            # self.to_h2_mats.append(orth_coeff)
            # self.from_h2_mats.append(torch.linalg.inv(self.to_h2_mats[-1]))

            for _ind in range(n):
                # get one of the Legendre polynomials
                poly = PolynomClass([0. for i in range(_ind)] + [1.], domain=self.domain[d])

                # convert to monomial coefficients
                coeffs = torch.zeros((n,))
                coeffs[:_ind + 1] = torch.from_numpy(poly.convert(domain=[-1, 1], kind=Polynomial).coef)
                self.to_monomial_mats[-1][:, _ind] = coeffs

                # convert to [-1, 1] domain
                coeffs = torch.zeros((n,))
                coeffs[:_ind + 1] = torch.from_numpy(poly.convert(domain=[-1, 1], kind=PolynomClass).coef)
                self.to_oneone_mats[-1][:, _ind] = coeffs

                # convert to hermite polynomials
                coeffs = torch.zeros((n,))
                coeffs[:_ind + 1] = torch.from_numpy(poly.convert(domain=[-1, 1], kind=HermiteE).coef)
                self.to_hermite_mats[-1][:, _ind] = coeffs

                # Do all of the above backwards

                # convert monomial to polyclass coefficients
                poly = Polynomial([0. for i in range(_ind)] + [1.], domain=[-1, 1])
                coeffs = torch.zeros((n,))
                coeffs[:_ind + 1] = torch.from_numpy(poly.convert(domain=self.domain[d], kind=PolynomClass).coef)
                self.to_polyclass_mats[-1][:, _ind] = coeffs

                # convert [-1, 1] domain to self.domain
                poly = PolynomClass([0. for i in range(_ind)] + [1.], domain=[-1, 1])
                coeffs = torch.zeros((n,))
                coeffs[:_ind + 1] = torch.from_numpy(poly.convert(domain=self.domain[d], kind=PolynomClass).coef)
                self.to_domain_mats[-1][:, _ind] = coeffs

                # convert hermite to polyclass
                poly = HermiteE([0. for i in range(_ind)] + [1.], domain=[-1, 1])
                coeffs = torch.zeros((n,))
                coeffs[:_ind + 1] = torch.from_numpy(poly.convert(domain=self.domain[d], kind=PolynomClass).coef)
                self.from_hermite_mats[-1][:, _ind] = coeffs

    def get_polyclass_coefficients(self, coeffs):
        """Builds a coefficient tensor in the chosen polynomial class from a monomial coefficient tensor.

        Returns:
            coeff (torch.tensor): tensor of same shape as self.coefficients which is built from the argument tensor.
        """

        idx = list(slice(None, s, None) for s in coeffs.shape)

        # coeff = torch.tensor(coeffs.clone().detach().requires_grad_(True))

        coeff = torch.zeros_like(coeffs)
        # coeff[:,:] = coeffs
        coeff[:] = coeffs

        for i in range(self.dim):
            L_i = self.to_polyclass_mats[i][:coeffs.shape[i], :coeffs.shape[i]]

            idx = list(slice(None, s, None) for s in coeffs.shape)
            idx2 = list(slice(None, s, None) for s in coeffs.shape)

            for l in range(coeffs.shape[i]):
                idx2[i] = l
                _add = 0
                for k in range(coeffs.shape[i]):
                    idx[i] = k
                    _add += L_i[l, k] * coeff[idx]  # order d-1
                coeff[idx2] = _add  # add order d-1 to the coefficients

        return coeff

    def get_onb_coefficients(self, coeffs):
        """Builds a coefficient tensor in the chosen polynomial class from a monomial coefficient tensor.

        Returns:
            coeff (torch.tensor): tensor of same shape as self.coefficients which is built from the argument tensor.
        """

        idx = list(slice(None, s, None) for s in coeffs.shape)

        # coeff = torch.tensor(coeffs.clone().detach().requires_grad_(True))

        coeff = torch.zeros_like(coeffs)
        # coeff[:,:] = coeffs
        coeff[:] = coeffs

        for i in range(self.dim):
            L_i = self.to_onb_mats[i][:coeffs.shape[i], :coeffs.shape[i]]

            idx = list(slice(None, s, None) for s in coeffs.shape)
            idx2 = list(slice(None, s, None) for s in coeffs.shape)

            for l in range(coeffs.shape[i]):
                idx2[i] = l
                _add = 0
                for k in range(coeffs.shape[i]):
                    idx[i] = k
                    _add += L_i[l, k] * coeff[idx]  # order d-1
                coeff[idx2] = _add  # add order d-1 to the coefficients

        return coeff

    def from_onb_coefficients(self, coeffs):
        """Builds a coefficient tensor in the chosen polynomial class from a monomial coefficient tensor.

        Returns:
            coeff (torch.tensor): tensor of same shape as self.coefficients which is built from the argument tensor.
        """

        idx = list(slice(None, s, None) for s in coeffs.shape)

        # coeff = torch.tensor(coeffs.clone().detach().requires_grad_(True))

        coeff = torch.zeros_like(coeffs)
        # coeff[:,:] = coeffs
        coeff[:] = coeffs

        for i in range(self.dim):
            L_i = self.from_onb_mats[i][:coeffs.shape[i], :coeffs.shape[i]]

            idx = list(slice(None, s, None) for s in coeffs.shape)
            idx2 = list(slice(None, s, None) for s in coeffs.shape)

            for l in range(coeffs.shape[i]):
                idx2[i] = l
                _add = 0
                for k in range(coeffs.shape[i]):
                    idx[i] = k
                    _add += L_i[l, k] * coeff[idx]  # order d-1
                coeff[idx2] = _add  # add order d-1 to the coefficients

        return coeff

    def get_hermite_coefficients(self, coeffs):
        """Builds a coefficient tensor in the chosen polynomial class from a monomial coefficient tensor.

        Returns:
            coeff (torch.tensor): tensor of same shape as self.coefficients which is built from the argument tensor.
        """

        idx = list(slice(None, s, None) for s in coeffs.shape)

        # coeff = torch.tensor(coeffs.clone().detach().requires_grad_(True))

        coeff = torch.zeros_like(coeffs)
        # coeff[:,:] = coeffs
        coeff[:] = coeffs

        for i in range(self.dim):
            L_i = self.to_hermite_mats[i][:coeffs.shape[i], :coeffs.shape[i]]

            idx = list(slice(None, s, None) for s in coeffs.shape)
            idx2 = list(slice(None, s, None) for s in coeffs.shape)

            for l in range(coeffs.shape[i]):
                idx2[i] = l
                _add = 0
                for k in range(coeffs.shape[i]):
                    idx[i] = k
                    _add += L_i[l, k] * coeff[idx]  # order d-1
                coeff[idx2] = _add  # add order d-1 to the coefficients

        return coeff

    def get_h2_coefficients(self, coeffs):
        """Builds a coefficient tensor in the chosen polynomial class from a monomial coefficient tensor.

        Returns:
            coeff (torch.tensor): tensor of same shape as self.coefficients which is built from the argument tensor.
        """

        idx = list(slice(None, s, None) for s in coeffs.shape)

        # coeff = torch.tensor(coeffs.clone().detach().requires_grad_(True))

        coeff = torch.zeros_like(coeffs)
        # coeff[:,:] = coeffs
        coeff[:] = coeffs

        for i in range(self.dim):
            L_i = self.to_h2_mats[i][:coeffs.shape[i], :coeffs.shape[i]]

            idx = list(slice(None, s, None) for s in coeffs.shape)
            idx2 = list(slice(None, s, None) for s in coeffs.shape)

            for l in range(coeffs.shape[i]):
                idx2[i] = l
                _add = 0
                for k in range(coeffs.shape[i]):
                    idx[i] = k
                    _add += L_i[l, k] * coeff[idx]  # order d-1
                coeff[idx2] = _add  # add order d-1 to the coefficients

        return coeff

    def get_monomial_coefficients(self, coeffs):
        """Returns the coefficient tensor in monomial basis

        Returns:
            coeff (torch.tensor): tensor of same shape as self.coefficients
        """
        idx = list(slice(None, s, None) for s in coeffs.shape)

        # coeff = torch.tensor(coeffs.clone().detach().requires_grad_(True))  # copy.copy(coeffs)
        coeff = torch.zeros_like(coeffs)
        # coeff[:,:] = coeffs
        coeff[:] = coeffs
        for i in range(self.dim):
            L_i = self.to_monomial_mats[i][:coeffs.shape[i], :coeffs.shape[i]]

            idx = list(slice(None, s, None) for s in coeffs.shape)
            idx2 = list(slice(None, s, None) for s in coeffs.shape)

            for l in range(coeffs.shape[i]):
                idx2[i] = l
                _add = 0
                for k in range(coeffs.shape[i]):
                    idx[i] = k
                    _add += L_i[l, k] * coeff[idx]  # order d-1
                coeff[idx2] = _add  # add order d-1 to the coefficients

        return coeff

    def add(self, P1, P2, a=1., b=1.):

        if P2 is not None:

            assert len(P1.domain) == len(P2.domain)
            for dom1, dom2 in zip(P1.domain, P2.domain):
                for i in [0, 1]:
                    assert dom1[i] == dom2[i]

            # Convert P1 and P2 to Monomials
            C1 = self.get_monomial_coefficients(P1.coefficients)
            C2 = self.get_monomial_coefficients(P2.coefficients)

            shapes = [max(s1, s2) for s1, s2 in zip(C1.shape, C2.shape)]

            # TODO: make this sparse later

            C = torch.zeros(shapes)
            C[tuple(slice(None, s, None) for s in C1.shape)] += a * C1
            C[tuple(slice(None, s, None) for s in C2.shape)] += b * C2

            # convert class to poln
            C = self.get_polyclass_coefficients(C)

            degrees = [s - 1 for s in C.shape]

            return Custom_Polynomial(degrees, P1.domain, C)

        else:
            assert a == 1
            return P1

    def square_poly(self, P):
        """
        return a Custom_Polynomial representing the squared polynomial P^2
        """
        # convert P to monomial
        C = self.get_monomial_coefficients(P.coefficients)

        # construct P^2 via monomial representation
        C_squared = torch.zeros([2 * (s - 1) + 1 for s in C.shape])

        # TODO: sparsity
        for alpha in product(*[range(s) for s in C.shape]):
            for beta in product(*[range(s) for s in C.shape]):
                idx = tuple(a + b for a, b in zip(alpha, beta))  # = alpha + beta
                C_squared[idx] += C[alpha] * C[beta]

        # convert P^2 to Custom_polynomial class
        C = self.get_polyclass_coefficients(C_squared)
        degrees = [s - 1 for s in C_squared.shape]
        return Custom_Polynomial(degrees, P.domain, C)

    def symbolic_laplace(self, P):
        """Returns a polynomial of the same class as self, defined by the Laplace of self.

        Returns:
            Polynomial(self.degrees, self.domain, Laplace_coeffs)
        """
        # coeff = self.coefficients
        coeff = torch.zeros(P.coefficients.shape)

        # First: transform coefficients into monomial coefficients
        transformed_coeffs = self.get_monomial_coefficients(P.coefficients)

        # Apply Laplace to transformed coeffs
        for i in range(self.dim):

            degree = P.degrees[i]
            L_i = torch.diag(torch.tensor([(k + 2.0) * (k + 1.0) for k in range(degree - 1)]), +2)  # .T

            idx = list(slice(None, d + 1, None) for d in P.degrees)
            idx2 = list(slice(None, d + 1, None) for d in P.degrees)

            sparse_indices = []
            for _idx_s in product(*[range(d + 1) for d in [degree, degree]]):
                if L_i[_idx_s] != 0:
                    sparse_indices.append(_idx_s)

            for _idx_s in sparse_indices:
                l, k = _idx_s
                idx[i] = k
                idx2[i] = l
                coeff[idx2] += L_i[_idx_s] * transformed_coeffs[idx]

        # Transform back to polyclass basis
        coeff = self.get_polyclass_coefficients(coeff)

        return Custom_Polynomial(P.degrees, self.domain, coeff)

    def symbolic_DV_x(self, P):
        """Returns a polynomial of the same class as self, defined by x*grad_self.

        Returns:
            Polynomial(self.degrees, self.domain, xDV_coeffs)
        """
        # coeff = self.coefficients
        coeff = torch.zeros(P.coefficients.shape)  # torch.zeros(self.coefficients.shape)

        # First: transform coefficients into monomial coefficients
        transformed_coeffs = self.get_monomial_coefficients(P.coefficients)

        for i in range(self.dim):
            degree = P.degrees[i]
            L_i = torch.diag(torch.tensor([k for k in range(degree + 1)]))

            idx = list(slice(None, d + 1, None) for d in P.degrees)
            idx2 = list(slice(None, d + 1, None) for d in P.degrees)

            sparse_indices = []
            for _idx_s in product(*[range(d + 1) for d in [degree, degree]]):
                if L_i[_idx_s] != 0:
                    sparse_indices.append(_idx_s)

            for _idx_s in sparse_indices:
                l, k = _idx_s
                idx[i] = k
                idx2[i] = l
                coeff[idx2] += L_i[_idx_s] * transformed_coeffs[idx]

        # Transform back to polyclass basis
        coeff = self.get_polyclass_coefficients(coeff)

        return Custom_Polynomial(P.degrees, self.domain, coeff)

    def symbolic_full(self, P):
        """Returns a polynomial of the same class as self, defined by
            Laplace self + x*grad_self.

        Returns:
            Polynomial(self.degrees, self.domain, coeffs)
        """
        # coeff = self.coefficients
        coeff = torch.zeros(P.coefficients.shape)  # torch.zeros(self.coefficients.shape)

        # First: transform coefficients into monomial coefficients
        transformed_coeffs = self.get_monomial_coefficients(P.coefficients)

        for i in range(self.dim):
            degree = P.degrees[i]
            L_i = torch.diag(torch.tensor([k for k in range(degree + 1)], dtype=torch.float64))
            L_i += torch.diag(torch.tensor([(k + 2.0) * (k + 1.0) for k in range(degree - 1)]), +2)

            idx = list(slice(None, d + 1, None) for d in P.degrees)
            idx2 = list(slice(None, d + 1, None) for d in P.degrees)

            sparse_indices = []
            for _idx_s in product(*[range(d + 1) for d in [degree, degree]]):
                if L_i[_idx_s] != 0:
                    sparse_indices.append(_idx_s)

            for _idx_s in sparse_indices:
                l, k = _idx_s
                idx[i] = k
                idx2[i] = l
                coeff[idx2] += L_i[_idx_s] * transformed_coeffs[idx]

        # Transform back to polyclass basis
        coeff = self.get_polyclass_coefficients(coeff)

        return Custom_Polynomial(P.degrees, self.domain, coeff, cache=(P.last_x, P.cache, P.cache_dx, P.cache_dxx))

    def symbolic_full_mat(self, degrees):

        vector_shape = (degrees[0] + 1) * (degrees[1] + 1)
        L = torch.zeros((vector_shape, vector_shape))
        for i in range(0, self.dim):
            degree = degrees[i]
            # x*nabla V part
            L_i = torch.diag(torch.tensor([k for k in range(degree + 1)], dtype=torch.float64))
            # Laplace part
            L_i += torch.diag(torch.tensor([(k + 2.0) * (k + 1.0) for k in range(degree - 1)]), +2)
            L_i_trafo = self.to_polyclass_mats[i][:degree + 1, :degree + 1] @ L_i @ self.to_monomial_mats[i][:degree + 1, :degree + 1]
            # TODO: Assemble tensor instead of Kronecker product
            if i == 0:
                L += torch.kron(L_i_trafo, torch.eye(degrees[1] + 1))
            elif i == 1:
                L += torch.kron(torch.eye(degrees[0] + 1), L_i_trafo)
        return L

    def symbolic_DV_2_mat(self, W):

        # THIS ONLY WORKS IN 2D
        # Also TODO: truncate polynomial degree properly to deg-1 when using gradient to save DoFs
        assert self.dim == 2

        # shape of vectorized coefficients
        vector_dim = (W.degrees[0] + 1) * (W.degrees[1] + 1)
        # new_vector_dim = (2*(W.degrees[0]-1)+1)*(2*(W.degrees[1]-1)+1)

        # Initialize linear mapping onto higher space
        N_W = torch.zeros((4 * (W.degrees[0] + 1) * (W.degrees[1] + 1), vector_dim))
        # N_W = torch.zeros(( (2*vector_dim-1) , vector_dim ))
        # N_W = torch.zeros(( (2*(W.degrees[0]-1)+1)*(2*(W.degrees[1]-1)+1) , vector_dim ))

        # get coefficients of the initial condition
        W_coeffs = copy.deepcopy(W.coefficients)

        # we will have to calculate and add different derivatives, so we store them in a list
        Dalpha_list = []
        Dalpha_W_list = []

        for alpha in range(self.dim):

            # get degrees of current solution
            degree = W.degrees[alpha]

            # get derivative matrix in alpha-th dimension (this corresponds to what was previously called L_i)
            D_alpha = torch.diag(torch.tensor([(k + 1.) for k in range(degree)]), +1)

            # Transform to operate on polyclass coefficients
            D_alpha = self.to_polyclass_mats[alpha][:degree + 1, :degree + 1] @ D_alpha @ self.to_monomial_mats[alpha][:degree + 1, :degree + 1]

            # initialize the derivative in dimension i performed on W
            Dalpha_W = torch.zeros(W_coeffs.shape)

            idx = list(slice(None, d + 1, None) for d in W.degrees)
            idx2 = list(slice(None, d + 1, None) for d in W.degrees)

            for l in range(W.degrees[alpha] + 1):
                idx2[alpha] = l
                _add = 0
                for k in range(W.degrees[alpha] + 1):
                    idx[alpha] = k
                    _add += D_alpha[l, k] * W_coeffs[idx]  # order d-1
                Dalpha_W[idx2] += _add  # add order d-1 to the coefficients

            # transform back to polyclass (Legendre) coefficients
            Dalpha_W = self.get_polyclass_coefficients(Dalpha_W)

            # ------- this part works only in 2D ----------------
            # store in vectorized form
            Dalpha_W_list.append(torch.reshape(Dalpha_W, (vector_dim,)))

            # Now store the D_alpha as applied to the whole coefficients vector
            if alpha == 0:
                Dalpha_list.append(torch.kron(D_alpha, torch.eye(W.degrees[1] + 1)))
            elif alpha == 1:
                Dalpha_list.append(torch.kron(torch.eye(W.degrees[0] + 1), D_alpha))
            else:
                print("index should be 1 or 0 in 2D!")
                exit()

            for mu in range(vector_dim):
                for beta in range(vector_dim):
                    N_W[mu + beta, :] += Dalpha_W_list[alpha][mu] * Dalpha_list[alpha][beta, :]
            # ----------------------------------------------------

        # Finally, get the projected Operator by truncating all higher order terms (works only for Legendre polynomials)
        N_w_proj = N_W[:vector_dim, :]

        return 2 * N_w_proj, N_W

    def symbolic_DV_2_mat_new(self, W):

        # THIS ONLY WORKS IN 2D
        # Also TODO: truncate polynomial degree properly to deg-1 when using gradient to save DoFs
        assert self.dim == 2

        # get coefficients of the initial condition
        A_tilde = copy.deepcopy(W.coefficients)

        # shape of vectorized coefficients
        vector_dim = A_tilde.shape[0] * A_tilde.shape[1]

        # Initialize linear mapping onto higher space
        N_W = torch.zeros((4 * A_tilde.shape[0] * A_tilde.shape[1], vector_dim))

        # we will have to calculate and add different derivatives, so we store them in a list
        M_mu_B_list = []
        M_mu_B_A_tilde_list = []

        for mu in range(self.dim):
            # get degrees of current solution
            degree = W.degrees[mu]

            # get monomial-derivative matrix in mu-th dimension (this corresponds to what was previously called L_i)
            M_mu = torch.diag(torch.tensor([(k + 1.) for k in range(degree)]), +1)

            # Transform to operate on monomial coefficients
            M_mu_B = M_mu @ self.to_monomial_mats[mu][:degree + 1, :degree + 1]

            # initialize the derivative in dimension i performed on W
            M_mu_B_A_tilde = torch.zeros(A_tilde.shape)

            idx = list(slice(None, d + 1, None) for d in W.degrees)
            idx2 = list(slice(None, d + 1, None) for d in W.degrees)

            for l in range(A_tilde.shape[mu]):
                idx2[mu] = l
                _add = 0
                for k in range(A_tilde.shape[mu]):
                    idx[mu] = k
                    _add += M_mu_B[l, k] * A_tilde[idx]  # order d-1
                M_mu_B_A_tilde[idx2] += _add  # add order d-1 to the coefficients

            # ------- this part works only in 2D ----------------
            # store in vectorized form
            M_mu_B_A_tilde_list.append(torch.reshape(M_mu_B_A_tilde, (vector_dim,)))

            # Now store the D_alpha as applied to the whole coefficients vector
            if mu == 0:
                M_mu_B_list.append(torch.kron(M_mu_B, self.to_monomial_mats[1][:W.degrees[1] + 1, :W.degrees[1] + 1]))
            elif mu == 1:
                M_mu_B_list.append(torch.kron(self.to_monomial_mats[0][:W.degrees[0] + 1, :W.degrees[0] + 1], M_mu_B))
            else:
                print("index should be 1 or 0 in 2D!")
                exit()

            for i in range(vector_dim):
                for j in range(vector_dim):
                    N_W[i + j, :] += M_mu_B_A_tilde_list[mu][i] * M_mu_B_list[mu][j, :]

            # Transform back to polynomial basis
            # B = torch.kron(self.to_polyclass_mats[0][:W.degrees[0]+1,:W.degrees[0]+1], self.to_polyclass_mats[1][:W.degrees[1]+1,:W.degrees[1]+1])
            # BN_W = B @ N_W
            print(N_W)
            BN_W = self.get_polyclass_coefficients(N_W)

            # Finally, get the projected Operator by truncating all higher order terms (works only for Legendre polynomials)
            PBN_W = BN_W[:vector_dim, :]

            return PBN_W, BN_W

    def symbolic_DV_2_mat_2D(self, W):

        # THIS ONLY WORKS IN 2D
        # Also TODO: truncate polynomial degree properly to deg-1 when using gradient to save DoFs
        assert self.dim == 2

        # get coefficients of the initial condition
        A_tilde = copy.deepcopy(W.coefficients)

        # shape of vectorized coefficients
        vector_dim = A_tilde.shape[0] * A_tilde.shape[1]

        # Initialize linear mapping onto higher space
        N_W = torch.zeros(((2 * A_tilde.shape[0] - 1) * (2 * A_tilde.shape[1] - 1), vector_dim))

        M_1 = torch.kron(torch.diag(torch.tensor([(k + 1.) for k in range(W.degrees[0])]), +1), torch.eye(W.degrees[1] + 1))
        M_2 = torch.kron(torch.eye(W.degrees[0] + 1), torch.diag(torch.tensor([(k + 1.) for k in range(W.degrees[1])]), +1))

        B = torch.kron(self.to_monomial_mats[0][:W.degrees[0] + 1, :W.degrees[0] + 1], self.to_monomial_mats[1][:W.degrees[1] + 1, :W.degrees[1] + 1])

        M_1_B = M_1 @ B
        M_2_B = M_2 @ B

        # vectorize A_tilde and apply the transformation
        A_til_vec = torch.reshape(A_tilde, (vector_dim,))
        M_1_B_A_til_vec = M_1_B @ A_til_vec
        M_2_B_A_til_vec = M_2_B @ A_til_vec

        for i in range(vector_dim):
            for j in range(vector_dim):
                N_W[i + j, :] += M_1_B_A_til_vec[i] * M_1_B[j, :]
                N_W[i + j, :] += M_2_B_A_til_vec[i] * M_2_B[j, :]

        B_inv = torch.kron(self.to_monomial_mats[0][:2 * W.degrees[0] + 1, :2 * W.degrees[0] + 1], self.to_monomial_mats[1][:2 * W.degrees[1] + 1, :2 * W.degrees[1] + 1])

        BN_W = B_inv @ N_W

        # Finally, get the projected Operator by truncating all higher order terms (works only for Legendre polynomials)
        PBN_W = BN_W[:vector_dim, :]

        return PBN_W, BN_W

    def symbolic_DV_2(self, P):

        assert self.dim == 2

        # coeff = self.coefficients
        coeff = torch.zeros(P.coefficients.shape)

        # First: transform coefficients into monomial coefficients
        transformed_coeffs = self.get_monomial_coefficients(P.coefficients)

        contributions = []

        for i in range(P.dim):
            degree = P.degrees[i]

            coeff = torch.zeros(transformed_coeffs.shape)

            L_i = torch.diag(torch.tensor([(k + 1) for k in range(degree)]), +1)

            idx = list(slice(None, d + 1, None) for d in P.degrees)
            idx2 = list(slice(None, d + 1, None) for d in P.degrees)

            for l in range(P.degrees[i] + 1):
                idx2[i] = l
                _add = 0
                for k in range(P.degrees[i] + 1):
                    idx[i] = k
                    _add += L_i[l, k] * transformed_coeffs[idx]  # order d-1
                coeff[idx2] += _add  # add order d-1 to the coefficients

            D_xi_P = Custom_Polynomial(P.degrees, P.domain, coeff)  # , cache=(P.last_x, P.cache, P.cache_dx, P.cache_dxx))

            idx_sparse = []
            for idx in product(*[range(d + 1) for d in D_xi_P.degrees]):
                if D_xi_P.coefficients[idx] != 0:
                    idx_sparse.append(idx)

            coeff_i = torch.zeros([2 * d + 1 for d in P.degrees])
            for idx_1 in idx_sparse:
                i1, j1 = idx_1
                for idx_2 in idx_sparse:
                    i2, j2 = idx_2
                    coeff_i[i1 + i2, j1 + j2] += D_xi_P.coefficients[i1, j1] * D_xi_P.coefficients[i2, j2]

            # D_xi_P_2 = Polynomial([2 * d for d in P.degrees], coeff_i)  # this is D_i

            contributions.append(coeff_i)

        # sum_i ( D_xi P ) ^2 coefficients
        res_monomial_coeff = sum(contributions)

        poly_coeffs = self.get_polyclass_coefficients(res_monomial_coeff)

        return Custom_Polynomial([2 * d for d in P.degrees], P.domain, poly_coeffs)  # Polynomial([2 * d for d in P.degrees], poly_coeffs)

    def linearized_rhs(self, W):
        """returns the linearized HJB rhs as a matrix. linearization is performed in W.

        Args:
            W (PolyClass): Polynom at which the linearization is to be performed

        Returns:
            h (torch.tensor): linear operator in matrix form
        """
        L = self.symbolic_full_mat(W.degrees)
        N_d, _ = self.symbolic_DV_2_mat_2D(W)
        H = L - N_d
        return H


class Potential(object):

    def __init__(self, domain, degrees):
        self.domain = domain
        self.degrees = degrees

        self.local_minima = None
        self.local_hessians = None

    def get_domain(self):
        return self.domain

    def get_degrees(self):
        return self.degrees

    def __call__(self, x):
        pass

    def grad(self, x):
        pass

    def get_local_minima(self, init_guesses):
        """
        updates self.local_mininma
                self.local_hessians
        """
        pass


class Custom_Polynomial(Potential):

    def __init__(self, degrees, domain, coefficients=None, cache=None):
        super().__init__(domain, degrees)
        """Fully tensorized polynomial class.

        Args:
            degrees (list): list of degrees in every dimension. Length of the list determines dimension of underlying state space.
            coefficients (torch.tensor, optional): Coefficient tensor of the polynomial of dimension (degrees[0] + 1, ..., degrees[d] + 1) where d = len(degrees)
                                                    For example, if d=3, then the ijk-th entry is equal to the coefficient of the basis function b_i(x_1)*b_j(x_2)*b_k(x_3)
                                                    where {b_i}_i is the chosen 1D polynomial basis (Legendre, Chebychev, ...)
                                                    If the basis is monomial, for instance, entry ijk is the coefficient of x^i*y^j*z^k
            domain (list, optional): intervals on which the chosen 1D bases are orthonormal (in case of i.e. Legendre). Defaults to [-5, 5].
                                     Example: [[-1, 1], [-2, 2], [4, 8]] for a 3D problem.
        """

        self.dim = len(degrees)
        self.e = {}
        for i in range(max(degrees) + 1):
            self.e[i] = torch.zeros((torch.tensor(max(degrees) + 1, dtype=int)).tolist())
            self.e[i][i] = 1

        # self.degrees = degrees
        self.coefficients = coefficients

        self.last_x = None
        # self.domain = domain
        self.cache, self.cache_dx, self.cache_dxx = None, None, None

        if cache is not None:
            assert len(cache) == 4
            self.last_x, self.cache, self.cache_dx, self.cache_dxx = cache

        self.to_oneone_mats = []
        self.to_domain_mats = []
        for d in range(self.dim):
            # current dofs
            n = degrees[d] + 1
            self.to_domain_mats.append(torch.zeros(n, n))
            self.to_oneone_mats.append(torch.zeros(n, n))

            for _ind in range(n):
                # get one of the Legendre polynomials
                poly = PolynomClass([0. for i in range(_ind)] + [1.], domain=self.domain[d])

                coeffs = torch.zeros((n,))
                coeffs[:_ind + 1] = torch.from_numpy(poly.convert(domain=[-1, 1], kind=PolynomClass).coef)

                self.to_oneone_mats[-1][:, _ind] = coeffs

            self.to_domain_mats[-1] = torch.linalg.inv(self.to_oneone_mats[-1])

    def __call__(self, x):
        """A call of the polynomial for batched input x.

        Args:
            x (torch.tensor): batched input of shape (dim, batch_size).

        Returns:
            res (torch.tensor): result of the polynomial evaluation of shape (batch_size,)
        """
        cache, _, _ = self.get_cache(x)
        # return sum( self.coefficients[idx] * prod( [Legendre(self.e[idx[i]])(x[i,:]) for i in range(self.dim)]) for idx in product(*[range(d+1) for d in self.degrees]))
        res = sum(self.coefficients[idx] * prod([cache[i][idx[i]] for i in range(self.dim)]) for idx in product(*[range(d + 1) for d in self.degrees]))
        return res

    def get_local_minima(self, init_guess):
        """returns list of local minima of the polynomial

        Args:
            init_guess (list): initial guesses for local minima (minima of previous potential)

        Returns:
            new_mins (list): local minima
            new_H_invs (list): list of Hessians in local minima
        """
        new_mins = []
        new_H_invs = []

        for guess in init_guess:
            res = minimize(self.__call__, guess, method="cg", options={"gtol": 1e-8, "disp": True}).result
            new_min = res.x
            if sum([torch.linalg.norm(new_min - loc_new_min) < 1e-6 for loc_new_min in new_mins]) == 0:
                new_mins.append(new_min)
                new_H_invs.append(res.hess_inv)
        return new_mins, new_H_invs

    def set_coefficients(self, coefficients):
        """Set the coefficient tensor of the polynomial

        Args:
            coefficients (torch.tensor): must have shape (self.degree[0] + 1, ..., self.degree[dim] + 1)
        """
        for i in range(self.dim):
            assert coefficients.shape[i] == self.degrees[i] + 1

        self.coefficients = coefficients

    def set_cache(self, last_x, cache, cache_dx, cache_dxx):
        self.last_x = last_x
        self.cache, self.cache_dx, self.cache_dxx = cache, cache_dx, cache_dxx

    def fit_model(self, callable_fun):
        """fit the function callable_fun

        Args:
            callable_fun (function): function mapping from torch.tensor (the coefficient tensor of the polynomial) to
        """

        pass

    def func(self, V):
        pass

    def __compute_cache(self, x, der_x=False, der_xx=False):
        """Sets the cache of 1D polynomial evaluations to a specific batch of inputs x.
        The cache is a list of length self.dim
            [cache_1 , ... , cache_dim]
        where cache_i is again a list of all basis evaluations of the corresponding dimension.
        In total
            self.cache[i][j][k]
        now is the
            evaluation in the i-th dimension of the j-th basis function to the k-th input (to the k-th input's i-th component).

        Additionaly, the caches for first and second derivatives are set, such that
            self.cache_dx[i][j][k]
            self.cache_dxx[i][j][k]
        are the
            evaluation in the i-th dimension of the j-th basis function's first/second derivative to the k-th input (to the k-th input's i-th component).

        Args:
            x (torch.tensor): Tensor of batched inputs, shape (dim, batch_size)
            der_x / der_xx (bool): If True, caches for first and second derivatives are computed too.
        """
        # self.cache = [ [Chebyshev(self.e[k])(x[i,:]) for k in range(self.degrees[i]+1) ] for i in range(self.dim) ]
        # self.cache = [ [Legendre(self.e[k])(x[i,:]) for k in range(self.degrees[i]+1) ] for i in range(self.dim) ]

        self.cache = [[PolynomClass(self.e[k], domain=self.domain[i])(x[i, :]) for k in range(self.degrees[i] + 1)] for i in range(self.dim)]
        if der_x == True:
            self.cache_dx = [[PolynomClass(self.to_domain_mats[i] @ torch.cat((torch.tensor(PolyDeriviate(self.to_oneone_mats[i] @ self.e[k][:self.degrees[i] + 1], m=1)), torch.tensor([0.])), 0), domain=self.domain[i])(x[i, :]) for k in range(self.degrees[i] + 1)] for i in range(self.dim)]
        if der_xx == True:
            self.cache_dxx = [[PolynomClass(self.to_domain_mats[i] @ torch.cat((torch.tensor(PolyDeriviate(self.to_oneone_mats[i] @ self.e[k][:self.degrees[i] + 1], m=2)), torch.tensor([0., 0.])), 0), domain=self.domain[i])(x[i, :]) for k in range(self.degrees[i] + 1)] for i in range(self.dim)]
        self.last_x = x

    def get_cache(self, x):
        """ Returns the cache that is computed via the __compute_cash method.
        """
        if self.last_x is not None and x.shape == self.last_x.shape and torch.allclose(x, self.last_x):
            return self.cache, self.cache_dx, self.cache_dxx
        else:
            self.__compute_cache(x, der_x=True, der_xx=True)
            return self.cache, self.cache_dx, self.cache_dxx

    def grad(self, x):
        """evaluate the gradient of the polynomial in x.

        Args:
            x (torch.tensor): batched inputs of shape (dim, batch_size)

        Returns:
            res (torch.tensor): gradient output of shape (dim,batch_size)
        """
        res = torch.zeros(x.shape)
        cache, cache_dx, _ = self.get_cache(x)  # [ [Legendre(self.e[k])(x[i,:]) for k in range(self.degrees[i]+1) ] for i in range(self.dim) ]

        for k in range(self.dim):
            res[k, :] = sum(self.coefficients[idx] * prod([cache[i][idx[i]] for i in range(k)] +
                                                            [cache_dx[k][idx[k]]] +
                                                            [cache[i][idx[i]] for i in range(k + 1, self.dim)]
                                                        ) for idx in product(*[range(d + 1) for d in self.degrees]))
        return res

    def laplace(self, x):
        """evaluate the Laplace of the polynomial in x.

        Args:
            x (torch.tensor): batched inputs of shape (dim, batch_size)

        Returns:
            res (torch.tensor): laplace output of shape (batch_size,)
        """
        cache, _, cache_dxx = self.get_cache(x)
        res = sum(sum(self.coefficients[idx] * prod([cache[i][idx[i]] for i in range(k)] +
                                                        [cache_dxx[k][idx[k]]] +
                                                        [cache[i][idx[i]] for i in range(k + 1, self.dim)]
                                                    )
                    for idx in product(*[range(d + 1) for d in self.degrees]))
                for k in range(self.dim)
        )

        return res

    def symbolic_DV_2(self):

        assert self.dim == 2

        # 0 1         p_0
        # 0 2       p_1
        #   0 3
        #      k     p_k

        contributions = []

        for i in range(self.dim):
            degree = self.degrees[i]

            coeff = torch.zeros(self.coefficients.shape)

            L_i = torch.diag(torch.tensor([(k + 1) for k in range(degree)]), +1)

            idx = list(slice(None, d + 1, None) for d in self.degrees)
            idx2 = list(slice(None, d + 1, None) for d in self.degrees)

            for l in range(self.degrees[i] + 1):
                idx2[i] = l
                _add = 0
                for k in range(self.degrees[i] + 1):
                    idx[i] = k
                    _add += L_i[l, k] * self.coefficients[idx]  # order d-1
                coeff[idx2] += _add  # add order d-1 to the coefficients

            D_xi_P = Polynomial(self.degrees, coeff)  # this is D_i(self)

            idx_sparse = []
            for idx in product(*[range(d + 1) for d in D_xi_P.degrees]):
                if D_xi_P.coefficients[idx] != 0:
                    idx_sparse.append(idx)

            coeff_i = torch.zeros([2 * d + 1 for d in self.degrees])
            for idx_1 in idx_sparse:
                i1, j1 = idx_1
                for idx_2 in idx_sparse:
                    i2, j2 = idx_2
                    coeff_i[i1 + i2, j1 + j2] += D_xi_P.coefficients[i1, j1] * D_xi_P.coefficients[i2, j2]

            D_xi_P_2 = Polynomial([2 * d for d in self.degrees], coeff_i)  # this is D_i

            contributions.append(D_xi_P_2)

        return add(contributions[0], contributions[1])

    def inner_grad_grad_log_target(self, other, x):
        """
        The function other must provide .grad function

        returns   self.grad(x).T  other.grad(x)
        """
        return torch.einsum("ib, ib->b", self.grad(x), other.grad(x))

    def euclidean_norm_2_grad(self, x):
        """
        Returns   self.grad(x).T * self.grad(x)
        """
        # TODO: only compute self.grad(x) once via caching, together with the inner grad_grad_log_target term
        g = self.grad(x)
        return torch.einsum("ib, ib->b", g, g)


if __name__ == "__main__":

    degrees = [2, 2]
    domain = [[-1, 1], [-1, 1]]
    cpa = Custom_Polynom_Arithmetic([50, 50], domain)

    C0 = torch.tensor([
        [1., 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])
    print("before Laplace (mon):\n ", C0)
    C0 = cpa.get_polyclass_coefficients(C0)

    p = Custom_Polynomial(degrees, domain=domain, coefficients=C0)

    # check linear part
    L = cpa.symbolic_full_mat(degrees)
    print("L_mat = \n", L)
    flattenedCoeffs = torch.reshape(p.coefficients, ((degrees[0] + 1) * (degrees[1] + 1),))

    Laplace_coeffs = torch.reshape((L @ flattenedCoeffs), (degrees[0] + 1, degrees[1] + 1))
    Laplace_coeffs_mon = cpa.get_monomial_coefficients(Laplace_coeffs)

    print("After linear operator (mon):\n ", Laplace_coeffs_mon)

    sym_coeff = cpa.symbolic_full(p)
    print("After linear operator check (mon):\n ", Laplace_coeffs_mon)

    # check nonlinear part
    N_proj, N = cpa.symbolic_DV_2_mat_2D(p)
    print(N_proj.shape)
    print(N.shape)

    print(N_proj)
    print(N[:30, :])

    Nonlinear_coeffs = torch.reshape((N @ flattenedCoeffs), (degrees[0] + 1, degrees[1] + 1))
    Nonlinear_coeffs_mon = cpa.get_monomial_coefficients(Nonlinear_coeffs)

    print("After nonlinear operator (mon):\n ", Nonlinear_coeffs_mon)

