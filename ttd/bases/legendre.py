"""Tensor-product Legendre-polynomial basis (`TensorLegendreBasis`) plus the
basis-transform machinery used to re-project an xTT's coefficients onto a Legendre
basis with different degree or domain.
"""

from copy import deepcopy

import torch
from numpy.polynomial.legendre import leggauss

from ttd.modelclass import Custom_Polynom_Arithmetic


class TensorLegendreBasis:
    """
    A tensor product basis using Legendre polynomials in each dimension.
    Implementation is very naive with overhead in the evaluation, in particular the
    underlying _legendre_and_grad_all is called in __call__ and grad, which is not
    efficient.
    """
    def __init__(self, domain_list, deg_list, orthonormalize="L2", device="cpu"):
        assert len(domain_list) == len(deg_list)
        self.d = len(domain_list)
        self.device = device
        self.domain_list = domain_list
        self.deg_list = deg_list
        self.ndofs = [deg + 1 for deg in deg_list]  # P0,...,P_deg

        self.domain_bounds = (torch.tensor([domain_list[j][0] for j in range(self.d)]),
                               torch.tensor([domain_list[j][1] for j in range(self.d)]))
        self.orthonormalize = orthonormalize
        if orthonormalize is None:
            self.orthomalizations = [torch.eye(ndof, device=self.device) for ndof in self.ndofs]
        else:
            assert self.orthonormalize in ["L2", "H1", "H2"]
            self.orthomalizations = self._assemble_orthonormalisation(self.orthonormalize, 50, self.device)

    def prepare_save_state(self, to_cpu=False):
        state = {
            "basis_type": "TensorLegendreBasis",
            "domain_list": self.domain_list,
            "deg_list": self.deg_list,
            "device": "cpu" if to_cpu else self.device,
            "orthonormalize": self.orthonormalize,
            "orthomalizations": [t.cpu() if to_cpu else t for t in self.orthomalizations],
        }
        return state

    @classmethod
    def load_from_state(cls, state):
        assert state["basis_type"] == "TensorLegendreBasis"
        # Create empty object without recomputation
        obj = cls.__new__(cls)
        obj.domain_list = state["domain_list"]
        obj.deg_list = state["deg_list"]
        obj.device = state["device"]
        obj.d = len(obj.domain_list)
        obj.ndofs = [deg + 1 for deg in obj.deg_list]
        obj.domain_bounds = (torch.tensor([obj.domain_list[j][0] for j in range(obj.d)]),
                              torch.tensor([obj.domain_list[j][1] for j in range(obj.d)]))
        obj.orthonormalize = state["orthonormalize"]
        obj.orthomalizations = state["orthomalizations"]
        return obj

    def _scale_to_legendre_domain(self, x, a, b):
        # map [a,b] -> [-1,1]
        return 2 * (x - a) / (b - a) - 1

    def _legendre(self, z, deg):
        """Legendre polynomials on the reference domain.

        z: [B] in [-1,1]
        Returns:
          P : [B, deg+1]
        """
        B = z.shape[0]
        P = torch.zeros(B, deg + 1, device=self.device, dtype=z.dtype)
        P[:, 0] = 1.0
        if deg >= 1:
            P[:, 1] = z
        for n in range(1, deg):
            P[:, n + 1] = ((2 * n + 1) * z * P[:, n] - n * P[:, n - 1]) / (n + 1)

        return P

    def _legendre_and_grad_all(self, z, deg):
        """Legendre polynomials and their first derivative on the reference domain.

        z: [B] in [-1,1]
        Returns:
          P : [B, deg+1]
          dP_dz : [B, deg+1]  (derivative wrt z)
        """
        B = z.shape[0]
        P = torch.zeros(B, deg + 1, device=self.device, dtype=z.dtype)
        dP = torch.zeros_like(P)

        P[:, 0] = 1.0
        dP[:, 0] = 0.0
        if deg >= 1:
            P[:, 1] = z
            dP[:, 1] = 1.0

        for n in range(1, deg):
            P[:, n + 1] = ((2 * n + 1) * z * P[:, n] - n * P[:, n - 1]) / (n + 1)
            dP[:, n + 1] = ((2 * n + 1) * (P[:, n] + z * dP[:, n]) - n * dP[:, n - 1]) / (n + 1)

        return P, dP

    def _legendre_and_two_grads_all(self, z, deg):
        """Legendre polynomials and their first and second derivative on the reference domain.

        z: [B] in [-1,1]
        Returns:
          P   : [B, deg+1]
          dP  : [B, deg+1]  (d/dz)
          d2P : [B, deg+1]  (d^2/dz^2)
        """
        B = z.shape[0]
        P = torch.zeros(B, deg + 1, device=self.device, dtype=z.dtype)
        dP = torch.zeros_like(P)
        d2P = torch.zeros_like(P)

        P[:, 0] = 1.0
        dP[:, 0] = 0.0
        d2P[:, 0] = 0.0
        if deg >= 1:
            P[:, 1] = z
            dP[:, 1] = 1.0
            d2P[:, 1] = 0.0

        for n in range(1, deg):
            P[:, n + 1] = ((2 * n + 1) * z * P[:, n] - n * P[:, n - 1]) / (n + 1)
            dP[:, n + 1] = ((2 * n + 1) * (P[:, n] + z * dP[:, n]) - n * dP[:, n - 1]) / (n + 1)
            d2P[:, n + 1] = ((2 * n + 1) * (2 * dP[:, n] + z * d2P[:, n]) - n * d2P[:, n - 1]) / (n + 1)

        return P, dP, d2P

    def __call__(self, x, raw=False):
        """Evaluate the basis on the physical domain.

        x: [B, d]
        raw: if False, apply the orthonormalization transform; if True, don't.
        Returns: list of length d, each [B, n_basis_j]
        """
        B, d = x.shape
        assert d == self.d
        evals_list = []
        for j in range(d):
            a, b = self.domain_list[j]
            deg = self.deg_list[j]
            z = self._scale_to_legendre_domain(x[:, j], a, b)
            P = self._legendre(z, deg)
            if raw:
                evals_list.append(P)
            else:
                trafo = self.orthomalizations[j]
                evals_list.append(torch.einsum("ij, bj -> bi", trafo, P))

        return evals_list

    def grad(self, x, raw=False):
        """Evaluate the basis gradient on the physical domain.

        x: [B, d]
        raw: if False, apply the orthonormalization transform; if True, don't.
        Returns: list of length d, each [B, n_basis_j] with derivatives wrt x_j
        """
        B, d = x.shape
        assert d == self.d
        grads_list = []
        for j in range(d):
            a, b = self.domain_list[j]
            deg = self.deg_list[j]
            z = self._scale_to_legendre_domain(x[:, j], a, b)
            _, dP_dz = self._legendre_and_grad_all(z, deg)
            if raw:
                # chain rule dz/dx = 2/(b-a)
                grads_list.append(dP_dz * (2.0 / (b - a)))
            else:
                trafo = self.orthomalizations[j]
                grads_list.append(torch.einsum("ij, bj -> bi", trafo, dP_dz * (2.0 / (b - a))))
        return grads_list

    def call_and_grad(self, x, raw=False):
        """
        x: [B, d]
        Returns: lists of length d, each [B, n_basis_j]
        """
        B, d = x.shape
        assert d == self.d
        evals_list = []
        grads_list = []
        for j in range(d):
            a, b = self.domain_list[j]
            deg = self.deg_list[j]
            z = self._scale_to_legendre_domain(x[:, j], a, b)
            P, dP_dz = self._legendre_and_grad_all(z, deg)

            if raw:
                evals_list.append(P)
                # chain rule dz/dx = 2/(b-a)
                grads_list.append(dP_dz * (2.0 / (b - a)))
            else:
                trafo = self.orthomalizations[j]
                evals_list.append(torch.einsum("ij, bj -> bi", trafo, P))
                grads_list.append(torch.einsum("ij, bj -> bi", trafo, dP_dz * (2.0 / (b - a))))

        return evals_list, grads_list

    def call_and_grad_and_D2(self, x, raw=False):
        """
        x: [B, d]
        Returns: lists of length d, each [B, n_basis_j]
        """
        B, d = x.shape
        assert d == self.d
        evals_list = []
        grads_list = []
        D2_list = []
        for j in range(d):
            a, b = self.domain_list[j]
            deg = self.deg_list[j]
            z = self._scale_to_legendre_domain(x[:, j], a, b)
            P, dP_dz, d2P_dz2 = self._legendre_and_two_grads_all(z, deg)

            if raw:
                evals_list.append(P)
                grads_list.append(dP_dz * (2.0 / (b - a)))
                d2factor = (2.0 / (b - a)) ** 2
                D2_list.append(d2P_dz2 * d2factor)
            else:
                trafo = self.orthomalizations[j]
                evals_list.append(torch.einsum("ij, bj -> bi", trafo, P))
                grads_list.append(torch.einsum("ij, bj -> bi", trafo, dP_dz * (2.0 / (b - a))))
                d2factor = (2.0 / (b - a)) ** 2
                D2_list.append(torch.einsum("ij, bj -> bi", trafo, d2P_dz2 * d2factor))

        return evals_list, grads_list, D2_list

    def D2(self, x, raw=False):
        """
        x: [B, d]
        Returns: list of length d, each [B, n_basis_j] with second derivatives wrt x_j
        """
        _, d = x.shape
        assert d == self.d
        D2_list = []
        for j in range(d):
            a, b = self.domain_list[j]
            deg = self.deg_list[j]
            z = self._scale_to_legendre_domain(x[:, j], a, b)
            _, _, d2P_dz2 = self._legendre_and_two_grads_all(z, deg)
            factor = (2.0 / (b - a)) ** 2
            if raw:
                # chain rule: d^2/dx^2 = d^2/dz^2 * (dz/dx)^2
                D2_list.append(d2P_dz2 * factor)
            else:
                trafo = self.orthomalizations[j]
                D2_list.append(torch.einsum("ij, bj -> bi", trafo, d2P_dz2 * factor))
        return D2_list

    def _assemble_orthonormalisation(self, inner_product: str = "H2", n_quadrature: int = 200, device: str = "cpu"):
        dtype: torch.dtype = torch.float64

        assert inner_product in ["L2", "H1", "H2"]

        d = self.d

        a, b = self.domain_bounds
        a = a.reshape(-1, 1).to(device)
        b = b.reshape(-1, 1).to(device)

        # Gauss-Legendre quadrature on [a,b]
        xg, wg = leggauss(n_quadrature)
        xg = 0.5 * (b - a) * torch.tensor(xg, device=device) + 0.5 * (a + b)
        wg = 0.5 * (b - a) * torch.tensor(wg, device=device)
        xg_t = xg.T
        wg_t = wg.T

        terms = {}

        if inner_product == "L2":
            val = self.__call__(xg_t, raw=True)
            terms["0"] = val
        elif inner_product == "H1":
            val, grad = self.call_and_grad(xg_t, raw=True)
            terms["0"] = val
            terms["1"] = grad
        elif inner_product == "H2":
            val, grad, D2 = self.call_and_grad_and_D2(xg_t, raw=True)
            terms["0"] = val
            terms["1"] = grad
            terms["2"] = D2
        else:
            raise ValueError(f"Only inner product L2, H1, H2 supported but got, {inner_product}.")
        T_list = []

        for j in range(d):
            n_j = self.ndofs[j]
            # Gram matrix
            G = torch.zeros((n_j, n_j), dtype=dtype, device=device)

            def add_term(Psi, out):
                out += Psi.T @ (Psi * wg_t[:, j][:, None])

            if inner_product.upper() == "L2":
                add_term(terms["0"][j], G)
            elif inner_product.upper() == "H1":
                add_term(terms["0"][j], G)
                add_term(terms["1"][j], G)
            elif inner_product.upper() == "H2":
                add_term(terms["0"][j], G)
                add_term(terms["1"][j], G)
                add_term(terms["2"][j], G)
            else:
                raise ValueError(f"Unknown inner_product '{inner_product}'")

            G = 0.5 * (G + G.T)  # symmetrize

            # inverse square root via eigendecomposition
            evals, evecs = torch.linalg.eigh(G)
            invsqrt = (evecs * (evals.rsqrt())) @ evecs.T

            T_list.append(invsqrt)

        return T_list


def compute_basis_transform_Legendre(
    basis_from: TensorLegendreBasis,
    basis_to: TensorLegendreBasis,
    inner_product: str = "L2",
    n_quadrature: int = 200,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """
    Compute transformation matrix T such that beta = T @ alpha
    maps coefficients from basis_from to basis_to, restricted to the
    intersection domain of both bases.

    Args:
        basis_from: TensorLegendreBasis (source basis).
        basis_to:   TensorLegendreBasis (target basis).
        inner_product: "L2", "H1", or "H2".
        n_quadrature: number of Gauss-Legendre quadrature points.
        device, dtype: torch device/dtype.

    Returns:
        T: (d_to, d_from) torch.Tensor transformation matrix.
    """
    assert basis_from.d == basis_to.d
    d = basis_from.d

    a = []
    b = []
    for j in range(d):
        a_j_from, b_j_from = basis_from.domain_list[j]
        a_j_to, b_j_to = basis_to.domain_list[j]
        # intersection domain
        a_j = max(a_j_from, a_j_to)
        b_j = min(b_j_from, b_j_to)
        if b_j <= a_j:
            raise ValueError("Empty intersection domain between bases")
        a.append(a_j)
        b.append(b_j)
    a = torch.tensor(a, device=device).reshape(-1, 1)
    b = torch.tensor(b, device=device).reshape(-1, 1)

    # Gauss-Legendre quadrature on [a,b]
    xg, wg = leggauss(n_quadrature)
    xg = 0.5 * (b - a) * torch.tensor(xg, device=device) + 0.5 * (a + b)
    wg = 0.5 * (b - a) * torch.tensor(wg, device=device)
    xg_t = xg.T
    wg_t = wg.T

    terms_from = {}
    terms_to = {}

    if inner_product == "L2":
        val_from = basis_from(xg_t)  # (N, d_from)
        val_to = basis_to(xg_t)      # (N, d_to)
        terms_from["0"] = val_from
        terms_to["0"] = val_to
    elif inner_product == "H1":
        val_from, grad_from = basis_from.call_and_grad(xg_t)
        val_to, grad_to = basis_to.call_and_grad(xg_t)
        terms_from["0"] = val_from
        terms_to["0"] = val_to
        terms_from["1"] = grad_from
        terms_to["1"] = grad_to
    elif inner_product == "H2":
        val_from, grad_from, D2_from = basis_from.call_and_grad_and_D2(xg_t)
        val_to, grad_to, D2_to = basis_to.call_and_grad_and_D2(xg_t)
        terms_from["0"] = val_from
        terms_to["0"] = val_to
        terms_from["1"] = grad_from
        terms_to["1"] = grad_to
        terms_from["2"] = D2_from
        terms_to["2"] = D2_to
    else:
        raise ValueError("Only inner product L2, H1, H2 supported.")

    T_list = []

    for j in range(d):
        d_from, d_to = basis_from.ndofs[j], basis_to.ndofs[j]

        # Gram matrix and cross matrix
        G2 = torch.zeros((d_to, d_to), dtype=dtype, device=device)
        M = torch.zeros((d_to, d_from), dtype=dtype, device=device)

        def add_term(Psi1, Psi2, out):
            # Psi: (N, d)
            out += Psi1.T @ (wg_t[:, j].reshape(-1, 1) * Psi2)

        if inner_product.upper() == "L2":
            add_term(terms_to["0"][j], terms_to["0"][j], G2)
            add_term(terms_to["0"][j], terms_from["0"][j], M)

        elif inner_product.upper() == "H1":
            add_term(terms_to["0"][j], terms_to["0"][j], G2)
            add_term(terms_to["0"][j], terms_from["0"][j], M)

            add_term(terms_to["1"][j], terms_to["1"][j], G2)
            add_term(terms_to["1"][j], terms_from["1"][j], M)

        elif inner_product.upper() == "H2":
            add_term(terms_to["0"][j], terms_to["0"][j], G2)
            add_term(terms_to["0"][j], terms_from["0"][j], M)

            add_term(terms_to["1"][j], terms_to["1"][j], G2)
            add_term(terms_to["1"][j], terms_from["1"][j], M)

            add_term(terms_to["2"][j], terms_to["2"][j], G2)
            add_term(terms_to["2"][j], terms_from["2"][j], M)
        else:
            raise ValueError(f"Unknown inner_product '{inner_product}'")

        # Solve G2 beta = M alpha -> T = G2^{-1} M
        try:
            T = torch.linalg.solve(G2, M)  # preferred, if square/invertible
        except RuntimeError:
            T = torch.linalg.pinv(G2) @ M  # fallback: pseudoinverse
        T_list.append(T)
    return T_list


class orthpoly(object):

    def __init__(self, degrees, domain):
        """
            Permits lists of domains for the different dimensions, as well as lists of
            regularization norms for the different dimensions. 'domain' should be either
            a tuple [float,float] or a list [[float,float],...,[float,float]] of tuples.
            'norm' should be either a string or a list of strings.
        """
        self.d = len(degrees)
        self.degs = degrees

        self.ndofs = [deg + 1 for deg in degrees]

        self.cpa = Custom_Polynom_Arithmetic(degrees, domain)

        # polynomial basis without terminal condition
        self.coeffs = [self.cpa.to_monomial_mats[mu].T for mu in range(self.d)]

        self.coeffs_grad = coeffs_grad(self.coeffs)
        self.coeffs_lap = coeffs_grad(self.coeffs_grad)

        self.gradTmatrix_list = self.coeffs_grad

        self.a = [domain[k][0] for k in range(self.d)]
        self.b = [domain[k][1] for k in range(self.d)]

        self.domain = domain

    def __call__(self, x):
        """lifts the inputs to feature space.

        Parameters
        ----------
        x : lb.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of lb.tensor
            inputs lifted to feature space defined by the feature and
            basis_coeffs attributes.
            Query [i][j,k] is the k-th basis function evaluated at the j-th sample's
            i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []

        for k in range(self.d):
            exponents = torch.arange(0, self.degs[k] + 1, 1, dtype=torch.float64)
            embedded_data.append(x[:, k, None] ** exponents)
            if self.coeffs is not None:
                embedded_data[k] = embedded_data[k] @ self.coeffs[k].T
        return embedded_data

    def grad(self, x):
        """lifts the inputs to feature-derivative space.

        Parameters
        ----------
        input_data : lb.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of lb.tensor
            inputs lifted to feature-derivative space defined by the feature and
            grad_coeffs attributes.
            Query Query [i][j,k] is the first derivative of the k-th basis function evaluated
            at the j-th sample's i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []

        for k in range(self.d):
            exponents = torch.arange(0, self.degs[k] + 1, 1, dtype=torch.float64)
            embedded_data.append(x[:, k, None] ** exponents)
            if self.coeffs is not None:
                embedded_data[k] = torch.einsum("oi, bi -> bo", self.coeffs_grad[k], embedded_data[k])
        return embedded_data

    def laplace(self, x):
        """lifts the inputs to feature-second-derivative space.

        Parameters
        ----------
        input_data : lb.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of lb.tensor
            inputs lifted to feature-derivative-derivative space defined by the feature and
            grad_coeffs attributes.
            Query Query [i][jk] is the second derivative of the k-th basis function evaluated
            at the j-th sample's i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []

        for k in range(self.d):
            exponents = torch.arange(0, self.degs[k] + 1, 1, dtype=torch.float64)
            embedded_data.append(x[:, k, None] ** exponents)
            if self.coeffs is not None:
                embedded_data[k] = torch.einsum("oi, bi -> bo", self.coeffs_lap[k], embedded_data[k])
        return embedded_data


def coeffs_grad(coeffs):
    grad_coeffs = []
    for coeff in coeffs:
        grad_coeff = deepcopy(coeff)
        for j in range(coeff.shape[1] - 1):
            grad_coeff[:, j] = (j + 1) * grad_coeff[:, j + 1]
        grad_coeff[:, -1] = 0 * grad_coeff[:, -1]
        grad_coeffs.append(grad_coeff)
    return grad_coeffs
