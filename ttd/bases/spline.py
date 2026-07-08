"""B-spline and cubic-spline bases: `HighDegreeCSpline` (single-dimension), `BSplines`,
and their tensor-product extensions `TensorSplineBasis` / `TensorSplineBasis_Equidistant`.
"""

import torch
from numpy.polynomial.legendre import leggauss

from ttd.utils.timing import TicToc


class HighDegreeCSpline(torch.nn.Module):
    def __init__(self, knots, p=5, s=2, device="cpu", dtype=torch.float64, orthonormalize="H2"):
        super().__init__()
        self.degree = p
        self.smoothness = s
        self.knot_mult = p - s
        self.initial_knots = knots
        self.device = device
        self.dtype = dtype

        # construct knot vector with repeated knots
        self.knots = self.create_knot_vector(knots)
        self.knots = torch.tensor(self.knots, device=device, dtype=dtype)

        # number of B-spline basis functions
        self.n_basis = len(self.knots) - self.degree - 1

        if orthonormalize is None:
            self.T = None

        else:
            orthonormalize_options = ["H2", "H2semi", "H1", "H1semi", "L2"]
            assert (orthonormalize in orthonormalize_options)

            if orthonormalize == "H2":
                w0, w1, w2 = 1., 1., 1.
            elif orthonormalize == "H2semi":
                w0, w1, w2 = 0., 0., 1.
            elif orthonormalize == "H1":
                w0, w1, w2 = 1., 1., 0.
            elif orthonormalize == "H1semi":
                w0, w1, w2 = 0., 1., 0.
            elif orthonormalize == "L2":
                w0, w1, w2 = 1., 0., 0.

            self.T = self.__assemble_orthonormal_transform(w0, w1, w2)

    def create_knot_vector(self, grid_points):
        p = self.degree
        s = self.smoothness
        mult = p - s

        knots = []
        knots += [grid_points[0]] * (p + 1)  # p + 1 repetitions at left

        for x in grid_points[1:-1]:
            knots += [x] * mult  # p - s repetitions inside

        knots += [grid_points[-1]] * (p + 1)  # p + 1 repetitions at right

        return knots

    def bspline_basis_matrix(self, x, knots, degree):
        device, dtype = x.device, self.dtype
        b = x.shape[0]
        n_basis = len(knots) - degree - 1
        x_col = x[:, None]  # [b,1]

        # basis k=0 (piecewise constant)
        left = knots[:n_basis][None, :]
        right = knots[1:n_basis + 1][None, :]
        B0 = ((x_col >= left) & (x_col < right)).to(dtype)
        # handle the right endpoint correctly
        B0[:, -1] += (x == knots[-1]).to(dtype)

        # --- precomputation for the recursion ---
        i = torch.arange(n_basis, device=device)
        j = torch.arange(1, degree + 1, device=device)

        # --- safe recursion coefficients (avoiding .clamp_min()) ---

        # denominators without clamping
        denom1 = knots[i[:, None] + j[None, :]] - knots[i, None]
        denom2 = knots[i[:, None] + j[None, :] + 1] - knots[i[:, None] + 1]

        # safe calculation of alpha:
        # use a placeholder for division where denom is 0, then zero out the result
        safe_denom1 = torch.where(denom1 == 0, torch.ones_like(denom1), denom1)
        alpha = (x_col[:, :, None] - knots[i][None, :, None]) / safe_denom1[None, :, :]
        alpha = torch.where(denom1[None, :, :] == 0, torch.zeros_like(alpha), alpha)

        # safe calculation of beta
        safe_denom2 = torch.where(denom2 == 0, torch.ones_like(denom2), denom2)
        beta = (knots[i[:, None] + j[None, :] + 1][None, :, :] - x_col[:, :, None]) / safe_denom2[None, :, :]
        beta = torch.where(denom2[None, :, :] == 0, torch.zeros_like(beta), beta)

        # --- recursion loop ---
        # tensor to store basis functions of all orders
        B = torch.zeros((b, n_basis, degree + 1), device=device, dtype=dtype)
        B[:, :, 0] = B0

        # loop over degree k
        for k in range(1, degree + 1):
            term1 = alpha[:, :, k - 1] * B[:, :, k - 1]
            term2 = beta[:, :-1, k - 1] * B[:, 1:, k - 1]

            B[:, :, k] = term1
            B[:, :-1, k] += term2

        return B[:, :, degree]

    def forward(self, x):
        x = x.view(-1)  # flatten to [N]

        B = self.bspline_basis_matrix(x, self.knots, self.degree)

        if self.T is not None:
            B = B @ self.T
        return B

    @torch.no_grad()
    def _find_span_uniform(self, x: torch.Tensor) -> torch.Tensor:
        """Determine the span index i for each u=x (vectorized).
        Expects x as a 1D tensor [N].
        """
        U = self.knots
        p = self.degree
        n = self.n_basis
        # special case: u == U[-1] -> span = n - 1
        is_right = (x == U[-1])
        i = torch.bucketize(x, U, right=True) - 1
        i = i.clamp(min=p, max=n - 1)
        if is_right.any():
            i[is_right] = n - 1
        return i  # [N], dtype long (bucketize returns long)

    @torch.no_grad()
    def _basis_nonzero_uniform(self, x: torch.Tensor):
        """
        Vectorized De Boor/Cox (algorithm A2.2) implementation, over batch.
        Input:
        x: [N]  (1D, flatten first)
        Returns:
        idx: [N, p+1] long  -> global column indices of the nonzero basis functions
        vals:[N, p+1] float -> corresponding values
        """
        device = x.device
        dtype = x.dtype
        U = self.knots
        p = self.degree
        N = x.shape[0]

        span = self._find_span_uniform(x)  # [N], long
        # Nlocal: [N, p+1]
        Nlocal = torch.zeros((N, p + 1), device=device, dtype=dtype)
        Nlocal[:, 0] = 1.0

        left = torch.empty((N, p + 1), device=device, dtype=dtype)
        right = torch.empty((N, p + 1), device=device, dtype=dtype)

        for j in range(1, p + 1):
            # U[span + 1 - j] and U[span + j] -> shape [N]
            uj_left = U[span + 1 - j]  # advanced indexing -> [N]
            uj_right = U[span + j]  # [N]
            left[:, j] = x - uj_left
            right[:, j] = uj_right - x

            saved = torch.zeros(N, device=device, dtype=dtype)
            for r in range(0, j):
                denom = (right[:, r + 1] + left[:, j - r]).clamp_min(1e-18)
                temp = Nlocal[:, r] / denom
                Nlocal[:, r] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp
            Nlocal[:, j] = saved

        # indices: span - p ... span (global basis indices)
        offsets = torch.arange(p, -1, -1, device=device, dtype=torch.long)  # [p+1]
        idx = (span[:, None] - offsets[None, :]).to(torch.long)  # [N, p+1]

        return idx, Nlocal

    def bspline_basis_deriv_matrix(self, x):
        p = self.degree
        knots = self.knots
        dtype = self.dtype  # use dtype from the class

        xx = x.clone()
        # numerical safeguard for the right endpoint
        xx[x == knots[-1]] -= 1e-6

        if p == 0:
            return torch.zeros(x.shape[0], self.n_basis, device=x.device, dtype=dtype)

        B_p_minus_1 = self.bspline_basis_matrix(xx, knots, degree=p - 1)
        n_basis = self.n_basis

        # --- corrected derivative calculation ---

        # denominators for the first term: (t_{i+p} - t_i)
        denom1 = knots[p:p + n_basis] - knots[:n_basis]
        # denominators for the second term: (t_{i+p+1} - t_{i+1})
        denom2 = knots[p + 1:p + 1 + n_basis] - knots[1:n_basis + 1]

        # create safe reciprocals (1/denom), where 1/0 -> 0
        recip_denom1 = torch.zeros_like(denom1)
        mask1 = torch.abs(denom1) > 1e-12
        recip_denom1[mask1] = 1.0 / denom1[mask1]

        recip_denom2 = torch.zeros_like(denom2)
        mask2 = torch.abs(denom2) > 1e-12
        recip_denom2[mask2] = 1.0 / denom2[mask2]

        # B_p_minus_1 has shape [b, n_basis+1]
        # first term uses B_{i, p-1} for i=0..n_basis-1
        term1 = p * B_p_minus_1[:, :n_basis] * recip_denom1

        # second term uses B_{i+1, p-1} for i=0..n_basis-1
        term2 = p * B_p_minus_1[:, 1:n_basis + 1] * recip_denom2

        D = term1 - term2
        return D

    def bspline_basis_second_deriv_matrix(self, x):
        p = self.degree
        knots = self.knots
        device, dtype = x.device, self.dtype

        xx = x.clone()
        xx[x == knots[-1]] -= 1e-6

        if p < 2:
            return torch.zeros(xx.shape[0], self.n_basis, device=device, dtype=dtype)

        # --- first, get the first derivative of splines of degree p-1 ---
        # this requires basis functions of degree p-2
        B_p_minus_2 = self.bspline_basis_matrix(xx, knots, degree=p - 2)
        n_basis_p_minus_1 = len(knots) - (p - 1) - 1

        # denominators for the p-1 spline derivative
        denom1_p1 = knots[p - 1:p - 1 + n_basis_p_minus_1] - knots[:n_basis_p_minus_1]
        denom2_p1 = knots[p:p + n_basis_p_minus_1] - knots[1:n_basis_p_minus_1 + 1]

        # safe reciprocals for the p-1 spline derivative
        recip_d1_p1 = torch.zeros_like(denom1_p1)
        mask1_p1 = torch.abs(denom1_p1) > 1e-12
        recip_d1_p1[mask1_p1] = 1.0 / denom1_p1[mask1_p1]

        recip_d2_p1 = torch.zeros_like(denom2_p1)
        mask2_p1 = torch.abs(denom2_p1) > 1e-12
        recip_d2_p1[mask2_p1] = 1.0 / denom2_p1[mask2_p1]

        # calculate D_{p-1}
        term1_p1 = (p - 1) * B_p_minus_2[:, :n_basis_p_minus_1] * recip_d1_p1
        term2_p1 = (p - 1) * B_p_minus_2[:, 1:n_basis_p_minus_1 + 1] * recip_d2_p1
        D_p_minus_1 = term1_p1 - term2_p1

        # --- now get the second derivative using D_{p-1} ---
        n_basis = self.n_basis
        denom1 = knots[p:p + n_basis] - knots[:n_basis]
        denom2 = knots[p + 1:p + 1 + n_basis] - knots[1:n_basis + 1]

        # safe reciprocals for the final step
        recip_d1 = torch.zeros_like(denom1)
        mask1 = torch.abs(denom1) > 1e-12
        recip_d1[mask1] = 1.0 / denom1[mask1]

        recip_d2 = torch.zeros_like(denom2)
        mask2 = torch.abs(denom2) > 1e-12
        recip_d2[mask2] = 1.0 / denom2[mask2]

        # calculate D2
        term1 = p * D_p_minus_1[:, :n_basis] * recip_d1
        term2 = p * D_p_minus_1[:, 1:n_basis + 1] * recip_d2
        D2 = term1 - term2

        return D2

    def grad(self, x):
        D = self.bspline_basis_deriv_matrix(x)  # [b, n_basis]
        if self.T is not None:
            D = D @ self.T
        return D

    def grad2(self, x):
        """
        Second derivative of the (optionally orthonormalized) basis functions at points x.
        Returns: Tensor [b, n_basis]
        """
        D2 = self.bspline_basis_second_deriv_matrix(x)  # [b, n_basis]
        if self.T is not None:
            D2 = D2 @ self.T

        return D2

    def __distinct_intervals(self):
        """Return the open intervals [a,b) (strict jumps in the knot vector)."""
        k = self.knots
        mask = (k[1:] > k[:-1])
        a = k[:-1][mask]
        b = k[1:][mask]
        return a, b

    def __assemble_orthonormal_transform(self, w0, w1, w2):
        """
        Fully vectorized assembly of all Gram matrices.
        Returns: M0, M1, M2, G, G^{-1/2}
        """
        assert w0 != 0 or w1 != 0 or w2 != 0
        device = self.knots.device
        dtype = self.knots.dtype
        n = self.n_basis
        p = self.degree

        nq = p + 1  # gaussian quadrature is exact for 2nq-1 degree, hence for grammian at post degree 2p appear

        # list of distinct intervals
        a_list, b_list = self.__distinct_intervals()
        m = a_list.shape[0]
        if m == 0:
            raise RuntimeError("No distinct intervals found.")

        # reference nodes/weights, xi in [-1,1], wi positive
        xi_ref, wi_ref = leggauss(nq)
        xi_ref, wi_ref = torch.tensor(xi_ref, device=device, dtype=dtype), torch.tensor(wi_ref, device=device, dtype=dtype)

        # affine map for each interval: mid, half
        mid = 0.5 * (a_list + b_list)      # [m]
        half = 0.5 * (b_list - a_list)     # [m]

        # quadrature points as 2D (m x nq), then flattened:
        # x_{r,i} = mid[r] + half[r] * xi_ref[i], w_{r,i} = half[r] * wi_ref[i]
        x_all = mid[:, None] + half[:, None] * xi_ref[None, :]  # [m, nq]
        w_all = half[:, None] * wi_ref[None, :]                 # [m, nq]

        x_all = x_all.reshape(-1)   # [m*nq]
        w_all = w_all.reshape(-1)   # [m*nq]

        # evaluate basis and derivatives on all points in one batch
        B_all = None
        D_all = None
        D2_all = None
        if w0 != 0.0:
            B_all = self.bspline_basis_matrix(x_all, self.knots, self.degree)  # [N, n]
        if w1 != 0.0:
            D_all = self.bspline_basis_deriv_matrix(x_all)  # [N, n]
        if w2 != 0.0:
            D2_all = self.bspline_basis_second_deriv_matrix(x_all)  # [N, n]

        # weight multiplication: (B^T W B) with W = diag(w_all)
        w_col = w_all.to(device=device, dtype=dtype).unsqueeze(1)  # [N,1]

        if w0 != 0.0:
            M0 = (B_all * w_col).T @ B_all  # [n,n]
        else:
            M0 = torch.zeros(n, n)

        if w1 != 0.0:
            M1 = (D_all * w_col).T @ D_all
        else:
            M1 = torch.zeros(n, n)

        if w2 != 0.0:
            M2 = (D2_all * w_col).T @ D2_all
        else:
            M2 = torch.zeros(n, n)

        # combined G
        G = w0 * M0 + w1 * M1 + w2 * M2
        G = 0.5 * (G + G.T)  # symmetrize

        # inverse square root via eigendecomposition
        eps = 0.  # could be raised (e.g. 1e-12) for numerical stabilization
        evals, evecs = torch.linalg.eigh(G)
        evals_clipped = torch.clamp(evals, min=eps)
        invsqrt = (evecs * (evals_clipped.rsqrt())) @ evecs.T

        return invsqrt


def compute_basis_transform_spline(
    basis_from: HighDegreeCSpline,
    basis_to: HighDegreeCSpline,
    inner_product: str = "L2",
    n_quadrature: int = 40,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Compute transformation matrix T such that beta = T @ alpha
    maps coefficients from basis_from to basis_to, restricted to the
    intersection domain of both bases.

    Args:
        basis_from: HighDegreeCSpline (source basis).
        basis_to:   HighDegreeCSpline (target basis).
        inner_product: "L2", "H1", or "H2".
        n_quadrature: overridden below (basis_from.degree + basis_to.degree).
        device, dtype: torch device/dtype.

    Returns:
        T: (d_to, d_from) torch.Tensor transformation matrix.
    """
    n_quadrature = basis_from.degree + basis_to.degree
    knots_from = basis_from.initial_knots
    knots_to = basis_to.initial_knots

    # intersection domain
    a = max(knots_from[0], knots_to[0])
    b = min(knots_from[-1], knots_to[-1])

    if b <= a:
        raise ValueError("Empty intersection domain between bases")

    # collect all knots within [a,b]
    all_knots = torch.cat([knots_from, knots_to])
    all_knots_in_domain = all_knots[(all_knots >= a) & (all_knots <= b)]

    # unique knots, sorted
    unique_knots = torch.unique(all_knots_in_domain)

    mask = (unique_knots[1:] > unique_knots[:-1])
    a_list = unique_knots[:-1][mask]
    b_list = unique_knots[1:][mask]

    # Gauss-Legendre quadrature on the reference interval
    xg_ref, wg_ref = leggauss(n_quadrature)
    xg_ref = torch.tensor(xg_ref, device=device)
    wg_ref = torch.tensor(wg_ref, device=device)

    # affine map for each interval: mid, half
    mid = 0.5 * (a_list + b_list)      # [m]
    half = 0.5 * (b_list - a_list)     # [m]

    # quadrature points as 2D (m x nq), then flattened:
    # x_{r,i} = mid[r] + half[r] * xi_ref[i], w_{r,i} = half[r] * wi_ref[i]
    xg_t = mid[:, None] + half[:, None] * xg_ref[None, :]  # [m, nq]
    wg_t = half[:, None] * wg_ref[None, :]                 # [m, nq]

    xg_t = xg_t.reshape(-1)   # [m*nq]
    wg_t = wg_t.reshape(-1)   # [m*nq]

    # evaluations at quadrature points
    Phi_from = basis_from.forward(xg_t)            # (N, d_from)
    Phi_to = basis_to.forward(xg_t)                # (N, d_to)
    d_from, d_to = Phi_from.shape[1], Phi_to.shape[1]

    terms_from = {"0": Phi_from}
    terms_to = {"0": Phi_to}

    if inner_product.upper() in ("H1", "H2"):
        Phi_from_p = basis_from.grad(xg_t)
        Phi_to_p   = basis_to.grad(xg_t)
        terms_from["1"] = Phi_from_p
        terms_to["1"]   = Phi_to_p

    if inner_product.upper() == "H2":
        Phi_from_pp = basis_from.grad2(xg_t)
        Phi_to_pp   = basis_to.grad2(xg_t)
        terms_from["2"] = Phi_from_pp
        terms_to["2"]   = Phi_to_pp

    # Gram matrix and cross matrix
    G2 = torch.zeros((d_to, d_to), dtype=dtype, device=device)
    M = torch.zeros((d_to, d_from), dtype=dtype, device=device)

    def add_term(Psi1, Psi2, out):
        # Psi: (N, d)
        out += Psi1.T @ (wg_t[:, None] * Psi2)

    if inner_product.upper() == "L2":
        add_term(Phi_to, Phi_to, G2)
        add_term(Phi_to, Phi_from, M)
    elif inner_product.upper() == "H1":
        add_term(Phi_to, Phi_to, G2)
        add_term(terms_to["1"], terms_to["1"], G2)
        add_term(Phi_to, Phi_from, M)
        add_term(terms_to["1"], terms_from["1"], M)
    elif inner_product.upper() == "H2":
        add_term(Phi_to, Phi_to, G2)
        add_term(terms_to["1"], terms_to["1"], G2)
        add_term(terms_to["2"], terms_to["2"], G2)
        add_term(Phi_to, Phi_from, M)
        add_term(terms_to["1"], terms_from["1"], M)
        add_term(terms_to["2"], terms_from["2"], M)
    else:
        raise ValueError(f"Unknown inner_product '{inner_product}'")

    # solve G2 beta = M alpha -> T = G2^{-1} M
    try:
        T = torch.linalg.solve(G2, M)  # preferred, if square/invertible
    except RuntimeError:
        T = torch.linalg.pinv(G2) @ M  # fallback: pseudoinverse

    return T


def test_transform():
    import matplotlib.pyplot as plt
    p_list = [9]

    nknots = 5
    knots_from = torch.linspace(-1.1, 2, nknots)
    knots_to = torch.linspace(-1.15, 1.9, nknots)

    s = 3

    plt.figure(figsize=(12, 4))
    for i, p_local in enumerate(p_list):
        # two bases with slightly shifted domains
        B1 = HighDegreeCSpline(knots_from, p=p_local, s=s, orthonormalize="H2")
        B2 = HighDegreeCSpline(knots_to, p=p_local, s=s, orthonormalize="H2")

        # transformation matrix from B1 -> B2 under the H2 inner product
        T = compute_basis_transform_spline(B1, B2, inner_product="H2", n_quadrature=200)

        # random coefficients in basis 1
        alpha = 10 * torch.rand(B1.n_basis, dtype=torch.float64)
        if i > 3:
            alpha[1] = 0.1
            alpha[2] = -0.01
        beta = T @ alpha

        # intersection range
        a_int = max(B1.initial_knots[0], B2.initial_knots[0])
        b_int = min(B1.initial_knots[-1], B2.initial_knots[-1])

        # plot points
        x1 = torch.linspace(B1.initial_knots[0], B1.initial_knots[-1], 500)
        x2 = torch.linspace(B2.initial_knots[0], B2.initial_knots[-1], 500)

        f1 = (B1.forward(x1) @ alpha).detach().numpy()
        f2 = (B2.forward(x2) @ beta).detach().numpy()
        f1_p = (B1.grad(x1) @ alpha).detach().numpy()
        f2_p = (B2.grad(x2) @ beta).detach().numpy()
        f1_pp = (B1.grad2(x1) @ alpha).detach().numpy()
        f2_pp = (B2.grad2(x2) @ beta).detach().numpy()

        x1_np, x2_np = x1.detach().numpy(), x2.detach().numpy()

        # function
        plt.subplot(len(p_list), 3, 3 * i + 1)
        plt.plot(x1_np, f1, "r-", label="B1*alpha on domain 1")
        plt.plot(x2_np, f2, "b-", label="B2*beta on domain 2")
        plt.axvline(a_int, color="k", linestyle=":", label="Intersection")
        plt.axvline(b_int, color="k", linestyle=":")
        plt.title("Function")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        # first derivative
        plt.subplot(len(p_list), 3, 3 * i + 2)
        plt.plot(x1_np, f1_p, "r-", label="B1*alpha on domain 1")
        plt.plot(x2_np, f2_p, "b-", label="B2*beta on domain 2")
        plt.axvline(a_int, color="k", linestyle=":", label="Intersection")
        plt.axvline(b_int, color="k", linestyle=":")
        plt.title("First derivative")
        plt.xlabel("x")
        plt.ylabel("f'(x)")

        # second derivative
        plt.subplot(len(p_list), 3, 3 * i + 3)
        plt.plot(x1_np, f1_pp, "r-", label="B1*alpha on domain 1")
        plt.plot(x2_np, f2_pp, "b-", label="B2*beta on domain 2")
        plt.axvline(a_int, color="k", linestyle=":", label="Intersection")
        plt.axvline(b_int, color="k", linestyle=":")
        plt.title("Second derivative")
        plt.xlabel("x")
        plt.ylabel("f''(x)")

    plt.show()


if __name__ == "__main__":
    test_transform()




# ---------------------------------------------------------------------------
# B-spline bases
# ---------------------------------------------------------------------------


class BSplines(object):
    def __init__(self, degrees, knots, allLevels=True):
        """
        degrees: list of B-spline degrees, one per dimension.
        knots: list of 1D knot tensors, one per dimension (sorted internally).
        allLevels: if True, embed_data concatenates basis functions of every degree
            0..degrees[k] per dimension; if False, only the top (full) degree.
        """
        self.d = len(degrees)
        self.degrees = degrees
        self.allLevels = allLevels
        self.knots = [torch.sort(loc_knots).values for loc_knots in knots]
        emb = self.embed_data(torch.ones(2, self.d))
        self.degs = [e.shape[1] - 1 for e in emb]

        self.domain = [(min(loc_knots), max(loc_knots)) for loc_knots in knots]

        self.coeffs = [torch.ones(self.degs[k] + 1) for k in range(self.d)]

        self.a = [self.domain[k][0] for k in range(self.d)]
        self.b = [self.domain[k][1] for k in range(self.d)]

    def buildBx(self, x):
        """computes the Bspline evaluations via the Cox de Boor formula.

        Args:
            x (torch.tensor): input of shape (b,d)

        Returns:
            Bx (list): list indexed by dimension such that Bx[k][l,i,j] is the Spline of interval i and degree j evaluated at
            the l-th sample's k-th component.
        """
        assert x.shape[1] == self.d

        knots = self.knots
        m = len(knots[0])

        Bx = [torch.zeros((x.shape[0], m - 1, self.degrees[k] + 1), dtype=x.dtype, device=x.device) for k in range(self.d)]
        for k in range(self.d):
            for i in range(m - 1):
                Bx[k][:, i, 0] = ((x[:, k] > knots[k][i]) & (x[:, k] <= knots[k][i + 1])).float()

        # use Cox-de Boor recursion formula
        for k in range(self.d):
            for j in range(1, self.degrees[k] + 1):
                for i in range(m - 1 - j):
                    denom1 = knots[k][i + j] - knots[k][i]
                    denom2 = knots[k][i + j + 1] - knots[k][i + 1]
                    coeff1 = torch.where(denom1 != 0, (x[:, k] - knots[k][i]) / denom1, torch.tensor(0.0))
                    coeff2 = torch.where(denom2 != 0, (knots[k][i + j + 1] - x[:, k]) / denom2, torch.tensor(0.0))
                    Bx[k][:, i, j] = coeff1 * Bx[k][:, i, j - 1] + coeff2 * Bx[k][:, i + 1, j - 1]
        return Bx

    def embed_data(self, x):
        """evaluate the B-Splines at the input data x

        Args:
            x (torch.tensor): shape (b,d)

        Returns:
            Bx (torch.tensor): list indexed by dimension, Bx[k][i,j] is the j-th basis
                function evaluated at the i-th sample's k-th component.
        """
        Bx = self.buildBx(x)
        m = len(self.knots[0])
        if self.allLevels:
            return [torch.cat([Bx[k][:, :m - 1 - j, j] for j in range(self.degrees[k] + 1)], dim=1) for k in range(len(Bx))]
        else:
            return [Bx[k][:, :m - 1 - self.degrees[k], self.degrees[k]] for k in range(len(Bx))]

    def __call__(self, x):
        with TicToc(key=" o Building Bx for call ", do_print=False, accumulate=True, sec_key="BSplines: "):
            Bx = self.embed_data(x)
        embedded_data = [torch.einsum("i, bi -> bi", self.coeffs[k], Bx[k]) for k in range(self.d)]
        # NOTE: returns Bx, not embedded_data -- self.coeffs (always 1s) is currently unused here
        return Bx

    def derivative_embed_data(self, x):
        assert x.shape[1] == self.d

        knots = self.knots
        m = len(knots[0])
        Bx = self.buildBx(x)

        Bx_d = [torch.zeros((x.shape[0], m - 1, self.degrees[k] + 1), dtype=x.dtype, device=x.device) for k in range(self.d)]

        # use Cox-de Boor recursion formula
        for k in range(self.d):
            for j in range(1, self.degrees[k] + 1):
                for i in range(m - 1 - j):
                    denom1 = knots[k][i + j] - knots[k][i]
                    denom2 = knots[k][i + j + 1] - knots[k][i + 1]
                    coeff1 = torch.where(denom1 != 0, 1 / denom1, torch.tensor(0.0))
                    coeff2 = torch.where(denom2 != 0, 1 / denom2, torch.tensor(0.0))
                    Bx_d[k][:, i, j] = j * (coeff1 * Bx[k][:, i, j - 1] - coeff2 * Bx[k][:, i + 1, j - 1])
        if self.allLevels:
            return [torch.cat([Bx_d[k][:, :m - 1 - j, j] for j in range(self.degrees[k] + 1)], dim=1) for k in range(len(Bx))]
        else:
            return [Bx_d[k][:, :m - 1 - self.degrees[k], self.degrees[k]] for k in range(len(Bx))]

    def grad(self, x):
        with TicToc(key=" o Building derivative ", do_print=False, accumulate=True, sec_key="BSplines: "):
            Bx = self.derivative_embed_data(x)
        return Bx


class TensorSplineBasis_Equidistant:
    def __init__(self, domain, nknots, p_list, s_list, device = "cpu", orthonormalize = "H2"):
        self.d = len(domain)
        self.device = device
        self.p_list = p_list
        self.s_list = s_list
        self.nknots = nknots

        self.unique_spline_params = {}
        idx_to_spline_type = {}
        for i in range(self.d):
            nk, p, s = nknots[i], p_list[i], s_list[i]
            if (nk,p,s) not in self.unique_spline_params:
                self.unique_spline_params[(nk,p,s)] = [i]
            else:
                self.unique_spline_params[(nk,p,s)].append(i)
            idx_to_spline_type[i] = (nk,p,s)
        for key in self.unique_spline_params:
            self.unique_spline_params[key] = torch.tensor(self.unique_spline_params[key], device=device, dtype=torch.long)
        
        self.reference_splines = {}
        for key in self.unique_spline_params.keys():
            nk,p,s = key
            self.reference_splines[key] = HighDegreeCSpline(knots=torch.linspace(-1, 1, nk, device=device), p=p, s=s, device=device, orthonormalize = None)


        self.domain_bounds = (torch.tensor([domain[i][0] for i in range(self.d)], device=device),
                              torch.tensor([domain[i][-1] for i in range(self.d)],device=device))
        # transformation of tensor domain to [-1,1]^d
        a = self.domain_bounds[0]
        b = self.domain_bounds[1]
        self.scale = 2.0 / (b - a)
        self.shift = -(b + a) / (b - a)

        self.ndofs =  [self.reference_splines[idx_to_spline_type[i]].n_basis for i in range(self.d)]

        self.orthonormalize = orthonormalize
        self.orthonormalize_transforms = None
        if self.orthonormalize is not None:
            self.orthonormalize_transforms = []
            for i in range(self.d):
                spline_i = HighDegreeCSpline(knots=torch.linspace(domain[i][0], domain[i][1], nknots[i], device=device), p=self.p_list[i], s=self.s_list[i], device=device, orthonormalize = orthonormalize)
                self.orthonormalize_transforms.append(spline_i.T)

        if device != "cpu":
            self.T_blocks = {}
            for key in self.unique_spline_params:
                indices = self.unique_spline_params[key]
                T_block = torch.block_diag(*[self.orthonormalize_transforms[i] for i in indices])
                self.T_blocks[key] = T_block

    def __call__(self, x):
        """
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j]
        """
        b, d = x.shape
        assert d == self.d

        # transformation of tensor domain to [-1,1]^d
        x_ref = x * self.scale + self.shift 
        evaluations = [None] * d
        for key, indices in self.unique_spline_params.items():
            x_ref_sub = x_ref[:, indices].contiguous() # b x d_key
            val = self.reference_splines[key](x_ref_sub.view(-1)) # [b*d_key, n_basis]

            if self.device != "cpu":
            
                if self.orthonormalize_transforms is not None:
                    T_block = self.T_blocks[key]
                    val = val.reshape(b, -1) @ T_block
                val = val.reshape(b, len(indices), -1)
                for loc_i, i  in enumerate(indices):
                    evaluations[i] = val[:, loc_i] # b x n_basis
            else:
                val = val.reshape(*x_ref_sub.shape, -1)
                if self.orthonormalize_transforms is not None:
                    for loc_i, i  in enumerate(indices):
                        evaluations[i] = val[:, loc_i] @ self.orthonormalize_transforms[i]
                else:
                    for loc_i, i  in enumerate(indices):
                        evaluations[i] = val[:, loc_i] # b x n_basis
                

        return evaluations
    
    def grad(self, x):
        """
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j] with derivatives
        """
        b, d = x.shape
        assert d == self.d

        # transformation of tensor domain to [-1,1]^d
        x_ref = x * self.scale + self.shift 
        gradients = [None] * d
        for key, indices in self.unique_spline_params.items():
            x_ref_sub = x_ref[:, indices].contiguous() # b x d_key
            grad_ref = self.reference_splines[key].grad(x_ref_sub.view(-1)) # [b*d_key, n_basis]
            grad_ref = grad_ref.reshape(*x_ref_sub.shape, -1)
            if self.orthonormalize_transforms is not None:
                for loc_i, i  in enumerate(indices):
                    gradients[i] = grad_ref[:, loc_i] @ self.orthonormalize_transforms[i] * self.scale[i]
            else:
                for loc_i, i  in enumerate(indices):
                    gradients[i] = self.scale[i]*grad_ref[:, loc_i] # b x n_basis
        return gradients
    
    
    def D2(self, x):
        """
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j] with second derivatives
        """
        b, d = x.shape
        assert d == self.d

        # transformation of tensor domain to [-1,1]^d
        x_ref = x * self.scale + self.shift 
        D2s = [None] * d
        for key, indices in self.unique_spline_params.items():
            x_ref_sub = x_ref[:, indices].contiguous() # b x d_key
            D2_ref = self.reference_splines[key].grad2(x_ref_sub.view(-1)) # [b*d_key, n_basis]
            D2_ref = D2_ref.reshape(*x_ref_sub.shape, -1) # b x d_key x n_basis
            if self.orthonormalize_transforms is not None:
                for loc_i, i  in enumerate(indices):
                    D2s[i] =  D2_ref[:, loc_i] @ self.orthonormalize_transforms[i] * (self.scale[i]**2)
            else:
                for loc_i, i  in enumerate(indices):
                    D2s[i] = (self.scale[i]**2) * D2_ref[:, loc_i] # b x n_basis
        return D2s


class TensorSplineBasis:
    def __init__(self, knots_list, p_list, s_list, device="cpu", orthonormalize = "H2"):
        self.d = len(knots_list)
        self.device = device
        self.splines = []
        self.knots_list = knots_list
        self.p_list = p_list 
        self.s_list = s_list
        self.orthonormalize = orthonormalize
        for j in range(self.d):
            spline = HighDegreeCSpline(
                knots=knots_list[j].clone().detach().to(device),
                p=p_list[j],
                s=s_list[j],
                device=device, 
                orthonormalize = self.orthonormalize
            )
            self.splines.append(spline)


        self.domain_bounds = (torch.tensor([knots_list[j][0].item() for j in range(self.d)]),
                               torch.tensor([knots_list[j][-1].item() for j in range(self.d)]))

        self.ndofs = [spline.n_basis for spline in self.splines]
    def __call__(self, x):
        """
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j]
        """
        b, d = x.shape
        assert d == self.d
        evaluations = []

        for j in range(d):
            xj = x[:, j]
            evals = self.splines[j](xj)
            evaluations.append(evals)

        return evaluations
    
    def new_call(self, x):
        _, d = x.shape
        assert d == self.d
        #futures = []

        return [self.splines[j](x[:,j]) for j in range(d)]
        
    
    def new_call_parallel(self, x):
        _, d = x.shape
        assert d == self.d
        futures = []
        for j in range(d):
            xj = x[:, j]
            futures.append(torch.jit.fork(self.splines[j], xj))
        # collect
        return [torch.jit.wait(f) for f in futures]

    
    def grad(self, x):
        """
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j] with derivatives
        """
        b, d = x.shape
        assert d == self.d
        gradients = []
        for j in range(d):
            xj = x[:, j]
            grads = self.splines[j].grad(xj)
            gradients.append(grads)
        return gradients
    
    def D2(self, x):
        """
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j] with second derivatives
        """
        b, d = x.shape
        assert d == self.d
        D2s = []
        for j in range(d):
            xj = x[:, j]
            D2_j = self.splines[j].grad2(xj)
            D2s.append(D2_j)
        return D2s
