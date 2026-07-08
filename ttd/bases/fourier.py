"""Extended-Fourier basis: an orthonormal trigonometric basis on an interval, augmented
with optional linear/quadratic terms, plus its tensor-product (multi-dimensional)
extension `TensorExtendedFourierBasis`.
"""

import math
from typing import Optional, Tuple

import torch


class OrthonormalExtendedFourierBasis(torch.nn.Module):
    def __init__(
        self,
        domain: Tuple[float, float] = (0.0, 1.0),
        n_basis: Optional[int] = None,
        n_harmonics: Optional[int] = None,
        include_linear: bool = True,
        include_quadratic: bool = True,
        device: str = "cpu",
        dtype=torch.float64,
        ordering: str = "const_lin_quad_cos_sin",
        orthonormalize: str = "L2",
    ):
        super().__init__()
        a, b = float(domain[0]), float(domain[1])
        if b <= a:
            raise ValueError("domain must satisfy b > a")
        self.a = a
        self.b = b
        self.L = float(b - a)
        self.device = torch.device(device)
        self.dtype = dtype
        self.ordering = ordering
        self.orthonormalize = orthonormalize.upper()
        self.include_linear = bool(include_linear)
        self.include_quadratic = bool(include_quadratic)

        if n_basis is None and n_harmonics is None:
            raise ValueError("Provide either n_basis or n_harmonics")

        offset = 1 + int(self.include_linear) + int(self.include_quadratic)
        if n_basis is not None:
            m = math.ceil(max(0, (n_basis - offset)) / 2)
            produced = offset + 2 * m
            self.requested_n_basis = n_basis
        else:
            m = int(n_harmonics)
            produced = offset + 2 * m
            self.requested_n_basis = produced

        self.n_harmonics = m
        self.n_raw = min(produced, self.requested_n_basis)

        # k indices and angular frequencies
        if m > 0:
            k = torch.arange(1, m + 1, device=self.device, dtype=self.dtype)
            omega_x = 2.0 * math.pi * k / float(self.L)
            self.register_buffer("_k_tensor", k)
            self.register_buffer("_omega_x", omega_x)
        else:
            self.register_buffer("_k_tensor", torch.empty((0,), device=self.device, dtype=self.dtype))
            self.register_buffer("_omega_x", torch.empty((0,), device=self.device, dtype=self.dtype))

        # build Gram matrix for the first n_raw functions (vectorized)
        G = self._build_gram_matrix(self.n_raw)

        vals, vecs = torch.linalg.eigh(G)
        if torch.any(vals <= 0):
            raise RuntimeError("Gram matrix not positive definite (eigenvalues <= 0)")
        inv_sqrt = vecs @ torch.diag(vals.rsqrt()) @ vecs.T

        self.register_buffer("_G", G)
        self.register_buffer("_G_inv_sqrt", inv_sqrt)

    def _build_gram_matrix(self, n: int) -> torch.Tensor:
        L = self.L
        # maximum possible size (all functions)
        max_n = 1 + int(self.include_linear) + int(self.include_quadratic) + 2 * self.n_harmonics
        G_full = torch.zeros((max_n, max_n), dtype=self.dtype, device=self.device)
        
        if max_n == 0:
            return G_full[:n, :n]

        idx_const = 0
        current_idx = 1
        idx_lin, idx_quad = None, None
        if self.include_linear:
            idx_lin = current_idx
            current_idx += 1
        if self.include_quadratic:
            idx_quad = current_idx
            current_idx += 1
        idx_trig_start = current_idx

        # correct for L2 + H1semi + H2semi
        # constant
        G_full[idx_const, idx_const] = L                                    # int_a^b 1 dx = L, no derivatives required

        if self.include_linear:
            if self.orthonormalize == "L2":
                G_full[idx_lin, idx_lin] = L**3 / 12.0  # ∫(x-mid)² dx
            elif self.orthonormalize == "H1":
                # ∫[(x-mid)² + 1²] dx = L³/12 + L
                G_full[idx_lin, idx_lin] = L**3 / 12.0 + L
            elif self.orthonormalize == "H2":
                # ∫[(x-mid)² + 1² + 0²] dx = L³/12 + L
                G_full[idx_lin, idx_lin] = L**3 / 12.0 + L

            # Cross term between constant and linear: integral of 1*(x-mid) dx = 0, and
            # it is zero for all derivative terms too.

            # cross <x-mid, sin(k)> = - L^2/(2*pi*k) (always negative)
            # cross terms with cos and (x-mid) are zero in L2(odd and even function), H1 (linear function is constant and orthogonal to cos) and H2 ( linear function is 0)
            if self.n_harmonics > 0:
                k_vals = torch.arange(1, self.n_harmonics + 1, device=self.device, dtype=self.dtype)
                # always negative: -L^2/(2*pi*k)
                cross_term_lin_sin = - (L**2 / (2.0 * math.pi * k_vals))
                for j in range(self.n_harmonics):
                    idx_sin = idx_trig_start + 2*j + 1
                    G_full[idx_lin, idx_sin] = cross_term_lin_sin[j]
                    G_full[idx_sin, idx_lin] = cross_term_lin_sin[j]

            
        # quadratic q = (x-mid)^2
        if self.include_quadratic:
            # diagonal quadratic - quadratic
            if self.orthonormalize == "L2":
                G_full[idx_quad, idx_quad] = L**5 / 80.0  # ∫(x-mid)⁴ dx
            elif self.orthonormalize == "H1":
                # ∫[(x-mid)⁴ + (2(x-mid))²] dx = L⁵/80 + 4*L³/12
                G_full[idx_quad, idx_quad] = L**5 / 80.0 + 4 * L**3 / 12.0
            elif self.orthonormalize == "H2":
                # ∫[(x-mid)⁴ + (2(x-mid))² + 2²] dx = L⁵/80 + 4*L³/12 + 4L
                G_full[idx_quad, idx_quad] = L**5 / 80.0 + 4 * L**3 / 12.0 + 4 * L


            # constant * quadratic cross term
            # integral of [constant * quadratic + constant' * quadratic' + constant'' * quadratic''] dx
            # L2 :  ∫ 1 * (x-mid)^2 dx = L^3/12
            # H1 :  ∫[1 * (x-mid)^2 + 0 * 2(x-mid) + 0 * 2] dx = L^3/12
            # H2 :  ∫[1 * (x-mid)^2 + 0 * 2(x-mid) + 0 * 2] dx = L^3/12
            G_full[idx_const, idx_quad] = L**3 / 12.0
            G_full[idx_quad, idx_const] = L**3 / 12.0

            # linear - quadratic contribution is always zero for any L2, H1 or H2
            # L2 : ∫ (x-mid) * (x-mid)^2 dx = ∫ (x-mid)^3 dx = 0 (odd function)
            # H1 : ∫[ (x-mid)^3 + 1 * 2(x-mid) ] dx
            #       first term: ∫ (x-mid)^3 dx = 0
            #       second term: ∫ 2(x-mid) dx = 0
            # H2: ∫[ (x-mid)^3 + 1 * 2(x-mid) + 0 * 2 ] dx = 0
            # no entry needed


            if self.n_harmonics > 0:
                k_vals = torch.arange(1, self.n_harmonics + 1, device=self.device, dtype=self.dtype)
                omega_k = 2.0 * math.pi * k_vals / L
                if self.orthonormalize == "L2":
                    # ∫ q * cos dx = L^3 / (2*pi^2*k^2) (always positive!)
                    cross_term = L**3 / (2.0 * math.pi**2 * k_vals**2)
                elif self.orthonormalize == "H1" or self.orthonormalize == "H2":
                    # term 1: ∫ q * cos dx
                    term1 = L**3 / (2.0 * math.pi**2 * k_vals**2)
                    # term 2: ∫ q' * cos' dx = ∫ 2(x-mid) * (-omega*sin) dx
                    # = -2*omega * ∫ (x-mid)*sin dx = -2*omega * [-L^2/(2*pi*k)]
                    term2 = 2 * omega_k * (L**2 / (2.0 * math.pi * k_vals))
                    # the "H2" term 3: ∫ q'' * cos'' dx = 0, so it's omitted
                    cross_term = term1 + term2

                for j in range(self.n_harmonics):
                    idx_cos = idx_trig_start + 2*j
                    if idx_cos < max_n:
                        G_full[idx_quad, idx_cos] = cross_term[j]
                        G_full[idx_cos, idx_quad] = cross_term[j]


        # trigonometric diagonal entries; by orthogonality there are no cross terms between
        # trigonometric components.
        for j in range(self.n_harmonics):
            idx_cos = idx_trig_start + 2*j
            idx_sin = idx_trig_start + 2*j + 1
            
            k = j + 1
            omega = 2.0 * math.pi * k / L
            base = L / 2.0

            if self.orthonormalize == "L2":
                norm_sq = base
            elif self.orthonormalize == "H1":
                norm_sq = base + (omega**2) * base
            elif self.orthonormalize == "H2":
                norm_sq = base + (omega**2) * base + (omega**4) * base
            else:
                raise ValueError("orthonormalize must be L2, H1, or H2")

            G_full[idx_cos, idx_cos] = norm_sq
            G_full[idx_sin, idx_sin] = norm_sq

        # trim to the desired size n
        return G_full[:n, :n]


    def _raw_basis_at(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw (non-orthonormalized) basis evaluated at x.
        Output shape: [N, n_raw]
        Vectorized, minimal python loops.
        """
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.view(-1)
        x = x.to(device=self.device, dtype=self.dtype).view(-1)
        N = x.shape[0]

        parts = []
        # constant
        parts.append(torch.ones((N, 1), device=self.device, dtype=self.dtype))

        mid = (self.a + self.b) / 2.0
        t = (x - self.a) / self.L  # normalized coordinate in [0,1]
        # linear
        if self.include_linear and len(parts) < self.n_raw:
            parts.append((x - mid).view(N, 1))

        # quadratic
        if self.include_quadratic and len(parts) < self.n_raw:
            parts.append(((x - mid) ** 2).view(N, 1))

        # harmonics: compute all angles at once [N, m]
        if self.n_harmonics > 0 and len(parts) < self.n_raw:
            k = self._k_tensor  # shape (m,)
            # angles: [N, m]
            angles = 2.0 * math.pi * (t.unsqueeze(1) * k.unsqueeze(0))  # broadcasting
            cos_terms = torch.cos(angles)
            sin_terms = torch.sin(angles)
            # interleave cos, sin but cap at n_raw
            # build a [N, 2*m] interleaved matrix and then slice appropriate number of columns
            interleaved = torch.empty((N, 2 * k.shape[0]), device=self.device, dtype=self.dtype)
            interleaved[:, 0::2] = cos_terms
            interleaved[:, 1::2] = sin_terms
            # how many trig columns we still need
            need = self.n_raw - len(parts)
            if need > 0:
                parts.append(interleaved[:, :need])

        # concat and ensure final width equals n_raw
        if len(parts) == 0:
            return torch.empty((N, 0), device=self.device, dtype=self.dtype)
        Phi = torch.cat(parts, dim=1)

        # sometimes last chunk might have more columns than needed (if we added combined cos/sin block)
        if Phi.shape[1] > self.n_raw:
            Phi = Phi[:, : self.n_raw]
        return Phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Phi = self._raw_basis_at(x)  # [N, n_raw]
        Psi = Phi @ self._G_inv_sqrt  # [N, n_raw] orthonormalized
        return Psi

    def derivative(self, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """
        Derivative of order `order` for each basis at x. Returns [N, n_raw].
        Uses vectorized phase-shifting patterns for trig derivatives.
        """
        if order < 0:
            raise ValueError("order must be nonnegative")
        if order == 0:
            return self.forward(x)

        if x.ndim == 2 and x.shape[1] == 1:
            x = x.view(-1)
        x = x.to(device=self.device, dtype=self.dtype).view(-1)
        N = x.shape[0]
        parts = []
        mid = (self.a + self.b) / 2.0
        t = (x - self.a) / self.L

        # constant derivative -> zero
        parts.append(torch.zeros((N, 1), device=self.device, dtype=self.dtype))

        # linear derivative
        if self.include_linear and len(parts) < self.n_raw:
            if order == 1:
                parts.append(torch.ones((N, 1), device=self.device, dtype=self.dtype))
            else:
                parts.append(torch.zeros((N, 1), device=self.device, dtype=self.dtype))

        # quadratic derivative
        if self.include_quadratic and len(parts) < self.n_raw:
            if order == 1:
                parts.append((2.0 * (x - mid)).view(N, 1))
            elif order == 2:
                parts.append(torch.full((N, 1), 2.0, device=self.device, dtype=self.dtype))
            else:
                parts.append(torch.zeros((N, 1), device=self.device, dtype=self.dtype))

        # harmonics: vectorized
        if self.n_harmonics > 0 and len(parts) < self.n_raw:
            k = self._k_tensor  # shape (m,)
            omega = self._omega_x  # shape (m,)
            angles = 2.0 * math.pi * (t.unsqueeze(1) * k.unsqueeze(0))  # [N, m]

            # derivative factor: omega**order
            # phase shift depends on order % 4
            r = order % 4
            ox_pow = (omega ** order).view(1, -1)  # [1, m]
            if r == 0:
                cos_part = torch.cos(angles) * ox_pow
                sin_part = torch.sin(angles) * ox_pow
            elif r == 1:
                cos_part = -torch.sin(angles) * ox_pow
                sin_part = torch.cos(angles) * ox_pow
            elif r == 2:
                cos_part = -torch.cos(angles) * ox_pow
                sin_part = -torch.sin(angles) * ox_pow
            else:  # r == 3
                cos_part = torch.sin(angles) * ox_pow
                sin_part = -torch.cos(angles) * ox_pow

            interleaved = torch.empty((N, 2 * k.shape[0]), device=self.device, dtype=self.dtype)
            interleaved[:, 0::2] = cos_part
            interleaved[:, 1::2] = sin_part
            need = self.n_raw - len(parts)
            if need > 0:
                parts.append(interleaved[:, :need])

        if len(parts) == 0:
            return torch.empty((N, 0), device=self.device, dtype=self.dtype)

        Phi_r = torch.cat(parts, dim=1)
        if Phi_r.shape[1] > self.n_raw:
            Phi_r = Phi_r[:, : self.n_raw]
        Psi_r = Phi_r @ self._G_inv_sqrt
        return Psi_r

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.derivative(x, order=2)


class OrthonormalFourierBasis(torch.nn.Module):
    def __init__(self,
                 domain=(0.0, 1.0),
                 n_basis=None,
                 n_harmonics=None,
                 device="cpu",
                 dtype=torch.float64,
                 ordering="const_cos_sin",
                 orthonormalize="L2"):
        super().__init__()
        a, b = float(domain[0]), float(domain[1])
        if b <= a:
            raise ValueError("domain must satisfy b > a")

        self.a, self.b = a, b
        self.L = torch.tensor(b - a)
        self.device = device
        self.dtype = dtype
        self.ordering = ordering
        self.orthonormalize = orthonormalize.upper()

        if n_basis is None and n_harmonics is None:
            raise ValueError("Provide either n_basis or n_harmonics")

        if n_basis is not None:
            m = math.ceil(max(0, (n_basis - 1)) / 2)
            produced = 1 + 2 * m
            self.requested_n_basis = n_basis
        else:
            m = int(n_harmonics)
            produced = 1 + 2 * m
            self.requested_n_basis = produced

        self.n_harmonics = m
        self.n_basis = produced

        if m > 0:
            k = torch.arange(1, m + 1, device=device, dtype=dtype)
            self.register_buffer("_k_tensor", k)
        else:
            self.register_buffer("_k_tensor", torch.empty((0,), device=device, dtype=dtype))

        # constant basis norm
        self.register_buffer("_norm_const", torch.tensor(1.0 / math.sqrt(self.L),
                                                           device=device, dtype=dtype))

        # frequency-dependent norms
        if m > 0:
            omega = 2.0 * math.pi * k / self.L
            if self.orthonormalize == "L2":
                norms = torch.sqrt(self.L/2) * torch.ones_like(omega)
            elif self.orthonormalize == "H1":
                norms = torch.sqrt(self.L/2 * (1 + omega**2))
            elif self.orthonormalize == "H2":
                norms = torch.sqrt(self.L/2 * (1 + omega**2 + omega**4))
            else:
                raise ValueError("orthonormalize must be L2, H1, or H2")
            self.register_buffer("_norm_harm_k", 1.0 / norms)
        else:
            self.register_buffer("_norm_harm_k", torch.empty((0,), device=device, dtype=dtype))

    def _assemble(self, const, cos_terms, sin_terms):
        """Helper to assemble the basis matrix."""
        if self.ordering == "const_cos_sin":
            parts = [const]
            for j in range(self.n_harmonics):
                parts.append(cos_terms[:, j:j+1])
                parts.append(sin_terms[:, j:j+1])
            B = torch.cat(parts, dim=1)
        else:
            B = torch.cat([const, cos_terms, sin_terms], dim=1)

        if B.shape[1] > self.requested_n_basis:
            B = B[:, :self.requested_n_basis]
        return B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the basis functions at x, accounting for orthonormalize."""
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.view(-1)
        x = x.view(-1).to(device=self._norm_const.device, dtype=self.dtype)
        N = x.shape[0]

        t = (x - self.a) / self.L
        if self.n_harmonics > 0:
            angles = 2.0 * math.pi * torch.outer(t, self._k_tensor)
            cos_terms = torch.cos(angles) * self._norm_harm_k
            sin_terms = torch.sin(angles) * self._norm_harm_k
        else:
            cos_terms = torch.empty((N, 0), device=self.device, dtype=self.dtype)
            sin_terms = torch.empty((N, 0), device=self.device, dtype=self.dtype)

        const = torch.full((N, 1), float(self._norm_const), device=self.device, dtype=self.dtype)
        return self._assemble(const, cos_terms, sin_terms)

    def derivative(self, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Derivative of the basis functions of arbitrary order under Sobolev orthonormalization."""
        if order < 0:
            raise ValueError("order must be nonnegative")
        if order == 0:
            return self.forward(x)

        if x.ndim == 2 and x.shape[1] == 1:
            x = x.view(-1)
        x = x.view(-1).to(device=self._norm_const.device, dtype=self.dtype)
        N = x.shape[0]

        t = (x - self.a) / self.L
        if self.n_harmonics > 0:
            angles = 2.0 * math.pi * torch.outer(t, self._k_tensor)  # [N, m]
            omega = 2.0 * math.pi * self._k_tensor / self.L  # [m]

            # derivative: swap cos/sin & signs, accounting for the norms
            if order % 4 == 1:
                cos_terms = -torch.sin(angles) * (self._norm_harm_k * omega)
                sin_terms = torch.cos(angles) * (self._norm_harm_k * omega)
            elif order % 4 == 2:
                cos_terms = -torch.cos(angles) * (self._norm_harm_k * omega**2)
                sin_terms = -torch.sin(angles) * (self._norm_harm_k * omega**2)
            elif order % 4 == 3:
                cos_terms = torch.sin(angles) * (self._norm_harm_k * omega**3)
                sin_terms = -torch.cos(angles) * (self._norm_harm_k * omega**3)
            else:  # order % 4 == 0
                cos_terms = torch.cos(angles) * (self._norm_harm_k * omega**4)
                sin_terms = torch.sin(angles) * (self._norm_harm_k * omega**4)
        else:
            cos_terms = torch.empty((N, 0), device=self.device, dtype=self.dtype)
            sin_terms = torch.empty((N, 0), device=self.device, dtype=self.dtype)

        const = torch.zeros((N, 1), device=self.device, dtype=self.dtype)  # constant term drops out
        return self._assemble(const, cos_terms, sin_terms)

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.derivative(x, order=2)


from numpy.polynomial.legendre import leggauss


def compute_basis_transform(
    basis_from: OrthonormalExtendedFourierBasis,
    basis_to: OrthonormalExtendedFourierBasis,
    inner_product: str = "L2",
    n_quadrature: int = 200,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Compute transformation matrix T such that beta = T @ alpha
    maps coefficients from basis_from to basis_to, restricted to the
    intersection domain of both bases.

    Args:
        basis_from: OrthonormalExtendedFourierBasis (source basis).
        basis_to:   OrthonormalExtendedFourierBasis (target basis).
        inner_product: "L2", "H1", or "H2".
        n_quadrature: number of Gauss-Legendre quadrature points.
        device, dtype: torch device/dtype.

    Returns:
        T: (d_to, d_from) torch.Tensor transformation matrix.
    """
    # intersection domain
    a = max(basis_from.a, basis_to.a)
    b = min(basis_from.b, basis_to.b)
    if b <= a:
        raise ValueError("Empty intersection domain between bases")

    # Gauss-Legendre quadrature on [a,b]
    xg, wg = leggauss(n_quadrature)
    xg = 0.5 * (b - a) * xg + 0.5 * (a + b)
    wg = 0.5 * (b - a) * wg
    xg_t = torch.tensor(xg, device=device, dtype=dtype)
    wg_t = torch.tensor(wg, device=device, dtype=dtype)

    # evaluations at quadrature points
    Phi_from = basis_from.forward(xg_t)            # (N, d_from)
    Phi_to = basis_to.forward(xg_t)                # (N, d_to)
    d_from, d_to = Phi_from.shape[1], Phi_to.shape[1]

    terms_from = {"0": Phi_from}
    terms_to = {"0": Phi_to}

    if inner_product.upper() in ("H1", "H2"):
        Phi_from_p = basis_from.derivative(xg_t, 1)
        Phi_to_p   = basis_to.derivative(xg_t, 1)
        terms_from["1"] = Phi_from_p
        terms_to["1"]   = Phi_to_p

    if inner_product.upper() == "H2":
        Phi_from_pp = basis_from.derivative(xg_t, 2)
        Phi_to_pp   = basis_to.derivative(xg_t, 2)
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


class TensorExtendedFourierBasis:
    def __init__(self, domains, n_basis_list, orthonormalize="L2", include_linear: bool = True, include_quadratic: bool = True, device="cpu", dtype=torch.float64):
        """
        Tensor-product Fourier basis in d dimensions.

        Args:
            domains: list of (a,b) intervals, one per dimension
            n_basis_list: list of the desired number of basis functions, one per dimension
            device, dtype: PyTorch options
        """
        self.d = len(domains)
        assert len(n_basis_list) == self.d, "n_basis_list must have the same length as domains"
        self.device = device
        self.dtype = dtype
        self.orthonormalize = orthonormalize.upper()
        self.include_linear = include_linear
        self.include_quadratic = include_quadratic

        self.bases = []
        if not include_linear and not include_quadratic:
            for j in range(self.d):
                fb = OrthonormalFourierBasis(
                    domain=domains[j],
                    n_basis=n_basis_list[j],
                    device=device,
                    dtype=dtype,
                    orthonormalize=self.orthonormalize,
                )
                self.bases.append(fb)
        elif include_linear and include_quadratic:
            for j in range(self.d):
                fb = OrthonormalExtendedFourierBasis(
                    domain=domains[j],
                    n_basis=n_basis_list[j],
                    device=device,
                    dtype=dtype,
                    orthonormalize=self.orthonormalize,
                )
                self.bases.append(fb)
        else:
            raise NotImplementedError("Other cases not verified so far")

        self.domain_bounds = (
            torch.tensor([domains[j][0] for j in range(self.d)], device=device, dtype=dtype),
            torch.tensor([domains[j][1] for j in range(self.d)], device=device, dtype=dtype),
        )

        self.ndofs = [fb.requested_n_basis for fb in self.bases]

    def prepare_save_state(self, to_cpu=False):
        state = {
            "basis_type": "TensorExtendedFourierBasis",
            "d": self.d,
            "domain_bounds": (
                self.domain_bounds[0].cpu() if to_cpu else self.domain_bounds[0],
                self.domain_bounds[1].cpu() if to_cpu else self.domain_bounds[1],
            ),
            "domains": [(float(self.domain_bounds[0][j]), float(self.domain_bounds[1][j])) for j in range(self.d)],
            "n_basis_list": [fb.requested_n_basis for fb in self.bases],
            "orthonormalize": self.orthonormalize,
            "include_linear": self.include_linear,
            "include_quadratic": self.include_quadratic,
            "device": "cpu" if to_cpu else self.device,
            # save each univariate Fourier basis too
            "fourier_bases": [
                {
                    "domain": (fb.a, fb.b),
                    "requested_n_basis": fb.requested_n_basis,
                    "n_harmonics": fb.n_harmonics,
                    "n_raw": fb.n_raw,
                    "include_linear": fb.include_linear,
                    "include_quadratic": fb.include_quadratic,
                    "orthonormalize": fb.orthonormalize,
                    "_G": fb._G.cpu() if to_cpu else fb._G,
                    "_G_inv_sqrt": fb._G_inv_sqrt.cpu() if to_cpu else fb._G_inv_sqrt,
                    "_k_tensor": fb._k_tensor.cpu() if to_cpu else fb._k_tensor,
                    "_omega_x": fb._omega_x.cpu() if to_cpu else fb._omega_x,
                }
                for fb in self.bases
            ],
        }
        return state

    @classmethod
    def load_from_state(cls, state):
        assert state["basis_type"] == "TensorExtendedFourierBasis"

        # create an empty object without calling __init__()
        obj = cls.__new__(cls)
        obj.d = state["d"]
        obj.device = state["device"]
        obj.orthonormalize = state["orthonormalize"]
        obj.include_linear = state["include_linear"]
        obj.include_quadratic = state["include_quadratic"]
        obj.domain_bounds = state["domain_bounds"]

        obj.bases = []


        for fb_state in state["fourier_bases"]:
            fb = OrthonormalExtendedFourierBasis.__new__(OrthonormalExtendedFourierBasis)
            torch.nn.Module.__init__(fb)
            fb.a, fb.b = fb_state["domain"]
            fb.L = fb.b - fb.a
            fb.device = torch.device(obj.device)
            fb.dtype = torch.float64
            fb.ordering = "const_lin_quad_cos_sin"  # default
            fb.orthonormalize = fb_state["orthonormalize"]
            fb.include_linear = fb_state["include_linear"]
            fb.include_quadratic = fb_state["include_quadratic"]
            fb.requested_n_basis = fb_state["requested_n_basis"]
            fb.n_harmonics = fb_state["n_harmonics"]
            fb.n_raw = fb_state["n_raw"]

            fb._G = fb_state["_G"].to(obj.device)
            fb._G_inv_sqrt = fb_state["_G_inv_sqrt"].to(obj.device)
            fb.register_buffer("_k_tensor", fb_state["_k_tensor"].to(obj.device))
            fb.register_buffer("_omega_x", fb_state["_omega_x"].to(obj.device))

            obj.bases.append(fb)

        obj.ndofs = [fb.requested_n_basis for fb in obj.bases]
        return obj

    def __call__(self, x):
        """
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j]
        """
        b, d = x.shape
        assert d == self.d
        evaluations = []
        for j in range(d):
            evals = self.bases[j](x[:, j])
            evaluations.append(evals)
        return evaluations

    def grad(self, x):
        """
        First derivatives.
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j]
        """
        b, d = x.shape
        assert d == self.d
        grads = []
        for j in range(d):
            g = self.bases[j].derivative(x[:, j], order=1)
            grads.append(g)
        return grads

    def D2(self, x):
        """
        Second derivatives.
        x: [b, d]
        Returns: list of d tensors [b, n_basis_j]
        """
        b, d = x.shape
        assert d == self.d
        D2s = []
        for j in range(d):
            D2_j = self.bases[j].second_derivative(x[:, j])
            D2s.append(D2_j)
        return D2s


def plot_fourier_basis():
    import matplotlib.pyplot as plt
    domain = (-2.3, 2.4)
    mod = OrthonormalFourierBasis(domain=domain, n_basis=5, orthonormalize="H2")

    x = torch.linspace(*domain, 400)

    B0 = mod(x)                    # basis
    B1 = mod.derivative(x, 1)      # first derivative
    B2 = mod.second_derivative(x)  # second derivative

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for j in range(B0.shape[1]):
        axes[0].plot(x.numpy(), B0[:, j].numpy(), label=f"phi{j}")
    axes[0].set_title("Fourier basis functions")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(True)

    for j in range(B1.shape[1]):
        axes[1].plot(x.numpy(), B1[:, j].numpy(), label=f"phi{j}'")
    axes[1].set_title("First derivatives")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(True)

    for j in range(B2.shape[1]):
        axes[2].plot(x.numpy(), B2[:, j].numpy(), label=f"phi{j}''")
    axes[2].set_title("Second derivatives")
    axes[2].legend(ncol=4, fontsize=8)
    axes[2].grid(True)

    axes[2].set_xlabel("x")
    plt.tight_layout()
    plt.show()

def test_transform():
    import matplotlib.pyplot as plt
    ndofs_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    plt.figure(figsize=(12, 4))
    for i, ndofs in enumerate(ndofs_list):
        # two bases with slightly shifted domains
        B1 = OrthonormalExtendedFourierBasis(domain=(0.0, 1.0), n_basis=ndofs, orthonormalize="H2")
        B2 = OrthonormalExtendedFourierBasis(domain=(0.05, 1.09), n_basis=ndofs, orthonormalize="H2")

        T = compute_basis_transform(B1, B2, inner_product="H2", n_quadrature=200)

        # random coefficients in basis 1
        alpha = 10 * torch.rand(B1.requested_n_basis, dtype=torch.float64)
        if i > 3:
            alpha[1] = 0.1
            alpha[2] = -0.01
        beta = T @ alpha

        # intersection range
        a_int = max(B1.a, B2.a)
        b_int = min(B1.b, B2.b)

        # plot points
        x1 = torch.linspace(B1.a, B1.b, 500)
        x2 = torch.linspace(B2.a, B2.b, 500)

        f1 = (B1.forward(x1) @ alpha).detach().numpy()
        f2 = (B2.forward(x2) @ beta).detach().numpy()
        f1_p = (B1.derivative(x1, 1) @ alpha).detach().numpy()
        f2_p = (B2.derivative(x2, 1) @ beta).detach().numpy()
        f1_pp = (B1.derivative(x1, 2) @ alpha).detach().numpy()
        f2_pp = (B2.derivative(x2, 2) @ beta).detach().numpy()

        x1_np, x2_np = x1.detach().numpy(), x2.detach().numpy()

        # function
        plt.subplot(len(ndofs_list), 3, 3 * i + 1)
        plt.plot(x1_np, f1, "r-", label="B1*alpha on domain 1")
        plt.plot(x2_np, f2, "b-", label="B2*beta on domain 2")
        plt.axvline(a_int, color="k", linestyle=":", label="Intersection")
        plt.axvline(b_int, color="k", linestyle=":")
        plt.title("Function")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        # first derivative
        plt.subplot(len(ndofs_list), 3, 3 * i + 2)
        plt.plot(x1_np, f1_p, "r-", label="B1*alpha on domain 1")
        plt.plot(x2_np, f2_p, "b-", label="B2*beta on domain 2")
        plt.axvline(a_int, color="k", linestyle=":", label="Intersection")
        plt.axvline(b_int, color="k", linestyle=":")
        plt.title("First derivative")
        plt.xlabel("x")
        plt.ylabel("f'(x)")

        # second derivative
        plt.subplot(len(ndofs_list), 3, 3 * i + 3)
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
