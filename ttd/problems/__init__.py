"""Problem definitions for TTD."""

from ttd.problems.base import (
    GaussianMixture,
    GeneralizedProblem,
    StandardNormal,
    SymmetricGaussianMixture2D,
    Target,
    absshape_reguarlization,
    const_regularization,
    get_Basis,
    get_basis_info,
    handle_rank,
    scheduling,
)
from ttd.problems.concrete import (
    Funnel,
    GaussianProblem,
    GinzburgLandau,
    Kitagawa,
    Multiwell,
)
from ttd.problems.quadrature import (
    compute_score_by_hermite_quadrature,
    compute_score_by_legendre_quadrature,
)
