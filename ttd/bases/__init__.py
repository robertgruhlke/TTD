"""Basis functions for tensor-train fitting."""

from ttd.bases.fourier import TensorExtendedFourierBasis, compute_basis_transform
from ttd.bases.legendre import (
    TensorLegendreBasis,
    compute_basis_transform_Legendre,
    orthpoly,
)
from ttd.bases.rank_rules import (
    Absolute_Singularvalue_Tresholding,
    Dörfler_Adaptivity,
    Relative_Singularvalue_Tresholding,
    Rule,
    Threshold,
)
from ttd.bases.spline import (
    BSplines,
    HighDegreeCSpline,
    TensorSplineBasis,
    TensorSplineBasis_Equidistant,
    compute_basis_transform_spline,
)
