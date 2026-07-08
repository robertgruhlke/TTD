"""Research scratch: standalone demo/benchmark functions for the basis and xTT
machinery (fitting, gradients, Hessians, speed comparisons, basis-transform checks).
Not a unit-test suite — each function is meant to be run one at a time.
"""

import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

from ttd.bases.fourier import TensorExtendedFourierBasis
from ttd.bases.legendre import TensorLegendreBasis
from ttd.bases.spline import (
    HighDegreeCSpline,
    TensorSplineBasis,
    TensorSplineBasis_Equidistant,
)
from ttd.policies import Langevin_policy, quadratic_annealed_Langevin
from ttd.problems.base import GeneralizedProblem, get_Basis, get_basis_info, handle_rank
from ttd.problems.concrete import Funnel, Multiwell
from ttd.problems.quadrature import compute_score_by_legendre_quadrature
from ttd.sde import simulate_SDE
from ttd.solvers.als import ALS_H1, ALS_L2
from ttd.solvers.backward_iteration import min_max_per_dim, train
from ttd.tt.extended import Extended_TensorTrain, projected_presentation_change
from ttd.utils.plotting import plotSamples
from ttd.xftt import xFTT


def test_1d():
    device = "cpu"

    # 1D grid points
    a, b = -1., 1.
    grid_points = [torch.linspace(a, b, 3)]  # single dimension
    p = [5]  # local polynomial degree
    s = [2]  # spline smoothness

    spline_basis = TensorSplineBasis(grid_points, p, s, device)

    # evaluation grid
    n = 200
    x_vals = torch.linspace(a, b, n).reshape(-1, 1)

    # evaluate basis (list with one element)
    [B] = spline_basis(x_vals)

    # random coefficients
    coeffs = torch.randn(B.shape[1], device=device)

    # contract
    y = B @ coeffs

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals.numpy(), y.detach().numpy(), lw=2)
    plt.title("Random 1D Tensor Spline")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()


def test_2d():
    device = "cpu"
    a1, b1 = -1., 1.
    a2, b2 = -1., 1.
    grid_points = [torch.linspace(a1, b1, 4), torch.linspace(a2, b2, 5)]
    p = [5, 5]  # local polynomial degree
    s = [2, 2]  # spline smoothness

    spline_basis = TensorSplineBasis(grid_points, p, s, device)

    # evaluation grid
    n = 500
    x_vals = torch.linspace(-1, 1, n)
    y_vals = torch.linspace(-1, 1, n)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # evaluate basis
    Bx, By = spline_basis(XY)

    n_basis = [Bx.shape[1], By.shape[1]]
    print("Number of basis functions per dimension:", n_basis)
    # random coefficients
    coeffs = torch.randn(n_basis[0], n_basis[1], device=device)

    # contract
    Z = torch.einsum("bi,bj,ij->b", Bx, By, coeffs).reshape(n, n)

    # plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X.numpy(), Y.numpy(), Z.detach().numpy(), cmap="viridis")
    ax.set_title("Random 2D Tensor Spline")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    plt.show()


def fit_tensor_spline(spline_basis, g, N=2000, device="cpu"):
    d = len(spline_basis.splines)

    # 1. sample points
    a, b = -1., 1.
    X_train = torch.rand(N, d, device=device) * (b - a) + a
    y_train = g(X_train)

    # 2. evaluate basis
    Bx, By = spline_basis(X_train)
    n_basis = [Bx.shape[1], By.shape[1]]

    # 3. build the full design matrix Phi with a Kronecker product
    # Phi[k, i*n2 + j] = Bx[k,i] * By[k,j]
    Phi = torch.einsum("ki,kj->kij", Bx, By).reshape(N, -1)

    # 4. solve least squares Phi * coeffs ~ y_train
    coeffs_vec, *_ = torch.linalg.lstsq(Phi, y_train)
    coeffs = coeffs_vec.view(n_basis)

    return coeffs


def test_fit_2d():
    # target function
    def g(X):
        return torch.sin(2 * X[:, 0]) * torch.cos(3 * X[:, 1])

    torch.manual_seed(0)
    device = "cpu"
    grid_points = [torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 5)]
    p = [5, 5]
    s = [2, 2]

    spline_basis = TensorSplineBasis(grid_points, p, s, device)

    # fit
    coeffs = fit_tensor_spline(spline_basis, g)

    # visualization grid
    n = 200
    x_vals = torch.linspace(-1, 1, n)
    y_vals = torch.linspace(-1, 1, n)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # original
    Z_true = g(XY).reshape(n, n)

    # approximation
    Bx, By = spline_basis(XY)
    Z_approx = torch.einsum("bi,bj,ij->b", Bx, By, coeffs).reshape(n, n)

    # plot
    fig = plt.figure(figsize=(12, 5))

    # left: original
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X.numpy(), Y.numpy(), Z_true.detach().numpy(), cmap="viridis")
    ax1.set_title("Original function $g(x,y)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x,y)")

    # right: approximation
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(X.numpy(), Y.numpy(), Z_approx.detach().numpy(), cmap="viridis")
    ax2.set_title("Tensor Spline Approximation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("f(x,y)")

    plt.tight_layout()
    plt.show()


def test_spline_als():
    torch.manual_seed(0)
    device = "cpu"
    grid_points = [torch.linspace(-2.5, 2.5, 7), torch.linspace(-2.5, 2.5, 7)]
    p = [5, 5]
    s = [2, 2]

    tensor_basis_functions = TensorSplineBasis(grid_points, p, s, device)
    ranks = [1, 3, 1]
    xtt = Extended_TensorTrain(tensor_basis_functions, ranks)

    # target
    def g(X):
        return torch.sin(1 * X[:, 0]) * torch.cos(4 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 1] + 5 * torch.exp(-X[:, 0] ** 2) * torch.exp(-X[:, 1] ** 2)

    d = len(tensor_basis_functions.splines)
    # 1. sample points
    B = 4000
    a, b = -2.5, 2.5
    X_train = torch.rand(B, d, device=device) * (b - a) + a

    y_train = g(X_train).reshape(-1, 1)

    ALS_L2(X_train, y_train, iterations=100, tol=1e-6, reg_param=0., xTT=xtt, verbose=True)

    # visualization grid
    n = 200
    x_vals = torch.linspace(a, b, n)
    y_vals = torch.linspace(a, b, n)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # original
    Z_true = g(XY).reshape(n, n)

    # approximation
    Z_approx = xtt(XY).reshape(n, n)

    # plot
    fig = plt.figure(figsize=(12, 5))

    # left: original
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X.numpy(), Y.numpy(), Z_true.detach().numpy(), cmap="viridis")
    ax1.set_title("Original function $g(x,y)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x,y)")

    # right: approximation
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(X.numpy(), Y.numpy(), Z_approx.detach().numpy(), cmap="viridis")
    ax2.set_title("Tensor Spline Approximation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("f(x,y)")

    plt.tight_layout()
    plt.show()


def test_equiv_splines():
    grid_points = torch.linspace(-2.5, 2.5, 5)
    spline = HighDegreeCSpline(grid_points, p=5, s=2, device="cpu", dtype=torch.float64, orthonormalize="H2")

    N = 50000
    d = 100

    x_1d = grid_points[0] + (grid_points[-1] - grid_points[0]) * torch.rand(N)
    x_batched = grid_points[0] + (grid_points[-1] - grid_points[0]) * torch.rand(N * d)

    # --- standard ---
    start = time.perf_counter()
    for _ in range(d):
        y = spline.bspline_basis_matrix(x_1d, spline.knots, spline.degree)
    end = time.perf_counter()
    time_baseline = end - start

    # --- fast ---
    start = time.perf_counter()
    y_old = spline.bspline_basis_matrix(x_batched, spline.knots, spline.degree)
    end = time.perf_counter()
    time_fast_1 = end - start

    # --- fast (gpu) ---
    start = time.perf_counter()
    y_old = spline.bspline_basis_matrix_fast_gpu(x_batched, spline.knots, spline.degree)
    end = time.perf_counter()
    time_fast_2 = end - start

    print(f"baseline: {time_baseline:.4f} sec")
    print(f"batched     spline  : {time_fast_1:.4f} sec")
    print(f"batched gpu spline  : {time_fast_2:.4f} sec")
    print(f"Speed gain  base / batched spline : {time_baseline/time_fast_1:.2f}x faster")
    print(f"Speed gain  base / batched gpu spline : {time_baseline/time_fast_2:.2f}x faster")


def test_derivative_spline():

    torch.manual_seed(0)

    grid_points = torch.linspace(-2.3, 4., 3)
    spline = HighDegreeCSpline(grid_points, p=3, s=2, device="cpu", dtype=torch.float64, orthonormalize="H2")

    N = 5000
    x_1d = torch.linspace(-2.3, 4., N)
    y = spline.bspline_basis_matrix(x_1d, spline.knots, spline.degree)
    dy = spline.bspline_basis_deriv_matrix(x_1d)

    h = 1e-6
    dy_plus_eps = spline.bspline_basis_matrix(x_1d + h, spline.knots, spline.degree)
    dy_minus_eps = spline.bspline_basis_matrix(x_1d - h, spline.knots, spline.degree)

    dy_approx = (dy_plus_eps - y) / h
    dy_approx[-1] = (y[-1] - dy_minus_eps[-1]) / h

    coeff = torch.randn(spline.n_basis)

    plt.subplot(1, 2, 1)
    plt.plot(x_1d.numpy(), (y @ coeff).detach().numpy(), label="spline basis")
    plt.subplot(1, 2, 2)
    plt.plot(x_1d.numpy(), (dy @ coeff).detach().numpy(), label="spline derivative")
    plt.plot(x_1d.numpy(), (dy_approx @ coeff).detach().numpy(), "--", label="finite difference")

    plt.grid(visible=True, which="both")
    plt.legend()
    plt.show()

def test_eval_xtt():
    device = "cpu"

    dim = 10
    N = 50000
    rank = 20

    grid_points = [torch.linspace(-2.5, 2.5, 5) for _ in range(dim)]
    p = [5] * dim
    s = [2] * dim

    device = "cpu"

    tensor_basis_functions = TensorSplineBasis(grid_points, p, s, device)
    ranks = [1] + [rank] * (dim - 1) + [1]
    xtt = Extended_TensorTrain(tensor_basis_functions, ranks, eps=0.1, device=device)
    xtt.tt.set_core(0)

    # lower and upper bounds per dimension
    lower = torch.tensor([g[0] for g in grid_points])
    upper = torch.tensor([g[-1] for g in grid_points])

    # uniform sampling in each dimension
    x = lower + (upper - lower) * torch.rand(N, dim)

    start = time.perf_counter()
    w = xtt.tensor_basis_functions(x)
    end = time.perf_counter()
    time_basis = end - start

    start = time.perf_counter()
    w2 = xtt.tensor_basis_functions.new_call(x)
    end = time.perf_counter()
    time_basis_list_comp = end - start

    start = time.perf_counter()
    w2 = xtt.tensor_basis_functions.new_call_parallel(x)
    end = time.perf_counter()
    time_basis_paralell = end - start

    print(f"basis evaluation old call: {time_basis:.4f} sec")
    print(f"basis evaluation new call : {time_basis_list_comp:.4f} sec")
    print(f"basis evaluation parallel call : {time_basis_paralell:.4f} sec")

    # --- standard ---
    start = time.perf_counter()
    d_base = xtt.tt.dot_rank_one(w)
    end = time.perf_counter()
    time_baseline = end - start

    # --- fast einsum ---
    start = time.perf_counter()
    d_new_torch_einsum = xtt.tt.dot_rank_one_new(w, "torch")
    end = time.perf_counter()
    time_fast = end - start

    start = time.perf_counter()
    d_new_no_einsum = xtt.tt.dot_rank_one_new(w, None)
    end = time.perf_counter()
    time_fast3 = end - start

    # comparison
    print(f"baseline: {time_baseline:.4f} sec")
    print(f"new version with torch einsum  : {time_fast:.4f} sec")
    print(f"new version non einsum  : {time_fast3:.4f} sec")
    print(f"Speed gain  base / with torch einsum : {time_baseline/time_fast:.2f}x faster")
    print(f"Speed gain  base / without    einsum : {time_baseline/time_fast3:.2f}x faster")
    print("err = ", torch.linalg.norm(d_base - d_new_torch_einsum))
    print("err = ", torch.linalg.norm(d_base - d_new_no_einsum))


def basis_representation_change_Legendre():
    device = "cpu"

    d = 2
    lim_from = [-1.1, 1.1]
    lim_to = [-1.2, 1.05]

    overlap = [-1.1, 1.05]

    orthonormalization = "H2"

    basis_info_from = {
        "type": "TensorLegendreBasis",
        "lims": [lim_from for _ in range(d)],
        "deg": [5] * d,
        "orthonormalization": orthonormalization,
        "device": device,
    }
    _, basis_from = get_Basis(basis_info_from)

    basis_info_to = {
        "type": "TensorLegendreBasis",
        "lims": [lim_to for _ in range(d)],
        "deg": [5] * d,
        "orthonormalization": orthonormalization,
        "device": device,
    }

    _, basis_to = get_Basis(basis_info_to)

    ranks = handle_rank(rank=2, d=d)
    xtt_from = Extended_TensorTrain(basis_from, ranks, eps=1, device=device)
    xtt_from.tt.set_core(0)
    xtt_to = Extended_TensorTrain(basis_to, ranks, eps=1, device=device)

    projected_presentation_change(xtt_from, xtt_to, inner_univariate_product="H2", n_quadrature=200)

    if d == 1:

        x_vals_from = torch.linspace(lim_from[0], lim_from[1], 200).reshape(-1, 1)
        y_from = xtt_from(x_vals_from)

        x_vals_to = torch.linspace(lim_to[0], lim_to[1], 200).reshape(-1, 1)
        y_to = xtt_to(x_vals_to)

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals_from.numpy(), y_from.numpy(), label="xtt from ", lw=2)
        plt.plot(x_vals_to.numpy(), y_to.detach().numpy(), "--", label="xtt to", lw=2)

        plt.plot([overlap[0]] * 2, [-1, 1], "k--")
        plt.plot([overlap[1]] * 2, [-1, 1], "k--")

        plt.title("Function Approximation with Tensor Train Spline")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()

    if d == 2:

        # build a grid of points in the rectangle [lower_bound, upper_bound]
        n_grid = 50
        x1_from = torch.linspace(lim_from[0], lim_from[1], n_grid)
        x2_from = torch.linspace(lim_from[0], lim_from[1], n_grid)
        X1, X2 = torch.meshgrid(x1_from, x2_from, indexing="ij")
        grid_points = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)
        Y_from = xtt_from(grid_points).detach().reshape(n_grid, n_grid)

        fig = plt.figure(figsize=(12, 12))
        # true function surface
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot_surface(X1.numpy(), X2.numpy(), Y_from.numpy(),
                        cmap="viridis", alpha=0.8)
        ax1.set_title(" xTT from on its original domain")

        # build a grid of points in the rectangle [lower_bound, upper_bound]
        n_grid = 50
        x1_to = torch.linspace(lim_to[0], lim_to[1], n_grid)
        x2_to = torch.linspace(lim_to[0], lim_to[1], n_grid)
        X1, X2 = torch.meshgrid(x1_to, x2_to, indexing="ij")
        grid_points = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)
        Y_to = xtt_to(grid_points).detach().reshape(n_grid, n_grid)

        # true function surface
        ax1 = fig.add_subplot(2, 2, 2, projection="3d")
        ax1.plot_surface(X1.numpy(), X2.numpy(), Y_to.numpy(),
                        cmap="viridis", alpha=0.8)
        ax1.set_title(" xTT to on its original domain ")

        # add the overlapping domain

        # build a grid of points in the rectangle [lower_bound, upper_bound]
        n_grid = 50
        x1 = torch.linspace(overlap[0], overlap[1], n_grid)
        x2 = torch.linspace(overlap[0], overlap[1], n_grid)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        grid_points = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)
        Y_from_overlap = xtt_from(grid_points).detach().reshape(n_grid, n_grid)
        Y_to_overlap = xtt_to(grid_points).detach().reshape(n_grid, n_grid)

        # true function surface
        ax1 = fig.add_subplot(2, 2, 3, projection="3d")
        ax1.plot_surface(X1.numpy(), X2.numpy(), Y_from_overlap.numpy(),
                        cmap="viridis", alpha=0.8)
        ax1.set_title(" xTT from on overlap domain")

        ax1 = fig.add_subplot(2, 2, 4, projection="3d")
        ax1.plot_surface(X1.numpy(), X2.numpy(), Y_to_overlap.numpy(),
                        cmap="viridis", alpha=0.8)
        ax1.set_title(" xTT to on overlap domain")

        plt.show()

def test_fitting_with_ALS_H1_loss():
    d = 4

    device = "cpu"
    sample_mode = 0
    basis_choice = 1

    reg_init = 1e-8
    timeSteps = 1000

    plot_weighted_error = False
    use_policy_samples = True

    B = 4000
    plot_initial_fit = False  # plots then exit

    maximal_computational_domain = [[-6.5, 5.], [-5., 5.]]

    maximal_computational_domain = [[-3, 3] for _ in range(d)]

    maximal_plot_domain = [[-10, 10.], [-15., 15.]]

    if basis_choice == 0:
        nknots_x1 = 5
        nknots_remain = 3
        deg_x1 = 10
        deg_remain = 2
        s = 1
        basis_info = {
            "type": "TensorSplineBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],  # does not matter
            "nknots": [nknots_x1] + [nknots_remain] * (d - 1),
            "deg": [deg_x1] + [deg_remain] * (d - 1),
            "s": [s] * d,
            "device": "cpu",
            "orthonormalize": "H2",
        }
        basis_info_x1 = {
            "type": "TensorSplineBasis",
            "lims": None,
            "nknots": [nknots_x1],
            "deg": [deg_x1],
            "orthonormalize": "H2",
            "s": [s],
            "device": device,
        }

        basis_info_x2 = {
            "type": "TensorSplineBasis",
            "nknots": [nknots_remain],
            "lims": None,
            "deg": [deg_remain],
            "orthonormalize": "H2",
            "s": [s],
            "device": device,
        }
    elif basis_choice == 1:
        include_linear = True
        include_quadratic = True

        num_trigonmetric = 20
        num_trigonmetric_remain = 20

        basis_info = {
            "type": "TensorExtendedFourierBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + num_trigonmetric]
            + [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + num_trigonmetric_remain] * (d - 1),
            "device": device,
            "orthonormalize": "H2",
            "include_linear": include_linear,
            "include_quadratic": include_quadratic,
        }

        basis_info_x1 = {
            "type": "TensorExtendedFourierBasis",
            "lims": None,
            "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + num_trigonmetric],
            "orthonormalize": "H2",
            "device": device,
            "include_linear": include_linear,
            "include_quadratic": include_quadratic,
        }

        basis_info_x2 = {
            "type": "TensorExtendedFourierBasis",
            "lims": None,
            "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + num_trigonmetric_remain],
            "orthonormalize": "H2",
            "device": device,
            "include_linear": include_linear,
            "include_quadratic": include_quadratic,
        }
    elif basis_choice == 2:
        deg_x1 = 20
        deg_x_rest = 2

        basis_info = {
            "type": "TensorLegendreBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "deg": [deg_x1] + [deg_x_rest] * (d - 1),
            "orthonormalize": "H2",
            "device": "cpu",
        }

        basis_info_x1 = {
            "type": "TensorLegendreBasis",
            "lims": None,
            "deg": [deg_x1],
            "orthonormalize": "H2",
            "device": device,
        }

        basis_info_x2 = {
            "type": "TensorLegendreBasis",
            "lims": None,
            "deg": [deg_x_rest],
            "orthonormalize": "H2",
            "device": device,
        }

    problem = Funnel(d, basis_info, rank=2, timeSteps=timeSteps, batch_size=B, T=3, reg=0)
    problem = Multiwell(d, n_double_wells=d, basis_info=basis_info, timeSteps=timeSteps, batch_size=B, T=2., reg=1e-4)

    myxFTT = xFTT(problem, None, p_gradient_extension=0.25, device=device, initial_policy=Langevin_policy)
    batch_size = problem.batch_size
    timeSteps = problem.timeSteps
    stepSizes = problem.stepSizes
    times = problem.times
    # we get the whole sample trajectory with the current drift f + sigma u
    # TODO: avoid returning sig_u, since it is only time-dependent
    training_data = simulate_SDE(problem, myxFTT.u, stepSizes, batch_size,
                                              simulation_device=myxFTT.device, computation_device=myxFTT.device)

    def masking_data(some_training_data, domain):
        a1, b1 = domain[0]
        X_u, xi, sig_u, u_val, tp = some_training_data
        mask_final_samples = (
            (X_u[..., 0] > a1) & (X_u[..., 0] < b1)
        ).all(dim=0)

        for i in range(1, d):
            ai, bi = domain[i]
            mask_final_samples = mask_final_samples & (
                (X_u[..., i] > ai) & (X_u[..., i] < bi)
            ).all(dim=0)
        return X_u[:, mask_final_samples], xi[:, mask_final_samples], sig_u, u_val[:, mask_final_samples], tp

    def masking_samples(samples, domain):
        a1, b1 = domain[0]
        mask_final_samples = (samples[:, 0] > a1) & (samples[:, 0] < b1)
        for i in range(1, d):
            ai, bi = domain[i]
            mask_final_samples = mask_final_samples & (samples[:, i] > ai) & (samples[:, i] < bi)
        return samples[mask_final_samples]

    training_data = masking_data(training_data, maximal_computational_domain)
    X_u, _, _, _, _ = training_data

    def ancestral_sample_funnel_numpy(num_samples=5000, D=10):
        num_x_dims = D - 1
        v = np.random.normal(loc=0, scale=3, size=num_samples)
        std_x = np.exp(v / 2)
        x = np.random.normal(loc=0, scale=std_x[:, np.newaxis], size=(num_samples, num_x_dims))
        samples = np.concatenate([v[:, np.newaxis], x], axis=1)
        return samples

    direct_samples = ancestral_sample_funnel_numpy(num_samples=B, D=d)
    direct_samples = masking_samples(torch.tensor(direct_samples), maximal_computational_domain)

    if d < 3:
        plotSamples(direct_samples, X_u[-1], label="_training", bins=100, label1="reference masked samples", label2="initial policy masked samples")

    if use_policy_samples:
        X_train = X_u[-1]
    else:
        X_train = direct_samples

    target = lambda x: problem.target(x).reshape(-1, 1)
    Dtarget = lambda x: problem.target.grad(x)

    # initial fitting via ALS L2
    Y_vals = target(X_train)
    DY_vals = Dtarget(X_train)

    lims = min_max_per_dim(X_train)
    print(lims)

    lower_bound = torch.tensor([lim[0] for lim in lims])
    upper_bound = torch.tensor([lim[1] for lim in lims])
    domain_width = torch.abs(upper_bound - lower_bound)
    q = 0.15
    lims_extended = [[lim_i[0] - q * domain_width[0], lim_i[1] + q * domain_width[i]] for i, lim_i in enumerate(lims)]
    tensor_basis, ranks = problem.getTTparams(basis_info)
    basis_info = get_basis_info(tensor_basis)
    basis_info["lims"] = lims_extended
    tensor_basis, ranks = problem.getTTparams(basis_info)
    xTT_L2 = Extended_TensorTrain(tensor_basis, ranks, eps=0., comps=None, device=basis_info["device"])
    xTT_H1 = Extended_TensorTrain(tensor_basis, ranks, eps=0., comps=None, device=basis_info["device"])
    L2_UPDATE_LOSS = lambda surrogate: ALS_L2(X_train, Y_vals, iterations=100, tol=1e-4, reg_param_init=reg_init, xTT=surrogate, verbose=False, choose_SD=False, adaptive_reguarlization=True, verbose_adaptive_reg=False)
    H1_UPDATE_LOSS = lambda surrogate: ALS_H1(X_train, Y_vals, DY_vals, iterations=100, tol=1e-4, reg_param_init=reg_init, xTT=surrogate, verbose=False, choose_SD=False, adaptive_reguarlization=True, verbose_adaptive_reg=True)

    L2_loss, _, reg_param_updated_L2 = L2_UPDATE_LOSS(xTT_L2)

    H1_loss, _, reg_param_updated_H1 = H1_UPDATE_LOSS(xTT_H1)
    print("L2 loss = ", L2_loss, " H1 loss = ", H1_loss)
    print("regparams = ", reg_param_updated_L2, reg_param_updated_L2)

    if d == 1:
        x = torch.linspace(lims[0][0], lims[0][1], 500, device=device).reshape(-1, 1)
        vals_true = target(x)
        grads_true = Dtarget(x)

        vals_xtt_L2 = xTT_L2(x).reshape(-1, 1)
        vals_xtt_H1 = xTT_H1(x).reshape(-1, 1)

        grads_xtt_L2 = xTT_L2.grad(x).reshape(-1, 1)
        grads_xtt_H1 = xTT_H1.grad(x).reshape(-1, 1)

        plt.subplot(2, 2, 1)
        plt.plot(x, vals_true, label="True solution")
        plt.plot(x, vals_xtt_L2, label="Approx solution L2 fitting")
        plt.plot(x, vals_xtt_H1, label="Approx solution H1 fitting")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.semilogy(x, torch.abs(vals_true - vals_xtt_L2), label="Error solution L2 fitting")
        plt.semilogy(x, torch.abs(vals_true - vals_xtt_H1), label="Error solution H1 fitting")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(x, grads_true, label="True grad")
        plt.plot(x, grads_xtt_L2, label="Approx grad L2 fitting")
        plt.plot(x, grads_xtt_H1, label="Approx grad H1 fitting")
        plt.legend()

        coeff_error = torch.linalg.norm(xTT_L2.tt.comps[0] - xTT_H1.tt.comps[0])

        plt.subplot(2, 2, 4)
        plt.semilogy(x, torch.abs(grads_true - grads_xtt_L2), label="Error grad L2 fitting")
        plt.semilogy(x, torch.abs(grads_true - grads_xtt_H1), label="Error grad H1 fitting")
        plt.title("Coefficient distance = " + str(round(coeff_error.item(), 5)))
        plt.legend()

        print("coefficient distance : ", torch.linalg.norm(xTT_L2.tt.comps[0] - xTT_H1.tt.comps[0]))

        plt.show()

    if d == 2:
        # build a grid of points in the rectangle [lower_bound, upper_bound]
        n_grid = 200
        x1 = torch.linspace(lims[0][0], lims[0][1], n_grid)
        x2 = torch.linspace(lims[1][0], lims[1][1], n_grid)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        grid_points = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

        # compute the true value and approximation on the grid
        Y_true = problem.target(grid_points).reshape(n_grid, n_grid)

        exp_minus_V = torch.exp(-Y_true)
        Y_L2_approx = xTT_L2(grid_points).detach().reshape(n_grid, n_grid)
        Y_H1_approx = xTT_H1(grid_points).detach().reshape(n_grid, n_grid)

        DY_true = problem.target.grad(grid_points).reshape(n_grid, n_grid, d)
        DY_L2_approx = xTT_L2.grad(grid_points).detach().reshape(n_grid, n_grid, d)
        DY_H1_approx = xTT_H1.grad(grid_points).detach().reshape(n_grid, n_grid, d)

        # 3D plot: true function vs. approximation
        fig, axs = plt.subplots(3, 3, figsize=(12, 10))
        if plot_weighted_error:
            fig.suptitle("Approximation on extended domain by factor q = " + str(q) + " and evaluation on orginal domain - weighted Errors", fontsize=16)
        else:
            fig.suptitle("Approximation on extended domain by factor q = " + str(q) + " and evaluation on orginal domain - uniform Errors", fontsize=16)

        fig.delaxes(axs[1, 0])  # remove the old 2D axis
        fig.delaxes(axs[2, 0])  # remove the old 2D axis

        # true function surface
        fig.delaxes(axs[0, 0])  # remove the old 2D axis
        axs[0, 0] = fig.add_subplot(3, 3, 1, projection="3d")
        axs[0, 0].plot_surface(X1.numpy(), X2.numpy(), Y_true.numpy(),
                        cmap="viridis", alpha=0.8)
        axs[0, 0].set_title("True function")
        axs[0, 0].set_xlabel("x1")
        axs[0, 0].set_ylabel("x2")

        fig.delaxes(axs[0, 1])  # remove the old 2D axis
        axs[0, 1] = fig.add_subplot(3, 3, 2, projection="3d")
        axs[0, 1].plot_surface(X1.numpy(), X2.numpy(), Y_L2_approx.numpy(),
                        cmap="viridis", alpha=0.8)
        axs[0, 1].set_title("Approx function in L2")
        axs[0, 1].set_xlabel("x1")
        axs[0, 1].set_ylabel("x2")

        fig.delaxes(axs[0, 2])  # remove the old 2D axis
        axs[0, 2] = fig.add_subplot(3, 3, 3, projection="3d")
        axs[0, 2].plot_surface(X1.numpy(), X2.numpy(), Y_H1_approx.numpy(),
                        cmap="viridis", alpha=0.8)
        axs[0, 2].set_title("Approx function in H1")
        axs[0, 2].set_xlabel("x1")
        axs[0, 2].set_ylabel("x2")

        if plot_weighted_error:
            cf_val_L2 = axs[1, 1].contourf(X1.numpy(), X2.numpy(), (torch.abs(Y_true - Y_L2_approx) * exp_minus_V).numpy(),
                            cmap="viridis", alpha=0.8)
        else:
            cf_val_L2 = axs[1, 1].contourf(X1.numpy(), X2.numpy(), (torch.abs(Y_true - Y_L2_approx)).numpy(),
                            cmap="viridis", alpha=0.8)
        axs[1, 1].set_title("Approx error for fitting with ALS L2")
        axs[1, 1].set_xlabel("x1")
        axs[1, 1].set_ylabel("x2")
        axs[1, 1].set_aspect("equal")

        if plot_weighted_error:
            cf_grad_L2 = axs[2, 1].contourf(X1.numpy(), X2.numpy(), (torch.linalg.norm(DY_true - DY_L2_approx, dim=2) * exp_minus_V).numpy(),
                            cmap="viridis", alpha=0.8)
        else:
            cf_grad_L2 = axs[2, 1].contourf(X1.numpy(), X2.numpy(), (torch.linalg.norm(DY_true - DY_L2_approx, dim=2)).numpy(),
                            cmap="viridis", alpha=0.8)
        axs[2, 1].set_title("Grad Approx error for fitting with ALS L2")
        axs[2, 1].set_xlabel("x1")
        axs[2, 1].set_ylabel("x2")
        axs[2, 1].set_aspect("equal")

        if plot_weighted_error:
            cf_val_H1 = axs[1, 2].contourf(X1.numpy(), X2.numpy(), (torch.abs(Y_true - Y_H1_approx) * exp_minus_V).numpy(),
                            cmap="viridis", alpha=0.8)
        else:
            cf_val_H1 = axs[1, 2].contourf(X1.numpy(), X2.numpy(), (torch.abs(Y_true - Y_H1_approx) * exp_minus_V).numpy(),
                            cmap="viridis", alpha=0.8)
        axs[1, 2].set_title("Approx error for fitting with ALS H1")
        axs[1, 2].set_xlabel("x1")
        axs[1, 2].set_ylabel("x2")
        axs[1, 2].set_aspect("equal")

        if plot_weighted_error:
            cf_grad_H1 = axs[2, 2].contourf(X1.numpy(), X2.numpy(), (torch.linalg.norm(DY_true - DY_H1_approx, dim=2) * exp_minus_V).numpy(),
                            cmap="viridis", alpha=0.8)
        else:
            cf_grad_H1 = axs[2, 2].contourf(X1.numpy(), X2.numpy(), (torch.linalg.norm(DY_true - DY_H1_approx, dim=2)).numpy(),
                            cmap="viridis", alpha=0.8)
        axs[2, 2].set_title("Grad Approx error for fitting with ALS H1")
        axs[2, 2].set_xlabel("x1")
        axs[2, 2].set_ylabel("x2")
        axs[2, 2].set_aspect("equal")

        # shared colorbar for (1,1) and (1,2)
        fig.colorbar(cf_val_L2, ax=[axs[1, 1]], shrink=0.8, orientation="vertical")
        # shared colorbar for (2,1)
        fig.colorbar(cf_grad_L2, ax=[axs[2, 1]], shrink=0.8, orientation="vertical")

        # shared colorbar for (1,2)
        fig.colorbar(cf_val_H1, ax=[axs[1, 2]], shrink=0.8, orientation="vertical")
        # shared colorbar for (2,2)
        fig.colorbar(cf_grad_H1, ax=[axs[2, 2]], shrink=0.8, orientation="vertical")

        plt.show()

    exit()

    print(" with error ", l_post, " and rank = ", xTT.rank)


def fitting_of_funnel_via_samples(B=8000, timeSteps=4000, deg_x1=4, deg_x_rest=4,
                                no_iters=2, rank_modification_frequency=10, delta_rank_adapt=1e-4):
    d = 3

    device = "cpu"
    sample_mode = 0
    basis_choice = 2

    reg_init = 1e-8

    plot_initial_fit = False  # plots then exit

    fit_1D = False
    plot_1d_fittings = False

    analyse_ranks = False

    maximal_computational_domain = [[-6.5, 5.], [-4., 4.]]
    maximal_plot_domain = [[-10, 10.], [-15., 15.]]

    if basis_choice == 0:
        nknots_x1 = 3
        nknots_remain = 3
        deg_x1 = 6
        deg_remain = 2
        s = 1
        basis_info = {
            "type": "TensorSplineBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],  # does not matter
            "nknots": [nknots_x1] + [nknots_remain] * (d - 1),
            "deg": [deg_x1] + [deg_remain] * (d - 1),
            "s": [s] * d,
            "device": "cpu",
            "orthonormalize": "H2",
        }
        basis_info_x1 = {
            "type": "TensorSplineBasis",
            "lims": None,
            "nknots": [nknots_x1],
            "deg": [deg_x1],
            "orthonormalize": "H2",
            "s": [s],
            "device": device,
        }

        basis_info_x2 = {
            "type": "TensorSplineBasis",
            "nknots": [nknots_remain],
            "lims": None,
            "deg": [deg_remain],
            "orthonormalize": "H2",
            "s": [s],
            "device": device,
        }
    elif basis_choice == 1:
        include_linear = True
        include_quadratic = True

        num_trigonmetric = 8
        num_trigonmetric_remain = 8

        basis_info = {
            "type": "TensorExtendedFourierBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + num_trigonmetric]
            + [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + num_trigonmetric_remain] * (d - 1),
            "device": device,
            "orthonormalize": "H2",
            "include_linear": include_linear,
            "include_quadratic": include_quadratic,
        }

        basis_info_x1 = {
            "type": "TensorExtendedFourierBasis",
            "lims": None,
            "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + num_trigonmetric],
            "orthonormalize": "H2",
            "device": device,
            "include_linear": include_linear,
            "include_quadratic": include_quadratic,
        }

        basis_info_x2 = {
            "type": "TensorExtendedFourierBasis",
            "lims": None,
            "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + num_trigonmetric_remain],
            "orthonormalize": "H2",
            "device": device,
            "include_linear": include_linear,
            "include_quadratic": include_quadratic,
        }
    elif basis_choice == 2:
        basis_info = {
            "type": "TensorLegendreBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "deg": [deg_x1] + [deg_x_rest] * (d - 1),
            "orthonormalize": "H2",
            "device": "cpu",
        }

        basis_info_x1 = {
            "type": "TensorLegendreBasis",
            "lims": None,
            "deg": [deg_x1],
            "orthonormalize": "H2",
            "device": device,
        }

        basis_info_x2 = {
            "type": "TensorLegendreBasis",
            "lims": None,
            "deg": [deg_x_rest],
            "orthonormalize": "H2",
            "device": device,
        }

    problem = Funnel(d, basis_info, rank=2, timeSteps=timeSteps, batch_size=B, T=3, reg=0)

    if analyse_ranks:
        p_t = compute_score_by_legendre_quadrature(problem, nP=200, L=10)

        # build a grid of points in the rectangle [lower_bound, upper_bound]
        n_grid = 50
        lim = 4
        x1 = torch.linspace(-lim, lim, n_grid)
        x2 = torch.linspace(-lim, lim, n_grid)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        grid_points = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

        times = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.25, 0.75, 0.5, 1., 1.5, 2., 3., 4.]
        estimated_ranks = []

        plot_log_pt = True
        if plot_log_pt:
            fig = plt.figure(figsize=(6 * len(times), 5))

        for i_t, t in enumerate(times):
            if t == 0.:
                Y_p_t = torch.exp(-problem.target(grid_points)).reshape(n_grid, n_grid)
                log_Y_p_t = torch.log(Y_p_t)
            else:
                # compute the true value and approximation on the grid
                Y_p_t = p_t(t=torch.tensor(t), x=grid_points).reshape(n_grid, n_grid)
                log_Y_p_t = torch.log(Y_p_t)
            if plot_log_pt:
                # true function surface
                ax1 = fig.add_subplot(3, len(times), 1 + i_t)
                ax1.contourf(X1.numpy(), X2.numpy(), Y_p_t.numpy(),
                                cmap="viridis", alpha=0.8)

                ax1 = fig.add_subplot(3, len(times), 1 + i_t + len(times))
                ax1.contourf(X1.numpy(), X2.numpy(), log_Y_p_t.numpy(),
                                cmap="viridis", alpha=0.8)

            s_Phi_t = np.linalg.svd(np.array(log_Y_p_t), compute_uv=False)
            N = 10

            relative_sing_values = s_Phi_t[:N] / s_Phi_t[0]
            tol_sing = 1e-6
            for pos, rsing in enumerate(relative_sing_values):
                if rsing < tol_sing:
                    break
            if plot_log_pt:
                ax1 = fig.add_subplot(3, len(times), 1 + i_t + 2 * len(times))
                ax1.semilogy(range(1, N + 1), relative_sing_values, label=f"Phi (log p_t) at t={t}")
                ax1.scatter([pos + 1], relative_sing_values[pos], color="red", label=f"1st SV < {tol_sing}")
                ax1.legend()
                ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
                plt.grid(visible=True)

            estimated_ranks.append(pos)

            nan_mask = torch.isnan(log_Y_p_t)
            inf_mask = torch.isinf(log_Y_p_t)

            print("Number of NaNs:", nan_mask.sum().item())
            print("Number of Infs:", inf_mask.sum().item())

            # optional: indices of the first NaNs/Infs
            print("Indices NaNs:", nan_mask.nonzero(as_tuple=True))
            print("Indices Infs:", inf_mask.nonzero(as_tuple=True))
        if plot_log_pt:
            plt.tight_layout()
            plt.show()

        plt.plot(times, estimated_ranks, "ro-", label="estimated ranks")
        plt.grid(visible=True)
        plt.show()
        exit()

    myxFTT = xFTT(problem, None, p_gradient_extension=0.25, device=device, initial_policy=quadratic_annealed_Langevin)

    batch_size = problem.batch_size
    timeSteps = problem.timeSteps
    stepSizes = problem.stepSizes
    times = problem.times

    # we get the whole sample trajectory with the current drift f + sigma u
    # TODO: avoid returning sig_u, since it is only time-dependent
    training_data = simulate_SDE(problem, myxFTT.u, stepSizes, batch_size,
                                              simulation_device=myxFTT.device, computation_device=myxFTT.device)

    def masking_data(some_training_data, domain):
        a1, b1 = domain[0]
        a2, b2 = domain[1]
        X_u, xi, sig_u, u_val, tp = some_training_data
        mask_final_samples = (
            (X_u[..., 0] > a1) & (X_u[..., 0] < b1) &
            (X_u[..., 1] > a2) & (X_u[..., 1] < b2)
        ).all(dim=0)
        return X_u[:, mask_final_samples], xi[:, mask_final_samples], sig_u, u_val[:, mask_final_samples], tp

    training_data = masking_data(training_data, maximal_computational_domain)

    def ancestral_sample_funnel_numpy(num_samples=5000, D=10):
        num_x_dims = D - 1
        v = np.random.normal(loc=0, scale=3, size=num_samples)
        std_x = np.exp(v / 2)
        x = np.random.normal(loc=0, scale=std_x[:, np.newaxis], size=(num_samples, num_x_dims))
        samples = np.concatenate([v[:, np.newaxis], x], axis=1)
        return samples

    direct_samples = ancestral_sample_funnel_numpy(num_samples=B, D=d)

    X_u, _, _, _, _ = training_data

    initial_policy_samples = X_u[-1].clone()

    initial_policy_samples_mask = (initial_policy_samples[:, 0] > -10) & (initial_policy_samples[:, 0] < 10) & (initial_policy_samples[:, 1] > -10) & (initial_policy_samples[:, 1] < 10)
    direct_samples_mask = (direct_samples[:, 0] > -10) & (direct_samples[:, 0] < 10) & (direct_samples[:, 1] > -10) & (direct_samples[:, 1] < 10)
    if d < 3:
        plotSamples(direct_samples[direct_samples_mask], initial_policy_samples[initial_policy_samples_mask], label="_ref_init", bins=100, label1="reference samples", label2="initial policy")

    if sample_mode == 0:
        samples = X_u[-1]
    elif sample_mode == 1:
        samples = torch.tensor(direct_samples)
    else:
        lims = min_max_per_dim(X_u[-1])
        lower_bound = torch.tensor([lim[0] for lim in lims])
        upper_bound = torch.tensor([lim[1] for lim in lims])
        domain_width = torch.abs(upper_bound - lower_bound)
        q = 0.0
        lower_bound_extended = lower_bound - q * domain_width
        upper_bound_extended = upper_bound + q * domain_width
        # (1) uniform samples in [0,1]^d
        U = torch.rand(X_u[-1].shape)
        # (2) scale to [lower_bound, upper_bound] (broadcast)
        samples = lower_bound_extended + (upper_bound_extended - lower_bound_extended) * U

    if fit_1D:
        lims = min_max_per_dim(samples)
        basis_info_x1["lims"] = [lims[0]]
        basis_info_x2["lims"] = [lims[1]]
        lower_bound = torch.tensor([lim[0] for lim in lims])
        upper_bound = torch.tensor([lim[1] for lim in lims])

        domain_width = torch.abs(upper_bound - lower_bound)
        q = 0.0
        lower_bound_extended = lower_bound - q * domain_width
        upper_bound_extended = upper_bound + q * domain_width
        # (1) uniform samples in [0,1]^d
        U = torch.rand(B, d)
        # (2) scale to [lower_bound, upper_bound] (broadcast)
        X_uniform = lower_bound_extended + (upper_bound_extended - lower_bound_extended) * U

        x1_vals = torch.linspace(lower_bound[0], upper_bound[0], 100).reshape(-1, 1)
        x2_vals = torch.linspace(lower_bound[1], upper_bound[1], 100).reshape(-1, 1)

        # funnel area
        nu = 3.
        C_1_1 = lambda x1: x1 ** 2 / (2. * nu ** 2)
        C_1_2 = lambda x1: 1. / torch.exp(x1)

        C_2_1 = lambda x2: torch.ones_like(x2)
        C_2_2 = lambda x2: x2 ** 2 / 2.

        _, basis_x1 = get_Basis(basis_info_x1)
        _, basis_x2 = get_Basis(basis_info_x2)
        ranks_1d = handle_rank(rank=0, d=1)
        xtt_C_1_1 = Extended_TensorTrain(basis_x1, ranks_1d, eps=0, device=device)
        xtt_C_1_1.tt.set_core(0)
        xtt_C_1_2 = Extended_TensorTrain(basis_x1, ranks_1d, eps=0, device=device)
        xtt_C_1_2.tt.set_core(0)

        xtt_C_2_1 = Extended_TensorTrain(basis_x2, ranks_1d, eps=0, device=device)
        xtt_C_2_1.tt.set_core(0)
        xtt_C_2_2 = Extended_TensorTrain(basis_x2, ranks_1d, eps=0, device=device)
        xtt_C_2_2.tt.set_core(0)

        x1_uniform = X_uniform[:, 0].reshape(-1, 1)
        x2_uniform = X_uniform[:, 1].reshape(-1, 1)

        iterations = 100
        reg = 1e-6
        tol = 1e-6
        # run ALS
        L_post, k, new_reg_param = ALS_L2(x1_uniform, C_1_1(x1_uniform), iterations, tol, reg, xtt_C_1_1, verbose=True, choose_SD=False, adaptive_reguarlization=True)
        L_post, k, new_reg_param = ALS_L2(x1_uniform, C_1_2(x1_uniform), iterations, tol, reg, xtt_C_1_2, verbose=True, choose_SD=False, adaptive_reguarlization=True)
        L_post, k, new_reg_param = ALS_L2(x2_uniform, C_2_1(x2_uniform), iterations, tol, reg, xtt_C_2_1, verbose=True, choose_SD=False, adaptive_reguarlization=True)
        L_post, k, new_reg_param = ALS_L2(x2_uniform, C_2_2(x2_uniform), iterations, tol, reg, xtt_C_2_2, verbose=True, choose_SD=False, adaptive_reguarlization=True)

        relative_error = torch.linalg.norm(C_1_2(x1_uniform) - xtt_C_1_2(x1_uniform)) / torch.linalg.norm(C_1_2(x1_uniform))
        print("relativ error C_{12} :  ", relative_error)
        relative_error = torch.linalg.norm(C_2_2(x2_uniform) - xtt_C_2_2(x2_uniform)) / torch.linalg.norm(C_2_2(x2_uniform))
        print("relativ error C_{22} :  ", relative_error)

        if plot_1d_fittings:
            plt.subplot(2, 4, 1)
            plt.plot(x1_vals, C_1_1(x1_vals), label="C_1_1")
            plt.plot(x1_vals, xtt_C_1_1(x1_vals), "--", label="C_1_1 approximation")
            plt.legend()
            plt.subplot(2, 4, 5)
            plt.plot(x1_vals, torch.abs(C_1_1(x1_vals) - xtt_C_1_1(x1_vals)), label="C_1_1 error")
            plt.legend()

            plt.subplot(2, 4, 2)
            plt.plot(x1_vals, C_1_2(x1_vals), label="C_1_2")
            plt.plot(x1_vals, xtt_C_1_2(x1_vals), "--", label="C_1_2 approximation")
            plt.subplot(2, 4, 6)
            plt.plot(x1_vals, torch.abs(C_1_2(x1_vals) - xtt_C_1_2(x1_vals)), label="C_1_2 error")

            plt.legend()

            plt.subplot(2, 4, 3)
            plt.plot(x2_vals, C_2_1(x2_vals), label="C_2_1")
            plt.plot(x2_vals, xtt_C_2_1(x2_vals), "--", label="C_2_1 approximation")
            plt.legend()
            plt.subplot(2, 4, 7)
            plt.plot(x1_vals, torch.abs(C_2_1(x2_vals) - xtt_C_2_1(x2_vals)), label="C_2_1 error")
            plt.legend()

            plt.subplot(2, 4, 4)
            plt.plot(x2_vals, C_2_2(x2_vals), label="C_2_2")
            plt.plot(x2_vals, xtt_C_2_2(x2_vals), "--", label="C_2_2 approximation")
            plt.legend()
            plt.subplot(2, 4, 8)
            plt.plot(x2_vals, torch.abs(C_2_2(x2_vals) - xtt_C_2_2(x2_vals)), label="C_2_2 error")
            plt.legend()

            plt.show()

        comps_1st = torch.cat([xtt_C_1_1.tt.comps[0], xtt_C_1_2.tt.comps[0]], dim=2)
        comps_last = torch.cat([xtt_C_2_1.tt.comps[0], xtt_C_2_2.tt.comps[0]], dim=0)

        print("shape = ", comps_1st.shape)
        print("shape = ", comps_last.shape)

    # initial fitting via ALS L2
    Y_from_samples = problem.target(samples).reshape(-1, 1)
    lims = min_max_per_dim(samples)

    print("lims for ALS L2  : ", lims)
    tensor_basis, ranks = problem.getTTparams(basis_info)
    basis_info = get_basis_info(tensor_basis)
    basis_info["lims"] = lims
    tensor_basis, ranks = problem.getTTparams(basis_info)

    if fit_1D:
        xtt_from_1d_fittings = Extended_TensorTrain(tensor_basis, [1, 2, 1], eps=0., comps=[comps_1st, comps_last], device=basis_info["device"])
        xtt_from_1d_fittings.tt.set_core(0)
        print("debug xtt 1d fitting : ", xtt_from_1d_fittings.tensor_basis_functions.domain_bounds)

    xTT = Extended_TensorTrain(tensor_basis, ranks, eps=0., comps=None, device=basis_info["device"])
    L2_UPDATE_LOSS = lambda surrogate: ALS_L2(samples, Y_from_samples, iterations=100, tol=1e-4, reg_param_init=reg_init, xTT=surrogate, verbose=False, choose_SD=False, adaptive_reguarlization=True, verbose_adaptive_reg=False)

    l_post, _, reg_param_updated = L2_UPDATE_LOSS(xTT)

    print(" with error ", l_post, " and rank = ", xTT.rank)

    # update the regularization parameter proposed by ALS L2
    problem.reg = reg_param_updated

    if plot_initial_fit and d == 2:
        training_data_valid = simulate_SDE(problem, myxFTT.u, stepSizes, batch_size,
                                              simulation_device=myxFTT.device, computation_device=myxFTT.device)

        samples_validation = training_data_valid[0][-1]
        Y_from_valid_samples = problem.target(samples_validation).reshape(-1, 1)

        _, sings, v = torch.linalg.svd(xTT.tt.full())

        # evaluation grid
        n = 500
        x_vals = torch.linspace(lims[0][0], lims[0][1], n)
        y_vals = torch.linspace(lims[1][0], lims[1][1], n)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
        XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

        # contract
        Z_ref = problem.target(XY).reshape(X.shape)
        Z_xTT = xTT(XY).reshape(X.shape)

        # plot
        fig = plt.figure(figsize=(120, 6))

        gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1], wspace=0.95, hspace=0.45)

        ax = fig.add_subplot(gs[0, 0], projection="3d")

        ax.plot_surface(X.numpy(), Y.numpy(), Z_ref.detach().numpy(), cmap="viridis")
        ax.scatter(samples[:, 0], samples[:, 1], Y_from_samples.flatten(), marker=5, color="red")
        ax.set_title("True")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax = fig.add_subplot(gs[0, 1], projection="3d")
        ax.plot_surface(X.numpy(), Y.numpy(), Z_xTT.detach().numpy(), cmap="viridis")

        ax.set_title("Approx")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")

        ax = fig.add_subplot(gs[0, 2], projection="3d")
        ax.plot_surface(X.numpy(), Y.numpy(), torch.abs(Z_ref - Z_xTT).detach().numpy(), cmap="viridis")

        ax = fig.add_subplot(gs[0, 3], projection="3d")
        Y_xTT_on_samples = xTT(samples)
        ax.scatter(samples[:, 0], samples[:, 1], torch.abs(Y_from_samples - Y_xTT_on_samples).flatten(), marker=6, color="red")

        ax.set_title("Pointwise error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")

        ax = fig.add_subplot(gs[0, 4], projection="3d")
        Y_xTT_on_valid_samples = xTT(samples_validation)
        ax.scatter(samples_validation[:, 0], samples_validation[:, 1], torch.abs(Y_from_valid_samples - Y_xTT_on_valid_samples).flatten(), marker=6, color="red")

        ax.set_title("Pointwise error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")

        if fit_1D:
            Z_xTT_from_1d_fit = xtt_from_1d_fittings(XY).reshape(X.shape)

            # plot
            ax = fig.add_subplot(gs[1, 0], projection="3d")
            ax.plot_surface(X.numpy(), Y.numpy(), Z_ref.detach().numpy(), cmap="viridis")
            ax.set_title("True")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            ax = fig.add_subplot(gs[1, 1], projection="3d")
            ax.plot_surface(X.numpy(), Y.numpy(), Z_xTT_from_1d_fit.detach().numpy(), cmap="viridis")
            ax.set_title("Approx from 1D fittings")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f(x,y)")

            ax = fig.add_subplot(gs[1, 2], projection="3d")
            ax.plot_surface(X.numpy(), Y.numpy(), torch.abs(Z_ref - Z_xTT_from_1d_fit).detach().numpy(), cmap="viridis")
            ax.set_title("Pointwise error")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f(x,y)")

            ax = fig.add_subplot(gs[1, 3], projection="3d")
            xTT_fit1d_on_samples = xtt_from_1d_fittings(samples)
            ax.scatter(samples[:, 0], samples[:, 1], torch.abs(Y_from_samples - xTT_fit1d_on_samples).flatten(), marker=5, color="red")

            ax = fig.add_subplot(gs[1, 4], projection="3d")
            xTT_fit1d_on_valid_samples = xtt_from_1d_fittings(samples_validation)
            ax.scatter(samples[:, 0], samples[:, 1], torch.abs(Y_from_valid_samples - xTT_fit1d_on_valid_samples).flatten(), marker=6, color="red")

            ax.set_title("Pointwise error")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f(x,y)")

            _, sings, v = torch.linalg.svd(xtt_from_1d_fittings.tt.full())
            print(" singular values from 1d fitting  = ", sings)

            ax = fig.add_subplot(gs[2, 3:5])

            errors = torch.abs(Y_from_valid_samples - xTT_fit1d_on_valid_samples).flatten().numpy()
            # choose log-spaced bins
            bins = np.logspace(np.log10(errors.min()), np.log10(errors.max()), 50)
            ax.hist(errors, density=False, bins=bins, label="validation error,  1d fittings based")

            errors = torch.abs(Y_from_valid_samples - Y_xTT_on_valid_samples).flatten().numpy()
            # choose log-spaced bins
            bins = np.logspace(np.log10(errors.min()), np.log10(errors.max()), 50)
            ax.hist(errors, density=False, bins=bins, label="validation error,  ALS_L2 based")
            ax.set_xscale("log")
            plt.legend()

        plt.show()

        exit()

    # first iteration of the backward learning principle
    iteration = 0
    train(problem, myxFTT, iteration, training_data, tol=1e-3, verbose=False, plots=False, adaptive_reguarlization=True, initial_xTT=xTT, rank_modification_frequency=5, delta_rank_adapt=delta_rank_adapt)

    # flag to allow/disable linear_extension inside the training loop
    for iteration in range(0, no_iters):
        if isinstance(problem, GeneralizedProblem):
            training_data = simulate_SDE(problem, myxFTT.u, problem.stepSizes, problem.batch_size, simulation_device="cpu", computation_device="cpu")

            final_samples = training_data[0][-1]
            # restrict to the plotting domain
            plot_data = masking_data(training_data, maximal_plot_domain)
            training_data = masking_data(training_data, maximal_computational_domain)

            maximal_computational_domain
            final_samples = plot_data[0][-1]
            a1, b1 = maximal_plot_domain[0]
            a2, b2 = maximal_plot_domain[1]

            initial_policy_samples_mask = (initial_policy_samples[:, 0] > a1) & (initial_policy_samples[:, 0] < b1) & (initial_policy_samples[:, 1] > a2) & (initial_policy_samples[:, 1] < b2)
            direct_samples_mask = (direct_samples[:, 0] > a1) & (direct_samples[:, 0] < b1) & (direct_samples[:, 1] > a2) & (direct_samples[:, 1] < b2)

            plotSamples(final_samples, initial_policy_samples[initial_policy_samples_mask], label="1", bins=100, label1=f"{iteration+1}st outerloop samples", label2="initial policy")
            plotSamples(direct_samples[direct_samples_mask], initial_policy_samples[initial_policy_samples_mask], label="_ref_init", bins=100, label1="reference samples", label2="initial policy")
            plotSamples(direct_samples[direct_samples_mask], final_samples, label=f"_ref_outerloop{iteration+1}", bins=100, label1="reference samples", label2=f"{iteration+1}st outerloop samples")

            if iteration + 1 != no_iters:
                train(problem, myxFTT, iteration, training_data, verbose=False, plots=False, adaptive_reguarlization=True, initial_xTT=xTT, rank_modification_frequency=rank_modification_frequency, delta_rank_adapt=delta_rank_adapt)


def fitting_of_funnel():
    torch.manual_seed(0)
    d = 2
    N = 10000
    device = "cpu"
    include_linear = True
    include_quadratic = True
    basis_info = {
        "type": "TensorExtendedFourierBasis",
        "lims": [[-2.3, 2.3] for _ in range(d)],
        "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + 10] * d,
        "device": device,
        "orthonormalize": "H2",
        "include_linear": include_linear,
        "include_quadratic": include_quadratic,
    }

    problem = Funnel(d, basis_info, rank=2, timeSteps=1000, batch_size=1000, T=3, reg=0)

    myxFTT = xFTT(problem, None, p_gradient_extension=0.0, device=device)

    batch_size = problem.batch_size
    timeSteps = problem.timeSteps
    stepSizes = problem.stepSizes
    times = problem.times

    # we get the whole sample trajectory with the current drift f + sigma u
    # TODO: avoid returning sig_u, since it is only time-dependent
    X_u, xi, sig_u, u_val, tp = simulate_SDE(problem, myxFTT.u, stepSizes, batch_size,
                                              simulation_device=myxFTT.device, computation_device=myxFTT.device)

    X = X_u[-1]

    def ancestral_sample_funnel_numpy(num_samples=5000, D=10):
        num_x_dims = D - 1
        v = np.random.normal(loc=0, scale=3, size=num_samples)
        std_x = np.exp(v / 2)
        x = np.random.normal(loc=0, scale=std_x[:, np.newaxis], size=(num_samples, num_x_dims))
        samples = np.concatenate([v[:, np.newaxis], x], axis=1)
        return samples

    direct_samples = ancestral_sample_funnel_numpy(num_samples=5000, D=d)

    plot_samples = False
    if plot_samples:
        plt.figure(figsize=(8, 8))
        plt.plot(direct_samples[:, 0], direct_samples[:, 1], "ob", alpha=0.3, markersize=5, label="reference samples")
        plt.plot(X[:, 0], X[:, 1], "or", alpha=0.3, markersize=5, label="Langevin samples")

        plt.xlabel("$x_0$ (first dimension of x)")
        plt.ylabel("$v$ (variance-controlling variable)")
        plt.grid(True)

        plt.xlim(-15, 10)
        plt.ylim(-20, 20)
        plt.gca().set_aspect("equal", adjustable="box")
        exit()

    lims = min_max_per_dim(X)
    lower_bound = torch.tensor([lim[0] for lim in lims])
    upper_bound = torch.tensor([lim[1] for lim in lims])

    domain_width = torch.abs(upper_bound - lower_bound)
    q = 0.0

    lower_bound_extended = lower_bound - q * domain_width
    upper_bound_extended = upper_bound + q * domain_width

    # (1) uniform samples in [0,1]^d
    U = torch.rand(N, d)
    # (2) scale to [lower_bound, upper_bound] (broadcast)
    X_uniform = lower_bound_extended + (upper_bound_extended - lower_bound_extended) * U

    x1_vals = torch.linspace(lower_bound[0], upper_bound[0], 100).reshape(-1, 1)
    x2_vals = torch.linspace(lower_bound[1], upper_bound[1], 100).reshape(-1, 1)

    # funnel area
    nu = 3.
    C_1_1 = lambda x1: x1 ** 2 / (2 * nu ** 2)
    C_1_2 = lambda x1: 1. / torch.exp(x1)

    C_2_1 = lambda x2: torch.ones_like(x2)
    C_2_2 = lambda x2: x2 ** 2 / 2

    basis_info_x1 = {
        "type": "TensorLegendreBasis",
        "lims": [[lower_bound_extended[0], upper_bound_extended[0]] for _ in range(1)],
        "deg": [20],
        "orthonormalize": "H2",
        "device": device,
    }

    _, basis_x1 = get_Basis(basis_info_x1)
    ranks = handle_rank(rank=2, d=1)
    xtt_C_1_1 = Extended_TensorTrain(basis_x1, ranks, eps=1, device=device)
    xtt_C_1_1.tt.set_core(0)

    xtt_C_1_2 = Extended_TensorTrain(basis_x1, ranks, eps=1, device=device)
    xtt_C_1_2.tt.set_core(0)

    x1_uniform = X_uniform[:, 0].reshape(-1, 1)

    iterations = 100
    reg = 1e-12
    tol = 1e-6
    # run ALS
    L_post, k, new_reg_param = ALS_L2(x1_uniform, C_1_1(x1_uniform), iterations, tol, reg, xtt_C_1_1, verbose=True, choose_SD=False, adaptive_reguarlization=True)
    L_post, k, new_reg_param = ALS_L2(x1_uniform, C_1_2(x1_uniform), iterations, tol, reg, xtt_C_1_2, verbose=True, choose_SD=False, adaptive_reguarlization=True)

    relative_error = torch.linalg.norm(C_1_2(x1_uniform) - xtt_C_1_2(x1_uniform)) / torch.linalg.norm(C_1_2(x1_uniform))
    print("relativ error : ", relative_error)

    plt.subplot(2, 3, 1)
    plt.plot(x1_vals, C_1_1(x1_vals), label="C_1_1")
    plt.plot(x1_vals, xtt_C_1_1(x1_vals), "--", label="C_1_1 approximation")
    plt.legend()
    plt.subplot(2, 3, 4)
    plt.plot(x1_vals, torch.abs(C_1_1(x1_vals) - xtt_C_1_1(x1_vals)), label="C_1_1 error")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(x1_vals, C_1_2(x1_vals), label="C_1_2")
    plt.plot(x1_vals, xtt_C_1_2(x1_vals), "--", label="C_1_2 approximation")
    plt.subplot(2, 3, 5)
    plt.plot(x1_vals, torch.abs(C_1_2(x1_vals) - xtt_C_1_2(x1_vals)), label="C_1_2 error")

    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(x2_vals, C_2_1(x2_vals), label="C_2_1")
    plt.plot(x2_vals, C_2_2(x2_vals), label="C_2_2")
    plt.legend()

    plt.show()

    exit()

    # (1) uniform samples in [0,1]^d
    U = torch.rand(N, d)
    # (2) scale to [lower_bound, upper_bound] (broadcast)
    X_uniform = lower_bound + (upper_bound - lower_bound) * U
    X = X_uniform.clone()

    tensor_basis, ranks = problem.getTTparams()
    basis_info = get_basis_info(tensor_basis)
    basis_info["lims"] = lims
    tensor_basis, _ = problem.getTTparams(basis_info)
    comps = None

    Y = problem.target(X).reshape(-1, 1)
    tol = 1e-6
    iterations = 5000

    # fit the initial xTT
    xTT = Extended_TensorTrain(tensor_basis, ranks, eps=1., device=device)
    xTT.tt.set_core(0)  # set the central core
    # run ALS
    L_post, k = ALS_L2(X, Y, iterations, tol, reg, xTT, verbose=True, choose_SD=False)

    
def test_stack_based_grad_speed():
    device = "cpu"

    torch.manual_seed(0)
    d = 50
    N = 5000
    device = "cpu"
    include_linear = True
    include_quadratic = True
    basis_info = {
        "type": "TensorExtendedFourierBasis",
        "lims": [[-2.3, 2.3] for _ in range(d)],
        "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + 10] * d,
        "device": device,
        "orthonormalize": "H2",
        "include_linear": include_linear,
        "include_quadratic": include_quadratic,
    }

    domain, basis = get_Basis(basis_info)
    ranks = handle_rank(rank=2, d=d)

    xtt = Extended_TensorTrain(basis, ranks, eps=1, device=device)
    xtt.tt.set_core(0)

    lower_bound, upper_bound = xtt.tensor_basis_functions.domain_bounds
    # (1) uniform samples in [0,1]^d
    U = torch.rand(N, d)
    # (2) scale to [lower_bound, upper_bound] (broadcast)
    X = lower_bound + (upper_bound - lower_bound) * U

    start = time.perf_counter()
    D1 = xtt.grad(X, use_stacks=False)
    end = time.perf_counter()
    time_1 = end - start

    # --- fast, without einsum ---
    start = time.perf_counter()
    D2 = xtt.grad(X)
    end = time.perf_counter()
    time_2 = end - start

    # comparison
    print("########## gradient evaluation ##########")
    print(f"baseline: {time_1:.4f} sec")
    print(f"new   : {time_2:.4f} sec")
    print(f"Speed gain  baseline / new : {time_1/time_2:.2f}x faster")
    print("err = ", torch.linalg.norm(D1 - D2))

    start = time.perf_counter()
    H1 = xtt.hessian(X, use_stacks=False)
    end = time.perf_counter()
    time_1 = end - start

    start = time.perf_counter()
    H3 = xtt.hessian(X)
    end = time.perf_counter()
    time_3 = end - start

    # comparison
    print("########## Hessian evaluation ##########")
    print(f"baseline: {time_1:.4f} sec")
    print(f"Hessian double stack    : {time_3:.4f} sec")
    print(f"Speed gain  baseline / Hessian double stack : {time_1/time_3:.2f}x faster")
    print("err = ", torch.linalg.norm(H1 - H3))


def test_grad_speed():
    device = "cpu"

    dim = 5
    N = 100
    rank = 5

    grid_points = [torch.linspace(-2.5, 2.5, 7) for _ in range(dim)]
    p = [5] * dim
    s = [2] * dim

    device = "cpu"

    tensor_basis_functions = TensorSplineBasis(grid_points, p, s, device)
    ranks = [1] + [rank] * (dim - 1) + [1]
    xtt = Extended_TensorTrain(tensor_basis_functions, ranks, eps=1e-1, device=device)
    xtt.tt.set_core(0)

    # lower and upper bounds per dimension
    lower = torch.tensor([g[0] for g in grid_points])
    upper = torch.tensor([g[-1] for g in grid_points])

    # uniform sampling in each dimension
    x = lower + (upper - lower) * torch.rand(N, dim)

    # warm up
    xtt.grad_old(x)
    xtt.grad(x)

    # --- standard ---
    start = time.perf_counter()
    d_base = xtt.grad_old(x)
    end = time.perf_counter()
    time_baseline = end - start

    # --- fast einsum ---
    start = time.perf_counter()
    d_new = xtt.grad(x, use_einsum=True)
    end = time.perf_counter()
    time_fast = end - start

    # --- fast, without einsum ---
    start = time.perf_counter()
    d_new_2 = xtt.grad(x, use_einsum=False)
    end = time.perf_counter()
    time_fast_2 = end - start

    # comparison
    print(f"baseline: {time_baseline:.4f} sec")
    print(f"fast einsum version   : {time_fast:.4f} sec")
    print(f"fast  non einsum version   : {time_fast_2:.4f} sec")

    print(f"Speed gain  base / einsum : {time_baseline/time_fast:.2f}x faster")
    print(f"Speed gain  base / without einsum : {time_baseline/time_fast_2:.2f}x faster")
    print(f"Speed gain  einsum / without einsum : {time_fast/time_fast_2:.2f}x faster")

    print("err = ", torch.linalg.norm(d_base - d_new))


def test_batched_splineEvaluation_speed():
    device = "cpu"
    dim = 5
    N = 10000

    domain = [torch.tensor([-2.5 - 0.1 * torch.rand(1).item(), 2.5 + 0.1 * torch.rand(1).item()], device=device) for _ in range(dim)]

    nknots = [3] * dim
    grid_points = [torch.linspace(domain[i][0], domain[i][1], nknots[i], device=device) for i in range(dim)]

    p_list = [6] * dim
    s_list = [2] * dim

    basis_old = TensorSplineBasis(grid_points, p_list, s_list, device, orthonormalize="H2")
    basis_new = TensorSplineBasis_Equidistant(domain, nknots, p_list, s_list, device, orthonormalize="H2")

    # lower and upper bounds per dimension
    lower = torch.tensor([g[0] for g in grid_points], device=device)
    upper = torch.tensor([g[-1] for g in grid_points], device=device)

    # uniform sampling in each dimension
    x = lower + (upper - lower) * torch.rand(N, dim, device=device)

    # --- values, standard ---
    print("---- Values ----")
    start = time.perf_counter()
    basis_val_old = basis_old(x)
    end = time.perf_counter()
    time_baseline_val = end - start

    # --- fast ---
    start = time.perf_counter()
    basis_val_new = basis_new(x)
    end = time.perf_counter()
    time_fast_val = end - start

    print(f"basis evaluation old call: {time_baseline_val:.4f} sec")
    print(f"basis evaluation new parallel call : {time_fast_val:.4f} sec")
    print(f"Speed gain  base / with torch einsum : {time_baseline_val/time_fast_val:.2f}x faster")
    print("With error err = ", sum(torch.linalg.norm(basis_val_old[i] - basis_val_new[i]) for i in range(dim)))

    # --- gradient ---
    print("---- Gradients ----")
    # --- standard ---
    start = time.perf_counter()
    basis_grad_old = basis_old.grad(x)
    end = time.perf_counter()
    time_baseline_grad = end - start
    # --- fast ---
    start = time.perf_counter()
    basis_grad_new = basis_new.grad(x)
    end = time.perf_counter()
    time_fast_grad = end - start
    print(f"basis GRADIENT evaluation old call: {time_baseline_grad:.4f} sec")
    print(f"basis GRADIENT evaluation new parallel call : {time_fast_grad:.4f} sec")
    print(f"Speed gain  base / with torch einsum : {time_baseline_grad/time_fast_grad:.2f}x faster")
    print("With error err = ", sum(torch.linalg.norm(basis_grad_old[i] - basis_grad_new[i]) for i in range(dim)))

    # --- 2nd derivative ---
    print("---- 2nd Derivatives ----")
    # --- standard ---
    start = time.perf_counter()
    basis_d2_old = basis_old.D2(x)
    end = time.perf_counter()
    time_baseline_d2 = end - start
    # --- fast ---
    start = time.perf_counter()
    basis_d2_new = basis_new.D2(x)
    end = time.perf_counter()
    time_fast_d2 = end - start
    print(f"basis 2nd DERIVATIVE evaluation old call: {time_baseline_d2:.4f} sec")
    print(f"basis 2nd DERIVATIVE evaluation new parallel call : {time_fast_d2:.4f} sec")
    print(f"Speed gain  base / with torch einsum : {time_baseline_d2/time_fast_d2:.2f}x faster")
    print("With error err = ", sum(torch.linalg.norm(basis_d2_old[i] - basis_d2_new[i]) for i in range(dim)))


def test_batched_equidist_splineEvaluation():
    device = "cpu"
    dim = 3
    N = 5

    domain = [torch.tensor([-2.5, 2.5 + 0.1 * torch.rand(1).item()], device=device) for _ in range(dim)]

    nknots = [3] * dim
    grid_points = [torch.linspace(domain[i][0], domain[i][1], nknots[i]) for i in range(dim)]

    p_list = [6] * 2 + [4] * (dim - 2)
    s_list = [4] * 2 + [2] * (dim - 2)

    device = "cpu"

    basis_old = TensorSplineBasis(grid_points, p_list, s_list, device, orthonormalize="H2")
    basis_new = TensorSplineBasis_Equidistant(domain, nknots, p_list, s_list, device, orthonormalize="H2")

    # lower and upper bounds per dimension
    lower = torch.tensor([g[0] for g in grid_points])
    upper = torch.tensor([g[-1] for g in grid_points])

    # uniform sampling in each dimension
    x = lower + (upper - lower) * torch.rand(N, dim)

    basis_vals_old = basis_old(x)
    basis_vals_new = basis_new(x)

    for b_old, b_new in zip(basis_vals_old, basis_vals_new):
        print("err = ", torch.linalg.norm(b_old - b_new))

    print("---- Gradients ----")

    basis_grads_old = basis_old.grad(x)
    basis_grads_new = basis_new.grad(x)
    for g_old, g_new in zip(basis_grads_old, basis_grads_new):
        print("err grad = ", torch.linalg.norm(g_old - g_new))

    print("---- 2nd Derivatives ----")

    basis_d2_old = basis_old.D2(x)
    basis_d2_new = basis_new.D2(x)
    for d2_old, d2_new in zip(basis_d2_old, basis_d2_new):
        print("err d2 = ", torch.linalg.norm(d2_old - d2_new))

    exit()
   
def compare_accuracy_extended_fourier():
    torch.manual_seed(0)
    d = 1
    N = 1000
    device = "cpu"

    X = 1.41 * torch.randn(N, d)

    lims = min_max_per_dim(X)

    lower_bound = torch.tensor([lim[0] for lim in lims])
    upper_bound = torch.tensor([lim[1] for lim in lims])

    # (1) uniform samples in [0,1]^d
    U = torch.rand(N, d)
    # (2) scale to [lower_bound, upper_bound] (broadcast)
    X_uniform = lower_bound + (upper_bound - lower_bound) * U

    X = X_uniform.clone()

    def get_error_fit(problem):
        Y = problem.target(X_uniform).reshape(-1, 1)

        tensor_basis, ranks = problem.getTTparams()

        basis_info = get_basis_info(tensor_basis)
        basis_info["lims"] = lims
        tensor_basis, _ = problem.getTTparams(basis_info)
        comps = None
        tol = 1e-6
        iterations = 5000

        xTT = Extended_TensorTrain(tensor_basis, ranks, eps=1., device=device)
        xTT.tt.set_core(0)  # set the central core

        # run ALS to fit xTT
        L_post, k = ALS_L2(X, Y, iterations, tol, reg, xTT, verbose=True, choose_SD=False)

        Y = problem.target(X).reshape(-1, 1)
        print("Error on desired samples : ", 1. / N * torch.linalg.norm(xTT(X) - Y))
        return 1. / N * torch.linalg.norm(xTT(X) - Y)

    mean = torch.zeros(d)
    cov = torch.eye(d) * 2
    rank = 2
    reg = 0.

    err_without_linear = []
    err_with_linear = []

    n_fourier_list = [4, 8, 16, 24, 32, 48, 64, 128]

    for n_fourier in n_fourier_list:
        include_linear = False
        basis_info = {
            "type": "TensorExtendedFourierBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "n_basis": [1 + (1 if include_linear else 0) + n_fourier] * d,
            "device": "cpu",
            "orthonormalize": "H2",
            "include_linear": include_linear,
        }
        problem = Multiwell(d, n_double_wells=d, basis_info=basis_info, timeSteps=200, batch_size=5000, T=2., reg=1e-4)

        err_without_linear.append(get_error_fit(problem))

    for n_fourier in n_fourier_list:
        include_linear = True
        basis_info = {
            "type": "TensorExtendedFourierBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "n_basis": [1 + (1 if include_linear else 0) + n_fourier] * d,
            "device": "cpu",
            "orthonormalize": "H2",
            "include_linear": include_linear,
        }
        problem = Multiwell(d, n_double_wells=d, basis_info=basis_info, timeSteps=200, batch_size=5000, T=2., reg=1e-4)

        err_with_linear.append(get_error_fit(problem))

    plt.plot(n_fourier_list, err_with_linear, "-o", label="with linear")
    plt.plot(n_fourier_list, err_without_linear, "-o", label="without linear")
    plt.yscale("log")
    plt.xlabel("Number of additional Fourier basis functions ")
    plt.ylabel("Fitting error")
    plt.legend()
    plt.grid(visible=True, which="both")
    plt.title("Accuracy of Fourier basis fitting for 1D Gaussian problem")
    plt.show()

    print(err_with_linear)
    print(err_without_linear)


def test_fourier_basis_representation_change():
    torch.manual_seed(0)
    d = 1
    N = 1000
    device = "cpu"
    include_linear = True
    include_quadratic = True
    basis_info = {
        "type": "TensorExtendedFourierBasis",
        "lims": [[-2.3, 2.3] for _ in range(d)],
        "n_basis": [1 + (1 if include_linear else 0) + (1 if include_quadratic else 0) + 10] * d,
        "device": device,
        "orthonormalize": "H2",
        "include_linear": include_linear,
        "include_quadratic": include_quadratic,
    }

    reg = 0.

    problem = Multiwell(d, n_double_wells=d, basis_info=basis_info, timeSteps=200, batch_size=5000, T=2., reg=1e-4)

    myxFTT = xFTT(problem, None, p_gradient_extension=0.0, device=device)

    batch_size = problem.batch_size
    timeSteps = problem.timeSteps
    stepSizes = problem.stepSizes
    times = problem.times
    # we get the whole sample trajectory with the current drift f + sigma u
    # TODO: avoid returning sig_u, since it is only time-dependent
    X_u, xi, sig_u, u_val, tp = simulate_SDE(problem, myxFTT.u, stepSizes, batch_size, device=myxFTT.device)

    X = X_u[-1]
    lims = min_max_per_dim(X)

    lower_bound = torch.tensor([lim[0] for lim in lims])
    upper_bound = torch.tensor([lim[1] for lim in lims])

    # (1) uniform samples in [0,1]^d
    U = torch.rand(N, d)
    # (2) scale to [lower_bound, upper_bound] (broadcast)
    X_uniform = lower_bound + (upper_bound - lower_bound) * U
    X = X_uniform.clone()

    tensor_basis, ranks = problem.getTTparams()
    basis_info = get_basis_info(tensor_basis)
    basis_info["lims"] = lims
    tensor_basis, _ = problem.getTTparams(basis_info)
    comps = None

    Y = problem.target(X).reshape(-1, 1)
    tol = 1e-6
    iterations = 5000

    # fit the initial xTT
    xTT = Extended_TensorTrain(tensor_basis, ranks, eps=1., device=device)
    xTT.tt.set_core(0)  # set the central core
    # run ALS
    L_post, k = ALS_L2(X, Y, iterations, tol, reg, xTT, verbose=True, choose_SD=False)

    # get the next initial candidate:
    previous_xTT = xTT
    X_next = X_u[-2]

    lower_bound_next = torch.tensor([lim[0] for lim in lims])
    upper_bound_next = torch.tensor([lim[1] for lim in lims])
    lims = min_max_per_dim(X_next)
    basis_info = get_basis_info(previous_xTT.tensor_basis_functions)
    basis_info["lims"] = lims
    tensor_basis, ranks = problem.getTTparams(basis_info)
    xTT_next = Extended_TensorTrain(tensor_basis, ranks, comps=None, device=device)

    projected_presentation_change(previous_xTT, xTT_next, "H2", n_quadrature=100)

    if d == 1:

        x_vals = torch.linspace(lower_bound[0], upper_bound[0], 200).reshape(-1, 1)
        y_true = problem.target(x_vals)
        y_approx = xTT(x_vals)

        x_vals_next = torch.linspace(lower_bound_next[0], upper_bound_next[0], 200).reshape(-1, 1)
        y_next = xTT_next(x_vals_next)

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals.numpy(), y_true.numpy(), label="True function", lw=2)
        plt.plot(x_vals.numpy(), y_approx.detach().numpy(), "--", label="TT Approximation", lw=2)
        plt.plot(x_vals_next.numpy(), y_next.detach().numpy(), "--", label="TT next init", lw=2)
        plt.title("Function Approximation with Tensor Train Spline")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()


def test_multivariate_spline_fitting():
    torch.manual_seed(0)
    d = 1
    N = 1000
    device = "cpu"
    basis_choice = 0

    if basis_choice == 0:
        basis_info = {
            "type": "TensorSplineBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],  # does not matter
            "nknots": [3] * d,
            "deg": [4] * d,
            "s": [1] * d,
            "device": device,
            "orthonormalize": None,
        }
    elif basis_choice == 1:
        basis_info = {
            "type": "TensorLegendreBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "deg": [4] * d,
            "device": device,
        }
    elif basis_choice == 2:
        basis_info = {
            "type": "TensorSplineBasis_Equidistant",
            "device": device,
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "nknots": [3] * d,
            "deg": [2] * d,
            "s": [1] * d,
            "orthonormalize": "L2",
        }
    elif basis_choice == 3:
        include_linear = True
        basis_info = {
            "type": "TensorExtendedFourierBasis",
            "lims": [[-2.3, 2.3] for _ in range(d)],
            "n_basis": [1 + (1 if include_linear else 0) + 128] * d,
            "device": "cpu",
            "orthonormalize": "H2",
            "include_linear": include_linear,
        }

    mean = torch.zeros(d)
    cov = torch.eye(d) * 2
    rank = 2

    reg = 0.

    problem = Multiwell(d, n_double_wells=d, basis_info=basis_info, timeSteps=200, batch_size=5000, T=2., reg=1e-4)

    myxFTT = xFTT(problem, None, p_gradient_extension=0.0, device=device)

    batch_size = problem.batch_size
    timeSteps = problem.timeSteps
    stepSizes = problem.stepSizes
    times = problem.times

    X = 0.75 * 1.41 * torch.randn(N, d)

    lims = min_max_per_dim(X)

    lower_bound = torch.tensor([lim[0] for lim in lims])
    upper_bound = torch.tensor([lim[1] for lim in lims])

    # (1) uniform samples in [0,1]^d
    U = torch.rand(N, d)
    # (2) scale to [lower_bound, upper_bound] (broadcast)
    X_uniform = lower_bound + (upper_bound - lower_bound) * U

    if d == 1:
        positive_mask_1d = X_uniform[:, 0] >= -0.02
        X_uniform = X_uniform[positive_mask_1d]
        X_uniform[0] = -1.5

    print("X_uniform shape ", X_uniform.shape)

    tensor_basis, ranks = problem.getTTparams()

    basis_info = get_basis_info(tensor_basis)
    basis_info["lims"] = lims
    tensor_basis, _ = problem.getTTparams(basis_info)
    comps = None

    Y_uniform = problem.target(X_uniform).reshape(-1, 1)

    print("Y uniform shape = ", Y_uniform.shape)
    tol = 1e-6
    iterations = 5000

    Y = Y_uniform

    # fit the initial xTT
    xTT = Extended_TensorTrain(tensor_basis, ranks, eps=1., device=device)
    xTT.tt.set_core(0)  # set the central core
    # run ALS
    L_post, k, reg_param = ALS_L2(X_uniform, Y, iterations, tol, reg, xTT, verbose=True, choose_SD=False)

    if d == 1:

        x_vals = torch.linspace(lower_bound[0], upper_bound[0], 200).reshape(-1, 1)
        y_true = problem.target(x_vals)
        y_approx = xTT(x_vals)

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals.numpy(), y_true.numpy(), label="True function", lw=2)
        plt.plot(x_vals.numpy(), y_approx.detach().numpy(), "--", label="Approximation with splines", lw=2)
        plt.scatter(X_uniform.numpy(), Y_uniform.numpy(), color="red", s=20, label="Training data")
        plt.title("Function Approximation with Spline basis functions")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.ylim(-2, 20)
        plt.xlim(-2.5, 2.5)
        plt.legend()
        plt.grid(True)
        plt.savefig("failed_spline_fitting.pdf")

    if d == 2:

        # build a grid of points in the rectangle [lower_bound, upper_bound]
        n_grid = 50
        x1 = torch.linspace(lower_bound[0], upper_bound[0], n_grid)
        x2 = torch.linspace(lower_bound[1], upper_bound[1], n_grid)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        grid_points = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)

        # compute the true value and approximation on the grid
        Y_true = problem.target(grid_points).reshape(n_grid, n_grid)
        Y_approx = xTT(grid_points).detach().reshape(n_grid, n_grid)

        # 3D plot: true function vs. approximation
        fig = plt.figure(figsize=(12, 5))

        # true function surface
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(X1.numpy(), X2.numpy(), Y_true.numpy(),
                        cmap="viridis", alpha=0.8)
        ax1.scatter(X_uniform[:, 0].numpy(), X_uniform[:, 1].numpy(),
                   0 * Y_uniform.numpy(), color="red", s=15, label="Training data")
        ax1.set_title("True function with training data")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("f(x)")

        # approximation surface
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(X1.numpy(), X2.numpy(), Y_approx.numpy(),
                        cmap="cividis", alpha=0.8)
        ax2.scatter(X_uniform[:, 0].numpy(), X_uniform[:, 1].numpy(),
                   Y_uniform[:, 0].numpy(), color="red", s=15, label="Training data")
        ax2.set_title("TT Approximation with training data")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.set_zlabel("f(x)")

        plt.tight_layout()
        plt.show()


def test_save_and_load_xTT():

    domain = [[-2., 3], [-1., 4.], [1., 2.1]]

    basis_choice = "TensorExtendedFourierBasis"  # or "TensorLegendreBasis"

    if basis_choice == "TensorLegendreBasis":
        deg_list = [5, 4, 3]
        tensor_basis = TensorLegendreBasis(domain, deg_list, orthonormalize="H2", device="cpu")
    elif basis_choice == "TensorExtendedFourierBasis":
        tensor_basis = TensorExtendedFourierBasis(domain, [6, 4, 5], orthonormalize="H2")

    xTT = Extended_TensorTrain(tensor_basis, ranks=[1, 2, 3, 1], eps=1, device="cpu")
    xTT.tt.set_core(0)
    xTT.save("myxTT.pt", to_cpu=True)

    xTT_loaded = Extended_TensorTrain.load("myxTT.pt", map_location="cpu", set_core_position_to=0)

    sample = torch.randn(10, len(domain))

    print(torch.linalg.norm(xTT(sample) - xTT_loaded(sample)))


if __name__ == "__main__":

    test_fitting_with_ALS_H1_loss()

    exit()

    test_multivariate_spline_fitting()
    exit()

    # test_save_and_load_xTT()

    # exit()

    fitting_of_funnel_via_samples()

    exit()

    fitting_of_funnel()

    exit()

    basis_representation_change_Legendre()
    exit()

    fitting_of_funnel()

    exit()

    test_stack_based_grad_speed()
    
    exit()

    test_fourier_basis_representation_change()

    exit()

    test_multivariate_spline_fitting()
    exit()

    compare_accuracy_extended_fourier()

    exit()

    test_multivariate_spline_fitting()
    exit()

    test_batched_splineEvaluation_speed()

    exit()

    test_batched_equidist_splineEvaluation()

    exit()

    test_derivative_spline()

    exit()

    test_equiv_splines() 

    exit()
    test_eval_xtt()
    exit()
    test_grad_speed()
    exit()

    test_spline_als()

    exit()
    
    #test_1d()
    test_2d()
