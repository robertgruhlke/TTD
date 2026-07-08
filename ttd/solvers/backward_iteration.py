"""Backward-in-time policy iteration.

Fits the value function V_n on a decreasing time grid via ALS regression against
simulated trajectories, then derives the control u_n = -sigma^T grad V_n from it.
`train()` is the live per-sweep entry point used by run_TTD.py; `train_adaptive_basis()`
and `run_backward_iteration()` are kept for reference but unused by the current example.
"""

import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore, Style

from ttd.bases.rank_rules import Relative_Singularvalue_Tresholding
from ttd.problems.base import GeneralizedProblem, get_basis_info
from ttd.problems.concrete import GaussianProblem, Multiwell
from ttd.sde import simulate_SDE
from ttd.solvers.als import (
    ALS_L2,
    ALS_GeneralBasis_fast_vectorized,
    optimize_and_choose_proper_basis_and_rank,
    optimize_and_choose_proper_rank,
)
from ttd.tt.extended import Extended_TensorTrain, projected_presentation_change
from ttd.utils.plotting import plotSamples
from ttd.xftt import xFTT


def get_h_term(xFTT_t, f, u_val, X, time):
    """
    This is a helper variable to define the term h from the manuscript
    {h}^u_n = \left(\div (f) + \frac{1}{2} \big\|\sigma^\top \nabla \widehat{V}_n\big\|^2 +  u \cdot \sigma^\top \nabla \widehat{V}_n\right)(\widehat{X}^u_n, t_n).
    """
    xFTT_t_u = xFTT_t.u(X)
    h = f.div(X, time) + 0.5 * torch.linalg.norm(-xFTT_t_u, axis=1) ** 2 + torch.einsum("bi, bi -> b", u_val, -xFTT_t_u)
    return h.reshape(-1, 1)


def min_max_per_dim(X):
    """
    Compute per-dimension min and max from a 2D torch tensor.

    Args:
        X (torch.Tensor): shape (b, dim)

    Returns:
        list of [min_i, max_i] for each dimension i
    """
    mins = torch.amin(X, dim=0)
    maxs = torch.amax(X, dim=0)
    return [[mins[i].item(), maxs[i].item()] for i in range(X.shape[1])]


def fit_initial_value_function(problem, dim, training_data, reg, K_unif, tol, device="cpu"):
    """
    Fit an initial value-function surrogate via ALS_L2 on samples drawn uniformly over
    the bounding box of the terminal-time training samples, resetting the RNG first.
    """
    torch.manual_seed(0)

    X_T = training_data[0][-1].to(device)
    lims = min_max_per_dim(X_T)
    lower_bound = torch.tensor([lim[0] for lim in lims], device=device)
    upper_bound = torch.tensor([lim[1] for lim in lims], device=device)
    X_samples = lower_bound + (upper_bound - lower_bound) * torch.rand(K_unif, dim, device=device)
    Y_samples = problem.target(X_samples).reshape(-1, 1)

    tensor_basis, ranks = problem.getTTparams()
    basis_info = get_basis_info(tensor_basis)
    basis_info["lims"] = lims
    tensor_basis, _ = problem.getTTparams(basis_info)

    xTT = Extended_TensorTrain(tensor_basis, ranks, eps=0., device=device)
    xTT.tt.set_core(0)
    _, _, reg_param_updated = ALS_L2(
        X_samples, Y_samples, iterations=1000, tol=tol, reg_param_init=reg, xTT=xTT,
        verbose=False, choose_SD=False, adaptive_reguarlization=True, verbose_adaptive_reg=False,
    )
    return xTT, reg_param_updated


def run_backward_iteration(problem, no_iters=1, plots=False):
    plot_finalsamples = True

    # linear_extension could also be Dictionary(basis_list=[problem.target]) or
    # Dictionary(basis_list=[problem.start, problem.target]); plain None works best so far.
    linear_extension = None

    myxFTT = xFTT(problem, linear_extension, p_gradient_extension=0.1)
    training_data = simulate_SDE(problem, myxFTT.u, problem.stepSizes, problem.batch_size, simulation_device="cpu", computation_device="cpu")

    reg_init = 1e-8
    X_u, _, _, _, _ = training_data

    # initial fitting via ALS_L2 on uniform samples over the empirical bounding box
    X = X_u[-1]
    lims = min_max_per_dim(X)
    lower_bound = torch.tensor([lim[0] for lim in lims])
    upper_bound = torch.tensor([lim[1] for lim in lims])
    domain_width = torch.abs(upper_bound - lower_bound)
    q = 0.0
    lower_bound_extended = lower_bound - q * domain_width
    upper_bound_extended = upper_bound + q * domain_width

    # uniform samples in [0,1]^d, rescaled to [lower_bound, upper_bound]
    U = torch.rand(X.shape)
    X = lower_bound_extended + (upper_bound_extended - lower_bound_extended) * U

    Y = problem.target(X).reshape(-1, 1)

    tensor_basis, ranks = problem.getTTparams()
    lims = min_max_per_dim(X)
    basis_info = get_basis_info(tensor_basis)
    basis_info["lims"] = lims
    tensor_basis, _ = problem.getTTparams(basis_info)
    xTT = Extended_TensorTrain(tensor_basis, ranks, eps=0., comps=None, device=basis_info["device"])

    _, _, reg_param_updated = ALS_L2(X, Y, iterations=100, tol=1e-2, reg_param_init=reg_init, xTT=xTT, verbose=False, choose_SD=False, adaptive_reguarlization=True, verbose_adaptive_reg=True)

    # update the regularization parameter proposed by ALS_L2
    problem.reg = reg_param_updated

    # first iteration of the backward learning principle
    iteration = 0
    train(problem, myxFTT, iteration, training_data, verbose=False, plots=plots, adaptive_reguarlization=True, initial_xTT=xTT)

    for iteration in range(1, no_iters):
        if isinstance(problem, GeneralizedProblem):
            training_data = simulate_SDE(problem, myxFTT.u, problem.stepSizes, problem.batch_size, simulation_device="cpu", computation_device="cpu")
            train(problem, myxFTT, iteration, training_data, verbose=False, plots=plots, adaptive_reguarlization=True)

    test_batch = 5000
    X_u, _, _, _, _ = simulate_SDE(problem, myxFTT.u, problem.stepSizes, test_batch)

    try:
        reference_samples = problem.target.sample(test_batch)
    except NotImplementedError:
        reference_samples = None
    if problem.dim == 1 or problem.dim == 2 and plot_finalsamples:
        plotSamples(X_u[problem.timeSteps], reference_samples, label="final", bins=100)


def build_xTT(t, problem, X, previous_xTT=None, verbose=False, device="cpu"):
    if previous_xTT is None:
        tensor_basis, ranks = problem.getTTparams()
        lims = min_max_per_dim(X)
        basis_info = get_basis_info(tensor_basis)
        basis_info["lims"] = lims
        tensor_basis, _ = problem.getTTparams(basis_info)
        xTT = Extended_TensorTrain(tensor_basis, ranks, eps=1e-4, comps=None, device=device)
    else:
        lims = min_max_per_dim(X)
        lower_bound = torch.tensor([lim[0] for lim in lims])
        upper_bound = torch.tensor([lim[1] for lim in lims])

        domain_width = torch.abs(upper_bound - lower_bound)
        q = 0.0

        lower_bound_extended = lower_bound - q * domain_width
        upper_bound_extended = upper_bound + q * domain_width

        lims = [[lower_bound_extended[i].item(), upper_bound_extended[i].item()] for i in range(len(lims))]

        basis_info = get_basis_info(previous_xTT.tensor_basis_functions)
        basis_info["lims"] = lims
        tensor_basis, ranks = problem.getTTparams(basis_info)
        xTT = Extended_TensorTrain(tensor_basis, previous_xTT.tt.rank, comps=None, device=device)

        # make basis representation change as initial fit between neighboured xTT
        projected_presentation_change(previous_xTT, xTT, "H2", n_quadrature=100)

    return xTT


def train_adaptive_basis(problem, xFTT, iteration, training_data, verbose=False, plots=False, initial_xTT=None, tol=1e-4, loss_atol=1e-4, adaptive_reguarlization=False, rank_modification_frequency=0, delta_rank_adapt=None, n_start=None, n_end=None):
    N = problem.timeSteps
    stepSizes = problem.stepSizes
    times = problem.times

    if n_start is None:
        n_start = 0
    if n_end is None:
        n_end = N

    # We get the whole sample trajectory with the current drift f + sigma u
    # TODO: avoid return of sig_u as it is only time dependent
    X_u, xi, sig_u, u_val, tp = training_data

    if plots:
        plt.subplot(1, 2, 1)
        plt.plot(tp.cpu(), X_u[:, :, 0].cpu())
        plt.subplot(1, 2, 2)
        finalSamples = X_u[N].cpu()
        targetSamples = None
        plotSamples(finalSamples, targetSamples, label=iteration)

    # set xFTT_T.V = log p_target and xFTT_T.u = sigma^T nabla log target
    if n_end == N:
        # init reg_val, TODO: make reg_val input of training not of problem
        xFTT.reg_value[N - 1] = problem.reg_params(times[N - 1])
        xFTT.update(N, problem.target,
                    torch.zeros(xFTT.linear_extension.ndofs, device=xFTT.device) if xFTT.linear_extension is not None else 0.)
        # initialize learned or random TensorTrain at last timestep
        xTT = build_xTT(times[N - 1], problem, X=X_u[n_end - 1 - n_start].to(xFTT.device), previous_xTT=initial_xTT, verbose=verbose, device=xFTT.device)
        xTT.tt.set_core(0)
        xFTT.update(N - 1, xTT, xFTT.xFTT_list[N].c_linear)

    for n in range(n_end - 1, n_start - 1, -1):
        train_time_start = time.time()
        print("n = %d " % n)

        h_n_plus_1 = get_h_term(xFTT.xFTT_list[n + 1], problem.f, u_val[n + 1 - n_start].to(xFTT.device), X_u[n + 1 - n_start].to(xFTT.device), tp[n + 1 - n_start])
        times[n] = times[n + 1] - stepSizes[n]

        if n < N - 1:
            xTT = build_xTT(times[n], problem, X_u[n - n_start].to(xFTT.device), previous_xTT=xFTT.xFTT_list[n + 1].xTT, verbose=False, device=xFTT.device)
            xFTT.update(n, xTT, xFTT.xFTT_list[n + 1].c_linear)

        y = -h_n_plus_1 * stepSizes[n] + xFTT.xFTT_list[n + 1].V(X_u[n + 1 - n_start].to(xFTT.device))
        Sigma = sqrt(stepSizes[n]) * torch.einsum("bi,ji->bj", xi[n + 1 - n_start].to(xFTT.device), sig_u[n - n_start].to(xFTT.device))

        ALS_SOLVER = ALS_GeneralBasis_fast_vectorized

        MY_UPDATE_LOSS = lambda surrogate: ALS_SOLVER(
            X_u[n - n_start].to(xFTT.device), y, Sigma, surrogate,
            iterations=500, tol=tol, reg_param_init=xFTT.reg_value[n], verbose=False, adaptive_reguarlization=adaptive_reguarlization, verbose_adaptive_reg=False
        )

        xFTT.xFTT_list[n], curr_res, k_needed1, reg_value_new, loss_values = \
            optimize_and_choose_proper_basis_and_rank(xFTT.xFTT_list[n], MY_UPDATE_LOSS, delta=delta_rank_adapt, loss_atol=loss_atol, verbose_level=verbose, number_basis_increasements=10)

        # check singular values:
        xFTT.logs["singular_values"].append([])

        for pos in range(xFTT.xFTT_list[n].xTT.tt.n_comps - 1):
            xFTT.xFTT_list[n].xTT.tt.set_core(pos)
            c = xFTT.xFTT_list[n].xTT.tt.comps[pos]
            s = c.shape
            c = c.reshape(s[0] * s[1], s[2])
            u, sigma, v = torch.linalg.svd(c, full_matrices=False)
            xFTT.logs["singular_values"][-1].append(sigma.to("cpu").tolist())
        xFTT.xFTT_list[n].xTT.tt.set_core(0)

        train_time_end = time.time()

        xFTT.logs["loss"].append(curr_res)
        xFTT.logs["sweeps"].append(loss_values)
        xFTT.logs["time"].append(train_time_end - train_time_start)

        if adaptive_reguarlization:
            xFTT.reg_value[n - 1] = reg_value_new
        else:
            xFTT.reg_value[n - 1] = xFTT.reg_value[n]

        # TODO: optional rank truncation of the xTT subobject of xFTT.xFTT_list[n]

        if verbose:
            print("At time {n}: ALS algebraic residual error: {c}{i}{rep} with rank = {r} and fdims = {f}".format(n=n, c=Fore.RED, i=curr_res, rep=Style.RESET_ALL, r=xFTT.xFTT_list[n].xTT.tt.rank, f=xFTT.xFTT_list[n].xTT.tt.dims))
            print("Expected runtime: %.2f min" % (np.mean(xFTT.logs["time"]) * n / 60))


def train(problem, xFTT, iteration, training_data, verbose=False, plots=False, initial_xTT=None, tol=1e-4, adaptive_reguarlization=False, rank_modification_frequency=0, delta_rank_adapt=None, n_start=None, n_end=None, delta_rank_short=0., rank_adapt_verbose=False):
    N = problem.timeSteps
    stepSizes = problem.stepSizes
    times = problem.times

    if n_start is None:
        n_start = 0
    if n_end is None:
        n_end = N

    # We get the whole sample trajectory with the current drift f + sigma u
    # TODO: avoid return of sig_u as it is only time dependent
    X_u, xi, sig_u, u_val, tp = training_data

    if plots:
        plt.subplot(1, 2, 1)
        plt.plot(tp.cpu(), X_u[:, :, 0].cpu())
        plt.subplot(1, 2, 2)
        finalSamples = X_u[N].cpu()
        targetSamples = None
        plotSamples(finalSamples, targetSamples, label=iteration)

    # set xFTT_T.V = log p_target and xFTT_T.u = sigma^T nabla log target
    if n_end == N:
        # init reg_val, TODO: make reg_val input of training not of problem
        xFTT.reg_value[N - 1] = problem.reg_params(times[N - 1])
        xFTT.update(N, problem.target,
                    torch.zeros(xFTT.linear_extension.ndofs, device=xFTT.device) if xFTT.linear_extension is not None else 0.)
        # initialize learned or random TensorTrain at last timestep
        xTT = build_xTT(times[N - 1], problem, X=X_u[n_end - 1 - n_start].to(xFTT.device), previous_xTT=initial_xTT, verbose=verbose, device=xFTT.device)
        xTT.tt.set_core(0)
        xFTT.update(N - 1, xTT, xFTT.xFTT_list[N].c_linear)

    iterations_without_rank_checkup = 0

    for n in range(n_end - 1, n_start - 1, -1):
        train_time_start = time.time()
        print("n = %d " % n)

        h_n_plus_1 = get_h_term(xFTT.xFTT_list[n + 1], problem.f, u_val[n + 1 - n_start].to(xFTT.device), X_u[n + 1 - n_start].to(xFTT.device), tp[n + 1 - n_start])
        times[n] = times[n + 1] - stepSizes[n]

        if n < N - 1:
            xTT = build_xTT(times[n], problem, X_u[n - n_start].to(xFTT.device), previous_xTT=xFTT.xFTT_list[n + 1].xTT, verbose=False, device=xFTT.device)
            xFTT.update(n, xTT, xFTT.xFTT_list[n + 1].c_linear)

        y = -h_n_plus_1 * stepSizes[n] + xFTT.xFTT_list[n + 1].V(X_u[n + 1 - n_start].to(xFTT.device))
        Sigma = sqrt(stepSizes[n]) * torch.einsum("bi,ji->bj", xi[n + 1 - n_start].to(xFTT.device), sig_u[n - n_start].to(xFTT.device))

        ALS_SOLVER = ALS_GeneralBasis_fast_vectorized

        MY_UPDATE_LOSS = lambda surrogate: ALS_SOLVER(
            X_u[n - n_start].to(xFTT.device), y, Sigma, surrogate,
            iterations=500, tol=tol, reg_param_init=xFTT.reg_value[n], verbose=verbose, adaptive_reguarlization=adaptive_reguarlization, verbose_adaptive_reg=False
        )

        if rank_modification_frequency > 0 and iterations_without_rank_checkup == rank_modification_frequency - 1:
            assert xFTT.xFTT_list[n].linear_extension is None
            assert delta_rank_adapt is not None
            curr_res, k_needed1, reg_value_new, loss_values = optimize_and_choose_proper_rank(xFTT.xFTT_list[n], MY_UPDATE_LOSS, delta=delta_rank_adapt, number_of_rank_iterations=10, verbose_level=1)
            iterations_without_rank_checkup = 0
        else:
            # else just perform a single update step
            curr_res, k_needed1, reg_value_new, loss_values = MY_UPDATE_LOSS(xFTT.xFTT_list[n])
            iterations_without_rank_checkup += 1

            if delta_rank_short > 0.:
                rank_rule = Relative_Singularvalue_Tresholding(delta_rank_short, maxranks=[None] * (problem.dim - 1), dims=xFTT.xFTT_list[n].xTT.tensor_basis_functions.ndofs, rankincr=0, verbose=rank_adapt_verbose)
                xFTT.xFTT_list[n].xTT.modify_ranks(rank_rule, verbose=rank_adapt_verbose)
                xFTT.xFTT_list[n].xTT.tt.set_core(0)

        # check singular values:
        xFTT.logs["singular_values"].append([])

        for pos in range(xFTT.xFTT_list[n].xTT.tt.n_comps - 1):
            xFTT.xFTT_list[n].xTT.tt.set_core(pos)
            c = xFTT.xFTT_list[n].xTT.tt.comps[pos]
            s = c.shape
            c = c.reshape(s[0] * s[1], s[2])
            u, sigma, v = torch.linalg.svd(c, full_matrices=False)
            xFTT.logs["singular_values"][-1].append(sigma.to("cpu").tolist())
        xFTT.xFTT_list[n].xTT.tt.set_core(0)

        train_time_end = time.time()

        xFTT.logs["loss"].append(curr_res.item())
        xFTT.logs["sweeps"].append(loss_values)
        xFTT.logs["time"].append(train_time_end - train_time_start)

        if adaptive_reguarlization:
            xFTT.reg_value[n - 1] = reg_value_new
        else:
            xFTT.reg_value[n - 1] = xFTT.reg_value[n]

        # TODO: optional rank truncation of the xTT subobject of xFTT.xFTT_list[n]

        if plots:
            # Check if the current step should be plotted
            if n % 20 == 0 or n in [0, 1, N - 1]:
                # --- Figure Setup ---
                if xFTT.linear_extension:
                    plt.figure(figsize=(12, 4))
                else:
                    plt.figure(figsize=(8, 4))

                basis_info = get_basis_info(xFTT.xFTT_list[n].xTT.tensor_basis_functions)
                limits = basis_info["lims"]

                # --- Plot 1: Potential V(x) ---
                if xFTT.linear_extension:
                    plt.subplot(1, 3, 1)
                else:
                    plt.subplot(1, 2, 1)

                # Create grid tensor on the specified device
                x_grid = torch.linspace(limits[0][0], limits[0][1], 800, device=xFTT.device).reshape(-1, 1)

                # Calculate potential on the grid and for samples
                Vx_grid = xFTT.xFTT_list[n].V(x_grid).detach()
                V_samples = xFTT.xFTT_list[n].V(X_u[n]).detach()
                Vmin, Vmax = min(V_samples), max(V_samples)

                # Plot potential V(x) - move tensors to CPU for plotting
                plt.plot(x_grid.cpu().numpy(), Vx_grid.cpu().numpy(), label=f"V at n = {n} t = {times[n].item():.4f}")

                # Create another grid for plotting reference potentials
                x_grid_limit_values = torch.linspace(-3, 3, 400, device=xFTT.device).reshape(-1, 1)

                # Plot target potential V_target - move to CPU
                plt.plot(x_grid_limit_values.cpu().numpy(), problem.target(x_grid_limit_values).detach().cpu().numpy(), label="V_target", color="gray", alpha=0.5)

                # --- Normalization Constant and Reference Potential ---
                if isinstance(problem.target, Multiwell):
                    assert problem.dim == 1
                    M = 73.1858236819160799233138441524096759292893782348385136818630588128474895165
                    shift = 0.1
                    Vmin, Vmax = -4 - shift, 0 + shift
                elif isinstance(problem.target, GaussianProblem):
                    # Ensure covariance matrix is on CPU for numpy operations
                    M = (2 * np.pi) ** (problem.dim / 2) * np.sqrt(torch.linalg.det(problem.target.cov.cpu()).numpy())
                    Vmin, Vmax = 0, 4
                else:
                    raise NotImplementedError("Other reference M not implemented")

                # Plot samples - move to CPU
                plt.plot(X_u[n].cpu().numpy(), 0 * X_u[n].cpu().numpy(), "o", color="orange", markersize=2, label=f"samples at n={n}")

                # --- Plot formatting and learning window ---
                learning_window = min_max_per_dim(X_u[n])
                plt.xlim(-3.5, 3.5)
                plt.ylim(-1, 10.)

                # --- Plot 2: Linear extension contributions (if applicable) ---
                if xFTT.linear_extension is not None:
                    plt.subplot(1, 3, 2)
                    xFTT_t = xFTT.xFTT_list[n]
                    v_xTT_contrib_x_grid = xFTT_t.xTT(x_grid).detach()
                    v_linear_contrib_x_grid = xFTT.linear_extension(x_grid, xFTT_t.t).detach() @ xFTT_t.c_linear

                    # Move tensors to CPU for plotting
                    plt.plot(x_grid.cpu().numpy(), v_xTT_contrib_x_grid.cpu().numpy(), label=f"xTT contrib at n = {n} t = {times[n].item():.4f}")
                    plt.plot(x_grid.cpu().numpy(), v_linear_contrib_x_grid.cpu().numpy(), label=f"linear at n = {n} c = {xFTT.xFTT_list[n].c_linear[:]}")
                    plt.legend()
                    plt.subplot(1, 3, 3)
                else:
                    plt.subplot(1, 2, 2)

                # --- Plot 3 (or 2): Drift u(x) ---
                x_grid_ext = torch.linspace(limits[0][0] - 1, limits[0][1] + 1, 400, device=xFTT.device).reshape(-1, 1)
                DVx_grid = xFTT.xFTT_list[n].u(x_grid_ext).detach()

                # Move tensor to CPU for plotting
                plt.plot(x_grid_ext.cpu().numpy(), DVx_grid.cpu().numpy(), label=f"u at n = {n}")

                if basis_info["type"] == "TensorSplineBasis":
                    grid_points_equidistant = torch.linspace(limits[0][0], limits[0][1], basis_info["nknots"][0], device=xFTT.device)
                    # Move grid points to CPU for plotting
                    plt.scatter(grid_points_equidistant.cpu().numpy(), 0 * grid_points_equidistant.cpu().numpy(),
                                label="equidistant grid points", color="red", s=10)

                # Shade the linear extension regions
                plt.axvspan(limits[0][0] - 1, learning_window[0][0], facecolor="blue", alpha=0.2, label="linear extension")
                plt.axvspan(learning_window[0][1], limits[0][1] + 1, facecolor="blue", alpha=0.2)

                # --- Save Figure ---
                plt.savefig(f"plots/V_iter{iteration}_at_n_{n}.pdf")
                plt.close()

        if verbose:
            print("At time {n}: ALS algebraic residual error: {c}{i}{rep} with rank = {r} and fdims = {f}".format(n=n, c=Fore.RED, i=curr_res, rep=Style.RESET_ALL, r=xFTT.xFTT_list[n].xTT.tt.rank, f=xFTT.xFTT_list[n].xTT.tt.dims))
            print("Expected runtime: %.2f min" % (np.mean(xFTT.logs["time"]) * n / 60))
