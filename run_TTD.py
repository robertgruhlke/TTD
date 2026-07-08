"""Solve a sampling problem with TTD and report path statistics.

Supports all problems implemented in ttd/problems/concrete.py via `--problem`.
All hyperparameters are CLI flags; run with `--help` to list them. Defaults
reproduce the original 2D double-well (Multiwell) example.
"""

import argparse

import torch

from ttd.evaluation import compute_path_statistics
from ttd.policies import annealed_Langevin
from ttd.problems.concrete import (
    Funnel,
    GaussianProblem,
    GinzburgLandau,
    Kitagawa,
    Multiwell,
)
from ttd.sde import simulate_SDE
from ttd.solvers.backward_iteration import fit_initial_value_function, train
from ttd.utils.progress import train_progress_bar
from ttd.xftt import xFTT


def build_basis_info(args, n_double_wells, device):
    """Extended-Fourier basis_info. For "multiwell"/"ginzburglandau", coordinates
    outside n_double_wells (i.e. the plain-Gaussian tail) get a smaller degree
    (n_basis_secondary) since they don't need the extra shape flexibility.
    """
    include_linear = args.include_linear
    include_quadratic = args.include_quadratic
    degree_main = 1 + int(include_linear) + int(include_quadratic) + args.n_basis_main

    if args.problem in ("multiwell", "ginzburglandau"):
        degree_secondary = args.n_basis_secondary
        n_basis = [degree_main] * min(args.d, n_double_wells) + [degree_secondary] * max(0, args.d - n_double_wells)
    else:
        n_basis = [degree_main] * args.d

    return {
        "type": "TensorExtendedFourierBasis",
        "lims": [[-args.basis_lim, args.basis_lim] for _ in range(args.d)],
        "n_basis": n_basis,
        "device": device,
        "orthonormalize": args.orthonormalize,
        "include_linear": include_linear,
        "include_quadratic": include_quadratic,
    }


def build_problem(args, basis_info, n_double_wells, reg):
    common = dict(basis_info=basis_info, rank=args.rank, timeSteps=args.N,
                  batch_size=args.batch_size, T=args.T, reg=reg)

    if args.problem == "multiwell":
        delta = args.delta if args.delta is not None else 2.0
        return Multiwell(dim=args.d, n_double_wells=n_double_wells, alpha=args.alpha,
                          delta=delta, x_shift=args.x_shift, tilt=args.tilt, **common)
    elif args.problem == "funnel":
        return Funnel(dim=args.d, **common)
    elif args.problem == "gaussian":
        mean = torch.zeros(args.d)
        cov = torch.eye(args.d) * args.gaussian_var
        return GaussianProblem(mean=mean, cov=cov, **common)
    elif args.problem == "ginzburglandau":
        delta = args.delta if args.delta is not None else 1.0
        return GinzburgLandau(dim=args.d, n_double_wells=n_double_wells, beta=args.beta,
                               kappa=args.kappa, delta=delta, **common)
    elif args.problem == "kitagawa":
        if args.d != args.T_model:
            print("warning: Kitagawa expects dim == T_model (got dim=%d, T_model=%d)" % (args.d, args.T_model))
        return Kitagawa(dim=args.d, T_model=args.T_model, sigma_v=args.sigma_v, sigma_w=args.sigma_w,
                         nonlinear_strength=args.nonlinear_strength, data_seed=args.data_seed,
                         device=args.device, **common)
    else:
        raise ValueError("Unknown problem %r" % args.problem)


def run(args):
    torch.manual_seed(args.seed)

    device = args.device
    n_double_wells = args.n_double_wells if args.n_double_wells is not None else args.d
    reg = args.reg if args.reg is not None else 1e-5 / (10 * 6 ** args.d)

    basis_info = build_basis_info(args, n_double_wells, device)
    problem = build_problem(args, basis_info, n_double_wells, reg)

    myxFTT = xFTT(problem=problem, linear_extension=None, p_gradient_extension=args.p_gradient_extension,
                  device=device, initial_policy=annealed_Langevin)
    training_data = simulate_SDE(problem, myxFTT.u, problem.stepSizes, problem.batch_size,
                                  simulation_device=device, computation_device=device)

    xTT, reg_param_updated = fit_initial_value_function(problem, args.d, training_data, reg, args.K_unif, args.tol, device)
    problem.reg = reg_param_updated

    for iteration in range(args.no_iters):
        if iteration > 0:
            training_data = simulate_SDE(problem, myxFTT.u, problem.stepSizes, problem.batch_size,
                                          simulation_device=device, computation_device=device)
            selection = ~(torch.abs(training_data[0]) > args.filter_traj).any([0, 2])
            training_data = (training_data[0][:, selection, :], training_data[1][:, selection, :],
                              training_data[2], training_data[3][:, selection, :], training_data[4])

        desc = "iter %d/%d" % (iteration + 1, args.no_iters)
        if args.log_mode == "progress":
            with train_progress_bar(args.N, desc):
                train(problem, myxFTT, iteration, training_data, verbose=False, plots=False,
                      initial_xTT=xTT, tol=args.tol, adaptive_reguarlization=True)
        else:
            train(problem, myxFTT, iteration, training_data, verbose=True, plots=False,
                  initial_xTT=xTT, tol=args.tol, adaptive_reguarlization=True)
        print("finished %s" % desc)

    try:
        Z_reference, norm_reference = problem.analytic_reference()
    except AttributeError:
        Z_reference, norm_reference = None, None

    stats = compute_path_statistics(
        problem, myxFTT.u, problem.stepSizes, args.eval_batch_size, device,
        Z_reference=Z_reference, norm_reference=norm_reference, filter_traj=args.filter_traj,
        seed=args.eval_seed, verbose=True, progress_bar=(args.log_mode == "progress"),
    )
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--problem", type=str, default="multiwell",
                         choices=["multiwell", "funnel", "gaussian", "ginzburglandau", "kitagawa"],
                         help="which problem from ttd/problems/concrete.py to solve")
    parser.add_argument("--log_mode", type=str, default="progress", choices=["progress", "detailed"],
                         help='"progress": tqdm bars; "detailed": full per-step ALS output')
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="training seed")

    # generic problem / TT / training hyperparameters
    parser.add_argument("--d", type=int, default=2, help="problem dimension")
    parser.add_argument("--rank", type=int, default=2, help="TT rank")
    parser.add_argument("--N", type=int, default=512, help="number of Euler-Maruyama timesteps")
    parser.add_argument("--batch_size", type=int, default=32768, help="training batch size")
    parser.add_argument("--K_unif", type=int, default=32768, help="uniform samples for the initial ALS_L2 fit")
    parser.add_argument("--eval_batch_size", type=int, default=524288)
    parser.add_argument("--eval_seed", type=int, default=12345)
    parser.add_argument("--no_iters", type=int, default=2, help="number of outer policy-iteration sweeps")
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=None,
                         help="ALS regularization; default 1e-5 / (10 * 6**d)")
    parser.add_argument("--p_gradient_extension", type=float, default=0.1)
    parser.add_argument("--filter_traj", type=float, default=2.5)

    # basis hyperparameters (extended-Fourier)
    parser.add_argument("--basis_lim", type=float, default=2.3, help="basis domain is [-basis_lim, basis_lim]^d")
    parser.add_argument("--n_basis_main", type=int, default=10,
                         help="extra trigonometric degree for the primary coordinates")
    parser.add_argument("--n_basis_secondary", type=int, default=3,
                         help="basis degree for coordinates outside n_double_wells (multiwell/ginzburglandau only)")
    parser.add_argument("--orthonormalize", type=str, default="H2")
    parser.add_argument("--include_linear", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_quadratic", action=argparse.BooleanOptionalAction, default=True)

    # multiwell / ginzburglandau
    parser.add_argument("--n_double_wells", type=int, default=None, help="default: equal to --d")
    parser.add_argument("--delta", type=float, default=None,
                         help="double-well separation (multiwell, default 2.0) or coupling (ginzburglandau, default 1.0)")

    # multiwell only
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--x_shift", type=float, default=0.0)
    parser.add_argument("--tilt", type=float, default=0.0)

    # gaussian only
    parser.add_argument("--gaussian_var", type=float, default=1.0, help="isotropic covariance scale")

    # ginzburglandau only
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=1.0)

    # kitagawa only
    parser.add_argument("--T_model", type=int, default=20)
    parser.add_argument("--sigma_v", type=float, default=1.0)
    parser.add_argument("--sigma_w", type=float, default=1.0)
    parser.add_argument("--nonlinear_strength", type=float, default=0.3)
    parser.add_argument("--data_seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
