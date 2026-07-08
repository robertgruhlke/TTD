"""Path-statistics diagnostics for a trained control.

Girsanov-weight importance-sampling estimate of the target's normalizing constant Z
and E[||X_T||^2], simulated forward from the prior under the trained control.
"""

import math

import torch
from tqdm import tqdm

from ttd.sde import sigma_func


def _log_p_prior(x, dim):
    return -0.5 * torch.sum(x ** 2, 1) - dim / 2 * torch.log(torch.tensor(2 * math.pi))


def compute_path_statistics(problem, u, stepSizes, batch_size, device="cpu",
                             Z_reference=None, norm_reference=None, filter_traj=None,
                             seed=None, verbose=True, progress_bar=False):
    if seed is not None:
        torch.manual_seed(seed)

    N = len(stepSizes)
    X_u = problem.start.sample(batch_size).to(device)
    selection = torch.tensor([True] * batch_size, device=device)
    log_weights = -_log_p_prior(X_u, problem.dim)
    t_n = torch.tensor(0.)

    steps = tqdm(range(N), desc="evaluation", leave=False) if progress_bar else range(N)
    for n in steps:
        delta_t = torch.tensor(stepSizes[n])
        u_n = u(X_u, t_n)

        X_u_new = (X_u + delta_t * (problem.f(X_u, t_n) + u_n.reshape(-1, problem.dim) @ sigma_func(X_u, t_n).T)
                   + torch.sqrt(delta_t) * torch.randn(batch_size, problem.dim, device=device) @ sigma_func(X_u, t_n).T)

        mu_1 = X_u_new - problem.f(X_u_new, t_n + delta_t) * delta_t
        sigma_1 = sigma_func(X_u_new, t_n + delta_t)[0, 0]
        mu_2 = X_u + (problem.f(X_u, t_n) + u_n @ sigma_func(X_u, t_n).T) * delta_t
        sigma_2 = sigma_func(X_u, t_n)[0, 0]

        log_weights += (- torch.sum((X_u - mu_1) ** 2, 1) / (2 * sigma_1 ** 2 * delta_t)
                        + torch.sum((X_u_new - mu_2) ** 2, 1) / (2 * sigma_2 ** 2 * delta_t)
                        - torch.log(sigma_1) + torch.log(sigma_2))

        t_n = t_n + delta_t
        X_u = X_u_new

        if filter_traj is not None:
            selection = selection * ~(torch.abs(X_u) > filter_traj).any([1])

    log_weights = log_weights + (-problem.target(X_u).squeeze())
    log_weights = log_weights[selection]
    weights = torch.exp(log_weights)

    Z_estimate = torch.mean(weights).item()
    weights_variance = torch.var(weights).item()
    weights_log_variance = torch.var(log_weights).item()
    ESS = (torch.sum(weights) ** 2 / (torch.sum(weights ** 2) * sum(selection))).item()
    expected_norm = torch.mean(torch.sum(X_u[selection, :] ** 2, 1)).item()
    n_selected = int(sum(selection))

    stats = {
        "estimated_Z": Z_estimate,
        "Var_w": weights_variance,
        "Var_log_w": weights_log_variance,
        "ESS": ESS,
        "expected_norm": expected_norm,
        "n_selected": n_selected,
    }

    if verbose:
        print("\n--- path statistics (%d / %d samples selected) ---" % (n_selected, batch_size))
        print("estimated Z:   %.6e" % Z_estimate)
        if Z_reference:
            print("Z reference:   %.6e  (rel. err %.3e)" % (Z_reference, abs(Z_estimate - Z_reference) / Z_reference))
        print("Var(w):        %.6e" % weights_variance)
        print("Var(log w):    %.6e" % weights_log_variance)
        print("ESS:           %.3f" % ESS)
        print("expected norm: %.6e" % expected_norm)
        if norm_reference:
            print("norm ref:      %.6e  (rel. err %.3e)" % (norm_reference, abs(expected_norm - norm_reference) / norm_reference))

    return stats
