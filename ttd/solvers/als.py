"""Alternating least squares (ALS) solvers for fitting a value function in extended
tensor-train (xTT) format.

`ALS_L2`/`ALS_H1` fit a plain xTT to (gradient) data; `ALS_GeneralBasis_fast_vectorized`
fits the BSDE-style regression used by `backward_iteration.train()`. Also includes
rank/basis-degree adaptation (`optimize_and_choose_proper_rank`,
`optimize_and_choose_proper_basis_and_rank`).
"""

import matplotlib.pyplot as plt
import torch
from colorama import Fore, Style

torch.set_printoptions(precision=10)

from copy import deepcopy

from ttd.tt.core import TensorTrain
from ttd.tt.extended import Extended_TensorTrain


def plot_als_info(losses, reg_contribs, reg_params_hist, d):
    sweep_len = 2 * (d - 1)
    steps = range(len(losses))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Loss
    axs[0].semilogy(steps, losses, label="loss behaviour")
    for k in range(0, len(losses), sweep_len):
        if k == 0:
            axs[0].axvline(k, color="k", linestyle="--", linewidth=0.7, alpha=0.5, label="sweep block")
        else:
            axs[0].axvline(k, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
    axs[0].legend()

    # 2. Regularization contribution
    axs[1].semilogy(range(len(reg_contribs)), reg_contribs, label="reg behaviour")
    for k in range(0, len(reg_contribs), sweep_len):
        if k == 0:
            axs[1].axvline(k, color="k", linestyle="--", linewidth=0.7, alpha=0.5, label="sweep block")
        else:
            axs[1].axvline(k, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
    axs[1].legend()

    # 3. Regularization parameter
    axs[2].semilogy(range(len(reg_params_hist)), reg_params_hist, label="reg magnitude history")
    for k in range(0, len(reg_params_hist), sweep_len):
        if k == 0:
            axs[2].axvline(k, color="k", linestyle="--", linewidth=0.7, alpha=0.5, label="sweep block")
        else:
            axs[2].axvline(k, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("plots/loss_regcontrib.pdf")
    plt.close(fig)


def ALS_L2(x, y, iterations, tol, reg_param_init, xTT, verbose=True, choose_SD=False, adaptive_reguarlization=True, verbose_adaptive_reg=False):
    """
    Fast ALS without loops over samples.
    """
    b, d = x.shape
    w = xTT.tensor_basis_functions(x)  # list of [b, n_i]

    reg_param = reg_param_init
    reg_magnitude_reduction = 1e-1

    # only used for adaptive reg_param choice
    losses = []
    reg_contribs = []
    reg_params_hist = []

    def update(core_pos, local_solution):
        xTT.tt.comps[core_pos] = local_solution.reshape(xTT.tt.comps[core_pos].shape)

    def loc_solve_TT(mu, A, reg_param, compute_post_err=True, compute_pre_err=False):
        """Optimizes a single TT core using precomputed contractions."""
        if reg_param is None or reg_param == 0.:
            V = torch.linalg.lstsq(A, y, rcond=None)[0]
        else:
            Rmat = torch.eye(A.shape[1], device=x.device)
            Ahat = torch.cat([A, torch.sqrt(torch.tensor(reg_param)) * Rmat], dim=0)
            yhat = torch.cat([y, torch.zeros((A.shape[1], 1), device=x.device)], dim=0)
            V = torch.linalg.lstsq(Ahat, yhat, rcond=None)[0]
        # update the xTT at position mu
        update(mu, V)
        err = (1.0 / b) * torch.norm(A @ V - y) ** 2
        rel_err = torch.norm(A @ V - y) ** 2 / torch.norm(y) ** 2
        return err, rel_err

    # local direct based solver, in case speed matters and stability is ensured
    def loc_solve_TT_direct_reguarization_aware(core_pos, A, reg_param, get_post_error=False, verbose_adaptive_reg=False, get_pre_error=False):
        # compute A^T A with assembling of A
        G = A.T @ A
        Ay = A.T @ y

        # only compute pre error at first step
        if verbose_adaptive_reg or get_pre_error:
            if len(losses) == 0:
                # current solution:
                solution_pre = xTT.tt.comps[core_pos].flatten()
                residual = A @ solution_pre.flatten() - y.flatten()
                loss_pre = torch.sum(residual * residual).item() / b  # = ||residual||^2 / b
                reg_pre = torch.sum(solution_pre * solution_pre).item()  # = ||solution_pre||^2

        if reg_param is None or reg_param == 0.:
            solution = torch.linalg.solve(G, Ay)
        else:
            # ridge: [M; sqrt(lam) I] * sol = [y; 0] normal equation (M.T M + lam I) sol = y
            R = torch.as_tensor(reg_param, device=A.device, dtype=A.dtype) * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
            solution = torch.linalg.solve(G + R, Ay)

        residual = A @ solution.flatten() - y.flatten()
        loss_post = torch.sum(residual * residual).item() / b  # = ||residual||^2 / b
        reg_post = torch.sum(solution * solution).item()

        old_reg_param = reg_param
        reg_param = reg_magnitude_reduction * loss_post / reg_post

        if verbose_adaptive_reg:
            print("=====================")
            if len(losses) == 0:
                print("loss_pre = ", loss_pre)
                print("reg_pre = ", reg_pre)
            print("loss_post = ", loss_post)
            print("reg_post = ", reg_post)
            print(" updated suggested for reg_param = ", reg_param, " ( before : ", old_reg_param, ")")
            print("=====================")

            if len(losses) == 0:
                losses.append(loss_pre)
            if len(reg_contribs) == 0:
                reg_contribs.append(reg_pre)
            if len(reg_params_hist) == 0:
                reg_params_hist.append(old_reg_param)  # init reg param value

            losses.append(loss_post)
            reg_contribs.append(reg_post)
            reg_params_hist.append(reg_param)

        update(core_pos, solution)

        if get_post_error:
            # training error for monitoring (based on the M used)
            if get_pre_error:
                return loss_post, loss_pre, reg_param
            else:
                return loss_post, None, reg_param
        else:
            if get_pre_error:
                return None, loss_pre, reg_param
            else:
                return None, None, reg_param

    if d == 1:
        assert choose_SD == False
        A = w[0]
        used_iteration = 1
        if not adaptive_reguarlization:
            err_post = loc_solve_TT(0, A, reg_param, True).item()
            if verbose:
                print("core 0 solve Error =", err_post)
        else:  # adaptive_reguarlization is true
            err_pre = None
            for k in range(iterations):
                if k == 0:
                    err_post, err_pre, reg_param = loc_solve_TT_direct_reguarization_aware(0, A, reg_param, True, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=True)
                else:
                    err_post, _, reg_param = loc_solve_TT_direct_reguarization_aware(0, A, reg_param, True, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=False)

                # stopping criterion
                denom = abs(err_pre) + 1e-12
                if abs(err_pre - err_post) / denom < tol:
                    if verbose:
                        print(f"breaking at k = {k+1} / {iterations}, loss = {err_post}, original loss = {err_pre}.")
                        print(f"         reg_param = {reg_param} with init reg_param  = {reg_param_init}.")
                    used_iteration = k + 1
                    break
                err_pre = err_post
        # set core to 0 (does nothing) to handle None type
        xTT.tt.set_core(0)
        if adaptive_reguarlization:
            return err_post, used_iteration, reg_param
        else:
            return err_post, used_iteration, None

    
    def compute_left_right(mu):
        # Left-to-right contractions
        L = torch.ones(b, 1, device=x.device).reshape(b, 1, 1)
        for k in range(mu):
            contr = torch.einsum("bj, rjR->brR", w[k], xTT.tt.comps[k])
            L = torch.einsum("brs, bsR->brR", L, contr)

        # Right-to-left contractions
        R = torch.ones(b, 1, device=x.device).reshape(b, 1, 1)
        for k in range(d - 1, mu, -1):
            contr = torch.einsum("bj, rjR->brR", w[k], xTT.tt.comps[k])
            R = torch.einsum("brs, bsR->brR", contr, R)
        return L[:, 0, :], R[:, :, 0]

    def get_current_loss():
        y_curr = xTT(x)
        err = (1.0 / b) * torch.norm(y_curr - y) ** 2
        return err

    def assemble_A(mu, L, R):
        return torch.einsum("br, bm, bs -> brms", L, w[mu], R).reshape(b, -1)

    def sweeping_with_fixed_reg():
        # === ALS main loop ===
        sweep_loop_indices = list(range(d - 1)) + list(range(d - 1, 0, -1))

        initial_loss = get_current_loss()
        for k in range(iterations):
            L_pre = get_current_loss()
            for mu in sweep_loop_indices:
                xTT.tt.set_core(mu)
                L, R = compute_left_right(mu)
                A = assemble_A(mu, L, R)
                L_post, L_post_rel = loc_solve_TT(mu, A, reg_param)
            if verbose:
                print(f"Sweep {k} loss =", L_post.item(), " relative loss =", L_post_rel.item())
            if torch.abs(L_pre - L_post) / torch.abs(L_pre) < tol or L_post.item() < 0.1 * tol:
                print("Early stopping at k =", k,
                    "with final loss =", L_post,
                    "(initial loss =", initial_loss, ")")
                break

        xTT.refresh_rank()  # just to be sure ranks are updated correctly in the xTT with respect to its tt

        return L_post, k, reg_param_init

    def sweeping_with_adaptive_reg(reg_init):
        loss_post = None
        reg_param = reg_init

        # set non-orthogonal core at core_pos = 0
        xTT.tt.set_core(0)

        loss_init = None

        for k in range(iterations):
            loss_post_from_compute = None
            for core_pos in range(d - 1):
                L, R = compute_left_right(core_pos)
                A = assemble_A(core_pos, L, R)
                if k == 0 and core_pos == 0:
                    _, loss_pre, reg_param = loc_solve_TT_direct_reguarization_aware(core_pos, A, reg_param, get_post_error=False, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=True)
                    loss_init = loss_pre
                else:
                    _, _, reg_param = loc_solve_TT_direct_reguarization_aware(core_pos, A, reg_param, get_post_error=False, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=False)
                # set non-orthogonal position at core_pos + 1 (moves core one position to the right)
                xTT.tt.set_core(core_pos + 1)

            for core_pos in range(d - 1, 0, -1):
                L, R = compute_left_right(core_pos)
                A = assemble_A(core_pos, L, R)
                # only compute loss_post at the last iteration
                loss_post_from_compute, _, reg_param = loc_solve_TT_direct_reguarization_aware(core_pos, A, reg_param, get_post_error=(core_pos == 1), verbose_adaptive_reg=verbose_adaptive_reg)
                # set non-orthogonal position at core_pos - 1 (moves core one position to the left)
                xTT.tt.set_core(core_pos - 1)

            loss_post = loss_post_from_compute
            if verbose:
                print(f"sweep {k}: loss = {loss_post} (prev {loss_pre})")

            # stopping criterion
            denom = abs(loss_pre) + 1e-12
            if abs(loss_pre - loss_post) / denom < tol:
                if verbose:
                    print(f"breaking at k = {k+1} / {iterations}, loss = {loss_post}, original loss = {loss_init}.")
                    print(f"         reg_param = {reg_param} with init reg_param  = {reg_param_init}.")
                break

            loss_pre = loss_post

        if verbose_adaptive_reg:
            plot_als_info(losses, reg_contribs, reg_params_hist, d)

        xTT.refresh_rank()  # just to be sure ranks are updated correctly in the xTT with respect to its tt

        return loss_post, k, reg_param

    if not adaptive_reguarlization:
        return sweeping_with_fixed_reg()
    else:
        return sweeping_with_adaptive_reg(reg_param_init)


def ALS_H1(x, y, y_grad, iterations, tol, reg_param_init, xTT, verbose=True, choose_SD=False, adaptive_reguarlization=True, verbose_adaptive_reg=False):
    """
    Fast ALS without loops over samples.
    """
    b, d = x.shape
    w = xTT.tensor_basis_functions(x)  # list of [b, n_i]
    Dw = xTT.tensor_basis_functions.grad(x)  # list of [b, n_i]

    Y = torch.cat([y] + [y_grad[:, j:j + 1] for j in range(d)], dim=0)
    Y_norms = [torch.linalg.norm(y, "fro")] + [torch.linalg.norm(y_grad[:, j:j + 1]) for j in range(d)]

    reweight = False

    reg_param = reg_param_init
    reg_magnitude_reduction = 1e-1

    # only used for adaptive reg_param choice
    losses = []
    reg_contribs = []
    reg_params_hist = []

    def update(core_pos, local_solution):
        xTT.tt.comps[core_pos] = local_solution.reshape(xTT.tt.comps[core_pos].shape)

    def loc_solve_TT(mu, A, reg_param, compute_post_err=True, compute_pre_err=False):
        """Optimizes a single TT core using precomputed contractions."""
        if reg_param is None or reg_param == 0.:
            V = torch.linalg.lstsq(A, Y, rcond=None)[0]
        else:
            Rmat = torch.eye(A.shape[1], device=x.device)
            Ahat = torch.cat([A, torch.sqrt(torch.tensor(reg_param)) * Rmat], dim=0)
            yhat = torch.cat([Y, torch.zeros((A.shape[1], 1), device=x.device)], dim=0)
            V = torch.linalg.lstsq(Ahat, yhat, rcond=None)[0]
        # update the xTT at position mu
        update(mu, V)
        err = (1.0 / b) * torch.norm(A @ V - Y) ** 2
        rel_err = torch.norm(A @ V - Y) ** 2 / torch.norm(y) ** 2
        return err, rel_err

    # local direct based solver, in case speed matters and stability is ensured
    def loc_solve_TT_direct_reguarization_aware(core_pos, A, reg_param, weights=None, get_post_error=False, verbose_adaptive_reg=False, get_pre_error=False):
        # compute A^T A with assembling of A
        G = A.T @ A

        if weights is None:
            Y = torch.cat([y] + [y_grad[:, j:j + 1] for j in range(d)], dim=0)
        else:
            Y = torch.cat([y] + [weights[j + 1] * y_grad[:, j:j + 1] for j in range(d)], dim=0)

        AY = A.T @ Y

        # only compute pre error at first step
        if verbose_adaptive_reg or get_pre_error:
            if len(losses) == 0:
                # current solution:
                solution_pre = xTT.tt.comps[core_pos].flatten()
                residual = A @ solution_pre.flatten() - Y.flatten()
                loss_pre = torch.sum(residual * residual).item() / b  # = ||residual||^2 / b
                reg_pre = torch.sum(solution_pre * solution_pre).item()  # = ||solution_pre||^2

        if reg_param is None or reg_param == 0.:
            solution = torch.linalg.solve(G, AY)
        else:
            # ridge: [M; sqrt(lam) I] * sol = [y; 0] normal equation (M.T M + lam I) sol = y
            R = torch.as_tensor(reg_param, device=A.device, dtype=A.dtype) * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
            solution = torch.linalg.solve(G + R, AY)

        residual = A @ solution.flatten() - Y.flatten()
        loss_post = torch.sum(residual * residual).item() / b  # = ||residual||^2 / b
        reg_post = torch.sum(solution * solution).item()

        old_reg_param = reg_param
        reg_param = reg_magnitude_reduction * loss_post / reg_post

        if verbose_adaptive_reg:
            print("=====================")
            if len(losses) == 0:
                print("loss_pre = ", loss_pre)
                print("reg_pre = ", reg_pre)
            print("loss_post = ", loss_post)
            print("reg_post = ", reg_post)
            print(" updated suggested for reg_param = ", reg_param, " ( before : ", old_reg_param, ")")
            print("=====================")

            if len(losses) == 0:
                losses.append(loss_pre)
            if len(reg_contribs) == 0:
                reg_contribs.append(reg_pre)
            if len(reg_params_hist) == 0:
                reg_params_hist.append(old_reg_param)  # init reg param value

            losses.append(loss_post)
            reg_contribs.append(reg_post)
            reg_params_hist.append(reg_param)

        update(core_pos, solution)

        if get_post_error:
            # training error for monitoring (based on the M used)
            if get_pre_error:
                return loss_post, loss_pre, reg_param
            else:
                return loss_post, None, reg_param
        else:
            if get_pre_error:
                return None, loss_pre, reg_param
            else:
                return None, None, reg_param

    if d == 1:
        assert choose_SD == False
        A_eval = w[0]
        A_grad = Dw[0]
        A = torch.cat([A_eval, A_grad], dim=0)
        used_iteration = 1
        if not adaptive_reguarlization:
            err_post = loc_solve_TT(0, A, reg_param, True).item()
            if verbose:
                print("core 0 solve Error =", err_post)
        else:  # adaptive_reguarlization is true
            err_pre = None
            for k in range(iterations):
                if k == 0:
                    err_post, err_pre, reg_param = loc_solve_TT_direct_reguarization_aware(0, A, reg_param, True, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=True)
                else:
                    err_post, _, reg_param = loc_solve_TT_direct_reguarization_aware(0, A, reg_param, True, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=False)

                # stopping criterion
                denom = abs(err_pre) + 1e-12
                if abs(err_pre - err_post) / denom < tol:
                    if verbose:
                        print(f"breaking at k = {k+1} / {iterations}, loss = {err_post}, original loss = {err_pre}.")
                        print(f"         reg_param = {reg_param} with init reg_param  = {reg_param_init}.")
                    used_iteration = k + 1
                    break
                err_pre = err_post
        # set core to 0 (does nothing) to handle None type
        xTT.tt.set_core(0)
        if adaptive_reguarlization:
            return err_post, used_iteration, reg_param
        else:
            return err_post, used_iteration, None

    
    def compute_left_right(mu, deriv_direction):
        # if deriv_direction == -1 then no derivative is taken
        # Left-to-right contractions
        L = torch.ones(b, 1, device=x.device).reshape(b, 1, 1)
        for k in range(mu):
            # if k == deriv_direction then gradients of basis functions required
            if deriv_direction == k:
                contr = torch.einsum("bj, rjR->brR", Dw[k], xTT.tt.comps[k])
            else:
                contr = torch.einsum("bj, rjR->brR", w[k], xTT.tt.comps[k])
            L = torch.einsum("brs, bsR->brR", L, contr)

        # Right-to-left contractions
        R = torch.ones(b, 1, device=x.device).reshape(b, 1, 1)
        for k in range(d - 1, mu, -1):
            if deriv_direction == k:
                contr = torch.einsum("bj, rjR->brR", Dw[k], xTT.tt.comps[k])
            else:
                contr = torch.einsum("bj, rjR->brR", w[k], xTT.tt.comps[k])
            R = torch.einsum("brs, bsR->brR", contr, R)
        return L[:, 0, :], R[:, :, 0]  # L and R w.r.t. deriv_direction

    def compute_all_left_right(mu):
        # left and right part without derivatives
        L, R = compute_left_right(mu, -1)
        L_list, R_list = [L], [R]
        for deriv_direction in range(d):
            L, R = compute_left_right(mu, deriv_direction)
            L_list.append(L)
            R_list.append(R)
        return L_list, R_list

    def get_current_loss():
        y_curr = xTT(x)
        err = (1.0 / b) * torch.norm(y_curr - y) ** 2
        return err

    def assemble_A(mu, L_list, R_list, reweight=True):
        A_list = []
        for i in range(d + 1):
            L, R = L_list[i], R_list[i]
            derive_direction = i - 1
            if derive_direction == mu:
                A = torch.einsum("br, bm, bs -> brms", L, Dw[mu], R).reshape(b, -1)
            else:
                A = torch.einsum("br, bm, bs -> brms", L, w[mu], R).reshape(b, -1)
            A_list.append(A)
        if reweight:
            A_norms_list = [torch.linalg.norm(A_i, "fro") for A_i in A_list]
            V_norm = A_norms_list[0]  # magnitude of value components
            eps = 1e-12

            Y_0_norm = Y_norms[0]

            weights = [V_norm * Y_0_norm / (mat_norm * y_norm + eps) for mat_norm, y_norm in zip(A_norms_list, Y_norms)]

            A_list_reweighted = [A_i * w for w, A_i in zip(weights, A_list)]
            A = torch.cat(A_list_reweighted, dim=0)
            return A, weights
        else:
            A = torch.cat(A_list, dim=0)
            return A, None

    def sweeping_with_fixed_reg():
        # === ALS main loop ===
        sweep_loop_indices = list(range(d - 1)) + list(range(d - 1, 0, -1))

        initial_loss = get_current_loss()
        for k in range(iterations):
            L_pre = get_current_loss()
            for mu in sweep_loop_indices:
                xTT.tt.set_core(mu)
                L_list, R_list = compute_all_left_right(mu)
                A, weights = assemble_A(mu, L_list, R_list)
                L_post, L_post_rel = loc_solve_TT(mu, A, reg_param)
            if verbose:
                print(f"Sweep {k} loss =", L_post.item(), " relative loss =", L_post_rel.item())
            if torch.abs(L_pre - L_post) / torch.abs(L_pre) < tol or L_post.item() < 0.1 * tol:
                print("Early stopping at k =", k,
                    "with final loss =", L_post,
                    "(initial loss =", initial_loss, ")")
                break

        return L_post, k, reg_param_init

    def sweeping_with_adaptive_reg(reg_init):
        loss_post = None
        reg_param = reg_init

        # set non-orthogonal core at core_pos = 0
        xTT.tt.set_core(0)

        loss_init = None

        for k in range(iterations):
            loss_post_from_compute = None
            for core_pos in range(d - 1):
                L_list, R_list = compute_all_left_right(core_pos)
                A, weights = assemble_A(core_pos, L_list, R_list, reweight)

                if k == 0 and core_pos == 0:
                    _, loss_pre, reg_param = loc_solve_TT_direct_reguarization_aware(core_pos, A, reg_param, weights, get_post_error=False, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=True)
                    loss_init = loss_pre
                else:
                    _, _, reg_param = loc_solve_TT_direct_reguarization_aware(core_pos, A, reg_param, weights, get_post_error=False, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=False)
                # set non-orthogonal position at core_pos + 1 (moves core one position to the right)
                xTT.tt.set_core(core_pos + 1)

            for core_pos in range(d - 1, 0, -1):
                L_list, R_list = compute_all_left_right(core_pos)
                A, weights = assemble_A(core_pos, L_list, R_list, reweight)
                # only compute loss_post at the last iteration
                loss_post_from_compute, _, reg_param = loc_solve_TT_direct_reguarization_aware(core_pos, A, reg_param, weights, get_post_error=(core_pos == 1), verbose_adaptive_reg=verbose_adaptive_reg)
                # set non-orthogonal position at core_pos - 1 (moves core one position to the left)
                xTT.tt.set_core(core_pos - 1)

            loss_post = loss_post_from_compute
            if verbose:
                print(f"sweep {k}: loss = {loss_post} (prev {loss_pre})")

            # stopping criterion
            denom = abs(loss_pre) + 1e-12
            if abs(loss_pre - loss_post) / denom < tol:
                if verbose:
                    print(f"breaking at k = {k+1} / {iterations}, loss = {loss_post}, original loss = {loss_init}.")
                    print(f"         reg_param = {reg_param} with init reg_param  = {reg_param_init}.")
                break

            loss_pre = loss_post

        if verbose_adaptive_reg:
            plot_als_info(losses, reg_contribs, reg_params_hist, d)

        return loss_post, k, reg_param

    if not adaptive_reguarlization:
        return sweeping_with_fixed_reg()
    else:
        return sweeping_with_adaptive_reg(reg_param_init)


def ALS_GeneralBasis_fast_vectorized(x, y, Sigma, xFTT_t, iterations, tol, reg_param_init, adaptive_reguarlization=True, verbose=True, verbose_adaptive_reg=False):
    """
    ALS with left/right stacks for fast matrix assembly, using incremental environments
    instead of rebuilding the design matrix from scratch at every core.

    Notation:
    - Left-Stacks[i][pos]  has shape (b, 1, r_pos) and represents the contraction over cores [0..pos-1]
    - Right-Stacks[i][pos] has shape (b, r_pos, 1) and represents the contraction over cores [pos..d-1]
      (matching the assembly line: einsum("br, bm, bR -> brmR",
                                           Left[:,0,:], w_ij_at_pos, Right[:,:,0]))
    """
    device = x.device
    dtype = x.dtype
    b, d = x.shape

    reg_param = reg_param_init
    reg_magnitude_reduction = 1e-1

    # only used for adaptive reg_param choice
    losses = []
    reg_contribs = []
    reg_params_hist = []

    # ---------------------------
    # linear extension (optional)
    # ---------------------------
    linear_extension = xFTT_t.linear_extension
    if linear_extension is not None:
        B = (xFTT_t.linear_extension(x, xFTT_t.t)
             + torch.einsum("bd,bdn->bn", Sigma, xFTT_t.linear_extension.grad(x, xFTT_t.t))).to(device=device, dtype=dtype)

    # ---------------------------
    # necessary precomputations
    # ---------------------------
    w = xFTT_t.xTT.tensor_basis_functions(x)  # list/tuple: w[j] in R^{b x m_j}
    w_dot = xFTT_t.xTT.tensor_basis_functions.grad(x)  # list/tuple: w_dot[j] in R^{b x m_j}

    sigma_w_dot = [torch.einsum("b,bi->bi", Sigma[:, j], w_dot[j]) for j in range(d)]

    def w_at(i, j):
        # i == -1 never equals j, so plain basis w[j]; otherwise gradient only at position i == j
        return sigma_w_dot[j] if i == j else w[j]

    def local_lstsq_solve_with_M(core_pos, M, reg_param, get_error=False):
        if linear_extension is not None:
            M = torch.cat([M, B], dim=1)
        if reg_param is None:
            solution = torch.linalg.lstsq(M, y, rcond=None)[0]
        else:
            # ridge: [M; sqrt(lam) I] * sol = [y; 0]
            lam_sqrt = torch.sqrt(torch.as_tensor(reg_param, device=device, dtype=dtype))
            Reye = torch.eye(M.shape[1], device=device, dtype=dtype)
            Ahat = torch.cat([M, lam_sqrt * Reye], dim=0)
            yhat = torch.cat([y, torch.zeros((M.shape[1], 1), device=device, dtype=dtype)], dim=0)
            solution = torch.linalg.lstsq(Ahat, yhat, rcond=None)[0]
        update(core_pos, solution)
        if get_error:
            # training error for monitoring (based on the M used)
            err = (torch.linalg.norm(M @ solution - y) ** 2) / b
            return err
        return None

    def local_lstsq_solve_with_M_reguarization_aware(core_pos, M, reg_param, get_error=False, verbose_adaptive_reg=False, get_pre_error=True):
        if linear_extension is not None:
            M = torch.cat([M, B], dim=1)

        # only compute pre error at first step
        if len(losses) == 0 or get_pre_error:
            # current solution:
            if linear_extension is not None:
                solution_pre = torch.concatenate([xFTT_t.xTT.tt.comps[core_pos].flatten(), xFTT_t.c_linear]).flatten()
            else:
                solution_pre = xFTT_t.xTT.tt.comps[core_pos].flatten()
            residual = M @ solution_pre - y.flatten()
            loss_pre = torch.sum(residual * residual).item() / b  # = ||residual||^2 / b
            reg_pre = torch.sum(solution_pre * solution_pre).item()  # = ||solution_pre||^2

        if reg_param is None or reg_param == 0.:
            solution = torch.linalg.lstsq(M, y, rcond=None)[0]
        else:
            # ridge: [M; sqrt(lam) I] * sol = [y; 0]
            lam_sqrt = torch.sqrt(torch.as_tensor(reg_param, device=device, dtype=dtype))
            Reye = torch.eye(M.shape[1], device=device, dtype=dtype)
            Ahat = torch.cat([M, lam_sqrt * Reye], dim=0)
            yhat = torch.cat([y, torch.zeros((M.shape[1], 1), device=device, dtype=dtype)], dim=0)
            solution = torch.linalg.lstsq(Ahat, yhat, rcond=None)[0]

        residual = M @ solution.flatten() - y.flatten()
        loss_post = torch.sum(residual * residual).item() / b  # = ||residual||^2 / b
        reg_post = torch.sum(solution * solution).item()

        old_reg_param = reg_param
        reg_param = reg_magnitude_reduction * loss_post / reg_post

        if verbose_adaptive_reg:
            print("=====================")
            if len(losses) == 0:
                print("loss_pre = ", loss_pre)
                print("reg_pre = ", reg_pre)
            print("loss_post = ", loss_post)
            print("reg_post = ", reg_post)
            print(" updated suggested for reg_param = ", reg_param, " ( before : ", old_reg_param, ")")
            print("=====================")

            if len(losses) == 0:
                losses.append(loss_pre)
            if len(reg_contribs) == 0:
                reg_contribs.append(reg_pre)
            if len(reg_params_hist) == 0:
                reg_params_hist.append(old_reg_param)  # init reg param value

            losses.append(loss_post)
            reg_contribs.append(reg_post)
            reg_params_hist.append(reg_param)

        update(core_pos, solution)

        if get_error:
            # training error for monitoring (based on the M used)
            if get_pre_error:
                return loss_post, loss_pre, reg_param
            else:
                return loss_post, None, reg_param
        else:
            if get_pre_error:
                return None, loss_pre, reg_param
            else:
                return None, None, reg_param

    def update(core_pos, local_solution):
        if linear_extension is not None:
            ndofs = linear_extension.ndofs
            core_vec = local_solution[:-ndofs]
            xFTT_t.c_linear = local_solution[-ndofs:].squeeze(1)
        else:
            core_vec = local_solution
        # back into core shape
        core_shape = xFTT_t.xTT.tt.comps[core_pos].shape
        xFTT_t.xTT.tt.comps[core_pos] = core_vec.reshape(core_shape).contiguous()


    # ---------------------------
    # special case d==1: ALS boils down to a single least-squares solution
    # ---------------------------
    if d == 1:
        # A = w_(-1,0) + w_(0,0)
        r, _, Rr = xFTT_t.xTT.tt.comps[0].shape
        assert r == 1 and Rr == 1, "For d=1 the TT core at core_pos must be a vector"
        A = w_at(-1, 0) + w_at(0, 0)
        used_iteration = 1
        if not adaptive_reguarlization:
            err_post = local_lstsq_solve_with_M(0, A, reg_param, True).item()
            if verbose:
                print("core 0 solve Error =", err_post)

        else:  # adaptive_reguarlization is true
            for k in range(iterations):
                err_post, err_pre, reg_param = local_lstsq_solve_with_M_reguarization_aware(0, A, reg_param, True, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=True)
                # stopping criterion
                denom = abs(err_pre) + 1e-12
                if abs(err_pre - err_post) / denom < tol:
                    if verbose:
                        print(f"breaking at k = {k+1} / {iterations}, loss = {err_post}, original loss = {err_pre}.")
                        print(f"         reg_param = {reg_param} with init reg_param  = {reg_param_init}.")
                    used_iteration = k + 1
                    break

        # set core to 0 (does nothing) to handle None type
        xFTT_t.xTT.tt.set_core(0)
        if adaptive_reguarlization:
            return err_post, used_iteration, reg_param, err_post
        else:
            return err_post, used_iteration, None, err_post

    # ---------------------------
    # higher-dimensional case d > 1: sweeps necessary in ALS / ASD
    # ---------------------------

    # W[j] = w_at[:, j]
    num_i = d + 1

    def build_W():
        """
        Build W[j] of shape (d, b, m) for each j in [0..d-1].
        Returns a list of length d.
        """
        d = len(w)
        W = []
        for j in range(d):
            # baseline: repeat w[j] (b, m) d times along axis 0
            Wj = w[j].unsqueeze(0).expand(d + 1, -1, -1).clone()  # (d, b, m)
            # overwrite row i=j
            Wj[j + 1] = sigma_w_dot[j]  # (b, m)
            W.append(Wj)
        return W
    W = build_W()

    # ---------------------------
    # build right-stacks once, backwards (from the current TT)
    # Right[i][pos] = environment of the right-hand side from pos onward (inclusive)
    # ---------------------------
    def build_right_stacks():
        Right = [None] * (d + 1)
        Right[d] = torch.ones(num_i, b, 1, 1, device=device, dtype=dtype)  # (num_i, b, 1, 1)

        for j in range(d - 1, 0, -1):
            core = xFTT_t.xTT.tt.comps[j]  # (r, m, R)
            r, m, R = core.shape

            # (1) compute C0 and Csig: (b, r, R)
            C0 = torch.einsum("bm,rmR->brR", w_at(-1, j), core)  # use w_at(-1,j) = W[j][-1] == w[j], shape (b, r, R)
            Csig = torch.einsum("bm,rmR->brR", sigma_w_dot[j], core)  # (b, r, R)

            # (2) build contract tensor (num_i, b, r, R)
            contract = C0.unsqueeze(0).expand(num_i, -1, -1, -1).clone()
            idx = j + 1
            contract[idx] = Csig  # set the i==j slot

            # (3) flatten (num_i*b, r, R) and multiply with Right[j+1] flattened (num_i*b, R, 1)
            left_flat = contract.reshape(num_i * b, r, R)  # (num_i*b, r, R)
            right_next = Right[j + 1].reshape(num_i * b, R, 1)  # (num_i*b, R, 1)

            res_flat = torch.bmm(left_flat, right_next)  # (num_i*b, r, 1)
            Right[j] = res_flat.view(num_i, b, r, 1)  # back to (num_i, b, r, 1)

        return Right

    # ---------------------------
    # build left-stacks incrementally (during the right sweep)
    # Left[i][0] = identity, Left[i][pos+1] = Left[i][pos] o core_pos (using w_at(i, core_pos))
    # ---------------------------
    def init_left_stacks_empty():
        Left = [None] * (d + 1)
        Left[0] = torch.ones(num_i, b, 1, 1, device=device, dtype=dtype)
        return Left

    def push_left_one_pos(Left, pos):
        """Vectorized update of Left[pos+1] from Left[pos] after optimizing core 'pos'."""
        L_prev = Left[pos]  # (num_i, b, r, s)
        core = xFTT_t.xTT.tt.comps[pos]  # (s, m, R)

        # standard contraction for all i
        C0 = torch.einsum("bm, smR -> b s R", w[pos], core)  # (b, s, R)
        C0 = C0.unsqueeze(0).expand(num_i, -1, -1, -1)  # (num_i, b, s, R)

        # Left_prev: (num_i, b, r, s) -> einsum over s
        Left_next = torch.einsum("ibrs, ibsR -> ibrR", L_prev, C0)  # (num_i, b, r, R)

        # special case i == pos
        Csig = torch.einsum("bm, smR -> b s R", sigma_w_dot[pos], core)
        Left_next[pos] = torch.einsum("brs, b s R -> brR", L_prev[pos], Csig)

        Left[pos + 1] = Left_next

    # ---------------------------
    # right-stacks incrementally (during the left sweep)
    # ---------------------------
    def init_right_stacks_empty():
        Right = [None] * (d + 1)
        Right[d] = torch.ones(num_i, b, 1, 1, device=device, dtype=dtype)
        return Right

    def push_right_one_pos(Right, pos):
        """After optimizing core 'pos': update Right[i][pos] from Right[i][pos+1]."""
        R_next = Right[pos + 1]  # num_i x b x s x R  with R = 1
        core = xFTT_t.xTT.tt.comps[pos]  # r x m x s
        C0 = torch.einsum("bm, rms -> brs", w[pos], core)
        C0 = C0.unsqueeze(0).expand(num_i, -1, -1, -1).clone()  # (num_i, b, r, s)
        C0[pos + 1] = torch.einsum("bm, rms -> brs", sigma_w_dot[pos], core)
        Right[pos] = torch.einsum("ibrs, ibsR -> ibrR", C0, R_next)

    # local direct based solver, in case speed matters and stability is ensured
    def local_solve_direct_reguarization_aware(core_pos, LeftStacks, RightStacks, reg_param, get_error=False, verbose_adaptive_reg=False, get_pre_error=False):
        # compute A^T A with assembling of A
        M = assemble_A_from_stacks(core_pos, LeftStacks, RightStacks)
        if linear_extension is not None:
            M = torch.cat([M, B], dim=1)
        G = M.T @ M
        My = M.T @ y

        # only compute pre error at first step
        if verbose_adaptive_reg or get_pre_error:
            if len(losses) == 0:
                # current solution:
                if linear_extension is not None:
                    solution_pre = torch.concatenate([xFTT_t.xTT.tt.comps[core_pos].flatten(), xFTT_t.c_linear])
                else:
                    solution_pre = xFTT_t.xTT.tt.comps[core_pos].flatten()
                residual = M @ solution_pre.flatten() - y.flatten()
                loss_pre = torch.sum(residual * residual).item() / b  # = ||residual||^2 / b
                reg_pre = torch.sum(solution_pre * solution_pre).item()  # = ||solution_pre||^2

        if reg_param is None or reg_param == 0.:
            solution = torch.linalg.solve(G, My)
        else:
            # ridge: [M; sqrt(lam) I] * sol = [y; 0] normal equation (M.T M + lam I) sol = y
            R = torch.as_tensor(reg_param, device=device, dtype=dtype) * torch.eye(M.shape[1], device=device, dtype=dtype)
            solution = torch.linalg.solve(G + R, My)

        residual = M @ solution.flatten() - y.flatten()
        loss_post = torch.sum(residual * residual).item() / b  # = ||residual||^2 / b
        reg_post = torch.sum(solution * solution).item()

        old_reg_param = reg_param
        reg_param = reg_magnitude_reduction * loss_post / reg_post

        if verbose_adaptive_reg:
            print("=====================")
            if len(losses) == 0:
                print("loss_pre = ", loss_pre)
                print("reg_pre = ", reg_pre)
            print("loss_post = ", loss_post)
            print("reg_post = ", reg_post)
            print(" updated suggested for reg_param = ", reg_param, " ( before : ", old_reg_param, ")")
            print("=====================")

            if len(losses) == 0:
                losses.append(loss_pre)
            if len(reg_contribs) == 0:
                reg_contribs.append(reg_pre)
            if len(reg_params_hist) == 0:
                reg_params_hist.append(old_reg_param)  # init reg param value

            losses.append(loss_post)
            reg_contribs.append(reg_post)
            reg_params_hist.append(reg_param)

        update(core_pos, solution)

        if get_error:
            # training error for monitoring (based on the M used)
            err = (torch.linalg.norm(M @ solution - y) ** 2) / b
            if get_pre_error:
                return err, loss_pre, reg_param
            else:
                return err, None, reg_param
        else:
            if get_pre_error:
                return None, loss_pre, reg_param
            else:
                return None, None, reg_param

    # ---------------------------
    # build the design matrix A for core 'core_pos' quickly (sum over i)
    # A has shape (b, r*m*R) with r/R the TT ranks at core_pos
    # ---------------------------
    def assemble_A_from_stacks(core_pos, LeftStacks, RightStacks):
        A = torch.einsum("ibr, ibm, ibR -> brmR", LeftStacks[core_pos][:, :, 0, :], W[core_pos], RightStacks[core_pos + 1][:, :, :, 0])
        A = A.reshape(b, -1)  # b x (r * m * R)
        return A

    def sweeping_with_adaptive_reg(reg_init):
        loss_values = []

        local_solve = local_solve_direct_reguarization_aware
        loss_post = None
        reg_param = reg_init

        # set non-orthogonal core at core_pos = 0
        xFTT_t.xTT.tt.set_core(0)

        loss_init = None
        # build right-stacks once (from the current TT state before the sweep)
        RightStacks = build_right_stacks()

        for k in range(iterations):
            # ---------- right sweep (0 .. d-2) ----------
            # init left-stacks (filled during the sweep)
            LeftStacks = init_left_stacks_empty()

            loss_post_from_compute = None
            for core_pos in range(d - 1):
                if k == 0 and core_pos == 0:
                    _, loss_pre, reg_param = local_solve(core_pos, LeftStacks, RightStacks, reg_param, get_error=False, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=True)
                    loss_init = loss_pre
                    loss_values.append(loss_pre)
                else:
                    _, _, reg_param = local_solve(core_pos, LeftStacks, RightStacks, reg_param, get_error=False, verbose_adaptive_reg=verbose_adaptive_reg, get_pre_error=False)
                # set non-orthogonal position at core_pos + 1 (moves core one position to the right)
                xFTT_t.xTT.tt.set_core(core_pos + 1)
                # extend Left by exactly this (updated) core
                push_left_one_pos(LeftStacks, core_pos)

            # ---------- left sweep (d-1 .. 1) ----------
            # for the left sweep we reuse the already filled LeftStacks
            # and build Right incrementally, starting at pos=d (identity)
            RightStacks = init_right_stacks_empty()

            for core_pos in range(d - 1, 0, -1):
                # only compute loss_post at the last iteration
                loss_post_from_compute, _, reg_param = local_solve(core_pos, LeftStacks, RightStacks, reg_param, get_error=(core_pos == 1), verbose_adaptive_reg=verbose_adaptive_reg)
                # set non-orthogonal position at core_pos - 1 (moves core one position to the left)
                xFTT_t.xTT.tt.set_core(core_pos - 1)
                # extend Right by exactly this (updated) core (for the next step to the left)
                push_right_one_pos(RightStacks, core_pos)  # builds Right[i][core_pos-1] from Right[i][core_pos]

            loss_post = loss_post_from_compute
            loss_values.append(loss_post.item())
            if verbose:
                print(f"sweep {k}: loss = {loss_post} (prev {loss_pre})")

            # stopping criterion
            denom = abs(loss_pre) + 1e-12
            if abs(loss_pre - loss_post) / denom < tol:
                if verbose:
                    print(f"breaking at k = {k+1} / {iterations}, loss = {loss_post}, original loss = {loss_init}.")
                    print(f"         reg_param = {reg_param} with init reg_param  = {reg_param_init}.")
                break

            loss_pre = loss_post

        if verbose_adaptive_reg:
            plot_als_info(losses, reg_contribs, reg_params_hist, d)

        return loss_post, k, reg_param, loss_values

    if adaptive_reguarlization:
        return sweeping_with_adaptive_reg(reg_param)
    else:
        raise ValueError("Adaptive regularization has other return values and should not be used anymore.")


from ttd.bases.rank_rules import (
    Absolute_Singularvalue_Tresholding,
    Relative_Singularvalue_Tresholding,
)
from ttd.problems.base import get_Basis, get_basis_info


def optimize_and_choose_proper_basis_and_rank(surrogate_init, training_loss, delta=1e-5, loss_atol=1e-6, max_deg=30, verbose_level=0, number_basis_increasements=10):
    d = surrogate_init.xTT.d

    surrogate_former = deepcopy(surrogate_init)
    loss_value, iteration_number, reg_param_updated, loss_values = training_loss(surrogate_former)

    initial_degrees = [c.shape[1] - 1 for c in surrogate_former.xTT.tt.comps]

    reached_minimal_degree = initial_degrees == [2] * d

    if verbose_level > 0:
        print(" Initial loss value : ", loss_value, " with requestes loss_atol = ", loss_atol)
        print(" Initial basis functions degrees : ", initial_degrees)

    if loss_value < loss_atol:
        surrogate_trial = surrogate_former
        loss_value_trial = loss_value

        while loss_value_trial < loss_atol:
            surrogate_former = deepcopy(surrogate_trial)
            if reached_minimal_degree:
                break
            # TODO: generalize to refined basis reduction, here just uniform decrease
            # try to reduce basis functions
            former_xTT = surrogate_former.xTT

            basis_info = get_basis_info(former_xTT.tensor_basis_functions)
            old_degs = basis_info["deg"]
            basis_info["deg"] = [max(2, deg - 1) for deg in old_degs]

            reached_minimal_degree = basis_info["deg"] == [2] * d

            _, basis = get_Basis(basis_info)

            shrinked_comps_list = [c[:, :basis_info["deg"][i] + 1, :] for i, c in enumerate(former_xTT.tt.comps)]
            shrinked_dims = [c.shape[1] for c in shrinked_comps_list]

            shrinked_tt = TensorTrain(shrinked_dims, shrinked_comps_list, device=former_xTT.tt.device, deep_copy=True)
            shrinked_tt.rank_truncation(shrinked_tt.uranks)  # reduce to maximal possible rank
            shrinked_xtt = Extended_TensorTrain(basis, shrinked_tt.rank, shrinked_tt.comps, device=former_xTT.device)
            surrogate_trial.xTT = shrinked_xtt
            loss_value_trial, iteration_number_trial, reg_param_updated_trial, loss_values_trial = training_loss(surrogate_trial)

            if verbose_level:
                print("shrinked basis functions to deg: ", basis_info["deg"])
                print("found new loss : ", loss_value_trial, " with requestes loss_atol = ", loss_atol)

                if reached_minimal_degree:
                    print(" REACH MINIMAL DEGREE")

            if loss_value_trial < loss_atol or (reached_minimal_degree and loss_value_trial < loss_atol):
                if reached_minimal_degree:
                    if verbose_level > 0:
                        print("Basis functions degree reached minimum and loss still met loss tol: accepted.")
                elif verbose_level > 0:
                    print("Basis functions reduction still met loss tol, try to reduce further.")
                loss_value = loss_value_trial
                reg_param_updated = reg_param_updated_trial
                iteration_number = iteration_number_trial
                loss_values = loss_values_trial
            else:
                if verbose_level > 0:
                    if reached_minimal_degree:
                        print(" basis functions reduction reached minimal possible.")

                    if loss_atol <= loss_value_trial:
                        print(" basis functions reduction led to loss increase above tol, revert to former surrogate.")

        # loss atol no longer met; surrogate_former was the last candidate to reach loss_atol or the degree hit its minimum

        if verbose_level > 0:
            print(" Final basis functions degrees : ", [c.shape[1] - 1 for c in surrogate_former.xTT.tt.comps])
            if reached_minimal_degree:
                print("Stopp reason: Reached minimal polynomial degrees. ")
        if d > 1:
            if verbose_level > 0:
                print(" Now try to reduce ranks to reach proper ranks. ")

                for pos in range(surrogate_former.xTT.tt.n_comps - 1):
                    surrogate_former.xTT.tt.set_core(pos)
                    c = surrogate_former.xTT.tt.comps[pos]
                    s = c.shape
                    c = c.reshape(s[0] * s[1], s[2])
                    _, sigma, _ = torch.linalg.svd(c, full_matrices=False)
                    print(f"Core {pos}: relative singular values: {sigma/sigma[0]}", " for rtol_delta = ", delta)

            rank_rule = Relative_Singularvalue_Tresholding(delta, maxranks=[None] * (d - 1), dims=surrogate_former.xTT.tensor_basis_functions.ndofs, rankincr=0, verbose=verbose_level > 0)
            original_rank_list, new_rank = surrogate_former.xTT.modify_ranks(rank_rule, verbose=verbose_level > 1)
            if verbose_level > 0:
                print("Rank Analysis after basis reduction:")
                print("Update rank old rank = ", original_rank_list, " new rank = ", new_rank)

        return surrogate_former, loss_value, iteration_number, reg_param_updated, loss_values

    else:
        surrogate_trial = surrogate_former
        loss_value_trial = loss_value

        loss_history_trials = []

        while loss_value_trial > loss_atol:
            surrogate_former = deepcopy(surrogate_trial)

            # attempt degree increase, simply increase all for now
            # TODO: generalize to refined basis increase, here just uniform increase
            former_xTT = surrogate_former.xTT

            basis_info = get_basis_info(former_xTT.tensor_basis_functions)
            old_degs = basis_info["deg"]
            basis_info["deg"] = [min(max_deg, deg + 1) for deg in old_degs]

            reached_maximal_degree = basis_info["deg"] == [max_deg] * d

            _, basis = get_Basis(basis_info)

            increased_deg_comps_list = []
            for c in former_xTT.tt.comps:
                rm1, m, r = c.shape
                increased_deg_comps_list.append(torch.zeros((rm1, m + 1, r), device=former_xTT.device))
                increased_deg_comps_list[-1][:, :-1, :] = c
            increased_dims = [c.shape[1] for c in increased_deg_comps_list]
            increased_deg_tt = TensorTrain(increased_dims, increased_deg_comps_list, device=former_xTT.tt.device, deep_copy=True)
            increased_deg_tt.set_core(0)

            increased_deg_xtt = Extended_TensorTrain(basis, increased_deg_tt.rank, increased_deg_tt.comps, device=former_xTT.device)
            surrogate_trial.xTT = increased_deg_xtt
            loss_value_trial, iteration_number_trial, reg_param_updated_trial, loss_values_trial = training_loss(surrogate_trial)

            loss_history_trials.append(loss_value_trial)

            if verbose_level:
                print("increased basis functions to deg: ", basis_info["deg"])
                print("found new loss : ", loss_value_trial, " with requestes loss_atol = ", loss_atol)

            if loss_value_trial > loss_atol and not reached_maximal_degree:
                if verbose_level > 0:
                    print("Basis functions increasement still not met loss tol, try to increase further.")
                loss_value = loss_value_trial
                reg_param_updated = reg_param_updated_trial
                iteration_number = iteration_number_trial
                loss_values = loss_values_trial
            else:
                if verbose_level > 0:
                    if loss_atol > loss_value_trial:
                        print("Basis functions increase led to loss decrease above tol, accept trial surrogate.")

                        plt.semilogy(loss_history_trials, label="loss history")
                        plt.semilogy([0, len(loss_history_trials) - 1], [loss_atol, loss_atol], "--", label="target loss tolerance")
                        plt.legend()
                        plt.grid(True)
                        plt.show()

                        return surrogate_trial, loss_value, iteration_number, reg_param_updated, loss_values

                    if reached_maximal_degree:
                        print(f"Basis function reached maximum allowed degree, but loss tolerance is not met. loss_value = {loss_value_trial}, loss tol = {loss_atol}")

                        plt.semilogy(loss_history_trials, label="loss history")
                        plt.semilogy([0, len(loss_history_trials) - 1], [loss_atol, loss_atol], "--", label="target loss tolerance")
                        plt.legend()
                        plt.grid(True)
                        plt.show()
                        break

            # attempt rank increase

        # loss_value not met: should increase polynomial degree and ranks and track for proper ranks
        # TODO: not implemented yet
        raise NotImplementedError(" OTHER CASE NOT IMPLEMENTED YET. ")


def optimize_and_choose_proper_rank(xtt_surrogate, training_loss, delta=1e-5, number_of_rank_iterations=10, verbose_level=0):
    d = xtt_surrogate.xTT.d

    rank_rule = Absolute_Singularvalue_Tresholding(delta, maxranks=[None] * (d - 1), dims=xtt_surrogate.xTT.tensor_basis_functions.ndofs, rankincr=1, verbose=True)

    loss_post, iteration_number, reg_param_updated, loss_values = training_loss(xtt_surrogate)
    original_rank_list, new_rank = xtt_surrogate.xTT.modify_ranks(rank_rule, verbose=verbose_level > 1)

    original_rank = torch.tensor(original_rank_list, dtype=torch.int64)  # as tensor for special print

    if verbose_level > 1:
        print("1st Rank Analysis:")
        print("old rank = ", original_rank_list, " new rank = ", new_rank)
        print("loss error: ", loss_post)

    former_rank_collection = {pos: [original_rank_list[pos]] for pos in range(1, d)}

    max_ranks_adaptive = [1] + [None] * (d - 1) + [1]
    for pos in range(1, d):
        # if rank shortening detected break
        if new_rank[pos] < original_rank_list[pos]:
            max_ranks_adaptive[pos] = new_rank[pos]

    for k in range(number_of_rank_iterations):
        # if all max_ranks have been fixed, finish
        if None not in max_ranks_adaptive:
            if torch.equal(torch.tensor(max_ranks_adaptive[1:-1], dtype=torch.int64), original_rank):
                print("{c}No rank modification suggested!{r} Keep rank r = {c}{rank}{r}".format(rank=original_rank_list[1:-1], c=Fore.GREEN, r=Style.RESET_ALL))
            else:
                print("{c}Suggested rank update{r} : rank r = {c1}{rank}{r} -> {c}{rankn}{r}".format(rank=original_rank_list[1:-1], rankn=max_ranks_adaptive[1:-1], c1=Fore.RED, c=Fore.GREEN, r=Style.RESET_ALL))

            loss_post, iteration_number, reg_param_updated, loss_values = training_loss(xtt_surrogate)
            if verbose_level > 1:
                print("loss error: ", loss_post)
            return loss_post, iteration_number, reg_param_updated, loss_values

        loss_post, iteration_number, reg_param_updated, loss_values = training_loss(xtt_surrogate)
        # old_rank is the same as the former new_rank
        old_rank_list, new_rank_list = xtt_surrogate.xTT.modify_ranks(rank_rule, verbose=verbose_level > 1)

        if verbose_level > 1:
            print(f"{k+2}th Rank Analysis : ")
            print("old rank = ", old_rank_list, " new rank = ", new_rank_list)
            print("loss error : ", loss_post)

        for pos, former_ranks_at_pos in former_rank_collection.items():
            rank_at_pos = new_rank_list[pos]
            if new_rank_list[pos] in former_ranks_at_pos or new_rank_list[pos] < original_rank_list[pos]:
                if verbose_level > 1 and max_ranks_adaptive[pos] is None:
                    print("{c}Fixing rank at pos={pos}{r} : rank r = {c}{rank}{r}".format(pos=pos, rank=rank_at_pos, c=Fore.GREEN, r=Style.RESET_ALL))
                max_ranks_adaptive[pos] = rank_at_pos
            else:
                former_rank_collection[pos].append(rank_at_pos)

            # set a new rule, with sub-max-ranks possibly non-None
            rank_rule = Absolute_Singularvalue_Tresholding(delta, maxranks=max_ranks_adaptive[1:-1], dims=xtt_surrogate.xTT.tensor_basis_functions.ndofs, rankincr=1, verbose=True)

    if verbose_level > 0:
        print("{c1}Maximum number_of_rank_iterations reached.{r} Suggested rank: {c1}{rank}{r}".format(rank=new_rank_list, c1=Fore.RED, r=Style.RESET_ALL))
    # perform final fitting
    loss_post, iteration_number, reg_param_updated, loss_values = training_loss(xtt_surrogate)

    print(" in optimize and choose rank: ", xtt_surrogate.xTT.rank)
    if verbose_level > 1:
        print("loss error: ", loss_post)
    return loss_post, iteration_number, reg_param_updated, loss_values
