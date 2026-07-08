"""Diagnostics for the Gaussian-mixture targets in `ttd.problems.base`:
`ReferenceSolutionGaussianMixture` gives the closed-form, analytically-diffused
density/score at any time (no TT fit needed), used to visualize and quantify error
against a trained xTT approximation (`animate_comparison`, `coeff_projection_test`).
"""

from math import exp, sqrt

import matplotlib.pyplot as plt
import torch
from matplotlib import animation

from ttd.problems.base import GaussianMixture, SymmetricGaussianMixture2D
from ttd.utils.sampling import get_normalization


class ReferenceSolutionGaussianMixture(object):

    def __init__(self, target):
        assert isinstance(target, SymmetricGaussianMixture2D) or isinstance(target, GaussianMixture)
        self.target = target
        self.dim = self.target.dim
        self.means = self.target.means
        self.numPoints = len(self.means)
        if isinstance(target, SymmetricGaussianMixture2D):
            self.covs = [torch.eye(self.dim) * 1. / var for var in self.target.vars]
        elif isinstance(target, GaussianMixture):
            self.covs = [cov for cov in target.covs]

    def nlog_pi(self, t):
        means_t = [mu * exp(-t) for mu in self.means]
        covs_t = [(exp(-2 * t)) * cov + (1 - exp(-2 * t)) * torch.eye(self.dim) for cov in self.covs]
        return GaussianMixture(means_t, covs_t)

    def nlog(self, x, t):
        nlog_pi_t = self.nlog_pi(t)
        return nlog_pi_t(x)

    def score(self, x, t):
        nlog_pi_t = self.nlog_pi(t)
        return nlog_pi_t.grad(x)

    def pi(self, t):
        return lambda x: torch.exp(-self.nlog_pi(t)(x))

    def sample_pi(self, t, N):
        """generate N samples from the posterior exp(-Phi) defined by the potential.

        Args:
            N (int): number of samples

        Returns:
            samples (torch.tensor): sample matrix of shape (D,N)
        """
        means_t = [mu * exp(-t) for mu in self.means]
        covs_t = [(exp(-2 * t)) * cov + (1 - exp(-2 * t)) * torch.eye(self.dim) for cov in self.covs]
        Ls_t = [torch.linalg.cholesky(cov) for cov in covs_t]

        samples = torch.zeros((N, 2))
        # uniform distribution over the individual Gaussians
        indices = torch.randint(low=0, high=self.numPoints, size=(N,))
        counts = torch.bincount(indices).tolist()
        counts += [0] * (self.numPoints - len(counts))
        counts = torch.tensor(counts)
        last_count = 0
        count = 0
        for i in range(self.numPoints):
            count += counts[i]
            # samples[:,last_count:count] = gaussians[i].sample((2,counts[i]))
            samples[last_count:count, :] = means_t[i][None, :] + torch.einsum("ij,bj->bi", Ls_t[i], torch.randn((counts[i], 2)))
            last_count += counts[i]

        return samples

    def targetVector(self, t, sigma_vals, x):
        return self.nlog(x, t) + torch.einsum("bd,bd->b", sigma_vals, self.score(x, t))


def animate_comparison(times, ref, apprx, apprx_score, finalTime, normalized_ref, xlim, ylim, full_times=None, max_z=0.25):

    X, Y = torch.meshgrid(torch.linspace(-3, 3, 50), torch.linspace(-3, 3, 50))
    x = torch.stack([X.flatten(), Y.flatten()]).T

    ref_data = {}
    apprx_data = {}
    error_data = {}

    for t in times:
        if normalized_ref:
            Z1 = ref.pi(finalTime - t)(x).reshape(X.shape)
        else:
            nlog_ref = ref.nlog_pi(finalTime - t)
            norm = get_normalization(nlog_ref, xlim, ylim, mode="nquad")
            print("normalization of ref at t={t} = {norm}".format(t=t, norm=norm))
            Z1 = 1. / norm * torch.exp(-nlog_ref(x).reshape(X.shape))

        ref_data[finalTime - t] = Z1

        Z2 = apprx(t, x).reshape(X.shape)
        apprx_data[t] = Z2
        error_data[t] = torch.abs(Z1 - Z2)

    error_times = times if full_times is None else full_times
    score_errors = []
    time_to_score_error_dict = {}
    for stime in error_times:
        N = 1000
        testVals = ref.sample_pi(finalTime - stime, N)
        error = torch.sqrt(sum(torch.linalg.norm(ref.score(testVals, finalTime - stime) - apprx_score(testVals, stime), dim=1) ** 2) / N)
        score_errors.append(error)
        print(f"L2 error at time {stime} is {error}")
        time_to_score_error_dict[stime] = error

    vmin, vmax = torch.tensor(torch.inf), -torch.tensor(torch.inf)
    for _, err in error_data.items():
        vmin = torch.min(vmin, torch.min(err))
        vmax = torch.max(vmax, torch.max(err))

    fig = plt.figure(figsize=(26, 6))

    levels = torch.linspace(vmin, vmax, 41)
    kw = dict(levels=levels.detach().numpy(), cmap="coolwarm", vmin=vmin, vmax=vmax, origin="lower")

    # animate(times[0])
    gca = fig.add_subplot(1, 4, 4, projection="3d")
    map = gca.contourf(X.detach().numpy(), Y.detach().numpy(), error_data[times[0]].numpy(), **kw)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(map, cax=cbar_ax)

    # gca = fig.add_subplot(1, 4, 3)
    # gca.plot(error_times, score_errors, "-k", label="Score L2(mu_t) error")

    def animate(t):

        print(t)
        # plt.gca().set_title("                            ")

        gca = fig.add_subplot(1, 4, 1, projection="3d")
        # gca.clear()
        Z1 = ref_data[finalTime - t]
        # Plot the 3D surface
        gca.plot_surface(X, Y, Z1, edgecolor="royalblue", lw=0.5, rstride=4, cstride=4, alpha=0.3)
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph
        gca.contourf(X.detach().numpy(), Y.detach().numpy(), Z1.detach().numpy(), zdir="z", offset=-2 * max_z, cmap="coolwarm")
        gca.contourf(X.detach().numpy(), Y.detach().numpy(), Z1.detach().numpy(), zdir="x", offset=xlim[0], cmap="coolwarm")
        gca.contourf(X.detach().numpy(), Y.detach().numpy(), Z1.detach().numpy(), zdir="y", offset=ylim[1], cmap="coolwarm")
        # gca.set_title("t =" + str(t))
        gca.set(xlim=xlim, ylim=ylim, zlim=(-1, max_z),
            xlabel="X", ylabel="Y", zlabel="Z")

        # update

        # plt.gca().set_title("                            ")

        gca = fig.add_subplot(1, 4, 2, projection="3d")
        # gca.clear()
        Z2 = apprx_data[t]
        # Plot the 3D surface
        gca.plot_surface(X, Y, Z2, edgecolor="royalblue", lw=0.5, rstride=4, cstride=4, alpha=0.3)
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph
        gca.contourf(X.detach().numpy(), Y.detach().numpy(), Z2.detach().numpy(), zdir="z", offset=-2 * max_z, cmap="coolwarm")
        gca.contourf(X.detach().numpy(), Y.detach().numpy(), Z2.detach().numpy(), zdir="x", offset=xlim[0], cmap="coolwarm")
        gca.contourf(X.detach().numpy(), Y.detach().numpy(), Z2.detach().numpy(), zdir="y", offset=ylim[1], cmap="coolwarm")
        # gca.set_title("t =" + str(t))
        gca.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-1, max_z),
            xlabel="X", ylabel="Y", zlabel="Z")

        gca = fig.add_subplot(1, 4, 3)
        gca.plot(error_times, score_errors, "-k", label="Score L2(mu_t) error")
        gca.plot([t], time_to_score_error_dict[t], "ob", markersize=5)

        # plt.gca().set_title("                            ")
        gca = fig.add_subplot(1, 4, 4)
        # gca.clear()
        err = error_data[t]
        map = gca.contourf(X.detach().numpy(), Y.detach().numpy(), err.detach().numpy(), **kw)
        # plt.colorbar(map, label="pointwise error", orientation="vertical")

    forwardbackwardtimes = times  # + times[::-1]

    ani = animation.FuncAnimation(fig, animate, interval=800, frames=forwardbackwardtimes)
    ani.save("comparison.gif", writer="pillow")


def coeff_projection_test():
    # log rho_t
    #
    #
    #   -log rho_*   -x^2/2

    #  int   log rho_t  *v rho_t  = int v w rho_t
    #              RHS            =  Galerkin mass matrix
    #                 2                   2x2
    #   c_start_t   c_tar_t
    #

    target = SymmetricGaussianMixture2D(means=[torch.tensor([sqrt(2.), sqrt(2.)]),
                                              torch.tensor([sqrt(2.), -sqrt(2.)]),
                                              torch.tensor([-sqrt(2.), sqrt(2.)]),
                                              torch.tensor([-sqrt(2.), -sqrt(2.)])],
                                    vars=[4., 4., 4., 4.])

    start = lambda x: 0.5 * torch.einsum("bi,bi->b", x, x)

    reference = ReferenceSolutionGaussianMixture(target)

    target = reference.nlog_pi(0)
    start = reference.nlog_pi(200)

    tar_grad = lambda x: reference.score(x, 0)
    start_grad = lambda x: reference.score(x, 200)

    N = 50000

    t = 0.5
    T = 10.
    ts = torch.linspace(0, T, 30)

    c_tar = []
    c_start = []
    score_errs = []
    value_errs = []

    for t in ts:
        # samples = reference.sample_pi(t, N)

        print("at t = ", t.item())

        fspace = [target, start]
        fgrad_space = [tar_grad, start_grad]

        A = torch.zeros(len(fspace), len(fspace))
        b = torch.zeros(len(fspace),)

        samples = reference.sample_pi(T - t, N)

        for i, v_t in enumerate(zip(fspace, fgrad_space)):
            v, vgrad = v_t
            for j, w_t in enumerate(zip(fspace, fgrad_space)):
                w, wgrad = w_t
                # samples = reference.sample_pi(T - t, N)
                A[i, j] = 1. / N * sum(v(samples) * w(samples))  \
                #         + 1. / N * sum(torch.einsum("bi,bi->b", vgrad(samples), wgrad(samples)))

                # empirical inner product of L^2(rho_t)

            samples = reference.sample_pi(T - t, N)
            b[i] = 1. / N * sum(reference.nlog_pi(T - t)(samples) * v(samples))

        c_t = torch.linalg.solve(A, b)
        c_tar.append(c_t[0])
        c_start.append(c_t[1])

        samples = reference.sample_pi(T - t, N)

        print("Do score computation")
        samples = reference.sample_pi(T - t, N)
        diff_value = reference.nlog(samples, T - t) - (c_t[0] * reference.nlog(samples, 0) + c_t[1] * reference.nlog(samples, 200))
        value_fct_err = torch.sqrt(sum(torch.abs(diff_value) / N))
        value_errs.append(value_fct_err)

        diff_score = reference.score(samples, T - t) - (c_t[0] * reference.score(samples, 0) + c_t[1] * reference.score(samples, 200))
        score_err = torch.sqrt(sum(torch.norm(diff_score, dim=1)) / N)
        score_errs.append(score_err)

    print("c_tar : \n", c_tar)
    print("c_start : \n", c_start)
    print("FINISHED computation")

    plt.subplot(1, 3, 1)
    plt.plot(ts, c_start, "b-", label="c_start")
    plt.plot(ts, c_tar, "r-", label="c_target")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(ts, value_errs, "b-", label="value error")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(ts, score_errs, "b-", label="score error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    coeff_projection_test()

