"""Sample-generation utilities: `get_normalization` (2D numerical-quadrature
normalizing constant) and `rejectionSampler` (rejection sampling given a known
majorant of the target/proposal density ratio).
"""

import torch
from scipy import integrate


def get_normalization(target, xlim, ylim, mode="nquad"):

    rho = lambda x0, x1: target.unnormalizedDensity(torch.tensor([[x0, x1]]))

    x0, x1 = xlim
    y0, y1 = ylim

    len_x = x1 - x0
    len_y = y1 - y0

    if mode == "nquad":
        res = integrate.nquad(rho, [[x0, x1], [y0, y1]],
                    full_output=True)
        val, err, _ = res
        print("quad error = ", err)

    elif mode == "rectangle":
        N = 2000

        area = (len_x) / N * (len_y) / N
        val = 0.

        xmids = torch.tensor([x0 + i * (len_x) / N + (len_x) / (2 * N) for i in range(N)])
        ymids = torch.tensor([y0 + j * (len_y) / N + (len_y) / (2 * N) for j in range(N)])

        X, Y = torch.meshgrid(xmids, ymids)

        x = torch.stack([X.flatten(), Y.flatten()]).T
        print(x.shape)
        val = sum(target.unnormalizedDensity(x)) * area

    elif mode == "MC":
        N = 5000
        x = torch.rand((N, 2)) - 0.5  # * torch.tensor([[2*xlim],[2*ylim]])
        x[:, 0] *= len_x
        x[:, 1] *= len_y

        area = len_x * len_y
        val = torch.sum(target.unnormalizedDensity(x)) / N * area

    else:
        raise NotImplementedError("Other quadrature schemes are not implemented")

    return val


def rejectionSampler(N, target_density, proposal_density, M):
    target_samples = []
    while len(target_samples) < N:
        # print("rejection loop start")
        N_new = N - len(target_samples)

        proposal_samples = proposal_density.sample(N_new)

        # proposal_samples = torch.randn(N_new, 2)

        U = torch.rand(N_new)
        Q = target_density(proposal_samples) / (M * proposal_density.normalizedDensity(proposal_samples))

        indices = U < Q
        added_samples = proposal_samples[indices]

        if len(target_samples) == 0:
            target_samples = added_samples
        else:
            target_samples = torch.cat([target_samples, added_samples], dim=0)
    return target_samples

