"""Plot helpers: a 2D vector-field plot, a histogram/scatter comparison of two sample
sets, and the analytic score of a Gaussian.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch


def vectorplot(domain, gridWidth, vectorField, ax, key):
    # plt.clf()
    assert len(domain) == 2
    feature_x = np.arange(domain[0][0], domain[0][1], gridWidth)
    feature_y = np.arange(domain[1][0], domain[1][1], gridWidth)

    x, y = np.meshgrid(feature_x, feature_y)
    z_long = np.array(list(itertools.product(feature_x, feature_y)))
    z_long = np.array(vectorField(torch.tensor(z_long)), dtype=np.float64)

    # plot normalized direction vectors
    # u = (z_long[:,0]/np.linalg.norm(z_long,axis=1)*gridWidth).reshape((len(feature_x), len(feature_y))).T
    # v = (z_long[:,1]/np.linalg.norm(z_long,axis=1)*gridWidth).reshape((len(feature_x), len(feature_y))).T

    scale = 3. * np.max(np.linalg.norm(z_long, axis=1))

    u = (z_long[:, 0] / scale).reshape((len(feature_x), len(feature_y))).T
    v = (z_long[:, 1] / scale).reshape((len(feature_x), len(feature_y))).T

    # ax.set_aspect(1)
    # ax.plot(feature_x, feature_y, c='k')
    ax.clear()
    ax.set_title(f"Vector plot at time {key}")
    ax.quiver(x, y, u, v, units="xy", scale=0.5, color="gray")


def plotSamples(samples, samples2=None, label=0, bins=100, label1="samples", label2="target samples"):

    if samples.shape[1] == 1:
        # plt.figure()
        # plt.plot(samples[:,0],samples[:,0]*0, 'b.',markersize=2, label=label1)
        plt.hist(samples[:, 0], bins=bins, density=True, alpha=0.5, label="Histogram of " + label1)
        if samples2 is not None:
            plt.hist(samples2[:, 0], bins=bins, density=True, alpha=0.5, label="Histogram of " + label2)
            # plt.plot(samples2[:,0],samples2[:,0]*0,'r.',markersize=2, label="target samples")
        plt.legend()
        plt.savefig(f"samples{label}.png")
        plt.close()
        return
    elif samples.shape[1] == 2:
        print("Plotting with label ", label)
        # plt.figure()
        plt.plot(samples[:, 0], samples[:, 1], "b.", markersize=2, label=label1)
        if samples2 is not None:
            plt.plot(samples2[:, 0], samples2[:, 1], "r.", markersize=2, label=label2)
        # if label is None:
        # plt.legend([label1])
        plt.legend()
        # else:
        #     plt.legend([f"samples after {label} iterations"])
        plt.savefig(f"samples{label}.png")
        plt.close()
    else:
        print("Plotting samples with shape ", samples.shape)
        raise NotImplementedError("Plotting for samples with more than 2 dimensions is not implemented yet.")


def gradLogGaussian(x, t, Sigma, mean):
    dim = x.shape[1]
    Sigma_t = np.exp(-2 * t) * Sigma + (1 - np.exp(-2 * t)) * np.eye(dim)
    M_t = np.exp(-t) * mean
    # Z = np.sqrt(np.linalg.det(2*np.pi*Sigma_t))
    diff = x - M_t    # b x d
    # log =  np.einsum("bi,ij,bj -> b", diff, np.linalg.inv(Sigma_t), diff)
    gradlog = -np.einsum("ij,bj -> bi", np.linalg.inv(Sigma_t), diff)
    # return np.exp(-0.5 * log) / Z
    return gradlog
