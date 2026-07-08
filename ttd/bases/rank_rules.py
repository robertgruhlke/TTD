"""Rank-update heuristics used by `Extended_TensorTrain.modify_ranks`: given the singular
values of a TT core, decide the new rank at that position (Dörfler marking, relative/
absolute singular-value thresholding).
"""

import torch
from colorama import Fore, Style


def prod(values):
    res = values[0]
    for i in range(1, len(values)):
        res = res * values[i]
    return res


class Rule(object):
    def __eval__(self, sigma, pos):
        pass


class Threshold(Rule):
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, u, sigma, v, pos):
        return torch.max([torch.sum(sigma > self.delta), 1])


class Dörfler_Adaptivity(Rule):
    """
        Adaptivity rank rule:

            - Dörfler condition is fulfilled if there exists L s.t.:

                    delta * (sum k=0^L  sigma[k]**2) >= sum_{k=L+1} sigma[k]**2

            - ranks have an upper bound
            - if the Dörfler condition holds, cutoff all singular values with index > L+1.
              In particular keep sigma[L+1] as a threshold singular value keeping track
              of a max rank needed. It can be rounded later.

            - if the Dörfler condition is not fulfilled for any L in [0,...,len(sigma)-1],
              then the new rank is increased or stays the same, i.e.

                    new rank = min(maxrank, oldrank + rankincr*)

              here
                    rankincr* = min(rankincr, max possible rank increase)

              with
                    max possible rank increase = difference of shapes of v
    """
    def __init__(self, delta, maxranks, dims, rankincr=2, verbose=False):
        self.delta = delta
        self.rankincr = rankincr
        self.verbose = verbose

        self.maxranks = [0] * (len(dims) - 1)
        if None in maxranks:
            for k in range(len(self.maxranks)):
                urank = min(prod(dims[:k + 1]), prod(dims[k + 1:]))
                if maxranks[k] is None:
                    self.maxranks[k] = urank
                else:
                    self.maxranks[k] = urank if maxranks[k] > urank else maxranks[k]
        else:
            self.maxranks = [0.] * len(maxranks)
            for k in range(len(maxranks)):
                urank = min(prod(dims[:k + 1]), prod(dims[k + 1:]))  # upper bound of rank due to the unfolding
                if maxranks[k] > urank:
                    print("Warning upper limit of rank exceeded ({a} > {b}). Choose upper limit as max rank.".format(a=maxranks[k], b=urank))
                    self.maxranks[k] = urank
                else:
                    self.maxranks[k] = maxranks[k]

    def __call__(self, u, sigma, v, pos):
        umax = abs(max(u.shape[0], u.shape[1]) - len(sigma))

        for k in range(1, len(sigma)):
            l = torch.sum(sigma[:k] ** 2)
            r = torch.sum(sigma[k:] ** 2)

            if self.delta ** 2 * l >= r:
                if self.verbose:
                    print("Dörfer rank recommendation: {c1} old rank = {r1} {r} -> {c2}new rank = {r2}{r}".format(r=Style.RESET_ALL, c1=Fore.RED, c2=Fore.GREEN, r1=len(sigma), r2=k))
                return k

        # else a rank increase is performed
        newrank = min(self.maxranks[pos], len(sigma) + min(umax, self.rankincr))

        if self.verbose:
            if len(sigma) == newrank:
                print("Dörfer rank recommendation: {c} old rank = new rank = {rank}{r}".format(r=Style.RESET_ALL, c=Fore.RED, rank=len(sigma)))
            else:
                print("Dörfer rank recommendation: {c1} old rank = {r1} {r} -> {c2}new rank = {r2}{r}".format(r=Style.RESET_ALL, c1=Fore.RED, c2=Fore.GREEN, r1=len(sigma), r2=newrank))

        return newrank


class Relative_Singularvalue_Tresholding(Rule):

    def __init__(self, delta, maxranks, dims, rankincr=2, verbose=False):
        self.delta = delta
        self.rankincr = rankincr
        self.verbose = verbose

        self.maxranks = [0] * (len(dims) - 1)
        if None in maxranks:
            for k in range(len(self.maxranks)):
                urank = min(prod(dims[:k + 1]), prod(dims[k + 1:]))
                if maxranks[k] is None:
                    self.maxranks[k] = urank
                else:
                    self.maxranks[k] = urank if maxranks[k] > urank else maxranks[k]
        else:
            self.maxranks = [0.] * len(maxranks)
            for k in range(len(maxranks)):
                urank = min(prod(dims[:k + 1]), prod(dims[k + 1:]))  # upper bound of rank due to the unfolding
                if maxranks[k] > urank:
                    print("Warning upper limit of rank exceeded ({a} > {b}). Choose upper limit as max rank.".format(a=maxranks[k], b=urank))
                    self.maxranks[k] = urank
                else:
                    self.maxranks[k] = maxranks[k]

    def __call__(self, u, sigma, v, pos):
        umax = abs(max(u.shape[0], u.shape[1]) - len(sigma))
        for k in range(1, len(sigma)):
            if sigma[k] / sigma[0] < self.delta:
                if self.verbose:
                    print("Relative_Singularvalue_Tresholding rank recommendation: {c1} old rank = {r1} {r} -> {c2}new rank = {r2}{r}".format(r=Style.RESET_ALL, c1=Fore.RED, c2=Fore.GREEN, r1=len(sigma), r2=k))
                return k

        # else a rank increase is performed
        newrank = min(self.maxranks[pos], len(sigma) + min(umax, self.rankincr))
        if self.verbose:
            if len(sigma) == newrank:
                print("Relative_Singular rank recommendation: {c} old rank = new rank = {rank}{r}".format(r=Style.RESET_ALL, c=Fore.RED, rank=len(sigma)))
            else:
                print("Relative_Singular rank recommendation: {c1} old rank = {r1} {r} -> {c2}new rank = {r2}{r}".format(r=Style.RESET_ALL, c1=Fore.RED, c2=Fore.GREEN, r1=len(sigma), r2=newrank))

        return newrank


class Absolute_Singularvalue_Tresholding(Rule):

    def __init__(self, delta, maxranks, dims, rankincr=2, verbose=False):
        self.delta = delta
        self.rankincr = rankincr
        self.verbose = verbose

        self.maxranks = [0] * (len(dims) - 1)
        if None in maxranks:
            for k in range(len(self.maxranks)):
                urank = min(prod(dims[:k + 1]), prod(dims[k + 1:]))
                if maxranks[k] is None:
                    self.maxranks[k] = urank
                else:
                    self.maxranks[k] = urank if maxranks[k] > urank else maxranks[k]
        else:
            self.maxranks = [0.] * len(maxranks)
            for k in range(len(maxranks)):
                urank = min(prod(dims[:k + 1]), prod(dims[k + 1:]))  # upper bound of rank due to the unfolding
                if maxranks[k] > urank:
                    print("Warning upper limit of rank exceeded ({a} > {b}). Choose upper limit as max rank.".format(a=maxranks[k], b=urank))
                    self.maxranks[k] = urank
                else:
                    self.maxranks[k] = maxranks[k]

    def __call__(self, u, sigma, v, pos):
        if self.verbose:
            print("singular value at pos %d" % pos, sigma)
            print("ratio", sigma[1:] / sigma[0])
            print("delta", self.delta)

        umax = abs(max(u.shape[0], u.shape[1]) - len(sigma))
        for k in range(1, len(sigma)):
            if sigma[k] < self.delta:
                if self.verbose:
                    print("Absolute Singularvalue_Tresholding rank recommendation: {c1} old rank = {r1} {r} -> {c2}new rank = {r2}{r}".format(r=Style.RESET_ALL, c1=Fore.RED, c2=Fore.GREEN, r1=len(sigma), r2=k))
                return k

        # else a rank increase is performed
        newrank = min(self.maxranks[pos], len(sigma) + min(umax, self.rankincr))
        if self.verbose:
            if len(sigma) == newrank:
                print("Absolute Singular rank recommendation: {c} old rank = new rank = {rank}{r}".format(r=Style.RESET_ALL, c=Fore.RED, rank=len(sigma)))
            else:
                print("Absolute Singular rank recommendation: {c1} old rank = {r1} {r} -> {c2}new rank = {r2}{r}".format(r=Style.RESET_ALL, c1=Fore.RED, c2=Fore.GREEN, r1=len(sigma), r2=newrank))

        return newrank
