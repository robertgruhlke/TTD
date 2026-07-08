"""Miscellaneous small numerical helpers."""

import torch


def nearest_index(t_list: torch.Tensor, t: torch.Tensor):
    """
    Find the index of the nearest value from t_list for each t.
    Assumption: t in [t_list[0], t_list[-1]].

    Args:
        t_list : 1D Tensor [n], sorted
        t      : Tensor [batch], values in [t0, tN]

    Returns:
        indices : Tensor [batch] with the nearest-neighbour indices
    """
    device = t.device
    t_list = t_list.to(device)

    # Binary search
    j = torch.searchsorted(t_list, t, right=True)  # candidate to the right
    j_left = torch.clamp(j - 1, 0, len(t_list) - 1)
    j_right = torch.clamp(j, 0, len(t_list) - 1)

    # Compare distances
    dist_left = torch.abs(t - t_list[j_left])
    dist_right = torch.abs(t - t_list[j_right])

    choose_right = dist_right < dist_left
    indices = torch.where(choose_right, j_right, j_left)

    return indices
