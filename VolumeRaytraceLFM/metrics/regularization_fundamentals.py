'''Regularization functions that can use used in the optimization process.'''
import torch


def l1(data):
    return torch.abs(data).mean()


def l2(data):
    return torch.pow(data, 2).mean()


def linfinity(data, weight=1.0):
    return weight * torch.max(torch.abs(data))


def elastic_net(data, weight1=1.0, weight2=1.0):
    l1_term = torch.abs(data).sum()
    l2_term = torch.pow(data, 2).sum()
    return weight1 * l1_term + weight2 * l2_term


def total_variation_3d_volumetric(data):
    """
    Computes the Total Variation regularization for a 4D tensor representing volumetric data.
    Args:
        data (torch.Tensor): Input 3D tensor with shape [depth, height, width].
    Returns:
        torch.Tensor: Computed Total Variation regularization term.
    """
    # Calculate the differences between adjacent elements along each spatial dimension
    diff_depth = torch.pow(data[1:, :, :] - data[:-1, :, :], 2).mean()
    diff_height = torch.pow(data[:, 1:, :] - data[:, :-1, :], 2).mean()
    diff_width = torch.pow(data[:, :, 1:] - data[:, :, :-1], 2).mean()

    tv_reg = diff_depth + diff_height + diff_width
    return tv_reg
