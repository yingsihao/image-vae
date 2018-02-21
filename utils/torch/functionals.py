import torch
import torch.nn.functional as F

from .misc import to_var


def conv_cross2d(inputs, weights, bias = None, stride = 1, padding = 0, dilation = 1, groups = 1):
    outputs = []
    for input, weight in zip(inputs, weights):
        outputs.append(F.conv2d(input.unsqueeze(0), weight, bias, stride, padding, dilation, groups))
    return torch.cat(outputs, 0)


def gaussian_sampler(mean, log_var):
    eps = to_var(torch.normal(torch.zeros(mean.size()), torch.ones(mean.size())))
    return mean + eps * torch.exp(log_var / 2.)


def kld_loss(mean, log_var):
    return -.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var)) / mean.size(0)
