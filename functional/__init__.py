import torch
from torch import Tensor
from torch.nn import functional as F

def best_reduce(metric: Tensor, values: Tensor) -> Tensor:
    """
    Using an at least 2D metric array we want to return a tensor concatinating
    all the best results
    """
    best = metric.T.argmin(dim=(1))
    tensors = torch.stack([values[b][i] for i, b in enumerate(best)])
    return tensors

def vae_kl_loss(mu, log_var):
    """
    Kullback-Leibler divergence how similar is the sample distribution to a normal distribution
    """
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)