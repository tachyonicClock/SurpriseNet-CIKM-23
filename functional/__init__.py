import torch
from torch import Tensor

def best_reduce(metric: Tensor, values: Tensor) -> Tensor:
    """
    Using an at least 2D metric array we want to return a tensor concatinating
    all the best results
    """
    best = metric.T.argmin(dim=(1))
    tensors = torch.stack([values[b][i] for i, b in enumerate(best)])
    return tensors