
import torch
from torch import Tensor


def softTargetCrossEntropy(input: Tensor, target: Tensor) -> Tensor:
    """
    `softTargetCrossEntropy` supports probabilistic targets which are not 
    supported by `nn.CrossEntropy` 
    """
    assert target.shape == input.shape, \
        f"Expected input.shape ({input.shape}) to match target.shape ({target.shape})"
    log_probs = torch.nn.functional.log_softmax(input, dim = 1)
    return  -(target * log_probs).sum() / input.shape[0]