import array
from torch import Tensor
import random
import torch


def dropout(input: Tensor, probability: Tensor, batch_size: int = 1, training: bool = True) -> Tensor:
    """
    Apply dropout where elements are zeroed or scaled with a probability.
    Adaptation of https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch

    Args:
        probability (Tensor): Probability of an element to be zeroed
        training (bool, optional): Is dropout in training or testing mode. Defaults to True.
    """
    if training:
        binomial = torch.distributions.binomial.Binomial(probs=1.0-probability)
        return input * binomial.sample([batch_size]) * (1.0/(1.0-probability))
    return input


class NaiveDropout(torch.nn.Module):
    """
    Naive dropout similar torch.nn.dropout()
    """

    probability: float

    def __init__(self, probability: float = 0.5) -> None:
        super().__init__()
        if 1.0 < probability < 0.0:
            raise ValueError("Probability should be between 0 and 1")
        self.probability = probability

    def forward(self, input: Tensor) -> Tensor:
        probabilities = torch.ones(input.shape[1], device=input.device) * self.probability
        batch_size = input.shape[0]
        return dropout(input, probabilities, batch_size, self.training)

class ConditionedDropout(torch.nn.Module):
    """Dropout conditioned on groups"""

    group_ids: Tensor     # Group that a unit is assigned to
    active_group: int = 0 # The current group that is active

    p_active: float     # Probability of drop given the unit is in the group
    p_inactive: float    # Probability of drop given the unit is out of the group
    n_groups: int         # Total number of groups

    def __init__(self, in_features: int, n_groups: int, p_active: float, p_inactive: float) -> None:


        group_ids = torch.zeros(in_features)
        # Assign units to groups equally
        for i, _ in enumerate(group_ids):
            group_ids[i] = i % n_groups

        # Randomize assignment by shuffling
        group_ids = group_ids[torch.randperm(group_ids.size(0))]

        self.p_active = p_active
        self.p_inactive = p_inactive
        self.n_groups = n_groups

        super().__init__()
        self.register_buffer("group_ids", group_ids)


    def set_active_group(self, active_group: int):
        """Set the active group"""

        if 0 > active_group > self.n_groups:
            raise ValueError(f"The group {active_group} does not exist")

        self.active_group = active_group

    def forward(self, input: Tensor) -> Tensor:
        # Mask true if they are in the active group
        if input.dim() != 2:
            raise ValueError(f"Expected only 2D got {input.dim()}D")

        mask = self.group_ids.eq(self.active_group)
        probability = mask * self.p_active + ~mask * self.p_inactive
    
        return dropout(input, probability, input.shape[0], self.training)
