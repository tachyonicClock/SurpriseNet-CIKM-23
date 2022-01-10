from torch import Tensor
import torch


def dropout(input: Tensor, probability: Tensor, training: bool = True) -> Tensor:
    """
    Apply dropout where elements are zeroed or scaled with a probability.
    Adaptation of https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch

    Args:
        probability (Tensor): Probability of an element to be zeroed
        training (bool, optional): Is dropout in training or testing mode. Defaults to True.
    """
    if probability.shape != input.shape:
        raise ValueError("Input and probabilities should be the same shape")

    if training:
        binomial = torch.distributions.binomial.Binomial(probs=1.0-probability)
        return input * binomial.sample() * (1.0/(1.0-probability))
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
        return dropout(input, torch.ones(input.shape) * self.probability, self.training)

class ConditionedDropout(torch.nn.Module):
    """Dropout conditioned on groups"""

    group_ids: Tensor     # Group that a unit is assigned to
    active_group: int = 0 # The current group that is active

    p_active: float     # Probability of drop given the unit is in the group
    p_inactive: float    # Probability of drop given the unit is out of the group
    n_groups: int         # Total number of groups

    def __init__(self, in_features: int, n_groups: int, p_active: float, p_inactive: float) -> None:

        # Assign to each group with probability 1/n_groups aka roughly equally
        self.group_ids = torch.randint(n_groups, (in_features,))

        self.p_active = p_active
        self.p_inactive = p_inactive
        self.n_groups = n_groups

        super().__init__()

    def set_active_group(self, active_group: int):
        """Set the active group"""

        if 0 > active_group > self.n_groups:
            raise ValueError(f"The group {active_group} does not exist")

        self.active_group = active_group

    def forward(self, input: Tensor) -> Tensor:
        if self.group_ids.shape != input.shape:
            raise ValueError(f"Input a different shape then expected. Expected {self.group_ids.shape} got {input.shape}")
        
        # Mask true if they are in the active group
        mask = self.group_ids.eq(self.active_group)
        probability = mask * self.p_active + ~mask * self.p_inactive
        
        return dropout(input, probability, self.training)
