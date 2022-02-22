import typing
from torch import Tensor

# An example that a supervised model can learn from. Contains a label and an input
LabeledExample = typing.Tuple[int, Tensor]

