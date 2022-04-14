from torch import nn, Tensor
from network.trait import Classifier, PackNetComposite
import network.module.packnet as pn

class ClassifyHead(Classifier, nn.Module):

    def __init__(self, bottleneck_width: int, n_classes: int) -> None:
        super().__init__()
        self.lin = nn.Linear(bottleneck_width, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(x) 
    
    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x)

class PN_ClassifyHead(ClassifyHead, PackNetComposite):
    def __init__(self, bottleneck_width: int, n_classes: int) -> None:
        super().__init__(bottleneck_width, n_classes)
        self.lin = pn.wrap(self.lin)