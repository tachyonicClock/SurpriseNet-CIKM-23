import io
from matplotlib.figure import Figure
import torch
from torch import Tensor
from torch.nn import functional as F
from PIL import Image

def best_reduce(metric: Tensor, values: Tensor) -> Tensor:
    """
    Using an at least 2D metric array we want to return a tensor concatinating
    all the best results
    """
    best = metric.T.argmin(dim=(1))
    tensors = torch.stack([values[b][i] for i, b in enumerate(best)])
    return tensors

def MSE(x_hat: Tensor, x: Tensor) -> Tensor:
    """Mean sum of pixel squared error"""
    loss = F.mse_loss(x, x_hat, reduction="none")
    loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss

def MRAE(x_hat: Tensor, x: Tensor, reduce_batch=True) -> Tensor:
    """Relative absolute error"""
    start = 0 if reduce_batch else 1
    x = x.flatten(start)
    x_hat = x_hat.flatten(start)
    return (x_hat - x).abs().sum(dim=[1])/x.abs().sum(dim=[1])

def figure_to_image(fig: Figure) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)
