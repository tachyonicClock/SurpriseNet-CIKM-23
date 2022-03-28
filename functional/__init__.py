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

def recon_loss(x_hat: Tensor, x: Tensor) -> Tensor:
    # Mean sum of pixel squared error
    loss = F.mse_loss(x, x_hat, reduction="none")
    loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss

def figure_to_image(fig: Figure) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)
