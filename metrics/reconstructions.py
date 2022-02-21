from typing import Dict
import typing
from matplotlib.axes import Axes
import torch
from avalanche.training.plugins import StrategyPlugin
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_definitions import MetricValue

from avalanche.training.strategies.base_strategy import BaseStrategy
from network.trait import HasFeatureMap, IsGenerative
import torchvision.transforms as T
from PIL import Image

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCExperience
from torchvision import transforms
import numpy.random as random
import io


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class GenerateReconstruction(PluginMetric):
    to_img = T.ToPILImage()
    examples_per_experience: int
    metric_name = "ExperienceReconstruction"

    def __init__(self, scenario, examples_per_experience=1, every_n_epochs=1):
        self.examples_per_experience = examples_per_experience

        # Randomly save a sample of images to use to generate reconstructions
        self.patterns = {}
        for i, experience in enumerate(scenario.test_stream):
            experience: NCExperience = experience
            self.patterns[i] = []

            # Get a sample of images from each task
            for _ in range(self.examples_per_experience):
                x, y, _ = experience.dataset[random.randint(
                    len(experience.dataset))]
                self.patterns[i].append((x, y))

        self.n_experiences = len(scenario.test_stream)

    def add_image(self, axes: typing.Sequence[Axes], input: torch.Tensor, output: torch.Tensor, label: int, pred: int):
        axes[0].set_ylabel(f"Class: {label}")
        axes[1].set_ylabel(f"Prediction: {pred}")
        axes[0].set_title(f"Input")
        axes[1].set_title(f"Reconstruction")
        input, output = input.cpu(), output.cpu()
        axes[0].imshow(input.view(28, 28))
        axes[1].imshow(output.view(28, 28))

    @torch.no_grad()
    def after_eval_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        assert isinstance(strategy.model,
                          IsGenerative), "Network must be generative"

        plt.ioff()
        fig = plt.figure(constrained_layout=True, figsize=(6, 20))
        task_figs = fig.subfigures(self.n_experiences, 1)

        for task_id, patterns in self.patterns.items():
            sub_fig = task_figs[task_id]
            img_figs = task_figs[task_id].subplots(
                self.examples_per_experience, 2, squeeze=False)
            sub_fig.suptitle(f"Experience {task_id}")

            for i, p in enumerate(patterns):
                x, y = p
                x: torch.Tensor = x.to(strategy.device)
                
                x_hat, y_hat = strategy.model(x)
                self.add_image(img_figs[i], x, x_hat, y, torch.argmax(y_hat))
        plt.ion()
        return MetricValue(self, "Reconstructions", fig2img(plt), x_plot=strategy.clock.train_exp_counter)

    def reset(self, **kwargs) -> None:
        pass

    def result(self, **kwargs):
        pass
