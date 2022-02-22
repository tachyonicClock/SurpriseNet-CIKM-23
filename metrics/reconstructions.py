from typing import Dict

from matplotlib.figure import Figure
from mltypes import *

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

    def sample_class_exemplars(self, experience: NCExperience) -> typing.Sequence[LabeledExample]:
        """Find an exemplar from each class in an experience"""
        n_patterns = len(experience.classes_in_this_experience)
        class_examples = {}

        while True:
            # Find each class in a monte carlo way
            x, y, _ = experience.dataset[random.randint(
                len(experience.dataset))]
            class_examples[y] = (x, y)

            # Exit when one of each class is found
            if len(class_examples) >= n_patterns:
                break

        return list(class_examples.values())

    def get_examples(self, test_stream, n_exemplars=1) -> typing.Dict[int, typing.Sequence[LabeledExample]]:
        """Sample n_exemplars from each class.

        Args:
            test_stream (Any): An avalanche test_stream containing test data
            n_exemplars (int, optional): Number of exemplars per class. 
                Defaults to 1.
        Returns:
            typing.Dict[int, torch.Tensor]: Returns a dictionary mapping 
                experience to a list of exemplars
        """
        assert n_exemplars >= 1, "n_exemplars should be positive and non-zero"

        patterns = {}
        for i, experience in enumerate(test_stream):
            experience: NCExperience = experience
            # Sample n_expempars from each class
            patterns[i] = [x
                           for _ in range(n_exemplars)
                           for x in self.sample_class_exemplars(experience)
                           ]

        return patterns


    def add_image(self, axes: typing.Sequence[Axes], input: torch.Tensor, output: torch.Tensor, label: int, pred: int):
        for axe in axes:
            axe.get_xaxis().set_ticks([])
            axe.get_yaxis().set_ticks([])

        axes[0].set_ylabel(f"Class: {label}")
        axes[1].set_ylabel(f"Prediction: {pred}")
        axes[0].set_title(f"Input")
        axes[1].set_title(f"Reconstruction")
        input, output = input.cpu(), output.cpu()
        axes[0].imshow(input.view(28, 28))
        axes[1].imshow(output.view(28, 28))

    @torch.no_grad()
    def after_eval_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        # assert isinstance(strategy.model,
        #                   IsGenerative), "Network must be generative"

        n_tasks = len(self.patterns)
        n_patterns_per_task = len(self.patterns[0])

        scale = 2
        fig, ax = plt.subplots(constrained_layout=False, figsize=(scale*n_tasks*2, scale*n_patterns_per_task))
        ax.set_axis_off()
        task_figs = fig.subfigures(1, self.n_experiences)

        for task_id, task_patterns in self.patterns.items():
            sub_fig = task_figs[task_id]
            img_figs = task_figs[task_id].subplots(len(task_patterns), 2, squeeze=False)

            sub_fig.suptitle(f"Experience {task_id}")
            for i, p in enumerate(task_patterns):
                x, y = p
                x: torch.Tensor = x.to(strategy.device)

                # Pass data through auto-encoder
                x_hat, y_hat = strategy.model(x)

                self.add_image(img_figs[i], x, x_hat, y, torch.argmax(y_hat))

        metric = MetricValue(self, "Reconstructions", fig2img(fig), x_plot=strategy.clock.train_exp_counter)
        return metric

    def reset(self, **kwargs) -> None:
        pass

    def result(self, **kwargs):
        pass

    def __init__(self, scenario, examples_per_class=1, seed=42, every_n_epochs=1):
        # A sample of images to use to generate reconstructions with
        state = random.get_state()
        random.seed(seed)
        self.patterns = self.get_examples(scenario.test_stream, examples_per_class)
        self.n_experiences = len(scenario.test_stream)
        random.set_state(state)
