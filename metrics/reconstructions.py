import io
import typing

import matplotlib.pyplot as plt
import numpy.random as random
import torch
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCExperience
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_definitions import MetricValue
from matplotlib.axes import Axes
from mltypes import *
from network.trait import Generative, PackNetModule
from PIL import Image


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class GenerateReconstruction(PluginMetric):
    examples_per_experience: int
    metric_name = "ExperienceReconstruction"

    def sample_class_exemplars(self, experience: NCExperience) \
            -> typing.Sequence[LabeledExample]:
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

    def add_image(
            self,
            axes:   typing.Sequence[Axes],
            input:  torch.Tensor, output: torch.Tensor,
            label:  int, pred:   int):

        # Hide axis
        for axe in axes:
            axe.get_xaxis().set_ticks([])
            axe.get_yaxis().set_ticks([])

        axes[0].set_ylabel(f"Class: {label}")
        axes[1].set_ylabel(f"Prediction: {pred}")
        axes[0].set_title(f"Input")
        axes[1].set_title(f"Reconstruction")
        input, output = input.cpu(), output.cpu()
        axes[0].imshow(input.squeeze())
        axes[1].imshow(output.squeeze())

    def get_examples(self, test_stream, n_exemplars=1) \
            -> typing.Dict[int, typing.Sequence[LabeledExample]]:
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

    @torch.no_grad()
    def after_eval_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        model = strategy.model
        assert isinstance(model, Generative), "Network must be generative"

        n_tasks = len(self.patterns)
        n_patterns_per_task = len(self.patterns[0])

        scale = 2
        fig, ax = plt.subplots(constrained_layout=False, figsize=(
            scale*n_tasks*2, scale*n_patterns_per_task))
        ax.set_axis_off()
        task_figs = fig.subfigures(1, self.n_experiences)

        # For each 
        for task_id, task_patterns in self.patterns.items():
            task_fig = task_figs[task_id]
            task_fig.suptitle(f"Experience {task_id}")

            if isinstance(model, PackNetModule):
                print("USING SUBSET")
                model.use_task_subset(task_id)

            task_plots = task_fig.subplots(len(task_patterns), 2, squeeze=False)
            for pattern, pattern_plot in zip(task_patterns, task_plots):
                x, y = pattern
                x: torch.Tensor = x.to(strategy.device)

                # Pass data through auto-encoder
                out = strategy.model.forward(x.unsqueeze(0))

                self.add_image(pattern_plot, x, out.x_hat, y, torch.argmax(out.y_hat))

        if isinstance(model, PackNetModule):
            model.use_top_subset()

        x_plot = strategy.clock.train_exp_counter
        metric = MetricValue(self, "Reconstructions", fig2img(fig), x_plot)

        plt.close("all")
        return metric

    def reset(self, **kwargs) -> None:
        pass

    def result(self, **kwargs):
        pass

    def __init__(self, scenario, examples_per_class=1, seed=42, every_n_epochs=1):
        # A sample of images to use to generate reconstructions with
        state = random.get_state()
        random.seed(seed)
        self.patterns = self.get_examples(
            scenario.test_stream, examples_per_class)
        self.n_experiences = len(scenario.test_stream)
        random.set_state(state)

class GenerateSamples(PluginMetric):
    rows: int # How may columns to generate
    cols: int # How many rows to generate

    img_size: float = 2.0 # How big each image should be matplotlib units

    def add_image(self, axes: Axes, model: Generative):
        # Randomly generate the latent dimension
        gen_z = model.sample_z().to(self.device)
        # print("add_image", gen_z)
        # Use the generated z to generate a pattern
        gen_x: Tensor = model.decode(gen_z)
        # Use the generated pattern to classify the instance
        gen_y = torch.argmax(model.classify(gen_x))

        axes.imshow(gen_x.cpu().squeeze())
        axes.set_axis_off()
        axes.set_title(f"Prediction {gen_y}")

    @torch.no_grad()
    def after_eval_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        assert isinstance(strategy.model,
                          Generative), "Network must be generative"
        self.device = strategy.device
        plt.ioff()
        fig, axes = plt.subplots(
            nrows=self.rows, ncols=self.cols,
            constrained_layout=True,
            figsize=(self.cols*self.img_size, self.rows*self.img_size))

        # Add image by sampling for each row and column
        for rows in axes:
            for ax in rows:
                self.add_image(ax, strategy.model)

        metric = MetricValue(self, "Sample", fig2img(fig), x_plot=strategy.clock.train_exp_counter)
        plt.close(fig)
        plt.ion()
        return metric

    def reset(self, **kwargs) -> None:
        pass

    def result(self, **kwargs):
        pass

    def __init__(self, rows, cols, img_size=2.0):
        self.rows = rows
        self.cols = cols
        self.img_size = img_size
