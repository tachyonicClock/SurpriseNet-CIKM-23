import io
import typing

import matplotlib.pyplot as plt
import numpy.random as random
import torch
from avalanche.benchmarks.scenarios.deprecated.new_classes.nc_scenario import (
    NCExperience,
)
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_definitions import MetricValue
from experiment.strategy import ForwardOutput, Strategy
from matplotlib.axes import Axes
from network.trait import (
    Classifier,
    MultiOutputNetwork,
    Samplable,
)
from PIL import Image
from torchvision.transforms.functional import to_pil_image

plt.ioff()


def MRAE(x_hat: torch.Tensor, x: torch.Tensor, reduce_batch=True) -> torch.Tensor:
    """Relative absolute error"""
    start = 0 if reduce_batch else 1
    x = x.flatten(start)
    x_mu = x.mean()
    x_hat = x_hat.flatten(start)
    return (x_hat - x).abs().sum(dim=[start]) / (x - x_mu).abs().sum()


def figure_to_image(fig: plt.Figure) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


LabeledExample = typing.Tuple[int, torch.Tensor]


def hide_axis(axes: Axes):
    # Hide axis
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])


def to_image(img: torch.Tensor) -> torch.Tensor:
    return to_pil_image(img.squeeze())


class GenerateReconstruction(PluginMetric):
    examples_per_experience: int
    metric_name = "ExperienceReconstruction"

    def sample_class_exemplars(
        self, experience: NCExperience
    ) -> typing.Sequence[LabeledExample]:
        """Find an exemplar from each class in an experience"""
        n_patterns = len(experience.classes_in_this_experience)
        class_examples = {}

        while True:
            # Find each class in a monte carlo way
            x, y, _ = experience.dataset[random.randint(len(experience.dataset))]
            class_examples[y] = (x, y)

            # Exit when one of each class is found
            if len(class_examples) >= n_patterns:
                break

        return list(class_examples.values())

    def add_image(
        self,
        axes: typing.Sequence[Axes],
        input: torch.Tensor,
        output: torch.Tensor,
        label: int,
        pred: int,
        pred_exp: int,
    ):
        axes[0].set_ylabel(f"Class={label}")
        axes[1].set_ylabel(f"{pred} using {pred_exp}")
        axes[0].set_title("Input")
        loss = float(MRAE(input.reshape(output.shape), output))
        axes[1].set_title(f"Loss={loss:.04}")

        for ax in axes:
            hide_axis(ax)

        input, output = to_image(input), to_image(output)

        axes[0].imshow(input)
        axes[1].imshow(output)

    def get_examples(
        self, test_stream, n_exemplars=1
    ) -> typing.Dict[int, typing.Sequence[LabeledExample]]:
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
            patterns[i] = [
                x
                for _ in range(n_exemplars)
                for x in self.sample_class_exemplars(experience)
            ]

        return patterns

    @torch.no_grad()
    def after_eval(self, strategy: Strategy):
        model = strategy.model
        assert isinstance(model, MultiOutputNetwork)

        n_tasks = len(self.patterns)
        n_patterns_per_task = len(self.patterns[0])

        scale = 2
        fig, ax = plt.subplots(
            constrained_layout=False,
            figsize=(scale * n_tasks * 2, scale * n_patterns_per_task),
        )
        ax.set_axis_off()
        task_figs = fig.subfigures(1, self.n_experiences)

        # For each
        for task_id, task_patterns in self.patterns.items():
            if self.n_experiences == 1:
                task_fig = task_figs
            else:
                task_fig = task_figs[task_id]
            task_fig.suptitle(f"Experience {task_id}")

            # Let the task oracle task inference strategy know what task we
            # are using
            strategy.experience.current_experience = task_id

            task_plots = task_fig.subplots(len(task_patterns), 2, squeeze=False)
            for pattern, pattern_plot in zip(task_patterns, task_plots):
                x, y = pattern
                x: torch.Tensor = x.to(strategy.device)

                # Pass data through auto-encoder
                out: ForwardOutput = strategy.model.multi_forward(x.unsqueeze(0))

                class_prediction = "NA"
                if out.y_hat is not None:
                    class_prediction = torch.argmax(out.y_hat)

                experience_prediction = "NA"
                if out.pred_exp_id is not None:
                    experience_prediction = int(out.pred_exp_id)

                self.add_image(
                    pattern_plot,
                    x,
                    out.x_hat,
                    y,
                    class_prediction,
                    experience_prediction,
                )

        x_plot = strategy.clock.train_exp_counter
        metric = MetricValue(self, "Reconstructions", figure_to_image(fig), x_plot)

        plt.close("all")
        return metric

    def reset(self, **kwargs) -> None:
        pass

    def result(self, **kwargs):
        pass

    def __init__(self, scenario, examples_per_class=1, seed=42):
        # A sample of images to use to generate reconstructions with
        state = random.get_state()
        random.seed(seed)
        self.patterns = self.get_examples(scenario.test_stream, examples_per_class)
        self.n_experiences = len(scenario.test_stream)
        random.set_state(state)


class GenerateSamples(PluginMetric):
    rows: int  # How may columns to generate
    cols: int  # How many rows to generate

    img_size: float = 2.0  # How big each image should be matplotlib units

    def add_image(self, axes: Axes, model: Samplable):
        # Randomly generate the latent dimension
        gen_x = model.sample(1)

        if isinstance(model, Classifier):
            gen_y = torch.argmax(model.classify(gen_x))
        else:
            gen_y = -1

        axes.imshow(to_image(gen_x.cpu()))
        hide_axis(axes)
        axes.set_title(f"Prediction {gen_y}")

    @torch.no_grad()
    def after_eval(self, strategy: Strategy):
        assert isinstance(strategy.model, Samplable), "Network must be `Samplable`"

        self.device = strategy.device
        fig, axes = plt.subplots(
            nrows=self.rows,
            ncols=self.cols,
            constrained_layout=True,
            figsize=(self.cols * self.img_size, self.rows * self.img_size),
        )

        # Add image by sampling for each row and column
        for task_id, rows in enumerate(axes):
            for ax in rows:
                if self.rows_are_experiences:
                    t_available = min(strategy.model.subset_count() - 1, task_id)
                    strategy.model.activate_task_id(t_available)
                    ax.set_ylabel(f"Subnet {t_available}")
                self.add_image(ax, strategy.model)

        metric = MetricValue(
            self, "Sample", figure_to_image(fig), x_plot=strategy.clock.total_iterations
        )
        plt.close(fig)
        return metric

    def reset(self, **kwargs) -> None:
        pass

    def result(self, **kwargs):
        pass

    def __init__(self, rows, cols, img_size=2.0, rows_are_experiences=False):
        """_summary_

        :param rows: _description_
        :param cols: _description_
        :param img_size: _description_, defaults to 2.0
        :param rows_are_experiences: Should rows be experiences, defaults to False
        """
        self.rows = rows
        self.cols = cols
        self.img_size = img_size
        self.rows_are_experiences = rows_are_experiences
