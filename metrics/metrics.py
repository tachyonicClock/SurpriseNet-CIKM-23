import typing as t
from matplotlib.patches import Patch

import numpy as np
import torch
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.training.templates import SupervisedTemplate
from experiment.loss import LossObjective
from experiment.strategy import Strategy
from matplotlib import pyplot as plt
from network.trait import PackNet
from torch import Tensor
import seaborn as sns


from .reconstructions import figure_to_image


class _MyMetric(PluginMetric[float]):
    def before_training(self, strategy):
        self.strategy = strategy

    @property
    def train_experience(self) -> int:
        """
        The experience that is currently being trained on. Or was previously
        trained on if we are in the eval phase
        """
        return self.strategy.clock.train_exp_counter

    @property
    def test_experience(self) -> int:
        """
        The experience we are currently being tested on
        """
        return self.strategy.experience.current_experience

    def reset(self):
        pass

    def result(self):
        pass


class EpochClock(_MyMetric):
    def after_training_epoch(self, strategy) -> MetricValue:
        clock = strategy.clock
        epoch = clock.train_exp_epochs
        step = strategy.clock.total_iterations
        return MetricValue(
            self, f"clock/{self.train_experience:04d}_epoch", epoch, step
        )

    def result(self, **kwargs):
        return super().result(**kwargs)

    def reset(self, **kwargs) -> None:
        return super().reset(**kwargs)


class ExperienceIdentificationCM(_MyMetric):
    def __init__(self, n_experiences: int) -> None:
        self.n_experiences = n_experiences
        self.cm: Tensor = torch.zeros((n_experiences, n_experiences)).int()
        self.reset()
        print("ExperienceIdentificationCM")

    def result(self):
        fig, ax = plt.subplots()
        ax.imshow(self.cm)
        ax.set_ylabel("True Experience")
        ax.set_xlabel("Predicted Experience")
        return figure_to_image(fig)

    def reset(self):
        self.cm.zero_()

    def before_eval(self, strategy):
        self.reset()

    def after_eval_iteration(self, strategy: Strategy) -> "MetricResult":
        exp_id = strategy.last_forward_output.exp_id
        pred_exp_id = strategy.last_forward_output.pred_exp_id
        assert pred_exp_id is not None, "Strategy did not output pred_exp_id"

        # Update the confusion matrix
        for i, j in zip(exp_id, pred_exp_id):
            self.cm[i, j] += 1

    def after_eval(self, strategy: Strategy):
        task_id_accuracy = float(self.cm.diag().sum() / self.cm.sum())
        return [
            MetricValue(
                self,
                "ExperienceIdentificationCM",
                self.result(),
                strategy.clock.total_iterations,
            ),
            MetricValue(
                self,
                "TaskIdAccuracy",
                task_id_accuracy,
                strategy.clock.total_iterations,
            ),
        ]


class SubsetRecognition(_MyMetric):
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.cm = None
        self.reset()

    def result(self):
        fig, ax = plt.subplots()
        ax.imshow(self.cm)
        ax.set_ylabel("Subset Used")
        ax.set_xlabel("True Label")

        # Add counts
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{int(self.cm[i, j])}",
                    ha="center",
                    va="center",
                    color="blue",
                )

        # Use an integer tick per cell
        ax.set_xticks(np.arange(self.cm.shape[1], dtype=int))
        ax.set_yticks(np.arange(self.cm.shape[0], dtype=int))
        return figure_to_image(fig)

    def reset(self):
        pass

    def before_eval(self, strategy) -> "MetricResult":
        model: PackNet = strategy.model
        self.cm = np.zeros((model.subset_count(), self.n_classes))

    def after_eval_iteration(self, strategy: Strategy) -> "MetricResult":
        subsets_used = strategy.last_forward_output.pred_exp_id
        true_ys = strategy.last_forward_output.y

        for subset, y in zip(subsets_used, true_ys):
            t = min(int(subset), self.cm.shape[0] - 1)
            self.cm[t, int(y)] += 1

    def after_eval(self, strategy: Strategy):
        return MetricValue(
            self, "SubsetRecognition", self.result(), strategy.clock.total_iterations
        )


class ConditionalMetrics(_MyMetric):
    correct_and_correct_task_id: int
    correct_task_id: int
    correct_and_wrong_task_id: int
    wrong_task_id: int

    def __init__(self):
        self.reset()

    @property
    def correct_given_correct_task_id(self) -> float:
        """P(correct|correct exp id)"""
        return float(self.correct_and_correct_task_id / (self.correct_task_id + 1))

    @property
    def correct_given_wrong_task_id(self) -> float:
        """P(correct|wrong exp id)"""
        return float(self.correct_and_wrong_task_id / (self.wrong_task_id + 1))

    @property
    def task_id_accuracy(self) -> float:
        """P(correct_exp_id)"""
        return float(
            self.correct_task_id / (self.wrong_task_id + self.correct_task_id + 1)
        )

    def update(self, y: Tensor, y_hat: Tensor, task_label: Tensor, task_pred: Tensor):
        correct_class = y_hat.eq(y)
        correct_task = task_pred.eq(task_label)

        self.correct_and_correct_task_id += (
            correct_class * correct_task
        ).count_nonzero()

        self.correct_and_wrong_task_id += (
            correct_class * ~correct_task
        ).count_nonzero()

        self.correct_task_id += correct_task.count_nonzero()
        self.wrong_task_id += (~correct_task).count_nonzero()

    def after_eval_iteration(self, strategy: Strategy) -> "MetricResult":
        exp_id = strategy.experience.current_experience
        if exp_id >= strategy.clock.train_exp_counter:
            return
        out = strategy.last_forward_output
        y_hat = out.y_hat.argmax(dim=1).cpu()
        task_label = exp_id * torch.ones(y_hat.size()).int()
        self.update(strategy.mb_y.cpu(), y_hat, task_label, out.pred_exp_id.cpu())

    def before_eval(self, strategy):
        self.reset()

    def after_eval(self, strategy: Strategy):
        step = strategy.clock.total_iterations

        return [
            MetricValue(
                self,
                "Conditional/P(correct|correct_task_id)",
                self.correct_given_correct_task_id,
                step,
            ),
            MetricValue(
                self,
                "Conditional/P(correct|!correct_task_id)",
                self.correct_given_wrong_task_id,
                step,
            ),
            MetricValue(
                self, "Conditional/P(correct_task_id)", self.task_id_accuracy, step
            ),
        ]

    def reset(self):
        self.correct_and_correct_task_id = 0
        self.correct_task_id = 0
        self.correct_and_wrong_task_id = 0
        self.wrong_task_id = 0

    def result(self, **kwargs):
        return None


class NoveltyScoreKde(_MyMetric):
    def __init__(self, scenario_composition: t.List[t.List[int]], n_classes: int):
        super().__init__()
        self.subset_novelty_score: t.Dict[int, t.Dict[t.List[float]]] = {}
        self.subset_composition = scenario_composition
        self.n_classes = n_classes

    def reset(self):
        self.subset_novelty_score = {}

    def before_eval(self, strategy: Strategy) -> MetricResult:
        self.reset()

    def after_eval_iteration(self, strategy: Strategy) -> MetricResult:
        new_novelty_scores = strategy.last_forward_output.novelty_scores

        if new_novelty_scores is None:
            return

        for subset, novelty_scores in new_novelty_scores.items():
            for y, novelty in zip(strategy.last_forward_output.y, novelty_scores):
                self.subset_novelty_score.setdefault(subset, {}).setdefault(
                    int(y), []
                ).append(float(novelty.item()))

    def after_eval(self, strategy: SupervisedTemplate) -> MetricResult:
        subsets = sorted(self.subset_novelty_score.keys())
        rows = len(subsets)

        if rows == 0:
            return

        fig, axes = plt.subplots(
            rows, 1, figsize=(10, 2 * rows), squeeze=False, sharex=True, sharey=True
        )
        fig.tight_layout(pad=0.0)  # Remove padding between subplots
        axes = axes.flatten()

        for subset, class_novelty in self.subset_novelty_score.items():
            for class_label, novelty in class_novelty.items():
                in_task = class_label in self.subset_composition[subset]

                # Plot Kernel Density Estimation of novelty scores
                sns.kdeplot(
                    novelty,
                    color=f"C{class_label}",
                    fill=in_task,
                    ax=axes[subset],
                )

                if in_task:
                    # Plot median of novelty scores if the class is in the task
                    axes[subset].axvline(
                        x=np.median(novelty), color=f"C{class_label}", linestyle="-"
                    )
                else:
                    # Plot median of novelty scores if the class is not in the task
                    axes[subset].axvline(
                        x=np.median(novelty),
                        color=f"C{class_label}",
                        linestyle="--",
                        alpha=0.5,
                    )

            axes[subset].set_ylabel(f"Subset {subset+1} Density")

        # Add basic legend
        class_labels = list(range(self.n_classes))
        patches = [Patch(color=f"C{y}") for y in class_labels]
        axes[0].legend(
            patches, class_labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.4)
        )

        # Save figure to image and emit metric
        img = figure_to_image(fig)
        return MetricValue(
            self, "NoveltyScoreKde", img, strategy.clock.total_iterations
        )


class LossObjectiveMetric(_MyMetric):
    n_samples: int
    loss_sum: float

    def __init__(self, name: str, loss_part: LossObjective, on_iteration: bool = False):
        super().__init__()
        self.loss_part = loss_part
        self.name = name
        self.reset()
        self.on_iteration = on_iteration

    def after_training_iteration(self, strategy: Strategy) -> "MetricResult":
        self.loss_sum += float(self.loss_part.loss)
        self.n_samples += 1
        step = strategy.clock.train_iterations

        if self.on_iteration:
            return MetricValue(
                self, f"TrainLossPartsMB/{self.name}", float(self.loss_part.loss), step
            )

    def reset(self):
        self.n_samples = 0.0
        self.loss_sum = 0.0

    def result(self):
        return self.loss_sum / self.n_samples

    def after_training_epoch(self, strategy: Strategy) -> "MetricResult":
        step = strategy.clock.train_iterations
        value = self.result()
        self.reset()
        return MetricValue(self, f"TrainLossPart/{self.name}", value, step)


class EvalLossObjectiveMetric(_MyMetric):
    n_samples: int
    loss_sum: float

    def __init__(self, name: str, loss_part: LossObjective):
        super().__init__()
        self.loss_part = loss_part
        self.name = name
        self.reset()

    def after_eval_iteration(self, strategy: Strategy) -> "MetricResult":
        self.loss_sum += float(self.loss_part.loss)
        self.n_samples += 1

    def reset(self):
        self.n_samples = 0.0
        self.loss_sum = 0.0

    def result(self):
        return self.loss_sum / self.n_samples

    def after_eval_exp(self, strategy: Strategy) -> "MetricResult":
        step = strategy.clock.total_iterations
        value = self.result()
        self.reset()
        return MetricValue(
            self,
            f"EvalLossPart/Experience_{self.test_experience}/{self.name}",
            value,
            step,
        )
