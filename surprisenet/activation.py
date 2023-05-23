from io import BytesIO
import typing as t
from abc import ABC, abstractmethod

from click import secho
import graphviz
from torch import Tensor
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor


from experiment.strategy import Strategy
from network.trait import MultiOutputNetwork, SurpriseNet

TreeType = t.Dict[int, t.Optional[int]]


class ActivationStrategy(ABC):
    """A strategy defining how to activate subsets for a given task."""

    @abstractmethod
    def task_activation(self, task_id: int) -> t.Sequence[int]:
        """Given a task id, or a infered task id, return the list of subsets
        that should be activated for the task.

        :param task_id: A task id or a infered task id.
        :return: A list of subset ids.
        """


class NaiveSurpriseNetActivation(ActivationStrategy):
    def task_activation(self, task_id: int) -> t.Sequence[int]:
        return list(range(task_id + 1))


class SurpriseNetTreeActivation(ActivationStrategy):
    def __init__(self, writer: SummaryWriter):
        self.tree: TreeType = {0: None}
        self.writer = writer

    def get_subset_path(self, node: int) -> t.List[int]:
        path = []
        while node is not None:
            path.append(node)
            node = self.tree[node]
        return path

    def task_activation(self, task_id: int) -> t.Sequence[int]:
        task_id = int(task_id)
        if task_id not in self.tree:
            raise ValueError(f"Task {task_id} not found in the tree.")
        return self.get_subset_path(task_id)

    @torch.no_grad()
    def _search_best_parent(self, strategy: Strategy) -> int:
        network: t.Union[SurpriseNet, MultiOutputNetwork] = strategy.model
        aggregate_novelty_score: t.Dict[int, float] = {}
        count: t.Dict[int, int] = {}
        training = network.training
        network.eval()

        secho("Determining best subset parent", fg="green")
        for x, y, _ in strategy.dataloader:
            x = x.to(strategy.device)
            y = y.to(strategy.device)

            out = network.multi_forward(x)
            assert out.novelty_scores is not None, "Novelty scores must be calculated"

            for task_id, novelty_score in out.novelty_scores.items():
                novelty_score: Tensor = novelty_score
                aggregate_novelty_score.setdefault(task_id, 0.0)
                count.setdefault(task_id, 0)

                aggregate_novelty_score[task_id] += float(novelty_score.sum().item())
                count[task_id] += len(novelty_score)

        # Average
        for task_id in aggregate_novelty_score:
            aggregate_novelty_score[task_id] /= count[task_id]

        # Print All
        for task_id, score in aggregate_novelty_score.items():
            secho(f"{task_id}: {score}", fg="blue")

        network.train(training)
        # Return the best parent subset by finding the minimum novelty score
        return min(aggregate_novelty_score, key=aggregate_novelty_score.get)

    @staticmethod
    def plot_graph(tree: TreeType) -> Tensor:
        graph = graphviz.Digraph()
        for node, parent in tree.items():
            graph.node(str(node))
            if parent is not None:
                graph.edge(str(parent), str(node))

        # Rasterize the graph
        img = Image.open(BytesIO(graph.pipe(format="png")))
        return to_tensor(img)

    def before_training_exp(self, strategy: Strategy):
        """Before training a new task, we need to determine the best subsets
        to inherit from.
        """
        network: SurpriseNet = strategy.model
        task_id: int = network.subset_count()
        if task_id == 0:
            return
        parent = self._search_best_parent(strategy)
        self.tree[task_id] = parent
        subsets = self.get_subset_path(task_id)
        secho(f"Building {task_id} from {subsets}", fg="green")

        self.writer.add_image("InheritanceTree", self.plot_graph(self.tree), task_id)
