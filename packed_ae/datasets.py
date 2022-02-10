from __future__ import annotations  
# Python typing is so bad https://www.python.org/dev/peps/pep-0563/

import typing
from random import shuffle

import torch.utils.data
from torchvision.datasets.cifar import CIFAR100

from config import *
from utils import *

class TaskGroup:
    """A task group defines contains multiple task datasets"""

    class Task:
        """A task is a group of classes"""

        train_set: typing.Any          # Training data set
        val_set: typing.Any            # Validation data set
        test_set: typing.Any           # Testing data set
        classes: typing.Sequence[int]  # IDs of the classes present
        task_id: int                   # ID of the task

        def __init__(self, task_id: int, classes: typing.Sequence[int]) -> None:
            """Define the classes present in the task"""
            self.classes = classes
            self.task_id = task_id

    tasks: typing.Sequence[Task]

    def __init__(self, n_classes, n_tasks) -> None:
        """Create a group of subtasks by randomly splitting n_classes into n_tasks """

        assert n_classes % n_tasks == 0, f"{n_tasks}(n_tasks) must divide evenly into {n_classes}(n_classes)"
        task_size = n_classes // n_tasks
        labels = list(range(n_classes))
        shuffle(labels)
        self.tasks = [TaskGroup.Task(i, labels[task_size*i:task_size*(i+1)])
                      for i in range(n_tasks)]

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, index) -> Task:
        return self.tasks[index]

    @classmethod
    def from_class_split(cls: TaskGroup,
                         n_classes: int,
                         n_tasks:   int,
                         train_set: torch.utils.data.Dataset,
                         test_set:  torch.utils.data.Dataset,
                         val_set:   torch.utils.data.Dataset) -> TaskGroup:
        task_group = TaskGroup(n_classes, n_tasks)

        for task in task_group:
            def _in_task(set):
                index_where(lambda x: x in task.classes, set.targets)

            task.test_set = torch.utils.data.Subset(
                train_set, _in_task(test_set))
            task.train_set = torch.utils.data.Subset(
                train_set, _in_task(train_set))
            task.val_set = torch.utils.data.Subset(
                train_set, _in_task(val_set))

        return task_group


def split_cifar100(n_tasks) -> TaskGroup:
    """Split cifar100 into n_tasks"""
    train_set = CIFAR100(root=DATASET_ROOT, download=True, train=True)
    test_set = CIFAR100(root=DATASET_ROOT, download=True, train=False)

    val_size = 30*100  # validation set size
    val_set, test_set = torch.utils.data.random_split(
        train_set, [val_size, len(test_set)-val_size])

    return TaskGroup.from_class_split(100, n_tasks, train_set, test_set, val_set)
