import random
import typing as t

import numpy as np
import torchvision.transforms as T
from avalanche.benchmarks.generators import dataset_benchmark
from torch.utils import data


def _schedule(
    class_count: int, task_count: int, width: float
) -> t.Tuple[t.List[t.List[float]], t.List[t.List[int]]]:
    """

    Based on the code from:
    https://github.com/deepmind/deepmind-research/tree/master/continual_learning
    """

    def _gaussian(peak: int, position: int):
        """What is the probability of a Gaussian with peak at `peak` and width `width`
        at position `position`?
        """
        out = np.exp(
            -((position / task_count - peak / task_count) ** 2 / (2 * (width) ** 2))
        )
        return out

    schedule_length = task_count

    labels = np.arange(class_count)
    # TODO: Uncomment this line to make the schedule random
    # labels = np.random.permutation(labels)

    # Each class label appears according to a Gaussian probability distribution
    # with peaks spread evenly over the schedule
    peak_every = schedule_length // class_count
    peaks = range(peak_every // 2, schedule_length, peak_every)
    micro_task_count = schedule_length

    label_schedule = []
    probabilities: t.List[t.List[float]] = [
        [0] * micro_task_count for _ in range(class_count)
    ]

    for micro_task_i in range(0, micro_task_count):
        lbls = []
        # make sure lbls isn't empty
        while lbls == []:
            for j in range(len(peaks)):
                peak = peaks[j]
                p = _gaussian(peak, micro_task_i)
                probabilities[int(labels[j])][micro_task_i] = p
                if np.random.binomial(1, p):
                    lbls.append(int(labels[j]))

        label_schedule.append(lbls)

    return probabilities, label_schedule


def _get_indices(targets: t.List[int]) -> t.Dict[int, t.List[int]]:
    """Get the indices of each class in a list of targets.

    :param targets: A list of targets.
    :return: A dictionary mapping each class to a list of indices.
    """
    indices: t.Dict[int, t.List[int]] = {}
    for i, target in enumerate(targets):
        target = int(target)
        if target not in indices:
            indices[target] = []
        indices[target].append(i)
    return indices


def _gs_indices(
    class_indices: t.Dict[int, t.List[int]],
    microtask_count: int,
    width: float,
    instances_in_task: int,
) -> t.List[t.List[int]]:
    class_count = len(class_indices)

    _, label_schedule = _schedule(class_count, microtask_count, width)

    micro_tasks: t.List[t.List[int]] = []
    for class_composition in label_schedule:
        n = instances_in_task
        micro_task = []

        # Sample from each class
        for class_idx in class_composition:
            micro_task.extend(random.choices(class_indices[class_idx], k=n))

        # Ensure only n samples are selected
        micro_task = random.choices(micro_task, k=n)
        micro_tasks.append(micro_task)

    return micro_tasks


def _print_stats(micro_task_indices: t.List[t.List[int]]):
    total_size = 0
    for micro_task in micro_task_indices:
        total_size += len(micro_task)

    unique_instances = set()
    for micro_task in micro_task_indices:
        unique_instances.update(micro_task)

    print("Gaussian Schedule Stats:")
    print(f"  Micro tasks: {len(micro_task_indices)}")
    print(f"  Total size: {total_size}")
    print(f"  Unique Instances: {len(unique_instances)}")


def gaussian_schedule_dataset(
    train_targets: t.List[int],
    train_dataset: data.Dataset,
    test_dataset: data.Dataset,
    width: float,
    microtask_count: int,
    instances_in_task: int,
    train_transform: T.Compose = None,
    eval_transform: T.Compose = None,
    verbose: bool = True,
):
    """Create a dataset benchmark with a Gaussian schedule.

    :param train_targets: A list of targets for the training dataset.
    :param train_dataset: The training dataset.
    :param test_dataset: The test dataset.
    :param width: The width of the Gaussian distribution.
    :param microtask_count: The number of microtasks.
    :param instances_in_task: The number of instances in each microtask.
    :param train_transform: The transform to apply to the training dataset.
    :param eval_transform: The transform to apply to the evaluation dataset.
    :param verbose: Whether to print statistics about the generated benchmark.
    :return: A dataset benchmark.
    """
    assert len(train_dataset) == len(
        train_targets
    ), "The labels `train_targets` are mapped to `train_dataset` and must be the same length"

    class_indices = _get_indices(train_targets)
    micro_task_indices = _gs_indices(
        class_indices,
        microtask_count=microtask_count,
        width=width,
        instances_in_task=instances_in_task,
    )

    if verbose:
        _print_stats(micro_task_indices)

    micro_tasks = []
    for micro_task in micro_task_indices:
        micro_tasks.append(data.Subset(train_dataset, micro_task))

    return dataset_benchmark(
        micro_tasks,
        [test_dataset],
        complete_test_set_only=True,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
