import typing as t
from torch.utils import data
from avalanche.benchmarks.generators import dataset_benchmark
import torchvision.transforms as T
import random
import numpy as np

def _schedule(class_count: int, task_count: int, increment_per_task: int) -> t.Tuple[t.List[t.List[float]], t.List[t.List[int]]]:
    """

    Based on the code from:
    https://github.com/deepmind/deepmind-research/tree/master/continual_learning
    """

    def _gaussian(peak: int, position: int):
        """What is the probability of a Gaussian with peak at `peak` and width `width`
        at position `position`?
        """
        out = np.exp(- ((position - peak)**2 / (2 * 50**2)))
        return out

    schedule_length = task_count * increment_per_task
    labels = np.random.permutation(np.arange(class_count))

    # Each class label appears according to a Gaussian probability distribution
    # with peaks spread evenly over the schedule
    peak_every = schedule_length // class_count
    peaks = range(peak_every // 2, schedule_length, peak_every)
    micro_task_count = schedule_length // increment_per_task

    label_schedule = []
    probabilities: t.List[t.List[float]] = [
        [0] * micro_task_count for _ in range(class_count)]

    for micro_task_i in range(0, micro_task_count):

        lbls = []
        # probabilities = {}
        # make sure lbls isn't empty
        while lbls == []:
            for j in range(len(peaks)):
                peak = peaks[j]
                p = _gaussian(peak, micro_task_i * increment_per_task)
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
    task_count: int,
    increment_per_task: int,
    batch_size: int
) -> t.List[t.List[int]]:
    class_count = len(class_indices)

    _, label_schedule = _schedule(
        class_count,
        task_count,
        increment_per_task)

    class_order: t.List[int] = np.random.shuffle(list(class_indices.keys()))
    micro_tasks: t.List[t.List[int]] = []
    for class_composition in label_schedule:
        n = increment_per_task * batch_size
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
    microtask_size: int,
    train_transform: T.Compose = None,
    eval_transform: T.Compose = None,
    verbose: bool = True
):
    """Create a dataset benchmark with a Gaussian schedule.

    :param train_targets: List of labels for `train_dataset`
    :param train_dataset: A training dataset split into micro tasks
    :param test_dataset: A test dataset left unchanged
    :param microtask_size: Number of instances kept in each micro task
    :param train_transform: Transform the training data, defaults to None
    :param eval_transform: Transform the evaluation data, defaults to None
    :return: _description_
    """
    assert len(train_dataset) == len(train_targets), \
        "The labels `train_targets` are mapped to `train_dataset` and must be the same length"

    class_indices = _get_indices(train_targets)
    micro_task_indices = _gs_indices(
        class_indices, 200, 5, microtask_size)

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
