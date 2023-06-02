from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from avalanche.benchmarks.generators.benchmark_generators import dataset_benchmark
import typing as t
from random import shuffle

from torch.utils import data


def _partition(lst: t.List[t.Any], size: int) -> t.List[t.List[t.Any]]:
    """Partition a list into chunks of size `size`.
    :param lst: A list to partition.
    :param size: The size of each chunk.
    :yield: A generator of chunks.
    """
    return [lst[i : i + size] for i in range(0, len(lst), size)]


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


def _random_task_classes(
    task_count: int, class_count: int, drop_remainder=False
) -> t.List[t.List[int]]:
    """Assign each class to a random task.

    :param task_count: The number of tasks.
    :param class_count: The number of classes.
    :return: A dictionary mapping each task to a list of classes.
    """
    if class_count % task_count != 0 and not drop_remainder:
        raise ValueError("The classes need to be equally divided among tasks.")
    class_order = list(range(class_count))
    shuffle(class_order)
    if drop_remainder and class_count % task_count != 0:
        class_order = class_order[: -(class_count % task_count)]
    return _partition(class_order, class_count // task_count)


def _class_sampled_validation_split(
    dataset: data.Dataset, validation_split: float
) -> t.Tuple[data.Dataset, t.Optional[data.Dataset]]:
    """Split a dataset into a training and validation set. Where the validation
    set is sampled from the training set so that each class is represented
    the same amount in both training and validation sets.
    """
    # Require `targets` attribute on `dataset`
    if not hasattr(dataset, "targets"):
        raise ValueError("Dataset must have a `targets` attribute.")
    if not (0.0 <= validation_split < 1.0):
        raise ValueError("Validation split must be between 0 and 1, or 0.")
    if not hasattr(dataset, "targets") or not isinstance(
        dataset.targets, list  # type: ignore
    ):
        raise ValueError("Dataset must have a `targets` attribute that is a list.")
    if validation_split == 0.0:
        return dataset, None

    # Determine indices for the split
    targets: t.List[int] = dataset.targets  # type: ignore
    indices = _get_indices(targets)  # type: ignore
    for _, idx in indices.items():
        shuffle(idx)

    # Determine the splits for each class
    train_indices, val_indices = [], []
    for _, idx in indices.items():
        split_index = int(len(idx) * validation_split)
        train_indices.extend(idx[split_index:])
        val_indices.extend(idx[:split_index])

    # Split the dataset
    train_dataset = data.Subset(dataset, train_indices)
    val_dataset = data.Subset(dataset, val_indices)

    # Update the targets
    train_dataset.targets = [targets[i] for i in train_indices]  # type: ignore
    val_dataset.targets = [targets[i] for i in val_indices]  # type: ignore
    return train_dataset, val_dataset


def _split_dataset_by_class(
    dataset: data.Dataset, targets: t.List[int]
) -> t.List[data.Dataset]:
    """Split a dataset into a dataset for each class.

    :param dataset: The dataset to split.
    :param targets: A list of targets or labels.
    :return: A list of datasets, one for each class.
    """
    indices = _get_indices(targets)
    return [data.Subset(dataset, indices[i]) for i in range(len(indices))]


class DSADS(Dataset):
    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root
        self.data_filename = os.path.join(self.root, "HAR", "DSADS", "dsads.feat")
        df = pd.read_csv(self.data_filename)

        # Turn the ActivityName column into a categorical variable
        df["ActivityName"] = df["ActivityName"].astype("category")
        self.class_names: t.List[str] = df["ActivityName"].cat.categories.tolist()
        # Replace the column with the categorical variable
        df["ActivityName"] = df["ActivityName"].cat.codes

        self.dataset_features = torch.Tensor(df.values[:, :-1])
        self.targets = list(map(int, df.values[:, -1]))
        self.memory_usage_bytes = (
            self.dataset_features.element_size() * self.dataset_features.nelement()
            + 4 * len(self.targets)
        )

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset_features[index], int(self.targets[index])


def split_DSADS(
    dsads: DSADS, task_count: int, fixed_class_order: t.Optional[t.List[int]] = None
) -> t.Tuple[t.List[data.Dataset], t.List[data.Dataset]]:
    """Split DSADS into a continual learning training and testing set."""
    train_dataset, val_dataset = _class_sampled_validation_split(dsads, 0.3)
    train_class_dataset = _split_dataset_by_class(train_dataset, train_dataset.targets)
    test_class_dataset = _split_dataset_by_class(val_dataset, val_dataset.targets)

    if fixed_class_order is None:
        task_composition = _random_task_classes(
            task_count, len(dsads.class_names), drop_remainder=True
        )
    else:
        task_composition = _partition(
            fixed_class_order, len(fixed_class_order) // task_count
        )
        # Drop the remainder
        task_composition = task_composition[:task_count]

    train_task_datasets = []
    test_task_datasets = []
    for classes in task_composition:
        train_task_datasets.append(
            data.ConcatDataset([train_class_dataset[i] for i in classes])
        )
        test_task_datasets.append(
            data.ConcatDataset([test_class_dataset[i] for i in classes])
        )
    return train_task_datasets, test_task_datasets


def avalanche_DSADS(
    root: str, task_count: int, fixed_class_order: t.Optional[t.List[int]] = None
):
    dsads = DSADS(root)
    scenario = dataset_benchmark(*split_DSADS(dsads, task_count, fixed_class_order))
    scenario.n_classes = len(dsads.class_names)
    return scenario
