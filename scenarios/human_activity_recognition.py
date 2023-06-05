from torch.utils.data import Dataset
import os
import torch
from torch import Tensor
from avalanche.benchmarks.generators.benchmark_generators import dataset_benchmark
import typing as t
from random import shuffle
import random
from torch.utils import data
import numpy as np
from scipy.io import loadmat


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

    state = random.getstate()
    # Ensure the same split is used each time
    random.seed(0)
    for _, idx in indices.items():
        shuffle(idx)
    random.setstate(state)

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
    def __init__(self, root: str, train=False) -> None:
        super().__init__()
        self.root = root
        self.data_filename = os.path.join(self.root, "HAR", "DSADS", "dsads.mat")

        self.class_names = [
            "ascending stairs",
            "cycling on an exercise bike in horizontal positions",
            "cycling on an exercise bike in vertical positions",
            "descending stairs",
            "exercising on a cross trainer",
            "exercising on a stepper",
            "jumping",
            "lying on back side",
            "lying on right side",
            "moving around in an elevator",
            "playing basketball",
            "rowing",
            "running on a treadmill3",
            "sitting",
            "standing",
            "standing in an elevator still",
            "walking in a parking lot",
            "walking on a treadmill1",
            "walking on a treadmill2",
        ]

        """
        dsads.mat: 9120 * 408.
        Columns 0->404 are features, listed in the order of 'Torso', 'Right Arm',
        'Left Arm', 'Right Leg', and 'Left Leg'. Each position contains 81
        columns of features.
        Columns 405->407 are labels.
        Column 405 is the activity sequence indicating the executing of
        activities (usually not used in experiments).
        Column 406 is the activity label (1~19).
        Column 407 denotes the person (1~8).
        """

        mat = loadmat("/local/scratch/antonlee/datasets/HAR/DSADS/dsads.mat")
        np_data = mat["data_dsads"]
        # Remove column 405->407 because they are label information
        label = np_data[:, 406] - 1.0
        self.participants = np_data[:, 407].astype(int)
        np_data = np.delete(np_data, [405, 406, 407], axis=1)

        self.data = torch.from_numpy(np_data).float()
        self.targets = list(map(int, label))
        self.memory_usage_bytes = (
            self.data.element_size() * self.data.nelement() + 4 * len(self.targets)
        )

        # Remove participants 8 and 9 from the training set and 1-7 from the
        # testing set.
        test_mask = np.isin(self.participants, [8, 9])
        if train:
            self.data = self.data[~test_mask]
            self.targets = np.array(self.targets)[~test_mask]
            self.participants = self.participants[~test_mask]
        else:
            self.data = self.data[test_mask]
            self.targets = np.array(self.targets)[test_mask]
            self.participants = self.participants[test_mask]

        # Scale the data to [0, 1]
        self.data: Tensor = (self.data - self.data.min()) / (
            self.data.max() - self.data.min()
        )
        self.data = self.data.clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], int(self.targets[index])


class PANMAP2(Dataset):
    def __init__(self, root: str, train=True) -> None:
        super().__init__()

        self.root = root
        self.data_filename = os.path.join(root, "HAR", "PAMAP2", "pamap.mat")

        mat = loadmat(self.data_filename)["data_pamap"]

        # Columns 1~243 are features
        self.data = torch.tensor(mat[:, :243]).float()
        self.targets = torch.tensor(mat[:, 243] - 1.0).long()
        self.person = torch.tensor(mat[:, 244].astype(int))

        # Some labels are missing, so we remap the remaining labels to 0~11
        remapping = {
            11: 7,
            12: 8,
            15: 9,
            16: 10,
            23: 11,
        }
        for k, v in remapping.items():
            self.targets[self.targets == k] = v

        # Remap the labels to 0~11
        self.n_classes = 12

        # Remove person 9,8 from the training set and 1-7 from the testing set.
        mask = torch.isin(self.person, torch.tensor([9, 8]))
        if train:
            mask = ~mask

        self.data = self.data[mask]
        self.targets = self.targets[mask].tolist()
        self.person = self.person[mask]

        # Scale the data to [0, 1]
        self.data: Tensor = (self.data - self.data.min()) / (
            self.data.max() - self.data.min()
        )
        self.data = self.data.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> t.Tuple[Tensor, Tensor]:
        return self.data[index], self.targets[index]


def split_HAR(
    train_dataset: t.Union[DSADS, PANMAP2],
    test_dataset: t.Union[DSADS, PANMAP2],
    task_count: int,
    fixed_class_order: t.Optional[t.List[int]] = None,
) -> t.Tuple[t.List[data.Dataset], t.List[data.Dataset]]:
    """
    Split DSADS into a continual learning training and testing set.
    """
    if type(train_dataset) != type(test_dataset):
        raise ValueError("Train and test datasets must be of the same type.")

    train_class_dataset = _split_dataset_by_class(train_dataset, train_dataset.targets)
    test_class_dataset = _split_dataset_by_class(test_dataset, test_dataset.targets)

    if fixed_class_order is None:
        task_composition = _random_task_classes(
            task_count, len(train_dataset.class_names), drop_remainder=True
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
    dsads_train = DSADS(root, train=True)
    dsads_test = DSADS(root, train=False)
    scenario = dataset_benchmark(
        *split_HAR(dsads_train, dsads_test, task_count, fixed_class_order)
    )
    scenario.n_classes = len(dsads_test.class_names)
    return scenario


def avalanche_PAMAP2(
    root: str, task_count: int, fixed_class_order: t.Optional[t.List[int]] = None
):
    pamap2_train = PANMAP2(root, train=True)
    pamap2_test = PANMAP2(root, train=False)
    scenario = dataset_benchmark(
        *split_HAR(pamap2_train, pamap2_test, task_count, fixed_class_order)
    )
    scenario.n_classes = pamap2_test.n_classes
    return scenario
