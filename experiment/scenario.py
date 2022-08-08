import os
import typing as t

import torch
import torchvision.transforms as T
from avalanche.benchmarks import NCScenario, nc_benchmark
from avalanche.benchmarks.classic import (SplitCIFAR10, SplitCIFAR100,
                                          SplitFMNIST)
from avalanche.benchmarks.datasets import CORe50Dataset
from torch import Tensor
from torch.utils.data import Dataset

def scenario(
    dataset: t.Literal["FMNIST", "CIFAR10", "CIFAR100", "CORe50_NC", "EmbeddedCIFAR100"], 
    dataset_root: str,
    n_experiences: int = 5, 
    fixed_class_order: bool = True) -> NCScenario:
    """Generate a new scenario.

    Note:
     - Data is not normalized

    :param dataset: The dataset used to generate the class incremental scenario
    :param dataset_root: Path to download data to
    :param n_experiences: Split the classes into n experiences, defaults to 5
    :param fixed_class_order: Should the order of classes be fixed (sequential) or random, defaults to True
    :return: A new scenario
    """

    cifar_train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ]
    )
    cifar_eval_transform = T.Compose(
        [
            T.ToTensor()
        ]
    )
    fmnist_transform = T.transforms.Compose([
        T.transforms.Resize((32, 32)),
        T.transforms.ToTensor(), 
    ])

    core50_train_transform = T.Compose(
        [
            T.RandomCrop(128, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ]
    )
    core50_eval_transform = T.Compose(
        [
            T.transforms.Resize((128, 128)),
            T.ToTensor()
        ]
    )

    if dataset == "FMNIST":
        return SplitFMNIST(
            n_experiences=n_experiences,
            fixed_class_order=list(range(10)) if fixed_class_order else None,
            return_task_id=False,
            train_transform=fmnist_transform,
            eval_transform=fmnist_transform,
            dataset_root=dataset_root
        )
    elif dataset == "CIFAR100":
        return SplitCIFAR100(
            n_experiences=n_experiences,
            fixed_class_order = list(range(100)) if fixed_class_order else None,
            return_task_id=False,
            train_transform=cifar_train_transform,
            eval_transform=cifar_eval_transform,
            dataset_root=dataset_root
        )
    elif dataset == "CIFAR10":
        return SplitCIFAR10(
            n_experiences=n_experiences,
            fixed_class_order = list(range(10)) if fixed_class_order else None,
            return_task_id=False,
            train_transform=cifar_train_transform,
            eval_transform=cifar_eval_transform,
            dataset_root=dataset_root
        )
    elif dataset == "CORe50_NC":
        train_set = CORe50Dataset(root=dataset_root, train=True, transform=core50_train_transform)
        test_set  = CORe50Dataset(root=dataset_root, train=False, transform=core50_eval_transform)

        default_core50 = [19, 30, 34, 22, 36, 23, 16, 15, 14, 49, 11, 3, 33, 28, 7, 35, 27, 18, 45, 8, 32, 9, 42, 48, 20, 17, 12, 10, 2, 21, 25, 43, 6, 1, 24, 38, 26, 44, 13, 41, 31, 40, 47, 0, 4, 37, 5, 29, 39, 46]

        return NCScenario(train_set, test_set,
            task_labels=False,
            n_experiences=n_experiences,
            # CORE50 orders classes in a meaningful way, so we need to use a fixed random
            # order to be representative
            fixed_class_order=default_core50 if fixed_class_order else None
        )
    else:
        raise NotImplementedError("Dataset not implemented")
