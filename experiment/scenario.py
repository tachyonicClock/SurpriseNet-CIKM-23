import os
import typing as t

import torch
import torchvision.transforms as T
from avalanche.benchmarks import NCScenario, nc_benchmark
from avalanche.benchmarks.classic import (SplitCIFAR10, SplitCIFAR100,
                                          SplitFMNIST, SplitMNIST)
from avalanche.benchmarks.datasets import CORe50Dataset
from torch import Tensor
from torch.utils.data import Dataset

def scenario(
    dataset: t.Literal["FMNIST", "CIFAR10", "CIFAR100", "CORe50_NC", "MNIST"], 
    dataset_root: str,
    n_experiences: int,
    supplied_class_order: t.List[int]) -> NCScenario:
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
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ]
    )
    core50_eval_transform = T.Compose(
        [
            T.ToTensor()
        ]
    )
    if dataset == "MNIST":
        return SplitMNIST(
            n_experiences=n_experiences,
            fixed_class_order=supplied_class_order,
            return_task_id=False,
            train_transform=fmnist_transform,
            eval_transform=fmnist_transform,
            dataset_root=dataset_root
        )
    elif dataset == "FMNIST":
        return SplitFMNIST(
            n_experiences=n_experiences,
            fixed_class_order=supplied_class_order,
            return_task_id=False,
            train_transform=fmnist_transform,
            eval_transform=fmnist_transform,
            dataset_root=dataset_root
        )
    elif dataset == "CIFAR100":
        return SplitCIFAR100(
            n_experiences=n_experiences,
            fixed_class_order = supplied_class_order,
            return_task_id=False,
            train_transform=cifar_train_transform,
            eval_transform=cifar_eval_transform,
            dataset_root=dataset_root
        )
    elif dataset == "CIFAR10":
        return SplitCIFAR10(
            n_experiences=n_experiences,
            fixed_class_order = supplied_class_order,
            return_task_id=False,
            train_transform=cifar_train_transform,
            eval_transform=cifar_eval_transform,
            dataset_root=dataset_root
        )
    elif dataset == "CORe50_NC":
        train_set = CORe50Dataset(root=dataset_root, train=True, transform=core50_train_transform, mini=True)
        test_set  = CORe50Dataset(root=dataset_root, train=False, transform=core50_eval_transform, mini=True)

        return NCScenario(train_set, test_set,
            task_labels=False,
            n_experiences=n_experiences,
            # CORE50 orders classes in a meaningful way, so we need to use a fixed random
            # order to be representative
            fixed_class_order=supplied_class_order
        )
    else:
        raise NotImplementedError("Dataset not implemented")
