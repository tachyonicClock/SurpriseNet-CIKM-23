import os
import typing as t
from sympy import false

import torch
import torchvision.transforms as T
from avalanche.benchmarks import NCScenario, nc_benchmark
from avalanche.benchmarks.classic import (SplitCIFAR10, SplitCIFAR100,
                                          SplitFMNIST, SplitMNIST)
from avalanche.benchmarks.datasets import CORe50Dataset
from torch import Tensor
from torch.utils.data import Dataset

MEAN_AND_STD = {
    "FMNIST": ((0.2861), (0.3530)),
    "CIFAR10": ((0.4915, 0.4822, 0.4466), (0.2470, 0.2435, 0.2616)),
    "CIFAR100": ((0.5070, 0.4865, 0.4408), (0.2673, 0.2564, 0.2761)),
    "CORE50": ((0.6001, 0.5721, 0.5417), (0.1965, 0.2066, 0.2183))
}

EVAL_TRANSFORM = {
    "FMNIST": T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ]),
    "CIFAR10": T.Compose([
        T.ToTensor(),
    ]),
    "CIFAR100": T.Compose([
        T.ToTensor(),
    ]),
    "CORE50": T.Compose([
        T.ToTensor(),
    ])
}

TRAIN_TRANSFORMS = {
    "FMNIST": T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ]),
    "CIFAR10": T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]),
    "CIFAR100": T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]),
    "CORE50": T.Compose([
        T.RandomCrop(128, padding=16),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
}


def scenario(
    dataset: t.Literal["FMNIST", "CIFAR10", "CIFAR100", "M_CORe50_NC", "MNIST", "CORe50_NC"], 
    dataset_root: str,
    n_experiences: int,
    supplied_class_order: t.List[int],
    normalize = False,
    ) -> NCScenario:
    """Generate a new scenario.

    Note:
     - Data is not normalized

    :param dataset: The dataset used to generate the class incremental scenario
    :param dataset_root: Path to download data to
    :param n_experiences: Split the classes into n experiences, defaults to 5
    :param fixed_class_order: Should the order of classes be fixed (sequential) or random, defaults to True
    :param normalize: Should the data be normalized, defaults to False
    :return: A new scenario
    """

    eval_transform = EVAL_TRANSFORM[dataset]
    train_transform = TRAIN_TRANSFORMS[dataset]

    if normalize:
        eval_transform.transforms.append(T.Normalize(*MEAN_AND_STD[dataset]))
        train_transform.transforms.append(T.Normalize(*MEAN_AND_STD[dataset]))

    if dataset == "MNIST":
        return SplitMNIST(
            n_experiences=n_experiences,
            fixed_class_order=supplied_class_order,
            return_task_id=False,
            train_transform=eval_transform,
            eval_transform=train_transform,
            dataset_root=dataset_root
        )
    elif dataset == "FMNIST":
        return SplitFMNIST(
            n_experiences=n_experiences,
            fixed_class_order=supplied_class_order,
            return_task_id=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root
        )
    elif dataset == "CIFAR100":
        return SplitCIFAR100(
            n_experiences=n_experiences,
            fixed_class_order = supplied_class_order,
            return_task_id=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root
        )
    elif dataset == "CIFAR10":
        return SplitCIFAR10(
            n_experiences=n_experiences,
            fixed_class_order = supplied_class_order,
            return_task_id=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root
        )
    elif dataset == "M_CORe50_NC" or dataset == "CORe50_NC":
        mini = dataset == "M_CORe50_NC"
        train_set = CORe50Dataset(root=dataset_root, train=True, transform=train_transform, mini=mini)
        test_set  = CORe50Dataset(root=dataset_root, train=False, transform=eval_transform, mini=mini)

        return NCScenario(train_set, test_set,
            task_labels=False,
            n_experiences=n_experiences,
            # CORE50 orders classes in a meaningful way, so we need to use a fixed random
            # order to be representative
            fixed_class_order=supplied_class_order
        )
    else:
        raise NotImplementedError("Dataset not implemented")
