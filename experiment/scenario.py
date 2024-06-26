import typing as t
import torch
import torchvision.transforms as T
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks import NCScenario
from avalanche.benchmarks.classic import (
    SplitCIFAR10,
    SplitCIFAR100,
    SplitFMNIST,
    SplitMNIST,
)
from avalanche.benchmarks.datasets import CORe50Dataset
from torchvision.datasets import MNIST

from scenarios.human_activity_recognition import (
    avalanche_DSADS,
    avalanche_PAMAP2,
)

MEAN_AND_STD = {
    "FMNIST": ((0.2861), (0.3530)),
    "CIFAR10": ((0.4915, 0.4822, 0.4466), (0.2470, 0.2435, 0.2616)),
    "CIFAR100": ((0.5070, 0.4865, 0.4408), (0.2673, 0.2564, 0.2761)),
    "CORe50_NC": ((0.6001, 0.5721, 0.5417), (0.1965, 0.2066, 0.2183)),
    "M_CORe50_NC": ((0.6001, 0.5721, 0.5417), (0.1965, 0.2066, 0.2183)),
}

EVAL_TRANSFORM = {
    "FMNIST": T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
        ]
    ),
    "MNIST": T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
        ]
    ),
    "CIFAR10": T.Compose(
        [
            T.ToTensor(),
        ]
    ),
    "CIFAR100": T.Compose(
        [
            T.ToTensor(),
        ]
    ),
    "CORe50_NC": T.Compose(
        [
            T.ToTensor(),
        ]
    ),
    "M_CORe50_NC": T.Compose(
        [
            T.ToTensor(),
        ]
    ),
    "DSADS": T.Compose(
        [
            T.ToTensor(),
        ]
    ),
    "PAMAP2": T.Compose(
        [
            T.ToTensor(),
        ]
    ),
}


def dequantize(x: torch.Tensor) -> torch.Tensor:
    """Dequantize a tensor originating from pixel values. We inject noise
    to avoid the same pixel values to be always mapped to the same float
    value. The values become, sort of, continuous.

    :param x: An input tensor with float values in [0, 1]
    :return: A dequantized tensor with float values in [0, 1]
    """
    return (x * 255 + torch.rand_like(x)) / 256


TRAIN_TRANSFORMS = {
    "FMNIST": T.Compose(
        [
            T.ToTensor(),
            T.Resize((32, 32), antialias=True),
            T.RandomHorizontalFlip(),
            dequantize,
        ]
    ),
    "MNIST": T.Compose(
        [
            T.ToTensor(),
            T.Resize((32, 32)),
            dequantize,
        ]
    ),
    "CIFAR10": T.Compose(
        [
            T.ToTensor(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            dequantize,
        ]
    ),
    "CIFAR100": T.Compose(
        [
            T.ToTensor(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            dequantize,
        ]
    ),
    "CORe50_NC": T.Compose(
        [
            T.ToTensor(),
            T.RandomCrop(128, padding=16),
            T.RandomHorizontalFlip(),
            dequantize,
        ]
    ),
    "M_CORe50_NC": T.Compose(
        [
            T.ToTensor(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            dequantize,
        ]
    ),
    "DSADS": T.Compose(
        [
            T.ToTensor(),
        ]
    ),
    "PAMAP2": T.Compose(
        [
            T.ToTensor(),
        ]
    ),
}


def split_scenario(
    dataset: t.Literal[
        "FMNIST",
        "CIFAR10",
        "CIFAR100",
        "M_CORe50_NC",
        "MNIST",
        "CORe50_NC",
        "DSADS",
    ],
    dataset_root: str,
    n_experiences: int,
    supplied_class_order: t.List[int],
    normalize: bool,
) -> NCScenario:
    """Generate a new scenario."""

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
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )
    elif dataset == "FMNIST":
        return SplitFMNIST(
            n_experiences=n_experiences,
            fixed_class_order=supplied_class_order,
            return_task_id=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )
    elif dataset == "CIFAR100":
        return SplitCIFAR100(
            n_experiences=n_experiences,
            fixed_class_order=supplied_class_order,
            return_task_id=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )
    elif dataset == "CIFAR10":
        return SplitCIFAR10(
            n_experiences=n_experiences,
            fixed_class_order=supplied_class_order,
            return_task_id=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )
    elif dataset == "DSADS":
        return avalanche_DSADS(
            dataset_root,
            n_experiences,
            fixed_class_order=supplied_class_order,
        )
    elif dataset == "PAMAP2":
        return avalanche_PAMAP2(
            dataset_root,
            n_experiences,
            fixed_class_order=supplied_class_order,
        )
    elif dataset == "M_CORe50_NC" or dataset == "CORe50_NC":
        # Determine if we are using the mini version of CORe50
        core_mini = dataset == "M_CORe50_NC"
        train_set = nc_benchmark(
            CORe50Dataset(
                root=dataset_root, train=True, transform=train_transform, mini=core_mini
            )
        )
        assert hasattr(train_set, "targets"), "Targets not found"
        test_set = nc_benchmark(
            CORe50Dataset(
                root=dataset_root, train=False, transform=eval_transform, mini=core_mini
            )
        )
        assert hasattr(test_set, "targets"), "Targets not found"

        return NCScenario(
            train_dataset=train_set,
            test_dataset=test_set,
            task_labels=False,
            n_experiences=n_experiences,
            fixed_class_order=supplied_class_order,
        )
    else:
        raise NotImplementedError("Dataset not implemented")
