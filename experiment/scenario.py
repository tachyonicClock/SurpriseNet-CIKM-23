import typing as t
import gin

import torchvision.transforms as T 

from avalanche.benchmarks.classic import SplitFMNIST, SplitCIFAR100, SplitCIFAR10, CORe50
from avalanche.benchmarks import NCScenario

def scenario(
    dataset: t.Literal["FMNIST", "CIFAR10", "CIFAR100", "CORe50_NC"], 
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
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ]
    )
    core50_eval_transform = T.Compose(
        [
            T.transforms.Resize((32, 32)),
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
        return CORe50(
            scenario="nicv2_391",
            train_transform=core50_train_transform,
            eval_transform=core50_eval_transform,
            dataset_root=dataset_root
        )
    pass
