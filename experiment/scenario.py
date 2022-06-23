import os
import typing as t
from torch import Tensor
import torch

from torch.utils.data import Dataset
import torchvision.transforms as T 

from avalanche.benchmarks.classic import SplitFMNIST, SplitCIFAR100, SplitCIFAR10, CORe50
from avalanche.benchmarks import NCScenario
from avalanche.benchmarks import nc_benchmark


class FeatureMapDataset(Dataset):
    """
    Dataset from extracting feature maps from a dataset.
    """
    
    def __init__(self, data: t.List[t.Tuple[Tensor, Tensor]], targets: Tensor):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> t.Tuple[Tensor, Tensor]:
        x, y = self.data[idx]
        return x.flatten(), y

def BI_CIFAR100(dataset_root: str, n_experiences: int, fixed_class_order: bool) -> NCScenario:
    """
    Create a scenario for a Brain Inspired CIFAR100 scenario where the classes 
    are split into n_experiences. CIFAR100 has been pre-processed by creating
    an embedded representation of the image using a pre-trained ResNet50.

    It is called brain inspired because creating an embedding of the image
    is analogous to the early stable regions of the brain visual cortex 
    (van de Ven et al., 2020).


    van de Ven, G. M., Siegelmann, H. T., & Tolias, A. S. (2020). 
    Brain-inspired replay for continual learning with artificial neural 
    networks. Nature Communications, 11(1), 4069. 
    https://doi.org/10.1038/s41467-020-17866-2
    """
    saved_train_embeddings = torch.load(os.path.join(dataset_root, "cifar100_features", "train_feature_maps.pt"))
    saved_test_embeddings = torch.load(os.path.join(dataset_root, "cifar100_features", "test_feature_maps.pt"))
    print("Loaded datasets")

    # Create a dataset from the feature maps
    train_dataset = FeatureMapDataset(saved_train_embeddings["data"], saved_train_embeddings["targets"])
    test_dataset = FeatureMapDataset(saved_test_embeddings["data"], saved_test_embeddings["targets"])

    print("Setup datasets")

    benchmark = nc_benchmark(
        train_dataset, 
        test_dataset,
        shuffle=True,
        n_experiences=n_experiences, 
        task_labels=False,
        fixed_class_order=list(range(100)) if fixed_class_order else None,)
    print("Setup benchmarks")

    return benchmark

def scenario(
    dataset: t.Literal["FMNIST", "CIFAR10", "CIFAR100", "CORe50_NC", "BI_CIFAR100"], 
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
    elif dataset == "BI_CIFAR100":
        return BI_CIFAR100(
            dataset_root=dataset_root,
            n_experiences=n_experiences,
            fixed_class_order=fixed_class_order
        )
    else:
        return NotImplementedError("Dataset not implemented")