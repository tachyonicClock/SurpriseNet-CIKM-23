import typing as t
import torch
from torch import nn, Tensor

from tqdm import tqdm
import os
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets.cifar import CIFAR100

dataset_dir = os.environ.get("DATASETS")
device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = os.path.join(dataset_dir, "cifar100_features")

class FeatureExtractor(nn.Module):
    """A feature extractor using ResNet"""

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

# Load pre-trained feature extractor. Trained on ImageNet.
model = FeatureExtractor().to(device).eval()

cifar100_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Import cifar100 dataset
train_dataset = CIFAR100(dataset_dir, train=True, download=True, transform=cifar100_transforms)
test_dataset = CIFAR100(dataset_dir, train=False, download=True, transform=cifar100_transforms)

# Wrap cifar100 dataset in loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

@torch.no_grad()
def extract_features(loader: t.Iterable[t.Tuple[Tensor, Tensor]]) -> t.List[t.Tuple[Tensor, Tensor]]:
    """ Extract features from the dataset using the pre-trained model. """

    feature_maps: t.List[t.Tuple[Tensor, Tensor]] = []

    # Extract all features from cifar100 dataset
    for batch in tqdm(loader):
        x, y = batch[0].to(device), batch[1].to(device)

        # Get features from pre-trained model
        features: Tensor = model(x)
        # features = features.flatten(1)

        for i in range(features.shape[0]):
            label = y[i]
            feature_maps.append((features[i].cpu(), label.cpu()))
    
    return feature_maps

with torch.no_grad():
    test_feature_maps = extract_features(test_loader)
    train_feature_maps = extract_features(train_loader)

    # Save feature_maps
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(test_feature_maps[0][0].shape)

    torch.save(
        dict(data=test_feature_maps, targets=test_dataset.targets), 
        os.path.join(save_dir, "test_feature_maps.pt"))
    torch.save(
        dict(data=train_feature_maps, targets=train_dataset.targets),
        os.path.join(save_dir, "train_feature_maps.pt"))