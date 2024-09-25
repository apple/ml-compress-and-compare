"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Cifar10 classification dataset.
"""

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
from image_compression_experiments.compression_utils import set_seed


CIFAR_MEAN = np.array([125.3, 123.0, 113.9])
CIFAR_STD = np.array([63.0, 62.1, 66.7])

CIFAR_NORMALIZE = transforms.Normalize(
    mean=[x / 255.0 for x in CIFAR_MEAN], std=[x / 255.0 for x in CIFAR_STD]
)
CIFAR_UNNORMALIZE = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0 / x for x in CIFAR_STD]),
        transforms.Normalize(
            mean=[-x / 255.0 for x in CIFAR_MEAN], std=[1.0, 1.0, 1.0]
        ),
    ]
)
CIFAR_RANDOM_CROP = transforms.RandomCrop(32, padding=4)
CIFAR_RANDOM_FLIP = transforms.RandomHorizontalFlip()

CIFAR_LABELS = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_cifar10(batch_size=128, directory="./datasets"):
    """Loads the test and train splits of the CelebA dataset.

    Args:
        batch_size (int): dataloader batch size
        directory (str): location of the CelebA dataset

    Returns: the CIFAR-10 test and train dataloaders.
    """
    test_dataset, test_dataloader = load_split(
        test_transform(), batch_size, directory, train=False
    )
    train_dataset, train_dataloader = load_split(
        train_transform(augmentation=True), batch_size, directory, train=True
    )
    return test_dataloader, train_dataloader


def load_split(transform, batch_size, directory, train):
    """Loads CIFAR-10 dataset and dataloader.

    Args:
        transform (torchvision transform): transform applied to images
        batch_size (int): dataloader batch size
        directory (str): location of the CelebA dataset
        train (bool): whether to load train or test data split

    Returns: dataset and dataloader for the CIFAR-10 data
    """
    dataset = datasets.CIFAR10(
        directory, train=train, transform=transform, download=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataset, dataloader


def train_transform(augmentation):
    """CIFAR-10 image transform for training data.

    Images are normalized and converted to tensors. If augmentation, then
    images are also randomly flipped and croped.
    """
    train_transform = transforms.Compose([])
    if augmentation:
        train_transform.transforms.append(CIFAR_RANDOM_CROP)
        train_transform.transforms.append(CIFAR_RANDOM_FLIP)
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(CIFAR_NORMALIZE)
    return train_transform


def test_transform():
    """CIFAR-10 image transform for test data."""
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            CIFAR_NORMALIZE,
        ]
    )
    return test_transform


def visualization_transform(image=False):
    """CIFAR-10 image transform for visualization. Unnormalizes image tensor from
    dataloader or dataset. If image, converts the tensor to a PIL image."""
    transform = [CIFAR_UNNORMALIZE]
    if image:
        transform.append(transforms.ToPILImage())
    return transforms.Compose(transform)
