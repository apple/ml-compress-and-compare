"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

ImageNet classification dataset.
"""

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
from torchvision import transforms
from image_compression_experiments.compression_utils import set_seed


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_imagenet(batch_size=256, directory="./datasets/"):
    """Loads the train and validation splits of the ImageNet dataset.

    Args:
        batch_size (int): dataloader batch size
        directory (str): location of the dataset

    Returns: the validation and train dataloaders.
    """
    val_dataset, val_dataloader = load_split(
        test_transform(), batch_size, directory, train=False
    )
    train_dataset, train_dataloader = load_split(
        train_transform(), batch_size, directory, train=True
    )
    return val_dataloader, train_dataloader


def load_split(transform, batch_size, directory, train=True):
    """Loads ImageNet split dataset and dataloader.

    Args:
        transform (torchvision transform): transform applied to images
        batch_size (int): dataloader batch size
        directory (str): location of the imagenet dataset
        train (bool): whether to load the training or validation set

    Returns: dataset and dataloader for ImageNet data
    """
    split = "training"
    if train is False:
        split = "validation"
    data_directory = os.path.join(directory, f"imagenet-2.0.0/data/raw/raw/{split}")
    dataset = datasets.ImageFolder(data_directory, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, pin_memory=True, num_workers=8
    )
    return dataset, dataloader


def train_transform():
    """ImageNet image transform for train data."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform


def test_transform():
    """ImageNet image transform for test data."""
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return test_transform


def visualization_transform(image=False):
    """CIFAR-10 image transform for visualization. Unnormalizes image tensor from
    dataloader or dataset. If image, converts the tensor to a PIL image."""
    transform = [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0 / x for x in IMAGENET_STD]),
        transforms.Normalize(mean=[-x for x in IMAGENET_MEAN], std=[1.0, 1.0, 1.0]),
    ]
    if image:
        transform.append(transforms.ToPILImage())
    return transforms.Compose(transform)
