"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

CelebA classification dataset.
"""


import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from image_compression_experiments.compression_utils import set_seed


CELEBA_MEAN = [0.5, 0.5, 0.5]
CELEBA_STD = [0.5, 0.5, 0.5]

CELEBA_ATTR = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


def load_celeba(batch_size=256, directory="./datasets/", attribute="Blond_Hair"):
    """Loads the test, validation, and train splits of the CelebA dataset.

    Args:
        batch_size (int): dataloader batch size
        directory (str): location of the CelebA dataset
        attribute (str): the target attribute from CELEBA_ATTR

    Returns: the CelebA test, validation, and train dataloaders.
    """
    test_dataset, test_dataloader = load_split(
        test_transform(), batch_size, "test", attribute, False, directory
    )
    val_dataset, val_dataloader = load_split(
        test_transform(), batch_size, "valid", attribute, False, directory
    )
    train_dataset, train_dataloader = load_split(
        train_transform(), batch_size, "train", attribute, True, directory
    )
    return test_dataloader, val_dataloader, train_dataloader


class CelebABinaryClassificationDataset(datasets.CelebA):
    """CelebA Dataset for binary attribute classification. Returns CelebA
    images and a single attribute as the label."""

    def __init__(self, directory, split, transform, download, attribute_index=9):
        super().__init__(directory, split=split, transform=transform, download=download)
        self.attribute_index = attribute_index
        self.classes = [
            f"Not {CELEBA_ATTR[attribute_index]}",
            CELEBA_ATTR[attribute_index],
        ]

    def __getitem__(self, idx):
        image, attributes = super().__getitem__(idx)
        label = attributes[self.attribute_index]
        return image, label


def load_split(transform, batch_size, split, attribute, balance_classes, directory):
    """Loads CelebA.

    Args:
        transform (torchvision transform): transform applied to images
        batch_size (int): dataloader batch size
        split ('test', 'valid', or 'train'): the dataset split to load
        attribute (str): the target attribute from CELEBA_ATTR
        balance_classes (bool): whether to balance the dataloader classes
        directory (str): location of the CelebA dataset

    Returns: CelebA dataset and dataloader for the data split
    """
    attribute_index = CELEBA_ATTR.index(attribute)
    dataset = CelebABinaryClassificationDataset(
        directory,
        split=split,
        transform=transform,
        download=True,
        attribute_index=attribute_index,
    )
    sampler = None
    shuffle = split == "train"
    if balance_classes:
        temp_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=8,
        )
        labels = (
            torch.cat([l for _, l in tqdm(temp_dataloader)], dim=0).detach().numpy()
        )
        counts = [len(dataset) - np.sum(labels), np.sum(labels)]
        class_weights = [1 / count for count in counts]
        sample_weights = [class_weights[i] for i in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(dataset), replacement=True
        )
        shuffle = False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=8,
        sampler=sampler,
    )
    return dataset, dataloader


def train_transform():
    """CelebA image transform for training data."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CELEBA_MEAN, std=CELEBA_STD),
        ]
    )
    return train_transform


def test_transform():
    """CelebA image transform for test data."""
    return train_transform()


def visualization_transform(image=False):
    """CelebA image transform for visualization. Unnormalizes image tensor from
    dataloader or dataset. If image, converts the tensor to a PIL image."""
    transform = [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0 / x for x in CELEBA_STD]),
        transforms.Normalize(mean=[-x for x in CELEBA_MEAN], std=[1.0, 1.0, 1.0]),
    ]
    if image:
        transform.append(transforms.ToPILImage())
    return transforms.Compose(transform)
