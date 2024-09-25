"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Image classification models.
"""

import torch
import torchvision.models as models


def load_model(name, pretrained=True):
    """Loads model.

    Args:
        name (str): String corresponding to PyTorch architecture.
        pretrained (bool): Whether to use pretrained weights.

    Returns: Model.
    """
    weights = None
    if pretrained:
        weights = "IMAGENET1K_V1"
    model = models.get_model(name, weights=weights)
    criterion, optimizer, scheduler, batch_size, epoch = get_training_parameters(
        name, model, pretrained
    )

    return model, criterion, optimizer, scheduler, batch_size, epoch


def get_training_parameters(name, model, pretrained):
    """Loads model training paramters

    Args:
        name (str): String corresponding to PyTorch architecture.
        model (PyTorch module): Model to train
        pretrained (bool): Whether to use pretrained weights.

    Returns: Criterion, optimizer, scheduler, batch_size, and starting epoch.
    """
    epoch = 0
    if name.startswith("resnet"):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=30)
        batch_size = 256
        if pretrained:
            epoch = 90
    elif name == "mobilenet_v2":
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.045, momentum=0.9, weight_decay=4e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.98, step_size=1)
        batch_size = 256
        if pretrained:
            epoch = 300
    else:
        raise ValueError(f"Unknown model name: {name}")
    return criterion, optimizer, scheduler, batch_size, epoch
