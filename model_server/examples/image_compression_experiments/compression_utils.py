"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Utility functions for compression.
"""

import random
import numpy as np
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm
import gc


def get_prunable_parameters(model):
    """Returns the prunable modules and module names in the model."""
    layers_to_prune = []
    module_names = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            if hasattr(module, "weight") or hasattr(module, "weight_orig"):
                layers_to_prune.append((module, "weight"))
                module_names.append(name)
    return module_names, layers_to_prune


def compute_model_sparsity(model):
    """Computes the percentage of prunable weights set to 0."""
    prunable_modules = get_prunable_parameters(model)[1]

    zero_counts = [float(torch.sum(m[0].weight == 0)) for m in prunable_modules]
    num_elements = [float(m[0].weight.nelement()) for m in prunable_modules]

    sparsities = []
    for zeroes, count in zip(zero_counts, num_elements):
        sparsities.append((zeroes / count))

    print(f"{int(sum(zero_counts))} pruned of {int(sum(num_elements))} parameters")
    global_sparsity = sum(zero_counts) / sum(num_elements)
    return global_sparsity


def make_pruning_permanent(model):
    """Removes additional pruning parameters and sets pruned weights to 0."""
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            prune.remove(module, "weight")


def _reset_batchnorm(m):
    """Reset batch normalization parameters on module m."""
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.reset_running_stats()


def recalibrate(model, dataloader, device="cuda"):
    """Update batch norm layers of model using data in dataloader."""
    model.apply(_reset_batchnorm)
    model.train()
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        model(images)
    model.eval()


def set_seed(seed=0):
    """Sets torch, random, and numpy seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def empty_device_cache():
    """Empties the cuda and mps caches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        from torch import mps

        mps.empty_cache()
    gc.collect()
