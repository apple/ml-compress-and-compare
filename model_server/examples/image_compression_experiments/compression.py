"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Compression functions.
"""

import copy
import numpy as np
import random
import torch
import torch.nn.utils.prune as prune
from torch.quantization import quantize_fx

from image_compression_experiments import compression_utils, sensitivity


def quantize(model, amount, data, copy_model=True, permanent=False, backend="qnnpack"):
    """Applies 8-bit quantization to the model.

    Args:
        model (PyTorch module): Model to prune.
        amount: Unused parameter for compression function consistency.
        data (None or torch.Tensor): Data used for quantization.
        copy_model (bool): If True, makes a copy of the current model.
        permanent (bool): Unused parameter for compression function consistency.
        backend (str): Quantization backend.

    Returns: Quantized model.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if copy_model:
        model = copy.deepcopy(model)
    model.eval().to("cpu")  # no gpu support for quanitzation
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
    example_inputs, _ = next(iter(data))
    model_prepared = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs)
    with torch.no_grad():  # calibrate the model
        for item, label in data:
            item = item.to("cpu")
            model_prepared(item)
    model_quantized = quantize_fx.convert_fx(model_prepared)
    return model_quantized


def randompruning(model, amount, data=None, copy_model=True, permanent=False):
    """Applies random pruning to the model.

    Args:
        model (PyTorch module): Model to prune.
        amount (float): A value between 0 and 1 indicating what percentage of the
            model to prune.
        data: Unused parameter for compression function consistency.
        copy_model (bool): If True, makes a copy of the current model.
        permanent (bool): If True, removes additional pruning weights and sets
            the weight parameter to 0.

    Returns: Randomly pruned model.
    """
    compression_utils.set_seed()
    return _prune(
        prune.RandomUnstructured,
        model,
        amount,
        copy_model=copy_model,
        permanent=permanent,
    )


def magnitudepruning(model, amount, data=None, copy_model=True, permanent=False):
    """Applies global magnitude pruning to the model.

    Args:
        model (PyTorch module): Model to prune.
        amount (float): A value between 0 and 1 indicating what percentage of the
            model to prune.
        data: Unused parameter for compression function consistency.
        copy_model (bool): If True, makes a copy of the current model.
        permanent (bool): If True, removes additional pruning weights and sets
            the weight parameter to 0.

    Returns: Magnitude pruned model.
    """
    compression_utils.set_seed()
    return _prune(
        prune.L1Unstructured, model, amount, copy_model=copy_model, permanent=permanent
    )


def gradientpruning(model, amount, data, copy_model=True, permanent=False):
    """Applies global gradient pruning to the model.

    Args:
        model (PyTorch module): Model to prune.
        amount (float): A value between 0 and 1 indicating what percentage of the
            model to prune.
        data (PyTorch DataLoader): Data to compute gradient with.
        copy_model (bool): If True, makes a copy of the current model.
        permanent (bool): If True, removes additional pruning weights and sets
            the weight parameter to 0.

    Returns: Gradient pruned model.
    """

    def taylor_approximation_importance(model, parameters_to_prune):
        importance_scores = sensitivity.taylor_approximation_sensitivity(
            model, data, parameters_to_prune
        )
        return importance_scores

    compression_utils.set_seed()
    return _prune(
        prune.L1Unstructured,
        model,
        amount,
        taylor_approximation_importance,
        copy_model=copy_model,
        permanent=permanent,
    )


def _prune(
    pruning_method,
    model,
    amount,
    importance_fn=lambda m, p: None,
    copy_model=True,
    permanent=False,
):
    """Prunes the model"""
    if copy_model:
        model = copy.deepcopy(model)
    parameters_to_prune = compression_utils.get_prunable_parameters(model)[1]
    importance_scores = importance_fn(model, parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount=amount,
        importance_scores=importance_scores,
    )
    if permanent:
        compression_utils.make_pruning_permanent(model)
    return model
