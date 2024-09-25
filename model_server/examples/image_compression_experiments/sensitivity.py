"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Sensitivity calculation for data-aware compression methods.
"""

import torch
import torch.nn.functional as F
from image_compression_experiments import compression_utils


def taylor_approximation_sensitivity(
    model,
    data,
    parameters,
    aggregation_function="max",
    gradient_function="max",
    normalization_function="mean",
):
    """
    Calculates sensitivities via taylor approximation approach.

    Args:
        model: A torch.nn.Module object that zero_grad can be called on
        data: Iterable such as a dataloader containing data inputs
            whose sensitivities should be aggregated
        parameters: Iterable of tuples (module, param_name)
        aggregation_function: How to aggregate results from multiple instances (can be
            max, sum, mean)
        gradient_function: How to aggregate outputs of the model_forward_fn before
            computing gradients
        normalization_function: How to normalize results at each layer by dividing each
            value by either mean or max at the layer

    Returns:
        A dictionary mapping (module, name) tuples to sensitivity
        scores for each parameter
    """
    device = "cpu"
    model.to(device)
    model.eval()
    layer_sensitivities = {}
    instance_count = 0

    for idx, (inputs, _) in enumerate(data):
        model.zero_grad()
        batch_instance_count = 0

        # Some assumptions to take note of about what the input format is - these should
        # be documented or encapsulated in some other function
        if isinstance(inputs, torch.Tensor):
            batch_instance_count = inputs.shape[0]
            if isinstance(model, torch.nn.DataParallel):
                model = model.module.to(device)
            outputs = model(inputs.to(device))
            # outputs = F.softmax(outputs, dim=1)
        else:
            raise ValueError(f"Unhandled input type {type(inputs)}")
        instance_count += batch_instance_count

        for j in range(batch_instance_count):
            # Compute the gradients for an instance
            model.zero_grad()
            if gradient_function == "max":
                outputs[j].max().backward(retain_graph=True)
            elif gradient_function == "sum":
                outputs[j].sum().backward(retain_graph=True)
            elif gradient_function == "top10":
                torch.topk(outputs[j], 10).values.sum().backward(retain_graph=True)
            elif gradient_function == "entropy":
                probs = F.softmax(outputs[j], 0)
                entropy = (-torch.log2(probs) * probs).sum()
                entropy.backward(retain_graph=True)
            else:
                raise ValueError(f"Unknown gradient function: {gradient_function}")

            for k, layer in enumerate(parameters):
                # Compute the activations of the instance for each layer
                try:
                    sens = torch.abs(
                        getattr(layer[0], "weight").grad.detach()
                        * getattr(layer[0], "weight").detach()
                    ).cpu()
                except:  # pruning an already pruned model
                    sens = torch.abs(
                        getattr(layer[0], "weight_orig").grad.detach()
                        * getattr(layer[0], "weight").detach()
                    ).cpu()
                if layer not in layer_sensitivities:
                    layer_sensitivities[layer] = sens
                elif aggregation_function == "max":
                    layer_sensitivities[layer] = torch.maximum(
                        layer_sensitivities[layer], sens
                    )
                elif aggregation_function in ("sum", "mean", "average"):
                    layer_sensitivities[layer] += sens
                else:
                    raise ValueError(
                        f"Unknown aggregation func '{aggregation_function}'"
                    )

        compression_utils.empty_device_cache()

    if aggregation_function in ("mean", "average"):
        layer_sensitivities = {
            k: x / instance_count for k, x in layer_sensitivities.items()
        }
    if normalization_function == "mean":
        layer_sensitivities = {
            k: x / x.mean() for k, x in layer_sensitivities.items()
        }  # x / x.mean()
    elif normalization_function == "max":
        layer_sensitivities = {
            k: x / x.max() for k, x in layer_sensitivities.items()
        }  # x / x.mean()

    return layer_sensitivities
