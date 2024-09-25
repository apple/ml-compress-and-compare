"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import torch
from torch.nn.utils import prune


def global_unstructured_patch(
    parameters, pruning_method, importance_scores=None, **kwargs
):
    """
    Fixes a very strange bug in torch.nn.utils.prune.global_unstructured where setting
    the mask tensors to zero doesn't set all parameters to zero.
    """
    importance_scores = importance_scores if importance_scores is not None else {}
    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = prune.PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            "the `pruning_method`. Found method {} of type {}".format(
                pruning_method, method.PRUNING_TYPE
            )
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask

    # Check that the amount of units to prune is not > than the number of
    # parameters in t
    t = relevant_importance_scores
    tensor_size = t.nelement()
    # Compute number of units to prune: amount if int,
    # else amount * tensor_size
    nparams_toprune = prune._compute_nparams_toprune(method.amount, tensor_size)
    # This should raise an error if the number of units to prune is larger
    # than the number of units in the tensor
    prune._validate_pruning_amount(nparams_toprune, tensor_size)

    final_mask = default_mask.clone(memory_format=torch.contiguous_format)

    if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
        # largest=True --> top k; largest=False --> bottom k
        # Prune the smallest k
        normed_params = torch.abs(t).view(-1)
        topk = torch.topk(normed_params, k=nparams_toprune, largest=False)
        # topk will have .indices and .values
        # unique_indices = np.unique(topk.indices.cpu().numpy(), return_counts=True)
        # print("Indices:", len(topk.indices), [(x, y) for x, y in zip(*unique_indices) if y > 1][:100], topk.indices.max().item(), topk.indices.min().item())

        # THIS LINE HAD THE BUG
        # final_mask.view(-1)[topk.indices] = 0
        # THESE LINES ARE THE FIX
        last_value = topk.values[-1]
        final_mask = torch.where(normed_params <= last_value, 0, final_mask)

    # final_mask = container.compute_mask(relevant_importance_scores, default_mask)
    # print(tensor_size, nparams_toprune, "final num zeroed", ((1 - final_mask) > 0).sum().item(), len(final_mask), len(default_mask), ((1 - final_mask) > 0).sum().item() / len(default_mask))

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # print("zeroed out:", (1 - param_mask).sum().item())
        # print("has mask:", hasattr(module, name + "_mask"))
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        prune.custom_from_mask(module, name, mask=param_mask)

        # if (getattr(module, name) == 0).sum().item() != (1 - param_mask).sum().item():
        #     print("ERROR")

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param


def reset_pruning(pruned_layers):
    for layer, param_name in pruned_layers:
        for k, hook in layer._forward_pre_hooks.items():
            if (
                isinstance(hook, torch.nn.utils.prune.BasePruningMethod)
                and hook._tensor_name == param_name
            ):
                # delete and reset
                if hasattr(layer, param_name + "_mask"):
                    delattr(layer, param_name + "_mask")
                if param_name + "_mask" in layer._buffers:
                    del layer._buffers[param_name + "_mask"]
                setattr(layer, param_name, layer._parameters[param_name + "_orig"])
                del layer._parameters[param_name + "_orig"]
                del layer._forward_pre_hooks[k]
                break


def calculate_sparsity(prunable_modules):
    zero_counts = [float(torch.sum(m[0].weight == 0)) for m in prunable_modules]
    num_elements = [float(m[0].weight.nelement()) for m in prunable_modules]

    sparsities = []
    for zeroes, count in zip(zero_counts, num_elements):
        sparsities.append((zeroes / count) * 100.0)

    global_sparsity = sum(zero_counts) / sum(num_elements) * 100.0

    return sparsities, global_sparsity
